"""Original v7 worker functions with observation hooks and bounded saves."""
import argparse
import inspect
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from control_common import *
from retention_base import SaveRetention, verify_runtime_uuid


def validate_environment(config):
    require("torch" not in sys.modules and os.environ.get("CUDA_VISIBLE_DEVICES") == ("" if config["mode"] == "preflight" else config["physical_gpu"])
            and os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
            and not any(k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")), "GPU environment not set before import")


class BoundedRetention(SaveRetention):
    def __init__(self, torch, work, report, data, epochs):
        super().__init__(torch, work, report)
        self.epochs = epochs
        view = Path(data).parent
        self.caches = {view / "labels/train.cache", view / "labels/val.cache"}

    def save(self, obj, destination, *args, **kwargs):
        require(isinstance(destination, (str, bytes, os.PathLike)), "Undeclared file-like torch.save")
        path = Path(os.fsdecode(destination)).absolute()
        allowed_weight = path.parent == self.weights and (path.name in ("init.pt", "last.pt", "best.pt")
                         or __import__("re").fullmatch(r"(?:epoch|best)_\d{3}\.pt", path.name))
        require(path in self.caches or allowed_weight, "torch.save escaped current weights/cache whitelist")
        match = __import__("re").fullmatch(r"(?:epoch|best)_(\d{3})\.pt", path.name)
        if match:
            require(isinstance(obj, dict) and obj.get("epoch") == int(match.group(1))
                    and 0 <= obj["epoch"] < self.epochs, "Numbered save epoch differs")
        return super().save(obj, destination, *args, **kwargs)


def execute(config, work, root, evidence):
    base = root / "run_package"
    sys.path.insert(0, str(base))
    require("study_common" not in sys.modules, "Legacy helper imported too early")
    legacy = load("pinned_stepmatched_worker", base / "study_worker.py")
    if config["mode"] == "preflight":
        from structure_check import cpu_preflight
        cpu_preflight(config, work, root, legacy)
        evidence["torch_device_binding"] = dict(physical_gpu=config["physical_gpu"], visible_devices=0, cpu_only=True)
        return
    original_install = legacy.install_v7
    holder = {}
    retention = dict(events=[], hook_restored=False, minimum_free_bytes=6 * 1024 ** 3)
    success = False

    def install(*args, **kwargs):
        require(not holder, "Repeated model environment installation")
        torch = original_install(*args, **kwargs)
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one CUDA device required")
        uid, scope = verify_runtime_uuid(getattr(torch.cuda.get_device_properties(0), "uuid", None), config["gpu_binding"]["uuid"])
        require("V100" in torch.cuda.get_device_name(0), "Expected V100")
        evidence["torch_device_binding"] = dict(uuid=uid, uuid_verification=scope, visible_devices=1,
                                                  physical_gpu=config["physical_gpu"], name=torch.cuda.get_device_name(0))
        holder["torch"] = torch
        if config["mode"] == "infer":
            from structure_check import describe_model
            trusted_load = torch.load
            def checked_load(*a, **kw):
                obj = trusted_load(*a, **kw)
                if a and isinstance(a[0], (str, Path)) and Path(a[0]).resolve() == Path(config["weight"]):
                    checkpoint_model = obj.get("ema") if obj.get("ema") is not None else obj.get("model")
                    architecture = work.parent.name.rsplit("_eval_", 1)[0]
                    architecture = architecture[6:] if architecture.startswith("smoke_") else architecture
                    architecture = architecture.split("_b80_s", 1)[0]
                    evidence["loaded_architecture"] = describe_model(checkpoint_model, architecture, root)
                return obj
            torch.load = checked_load
            holder["trusted_load"] = trusted_load

        versions = legacy.environment(torch)["versions"]
        require(all(versions.get(k) == value for k, value in protocol()["runtime_versions"].items()),
                "Runtime versions differ from accepted compact80 baseline")
        if config["mode"] == "train":
            import train
            from structure_check import describe_model
            old_model = train.Model
            holder["native_model"] = old_model
            holder["train_module"] = train
            def checked_model(*a, **kw):
                model = old_model(*a, **kw)
                report = describe_model(model, config["architecture"], root)
                dump(work / "architecture_evidence.json", report)
                return model
            train.Model = checked_model
            require(Path(train.__file__).resolve() == root / "source/train.py" and sha(train.__file__) == TRAIN_SHA,
                    "Wrong native train module")
            trace = StepTrace(config, work)
            holder["trace"] = trace
            old_step = torch.optim.SGD.step
            holder["old_step"] = old_step

            def step(optimizer, *args, **kwargs):
                frame = inspect.currentframe().f_back
                while frame is not None and not (frame.f_code.co_name == "train"
                        and Path(frame.f_code.co_filename).resolve() == root / "source/train.py"):
                    frame = frame.f_back
                require(frame is not None, "SGD step outside pinned training function")
                values = frame.f_locals
                context = {k: int(values[k]) for k in ("epoch", "i", "ni", "accumulate", "nb", "nw", "nbs", "total_batch_size")}
                context["actual_dataset_samples"] = len(values["dataset"])
                del frame, values
                trace.check(context)
                lrs = [float(group["lr"]) for group in optimizer.param_groups]
                result = old_step(optimizer, *args, **kwargs)
                trace.completed(context, lrs)
                return result

            torch.optim.SGD.step = step
            save = BoundedRetention(torch, work, retention, config["data"], config["epochs"])
            holder["save"] = save
            save.install()
        return torch

    legacy.install_v7 = install
    try:
        (legacy.train_v7 if config["mode"] == "train" else legacy.infer)(config)
        require(holder, "Worker did not install pinned runtime")
        success = True
    finally:
        legacy.install_v7 = original_install
        if "trusted_load" in holder:
            holder["torch"].load = holder["trusted_load"]
        if "native_model" in holder:
            holder["train_module"].Model = holder["native_model"]
        if "old_step" in holder:
            holder["torch"].optim.SGD.step = holder["old_step"]
            holder["trace"].restored = holder["torch"].optim.SGD.step is holder["old_step"]
        if "save" in holder:
            holder["save"].restore()
        try:
            if "trace" in holder:
                holder["trace"].close(success)
        finally:
            if config["mode"] == "train":
                retention.update(success=success, checkpoint_deletion_performed=False)
                dump(work / "worker_retention_audit.json", retention)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-package-sha256", required=True)
    args = parser.parse_args()
    verify_package(args.expected_package_sha256)
    require(sys.platform == "linux", "Linux laboratory execution only")
    config = read(regular(args.config))
    validate_environment(config)
    root, work = validate_config(config, args.config.absolute(), args.expected_package_sha256)
    evidence = dict(status="started", config_sha256=sha(args.config), package_sha256=args.expected_package_sha256,
                    started_unix_time=time.time(), actual_exitcode=1, unchanged_legacy_train_infer_functions=True,
                    environment_verified_before_torch=True)
    code = 1
    def interrupted(*args):
        raise KeyboardInterrupt("Terminated own worker")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        execute(config, work, root, evidence)
        code = 0
        evidence["status"] = "complete"
    except BaseException as exc:
        evidence.update(error=str(exc), traceback=traceback.format_exc(), status="failed")
        traceback.print_exc()
    finally:
        evidence.update(actual_exitcode=code, finished_unix_time=time.time())
        dump(work / "worker_execution.json", evidence)
    return code


if __name__ == "__main__":
    sys.exit(main())
