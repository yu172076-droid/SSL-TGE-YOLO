"""Run one pinned legacy YOLOv7 worker with isolated GPU and checkpoint retention.

The training/inference functions are imported unchanged from the replica's frozen
run_package. This wrapper only checks bindings and intercepts redundant numbered
torch.save calls. It never removes checkpoints or touches another study.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parent
MAPPING = {202: "2", 303: "3"}
POLICY = "suppress_numbered_epoch_and_best_v1"
MIN_FREE = 6 * 1024 ** 3
PINNED = {
    "study_worker.py": "71096b90e2662ac8c0c397af13e8eb5ef18b331bc740c32058a19ada13b28f39",
    "study_common.py": "142c50214d822977d445c18f51b798b2a618548c1c857bf5e6b03e23772274e0",
    "run_internal_study.py": "3d7179052d560032bdc0182145dcb9ad4318bdc887d404e63c0449398de49e88",
}
NUMBERED = re.compile(r"(?:epoch|best)_(\d{3})\.pt\Z")
RANK_NAMES = ("RANK", "WORLD_SIZE", "LOCAL_RANK")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def regular(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in [path] + list(path.parents)), "Symlink rejected: " + str(path))
    require(path.is_file(), "Missing regular file: " + str(path))
    return path


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_gpu_uuid(value):
    """Compare CUDA and NVML identities without relying on enumeration order."""
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, bytes):
        require(len(value) == 16, "Runtime GPU UUID bytes must contain exactly 16 bytes")
        return str(uuid.UUID(bytes=value))
    require(isinstance(value, str), "Unsupported GPU UUID representation")
    token = value[4:] if value[:4].lower() == "gpu-" else value
    try:
        return str(uuid.UUID(token))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError("Unparseable GPU UUID; physical identity was not established") from exc


def verify_runtime_uuid(runtime_uuid, parent_uuid):
    if runtime_uuid is None:
        return None, "unavailable_environment_binding_only"
    actual, expected = normalize_gpu_uuid(runtime_uuid), normalize_gpu_uuid(parent_uuid)
    require(actual == expected, "Visible CUDA GPU UUID differs from the parent NVML physical GPU")
    return "GPU-" + actual, "matched_parent_uuid"


def validate_environment(config):
    """Called before importing the legacy worker, torch, or project modules."""
    seed, physical = config.get("parallel_seed"), str(config.get("physical_gpu"))
    require(seed in MAPPING and physical == MAPPING[seed], "Only seed202/GPU2 and seed303/GPU3 are authorized")
    require(str(config.get("device")) == physical, "Native device differs")
    if config.get("mode") == "train":
        require(config.get("seed") == seed, "Training seed differs")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == physical, "CUDA_VISIBLE_DEVICES must select exactly the assigned physical GPU")
    require(os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", "CUDA_DEVICE_ORDER must be PCI_BUS_ID")
    require(not any(name in os.environ for name in RANK_NAMES), "DDP rank variables must be absent")
    require("torch" not in sys.modules, "Torch was imported before environment verification")
    binding = config.get("gpu_binding", {})
    require(str(binding.get("index")) == physical and binding.get("uuid", "").startswith("GPU-")
            and "V100" in binding.get("name", "") and binding.get("pci_bus_id"), "Missing assigned physical V100 identity")
    return dict(CUDA_VISIBLE_DEVICES=physical, CUDA_DEVICE_ORDER="PCI_BUS_ID", rank_vars_absent=True,
                rank_variable_names=list(RANK_NAMES), verified_before_torch_import=True)


def validate_config(config, config_path, package, expected):
    require(config.get("backend") == "v7" and config.get("mode") in ("train", "infer"), "Only v7 train/infer are allowed")
    require(config.get("parallel_manifest_sha256") == expected, "Config/new package manifest differs")
    require(config.get("snapshot_policy") == POLICY and config.get("minimum_free_bytes") == MIN_FREE, "Checkpoint/disk reserve policy differs")
    require(config.get("legacy_worker_sha256") == PINNED["study_worker.py"]
            and config.get("legacy_runner_sha256") == PINNED["run_internal_study.py"], "Legacy worker/runner pin differs")
    require(config.get("worker_sha256") == sha(package / "worker_entry.py")
            and config.get("runner_sha256") == sha(package / "run_queue.py"), "Parallel source bindings differ")
    study, work, base = [Path(config[k]).absolute() for k in ("study", "work", "base_package")]
    for path in (study, work, base):
        require(path == path.resolve() and not path.is_symlink(), "Study/work/package must be real paths")
    require(base == study / "run_package" and study / "jobs" in work.parents
            and work.name.startswith("attempt_") and work.is_dir(), "Worker must belong to this private attempt")
    require(config_path == work / "worker_config.json", "Config is not inside its own attempt")
    require(config.get("study_protocol_sha256") == sha(regular(study / "study.json"))
            and config.get("dataset_manifest_sha256") == sha(regular(study / "manifest.json")), "Frozen study/data binding differs")
    metadata = read(regular(study / "replica_metadata.json"))
    require(metadata.get("schema") == "parallel_b144_replica_v1" and metadata.get("seed") == config["parallel_seed"]
            and str(metadata.get("physical_gpu")) == config["physical_gpu"]
            and metadata.get("package_manifest_sha256") == expected
            and Path(metadata["replica_study"]).absolute() == study, "Private replica ownership differs")
    for name, digest in PINNED.items():
        require(sha(regular(base / name)) == digest, "Pinned legacy source changed: " + name)
    if config["mode"] == "train":
        require(config.get("budget") == 144 and config.get("workers") == 4, "Only budget144/workers4 are allowed")
        require(config.get("architecture") in ("teacher", "compact", "wide")
                and config.get("method") in ("supervised", "ssl")
                and not (config["architecture"] == "teacher" and config["method"] != "supervised"), "Undeclared train job")
        require(config.get("epochs") == 300 and config.get("batch") == (4 if config["architecture"] == "teacher" else 8), "Epochs/batch differ")
    return study, work, base


class SaveRetention:
    """Only suppress epoch/best_NNN for the exact current weights directory."""
    def __init__(self, torch, work, report):
        self.torch, self.work, self.report = torch, Path(work).resolve(), report
        self.weights = self.work / "train" / "weights"
        self.original = torch.save
        self.installed = False
        source = inspect.getsourcefile(self.original)
        self.report.update(original_torch_save_function=self.original.__module__ + "." + self.original.__qualname__,
                           original_torch_save_source_file=str(source) if source else None,
                           original_torch_save_source_sha256=sha(source) if source and Path(source).is_file() else None,
                           original_torch_save_function_source_sha256=hashlib.sha256(inspect.getsource(self.original).encode("utf-8")).hexdigest(),
                           hook_restored=False)

    def install(self):
        require(not self.installed, "Save hook installed more than once")
        self.torch.save = self.save
        self.installed = True

    def restore(self):
        if self.installed:
            self.torch.save = self.original
            self.report["hook_restored"] = self.torch.save is self.original
            self.installed = False

    def save(self, obj, destination, *args, **kwargs):
        path = None
        if isinstance(destination, (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(destination)).absolute()
            require(not any(p.is_symlink() for p in [path] + list(path.parents)), "Checkpoint destination symlink rejected")
        epoch = obj.get("epoch") if isinstance(obj, dict) else None
        match = NUMBERED.fullmatch(path.name) if path is not None and path.parent == self.weights else None
        event = dict(unix_time=time.time(), monotonic_ns=time.monotonic_ns(),
                     path=str(path) if path else "<file-like>",
                     relative_path=path.relative_to(self.work).as_posix() if path is not None and self.work in path.parents else None,
                     checkpoint_epoch=epoch if type(epoch) is int else None,
                     epoch=int(match.group(1)) if match else None)
        if match and type(epoch) is int and epoch == int(match.group(1)) and 0 <= epoch < 300:
            event.update(action="suppress", success=True, reason="exact_current_weights_numbered_checkpoint_with_matching_epoch")
            self.report["events"].append(event)
            return None
        try:
            free = shutil.disk_usage(self.work).free
            event["free_bytes_before_save"] = free
            require(free >= MIN_FREE, "Free disk below 6 GiB; checkpoint save stopped without deleting files")
            result = self.original(obj, destination, *args, **kwargs)
            event.update(action="save", success=True, free_bytes=free, reason="original_save_preserved", bytes=path.stat().st_size if path and path.is_file() else None)
            self.report["events"].append(event)
            return result
        except BaseException as exc:
            event.update(action="failure", success=False, error_type=type(exc).__name__, error=str(exc))
            self.report["events"].append(event)
            raise


def execute(config, work, base, evidence, retention):
    sys.path.insert(0, str(base))
    # Do not accidentally reuse a common helper imported from a different study.
    require("study_common" not in sys.modules, "study_common was already imported")
    worker = load("pinned_parallel_worker", base / "study_worker.py")
    common = sys.modules.get("study_common")
    require(common is not None and Path(common.__file__).resolve() == base / "study_common.py", "Legacy common module came from another package")
    original_install = worker.install_v7
    holder = []

    def checked_install(*args, **kwargs):
        require(not holder, "Legacy worker attempted repeated model-environment setup")
        torch = original_install(*args, **kwargs)
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Expected exactly one visible CUDA device")
        name = torch.cuda.get_device_name(0)
        require("V100" in name, "Visible CUDA device is not the assigned V100")
        properties = torch.cuda.get_device_properties(0)
        runtime_uuid = getattr(properties, "uuid", None)
        runtime_uuid, uuid_verification = verify_runtime_uuid(runtime_uuid, config["gpu_binding"]["uuid"])
        evidence["torch_device_binding"] = dict(visible_device_count=1, selected_device_index=0,
                name=name, physical_gpu=config["physical_gpu"], uuid=runtime_uuid,
                uuid_verification=uuid_verification,
                parent_gpu_uuid=config["gpu_binding"]["uuid"],
                binding_scope="physical identity queried by parent; environment set before import; one CUDA device visible")
        if config["mode"] == "train":
            hook = SaveRetention(torch, work, retention)
            hook.install()
            holder.append(hook)
        else:
            holder.append(None)
        return torch

    worker.install_v7 = checked_install
    try:
        (worker.train_v7 if config["mode"] == "train" else worker.infer)(config)
        require(len(holder) == 1, "Original worker did not initialize its checked model environment exactly once")
    finally:
        worker.install_v7 = original_install
        if holder and holder[0] is not None:
            holder[0].restore()
            evidence["torch_save_hook_restored"] = retention["hook_restored"]
        else:
            evidence["torch_save_hook_restored"] = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    require(platform.system() == "Linux", "Execution requires the laboratory Linux server")
    expected, package = args.expected_manifest_sha256.lower(), args.package.absolute()
    require(package == ROOT and not package.is_symlink(), "The selected package is not this worker package")
    load("parallel_worker_verifier", regular(package / "verify_package.py")).verify(package, expected)
    config_path = regular(args.config)
    config = read(config_path)
    environment = validate_environment(config)
    study, work, base = validate_config(config, config_path, package, expected)
    execution_path, retention_path = work / "worker_execution.json", work / "worker_retention_audit.json"
    require(not execution_path.exists() and not retention_path.exists(), "Use a new attempt; worker evidence already exists")
    evidence = dict(schema="parallel_b144_worker_execution_v1", status="started", actual_exitcode=None,
        pid=os.getpid(), started_unix_time=time.time(), mode=config["mode"], config_sha256=sha(config_path),
        package_manifest_sha256=expected, study=str(study), work=str(work), parallel_seed=config["parallel_seed"],
        physical_gpu=config["physical_gpu"], gpu_binding=config["gpu_binding"], environment_binding=environment,
        source_bindings=dict(legacy_worker_sha256=PINNED["study_worker.py"], legacy_runner_sha256=PINNED["run_internal_study.py"],
                             legacy_common_sha256=PINNED["study_common.py"], wrapper_sha256=sha(__file__)),
        original_training_and_inference_functions_unchanged=True, checkpoint_deletion_performed=False)
    retention = dict(schema="parallel_b144_worker_retention_v1", policy=POLICY, current_work=str(work),
                     weights_root=str(work / "train/weights"), minimum_free_bytes=MIN_FREE, events=[],
                     status="started", actual_exitcode=None, checkpoint_deletion_performed=False, hook_restored=False)
    code = 1
    def interrupted(signum, frame):
        raise KeyboardInterrupt("signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    try:
        execute(config, work, base, evidence, retention)
        code = 0
        evidence["status"] = retention["status"] = "completed"
    except BaseException as exc:
        code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        evidence.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        retention.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
    finally:
        evidence.update(actual_exitcode=code, finished_unix_time=time.time())
        evidence["duration_seconds"] = evidence["finished_unix_time"] - evidence["started_unix_time"]
        retention.update(actual_exitcode=code, finished_unix_time=time.time(),
                         saved_count=sum(e["action"] == "save" for e in retention["events"]),
                         suppressed_count=sum(e["action"] == "suppress" for e in retention["events"]))
        dump(execution_path, evidence)
        if config["mode"] == "train":
            dump(retention_path, retention)
    return code


if __name__ == "__main__":
    sys.exit(main())
