"""One isolated 144-label seed queue; display-only unless --execute is supplied."""
import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
GIB = 1024 ** 3
START_FREE = 12 * GIB
STOP_FREE = 6 * GIB
STUDY_SHA = "2e5b071a355e57e7b36bc4b17268c245eddb31b90dcc41b013450bea5aabfcd0"
DATA_SHA = "25ad1cac4bc842090b4044317654d3ac136f6d6ddc50c7b6ec61534394e7dc4d"
MAPPING = {202: "2", 303: "3"}
POLICY = "suppress_numbered_epoch_and_best_v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def regular(path):
    path = Path(path).absolute()
    for parent in list(path.parents) + [path]:
        require(not parent.is_symlink(), "Symlink rejected: " + str(parent))
    require(path.is_file(), "Missing regular file: " + str(path))
    return path


def artifact(base, relative):
    rel = Path(relative)
    require(not rel.is_absolute() and ".." not in rel.parts and "\\" not in relative, "Unsafe artifact path")
    return regular(base / rel)


class Lock:
    def __init__(self, path):
        self.path, self.handle = path, None
    def __enter__(self):
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        require(not self.path.is_symlink(), "Lock symlink refused")
        self.handle = self.path.open("a+b")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise RuntimeError("This private study/GPU queue is already locked; no duplicate launch")
        return self
    def __exit__(self, *args):
        if self.handle:
            self.handle.close()


def gpu_check(gpu, idle=True):
    result = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,utilization.gpu",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15, check=True)
    rows = list(csv.reader(result.stdout.strip().splitlines()))
    require(len(rows) == 1 and len(rows[0]) == 7, "Unexpected GPU identity response")
    index, uid, bus, name, total, used, util = [x.strip() for x in rows[0]]
    require(index == gpu and "V100" in name and uid.startswith("GPU-"), "Expected assigned single V100 only")
    if idle:
        require(float(used) <= 200 and float(util) <= 5, "Assigned GPU is busy; do not stop or take over its processes")
    return dict(index=index, uuid=uid, pci_bus_id=bus, name=name, memory_total_MiB=float(total),
                memory_used_MiB=float(used), utilization_percent=float(util), observed_unix_time=time.time())


def free_check(path, threshold=START_FREE):
    free = shutil.disk_usage(path).free
    require(free >= threshold, "Free disk {} below fixed safety reserve {} bytes; no automatic deletion".format(free, threshold))
    return free


def inherit_gates(source, target):
    """Copy original gate records as references, never launch/relabel old smoke jobs."""
    names = ["preflight_v7"] + ["smoke_{}_b80_s101_supervised".format(a) for a in ("teacher", "compact", "wide")]
    evidence = []
    for name in names:
        folder = source / "jobs" / name
        original = regular(folder / "completed.json")
        record = read(original)
        work = Path(record["work"])
        require(work.is_absolute() and source in work.resolve().parents, "Inherited gate work outside original study")
        checked = []
        for relative, value in record["output_hashes"].items():
            file = artifact(folder, relative)
            require(sha(file) == value, "Inherited preflight/smoke artifact changed: " + str(file))
            checked.append(dict(path=str(file), sha256=value, bytes=file.stat().st_size))
        if name != "preflight_v7":
            require(record["input"].get("epochs") == 2 and record["input"].get("seed") == 101, "Inherited smoke setting differs")
            require(sha(work / "train/weights/last.pt") in record["output_hashes"].values(), "Inherited smoke last.pt not verified")
        destination = target / "jobs" / name / "completed.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            require(sha(destination) == sha(original), "Inherited gate copy changed")
        else:
            shutil.copy2(original, destination)
        require(sha(destination) == sha(original), "Inherited gate copy mismatch")
        evidence.append(dict(job=name, original_completed=str(original), completed_sha256=sha(original),
                             original_work=str(work), reference_only=True, new_training_run=False, verified_artifacts=checked))
    existing = target / "inherited_gate_evidence.json"
    body = dict(schema="parallel_b144_inherited_gates_v1", source_study=str(source), records=evidence,
                purpose="Read-only inheritance of already verified preflight and smoke; excluded from new formal job count")
    if existing.exists():
        require(read(existing) == body, "Inherited gate evidence differs; preserve and inspect")
    else:
        dump(existing, body)


def stop(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def run_child(command, work, env, study):
    checks = []
    process = None
    code = None
    failure = None
    try:
        with (work / "output.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=work, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while True:
                checks.append(dict(unix_time=time.time(), free_bytes=free_check(study, STOP_FREE)))
                try:
                    code = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    pass
    except BaseException as exc:
        failure = dict(type=type(exc).__name__, message=str(exc))
        if process is not None:
            stop(process)
            code = process.returncode
        raise
    finally:
        dump(work / "parent_resource_checks.json", dict(check_interval_s=30, start_minimum_free_bytes=START_FREE,
             runtime_minimum_free_bytes=STOP_FREE, samples=checks, error=failure, child_pid=process.pid if process else None,
             actual_child_exitcode=code))
        dump(work / "exit_status.json", dict(returncode=code, unix_time=time.time(), parent_error=failure))
    return code


def make_study(base, root, options, manifest_sha, gpu_binding):
    class IsolatedStudy(base.Study):
        def task(self, identifier, config):
            require(config.get("backend") == "v7" and config.get("mode") in ("train", "infer"), "Only declared v7 jobs allowed")
            config = dict(config, study=str(self.root), study_protocol_sha256=sha(self.root / "study.json"),
                worker_sha256=sha(ROOT / "worker_entry.py"), runner_sha256=sha(__file__),
                legacy_worker_sha256=sha(self.root / "run_package/study_worker.py"),
                legacy_runner_sha256=sha(self.root / "run_package/run_internal_study.py"),
                base_package=str(self.root / "run_package"), parallel_seed=options.parallel_seed,
                physical_gpu=options.device, gpu_binding=gpu_binding, snapshot_policy=POLICY,
                minimum_free_bytes=STOP_FREE, parallel_manifest_sha256=manifest_sha)
            folder = self.root / "jobs" / identifier
            completed = folder / "completed.json"
            if completed.is_file():
                saved = read(completed)
                # The observed utilization timestamp changes between launches;
                # the physical identity is immutable, the prior observation is retained.
                proposed = dict(config)
                proposed["gpu_binding"] = saved["input"]["gpu_binding"]
                require(saved["input"] == proposed and saved["input"]["gpu_binding"]["uuid"] == gpu_binding["uuid"], "Completed job protocol/GPU differs")
                for path, value in saved["output_hashes"].items():
                    require(sha(artifact(folder, path)) == value, "Completed artifact changed")
                saved_work = Path(saved["work"]).absolute()
                require(saved_work == saved_work.resolve() and folder in saved_work.parents
                        and self.root in saved_work.parents, "Formal work escaped its private job")
                return Path(saved["work"])
            free_check(self.root)
            gpu_check(options.device, idle=True)
            folder.mkdir(parents=True, exist_ok=True)
            attempts = sorted(folder.glob("attempt_*"))
            require(not attempts or self.args.retry_failed, "Failed/partial attempt preserved; inspect then explicitly --retry-failed")
            work = folder / ("attempt_{:02d}".format(len(attempts) + 1))
            work.mkdir(exist_ok=False)
            full = dict(config, work=str(work))
            dump(work / "worker_config.json", full)
            env = os.environ.copy()
            for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
                env.pop(name, None)
            env.update(CUDA_VISIBLE_DEVICES=options.device, CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONDONTWRITEBYTECODE="1",
                PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", WANDB_MODE="disabled", YOLO_AUTOINSTALL="false",
                MPLCONFIGDIR=str(work / "plot_config"), YOLO_CONFIG_DIR=str(work / "ultralytics_settings"),
                OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
            command = [sys.executable, "-B", str(ROOT / "worker_entry.py"), "--config", str(work / "worker_config.json"),
                       "--package", str(ROOT), "--expected-manifest-sha256", manifest_sha]
            dump(work / "started.json", dict(command=command, unix_time=time.time(), input=config,
                 visible_device=options.device, cuda_device_order=env["CUDA_DEVICE_ORDER"]))
            print("START", identifier, "physical GPU", options.device, "log:", work / "output.log", flush=True)
            code = run_child(command, work, env, self.root)
            require(code == 0, "Worker failed; inspect preserved attempt: " + identifier)
            required = {"train": ["train/weights/last.pt", "train/weights/best.pt", "train/weights/init.pt", "training_settings.json", "seed_applied.json", "worker_retention_audit.json"],
                        "infer": ["predictions.json", "inference_metadata.json"]}[config["mode"]] + ["worker_execution.json", "parent_resource_checks.json", "exit_status.json"]
            hashes = {p.relative_to(folder).as_posix(): sha(p) for p in [regular(work / r) for r in required]}
            dump(completed, dict(input=config, work=str(work), output_hashes=hashes))
            print("DONE", identifier, flush=True)
            return work
    return IsolatedStudy(root, options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--seed", type=int, choices=(202, 303), required=True)
    parser.add_argument("--gpu", choices=("2", "3"), required=True)
    parser.add_argument("--expected-package-sha256", required=True)
    parser.add_argument("--return-root", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    require(MAPPING[args.seed] == args.gpu, "Only seed202->GPU2 and seed303->GPU3 are authorized")
    load("parallel_verifier", ROOT / "verify_package.py").verify(ROOT, args.expected_package_sha256)
    jobs = [dict(architecture=a, budget=144, seed=args.seed, method=m, epochs=300)
            for a, m in [("teacher", "supervised"), ("compact", "supervised"), ("compact", "ssl"), ("wide", "supervised"), ("wide", "ssl")]]
    print(json.dumps(dict(seed=args.seed, physical_gpu=args.gpu, formal_jobs=5, queue=jobs,
        workers=4, data_manifest_sha256=DATA_SHA, checkpoint_policy=POLICY,
        independent_generalization_test=False, execute=args.execute), ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        print("PLAN ONLY; no output or GPU access", flush=True)
        return 0
    require(platform.system() == "Linux" and args.return_root is not None, "Execution requires Linux and --return-root")
    root = args.study.absolute()
    require(not root.is_symlink() and root.resolve() == root, "Private study symlink refused")
    metadata = read(regular(root / "replica_metadata.json"))
    complete = regular(root / "prepare_completed.json")
    side = regular(root / "prepare_completed.sha256").read_text("ascii").split()[0]
    require(sha(complete) == side, "Preparation report SHA differs")
    preparation = read(complete)
    require(preparation.get("schema") == "parallel_b144_prepare_completed_v1"
            and preparation.get("state") == "complete"
            and preparation.get("replica_metadata_sha256") == sha(root / "replica_metadata.json")
            and preparation.get("seed") == args.seed and str(preparation.get("physical_gpu")) == args.gpu
            and preparation.get("replica_study") == str(root)
            and preparation.get("package_manifest_sha256") == args.expected_package_sha256
            and preparation.get("frozen_file_count") == 3238
            and preparation.get("copied_file_count") == 3247
            and preparation.get("no_training_started") is True
            and preparation.get("original_project_not_written") is True,
            "Preparation completion does not bind this verified replica")
    require(metadata["seed"] == args.seed and str(metadata["physical_gpu"]) == args.gpu
            and metadata["package_manifest_sha256"] == args.expected_package_sha256, "Prepared replica binding differs")
    require(Path(metadata["replica_study"]).resolve() == root, "Replica was moved; completed absolute paths must not be silently reused")
    require(sha(root / "study.json") == STUDY_SHA and sha(root / "manifest.json") == DATA_SHA, "Frozen input identity differs")
    source = Path(metadata["source_study"]).resolve()
    returns = args.return_root.absolute()
    original = Path(read(root / "study.json")["source_project"]).resolve()
    for protected in (source, root, ROOT, original):
        require(returns != protected and protected not in returns.parents and returns not in protected.parents, "Return location overlaps source/private study/project/package")
    require(not returns.is_symlink(), "Return root symlink refused")
    base_package = root / "run_package"
    for name, value in read(ROOT / "expected_base_package.json").items():
        require(sha(artifact(base_package, name)) == value, "Base package changed")
    options = SimpleNamespace(device=args.gpu, parallel_seed=args.seed, workers=4, retry_failed=args.retry_failed, v8_python=None)
    exitcode = 1
    def interrupt(signum, frame):
        raise KeyboardInterrupt("signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupt)
    with Lock(root / "parallel_queue.lock"), Lock(root.parent / ".parallel_b144_locks" / ("gpu" + args.gpu + ".lock")):
        free_check(root)
        binding = gpu_check(args.gpu, idle=True)
        inherit_gates(source, root)
        sys.path.insert(0, str(base_package))
        base = load("pinned_parallel_study", base_package / "run_internal_study.py")
        study = make_study(base, root, options, args.expected_package_sha256, binding)
        audit = load("parallel_audit_export", ROOT / "audit_export.py")
        returns.mkdir(parents=True, exist_ok=True)
        invocation = root / ("parallel_invocation_" + str(time.time_ns()) + ".json")
        status = dict(seed=args.seed, physical_gpu=args.gpu, gpu_binding=binding, command=sys.argv, pid=os.getpid(),
                      manifest_sha256=args.expected_package_sha256, queue=jobs, state="running", started_unix_time=time.time())
        dump(invocation, status)
        audits = []
        try:
            for position, item in enumerate(jobs, start=1):
                study.training(**item)
                audits.append(audit.audit_training(root, item))
                study.summaries()
                dump(root / "parallel_progress.json", dict(seed=args.seed, physical_gpu=args.gpu,
                    selected_jobs=5, verified_jobs=position, audits=audits, last_verified_job=audit.identifier(item),
                    package_manifest_sha256=args.expected_package_sha256, unix_time=time.time()))
                if position in (1, 3, 5):
                    if position == 5:
                        status.update(state="complete", verified_jobs=5, finished_unix_time=time.time(), exitcode=0)
                        dump(invocation, status)
                    out = returns / ("b144_s{}_{}_of5_{}".format(args.seed, position, time.time_ns()))
                    audit.export(root, out, base_package, newpackage=ROOT)
                    print("MILESTONE_RETURN", out, flush=True)
            require(sha(root / "study.json") == STUDY_SHA and sha(root / "manifest.json") == DATA_SHA, "Post-run freeze changed")
            status.update(state="complete", verified_jobs=5)
            exitcode = 0
        except BaseException as exc:
            status.update(state="failed_or_interrupted", error=str(exc), traceback=traceback.format_exc(), verified_jobs=len(audits))
            raise
        finally:
            status.update(finished_unix_time=time.time(), exitcode=exitcode)
            dump(invocation, status)
    print("PARALLEL_B144_COMPLETE", args.seed, "GPU", args.gpu, flush=True)
    return exitcode


if __name__ == "__main__":
    sys.exit(main())
