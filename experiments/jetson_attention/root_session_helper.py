"""Ordinary-user compact FP16 one-native-image-at-a-time root-sampled pilot.

The historical B3 preprocessing, forward/NMS and timing functions are imported
unchanged after independent hash verification. No sudo or telemetry child runs
here. Power/clock data are joined later; exit 0 is not formal experimental pass.
"""
import argparse
import contextlib
import csv
import datetime
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import stat
import statistics
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parent
OLD_MANIFEST_SHA = "84cadaf73d7073d1fac479ef8795571b2e33534fbf58e8b3795f2ae674fea510"
OLD_DRIVER_SHA = "48f7861f3054b34295894e1c9e8f67aae0dcaf707d487ba6f5131f3c3f17c938"
OLD_TELEMETRY_SHA = "2468ee031acb08e55f4d421ffa4f11e95b753f5c9d06f66033ceb2983f5249b9"
GIB = 1024 ** 3
PROTOCOL = "native_image_stream_one_at_a_time_v1"
ROOT_FRESH_NS = 3500000000
VM_COUNTERS = ("pswpin", "pswpout", "pgmajfault")
PINNED_RUNTIME_FLAGS = {"torch_num_threads": 6, "opencv_num_threads": 1,
                        "cudnn_benchmark": False, "cudnn_deterministic": False,
                        "cudnn_allow_tf32": True, "cuda_matmul_allow_tf32": False}
RUNTIME_FLAGS_OLD_REPORT_SHA = "5698b5bb035b2ea03a9c7d51005996a6290603f98ff44bbb77b5a80bca92d22e"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_native(row, inputs, cv2):
    """Decode exactly one original-size BGR uint8 image; never resize/cache it."""
    image = cv2.imread(str(inputs / row["edge_image"]), cv2.IMREAD_COLOR)
    require(image is not None and image.dtype.name == "uint8",
            "Native image decode failed: " + row["edge_image"])
    expected_shape = [int(row["height"]), int(row["width"]), 3]
    require(list(image.shape) == expected_shape, "Native dimensions changed")
    require(int(image.nbytes) == expected_shape[0] * expected_shape[1] * 3,
            "Decoded native byte count differs")
    return image


def sequential_input_check(rows, inputs, cv2, report, monitor, max_seconds=300):
    """Validate every decode and release it before opening the next image."""
    maximum = max(int(row["width"]) * int(row["height"]) * 3 for row in rows)
    decoded = {"input_protocol": PROTOCOL,
               "policy": "Read and decode one native image, validate, release; no predecoded image collection",
               "all_images_predecoded": False, "max_resident_native_images": 1,
               "max_native_image_bytes": maximum, "max_live_native_bytes": 0,
               "preflight_verified_images": 0, "preflight_cumulative_native_bytes": 0,
               "cumulative_bytes_are_not_simultaneous_resident_bytes": True,
               "native_images_retained_after_preflight": 0, "preflight_image_order": []}
    report["decoded_image_RAM"] = decoded
    started_ns = time.monotonic_ns()
    for row in rows:
        if (time.monotonic_ns() - started_ns) / 1e9 >= max_seconds:
            raise PhaseTimeout("Sequential input validation exceeded fixed 300-second cap")
        monitor.check_failure()
        image = None
        try:
            image = read_native(row, inputs, cv2)
            decoded["max_live_native_bytes"] = max(decoded["max_live_native_bytes"], int(image.nbytes))
            decoded["preflight_cumulative_native_bytes"] += int(image.nbytes)
            decoded["preflight_verified_images"] += 1
            decoded["preflight_image_order"].append(row["id"])
        finally:
            del image
        monitor.tick("sequential_input_check", minimum=3 * GIB, force=True)
    require(decoded["max_live_native_bytes"] == maximum, "Maximum native image size differs")
    return decoded


def run_stream_window(name, min_frames, min_seconds, max_seconds, rows, inputs, cv2,
                      infer, report, records, predictions, progress, monitor):
    """Whole-window scope includes input read/decode, the unchanged call and observer checks."""
    start_ns = time.monotonic_ns()
    window = {"start_monotonic_ns": start_ns, "end_monotonic_ns": None,
              "min_frames": min_frames, "min_seconds": min_seconds, "complete": False,
              "input_protocol": PROTOCOL, "max_resident_native_images": 1}
    report["windows"][name] = window
    progress(name + " started: minimum {} images AND {:.0f}s; one native image at a time".format(min_frames, min_seconds))
    last_progress = start_ns
    try:
        while True:
            elapsed = (time.monotonic_ns() - start_ns) / 1e9
            if elapsed >= max_seconds:
                raise PhaseTimeout(name + " exceeded maximum wall duration")
            if len(records) >= min_frames and elapsed >= min_seconds:
                break
            monitor.check_failure()
            index = len(records) % len(rows)
            row = rows[index]
            image = None
            pipeline_start_ns = time.monotonic_ns()
            input_read_start_ns = pipeline_start_ns
            try:
                image = read_native(row, inputs, cv2)
                input_read_end_ns = time.monotonic_ns()
                native_bytes = int(image.nbytes)
                record, values = infer(image, row)
            finally:
                # No original array remains between iterations, even after an error.
                del image
                pipeline_end_ns = time.monotonic_ns()
            record.update(iteration=len(records) + 1, image_order=index, cycle=len(records) // len(rows),
                          pipeline_start_monotonic_ns=pipeline_start_ns,
                          pipeline_end_monotonic_ns=pipeline_end_ns,
                          input_read_start_monotonic_ns=input_read_start_ns,
                          input_read_end_monotonic_ns=input_read_end_ns,
                          input_read_decode_ms=(input_read_end_ns - input_read_start_ns) / 1e6,
                          pipeline_ms=(pipeline_end_ns - pipeline_start_ns) / 1e6,
                          native_image_bytes=native_bytes)
            records.append(record)
            if predictions is not None:
                predictions[row["id"]] = {"image_id": row["id"], "image": row["edge_image"],
                    "image_sha256": row["image_sha256"], "last_measurement_iteration": len(records),
                    "detections_xyxy_conf_class": values}
            # Retain a completed original one_frame result before surfacing an observer latch.
            try:
                monitor.check_failure()
            except Exception:
                report["completed_frame_retained_after_observer_latch"] = True
                raise
            stamp = time.monotonic_ns()
            if stamp - last_progress >= 30 * 1000000000:
                progress(name + ": {} images, {:.1f}s".format(len(records), (stamp - start_ns) / 1e9))
                last_progress = stamp
        window["complete"] = True
    finally:
        window["end_monotonic_ns"] = time.monotonic_ns()
        window["frames"] = len(records)
        window["duration_s"] = (window["end_monotonic_ns"] - start_ns) / 1e9
        progress(name + " ended: {} images, {:.2f}s, complete={}".format(len(records), window["duration_s"], window["complete"]))


def latency_summary(old, records):
    result = old.latency_summary(records)
    if not records:
        return result
    for key in ("input_read_decode_ms", "pipeline_ms"):
        values = sorted(record[key] for record in records)
        require(all(math.isfinite(value) and value >= 0 for value in values), "Invalid stream timing")
        def percentile(p):
            position = (len(values) - 1) * p
            low, high = int(math.floor(position)), int(math.ceil(position))
            return values[low] + (values[high] - values[low]) * (position - low)
        result[key] = {"mean": statistics.mean(values), "median": statistics.median(values),
                       "p90": percentile(0.90), "p95": percentile(0.95), "p99": percentile(0.99),
                       "min": min(values), "max": max(values)}
    result["whole_window_throughput_note"] = "Latency reciprocals are not N/complete stream window; use image_window_throughput_per_s only"
    return result


def save_csv(path, records):
    fields = ["iteration", "image_id", "image", "image_order", "cycle", "start_monotonic_ns",
              "end_monotonic_ns", "processing_ms", "forward_cuda_event_ms", "detection_count",
              "validation_and_record_conversion_ms", "all_outputs_finite",
              "pipeline_start_monotonic_ns", "pipeline_end_monotonic_ns",
              "input_read_start_monotonic_ns", "input_read_end_monotonic_ns",
              "input_read_decode_ms", "pipeline_ms", "native_image_bytes"]
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


class MemoryGate(Exception):
    pass


class ResourceReview(Exception):
    pass


class RootSessionError(Exception):
    pass


class PhaseTimeout(Exception):
    pass


def owned_json(path):
    """Read one atomic root-owned regular JSON file without following symlinks."""
    descriptor = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if metadata.st_uid != 0 or not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o022:
            raise RootSessionError("Root session JSON ownership/mode invalid: " + str(path))
        raw = stream.read(1048577)
    if len(raw) > 1048576:
        raise RootSessionError("Root session JSON exceeds finite read bound")
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


class RootSession:
    def __init__(self, path, expected, boot_id):
        self.path = path
        self.expected = expected
        self.boot_id = boot_id
        metadata = path.stat()
        if path.is_symlink() or metadata.st_uid != 0 or metadata.st_mode & 0o022:
            raise RootSessionError("Root session directory must be owned by root and not group/other writable")
        self.ready, self.ready_sha = owned_json(path / "ready.json")
        ready = self.ready
        required = (ready.get("boot_id") == boot_id and ready.get("manifest_sha256") == expected
                    and ready.get("euid") == 0 and isinstance(ready.get("session_id"), str)
                    and len(ready["session_id"]) >= 16 and isinstance(ready.get("sampler_pid"), int)
                    and ready["sampler_pid"] > 1 and isinstance(ready.get("ready_monotonic_ns"), int)
                    and 0 < ready["ready_monotonic_ns"] <= time.monotonic_ns())
        if not required:
            raise RootSessionError("Root ready identity/boot/package binding invalid")
        if ready.get("script_sha256") != sha(ROOT / "root_sampler.py"):
            raise RootSessionError("Root sampler script differs from this frozen package")
        if Path(ready.get("output", "")).resolve() != path:
            raise RootSessionError("Root ready output path differs")
        if ready.get("mode_query_ok") is not True or ready.get("clock_query_ok") is not True:
            raise RootSessionError("Root sampler mode/clock readiness not confirmed")
        self.check()

    def check(self):
        live, live_sha = owned_json(self.path / "live.json")
        now = time.monotonic_ns()
        for key, value in (("session_id", self.ready["session_id"]), ("boot_id", self.boot_id),
                           ("sampler_pid", self.ready["sampler_pid"]), ("euid", 0),
                           ("manifest_sha256", self.expected)):
            if live.get(key) != value:
                raise RootSessionError("Root live identity mismatch: " + key)
        for key in ("last_tegrastats_monotonic_ns", "last_sysfs_monotonic_ns", "last_power_tegrastats_monotonic_ns"):
            stamp = live.get(key)
            if not isinstance(stamp, int) or stamp <= 0 or not 0 <= now - stamp <= ROOT_FRESH_NS:
                raise RootSessionError("Root sampling is stale/missing: " + key)
        if (live.get("ready_written") is not True or live.get("sysfs_sample_count", 0) < 1
                or live.get("power_line_count", 0) < 1
                or live.get("last_sysfs_valid_gpu_emc") is not True
                or live.get("last_sysfs_valid_vdd_pair") is not True
                or live.get("sampler_status") != "collecting"):
            raise RootSessionError("Root sampler is not actively ready")
        return {"checked_monotonic_ns": now, "live_sha256": live_sha, "live": live}


def process_faults(raw):
    # After the final ')' come state=field3, ..., minflt=field10, majflt=field12.
    tail = raw[raw.rfind(")") + 1:].split()
    require(len(tail) >= 10, "Malformed /proc/self/stat")
    return {"minflt": int(tail[7]), "majflt": int(tail[9])}


def memory_sample(phase):
    memory_raw = Path("/proc/meminfo").read_text(encoding="ascii")
    vm_raw = Path("/proc/vmstat").read_text(encoding="ascii")
    status_raw = Path("/proc/self/status").read_text(encoding="ascii")
    stat_raw = Path("/proc/self/stat").read_text(encoding="ascii")
    memory, process, counters = {}, {}, {}
    for line in memory_raw.splitlines():
        parts = line.replace(":", "").split()
        if len(parts) >= 2 and parts[1].isdigit():
            memory[parts[0] + "_bytes"] = int(parts[1]) * (1024 if len(parts) > 2 and parts[2] == "kB" else 1)
    for line in status_raw.splitlines():
        parts = line.replace(":", "").split()
        if len(parts) >= 2 and parts[0] in ("VmRSS", "VmHWM", "VmSwap"):
            process[parts[0] + "_bytes"] = int(parts[1]) * 1024
    for line in vm_raw.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in VM_COUNTERS:
            counters[parts[0]] = int(parts[1])
    return {"phase": phase, "monotonic_ns": time.monotonic_ns(), "utc": utc(),
            "proc_meminfo": memory, "process_status": process,
            "process_faults": process_faults(stat_raw), "vm_counters": counters,
            "raw": {"meminfo": memory_raw, "vmstat": vm_raw, "self_status": status_raw, "self_stat": stat_raw}}


class Monitor:
    """One-second observer; workers latch failures, main owns stopping CUDA work."""
    def __init__(self, root_session, report):
        self.root = root_session
        self.report = report
        self.samples = []
        self.last_interval_sample = None
        self.swap_positive_intervals = 0
        self.next_ns = 0
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.phase = "preflight"
        self.minimum = GIB
        self.failure = None
        self.previous_sample_ns = None

    def set_phase(self, phase, minimum=GIB):
        with self.lock:
            self.phase, self.minimum = phase, minimum

    def check_failure(self):
        with self.lock:
            failure = self.failure
        if failure is not None:
            raise failure

    def latch(self, failure):
        with self.lock:
            if self.failure is None:
                self.failure = failure

    def start(self):
        require(self.thread is None, "Memory observer already started")
        self.thread = threading.Thread(target=self.worker, name="loadsync-readonly-observer", daemon=True)
        self.thread.start()

    def worker(self):
        while not self.stop_event.is_set():
            started = time.monotonic()
            try:
                with self.lock:
                    self.sample_once(self.phase, self.minimum)
            except Exception as exc:
                self.latch(exc)
            # Actual gaps are recorded; no strict 1 Hz/no-gap assertion.
            self.stop_event.wait(max(0.01, 1.0 - (time.monotonic() - started)))

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
            require(not self.thread.is_alive(), "Memory observer did not stop within 5 seconds")

    def tick(self, phase, minimum=GIB, force=False):
        with self.lock:
            self.check_failure()
            now = time.monotonic_ns()
            if not force and now < self.next_ns:
                return None
            try:
                return self.sample_once(phase, minimum)
            except Exception as exc:
                self.latch(exc)
                raise

    def sample_once(self, phase, minimum):
        row = memory_sample(phase)
        self.samples.append(row)
        if self.previous_sample_ns is not None:
            row["actual_sample_gap_s"] = (row["monotonic_ns"] - self.previous_sample_ns) / 1e9
        self.previous_sample_ns = row["monotonic_ns"]
        self.next_ns = row["monotonic_ns"] + 1000000000
        available = row["proc_meminfo"].get("MemAvailable_bytes")
        row["gate"] = {"minimum_MemAvailable_bytes": minimum, "passed": available is not None and available >= minimum}
        try:
            row["root_live_check"] = self.root.check()
        except Exception as exc:
            row["root_live_error"] = str(exc)
            raise RootSessionError(str(exc))
        if not row["gate"]["passed"]:
            raise MemoryGate("{} MemAvailable {} < fixed {} bytes".format(phase, available, minimum))
        previous = self.last_interval_sample
        if previous is None or row["monotonic_ns"] - previous["monotonic_ns"] >= 750000000:
            if previous is not None:
                row["interval_s"] = (row["monotonic_ns"] - previous["monotonic_ns"]) / 1e9
                row["vm_counter_deltas"] = {key: row["vm_counters"].get(key, 0) - previous["vm_counters"].get(key, 0)
                                            for key in VM_COUNTERS}
                if any(value < 0 for value in row["vm_counter_deltas"].values()):
                    raise ResourceReview("System vmstat counter decreased; review session continuity")
                positive = row["vm_counter_deltas"]["pswpin"] > 0 or row["vm_counter_deltas"]["pswpout"] > 0
                self.swap_positive_intervals = self.swap_positive_intervals + 1 if positive else 0
                row["consecutive_swap_traffic_intervals"] = self.swap_positive_intervals
                if self.swap_positive_intervals >= 2:
                    raise ResourceReview("Positive global swap traffic in two consecutive sampled intervals; not a process attribution")
            self.last_interval_sample = row
        return row


def guard_paths(parser, args):
    try:
        device = Path("/proc/device-tree/model").read_bytes().replace(b"\0", b"").decode("utf-8", "replace")
    except OSError:
        device = ""
    if (platform.system() != "Linux" or platform.machine().lower() not in ("aarch64", "arm64")
            or "jetson" not in device.lower()):
        parser.error("Run on the Linux aarch64 Jetson only. No output created.")
    if os.geteuid() == 0:
        parser.error("Run inference as the ordinary B3 user, never root. No output created.")
    returns = ROOT.parent / "returns"
    require(not returns.is_symlink() and returns.is_dir() and returns.resolve() == returns,
            "Pre-existing real returns directory required")
    for name in ("inputs", "stage_b_package", "probe_report", "smoke_report", "old_b3_package", "root_session"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    output = args.output.expanduser()
    require(not output.is_symlink(), "Output must not be a symlink")
    output = output.resolve()
    protected = (ROOT, args.inputs, args.stage_b_package, args.old_b3_package,
                 args.probe_report.parent, args.smoke_report.parent, args.root_session)
    require(output.parent == returns and not output.exists(), "Output must be a fresh direct child of returns")
    require(args.root_session.parent == returns, "Root sampler session must be a direct child of returns")
    require(not any(output == path or path in output.parents or output in path.parents for path in protected),
            "Output overlaps frozen inputs, packages or old evidence")
    for ancestor in output.parents:
        require(not any((ancestor / name).exists() for name in ("benchmark_report.json", "infer_report.json", "runtime_probe.json",
                                                             "smoke_report.json", "root_sampler_report.json")),
                "Output must not be nested in old returned evidence")
    return output, device


def validated_old(path):
    require(sha(path / "delivery_manifest.json") == OLD_MANIFEST_SHA, "Historical B3 manifest changed")
    manifest = json.loads((path / "delivery_manifest.json").read_text(encoding="utf-8"))
    require(sha(path / "verify_b3_package.py") == manifest["files"]["verify_b3_package.py"], "Old verifier changed")
    verifier = import_file("pinned_old_b3_verifier", path / "verify_b3_package.py")
    checked = verifier.verify(path, OLD_MANIFEST_SHA)
    require(sha(path / "benchmark_historical_compact.py") == OLD_DRIVER_SHA, "Historical B3 driver changed")
    require(sha(path / "telemetry.py") == OLD_TELEMETRY_SHA, "Historical meminfo helper changed")
    return (import_file("pinned_old_b3_driver", path / "benchmark_historical_compact.py"),
            import_file("pinned_old_b3_meminfo", path / "telemetry.py"), checked)


def snapshot_new(output, old, args, report):
    old.snapshot(output, args.stage_b_package, args.inputs, args.probe_report, args.smoke_report, report)
    destination = output / "source_snapshot/new_streamsync"
    destination.mkdir()
    records = {}
    for source in sorted(ROOT.iterdir()):
        if source.is_file():
            target = destination / source.name
            shutil.copyfile(source, target)
            require(sha(source) == sha(target), "New payload snapshot hash mismatch")
            records[source.name] = sha(target)
    write_json(destination / "snapshot_manifest.json", records)
    report["new_source_snapshot_sha256"] = sha(destination / "snapshot_manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root-session", "output", "inputs", "stage-b-package", "probe-report", "smoke-report", "old-b3-package"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        output, device = guard_paths(parser, args)
        verifier = import_file("streamsync_verifier", ROOT / "verify_package.py")
        package_verification = verifier.verify(ROOT, args.expected_manifest_sha256)
        old, memory_module, old_verification = validated_old(args.old_b3_package)
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        root_session = RootSession(args.root_session, args.expected_manifest_sha256, boot_id)
    except (OSError, ValueError, RootSessionError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    output.mkdir(exist_ok=False)
    report = {"schema": "jetson_b3_streamsync_inference_v1", "status": "failed",
              "purpose": "historical_compact_FP16_stream_input_load_synchronous_single_pilot", "time_utc": utc(),
              "uid": os.getuid(), "euid": os.geteuid(), "pid": os.getpid(), "output": str(output),
              "boot_id": boot_id, "device_model": device,
              "python_executable": str(Path(sys.executable).resolve()), "python_version": sys.version,
              "script_sha256": sha(__file__), "new_manifest_sha256": args.expected_manifest_sha256,
              "manifest_sha256": args.expected_manifest_sha256,
              "package_verification": package_verification, "old_b3_package_verification": old_verification,
              "root_session_path": str(args.root_session), "root_session_id": root_session.ready["session_id"],
              "root_ready_sha256": root_session.ready_sha, "root_ready": root_session.ready,
              "benchmark_completed": False, "formal_evaluation_completed": False, "passed_all": False,
              "AP_measured": False, "video_FPS_measured": False, "tracking_executed": False,
              "TensorRT_executed": False, "installation_performed": False,
              "power_or_clock_configuration_changed": False, "replicate_count": 1, "precision": "FP16",
              "forward_calls_started": 0, "any_forward_started": False, "models": {}, "windows": {}, "warnings": [],
              "module_power_and_efficiency": None,
              "power_binding_note": "Infer report does not calculate power; root sampler/binder must join the identical monotonic windows later",
              "config": {"batch": 1, "input_shape": [1, 3, 640, 640], "precision": "FP16", "conf": 0.25,
                         "iou": 0.65, "max_det": 300, "all_native_images_predecoded": False,
                         "input_protocol": PROTOCOL, "max_resident_native_images": 1,
                         "firefox_must_remain_open": True,
                         "baseline_min_s": 60, "warmup_min_s": 60, "warmup_min_frames": 30,
                         "measure_min_s": 60, "measure_min_frames": 300, "max_phase_s": 300,
                         "pre_first_forward_min_MemAvailable_bytes": 3 * GIB,
                         "warmup_measurement_min_MemAvailable_bytes": GIB,
                         "root_sample_fresh_max_s": 3.5,
                         "processing_scope": "Unmodified historical B3 one_frame preprocessing/forward/FP32 native NMS/coordinate restore to CPU on one decoded native image; validation outside old processing timer",
                         "input_read_scope": "File read and native IMREAD_COLOR decode plus dtype/shape/nbytes validation; no resize or persistent image collection",
                         "pipeline_scope": "Read/decode/validate one native image, unchanged one_frame including initial/final CUDA synchronization and output validation, then release native image",
                         "window_scope": "Complete sequential-native-image window includes read/decode, old one_frame, image release, validation/record and monitor overhead; not RAM-image or video FPS",
                         "monitor_scope": "Read-only background observer targets 1Hz and latches failures; original one_frame code/timer unchanged; actual gaps and observer contention remain part of complete-window scope"}}
    console = io.StringIO()
    measured, warmed, predictions = [], [], {}
    monitor = Monitor(root_session, report)
    exitcode = 1
    previous_handlers = {}
    progress_stream = sys.stdout

    def progress(message):
        progress_stream.write("[streamsync] " + message + "\n")
        progress_stream.flush()

    def interrupted(signum, frame):
        if signum == getattr(signal, "SIGALRM", None):
            raise PhaseTimeout("Phase exceeded fixed 300-second failure cap")
        raise KeyboardInterrupt("signal " + str(signum))

    try:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
            previous_handlers[signum] = signal.signal(signum, interrupted)
        monitor.tick("before_model_load", force=True)
        loader, smoke = old.validate_previous(args.stage_b_package, args.probe_report, args.smoke_report, report)
        require(sha(args.inputs / "input_manifest.json") == old.INPUT_SHA, "Frozen input manifest changed")
        checked = subprocess.run([sys.executable, "-B", str(args.stage_b_package / "verify_edge_inputs.py"),
                                  "--inputs", str(args.inputs), "--expected-manifest-sha", old.INPUT_SHA],
                                 capture_output=True, text=True, errors="replace", timeout=120, check=False)
        report["input_verification"] = {"returncode": checked.returncode, "stdout": checked.stdout, "stderr": checked.stderr}
        require(checked.returncode == 0, "Frozen 149-image/301-asset verification failed")
        rows = json.loads((args.inputs / "input_manifest.json").read_text(encoding="utf-8"))["rows"]
        require(len(rows) == 149 and len({row["id"] for row in rows}) == 149, "Exactly 149 frozen images required")
        report["input_manifest_sha256"] = old.INPUT_SHA
        report["image_order"] = [row["id"] for row in rows]
        snapshot_new(output, old, args, report)
        (output / "root_ready_snapshot.json").write_text(json.dumps(root_session.ready, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for name in ("MPLCONFIGDIR", "XDG_CACHE_HOME", "TORCH_HOME", "CUDA_CACHE_PATH", "TMPDIR", "TEMP", "TMP"):
            target = output / "cache" / name.lower()
            target.mkdir(parents=True, exist_ok=True)
            os.environ[name] = str(target)
        for target, name in ((socket.socket, "connect"), (socket.socket, "connect_ex"),
                             (socket, "create_connection"), (socket, "getaddrinfo")):
            setattr(target, name, old.deny_network)
        sys.path.insert(0, str(loader.SOURCE))
        with contextlib.redirect_stdout(console), contextlib.redirect_stderr(console):
            import torch
            import cv2
            import numpy as np
            from models.yolo import IDetect
            from models.common import Conv
            from utils.datasets import letterbox
            from utils.general import non_max_suppression, scale_coords
            for name in ("models.yolo", "models.common", "utils.datasets", "utils.general", "sort"):
                require(loader.SOURCE in Path(sys.modules[name].__file__).resolve().parents, "Runtime source outside frozen stage B")
            require(torch.cuda.is_available() and torch.cuda.device_count() > 0, "CUDA must be available")
            report["environment"] = {"torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
                                     "opencv": cv2.__version__, "numpy": np.__version__, "cuda_device": torch.cuda.get_device_name(0),
                                     "torch_num_threads": torch.get_num_threads(), "opencv_num_threads": cv2.getNumThreads(),
                                     "cudnn_benchmark": torch.backends.cudnn.benchmark,
                                     "cudnn_deterministic": torch.backends.cudnn.deterministic,
                                     "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                                     "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                                     "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
            for name in ("torch", "torch_cuda", "opencv", "numpy", "cuda_device"):
                require(report["environment"][name] == smoke.get("environment", {}).get(name), "Runtime changed since B2: " + name)
            for name, expected_flag in PINNED_RUNTIME_FLAGS.items():
                require(report["environment"].get(name) == expected_flag, "Runtime execution flag changed since reviewed B3: " + name)
            report["runtime_flag_binding"] = {"reviewed_old_report_sha256": RUNTIME_FLAGS_OLD_REPORT_SHA,
                                               "expected": PINNED_RUNTIME_FLAGS, "matched": True,
                                               "B2_binding_scope": "B2 binds five runtime/version/device fields; additional execution flags bind reviewed B3 pilot"}
            root_session.check()
            model, dtype = loader.load_model(torch, IDetect, Conv, "FP16", report)
            max_native_image_bytes = max(int(row["width"]) * int(row["height"]) * 3 for row in rows)
            report["config"]["sequential_input_capacity_min_MemAvailable_bytes"] = 3 * GIB + max_native_image_bytes
            report["sequential_input_capacity_gate"] = {"max_native_image_bytes": max_native_image_bytes,
                "minimum_bytes": 3 * GIB + max_native_image_bytes,
                "rule": "MemAvailable >= unchanged 3 GiB first-forward reserve + one maximum native image"}
            initial_sample = monitor.tick("before_sequential_input_check", minimum=3 * GIB, force=True)
            available = initial_sample["proc_meminfo"].get("MemAvailable_bytes")
            capacity_passed = available is not None and available >= 3 * GIB + max_native_image_bytes
            report["sequential_input_capacity_gate"].update(
                MemAvailable_bytes=available, passed=capacity_passed)
            if not capacity_passed:
                raise MemoryGate("Sequential input capacity gate: MemAvailable {} < {} bytes".format(
                    available, 3 * GIB + max_native_image_bytes))
            signal.alarm(300)
            sequential_input_check(rows, args.inputs, cv2, report, monitor)
            signal.alarm(0)
            monitor.tick("post_sequential_input_check_before_first_forward", minimum=3 * GIB, force=True)
            torch.cuda.synchronize()
            monitor.set_phase("baseline", 3 * GIB)
            monitor.start()
            start_ns = time.monotonic_ns()
            report["windows"]["baseline"] = {"start_monotonic_ns": start_ns, "end_monotonic_ns": None,
                "model_loaded": True, "all_images_predecoded": False, "native_images_retained": 0,
                "input_protocol": PROTOCOL, "inference_executed": False, "complete": False}
            progress("Loaded-idle baseline started: 60s, no inference")
            while (time.monotonic_ns() - start_ns) < 60000000000:
                monitor.check_failure()
                time.sleep(0.1)
            end_ns = time.monotonic_ns()
            report["windows"]["baseline"].update(end_monotonic_ns=end_ns, duration_s=(end_ns - start_ns) / 1e9, complete=True)
            monitor.tick("immediately_before_first_forward", minimum=3 * GIB, force=True)
            events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            monitor.set_phase("warmup")

            def infer(image, row):
                monitor.check_failure()
                if not report["any_forward_started"]:
                    monitor.tick("first_decoded_image_before_first_forward", minimum=3 * GIB, force=True)
                report["forward_calls_started"] += 1
                report["any_forward_started"] = True
                record, values = old.one_frame(model, dtype, image, row, torch, np, letterbox,
                                               non_max_suppression, scale_coords, loader, events)
                return record, values

            with torch.no_grad():
                signal.alarm(300)
                run_stream_window("warmup", 30, 60, 300, rows, args.inputs, cv2,
                                  infer, report, warmed, None, progress, monitor)
                signal.alarm(0)
                monitor.tick("postwarm", force=True)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(0)
                monitor.set_phase("measurement")
                signal.alarm(300)
                run_stream_window("measurement", 300, 60, 300, rows, args.inputs, cv2,
                                  infer, report, measured, predictions, progress, monitor)
                signal.alarm(0)
            report["CUDA_memory_measurement_peak_MiB"] = {
                "allocated": torch.cuda.max_memory_allocated(0) / (1024 ** 2),
                "reserved": torch.cuda.max_memory_reserved(0) / (1024 ** 2),
                "scope": "reset after warmup before measurement; allocator only, not additive to unified board RAM/RSS"}
            require(len(predictions) == 149, "Measurement did not visit all 149 frozen images")
            require("NMS time limit" not in console.getvalue(), "Native NMS time-limit warning")
            monitor.tick("post_measurement", force=True)
            monitor.set_phase("right_bracket_after_measurement")
            time.sleep(1.5)
            monitor.tick("right_bracket_after_measurement", force=True)
            report["benchmark_completed"] = True
            report["status"] = "streamsync_measurement_completed_review_required"
            exitcode = 0
    except MemoryGate as exc:
        report["status"] = "blocked_memory_gate"
        report["error"] = str(exc)
        exitcode = 3
    except ResourceReview as exc:
        report["status"] = "resource_review_required"
        report["error"] = str(exc)
        exitcode = 3
    except RootSessionError as exc:
        report["status"] = "blocked_root_session"
        report["error"] = str(exc)
        exitcode = 3
    except KeyboardInterrupt as exc:
        report["status"] = "interrupted_partial_run"
        report["error"] = str(exc)
        exitcode = 130
    except PhaseTimeout as exc:
        report["status"] = "failed_phase_timeout"
        report["error"] = str(exc)
        exitcode = 124
    except Exception as exc:
        report["error"] = "{}: {}".format(type(exc).__name__, exc)
        report["traceback"] = traceback.format_exc()
        exitcode = 1
    finally:
        signal.alarm(0)
        for signum in previous_handlers:
            signal.signal(signum, signal.SIG_IGN)
        try:
            monitor.stop()
        except Exception as exc:
            report["observer_cleanup_error"] = str(exc)
            if exitcode == 0:
                exitcode = 1
                report["status"] = "failed_observer_cleanup"
        if exitcode == 0 and monitor.failure is not None:
            report["status"] = ("blocked_memory_gate" if isinstance(monitor.failure, MemoryGate) else
                                "blocked_root_session" if isinstance(monitor.failure, RootSessionError) else
                                "resource_review_required")
            report["error"] = str(monitor.failure)
            exitcode = 3
        for window in report["windows"].values():
            if window.get("end_monotonic_ns") is None:
                window["end_monotonic_ns"] = time.monotonic_ns()
                window["complete"] = False
                window["duration_s"] = (window["end_monotonic_ns"] - window["start_monotonic_ns"]) / 1e9
        report["actual_self_exit_code"] = exitcode
        report["time_finished_utc"] = utc()
        report["forward_not_started"] = not report["any_forward_started"]
        report["memory_checks"] = [{key: value for key, value in row.items() if key != "raw"} for row in monitor.samples]
        gaps = [row["actual_sample_gap_s"] for row in monitor.samples if "actual_sample_gap_s" in row]
        report["memory_monitor"] = {"target_interval_s": 1, "sample_count": len(monitor.samples),
            "max_actual_sample_gap_s": max(gaps) if gaps else None,
            "gaps_over_1_5s": sum(gap > 1.5 for gap in gaps), "strict_1Hz_asserted": False,
            "thread_started": monitor.thread is not None,
            "thread_stopped": monitor.thread is None or not monitor.thread.is_alive(),
            "failure_latch": ({"type": type(monitor.failure).__name__, "message": str(monitor.failure)}
                              if monitor.failure is not None else None),
            "consecutive_swap_traffic_intervals": monitor.swap_positive_intervals,
            "counter_scope": "Global pswpin/out/pgmajfault; process minflt/majflt/VmSwap separately; no process attribution from global counters"}
        report["measurement_latency"] = latency_summary(old, measured)
        report["warmup_latency"] = latency_summary(old, warmed)
        window = report["windows"].get("measurement", {})
        if window.get("complete"):
            report["image_window_throughput_per_s"] = len(measured) / window["duration_s"]
        report["throughput_scope"] = "N/complete sequential-native-image window includes file read/decode, old processing, image release, monitor and record overhead; not RAM-image throughput or video FPS"
        report["limitations"] = ["One new pilot only, no formal replicates/AP/clock-stability claim",
            "Root sampler live freshness is checked, but power requires the later binder to verify raw sample coverage",
            "All /proc/vmstat swap traffic is global, not attribution to this process; occupancy alone is not swap traffic",
            "Observer samples concurrently with unchanged one_frame code; observer contention is included in timings; actual gaps retained, no strict 1Hz or zero-overhead claim",
            "Native images are decoded sequentially in-window; this new protocol cannot be pooled with the historical all-images-in-RAM pilot",
            "Firefox must remain open for the user's transfer/command workflow; contention remains part of this operating condition",
            "No silent input/threshold changes; first-forward attempted counter means entry into old.one_frame, not an added model hook"]
        save_csv(output / "frame_latency.csv", measured)
        save_csv(output / "warmup_frame_latency.csv", warmed)
        write_json(output / "predictions.json", [predictions[key] for key in report.get("image_order", []) if key in predictions])
        write_json(output / "measurement_window.json", report["windows"])
        with (output / "process_memory_samples.jsonl").open("x", encoding="utf-8") as stream:
            for row in monitor.samples:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        (output / "console.log").write_text(console.getvalue(), encoding="utf-8")
        write_json(output / "run_metadata.json", {"argv": sys.argv, "config": report["config"],
            "environment": report.get("environment"), "root_session_id": report["root_session_id"],
            "root_ready_sha256": report["root_ready_sha256"], "script_sha256": report["script_sha256"]})
        artifacts = [path for path in output.rglob("*") if path.is_file() and "cache" not in path.relative_to(output).parts]
        report["artifacts"] = [{"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size,
                                "sha256": sha(path)} for path in sorted(artifacts)]
        report["artifact_binding_scope"] = "All primary infer artifacts/source snapshots before report; excludes isolated runtime cache and report/sidecar"
        write_json(output / "infer_report.json", report)
        (output / "infer_report.sha256").write_text(sha(output / "infer_report.json") + "  infer_report.json\n", encoding="ascii")
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    print(json.dumps({"status": report["status"], "actual_self_exit_code": exitcode,
                      "benchmark_completed": report["benchmark_completed"], "passed_all": False,
                      "forward_not_started": report["forward_not_started"], "output": str(output)}, ensure_ascii=False), flush=True)
    return exitcode


if __name__ == "__main__":
    sys.exit(main())
