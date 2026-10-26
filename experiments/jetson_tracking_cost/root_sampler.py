"""Bounded root-only Jetson telemetry for an independently run normal-user video measurement.

Run with isolated Python through terminal sudo. No model, inference, IPC command
execution, internal sudo, system configuration change or sysfs write is allowed.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid


ROOT = Path(__file__).resolve().parent
PAYLOADS = {"root_sampler.py", "video_formal.py", "bind_results.py", "verify_package.py", "README.md"}
MAX_SECONDS = 1800
READY_DEADLINE_SECONDS = 30
POWER_RAIL = re.compile(r"\bVDD_IN\s+(\d+(?:\.\d+)?)mW/(\d+(?:\.\d+)?)mW\b")
VM_COUNTERS = ("pswpin", "pswpout", "pgmajfault")


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def failure(exc):
    return {"type": type(exc).__name__, "message": str(exc), "errno": getattr(exc, "errno", None)}


def finite(raw, positive=False):
    try:
        value = float(raw.strip())
        return value if math.isfinite(value) and (value > 0 if positive else value >= 0) else None
    except (AttributeError, ValueError):
        return None


def read_node(path):
    result = {"path": str(path), "monotonic_ns": time.monotonic_ns(), "time_utc": utc()}
    try:
        path = Path(path)
        result["resolved_path"] = str(path.resolve())
        result["mode"] = stat.filemode(path.stat().st_mode)
        result["raw"] = path.read_bytes().decode("utf-8", "replace")
    except OSError as exc:
        result["error"] = failure(exc)
    result["finished_monotonic_ns"] = time.monotonic_ns()
    return result


def verify_package(expected):
    # Only the package verifier changes; all privileged sampling stays identical.
    import importlib.util
    spec = importlib.util.spec_from_file_location("c4_root_verify", str(ROOT / "verify_package.py"))
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    manifest = verifier.verify(ROOT, expected)
    return manifest, (ROOT / "delivery_manifest.json").read_bytes(), {r["path"]: (ROOT / r["path"]).read_bytes() for r in manifest["files"]}


class NewArtifacts:
    """All new writes use an anchored FD; output-parent renames cannot redirect them."""
    def __init__(self, output):
        parent = ROOT.parent / "returns"
        if not output.is_absolute() or output.parent != parent or output.name in ("", ".", ".."):
            raise ValueError("Output must be a new absolute direct child of " + str(parent))
        if parent.is_symlink() or not parent.is_dir() or parent.resolve() != parent:
            raise ValueError("Pre-existing returns parent must be a real directory without symlinks")
        if os.path.lexists(str(output)):
            raise ValueError("Output exists or is a symlink; select a new name")
        parent_fd = os.open(str(parent), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.mkdir(output.name, 0o755, dir_fd=parent_fd)
            self.fd = os.open(output.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
        finally:
            os.close(parent_fd)
        self.path = output
        self.names = set()

    @staticmethod
    def check_name(name):
        if "/" in name or "\\" in name or name in ("", ".", ".."):
            raise ValueError("Artifact filename must be a simple controlled basename")

    def open_new(self, name):
        self.check_name(name)
        descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=self.fd)
        self.names.add(name)
        return os.fdopen(descriptor, "wb")

    def write(self, name, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        with self.open_new(name) as stream:
            stream.write(data)

    def replace_json(self, name, value):
        self.check_name(name)
        temporary = "." + name + ".tmp." + uuid.uuid4().hex
        data = (json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=self.fd)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
            os.replace(temporary, name, src_dir_fd=self.fd, dst_dir_fd=self.fd)
            self.names.add(name)
        finally:
            try:
                os.unlink(temporary, dir_fd=self.fd)
            except FileNotFoundError:
                pass

    def read(self, name):
        self.check_name(name)
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=self.fd)
        with os.fdopen(descriptor, "rb") as stream:
            return stream.read()

    def inventory(self):
        result = []
        for name in sorted(self.names):
            data = self.read(name)
            result.append({"path": name, "bytes": len(data), "sha256": sha(data)})
        return result


class StopSampler(Exception):
    def __init__(self, signum):
        self.signum = signum
        super().__init__("signal " + str(signum))


class Sampler:
    def __init__(self, artifacts, session_id, boot_id, manifest_sha, script_sha):
        self.artifacts = artifacts
        self.session_id = session_id
        self.boot_id = boot_id
        self.manifest_sha = manifest_sha
        self.script_sha = script_sha
        self.commands = []
        self.sensors = []
        self.discovery_errors = []
        self.vdd_binding = None
        self.gpu_paths = None
        self.emc_path = "/sys/kernel/debug/bpmp/debug/clk/emc/rate"
        self.fan_paths = []
        self.process = None
        self.reader = None
        self.reader_error = None
        self.lock = threading.Lock()
        self.tegrastats_count = 0
        self.valid_power_count = 0
        self.sysfs_count = 0
        self.last_tegrastats_ns = None
        self.last_power_tegrastats_ns = None
        self.last_sysfs_ns = None
        self.last_valid_gpu_emc = False
        self.last_valid_vdd_pair = False
        self.ready = None
        self.status = "preparing"
        self.handles = []
        self.tegra_command = None
        self.timeout = self.binary("timeout")

    @staticmethod
    def binary(name):
        if name not in ("timeout", "nvpmodel", "jetson_clocks", "tegrastats"):
            return None
        for directory in ("/usr/bin", "/usr/sbin"):
            candidate = Path(directory) / name
            try:
                resolved = candidate.resolve(strict=True)
                metadata = resolved.stat()
                if (metadata.st_uid == 0 and stat.S_ISREG(metadata.st_mode)
                        and not metadata.st_mode & 0o022 and os.access(str(resolved), os.X_OK)):
                    return str(resolved)
            except OSError:
                pass
        return None

    def query(self, name, binary, arguments):
        row = {"name": name, "time_started_utc": utc(), "monotonic_ns": time.monotonic_ns(),
               "euid": os.geteuid(), "returncode": None, "bound_seconds": 8,
               "stdout_file": name + ".stdout.txt", "stderr_file": name + ".stderr.txt"}
        self.commands.append(row)
        target = self.binary(binary)
        row["target_argv"] = [target] + arguments
        if not target or not self.timeout:
            row["status"] = "missing_system_binary_or_timeout"
            row["argv"] = None
            row["stdout"], row["stderr"] = "", ""
            self.artifacts.write(row["stdout_file"], b"")
            self.artifacts.write(row["stderr_file"], b"")
            return row
        argv = [self.timeout, "--signal=TERM", "--kill-after=2s", "8s", target] + arguments
        row["argv"] = argv
        process = None
        try:
            with self.artifacts.open_new(row["stdout_file"]) as stdout, self.artifacts.open_new(row["stderr_file"]) as stderr:
                process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                           start_new_session=True)
                row["pid"] = process.pid
                row["returncode"] = process.wait(timeout=12)
                row["status"] = "exited"
        except BaseException as exc:
            row["status"] = "interrupted_or_failed"
            row["error"] = failure(exc)
            if process is not None:
                self.stop_group(process)
                row["returncode"] = process.poll()
            raise
        finally:
            row["finished_monotonic_ns"] = time.monotonic_ns()
            row["time_finished_utc"] = utc()
            for field in ("stdout", "stderr"):
                try:
                    row[field] = self.artifacts.read(row[field + "_file"]).decode("utf-8", "replace")
                except OSError as exc:
                    row[field + "_error"] = failure(exc)
        return row

    @staticmethod
    def stop_group(process):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=3)

    def directories(self, parent, prefix, limit=32):
        try:
            matched = sorted(path for path in Path(parent).iterdir() if path.name.startswith(prefix) and path.is_dir())
            if len(matched) > limit:
                self.discovery_errors.append({"parent": parent, "reason": "finite_limit", "matched": len(matched), "limit": limit})
            return matched[:limit]
        except OSError as exc:
            self.discovery_errors.append({"parent": parent, "error": failure(exc)})
            return []

    def discover(self):
        seen = set()
        bindings = []
        for parent in ("/sys/bus/i2c/drivers/ina3221/1-0040/hwmon", "/sys/class/hwmon"):
            for directory in self.directories(parent, "hwmon"):
                try:
                    resolved = directory.resolve()
                    if str(resolved) in seen or not str(resolved).startswith("/sys/"):
                        continue
                    seen.add(str(resolved))
                except OSError as exc:
                    self.discovery_errors.append({"path": str(directory), "error": failure(exc)})
                    continue
                sensor = {"path": str(directory), "resolved_path": str(resolved), "fields": {},
                          "driver_symlink_path": str(directory / "device/driver")}
                try:
                    driver = directory / "device/driver"
                    sensor["driver_exists"] = driver.exists()
                    sensor["driver_resolved_path"] = str(driver.resolve())
                except OSError as exc:
                    sensor["driver_error"] = failure(exc)
                for field in ("name", "in1_label", "in1_input", "curr1_input", "in2_label", "in2_input", "curr2_input",
                              "in3_label", "in3_input", "curr3_input"):
                    sensor["fields"][field] = read_node(directory / field)
                self.sensors.append(sensor)
                name = sensor["fields"]["name"].get("raw", "").strip()
                label = sensor["fields"]["in1_label"].get("raw", "").strip()
                if name == "ina3221" and label == "VDD_IN" and "/1-0040/" in str(resolved):
                    bindings.append({"sensor_path": str(directory), "sensor_resolved_path": str(resolved),
                                     "driver_resolved_path": sensor.get("driver_resolved_path"),
                                     "name": name, "channel": 1, "label": label,
                                     "voltage_path": str(directory / "in1_input"),
                                     "current_path": str(directory / "curr1_input"),
                                     "units_note": "INA3221 channel 1 bus voltage/current raw sysfs readings; no power calculation"})
                try:
                    for node in sorted(directory.iterdir()):
                        if re.fullmatch(r"(?:fan\d+_input|pwm\d+)", node.name) and len(self.fan_paths) < 16:
                            self.fan_paths.append(str(node))
                except OSError as exc:
                    self.discovery_errors.append({"fan_parent": str(directory), "error": failure(exc)})
        if len(bindings) == 1:
            self.vdd_binding = bindings[0]
        else:
            self.discovery_errors.append({"reason": "expected_one_exact_ina3221_1_0040_channel1_VDD_IN", "bindings": bindings})
        candidates = [Path("/sys/devices/17000000.ga10b/devfreq/17000000.ga10b"),
                      Path("/sys/devices/platform/17000000.ga10b/devfreq/17000000.ga10b")]
        candidates += self.directories("/sys/class/devfreq", "")
        self.gpu_discovery = []
        for directory in candidates:
            if "17000000.ga10b" not in str(directory):
                continue
            paths = {field: str(directory / field) for field in ("cur_freq", "min_freq", "max_freq")}
            reads = {field: read_node(path) for field, path in paths.items()}
            self.gpu_discovery.append({"directory": str(directory), "reads": reads})
            if self.gpu_paths is None and all(finite(row.get("raw"), positive=True) is not None for row in reads.values()):
                self.gpu_paths = paths
        self.emc_discovery = read_node(self.emc_path)

    def base(self):
        return {"session_id": self.session_id, "boot_id": self.boot_id, "euid": os.geteuid(),
                "sampler_pid": os.getpid(), "manifest_sha256": self.manifest_sha,
                "script_sha256": self.script_sha, "output": str(self.artifacts.path)}

    def live(self):
        with self.lock:
            result = self.base()
            result.update({"schema": "jetson_video_formal_live_v1", "time_utc": utc(), "monotonic_ns": time.monotonic_ns(),
                           "last_tegrastats_monotonic_ns": self.last_tegrastats_ns,
                           "last_power_tegrastats_monotonic_ns": self.last_power_tegrastats_ns,
                           "last_sysfs_monotonic_ns": self.last_sysfs_ns,
                           "tegrastats_line_count": self.tegrastats_count, "sysfs_sample_count": self.sysfs_count,
                           "power_line_count": self.valid_power_count, "ready_written": self.ready is not None,
                           "sampler_status": self.status, "last_sysfs_valid_gpu_emc": self.last_valid_gpu_emc,
                           "last_sysfs_valid_vdd_pair": self.last_valid_vdd_pair,
                           "tegrastats_pid": self.process.pid if self.process else None})
        self.artifacts.replace_json("live.json", result)

    def read_telemetry(self, source, jsonl, original):
        try:
            for raw_bytes in iter(source.readline, b""):
                timestamp = time.monotonic_ns()
                raw = raw_bytes.decode("utf-8", "replace")
                with self.lock:
                    self.tegrastats_count += 1
                    sequence = self.tegrastats_count
                row = {"session_id": self.session_id, "boot_id": self.boot_id, "monotonic_ns": timestamp,
                       "time_utc": utc(), "raw": raw, "euid": os.geteuid(), "sequence": sequence,
                       "timestamp_scope": "Parent reader receipt after readline; sensor acquisition time/buffering is not asserted"}
                jsonl.write((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
                jsonl.flush()
                original.write(raw_bytes)
                original.flush()
                match = POWER_RAIL.search(raw)
                valid = bool(match and finite(match.group(1), positive=True) is not None
                             and finite(match.group(2)) is not None)
                with self.lock:
                    self.last_tegrastats_ns = timestamp
                    if valid:
                        self.valid_power_count += 1
                        self.last_power_tegrastats_ns = timestamp
        except BaseException as exc:
            with self.lock:
                self.reader_error = failure(exc)
        finally:
            source.close()

    def start_telemetry(self):
        executable = self.binary("tegrastats")
        if executable is None or self.timeout is None:
            raise ValueError("Missing trusted tegrastats or GNU timeout")
        argv = [self.timeout, "--signal=TERM", "--kill-after=2s", str(MAX_SECONDS) + "s", executable,
                "--interval", "1000", "--verbose"]
        jsonl = self.artifacts.open_new("root_tegrastats.jsonl")
        original = self.artifacts.open_new("root_tegrastats.stdout.txt")
        stderr = self.artifacts.open_new("root_tegrastats.stderr.txt")
        self.handles.extend((jsonl, original, stderr))
        self.tegra_command = {"argv": argv, "euid": os.geteuid(), "time_started_utc": utc(),
                              "monotonic_ns": time.monotonic_ns(), "returncode": None,
                              "stdout_file": "root_tegrastats.stdout.txt", "stderr_file": "root_tegrastats.stderr.txt",
                              "jsonl_file": "root_tegrastats.jsonl", "bound_seconds": MAX_SECONDS,
                              "reader_timestamp_scope": "Receipt after pipe readline; includes unknown producer/pipe buffering"}
        self.process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=stderr,
                                        start_new_session=True, bufsize=0)
        self.tegra_command["pid"] = self.process.pid
        self.reader = threading.Thread(target=self.read_telemetry, args=(self.process.stdout, jsonl, original), daemon=True)
        self.reader.start()

    def sysfs_sample(self, stream):
        timestamp = time.monotonic_ns()
        row = {"session_id": self.session_id, "boot_id": self.boot_id, "monotonic_ns": timestamp,
               "time_utc": utc(), "euid": os.geteuid(), "sequence": self.sysfs_count + 1,
               "vdd_in": None, "gpu": {}, "emc_rate": read_node(self.emc_path), "cpu_freq": [],
               "meminfo": read_node("/proc/meminfo"), "vmstat": read_node("/proc/vmstat"),
               "fan": [read_node(path) for path in self.fan_paths]}
        if self.vdd_binding:
            row["vdd_in"] = {"binding": self.vdd_binding,
                             "voltage": read_node(self.vdd_binding["voltage_path"]),
                             "current": read_node(self.vdd_binding["current_path"]),
                             "sampling_note": "Sequential paired reads with individual timestamps; not atomic; no calculated power"}
        if self.gpu_paths:
            row["gpu"] = {field: read_node(path) for field, path in self.gpu_paths.items()}
        for cpu in range(6):
            prefix = "/sys/devices/system/cpu/cpu{}/cpufreq/".format(cpu)
            row["cpu_freq"].append({"cpu": cpu, "fields": {
                field: read_node(prefix + field) for field in ("scaling_cur_freq", "scaling_min_freq", "scaling_max_freq", "scaling_governor")}})
        row["vm_counters"] = {}
        for line in row["vmstat"].get("raw", "").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] in VM_COUNTERS:
                try:
                    row["vm_counters"][parts[0]] = int(parts[1])
                except ValueError:
                    pass
        row["finished_monotonic_ns"] = time.monotonic_ns()
        stream.write((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
        stream.flush()
        gpu_valid = bool(row["gpu"]) and all(finite(value.get("raw"), positive=True) is not None for value in row["gpu"].values())
        emc_valid = finite(row["emc_rate"].get("raw"), positive=True) is not None
        pair = row["vdd_in"]
        pair_valid = bool(pair and finite(pair["voltage"].get("raw"), positive=True) is not None
                          and finite(pair["current"].get("raw"), positive=True) is not None)
        with self.lock:
            self.sysfs_count += 1
            self.last_sysfs_ns = timestamp
            self.last_valid_gpu_emc = bool(gpu_valid and emc_valid)
            self.last_valid_vdd_pair = bool(pair_valid)
        return row

    def maybe_ready(self, mode_ok, clocks_ok):
        if self.ready is not None:
            return
        with self.lock:
            allowed = (self.valid_power_count >= 2 and self.last_valid_gpu_emc and self.last_valid_vdd_pair
                       and self.vdd_binding is not None and mode_ok and clocks_ok and self.reader_error is None)
        if allowed:
            ready = self.base()
            ready.update({"schema": "jetson_video_formal_ready_v1", "ready_monotonic_ns": time.monotonic_ns(),
                          "time_utc": utc(), "vdd_in_binding": self.vdd_binding, "mode_query_ok": mode_ok,
                          "clock_query_ok": clocks_ok, "power_line_count_at_ready": self.valid_power_count,
                          "ready_scope": "Root sampling sources available; this is not a benchmark or experiment pass"})
            self.artifacts.write("ready.json", json.dumps(ready, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            self.ready = ready
            self.live()
            print("ROOT_SAMPLER_READY " + str(self.artifacts.path / "ready.json") + " session_id=" + self.session_id, flush=True)

    def stop_telemetry(self):
        if self.process:
            self.stop_group(self.process)
            if self.tegra_command:
                self.tegra_command["returncode"] = self.process.poll()
                self.tegra_command["subprocess_finished_monotonic_ns"] = time.monotonic_ns()
        if self.reader:
            self.reader.join(timeout=4)
            if self.reader.is_alive():
                self.reader_error = {"type": "ReaderJoinTimeout", "message": "Reader did not finish within four seconds"}
        for handle in self.handles:
            if not handle.closed:
                handle.flush()
                handle.close()
        if self.tegra_command:
            # Parent receipt timestamps can follow the child's exit while the
            # reader drains its last pipe line. Bound sampling after that join.
            self.tegra_command["finished_monotonic_ns"] = time.monotonic_ns()
            self.tegra_command["time_finished_utc"] = utc()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    model = read_node("/proc/device-tree/model").get("raw", "").replace("\0", "")
    if (platform.system() != "Linux" or platform.machine().lower() not in ("aarch64", "arm64")
            or "jetson" not in model.lower()):
        parser.error("Run only on the Linux aarch64 Jetson. No output created.")
    if os.geteuid() != 0:
        parser.error("Start this frozen sampler once through terminal sudo, with isolated Python. No output created.")
    if not sys.flags.isolated:
        parser.error("Use python3.8 -I -B for the root sampler. No output created.")
    try:
        manifest, manifest_raw, payloads = verify_package(args.expected_manifest_sha256)
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        uuid.UUID(boot_id)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error("Preflight verification failed before output creation: " + str(exc))
    previous_umask = os.umask(0o022)
    try:
        artifacts = NewArtifacts(args.output)
    except (OSError, ValueError) as exc:
        os.umask(previous_umask)
        parser.error(str(exc))
    session_id = str(uuid.uuid4())
    sampler = Sampler(artifacts, session_id, boot_id, args.expected_manifest_sha256.lower(), sha(payloads["root_sampler.py"]))
    report = sampler.base()
    report.update({"schema": "jetson_video_formal_root_sampler_v1", "status": "started", "exitcode": None,
                   "time_started_utc": utc(), "started_monotonic_ns": time.monotonic_ns(),
                   "max_sampling_seconds": MAX_SECONDS, "readiness_deadline_seconds": READY_DEADLINE_SECONDS,
                   "model_raw": model, "python_executable": sys.executable, "python_version": sys.version,
                   "isolated_python_flag": sys.flags.isolated, "package_manifest": manifest,
                   "scope": "Read-only root sensor sampler; normal-user inference is run independently in another terminal",
                   "limits": ["No model/torch load, inference/training, IPC commands, internal sudo, installation, sysfs writes, clock or power-mode changes",
                              "No chmod/chown, sudoers edits, mounts, I2C probing, global stop or unrelated process kill",
                              "Writes only newly created output artifacts, anchored by directory FD with O_NOFOLLOW",
                              "Tegrastats timestamps are parent receipt after readline; acquisition time and producer buffering are not asserted",
                              "Sysfs readings are sequential with individual timestamps; no power/efficiency calculation",
                              "Stopped sampling is not a successful experiment; phase windows and metrics require separate offline binding"]})
    previous_handlers = {}
    exitcode = 0
    start_mode = None
    start_clocks = None
    first_sample = None
    last_sample = None
    finalized = b""

    def interrupted(signum, frame):
        raise StopSampler(signum)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
            previous_handlers[signum] = signal.signal(signum, interrupted)
        signal.alarm(MAX_SECONDS)
        artifacts.write("source_root_sampler_snapshot.py", payloads["root_sampler.py"])
        artifacts.write("reviewed_delivery_manifest.json", manifest_raw)
        report["environment_reads"] = [read_node(path) for path in ("/etc/nv_tegra_release", "/proc/swaps")]
        sampler.discover()
        start_mode = sampler.query("start_nvpmodel_query", "nvpmodel", ["-q"])
        start_clocks = sampler.query("start_jetson_clocks_show", "jetson_clocks", ["--show"])
        mode_ok = bool(start_mode.get("returncode") == 0 and re.search(r"(?m)^NV Power Mode:\s*15W\s*$", start_mode.get("stdout", "")))
        clocks_ok = bool(start_clocks.get("returncode") == 0 and start_clocks.get("stdout", "").strip())
        report["initial_queries_ok"] = {"mode_15W": mode_ok, "clocks_show": clocks_ok}
        sampler.start_telemetry()
        sampler.status = "collecting"
        sampler.live()
        readiness_deadline = time.monotonic() + READY_DEADLINE_SECONDS
        with artifacts.open_new("root_sysfs_samples.jsonl") as stream:
            next_sample = time.monotonic()
            while True:
                now = time.monotonic()
                if now >= next_sample:
                    last_sample = sampler.sysfs_sample(stream)
                    if first_sample is None:
                        first_sample = last_sample
                    sampler.maybe_ready(mode_ok, clocks_ok)
                    sampler.live()
                    next_sample = now + 1
                if sampler.reader_error:
                    raise RuntimeError("Tegrastats reader failed: " + str(sampler.reader_error))
                if sampler.process.poll() is not None:
                    if sampler.process.returncode == 124:
                        report["status"] = "time_limit_reached_partial_sampling"
                        report["exit_reason"] = "owned_tegrastats_timeout"
                        exitcode = 124
                        break
                    raise RuntimeError("Tegrastats exited before operator stop: " + str(sampler.process.returncode))
                if sampler.ready is None and now >= readiness_deadline:
                    report["status"] = "readiness_requirements_not_met"
                    report["exit_reason"] = "no_ready_after_30_seconds"
                    exitcode = 3
                    break
                time.sleep(0.05)
    except StopSampler as exc:
        if exc.signum == signal.SIGINT:
            report["status"] = "stopped_by_operator"
            report["exit_reason"] = "SIGINT"
            exitcode = 0
        elif exc.signum == signal.SIGALRM:
            report["status"] = "time_limit_reached_partial_sampling"
            report["exit_reason"] = "sampler_alarm_1800_seconds"
            exitcode = 124
        else:
            report["status"] = "terminated_partial_sampling"
            report["exit_reason"] = "SIGTERM"
            exitcode = 143
    except KeyboardInterrupt:
        report["status"] = "stopped_by_operator"
        report["exit_reason"] = "KeyboardInterrupt"
        exitcode = 0
    except Exception as exc:
        report["status"] = "failed_partial_sampling"
        report["exit_reason"] = "exception"
        report["error"] = failure(exc)
        exitcode = 1
    finally:
        signal.alarm(0)
        for signum in previous_handlers:
            signal.signal(signum, signal.SIG_IGN)
        try:
            sampler.stop_telemetry()
        except BaseException as exc:
            report["cleanup_error"] = failure(exc)
            if exitcode == 0:
                exitcode = 1
                report["status"] = "cleanup_failed_partial_sampling"
        # Final read-only queries bracket the sampled interval. They do not
        # change mode or clocks and are each separately bounded to eight seconds.
        for name, binary, options in (("end_nvpmodel_query", "nvpmodel", ["-q"]),
                                      ("end_jetson_clocks_show", "jetson_clocks", ["--show"])):
            try:
                sampler.query(name, binary, options)
            except BaseException as exc:
                report.setdefault("end_query_errors", []).append(failure(exc))
        sampler.status = report["status"]
        try:
            sampler.live()
        except BaseException as exc:
            report["final_live_error"] = failure(exc)
        report.update({"exitcode": exitcode, "time_finished_utc": utc(), "finished_monotonic_ns": time.monotonic_ns(),
                       "queries": sampler.commands, "tegrastats_command": sampler.tegra_command,
                       "ready": sampler.ready, "sensors": sampler.sensors, "vdd_in_binding": sampler.vdd_binding,
                       "ready_sha256": sha(artifacts.read("ready.json")) if sampler.ready is not None else None,
                       "sampling_started_monotonic_ns": sampler.tegra_command.get("monotonic_ns") if sampler.tegra_command else None,
                       "sampling_finished_monotonic_ns": sampler.tegra_command.get("finished_monotonic_ns") if sampler.tegra_command else None,
                       "gpu_paths": sampler.gpu_paths, "gpu_discovery": getattr(sampler, "gpu_discovery", []),
                       "emc_path": sampler.emc_path, "emc_discovery": getattr(sampler, "emc_discovery", None),
                       "fan_paths": sampler.fan_paths, "discovery_errors": sampler.discovery_errors,
                       "tegrastats_line_count": sampler.tegrastats_count, "positive_VDD_IN_line_count": sampler.valid_power_count,
                       "sysfs_sample_count": sampler.sysfs_count, "reader_error": sampler.reader_error,
                       "last_tegrastats_monotonic_ns": sampler.last_tegrastats_ns,
                       "last_power_tegrastats_monotonic_ns": sampler.last_power_tegrastats_ns,
                       "last_sysfs_monotonic_ns": sampler.last_sysfs_ns})
        if first_sample is not None and last_sample is not None:
            report["sysfs_sample_span_ns"] = last_sample["monotonic_ns"] - first_sample["monotonic_ns"]
            report["vm_counter_deltas_whole_sampler_only"] = {
                key: last_sample["vm_counters"][key] - first_sample["vm_counters"][key]
                if key in first_sample["vm_counters"] and key in last_sample["vm_counters"] else None for key in VM_COUNTERS}
        report["artifacts"] = artifacts.inventory()
        finalized = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        artifacts.write("root_sampler_report.json", finalized)
        artifacts.write("root_sampler_report.sha256", sha(finalized) + "  root_sampler_report.json\n")
        os.close(artifacts.fd)
        os.umask(previous_umask)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    print(json.dumps({"status": report["status"], "exitcode": exitcode, "output": str(args.output),
                      "session_id": session_id, "report_sha256": sha(finalized)}, ensure_ascii=False), flush=True)
    return exitcode


if __name__ == "__main__":
    sys.exit(main())
