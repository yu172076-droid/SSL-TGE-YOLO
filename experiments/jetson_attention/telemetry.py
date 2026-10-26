"""Read-only, process-owned Jetson telemetry; standard library only."""
import datetime
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import threading
import time


MIB = 1024.0 * 1024.0
POWER_SCOPE = "total_module_not_wall_or_carrier_board"
RAIL = re.compile(r"\bVDD_IN\s+(\d+(?:\.\d+)?)mW/(\d+(?:\.\d+)?)mW\b")


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def meminfo():
    result = {}
    try:
        with open("/proc/meminfo", encoding="ascii") as stream:
            for line in stream:
                name, value = line.split(":", 1)
                parts = value.split()
                if len(parts) >= 2 and parts[1] == "kB":
                    result[name + "_bytes"] = int(parts[0]) * 1024
    except (OSError, ValueError):
        pass
    return result


def rss_bytes():
    try:
        with open("/proc/self/statm", encoding="ascii") as stream:
            return int(stream.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def fan_snapshot():
    result = []
    for directory in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            name = (directory / "name").read_text(encoding="ascii").strip()
        except OSError:
            name = None
        paths = sorted(directory.glob("fan*_input")) + sorted(directory.glob("pwm1"))
        for path in paths:
            try:
                result.append({"hwmon_name": name, "path": str(path),
                               "raw": path.read_text(encoding="ascii").strip()})
            except OSError as exc:
                result.append({"hwmon_name": name, "path": str(path), "error": str(exc)})
    return {"fields": result, "available": bool(result),
            "note": "Read-only sysfs raw fan/pwm values; absent paths mean unavailable, not fan stopped"}


def parse_line(raw):
    result = {"module_power_VDD_IN_current_mW": None,
              "module_power_VDD_IN_average_field_mW": None,
              "power_scope": POWER_SCOPE,
              "temperatures_C": {}, "board_ram_reported_MB": None,
              "board_ram_MiB": None,
              "board_ram_unit_note": "Raw tegrastats MB retained; no MiB conversion asserted"}
    rail = RAIL.search(raw)
    if rail:
        result["module_power_VDD_IN_current_mW"] = float(rail.group(1))
        result["module_power_VDD_IN_average_field_mW"] = float(rail.group(2))
    ram = re.search(r"\bRAM\s+(\d+)/(\d+)MB", raw)
    if ram:
        result["board_ram_reported_MB"] = {"used": int(ram.group(1)), "total": int(ram.group(2))}
    for name, value in re.findall(r"([A-Za-z0-9_]+)@(-?\d+(?:\.\d+)?)C\b", raw):
        result["temperatures_C"][name] = float(value)
    for name in ("GR3D_FREQ", "EMC_FREQ"):
        field = re.search(r"\b" + name + r"\s+(\S+)", raw)
        result[name + "_raw"] = field.group(1) if field else None
    return result


class Telemetry:
    """Keep all samples in RAM; terminate only the tegrastats PID we start."""
    def __init__(self):
        self.records = []
        self.rss_records = []
        self.errors = []
        self.process = None
        self.thread = None
        self.rss_thread = None
        self.stop_event = threading.Event()
        self.command = None

    def start(self):
        self.rss_thread = threading.Thread(target=self._rss_worker, daemon=True)
        self.rss_thread.start()
        executable = shutil.which("tegrastats")
        if executable is None:
            self.errors.append("tegrastats not found on PATH; no power values inferred")
            return
        self.command = [executable, "--interval", "1000"]
        try:
            self.process = subprocess.Popen(self.command, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, universal_newlines=True,
                                            bufsize=1)
            self.thread = threading.Thread(target=self._reader, daemon=True)
            self.thread.start()
        except OSError as exc:
            self.errors.append("tegrastats start failed: " + str(exc))

    def _rss_worker(self):
        while not self.stop_event.is_set():
            self.rss_records.append({"monotonic_ns": time.monotonic_ns(), "utc": utc_now(),
                                     "process_rss_bytes": rss_bytes(), "proc_meminfo": meminfo(),
                                     "fan_sysfs": fan_snapshot()})
            self.stop_event.wait(1.0)

    def _reader(self):
        try:
            for line in iter(self.process.stdout.readline, ""):
                stamp = time.monotonic_ns()
                raw = line.rstrip("\r\n")
                record = {"monotonic_ns": stamp, "utc": utc_now(), "raw": raw,
                          "process_rss_bytes": rss_bytes()}
                record.update(parse_line(raw))
                self.records.append(record)
        except Exception as exc:
            self.errors.append("tegrastats reader failed: " + str(exc))

    def stop(self):
        self.stop_event.set()
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            if self.thread is not None:
                self.thread.join(timeout=5)
            if self.process.stdout is not None:
                self.process.stdout.close()
        if self.rss_thread is not None:
            self.rss_thread.join(timeout=3)

    def window(self, start_ns, end_ns):
        duration = (end_ns - start_ns) / 1e9
        records = [r for r in self.records if start_ns <= r["monotonic_ns"] <= end_ns]
        valid = [r for r in records if r["module_power_VDD_IN_current_mW"] is not None]
        times = [r["monotonic_ns"] for r in valid]
        values_w = [r["module_power_VDD_IN_current_mW"] / 1000.0 for r in valid]
        gaps = [(b - a) / 1e9 for a, b in zip(times, times[1:])]
        span = (times[-1] - times[0]) / 1e9 if len(times) > 1 else 0.0
        first_gap = (times[0] - start_ns) / 1e9 if times else None
        last_gap = (end_ns - times[-1]) / 1e9 if times else None
        adequate = (duration > 0 and len(times) >= max(2, int(math.floor(duration * 0.8)))
                    and span / duration >= 0.8 and max(gaps or [float("inf")]) <= 2.5
                    and first_gap <= 2.5 and last_gap <= 2.5)
        sample_mean = statistics.mean(values_w) if values_w else None
        weighted = self._trapezoidal_mean(start_ns, end_ns)
        rss = [r["process_rss_bytes"] for r in self.rss_records
               if start_ns <= r["monotonic_ns"] <= end_ns and r["process_rss_bytes"] is not None]
        temperatures = {}
        ram_values = [r["board_ram_reported_MB"]["used"] for r in records
                      if r["board_ram_reported_MB"] is not None]
        ram_totals = sorted({r["board_ram_reported_MB"]["total"] for r in records
                             if r["board_ram_reported_MB"] is not None})
        for record in records:
            for name, value in record["temperatures_C"].items():
                temperatures.setdefault(name, []).append(value)
        trends = {}
        for name, values in temperatures.items():
            trends[name] = {"sample_count": len(values), "min_C": min(values), "max_C": max(values),
                            "first10_mean_C": statistics.mean(values[:10]),
                            "last10_mean_C": statistics.mean(values[-10:]),
                            "last10_minus_first10_C": statistics.mean(values[-10:]) - statistics.mean(values[:10])}
        return {"start_monotonic_ns": start_ns, "end_monotonic_ns": end_ns, "duration_s": duration,
                "tegrastats_lines": len(records), "valid_current_power_samples": len(valid),
                "module_power_scope": POWER_SCOPE, "rail_used": "VDD_IN current mW only",
                "power_sample_span_s": span, "power_sample_span_fraction": span / duration if duration else 0,
                "first_sample_gap_s": first_gap, "last_sample_gap_s": last_gap,
                "max_internal_sample_gap_s": max(gaps) if gaps else None,
                "power_coverage_adequate": adequate,
                "module_power_VDD_IN_sample_mean_W_unqualified": sample_mean,
                "module_power_VDD_IN_mean_W": sample_mean if adequate else None,
                "module_power_VDD_IN_trapezoidal_mean_W": weighted,
                "primary_power_method": "Arithmetic mean of every valid current VDD_IN sample received inside the complete window; qualified only with >=80% span/count and <=2.5s edge/internal gaps",
                "weighted_method": "Auxiliary linear interpolation at exact boundaries using bracketing samples <=2.5s apart, trapezoidal integration; unavailable if unbracketed or gaps >2.5s. Receipt timestamps, not hardware sample timestamps.",
                "process_RSS_sample_count": len(rss), "process_RSS_sampled_peak_MiB": max(rss) / MIB if rss else None,
                "board_RAM_peak_reported_MB": max(ram_values) if ram_values else None,
                "board_RAM_totals_reported_MB": ram_totals,
                "board_RAM_peak_MiB": None,
                "board_RAM_unit_note": "Original tegrastats MB retained; no MiB unit assertion or sum with allocator/RSS",
                "temperatures": trends,
                "clock_or_thermal_stability_asserted": False}

    def _trapezoidal_mean(self, start_ns, end_ns):
        points = [(r["monotonic_ns"], r["module_power_VDD_IN_current_mW"] / 1000.0)
                  for r in self.records if r["module_power_VDD_IN_current_mW"] is not None]
        before = [p for p in points if p[0] <= start_ns]
        after = [p for p in points if p[0] >= end_ns]
        if not before or not after or end_ns <= start_ns:
            return None
        inside = [p for p in points if start_ns < p[0] < end_ns]
        selected = [before[-1]] + inside + [after[0]]
        if any((b[0] - a[0]) / 1e9 > 2.5 or b[0] <= a[0] for a, b in zip(selected, selected[1:])):
            return None
        def interpolate(left, right, stamp):
            if left[0] == right[0]:
                return left[1]
            return left[1] + (right[1] - left[1]) * (stamp - left[0]) / (right[0] - left[0])
        if inside:
            left = (start_ns, interpolate(selected[0], selected[1], start_ns))
            right = (end_ns, interpolate(selected[-2], selected[-1], end_ns))
            clipped = [left] + inside + [right]
        else:
            clipped = [(start_ns, interpolate(selected[0], selected[-1], start_ns)),
                       (end_ns, interpolate(selected[0], selected[-1], end_ns))]
        energy = sum((a[1] + b[1]) * 0.5 * (b[0] - a[0]) / 1e9 for a, b in zip(clipped, clipped[1:]))
        return energy / ((end_ns - start_ns) / 1e9)
