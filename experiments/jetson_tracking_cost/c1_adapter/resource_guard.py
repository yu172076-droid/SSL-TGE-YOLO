"""Resource observer extracted unchanged from accepted short-video and streamsync helpers."""
import datetime
import json
from pathlib import Path
import threading
import time
GIB = 1024 ** 3
VM_KEYS = ('pswpin','pswpout','pgmajfault')
VM_COUNTERS = VM_KEYS

def require(ok, message):
    if not ok:
        raise ValueError(message)

class ResourceStop(Exception):
    pass

class Monitor:
    """Local normal-user memory observer only; no fake root readiness or power."""
    def __init__(self, sampler, sample_path=None):
        self.sampler = sampler
        self.samples = []
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.phase, self.minimum = "preflight", GIB
        self.failure = None
        self.previous_interval = None
        self.streak = 0
        self.sample_path = sample_path
        if sample_path is not None:
            sample_path.open("x", encoding="utf-8").close()

    def set_phase(self, phase, minimum=GIB):
        with self.lock:
            self.phase, self.minimum = phase, minimum

    def check(self):
        with self.lock:
            if self.failure is not None:
                raise self.failure

    def tick(self, phase=None, minimum=None):
        with self.lock:
            self.check()
            row = self.sampler(phase or self.phase)
            threshold = self.minimum if minimum is None else minimum
            row["gate"] = dict(minimum_MemAvailable_bytes=threshold,
                               passed=row["proc_meminfo"]["MemAvailable_bytes"] >= threshold)
            if self.samples:
                row["actual_sample_gap_s"] = (row["monotonic_ns"] - self.samples[-1]["monotonic_ns"]) / 1e9
            self.samples.append(row)
            try:
                if not row["gate"]["passed"]:
                    raise ResourceStop("{} MemAvailable below {} bytes".format(row["phase"], threshold))
                previous = self.previous_interval
                if previous is None or row["monotonic_ns"] - previous["monotonic_ns"] >= 750000000:
                    if previous is not None:
                        delta = {k: row["vm_counters"][k] - previous["vm_counters"][k] for k in VM_KEYS}
                        row["vm_counter_deltas"] = delta
                        if any(v < 0 for v in delta.values()):
                            raise ResourceStop("Global vmstat counters decreased")
                        self.streak = self.streak + 1 if delta["pswpin"] > 0 or delta["pswpout"] > 0 else 0
                        row["consecutive_positive_global_swap_intervals"] = self.streak
                        if self.streak >= 2:
                            raise ResourceStop("Two consecutive positive global swap intervals; no process attribution")
                    self.previous_interval = row
            except Exception as exc:
                self.failure = exc
                raise
            finally:
                if self.sample_path is not None:
                    with self.sample_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            return row

    def start(self):
        require(self.thread is None, "Observer already started")
        def observe():
            while not self.stop_event.is_set():
                started = time.monotonic()
                try:
                    self.tick()
                except Exception as exc:
                    with self.lock:
                        if self.failure is None:
                            self.failure = exc
                self.stop_event.wait(max(.01, 1 - (time.monotonic() - started)))
        self.thread = threading.Thread(target=observe, daemon=True, name="video-short-memory")
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
            require(not self.thread.is_alive(), "Observer failed to stop")

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

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
