"""Offline standard-library binding of one complete video measurement and power.

No model/device/subprocess execution. An incomplete binding exits 3 and preserves
diagnostics in an exclusively created output directory. Qualified power uses the
unchanged, hash-pinned historical Telemetry.window implementation.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import statistics
import sys
import traceback


ROOT = Path(__file__).absolute().parent
OLD_MANIFEST_SHA = "84cadaf73d7073d1fac479ef8795571b2e33534fbf58e8b3795f2ae674fea510"
OLD_TELEMETRY_SHA = "2468ee031acb08e55f4d421ffa4f11e95b753f5c9d06f66033ceb2983f5249b9"
STAGE_B_SHA = "8754af37bbbb84d8744f52dada039cb649ce2237b8f634999faeb06bb0b54c53"
PROBE_SHA = "1f1cb9976d070d97b522f00cb820dadd2162da9efa6b508022229d8c32d27705"
SMOKE_SHA = "78636ba545e2b668919db32b48b5f5a7455c6a7869bd574436092bccd3eacad6"
WEIGHT_SHA = "42223d181d2fa79f96e6d1d67c4cd3d0faecbad1335fb274d4a6d9fbe9ccde93"
PAYLOADS = {"root_sampler.py", "video_formal.py", "bind_results.py", "verify_package.py", "README.md"}
GIB = 1024 ** 3
VM_COUNTERS = ("pswpin", "pswpout", "pgmajfault")
PROTOCOL = "native_pair_complete_passes_online_tracking_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def no_links(path, must_exist=True):
    """Reject symlinks and Windows junction/reparse points before resolution."""
    path = Path(os.path.abspath(str(path)))
    for part in list(reversed(path.parents)) + [path]:
        try:
            meta = os.lstat(str(part))
        except FileNotFoundError:
            if must_exist or part != path:
                raise
            continue
        require(not stat.S_ISLNK(meta.st_mode) and not (getattr(meta, "st_file_attributes", 0) & 0x400),
                "Symlink/reparse path rejected: " + str(part))
    return path


def relative_file(root, relative):
    require(isinstance(relative, str) and relative and "\\" not in relative and ":" not in relative,
            "Invalid relative artifact name")
    parts = relative.split("/")
    require(not PurePosixPath(relative).is_absolute() and all(p not in ("", ".", "..") for p in parts),
            "Artifact path traversal rejected: " + relative)
    path = no_links(root.joinpath(*parts))
    require(path.is_file() and stat.S_ISREG(os.lstat(str(path)).st_mode), "Artifact is not a regular file: " + relative)
    return path


def sha(path):
    path = no_links(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("Non-finite JSON constant: " + value)


def decode_json(raw):
    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)


def read_json(path):
    return decode_json(no_links(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def jsonl(path):
    with no_links(path).open(encoding="utf-8") as stream:
        return [decode_json(line) for line in stream if line.strip()]


def number(value, positive=False):
    require(not isinstance(value, bool), "Boolean supplied as numeric value")
    result = float(value)
    require(math.isfinite(result) and (not positive or result > 0), "Invalid numeric sample")
    return result


def node_number(node, positive=False):
    if not isinstance(node, dict) or "raw" not in node:
        return None
    try:
        return number(node["raw"].strip(), positive)
    except (ValueError, TypeError, AttributeError):
        return None


def file_rows(manifest):
    rows = manifest.get("files")
    if isinstance(rows, dict):
        return [dict(path=key, sha256=value) for key, value in rows.items()]
    require(isinstance(rows, list), "Package file manifest must be a list/dict")
    return rows


def verify_package(directory, expected, new=True):
    spec = importlib.util.spec_from_file_location("c4_offline_verify", str(ROOT / "verify_package.py"))
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    manifest = verifier.verify(directory, expected)
    return manifest, {r["path"]: r["sha256"] for r in manifest["files"]}


def sidecar(directory, filename, expected=None):
    path = relative_file(directory, filename)
    actual = sha(path)
    fields = relative_file(directory, Path(filename).with_suffix(".sha256").as_posix()).read_text("ascii").split()
    require(len(fields) == 2 and fields[0] == actual and fields[1].lstrip("*") == Path(filename).name,
            "Invalid SHA sidecar: " + filename)
    require(expected is None or actual == expected, "Unexpected pinned SHA: " + filename)
    return actual


def report_artifacts(directory, filename, bindings):
    report_sha = sidecar(directory, filename)
    report = read_json(relative_file(directory, filename))
    rows = report.get("artifacts")
    require(isinstance(rows, list), "Missing artifact manifest: " + filename)
    seen = set()
    for row in rows:
        relative = row["path"]
        require(relative not in seen and relative not in (filename, str(Path(filename).with_suffix(".sha256"))),
                "Duplicate/self-referential artifact manifest")
        seen.add(relative)
        path = relative_file(directory, relative)
        actual = sha(path)
        require(actual == row["sha256"] and path.stat().st_size == row["bytes"], "Returned artifact binding mismatch: " + relative)
        bindings.append(dict(root=str(directory), path=relative, sha256=actual, bytes=path.stat().st_size))
    # Walk without following links, rejecting even unlisted link/reparse nodes.
    for parent, dirs, files in os.walk(str(directory), followlinks=False):
        for name in dirs + files:
            no_links(Path(parent) / name)
    bindings.append(dict(root=str(directory), path=filename, sha256=report_sha, bytes=(directory / filename).stat().st_size))
    return report, report_sha, seen


def identity(row, session, boot, manifest=None):
    require(row.get("session_id") == session and row.get("boot_id") == boot and row.get("euid") == 0,
            "Root session/boot/euid identity mismatch")
    if manifest is not None:
        require(row.get("manifest_sha256") == manifest, "Root manifest identity mismatch")


def ordered(rows, name):
    require(rows and all(isinstance(row.get("monotonic_ns"), int) and row["monotonic_ns"] > 0 for row in rows),
            name + " lacks monotonic timestamps")
    require(all(a["monotonic_ns"] < b["monotonic_ns"] for a, b in zip(rows, rows[1:])), name + " timestamps not strictly increasing")


def sample_gap_summary(stamps, start=None, end=None):
    gaps = [(b - a) / 1e9 for a, b in zip(stamps, stamps[1:])]
    return dict(count=len(stamps), actual_max_gap_s=max(gaps) if gaps else None,
                gaps_over_1_5s=sum(x > 1.5 for x in gaps), strict_1Hz_asserted=False,
                first_gap_s=(stamps[0] - start) / 1e9 if stamps and start is not None else None,
                last_gap_s=(end - stamps[-1]) / 1e9 if stamps and end is not None else None,
                sample_span_s=(stamps[-1] - stamps[0]) / 1e9 if len(stamps) > 1 else 0.0)


def frequency_summary(nodes, unit):
    valid = [node_number(node, True) for node in nodes]
    finite = [x for x in valid if x is not None]
    return dict(unit=unit, count=len(nodes), valid_count=len(finite), invalid_count=len(nodes) - len(finite),
                minimum=min(finite) if finite else None, maximum=max(finite) if finite else None,
                raw_values=[node.get("raw") if isinstance(node, dict) else None for node in nodes],
                DVFS_observation_only=True, lock_or_stability_asserted=False)


def proc_values(raw, colon=False):
    result = {}
    for line in raw.splitlines():
        parts = line.split(":" if colon else None, 1)
        if len(parts) == 2:
            result[parts[0]] = parts[1].strip()
    return result



def validate_root(root, infer_report, root_report, payload, expected, telemetry_module):
    require(root_report.get("schema") == "jetson_video_formal_root_sampler_v1" and root_report.get("euid") == 0
            and root_report.get("isolated_python_flag") == 1 and root_report.get("status") == "stopped_by_operator"
            and root_report.get("exitcode") == 0 and root_report.get("max_sampling_seconds") == 1800, "Root sampler not qualified/stopped")
    require(root_report.get("manifest_sha256") == expected and root_report.get("script_sha256") == payload["root_sampler.py"], "Root package/source mismatch")
    require(sha(relative_file(root, "source_root_sampler_snapshot.py")) == payload["root_sampler.py"]
            and sha(relative_file(root, "reviewed_delivery_manifest.json")) == expected, "Root frozen snapshot mismatch")
    ready_sha = sidecar_value = sha(relative_file(root, "ready.json"))
    ready = read_json(relative_file(root, "ready.json"))
    session, boot = root_report.get("session_id"), root_report.get("boot_id")
    require(isinstance(session, str) and len(session) >= 16 and isinstance(boot, str) and boot, "Missing session/boot IDs")
    identity(root_report, session, boot, expected)
    identity(ready, session, boot, expected)
    require(root_report.get("ready") == ready == infer_report.get("root_ready")
            and root_report.get("ready_sha256") == ready_sha == infer_report.get("root_ready_sha256")
            and infer_report.get("root_session_id") == session and infer_report.get("boot_id") == boot,
            "Ready SHA/session/boot cross-binding mismatch")
    require(ready.get("script_sha256") == payload["root_sampler.py"] and ready.get("mode_query_ok") is True
            and ready.get("clock_query_ok") is True, "Root readiness incomplete")
    live = read_json(relative_file(root, "live.json"))
    identity(live, session, boot, expected)
    require(live.get("sampler_status") == "stopped_by_operator" and live.get("sampler_pid") == ready.get("sampler_pid"), "Final live status/PID mismatch")
    queries = root_report.get("queries", [])
    required_queries = {"start_nvpmodel_query", "start_jetson_clocks_show", "end_nvpmodel_query", "end_jetson_clocks_show"}
    require(len(queries) == 4 and {q.get("name") for q in queries} == required_queries
            and all(q.get("returncode") == 0 and q.get("euid") == 0 and q.get("stdout", "").strip() for q in queries),
            "Root mode/clock queries incomplete or failed")
    modes = [q["stdout"].strip() for q in queries if q["name"].endswith("nvpmodel_query")]
    require(len(modes) == 2 and modes[0] == modes[1] and re.search(r"(?m)^NV Power Mode:\s*15W\s*$", modes[0]), "Power mode changed or differs from 15W")
    require(root_report.get("reader_error") is None and root_report.get("cleanup_error") is None
            and root_report.get("final_live_error") is None, "Root sampler cleanup/reader failed")
    require(isinstance(root_report.get("tegrastats_command", {}).get("returncode"), int), "Root telemetry subprocess has not exited")
    root_rows = jsonl(relative_file(root, "root_tegrastats.jsonl"))
    sysfs = jsonl(relative_file(root, "root_sysfs_samples.jsonl"))
    ordered(root_rows, "Root tegrastats")
    ordered(sysfs, "Root sysfs")
    lower, upper = root_report["sampling_started_monotonic_ns"], root_report["sampling_finished_monotonic_ns"]
    require(root_report["started_monotonic_ns"] <= lower < upper <= root_report["finished_monotonic_ns"], "Root sampler bounds invalid")
    for row in root_rows + sysfs:
        identity(row, session, boot)
        require(lower <= row["monotonic_ns"] <= upper, "Root sample outside actual sampler lifecycle")
    require(len(root_rows) == root_report["tegrastats_line_count"] and len(sysfs) == root_report["sysfs_sample_count"], "Root row count mismatch")
    require("".join(row["raw"] for row in root_rows) == relative_file(root, "root_tegrastats.stdout.txt").read_text("utf-8"), "Root raw stream differs from JSONL")
    binding = ready["vdd_in_binding"]
    require(binding == root_report.get("vdd_in_binding") and binding.get("name") == "ina3221"
            and binding.get("channel") == 1 and binding.get("label") == "VDD_IN"
            and "/1-0040/" in binding.get("sensor_resolved_path", "")
            and binding.get("driver_resolved_path") == "/sys/bus/i2c/drivers/ina3221"
            and binding.get("voltage_path", "").endswith("/in1_input")
            and binding.get("current_path", "").endswith("/curr1_input"), "Exact INA3221 VDD_IN binding differs")
    parsed_root, invalid_temperature_count, positive = [], 0, 0
    for row in root_rows:
        parsed = telemetry_module.parse_line(row["raw"])
        p = parsed.get("module_power_VDD_IN_current_mW")
        if p is not None and (not math.isfinite(p) or p <= 0):
            parsed["module_power_VDD_IN_current_mW"] = None
        elif p is not None:
            positive += 1
        valid_temps = {}
        for name, value in parsed["temperatures_C"].items():
            if math.isfinite(value) and value != -256.0 and -100 < value < 200:
                valid_temps[name] = value
            else:
                invalid_temperature_count += 1
        parsed["temperatures_C"] = valid_temps
        parsed.update(monotonic_ns=row["monotonic_ns"], process_rss_bytes=None)
        parsed_root.append(parsed)
    require(positive == root_report["positive_VDD_IN_line_count"] and positive > 0, "Root current-power count mismatch")
    require(lower <= ready["ready_monotonic_ns"] <= upper, "Root ready outside sampling lifecycle")
    for q in queries:
        option = "-q" if q["name"].endswith("nvpmodel_query") else "--show"
        require(q.get("target_argv", [None])[-1] == option and len(q["target_argv"]) == 2, "Root query is not read-only allowlisted")
    return session, boot, ready, ready_sha, lower, upper, root_rows, sysfs, parsed_root, invalid_temperature_count


def validate_memory(rows, report, session, boot, expected, ready, payload):
    ordered(rows, "Process memory")
    observer = report["memory_observer"]
    require(observer["failure"] is None and observer["samples"] == len(rows) and observer["thread_started"] is True and observer["thread_stopped"] is True, "Resource observer incomplete/failed")
    required = {"before_model_load", "model_loaded_before_decoder_open", "decoder_capacity_after_baseline", "first_decoded_frame_before_first_forward", "postwarm", "post_measurement", "right_bracket_after_measurement"}
    require(required <= {r["phase"] for r in rows}, "Mandatory resource gates absent")
    require({r["phase"] for r in rows} <= required | {"baseline", "decoder_open_before_first_forward", "warmup", "measurement"}, "Unknown resource phase")
    import startup_monitor
    activation=report["formal_guard_activation"]
    require(report["startup_policy"]==activation["policy"]==startup_monitor.POLICY,"Startup policy mismatch")
    require(report["windows"]["warmup"]["end_monotonic_ns"]<=activation["monotonic_ns"]<=report["windows"]["measurement"]["start_monotonic_ns"],"Formal guard not activated before measurement")
    before=[r for r in rows if r["monotonic_ns"]<activation["monotonic_ns"]]
    proof=startup_monitor.stable_tail(before,report["windows"]["warmup"]["start_monotonic_ns"])
    require(proof==activation["stability"] and proof["ready"],"Stable-tail proof differs")
    initial_oom=startup_monitor.oom(rows[0])
    require(initial_oom is not None,"Missing initial OOM counter")
    previous, streak = None, 0
    for i, row in enumerate(rows):
        require(row["startup_policy"]==startup_monitor.POLICY and type(row["paging_failure_enforced"]) is bool
            and row["paging_failure_enforced"]==(row["monotonic_ns"]>=activation["monotonic_ns"])
            and row["oom_kill_counter"]==startup_monitor.oom(row)==initial_oom,"Phase enforcement or OOM evidence differs")
        threshold = 3 * GIB + 24883200 if row["phase"] == "decoder_capacity_after_baseline" else (3 * GIB if row["phase"] in ("model_loaded_before_decoder_open", "decoder_open_before_first_forward", "first_decoded_frame_before_first_forward") else GIB)
        available = row["proc_meminfo"]["MemAvailable_bytes"]
        require(row["gate"]["minimum_MemAvailable_bytes"] == threshold and row["gate"]["passed"] is True and available >= threshold, "Memory gate changed/failed")
        mem = proc_values(row["raw"]["meminfo"], True)
        vm = proc_values(row["raw"]["vmstat"])
        status = proc_values(row["raw"]["self_status"], True)
        require(int(mem["MemAvailable"].split()[0]) * 1024 == available and all(int(vm[k]) == row["vm_counters"][k] for k in VM_COUNTERS), "Raw memory/vm binding differs")
        for key in ("VmRSS", "VmHWM", "VmSwap"):
            if key in status:
                require(int(status[key].split()[0]) * 1024 == row["process_status"][key + "_bytes"], "Raw process memory differs")
        tail = row["raw"]["self_stat"].rsplit(")", 1)[1].split()
        require(row["process_faults"] == {"minflt": int(tail[7]), "majflt": int(tail[9])}, "Raw process faults differ")
        if i:
            require(math.isclose(number(row["actual_sample_gap_s"]), (row["monotonic_ns"] - rows[i - 1]["monotonic_ns"]) / 1e9, abs_tol=1e-9), "Observer gap differs")
        check = row["root_live_check"]
        live = check["live"]
        identity(live, session, boot, expected)
        require(live["script_sha256"] == payload["root_sampler.py"] and live["sampler_pid"] == ready["sampler_pid"]
                and live["ready_written"] is True and live["sampler_status"] == "collecting" and live["last_sysfs_valid_gpu_emc"] is True
                and live["last_sysfs_valid_vdd_pair"] is True and live.get("power_line_count", 0) >= 1 and live.get("sysfs_sample_count", 0) >= 1, "Root live check incomplete")
        require(hashlib.sha256((json.dumps(live, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")).hexdigest() == check["live_sha256"], "Captured live hash mismatch")
        for key in ("last_tegrastats_monotonic_ns", "last_power_tegrastats_monotonic_ns", "last_sysfs_monotonic_ns"):
            require(0 <= check["checked_monotonic_ns"] - live[key] <= 3500000000, "Stale root sampling")
        if previous is None or row["monotonic_ns"] - previous["monotonic_ns"] >= 750000000:
            if previous is not None:
                delta = {k: row["vm_counters"][k] - previous["vm_counters"][k] for k in VM_COUNTERS}
                require(all(v >= 0 for v in delta.values()) and delta == row["vm_counter_deltas"], "VM delta changed/decreased")
                streak = streak + 1 if delta["pswpin"] > 0 or delta["pswpout"] > 0 else 0
                require((not row["paging_failure_enforced"] or streak < 2) and row["consecutive_positive_global_swap_intervals"] == streak, "Formal consecutive global swap guard failed")
            previous = row
    return dict(startup_policy=startup_monitor.POLICY, formal_guard_activation=activation, minimum_MemAvailable_bytes=min(r["proc_meminfo"]["MemAvailable_bytes"] for r in rows), maximum_sampled_RSS_MiB=max(r["process_status"].get("VmRSS_bytes", 0) for r in rows) / 1024 ** 2,
                actual_sampling=sample_gap_summary([r["monotonic_ns"] for r in rows]), global_swap_scope="Global counters; allowed isolated traffic is retained, no process attribution")


def validate_video_frames(infer, phase, window, driver):
    with relative_file(infer, phase + "_frames.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    predictions = jsonl(relative_file(infer, phase + "_predictions.jsonl"))
    require(len(rows) == len(predictions) == window["frames"] and len(rows) >= (751 if phase == "warmup" else 300), "Frame/prediction count differs")
    previous_end = window["start_monotonic_ns"]
    numeric = []
    for i, (r, p) in enumerate(zip(rows, predictions), 1):
        require(r["phase"] == p["phase"] == phase and int(r["iteration"]) == p["iteration"] == i
                and int(r["pass_index"]) == p["pass_index"] and int(r["source_decode_ordinal"]) == p["source_decode_ordinal"] == int(r["image_id"]), "Frame ordinal/prediction identity differs")
        begin, read_end, process_start, process_end, end = [int(r[k]) for k in ("pipeline_start_monotonic_ns", "read_end_monotonic_ns", "start_monotonic_ns", "end_monotonic_ns", "pipeline_end_monotonic_ns")]
        require(previous_end <= begin == int(r["read_start_monotonic_ns"]) <= read_end <= process_start <= process_end <= end <= window["end_monotonic_ns"], "Frame timing order/bounds invalid")
        previous_end = end
        for k, want in (("input_read_decode_ms", (read_end - begin) / 1e6), ("processing_ms", (process_end - process_start) / 1e6), ("pipeline_ms", (end - begin) / 1e6)):
            require(math.isclose(number(r[k]), want, abs_tol=1e-7), "Frame timing differs: " + k)
        require(0 <= number(r["forward_cuda_event_ms"]) <= number(r["processing_ms"]) + 1 and int(r["native_frame_bytes"]) == 24883200
                and r["all_outputs_finite"] == "True" and int(r["detection_count"]) == len(p["detections_xyxy_conf_class"]), "Frame native size/finite/forward/count differs")
        driver.validate_values(p["detections_xyxy_conf_class"])
        payload=p["tracking_payload"];ordinal=int(r["source_decode_ordinal"])
        require(payload["clip"]==("R04" if ordinal<=750 else "R02") and payload["clip_frame_1"]==(ordinal-1)%750+1,"Online clip/frame mapping")
        require(int(r["track_count"])==len(payload["tracks"]),"Online output count")
        require(payload["mode"]==driver.EXPECTED_MODE,"Online mode differs from scheduled mode")
        require(payload["tracker_reset_requested"]==(ordinal in (1,751) and payload["mode"]!="detector_only"),"Online reset marker differs")
        require(len({t[4] for t in payload["tracks"]})==len(payload["tracks"]),"Duplicate online track IDs")
        if payload["mode"]=="detector_only":require(not payload["tracks"],"Detection-only emitted tracks")
        for t in payload["tracks"]:
            require(len(t)==7 and all(math.isfinite(v) for v in t) and 0<=t[0]<t[2]<=3840 and 0<=t[1]<t[3]<=2160 and t[4]>=1 and t[4]==int(t[4]) and t[6]==0,"Online output geometry/ID")
        numeric.append({k: number(r[k]) for k in ("processing_ms", "input_read_decode_ms", "pipeline_ms", "forward_cuda_event_ms", "tracking_cpu_ms", "combined_processing_ms")})
    consumed, previous_close = 0, window["start_monotonic_ns"]
    for index, pr in enumerate(window["passes"], 1):
        count = pr["frames"]
        group = rows[consumed:consumed + count]
        require(pr["pass_index"] == index and 0 < count <= 1500 and len(group) == count
                and [int(r["source_decode_ordinal"]) for r in group] == list(range(1, count + 1))
                and all(int(r["pass_index"]) == index for r in group), "Pass order/frame coverage differs")
        require(previous_close <= pr["open_start_monotonic_ns"] <= pr["open_end_monotonic_ns"] <= int(group[0]["read_start_monotonic_ns"])
                and int(group[-1]["pipeline_end_monotonic_ns"]) <= pr["close_end_monotonic_ns"] <= pr["record_flush_end_monotonic_ns"] <= window["end_monotonic_ns"], "Pass open/close/flush outside timing window")
        if pr["complete"] or phase == "measurement":
            require(pr["complete"] is True and pr["eof_seen"] is True and count == 1500
                    and int(group[-1]["pipeline_end_monotonic_ns"]) <= pr["eof_read_start_monotonic_ns"] <= pr["eof_read_end_monotonic_ns"] <= pr["close_end_monotonic_ns"], "Formal pass incomplete/missing EOF")
        else:
            require(index == len(window["passes"]) and phase == "warmup", "Partial non-final warmup pass")
        previous_close = pr["record_flush_end_monotonic_ns"]
        consumed += count
    require(consumed == len(rows), "Pass count does not cover frames")
    return rows, driver.latency(numeric)


def audit(args):
    expected = args.expected_manifest_sha256
    root, infer = no_links(args.root_session), no_links(args.infer_output)
    _, payload = verify_package(ROOT, expected, True)
    spec = importlib.util.spec_from_file_location("c4_offline_power", str(ROOT / "telemetry.py"))
    telemetry_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(telemetry_module)
    spec = importlib.util.spec_from_file_location("c4_offline_driver", str(ROOT / "video_formal.py"))
    driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
    short, stream = driver.libraries()
    job, accepted, receipt = short.qualification(args, verify_actual=False)
    driver.validate_values = short.validate_values
    driver.EXPECTED_MODE=args.mode
    bindings = []
    rr, root_sha, _ = report_artifacts(root, "root_sampler_report.json", bindings)
    launcher, launcher_sha, assets = report_artifacts(infer, "launcher_report.json", bindings)
    infer_sha = sidecar(infer, "infer_report.json")
    ir = read_json(relative_file(infer, "infer_report.json"))
    require({"infer_report.json", "infer_report.sha256", "worker_console.txt", "measurement_window.json", "process_memory_samples.jsonl"} <= assets, "Missing worker evidence")
    require(launcher["schema"] == "jetson_video_formal_launcher_v1" and ir["schema"] == "jetson_video_formal_inference_v1" and launcher["child_exitcode"] == launcher["launcher_exitcode"] == ir["actual_self_exit_code"] == 0 and ir["benchmark_completed"] is True and launcher["nms_time_limit_warning"] is False and not launcher.get("watchdog_triggered"), "Unfinished launcher/worker")
    require("NMS time limit" not in relative_file(infer,"worker_console.txt").read_text(encoding="utf-8"), "NMS warning")
    require(launcher["normal_euid"] == ir["euid"] != 0 and launcher["child_pid"] == ir["pid"] and launcher["output"] == ir["output"] and launcher["root_session"] == ir["root_session_path"] == rr["output"] and launcher["manifest_sha256"] == ir["manifest_sha256"] == expected and launcher["script_sha256"] == ir["script_sha256"] == payload["video_formal.py"], "Source/process/session identity")
    require(launcher["mode"]==ir["mode"]==args.mode,"Mode identity differs")
    from types import SimpleNamespace
    an=driver.admission_name(SimpleNamespace(mode=args.mode,replicate=args.replicate))
    ad=read_json(relative_file(infer,"source_snapshot/admission/"+an))
    sidecar(infer,"source_snapshot/admission/"+an)
    require(ad["schema"]=="tracking_f1_admission_v1" and ad["manifest_sha256"]==expected and ad["mode"]==args.mode
            and ad["replicate"]==args.replicate and ad["minimum_quiet_seconds"]==600 and ad["boot_id"]==ir["boot_id"]
            and launcher["started_monotonic_ns"]-ad["prepared_monotonic_ns"]>=600*10**9,"Formal admission identity/600s")
    ctx=launcher["R2_startup_context"];before=ad["startup_context"]["memory"]["vm_counters"];after=ctx["memory"]["vm_counters"]
    quiet={k:after[k]-before[k] for k in before}
    require(quiet==ctx["quiet_interval_global_deltas"] and min(quiet.values())>=0 and quiet["pswpin"]==quiet["pswpout"]==0
            and ctx["firefox_tab_state"]==ad["startup_context"]["firefox_tab_state"]=="single-about-blank"
            and ctx["firefox_main_processes"] and ad["startup_context"]["firefox_main_processes"],"Formal quiet/Firefox declaration differs")
    require(launcher["job"] == ir["job"] == args.job and launcher["replicate"] == ir["replicate"] == args.replicate and launcher["weight_sha256"] == ir["weight_sha256"] == job["sha256"] and launcher["video_sha256"] == ir["video_sha256"] == driver.VIDEO_SHA and ir["video_unchanged"] is True, "Model/video/replicate differs")
    require(ir["input_protocol"] == PROTOCOL and ir["AP_measured"] is False and ir["tracking_executed"]==(args.mode!="detector_only") and ir["GT_used_by_tracker"] is False and ir["combined_same_process"] is True and ir["integrated_warmup_compatibility_passed"] is True and ir["TensorRT_executed"] is False and ir["replicate_count"] == 1, "Wrong scope")
    require(ir["config"] == dict(batch=1,input_shape=[1,3,640,640],precision="FP16",conf=.001,mode=args.mode,iou=.65,max_det=300,native_dimensions=[3840,2160],expected_frames_per_pass=1500,firefox_must_remain_open=True,backend="FFMPEG"), "Inference configuration")
    require(ir["environment"] == accepted["environment"] and ir["model"] == accepted["model"] and ir["models"] == {"FP16":accepted["model"]} and ir["runtime_flags"] == short.c1.RUNTIME_FLAGS, "Qualified runtime/model differs")
    require(ir["bundle_manifest_sha256"] == short.c1.BUNDLE_SHA and ir["input_manifest_sha256"] == short.c1.INPUT_SHA and ir["adapter_manifest_sha256"] == short.C1_SHA and ir["accepted_c2_report_sha256"] == short.C2_SHA and ir["accuracy_summary_sha256"] == receipt["accuracy_summary_sha256"] and ir["c3_parent_sha256"] == receipt["c3_parent_sha256"], "C1/C2/C3 provenance")
    require(ir["_forward_gate"] == dict(open=False,intercepted_calls=ir["forward_calls_started"]), "Unexpected forward")
    for name,pin in payload.items():
        require(sha(relative_file(infer,"source_snapshot/new_formal/"+name)) == pin, "C4 source snapshot differs")
    require(sha(relative_file(infer,"source_snapshot/new_formal/delivery_manifest.json")) == expected, "C4 manifest snapshot")
    require(sha(relative_file(infer,"source_snapshot/bundle/delivery_manifest.json")) == short.c1.BUNDLE_SHA, "Bundle snapshot manifest")
    bundle = read_json(relative_file(infer,"source_snapshot/bundle/delivery_manifest.json"))
    table = {r["path"]:r["sha256"] for r in bundle["files"]}
    for path,pin in table.items():
        if not path.startswith("models/"):
            require(sha(relative_file(infer,"source_snapshot/bundle/"+path)) == pin, "Bundle source snapshot")
    require(table[job["weight"]] == job["sha256"], "Bundle selected weight")
    require(sha(relative_file(infer,"source_snapshot/inputs/input_manifest.json")) == short.c1.INPUT_SHA, "Input manifest snapshot")
    for where in ("source_origins_before_load","source_origins_after_inference"):
        for name,origin in ir[where].items():
            prefix = "/home/nvidia/ssl_tge_yolo_edge_20261009/jetson_models_b80_s101_20261013/"
            require(origin["path"].startswith(prefix) and table.get(origin["path"][len(prefix):]) == origin["sha256"], "Loaded source origin differs")
    session, boot, ready, ready_sha, lower, upper, root_rows, sysfs, parsed, excluded = validate_root(root, ir, rr, payload, expected, telemetry_module)
    memory = jsonl(relative_file(infer, "process_memory_samples.jsonl"))
    resources = validate_memory(memory, ir, session, boot, expected, ready, payload)
    historical = telemetry_module.Telemetry()
    historical.records = parsed
    historical.rss_records = [dict(monotonic_ns=r["monotonic_ns"], process_rss_bytes=r["process_status"].get("VmRSS_bytes")) for r in memory]
    windows = ir["windows"]
    require(set(windows) == {"baseline", "warmup", "measurement"} and windows == read_json(relative_file(infer, "measurement_window.json")), "Window artifact/report differs")
    baseline = windows["baseline"]
    require(baseline["model_loaded"] is True and baseline["capture_closed"] is True and baseline["native_frames_retained"] == baseline["frames"] == 0 and baseline["inference_executed"] is False, "Baseline is not loaded-idle")
    summaries, previous_end, calls = {}, ready["ready_monotonic_ns"], 0
    for phase in ("baseline", "warmup", "measurement"):
        w = windows[phase]
        start, end = w["start_monotonic_ns"], w["end_monotonic_ns"]
        duration = (end - start) / 1e9
        require(w["complete"] is True and lower <= previous_end <= start < end <= upper
                and launcher["started_monotonic_ns"] <= start < end <= launcher["ended_monotonic_ns"]
                and 60 <= duration <= (90 if phase == "baseline" else (300 if phase == "warmup" else 600))
                and math.isclose(duration, number(w["duration_s"]), abs_tol=1e-8), "Window duration/sequence/source bounds invalid")
        previous_end = end
        power = historical.window(start, end)
        require(power["power_coverage_adequate"] is True and power["module_power_VDD_IN_mean_W"] is not None, "Same-window VDD_IN coverage inadequate: " + phase)
        inside = [r for r in sysfs if start <= r["monotonic_ns"] <= end]
        require(inside, "Same-window sysfs observations missing")
        for r in inside:
            pair = r.get("vdd_in") or {}
            require(pair.get("binding") == ready["vdd_in_binding"] and node_number(pair.get("voltage"), True) is not None and node_number(pair.get("current"), True) is not None, "VDD_IN sysfs binding differs")
        freq = dict(gpu={k: frequency_summary([r.get("gpu", {}).get(k, {}) for r in inside], "Hz") for k in ("cur_freq", "min_freq", "max_freq")}, emc_rate=frequency_summary([r.get("emc_rate", {}) for r in inside], "Hz"))
        require(freq["gpu"]["cur_freq"]["valid_count"] > 0 and freq["emc_rate"]["valid_count"] > 0, "GPU/EMC frequency observations missing")
        summaries[phase] = dict(window=w, power=power, frequencies=freq, sysfs_sampling=sample_gap_summary([r["monotonic_ns"] for r in inside], start, end))
        if phase != "baseline":
            require(w["input_protocol"] == PROTOCOL and w["max_resident_native_images"] == 1, "Video phase scope changed")
            rows, ls = validate_video_frames(infer, phase, w, driver)
            require(ls == ir[phase + "_latency"], "Reported latency differs from CSV")
            calls += len(rows)
            summaries[phase]["latency"] = ls
            if phase == "warmup":
                first = [r for r in memory if r["phase"] == "first_decoded_frame_before_first_forward"]
                require(len(first) == 1 and int(rows[0]["read_end_monotonic_ns"]) <= first[0]["monotonic_ns"] <= int(rows[0]["start_monotonic_ns"]), "Actual first-frame gate not before first forward")
    require(calls == ir["forward_calls_started"], "Forward call count differs")
    gate=ir["integrated_warmup_gate"]
    require(gate["passed"] is True and gate["forward_calls"]==windows["warmup"]["frames"]>=751
            and gate["online_updates"]==(0 if args.mode=="detector_only" else gate["forward_calls"])
            and gate["clips_seen"]==["R04","R02"]
            and windows["warmup"]["end_monotonic_ns"]<=gate["monotonic_ns"]<=windows["measurement"]["start_monotonic_ns"],"Warmup compatibility gate not before measurement")
    require(ir["online_updates_completed"]==(0 if args.mode=="detector_only" else calls),"Immediate update count differs")
    require(ir["pair_descriptor"]==read_json(ROOT/"video_pair.json"),"Pair descriptor provenance differs")
    complete=[a for a in ir["video_pair_capture_audits"] if a["complete"]]
    require(len(complete)>=len(windows["measurement"]["passes"])
            and all(a["clips"]==[dict(clip="R04",frames=750,eof_seen=True),dict(clip="R02",frames=750,eof_seen=True)] for a in complete),"Underlying pair EOF evidence differs")
    m = windows["measurement"]
    rate = m["frames"] / m["duration_s"]
    require(math.isclose(rate, ir["video_window_throughput_FPS"], abs_tol=1e-8), "Whole-window FPS differs")
    pw, bw = [summaries[p]["power"]["module_power_VDD_IN_mean_W"] for p in ("measurement", "baseline")]
    increment = pw - bw
    return dict(job=args.job, mode=args.mode, replicate=args.replicate, weight_sha256=job["sha256"], accuracy_summary_sha256=receipt["accuracy_summary_sha256"], status="video_measurement_completed_review_required", exitcode=0, passed_all=False, benchmark_completed=True, video_FPS_measured=True, power_measured=True,
                AP_measured=False, tracking_executed=args.mode!="detector_only", TensorRT_executed=False, replicate_count=1, input_protocol=PROTOCOL, manifest_sha256=expected,
                root_session_id=session, boot_id=boot, root_ready_sha256=ready_sha, root_report_sha256=root_sha, infer_report_sha256=infer_sha,
                launcher_report_sha256=launcher_sha, video_window_throughput_FPS=rate, measurement_frames=m["frames"], complete_passes=len(m["passes"]), resources=resources,
                excluded_invalid_temperature_samples=excluded, root_queries=rr["queries"], CUDA_allocator_postwarm_peak_MiB=ir["CUDA_allocator_postwarm_peak_MiB"],
                module_power_and_efficiency=dict(scope="total module VDD_IN including decode, Firefox and background; not wall/isolated-model power", primary_method="Qualified same-window arithmetic mean of current VDD_IN samples", module_power_mean_W=pw,
                    loaded_idle_baseline_module_power_mean_W=bw, signed_incremental_module_power_W=increment, FPS_per_module_W=rate / pw,
                    module_energy_per_frame_J=pw / rate, signed_incremental_energy_per_frame_J=increment / rate)), summaries, bindings


def main():
    parser = argparse.ArgumentParser(description="Offline video-only same-session evidence and power binder; no model/device commands")
    for n in ("root-session", "infer-output", "output"):
        parser.add_argument("--" + n, required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("detector_only","SORT","ByteTrack"))
    parser.add_argument("--job", required=True, choices=('compact_b80_s101_ssl', 'v8n_b80_s101_supervised', 'v8n_b80_s101_ssl'))
    parser.add_argument("--replicate", required=True, type=int, choices=(1, 2, 3))
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    out = no_links(args.output, must_exist=False)
    for p in (ROOT, args.root_session, args.infer_output):
        p = no_links(p)
        require(out != p and out not in p.parents and p not in out.parents, "Joined output overlaps evidence/package")
    require(not out.exists(), "Joined output must be new")
    out.mkdir(exist_ok=False)
    try:
        report, summaries, bindings = audit(args)
    except Exception as exc:
        report = dict(schema="jetson_video_formal_joined_v1", status="incomplete", exitcode=3, passed_all=False, benchmark_completed=False, video_FPS_measured=False,
                      power_measured=False, module_power_and_efficiency=None, error=dict(type=type(exc).__name__, message=str(exc)), traceback=traceback.format_exc())
        summaries, bindings = {}, []
    report.update(schema="jetson_video_formal_joined_v1", script_sha256=sha(__file__), limitations=["One selected model/process/device/video configuration; three repeats require all scheduled results and failures.", "Source video is not independent GT; no AP/tracking/TensorRT conclusions.", "Power timestamps are receipt times; no hardware latency, thermal steady state or locked clocks assertion.", "Native two-clip file decode,CPU detections,immediate same-process tracker and logging included; render/encode/camera excluded.", "Shell exit originals and Firefox state remain operator evidence; do not reconstruct."])
    write_json(out / "window_summary.json", summaries)
    write_json(out / "read_bindings.json", bindings)
    report["artifacts"] = [dict(path=n, bytes=(out / n).stat().st_size, sha256=sha(out / n)) for n in ("window_summary.json", "read_bindings.json")]
    write_json(out / "joined_report.json", report)
    (out / "joined_report.sha256").write_text(sha(out / "joined_report.json") + "  joined_report.json\n", encoding="ascii")
    print(json.dumps(dict(status=report["status"], exitcode=report["exitcode"], output=str(out)), ensure_ascii=False))
    return report["exitcode"]


if __name__ == "__main__":
    sys.exit(main())
