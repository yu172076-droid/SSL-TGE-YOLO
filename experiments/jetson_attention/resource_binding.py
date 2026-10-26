"""F1 steady resource proof with explicit bounded startup-wait phases."""
from reference_binder import *
from common import protocol
def validate_memory(rows, report, session, boot, expected, ready, payload):
    ordered(rows, "Process memory")
    observer = report["memory_observer"]
    require(observer["failure"] is None and observer["samples"] == len(rows) and observer["thread_started"] is True and observer["thread_stopped"] is True, "Resource observer incomplete/failed")
    required = {"before_model_load", "model_loaded_before_decoder_open", "decoder_capacity_after_baseline", "first_decoded_frame_before_first_forward", "postwarm", "post_measurement", "right_bracket_after_measurement", "model_load_ready"}
    require(required <= {r["phase"] for r in rows}, "Mandatory resource gates absent")
    require({r["phase"] for r in rows} <= required | {"baseline", "decoder_open_before_first_forward", "warmup", "measurement", "model_capacity_wait", "loaded_capacity_wait", "decoder_capacity_wait", "first_frame_capacity_wait", "decoder_startup_wait", "preflight"}, "Unknown resource phase (including explicit preflight allowance)")
    require(all(r["monotonic_ns"] < report["windows"]["baseline"]["start_monotonic_ns"] and r["paging_failure_enforced"] is False for r in rows if r["phase"] == "preflight"), "preflight is only legal before loaded-idle baseline, never formal")
    import startup_monitor
    require(report["capacity_policy"]==protocol()["capacity_policy"],"Capacity policy differs")
    activation=report["formal_guard_activation"]
    require(report["startup_policy"]==activation["policy"]==startup_monitor.POLICY,"Startup policy mismatch")
    require(report["windows"]["warmup"]["end_monotonic_ns"]<=activation["monotonic_ns"]<=report["windows"]["measurement"]["start_monotonic_ns"],"Formal guard not activated before measurement")
    before=[r for r in rows if r["monotonic_ns"]<activation["monotonic_ns"]]
    proof=startup_monitor.stable_tail(before,report["warmup_stability_start_monotonic_ns"])
    require(proof==activation["stability"] and proof["ready"],"Stable-tail proof differs")
    initial_oom=startup_monitor.oom(rows[0])
    require(initial_oom is not None,"Missing initial OOM counter")
    previous, streak = None, 0
    for i, row in enumerate(rows):
        require(row["startup_policy"]==startup_monitor.POLICY and type(row["paging_failure_enforced"]) is bool
            and row["paging_failure_enforced"]==(row["monotonic_ns"]>=activation["monotonic_ns"])
            and row["oom_kill_counter"]==startup_monitor.oom(row)==initial_oom,"Phase enforcement or OOM evidence differs")
        threshold = protocol()["predecoder_min_bytes"] if row["phase"] == "decoder_capacity_after_baseline" else (protocol()["first_frame_min_bytes"] if row["phase"] in ("model_loaded_before_decoder_open", "decoder_open_before_first_forward", "first_decoded_frame_before_first_forward") else (3*GIB if row["phase"]=="model_load_ready" else GIB))
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
