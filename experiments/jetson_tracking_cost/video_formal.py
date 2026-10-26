"""One bounded selected b80/s101 model FP16 video measurement, root sampled.

New output only. One process per scheduled mode/replicate; immediate online tracking when selected; no rendering or encoded output.
Exit zero means completed pending independent offline power/evidence binding.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
PROTOCOL = "native_pair_complete_passes_online_tracking_v1"
SHORT_MANIFEST = "c3187f6edb8f1232d9c4a3f52167792d38209ced74bf085a7f98c810a7695ce9"
SHORT_SCRIPT = "80cdd905fe857d401c16c34c658e7811133cad6a177b34073a602124a8e3f364"
STREAM_SCRIPT = "bb3f74a29b43226a18f88a8863b342a9ea8a63c4aecc71d166f91c29dfe07f9c"
VIDEO_SHA = "ed9996ffcea5f5547a45e9f78fbc5cdf28164412e2aa4d9c75e3f385a1135278"
WEIGHT_SHA = None
ARGS = None
VIDEO_BYTES, NATIVE_BYTES, VIDEO_FRAMES, GIB = 705, 24883200, 1500, 1024 ** 3
# Accepted short-report hashes are fixed before the new manifest is frozen.
SHORT_PARENT_SHA = "cd213826d72a021d624f4a0fc7210987e25da781da91c3b6408a70ae263c169a"
SHORT_WORKER_SHA = "3846d20dbc0cf27b57ff9f3152b49d360cc1c9b6b963a78fe25f57341c0aa40e"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def verify(expected):
    verifier = module("video_formal_verify", ROOT / "verify_package.py")
    return verifier.verify(ROOT, expected)


def short_qualification():
    short, stream = libraries()
    return short.qualification(ARGS)


def libraries():
    short = module("c4_formal_model_adapter", ROOT / "runtime_adapter.py")
    short.ARGS = ARGS
    stream = module("c4_root_session_helper", ROOT / "root_session_helper.py")
    stream.ROOT = ROOT
    return short, stream


def latency(rows):
    result = {}
    for key in ("input_read_decode_ms", "processing_ms", "forward_cuda_event_ms", "pipeline_ms", "tracking_cpu_ms", "combined_processing_ms"):
        values = sorted(float(r[key]) for r in rows)
        require(values and all(math.isfinite(v) and v >= 0 for v in values), "Invalid latency samples")
        pos = (len(values) - 1) * .95
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        result[key] = dict(mean=statistics.mean(values), median=statistics.median(values), p95=values[lo] + (values[hi] - values[lo]) * (pos - lo), min=min(values), max=max(values))
    return result


FIELDS = ["tracking_cpu_ms", "combined_processing_ms", "track_count", "phase", "iteration", "pass_index", "source_decode_ordinal", "image_id", "image", "start_monotonic_ns", "end_monotonic_ns", "processing_ms", "forward_cuda_event_ms", "detection_count", "validation_and_record_conversion_ms", "all_outputs_finite", "pipeline_start_monotonic_ns", "pipeline_end_monotonic_ns", "read_start_monotonic_ns", "read_end_monotonic_ns", "input_read_decode_ms", "pipeline_ms", "native_frame_bytes"]


def video_window(name, cv2, video, infer, short, monitor, output, report, clock=time.monotonic_ns):
    """Warmup can end after a frame; measurement stops only at a full EOF pass."""
    limit = 300 if name == "warmup" else 600
    minimum_frames = 751 if name == "warmup" else 300
    rows = []
    start = clock()
    window = dict(start_monotonic_ns=start, end_monotonic_ns=None, frames=0, complete=False, passes=[], input_protocol=PROTOCOL, max_resident_native_images=1)
    report["windows"][name] = window
    if name=="warmup":monitor.warmup_start_ns=start
    cap = None
    first = report["forward_calls_started"] == 0
    try:
        with (output / (name + "_frames.csv")).open("x", encoding="utf-8", newline="") as cs, (output / (name + "_predictions.jsonl")).open("x", encoding="utf-8") as ps:
            writer = csv.DictWriter(cs, fieldnames=FIELDS)
            writer.writeheader()
            stop = False
            while not stop:
                monitor.check()
                require((clock() - start) / 1e9 < limit, name + " phase timeout")
                pass_row = dict(pass_index=len(window["passes"]) + 1, open_start_monotonic_ns=clock(), frames=0, eof_seen=False, complete=False)
                window["passes"].append(pass_row)
                cap = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
                require(cap.isOpened() and cap.getBackendName() == "FFMPEG", "FFMPEG capture unavailable")
                pass_row["open_end_monotonic_ns"] = clock()
                ordinal = 0
                while True:
                    monitor.check()
                    require((clock() - start) / 1e9 < limit, name + " phase timeout")
                    frame = None
                    begin = clock()
                    ok, frame = cap.read()
                    if not ok:
                        del frame
                        pass_row["eof_seen"] = True
                        pass_row["eof_read_start_monotonic_ns"] = begin
                        pass_row["eof_read_end_monotonic_ns"] = clock()
                        require(ordinal == VIDEO_FRAMES, "Incomplete/truncated video pass")
                        pass_row["complete"] = True
                        break
                    ordinal += 1
                    try:
                        require(ordinal <= VIDEO_FRAMES and frame is not None and frame.dtype.name == "uint8"
                                and list(frame.shape) == [2160, 3840, 3] and int(frame.nbytes) == NATIVE_BYTES, "Native frame shape/count changed")
                        read_end = clock()
                        if first:
                            monitor.tick("first_decoded_frame_before_first_forward", 3 * GIB)
                            monitor.set_phase("warmup", GIB)
                            first = False
                        report["forward_calls_started"] += 1
                        record, values = infer(frame, dict(id=ordinal, edge_image="video_decode_ordinal/{}".format(ordinal)))
                        short.validate_values(values)
                    finally:
                        del frame
                        end = clock()
                    record.update(phase=name, iteration=len(rows) + 1, pass_index=pass_row["pass_index"], source_decode_ordinal=ordinal,
                                  pipeline_start_monotonic_ns=begin, pipeline_end_monotonic_ns=end, read_start_monotonic_ns=begin,
                                  read_end_monotonic_ns=read_end, input_read_decode_ms=(read_end - begin) / 1e6,
                                  pipeline_ms=(end - begin) / 1e6, native_frame_bytes=NATIVE_BYTES)
                    writer.writerow(record)
                    ps.write(json.dumps(dict(phase=name, iteration=len(rows) + 1, pass_index=pass_row["pass_index"], source_decode_ordinal=ordinal, detections_xyxy_conf_class=values, tracking_payload=report["_last_online_payload"]), allow_nan=False) + "\n")
                    rows.append(record)
                    pass_row["frames"] = ordinal
                    window["frames"] = len(rows)
                    monitor.check()
                    elapsed = (clock() - start) / 1e9
                    if name == "warmup" and len(rows) >= minimum_frames and elapsed >= 60:
                        # At most1 synchronous stability check/second after minimum warmup.
                        current=clock()
                        if current-getattr(monitor,"last_stability_check_ns",-10**20)>=10**9:
                            monitor.last_stability_check_ns=current
                            monitor.tick("warmup",GIB)
                            if monitor.tail()["ready"]:
                                stop=True
                                break
                cap.release()
                cap = None
                pass_row["close_end_monotonic_ns"] = clock()
                cs.flush()
                ps.flush()
                pass_row["record_flush_end_monotonic_ns"] = clock()
                elapsed = (clock() - start) / 1e9
                require(elapsed <= limit, name + " phase timeout at pass close")
                if len(rows) >= minimum_frames and elapsed >= 60:
                    if name!="warmup" or monitor.tail()["ready"]:stop=True
                print("{} frames={} elapsed={:.2f}s eof={}".format(name, len(rows), elapsed, pass_row["eof_seen"]), flush=True)
            cs.flush()
            ps.flush()
        require(name == "warmup" or all(p["complete"] for p in window["passes"]), "Measurement has partial pass")
        window["complete"] = True
    finally:
        if cap is not None:
            cap.release()
        window["end_monotonic_ns"] = clock()
        window["duration_s"] = (window["end_monotonic_ns"] - start) / 1e9
        window["frames"] = len(rows)
    require(60 <= window["duration_s"] <= limit, name + " window duration outside fixed bounds")
    report[name + "_latency"] = latency(rows)
    return window


def worker(args):
    output, video, expected = args.output, args.video, args.expected_manifest_sha256
    report = dict(job=args.job, mode=args.mode, replicate=args.replicate, schema="jetson_video_formal_inference_v1", status="incomplete", actual_self_exit_code=3, passed_all=False,
                  manifest_sha256=expected, script_sha256=sha(__file__), video_sha256=VIDEO_SHA, weight_sha256=WEIGHT_SHA,
                  input_protocol=PROTOCOL, pid=os.getpid(), euid=os.geteuid(), boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                  output=str(output), root_session_path=str(args.root_session), windows={}, models={}, forward_calls_started=0,
                  benchmark_completed=False, video_FPS_measured=False, AP_measured=False, power_measured=False, tracking_executed=False, TensorRT_executed=False,
                  replicate_count=1, config=dict(batch=1, input_shape=[1, 3, 640, 640], precision="FP16", conf=.001, mode=args.mode, iou=.65, max_det=300,
                      native_dimensions=[3840, 2160], expected_frames_per_pass=VIDEO_FRAMES, firefox_must_remain_open=True, backend="FFMPEG"),
                  timing_scope="Complete two-clip native file decode to CPU detections and same-process tracking/record flush; no render/encode/camera; root power qualification pending")
    state, monitor = {}, None
    code = 3
    try:
        def interrupt(signum, frame):
            raise KeyboardInterrupt("signal " + str(signum))
        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        verify(expected)
        short_qualification()
        short, stream = libraries()
        short.require_jetson()
        require(os.geteuid() != 0 and sha(video) == VIDEO_SHA and video.stat().st_size == VIDEO_BYTES, "Wrong user/video")
        before = video.stat()
        root_session = stream.RootSession(args.root_session, expected, report["boot_id"])
        report.update(root_ready=root_session.ready, root_ready_sha256=root_session.ready_sha, root_session_id=root_session.ready["session_id"])
        original_monitor = short.Monitor
        class FormalMonitor(original_monitor):
            def __init__(self, sampler, sample_path=None):
                def combined(phase):
                    row = sampler(phase)
                    row["root_live_check"] = root_session.check()
                    return row
                super().__init__(combined, sample_path)
        short.Monitor = FormalMonitor
        cv2, torch, monitor, infer = short.prepare_model(output, report, state)
        for source in sorted(ROOT.rglob("*")):
            if source.is_file():
                target = output / "source_snapshot/new_formal" / source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                require(sha(source) == sha(target), "Formal snapshot differs")
        report["accepted_short_report_sha256"] = short.C2_SHA
        (output / "opencv_build_information.txt").write_text(cv2.getBuildInformation(), encoding="utf-8")
        monitor.set_phase("baseline", GIB)
        if monitor.thread is None:
            monitor.start()
        start = time.monotonic_ns()
        window = dict(start_monotonic_ns=start, model_loaded=True, capture_closed=True, native_frames_retained=0, inference_executed=False, frames=0, complete=False)
        report["windows"]["baseline"] = window
        while (time.monotonic_ns() - start) / 1e9 < 60:
            monitor.check()
            time.sleep(.1)
        window.update(end_monotonic_ns=time.monotonic_ns(), complete=True)
        window["duration_s"] = (window["end_monotonic_ns"] - start) / 1e9
        require(window["duration_s"] <= 90, "Loaded-idle scheduling timeout")
        monitor.tick("decoder_capacity_after_baseline", 3 * GIB + NATIVE_BYTES)
        monitor.set_phase("decoder_open_before_first_forward", 3 * GIB)
        with torch.no_grad():
            video_window("warmup", cv2, video, infer, short, monitor, output, report)
            torch.cuda.synchronize()
            clips=sorted({r["clip"] for r in report["online_reset_events"] if r["stage"]=="warmup"})
            require(clips==["R02","R04"] and report["_forward_gate"]==dict(open=False,intercepted_calls=report["forward_calls_started"])
                    and report["online_updates_completed"]==(0 if args.mode=="detector_only" else report["forward_calls_started"]),"Integrated warmup compatibility failed")
            report["integrated_warmup_gate"]=dict(passed=True,forward_calls=report["forward_calls_started"],online_updates=report["online_updates_completed"],
                clips_seen=["R04","R02"],monotonic_ns=time.monotonic_ns())
            monitor.tick("postwarm", GIB)
            monitor.set_phase("measurement", GIB)
            report["formal_guard_activation"]=monitor.formal_activation
            report["startup_policy"]=monitor.formal_activation["policy"]
            torch.cuda.reset_peak_memory_stats(0)
            measured = video_window("measurement", cv2, video, infer, short, monitor, output, report)
            torch.cuda.synchronize()
        report["CUDA_allocator_postwarm_peak_MiB"] = dict(allocated=torch.cuda.max_memory_allocated(0) / 1024 ** 2, reserved=torch.cuda.max_memory_reserved(0) / 1024 ** 2, scope="reset after warmup; allocator only, not additive to process RSS")
        monitor.tick("post_measurement", GIB)
        # Allow a right-hand power bracket outside the measured window.
        bracket = time.monotonic()
        while time.monotonic() - bracket < 2:
            monitor.check()
            time.sleep(.1)
        monitor.tick("right_bracket_after_measurement", GIB)
        require(report["_forward_gate"] == dict(open=False, intercepted_calls=report["forward_calls_started"]), "Unexpected/extra forward")
        import pair_input
        require(pair_input.verify_pair()==report["pair_descriptor"],"Original pair changed")
        require(report["online_updates_completed"]==(0 if args.mode=="detector_only" else report["forward_calls_started"]),"Integrated online update count")
        report["integrated_warmup_compatibility_passed"]=True
        report["combined_same_process"]=True
        report["source_origins_after_inference"] = short.c1.source_origins(args.bundle.resolve(), short.select(args)["backend"])
        require(sha(args.bundle / short.select(args)["weight"]) == WEIGHT_SHA, "Weight changed")
        after = video.stat()
        report["video_unchanged"] = before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns and sha(video) == VIDEO_SHA
        require(report["video_unchanged"], "Video changed")
        report.update(status="video_measurement_completed_binding_required", benchmark_completed=True,
                      video_window_throughput_FPS=measured["frames"] / measured["duration_s"])
        code = 0
    except BaseException as exc:
        report.update(error=dict(type=type(exc).__name__, message=str(exc)), traceback=traceback.format_exc())
    finally:
        monitor = monitor or state.get("monitor")
        if monitor is not None:
            try:
                monitor.stop()
                monitor.check()
            except Exception as exc:
                report.update(status="resource_or_root_guard_stopped", error=str(exc), benchmark_completed=False)
                code = 3
            report["memory_observer"] = dict(samples=len(monitor.samples), failure=None if monitor.failure is None else str(monitor.failure), thread_started=monitor.thread is not None, thread_stopped=monitor.thread is None or not monitor.thread.is_alive(), strict_1Hz_asserted=False)
        report["actual_self_exit_code"] = code
        write(output / "measurement_window.json", report["windows"])
        write(output / "infer_report.json", report)
        (output / "infer_report.sha256").write_text(sha(output / "infer_report.json") + "  infer_report.json\n", encoding="ascii")
        print(json.dumps(dict(status=report["status"], actual_self_exit_code=code, benchmark_completed=report["benchmark_completed"])), flush=True)
    return code


def parent(args):
    verify(args.expected_manifest_sha256)
    short, stream = libraries()
    short.require_jetson()
    require(os.geteuid() != 0, "Ordinary user only")
    args.video = args.video.resolve(strict=True)
    require(sha(args.video) == VIDEO_SHA and args.video.stat().st_size == VIDEO_BYTES, "Pinned video missing/different")
    short_qualification()
    returns = PROJECT / "returns"
    args.output = args.output.absolute()
    args.root_session = args.root_session.resolve(strict=True)
    require(returns.is_dir() and not returns.is_symlink() and args.output.parent.resolve() == returns.resolve()
            and args.root_session.parent == returns.resolve() and args.root_session != args.output and not os.path.lexists(str(args.output)), "New output/direct root session required")
    stream.RootSession(args.root_session, args.expected_manifest_sha256, short.boot_id())
    args.output.mkdir(exist_ok=False)
    admission_dir = args.output / "source_snapshot/admission"
    admission_dir.mkdir(parents=True)
    for name in (admission_name(args), str(Path(admission_name(args)).with_suffix(".sha256"))):
        source = PROJECT / "returns" / name
        shutil.copyfile(source, admission_dir / name)
        require(sha(source) == sha(admission_dir / name), "Admission snapshot differs")
    report = dict(job=args.job, mode=args.mode, replicate=args.replicate, schema="jetson_video_formal_launcher_v1", manifest_sha256=args.expected_manifest_sha256, script_sha256=sha(__file__), input_protocol=PROTOCOL,
                  video_sha256=VIDEO_SHA, weight_sha256=WEIGHT_SHA, root_session=str(args.root_session), output=str(args.output), normal_euid=os.geteuid(),
                  child_exitcode=None, launcher_exitcode=3, status="incomplete", watchdog_s=1200, started_monotonic_ns=time.monotonic_ns(), passed_all=False)
    cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", "--expected-manifest-sha256", args.expected_manifest_sha256,
           "--root-session", str(args.root_session), "--video", str(args.video), "--output", str(args.output)]
    cmd += ["--mode",args.mode,"--job", args.job, "--replicate", str(args.replicate), "--bundle", str(args.bundle.resolve()), "--inputs", str(args.inputs.resolve())]
    report["worker_argv"] = cmd
    report["R2_startup_context"] = args.r2_startup_context
    write(args.output / "launch_metadata.json", report)
    child = None
    handlers = {}
    def interrupted(signum, frame):
        raise KeyboardInterrupt("launcher signal " + str(signum))
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, interrupted)
        with (args.output / "worker_console.txt").open("xb") as console:
            child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=console, stderr=subprocess.STDOUT)
            report["child_pid"] = child.pid
            print("Video worker {} started; root sampler stays in its terminal; wait for launcher summary.".format(child.pid), flush=True)
            try:
                child.wait(timeout=1200)
            except subprocess.TimeoutExpired:
                report["watchdog_triggered"] = True
                short.stop_child(child)
            report["child_exitcode"] = child.returncode
        body_path = args.output / "infer_report.json"
        body = read(body_path) if body_path.exists() else {}
        sc = args.output / "infer_report.sha256"
        bound = body_path.exists() and sc.exists() and sc.read_text(encoding="ascii").split() == [sha(body_path), "infer_report.json"]
        report["nms_time_limit_warning"] = "NMS time limit" in (args.output / "worker_console.txt").read_text(encoding="utf-8", errors="replace")
        success = (child.returncode == body.get("actual_self_exit_code") == 0 and body.get("benchmark_completed") is True and bound
                   and body.get("manifest_sha256") == args.expected_manifest_sha256 and body.get("input_protocol") == PROTOCOL
                   and body.get("video_sha256") == VIDEO_SHA and body.get("weight_sha256") == WEIGHT_SHA
                   and body.get("job") == args.job and body.get("mode")==args.mode and body.get("replicate") == args.replicate
                   and body.get("pid") == child.pid and body.get("euid") == os.geteuid() and body.get("root_session_path") == str(args.root_session)
                   and not report["nms_time_limit_warning"] and not report.get("watchdog_triggered"))
        report["launcher_exitcode"] = 0 if success else (124 if report.get("watchdog_triggered") else 3)
        if success:
            report["status"] = "video_measurement_completed_binding_required"
    except BaseException as exc:
        short.stop_child(child)
        report["child_exitcode"] = child.returncode if child is not None else None
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
    finally:
        for signum, original in handlers.items():
            signal.signal(signum, original)
    report["ended_monotonic_ns"] = time.monotonic_ns()
    report["artifacts"] = [dict(path=p.relative_to(args.output).as_posix(), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(args.output.rglob("*")) if p.is_file() and "cache" not in p.relative_to(args.output).parts]
    write(args.output / "launcher_report.json", report)
    (args.output / "launcher_report.sha256").write_text(sha(args.output / "launcher_report.json") + "  launcher_report.json\n", encoding="ascii")
    print(json.dumps(dict(status=report["status"], child_exitcode=report["child_exitcode"], launcher_exitcode=report["launcher_exitcode"], output=str(args.output))), flush=True)
    return report["launcher_exitcode"]


def r2_start_context():
    short, unused = libraries()
    memory = short.guard.memory_sample("r2_startup_outside_measurement")
    firefox = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            text = (path / "status").read_text()
            values = {line.partition(":")[0]: line.partition(":")[2].strip() for line in text.splitlines()}
            if int(values.get("Uid", "-1").split()[0]) == os.geteuid() and values.get("Name") == "firefox":
                firefox.append(dict(pid=int(path.name),RSS_status=values.get("VmRSS")))
        except (OSError, ValueError, IndexError):
            continue
    require(firefox, "Open Firefox with the declared single about:blank tab before R2 admission")
    return dict(memory=memory,uptime_raw=Path("/proc/uptime").read_text(),firefox_main_processes=firefox,firefox_tab_state="single-about-blank",tab_state_source="operator declaration; process presence checked, tab contents NOT machine-verified")


def admission_name(args):
    return "tracking_cost_admission_"+token(args)+".json"
def token(args):
    return "{}_20261018_{:02d}_f1_steady01".format(args.mode,args.replicate)
def qualified_prefix_receipt(args,row,used,receipt_name):
    from types import SimpleNamespace
    import common as c
    receipt=c.read(ROOT/receipt_name)
    pin=next(x for x in receipt["accepted"] if dict(mode=x["mode"],replicate=x["replicate"])==row)
    old=c.verified_module(PROJECT/receipt["original_packet"],receipt["original_manifest_sha256"],"video_formal.py","F1_accepted_original_driver")
    old.verify(receipt["original_manifest_sha256"])
    result=old.qualified_previous(SimpleNamespace(expected_manifest_sha256=receipt["original_manifest_sha256"]),row,used)
    require(result["joined_report_sha256"]==pin["joined_report_sha256"] and result["boot_id"]==pin["boot_id"],"Accepted original prefix differs")
    directory=PROJECT/"returns";token_old=pin["token"]
    for folder,stem,key in (("tracking_cost_","infer_report","infer_report_sha256"),("tracking_cost_","launcher_report","launcher_report_sha256"),("tracking_cost_root_","root_sampler_report","root_report_sha256")):
        body=c.report(directory/(folder+token_old),stem,pin[key])
    ir=c.read(directory/("tracking_cost_"+token_old)/"infer_report.json")
    require(ir["windows"]["warmup"]["frames"]==pin["warmup_frames"]>=751 and ir["windows"]["warmup"]["duration_s"]==pin["warmup_duration_s"]>=60,"Original warmup not equivalent")
    ad=directory/("tracking_cost_admission_"+token_old+".json")
    require(c.sha(ad)==pin["admission_sha256"],"Accepted original admission differs")
    import startup_monitor
    bridge=c.read(ROOT/"accepted_steady_bridge.json")
    bp=next(x for x in bridge["items"] if dict(mode=x["mode"],replicate=x["replicate"])==row)
    rawpath=directory/("tracking_cost_"+token_old)/"process_memory_samples.jsonl"
    require(c.sha(rawpath)==bp["memory_trace_sha256"] and c.sha(directory/("tracking_cost_"+token_old)/"infer_report.json")==bp["infer_report_sha256"],"Legacy steady trace pin differs")
    rows=[json.loads(x) for x in rawpath.read_text().splitlines()]
    before=[x for x in rows if x["monotonic_ns"]<=ir["windows"]["measurement"]["start_monotonic_ns"]]
    proof=startup_monitor.stable_tail(before,ir["windows"]["warmup"]["start_monotonic_ns"])
    require(proof==bp["proof"] and proof["ready"],"Retained original fails common stable-tail criterion")
    result["steady_bridge_trace_sha256"]=bp["memory_trace_sha256"]
    result["accepted_prefix_source_packet"]=receipt["original_packet"]
    return result

def qualified_original_prefix(args,row,used):
    return qualified_prefix_receipt(args,row,used,"accepted_f1_prefix.json")

def qualified_resume01_prefix(args,row,used):
    return qualified_prefix_receipt(args,row,used,"accepted_f1_resume01.json")

def qualified_previous(args,row,used):
    if row in read(ROOT/"protocol.json")["accepted_resume01_prefix"]:return qualified_resume01_prefix(args,row,used)
    if row in read(ROOT/"protocol.json")["accepted_original_prefix"]:return qualified_original_prefix(args,row,used)
    from types import SimpleNamespace
    import common as c
    item=SimpleNamespace(mode=row["mode"],replicate=row["replicate"])
    t=token(item);directory=PROJECT/"returns"
    joined=c.report(directory/("tracking_cost_joined_"+t),"joined_report")
    root=c.report(directory/("tracking_cost_root_"+t),"root_sampler_report",joined["root_report_sha256"])
    launcher=c.report(directory/("tracking_cost_"+t),"launcher_report",joined["launcher_report_sha256"])
    infer_path=directory/("tracking_cost_"+t)/"infer_report.json";body=c.read(infer_path)
    require(c.sha(infer_path)==joined["infer_report_sha256"] and infer_path.with_suffix(".sha256").read_text().split()==[c.sha(infer_path),infer_path.name],"Previous inference sidecar")
    require(joined["exitcode"]==root["exitcode"]==launcher["child_exitcode"]==launcher["launcher_exitcode"]==body["actual_self_exit_code"]==0
            and joined["benchmark_completed"] and joined["video_FPS_measured"] and joined["power_measured"]
            and joined["mode"]==body["mode"]==launcher["mode"]==row["mode"]
            and joined["replicate"]==body["replicate"]==launcher["replicate"]==row["replicate"]
            and all(x["manifest_sha256"]==args.expected_manifest_sha256 for x in (joined,root,launcher,body))
            and joined["script_sha256"]==sha(ROOT/"bind_results.py")
            and body["script_sha256"]==launcher["script_sha256"]==sha(__file__)
            and root["script_sha256"]==sha(ROOT/"root_sampler.py"),"Previous formal item failed/wrong mode/source")
    require(joined["root_session_id"]==root["session_id"]==body["root_session_id"]
            and joined["boot_id"]==root["boot_id"]==body["boot_id"] and joined["boot_id"] not in used,"Previous boot/session reuse")
    ap=directory/admission_name(item);ad=c.read(ap)
    require(ap.with_suffix(".sha256").read_text().split()==[c.sha(ap),ap.name] and ad["schema"]=="tracking_f1_admission_v1"
            and ad["manifest_sha256"]==args.expected_manifest_sha256 and ad["minimum_quiet_seconds"]==600 and ad["mode"]==item.mode
            and ad["replicate"]==item.replicate and ad["boot_id"]==body["boot_id"]
            and launcher["started_monotonic_ns"]-ad["prepared_monotonic_ns"]>=600*10**9,"Previous admission incomplete")
    for name in (ap.name,ap.with_suffix(".sha256").name):
        require(sha(directory/("tracking_cost_"+t)/"source_snapshot/admission"/name)==sha(directory/name),"Previous admission snapshot differs")
    expected=["F1_"+k+"_EXIT=0 mode="+item.mode+" rep={:02d}".format(item.replicate) for k in ("VERIFY","PREFLIGHT","INFER","ROOT","BIND")]
    require((directory/("tracking_cost_terminal_"+t+".txt")).read_text().splitlines()==expected,"Previous five shell exits not all0")
    used.add(joined["boot_id"])
    return dict(mode=item.mode,replicate=item.replicate,boot_id=body["boot_id"],joined_report_sha256=c.sha(directory/("tracking_cost_joined_"+t)/"joined_report.json"))
def unused_preupload_incident(directory):
    import re,common as c
    authorization=c.read(ROOT/"preupload_reissue_authorization.json")
    token_old=authorization["superseded_unused_token"]
    path=directory/authorization["original_log"]
    require(not path.is_symlink() and path.is_file(),"Preserve and provide original two upload-failure log lines")
    lines=path.read_text(encoding="utf-8").splitlines()
    require(len(lines)==2 and all(re.fullmatch(r"F1_VERIFY_EXIT=2(?: mode=detector_only rep=02)?",line.strip()) for line in lines),"Original preupload failure log differs; do not edit it")
    prohibited=["tracking_cost_"+token_old,"tracking_cost_root_"+token_old,"tracking_cost_joined_"+token_old,
        "tracking_cost_admission_"+token_old+".json","tracking_cost_admission_"+token_old+".sha256"]
    require(all(not os.path.lexists(str(directory/name)) for name in prohibited),"Prior identity has admission/measurement evidence; unused-identity reissue no longer applies")
    return dict(token=token_old,original_log=path.name,sha256=c.sha(path),recorded_verify_exit_codes=[2,2],
        admission_root_infer_joined_absent=True,measurement_replacement_quota_consumed=False)

def require_previous(args,prepare=False,status_check=False,complete_check=False):
    import common as c,pair_input
    p=c.read(ROOT/"protocol.json");directory=PROJECT/"returns"
    incident=unused_preupload_incident(directory)
    latest=read(ROOT/"accepted_latest_failure.json")
    c.report(directory/("tracking_cost_"+latest["token"]),"infer_report",latest["infer_report_sha256"])
    c.report(directory/("tracking_cost_"+latest["token"]),"launcher_report",latest["launcher_report_sha256"])
    c.report(directory/("tracking_cost_root_"+latest["token"]),"root_sampler_report",latest["root_report_sha256"])
    prior=c.PROJECT/p["accepted_T2_packet"]
    module=c.verified_module(prior,p["accepted_T2_manifest_sha256"],"common.py","F1_accepted_T2_common")
    module.verify(p["accepted_T2_manifest_sha256"])
    w=c.report(directory/"tracking_cache_full_20261017_t2_01","cache_worker_report",p["accepted_T2_worker_report_sha256"])
    tr=c.report(directory/"tracking_replay_full_20261017_t2_01","replay_report",p["accepted_T2_replay_report_sha256"])
    require(w["exitcode"]==tr["exitcode"]==0 and w["forward_calls_started"]==1530 and len(tr["results"])==4,"Accepted T2 prerequisite")
    pair_input.verify_pair()
    receipt=c.read(ROOT/"accepted_f1_prefix.json");failed=receipt["failed_original"]
    fdir=directory/("tracking_cost_"+failed["token"])
    fw=c.report(fdir,"infer_report",failed["infer_report_sha256"])
    c.report(fdir,"launcher_report",failed["launcher_report_sha256"])
    c.report(directory/("tracking_cost_root_"+failed["token"]),"root_sampler_report",failed["root_report_sha256"])
    require(fw["actual_self_exit_code"]==3 and fw["forward_calls_started"]==681 and "measurement" not in fw["windows"] and fw["boot_id"]==failed["boot_id"],"Original failed attempt evidence differs")
    latest=c.read(ROOT/"accepted_f1_resume01.json")["failed_resume01"]
    lastdir=directory/("tracking_cost_"+latest["token"])
    last=c.report(lastdir,"infer_report",latest["infer_report_sha256"])
    c.report(lastdir,"launcher_report",latest["launcher_report_sha256"])
    c.report(directory/("tracking_cost_root_"+latest["token"]),"root_sampler_report",latest["root_report_sha256"])
    require(last["actual_self_exit_code"]==3 and last["forward_calls_started"]==11 and "measurement" not in last["windows"]
        and last["boot_id"]==latest["boot_id"] and last["memory_observer"]["failure"]=="Two consecutive positive global swap intervals; no process attribution","Prior D2 paging failure evidence differs")
    schedule=p["schedule"];index=next(i for i,x in enumerate(schedule) if x==dict(mode=args.mode,replicate=args.replicate))
    require(not complete_check or index==8,"Queue check must use lastitem SORT03")
    require(complete_check or index>=5,"First five accepted; resume02 starts D2")
    used=set(p["excluded_boots"]);receipts=[]
    for row in schedule[:index+(1 if complete_check else 0)]:receipts.append(qualified_previous(args,row,used))
    if complete_check:return dict(status="F1_NINE_COMPLETED_REVIEW_REQUIRED",manifest_sha256=args.expected_manifest_sha256,items=receipts,total=9)
    require(args.firefox_state=="single-about-blank","Declared Firefox single blank required")
    boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip();now=time.monotonic_ns()
    require(boot not in used,"Normal fresh boot required for every formal item")
    ctx=r2_start_context();path=directory/admission_name(args)
    ctx["preupload_identity_reissue"]=incident
    if prepare:
        require(not os.path.lexists(str(path)) and not os.path.lexists(str(path.with_suffix(".sha256")))
                and not os.path.lexists(str(args.root_session)) and not os.path.lexists(str(args.output)),"New formal identities only")
        c.signed(path,dict(schema="tracking_f1_admission_v1",manifest_sha256=args.expected_manifest_sha256,
            mode=args.mode,replicate=args.replicate,boot_id=boot,prepared_monotonic_ns=now,minimum_quiet_seconds=600,startup_context=ctx))
        return dict(status="F1_ADMISSION_PREPARED",boot_id=boot)
    require(path.with_suffix(".sha256").read_text().split()==[c.sha(path),path.name],"Admission sidecar")
    ad=c.read(path);require(ad["schema"]=="tracking_f1_admission_v1" and ad["manifest_sha256"]==args.expected_manifest_sha256
        and ad["mode"]==args.mode and ad["replicate"]==args.replicate and ad["boot_id"]==boot and ad["minimum_quiet_seconds"]==600,"Admission identity/boot")
    elapsed=now-ad["prepared_monotonic_ns"];require(elapsed>=0,"Monotonic decrease")
    before=ad["startup_context"]["memory"]["vm_counters"];after=ctx["memory"]["vm_counters"]
    delta={k:after[k]-before[k] for k in before}
    require(min(delta.values())>=0 and delta["pswpin"]==delta["pswpout"]==0,"Quiet global paging")
    status=dict(status="F1_ADMISSION_READY" if elapsed>=600*10**9 else "F1_ADMISSION_WAIT",
        elapsed_s=elapsed/1e9,remaining_s=max(0,600*10**9-elapsed)/1e9,boot_id=boot,
        manifest_sha256=args.expected_manifest_sha256,quiet_interval_global_deltas=delta,read_only=True)
    if status_check:
        require(not os.path.lexists(str(args.root_session)) and not os.path.lexists(str(args.output)),"Status before root only")
        return status
    require(status["status"]=="F1_ADMISSION_READY","Wait incomplete; do not launch")
    ctx["quiet_interval_global_deltas"]=delta;ctx["admission_clock_check"]=status;args.r2_startup_context=ctx
    return status
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-manifest-sha256",required=True)
    for n in ("root-session","video","output"):parser.add_argument("--"+n,required=True,type=Path)
    parser.add_argument("--job",default="compact_b80_s101_ssl",choices=("compact_b80_s101_ssl",))
    parser.add_argument("--mode",required=True,choices=("detector_only","SORT","ByteTrack"))
    parser.add_argument("--replicate",required=True,type=int,choices=(1,2,3))
    parser.add_argument("--firefox-state",choices=("single-about-blank",))
    parser.add_argument("--bundle",type=Path,default=PROJECT/"jetson_models_b80_s101_20261013")
    parser.add_argument("--inputs",type=Path,default=PROJECT/"jetson_deployment_inputs_20261009")
    for n in ("worker","verify-package-only","verify-only","admission-status","queue-complete"):parser.add_argument("--"+n,action="store_true")
    args=parser.parse_args();sys.dont_write_bytecode=True
    global ARGS,WEIGHT_SHA
    verify(args.expected_manifest_sha256);ARGS=args
    require(sum((args.worker,args.verify_package_only,args.verify_only,args.admission_status,args.queue_complete))<=1,"Choose one optional stage")
    require(os.geteuid()!=0,"Inference ordinary user only")
    short,stream=libraries();WEIGHT_SHA=short.select(args)["sha256"]
    require(args.output.name=="tracking_cost_"+token(args) and args.root_session.name=="tracking_cost_root_"+token(args),"Exact scheduled output identity")
    require(args.video.resolve()==(ROOT/"video_pair.json").resolve(),"Frozen two-clip descriptor required")
    require(args.bundle.resolve()==(PROJECT/"jetson_models_b80_s101_20261013").resolve() and args.inputs.resolve()==(PROJECT/"jetson_deployment_inputs_20261009").resolve(),"Qualified bundle/input paths")
    require(dict(mode=args.mode,replicate=args.replicate) in read(ROOT/"protocol.json")["schedule"],"Scheduled formal item only")
    if args.verify_package_only:print("F1_PACKAGE_VERIFIED",flush=True);return 0
    if args.worker:return worker(args)
    short.prerequisites(args)
    if args.queue_complete:print(json.dumps(require_previous(args,complete_check=True)),flush=True);return 0
    if args.verify_only:print(json.dumps(require_previous(args,prepare=True)),flush=True);return 0
    if args.admission_status:print(json.dumps(require_previous(args,status_check=True)),flush=True);return 0
    require_previous(args)
    return parent(args)
if __name__=="__main__":raise SystemExit(main())
