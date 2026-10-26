"""R2 complete-pass timing core; warmup adds bounded stable-tail admission only."""
import csv,json,math,statistics,time
from common import require,capacity_wait,GIB,protocol
PROTOCOL="native_video_complete_passes_cpu_detection_v1"
VIDEO_FRAMES=1475
NATIVE_BYTES=24883200

def latency(rows):
    result = {}
    for key in ("input_read_decode_ms", "processing_ms", "forward_cuda_event_ms", "pipeline_ms"):
        values = sorted(float(r[key]) for r in rows)
        require(values and all(math.isfinite(v) and v >= 0 for v in values), "Invalid latency samples")
        pos = (len(values) - 1) * .95
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        result[key] = dict(mean=statistics.mean(values), median=statistics.median(values), p95=values[lo] + (values[hi] - values[lo]) * (pos - lo), min=min(values), max=max(values))
    return result

FIELDS = ["phase", "iteration", "pass_index", "source_decode_ordinal", "image_id", "image", "start_monotonic_ns", "end_monotonic_ns", "processing_ms", "forward_cuda_event_ms", "detection_count", "validation_and_record_conversion_ms", "all_outputs_finite", "pipeline_start_monotonic_ns", "pipeline_end_monotonic_ns", "read_start_monotonic_ns", "read_end_monotonic_ns", "input_read_decode_ms", "pipeline_ms", "native_frame_bytes"]

def video_window(name, cv2, video, infer, short, monitor, output, report, clock=time.monotonic_ns):
    """Warmup can end after a frame; measurement stops only at a full EOF pass."""
    limit = 300 if name == "warmup" else 600
    minimum_frames = 30 if name == "warmup" else 300
    rows = []
    start = clock()
    if name == "warmup":
        monitor.warmup_start_ns = start
    window = dict(start_monotonic_ns=start, end_monotonic_ns=None, frames=0, complete=False, passes=[], input_protocol=PROTOCOL, max_resident_native_images=1)
    report["windows"][name] = window
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
                            capacity_wait(monitor, "first_frame_capacity_wait", protocol()['first_frame_min_bytes'])
                            monitor.tick("first_decoded_frame_before_first_forward", protocol()['first_frame_min_bytes'])
                            monitor.set_phase("warmup", GIB)
                            monitor.warmup_start_ns = clock()
                            report["warmup_stability_start_monotonic_ns"] = monitor.warmup_start_ns
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
                    ps.write(json.dumps(dict(phase=name, iteration=len(rows) + 1, pass_index=pass_row["pass_index"], source_decode_ordinal=ordinal, detections_xyxy_conf_class=values), allow_nan=False) + "\n")
                    rows.append(record)
                    pass_row["frames"] = ordinal
                    window["frames"] = len(rows)
                    monitor.check()
                    elapsed = (clock() - start) / 1e9
                    # Warmup now closes only at a full EOF pass, allowing bounded
                    # continuation after close/flush if the freshest tail is unready.
                cap.release()
                cap = None
                pass_row["close_end_monotonic_ns"] = clock()
                cs.flush()
                ps.flush()
                pass_row["record_flush_end_monotonic_ns"] = clock()
                elapsed = (clock() - start) / 1e9
                require(elapsed <= limit, name + " phase timeout at pass close")
                if name == "warmup":
                    print("{} frames={} elapsed={:.2f}s eof={}".format(name, len(rows), elapsed, pass_row["eof_seen"]), flush=True)
                    if len(rows)>=minimum_frames and elapsed>=60 and (clock()-monitor.warmup_start_ns)/1e9>=60:
                        # Capture is closed and CSV/JSONL are flushed before the
                        # latest synchronous sample. A changed tail continues
                        # this same bounded warmup; it is never waved through.
                        with monitor.lock:
                            latest=monitor.tick("postwarm",GIB)
                            proof=monitor.tail()
                            if proof["ready"]:
                                ended=clock();window.update(complete=True,end_monotonic_ns=ended,duration_s=(ended-start)/1e9)
                                monitor.set_phase("measurement",GIB)
                                report["warmup_handoff"]=dict(policy=protocol()["handoff_policy"],accepted=True,latest_sample_monotonic_ns=latest["monotonic_ns"],warmup_end_monotonic_ns=ended,activation_monotonic_ns=monitor.formal_activation["monotonic_ns"],capture_closed=True,records_flushed=True,complete_EOF_passes=len(window["passes"]))
                                stop=True
                            else:
                                print("TRT_HANDOFF_CONTINUE_WARMUP elapsed_s={:.2f} cap_s=300 reason={}".format(elapsed,proof.get("reason")),flush=True)
                else:
                    if len(rows)>=minimum_frames and elapsed>=60:stop=True
                    print("{} frames={} elapsed={:.2f}s eof={}".format(name, len(rows), elapsed, pass_row["eof_seen"]), flush=True)
            cs.flush()
            ps.flush()
        require(name == "warmup" or all(p["complete"] for p in window["passes"]), "Measurement has partial pass")
        window["complete"] = True
    finally:
        if cap is not None:
            cap.release()
        if window["end_monotonic_ns"] is None:window["end_monotonic_ns"] = clock()
        window["duration_s"] = (window["end_monotonic_ns"] - start) / 1e9
        window["frames"] = len(rows)
    require(60 <= window["duration_s"] <= limit, name + " window duration outside fixed bounds")
    report[name + "_latency"] = latency(rows)
    return window
