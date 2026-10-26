"""C1 only: three frozen b80/s101 models, five native val images, FP32/FP16.
No AP, FPS, power, tracking or export results are produced.
Python 3.8 compatible. No package installation or automatic retries.
"""
import argparse
import datetime
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
BUNDLE_SHA = "a8e8e49b9589046559a2e4a216dce4c0194ac82b0dd9e21f5758b190322ead6d"
INPUT_SHA = "d1e04100fae1cd94e9847482fd41808179b20c343131f5f64c34ae39d6d2e39d"
PROTOCOL = "b80_s101_three_model_load_and_five_native_val_images_fp32_fp16_v1"
GIB = 1024 ** 3
IDS = [16, 33, 69, 85, 92]
CHILD_CAP_S = 300
RUNTIME_FLAGS = dict(torch_num_threads=6, opencv_num_threads=1,
                     cudnn_benchmark=False, cudnn_deterministic=False,
                     cudnn_allow_tf32=True, cuda_matmul_allow_tf32=False)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, obj):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def signed_report(output, name, obj):
    dest = output / (name + ".json")
    write(dest, obj)
    with (output / (name + ".sha256")).open("x", encoding="ascii") as stream:
        stream.write(sha(dest) + "  " + dest.name + "\n")


def safe_path(base, name):
    rel = PurePosixPath(name)
    require(name and not rel.is_absolute() and ".." not in rel.parts
            and "\\" not in name and ":" not in name, "Unsafe relative path: " + name)
    p = base.joinpath(*rel.parts)
    require(base.resolve() in p.resolve().parents and not p.is_symlink(), "Path escaped root")
    return p


def verify_tree(base, expected, count, payload=None):
    manifest_path = base / "delivery_manifest.json"
    require(sha(manifest_path) == expected, "Independent manifest SHA mismatch: " + str(base))
    require((base / "delivery_manifest.sha256").read_text(encoding="ascii").split()
            == [expected, "delivery_manifest.json"], "Manifest sidecar mismatch")
    m = read(manifest_path)
    rows = m["files"]
    names = [r["path"] for r in rows]
    require(len(rows) == count and len(set(names)) == count, "Manifest count/duplicates differ")
    if payload is not None:
        require(set(names) == set(payload), "Unexpected execution package payload")
    expected_names = set(names) | {"delivery_manifest.json", "delivery_manifest.sha256"}
    actual = {p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file()}
    require(actual == expected_names, "Missing/unlisted file(s): " + str(sorted(actual ^ expected_names)))
    for row in rows:
        p = safe_path(base, row["path"])
        require(p.stat().st_size == row["bytes"] and sha(p) == row["sha256"],
                "File size/SHA mismatch: " + row["path"])
    return m


def verify_all(args):
    package = verify_tree(ROOT, args.expected_package_manifest_sha, 4,
                          ["precheck.py", "resource_guard.py", "model_contract.json", "README.md"])
    contract = read(ROOT / "model_contract.json")
    require(contract["protocol"] == PROTOCOL and contract["bundle_manifest_sha256"] == BUNDLE_SHA
            and contract["input_manifest_sha256"] == INPUT_SHA, "Contract identity mismatch")
    bundle = args.bundle.resolve()
    inputs = args.inputs.resolve()
    delivery = verify_tree(bundle, BUNDLE_SHA, 697)
    table = {r["path"]: r for r in delivery["files"]}
    for job in contract["models"]:
        require(table[job["weight"]]["sha256"] == job["sha256"], "Contract weight differs")
    require(sha(inputs / "input_manifest.json") == INPUT_SHA, "Frozen input manifest differs")
    im = read(inputs / "input_manifest.json")
    require(im["image_count"] == 149 and im["split"] == "val" and len(im["rows"]) == 149,
            "Input split/count differs")
    require([r["id"] for r in im["rows"][:5]] == IDS, "Five fixed IDs differ")
    for row in im["rows"]:
        for path_key, hash_key in [("edge_image", "image_sha256"), ("edge_label", "label_sha256")]:
            require(sha(safe_path(inputs, row[path_key])) == row[hash_key], "Input asset changed")
    return contract, delivery, im["rows"][:5]


def require_jetson():
    device = Path("/proc/device-tree/model").read_bytes().replace(b"\0", b"").decode("utf-8", "replace")
    require(platform.system() == "Linux" and platform.machine().lower() in ("aarch64", "arm64")
            and "jetson" in device.lower(), "This command runs only on a Jetson")
    require(os.geteuid() != 0, "Run as the ordinary user, without sudo")
    require(Path(sys.executable).resolve() == Path("/usr/bin/python3.8").resolve(),
            "Use the previously qualified /usr/bin/python3.8")
    return device.strip()


def deny_network(*args, **kwargs):
    raise OSError("C1 network access disabled; no downloads or dependency installation")


def local_runtime(output):
    for name in ("MPLCONFIGDIR", "XDG_CACHE_HOME", "TORCH_HOME", "CUDA_CACHE_PATH",
                 "TMPDIR", "TEMP", "TMP", "YOLO_CONFIG_DIR"):
        p = output / "cache" / name.lower()
        p.mkdir(parents=True, exist_ok=False)
        os.environ[name] = str(p)
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", YOLO_OFFLINE="true",
                      YOLO_AUTOINSTALL="false", YOLO_VERBOSE="false",
                      WANDB_MODE="disabled", OMP_NUM_THREADS="6", MKL_NUM_THREADS="6",
                      OPENBLAS_NUM_THREADS="6", NUMEXPR_MAX_THREADS="6")
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        os.environ.pop(name, None)
    sys.dont_write_bytecode = True
    for target, name in [(socket.socket, "connect"), (socket.socket, "connect_ex"),
                         (socket, "create_connection"), (socket, "getaddrinfo")]:
        setattr(target, name, deny_network)


def import_source(bundle, backend):
    if backend == "v7":
        source = bundle / "frozen_source"
        sys.path.insert(0, str(source))
        from models.yolo import IDetect
        from models.common import Conv
        from utils.general import non_max_suppression, scale_coords
        return IDetect, Conv, non_max_suppression, scale_coords
    source = bundle / "ultralytics_8_4_132"
    require("ultralytics" not in sys.modules, "Ultralytics was imported before source isolation")
    spec = importlib.util.spec_from_file_location("ultralytics", str(source / "__init__.py"),
                                                  submodule_search_locations=[str(source)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["ultralytics"] = module
    spec.loader.exec_module(module)
    require(module.__version__ == "8.4.132", "Frozen Ultralytics version differs")
    from ultralytics.nn.modules.head import Detect
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes
    from ultralytics.utils import SETTINGS
    SETTINGS.update(sync=False, clearml=False, comet=False, dvc=False, mlflow=False,
                    raytune=False, wandb=False, vscode_msg=False, openvino_msg=False)
    return Detect, None, non_max_suppression, scale_boxes


def source_origins(bundle, backend):
    prefix = ("models", "utils", "sort") if backend == "v7" else ("ultralytics",)
    base = bundle / ("frozen_source" if backend == "v7" else "ultralytics_8_4_132")
    origins = {}
    for name, module in list(sys.modules.items()):
        if any(name == p or name.startswith(p + ".") for p in prefix):
            filename = getattr(module, "__file__", None)
            if filename:
                p = Path(filename).resolve()
                require(base in p.parents, "Mixed installed/frozen source: " + name)
                origins[name] = dict(path=str(p), sha256=sha(p))
    require(origins, "No model source origins found")
    return origins


def member(checkpoint):
    require(isinstance(checkpoint, dict), "Expected a checkpoint dictionary")
    require(checkpoint.get("ema") is None and checkpoint.get("optimizer") is None
            and checkpoint.get("updates") is None, "Expected stripped checkpoint actual None members")
    require(checkpoint.get("model") is not None, "Checkpoint model is missing")
    return checkpoint["model"]


def fuse_without_profile(model, backend, torch):
    # Native v7 fuse() ends with self.info(), whose thop branch forwards a
    # copied model with zeros. Suppress only that optional information callback.
    # Frozen source bytes remain unchanged; no dummy forward is allowed in C1.
    with torch.no_grad():
        if backend == "v7":
            old = model.__dict__.get("info")
            existed = "info" in model.__dict__
            model.info = lambda *a, **kw: None
            try:
                model = model.fuse()
            finally:
                if existed:
                    model.info = old
                else:
                    delattr(model, "info")
        else:
            model = model.fuse(verbose=False)
    return model


def load_model(bundle, job, precision, framework, report, torch):
    head_type, conv_type, nms, scale = framework
    weight = bundle / job["weight"]
    require(sha(weight) == job["sha256"], "Weight changed immediately before load")
    kw = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kw["weights_only"] = False  # Only the independently pinned trusted local objects.
    report["load_kwargs"] = kw
    checkpoint = torch.load(str(weight), **kw)
    model = member(checkpoint)
    require(isinstance(model, torch.nn.Module), "model member is not an nn.Module")
    head = model.model[-1]
    info = dict(checkpoint_member="model", checkpoint_keys=sorted(map(str, checkpoint)),
                epoch=checkpoint.get("epoch"), ema_is_none=checkpoint.get("ema") is None,
                optimizer_is_none=checkpoint.get("optimizer") is None,
                parameters_before_fusion=sum(p.numel() for p in model.parameters()),
                buffer_elements_before_fusion=sum(p.numel() for p in model.buffers()),
                stored_parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
                model_class=type(model).__module__ + "." + type(model).__name__,
                head_class=type(head).__module__ + "." + type(head).__name__,
                nc=getattr(head, "nc", None), names=getattr(model, "names", None),
                head_flags={k: bool(getattr(head, k, False))
                            for k in ("export", "end2end", "include_nms", "concat", "xyxy")},
                fusion="native on CPU; v7 info callback suppressed / v8 verbose=False")
    report["model"] = info
    report["_forward_gate"] = dict(open=False, intercepted_calls=0)
    def forward_guard(module, values):
        gate = report["_forward_gate"]
        require(gate["open"], "Unexpected forward before actual native-image gate")
        gate["intercepted_calls"] += 1
    model.register_forward_pre_hook(forward_guard)
    require(info["parameters_before_fusion"] == job["parameters"]
            and info["buffer_elements_before_fusion"] == job["buffer_elements"], "Parameter/buffer count mismatch")
    require(info["stored_parameter_dtypes"] == ["torch.float16"], "Stored parameter dtype differs")
    names = info["names"]
    require(names == ["fish"] or names == {0: "fish"} or names == {"0": "fish"}, "Expected only fish class")
    require(type(head) is head_type and head.nc == 1
            and not any(info["head_flags"].values()), "Head class/nc/output mode differs")
    require(head.stride.detach().cpu().tolist() == [8.0, 16.0, 32.0], "Head strides differ")
    if job["backend"] == "v7":
        require(head.no == 6, "Expected v7 decoded objectness plus one-class score")
        for m in model.modules():
            if type(m) in (torch.nn.Hardswish, torch.nn.LeakyReLU, torch.nn.ReLU,
                           torch.nn.ReLU6, torch.nn.SiLU):
                m.inplace = True
            elif type(m) is torch.nn.Upsample:
                m.recompute_scale_factor = None
            elif type(m) is conv_type:
                m._non_persistent_buffers_set = set()
    model = fuse_without_profile(model.float().eval(), job["backend"], torch).eval()
    model.requires_grad_(False)
    model = model.to("cuda:0")
    model = model.half() if precision == "FP16" else model.float()
    dtype = torch.float16 if precision == "FP16" else torch.float32
    head = model.model[-1]
    if job["backend"] == "v7":
        head.grid = [torch.zeros(1, device="cuda:0", dtype=dtype) for _ in range(head.nl)]
        head.stride = head.stride.to(device="cuda:0", dtype=torch.float32)
    else:
        head.shape = None  # Rebuild anchors in the first real-frame forward.
        head.anchors = torch.empty(0, device="cuda:0", dtype=dtype)
        head.strides = torch.empty(0, device="cuda:0", dtype=dtype)
    if torch.is_tensor(getattr(model, "stride", None)):
        model.stride = model.stride.to(device="cuda:0", dtype=torch.float32)
    require(all(p.device.type == "cuda" and p.device.index == 0 and p.dtype == dtype
                for p in model.parameters()), "Wrong parameter precision/device or CPU fallback")
    require(all(b.device.type == "cuda" for b in model.buffers()), "CPU model buffer")
    info.update(parameters_after_fusion=sum(p.numel() for p in model.parameters()),
                parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
                parameter_devices=sorted({str(p.device) for p in model.parameters()}),
                runtime_stride=head.stride.detach().cpu().tolist(),
                runtime_stride_dtype=str(head.stride.dtype),
                runtime_stride_device=str(head.stride.device),
                anchor_cache="invalidated; first real input rebuilds native cache")
    del checkpoint
    return model, dtype, nms, scale


def letterbox_native(frame, cv2):
    # Same fixed square geometry as frozen v7 letterbox(auto=False,scaleFill=False,
    # scaleup=True,stride=32), shared by both backends. No original-frame cache.
    h, w = frame.shape[:2]
    r = min(640 / h, 640 / w)
    size = (int(round(w * r)), int(round(h * r)))
    dw, dh = (640 - size[0]) / 2, (640 - size[1]) / 2
    image = frame if (w, h) == size else cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(image, int(round(dh - .1)), int(round(dh + .1)),
                              int(round(dw - .1)), int(round(dw + .1)),
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return image, (r, r), (dw, dh)


def tensor_records(value, torch, path="raw"):
    records = []
    if torch.is_tensor(value):
        require(value.device.type == "cuda" and bool(torch.isfinite(value).all().item()),
                "CPU/nonfinite raw tensor: " + path)
        records.append(dict(path=path, shape=list(value.shape), dtype=str(value.dtype), device=str(value.device)))
    elif isinstance(value, (tuple, list)):
        for i, v in enumerate(value):
            records.extend(tensor_records(v, torch, path + "[{}]".format(i)))
    elif isinstance(value, dict):
        for k, v in value.items():
            records.extend(tensor_records(v, torch, path + "." + str(k)))
    else:
        require(value is None, "Unexpected raw output type: " + str(type(value)))
    return records


def validate_values(values, width, height):
    require(len(values) <= 300, "Too many detections")
    for b in values:
        require(len(b) == 6 and all(math.isfinite(x) for x in b)
                and 0 <= b[0] <= b[2] <= width and 0 <= b[1] <= b[3] <= height
                and .25 < b[4] <= 1 and b[5] == 0, "Invalid restored xyxy/conf/class output")


def run_images(rows, inputs, job, model, dtype, nms, scale, monitor, report, output, torch, cv2, np):
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            monitor.check()
            require(sha(inputs / row["edge_image"]) == row["image_sha256"], "Image changed before decode")
            frame = cv2.imread(str(inputs / row["edge_image"]), cv2.IMREAD_COLOR)
            try:
                require(frame is not None and frame.dtype.name == "uint8"
                        and list(frame.shape) == [row["height"], row["width"], 3]
                        and frame.nbytes == row["height"] * row["width"] * 3, "Native decode differs")
                if index == 0:
                    monitor.tick("first_native_image_resident_before_first_forward", 3 * GIB)
                    monitor.set_phase("five_image_inference", GIB)
                padded, ratio, pad = letterbox_native(frame, cv2)
                rgb = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1))
                tensor = torch.from_numpy(rgb).to(device="cuda:0", dtype=dtype).unsqueeze(0) / 255.0
                require(list(tensor.shape) == [1, 3, 640, 640] and tensor.dtype == dtype
                        and bool(torch.isfinite(tensor).all().item()), "Invalid tensor input")
                monitor.check()
                torch.cuda.synchronize()
                report["forward_calls_started"] += 1
                report["_forward_gate"]["open"] = True
                try:
                    with torch.no_grad():
                        raw = model(tensor, augment=False)
                finally:
                    report["_forward_gate"]["open"] = False
                torch.cuda.synchronize()
                tensors = tensor_records(raw, torch)
                prediction = raw if torch.is_tensor(raw) else raw[0]
                expected = [1, 25200, 6] if job["backend"] == "v7" else [1, 5, 8400]
                require(list(prediction.shape) == expected and prediction.dtype == dtype,
                        "Decoded prediction shape/dtype differs")
                kwargs = dict(conf_thres=.25, iou_thres=.65, classes=None, agnostic=False, multi_label=False)
                if job["backend"] == "v8":
                    kwargs.update(nc=1, max_det=300, end2end=False)
                detections = nms(prediction.float().clone(), **kwargs)
                require(len(detections) == 1, "NMS batch differs")
                det = detections[0].float()
                require(det.ndim == 2 and det.shape[1] == 6 and det.device.type == "cuda",
                        "Invalid NMS shape/device")
                if len(det):
                    det[:, :4] = scale(tensor.shape[2:], det[:, :4], frame.shape, ratio_pad=(ratio, pad))
                torch.cuda.synchronize()
                values = det.cpu().tolist()
                validate_values(values, row["width"], row["height"])
                record = dict(image_id=row["id"], image_sha256=row["image_sha256"],
                              native_shape=list(frame.shape), native_bytes=int(frame.nbytes),
                              tensor_shape=list(tensor.shape), tensor_dtype=str(tensor.dtype),
                              ratio=list(ratio), pad=list(pad), raw_tensors=tensors,
                              decoded_shape=list(prediction.shape), nms_dtype=str(det.dtype),
                              detections_xyxy_conf_class=values, detection_count=len(values),
                              completed_monotonic_ns=time.monotonic_ns())
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                report["images"].append(dict(image_id=row["id"], detection_count=len(values)))
                report["completed_frames"] += 1
                del padded, rgb, tensor, raw, prediction, detections, det
                monitor.check()
            finally:
                del frame
            monitor.tick("after_image_{}".format(row["id"]), GIB)
            require(sha(inputs / row["edge_image"]) == row["image_sha256"], "Image changed after inference")


def artifacts(output, exclude=()):
    return [dict(path=p.relative_to(output).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
            for p in sorted(output.rglob("*")) if p.is_file()
            and p.relative_to(output).as_posix() not in exclude
            and not p.relative_to(output).as_posix().startswith("cache/")]


def common_report(args):
    return dict(protocol=PROTOCOL, status="failed", session_id=args.session,
                package_manifest_sha256=args.expected_package_manifest_sha,
                bundle_manifest_sha256=BUNDLE_SHA, input_manifest_sha256=INPUT_SHA,
                python=sys.version, python_executable=str(Path(sys.executable).resolve()),
                pid=os.getpid(), boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                start_monotonic_ns=time.monotonic_ns(),
                formal_accuracy_completed=False, formal_speed_completed=False,
                installation_performed=False, performance_metrics_generated=False)


def worker(args):
    output = args.output.resolve() / args.job
    report = common_report(args)
    monitor = None
    code = 1
    try:
        require_jetson()
        require(read(output / "launch.json")["session_id"] == args.session, "Parent session differs")
        contract, delivery, rows = verify_all(args)
        job = next(j for j in contract["models"] if args.job == j["id"] + "_" + args.precision)
        require(args.precision in ("FP32", "FP16"), "Unexpected precision")
        report.update(job=job["id"], precision=args.precision, weight_sha256=job["sha256"],
                      images=[], completed_frames=0, forward_calls_started=0,
                      runtime_flags=dict(RUNTIME_FLAGS), selected_image_ids=IDS)
        local_runtime(output)
        from resource_guard import Monitor, memory_sample
        monitor = Monitor(memory_sample, output / "process_memory_samples.jsonl")
        monitor.tick("before_framework_import", GIB)
        monitor.start()
        import torch
        import cv2
        import numpy as np
        import torchvision
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Expected one visible CUDA device")
        framework = import_source(args.bundle.resolve(), job["backend"])
        torch.set_num_threads(6)
        cv2.setNumThreads(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = False
        report["environment"] = dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                                    cv2=cv2.__version__, numpy=np.__version__, torchvision=torchvision.__version__,
                                    device=torch.cuda.get_device_name(0),
                                    visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                                    torch_threads=torch.get_num_threads(), cv2_threads=cv2.getNumThreads(),
                                    module_origins={m.__name__: str(Path(m.__file__).resolve())
                                                    for m in (torch, cv2, np, torchvision)})
        require(str(torch.__version__) == "2.1.0a0+41361538.nv23.06"
                and torch.version.cuda == "11.4" and cv2.__version__ == "4.13.0"
                and np.__version__ == "1.24.4" and torchvision.__version__ == "0.16.2",
                "Runtime version changed from accepted B1/B2; stop for review")
        report["source_origins_before_load"] = source_origins(args.bundle.resolve(), job["backend"])
        monitor.tick("before_checkpoint_load", GIB)
        model, dtype, nms, scale = load_model(args.bundle.resolve(), job, args.precision, framework, report, torch)
        capacity = 3 * GIB + max(r["width"] * r["height"] * 3 for r in rows)
        monitor.tick("model_loaded_capacity_for_largest_native_image", capacity)
        monitor.set_phase("before_first_native_decode", 3 * GIB)
        run_images(rows, args.inputs.resolve(), job, model, dtype, nms, scale,
                   monitor, report, output, torch, cv2, np)
        report["source_origins_after_inference"] = source_origins(args.bundle.resolve(), job["backend"])
        require(report["completed_frames"] == report["forward_calls_started"] == report["_forward_gate"]["intercepted_calls"] == 5, "Incomplete/extra five actual forwards")
        require(sha(args.bundle.resolve() / job["weight"]) == job["sha256"], "Weight changed")
        monitor.check()
        report["status"] = "passed"
        code = 0
    except BaseException as exc:
        report.update(error=type(exc).__name__ + ": " + str(exc), traceback=traceback.format_exc())
        code = 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        if monitor is not None:
            try:
                monitor.stop()
                monitor.check()
            except BaseException as exc:
                report.update(status="failed", guard_final_error=str(exc))
                code = 1
            report["resource_guard"] = dict(sample_count=len(monitor.samples), failure=str(monitor.failure)
                                            if monitor.failure else None,
                                            max_swap_streak=max([r.get("consecutive_positive_global_swap_intervals", 0)
                                                                 for r in monitor.samples] or [0]))
        report.update(end_monotonic_ns=time.monotonic_ns(), intended_self_exit_code=code)
        report["artifacts"] = artifacts(output, ["console.log"])
        signed_report(output, "worker_report", report)
    return code


def stop_owned(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)


def run_child(command, output, env):
    with (output / "console.log").open("x", encoding="utf-8") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 env=env, cwd=str(PROJECT), start_new_session=True)
        timed_out = False
        interrupted = False
        try:
            child.wait(timeout=CHILD_CAP_S)
        except subprocess.TimeoutExpired:
            timed_out = True
            stop_owned(child)
        except BaseException:
            interrupted = True
            stop_owned(child)
            raise
        finally:
            write(output / "exit_status.json",
                  dict(pid=child.pid, actual_child_exit_code=child.returncode,
                       timed_out=timed_out, interrupted=interrupted, cap_seconds=CHILD_CAP_S,
                       wrapper_exit_code=124 if timed_out else child.returncode))
    return 124 if timed_out else child.returncode


def check_child(output, name, session, precision, job, package_sha):
    r = read(output / "worker_report.json")
    require((output / "worker_report.sha256").read_text(encoding="ascii").split()
            == [sha(output / "worker_report.json"), "worker_report.json"], "Worker report sidecar differs")
    require(r["status"] == "passed" and r["session_id"] == session
            and r["job"] + "_" + r["precision"] == name and r["precision"] == precision
            and r["completed_frames"] == r["forward_calls_started"] == 5
            and r["selected_image_ids"] == IDS and r["intended_self_exit_code"] == 0
            and r["protocol"] == PROTOCOL and r["package_manifest_sha256"] == package_sha
            and r["bundle_manifest_sha256"] == BUNDLE_SHA and r["input_manifest_sha256"] == INPUT_SHA
            and r["weight_sha256"] == job["sha256"] and r["model"]["checkpoint_member"] == "model"
            and r["resource_guard"]["failure"] is None
            and r["_forward_gate"] == dict(open=False, intercepted_calls=5),
            "Worker report incomplete or misbound")
    for row in r["artifacts"]:
        p = safe_path(output, row["path"])
        require(p.stat().st_size == row["bytes"] and sha(p) == row["sha256"], "Worker artifact changed")
    predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    require([p["image_id"] for p in predictions] == IDS, "Prediction IDs/count differ")
    require(r["images"] == [dict(image_id=p["image_id"], detection_count=len(p["detections_xyxy_conf_class"]))
                            for p in predictions], "Prediction/report counts differ")
    for pred in predictions:
        h, w, channels = pred["native_shape"]
        require(channels == 3, "Prediction native channels differ")
        validate_values(pred["detections_xyxy_conf_class"], w, h)
    return r


def snapshot(output, bundle, delivery, inputs):
    pairs = [(p, "package/" + p.name) for p in ROOT.iterdir() if p.is_file()]
    pairs += [(bundle / r["path"], "bundle/" + r["path"]) for r in delivery["files"]
              if not r["path"].startswith("models/")]
    pairs += [(bundle / n, "bundle/" + n) for n in ["delivery_manifest.json", "delivery_manifest.sha256"]]
    pairs += [(inputs / "input_manifest.json", "inputs/input_manifest.json")]
    for source, name in pairs:
        dest = output / "source_snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as a, dest.open("xb") as b:
            shutil.copyfileobj(a, b)
        require(sha(source) == sha(dest), "Source snapshot copy differs")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--expected-package-manifest-sha", required=True)
    p.add_argument("--output", type=Path, default=PROJECT / "returns/models_precheck_20261014_01")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--job", help=argparse.SUPPRESS)
    p.add_argument("--precision", help=argparse.SUPPRESS)
    p.add_argument("--session", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        return worker(args)
    if args.verify_only:
        contract, delivery, rows = verify_all(args)
        print(json.dumps(dict(status="verified_only", bundle_files=len(delivery["files"]),
                              models=len(contract["models"]), selected_ids=[r["id"] for r in rows],
                              heavy_imports_performed=False)))
        return 0
    device = require_jetson()
    expected = (PROJECT / "returns/models_precheck_20261014_01").resolve()
    require(args.output.resolve() == expected and not expected.exists(), "Use new exact _01 return directory; no retries")
    for base in (ROOT, args.bundle.resolve(), args.inputs.resolve()):
        require(base != expected and base not in expected.parents and expected not in base.parents,
                "Output/input/package trees overlap")
    require(shutil.disk_usage(PROJECT).free >= GIB, "Need at least 1 GiB free disk")
    expected.mkdir(parents=True, exist_ok=False)
    args.session = str(uuid.uuid4())
    report = common_report(args)
    report.update(device=device, jobs=[], pending=[], diagnostic_count_comparison=[],
                  limitations=["Five images do not establish AP preservation",
                               "No FPS, energy, tracking, TensorRT or formal repeat claim",
                               "Frozen V8 is export-time environment snapshot, not independent training-time source pin"])
    code = 1
    try:
        contract, delivery, rows = verify_all(args)
        plan = [(j["id"] + "_" + precision, precision) for j in contract["models"]
                for precision in ("FP32", "FP16")]
        report["pending"] = [n for n, precision in plan]
        snapshot(expected, args.bundle.resolve(), delivery, args.inputs.resolve())
        write(expected / "plan.json", dict(session_id=args.session, protocol=PROTOCOL,
                                          jobs=report["pending"], child_cap_seconds=CHILD_CAP_S))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        for name, precision in plan:
            out = expected / name
            out.mkdir(exist_ok=False)
            command = [sys.executable, "-B", str(ROOT / "precheck.py"),
                       "--bundle", str(args.bundle.resolve()), "--inputs", str(args.inputs.resolve()),
                       "--expected-package-manifest-sha", args.expected_package_manifest_sha,
                       "--output", str(expected), "--worker", "--job", name,
                       "--precision", precision, "--session", args.session]
            write(out / "launch.json", dict(session_id=args.session, command=command,
                                           python_executable=sys.executable, start_monotonic_ns=time.monotonic_ns()))
            item = dict(name=name, precision=precision, status="started", actual_exit_code=None)
            report["jobs"].append(item)
            report["pending"].remove(name)
            print("C1 start " + name, flush=True)
            rc = run_child(command, out, env)
            item.update(actual_exit_code=read(out / "exit_status.json")["actual_child_exit_code"],
                        wrapper_exit_code=rc, status="failed" if rc != 0 else "checking_report")
            require(rc == 0, "Child failed; preserve partial output and STOP: " + name)
            job = next(j for j in contract["models"] if name == j["id"] + "_" + precision)
            r = check_child(out, name, args.session, precision, job, args.expected_package_manifest_sha)
            item.update(status="passed", images=r["images"], report_sha256=sha(out / "worker_report.json"))
            print("C1 passed " + name, flush=True)
        verify_all(args)
        for job in contract["models"]:
            a, b = [read(expected / (job["id"] + "_" + q) / "worker_report.json") for q in ("FP32", "FP16")]
            report["diagnostic_count_comparison"].append(dict(job=job["id"], fp32=a["images"], fp16=b["images"],
                                                              count_agreement_required=False,
                                                              formal_AP_comparison=False))
        report["status"] = "passed"
        code = 0
    except BaseException as exc:
        report.update(error=type(exc).__name__ + ": " + str(exc), traceback=traceback.format_exc())
        code = 130 if isinstance(exc, KeyboardInterrupt) else 1
    report.update(end_monotonic_ns=time.monotonic_ns(), intended_launcher_exit_code=code)
    report["artifacts"] = artifacts(expected)
    signed_report(expected, "precheck_report", report)
    print(json.dumps(dict(status=report["status"], output=str(expected), next_action="return_for_main_window_audit")))
    return code


if __name__ == "__main__":
    sys.exit(main())
