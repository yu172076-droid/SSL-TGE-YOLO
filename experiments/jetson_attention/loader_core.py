"""AT loader derived from accepted C1 load_model; explicit changed calls below.
Preserves native CPU fusion, no dummy forward, flags, dtype and grid handling.
"""
import inspect
from common import require,sha,weight_path
from runtime_adapter import c1,structural_member
fuse_without_profile=c1.fuse_without_profile
def load_model(job, precision, framework, report, torch):
    head_type, conv_type, nms, scale = framework
    weight = weight_path(job)
    require(sha(weight) == job["weight_sha256"], "Weight changed immediately before load")
    kw = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kw["weights_only"] = False  # Only the independently pinned trusted local objects.
    report["load_kwargs"] = kw
    checkpoint = torch.load(str(weight), **kw)
    model = structural_member(checkpoint, job, report, torch)
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
    require(info["parameters_before_fusion"] == job["parameters"], "Parameter count mismatch; buffers discovered separately")
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
