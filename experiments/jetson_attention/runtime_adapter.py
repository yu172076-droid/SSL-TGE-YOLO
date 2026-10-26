"""Same C1 native numerical path; AT structural discovery before fusion."""
import sys
from pathlib import Path
from common import ROOT, PROJECT, require, sha, module, protocol, weight_path

c1 = module('AT_P1_qualified_C1', ROOT / 'qualified/c1_adapter.py')


def framework():
    base = PROJECT / protocol()['attention_directory'] / 'source'
    require(not any(n == 'models' or n.startswith('models.') or n == 'utils' or n.startswith('utils.') for n in sys.modules), 'Model source imported before isolation')
    sys.path.insert(0, str(base))
    from models.yolo import IDetect
    from models.common import Conv
    from utils.general import non_max_suppression, scale_coords
    return IDetect, Conv, non_max_suppression, scale_coords


def origins():
    base = (PROJECT / protocol()['attention_directory'] / 'source').resolve()
    result = {}
    for name, value in list(sys.modules.items()):
        if any(name == prefix or name.startswith(prefix + '.') for prefix in ('models', 'utils')):
            filename = getattr(value, '__file__', None)
            if filename:
                p = Path(filename).resolve()
                require(base in p.parents, 'Mixed frozen/installed source: ' + name)
                result[name] = dict(path=str(p), sha256=sha(p))
    require(result, 'No frozen model origins found')
    return result


def structural_member(checkpoint, job, report, torch):
    model = c1.member(checkpoint)
    require(len(model.model) == 58, 'Expected 58 global nodes')
    attention = model.model[55]
    expected = {'identity': torch.nn.Identity}
    from models.common import CA_att, CBAM_att, Eca_layer
    expected.update(ca=CA_att, cbam=CBAM_att, eca=Eca_layer)
    require(type(attention) is expected[job['id']], 'Wrong pure attention class')
    require(getattr(attention, 'f', None) in (45, [45]) and getattr(attention, 'i', None) == 55, 'P4 source/node differs')
    require(sum(p.numel() for p in attention.parameters()) == job['module_parameters'], 'Attention parameter count differs')
    require(sum(p.numel() for p in model.parameters()) == job['parameters'], 'Unfused parameter count differs')
    if job['id'] == 'ca':
        require(attention.m[0].in_channels == 128 and attention.m[0].out_channels == 8
                and attention.conv_h.out_channels == attention.conv_w.out_channels == 128, 'CA width differs')
    elif job['id'] == 'eca':
        require(tuple(attention.conv.kernel_size) == (3,), 'ECA kernel differs')
    head = model.model[-1]
    require(list(head.f) == [54, 55, 56] and [m.in_channels for m in head.m] == [128, 128, 512], 'Head widths/inputs differ')
    report['AT_structure_before_fusion'] = dict(nodes=58, layer_index=55, source_index=45,
        module_class=type(attention).__module__ + '.' + type(attention).__name__,
        module_parameters=job['module_parameters'], parameters=job['parameters'],
        actual_buffer_elements=sum(b.numel() for b in model.buffers()),
        buffer_count_role='measured discovery, not an independently accepted prior buffer count',
        detect_input_channels=[m.in_channels for m in head.m], P4_runtime_shapes=[])
    return model


def load(job, precision, report, torch):
    fw = framework()
    report['source_origins_before_load'] = origins()
    # The core is copied from accepted C1 with one declared AT change: buffer
    # count is discovered, while strict pre-fusion parameters/structure remain.
    from loader_core import load_model
    model, dtype, nms, scale = load_model(job, precision, fw, report, torch)
    report['structure_hook_calls'] = 0
    def pre(module, values):
        require(len(values) == 1 and list(values[0].shape) == [1, 128, 40, 40], 'Actual P4 input shape differs')
    handles=[]
    def post(module, values, result):
        require(list(result.shape) == [1, 128, 40, 40] and result.dtype == dtype
                and result.device.type == 'cuda' and bool(torch.isfinite(result).all().item()), 'Actual P4 output differs')
        report['structure_hook_calls'] += 1
        if report.get('shape_hook_once'):
            for handle in handles:handle.remove()
        if not report['AT_structure_before_fusion']['P4_runtime_shapes']:
            report['AT_structure_before_fusion']['P4_runtime_shapes'].append(dict(input=[1,128,40,40], output=list(result.shape), dtype=str(result.dtype)))
    handles.append(model.model[55].register_forward_pre_hook(pre))
    handles.append(model.model[55].register_forward_hook(post))
    return model, dtype, nms, scale


def infer(frame, confidence, model, dtype, nms, scale, report, torch, cv2, np):
    padded, ratio, pad = c1.letterbox_native(frame, cv2)
    rgb = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1))
    tensor = torch.from_numpy(rgb).to(device='cuda:0', dtype=dtype).unsqueeze(0) / 255.
    require(list(tensor.shape) == [1,3,640,640] and bool(torch.isfinite(tensor).all().item()), 'Invalid native tensor')
    torch.cuda.synchronize()
    report['forward_calls_started'] += 1
    report['_forward_gate']['open'] = True
    try:
        with torch.no_grad():
            raw = model(tensor, augment=False)
    finally:
        report['_forward_gate']['open'] = False
    torch.cuda.synchronize()
    tensors = c1.tensor_records(raw, torch)
    prediction = raw if torch.is_tensor(raw) else raw[0]
    require(list(prediction.shape) == [1,25200,6] and prediction.dtype == dtype, 'Decoded shape/dtype differs')
    dets = nms(prediction.float().clone(), conf_thres=confidence, iou_thres=.65,
               classes=None, agnostic=False, multi_label=False)
    require(len(dets) == 1, 'NMS batch differs')
    det = dets[0].float()
    require(det.ndim == 2 and det.shape[1] == 6 and det.device.type == 'cuda', 'NMS shape/device differs')
    if len(det):
        det[:, :4] = scale(tensor.shape[2:], det[:, :4], frame.shape, ratio_pad=(ratio,pad))
    values = det.cpu().tolist()
    validate_values(values, frame.shape[1], frame.shape[0], confidence)
    return dict(native_shape=list(frame.shape), native_bytes=int(frame.nbytes), tensor_shape=list(tensor.shape),
                tensor_dtype=str(dtype), decoded_shape=list(prediction.shape), nms_dtype=str(det.dtype),
                raw_tensors=tensors, detections_xyxy_conf_class=values, detection_count=len(values))


def validate_values(values, width, height, confidence):
    import math
    require(len(values) <= 300, 'NMS max_det differs')
    for b in values:
        require(len(b) == 6 and all(isinstance(x,(int,float)) and math.isfinite(x) for x in b)
                and 0 <= b[0] <= b[2] <= width and 0 <= b[1] <= b[3] <= height
                and confidence < b[4] <= 1 and b[5] == 0, 'Invalid restored six-field detection')


def coco_boxes(row, values):
    validate_values(values, row['width'], row['height'], .001)
    predictions, excluded = [], []
    for x1,y1,x2,y2,score,cls in values:
        if x2 <= x1 or y2 <= y1:
            excluded.append(dict(image_id=row['id'], xyxy=[x1,y1,x2,y2], score=score))
        else:
            predictions.append(dict(image_id=row['id'], category_id=1, bbox=[x1,y1,x2-x1,y2-y1], score=score))
    return predictions, excluded
