"""Accepted native R2 timing/NMS path with AT shape discovery outside formal window."""
import math,time
from common import require
import runtime_adapter as adapter
c1=adapter.c1
def validate_values(values):return adapter.validate_values(values,3840,2160,.25)
def one_frame(frame,row,model,dtype,nms,scale,report,torch,cv2,np,events):
    torch.cuda.synchronize();start=time.monotonic_ns()
    padded,ratio,pad=c1.letterbox_native(frame,cv2)
    rgb=np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1))
    tensor=torch.from_numpy(rgb).to(device='cuda:0',dtype=dtype).unsqueeze(0)/255.
    begin,end_event=events;begin.record();report['_forward_gate']['open']=True
    try:raw=model(tensor,augment=False)
    finally:report['_forward_gate']['open']=False
    end_event.record()
    prediction=raw if torch.is_tensor(raw) else raw[0]
    detections=nms(prediction.float().clone(),conf_thres=.25,iou_thres=.65,classes=None,agnostic=False,multi_label=False)
    detection=detections[0].float()
    if len(detection):detection[:,:4]=scale(tensor.shape[2:],detection[:,:4],frame.shape,ratio_pad=(ratio,pad))
    cpu=detection.cpu();torch.cuda.synchronize();end=time.monotonic_ns()
    validation_start=time.monotonic_ns();forward=float(begin.elapsed_time(end_event))
    require(list(tensor.shape)==[1,3,640,640] and tensor.dtype==dtype and tensor.device.type=='cuda' and bool(torch.isfinite(tensor).all().item()),'Invalid native tensor')
    c1.tensor_records(raw,torch)
    require(list(prediction.shape)==[1,25200,6] and prediction.dtype==dtype,'Wrong decoded tensor')
    require(len(detections)==1 and detection.device.type=='cuda' and cpu.ndim==2 and cpu.shape[1]==6 and bool(torch.isfinite(cpu).all().item()),'Invalid NMS output')
    values=cpu.tolist();validate_values(values)
    require(math.isfinite(forward) and forward>0 and end>start,'Invalid diagnostic timing')
    return dict(image_id=row['id'],image=row['edge_image'],start_monotonic_ns=start,end_monotonic_ns=end,
        processing_ms=(end-start)/1e6,forward_cuda_event_ms=forward,detection_count=len(values),
        validation_and_record_conversion_ms=(time.monotonic_ns()-validation_start)/1e6,all_outputs_finite=True),values
