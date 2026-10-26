"""T1 low-score cache contract; does not modify the qualified C1/R2 validator."""
import math

def validate_values(values):
    if len(values)>300:
        raise ValueError("T1 cache has more than300 detections")
    for index,b in enumerate(values):
        if len(b)!=6 or not all(math.isfinite(x) for x in b):
            raise ValueError("T1 cache row{} requires6 finite fields".format(index))
        if not (0<=b[0]<=b[2]<=3840 and 0<=b[1]<=b[3]<=2160):
            raise ValueError("T1 cache row{} invalid native xyxy bounds".format(index))
        if not (.001<=b[4]<=1 and b[5]==0):
            raise ValueError("T1 cache row{} invalid confidence/class: {}/{}".format(index,b[4],b[5]))

def one_frame(frame,row,model,dtype,nms,scale,job,report,torch,cv2,np,events):
 # Keep the historical processing timer; whole-window N/T includes validation
 # and per-frame CPU detections/JSONL/CSV writes outside this diagnostic timer.
 torch.cuda.synchronize();start=time.monotonic_ns()
 padded,ratio,pad=c1.letterbox_native(frame,cv2)
 rgb=np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1))
 tensor=torch.from_numpy(rgb).to(device='cuda:0',dtype=dtype).unsqueeze(0)/255.
 event_start,event_end=events;event_start.record()
 report['_forward_gate']['open']=True
 try:raw=model(tensor,augment=False)
 finally:report['_forward_gate']['open']=False
 event_end.record()
 prediction=raw if torch.is_tensor(raw) else raw[0]
 kw=dict(conf_thres=.001,iou_thres=.65,classes=None,agnostic=False,multi_label=False)
 if job['backend']=='v8':kw.update(nc=1,max_det=300,end2end=False)
 detections=nms(prediction.float().clone(),**kw);detection=detections[0].float()
 if len(detection):detection[:,:4]=scale(tensor.shape[2:],detection[:,:4],frame.shape,ratio_pad=(ratio,pad))
 cpu=detection.cpu();torch.cuda.synchronize();end=time.monotonic_ns()
 validation_start=time.monotonic_ns();forward=float(event_start.elapsed_time(event_end))
 require(list(tensor.shape)==[1,3,640,640] and tensor.dtype==dtype and tensor.device.type=='cuda' and bool(torch.isfinite(tensor).all().item()),'Input shape/dtype/device/finite')
 c1.tensor_records(raw,torch)
 require(list(prediction.shape)==([1,25200,6] if job['backend']=='v7' else [1,5,8400]) and prediction.dtype==dtype,'Decoded shape/dtype')
 require(len(detections)==1 and detection.device.type=='cuda' and detection.device.index==0 and cpu.ndim==2 and cpu.shape[1]==6 and len(cpu)<=300 and bool(torch.isfinite(cpu).all().item()),'NMS CPU/CUDA shape/finite')
 values=cpu.tolist();validate_values(values)
 require(math.isfinite(forward) and forward>0 and end>start,'Diagnostic timing')
 return dict(image_id=row['id'],image=row['edge_image'],start_monotonic_ns=start,end_monotonic_ns=end,processing_ms=(end-start)/1e6,forward_cuda_event_ms=forward,detection_count=len(values),validation_and_record_conversion_ms=(time.monotonic_ns()-validation_start)/1e6,all_outputs_finite=True),values
