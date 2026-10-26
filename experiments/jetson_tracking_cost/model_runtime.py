"""C4 three-model adapter to the accepted complete-pass video/power protocol.
Only prepare_model imports GPU packages. No model forward during preparation.
"""
import argparse, hashlib, importlib.util, json, math, os
from pathlib import Path
import sys, time
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
C1_SHA='353475e038c4ef39958f9017568c05277d94f1036bfbd2b2de05c64b3c2bdff8'
C2_SHA='8525ef18c80edf83c90b766d861e909d0f43ea727a89e776f5f4192a08874ad0'
GIB=1024**3;NATIVE_BYTES=24883200
def require(v,msg):
 if not v:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def module(name,path):
 s=importlib.util.spec_from_file_location(name,str(path));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
c1=module('c4_c1',ROOT/'c1_adapter/precheck.py')
guard=module('c4_guard',ROOT/'c1_adapter/resource_guard.py')
Monitor=guard.Monitor
ARGS=None
require_jetson=c1.require_jetson
def boot_id():return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
def stop_child(child):
 if child is not None and child.poll() is None:
  child.terminate()
  try:child.wait(timeout=10)
  except Exception:child.kill();child.wait(timeout=10)
def validate_values(values):return c1.validate_values(values,3840,2160)

def select(args):
 contract=read(ROOT/'c1_adapter/model_contract.json')
 job=next((m for m in contract['models'] if m['id']==args.job),None)
 require(job is not None and args.replicate in (1,2,3),'Unknown job/replicate')
 return job

def qualification(args,verify_actual=True):
 job=select(args);q=read(ROOT/'qualification.json');ap=read(ROOT/'accuracy_summary.json');audit=read(ROOT/'c3_audit.json')
 require(sha(ROOT/'accuracy_summary.json')==q['accuracy_summary_sha256'] and (ROOT/'accuracy_summary.sha256').read_text(encoding='ascii').split()==[q['accuracy_summary_sha256'],'accuracy_summary.json'],'AP receipt hash')
 require(sha(ROOT/'c3_audit.json')==q['c3_audit_sha256'] and audit['status']=='passed' and not audit['failures'] and audit['original_return_unchanged'],'C3 audit receipt')
 require(ap['parent_export_report_sha256']==audit['parent_report_sha256']==q['c3_parent_sha256'] and ap['formal_accuracy_completed'] and ap['fixed_seed']==101 and ap['test_used_for_selection'] is False,'C3 AP provenance')
 gates=ap['validation_precision_gate'];require(len(gates)==3 and {v['job'] for v in gates}=={m['id'] for m in read(ROOT/'c1_adapter/model_contract.json')['models']},'Three val gates')
 for g in gates:
  pair={p:next(v for v in ap['metrics'] if v['job']==g['job'] and v['split']=='val' and v['precision']==p) for p in ('FP32','FP16')}
  delta={k:pair['FP16']['metrics'][k]-pair['FP32']['metrics'][k] for k in ('AP50_95','AP50')}
  require(g['accepted'] is True and g['max_allowed_drop']==.005 and g['validation_FP16_minus_FP32']==delta and all(math.isfinite(d) and d>=-.005 for d in delta.values()),'Val engineering precision gate')
 parent=read(ROOT/'accepted_c2_parent.json');wpath=ROOT/('accepted_c2_'+job['id']+'.json');worker=read(wpath)
 require(sha(ROOT/'accepted_c2_parent.json')==C2_SHA and sha(wpath)==q['c2_worker_sha256'][job['id']],'C2 accepted receipts')
 require(parent['status']=='passed' and not parent['pending'] and worker['status']=='passed' and worker['job']==job['id'] and worker['weight_sha256']==job['sha256'] and worker['precision']=='FP16' and worker['forward_calls_started']==130 and worker['video_unchanged'],'C2 qualification')
 if verify_actual:
  directory=PROJECT/'returns/models_video_short_20261014_01'
  require(sha(directory/'video_short_report.json')==C2_SHA,'Actual C2 parent differs')
  require((directory/'video_short_report.sha256').read_text(encoding='ascii').split()==[C2_SHA,'video_short_report.json'],'Actual C2 sidecar')
  for a in parent['artifacts']:
   p=c1.safe_path(directory,a['path']);require(p.stat().st_size==a['bytes'] and sha(p)==a['sha256'],'Actual C2 artifact differs')
 return job,worker,q

def prerequisites(args):
 ca=argparse.Namespace(bundle=args.bundle,inputs=args.inputs,expected_package_manifest_sha=C1_SHA)
 contract,delivery,smoke=c1.verify_all(ca)
 job,accepted,q=qualification(args)
 require(sha(args.bundle/job['weight'])==job['sha256'],'Selected weight differs')
 return job,accepted,q,delivery

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
 kw=dict(conf_thres=.25,iou_thres=.65,classes=None,agnostic=False,multi_label=False)
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

def model_signature(info):
 # Normalize only JSON's integer mapping keys for class names. All other
 # structure, checkpoint, shape/dtype/device and source checks stay exact.
 result = dict(info)
 names = result.get('names')
 if isinstance(names, dict):
  require(all(type(k) is int or isinstance(k, str) for k in names), 'Invalid class-name key type')
  normalized = {str(k): value for k, value in names.items()}
  require(len(normalized) == len(names), 'Colliding class-name keys')
  result['names'] = normalized
 return result

def prepare_model(output,report,state):
 args=ARGS;require(args is not None,'Runtime args not bound')
 job,accepted,q,delivery=prerequisites(args)
 c1.local_runtime(output)
 monitor=Monitor(guard.memory_sample,output/'process_memory_samples.jsonl');state['monitor']=monitor
 monitor.set_phase('before_model_load',GIB);monitor.tick();monitor.start()
 import torch
 import cv2
 import numpy as np
 import torchvision
 require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Single qualified CUDA device required')
 framework=c1.import_source(args.bundle.resolve(),job['backend'])
 torch.set_num_threads(6);cv2.setNumThreads(1)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
 torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
 environment=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cv2=cv2.__version__,numpy=np.__version__,torchvision=torchvision.__version__,device=torch.cuda.get_device_name(0),torch_threads=torch.get_num_threads(),cv2_threads=cv2.getNumThreads(),module_origins={m.__name__:str(Path(m.__file__).resolve()) for m in (torch,cv2,np,torchvision)})
 require(environment==accepted['environment'],'Runtime differs from accepted C2')
 report.update(environment=environment,runtime_flags=dict(c1.RUNTIME_FLAGS),job=job['id'],replicate=args.replicate,weight_sha256=job['sha256'],bundle_manifest_sha256=c1.BUNDLE_SHA,input_manifest_sha256=c1.INPUT_SHA,adapter_manifest_sha256=C1_SHA,accepted_c2_report_sha256=C2_SHA,accuracy_summary_sha256=q['accuracy_summary_sha256'],c3_parent_sha256=q['c3_parent_sha256'],source_origins_before_load=c1.source_origins(args.bundle.resolve(),job['backend']))
 model,dtype,nms,scale=c1.load_model(args.bundle.resolve(),job,'FP16',framework,report,torch)
 report['model']=model_signature(report['model'])
 require(report['model']==model_signature(accepted['model']),'Model structure differs from accepted C2')
 report['models']={'FP16':report['model']}
 cap=monitor.tick('model_loaded_before_decoder_open',3*GIB)
 require(cap['proc_meminfo']['MemAvailable_bytes']>=3*GIB+NATIVE_BYTES,'Pre-decoder capacity below 3GiB plus frame')
 c1.snapshot(output,args.bundle.resolve(),delivery,args.inputs.resolve())
 events=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
 def infer(frame,row):
  monitor.check()
  return one_frame(frame,row,model,dtype,nms,scale,job,report,torch,cv2,np,events)
 return cv2,torch,monitor,infer
