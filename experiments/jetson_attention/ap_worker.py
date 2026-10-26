"""Owned AT-P1 GPU subprocess. Diagnostic only, no AP/FPS/power claims."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
from common import *


def native_image(row, cv2):
    path = image_path(row)
    require(sha(path) == row['image_sha256'], 'Image changed before decode')
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    require(frame is not None and frame.dtype.name == 'uint8'
            and list(frame.shape) == [row['height'],row['width'],3]
            and frame.nbytes == row['height']*row['width']*3, 'Native image decode differs')
    return frame


def compatibility(rows, out, monitor, call, report, cv2):
    from readiness import capacity
    with (out/'compatibility_records.jsonl').open('x',encoding='utf-8') as stream:
        for index,row in enumerate(rows[:5]):
            monitor.check()
            frame=native_image(row,cv2)
            try:
                if index==0:
                    capacity(monitor,'first_image_capacity_wait',3*GIB)
                    monitor.tick('first_native_image_before_first_forward',3*GIB)
                    monitor.set_phase('compatibility_cold_start',GIB)
                record=call(frame,.25)
                record.update(image_id=row['id'],image_sha256=row['image_sha256'],phase='compatibility',completed_monotonic_ns=time.monotonic_ns())
                stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
                report['compatibility_images']+=1
            finally:
                del frame
            require(sha(image_path(row))==row['image_sha256'],'Image changed after compatibility')
            monitor.tick('compatibility_after_image',GIB)


def accuracy(rows, out, monitor, call, report, cv2):
    from runtime_adapter import coco_boxes
    predictions=dict(val=[],test=[])
    with (out/'image_records.jsonl').open('x',encoding='utf-8') as stream:
        for index,row in enumerate(rows):
            monitor.check()
            if index==5:
                from readiness import settle
                report['protected_diagnostic_admission']=settle(monitor,'accuracy_predictions')
            frame=native_image(row,cv2)
            try:
                if index==0:
                    from readiness import capacity
                    capacity(monitor,'first_image_capacity_wait',3*GIB)
                    monitor.tick('first_native_image_before_first_forward',3*GIB)
                    monitor.set_phase('accuracy_first_five_cold_start',GIB)
                record=call(frame,.001)
                values=record.pop('detections_xyxy_conf_class')
                boxes,excluded=coco_boxes(row,values)
                record.update(image_id=row['id'],split=row['split'],image_sha256=row['image_sha256'],
                              predictions=boxes,zero_area_excluded=excluded,completed_monotonic_ns=time.monotonic_ns())
                stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
                predictions[row['split']].extend(boxes)
                report['completed_by_split'][row['split']]+=1
            finally:
                del frame
            require(sha(image_path(row))==row['image_sha256'],'Image changed after prediction')
            monitor.tick('accuracy_after_image',GIB)
            if (index+1)%50==0:
                print('ATF1_COMMON_ACCURACY {}/298'.format(index+1),flush=True)
    for split in ('val','test'):
        write(out/('predictions_'+split+'.json'),predictions[split])


def short_video(out, monitor, call, report, cv2):
    from readiness import capacity,settle
    p=protocol();video=PROJECT/p['video_relative']
    cap=None
    try:
        cap=cv2.VideoCapture(str(video),cv2.CAP_FFMPEG)
        require(cap.isOpened() and cap.getBackendName()=='FFMPEG','FFMPEG capture unavailable')
        header=dict(backend=cap.getBackendName(),fps=cap.get(cv2.CAP_PROP_FPS),frames=cap.get(cv2.CAP_PROP_FRAME_COUNT),width=cap.get(cv2.CAP_PROP_FRAME_WIDTH),height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        require(header['frames']==1475 and header['width']==3840 and header['height']==2160 and abs(header['fps']-25)<.01,'Video header differs')
        report['video_header']=header
        with (out/'frame_records.jsonl').open('x',encoding='utf-8') as stream:
            for ordinal in range(1,131):
                monitor.check()
                if ordinal==31:
                    # Keep decoder position. No seek, skipped frame or extra
                    # forward; cold-start paging settles before protected probe.
                    report['protected_diagnostic_admission']=settle(monitor,'short_video_probe')
                ok,frame=cap.read()
                require(ok and frame is not None and frame.dtype.name=='uint8'
                        and list(frame.shape)==[2160,3840,3] and frame.nbytes==24883200,'Native short-video frame differs')
                try:
                    if ordinal==1:
                        capacity(monitor,'first_video_frame_capacity_wait',3*GIB)
                        monitor.tick('first_native_video_frame_before_first_forward',3*GIB)
                        monitor.set_phase('short_video_cold_warmup',GIB)
                    record=call(frame,.25)
                    phase='warmup' if ordinal<=30 else 'probe'
                    record.update(source_decode_ordinal=ordinal,phase=phase,completed_monotonic_ns=time.monotonic_ns())
                    stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
                    report['completed_by_phase'][phase]+=1
                finally:
                    del frame
                monitor.tick('short_video_after_frame',GIB)
                if ordinal in (30,80,130):
                    print('AT_P1_SHORT {}/130'.format(ordinal),flush=True)
    finally:
        if cap is not None:cap.release()
    require(sha(video)==p['video_sha256'],'Video changed after short inference')
    report['video_inputs_unchanged']=True


def execute(args):
    out=args.output.resolve();launch=read(out/'launch.json')
    report_value=dict(protocol=protocol()['protocol'],status='failed',session_id=launch['session_id'],
        model_id=args.model,kind=args.kind,precision=args.precision,package_manifest_sha256=args.expected_manifest,
        pid=os.getpid(),parent_pid=os.getppid(),boot_id=boot(),started_utc=now(),start_monotonic_ns=time.monotonic_ns(),
        compatibility_images=0,completed_by_split=dict(val=0,test=0),completed_by_phase=dict(warmup=0,probe=0),
        forward_calls_started=0,AP_evaluated=False,formal_accuracy_completed=False,video_FPS_measured=False,
        formal_cost_completed=False,power_measured=False,installation_performed=False)
    monitor=None;code=3
    try:
        platform_check()
        require(args.kind=='images','ATF1 authorizes common images only; no standalone short-video batch')
        require(launch['parent_pid']==os.getppid() and launch['boot_id']==boot()
                and launch['model']==args.model and launch['kind']==args.kind and launch['precision']==args.precision
                and launch['package_manifest_sha256']==args.expected_manifest,'Launch identity differs')
        rows=dependencies(args.expected_manifest)
        job=job_config(args.model)
        report_value.update(weight_sha256=job['weight_sha256'],attention_manifest_sha256=protocol()['attention_manifest_sha256'],
                            dataset_manifest_sha256=protocol()['dataset_manifest_sha256'],GT_sha256=protocol()['GT_sha256'],
                            video_sha256=protocol()['video_sha256'],runtime_flags=protocol()['runtime_flags'],Firefox_pids_before=firefox_present())
        import runtime_adapter as adapter
        adapter.c1.local_runtime(out)
        from resource_guard import memory_sample
        import resource_guard
        import startup_monitor
        from readiness import capacity,settle
        monitor=startup_monitor.build_monitor(resource_guard)(memory_sample,out/'process_memory_samples.jsonl')
        monitor.tick('before_framework_import',GIB);monitor.start()
        import torch
        import cv2
        import numpy as np
        import torchvision
        require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Expected one CUDA device')
        require((str(torch.__version__),torch.version.cuda,cv2.__version__,np.__version__,torchvision.__version__)
                ==('2.1.0a0+41361538.nv23.06','11.4','4.13.0','1.24.4','0.16.2'),'Qualified runtime changed')
        torch.set_num_threads(6);cv2.setNumThreads(1)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
        torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
        report_value['environment']=dict(torch=str(torch.__version__),cuda=torch.version.cuda,opencv=cv2.__version__,numpy=np.__version__,torchvision=torchvision.__version__,device=torch.cuda.get_device_name(0),torch_threads=torch.get_num_threads(),cv2_threads=cv2.getNumThreads())
        capacity(monitor,'before_checkpoint_load',3*GIB)
        model,dtype,nms,scale=adapter.load(job,args.precision,report_value,torch)
        capacity(monitor,'loaded_before_native_decode',3*GIB+24883200)
        def call(frame,confidence):
            monitor.check()
            result=adapter.infer(frame,confidence,model,dtype,nms,scale,report_value,torch,cv2,np)
            monitor.check()
            return result
        if args.kind=='images':
            accuracy(rows,out,monitor,call,report_value,cv2)
            require(report_value['compatibility_images']==0 and report_value['completed_by_split']==dict(val=149,test=149),'Incomplete image outputs')
            expected=298
            report_value['accuracy_predictions_exported']=True
        else:
            require(args.precision=='FP16','Short video is FP16 only')
            short_video(out,monitor,call,report_value,cv2)
            require(report_value['completed_by_phase']==dict(warmup=30,probe=100),'Incomplete short video')
            expected=130
        require(report_value['forward_calls_started']==report_value['_forward_gate']['intercepted_calls']==report_value['structure_hook_calls']==expected
                and report_value['_forward_gate']['open'] is False,'Unexpected/missing actual forwards')
        report_value['source_origins_after_inference']=adapter.origins()
        require(all(report_value['source_origins_after_inference'].get(k)==v for k,v in report_value['source_origins_before_load'].items()),'Loaded source origins changed during inference')
        require(sha(weight_path(job))==job['weight_sha256'],'Weight changed')
        report_value['Firefox_pids_after']=firefox_present()
        report_value['status']='passed';code=0
    except BaseException as exc:
        report_value.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        code=130 if isinstance(exc,KeyboardInterrupt) else 3
    finally:
        if monitor is not None:
            try:
                monitor.tick('post_diagnostic',GIB);monitor.stop();monitor.check()
            except BaseException as exc:
                report_value.update(status='failed',guard_final_error=str(exc));code=3
                monitor.stop()
            report_value['resource_guard']=dict(samples=len(monitor.samples),failure=str(monitor.failure) if monitor.failure else None,
                legacy_formal_activation_is_diagnostic_protection=True,activation=monitor.formal_activation,
                capacity_wait_used_s=getattr(monitor,'capacity_wait_used_s',0),capacity_admissions=getattr(monitor,'capacity_admissions',[]))
        report_value.update(end_monotonic_ns=time.monotonic_ns(),ended_utc=now(),intended_self_exit_code=code)
        report_value['artifacts']=artifacts(out,exclude=['worker_console.txt','process_exit.json','exit_status.json'])
        signed(out/'worker_report.json',report_value)
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-manifest',required=True)
    parser.add_argument('--model',choices=('identity','ca','cbam','eca'),required=True)
    parser.add_argument('--kind',choices=('images','short'),required=True)
    parser.add_argument('--precision',choices=('FP32','FP16'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    sys.exit(execute(parser.parse_args()))
