"""One fresh-boot formal native FP16 complete-file cost session; AP remains pending."""
import argparse,signal,sys,time,traceback
from common import *
import video_window as windows
from native_frame import one_frame,validate_values
def environment():
    import torch,cv2,numpy as np,torchvision
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Single CUDA device required')
    require((str(torch.__version__),torch.version.cuda,cv2.__version__,np.__version__,torchvision.__version__)==('2.1.0a0+41361538.nv23.06','11.4','4.13.0','1.24.4','0.16.2'),'Qualified environment changed')
    torch.set_num_threads(6);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    return torch,cv2,np,dict(torch=str(torch.__version__),cuda=torch.version.cuda,opencv=cv2.__version__,numpy=np.__version__,torchvision=torchvision.__version__,device=torch.cuda.get_device_name(0),torch_threads=torch.get_num_threads(),cv2_threads=cv2.getNumThreads())
def worker(args):
    p=paths(args.slot);row=slot_config(args.slot);job=job_config(row['model']);out=p['infer']
    state=dict(schema='ATF1_native_cost_inference_v1',status='incomplete',actual_self_exit_code=3,manifest_sha256=args.expected,
        slot=args.slot,model_id=row['model'],job=row['model'],replicate=row['replicate'],weight_sha256=job['weight_sha256'],
        boot_id=boot(),pid=os.getpid(),euid=os.geteuid(),windows={},forward_calls_started=0,shape_hook_once=True,
        capacity_policy=protocol()['capacity_policy'],video_sha256=protocol()['video_sha256'],AP_evaluated=False,AP_gate_pending=True,
        TensorRT_executed=False,tracking_executed=False,benchmark_completed=False,
        config=dict(batch=1,input_shape=[1,3,640,640],precision='native_PyTorch_FP16',conf=.25,iou=.65,max_det=300,native_dimensions=[3840,2160],expected_frames_per_pass=1475,backend='FFMPEG',firefox_must_remain_open=True))
    monitor=None;code=3
    try:
        dependencies(args.expected);platform_check();firefox_present()
        import control
        control.validate_admission(args.slot,args.expected)
        root=module('ATF1_root_session',ROOT/'root_session_helper.py');root.ROOT=ROOT
        session=root.RootSession(p['root'],args.expected,boot())
        state.update(root_ready=session.ready,root_ready_sha256=session.ready_sha,root_session_id=session.ready['session_id'])
        import runtime_adapter as adapter,resource_guard,startup_monitor
        adapter.c1.local_runtime(out)
        def sample(phase):
            r=resource_guard.memory_sample(phase);r['root_live_check']=session.check();return r
        monitor=startup_monitor.build_monitor(resource_guard)(sample,out/'process_memory_samples.jsonl')
        monitor.tick('before_model_load',GIB);monitor.start()
        torch,cv2,np,env=environment();state['environment']=env
        capacity_wait(monitor,'model_capacity_wait',3*GIB);monitor.tick('model_load_ready',3*GIB)
        model,dtype,nms,scale=adapter.load(job,'FP16',state,torch)
        capacity_wait(monitor,'loaded_capacity_wait',protocol()['predecoder_min_bytes'])
        monitor.tick('model_loaded_before_decoder_open',3*GIB)
        snapshot(out);events=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
        monitor.set_phase('baseline',GIB);start=time.monotonic_ns()
        baseline=dict(start_monotonic_ns=start,model_loaded=True,capture_closed=True,native_frames_retained=0,inference_executed=False,frames=0,complete=False);state['windows']['baseline']=baseline
        print('ATF1_LOADED_IDLE_60S',flush=True)
        while (time.monotonic_ns()-start)/1e9<60:monitor.check();time.sleep(.1)
        baseline.update(end_monotonic_ns=time.monotonic_ns(),complete=True);baseline['duration_s']=(baseline['end_monotonic_ns']-start)/1e9
        require(baseline['duration_s']<=90,'Loaded idle exceeded90s')
        capacity_wait(monitor,'decoder_capacity_wait',protocol()['predecoder_min_bytes'])
        monitor.tick('decoder_capacity_after_baseline',protocol()['predecoder_min_bytes']);monitor.set_phase('decoder_startup_wait',GIB)
        def infer(frame,source):
            monitor.check();return one_frame(frame,source,model,dtype,nms,scale,state,torch,cv2,np,events)
        with torch.no_grad():
            windows.video_window('warmup',cv2,PROJECT/protocol()['video_relative'],infer,sys.modules[__name__],monitor,out,state)
            require(state['structure_hook_calls']==1 and state['AT_structure_before_fusion']['P4_runtime_shapes'],'Warmup shape discovery failed')
            torch.cuda.synchronize();state.update(startup_policy=startup_monitor.POLICY,formal_guard_activation=monitor.formal_activation)
            torch.cuda.reset_peak_memory_stats(0)
            measured=windows.video_window('measurement',cv2,PROJECT/protocol()['video_relative'],infer,sys.modules[__name__],monitor,out,state)
        torch.cuda.synchronize()
        state['CUDA_allocator_postwarm_peak_MiB']=dict(allocated=torch.cuda.max_memory_allocated(0)/1024**2,reserved=torch.cuda.max_memory_reserved(0)/1024**2,scope='Torch allocator during formal window, not separate physical GPU RAM')
        monitor.tick('post_measurement',GIB);start=time.monotonic()
        while time.monotonic()-start<2:monitor.check();time.sleep(.1)
        monitor.tick('right_bracket_after_measurement',GIB)
        require(state['forward_calls_started']==state['_forward_gate']['intercepted_calls'] and state['structure_hook_calls']==1,'Extra/missing forwards or formal shape-hook overhead')
        state['source_origins_after_inference']=adapter.origins()
        require(sha(weight_path(job))==job['weight_sha256'] and sha(PROJECT/protocol()['video_relative'])==protocol()['video_sha256'],'Weight/video changed')
        firefox_present();state.update(status='ATF1_FORMAL_WINDOW_COMPLETED_BINDING_REQUIRED',benchmark_completed=True,video_unchanged=True,video_window_throughput_FPS=measured['frames']/measured['duration_s']);code=0
    except BaseException as exc:state.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        if monitor is not None:
            monitor.stop()
            if monitor.failure is not None:state.update(resource_error=str(monitor.failure),benchmark_completed=False);code=3
            state['memory_observer']=dict(samples=len(monitor.samples),failure=None if monitor.failure is None else str(monitor.failure),thread_started=monitor.thread is not None,thread_stopped=monitor.thread is None or not monitor.thread.is_alive())
        state['actual_self_exit_code']=code;write(out/'measurement_window.json',state['windows']);signed(out/'infer_report.json',state)
    print(state['status'],code,flush=True);return code
def parent(args):
    dependencies(args.expected);platform_check()
    import control
    ad=control.validate_admission(args.slot,args.expected);p=paths(args.slot);out=create_output(p['infer'].name);row=slot_config(args.slot)
    shutil.copyfile(p['admission'],out/'admission.json')
    result=dict(schema='ATF1_native_cost_launcher_v1',status='incomplete',exitcode=3,manifest_sha256=args.expected,boot_id=boot(),normal_euid=os.geteuid(),slot=args.slot,job=row['model'],replicate=row['replicate'],weight_sha256=job_config(row['model'])['weight_sha256'],started_monotonic_ns=time.monotonic_ns());code=3
    try:
        from supervisor import run_child
        result['child']=run_child([sys.executable,'-B',ROOT/'native_cost.py','--worker','--slot',str(args.slot),'--expected',args.expected],out)
        ir=get_report(out,'infer_report')
        require(ir['actual_self_exit_code']==0 and ir['benchmark_completed'] and ir['pid']==result['child']['pid'] and ir['boot_id']==result['boot_id'],'Cost worker identity/exit incomplete')
        result['nms_time_limit_warning']='NMS time limit' in (out/'worker_console.txt').read_text(encoding='utf-8',errors='replace');require(not result['nms_time_limit_warning'],'NMS time-limit warning')
        dependencies(args.expected);result['status']='ATF1_FORMAL_WINDOW_COMPLETED_BINDING_REQUIRED';code=0
    except BaseException as exc:result.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        result.update(exitcode=code,ended_monotonic_ns=time.monotonic_ns(),artifacts=artifacts(out));signed(out/'launcher_report.json',result)
    return code
if __name__=='__main__':
    def interrupt(sig,frame):raise KeyboardInterrupt('Signal '+str(sig))
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupt)
    ap=argparse.ArgumentParser();ap.add_argument('--worker',action='store_true');ap.add_argument('--slot',type=int,required=True);ap.add_argument('--expected',required=True);args=ap.parse_args()
    try:sys.exit(worker(args) if args.worker else parent(args))
    except (Exception,KeyboardInterrupt):traceback.print_exc();sys.exit(3)
