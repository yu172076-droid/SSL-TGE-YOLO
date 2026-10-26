"""Same-boot common predictions after rep01 root and cost binding are complete."""
import argparse,signal,sys,time,traceback,uuid
from common import *
def validate(model,expected):
    rows=dependencies(expected);job=job_config(model);out=PROJECT/'returns'/job['output_name'];r=get_report(out,'ap_report')
    require(r['status']=='ATF1_COMMON_PREDICTIONS_COMPLETED_REVIEW_REQUIRED' and r['model']==model and r['weight_sha256']==job['weight_sha256'] and r['manifest_sha256']==expected and r['exitcode']==0,'Common prediction parent incomplete')
    slot=next(s['slot'] for s in protocol()['schedule'] if s['model']==model and s['replicate']==1)
    joined=get_report(paths(slot)['joined'],'joined_report')
    require(r['cost_joined_sha256']==sha(paths(slot)['joined']/'joined_report.json') and r['boot_id']==joined['boot_id'],'Common predictions need same first-cost boot/weight')
    require(r['AP_evaluated'] is False and r['forward_calls']==596 and len(r['jobs'])==2,'Common prediction count/claim differs')
    from prediction_validation import validate_worker
    for item,precision in zip(r['jobs'],('FP32','FP16')):
        require(item['precision']==precision and item['exitcode']==0,'Common subprocess failed')
        child=out/precision;process=read(child/'process_exit.json');mapped=read(child/'exit_status.json')
        require(process['exitcode']==mapped['actual_exit_code']==0 and process['pid']==mapped['pid'] and not process['timed_out'] and not process['interrupted'],'Actual prediction process failed')
        require(sha(child/'worker_report.json')==item['report_sha256'],'Common worker report changed')
        validate_worker(child,model,'images',precision,r['session_id'],expected,r['boot_id'],rows)
    return r
def run(model,expected):
    rows=dependencies(expected);platform_check();firefox_present()
    slot=next(s['slot'] for s in protocol()['schedule'] if s['model']==model and s['replicate']==1)
    import control
    cost=control.completed(slot,expected,require_ap=False)
    require(cost['boot_id']==boot(),'Common predictions stay in first formal cost boot; root must already be stopped')
    out=create_output(job_config(model)['output_name']);session=str(uuid.uuid4());r=dict(status='incomplete',model=model,weight_sha256=job_config(model)['weight_sha256'],manifest_sha256=expected,boot_id=boot(),session_id=session,exitcode=3,cost_joined_sha256=sha(paths(slot)['joined']/'joined_report.json'),jobs=[],forward_calls=0,AP_evaluated=False,formal_cost_window_already_closed=True);code=3
    try:
        snapshot(out)
        for split in ('val','test'):shutil.copyfile(ROOT/('gt_'+split+'.json'),out/('gt_'+split+'.json'))
        from supervisor import run_child
        from prediction_validation import validate_worker
        for precision in ('FP32','FP16'):
            child=out/precision;child.mkdir()
            command=[sys.executable,'-B',ROOT/'ap_worker.py','--expected-manifest',expected,'--model',model,'--kind','images','--precision',precision,'--output',child]
            write(child/'launch.json',dict(parent_pid=os.getpid(),boot_id=boot(),model=model,kind='images',precision=precision,session_id=session,package_manifest_sha256=expected,command=list(map(str,command))))
            process=run_child(command,child,900)
            write(child/'exit_status.json',dict(pid=process['pid'],actual_exit_code=process['exitcode'],timed_out=process['timed_out'],interrupted=process['interrupted']))
            validate_worker(child,model,'images',precision,session,expected,boot(),rows)
            r['jobs'].append(dict(precision=precision,exitcode=process['exitcode'],report_sha256=sha(child/'worker_report.json')));r['forward_calls']+=298
            print('ATF1_COMMON_PREDICTIONS_PASSED '+model+' '+precision,flush=True)
        r['status']='ATF1_COMMON_PREDICTIONS_COMPLETED_REVIEW_REQUIRED';code=0
    except BaseException as exc:r.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:r.update(exitcode=code,artifacts=artifacts(out));signed(out/'ap_report.json',r)
    return code
if __name__=='__main__':
    def interrupt(sig,frame):raise KeyboardInterrupt('Signal '+str(sig))
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupt)
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--expected',required=True);a=ap.parse_args()
    try:sys.exit(run(a.model,a.expected))
    except (Exception,KeyboardInterrupt):traceback.print_exc();sys.exit(3)
