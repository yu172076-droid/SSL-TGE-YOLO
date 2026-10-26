"""PP1 detached formal queue launcher; same-boot, gated stages, actual child exit receipt."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import signal

from common import ROOT,boot,need,new_json,process_start_ticks,read,receipt,sha,verify


def status(project,config):
    out=project/'returns'/config['output_name']
    if not out.exists(): return dict(state='NOT_SUBMITTED')
    terminal=out/'terminal_exit.json'
    if terminal.exists():
        need(terminal.with_suffix('.sha256').read_text().split()[0]==sha(terminal),'Terminal receipt sidecar differs')
        end=read(terminal)
        need(sha(project/'returns'/(config['output_name']+'_console.txt'))==end['console_sha256'],'Closed console changed')
        if end['report_sha256'] is not None: need(sha(out/'report.json')==end['report_sha256'],'Closed report changed')
        return dict(state='COMPLETED_REVIEW_REQUIRED' if end['actual_pipeline_exit']==end['closure_exit']==0 else 'FAILED_STOPPED',
                    terminal=end,query_wall_ns=time.time_ns(),current_boot_id=boot())
    launch=out/'launch.json'
    if not launch.exists(): return dict(state='INTERRUPTED_OR_LAUNCH_FAILED',rerun_authorized=False)
    info=read(launch)
    actual_boot=boot(); actual_ticks=process_start_ticks(info['supervisor_PID'])
    alive=(isinstance(info['supervisor_start_ticks'],int) and actual_boot==info['boot_id']
           and actual_ticks==info['supervisor_start_ticks'])
    return dict(state='RUNNING' if alive else 'INTERRUPTED',PID=info['supervisor_PID'],rerun_authorized=False,
                session=info.get('session'),supervisor_start_ticks=info['supervisor_start_ticks'],
                current_boot_id=actual_boot,query_wall_ns=time.time_ns(),
                note='Missing terminal receipt is not success; retain all files')


def supervise(args,config,project):
    out=project/'returns'/config['output_name']
    need(read(out/'job_identity.json')['session']==args.session,'Wrong supervisor session')
    import fcntl
    code=3; actual=None; error=None
    try:
        locks=project/'locks'; locks.mkdir(exist_ok=True)
        with (locks/'pi_P1_active.lock').open('a+') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            model_bundle=project/'pi_cpu_prelease_20261021'
            python=project/'venvs/pi_cpu_cp313_20261021_01/bin/python'
            need(python.is_file(),'Accepted P1 venv Python missing; no install is authorized')
            need(sha(ROOT/'release.json')==config['release_sha256'],'PP1 release SHA differs')
            need(sha(ROOT/'core_manifest.json')==config['core_manifest_sha256'],'PP1 core SHA differs')
            command=[str(python),'-u','-B',str(ROOT/'run.py'),'--model-bundle',str(model_bundle),
                '--bundle-sha',config['model_bundle_manifest_sha256'],'--draft-sha',config['core_manifest_sha256'],
                '--release',str(ROOT/'release.json'),'--release-sha',config['release_sha256'],'--output',str(out),
                '--launcher-session',args.session,'--launcher-identity-sha',sha(out/'job_identity.json')]
            environment=os.environ.copy()
            for key in ('PYTHONPATH','PYTHONHOME'): environment.pop(key,None)
            environment.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
            process=subprocess.Popen(command,start_new_session=True,env=environment)
            try:
                actual=process.wait(timeout=config['whole_pipeline_timeout_s'])
            except subprocess.TimeoutExpired:
                # Dedicated group was created for this pipeline and inherited only by its own children.
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL); process.wait()
                raise
            need((out/'report.sha256').read_text().split()[0]==sha(out/'report.json'),'Pipeline report sidecar differs')
            need(read(out/'report.json')['exitcode']==actual,'Pipeline report/actual exit differs')
            code=actual
    except subprocess.TimeoutExpired: error='pipeline_timeout_own_dedicated_process_group_stopped_and_reaped'
    except Exception as exc: error=repr(exc)
    print('PI_PP1_PIPELINE_EXIT='+str(actual)+'; CLOSURE_EXIT='+str(code),flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    console=project/'returns'/(config['output_name']+'_console.txt')
    receipt(out,'terminal_exit',dict(schema='pi_PP1_actual_exit_closure_v1',session=args.session,
        actual_pipeline_exit=actual,closure_exit=code,error=error,boot_id=boot(),
        report_sha256=sha(out/'report.json') if (out/'report.json').exists() else None,
        console_sha256=sha(console),formal_execution_authorized=True,AP_review_required=False,
        power_measured=(code==0),power_scope='PMIC_output_rails_only',whole_board_power_measured=False))
    return code


def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['start','status','supervise'])
    p.add_argument('--project',required=True,type=Path); p.add_argument('--packet-sha',required=True)
    p.add_argument('--session',help=argparse.SUPPRESS); args=p.parse_args()
    config=verify(args.packet_sha); project=args.project.resolve()
    need(project.is_dir(),'Project missing')
    if args.action=='status': print(json.dumps(status(project,config),ensure_ascii=False)); return 0
    need(boot()==config['expected_boot_id'],'P0 boot changed; stop and return evidence')
    if args.action=='supervise': return supervise(args,config,project)
    returns=project/'returns'; need(returns.is_dir(),'Expected existing P0 returns directory')
    out=returns/config['output_name']; need(not out.exists(),'PP1 already submitted; use status and return evidence, no resubmission')
    console=returns/(config['output_name']+'_console.txt'); need(not console.exists(),'Existing console; do not overwrite')
    out.mkdir()
    session=str(uuid.uuid4())
    new_json(out/'job_identity.json',dict(session=session,packet_manifest_sha256=args.packet_sha,
        boot_id=boot(),created_wall_ns=time.time_ns()))
    command=['/usr/bin/nohup',sys.executable,'-u','-B',str(Path(__file__).resolve()),'supervise',
        '--project',str(project),'--packet-sha',args.packet_sha,'--session',session]
    with console.open('x',encoding='utf-8') as log:
        process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            close_fds=True,start_new_session=True)
    born=process_start_ticks(process.pid)
    new_json(out/'launch.json',dict(supervisor_PID=process.pid,supervisor_start_ticks=born,session=session,
        boot_id=boot(),command=command,launch_does_not_mean_completed=True))
    print(json.dumps(dict(state='SUBMITTED',PID=process.pid,output=str(out),console=str(console),
        note='Do not submit again after SSH reconnect; use status',formal_execution_authorized=True),ensure_ascii=False))
    return 0


if __name__=='__main__': raise SystemExit(main())
