"""Finite normal-user subprocesses; only the newly owned process group is stopped."""
import os,signal,subprocess,time
from common import write,require
def stop_owned(child):
    if child is not None and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
def run_child(command,out,timeout=1200):
    child=None;timed=False;interrupted=False;start=time.monotonic_ns()
    try:
        with (out/'worker_console.txt').open('x',encoding='utf-8') as stream:
            child=subprocess.Popen(list(map(str,command)),stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
            try:child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:timed=True;stop_owned(child)
    except BaseException:interrupted=True;stop_owned(child);raise
    finally:
        stop_owned(child)
        value=dict(pid=child.pid if child else None,exitcode=child.returncode if child else None,timed_out=timed,interrupted=interrupted,
                   started_monotonic_ns=start,ended_monotonic_ns=time.monotonic_ns(),watchdog_seconds=timeout)
        write(out/'process_exit.json',value)
    require(not timed and value['exitcode']==0,'Owned worker failed; stop batch and preserve output')
    return value
