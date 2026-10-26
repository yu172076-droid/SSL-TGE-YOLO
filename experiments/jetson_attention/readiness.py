"""Bounded capacity/stability admission for diagnostic prediction, not FPS."""
import time
from common import GIB, require


def capacity(monitor, phase, minimum, clock=time.monotonic, sleep=time.sleep):
    monitor.set_phase(phase, GIB)
    start = clock()
    remaining = max(0,180-getattr(monitor,'capacity_wait_used_s',0))
    last = -15
    while True:
        row = monitor.tick(phase,GIB)
        if row['proc_meminfo']['MemAvailable_bytes'] >= minimum:
            monitor.capacity_wait_used_s = getattr(monitor,'capacity_wait_used_s',0) + clock()-start
            admissions=getattr(monitor,'capacity_admissions',[])
            admissions.append(dict(phase=phase,target_bytes=minimum,available_bytes=row['proc_meminfo']['MemAvailable_bytes'],
                                   sample_monotonic_ns=row['monotonic_ns'],sample_index=len(monitor.samples)-1))
            monitor.capacity_admissions=admissions
            return row
        elapsed = clock()-start
        require(elapsed < remaining, 'Capacity wait exhausted: ' + phase)
        if elapsed-last >= 15:
            print('AT_CAPACITY_WAIT phase={} available={} target={} remaining_s={:.1f}'.format(phase,row['proc_meminfo']['MemAvailable_bytes'],minimum,max(0,remaining-elapsed)),flush=True)
            last=elapsed
        sleep(1)


def settle(monitor, phase, clock=time.monotonic, sleep=time.sleep):
    monitor.set_phase('diagnostic_settle',GIB)
    monitor.warmup_start_ns=time.monotonic_ns()
    started=clock()
    last=-15
    while True:
        with monitor.lock:
            monitor.tick('diagnostic_settle',GIB)
            proof=monitor.tail()
            if proof['ready']:
                # Legacy startup monitor calls this formal_activation. Here it
                # activates paging protection for diagnostics, never FPS/power.
                monitor.set_phase('measurement',GIB)
                monitor.set_phase(phase,GIB)
                return dict(scope='protected_diagnostic_predictions_not_formal_cost',
                            stability=proof,activation=monitor.formal_activation)
        elapsed=clock()-started
        require(elapsed<180,'Stable30 diagnostic settle exhausted')
        if elapsed-last>=15:
            print('AT_SETTLE phase={} elapsed_s={:.1f} remaining_s={:.1f} reason={}'.format(phase,elapsed,max(0,180-elapsed),proof.get('reason')),flush=True)
            last=elapsed
        sleep(1)
