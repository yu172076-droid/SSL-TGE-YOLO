"""Formal phase monitor and atomic readiness handoff; only read-only Pi probes."""
import json
from pathlib import Path
import threading
import time


def runtime_context(config):
    frequency=int(Path(config['frequency_path']).read_text(encoding='ascii').strip())
    governor=Path(config['governor_path']).read_text(encoding='ascii').strip()
    stat=Path('/proc/self/stat').read_text(encoding='utf-8').rsplit(')',1)[1].split()
    cpu_ticks=int(stat[11])+int(stat[12])
    if frequency<=0 or not governor or cpu_ticks<0:
        raise ValueError('Invalid live frequency/governor/process CPU counters')
    return dict(frequency_kHz=frequency,scaling_governor=governor,process_cpu_ticks=cpu_ticks)


class Guard:
    def __init__(self, output, release):
        from admission import validate_sample, formal_interval
        from admission_pf2 import readiness
        from resource_guard import linux_sample
        self.readiness_rule, self.validate, self.formal_rule = readiness, validate_sample, formal_interval
        self.sampler = linux_sample
        self.q = release
        self.required = release['resource_guard_required']
        self.lock = threading.RLock()
        self.event = threading.Event()
        self.rows, self.failure = [], None
        self.phase = 'idle'
        self.initial_oom = release.get('expected_oom_kill')
        self.boot = release.get('expected_boot_id')
        self.start_mask = None
        self.previous_effective = None
        self.paging_streak = 0
        self.start_ns = self.end_ns = None
        self.thread = None
        from pmic_reader import PMICLog
        self.pmic=PMICLog(output,release['pmic_config'],self.boot)
        self.file = (Path(output) / 'resources.jsonl').open('x', encoding='utf-8', buffering=1)
        self.waits = (Path(output) / 'readiness.jsonl').open('x', encoding='utf-8', buffering=1)
        if self.required:
            self.sync()
            self.thread = threading.Thread(target=self._loop, daemon=True, name='readonly_formal_monitor')
            self.thread.start()

    def _take_locked(self):
        try:
            row = self.sampler(self.q['sensor_config'])
            row.update(runtime_context(self.q['sensor_config']))
            self.pmic.capture(self.phase,row['boot_id'])
            error = self.validate(row, self.boot)
            if not error and row['scaling_governor']!=self.q['expected_governor']:
                error='governor_changed_from_release'
            previous = self.rows[-1] if self.rows else None
            if self.initial_oom is None:
                self.initial_oom = row['vmstat']['oom_kill']
            if not error and row['vmstat']['oom_kill'] != self.initial_oom:
                error = 'global_OOM_changed'
            if not error and previous:
                gap = (row['monotonic_ns'] - previous['monotonic_ns']) / 1e9
                if not 0 < gap <= self.q['runtime_policy']['max_gap_s']:
                    error = 'resource_sample_gap'
                elif any(row['vmstat'][k] < previous['vmstat'][k] for k in ('pswpin', 'pswpout', 'oom_kill')):
                    error = 'counter_decreased'
                dt = row['cpu_total_ticks'] - previous['cpu_total_ticks']
                di = row['cpu_idle_ticks'] - previous['cpu_idle_ticks']
                row['cpu_busy_percent'] = 100 * (dt - di) / dt if dt > 0 else None
            else:
                row['cpu_busy_percent'] = None
            if not error and row['MemAvailable_bytes'] < self.q['runtime_policy']['min_available_bytes']:
                error = 'runtime_capacity'
            if not error and row['temperature_C'] >= self.q['runtime_policy']['max_temperature_C']:
                error = 'runtime_temperature'
            if not error and row['throttled_mask'] & 15:
                error = 'current_supply_or_throttle_event'
            if not error and self.phase == 'measurement':
                if (row['throttled_mask'] & 0xF0000) & ~(self.start_mask & 0xF0000):
                    error = 'new_historical_supply_or_throttle_event'
                # Endpoints closer than 0.5 s remain raw evidence, not extra 1-Hz intervals.
                if self.previous_effective and (row['monotonic_ns'] - self.previous_effective['monotonic_ns']) / 1e9 >= .5:
                    result = self.formal_rule(self.previous_effective, row, self.q['runtime_policy'], self.boot,
                                              self.initial_oom, self.start_mask, self.paging_streak)
                    self.paging_streak = result['paging_streak']
                    self.previous_effective = row
                    if result['state'] == 'STOP':
                        error = result['reason']
            row['phase'] = self.phase
            self.rows.append(row)
            self.file.write(json.dumps(row, allow_nan=False) + '\n')
            self.file.flush()
            if error and self.failure is None:
                self.failure = {'reason': error, 'monotonic_ns': row['monotonic_ns']}
            return row
        except Exception as error:
            self.failure = self.failure or {'reason': 'resource_read_failed', 'error': repr(error)}
            return None

    def _loop(self):
        while not self.event.wait(1.0):
            with self.lock:
                self._take_locked()

    def check(self):
        with self.lock:
            if self.failure:
                raise RuntimeError('Resource guard stopped: ' + json.dumps(self.failure))
            if self.required and (not self.rows or (time.monotonic_ns() - self.rows[-1]['monotonic_ns']) / 1e9 > self.q['runtime_policy']['max_gap_s']):
                raise RuntimeError('Latest resource sample stale')

    def sync(self):
        with self.lock:
            row = self._take_locked() if self.required else None
            self.check()
            return row

    def set_phase(self, phase):
        with self.lock:
            self.check()
            self.phase = phase

    def wait_ready(self, kind, transition=False):
        policy = self.q['idle_policy' if kind == 'idle' else 'loaded_policy']
        started = time.monotonic_ns()
        while True:
            with self.lock:
                self.check()
                now = time.monotonic_ns()
                if self.required:
                    proof = self.readiness_rule(self.rows, policy, self.boot, self.initial_oom, started, now, kind == 'idle')
                else:
                    # Explicit local Windows validation. No synthetic Pi resource readings.
                    proof = dict(state='READY', reason='local_validation_resource_guard_unavailable', remaining_s=0,
                                 formal_execution_authorized=False)
                # The monitor is the only periodic producer. Do not add a second
                # 1-Hz sampler that creates tiny/zero-tick CPU intervals in idle.
                if proof['state']=='READY' and transition and self.required:
                    row=self._take_locked()
                    self.check()
                    now=time.monotonic_ns()
                    proof=self.readiness_rule(self.rows,policy,self.boot,self.initial_oom,started,now,False)
                if proof['state'] == 'STOP':
                    self.waits.write(json.dumps(dict(kind=kind,monotonic_ns=now,quiet_started_ns=started,**proof),allow_nan=False)+'\n')
                    self.waits.flush()
                    raise RuntimeError('Readiness stopped: ' + proof['reason'])
                if proof['state'] == 'READY':
                    if transition:
                        self.start_mask = self.rows[-1]['throttled_mask'] if self.required else None
                        self.previous_effective = self.rows[-1] if self.required else None
                        self.paging_streak = 0
                        self.phase = 'measurement'
                        self.start_ns = time.monotonic_ns()
                        proof.update(transition_committed=True,measurement_start_ns=self.start_ns)
                    self.waits.write(json.dumps(dict(kind=kind,monotonic_ns=now,quiet_started_ns=started,**proof),allow_nan=False)+'\n')
                    self.waits.flush()
                    if transition:
                        return self.start_ns
                    return proof
                self.waits.write(json.dumps(dict(kind=kind,monotonic_ns=now,quiet_started_ns=started,**proof),allow_nan=False)+'\n')
                self.waits.flush()
            print('RESOURCE_WAIT ' + json.dumps(proof), flush=True)
            self.event.wait(1.0)

    def finish_measurement(self):
        with self.lock:
            if self.required:
                self._take_locked()
            self.check()
            self.end_ns = time.monotonic_ns()
            self.phase = 'postmeasurement'
            self.pmic.capture('postmeasurement_endpoint',self.boot)
            return self.end_ns

    def close(self):
        self.event.set()
        if self.thread:
            self.thread.join(timeout=3)
            if self.thread.is_alive():
                self.failure = self.failure or {'reason': 'monitor_thread_did_not_stop'}
        self.file.close()
        self.waits.close()
        pmic_receipt=self.pmic.close()
        measured = [r for r in self.rows if r['phase'] == 'measurement']
        return dict(required=self.required, failure=self.failure, PMIC=pmic_receipt, samples=len(self.rows), measurement_samples=len(measured),
            runtime_min_MemAvailable_bytes=min((r['MemAvailable_bytes'] for r in measured), default=None),
            sampled_measurement_peak_RSS_bytes=max((r['process_RSS_bytes'] for r in measured), default=None),
            lifetime_VmHWM_bytes=max((r['process_VmHWM_bytes'] for r in self.rows), default=None),
            measurement_frequency_kHz_min=min((r['frequency_kHz'] for r in measured),default=None),
            measurement_frequency_kHz_max=max((r['frequency_kHz'] for r in measured),default=None),
            observed_governors=sorted(set(r['scaling_governor'] for r in self.rows)),
            historical_events_may_be_unobservable_when_already_sticky=True,
            global_paging_does_not_identify_culprit_process=True)
