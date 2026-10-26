"""Read-only, bounded Pi5 PMIC telemetry and per-rail integration. No rail sum."""
import bisect
import math
from pathlib import Path
import re
import json
import subprocess
import threading
import time

CHANNELS={
    '3V7_WL_SW':(0,8),'3V3_SYS':(1,9),'1V8_SYS':(2,10),'DDR_VDD2':(3,11),
    'DDR_VDDQ':(4,12),'1V1_SYS':(5,13),'0V8_SW':(6,14),'VDD_CORE':(7,15),
    '3V3_DAC':(17,20),'3V3_ADC':(18,21),'0V8_AON':(16,19),'HDMI':(22,23),
}
PRIMARY_RAILS=['VDD_CORE','DDR_VDD2','DDR_VDDQ']


def parse_adc(raw):
    fields={}
    for line in raw.splitlines():
        if not line.strip():continue
        m=re.fullmatch(r'\s*(\w+)\s+(current|volt)\((\d+)\)=([+-]?(?:\d+(?:\.\d*)?|\.\d+))(A|V)\s*',line)
        if not m:raise ValueError('Unrecognized PMIC output line')
        label,kind,channel,number,unit=m.groups();value=float(number)
        if label in fields or not math.isfinite(value) or value<0:raise ValueError('Duplicate or invalid PMIC field')
        if (kind,unit) not in (('current','A'),('volt','V')):raise ValueError('PMIC unit mismatch')
        fields[label]=dict(channel=int(channel),value=value,unit=unit)
    expected={name+suffix for name in CHANNELS for suffix in ('_A','_V')}|{'EXT5V_V','BATT_V'}
    if set(fields)!=expected:raise ValueError('PMIC channel set differs from qualified snapshot')
    rails={}
    for name,(ac,vc) in CHANNELS.items():
        amps,volts=fields[name+'_A'],fields[name+'_V']
        if amps['channel']!=ac or volts['channel']!=vc or amps['unit']!='A' or volts['unit']!='V' or volts['value']<=0:
            raise ValueError('PMIC paired channels differ')
        rails[name]=dict(current_A=amps['value'],voltage_V=volts['value'],power_W=amps['value']*volts['value'])
    if fields['EXT5V_V']['channel']!=24 or fields['EXT5V_V']['unit']!='V' or fields['EXT5V_V']['value']<=0 or fields['BATT_V']['channel']!=25:
        raise ValueError('PMIC voltage-only channel differs')
    return dict(rails=rails,voltage_only_V={name:fields[name]['value'] for name in ('EXT5V_V','BATT_V')})


def integrate(samples,start_ns,end_ns,max_gap_s=3.5):
    if not (type(start_ns) is int and type(end_ns) is int and end_ns>start_ns):raise ValueError('Invalid power window')
    if len(samples)<2:raise ValueError('Missing PMIC samples')
    times=[r['mid_monotonic_ns'] for r in samples]
    if any(b<=a for a,b in zip(times,times[1:])):raise ValueError('PMIC timestamps not increasing')
    left=bisect.bisect_right(times,start_ns)-1;right=bisect.bisect_left(times,end_ns)
    if left<0 or right>=len(times):raise ValueError('PMIC window not bracketed')
    selected=samples[left:right+1]
    if any((b['mid_monotonic_ns']-a['mid_monotonic_ns'])/1e9>max_gap_s for a,b in zip(selected,selected[1:])):
        raise ValueError('PMIC window sample gap')
    result={}
    for rail in CHANNELS:
        energy=0.
        for a,b in zip(selected,selected[1:]):
            t0,t1=a['mid_monotonic_ns'],b['mid_monotonic_ns']
            lo,hi=max(start_ns,t0),min(end_ns,t1)
            if hi<=lo:continue
            p0,p1=a['rails'][rail]['power_W'],b['rails'][rail]['power_W']
            if not all(math.isfinite(v) and v>=0 for v in (p0,p1)):raise ValueError('Invalid rail power')
            vlo=p0+(p1-p0)*(lo-t0)/(t1-t0);vhi=p0+(p1-p0)*(hi-t0)/(t1-t0)
            energy+=(vlo+vhi)/2*(hi-lo)/1e9
        result[rail]=dict(mean_W=energy/((end_ns-start_ns)/1e9),energy_J=energy)
    return dict(start_monotonic_ns=start_ns,end_monotonic_ns=end_ns,duration_s=(end_ns-start_ns)/1e9,
                bracketing_sample_midpoints_ns=[times[left],times[right]],samples_used=len(selected),
                maximum_gap_s=max((b['mid_monotonic_ns']-a['mid_monotonic_ns'])/1e9 for a,b in zip(selected,selected[1:])),
                method='piecewise_linear_power_trapezoid_clipped_to_exact_window',rails=result,
                channel_sum_computed=False,whole_board_power_measured=False)


class PMICLog:
    def __init__(self,directory,config,boot):
        self.config=config;self.boot=boot;self.rows=[];self.lock=threading.RLock();self.failed=False
        self.file=(Path(directory)/'pmic.jsonl').open('x',encoding='utf-8',buffering=1)

    def capture(self,phase,resource_boot):
        with self.lock:
            begin=time.monotonic_ns();wall=time.time_ns()
            record=dict(capture_start_monotonic_ns=begin,capture_start_wall_ns=wall,phase=phase,resource_boot_id=resource_boot)
            try:
                completed=subprocess.run([self.config['vcgencmd_path'],'pmic_read_adc'],capture_output=True,text=True,
                                         timeout=self.config['query_timeout_s'],check=False)
                end=time.monotonic_ns()
                record.update(capture_end_monotonic_ns=end,mid_monotonic_ns=(begin+end)//2,
                              query_duration_ns=end-begin,actual_exit=completed.returncode,stdout=completed.stdout,stderr=completed.stderr)
                if completed.returncode!=0:raise ValueError('PMIC query failed')
                if (end-begin)/1e9>self.config['max_query_wall_s']:raise ValueError('PMIC query wall time exceeded')
                if resource_boot!=self.boot:raise ValueError('PMIC associated resource boot changed')
                record.update(parse_adc(completed.stdout))
                record['valid']=True
            except Exception as error:
                self.failed=True;record.update(valid=False,error=repr(error))
                record.setdefault('capture_end_monotonic_ns',time.monotonic_ns())
                if isinstance(error,subprocess.TimeoutExpired):
                    record['timeout']=True;record['stdout']=str(error.stdout or '');record['stderr']=str(error.stderr or '')
                self.file.write(json.dumps(record,allow_nan=False)+'\n');self.file.flush()
                raise
            self.rows.append(record)
            self.file.write(json.dumps(record,allow_nan=False)+'\n');self.file.flush()
            return record

    def summary(self,baseline,start_ns,end_ns,frames):
        with self.lock:
            if self.failed:raise ValueError('PMIC failure latch set')
            before=integrate(self.rows,baseline['start_monotonic_ns'],baseline['end_monotonic_ns'],self.config['max_gap_s'])
            measured=integrate(self.rows,start_ns,end_ns,self.config['max_gap_s'])
            if before['duration_s']<60:raise ValueError('PMIC idle baseline shorter than 60s')
            for rail in PRIMARY_RAILS:
                if measured['rails'][rail]['mean_W']<=0:raise ValueError('Primary PMIC rail all zero')
            return dict(scope='PMIC_output_rails_only',primary_rails=PRIMARY_RAILS,frames=frames,
                        baseline=before,measurement=measured,
                        incremental_mean_W={rail:measured['rails'][rail]['mean_W']-before['rails'][rail]['mean_W'] for rail in CHANNELS},
                        independent_external_calibration_performed=False,channel_sum_computed=False,
                        whole_board_power_measured=False,rail_V_and_I_not_guaranteed_simultaneous=True,
                        sampling_overhead_part_of_this_new_co_measurement_protocol=True)

    def close(self):
        with self.lock:
            self.file.close()
            return dict(samples=len(self.rows),failure=self.failed,scope='PMIC_output_rails_only',whole_board_power_measured=False)
