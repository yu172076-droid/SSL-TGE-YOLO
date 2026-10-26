"""Fixed12 queue, bounded600s quiet admission and read-only completion checks."""
import argparse,json,sys,time,traceback
from common import *
def completed(slot,expected,require_ap=True):
    p=paths(slot);row=slot_config(slot);j=get_report(p['joined'],'joined_report');lr=get_report(p['infer'],'launcher_report',j['launcher_report_sha256']);ir=get_report(p['infer'],'infer_report',j['infer_report_sha256']);rr=get_report(p['root'],'root_sampler_report',j['root_report_sha256'])
    require(j['status']=='ATF1_COST_COMPLETED_PRECISION_REVIEW_PENDING' and j['exitcode']==lr['exitcode']==ir['actual_self_exit_code']==rr['exitcode']==0,'Previous formal cost incomplete')
    require(j['manifest_sha256']==lr['manifest_sha256']==ir['manifest_sha256']==rr['manifest_sha256']==expected and j['slot']==lr['slot']==ir['slot']==slot,'Previous slot/package differs')
    require(j['job']==ir['job']==lr['job']==row['model'] and j['replicate']==ir['replicate']==lr['replicate']==row['replicate'],'Previous model/rep differs')
    require(j['boot_id']==lr['boot_id']==ir['boot_id']==rr['boot_id'] and j['weight_sha256']==ir['weight_sha256']==lr['weight_sha256']==job_config(row['model'])['weight_sha256'],'Previous boot/weight differs')
    require(j['admission_sha256']==sha(p['admission'])==sha(p['infer']/'admission.json'),'Previous admission changed')
    prefix=exit_lines(slot,('VERIFY','PREFLIGHT','INFER','ROOT','BIND'))
    lines=p['terminal'].read_text(encoding='ascii').splitlines()
    if row['replicate']==1 and require_ap:
        require(lines==prefix+exit_lines(slot,('AP',)),'Actual cost+AP shell chain incomplete')
        import ap_export
        ap_export.validate(row['model'],expected)
    else:require(lines==prefix,'Actual five cost shell exits incomplete')
    return dict(slot=slot,model=row['model'],replicate=row['replicate'],boot_id=j['boot_id'],joined_report_sha256=sha(p['joined']/'joined_report.json'))
def previous(slot,expected):
    rows=[completed(i,expected) for i in range(1,slot)]
    require(len({r['boot_id'] for r in rows})==len(rows),'Prior formal boots reused')
    return rows
def validate_quiet(ad,slot,expected):
    import startup_monitor
    p=paths(slot);trace=[json.loads(s) for s in p['trace'].read_text(encoding='utf-8').splitlines()]
    require(ad['status']=='ATF1_READY' and ad['slot']==slot and ad['manifest_sha256']==expected and ad['trace_sha256']==sha(p['trace']),'Quiet identity/trace differs')
    proof=startup_monitor.stable_tail(trace,ad['started_monotonic_ns'])
    elapsed=(ad['ended_monotonic_ns']-ad['started_monotonic_ns'])/1e9
    require(proof==ad['stable_tail'] and proof['ready'] and 600<=elapsed<=900 and trace[-1]['monotonic_ns']==ad['ended_monotonic_ns'],'Quiet600/stable30 incomplete')
    require(trace[0]['monotonic_ns']>=ad['started_monotonic_ns'] and trace[-1]['proc_meminfo']['MemAvailable_bytes']>=3*GIB+24883200,'Quiet capacity/time differs')
    initial=startup_monitor.oom(trace[0]);prev=None
    require(initial is not None,'Quiet OOM counter absent')
    for r in trace:
        vm={s.split()[0]:int(s.split()[1]) for s in r['raw']['vmstat'].splitlines() if len(s.split())==2}
        mem={s.partition(':')[0]:s.partition(':')[2].strip() for s in r['raw']['meminfo'].splitlines()}
        require(r['proc_meminfo']['MemAvailable_bytes']>=GIB and int(mem['MemAvailable'].split()[0])*1024==r['proc_meminfo']['MemAvailable_bytes'] and startup_monitor.oom(r)==initial,'Quiet raw/capacity/OOM differs')
        require(all(vm[k]==r['vm_counters'][k] for k in ('pswpin','pswpout','pgmajfault')),'Quiet raw VM differs')
        if prev is not None:require(r['monotonic_ns']>prev['monotonic_ns'] and all(r['vm_counters'][k]>=prev['vm_counters'][k] for k in ('pswpin','pswpout','pgmajfault')),'Quiet counters/time decreased')
        prev=r
    require(ad['firefox_before'] and ad['firefox_after'] and ad['single_blank_tab_operator_declared'],'Firefox declaration absent')
    return proof
def validate_admission(slot,expected):
    verify(expected);p=paths(slot);row=slot_config(slot);ad=report(p['admission']);prior=previous(slot,expected)
    require(ad['boot_id']==boot() and ad['previous']==prior and ad['model']==row['model'] and ad['replicate']==row['replicate'],'Admission boot/prior/model differs')
    require(boot() not in [r['boot_id'] for r in prior],'Normal new boot required for every formal item')
    validate_quiet(ad,slot,expected)
    require(0<=time.monotonic_ns()-ad['ended_monotonic_ns']<=900*10**9,'Admission expired')
    firefox_present();return ad
def ready(slot,expected):
    dependencies(expected);platform_check();prior=previous(slot,expected);p=paths(slot);row=slot_config(slot)
    require(boot() not in [r['boot_id'] for r in prior],'Reboot normally before each formal item')
    require(float(Path('/proc/uptime').read_text().split()[0])<=1800,'Start preflight within30min of normal boot')
    require(not any(q.exists() for k,q in p.items() if k not in ('terminal','console')),'Existing item artifacts: preserve and stop')
    require(p['terminal'].read_text().splitlines()==exit_lines(slot,('VERIFY',)),'Preflight needs one actual successful verify')
    import resource_guard,startup_monitor
    monitor=startup_monitor.build_monitor(resource_guard)(resource_guard.memory_sample,p['trace'])
    start=time.monotonic_ns();monitor.warmup_start_ns=start;browser=firefox_present();last=-30
    while True:
        r=monitor.tick('quiet_wait',GIB);elapsed=(r['monotonic_ns']-start)/1e9;proof=monitor.tail()
        if elapsed>=600 and proof['ready'] and r['proc_meminfo']['MemAvailable_bytes']>=3*GIB+24883200:break
        require(elapsed<900,'Quiet readiness exhausted; no GPU started')
        if elapsed-last>=30:
            print('ATF1_WAIT elapsed_s={:.1f} remaining_min_s={:.1f} available_GiB={:.3f} stable30={}'.format(elapsed,max(0,600-elapsed),r['proc_meminfo']['MemAvailable_bytes']/GIB,proof['ready']),flush=True);last=elapsed
        time.sleep(1)
    ad=dict(status='ATF1_READY',manifest_sha256=expected,slot=slot,model=row['model'],replicate=row['replicate'],boot_id=boot(),weight_sha256=job_config(row['model'])['weight_sha256'],previous=prior,started_monotonic_ns=start,ended_monotonic_ns=r['monotonic_ns'],stable_tail=proof,trace_sha256=sha(p['trace']),firefox_before=browser,firefox_after=firefox_present(),single_blank_tab_operator_declared=True)
    signed(p['admission'],ad);print('ATF1_READY START_ROOT_THEN_INFER slot={:02d}'.format(slot),flush=True)
    return 0
def complete(expected):
    dependencies(expected);rows=[completed(i,expected) for i in range(1,13)]
    require(len({r['boot_id'] for r in rows})==12,'Twelve formal boots must differ')
    destination=PROJECT/'returns/ATF1_twelve_completed_20261020_01.json'
    signed(destination,dict(status='ATF1_TWELVE_COSTS_AND_FOUR_COMMON_PREDICTIONS_COMPLETED_REVIEW_REQUIRED',manifest_sha256=expected,rows=rows,cost_sessions=12,common_prediction_forwards=2384,AP_evaluated=False,AP_gate_pending=True))
    print('ATF1_TWELVE_COSTS_AND_FOUR_COMMON_PREDICTIONS_COMPLETED_REVIEW_REQUIRED',flush=True);return 0
def prefix(slot,expected,phase):
    verify(expected);p=paths(slot)
    required={'root':('VERIFY','PREFLIGHT'),'infer':('VERIFY','PREFLIGHT'),'bind':('VERIFY','PREFLIGHT','INFER','ROOT')}[phase]
    require(p['terminal'].read_text(encoding='ascii').splitlines()==exit_lines(slot,required),'Wrong/failed/repeated actual terminal prefix')
    require(not p[{'root':'root','infer':'infer','bind':'joined'}[phase]].exists(),'Output exists: preserve and stop')
    if phase in ('root','infer'):validate_admission(slot,expected)
    print('ATF1_PREFIX_VALID '+phase,flush=True);return 0
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('verify','ready','check','complete','prefix','finish'));ap.add_argument('--slot',type=int,default=1);ap.add_argument('--phase',choices=('root','infer','bind'));ap.add_argument('--expected',required=True);a=ap.parse_args()
    try:
        if a.action=='verify':dependencies(a.expected);print('ATF1_VERIFIED_ONLY no_PT_load no_GPU');code=0
        elif a.action=='ready':code=ready(a.slot,a.expected)
        elif a.action=='check':validate_admission(a.slot,a.expected);print('ATF1_ADMISSION_VALID');code=0
        elif a.action=='prefix':code=prefix(a.slot,a.expected,a.phase)
        elif a.action=='finish':completed(a.slot,a.expected);print('ATF1_ITEM_COMPLETED NEXT_FIXED_SLOT_REQUIRES_NORMAL_REBOOT',flush=True);code=0
        else:code=complete(a.expected)
        sys.exit(code)
    except (Exception,KeyboardInterrupt):traceback.print_exc();sys.exit(3)
