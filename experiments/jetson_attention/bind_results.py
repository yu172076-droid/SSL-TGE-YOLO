"""CPU-only root/window/raw-resource binding for ATF1 native measurements."""
import argparse,math,sys,traceback
from common import *
import reference_binder as b
import resource_binding
import video_window as driver
from native_frame import validate_values
driver.validate_values=validate_values
def audit(slot,expected):
    manifest=verify(expected);p=paths(slot);row=slot_config(slot);job=job_config(row['model']);payload={r['path']:r['sha256'] for r in manifest['files']};bindings=[]
    rr,root_sha,_=b.report_artifacts(p['root'],'root_sampler_report.json',bindings)
    lr,launch_sha,_=b.report_artifacts(p['infer'],'launcher_report.json',bindings)
    ir=get_report(p['infer'],'infer_report');infer_sha=sha(p['infer']/'infer_report.json')
    require(lr['schema']=='ATF1_native_cost_launcher_v1' and ir['schema']=='ATF1_native_cost_inference_v1'
        and lr['exitcode']==ir['actual_self_exit_code']==0 and lr['child']['exitcode']==0 and not lr['child']['timed_out'] and not lr['child']['interrupted'] and lr['child']['pid']==ir['pid'] and lr['normal_euid']==ir['euid']!=0,'Actual normal-user inference identity/exit differs')
    require(lr['manifest_sha256']==ir['manifest_sha256']==expected and ir['benchmark_completed'] and lr['boot_id']==ir['boot_id']==rr['boot_id'] and lr['slot']==ir['slot']==slot,'Cost session incomplete/misbound')
    require(lr['job']==ir['job']==row['model'] and lr['replicate']==ir['replicate']==row['replicate'] and lr['weight_sha256']==ir['weight_sha256']==job['weight_sha256'],'Checkpoint/model/rep differs')
    require(ir['video_sha256']==protocol()['video_sha256'] and ir['video_unchanged'] and ir['capacity_policy']==protocol()['capacity_policy'],'Video/capacity identity differs')
    require(ir['AP_evaluated'] is False and ir['AP_gate_pending'] is True and ir['TensorRT_executed'] is False and ir['tracking_executed'] is False,'Accuracy/backend scope differs')
    require(ir['config']==dict(batch=1,input_shape=[1,3,640,640],precision='native_PyTorch_FP16',conf=.25,iou=.65,max_det=300,native_dimensions=[3840,2160],expected_frames_per_pass=1475,backend='FFMPEG',firefox_must_remain_open=True),'Cost settings differ')
    require(not lr['nms_time_limit_warning'] and 'NMS time limit' not in (p['infer']/'worker_console.txt').read_text(encoding='utf-8',errors='replace'),'NMS warning')
    require(ir['model']['parameters_before_fusion']==job['parameters'] and ir['model']['checkpoint_member']=='model'
        and ir['AT_structure_before_fusion']['module_parameters']==job['module_parameters'] and ir['AT_structure_before_fusion']['detect_input_channels']==[128,128,512]
        and ir['structure_hook_calls']==1 and ir['shape_hook_once'] is True and ir['_forward_gate']['open'] is False,'Integrated structural compatibility failed')
    env=ir['environment']
    require([env[k] for k in ('torch','cuda','opencv','numpy','torchvision','torch_threads','cv2_threads')]==['2.1.0a0+41361538.nv23.06','11.4','4.13.0','1.24.4','0.16.2',6,1],'Qualified runtime/threads differ')
    origins=ir['source_origins_before_load']
    require(origins and all(ir['source_origins_after_inference'].get(k)==v for k,v in origins.items()),'Imported source changed during inference')
    source_rows={v['path'][7:]:v['sha256'] for v in read(PROJECT/protocol()['attention_directory']/'delivery_manifest.json')['files'] if v['path'].startswith('source/')}
    for collection in (origins,ir['source_origins_after_inference']):
        for value in collection.values():
            marker='/'+protocol()['attention_directory']+'/source/';normalized=value['path'].replace('\\','/')
            require(marker in normalized and source_rows.get(normalized.split(marker,1)[1])==value['sha256'],'Frozen imported source origin differs')
    for name,pin in payload.items():require(sha(p['infer']/'source_snapshot/packet'/name)==pin,'Packet snapshot differs: '+name)
    require(sha(p['infer']/'source_snapshot/packet/delivery_manifest.json')==expected,'Packet manifest snapshot differs')
    source=PROJECT/protocol()['attention_directory']/'source'
    for asset in source.rglob('*'):
        if asset.is_file():require(sha(p['infer']/'source_snapshot/attention_source'/asset.relative_to(source))==sha(asset),'Frozen source snapshot differs')
    session,boot_id,ready,ready_sha,lower,upper,root_rows,sysfs,parsed,excluded=b.validate_root(p['root'],ir,rr,payload,expected,module('ATF1_root_telemetry',ROOT/'telemetry.py'))
    memory=b.jsonl(p['infer']/'process_memory_samples.jsonl');resources=resource_binding.validate_memory(memory,ir,session,boot_id,expected,ready,payload)
    telemetry=module('ATF1_window_telemetry',ROOT/'telemetry.py').Telemetry();telemetry.records=parsed
    telemetry.rss_records=[dict(monotonic_ns=r['monotonic_ns'],process_rss_bytes=r['process_status'].get('VmRSS_bytes')) for r in memory]
    windows=ir['windows'];require(set(windows)=={'baseline','warmup','measurement'} and windows==read(p['infer']/'measurement_window.json'),'Window artifact differs')
    idle=windows['baseline'];require(idle['model_loaded'] and idle['capture_closed'] and idle['native_frames_retained']==idle['frames']==0 and idle['inference_executed'] is False,'Baseline not loaded idle')
    summaries={};previous_end=ready['ready_monotonic_ns'];calls=0
    for phase in ('baseline','warmup','measurement'):
        w=windows[phase];start,end=w['start_monotonic_ns'],w['end_monotonic_ns'];duration=(end-start)/1e9
        require(w['complete'] and lower<=previous_end<=start<end<=upper and lr['started_monotonic_ns']<=start<end<=lr['ended_monotonic_ns']
            and 60<=duration<=dict(baseline=90,warmup=300,measurement=600)[phase] and math.isclose(duration,w['duration_s'],abs_tol=1e-8),'Window duration/order/lifecycle differs')
        previous_end=end;power=telemetry.window(start,end)
        require(power['power_coverage_adequate'] and power['module_power_VDD_IN_mean_W'] is not None,'Inadequate same-window power '+phase)
        inside=[r for r in sysfs if start<=r['monotonic_ns']<=end];require(inside,'No sysfs window samples')
        for r in inside:
            pair=r.get('vdd_in') or {};require(pair.get('binding')==ready['vdd_in_binding'] and b.node_number(pair.get('voltage'),True) is not None and b.node_number(pair.get('current'),True) is not None,'VDD_IN raw binding invalid')
        frequency=dict(gpu={k:b.frequency_summary([r.get('gpu',{}).get(k,{}) for r in inside],'Hz') for k in ('cur_freq','min_freq','max_freq')},emc_rate=b.frequency_summary([r.get('emc_rate',{}) for r in inside],'Hz'))
        require(frequency['gpu']['cur_freq']['valid_count'] and frequency['emc_rate']['valid_count'],'GPU/EMC observation absent')
        summaries[phase]=dict(window=w,power=power,frequencies=frequency,sysfs_sampling=b.sample_gap_summary([r['monotonic_ns'] for r in inside],start,end))
        if phase!='baseline':
            require(w['input_protocol']==driver.PROTOCOL and w['max_resident_native_images']==1,'Wrong timing scope')
            frames,latencies=b.validate_video_frames(p['infer'],phase,w,driver);calls+=len(frames)
            require(latencies==ir[phase+'_latency'],'Frame latency recomputation differs');summaries[phase]['latency']=latencies
            if phase=='warmup':
                first=[r for r in memory if r['phase']=='first_decoded_frame_before_first_forward']
                require(len(first)==1 and int(frames[0]['read_end_monotonic_ns'])<=first[0]['monotonic_ns']<=ir['warmup_stability_start_monotonic_ns']<=int(frames[0]['start_monotonic_ns'])
                    and end-ir['warmup_stability_start_monotonic_ns']>=60*10**9,'Native first-frame/warmup gate differs')
    h=ir['warmup_handoff'];w=windows['warmup'];act=ir['formal_guard_activation'];post=[r for r in memory if r['phase']=='postwarm' and r['monotonic_ns']<act['monotonic_ns']]
    require(h['policy']==protocol()['handoff_policy'] and h['accepted'] and h['capture_closed'] and h['records_flushed'],'Warmup handoff policy failed')
    require(post and h['latest_sample_monotonic_ns']==post[-1]['monotonic_ns']==act['stability']['end_monotonic_ns']
        and w['passes'][-1]['record_flush_end_monotonic_ns']<=post[-1]['monotonic_ns']<=w['end_monotonic_ns']==h['warmup_end_monotonic_ns']<=act['monotonic_ns']==h['activation_monotonic_ns'],'Fresh after-close/flush handoff missing')
    require(h['complete_EOF_passes']==len(w['passes']) and all(x['complete'] and x['eof_seen'] and x['frames']==1475 for x in w['passes']),'Warmup must consist of full EOF passes')
    require(calls==ir['forward_calls_started']==ir['_forward_gate']['intercepted_calls'],'Unexpected actual forwards')
    admission=report(p['admission'])
    require(sha(p['infer']/'admission.json')==sha(p['admission']) and admission['boot_id']==boot_id and admission['weight_sha256']==job['weight_sha256'],'Admission chain differs')
    import control
    control.validate_quiet(admission,slot,expected)
    require(p['terminal'].read_text().splitlines()==exit_lines(slot,('VERIFY','PREFLIGHT','INFER','ROOT')),'Four actual pre-bind exits incomplete/nonzero')
    m=windows['measurement'];rate=m['frames']/m['duration_s'];require(math.isclose(rate,ir['video_window_throughput_FPS'],abs_tol=1e-8),'FPS differs')
    pw,bw=[summaries[x]['power']['module_power_VDD_IN_mean_W'] for x in ('measurement','baseline')]
    formal_memory=[r for r in memory if m['start_monotonic_ns']<=r['monotonic_ns']<=m['end_monotonic_ns']];require(formal_memory,'Formal memory samples absent')
    resources.update(formal_window_sampled_RSS_peak_MiB=max(r['process_status']['VmRSS_bytes'] for r in formal_memory)/1024**2,
        process_lifetime_VmHWM_peak_MiB=max(r['process_status']['VmHWM_bytes'] for r in memory)/1024**2,
        formal_window_minimum_MemAvailable_bytes=min(r['proc_meminfo']['MemAvailable_bytes'] for r in formal_memory),
        Torch_allocator_postwarm_peak_MiB=ir['CUDA_allocator_postwarm_peak_MiB'],memory_scope_note='RSS/VmHWM/Torch allocator/system are overlapping unified-memory views; do not sum')
    result=dict(schema='ATF1_cost_joined_v1',status='ATF1_COST_COMPLETED_PRECISION_REVIEW_PENDING',exitcode=0,manifest_sha256=expected,slot=slot,job=row['model'],replicate=row['replicate'],weight_sha256=job['weight_sha256'],root_session_id=session,boot_id=boot_id,
        root_report_sha256=root_sha,launcher_report_sha256=launch_sha,infer_report_sha256=infer_sha,admission_sha256=sha(p['admission']),resources=resources,
        video_FPS_measured=True,power_measured=True,AP_evaluated=False,AP_gate_pending=True,TensorRT_executed=False,tracking_executed=False,
        video_window_throughput_FPS=rate,measurement_frames=m['frames'],complete_passes=len(m['passes']),
        module_power_and_efficiency=dict(scope='same-window total module VDD_IN including browser/background, not wall/isolated GPU',module_power_mean_W=pw,loaded_idle_baseline_module_power_mean_W=bw,signed_incremental_module_power_W=pw-bw,FPS_per_module_W=rate/pw,module_energy_per_frame_J=pw/rate,signed_incremental_energy_per_frame_J=(pw-bw)/rate))
    return result,summaries,bindings
def main(args):
    verify(args.expected);p=paths(args.slot);out=create_output(p['joined'].name);code=3
    try:
        result,summaries,bindings=audit(args.slot,args.expected)
        write(out/'window_summary.json',summaries);write(out/'read_bindings.json',bindings);shutil.copyfile(p['terminal'],out/'terminal_before_bind.txt');code=0
    except BaseException as exc:result=dict(status='incomplete',exitcode=3,error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    result.update(passed_all=False,artifacts=artifacts(out));signed(out/'joined_report.json',result);print(result['status'],flush=True);return code
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--slot',type=int,required=True);ap.add_argument('--expected',required=True);a=ap.parse_args();sys.exit(main(a))
