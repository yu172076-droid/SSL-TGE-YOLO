"""ATF1 common predictions: official-evaluator input binding, no AP run."""
import json,math
from common import *

def jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]

def raw_counters(raw):
    return {p[0]:int(p[1]) for line in raw.splitlines() if len(p:=line.split())==2 and p[1].isdigit()}

def raw_memory(raw):
    answer={}
    for line in raw.splitlines():
        p=line.replace(':','').split()
        if len(p)>=2 and p[1].isdigit():
            answer[p[0]+'_bytes']=int(p[1])*(1024 if len(p)>2 and p[2]=='kB' else 1)
    return answer

def memory_evidence(directory, guard, admission=None):
    import startup_monitor
    rows=jsonl(directory/'process_memory_samples.jsonl')
    require(len(rows)==guard['samples'] and rows and guard['failure'] is None,'Memory evidence incomplete')
    previous=None;streak=0;initial_oom=None
    for index,r in enumerate(rows):
        if index:require(r['monotonic_ns']>rows[index-1]['monotonic_ns'],'Memory times not increasing')
        vm=raw_counters(r['raw']['vmstat']);mem=raw_memory(r['raw']['meminfo'])
        require(all(r['vm_counters'][k]==vm[k] for k in ('pswpin','pswpout','pgmajfault'))
                and r['proc_meminfo']==mem,'Raw/parsed memory counters differ')
        if initial_oom is None:initial_oom=vm['oom_kill']
        require(vm['oom_kill']==initial_oom,'OOM changed')
        require(r['gate']['passed'] and mem['MemAvailable_bytes']>=r['gate']['minimum_MemAvailable_bytes']>=GIB,'Capacity failure')
        if previous is None or r['monotonic_ns']-previous['monotonic_ns']>=750000000:
            if previous is not None:
                delta={k:r['vm_counters'][k]-previous['vm_counters'][k] for k in ('pswpin','pswpout','pgmajfault')}
                require(delta==r['vm_counter_deltas'] and all(v>=0 for v in delta.values()),'Interval deltas differ')
                streak=streak+1 if delta['pswpin']>0 or delta['pswpout']>0 else 0
                require(streak==r['consecutive_positive_global_swap_intervals'],'Paging streak differs')
                if r['paging_failure_enforced']:require(streak<2,'Protected diagnostic paging failure')
            previous=r
    if admission is not None:
        proof=admission['stability'];last=proof['end_monotonic_ns']
        prefix=[r for r in rows if r['monotonic_ns']<=last]
        actual=startup_monitor.stable_tail(prefix,proof['warmup_start_ns'])
        require(actual==proof and actual['ready'],'Stable30 proof differs')
        require(admission['activation']==guard['activation']
                and admission['activation']['stability']==proof
                and admission['activation']['monotonic_ns']>=last,'Protection activation differs')
        activation=admission['activation']['monotonic_ns']
        require(all(r['paging_failure_enforced']==(r['monotonic_ns']>=activation) for r in rows),'Protected interval phase differs')
    return rows

def validate_record(record,width,height,precision,confidence):
    from runtime_adapter import validate_values
    require(record['native_shape']==[height,width,3] and record['native_bytes']==height*width*3
            and record['tensor_shape']==[1,3,640,640] and record['tensor_dtype']==('torch.float16' if precision=='FP16' else 'torch.float32')
            and record['decoded_shape']==[1,25200,6] and record['nms_dtype']=='torch.float32','Native tensor record differs')
    if 'detections_xyxy_conf_class' in record:
        values=record['detections_xyxy_conf_class']
        validate_values(values,width,height,confidence)
        require(record['detection_count']==len(values),'Detection count differs')
    for raw in record['raw_tensors']:
        require(raw['device']=='cuda:0' and raw['dtype']==record['tensor_dtype'],'Raw CUDA output differs')

def validate_worker(out,model,kind,precision,session,expected,boot_id,rows):
    r=report(out/'worker_report.json');job=job_config(model)
    require(r['status']=='passed' and r['model_id']==model and r['kind']==kind and r['precision']==precision
            and r['session_id']==session and r['package_manifest_sha256']==expected and r['boot_id']==boot_id
            and r['weight_sha256']==job['weight_sha256'] and r['intended_self_exit_code']==0,'Worker identity/exit differs')
    require(r['attention_manifest_sha256']==protocol()['attention_manifest_sha256']
            and r['dataset_manifest_sha256']==protocol()['dataset_manifest_sha256'] and r['GT_sha256']==protocol()['GT_sha256']
            and r['video_sha256']==protocol()['video_sha256'],'Worker asset binding differs')
    for flag in ('AP_evaluated','formal_accuracy_completed','video_FPS_measured','formal_cost_completed','power_measured','installation_performed'):
        require(r[flag] is False,'Unsupported metric claim')
    verify_artifacts(out,r['artifacts'])
    require('NMS time limit' not in (out/'worker_console.txt').read_text(encoding='utf-8',errors='replace'),'NMS time limit warning')
    launch=read(out/'launch.json');exit_record=read(out/'exit_status.json')
    require(launch['session_id']==session and launch['boot_id']==boot_id and launch['model']==model and launch['kind']==kind and launch['precision']==precision
            and r['parent_pid']==launch['parent_pid'] and r['pid']==exit_record['pid']
            and exit_record['actual_exit_code']==0 and not exit_record['timed_out'] and not exit_record['interrupted'],'Actual child launch/exit differs')
    count=298 if kind=='images' else 130
    require(r['forward_calls_started']==r['_forward_gate']['intercepted_calls']==r['structure_hook_calls']==count
            and r['_forward_gate']['open'] is False,'Forward/structure hook count differs')
    structure=r['AT_structure_before_fusion']
    require(structure['parameters']==job['parameters'] and structure['module_parameters']==job['module_parameters']
            and structure['detect_input_channels']==[128,128,512] and structure['layer_index']==55 and structure['source_index']==45
            and structure['P4_runtime_shapes']==[dict(input=[1,128,40,40],output=[1,128,40,40],dtype='torch.float16' if precision=='FP16' else 'torch.float32')],'P4 evidence differs')
    require(r['model']['checkpoint_member']=='model' and r['model']['parameters_before_fusion']==job['parameters']
            and r['model']['stored_parameter_dtypes']==['torch.float16'],'Loaded model/member differs')
    source_rows={v['path'][len('source/'):]:v for v in read(PROJECT/protocol()['attention_directory']/'delivery_manifest.json')['files'] if v['path'].startswith('source/')}
    for origins in (r['source_origins_before_load'],r['source_origins_after_inference']):
        require(origins,'Empty imported source origins')
        for name,origin in origins.items():
            marker='/'+protocol()['attention_directory']+'/source/'
            normalized=origin['path'].replace('\\','/')
            require(marker in normalized,'Imported source path differs')
            rel=normalized.split(marker,1)[1]
            require(rel in source_rows and origin['sha256']==source_rows[rel]['sha256'],'Imported source SHA differs')
    memory=memory_evidence(out,r['resource_guard'],r['protected_diagnostic_admission'])
    capacities=r['resource_guard']['capacity_admissions']
    require([v['phase'] for v in capacities]==['before_checkpoint_load','loaded_before_native_decode','first_image_capacity_wait' if kind=='images' else 'first_video_frame_capacity_wait']
            and [v['target_bytes'] for v in capacities]==[3*GIB,3*GIB+24883200,3*GIB],'Capacity admission targets differ')
    for v in capacities:
        row=memory[v['sample_index']]
        require(row['monotonic_ns']==v['sample_monotonic_ns'] and row['proc_meminfo']['MemAvailable_bytes']==v['available_bytes']>=v['target_bytes'],'Capacity admission sample differs')
    require(any(v['phase']=='first_native_image_before_first_forward' if kind=='images' else v['phase']=='first_native_video_frame_before_first_forward' for v in memory),'Native first-frame gate absent')
    activation=r['resource_guard']['activation']['monotonic_ns']
    if kind=='images':
        require(r['compatibility_images']==0,'No separate compatibility forward allowed')
        images=jsonl(out/'image_records.jsonl')
        require(len(images)==298 and [(v['split'],v['image_id']) for v in images]==[(v['split'],v['id']) for v in rows]
                and r['completed_by_split']==dict(val=149,test=149) and r['accuracy_predictions_exported'],'Accuracy images incomplete')
        for ordinal,(record,row) in enumerate(zip(images,rows)):
            validate_record(record,row['width'],row['height'],precision,.001)
            require(record['image_sha256']==row['image_sha256'] and (record['completed_monotonic_ns']>activation)==(ordinal>=5),'Accuracy identity/protection order differs')
            require(record['detection_count']==len(record['predictions'])+len(record['zero_area_excluded'])<=300,'Accuracy detections missing')
            for box in record['predictions']:
                b=box['bbox']
                require(box['image_id']==row['id'] and box['category_id']==1 and len(b)==4
                        and all(isinstance(x,(int,float)) and math.isfinite(x) for x in b+[box['score']])
                        and .001<box['score']<=1 and b[0]>=0 and b[1]>=0 and b[2]>0 and b[3]>0
                        and b[0]+b[2]<=row['width']+1e-8 and b[1]+b[3]<=row['height']+1e-8,'Invalid COCO box')
        for split in ('val','test'):
            require(read(out/('predictions_'+split+'.json'))==[b for im in images if im['split']==split for b in im['predictions']],'Prediction JSON/records differ')
    else:
        video=jsonl(out/'frame_records.jsonl')
        require(len(video)==130 and [v['source_decode_ordinal'] for v in video]==list(range(1,131))
                and [v['phase'] for v in video]==['warmup']*30+['probe']*100
                and r['completed_by_phase']==dict(warmup=30,probe=100) and r['video_inputs_unchanged'],'Short-video sequence differs')
        for v in video:
            validate_record(v,3840,2160,'FP16',.25)
            require((v['completed_monotonic_ns']>activation)==(v['phase']=='probe'),'Short-video protection order differs')
    return r
