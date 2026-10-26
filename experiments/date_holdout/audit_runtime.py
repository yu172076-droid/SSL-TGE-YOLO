"""Strict success gates; low AP is a result, not a reason to tune/retry."""
import math
from pathlib import Path
from control_common import *

def check_job(root,job,expected):
    record,work=completed(root,job);config=read(work/'worker_config.json')
    require({k:v for k,v in config.items() if k!='work'}==record['input'],'Completed config differs')
    validate_config(config,work/'worker_config.json',expected)
    execution=read(work/'worker_execution.json')
    require(execution['package_sha256']==expected and execution['config_sha256']==sha(work/'worker_config.json'),'Execution source differs')
    require(execution['finished_unix_time']>=execution['started_unix_time'],'Bad execution timestamps')
    resources=read(work/'parent_resource_checks.json')
    require(resources['error'] is None and resources['samples'] and resources['actual_child_exitcode']==0
            and all(c['free_bytes']>=6*1024**3 for c in resources['samples']),'Disk guard failed')
    mode=config['mode']
    if mode=='preflight':
        p=read(work/'preflight.json')
        require(p['passed'] and p['cpu_only'] and p['initial_nc']==80 and p['initial_architecture_compatible'],'CPU gate failed')
        require([(m['architecture'],m['native_parameters'],m['prediction_shape'],m['all_finite']) for m in p['models']]
                ==[('teacher',37196556,[1,4032,6],True),('compact',4958807,[1,4032,6],True)],'CPU real models differ')
    elif mode=='train':
        summary=read(work/'optimizer_step_summary.json')
        nb,nw,steps=schedule(config['n_training_samples'],config['epochs'],config['batch'])
        require(summary['success'] and summary['hook_restored'] and summary['nb']==nb and summary['nw']==nw
                and summary['actual_successful_SGD_step_calls']==summary['expected_calls_from_pinned_source']==len(steps)
                and summary['trace_sha256']==sha(work/'optimizer_steps.jsonl'),'Optimizer summary differs')
        lines=(work/'optimizer_steps.jsonl').read_text(encoding='utf-8').splitlines();require(len(lines)==len(steps),'Trace length differs')
        for i,(line,reference) in enumerate(zip(lines,steps)):
            row=json.loads(line);c=row['context']
            require(row['update']==i+1 and row['optimizer_step_completed'] and tuple(c[k] for k in ('epoch','i','ni','accumulate'))==reference,'Trace step differs')
            require(len(row['learning_rates'])==3 and all(math.isfinite(x) and 0<=x<=.101 for x in row['learning_rates']),'Native SGD LR invalid')
        reference=[json.loads(line) for line in (Path(root)/'reference/ssl_optimizer_steps.jsonl').read_text(encoding='utf-8').splitlines()]
        require(len(reference)==len(lines),'Measured reference count differs')
        for actual,prior in zip(map(json.loads,lines),reference):
            require(actual['context']==prior['context'] and actual['update']==prior['update'] and max(abs(a-b) for a,b in zip(actual['learning_rates'],prior['learning_rates']))<=1e-12,'Measured reference context/LR differs')
        retention=read(work/'worker_retention_audit.json')
        require(retention['success'] and retention['hook_restored'] and not retention['checkpoint_deletion_performed'],'Retention hook failed')
        require({p.name for p in (work/'train/weights').iterdir()}=={'init.pt','best.pt','last.pt'},'Unexpected checkpoint retention')
        settings=read(work/'training_settings.json');o=settings['native_options']
        require(settings['training_seed']==config['seed'] and settings['precision']=='FP32' and settings['amp'] is False
                and o['epochs']==config['epochs'] and o['batch_size']==config['batch'] and o['resume'] is False,'Actual training options differ')
        wanted_weight=str(Path(root)/'initial/yolov7.pt') if config['architecture']=='teacher' else ''
        require(o['weights']==wanted_weight and o['img_size']==[640,640] and o['workers']==4,'Initialization/resolution/workers differ')
        calls=read(work/'seed_applied.json');require(len(calls)==1 and calls[0]['applied_training_seed']==config['seed'],'Seed hook failed')
        results=(work/'train/results.txt').read_text(encoding='utf-8').splitlines()
        require(len(results)==config['epochs'],'Native epoch rows incomplete')
        for line in results:
            cells=line.split();require(len(cells)>=10 and all(math.isfinite(float(x)) for x in cells[2:]),'Nonfinite native losses/metrics')
    else:
        pred=read(work/'predictions.json');wanted=set(config['ids']);index={r['id']:r for r in read(Path(root)/'manifest.json')}
        counts={}
        for row in pred:
            require(row['image_id'] in wanted and row['category_id']==1 and 0<=row['score']<=1,'Prediction ID/category/score differs')
            x,y,w,h=row['bbox'];r=index[row['image_id']]
            require(all(math.isfinite(v) for v in (x,y,w,h,row['score'])) and x>=0 and y>=0 and w>0 and h>0
                    and x+w<=r['width']+1e-6 and y+h<=r['height']+1e-6,'Invalid box')
            counts[row['image_id']]=counts.get(row['image_id'],0)+1
        require(not counts or max(counts.values())<=300,'NMS max_det differs')
        meta=read(work/'inference_metadata.json')
        require(meta['image_count']==len(wanted) and meta['checkpoint_sha256']==config['weight_sha256'] and meta['settings']==config,'Inference evidence differs')
        if '_eval_' in job:
            import numpy as np
            m=read(work/'coco_metrics.json');split=job.rsplit('_eval_',1)[1].split('_')[0]
            require(m['input']['gt_sha256']==sha(Path(root)/('gt_'+split+'.json')) and m['input']['predictions_sha256']==sha(work/'predictions.json')
                    and m['checkpoint_sha256']==config['weight_sha256'] and m['split']==split,'COCO input binding differs')
            require(all(v is None or math.isfinite(v) and 0<=v<=1 for v in m['metrics'].values()) and len(m['metrics'])==12,'COCO metrics invalid')
            with np.load(str(work/'coco_precision_recall.npz'),allow_pickle=False) as a:
                require(a['precision'].shape==(10,101,1,4,3) and a['recall'].shape==(10,1,4,3),'COCO array shape differs')
                require(all(np.isfinite(a[k]).all() and (a[k]>=-1).all() and (a[k]<=1).all() for k in ('precision','recall')),'COCO array invalid')
    return sha(Path(root)/'jobs'/job/'completed.json')

def audit_jobs(root,jobs,expected):
    return {job:check_job(root,job,expected) for job in jobs}
