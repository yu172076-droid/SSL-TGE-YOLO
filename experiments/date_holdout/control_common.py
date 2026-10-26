"""Identity, date-group isolation and task guards for a new independent study."""
import hashlib
import json
import math
from pathlib import Path
from resource_base import read,dump,sha,regular,require,load

PACKAGE=Path(__file__).resolve().parent
TRAIN_SHA='7078fcbc2085a75991282de2701a33d9796be1b5a1af1366139f069507c69f5e'
SEEDS=(101,202,303)
def protocol():return read(PACKAGE/'protocol.json')
def safe(root,rel):
    p=Path(rel)
    require(isinstance(rel,str) and not p.is_absolute() and '..' not in p.parts and '\\' not in rel,'Unsafe relative artifact')
    return regular(Path(root)/p)
def disjoint(a,b):
    a,b=Path(a).absolute(),Path(b).absolute()
    require(a!=b and a not in b.parents and b not in a.parents,'Protected paths overlap')
def verify_package(expected):
    m=regular(PACKAGE/'delivery_manifest.json')
    require(sha(m)==expected and (PACKAGE/'delivery_manifest.sha256').read_text('ascii').split()[0]==expected,'Package SHA/sidecar mismatch')
    rows=read(m)['files'];names={r['path'] for r in rows}
    require(len(names)==len(rows),'Duplicate package path')
    require({p.relative_to(PACKAGE).as_posix() for p in PACKAGE.rglob('*') if p.is_file()}==names|{'delivery_manifest.json','delivery_manifest.sha256'},'Unlisted package files')
    allowed_dirs={q.as_posix() for name in names for q in Path(name).parents if q!=Path('.')}
    for p in PACKAGE.rglob('*'):
        require(not p.is_symlink(),'Package symlink')
        if p.is_dir():require(p.relative_to(PACKAGE).as_posix() in allowed_dirs,'Unlisted package directory')
    for r in rows:
        p=safe(PACKAGE,r['path']);require(p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'Changed package payload')
    return expected
def assignment(seed,gpu,root):
    p=protocol();require(type(seed) is int and seed in SEEDS and p['mapping'][str(seed)]==gpu,'Seed/GPU mapping differs')
    root=Path(root).absolute()
    require(root==root.resolve() and str(root)==p['studies'][str(seed)],'Unexpected private study path')
    disjoint(root,PACKAGE)
    for source in p['source_studies'].values():disjoint(root,source)
    return root
def verify_root(root,expected):
    root=Path(root).absolute();p=protocol();owner=read(safe(root,'owner.json'));seed,gpu=owner['seed'],owner['gpu']
    assignment(seed,gpu,root)
    require(owner==dict(schema='hongyou_dayholdout_stepmatched_owner_v1',seed=seed,gpu=gpu,study=str(root),
        source_study=p['source_studies'][str(seed)],package_sha256=expected,protocol_sha256=sha(PACKAGE/'protocol.json')),'Owner differs')
    prepared=read(safe(root,'prepare_completed.json'))
    require(prepared['owner_sha256']==sha(root/'owner.json') and prepared['package_sha256']==expected
            and prepared['study_json_sha256']==sha(root/'study.json'),'Prepared identity differs')
    original=read(PACKAGE/'references'/('s'+str(seed))/'source_study.json')
    wanted=dict(original['frozen_hashes'])
    for name,ref in p['references'][str(seed)]['reference_files'].items():wanted['reference/'+name]=ref['sha256']
    info=read(root/'study.json')
    require(info['frozen_hashes']==wanted and info['counts']==dict(train=216,val=146,test=74)
            and info['training_seeds']==list(SEEDS) and info['budgets']==[80]
            and info['initial_checkpoint_sha256']==p['source_initial_checkpoint_sha256']
            and info['study_type']=='retrospective_date_heldout_manual_repetition_control','Frozen study scope differs')
    for rel,digest in wanted.items():require(sha(safe(root,rel))==digest,'Frozen input changed: '+rel)
    require(sha(root/'manifest.json')==p['dataset_manifest_sha256'] and sha(root/'label_budgets.json')==p['label_budgets_sha256'],'Data identity differs')
    rows=read(root/'manifest.json');dates={s:{r['capture_date'] for r in rows if r['split']==s} for s in ('train','val','test')}
    require(dates==dict(train={'2019-02-01','2019-02-05'},val={'2019-02-11'},test={'2019-02-23'}),'Date split changed')
    require(len(rows)==len({r['id'] for r in rows})==len({r['image_sha256'] for r in rows})==436,'Duplicate image identity')
    require(all('img (' not in r['source_image'] for r in rows),'Goldfish entered supplement')
    return owner
def verify_original_source(seed):
    p=protocol();source=Path(p['source_studies'][str(seed)]);ref=p['references'][str(seed)]
    inventory=read(PACKAGE/'references'/('s'+str(seed))/'source_export_manifest.json')
    require(sha(PACKAGE/'references'/('s'+str(seed))/'source_export_manifest.json')==ref['export_manifest_sha256'],'Accepted source manifest changed')
    entries=inventory['files']+inventory['omitted_server_only'];wanted={r['path']:r for r in entries}
    require(len(entries)==len(wanted),'Duplicate source inventory')
    actual={q.relative_to(source).as_posix() for q in source.rglob('*') if q.is_file()}
    require(actual==set(wanted),'Original source file inventory changed; no writes permitted')
    for i,(rel,row) in enumerate(wanted.items(),1):
        q=safe(source,rel);require(q.stat().st_size==row['bytes'] and sha(q)==row['sha256'],'Original accepted source changed: '+rel)
        if i%500==0:print('SOURCE_READ_ONLY_HASH_PROGRESS',seed,i,len(wanted),flush=True)
    for name,record in ref['reference_files'].items():
        if name!='source_export_manifest.json':require(sha(safe(source,record['path']))==record['sha256'],'Source reference differs')
    complete=read(source/'formal_complete.json');shell=read(source/'formal_shell_exit.json')
    require(complete['verified_jobs']==3 and complete['verified_tasks']==11 and shell['python_exit']==shell['tee_exit']==0,'Source not formally complete')
    require(sha(source/'jobs'/ref['source_SSL_job']/'attempt_01/train/weights/last.pt')==ref['SSL_checkpoint_sha256'],'Source SSL checkpoint differs')
    return len(wanted)
def train_items(seed):
    return [dict(job='compact_b80_s{}_stepmatched'.format(seed),mode='train',architecture='compact',
                 budget=80,seed=seed,method='supervised',epochs=300,batch=8)]
def cpu_job(seed):return 'preflight_stepmatched_s{}'.format(seed)
def formal_jobs(seed):
    job=train_items(seed)[0]['job'];return [cpu_job(seed),job,job+'_eval_val_640',job+'_eval_test_640']
def completed(root,job):
    folder=Path(root)/'jobs'/job;record=read(safe(folder,'completed.json'));work=Path(record['work'])
    require(work==folder/'attempt_01' and work==work.resolve(),'Unexpected attempt/work path')
    for rel,digest in record['output_hashes'].items():require(sha(safe(folder,rel))==digest,'Completed payload differs: '+job+'/'+rel)
    require(read(work/'exit_status.json')['returncode']==0 and read(work/'worker_execution.json')['actual_exitcode']==0
            and read(work/'worker_execution.json')['status']=='complete','Worker exit differs')
    return record,work
def expected_samples(root,seed,method):
    require(method=='supervised','Only manual-label repetition authorized')
    manual=read(Path(root)/'label_budgets.json')['80']['labeled_ids'];n=protocol()['references'][str(seed)]['SSL_samples']
    require(len(manual)==len(set(manual))==80 and n>=80,'Invalid reference sample count')
    return (manual*((n+79)//80))[:n],None
def validate_view(root,job,ids,texts):
    root=Path(root);view=root/'views'/job;body=read(safe(view,'view.json'));index={r['id']:r for r in read(root/'manifest.json')}
    require(body['input']==dict(ids=ids,label_text=texts) and body['n_training_samples']==len(ids),'View signature differs')
    require(set(ids)<={r['id'] for r in index.values() if r['split']=='train'},'Held-out image entered training')
    expected={}
    for pos,i in enumerate(ids):
        r=index[i];name=str(pos).zfill(6)+'_'+r['staged_name']
        expected['images/train/'+name]=r['image_sha256']
        expected['labels/train/'+Path(name).stem+'.txt']=hashlib.sha256(texts[str(i)].encode('utf-8')).hexdigest() if texts and str(i) in texts else r['label_sha256']
    for r in index.values():
        if r['split']=='val':
            expected['images/val/'+r['staged_name']]=r['image_sha256']
            expected['labels/val/'+Path(r['staged_name']).stem+'.txt']=r['label_sha256']
    expected['data.yaml']=sha(safe(view,'data.yaml'))
    require(body['hashes']==expected,'View image/label provenance differs')
    require(read(view/'data.yaml')==dict(train=str(view/'images/train'),val=str(view/'images/val'),nc=1,names=['fish']),'Native loader path differs')
    allowed=set(expected)|{'view.json','labels/train.cache','labels/val.cache'}
    require({q.relative_to(view).as_posix() for q in view.rglob('*') if q.is_file()}<=allowed,'Unlisted sample/cache in native view')
    for rel,digest in expected.items():require(sha(safe(view,rel))==digest,'View bytes changed')
    return view
def validate_config(config,path,expected):
    root=Path(config['study']).absolute();work=Path(config['work']).absolute();owner=verify_root(root,expected)
    seed,gpu=owner['seed'],owner['gpu']
    require(work==root/'jobs'/work.parent.name/'attempt_01' and work==work.resolve() and path==work/'worker_config.json','Worker path escaped private attempt')
    require(config['package_manifest_sha256']==expected and config['control_protocol_sha256']==sha(PACKAGE/'protocol.json')
            and config['worker_sha256']==sha(PACKAGE/'control_worker.py') and config['runner_sha256']==sha(PACKAGE/'run_queue.py')
            and config['legacy_worker_sha256']==sha(PACKAGE/'legacy/study_worker.py')
            and config['dataset_manifest_sha256']==protocol()['dataset_manifest_sha256']
            and config['study_protocol_sha256']==sha(root/'study.json')
            and config['physical_gpu']==config['device']==gpu and config['backend']=='v7','Worker source/data binding differs')
    b=config['gpu_binding'];require(b['index']==gpu and b['uuid'].startswith('GPU-') and b['pci_bus_id'] and 'V100' in b['name'],'GPU identity missing')
    job=work.parent.name;items={r['job']:r for r in train_items(seed)}
    if config['mode']=='preflight':require(job==cpu_job(seed) and config['seed']==seed,'CPU scope differs')
    elif config['mode']=='train':
        require(job in items,'Unplanned training job')
        for key in ('architecture','budget','seed','method','epochs','batch'):require(config[key]==items[job][key],'Train differs: '+key)
        ids,texts=expected_samples(root,seed,config['method']);view=validate_view(root,job,ids,texts)
        require(config['data']==str(view/'data.yaml') and config['data_view_sha256']==sha(view/'view.json')
                and config['n_training_samples']==len(ids) and config['workers']==4,'Loader differs')
        if config['epochs']==300:
            verify_control_gate(root,expected,seed)
    else:
        require(config['mode']=='infer' and config['imgsz']==640,'Inference mode/size differs')
        source,tail=job.rsplit('_eval_',1);require(source in items and tail in ('val_640','test_640'),'Unplanned evaluation')
        split=tail.split('_')[0];ids=[r['id'] for r in read(root/'manifest.json') if r['split']==split];iou=.65
        weight=completed(root,source)[1]/'train/weights/last.pt'
        require(config['ids']==ids and config['nms_iou']==iou and config['weight']==str(weight)
                and config['weight_sha256']==sha(weight),'Inference list/weight/protocol differs')
    return root,work
def control_gate_body(root,expected,seed):
    job=cpu_job(seed);completed(root,job);ref=protocol()['references'][str(seed)]
    return dict(schema='dateholdout_stepmatched_cpu_source_gate_v1',package_sha256=expected,seed=seed,study=str(root),
        source_export_manifest_sha256=ref['export_manifest_sha256'],source_SSL_trace_sha256=ref['measured_SSL_trace_sha256'],
        CPU_completed_sha256=sha(Path(root)/'jobs'/job/'completed.json'),expected_samples=ref['SSL_samples'],
        expected_optimizer_updates=ref['measured_SSL_updates'],no_teacher_or_SSL_training=True)
def verify_control_gate(root,expected,seed):
    require(read(safe(root,'control_runtime_gate.json'))==control_gate_body(root,expected,seed),'CPU/source gate absent or changed')
def bank_round(n,d):
    q,r=divmod(n,d);return q+int(2*r>d or (2*r==d and q%2==1))
def schedule(samples,epochs,batch):
    require(type(samples) is int and samples>0 and type(epochs) is int and epochs>0 and batch in (4,8),'Bad schedule')
    nb=(samples+batch-1)//batch;nw=max(3*nb,1000);late=64//batch
    rows=[]
    for ni in range(nb*epochs):
        accumulate=max(1,bank_round(nw+(late-1)*ni,nw)) if ni<=nw else late
        if ni%accumulate==0:rows.append((ni//nb,ni%nb,ni,accumulate))
    return nb,nw,rows
class StepTrace:
    def __init__(self,config,work):
        self.config,self.work=config,work
        self.nb,self.nw,self.expected=schedule(config['n_training_samples'],config['epochs'],config['batch'])
        self.reference=[json.loads(line) for line in (Path(config['study'])/'reference/ssl_optimizer_steps.jsonl').read_text(encoding='utf-8').splitlines()]
        require(len(self.reference)==len(self.expected)==protocol()['references'][str(config['seed'])]['measured_SSL_updates'],'Measured reference count differs')
        self.count,self.restored=0,False;self.file=(work/'optimizer_steps.jsonl').open('x',encoding='utf-8')
    def check(self,c):
        require(self.count<len(self.expected) and tuple(c[k] for k in ('epoch','i','ni','accumulate'))==self.expected[self.count],'SGD step positions differ')
        require(c['nb']==self.nb and c['nw']==self.nw and c['nbs']==64 and c['total_batch_size']==self.config['batch']
                and c['actual_dataset_samples']==self.config['n_training_samples'],'Actual loader/accumulation differs')
    def completed(self,c,lrs):
        self.check(c);require(len(lrs)==3 and all(math.isfinite(x) for x in lrs),'Invalid native learning rates')
        reference=self.reference[self.count]
        require(reference['optimizer_step_completed'] and reference['context']==c and max(abs(a-b) for a,b in zip(reference['learning_rates'],lrs))<=1e-12,'Actual SGD context/LR differs from accepted SSL measurement')
        self.file.write(json.dumps(dict(update=self.count+1,context=c,learning_rates=lrs,optimizer_step_completed=True),allow_nan=False)+'\n');self.file.flush();self.count+=1
    def close(self,success):
        self.file.close()
        dump(self.work/'optimizer_step_summary.json',dict(actual_successful_SGD_step_calls=self.count,expected_calls_from_pinned_source=len(self.expected),nb=self.nb,nw=self.nw,
             epochs=self.config['epochs'],samples=self.config['n_training_samples'],batch=self.config['batch'],hook_restored=self.restored,
             success=success and self.count==len(self.expected),trace_sha256=sha(self.work/'optimizer_steps.jsonl')))
        if success:require(self.count==len(self.expected),'Incomplete optimizer trace')
