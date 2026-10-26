"""Direct formal queue authorized by the user; no training smoke or manual mid-queue release."""
import argparse
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from control_common import *
from audit_runtime import audit_jobs
import resource_base as resources

def make_study(root,a,binding):
    sys.path.insert(0,str(root/'run_package'))
    legacy=load('pinned_date_study',root/'run_package/run_internal_study.py')
    class DateStudy(legacy.Study):
        def task(self,job,config):
            config=dict(config,study=str(root),physical_gpu=a.gpu,device=a.gpu,gpu_binding=binding,
                package_manifest_sha256=a.expected_package_sha256,control_protocol_sha256=sha(PACKAGE/'protocol.json'),
                worker_sha256=sha(PACKAGE/'control_worker.py'),runner_sha256=sha(PACKAGE/'run_queue.py'),
                legacy_worker_sha256=sha(PACKAGE/'legacy/study_worker.py'),study_protocol_sha256=sha(root/'study.json'))
            folder=root/'jobs'/job
            require(not folder.exists(),'Existing job preserved; no automatic retry/rewrite: '+job)
            resources.free_check(root);resources.gpu_check(a.gpu,idle=True)
            work=folder/'attempt_01';work.mkdir(parents=True,exist_ok=False)
            full=dict(config,work=str(work));dump(work/'worker_config.json',full)
            env=os.environ.copy()
            for key in ('RANK','WORLD_SIZE','LOCAL_RANK'):env.pop(key,None)
            env.update(CUDA_VISIBLE_DEVICES='' if config['mode']=='preflight' else a.gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID',
                PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1',WANDB_MODE='disabled',
                YOLO_AUTOINSTALL='false',MPLCONFIGDIR=str(work/'plot_config'),YOLO_CONFIG_DIR=str(work/'ultralytics_settings'),
                OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
            command=[sys.executable,'-B',str(PACKAGE/'control_worker.py'),'--config',str(work/'worker_config.json'),
                     '--expected-package-sha256',a.expected_package_sha256]
            dump(work/'started.json',dict(command=command,unix_time=time.time(),input=config))
            print('START',job,'GPU',a.gpu,'log',work/'output.log',flush=True)
            code=resources.run_child(command,work,env,root)
            require(code==0,'Worker failed; queue stopped and attempt retained')
            required=['worker_config.json','started.json','worker_execution.json','exit_status.json','parent_resource_checks.json']
            if config['mode']=='train':required+=['train/weights/init.pt','train/weights/best.pt','train/weights/last.pt','train/results.txt',
                'train/opt.yaml','train/hyp.yaml','training_settings.json','seed_applied.json','worker_retention_audit.json',
                'optimizer_steps.jsonl','optimizer_step_summary.json','architecture_evidence.json']
            elif config['mode']=='preflight':required+=['preflight.json']
            else:required+=['predictions.json','inference_metadata.json']
            hashes={p.relative_to(folder).as_posix():sha(p) for p in [regular(work/r) for r in required]}
            dump(folder/'completed.json',dict(input=config,work=str(work),output_hashes=hashes))
            print('DONE',job,flush=True)
            return work
        def evaluate(self,job,backend,weight,split,imgsz):
            super().evaluate(job,backend,weight,split,imgsz)
            identifier=job+'_eval_'+split+'_640';record,work=completed(root,identifier)
            metrics=read(work/'coco_metrics.json')
            metrics.update(date_group_disjoint_verified=True,source_clock_basis='Provider confirmation and visible dates',
                independent_generalization_test=False,full_source_video_sequence_identity_verified=False,
                control_wrapper_manifest_sha256=a.expected_package_sha256,
                experiment_scope='Retrospective date-held-out manual repetition control matched to measured SSL updates',
                main_checkpoint_rule='Nonformal 2-epoch smoke last, excluded from formal statistics' if job.startswith('smoke_') else 'Fixed 300 epochs, last.pt')
            dump(work/'coco_metrics.json',metrics)
            record['output_hashes'][(work/'coco_metrics.json').relative_to(root/'jobs'/identifier).as_posix()]=sha(work/'coco_metrics.json')
            dump(root/'jobs'/identifier/'completed.json',record)
    return DateStudy(root,SimpleNamespace(device=a.gpu,workers=4,retry_failed=False,v8_python=None))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=('formal',),required=True)
    ap.add_argument('--study',type=Path,required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--gpu',required=True)
    ap.add_argument('--expected-package-sha256',required=True);ap.add_argument('--execute',action='store_true')
    a=ap.parse_args();verify_package(a.expected_package_sha256);root=assignment(a.seed,a.gpu,a.study);ref=protocol()['references'][str(a.seed)]
    print('PLAN',dict(phase='formal',seed=a.seed,gpu=a.gpu,study=str(root),formal_trainings=1,epochs_each=300,
        unique_manual_images=80,repeated_samples=ref['SSL_samples'],measured_reference_updates=ref['measured_SSL_updates'],
        training_smoke_jobs=0,no_teacher_or_SSL_retraining=True),flush=True)
    if not a.execute:return
    require(sys.platform=='linux','Linux server only');verify_root(root,a.expected_package_sha256);verify_original_source(a.seed)
    require(not (root/'jobs').exists() and not (root/'formal_complete.json').exists(),'Existing attempts retained; no retry or duplicate execution')
    lockdir=Path(protocol()['gpu_lock_directory']);lockdir.mkdir(exist_ok=True)
    with resources.Lock(lockdir/('gpu_'+a.gpu+'.lock')),resources.Lock(root/'queue.lock'):
        resources.free_check(root);binding=resources.gpu_check(a.gpu,idle=True);study=make_study(root,a,binding)
        study.task(cpu_job(a.seed),dict(mode='preflight',backend='v7',seed=a.seed,dataset_manifest_sha256=protocol()['dataset_manifest_sha256']))
        audit_jobs(root,[cpu_job(a.seed)],a.expected_package_sha256)
        dump(root/'control_runtime_gate.json',control_gate_body(root,a.expected_package_sha256,a.seed))
        print('DATE_STEPMATCHED_CPU_SOURCE_GATE_PASS',a.seed,flush=True)
        item=train_items(a.seed)[0];job=item['job'];ids,texts=expected_samples(root,a.seed,'supervised')
        view=study.make_view(job,ids,texts);validate_view(root,job,ids,texts)
        config=dict(mode='train',backend='v7',architecture='compact',budget=80,seed=a.seed,method='supervised',epochs=300,
            batch=8,workers=4,data=str(view),data_view_sha256=sha(view.parent/'view.json'),
            dataset_manifest_sha256=protocol()['dataset_manifest_sha256'],n_training_samples=len(ids))
        work=study.task(job,config)
        for split in ('val','test'):study.evaluate(job,'v7',work/'train/weights/last.pt',split,640)
        accepted=audit_jobs(root,formal_jobs(a.seed),a.expected_package_sha256)
        source_count=verify_original_source(a.seed);verify_root(root,a.expected_package_sha256)
        results=[]
        for split in ('val','test'):
            m=read(completed(root,job+'_eval_'+split+'_640')[1]/'coco_metrics.json')
            results.append(dict(architecture='compact',method='stepmatched_manual_repetition',seed=a.seed,budget=80,
                split=split,metrics=m['metrics'],checkpoint_sha256=m['checkpoint_sha256']))
        dump(root/'formal_result_rows.json',results)
        dump(root/'formal_complete.json',dict(schema='dateholdout_stepmatched_complete_v1',seed=a.seed,gpu=a.gpu,
            package_sha256=a.expected_package_sha256,accepted_completed_sha256=accepted,verified_jobs=1,verified_tasks=4,
            formal_trainings=1,training_smoke_jobs=0,epochs_each=300,unique_manual_images=80,
            actual_training_samples=ref['SSL_samples'],matched_measured_reference_updates=ref['measured_SSL_updates'],
            source_export_manifest_sha256=ref['export_manifest_sha256'],source_SSL_trace_sha256=ref['measured_SSL_trace_sha256'],
            original_source_files_reverified=source_count,no_teacher_or_SSL_retraining=True,finished_unix_time=time.time()))
        print('DATE_STEPMATCHED_FORMAL_COMPLETE',a.seed,'verified_jobs=1/1 verified_tasks=4',flush=True)
if __name__=='__main__':main()
