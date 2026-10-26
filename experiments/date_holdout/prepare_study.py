"""Read accepted sources in place, create a new private manual control only."""
import argparse
import os
import shutil
import sys
import time
from control_common import *
import resource_base as resources
def prepare(root,source,seed,gpu,expected):
    verify_package(expected);assignment(seed,gpu,root)
    require(str(source)==protocol()['source_studies'][str(seed)] and source==source.resolve(),'Wrong source study')
    require(not root.exists(),'Target exists; retain partial, no automatic retry/adoption')
    count=verify_original_source(seed);original=read(source/'study.json');inputs=[]
    for rel,digest in original['frozen_hashes'].items():inputs.append((safe(source,rel),rel,digest))
    for name,record in protocol()['references'][str(seed)]['reference_files'].items():
        inputs.append((safe(PACKAGE,'references/s{}/{}'.format(seed,name)),'reference/'+name,record['sha256']))
    for origin,rel,digest in inputs:require(sha(origin)==digest,'Prepare source changed')
    resources.free_check(root.parent);root.mkdir(exist_ok=False)
    dump(root/'owner.json',dict(schema='hongyou_dayholdout_stepmatched_owner_v1',seed=seed,gpu=gpu,study=str(root),
        source_study=str(source),package_sha256=expected,protocol_sha256=sha(PACKAGE/'protocol.json')))
    for origin,rel,digest in inputs:
        dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        if rel.startswith('dataset/images/') or rel=='initial/yolov7.pt':
            try:os.link(str(origin),str(dest))
            except OSError:shutil.copy2(origin,dest)
        else:shutil.copy2(origin,dest)
        require(sha(regular(dest))==digest,'Prepared bytes differ')
    info=dict(original);info['study_type']='retrospective_date_heldout_manual_repetition_control'
    info['frozen_hashes']={rel:digest for origin,rel,digest in inputs}
    info.update(no_teacher_or_SSL_retraining=True,unique_manual_training_images=80,
        repeated_training_samples=protocol()['references'][str(seed)]['SSL_samples'],
        matched_measured_SSL_updates=protocol()['references'][str(seed)]['measured_SSL_updates'])
    dump(root/'study.json',info)
    require(verify_original_source(seed)==count,'Original source inventory changed during prepare')
    dump(root/'prepare_completed.json',dict(package_sha256=expected,owner_sha256=sha(root/'owner.json'),
        study_json_sha256=sha(root/'study.json'),source_study_sha256=sha(source/'study.json'),
        original_source_files_verified=count,source_files_written=False,completed_unix_time=time.time()))
    verify_root(root,expected);print('DATE_STEPMATCHED_PREPARE_COMPLETE',seed,flush=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',required=True,type=int);ap.add_argument('--gpu',required=True)
    ap.add_argument('--study',required=True,type=Path);ap.add_argument('--source-study',required=True,type=Path)
    ap.add_argument('--expected-package-sha256',required=True);ap.add_argument('--execute',action='store_true')
    a=ap.parse_args();verify_package(a.expected_package_sha256);root=assignment(a.seed,a.gpu,a.study)
    require(str(a.source_study)==protocol()['source_studies'][str(a.seed)],'Unexpected source')
    print('PREPARE_PLAN',a.seed,str(root),'new manual control; original accepted sources read only',flush=True)
    if a.execute:
        require(sys.platform=='linux','Linux server only');prepare(root,a.source_study,a.seed,a.gpu,a.expected_package_sha256)
if __name__=='__main__':main()
