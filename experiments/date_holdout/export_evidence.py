"""Export new evidence, preserving server weights and all original studies."""
import argparse
import shutil
import time
import zipfile
from control_common import *

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--study',type=Path,required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--gpu',required=True)
    ap.add_argument('--phase',choices=('formal',),required=True)
    ap.add_argument('--expected-package-sha256',required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--allow-incomplete',action='store_true')
    a=ap.parse_args();verify_package(a.expected_package_sha256);root=assignment(a.seed,a.gpu,a.study)
    verify_root(root,a.expected_package_sha256)
    shell=root/(a.phase+'_shell_exit.json');phase=root/'formal_complete.json'
    complete=phase.is_file() and shell.is_file() and read(shell)['python_exit']==read(shell)['tee_exit']==0
    require(complete or a.allow_incomplete,'Incomplete phase: preserve failure; explicit --allow-incomplete for diagnostic export')
    output=a.output.absolute();require(output==output.resolve() and output.is_dir(),'Output must be an existing real directory')
    disjoint(output,root);[disjoint(output,p) for p in protocol()['source_studies'].values()];disjoint(output,PACKAGE)
    folder=output/('dayholdout_stepmatched_{}_s{}_{}'.format(a.phase,a.seed,time.time_ns()));folder.mkdir(exist_ok=False)
    payload=[];omitted=[]
    for path in sorted(root.rglob('*')):
        require(not path.is_symlink(),'Export symlink refused')
        if not path.is_file():continue
        rel=path.relative_to(root).as_posix();digest=sha(path)
        item=dict(path=rel,bytes=path.stat().st_size,sha256=digest)
        if rel.startswith('dataset/images/') or rel.startswith('views/') and '/images/' in rel or path.suffix in ('.pt','.cache','.npy') or path.name.endswith('.lock'):
            item['reason']='Server-only image/checkpoint/cache/lock; hash only'
            omitted.append(item);continue
        destination=folder/rel;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination);require(sha(destination)==digest,'Export copy differs');payload.append(item)
    manifest=dict(schema='dateholdout_stepmatched_evidence_export_v1',study=str(root),seed=a.seed,gpu=a.gpu,phase=a.phase,
        package_sha256=a.expected_package_sha256,phase_complete=complete,
        files=payload,omitted_server_only=omitted,originals_deleted=False,
        actual_model_PT_loaded_by_main_window=False)
    dump(folder/'export_manifest.json',manifest)
    (folder/'export_manifest.sha256').write_text(sha(folder/'export_manifest.json')+'  export_manifest.json\n',encoding='ascii')
    for row in payload:
        require(sha(safe(root,row['path']))==row['sha256'],'Source changed during export')
    archive=folder.with_suffix('.zip')
    with zipfile.ZipFile(str(archive),'x',compression=zipfile.ZIP_DEFLATED) as z:
        for file in sorted(folder.rglob('*')):
            if file.is_file():z.write(str(file),folder.name+'/'+file.relative_to(folder).as_posix())
    print('EXPORT',folder,'payload',len(payload),'server-only',len(omitted),flush=True)
    print('ZIP',archive,'SHA256',sha(archive),flush=True)
    print('MANIFEST_SHA256',sha(folder/'export_manifest.json'),flush=True)
if __name__=='__main__':main()
