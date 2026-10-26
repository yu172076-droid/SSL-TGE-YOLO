"""Real CPU model and checkpoint compatibility checks, executed on the server."""
from pathlib import Path
from control_common import read,dump,sha,require,protocol

COUNTS={'teacher':37196556,'compact':4958807}
def describe_model(model,architecture,root):
    require(architecture in COUNTS,'Unplanned architecture')
    cfg=read(Path(root)/'configs'/(architecture+'.yaml'))
    count=sum(p.numel() for p in model.parameters())
    require(count==COUNTS[architecture] and model.yaml.get('nc')==1,'Native architecture count/class differs')
    for key in ('backbone','head'):require(model.yaml[key]==cfg[key],'Model graph differs from frozen config')
    stride=[int(x) for x in model.stride.tolist()]
    require(stride==[8,16,32],'Stride differs')
    return dict(architecture=architecture,native_parameters=count,nc=1,stride=stride,
                config_sha256=sha(Path(root)/'configs'/(architecture+'.yaml')),
                graph_matches_frozen_config=True,modules=[dict(name=n,type=type(m).__name__) for n,m in model.named_modules()])
def cpu_preflight(config,work,root,legacy):
    torch=legacy.install_v7(root)
    require(not torch.cuda.is_available() and torch.cuda.device_count()==0,'CPU preflight saw CUDA')
    torch.set_num_threads(4);torch.manual_seed(config['seed'])
    from models.yolo import Model
    env=legacy.environment(torch)
    require(all(env['versions'].get(k)==v for k,v in protocol()['runtime_versions'].items()),'CPU runtime versions differ')
    reports=[]
    for arch in ('teacher','compact'):
        model=Model(str(root/'configs'/(arch+'.yaml')),ch=3,nc=1).float().eval()
        report=describe_model(model,arch,root)
        with torch.no_grad():out=model(torch.zeros(1,3,256,256))[0]
        require(list(out.shape)==[1,4032,6] and bool(torch.isfinite(out).all()),'CPU forward shape/nonfinite output')
        report.update(input_shape=[1,3,256,256],prediction_shape=list(out.shape),all_finite=True)
        reports.append(report);del model,out
    initial=torch.load(str(root/'initial/yolov7.pt'),map_location='cpu')
    model=initial.get('model');require(model is not None and model.yaml.get('nc')==80,'Initial checkpoint is not the pinned 80-class model')
    import json
    cfg=read(root/'configs/teacher.yaml')
    for key in ('backbone','head'):
        actual=json.loads(json.dumps(model.yaml[key]));target=json.loads(json.dumps(cfg[key]))
        if key=='head' and actual[-1][2] in ('Detect','IDetect'):actual[-1][2]=target[-1][2]
        require(actual==target,'Initial full-YOLOv7 graph incompatible')
    import pycocotools.cocoeval
    import torch.utils.tensorboard
    dump(work/'preflight.json',dict(passed=True,environment=env,cpu_only=True,models=reports,
         initial_checkpoint_sha256=sha(root/'initial/yolov7.pt'),initial_nc=80,
         initial_architecture_compatible=True,official_pretraining_origin_independently_certified=False,
         date_manifest_sha256=sha(root/'manifest.json'),no_old_fish_teacher_reused=True))
