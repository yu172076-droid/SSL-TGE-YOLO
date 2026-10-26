"""Fresh-process training/inference worker. GPU branches require server smoke validation."""
import argparse
import importlib
import importlib.metadata
import inspect
import json
import os
import sys
import time
import random
import math
from pathlib import Path
from types import SimpleNamespace
from study_common import dump, read, sha


def install_v7(root, weight=None):
    sys.path.insert(0, str(root/'source'))
    # The project handles ImportError. Disable external experiment tracking explicitly.
    sys.modules['wandb'] = None
    import torch
    import models.experimental as experimental
    import utils.google_utils as google_utils
    original_load = torch.load

    def trusted_load(*args, **kwargs):
        if args and isinstance(args[0], (str, Path)):
            path = Path(args[0]).resolve()
            if root not in path.parents or not path.is_file():
                raise ValueError('Checkpoint must be an existing trusted study file: '+str(path))
            if 'weights_only' in inspect.signature(original_load).parameters:
                kwargs.setdefault('weights_only', False)
        return original_load(*args, **kwargs)

    torch.load = trusted_load

    def existing_only(path, *args, **kwargs):
        path = Path(path).resolve()
        if not path.is_file() or root not in path.parents:
            raise FileNotFoundError('Downloads disabled; missing staged study checkpoint: '+str(path))

    experimental.attempt_download = existing_only
    google_utils.attempt_download = existing_only
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def environment(torch):
    versions = {}
    for name in ['torch', 'torchvision', 'numpy', 'opencv-python', 'PyYAML', 'pycocotools', 'ultralytics']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': sys.version, 'versions': versions, 'cuda_version': torch.version.cuda,
            'cuda_available': torch.cuda.is_available(),
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}


def preflight(config):
    root = Path(config['study']).resolve()
    backend = config['backend']
    if backend == 'v7':
        torch = install_v7(root)
        from models.yolo import Model
        report = environment(torch)
        counts = {}
        for name, expected in [('compact', 4958807), ('wide', 6914175), ('teacher', 37196556)]:
            model = Model(str(root/'configs'/(name+'.yaml')), ch=3, nc=1)
            count = sum(p.numel() for p in model.parameters())
            counts[name] = count
            if count != expected:
                raise ValueError('Architecture parameter mismatch: {} {} != {}'.format(name, count, expected))
            model.eval()
            with torch.no_grad():
                model(torch.zeros(1, 3, 64, 64))
            del model
        initial = torch.load(str(root/'initial/yolov7.pt'), map_location='cpu')
        model = initial.get('model')
        if model is None or getattr(model, 'yaml', {}).get('nc') != 80:
            raise ValueError('Teacher initial checkpoint must be an 80-class YOLOv7 model, not a fish-trained checkpoint.')
        # Compare backbone/head layout to the verified full YOLOv7 architecture.
        expected_cfg = read(root/'configs/teacher.yaml')
        for key in ['backbone', 'head']:
            actual = json.loads(json.dumps(model.yaml.get(key)))
            target = json.loads(json.dumps(expected_cfg[key]))
            # Official deploy checkpoints may use Detect instead of IDetect. The train model still uses IDetect.
            if key == 'head' and actual and actual[-1][2] in ['Detect', 'IDetect']:
                actual[-1][2] = target[-1][2]
            if actual != target:
                raise ValueError('Teacher initial architecture differs: '+key)
        import pycocotools.cocoeval
        import torch.utils.tensorboard
        report['unfused_parameters'] = counts
        report['initial_sha256'] = sha(root/'initial/yolov7.pt')
        report['initial_80_class_compatible'] = True
    else:
        import torch
        from ultralytics import YOLO
        model = YOLO('yolov8n.yaml')
        report = environment(torch)
        report['bundled_yolov8n_yaml'] = model.model.yaml
        report['note'] = 'Built from package YAML; no pretrained model download or GPU training.'
    dump(Path(config['work'])/'preflight.json', report)


def train_v7(config):
    root = Path(config['study']).resolve()
    torch = install_v7(root)
    import yaml
    import train
    from utils.general import init_seeds as original_seed, set_logging
    from utils.torch_utils import select_device
    set_logging(-1)
    seed = int(config['seed'])
    seed_calls = []

    def independent_seed(ignored_native_seed):
        original_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        seed_calls.append({'native_requested_seed': ignored_native_seed, 'applied_training_seed': seed})
        dump(Path(config['work'])/'seed_applied.json', seed_calls)

    train.init_seeds = independent_seed
    # FP32 in both frameworks, avoiding an implicit automatic AMP model download in YOLOv8.
    train.amp = SimpleNamespace(GradScaler=lambda **kw: torch.cuda.amp.GradScaler(enabled=False),
                               autocast=lambda **kw: torch.cuda.amp.autocast(enabled=False))
    train.attempt_download = importlib.import_module('utils.google_utils').attempt_download
    weights = str(root/'initial/yolov7.pt') if config['architecture'] == 'teacher' else ''
    if weights and sha(weights) != read(root/'study.json')['initial_checkpoint_sha256']:
        raise ValueError('Teacher initial checkpoint changed.')
    values = dict(weights=weights, cfg=str(root/'configs'/(config['architecture']+'.yaml')),
                  data=config['data'], hyp=str(root/'configs/hyp.yaml'), epochs=config['epochs'],
                  batch_size=config['batch'], total_batch_size=config['batch'], img_size=[640, 640],
                  rect=False, resume=False, nosave=False, notest=False, noautoanchor=False, evolve=False,
                  bucket='', cache_images=False, image_weights=False, device=config['device'], multi_scale=False,
                  single_cls=False, adam=False, sync_bn=False, local_rank=-1, workers=config['workers'],
                  project=config['work'], entity=None, name='train', exist_ok=False, quad=False,
                  linear_lr=False, label_smoothing=0., upload_dataset=False, bbox_interval=-1,
                  save_period=-1, artifact_alias='latest', freeze=[0], v5_metric=False,
                  world_size=1, global_rank=-1, save_dir=str(Path(config['work'])/'train'))
    hyp = yaml.safe_load(Path(values['hyp']).read_text(encoding='utf-8'))
    dump(Path(config['work'])/'training_settings.json', {'native_options': values, 'hyp': hyp,
          'training_seed': seed, 'amp': False, 'precision': 'FP32', 'main_checkpoint': 'last.pt',
          'native_best_rule': '.1*native_AP50+.9*native_AP50_95; secondary only', 'environment': environment(torch)})
    device = select_device(config['device'], batch_size=config['batch'])
    train.train(hyp, argparse.Namespace(**values), device, tb_writer=None)
    if len(seed_calls) != 1 or seed_calls[0]['applied_training_seed'] != seed:
        raise ValueError('Training seed hook did not execute exactly once.')


def train_v8(config):
    import torch
    import numpy as np
    from ultralytics import YOLO
    root = Path(config['study']).resolve()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    # Use a bundled font instead of automatic font download; automatic dependency installation is disabled by the launcher.
    import matplotlib
    from ultralytics.utils import checks
    import ultralytics.data.utils as data_utils
    import ultralytics.utils.plotting as plotting
    font = Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'
    if not font.is_file():
        raise FileNotFoundError('Bundled Matplotlib font is missing.')
    checks.check_font = lambda *a, **kw: font
    data_utils.check_font = checks.check_font
    plotting.check_font = checks.check_font
    # Architecture-specific losses remain native; shared schedule/optimizer/augmentation are explicit.
    values = dict(data=config['data'], epochs=config['epochs'], batch=config['batch'], imgsz=640,
                  device=config['device'], workers=config['workers'], optimizer='SGD', lr0=.01, lrf=.01,
                  momentum=.937, weight_decay=.0005, warmup_epochs=3., warmup_momentum=.8, warmup_bias_lr=.1,
                  hsv_h=.015, hsv_s=.7, hsv_v=.4, degrees=0., translate=.1, scale=.5, shear=0.,
                  perspective=0., flipud=0., fliplr=.5, mosaic=1., mixup=.05, copy_paste=0.,
                  close_mosaic=0, cos_lr=True, nbs=64, seed=config['seed'], deterministic=True,
                  pretrained=False, amp=False, patience=0, cache=False, multi_scale=False, rect=False,
                  project=config['work'], name='train', exist_ok=False, save=True, save_period=-1, plots=True)
    dump(Path(config['work'])/'training_settings.json', {'ultralytics_options': values,
          'initialization': 'scratch from bundled yolov8n.yaml', 'main_checkpoint': 'last.pt',
          'precision': 'FP32', 'loss': 'native YOLOv8 detection loss; not claimed identical to YOLOv7',
          'environment': environment(torch)})
    model = YOLO('yolov8n.yaml')
    model.train(**values)
    # args.yaml and package version are returned; check the trainer actually kept the declared seed.
    if int(model.trainer.args.seed) != int(config['seed']):
        raise ValueError('YOLOv8 trainer seed differs from declared seed.')
    dump(Path(config['work'])/'seed_applied.json', {'applied_training_seed': int(model.trainer.args.seed)})


def infer(config):
    root = Path(config['study']).resolve()
    weight = Path(config['weight']).resolve()
    if sha(weight) != config['weight_sha256']:
        raise ValueError('Checkpoint hash changed.')
    backend = config['backend']
    manifest = read(root/'manifest.json')
    wanted = set(config['ids'])
    rows = [r for r in manifest if r['id'] in wanted]
    if len(rows) != len(wanted):
        raise ValueError('Unknown image IDs.')
    predictions, invalid = [], []
    if backend == 'v7':
        torch = install_v7(root, weight)
        import cv2
        import numpy as np
        from models.experimental import attempt_load
        from utils.datasets import letterbox
        from utils.general import non_max_suppression, scale_coords
        from utils.torch_utils import select_device
        device = select_device(config['device'])
        model = attempt_load(str(weight), map_location=device).float().eval()
        stride = int(model.stride.max())
        for index, row in enumerate(rows):
            original = cv2.imread(str(root/row['image_relative']))
            if original is None:
                raise ValueError('Unreadable frozen image: '+row['image_relative'])
            image = letterbox(original, config['imgsz'], stride=stride, auto=False)[0]
            image = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))
            tensor = torch.from_numpy(image).to(device).float().unsqueeze(0)/255.
            with torch.no_grad():
                output = model(tensor)[0]
                det = non_max_suppression(output, .001, config['nms_iou'])[0]
            if len(det):
                det[:, :4] = scale_coords(tensor.shape[2:], det[:, :4], original.shape)
            collect(row, det.detach().cpu().tolist(), predictions, invalid)
            if (index+1) % 100 == 0:
                print('Inference {}/{}'.format(index+1, len(rows)), flush=True)
    else:
        import torch
        from ultralytics import YOLO
        model = YOLO(str(weight))
        for index, row in enumerate(rows):
            result = model.predict(source=str(root/row['image_relative']), imgsz=config['imgsz'],
                                   conf=.001, iou=config['nms_iou'], max_det=300, device=config['device'],
                                   half=False, rect=False, augment=False, verbose=False)[0]
            det = [list(xyxy)+[float(score), float(cls)] for xyxy, score, cls in
                   zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist())]
            collect(row, det, predictions, invalid)
            if (index+1) % 100 == 0:
                print('Inference {}/{}'.format(index+1, len(rows)), flush=True)
    dump(Path(config['work'])/'predictions.json', predictions)
    dump(Path(config['work'])/'inference_metadata.json', {'checkpoint_sha256': sha(weight), 'settings': config,
          'environment': environment(torch), 'precision': 'FP32', 'square_letterbox': True,
          'invalid_zero_area_predictions_excluded': invalid, 'image_count': len(rows),
          'n_predictions': len(predictions), 'sequence_independence_verified': False})


def collect(row, detections, predictions, invalid):
    # Same clipping and zero-area policy for all backends. NMS max_det is 300 for both.
    for x1, y1, x2, y2, score, cls in detections:
        if not all(math.isfinite(x) for x in [x1, y1, x2, y2, score, cls]) or not 0 <= score <= 1:
            raise ValueError('Non-finite or invalid prediction; evaluation cancelled.')
        if cls != 0:
            raise ValueError('Unexpected class in single-class detector.')
        x1, x2 = max(0., min(row['width'], x1)), max(0., min(row['width'], x2))
        y1, y2 = max(0., min(row['height'], y1)), max(0., min(row['height'], y2))
        if x2 <= x1 or y2 <= y1:
            invalid.append({'image_id': row['id'], 'xyxy': [x1, y1, x2, y2], 'score': score})
            continue
        predictions.append({'image_id': row['id'], 'category_id': 1, 'bbox': [x1, y1, x2-x1, y2-y1], 'score': score})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, type=Path)
    a = ap.parse_args()
    config = read(a.config)
    if config['mode'] == 'preflight':
        preflight(config)
    elif config['mode'] == 'train':
        (train_v7 if config['backend'] == 'v7' else train_v8)(config)
    elif config['mode'] == 'infer':
        infer(config)
    else:
        raise ValueError('Unknown mode.')
