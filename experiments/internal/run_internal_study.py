"""Sequential, resumable experiment queue. Planning is default; --execute enables training."""
import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from study_common import copy_checked, dump, image_alias, match, normalized_to_xyxy, parse_label, read, sha, verify_frozen

HERE = Path(__file__).resolve().parent


class Study:
    def __init__(self, root, args):
        self.root, self.args = root, args
        self.info = verify_frozen(root)
        self.rows = read(root/'manifest.json')
        self.index = {r['id']: r for r in self.rows}
        self.budgets = read(root/'label_budgets.json')

    def task(self, identifier, config):
        folder = self.root/'jobs'/identifier
        config = dict(config, study=str(self.root))
        config['study_protocol_sha256'] = sha(self.root/'study.json')
        # Bind the runner version to resumed jobs.
        config['worker_sha256'] = sha(HERE/'study_worker.py')
        config['runner_sha256'] = sha(__file__)
        record_file = folder/'completed.json'
        if record_file.is_file():
            record = read(record_file)
            if record['input'] != config:
                raise ValueError('Completed job protocol changed; create a NEW study: '+identifier)
            for p, digest in record['output_hashes'].items():
                if not (folder/p).is_file() or sha(folder/p) != digest:
                    raise ValueError('Completed job artifact changed: '+identifier+'/'+p)
            return Path(record['work'])
        folder.mkdir(parents=True, exist_ok=True)
        attempts = sorted(folder.glob('attempt_*'))
        if attempts and not self.args.retry_failed:
            raise ValueError('Incomplete/failed job preserved. Inspect its log, then use --retry-failed: '+identifier)
        work = folder/('attempt_'+str(len(attempts)+1).zfill(2))
        work.mkdir()
        full = dict(config, work=str(work))
        dump(work/'worker_config.json', full)
        python = self.args.v8_python if config['backend'] == 'v8' else sys.executable
        if not python:
            raise ValueError('Provide --v8-python pointing to an existing YOLOv8 environment.')
        env = os.environ.copy()
        for name in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK']:
            env.pop(name, None)
        env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8', WANDB_MODE='disabled',
                   MPLCONFIGDIR=str(work/'plot_config'), YOLO_CONFIG_DIR=str(work/'ultralytics_settings'),
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        env['YOLO_AUTOINSTALL'] = 'false'
        if config['mode'] == 'preflight':
            env['CUDA_VISIBLE_DEVICES'] = ''
        command = [str(python), str(HERE/'study_worker.py'), '--config', str(work/'worker_config.json')]
        dump(work/'started.json', {'command': command, 'unix_time': time.time(), 'input': config})
        print('START', identifier, 'log:', work/'output.log', flush=True)
        with (work/'output.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
        dump(work/'exit_status.json', {'returncode': result.returncode, 'unix_time': time.time()})
        if result.returncode:
            raise RuntimeError('Job failed; queue stopped. No later experiment launched: '+str(work/'output.log'))
        required = {'preflight': ['preflight.json'], 'train': ['train/weights/last.pt', 'training_settings.json', 'seed_applied.json'],
                    'infer': ['predictions.json', 'inference_metadata.json']}[config['mode']]
        for relative in required:
            if not (work/relative).is_file():
                raise ValueError('Worker exited without required output: '+relative)
        hashes = {p.relative_to(folder).as_posix(): sha(p) for p in [work/r for r in required]}
        dump(record_file, {'input': config, 'work': str(work), 'output_hashes': hashes})
        print('DONE', identifier, flush=True)
        return work

    def require_preflight(self, backend):
        if not (self.root/'jobs'/('preflight_'+backend)/'completed.json').is_file():
            raise ValueError('Run CPU preflight for '+backend+' before launching GPU experiments.')

    def make_view(self, identifier, ids, label_text=None):
        """Unique filename per repeated sample; hidden GT never appears in unlabeled loaders."""
        folder = self.root/'views'/identifier
        signature = {'ids': ids, 'label_text': label_text}
        if (folder/'view.json').is_file():
            saved = read(folder/'view.json')
            if saved['input'] != signature:
                raise ValueError('Existing data view protocol differs: '+identifier)
            for relative, digest in saved['hashes'].items():
                if not (folder/relative).is_file() or sha(folder/relative) != digest:
                    raise ValueError('Data view changed: '+identifier+'/'+relative)
            return folder/'data.yaml'
        if folder.exists():
            raise ValueError('Incomplete data view preserved; use a new study or inspect: '+identifier)
        folder.mkdir(parents=True)
        hashes = {}
        for position, image_id in enumerate(ids):
            row = self.index[image_id]
            name = str(position).zfill(6)+'_'+row['staged_name']
            relative = 'images/train/'+name
            image_alias(self.root/row['image_relative'], folder/relative)
            hashes[relative] = row['image_sha256']
            label_relative = 'labels/train/'+Path(name).stem+'.txt'
            label = folder/label_relative
            label.parent.mkdir(parents=True, exist_ok=True)
            if label_text is not None and str(image_id) in label_text:
                label.write_text(label_text[str(image_id)], encoding='utf-8')
            else:
                copy_checked(self.root/row['gt_relative'], label, row['label_sha256'])
            parse_label(label)
            hashes[label_relative] = sha(label)
        # Validation images and GT labels are copied/aliased into the view for native training validation.
        for row in self.rows:
            if row['split'] != 'val':
                continue
            relative = 'images/val/'+row['staged_name']
            image_alias(self.root/row['image_relative'], folder/relative)
            hashes[relative] = row['image_sha256']
            label_relative = 'labels/val/'+Path(row['staged_name']).stem+'.txt'
            copy_checked(self.root/row['gt_relative'], folder/label_relative, row['label_sha256'])
            hashes[label_relative] = row['label_sha256']
        data = {'train': str(folder/'images/train'), 'val': str(folder/'images/val'), 'nc': 1, 'names': ['fish']}
        dump(folder/'data.yaml', data)
        hashes['data.yaml'] = sha(folder/'data.yaml')
        dump(folder/'view.json', {'input': signature, 'hashes': hashes, 'n_training_samples': len(ids)})
        return folder/'data.yaml'

    def training(self, architecture, budget, seed, method='supervised', epochs=300):
        backend = 'v8' if architecture == 'v8n' else 'v7'
        self.require_preflight(backend)
        if epochs == 300:
            smoke_id = 'smoke_{}_b{}_s{}_supervised'.format(architecture, self.info['budgets'][0], self.info['training_seeds'][0])
            if not (self.root/'jobs'/smoke_id/'completed.json').is_file():
                raise ValueError('Run the 2-epoch GPU smoke test for '+architecture+' first.')
        prefix = '{}_b{}_s{}_{}'.format(architecture, budget, seed, method)
        if epochs != 300:
            prefix = 'smoke_'+prefix
        labeled = self.budgets[str(budget)]['labeled_ids']
        texts = None
        ids = labeled
        if method in ['ssl', 'stepmatched']:
            pseudo = self.ensure_pseudo(budget, seed)
            if method == 'ssl':
                ids = labeled+[int(x) for x in sorted(pseudo['labels'], key=int)]
                texts = pseudo['labels']
            else:
                # Same number of samples/batches as SSL, balanced repetition of manual images only.
                target = len(labeled)+len(pseudo['labels'])
                ids = (labeled*((target+len(labeled)-1)//len(labeled)))[:target]
        data = self.make_view(prefix, ids, texts)
        work = self.task(prefix, {'mode': 'train', 'backend': backend, 'architecture': architecture,
                   'budget': budget, 'seed': seed, 'method': method, 'epochs': epochs,
                   'data': str(data), 'data_view_sha256': sha(data.parent/'view.json'),
                   'batch': 4 if architecture == 'teacher' else 8,
                   'workers': self.args.workers, 'device': self.args.device,
                   'dataset_manifest_sha256': sha(self.root/'manifest.json'), 'n_training_samples': len(ids)})
        weight = work/'train/weights/last.pt'
        if epochs == 300:
            for split in ['val', 'test']:
                self.evaluate(prefix, backend, weight, split, 640)
        return weight

    def ensure_pseudo(self, budget, seed):
        identifier = 'teacher_b{}_s{}_pseudo'.format(budget, seed)
        record_path = self.root/'pseudo'/identifier/'pseudo_manifest.json'
        teacher = self.training('teacher', budget, seed)
        unlabeled = self.budgets[str(budget)]['unlabeled_ids']
        raw = self.task(identifier, {'mode': 'infer', 'backend': 'v7', 'ids': unlabeled,
                        'weight': str(teacher), 'weight_sha256': sha(teacher), 'device': self.args.device,
                        'imgsz': 640, 'nms_iou': .45, 'dataset_manifest_sha256': sha(self.root/'manifest.json')})
        input_signature = {'teacher_sha256': sha(teacher), 'predictions_sha256': sha(raw/'predictions.json'),
                           'threshold': .5, 'nms_iou': .45, 'unlabeled_ids': unlabeled}
        if record_path.is_file():
            saved = read(record_path)
            if saved['input'] != input_signature:
                raise ValueError('Pseudo-label input changed.')
            if not saved['labels']:
                raise ValueError('No pseudo labels at predeclared .5. Do not silently change threshold or train a mislabeled SSL job.')
            return saved
        if record_path.parent.exists():
            raise ValueError('Incomplete pseudo-label output preserved: '+str(record_path.parent))
        record_path.parent.mkdir(parents=True)
        grouped = defaultdict(list)
        for p in read(raw/'predictions.json'):
            grouped[p['image_id']].append(p)
        labels, empty, quality = {}, [], {str(t): {'tp': 0, 'fp': 0, 'fn': 0} for t in [.3, .5, .7]}
        for image_id in unlabeled:
            row = self.index[image_id]
            text = []
            predictions = grouped[image_id]
            for p in sorted(predictions, key=lambda p: -p['score']):
                if p['score'] < .5:
                    continue
                x, y, w, h = p['bbox']
                text.append('0 {:.8f} {:.8f} {:.8f} {:.8f}\n'.format((x+w/2)/row['width'], (y+h/2)/row['height'],
                              w/row['width'], h/row['height']))
            if text:
                labels[str(image_id)] = ''.join(text)
            else:
                empty.append(image_id)
            truth = [normalized_to_xyxy(b, row['width'], row['height']) for b in parse_label(self.root/row['gt_relative'])]
            diagnostic_pred = [dict(p, xyxy=[p['bbox'][0], p['bbox'][1], p['bbox'][0]+p['bbox'][2], p['bbox'][1]+p['bbox'][3]]) for p in predictions]
            for t in [.3, .5, .7]:
                tp, fp, fn = match(diagnostic_pred, truth, t)
                for key, value in [('tp', tp), ('fp', fp), ('fn', fn)]:
                    quality[str(t)][key] += value
        for value in quality.values():
            tp, fp, fn = value['tp'], value['fp'], value['fn']
            value['precision'] = tp/(tp+fp) if tp+fp else None
            value['recall'] = tp/(tp+fn) if tp+fn else None
        saved = {'input': input_signature, 'labels': labels, 'empty_images_excluded': empty,
                 'n_pseudo_images': len(labels), 'n_pseudo_boxes': sum(len(t.splitlines()) for t in labels.values()),
                 'n_labeled_images': budget, 'n_combined': budget+len(labels),
                 'hidden_gt_diagnostic_only': quality,
                 'threshold_policy': 'Fixed .5 before execution; hidden training GT/test never selects threshold',
                 'matching': 'Confidence-ordered, best still-unmatched GT, IoU>=.5; no per-image maxDets cap for quality counts',
                 'teacher_updated_during_student_training': False}
        dump(record_path, saved)
        if not labels:
            raise ValueError('No pseudo labels at predeclared .5. Preserved diagnostic; do not silently lower threshold.')
        return saved

    def evaluate(self, identifier, backend, weight, split, imgsz):
        task_id = identifier+'_eval_'+split+'_'+str(imgsz)
        ids = [r['id'] for r in self.rows if r['split'] == split]
        raw = self.task(task_id, {'mode': 'infer', 'backend': backend, 'ids': ids,
                        'weight': str(weight), 'weight_sha256': sha(weight), 'device': self.args.device,
                        'imgsz': imgsz, 'nms_iou': .65, 'dataset_manifest_sha256': sha(self.root/'manifest.json')})
        metrics = raw/'coco_metrics.json'
        signature = {'gt_sha256': sha(self.root/('gt_'+split+'.json')), 'predictions_sha256': sha(raw/'predictions.json'),
                     'worker_sha256': sha(HERE/'study_worker.py'), 'evaluator_runner_sha256': sha(__file__)}
        if metrics.is_file():
            if read(metrics)['input'] != signature:
                raise ValueError('COCO evaluation artifacts changed.')
            return
        import numpy as np
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        gt = COCO(str(self.root/('gt_'+split+'.json')))
        predictions = read(raw/'predictions.json')
        if predictions:
            dt = gt.loadRes(predictions)
        else:
            dt = COCO()
            dt.dataset = {'images': gt.dataset['images'], 'categories': gt.dataset['categories'], 'annotations': []}
            dt.createIndex()
        ev = COCOeval(gt, dt, 'bbox')
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate(); ev.accumulate(); ev.summarize()
        keys = ['AP50_95', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
                'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large']
        scores = {k: float(v) if v >= 0 else None for k, v in zip(keys, ev.stats)}
        metric_record = {'input': signature, 'metrics': scores, 'split': split, 'imgsz': imgsz,
             'independent_generalization_test': False, 'area_definition': 'Original-image pixels; COCO 32^2/96^2; -1 -> null',
             'checkpoint_sha256': sha(weight), 'main_checkpoint_rule': 'Fixed 300 epochs, last.pt',
             'precision': 'FP32, square letterbox, confidence floor .001, NMS IoU .65, model max_det300, COCO maxDets100'}
        np.savez_compressed(raw/'coco_precision_recall.npz', precision=ev.eval['precision'], recall=ev.eval['recall'],
                            iou_thresholds=ev.params.iouThrs, recall_thresholds=ev.params.recThrs)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        curve = ev.eval['precision'][0, :, 0, 0, 2]
        valid = curve >= 0
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ev.params.recThrs[valid], curve[valid], label='COCO AP50 {:.3f}'.format(scores['AP50'] or 0))
        ax.set(xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1), title='Internal '+split+' | '+identifier)
        ax.legend(); fig.tight_layout(); fig.savefig(raw/'coco_PR50.png', dpi=150); plt.close(fig)
        dump(metrics, metric_record)
        completed = self.root/'jobs'/task_id/'completed.json'
        record = read(completed)
        for file in [metrics, raw/'coco_precision_recall.npz', raw/'coco_PR50.png']:
            record['output_hashes'][file.relative_to(completed.parent).as_posix()] = sha(file)
        dump(completed, record)

    def summaries(self):
        records = []
        for complete in sorted((self.root/'jobs').glob('*/completed.json')):
            record = read(complete)
            metrics = Path(record['work'])/'coco_metrics.json'
            if not metrics.is_file():
                continue
            job_id = complete.parent.name
            scores = read(metrics)
            model_id = job_id.rsplit('_eval_', 1)[0]
            train_record = self.root/'jobs'/model_id/'completed.json'
            if not train_record.is_file():
                continue
            settings = read(train_record)['input']
            records.append(dict(architecture=settings['architecture'], budget=settings['budget'], seed=settings['seed'],
                                method=settings['method'], split=scores['split'], imgsz=scores['imgsz'],
                                checkpoint_sha256=scores['checkpoint_sha256'], **scores['metrics']))
        dump(self.root/'summary_individual.json', records)
        grouped = defaultdict(list)
        for row in records:
            key = tuple(row[k] for k in ['architecture', 'budget', 'method', 'split', 'imgsz'])
            grouped[key].append(row)
        summary = []
        for key, rows in sorted(grouped.items()):
            seeds = sorted(r['seed'] for r in rows)
            item = dict(zip(['architecture', 'budget', 'method', 'split', 'imgsz'], key))
            item.update(training_seeds=seeds, n_independent_training_runs=len(seeds),
                        three_seed_complete=(seeds == sorted(self.info['training_seeds'])))
            for metric in ['AP50', 'AP50_95', 'AP_small', 'AP_medium', 'AP_large']:
                values = [r[metric]*100 for r in rows if r[metric] is not None]
                item[metric+'_mean_percent'] = statistics.mean(values) if values else None
                # Do not present one run or two runs as a completed 3-seed result.
                item[metric+'_sample_sd_pp'] = statistics.stdev(values) if len(values) >= 3 and item['three_seed_complete'] else None
                item[metric+'_valid_runs'] = len(values)
            summary.append(item)
        dump(self.root/'summary_mean_sd.json', summary)
        lines = ['# 去重内部对照结果', '', '序列独立性未验证。AP单位为百分数，标准差单位为百分点；未满三种子不填三次重复结论。', '',
                 '|模型|预算|方法|集合|输入|训练次数|AP50均值±样本标准差|AP50:95均值±样本标准差|',
                 '|---|---:|---|---|---:|---:|---:|---:|']
        def fmt(item, key):
            mean, sd = item[key+'_mean_percent'], item[key+'_sample_sd_pp']
            return '--' if mean is None else '{:.2f}'.format(mean)+(' ± {:.2f}'.format(sd) if sd is not None else '（未满三次）')
        for row in summary:
            lines.append('|{}|{}|{}|{}|{}|{}|{}|{}|'.format(row['architecture'], row['budget'], row['method'],
                          row['split'], row['imgsz'], row['n_independent_training_runs'], fmt(row, 'AP50'), fmt(row, 'AP50_95')))
        (self.root/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')

    def evidence_zip(self):
        destination = self.root.parent/(self.root.name+'_return.zip')
        if destination.exists():
            raise ValueError('Return ZIP exists; rename/archive it before making another version.')
        with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as z:
            for p in sorted(self.root.rglob('*')):
                if not p.is_file():
                    continue
                relative = p.relative_to(self.root)
                if relative.parts[0] in ['dataset', 'ground_truth', 'views', 'initial']:
                    continue
                if p.suffix.lower() in ['.pt', '.cache', '.pyc'] or any(x in relative.parts for x in ['plot_config', 'ultralytics_settings', '__pycache__']):
                    continue
                if p.suffix.lower() not in ['.py', '.yaml', '.json', '.log', '.txt', '.md', '.csv', '.png', '.npz']:
                    continue
                z.write(p, relative.as_posix())
        print('RETURN', destination, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', required=True, type=Path)
    ap.add_argument('--phase', required=True, choices=['preflight', 'smoke', 'pilot', 'core', 'v8', 'matched', 'resolutions', 'return'])
    ap.add_argument('--execute', action='store_true', help='Enable training/inference; without this flag only print the queue.')
    ap.add_argument('--v8-python', help='Full Python path in an existing isolated Ultralytics environment.')
    ap.add_argument('--include-v8', action='store_true', help='Include v8 in preflight/smoke/pilot/matched.')
    ap.add_argument('--device', default='0')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--retry-failed', action='store_true', help='Preserve failed attempts and use a new attempt directory.')
    a = ap.parse_args()
    if a.workers < 0 or (a.device != 'cpu' and not a.device.isdigit()):
        ap.error('Use nonnegative workers and a single GPU index (or cpu). DDP/multiple GPUs are not part of this protocol.')
    root = a.study.resolve()
    info = read(root/'study.json')
    first_budget, first_seed = info['budgets'][0], info['training_seeds'][0]
    pairs = [(first_budget, first_seed)] if a.phase in ['smoke', 'pilot'] else [(b, s) for b in info['budgets'] for s in info['training_seeds']]
    plan = []
    if a.phase == 'preflight':
        plan = [{'mode': 'preflight', 'backend': b} for b in ['v7']+(['v8'] if a.include_v8 else [])]
    elif a.phase in ['smoke', 'pilot', 'core', 'v8', 'matched']:
        for budget, seed in pairs:
            if a.phase in ['pilot', 'core']:
                plan.append({'architecture': 'teacher', 'budget': budget, 'seed': seed, 'method': 'supervised', 'epochs': 300})
            architectures = ['v8n'] if a.phase == 'v8' else ['compact', 'wide']+(['v8n'] if a.include_v8 else [])
            if a.phase == 'smoke':
                architectures = ['teacher']+architectures
            methods = ['stepmatched'] if a.phase == 'matched' else ['supervised'] if a.phase == 'smoke' else ['supervised', 'ssl']
            for arch in architectures:
                for method in methods:
                    plan.append({'architecture': arch, 'budget': budget, 'seed': seed, 'method': method,
                                 'epochs': 2 if a.phase == 'smoke' else 300})
    print(json.dumps({'phase': a.phase, 'planned_training_jobs': len(plan), 'queue': plan,
          'independent_generalization_test': False, 'note': 'Existing completed jobs are verified and reused; only 300-epoch jobs enter main comparisons.'}, ensure_ascii=False, indent=2))
    if a.phase not in ['preflight', 'return'] and not a.execute:
        print('Planning only. Add --execute to use the GPU.'); return
    study = Study(root, a)
    if a.phase == 'preflight':
        for config in plan:
            study.task('preflight_'+config['backend'], config)
    elif a.phase == 'return':
        study.summaries(); study.evidence_zip()
    elif a.phase == 'resolutions':
        # Predeclared diagnostic evaluation of the SAME checkpoints; no retraining or checkpoint selection on test.
        for file in sorted((root/'jobs').glob('*/completed.json')):
            record = read(file)
            settings = record['input']
            if settings['mode'] != 'train' or settings['epochs'] != 300 or settings['architecture'] == 'teacher':
                continue
            for size in [960, 1280]:
                study.evaluate(file.parent.name, settings['backend'], Path(record['work'])/'train/weights/last.pt', 'test', size)
        study.summaries()
    else:
        for config in plan:
            study.training(**config)
            study.summaries()
        # Recheck frozen sources/GT/images after the queue. No original-project mutation is needed.
        verify_frozen(root)
        dump(root/('postcheck_'+a.phase+'.json'), {'frozen_snapshot_unchanged': True, 'unix_time': time.time()})


if __name__ == '__main__':
    main()
