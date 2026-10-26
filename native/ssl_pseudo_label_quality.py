#!/usr/bin/env python3
"""
Experiment 3: pseudo-label quality evaluation (Table_7).

Run teacher on "unlabeled but GT-annotated" images, keep confidence,
match to GT at tau in {0.3, 0.5, 0.7} with IoU >= 0.5:
  precision = TP / (TP + FP)
  recall    = TP / (TP + FN)

Usage:
  python ssl_pseudo_label_quality.py \
    --weights runs/train/SSL-teacher-v3/weights/best.pt \
    --images-dir data/datasets/ssl/unlabeled/images \
    --gt-dir data/datasets/labels/train \
    --device 2
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.datasets import LoadImages
from utils.general import non_max_suppression, scale_coords, box_iou, check_img_size
from utils.torch_utils import select_device

THRESHOLDS = [0.3, 0.5, 0.7]
IOU_MATCH = 0.5


def load_model(weights_path, device):
    import torch.nn as nn
    weights_path = Path(weights_path)
    assert weights_path.exists(), 'Weights not found: {}'.format(weights_path)
    ckpt = torch.load(str(weights_path), map_location=device)
    model = ckpt['ema' if ckpt.get('ema') else 'model'].float()
    model.eval()
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True
        elif hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None
    return model


def read_gt_xyxy(label_path, img_w, img_h):
    boxes = []
    if not label_path.exists():
        return torch.zeros((0, 4), dtype=torch.float32)
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, xc, yc, w, h = (float(x) for x in parts[:5])
        x1 = (xc - w / 2) * img_w
        y1 = (yc - h / 2) * img_h
        x2 = (xc + w / 2) * img_w
        y2 = (yc + h / 2) * img_h
        boxes.append([x1, y1, x2, y2])
    return torch.tensor(boxes, dtype=torch.float32)


def greedy_match(pred_xyxy, pred_conf, gt_xyxy, tau, iou_match=IOU_MATCH):
    keep = pred_conf >= tau
    pred_xyxy = pred_xyxy[keep]
    pred_conf = pred_conf[keep]

    n_pred = pred_xyxy.shape[0]
    n_gt = gt_xyxy.shape[0]

    if n_gt == 0:
        return 0, n_pred, 0
    if n_pred == 0:
        return 0, 0, n_gt

    order = torch.argsort(pred_conf, descending=True)
    pred_xyxy = pred_xyxy[order]

    ious = box_iou(pred_xyxy, gt_xyxy)
    matched_gt = set()
    tp = fp = 0
    for i in range(n_pred):
        best_iou, best_j = ious[i].max(0)
        if best_iou.item() >= iou_match and best_j.item() not in matched_gt:
            tp += 1
            matched_gt.add(best_j.item())
        else:
            fp += 1
    fn = n_gt - len(matched_gt)
    return tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--images-dir', default='data/datasets/ssl/unlabeled/images')
    ap.add_argument('--gt-dir', default='data/datasets/labels/train')
    ap.add_argument('--conf-thres', type=float, default=0.001)
    ap.add_argument('--iou-thres', type=float, default=0.45)
    ap.add_argument('--img-size', type=int, default=640)
    ap.add_argument('--device', default='0')
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    gt_dir = Path(args.gt_dir)
    device = select_device(args.device)
    model = load_model(args.weights, device)
    stride = int(model.stride.max())
    imgsz = check_img_size(args.img_size, s=stride)

    dataset = LoadImages(str(images_dir), img_size=imgsz, stride=stride)
    total = len(dataset)

    stats = {t: {'tp': 0, 'fp': 0, 'fn': 0} for t in THRESHOLDS}

    for idx, (path, img, im0s, _) in enumerate(dataset):
        img = torch.from_numpy(img).to(device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            pred = model(img)[0]
        pred = non_max_suppression(pred, args.conf_thres, args.iou_thres)

        det = pred[0]
        p = Path(path)
        img_h, img_w = im0s.shape[:2]

        gt_xyxy = read_gt_xyxy(gt_dir / (p.stem + '.txt'), img_w, img_h).to(device)

        if det is not None and len(det):
            det = det.clone()
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
            pred_xyxy = det[:, :4]
            pred_conf = det[:, 4]
        else:
            pred_xyxy = torch.zeros((0, 4), device=device)
            pred_conf = torch.zeros((0,), device=device)

        for tau in THRESHOLDS:
            tp, fp, fn = greedy_match(pred_xyxy, pred_conf, gt_xyxy, tau)
            stats[tau]['tp'] += tp
            stats[tau]['fp'] += fp
            stats[tau]['fn'] += fn

        if (idx + 1) % 100 == 0:
            print('  Progress: {}/{}'.format(idx + 1, total))

    print('Evaluated {} images (GT dir: {})'.format(total, gt_dir))
    print('{:>6} {:>10} {:>10} {:>8} {:>8} {:>8}'.format('tau', 'precision', 'recall', 'TP', 'FP', 'FN'))
    for tau in THRESHOLDS:
        tp = stats[tau]['tp']
        fp = stats[tau]['fp']
        fn = stats[tau]['fn']
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        print('{:>6.2f} {:>10.4f} {:>10.4f} {:>8d} {:>8d} {:>8d}'.format(tau, prec, rec, tp, fp, fn))


if __name__ == '__main__':
    main()
