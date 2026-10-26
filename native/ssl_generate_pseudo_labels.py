#!/usr/bin/env python3
"""
Step 2: Generate pseudo-labels for unlabeled images using the trained teacher model.
Uses detect.py logic to run inference and save YOLO-format labels.

Usage: python ssl_generate_pseudo_labels.py --weights runs/train/SSL-teacher/weights/best.pt --device 0
"""
import argparse
import sys
from pathlib import Path

import torch
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device


def load_model(weights_path, device):
    """Load a trained YOLOv7 model, bypassing attempt_download's lowercase bug."""
    import torch.nn as nn
    weights_path = Path(weights_path)
    assert weights_path.exists(), f'Weights not found: {weights_path}'
    ckpt = torch.load(str(weights_path), map_location=device)
    model = ckpt['ema' if ckpt.get('ema') else 'model'].float()
    model.eval()
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True
        elif hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Teacher model weights')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', type=str, default='0', help='GPU device')
    parser.add_argument('--img-size', type=int, default=640, help='Inference image size')
    parser.add_argument('--unlabeled-dir', type=str, default=None, help='Unlabeled images dir')
    parser.add_argument('--output-dir', type=str, default=None, help='Output pseudo-label dir')
    opt = parser.parse_args()

    ssl_dir = ROOT / 'data' / 'datasets' / 'ssl'

    unlabeled_img = Path(opt.unlabeled_dir) if opt.unlabeled_dir else ssl_dir / 'unlabeled' / 'images'
    pseudo_lbl_dir = Path(opt.output_dir) if opt.output_dir else ssl_dir / 'pseudo_labels'

    pseudo_lbl_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(opt.device)

    # Load model
    model = load_model(opt.weights, device)
    stride = int(model.stride.max())
    imgsz = check_img_size(opt.img_size, s=stride)

    # Load images
    dataset = LoadImages(str(unlabeled_img), img_size=imgsz, stride=stride)

    total = len(dataset)
    count = 0
    empty_count = 0

    for path, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            pred = model(img)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres)

        p = Path(path)
        txt_path = pseudo_lbl_dir / (p.stem + '.txt')

        det = pred[0]
        if det is not None and len(det):
            gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

            with open(txt_path, 'w') as f:
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    f.write(f'{int(cls)} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}\n')
        else:
            # No detection: write empty label file
            open(txt_path, 'w').close()
            empty_count += 1

        count += 1
        if count % 100 == 0:
            print(f'  Progress: {count}/{total}')

    print(f'Done. Generated pseudo-labels: {count} images, {empty_count} with no detections.')
    print(f'Pseudo-labels saved to: {pseudo_lbl_dir}')


if __name__ == '__main__':
    main()
