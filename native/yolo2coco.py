#!/usr/bin/env python3
"""
Convert YOLO-format labels to COCO JSON (for torchvision Faster R-CNN).

Usage:
  python yolo2coco.py --images data/datasets/images/train --labels data/datasets/labels/train --out data/datasets/coco/instances_train.json
  python yolo2coco.py --images data/datasets/images/val   --labels data/datasets/labels/val   --out data/datasets/coco/instances_val.json
"""
import argparse
import json
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    img_dir = Path(args.images)
    lbl_dir = Path(args.labels)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    img_id = 0
    ann_id = 0
    cat_id = 1

    for img_file in sorted(img_dir.iterdir()):
        if img_file.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        im = cv2.imread(str(img_file))
        if im is None:
            print('WARN: cannot read {}'.format(img_file))
            continue
        h, w = im.shape[:2]
        img_id += 1
        images.append({'id': img_id, 'file_name': img_file.name, 'width': int(w), 'height': int(h)})

        lbl_file = lbl_dir / (img_file.stem + '.txt')
        if lbl_file.exists():
            for line in lbl_file.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls, xc, yc, bw, bh = (float(x) for x in parts[:5])
                x = (xc - bw / 2) * w
                y = (yc - bh / 2) * h
                bw_px = bw * w
                bh_px = bh * h
                ann_id += 1
                annotations.append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': cat_id,
                    'bbox': [round(x, 2), round(y, 2), round(bw_px, 2), round(bh_px, 2)],
                    'area': round(bw_px * bh_px, 2),
                    'iscrowd': 0,
                })

    coco = {
        'images': images,
        'annotations': annotations,
        'categories': [{'id': cat_id, 'name': 'fish'}],
    }
    with open(out_path, 'w') as f:
        json.dump(coco, f)
    print('Wrote {} images, {} annotations -> {}'.format(len(images), len(annotations), out_path))


if __name__ == '__main__':
    main()
