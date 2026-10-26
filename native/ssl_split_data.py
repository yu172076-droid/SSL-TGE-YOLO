#!/usr/bin/env python3
"""
SSL data split (parameterized): pick N labeled images (fixed seed), rest unlabeled.
Fixed seed -> smaller N is a subset of larger N (nested), for Table_6 ratio comparison.

Usage:
  python ssl_split_data.py --n-labeled 80
  python ssl_split_data.py --n-labeled 50  --ssl-dir data/datasets/ssl_50
  python ssl_split_data.py --n-labeled 100 --ssl-dir data/datasets/ssl_100
  python ssl_split_data.py --n-labeled 250 --ssl-dir data/datasets/ssl_250
"""
import argparse
import random
import shutil
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data' / 'datasets'
TRAIN_IMG = DATA_DIR / 'images' / 'train'
TRAIN_LBL = DATA_DIR / 'labels' / 'train'
VAL_IMG = DATA_DIR / 'images' / 'val'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-labeled', type=int, default=80)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ssl-dir', type=str, default='data/datasets/ssl')
    args = ap.parse_args()

    random.seed(args.seed)

    ssl_dir = ROOT / args.ssl_dir
    labeled_img = ssl_dir / 'labeled' / 'images'
    labeled_lbl = ssl_dir / 'labeled' / 'labels'
    unlabeled_img = ssl_dir / 'unlabeled' / 'images'

    for d in (labeled_img, labeled_lbl, unlabeled_img):
        d.mkdir(parents=True, exist_ok=True)

    images = sorted(f.name for f in TRAIN_IMG.iterdir()
                    if f.suffix.lower() in ('.jpg', '.jpeg', '.png'))

    print('Total training images: {}'.format(len(images)))

    random.shuffle(images)
    labeled = sorted(images[:args.n_labeled])
    unlabeled = sorted(images[args.n_labeled:])

    print('Labeled: {}, Unlabeled: {}'.format(len(labeled), len(unlabeled)))

    for img in labeled:
        shutil.copy2(TRAIN_IMG / img, labeled_img / img)
        lbl = Path(img).stem + '.txt'
        if (TRAIN_LBL / lbl).exists():
            shutil.copy2(TRAIN_LBL / lbl, labeled_lbl / lbl)

    for img in unlabeled:
        shutil.copy2(TRAIN_IMG / img, unlabeled_img / img)

    teacher_yaml = {
        'train': str(labeled_img),
        'val': str(VAL_IMG),
        'nc': 1,
        'names': ['fish'],
    }
    with open(ssl_dir / 'ssl_teacher.yaml', 'w') as f:
        yaml.dump(teacher_yaml, f, default_flow_style=False)

    student_yaml = {
        'train': str(ssl_dir / 'combined' / 'images'),
        'val': str(VAL_IMG),
        'nc': 1,
        'names': ['fish'],
    }
    with open(ssl_dir / 'ssl_student.yaml', 'w') as f:
        yaml.dump(student_yaml, f, default_flow_style=False)

    with open(ssl_dir / 'labeled_list.txt', 'w') as f:
        f.write('\n'.join(labeled))
    with open(ssl_dir / 'unlabeled_list.txt', 'w') as f:
        f.write('\n'.join(unlabeled))

    print('Done. Labeled: {}, Unlabeled: {}'.format(len(labeled), len(unlabeled)))
    print('Teacher yaml: {}'.format(ssl_dir / 'ssl_teacher.yaml'))
    print('Student yaml: {}'.format(ssl_dir / 'ssl_student.yaml'))


if __name__ == '__main__':
    main()
