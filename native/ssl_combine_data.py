#!/usr/bin/env python3
"""
Combine real labels + pseudo labels for student training (parameterized --ssl-dir).

Usage:
  python ssl_combine_data.py
  python ssl_combine_data.py --ssl-dir data/datasets/ssl_50
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ssl-dir', type=str, default='data/datasets/ssl')
    args = ap.parse_args()

    ssl_dir = ROOT / args.ssl_dir
    labeled_img = ssl_dir / 'labeled' / 'images'
    labeled_lbl = ssl_dir / 'labeled' / 'labels'
    unlabeled_img = ssl_dir / 'unlabeled' / 'images'
    pseudo_lbl = ssl_dir / 'pseudo_labels'
    combined_img = ssl_dir / 'combined' / 'images'
    combined_lbl = ssl_dir / 'combined' / 'labels'

    for d in (combined_img, combined_lbl):
        d.mkdir(parents=True, exist_ok=True)

    count_labeled = 0
    for lbl_file in labeled_lbl.iterdir():
        if lbl_file.suffix == '.txt' and lbl_file.stat().st_size > 0:
            stem = lbl_file.stem
            for ext in ('.jpg', '.jpeg', '.png'):
                img_file = labeled_img / (stem + ext)
                if img_file.exists():
                    shutil.copy2(img_file, combined_img / img_file.name)
                    break
            shutil.copy2(lbl_file, combined_lbl / lbl_file.name)
            count_labeled += 1

    count_pseudo = 0
    for lbl_file in pseudo_lbl.iterdir():
        if lbl_file.suffix == '.txt' and lbl_file.stat().st_size > 0:
            stem = lbl_file.stem
            for ext in ('.jpg', '.jpeg', '.png'):
                img_file = unlabeled_img / (stem + ext)
                if img_file.exists():
                    shutil.copy2(img_file, combined_img / img_file.name)
                    break
            shutil.copy2(lbl_file, combined_lbl / lbl_file.name)
            count_pseudo += 1

    print('Combined training set: {} labeled + {} pseudo-labeled = {} images'.format(
        count_labeled, count_pseudo, count_labeled + count_pseudo))
    print('Saved to: {} and {}'.format(combined_img, combined_lbl))


if __name__ == '__main__':
    main()
