"""Standard-library utilities shared by the internal-control experiment runners."""
import hashlib
import json
import math
import os
import shutil
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def outside(project, output):
    project, output = Path(project).resolve(), Path(output).resolve()
    if output == project or project in output.parents:
        raise ValueError('Output must be outside the original project.')
    return project, output


def copy_checked(source, dest, digest=None):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise ValueError('Refusing overwrite: ' + str(dest))
    shutil.copy2(source, dest)
    expected = digest or sha(source)
    if sha(dest) != expected:
        raise ValueError('Copied file hash mismatch: ' + str(dest))


def image_alias(source, dest):
    """Only staged images are hardlinked. Labels are always separate new files."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise ValueError('Refusing overwrite: ' + str(dest))
    try:
        os.link(str(source), str(dest))
    except OSError:
        shutil.copy2(source, dest)


def parse_label(path):
    if not Path(path).is_file():
        raise ValueError('Missing annotation; never interpreted as a negative: ' + str(path))
    boxes = []
    for number, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            a = [float(x) for x in line.split()]
        except ValueError:
            raise ValueError('Non-numeric annotation at line {}'.format(number))
        if (len(a) != 5 or not all(math.isfinite(x) for x in a) or a[0] != 0 or
                not all(0 <= x <= 1 for x in a[1:]) or a[3] <= 0 or a[4] <= 0):
            raise ValueError('Invalid single-class YOLO box at line {}'.format(number))
        boxes.append(tuple(a))
    return tuple(sorted(boxes))


def normalized_to_xyxy(box, width, height):
    _, x, y, w, h = box
    return [max(0, (x-w/2)*width), max(0, (y-h/2)*height),
            min(width, (x+w/2)*width), min(height, (y+h/2)*height)]


def iou(a, b):
    inter = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    aa = max(0, a[2]-a[0])*max(0, a[3]-a[1])
    bb = max(0, b[2]-b[0])*max(0, b[3]-b[1])
    return inter / (aa+bb-inter) if aa+bb-inter else 0.0


def match(predictions, truth, confidence, threshold=.5):
    """Confidence-ordered one-to-one match to the BEST STILL UNMATCHED GT."""
    unmatched = set(range(len(truth)))
    tp = fp = 0
    for p in sorted((p for p in predictions if p['score'] >= confidence), key=lambda p: -p['score']):
        candidates = [(iou(p['xyxy'], truth[j]), j) for j in sorted(unmatched)]
        value, j = max(candidates, default=(-1, -1))
        if value >= threshold:
            tp += 1
            unmatched.remove(j)
        else:
            fp += 1
    return tp, fp, len(unmatched)


def coco_truth(rows, root, split):
    images, annotations = [], []
    for row in rows:
        if row['split'] != split:
            continue
        images.append({'id': row['id'], 'file_name': row['staged_name'],
                       'width': row['width'], 'height': row['height']})
        boxes = parse_label(Path(root)/row['gt_relative'])
        for box in boxes:
            x1, y1, x2, y2 = normalized_to_xyxy(box, row['width'], row['height'])
            if x2 <= x1 or y2 <= y1:
                raise ValueError('GT becomes zero area after clipping: '+row['staged_name'])
            annotations.append({'id': len(annotations)+1, 'image_id': row['id'], 'category_id': 1,
                                'bbox': [x1, y1, x2-x1, y2-y1], 'area': (x2-x1)*(y2-y1), 'iscrowd': 0})
    return {'info': {'description': 'Hash-deduplicated internal controls; sequence independence unverified'},
            'licenses': [], 'images': images, 'annotations': annotations,
            'categories': [{'id': 1, 'name': 'fish'}]}


def verify_frozen(root):
    root = Path(root)
    metadata = read(root/'study.json')
    for relative, digest in metadata['frozen_hashes'].items():
        path = root/relative
        if not path.is_file() or sha(path) != digest:
            raise ValueError('Frozen file missing or changed: '+relative)
    return metadata
