"""Validate released split/label/identity metadata without loading a model."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
manifest = read(ROOT/'data/internal/manifest.json')
labels = read(ROOT/'data/internal/labels.json')
assert len(manifest) == 1486
assert Counter(x['split'] for x in manifest) == {'train':1188,'val':149,'test':149}
assert len({x['image_sha256'] for x in manifest}) == 1486
assert len({x['id'] for x in manifest}) == 1486
for row in manifest:
    label = labels[row['staged_name']]
    assert label['label_sha256'] == row['label_sha256']
    assert hashlib.sha256(label['text'].encode('utf-8')).hexdigest() == row['label_sha256']
    for line in label['text'].splitlines():
        if not line.strip(): continue
        values = [float(v) for v in line.split()]
        assert len(values) == 5 and values[0] == 0 and all(math.isfinite(v) for v in values)
budgets = read(ROOT/'data/internal/manual_label_budgets.json')
train_ids = {x['id'] for x in manifest if x['split'] == 'train'}
for n in [80,144]:
    ids = {int(Path(str(v)).stem) for v in budgets[str(n)]['labeled_ids']}
    assert len(ids) == n and ids <= train_ids
assert set(budgets['80']['labeled_ids']) <= set(budgets['144']['labeled_ids'])
date = read(ROOT/'data/date_holdout/manifest.json')
assert len(date) == 436 and Counter(x['split'] for x in date) == {'train':216,'val':146,'test':74}
assert {x['image_sha256'] for x in date} <= {x['image_sha256'] for x in manifest}
counts = {}
for name,expected_boxes,expected_tracks in [('R04',1188,26),('R02',1227,19)]:
    with (ROOT/f'data/tracking/{name}_boxes.csv').open(encoding='utf-8',newline='') as f:
        boxes = list(csv.DictReader(f))
    assert len(boxes) == expected_boxes
    assert len({r['xml_track_id'] for r in boxes}) == expected_tracks
    pairs = {(int(r['cvat_frame_0']),r['xml_track_id']) for r in boxes}
    assert len(pairs) == expected_boxes and all(0 <= n < 750 for n,_ in pairs)
    counts[name] = len(boxes)
print(json.dumps({'passed':True,'internal_images':1486,'date_images':436,'tracking_boxes':counts},indent=2))
