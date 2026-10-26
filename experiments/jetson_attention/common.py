"""AT-P1 immutable assets and read-only evidence. Python 3.8, stdlib only."""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
GIB = 1024 ** 3


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def signed(path, value):
    write(path, value)
    with Path(path).with_suffix('.sha256').open('x', encoding='ascii') as stream:
        stream.write(sha(path) + '  ' + Path(path).name + '\n')


def report(path):
    digest = sha(path)
    require(Path(path).with_suffix('.sha256').read_text(encoding='ascii').split() == [digest, Path(path).name], 'Report sidecar differs: ' + str(path))
    return read(path)


def safe(base, name):
    rel = PurePosixPath(name)
    require(name and not rel.is_absolute() and '..' not in rel.parts and '\\' not in name and ':' not in name, 'Unsafe relative path')
    target = base.joinpath(*rel.parts)
    require(base.resolve() in target.resolve().parents and not any(p.is_symlink() for p in [target] + list(target.parents)), 'Linked/escaped asset')
    return target


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def protocol():
    return read(ROOT / 'protocol.json')


def boot():
    return Path('/proc/sys/kernel/random/boot_id').read_text(encoding='ascii').strip()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def verify_tree(base, expected, expected_count):
    p = base / 'delivery_manifest.json'
    require(sha(p) == expected, 'Manifest differs: ' + str(base))
    require((base / 'delivery_manifest.sha256').read_text(encoding='ascii').split() == [expected, p.name], 'Manifest sidecar differs')
    value = read(p)
    rows = value['files']
    names = [r['path'] for r in rows]
    require(len(rows) == len(set(names)) == expected_count, 'Manifest count/duplicates differ')
    actual = {p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()}
    require(actual == set(names) | {'delivery_manifest.json', 'delivery_manifest.sha256'}, 'Missing/unlisted assets: ' + str(base))
    for row in rows:
        p = safe(base, row['path'])
        require(p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], 'Asset differs: ' + row['path'])
    return value


def verify_packet(expected):
    p = ROOT / 'delivery_manifest.json'
    require(sha(p) == expected, 'Independent AT-P1 packet manifest differs')
    value = read(p)
    require(value['schema'] == 'jetson_attention_ATF1_integrated_v1', 'Packet schema differs')
    return verify_tree(ROOT, expected, value['payload_files'])


def dependencies(expected):
    verify_packet(expected)
    p = protocol()
    attention = PROJECT / p['attention_directory']
    bundle = PROJECT / p['ECA_bundle_directory']
    accuracy = PROJECT / p['accuracy_directory']
    inputs = PROJECT / p['val_inputs_directory']
    am = verify_tree(attention, p['attention_manifest_sha256'], 275)
    verify_tree(bundle, p['ECA_bundle_manifest_sha256'], 697)
    verify_tree(accuracy, p['accuracy_manifest_sha256'], 311)
    require(sha(inputs / 'input_manifest.json') == p['val_input_manifest_sha256'], 'Val manifest differs')
    require(sha(accuracy / 'dataset_manifest.json') == p['dataset_manifest_sha256'], 'Dataset manifest differs')
    rows = read(accuracy / 'dataset_manifest.json')['rows']
    original = read(accuracy / 'original_split_manifest.json')
    require(sha(accuracy / 'original_split_manifest.json') == sha(bundle / 'protocol/manifest.json'), 'Original split differs')
    canonical = [r for split in ('val', 'test') for r in original if r['split'] == split]
    require(len(rows) == 298 and [{k: r[k] for k in c} for r, c in zip(rows, canonical)] == canonical, 'Dataset IDs/geometry differ')
    val = read(inputs / 'input_manifest.json')['rows']
    require([r['id'] for r in rows if r['split'] == 'val'] == [r['id'] for r in val], 'Val order differs')
    require([r['id'] for r in rows[:5]] == [16, 33, 69, 85, 92], 'Five compatibility IDs differ')
    for row in rows:
        image = image_path(row)
        label = safe(inputs, row['edge_label']) if row['split'] == 'val' else safe(accuracy, row['payload_label'])
        require(sha(image) == row['image_sha256'] and sha(label) == row['label_sha256'], 'Image/label changed')
    for split, boxes in [('val', 261), ('test', 263)]:
        gp = accuracy / ('gt_' + split + '.json')
        require(sha(gp) == p['GT_sha256'][split] == sha(ROOT / gp.name) == sha(bundle / 'protocol' / gp.name), 'GT differs')
        gt = read(gp)
        require([i['id'] for i in gt['images']] == [r['id'] for r in rows if r['split'] == split]
                and len(gt['annotations']) == boxes and gt['categories'] == [dict(id=1, name='fish')], 'GT identities/counts differ')
    indexed = {r['path']: r for r in am['files']}
    for m in p['models']:
        weight = weight_path(m)
        require(sha(weight) == m['weight_sha256'], 'Weight differs: ' + m['id'])
        if m['id'] != 'eca':
            require(indexed[m['weight_relative']]['sha256'] == m['weight_sha256'], 'Weight not in delivery')
            a = read(attention / m['architecture_evidence'])
            require(a['parameters'] == m['parameters'] and a['module_parameters'] == m['module_parameters']
                    and a['channels'] == 128 and a['detect_input_channels'] == [128, 128, 512]
                    and a['layer_index'] == 55 and a['source_index'] == 45
                    and sha(attention / m['config_relative']) == a['config_sha256'], 'Accepted architecture differs')
    video = PROJECT / p['video_relative']
    require(video.stat().st_size == 15511756 and sha(video) == p['video_sha256'], 'Video differs')
    return rows


def weight_path(job):
    p = protocol()
    base = PROJECT / (p['ECA_bundle_directory'] if job['id'] == 'eca' else p['attention_directory'])
    return safe(base, job['weight_relative'])


def image_path(row):
    p = protocol()
    return safe(PROJECT / p['val_inputs_directory'], row['edge_image']) if row['split'] == 'val' else safe(PROJECT / p['accuracy_directory'], row['payload_image'])


def job_config(model):
    return next(m for m in protocol()['models'] if m['id'] == model)


def platform_check():
    require(platform.system() == 'Linux' and platform.machine().lower() in ('aarch64', 'arm64'), 'Qualified Jetson only')
    require(os.geteuid() != 0, 'Ordinary user only; no sudo')
    require(Path(sys.executable).resolve() == Path('/usr/bin/python3.8').resolve(), 'Use qualified /usr/bin/python3.8')
    device = Path('/proc/device-tree/model').read_bytes().replace(b'\0', b'').decode('utf-8', 'replace')
    require('jetson' in device.lower(), 'Jetson device model missing')
    return device


def firefox_present():
    found = []
    for p in Path('/proc').glob('[0-9]*/comm'):
        try:
            if 'firefox' in p.read_text(encoding='utf-8').lower():
                found.append(int(p.parent.name))
        except OSError:
            pass
    require(found, 'Firefox process must remain open')
    return found


def artifacts(directory, exclude=()):
    return [dict(path=p.relative_to(directory).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
            for p in sorted(directory.rglob('*')) if p.is_file()
            and 'cache' not in p.relative_to(directory).parts and p.relative_to(directory).as_posix() not in exclude]


def verify_artifacts(directory, rows):
    require(len({r['path'] for r in rows}) == len(rows), 'Repeated artifact paths')
    for row in rows:
        p = safe(directory, row['path'])
        require(p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], 'Output asset differs: ' + row['path'])


def snapshot(output):
    roots = [(ROOT, 'packet'), (PROJECT / protocol()['attention_directory'] / 'source', 'attention_source')]
    for base, name in roots:
        for p in sorted(base.rglob('*')):
            if p.is_file():
                target = output / 'source_snapshot' / name / p.relative_to(base)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, target)
                require(sha(p) == sha(target), 'Snapshot changed')


def new_output(model):
    base = PROJECT / 'returns'
    require(base.is_dir() and not base.is_symlink(), 'Existing returns directory missing/linked')
    out = safe(base, job_config(model)['output_name'])
    require(out.parent == base and not os.path.lexists(str(out)), 'Output already exists: preserve and stop')
    require(shutil.disk_usage(PROJECT).free >= 2 * GIB, 'Need 2 GiB free disk')
    out.mkdir()
    return out

def verify(expected):
    return verify_packet(expected)

def capacity_wait(monitor,phase,minimum):
    from readiness import capacity
    return capacity(monitor,phase,minimum)

def get_report(directory,stem,pin=None):
    path=directory/(stem+'.json')
    value=report(path)
    require(pin is None or sha(path)==pin,'Pinned report SHA differs')
    verify_artifacts(directory,value.get('artifacts',[]))
    return value

def create_output(name):
    parent=PROJECT/'returns'
    require(parent.is_dir() and not parent.is_symlink(),'Returns directory absent/linked')
    out=safe(parent,name)
    require(out.parent==parent and not os.path.lexists(str(out)),'Output already exists: preserve and stop')
    require(shutil.disk_usage(PROJECT).free>=2*GIB,'Need2GiB free disk')
    out.mkdir()
    return out

def slot_config(slot):
    require(slot in range(1,13),'Only fixed12 slots authorized')
    return protocol()['schedule'][slot-1]

def paths(slot):
    row=slot_config(slot)
    token='{}_20261020_{:02d}_atf1'.format(row['model'],row['replicate'])
    parent=PROJECT/'returns'
    return {key:parent/name for key,name in dict(root='atf1_root_'+token,infer='atf1_infer_'+token,joined='atf1_joined_'+token,
      admission='atf1_admission_'+token+'.json',trace='atf1_quiet_'+token+'.jsonl',terminal='atf1_terminal_'+token+'.log',console='atf1_console_'+token+'.txt').items()}

def exit_lines(slot,phases):
    return ['ATF1_{}_EXIT=0 slot={:02d}'.format(phase,slot) for phase in phases]
