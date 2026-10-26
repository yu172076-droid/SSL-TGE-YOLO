"""Stdlib integrity helpers for this target-specific P1 release."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def need(ok,message):
    if not ok: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def new_json(path,data):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')


def receipt(out,name,data):
    p=Path(out)/(name+'.json'); new_json(p,data)
    with p.with_suffix('.sha256').open('x',encoding='ascii') as f: f.write(sha(p)+'  '+p.name+'\n')


def verify(pin):
    need(sha(ROOT/'manifest.json')==pin,'P1 packet manifest SHA differs')
    manifest=read(ROOT/'manifest.json'); names=set()
    for row in manifest['files']:
        rel=Path(row['path']); p=ROOT/rel
        need(not rel.is_absolute() and '..' not in rel.parts and row['path'] not in names,'Unsafe/duplicate P1 path')
        names.add(row['path'])
        need(not p.is_symlink() and p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'P1 payload changed: '+row['path'])
    need(manifest['schema']=='pi_PP1_SHORT300_release_packet_v1','Wrong packet schema')
    return read(ROOT/'config.json')


def boot(): return Path('/proc/sys/kernel/random/boot_id').read_text(encoding='ascii').strip()


def process_start_ticks(pid):
    try:
        # comm may contain spaces/parentheses; fields after last ')' begin with stat field3.
        values=Path('/proc/'+str(pid)+'/stat').read_text(encoding='utf-8').rsplit(')',1)[1].split()
        return int(values[19])
    except (OSError,ValueError,IndexError): return None
