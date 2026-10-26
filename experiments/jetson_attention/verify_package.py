"""Self-contained stdlib verifier, safe to load from root isolated Python."""
import hashlib,json,os
from pathlib import Path,PurePosixPath
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def verify(directory,expected):
    directory=Path(directory);p=directory/'delivery_manifest.json'
    if sha(p)!=expected:raise ValueError('Independent ATF1 manifest mismatch')
    if (directory/'delivery_manifest.sha256').read_text(encoding='ascii').split()!=[expected,p.name]:raise ValueError('Manifest sidecar mismatch')
    m=json.loads(p.read_text(encoding='utf-8'));rows=m['files'];names=[r['path'] for r in rows]
    if m['schema']!='jetson_attention_ATF1_integrated_v1' or len(rows)!=m['payload_files'] or len(names)!=len(set(names)):raise ValueError('Manifest identity/count mismatch')
    actual={q.relative_to(directory).as_posix() for q in directory.rglob('*') if q.is_file()}
    if actual!=set(names)|{'delivery_manifest.json','delivery_manifest.sha256'}:raise ValueError('Missing/extra packet files')
    for r in rows:
        rel=PurePosixPath(r['path'])
        if not r['path'] or rel.is_absolute() or '..' in rel.parts or '\\' in r['path'] or ':' in r['path']:raise ValueError('Unsafe packet path')
        q=directory.joinpath(*rel.parts)
        if any(x.is_symlink() for x in [q]+list(q.parents)) or q.stat().st_size!=r['bytes'] or sha(q)!=r['sha256']:raise ValueError('Packet payload changed '+r['path'])
    return m
