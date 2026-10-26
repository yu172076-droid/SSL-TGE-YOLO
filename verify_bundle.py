from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'manifests/file_manifest.json').read_text(encoding='utf-8'))
failures=[]
for row in manifest['files']:
    p=root/row['path']
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
        failures.append(row['path'])
print(json.dumps({'files_checked':len(manifest['files']),'passed':not failures,'failures':failures},indent=2))
raise SystemExit(bool(failures))
