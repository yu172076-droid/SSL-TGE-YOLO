"""Standard-library identity, source and immutable-output helpers."""
import hashlib,importlib.util,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
RETURNS=PROJECT/"returns"
def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda:f.read(1048576),b""):h.update(data)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def write(path,value):
    with Path(path).open("x",encoding="utf-8") as f:f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def signed(path,value):
    write(path,value)
    with Path(path).with_suffix(".sha256").open("x",encoding="ascii") as f:f.write(sha(path)+"  "+Path(path).name+"\n")
def module(name,path):
    s=importlib.util.spec_from_file_location(name,str(path));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def verify(expected):
    require(sha(ROOT/"delivery_manifest.json")==expected,"Independent T1 manifest differs")
    require((ROOT/"delivery_manifest.sha256").read_text().split()==[expected,"delivery_manifest.json"],"T1 manifest sidecar")
    rows=read(ROOT/"delivery_manifest.json")["files"]
    names={r["path"] for r in rows}
    require(len(names)==len(rows),"Duplicate T1 payload")
    actual=set()
    for p in ROOT.rglob("*"):
        require(not p.is_symlink(),"Linked T1 package")
        if p.is_file():actual.add(p.relative_to(ROOT).as_posix())
    require(actual==names|{"delivery_manifest.json","delivery_manifest.sha256"},"T1 extra/missing file")
    for r in rows:
        rel=Path(r["path"]);require(not rel.is_absolute() and ".." not in rel.parts and ":" not in r["path"] and "\\" not in r["path"],"Unsafe T1 payload")
        p=ROOT/rel;require(p.stat().st_size==r["bytes"] and sha(p)==r["sha256"],"Changed T1 asset "+r["path"])
    return rows
def report(base,stem,pin=None):
    p=Path(base)/(stem+".json")
    require(not p.is_symlink() and p.with_suffix(".sha256").read_text().split()==[sha(p),p.name],"Report sidecar differs "+stem)
    require(pin is None or sha(p)==pin,"Pinned report differs "+stem)
    value=read(p)
    for a in value.get("artifacts",[]):
        rel=Path(a["path"]);require(not rel.is_absolute() and ".." not in rel.parts,"Unsafe report asset")
        q=Path(base)/rel;require(not q.is_symlink() and q.stat().st_size==a["bytes"] and sha(q)==a["sha256"],"Report artifact differs "+str(rel))
    return value
def artifacts(base,exclude=()):
    return [dict(path=p.relative_to(base).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(base.rglob("*")) if p.is_file() and p.name not in exclude and "cache" not in p.relative_to(base).parts]
def new_output(name):
    require(RETURNS.is_dir() and not RETURNS.is_symlink(),"Real returns directory required")
    p=RETURNS/name;require(not os.path.lexists(str(p)),"New T1 output only; preserve prior attempt")
    p.mkdir();return p
def verified_module(packet,expected,filename,name):
    require(sha(packet/"delivery_manifest.json")==expected,"Prior package manifest differs")
    row=next(r for r in read(packet/"delivery_manifest.json")["files"] if r["path"]==filename)
    require(not (packet/filename).is_symlink() and sha(packet/filename)==row["sha256"],"Prior source differs before import")
    return module(name,packet/filename)
def prerequisites():
    p=read(ROOT/"protocol.json")
    t0packet=PROJECT/p["packet_T0"]
    t0=verified_module(t0packet,p["T0_manifest_sha256"],"preflight.py","T1_frozen_T0")
    t0.verify(p["T0_manifest_sha256"])
    r=report(RETURNS/"tracking_preflight_20261017_01","preflight_report",p["T0_report_sha256"])
    require(r["exitcode"]==0 and r["status"]=="tracking_preflight_completed_review_required" and r["previous_R2_total"]==9 and r["GT_validation"]["visible_boxes"]==2415,"Accepted T0 incomplete")
    gt=t0packet/p["GT_folder"]
    require(sha(gt/"freeze_manifest.json")==p["GT_manifest_sha256"],"Frozen GT identity differs")
    return p,gt
def r2_driver():
    p=read(ROOT/"protocol.json");packet=PROJECT/p["r2_packet"]
    d=verified_module(packet,p["r2_manifest_sha256"],"video_formal.py","T1_verified_R2")
    d.verify(p["r2_manifest_sha256"]);return d
