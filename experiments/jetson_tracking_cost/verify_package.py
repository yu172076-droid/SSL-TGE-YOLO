"""F1 immutable exact package verifier; standard library only."""
import argparse,hashlib,json,os,re,stat
from pathlib import Path,PurePosixPath
SCHEMA="jetson_tracking_formal_f1_package_v1"
PAYLOADS={'t1_vendor/bytetrack/__init__.py', 'accepted_latest_failure.json', 't1_vendor/bytetrack/byte_tracker.py', 'online_adapter.py', 'accepted_f1_prefix.json', 'accuracy_summary.json', 'accepted_f1_resume01.sha256', 'sort_core.py', 't1_vendor/bytetrack/utils/__init__.py', 'c1_adapter/delivery_manifest.json', 'accepted_f1_resume01.json', 'accepted_c2_v8n_b80_s101_ssl.json', 'accepted_c2_parent.json', 'qualification.json', 'preupload_reissue_authorization.sha256', 'root_session_helper.py', 'source_originals/byte_tracker.py', 'source_originals/metrics.py', 'c3_audit.json', 'accepted_c2_v8n_b80_s101_supervised.json', 'bind_results.py', 't1_vendor/bytetrack/utils/kalman_filter.py', 'accepted_steady_bridge.sha256', 'accepted_tracking_metrics.json', 'c1_adapter/model_contract.json', 'root_sampler.py', 'source_originals/sort.py', 'source_originals/matching.py', 't1_vendor/metrics.py', 'c1_adapter/resource_guard.py', 'tracker_adapter.py', 'accepted_f1_prefix.sha256', 'c1_adapter/delivery_manifest.sha256', 't1_vendor/bytetrack/utils/matching.py', 'accuracy_summary.sha256', 'retry_authorization.json', 'common.py', 'video_pair.json', 'protocol.json', 't1_vendor/bytetrack/basetrack.py', 'source_originals/basetrack.py', 'telemetry.py', 'source_originals/stracks.py', 'startup_monitor.py', 'preupload_reissue_authorization.json', 'source_originals/kalman_filter.py', 'c1_adapter/README.md', 'model_runtime.py', 'gpu_frame.py', 'source_originals/ops.py', 'accepted_tracking_metrics.sha256', 'README.md', 'verify_package.py', 'runtime_adapter.py', 'c1_adapter/precheck.py', 'video_formal.py', 'accepted_c2_compact_b80_s101_ssl.json', 't1_vendor/__init__.py', 't1_vendor/bytetrack/utils/stracks.py', 'pair_input.py', 't1_vendor/ops.py', 'accepted_steady_bridge.json'}

def require(v,msg):
 if not v:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def no_links(p):
 p=Path(os.path.abspath(str(p)))
 for x in list(reversed(p.parents))+[p]:
  s=os.lstat(str(x));require(not stat.S_ISLNK(s.st_mode) and not(getattr(s,'st_file_attributes',0)&0x400),'Linked/reparse package path')
 return p
def verify(root,expected):
 root=no_links(root)
 require(re.fullmatch('[0-9a-f]{64}',expected) is not None,'Expected independent SHA256')
 manifest=no_links(root/'delivery_manifest.json');require(sha(manifest)==expected,'Independent C4 manifest differs')
 require(no_links(root/'delivery_manifest.sha256').read_text(encoding='ascii').split()==[expected,'delivery_manifest.json'],'C4 sidecar')
 m=json.loads(manifest.read_text(encoding='utf-8'));rows=m['files']
 require(m['schema']==SCHEMA and len(rows)==len(PAYLOADS) and {r['path'] for r in rows}==PAYLOADS,'C4 exact payload set')
 actual=set()
 for parent,dirs,files in os.walk(str(root),followlinks=False):
  for name in dirs+files:no_links(Path(parent)/name)
  actual.update((Path(parent)/name).relative_to(root).as_posix() for name in files)
 require(actual==PAYLOADS|{'delivery_manifest.json','delivery_manifest.sha256'},'C4 missing/extra file')
 for r in rows:
  name=r['path'];parts=PurePosixPath(name).parts
  require(name and not PurePosixPath(name).is_absolute() and '..' not in parts and '\\' not in name and ':' not in name,'Unsafe payload path')
  p=no_links(root/name)
  require(stat.S_ISREG(p.stat().st_mode) and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'C4 asset differs: '+name)
 return m
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-manifest-sha256',required=True);a=p.parse_args()
 m=verify(Path(__file__).resolve().parent,a.expected_manifest_sha256)
 print(json.dumps(dict(status='verified_only',payloads=len(m['files']),heavy_imports_performed=False)))
