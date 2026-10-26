"""Explicit startup trace, stable tail admission, unchanged formal paging limit."""
import json,time
POLICY="startup_trace_stable30_formal_two_intervals_v1"
SECONDS=30
GIB=1024**3
def oom(row):
    values={p[0]:p[1] for line in row["raw"]["vmstat"].splitlines() if len(p:=line.split())==2}
    value=values.get("oom_kill")
    return int(value) if value is not None else None
def stable_tail(rows,warmup_start_ns):
    if not rows or warmup_start_ns is None:return dict(ready=False,reason="warmup not started")
    last=rows[-1];target=last["monotonic_ns"]-SECONDS*10**9
    candidates=[i for i,r in enumerate(rows) if warmup_start_ns<=r["monotonic_ns"]<=target]
    if not candidates:return dict(ready=False,reason="need30s observed warmup tail")
    tail=rows[candidates[-1]:];first=tail[0]
    gaps=[b["monotonic_ns"]-a["monotonic_ns"] for a,b in zip(tail,tail[1:])]
    if not gaps or min(gaps)<=0 or max(gaps)>3500000000:return dict(ready=False,reason="tail observation gap")
    delta={k:last["vm_counters"][k]-first["vm_counters"][k] for k in ("pswpin","pswpout")}
    if delta!={"pswpin":0,"pswpout":0}:return dict(ready=False,reason="tail has global paging",global_swap_delta=delta)
    counters=[oom(r) for r in tail]
    if any(x is None for x in counters) or max(counters)!=min(counters):return dict(ready=False,reason="OOM counter missing/changed")
    minimum=min(r["proc_meminfo"]["MemAvailable_bytes"] for r in tail)
    if minimum<GIB:return dict(ready=False,reason="tail capacity below1GiB")
    return dict(ready=True,policy=POLICY,start_monotonic_ns=first["monotonic_ns"],end_monotonic_ns=last["monotonic_ns"],
        duration_s=(last["monotonic_ns"]-first["monotonic_ns"])/1e9,samples=len(tail),max_sample_gap_s=max(gaps)/1e9,
        global_swap_delta=delta,oom_kill_delta=0,minimum_MemAvailable_bytes=minimum,warmup_start_ns=warmup_start_ns)
def build_monitor(guard):
    class Monitor(guard.Monitor):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.warmup_start_ns=None;self.formal_activation=None;self.initial_oom=None
        def tail(self):
            with self.lock:return stable_tail(self.samples,self.warmup_start_ns)
        def set_phase(self,phase,minimum=GIB):
            with self.lock:
                if phase=="measurement" and self.formal_activation is None:
                    proof=self.tail();guard.require(proof["ready"],"Formal admission needs observed stable30s warmup tail")
                    guard.require(0<=time.monotonic_ns()-proof["end_monotonic_ns"]<=3500000000,"Stable-tail proof is stale")
                    self.formal_activation=dict(monotonic_ns=time.monotonic_ns(),policy=POLICY,stability=proof)
                super().set_phase(phase,minimum)
        def tick(self,phase=None,minimum=None):
            with self.lock:
                self.check()
                row=self.sampler(phase or self.phase)
                threshold=self.minimum if minimum is None else minimum
                row["gate"]=dict(minimum_MemAvailable_bytes=threshold,passed=row["proc_meminfo"]["MemAvailable_bytes"]>=threshold)
                enforced=self.formal_activation is not None
                row["paging_failure_enforced"]=enforced;row["startup_policy"]=POLICY
                if self.samples:row["actual_sample_gap_s"]=(row["monotonic_ns"]-self.samples[-1]["monotonic_ns"])/1e9
                self.samples.append(row)
                try:
                    if not row["gate"]["passed"]:raise guard.ResourceStop("{} MemAvailable below {} bytes".format(row["phase"],threshold))
                    current=oom(row);row["oom_kill_counter"]=current
                    if current is None:raise guard.ResourceStop("Mandatory global oom_kill counter unavailable")
                    if self.initial_oom is None:self.initial_oom=current
                    if current!=self.initial_oom:raise guard.ResourceStop("Global oom_kill counter changed")
                    previous=self.previous_interval
                    if previous is None or row["monotonic_ns"]-previous["monotonic_ns"]>=750000000:
                        if previous is not None:
                            delta={k:row["vm_counters"][k]-previous["vm_counters"][k] for k in guard.VM_KEYS}
                            row["vm_counter_deltas"]=delta
                            if any(v<0 for v in delta.values()):raise guard.ResourceStop("Global vmstat counters decreased")
                            self.streak=self.streak+1 if delta["pswpin"]>0 or delta["pswpout"]>0 else 0
                            row["consecutive_positive_global_swap_intervals"]=self.streak
                            if self.streak>=2 and enforced:raise guard.ResourceStop("Two consecutive positive global swap intervals; no process attribution")
                        self.previous_interval=row
                except Exception as exc:
                    self.failure=exc;raise
                finally:
                    if self.sample_path is not None:
                        with self.sample_path.open("a",encoding="utf-8") as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
                return row
    return Monitor
