"""One process per mode; original detector and tracker cores remain fixed."""
import json,time
import common as c
def native_imports():
    import numpy,scipy,filterpy,lap
    found={name:dict(version=m.__version__,origin=m.__file__) for name,m in
        (("numpy",numpy),("scipy",scipy),("filterpy",filterpy),("lap",lap))}
    old=c.read(c.RETURNS/"tracking_cpu_probe_20261017_01/cpu_probe_report.json")["native_imports"]
    c.require(found==old,"Online probe native dependency versions/origins differ")
    return found
def json_scalar(value):
    import numpy as np
    if isinstance(value,np.generic):
        native=value.item()
        if type(native) in (bool,int,float):return native
    raise TypeError("Unsupported JSON value: "+type(value).__name__)
class Online:
    def __init__(self,mode,plan,report,infer):
        c.require(mode in plan["mode_order"],"Unknown online mode")
        import tracker_adapter
        self.adapter=tracker_adapter;self.mode=mode;self.plan=plan;self.report=report;self.infer=infer;self.tracker=None
        self.stage=None;self.updates=0;self.resets=[]
    def start(self,clip,stage):
        self.stage=(clip,stage)
        self.tracker=None if self.mode=="detector_only" else self.adapter.create(self.mode,self.plan)
        self.resets.append(dict(clip=clip,stage=stage,ID_reset_requested=self.mode!="detector_only"))
    def frame(self,frame,row):
        start=time.monotonic_ns();record,values=self.infer(frame,row)
        kept=[v for v in values if v[2]>v[0] and v[3]>v[1]]
        tick=time.monotonic_ns()
        if self.mode=="detector_only":tracks,clipping=[],[]
        else:
            tracks,clipping=self.adapter.update(self.mode,self.tracker,kept,self.plan);self.updates+=1
        end=time.monotonic_ns()
        record.update(tracking_cpu_ms=(end-tick)/1e6,combined_processing_ms=(end-start)/1e6)
        self.report["online_updates_completed"]=self.updates
        self.report["tracking_executed"]=self.updates>0
        return record,values,tracks,clipping
