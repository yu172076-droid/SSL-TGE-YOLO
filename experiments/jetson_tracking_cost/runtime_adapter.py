"""Qualified unchanged model loader plus fixed low-score online tracker wrapper."""
import json,types
from pathlib import Path
import common as c
ROOT=Path(__file__).resolve().parent
core=c.module("F1_qualified_model_runtime",ROOT/"model_runtime.py")
low=c.module("F1_low_score_GPU_frame",ROOT/"gpu_frame.py")
ARGS=None
c1=core.c1;guard=core.guard
import startup_monitor
Monitor=startup_monitor.build_monitor(guard)
C1_SHA=core.C1_SHA;C2_SHA=core.C2_SHA
select=core.select;qualification=core.qualification;prerequisites=core.prerequisites
require_jetson=core.require_jetson;stop_child=core.stop_child;boot_id=core.boot_id
validate_values=low.validate_values
def prepare_model(output,report,state):
    core.ARGS=ARGS;core.Monitor=Monitor
    core.validate_values=low.validate_values
    core.one_frame=types.FunctionType(low.one_frame.__code__,core.__dict__,"one_frame")
    cv2,torch,monitor,infer=core.prepare_model(output,report,state)
    import online_adapter,pair_input
    report["native_tracker_imports"]=online_adapter.native_imports()
    plan=c.read(ROOT/"protocol.json");descriptor=pair_input.verify_pair()
    report["pair_descriptor"]=descriptor;report["mode"]=ARGS.mode
    report["GT_used_by_tracker"]=False;report["online_updates_completed"]=0
    online=online_adapter.Online(ARGS.mode,plan,report,infer)
    report["online_reset_events"]=online.resets
    def combined(frame,row):
        ordinal=row["id"];clip="R04" if ordinal<=750 else "R02";frame1=(ordinal-1)%750+1
        phase="measurement" if "measurement" in report["windows"] else "warmup"
        if frame1==1:online.start(clip,phase)
        native_row=dict(id=ordinal,edge_image=clip+"/frame_{}".format(frame1))
        record,values,tracks,clipping=online.frame(frame,native_row)
        payload=dict(clip=clip,clip_frame_1=frame1,mode=ARGS.mode,tracks=tracks,clipping=clipping,
                     tracker_reset_requested=frame1==1 and ARGS.mode!="detector_only")
        report["_last_online_payload"]=json.loads(json.dumps(payload,allow_nan=False,default=online_adapter.json_scalar))
        record["track_count"]=len(tracks)
        return record,values
    return pair_input.CV2(cv2,descriptor,report),torch,monitor,combined
