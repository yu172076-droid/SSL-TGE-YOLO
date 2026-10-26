"""Single-class ndarray bridges; retain original tracker algorithm cores."""
from types import SimpleNamespace as NS
import numpy as np
import sort_core
from t1_vendor.bytetrack.byte_tracker import BYTETracker
from common import require

class Results:
    def __init__(self,rows):
        self.rows=np.asarray(rows,dtype=np.float32).reshape(-1,6)
        self.conf=self.rows[:,4]
        self.cls=self.rows[:,5]
        self.xyxy=self.rows[:,:4]
        self.xywh=self.xyxy.copy()
        self.xywh[:,:2]=(self.xyxy[:,:2]+self.xyxy[:,2:])/2
        self.xywh[:,2:]=self.xyxy[:,2:]-self.xyxy[:,:2]
    def __getitem__(self,idx):return Results(self.rows[idx])
    def __len__(self):return len(self.rows)

def create(name,protocol):
    if name=="SORT":
        sort_core.KalmanBoxTracker.count=0
        p=protocol["SORT"]
        return sort_core.Sort(max_age=p["max_age"],min_hits=p["min_hits"],iou_threshold=p["iou_threshold"])
    require(name=="ByteTrack","Unknown tracker")
    return BYTETracker(NS(**protocol["ByteTrack"]))

def update(name,tracker,values,protocol):
    raw=np.asarray(values,dtype=np.float32).reshape(-1,6)
    require(np.isfinite(raw).all() and ((raw[:,5]==0).all() if len(raw) else True),"Single-class finite cache required")
    require(not len(raw) or bool(((raw[:,2]>raw[:,0])&(raw[:,3]>raw[:,1])&(raw[:,0]>=0)&(raw[:,1]>=0)&(raw[:,2]<=3840)&(raw[:,3]<=2160)&(raw[:,4]>=0)&(raw[:,4]<=1)).all()),"Cache geometry/score outside authorized bounds")
    if name=="SORT":
        chosen=raw[raw[:,4]>=protocol["SORT"]["detection_threshold"]]
        result=tracker.update(chosen.copy())
        scores={t.id+1:float(t.bbox_history[-1][4]) for t in tracker.trackers}
        require(not len(result) or result.ndim==2 and result.shape[1]==9,"Frozen SORT fork must return9 columns")
        rows=[[*r[:4],float(r[8]),scores[int(r[8])],float(r[4])] for r in result]
    else:
        result=tracker.update(Results(raw.copy()))
        require(not len(result) or result.ndim==2 and result.shape[1]==8,"Frozen ByteTrack must return8 columns")
        rows=[[*r[:4],float(r[4]),float(r[5]),float(r[6])] for r in result]
    clean=[];clipped=[]
    for r in rows:
        require(all(np.isfinite(v) for v in r) and r[4]>=1 and r[4]==int(r[4]) and r[6]==0,"Tracker ID/class/finite")
        box=[max(0.,min(3840.,r[0])),max(0.,min(2160.,r[1])),max(0.,min(3840.,r[2])),max(0.,min(2160.,r[3]))]
        if box!=r[:4]:clipped.append(dict(id=int(r[4]),original_xyxy=r[:4],clipped_xyxy=box))
        if box[2]>box[0] and box[3]>box[1]:clean.append([*box,*r[4:]])
        else:clipped.append(dict(id=int(r[4]),excluded_zero_area=True))
    require(len({r[4] for r in clean})==len(clean),"Duplicate trackID in one frame")
    return clean,clipped

def synthetic_probe(protocol):
    # No real video or GT; demonstrate low-score association and empty-frame updates.
    frames=[[[100,100,140,130,.9,0]],[[101,100,141,130,.9,0]],
            [[102,100,142,130,.15,0]],[[103,100,143,130,.15,0]],[],
            [[105,100,145,130,.9,0]],[[106,100,146,130,.9,0]],[[107,100,147,130,.9,0]]]
    out={}
    for name in ("SORT","ByteTrack"):
        t=create(name,protocol);rows=[]
        for v in frames:
            values,clipped=update(name,t,v,protocol);rows.append(values)
        require(len(rows[0])==1 and rows[0][0][4]==1,"Native tracker start/reset ID")
        require((t.frame_count if name=="SORT" else t.frame_id)==8,"Empty-frame update missing")
        again=create(name,protocol);reset,unused=update(name,again,frames[0],protocol)
        require(reset[0][4]==1,"Tracker reset between clips")
        out[name]=dict(frame_count=8,tracks_per_frame=[len(r) for r in rows],ids_per_frame=[[int(x[4]) for x in r] for r in rows],reset_id=1)
    require(out["SORT"]["tracks_per_frame"][2]==0 and out["ByteTrack"]["tracks_per_frame"][2]==1,"Low-score stage not exercised")
    return out
