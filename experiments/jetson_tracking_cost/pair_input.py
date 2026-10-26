"""Sequential original4K R04 thenR02 capture; no concatenation,resize,cache."""
from pathlib import Path
import common as c
class PairCapture:
    def __init__(self,cv2,descriptor,report):
        self.cv2=cv2;self.descriptor=descriptor;self.report=report;self.index=0;self.frames=0;self.cap=None
        self.audit=dict(clips=[],complete=False)
        report.setdefault("video_pair_capture_audits",[]).append(self.audit)
        self.open()
    def open(self):
        row=self.descriptor["videos"][self.index]
        self.cap=self.cv2.VideoCapture(str(c.PROJECT/row["path"]),self.cv2.CAP_FFMPEG)
        c.require(self.cap.isOpened() and self.cap.getBackendName()=="FFMPEG"
                  and self.cap.get(self.cv2.CAP_PROP_FRAME_COUNT)==750
                  and abs(self.cap.get(self.cv2.CAP_PROP_FPS)-25.)<1e-6,"Pair clip header/backend")
        self.frames=0
        self.audit["clips"].append(dict(clip=row["clip"],frames=0,eof_seen=False))
    def isOpened(self):return self.cap is not None and self.cap.isOpened()
    def getBackendName(self):return "FFMPEG"
    def read(self):
        while self.cap is not None:
            ok,frame=self.cap.read()
            if ok:
                self.frames+=1;c.require(self.frames<=750,"Extra native clip frame")
                self.audit["clips"][-1]["frames"]=self.frames
                return ok,frame
            c.require(frame is None and self.frames==750,"Truncated pair clip")
            self.audit["clips"][-1]["eof_seen"]=True;self.cap.release();self.cap=None
            self.index+=1
            if self.index==2:
                self.audit["complete"]=True;return False,None
            self.open()
        return False,None
    def release(self):
        if self.cap is not None:self.cap.release();self.cap=None
class CV2:
    def __init__(self,original,descriptor,report):
        self.original=original;self.descriptor=descriptor;self.report=report
    def __getattr__(self,name):return getattr(self.original,name)
    def VideoCapture(self,path,backend):
        c.require(Path(path).resolve()==(c.ROOT/"video_pair.json").resolve() and backend==self.original.CAP_FFMPEG,"Frozen pair descriptor only")
        return PairCapture(self.original,self.descriptor,self.report)
def verify_pair():
    descriptor=c.read(c.ROOT/"video_pair.json")
    c.require(descriptor["order"]==["R04","R02"] and descriptor["frames_per_pass"]==1500,"Pair identity")
    for row in descriptor["videos"]:
        p=c.PROJECT/row["path"]
        c.require(not p.is_symlink() and p.stat().st_size==row["bytes"] and c.sha(p)==row["sha256"],"Pair original video changed")
    return descriptor
