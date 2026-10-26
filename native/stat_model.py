import torch
from models.yolo import Model
from thop import profile

def stat(cfg, name):
    model = Model(cfg, ch=3, nc=1)
    model.eval()
    model = model.to('cuda')
    dummy = torch.randn(1, 3, 640, 640).to('cuda')
    flops, params = profile(model, inputs=(dummy,), verbose=False)
    print(f'{name}: Params={params/1e6:.2f}M, GFLOPs={flops/1e9:.2f}G')

for cfg, name in [
    ('cfg/training/yolov7-tiny.yaml', 'YOLOv7-tiny'),
    ('cfg/training/yolov7-tiny-GhostNet.yaml', 'GhostNet'),
    ('cfg/training/yolov7-tiny-GhostNet-ECA.yaml', 'TGE-YOLO'),
]:
    try:
        stat(cfg, name)
    except Exception as e:
        print(f'{name}: ERROR {e}')
