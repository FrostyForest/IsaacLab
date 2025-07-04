from ultralytics import FastSAM
import time

source = "/home/linhai/图片/截图/截图 2025-07-01 21-31-53.png"
model = FastSAM("my_code/fast_sam/FastSAM-x.pt")
t1 = time.time()
everything_results = model(source, device="cuda", retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
t2 = time.time()
print("time", t2 - t1)
results = model(source, texts="a yellow cube")
