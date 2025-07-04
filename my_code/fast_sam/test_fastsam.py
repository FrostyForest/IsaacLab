from fastsam import FastSAM, FastSAMPrompt
import time

model = FastSAM("my_code/fast_sam/FastSAM-x.pt")
IMAGE_PATH = "/home/linhai/图片/截图/截图 2025-07-01 21-31-53.png"
DEVICE = "cuda"
t1 = time.time()
everything_results = model(
    IMAGE_PATH,
    device=DEVICE,
    retina_masks=True,
    imgsz=256,
    conf=0.4,
    iou=0.9,
)
t2 = time.time()
print("time", t2 - t1)
print(everything_results[0])
mask = everything_results
# prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)

# # everything prompt
# ann = prompt_process.everything_prompt()

# prompt_process.plot(annotations=ann,output_path='./output/dog.jpg',)
