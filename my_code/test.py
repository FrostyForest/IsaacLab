import torch
from transformers import AutoModel, AutoProcessor,BitsAndBytesConfig
from transformers.image_utils import load_image
import time

# 指定本地模型和处理器的路径
local_model_path = "/home/linhai/code/IsaacLab/my_code/local_siglip2/" # 替换成你实际的文件夹路径
# 从本地路径加载模型和处理器
model = AutoModel.from_pretrained(local_model_path, device_map="auto",torch_dtype=torch.bfloat16,).eval()
processor = AutoProcessor.from_pretrained(local_model_path)

# 加载图像 (这一步仍然会从网络下载图像，除非你也把图像下载到本地)
image_url = "/home/linhai/图片/10753_crt_V9wc8.jpg"
# 如果你想从本地加载图像:
# 1. 先下载图像: wget https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000285.jpg -O my_image.jpg
# 2. 然后使用本地路径: image = load_image("./my_image.jpg")
image = load_image(image_url)
candidate_labels = ["a girl with yellow hair", "a boy with yellow hair"]
texts = [f'{label}' for label in candidate_labels]

start_t=time.time()
inputs = processor(text=texts, images=image, padding="max_length", max_num_patches=64, return_tensors="pt").to("cuda")
end_t=time.time()
print(end_t-start_t)
with torch.no_grad():
    outputs = model(**inputs)
    breakpoint()
logits_per_image = outputs.logits_per_image
probs = torch.sigmoid(logits_per_image)
print(probs)
print(f"{probs[0][0]:.1%} that image 0 is '{candidate_labels[0]}'")