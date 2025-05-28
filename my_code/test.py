import torch
from transformers import AutoModel, AutoProcessor,BitsAndBytesConfig,Siglip2VisionModel,Siglip2TextModel,AutoTokenizer
from transformers.image_utils import load_image
import time
from torchvision.transforms import functional as F
import numpy as np
from torchvision import transforms

# 指定本地模型和处理器的路径
local_model_path = "/home/linhai/code/IsaacLab/my_code/local_siglip2/" # 替换成你实际的文件夹路径
# 从本地路径加载模型和处理器
model = AutoModel.from_pretrained(local_model_path, device_map="auto").eval()
processor = AutoProcessor.from_pretrained(local_model_path)

# 加载图像 (这一步仍然会从网络下载图像，除非你也把图像下载到本地)
image_url = "/home/linhai/图片/10753_crt_V9wc8.jpg"
# 如果你想从本地加载图像:
# 1. 先下载图像: wget https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000285.jpg -O my_image.jpg
# 2. 然后使用本地路径: image = load_image("./my_image.jpg")
image = load_image(image_url)
to_tensor_transform = transforms.PILToTensor()
image = to_tensor_transform(image)
images=image.expand(5,-1,-1,-1)
print(images.shape)
candidate_labels = ["a girl with yellow hair", "a boy with yellow hair"]
texts = [f'{label}' for label in candidate_labels]

#-----  测试同时获取文本和图像的embedding
start_t=time.time()
inputs = processor(text=texts, images=images, padding="max_length", max_num_patches=64, return_tensors="pt").to("cuda")
end_t=time.time()
print('创建输入用时',end_t-start_t)
start_t=time.time()
with torch.no_grad():
    outputs = model(**inputs)
end_t=time.time()
print('计算embedding用时',end_t-start_t)
logits_per_image = outputs.logits_per_image
probs = torch.sigmoid(logits_per_image)
image_embeddings = outputs.image_embeds#(batch_size,768)
text_embeddings = outputs.text_embeds
print(text_embeddings.shape)
print(probs)
print(f"{probs[0][0]:.1%} that image 0 is '{candidate_labels[0]}'")

# ---- 测试仅获取图像的feature
model = Siglip2VisionModel.from_pretrained(local_model_path, device_map="auto").eval()
processor = AutoProcessor.from_pretrained(local_model_path,device_map="auto")

s=time.time()
inputs = processor(images=images, return_tensors="pt",max_num_patches=64).to('cuda')
with torch.no_grad():
    outputs = model(**inputs)
e=time.time()
print('only image_processor',e-s)
last_hidden_state = outputs.last_hidden_state

pooled_output = outputs.pooler_output  # pooled features

print(pooled_output.shape)


# -------- 测试仅获取文本的feature
model = Siglip2TextModel.from_pretrained(local_model_path, device_map="auto").eval()
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

s=time.time()

inputs = tokenizer(texts,padding="max_length", return_tensors="pt").to('cuda')
with torch.no_grad():
    outputs = model(**inputs)
e=time.time()
print('only text_processor',e-s)
last_hidden_state = outputs.last_hidden_state

pooled_output = outputs.pooler_output  # pooled features

print(pooled_output.shape)
