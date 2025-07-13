from transformers import AutoModel, AutoProcessor, Siglip2VisionModel, Siglip2TextModel, AutoTokenizer
from PIL import Image
import torch
from torchvision.io import read_image
from torchvision.transforms import functional as F

clip_path = "/home/linhai/code/IsaacLab/my_code/local_siglip2/"
siglip_image_model = Siglip2VisionModel.from_pretrained(clip_path, device_map="auto").eval()
image_processor = AutoProcessor.from_pretrained(clip_path, device_map="auto")


# raw_image_data = Image.open('output_objects/object_2.png')
# 1. 直接读取为 PyTorch Tensor
#    返回的 Tensor 形状是 (Channels, Height, Width)
tensor_image = read_image("output_objects/object_2.png").permute(1, 2, 0)[:, :, :3].unsqueeze(0).expand(3, -1, -1, -1)

# 2. 检查结果
print("成功读取为 PyTorch Tensor！")
print(f"Tensor 类型: {type(tensor_image)}")
print(f"Tensor 形状 (Channels, Height, Width): {tensor_image.shape}")
print(f"数据类型: {tensor_image.dtype}")  # torch.uint8

# # 3. (可选但常用) 将数据类型转换为 float 并归一化到 [0, 1] 区间
# #    这是大多数模型所期望的输入格式
# normalized_tensor = tensor_image.float() / 255.0
# print("\n归一化后:")
# print(f"Tensor 形状: {normalized_tensor.shape}")
# print(f"数据类型: {normalized_tensor.dtype}") # torch.float32
# print(f"最小值: {normalized_tensor.min()}, 最大值: {normalized_tensor.max()}")
tensor2 = torch.ones_like(tensor_image)
img_list = [tensor_image, tensor2]

inputs = image_processor(images=img_list, max_num_patches=64, return_tensors="pt").to("cuda")

outputs = siglip_image_model(**inputs).pooler_output

print(outputs.shape)  # torch.Size([5, 768])
if torch.equal(outputs[0], outputs[1]) == True and torch.equal(outputs[0], outputs[3]) == False:
    print("顺序正确")
