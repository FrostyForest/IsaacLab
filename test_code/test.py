import torch
import torch.nn.functional as F
image_tensor = torch.randint(0, 256, size=(1, 224, 224,4), dtype=torch.uint8)


image_tensor_float = image_tensor.float().permute(0, 3, 1, 2)

# 使用 interpolate 进行放缩
resized_image_tensor = F.interpolate(image_tensor_float, size=(112, 112), mode='bilinear', align_corners=False)

# 将张量转换回 torch.uint8 类型 (可选)
resized_image_tensor_uint8 = resized_image_tensor.to(torch.uint8).permute(0,2,3,1)

print(resized_image_tensor_uint8.shape)