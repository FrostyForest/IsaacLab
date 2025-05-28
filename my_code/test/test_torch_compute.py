import torch
import torch.nn.functional as F

mapping_vectors = torch.tensor([
[1.0, 0.5, 0.5],  # 对应输入 0
[0.5, 1.0, 0.5],  # 对应输入 1
[0.5, 0.5, 1.0]   # 对应输入 2
],device='cuda')


weight_tensor=mapping_vectors[[0,1]]
print(weight_tensor)