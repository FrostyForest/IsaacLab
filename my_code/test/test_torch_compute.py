import torch
import torch.nn.functional as F

tensor_a=torch.ones((5,5))
tensor_b=torch.ones((5,5))
similarity = F.cosine_similarity(tensor_a, tensor_b, dim=-1)
print(similarity)