import sys
import os
import torch


t = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
print(t.shape)
t2 = t.reshape(2, 2, 2)
t3 = t2.permute(1, 0, 2)
t4 = t3.contiguous().reshape(2, 4)
print(t4)
