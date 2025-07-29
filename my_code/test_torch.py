import sys
import os
import torch


t = torch.ones(64, 128, 128, 1)
m = torch.mean(t).unsqueeze(0)
print(m.shape)
