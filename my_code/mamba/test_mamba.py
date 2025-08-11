import os

os.environ["TRITON_F32_DEFAULT"] = "ieee"
import torch
from mambapy.mamba2 import Mamba2, Mamba2Config

config = Mamba2Config(d_model=256, expand_factor=2, d_head=64, n_layers=8)
model = Mamba2(config).to("cuda")

B, L, D = 64, 1, 256
x = torch.randn(B, L, D).to("cuda")
y = model(x)

assert y.shape == x.shape
print(y.shape)
torch.save(model.state_dict(), "test_mamba.pt")
