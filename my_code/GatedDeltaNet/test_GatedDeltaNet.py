import torch
from fla.layers import GatedDeltaNet

bs, num_heads, seq_len, hidden_size = 16, 4, 2048, 512
gated_deltanet = (
    GatedDeltaNet(hidden_size=hidden_size, num_heads=num_heads, mode="chunk").cuda().to(dtype=torch.float16)
)

x = torch.randn(bs, seq_len, hidden_size, dtype=torch.float16).cuda()
y, _, cache = gated_deltanet(x)
print(y.shape)
print(cache.shape)
