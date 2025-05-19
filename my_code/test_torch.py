import torch.nn as nn
import torch
net=nn.Sequential(
            nn.Conv2d(in_channels=6,out_channels=12,kernel_size=3,stride=1,padding='same'),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=12, out_channels=18, kernel_size=3, stride=1, padding='same'),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=18, out_channels=24, kernel_size=5, stride=3, padding='valid'),
            nn.Flatten(),
            nn.LayerNorm(384),
            nn.ELU(),
            nn.Linear(in_features=384,out_features=10)
        )

test_t=torch.randn(10,6,64,64)
result=net(test_t)
print(result.shape)