import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        # Norm -> Activation -> Linear
        self.block1 = nn.Sequential(
            nn.Linear(feature_size, feature_size // 2),
            nn.LayerNorm(feature_size // 2),
            nn.ELU(),
            nn.Linear(feature_size // 2, feature_size),
        )

    def forward(self, x):
        # shortcut 路径是完全干净的恒等映射
        shortcut = x
        # 残差路径
        residual = self.block1(x)
        # 输出是干净的输入加上残差
        return shortcut + residual


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity
        return out


class CustomFeatureExtractor(nn.Module):
    def __init__(self, in_channels=4, output_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 96, stride=2)
        self.layer2 = self._make_layer(96, 128, stride=2)
        # self.layer3 = self._make_layer(128, 160, stride=2)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(128, output_dim))

    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride), BasicBlock(out_channels, out_channels, stride=1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        x = self.head(x)
        return x
