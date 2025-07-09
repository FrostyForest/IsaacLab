import torch
import timm

# --- 参数配置 ---
# 你可以换成 'resnet50', 'efficientnet_b0', 'convnext_tiny' 等
model_name = "mobilenetv3_small_050.lamb_in1k"
# 设置你想要的输入通道数，例如 5 (多光谱) 或 1 (灰度)
input_channels = 5
# 是否使用ImageNet预训练权重
use_pretrained = False

# --- 1. 创建模型并灵活设置输入通道 ---

# timm 会自动处理预训练权重的加载问题。
# 当 in_chans != 3 时，它会保留大部分预训练权重，
# 而第一个卷积层的权重会进行平均或随机初始化来匹配新的通道数。
# 这种策略通常比完全随机初始化要好得多。
model = timm.create_model(
    model_name,
    pretrained=use_pretrained,
    in_chans=input_channels,
)

print(f"成功创建模型: {model_name}")
# # 我们可以检查第一个卷积层的权重形状来验证
# first_conv_layer = model.conv1 # 对于ResNet是conv1, 对于EfficientNet是conv_stem
# print(f"第一个卷积层的权重形状: {first_conv_layer.weight.shape}")
# 输出应为 torch.Size([64, 5, 7, 7])，其中 5 就是我们设置的 in_chans

# --- 2. 方便地获取特征 ---

# --- 方法 A: 获取全局特征向量 (用于分类、检索等) ---
print("\n--- 方法A: 获取全局特征向量 ---")
feature_extractor_vector = timm.create_model(
    model_name,
    pretrained=use_pretrained,
    in_chans=input_channels,
    num_classes=0,  # 关键：移除分类头
    # features_only=True
)
feature_extractor_vector.eval()
dummy_input_vec = torch.randn(2, input_channels, 224, 224)  # Batch, C, H, W
features_vec = feature_extractor_vector(dummy_input_vec)


# print(f"模型输出的特征向量维度: {feature_extractor_vector.num_features}")
print(f"输出特征向量的形状: {features_vec.shape}")  # e.g., torch.Size([2, 1024])

# #定义保存路径
# save_path = model_name

# # 使用 torch.save() 保存模型的状态字典
# torch.save(model.state_dict(), save_path)
