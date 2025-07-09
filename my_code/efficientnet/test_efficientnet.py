import torch
import timm
from PIL import Image
from timm.data import create_transform
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

# ----------------- 1. 加载模型 -----------------
# 使用 timm.create_model 加载模型
# model_name: 你可以选择 'efficientnet_b0' 到 'efficientnet_b7', 甚至是 'tf_efficientnet_b0' (原始TF权重)
# pretrained=True: 加载在ImageNet上的预训练权重
# num_classes=0: 这是关键！移除分类头，模型将直接输出特征向量
local_weights_path = "my_code/efficientnet/efficientnet_b0_ra-3dd342df.pth"
model_name = "efficientnet_b0"
model = timm.create_model(model_name, pretrained=False, num_classes=0)

url = model.default_cfg["url"]
print(f"模型 '{model_name}' 的权重下载地址是:")
print(url)

# 将模型设置为评估模式
model.eval()

# 如果有GPU，将模型移到GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
state_dict = torch.load(local_weights_path, map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model = model.to(device)

print(f"模型 {model_name} 已加载，并准备在 {device} 上提取特征。")
print(f"模型输出的特征维度: {model.num_features}")  # 查看特征维度

# ----------------- 2. 准备图片和转换 -----------------
# timm 可以为特定模型自动创建合适的预处理转换
# data_config = timm.data.resolve_model_data_config(model)
# transforms = timm.data.create_transform_from_model(model, is_training=False)
config = model.default_cfg
# 使用 create_transform 并手动传入配置
# 这些配置项 (如 input_size, interpolation, mean, std, crop_pct) 都存储在 model.default_cfg 字典中
print(config)
transforms = create_transform(
    input_size=config["input_size"],
    is_training=False,
    mean=config["mean"],
    std=config["std"],
    interpolation=config["interpolation"],
    crop_pct=config["crop_pct"],
)


# 加载一张示例图片 (请替换为你自己的图片路径)
try:
    img = Image.open("/home/linhai/图片/截图/截图 2025-07-01 21-31-53.png").convert("RGB")
except FileNotFoundError:
    print("创建一张随机图片作为示例...")
    img = Image.fromarray((torch.rand(224, 224, 3) * 255).byte().numpy())


# 应用转换，将图片变为Tensor
# 转换会自动处理：resize, center crop, to_tensor, normalize
img_tensor = transforms(img)


# PyTorch模型需要一个batch维度，所以我们需要在前面增加一维
# [C, H, W] -> [B, C, H, W]，这里 B=1
input_tensor = img_tensor.unsqueeze(0).to(device)

# 从你的 config 中提取参数
config = {
    "input_size": (3, 224, 224),
    "interpolation": "bicubic",
    "crop_pct": 0.875,
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}

# 1. 计算参数
input_size = config["input_size"][1]  # 224
crop_pct = config["crop_pct"]
resize_size = int(input_size / crop_pct)  # 224 / 0.875 = 256

# 2. 获取插值方法
# 将字符串映射到 torchvision 的枚举类型
interpolation_map = {
    "bicubic": InterpolationMode.BICUBIC,
    "bilinear": InterpolationMode.BILINEAR,
    "nearest": InterpolationMode.NEAREST,
}
interpolation_method = interpolation_map[config["interpolation"]]

# 3. 创建 torchvision.transforms.Compose 管道
torch_transforms = T.Compose(
    [
        # Step 1: Resize
        # 将图像的短边缩放到 resize_size (256)，保持宽高比
        T.Resize(size=resize_size, interpolation=interpolation_method),
        # Step 2: CenterCrop
        # 从中心裁剪出 input_size (224x224)
        T.CenterCrop(size=input_size),
        # Step 4: Normalize
        # 使用 ImageNet 的均值和标准差进行归一化
        T.Normalize(mean=config["mean"], std=config["std"]),
    ]
)

test_tensor = torch.ones_like(input_tensor).to("cuda")
input_tensor = torch_transforms(test_tensor)


print(f"输入Tensor的形状: {input_tensor.shape}")

# ----------------- 3. 提取特征 -----------------
# 使用 torch.no_grad() 来禁用梯度计算，节省内存和计算资源
with torch.no_grad():
    features = model(input_tensor)

# ----------------- 4. 查看结果 -----------------
print(f"提取到的特征向量形状: {features.shape}")
# 对于 efficientnet_b0，输出形状应为 [1, 1280]
# [batch_size, num_features]

print("特征向量 (前10个值):")
print(features[0, :10])

# 你现在可以将这个 `features` tensor 用于你的下游任务了
# 如果需要转为Numpy数组:
features_numpy = features.cpu().numpy()
