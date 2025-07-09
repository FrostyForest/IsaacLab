# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# --- START OF FILE my_custom_lift_env.py ---
import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer, Siglip2TextModel, Siglip2VisionModel

from isaaclab.envs import ManagerBasedRLEnv

from .lift_env_cfg import LiftEnvCfg  # 导入你的配置类

# 假设 PREDEFINED_TARGETS 等定义在 mdp.string_target_defs 或类似地方
from .mdp import ID_TO_TARGET, NUM_TARGETS, PREDEFINED_TARGETS, TARGET_TO_ID
from fastsam import FastSAM, FastSAMPrompt
import chromadb
from chromadb.config import Settings
import timm
import torchvision.transforms.v2 as T
from torchvision.transforms import InterpolationMode


class LiftEnv(ManagerBasedRLEnv):
    # cfg: MyCustomLiftEnvCfg # 类型提示，指向你的配置类

    def __init__(self, cfg: LiftEnvCfg, **kwargs):  # : MyCustomLiftEnvCfg # 接收配置实例,

        # 在这里初始化你的自定义运行时状态变量
        # 暂时先设置环境数量和设备，先初始化环境变量，防止之后初始化顺序问题
        _num_envs = cfg.scene.num_envs  # 假设 cfg.scene.num_envs 总是可用的
        _device = cfg.sim.device  # 假设 cfg.sim.device 总是可用的
        self.feature_dim = 768
        self.current_target_ids_per_env = torch.full((_num_envs,), -1, dtype=torch.long, device=_device)
        self.encoded_task_goal_per_env = torch.zeros(
            _num_envs, NUM_TARGETS, dtype=torch.float32, device=_device  # 假设是 one-hot 编码
        )
        self.current_target_strings_per_env = ["a yellow cube"] * _num_envs
        self.current_target_state_per_env = torch.zeros(
            _num_envs, self.feature_dim, dtype=torch.float32, device=_device  # 假设是 one-hot 编码
        )

        # 加载siglip模型
        self.clip_path = "/home/linhai/code/IsaacLab/my_code/local_siglip2/"
        self.siglip_image_model = Siglip2VisionModel.from_pretrained(self.clip_path, device_map="auto").eval()
        self.image_processor = AutoProcessor.from_pretrained(self.clip_path, device_map="auto")
        self.siglip_text_model = Siglip2TextModel.from_pretrained(self.clip_path, device_map="auto").eval()
        self.text_processor = AutoTokenizer.from_pretrained(self.clip_path, device_map="auto")

        # self.sam_model = FastSAM("/home/linhai/code/IsaacLab/my_code/fast_sam/FastSAM-x.pt")  # 确保路径正确

        # settings = Settings(anonymized_telemetry=False)
        # self.client = chromadb.Client(settings=settings)
        # self.collection = self.client.get_or_create_collection(name="robot_seen_objects")

        self.rgb_extractor = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        self.rgb_processor = self._rgb_processor()
        self.rgb_extractor.eval()
        self.rgb_extractor.to("cuda")
        # 1. 调用父类的 __init__，这是至关重要的第一步
        #    父类的 __init__ 会处理场景的创建、管理器的设置等。
        #    在 super().__init__(cfg) 完成后，self.num_envs, self.device 等属性才可用。
        super().__init__(cfg, **kwargs)
        # self.sam_model.to(cfg.sim.device)
        rgb_weight_path = "/home/linhai/code/IsaacLab/my_code/efficientnet/efficientnet_b0_ra-3dd342df.pth"
        state_dict = torch.load(rgb_weight_path, map_location="cuda")
        self.rgb_extractor.load_state_dict(state_dict, strict=False)

        # 你可以在这里执行一些一次性的设置，如果它们依赖于完全初始化的环境
        # 但不适合放在事件中的逻辑（例如，只在环境对象创建时做一次的事情）
        print(f"MyCustomLiftEnv initialized with {self.num_envs} environments on device {self.device}.")
        print(f"Available string targets: {PREDEFINED_TARGETS}")

    # 你之前实现的 _reset_idx, _update_scene_based_on_target 等方法会在这里
    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        # 目标随机化现在由事件处理，但你可能还有其他特定于 MyCustomLiftEnv 的重置逻辑
        # 例如，如果你有一些计数器或其他每个 episode 需要重置的状态
        pass

    def _update_scene_based_on_target(self, env_ids: torch.Tensor):
        # ... (如之前的实现) ...
        pass

    def _rgb_processor(self):
        # 获得预训练模型预处理流程
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
                T.ToDtype(torch.float32, scale=True),
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
        return torch_transforms

    # 其他特定于你的环境的方法...


# --- END OF FILE my_custom_lift_env.py ---
