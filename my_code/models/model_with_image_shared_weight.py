# --- START OF REFACTORED FILE model_with_image.py ---

import torch
import torch.nn as nn
import timm
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

# 导入新的、模块化的归一化工具
# from skrl.resources.preprocessors.torch import RunningStandardScaler
from models import utils
from models.my_running_standard_scaler import my_RunningStandardScaler


class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
        reduction="sum",
        perfect_position=True,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        # --- 网络结构定义 (保持不变) ---
        self.state_net1 = nn.Sequential(
            nn.Linear(26, 80),
            nn.ELU(),
        )
        self.state_net2 = nn.Sequential(
            nn.Linear(15, 80),
            nn.ELU(),
        )
        self.depth_net = timm.create_model(
            "mobilenetv3_small_050.lamb_in1k", pretrained=False, in_chans=2, num_classes=0
        )
        self.mid_dim = 80 + 80 + 60 * 3 + 60 + 40 * 2

        self.block1 = utils.ResidualBlock(feature_size=self.mid_dim)
        self.block2 = utils.ResidualBlock(feature_size=self.mid_dim)
        self.action_block = utils.ResidualBlock(feature_size=self.mid_dim)
        self.rgb_feature_embedding = nn.Linear(in_features=1280, out_features=40)
        self.depth_feature_embedding = nn.Linear(in_features=1024, out_features=60)
        self.clip_embedding = nn.Linear(in_features=768, out_features=60)

        self.mean_layer = nn.Sequential(nn.Linear(self.mid_dim, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))
        self.value_layer = nn.Sequential(nn.Linear(self.mid_dim, 1))
        self.perfect_position = perfect_position

        # --- 重构后的归一化部分 ---
        # 定义需要被归一化的观测空间部分
        normalization_spaces = {
            "joint_pos": observation_space["joint_pos"],
            "joint_vel": observation_space["joint_vel"],
            "object_position": observation_space["object_position"],
            "ee_position": observation_space["ee_position"],
            "target_object_position": observation_space["target_object_position"],
            "actions": observation_space["actions"],
            "contact_force_left_finger": observation_space["contact_force_left_finger"],
            "contact_force_right_finger": observation_space["contact_force_right_finger"],
        }

        # 使用 nn.ModuleDict 来存储 RunningStandardScaler 实例
        # 这能确保它们被正确地注册为模型的子模块
        self.scalers = nn.ModuleDict(
            {key: my_RunningStandardScaler(size=space, device=device) for key, space in normalization_spaces.items()}
        )

        # 为深度观测值单独创建一个 scaler
        # 注意：size=1 是因为我们将对整个log转换后的特征图进行标准化，使用一个共享的均值和方差
        self.depth_scaler = my_RunningStandardScaler(size=1, device=device)
        self._shared_output = None

    # `_initialization_hook` 方法已被完全移除，因为 RunningStandardScaler 自行处理更新逻辑

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def _extract_features(self, space):
        """
        抽取并处理所有输入特征的私有方法，以减少代码重复。
        """
        # --- 向量状态归一化与处理 ---
        # 使用 self.scalers 来动态地归一化数据
        # self.training 标志会自动告知 scaler 是否需要更新其内部统计数据
        norm_joint_pos = self.scalers["joint_pos"](space["joint_pos"], train=self.training)
        norm_joint_vel = self.scalers["joint_vel"](space["joint_vel"], train=self.training)
        norm_actions = self.scalers["actions"](space["actions"], train=self.training)
        if self.perfect_position == True:
            norm_object_pos = self.scalers["object_position"](space["object_position_perfect"], train=self.training)
        else:
            norm_object_pos = self.scalers["object_position"](space["object_position"], train=self.training)
        norm_ee_pos = self.scalers["ee_position"](space["ee_position"], train=self.training)
        norm_target_pos = self.scalers["target_object_position"](space["target_object_position"], train=self.training)
        norm_left_finger_force = self.scalers["contact_force_left_finger"](
            space["contact_force_left_finger"], train=self.training
        )
        norm_right_finger_force = self.scalers["contact_force_right_finger"](
            space["contact_force_right_finger"], train=self.training
        )

        state_data1 = torch.cat([norm_joint_pos, norm_joint_vel, norm_actions], dim=-1).to(torch.float32)
        state_data2 = torch.cat(
            [norm_object_pos, norm_ee_pos, norm_target_pos, norm_left_finger_force, norm_right_finger_force], dim=-1
        ).to(torch.float32)

        feature1 = self.state_net1(state_data1)
        feature2 = self.state_net2(state_data2)

        # --- 图像与文本特征处理 ---
        # 对深度图先应用对数变换，然后再进行归一化
        depth_log = torch.log10(space["depth_obs"] + 1e-5)
        norm_depth = self.depth_scaler(depth_log, train=self.training).float()

        feature_depth = self.depth_feature_embedding(self.depth_net(norm_depth.permute(0, 3, 1, 2)))
        feature_rgb = self.rgb_feature_embedding(space["rgb_feature"].float()).reshape(-1, 80)
        text_clip_f = self.clip_embedding(space["text_clip_feature"])
        image_clip_f = self.clip_embedding(space["image_clip_feature"]).reshape(-1, 120)

        # --- 特征融合 ---
        combined_features = torch.cat(
            [feature1, feature2, feature_rgb, feature_depth, text_clip_f, image_clip_f], dim=-1
        )

        r1 = self.block1(combined_features)
        r2 = self.block2(r1)

        return r2

    def compute(self, inputs, role):
        space = self.tensor_to_space(inputs["states"], self.observation_space)

        # 策略网络和价值网络共享大部分计算
        # 如果 self._shared_output 存在，说明价值网络可以直接重用策略网络的计算结果
        # if self._shared_output is None:
        #     shared_output = self._extract_features(space)
        # else:
        #     shared_output = self._shared_output
        #     self._shared_output = None # 用完后清空
        shared_output = self._extract_features(space)
        if role == "policy":
            # self._shared_output = shared_output  # 缓存结果给价值网络使用
            r3 = self.action_block(shared_output)

            # with torch.no_grad():
            #     self.log_std_parameter.data.clamp_(max=0.0)
            return self.mean_layer(r3), self.log_std_parameter, {}

        elif role == "value":
            return self.value_layer(shared_output), {}


# --- END OF REFACTORED FILE ---
