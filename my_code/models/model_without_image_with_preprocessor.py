from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
import torch
import torch.nn as nn
from my_code.models import utils

# 导入我们重构好的、模块化的 RunningStandardScaler
# 假设你有一个 my_running_standard_scaler.py 文件，或者直接使用 skrl 提供的
from skrl.resources.preprocessors.torch import RunningStandardScaler


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
        init_log_std=0,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        # --- 网络结构定义 (保持不变) ---
        self.state_net1 = nn.Sequential(
            nn.Linear(26, 128),
            nn.ELU(),
        )
        self.state_net2 = nn.Sequential(
            nn.Linear(15, 128),
            nn.ELU(),
        )
        self.clip_embedding = nn.Linear(in_features=768, out_features=32)

        self.mid_dim = 128 * 2 + 32
        self.block1 = utils.ResidualBlock(feature_size=self.mid_dim)
        self.block2 = utils.ResidualBlock(feature_size=self.mid_dim)
        self.action_block = utils.ResidualBlock(feature_size=self.mid_dim)

        self.mean_layer = nn.Sequential(
            nn.Linear(self.mid_dim, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions) * init_log_std)
        self.value_layer = nn.Sequential(
            nn.Linear(self.mid_dim, 1),
        )

        # --- 重构后的归一化部分 ---
        # 1. 定义需要被归一化的观测空间部分
        normalization_spaces = {
            "joint_pos": observation_space["joint_pos"],
            "joint_vel": observation_space["joint_vel"],
            "object_position_perfect": observation_space["object_position_perfect"],
            "ee_position": observation_space["ee_position"],
            "target_object_position": observation_space["target_object_position"],
            "actions": observation_space["actions"],
            # contact_force 不需要归一化，所以不在这里定义
        }

        # 2. 使用 nn.ModuleDict 来存储 RunningStandardScaler 实例
        #    这能确保它们被正确地注册为模型的子模块，并随模型移动到正确的设备
        self.scalers = nn.ModuleDict(
            {key: RunningStandardScaler(size=space.shape, device=device) for key, space in normalization_spaces.items()}
        )
        print("归一化模块 (scalers) 已初始化。")

        # 缓存共享层的输出，以避免在 policy 和 value 角色中重复计算
        self._shared_output = None

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def _extract_features(self, inputs):
        """
        抽取并处理所有输入特征的私有方法，以减少代码重复。
        这个方法现在是模型的核心前向传播逻辑。
        """
        space = self.tensor_to_space(inputs["states"], self.observation_space)

        # --- 使用 self.scalers 来动态地归一化数据 ---
        # self.training 标志会由 skrl 的 agent.set_mode() 自动设置，
        # 告知 scaler 是否需要更新其内部的均值和方差统计数据。
        norm_joint_pos = self.scalers["joint_pos"](space["joint_pos"], train=self.training)
        norm_joint_vel = self.scalers["joint_vel"](space["joint_vel"], train=self.training)
        norm_actions = self.scalers["actions"](space["actions"], train=self.training)
        norm_object_pos = self.scalers["object_position_perfect"](space["object_position_perfect"], train=self.training)
        norm_ee_pos = self.scalers["ee_position"](space["ee_position"], train=self.training)
        norm_target_pos = self.scalers["target_object_position"](space["target_object_position"], train=self.training)

        # --- 特征拼接与处理 ---
        state_data1 = torch.cat([norm_joint_pos, norm_joint_vel, norm_actions], dim=-1).to(torch.float32)
        state_data2 = torch.cat(
            [
                norm_object_pos,
                norm_ee_pos,
                norm_target_pos,
                space["contact_force_left_finger"],  # contact_force 不经过归一化
                space["contact_force_right_finger"],
            ],
            dim=-1,
        ).to(torch.float32)

        feature1 = self.state_net1(state_data1)
        feature2 = self.state_net2(state_data2)
        clip_embed = self.clip_embedding(space["text_clip_feature"])

        # --- 特征融合 ---
        combined_features = torch.cat([feature1, feature2, clip_embed], dim=-1)

        r1 = self.block1(combined_features)
        r2 = self.block2(r1)

        return r2

    def compute(self, inputs, role):
        """
        compute 方法现在变得非常简洁，它只负责调用 act 方法。
        真正的计算逻辑在 act 和 _extract_features 中。
        这是为了与 skrl 的 GaussianMixin/DeterministicMixin 的 act 流程更好地集成。
        """
        if role == "policy":
            # 计算并缓存共享输出
            _shared_output = self._extract_features(inputs)
            # 从共享输出计算策略头
            r3 = self.action_block(_shared_output)
            mean = self.mean_layer(r3)
            return mean, self.log_std_parameter, {}
        elif role == "value":
            shared_output = self._extract_features(inputs)
            # 从共享输出计算价值头
            value = self.value_layer(shared_output)
            return value, {}
