import torch
import torch.nn as nn
import timm
import gymnasium as gym

# 导入 skrl 和 mambapy 的相关组件
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

# 假设 mambapy 在你的 python path 中
# from mambapy.mamba2 import Mamba2Block, Mamba2Config
from mambapy.mamba import Mamba, MambaConfig, RMSNorm  # <--- 导入 Mamba v1 相关类
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))
from my_code.models import utils

# 假设这些是你自己的工具文件
# from models import utils
# from models.my_running_standard_scaler import my_RunningStandardScaler
# 为了可运行性，我们用 skrl 自带的 scaler，并创建一个假的 utils
from my_code.models.my_running_standard_scaler import my_RunningStandardScaler

# class utils:
#     class ResidualBlock(nn.Module):
#         def __init__(self, feature_size):
#             super().__init__()
#             self.block = nn.Sequential(
#                 nn.Linear(feature_size, feature_size),
#                 nn.ReLU()
#             )
#         def forward(self, x):
#             return x + self.block(x)


class SharedMamba(GaussianMixin, DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        # --- PPO/skrl 参数 ---
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
        reduction="sum",
        # --- RNN/Mamba 相关的 skrl 参数 ---
        num_envs=1,
        sequence_length=16,  # PPO mini_batch size
        # --- 自定义模型参数 ---
        perfect_position=True,
        no_object_position=False,
        d_model=256,
        d_head=64,
        n_layers=6,
        expand_factor=2,
        single_step=False,
        activate_depth=True,  # 是否使用深度信息
        activate_contact_sensor=True,  # 是否使用力传感器
    ):
        # 1. 初始化基类
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.num_envs = num_envs
        self.sequence_length = sequence_length
        self.perfect_position = perfect_position
        self.no_object_position = no_object_position
        self.n_layers = n_layers
        self.activate_depth = activate_depth
        self.activate_contact_sensor = activate_contact_sensor

        self.dim_state_net1 = 48
        self.dim_state_net2 = 64
        self.dim_clip_feature = 48
        self.dim_rgb_feature = 32
        if activate_depth:
            self.dim_depth_feature = 64
        else:
            self.dim_depth_feature = 0
        self.single_step = single_step  # single step用于蒸馏
        # 2. 状态/图像/文本特征提取器 (来自 model_with_image_shared_weight.py)
        # 向量状态网络
        self.state_net1 = nn.Sequential(nn.Linear(26, self.dim_state_net1))
        if activate_contact_sensor:
            self.state_net2 = nn.Sequential(nn.Linear(19, self.dim_state_net2))
        else:
            self.state_net2 = nn.Sequential(nn.Linear(17, self.dim_state_net2))
        # 图像网络
        if activate_depth:
            self.depth_net = timm.create_model(
                "mobilenetv3_small_050.lamb_in1k", pretrained=False, in_chans=2, num_classes=0
            )
            self.depth_feature_embedding = nn.Linear(in_features=1024, out_features=self.dim_depth_feature)
        # 特征嵌入层
        self.rgb_feature_embedding = nn.Linear(in_features=1280, out_features=self.dim_rgb_feature)
        self.clip_embedding = nn.Linear(in_features=768, out_features=self.dim_clip_feature)

        # 计算特征融合后的维度
        feature_dim = (
            self.dim_state_net1
            + self.dim_state_net2
            + 2 * self.dim_rgb_feature
            + 3 * self.dim_clip_feature
            + self.dim_depth_feature
        )  # 3个clip feature+2个rgb feature+1个depth feature
        self.pre_mamba_mlp = nn.Linear(in_features=feature_dim, out_features=d_model)

        # 3. 实例化完整的、多层的 Mamba (v1) 模型
        self.mamba_config = MambaConfig(
            d_model=d_model,
            n_layers=self.n_layers,
            expand_factor=expand_factor,
            pscan=True,
            use_cuda=False,
            inner_layernorms=True,  # 强制使用纯torch pscan
        )
        self.mamba_backbone = Mamba(self.mamba_config)

        # 4. 策略和价值网络头
        self.policy_head = nn.Sequential(
            utils.ResidualBlock(feature_size=d_model), nn.Linear(d_model, self.num_actions)
        )
        self.value_head = nn.Sequential(nn.Linear(d_model, 1))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        # 5. 观测值归一化 (来自 model_with_image_shared_weight.py)
        normalization_spaces = {
            "joint_pos": observation_space["joint_pos"],
            "joint_vel": observation_space["joint_vel"],
            "object_position": observation_space["object_position_perfect"],
            "ee_position": observation_space["ee_position"],
            "ee_camera_orientation": observation_space["ee_camera_orientation"],
            "target_object_position": observation_space["target_object_position"],
            "actions": observation_space["actions"],
        }
        if activate_contact_sensor:
            normalization_spaces["contact_force_left_finger"] = observation_space["contact_force_left_finger"]
            normalization_spaces["contact_force_right_finger"] = observation_space["contact_force_right_finger"]

        self.scalers = nn.ModuleDict(
            {key: my_RunningStandardScaler(size=space, device=device) for key, space in normalization_spaces.items()}
        )
        if activate_depth:
            self.depth_scaler = my_RunningStandardScaler(size=1, device=device)

    def _get_single_layer_cache_size(self):
        # 计算 Mamba v1 单层缓存的大小
        config = self.mamba_config
        # h_cache shape: (B, ED, N)
        h_cache_size = config.d_inner * config.d_state
        # inputs_cache shape: (B, ED, d_conv-1)
        conv_cache_size = config.d_inner * (config.d_conv - 1)
        return h_cache_size + conv_cache_size

    def get_specification(self):
        single_layer_cache_size = self._get_single_layer_cache_size()
        total_cache_size = single_layer_cache_size * self.n_layers

        return {"rnn": {"sequence_length": self.sequence_length, "sizes": [(1, self.num_envs, total_cache_size)]}}

    def _pack_caches(self, caches_list):
        # caches_list: [(h1, c1), (h2, c2), ...]
        packed_tensors = []
        for h_cache, conv_cache in caches_list:
            batch_size = conv_cache.shape[0]
            # h_cache might be None initially
            if h_cache is None:
                h_cache_size = self.mamba_config.d_inner * self.mamba_config.d_state
                h_cache = torch.zeros(batch_size, h_cache_size, device=self.device)
            else:
                h_cache = h_cache.view(batch_size, -1)

            conv_flat = conv_cache.view(batch_size, -1)
            packed_tensors.append(torch.cat([h_cache, conv_flat], dim=1))
        return torch.cat(packed_tensors, dim=1).unsqueeze(0)

    def _unpack_caches(self, rnn_input_tensor):
        rnn_input_tensor = rnn_input_tensor.squeeze(0)

        single_layer_cache_size = self._get_single_layer_cache_size()
        batch_size = rnn_input_tensor.shape[0]

        per_layer_caches = torch.split(rnn_input_tensor, single_layer_cache_size, dim=1)

        unpacked_caches_list = []
        config = self.mamba_config
        h_cache_size = config.d_inner * config.d_state
        for packed_cache in per_layer_caches:
            h_flat = packed_cache[:, :h_cache_size]
            conv_flat = packed_cache[:, h_cache_size:]

            h_cache = h_flat.view(batch_size, config.d_inner, config.d_state)
            conv_cache = conv_flat.view(batch_size, config.d_inner, config.d_conv - 1)
            unpacked_caches_list.append((h_cache, conv_cache))

        return unpacked_caches_list

    def _extract_and_combine_features(self, space):
        # --- 向量状态归一化与处理 ---
        norm_joint_pos = self.scalers["joint_pos"](space["joint_pos"], train=self.training)
        norm_joint_vel = self.scalers["joint_vel"](space["joint_vel"], train=self.training)
        norm_actions = self.scalers["actions"](space["actions"], train=self.training)
        if self.perfect_position:
            norm_object_pos = self.scalers["object_position"](space["object_position_perfect"], train=self.training)
        else:
            norm_object_pos = self.scalers["object_position"](space["object_position"], train=self.training)
        if self.no_object_position:
            norm_object_pos = torch.zeros_like(space["object_position_perfect"])
        norm_ee_pos = self.scalers["ee_position"](space["ee_position"], train=self.training)
        norm_ee_camera_ori = self.scalers["ee_camera_orientation"](space["ee_camera_orientation"], train=self.training)
        norm_target_pos = self.scalers["target_object_position"](space["target_object_position"], train=self.training)
        if self.activate_contact_sensor:
            norm_left_finger_force = self.scalers["contact_force_left_finger"](
                space["contact_force_left_finger"], train=self.training
            )
            norm_right_finger_force = self.scalers["contact_force_right_finger"](
                space["contact_force_right_finger"], train=self.training
            )

        state_data1 = torch.cat([norm_joint_pos, norm_joint_vel, norm_actions], dim=-1).float()
        if self.activate_contact_sensor:
            state_data2 = torch.cat(
                [
                    norm_object_pos,
                    norm_ee_pos,
                    norm_ee_camera_ori,
                    norm_target_pos,
                    norm_left_finger_force,
                    norm_right_finger_force,
                ],
                dim=-1,
            ).float()
        else:
            state_data2 = torch.cat(
                [
                    norm_object_pos,
                    norm_ee_pos,
                    norm_ee_camera_ori,
                    norm_target_pos,
                    # norm_left_finger_force,
                    # norm_right_finger_force,
                ],
                dim=-1,
            ).float()
        feature1 = self.state_net1(state_data1)
        feature2 = self.state_net2(state_data2)

        # --- 图像与文本特征处理 ---
        if self.activate_depth:
            depth_log = torch.log10(space["depth_obs"] + 1e-5)
            norm_depth = self.depth_scaler(depth_log, train=self.training).float()
            feature_depth = self.depth_feature_embedding(self.depth_net(norm_depth.permute(0, 3, 1, 2)))
        feature_rgb = self.rgb_feature_embedding(space["rgb_feature"].float()).reshape(
            -1, 2 * self.dim_rgb_feature
        )  # 80 in original, seems a bug
        text_clip_f = self.clip_embedding(space["text_clip_feature"])
        image_clip_f = self.clip_embedding(space["image_clip_feature"]).reshape(-1, 2 * self.dim_clip_feature)

        # --- 特征融合 ---
        if self.activate_depth:
            combined_features = torch.cat(
                [feature1, feature2, feature_rgb, feature_depth, text_clip_f, image_clip_f], dim=-1
            )
        else:
            combined_features = torch.cat([feature1, feature2, feature_rgb, text_clip_f, image_clip_f], dim=-1)

        # 返回映射到 Mamba 维度的特征
        return self.pre_mamba_mlp(combined_features)

    def act(self, inputs, role):
        # 7. 实现 act 路由
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)
        # 可以添加其他 role，例如用于辅助任务
        # elif role == 'position': ...

    def compute(self, inputs, role):
        states = self.tensor_to_space(inputs["states"], self.observation_space)
        terminated = inputs.get("terminated", None)

        initial_cache_flat = inputs["rnn"][0]
        if self.training and not self.single_step:
            # --- 路径 A: PPO_RNN 序列更新模式 ---

            # 1. 获取序列的初始状态
            num_transitions_in_batch = initial_cache_flat.shape[1]
            indices = torch.arange(0, num_transitions_in_batch, self.sequence_length, device=self.device)
            initial_cache_for_seq = initial_cache_flat[:, indices, :]
            initial_caches = self._unpack_caches(initial_cache_for_seq)

            # 2. 重塑输入
            mamba_input = self._extract_and_combine_features(states)
            num_transitions = mamba_input.shape[
                0
            ]  # mini_batch_size = (memory_size * num_envs) / mini_batches,必须为self.sequence_length的整数倍
            num_sequences = num_transitions // self.sequence_length
            mamba_input = mamba_input.view(num_sequences, self.sequence_length, -1)

            if terminated is not None:
                terminated = terminated.view(num_sequences, self.sequence_length)

            # 3. 序列处理 (BPTT)
            outputs = []
            current_caches = initial_caches
            for i in range(self.sequence_length):
                x_step = mamba_input[:, i, :]
                step_caches = []
                if terminated is not None and i > 0:
                    finished_episodes_mask = terminated[:, i - 1].view(-1, 1, 1)
                    for h_cache, conv_cache in current_caches:
                        h_cache_reset = torch.where(finished_episodes_mask, 0.0, h_cache)
                        conv_cache_reset = torch.where(finished_episodes_mask, 0.0, conv_cache)
                        step_caches.append((h_cache_reset, conv_cache_reset))
                else:
                    step_caches = current_caches

                x_step, current_caches = self.mamba_backbone.step(x_step, step_caches)
                outputs.append(x_step.unsqueeze(1))

            mamba_output = torch.cat(outputs, dim=1).view(num_transitions, -1)
            final_caches = current_caches

        else:
            # --- 路径 B: 单步模式 (Rollout 或 BPTT 蒸馏) ---
            initial_caches = self._unpack_caches(initial_cache_flat)
            mamba_input = self._extract_and_combine_features(states)
            mamba_output, final_caches = self.mamba_backbone.step(mamba_input, initial_caches)
        final_packed_cache = self._pack_caches(
            current_caches if self.training and not self.single_step else final_caches
        )

        if role == "policy":
            mean = self.policy_head(mamba_output)
            return mean, self.log_std_parameter, {"rnn": [final_packed_cache]}
        elif role == "value":
            value = self.value_head(mamba_output)
            return value, {"rnn": [final_packed_cache]}
