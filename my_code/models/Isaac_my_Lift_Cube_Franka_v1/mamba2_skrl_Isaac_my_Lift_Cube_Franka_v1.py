import torch
import torch.nn as nn
import timm
import gymnasium as gym

# 导入 skrl 和 mambapy 的相关组件
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

# --- 核心修改：导入 Mamba v2 相关类 ---
from mambapy.mamba2 import Mamba2, Mamba2Config

# ------------------------------------

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))
from my_code.models import utils
from my_code.models.my_running_standard_scaler import my_RunningStandardScaler


class SharedMamba2(GaussianMixin, DeterministicMixin, Model):
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
        sequence_length=16,
        # --- 自定义模型参数 ---
        perfect_position=True,
        no_object_position=False,
        d_model=256,
        d_head=64,  # Mamba2 参数
        n_layers=6,
        expand_factor=2,
        single_step=False,
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
        self.single_step = single_step

        # --- 特征提取器 (保持不变) ---
        self.dim_state_net1 = 48
        self.dim_state_net2 = 64
        self.dim_clip_feature = 48
        self.dim_rgb_feature = 32
        self.dim_depth_feature = 64
        self.state_net1 = nn.Sequential(nn.Linear(26, self.dim_state_net1))
        self.state_net2 = nn.Sequential(nn.Linear(19, self.dim_state_net2))
        self.depth_net = timm.create_model(
            "mobilenetv3_small_050.lamb_in1k", pretrained=False, in_chans=2, num_classes=0
        )
        self.rgb_feature_embedding = nn.Linear(in_features=1280, out_features=self.dim_rgb_feature)
        self.depth_feature_embedding = nn.Linear(in_features=1024, out_features=self.dim_depth_feature)
        self.clip_embedding = nn.Linear(in_features=768, out_features=self.dim_clip_feature)
        feature_dim = (
            self.dim_state_net1
            + self.dim_state_net2
            + 2 * self.dim_rgb_feature
            + 3 * self.dim_clip_feature
            + self.dim_depth_feature
        )
        self.dim_mamba = d_model
        self.pre_mamba_mlp = nn.Linear(in_features=feature_dim, out_features=self.dim_mamba)

        # --- 核心修改：实例化 Mamba2 模型 ---
        self.mamba_config = Mamba2Config(
            d_model=self.dim_mamba,
            n_layers=self.n_layers,
            d_head=d_head,
            expand_factor=expand_factor,
            # Mamba2 依赖 Triton 内核，如果遇到硬件兼容性问题，需要设置环境变量
            # use_mem_eff_path=True, # 默认为 True
            # device=device,
            # dtype=self.dtype # 确保数据类型一致
        )
        self.mamba_backbone = Mamba2(self.mamba_config)
        # ------------------------------------

        # --- 策略和价值头 (保持不变) ---
        self.policy_head = nn.Sequential(
            utils.ResidualBlock(feature_size=self.dim_mamba), nn.Linear(self.dim_mamba, self.num_actions)
        )
        self.value_head = nn.Sequential(nn.Linear(self.dim_mamba, 1))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        # --- 归一化 (保持不变) ---
        normalization_spaces = {
            "joint_pos": observation_space["joint_pos"],
            "joint_vel": observation_space["joint_vel"],
            "object_position": observation_space["object_position_perfect"],
            "ee_position": observation_space["ee_position"],
            "ee_camera_orientation": observation_space["ee_camera_orientation"],
            "target_object_position": observation_space["target_object_position"],
            "actions": observation_space["actions"],
            "contact_force_left_finger": observation_space["contact_force_left_finger"],
            "contact_force_right_finger": observation_space["contact_force_right_finger"],
        }
        self.scalers = nn.ModuleDict(
            {key: my_RunningStandardScaler(size=space, device=device) for key, space in normalization_spaces.items()}
        )
        self.depth_scaler = my_RunningStandardScaler(size=1, device=device)

    # --- 核心修改：适配 Mamba2 的缓存结构 ---
    def _get_single_layer_cache_size(self):
        config = self.mamba_config
        # h_cache shape: (B, n_heads, d_head, d_state)
        h_cache_size = config.n_heads * config.d_head * config.d_state
        # conv_cache shape: (B, conv_dim, d_conv)
        conv_dim = config.d_inner + 2 * config.n_groups * config.d_state
        conv_cache_size = conv_dim * config.d_conv
        return h_cache_size + conv_cache_size

    def get_specification(self):
        single_layer_cache_size = self._get_single_layer_cache_size()
        total_cache_size = single_layer_cache_size * self.n_layers
        return {"rnn": {"sequence_length": self.sequence_length, "sizes": [(1, self.num_envs, total_cache_size)]}}

    def _pack_caches(self, caches_list):
        packed_tensors = []
        for h_cache, conv_cache in caches_list:
            batch_size = h_cache.shape[0]
            h_flat = h_cache.view(batch_size, -1)
            conv_flat = conv_cache.view(batch_size, -1)
            packed_tensors.append(torch.cat([h_flat, conv_flat], dim=1))
        return torch.cat(packed_tensors, dim=1).unsqueeze(0)

    def _unpack_caches(self, rnn_input_tensor):
        rnn_input_tensor = rnn_input_tensor.squeeze(0)
        single_layer_cache_size = self._get_single_layer_cache_size()
        batch_size = rnn_input_tensor.shape[0]
        per_layer_caches = torch.split(rnn_input_tensor, single_layer_cache_size, dim=1)

        unpacked_caches_list = []
        config = self.mamba_config
        h_cache_size = config.n_heads * config.d_head * config.d_state
        conv_dim = config.d_inner + 2 * config.n_groups * config.d_state

        for packed_cache in per_layer_caches:
            h_flat = packed_cache[:, :h_cache_size]
            conv_flat = packed_cache[:, h_cache_size:]
            h_cache = h_flat.view(batch_size, config.n_heads, config.d_head, config.d_state)
            conv_cache = conv_flat.view(batch_size, conv_dim, config.d_conv)
            unpacked_caches_list.append((h_cache, conv_cache))

        return unpacked_caches_list

    # ---------------------------------------------

    def _extract_and_combine_features(self, space):
        # (此方法保持不变)
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
        norm_left_finger_force = self.scalers["contact_force_left_finger"](
            space["contact_force_left_finger"], train=self.training
        )
        norm_right_finger_force = self.scalers["contact_force_right_finger"](
            space["contact_force_right_finger"], train=self.training
        )
        state_data1 = torch.cat([norm_joint_pos, norm_joint_vel, norm_actions], dim=-1).float()
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
        feature1 = self.state_net1(state_data1)
        feature2 = self.state_net2(state_data2)
        depth_log = torch.log10(space["depth_obs"] + 1e-5)
        norm_depth = self.depth_scaler(depth_log, train=self.training).float()
        feature_depth = self.depth_feature_embedding(self.depth_net(norm_depth.permute(0, 3, 1, 2)))
        feature_rgb = self.rgb_feature_embedding(space["rgb_feature"].float()).reshape(-1, 2 * self.dim_rgb_feature)
        text_clip_f = self.clip_embedding(space["text_clip_feature"])
        image_clip_f = self.clip_embedding(space["image_clip_feature"]).reshape(-1, 2 * self.dim_clip_feature)
        combined_features = torch.cat(
            [feature1, feature2, feature_rgb, feature_depth, text_clip_f, image_clip_f], dim=-1
        )
        return self.pre_mamba_mlp(combined_features)

    def act(self, inputs, role):
        # (此方法保持不变)
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        states = self.tensor_to_space(inputs["states"], self.observation_space)
        terminated = inputs.get("terminated", None)
        initial_cache_flat = inputs["rnn"][0]

        if self.training and not self.single_step:
            # --- 路径 A: PPO_RNN 序列更新模式 ---
            num_transitions_in_batch = initial_cache_flat.shape[1]
            indices = torch.arange(0, num_transitions_in_batch, self.sequence_length, device=self.device)
            initial_cache_for_seq = initial_cache_flat[:, indices, :]
            initial_caches = self._unpack_caches(initial_cache_for_seq)

            mamba_input = self._extract_and_combine_features(states)
            num_transitions = mamba_input.shape[0]
            num_sequences = num_transitions // self.sequence_length
            mamba_input = mamba_input.view(num_sequences, self.sequence_length, -1)

            if terminated is not None:
                terminated = terminated.view(num_sequences, self.sequence_length)

            outputs = []
            current_caches = initial_caches
            # --- 核心修改：Mamba2 step 调用 ---
            # mambapy Mamba2 的 step 接口是 `layer.step(x, cache)`
            # 而 Mamba2 类的 forward 方法可以处理 cache，但没有 step
            # 我们需要手动遍历层
            for i in range(self.sequence_length):
                x_step = mamba_input[:, i, :]

                # --- 逐层 step ---
                next_step_caches = []
                for layer_idx in range(self.n_layers):
                    layer = self.mamba_backbone.layers[layer_idx]
                    cache_for_layer = current_caches[layer_idx]

                    if terminated is not None and i > 0:
                        finished_mask = terminated[:, i - 1]
                        if finished_mask.any():
                            h, c = cache_for_layer
                            h[finished_mask] = 0
                            c[finished_mask] = 0
                            cache_for_layer = (h, c)

                    x_step, new_cache = layer.step(x_step, cache_for_layer)
                    next_step_caches.append(new_cache)
                current_caches = next_step_caches
                outputs.append(x_step.unsqueeze(1))

            mamba_output = torch.cat(outputs, dim=1).view(num_transitions, -1)
            final_caches = current_caches

        else:
            # --- 路径 B: 单步模式 (Rollout 或 BPTT 蒸馏) ---
            initial_caches = self._unpack_caches(initial_cache_flat)
            mamba_input = self._extract_and_combine_features(states)
            # [修复] 增加序列长度维度 (L=1)
            mamba_input_seq = mamba_input.unsqueeze(1)

            mamba_output, final_caches = self.mamba_backbone(mamba_input_seq, initial_caches)
            mamba_output = mamba_output.view(-1, self.dim_mamba)

        final_packed_cache = self._pack_caches(
            current_caches if self.training and not self.single_step else final_caches
        )

        if role == "policy":
            mean = self.policy_head(mamba_output)
            return mean, self.log_std_parameter, {"rnn": [final_packed_cache]}
        elif role == "value":
            value = self.value_head(mamba_output)
            return value, {"rnn": [final_packed_cache]}
