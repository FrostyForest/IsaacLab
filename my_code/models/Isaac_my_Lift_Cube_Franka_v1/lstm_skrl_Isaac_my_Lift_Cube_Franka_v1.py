import torch
import torch.nn as nn
import timm
import gymnasium as gym

# 导入 skrl 的相关组件
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

import sys
from pathlib import Path

# 假设你的项目结构是正确的
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))
from my_code.models import utils

# 为了可运行性，我们用 skrl 自带的 scaler
from my_code.models.my_running_standard_scaler import my_RunningStandardScaler


class SharedLSTM(GaussianMixin, DeterministicMixin, Model):
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
        # --- LSTM 相关的 skrl 参数 ---
        num_envs=1,
        sequence_length=16,
        # --- 自定义模型参数 ---
        perfect_position=True,
        no_object_position=False,
        lstm_hidden_size=256,
        lstm_num_layers=1,
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
        self.hidden_size = lstm_hidden_size
        self.num_layers = lstm_num_layers
        self.single_step = single_step

        # --- 特征提取器部分 (保持不变) ---
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

        # 将融合特征映射到 LSTM 的输入维度
        self.pre_lstm_mlp = nn.Linear(in_features=feature_dim, out_features=self.hidden_size)

        # 3. (MOD): 将 nn.RNN 替换为 nn.LSTM
        self.lstm = nn.LSTM(
            input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True
        )

        # 4. 策略和价值网络头 (保持不变)
        self.policy_head = nn.Sequential(
            utils.ResidualBlock(feature_size=self.hidden_size), nn.Linear(self.hidden_size, self.num_actions)
        )
        self.value_head = nn.Sequential(nn.Linear(self.hidden_size, 1))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        # 5. 观测值归一化 (保持不变)
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

    def get_specification(self):
        # 6. (MOD): 实现 get_specification 以包含 hidden state 和 cell state
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                # LSTM 有两个状态: hidden state (h) 和 cell state (c)
                "sizes": [
                    (self.num_layers, self.num_envs, self.hidden_size),  # hidden states (h)
                    (self.num_layers, self.num_envs, self.hidden_size),  # cell states (c)
                ],
            }
        }

    def _extract_and_combine_features(self, space):
        # (MOD): 方法名重命名为 _extract_and_combine_features
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

        # --- 图像与文本特征处理 ---
        depth_log = torch.log10(space["depth_obs"] + 1e-5)
        norm_depth = self.depth_scaler(depth_log, train=self.training).float()
        feature_depth = self.depth_feature_embedding(self.depth_net(norm_depth.permute(0, 3, 1, 2)))
        feature_rgb = self.rgb_feature_embedding(space["rgb_feature"].float()).reshape(-1, 2 * self.dim_rgb_feature)
        text_clip_f = self.clip_embedding(space["text_clip_feature"])
        image_clip_f = self.clip_embedding(space["image_clip_feature"]).reshape(-1, 2 * self.dim_clip_feature)

        # --- 特征融合 ---
        combined_features = torch.cat(
            [feature1, feature2, feature_rgb, feature_depth, text_clip_f, image_clip_f], dim=-1
        )

        # 返回映射到 LSTM 输入维度的特征
        return self.pre_lstm_mlp(combined_features)

    def act(self, inputs, role):
        # 7. 实现 act 路由 (保持不变)
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        # 8. 实现核心的 compute 方法 (适配 LSTM)

        # --- 步骤 1: 准备输入和缓存 ---
        states = self.tensor_to_space(inputs["states"], self.observation_space)
        terminated = inputs.get("terminated", None)

        # (MOD): 从 inputs["rnn"] 解包 hidden state 和 cell state
        hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

        # --- 步骤 2: 特征提取 ---
        lstm_input = self._extract_and_combine_features(states)

        # --- 步骤 3: 通过 LSTM 运行序列 ---
        if self.training and not self.single_step:
            # --- 路径 A: PPO_RNN 序列更新模式 ---
            num_transitions = lstm_input.shape[0]
            num_sequences = num_transitions // self.sequence_length

            if num_transitions % self.sequence_length != 0:
                raise ValueError(
                    f"Batch size ({num_transitions}) must be divisible by sequence length ({self.sequence_length})."
                )

            lstm_input = lstm_input.view(num_sequences, self.sequence_length, -1)

            # (MOD): 调整 hidden_states 和 cell_states 的形状
            hidden_states = hidden_states.view(self.num_layers, num_sequences, self.sequence_length, self.hidden_size)[
                :, :, 0, :
            ].contiguous()
            cell_states = cell_states.view(self.num_layers, num_sequences, self.sequence_length, self.hidden_size)[
                :, :, 0, :
            ].contiguous()

            if terminated is not None and torch.any(terminated):
                lstm_outputs = []
                terminated = terminated.view(num_sequences, self.sequence_length)
                indexes = (
                    [0]
                    + (terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist()
                    + [self.sequence_length]
                )

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    # (MOD): 传入和接收 (h, c) 元组
                    lstm_output_segment, (hidden_states, cell_states) = self.lstm(
                        lstm_input[:, i0:i1, :], (hidden_states, cell_states)
                    )

                    finished_episodes_mask = terminated[:, i1 - 1]
                    hidden_states[:, finished_episodes_mask, :] = 0
                    cell_states[:, finished_episodes_mask, :] = 0
                    lstm_outputs.append(lstm_output_segment)

                lstm_output = torch.cat(lstm_outputs, dim=1)
            else:
                lstm_output, (hidden_states, cell_states) = self.lstm(lstm_input, (hidden_states, cell_states))

            lstm_output = torch.flatten(lstm_output, start_dim=0, end_dim=1)
        else:
            # --- 路径 B: 单步模式 ---
            lstm_input = lstm_input.unsqueeze(1)
            lstm_output, (hidden_states, cell_states) = self.lstm(lstm_input, (hidden_states, cell_states))
            lstm_output = lstm_output.squeeze(1)

        # --- 步骤 4: 打包最终缓存 ---
        # (MOD): 将 hidden 和 cell state 都打包
        final_hidden_states = [hidden_states, cell_states]

        # --- 步骤 5: 根据 role 计算并返回结果 ---
        if role == "policy":
            mean = self.policy_head(lstm_output)
            return mean, self.log_std_parameter, {"rnn": final_hidden_states}
        elif role == "value":
            value = self.value_head(lstm_output)
            return value, {"rnn": final_hidden_states}
