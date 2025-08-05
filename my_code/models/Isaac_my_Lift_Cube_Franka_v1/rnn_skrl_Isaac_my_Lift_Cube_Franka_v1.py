import torch
import torch.nn as nn
import timm
import gymnasium as gym

# 导入 skrl 的相关组件
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))
from my_code.models import utils

# 为了可运行性，我们用 skrl 自带的 scaler
from skrl.resources.preprocessors.torch import RunningStandardScaler as my_RunningStandardScaler


class SharedRNN(GaussianMixin, DeterministicMixin, Model):
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
        # --- RNN 相关的 skrl 参数 ---
        num_envs=1,
        sequence_length=16,  # PPO mini_batch size
        # --- 自定义模型参数 ---
        perfect_position=True,
        no_object_position=False,
        rnn_hidden_size=256,  # 替换 d_model
        rnn_num_layers=1,  # 新增 RNN 层数参数
    ):
        # 1. 初始化基类
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.num_envs = num_envs
        self.sequence_length = sequence_length
        self.perfect_position = perfect_position
        self.no_object_position = no_object_position
        self.hidden_size = rnn_hidden_size
        self.num_layers = rnn_num_layers

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

        # 将融合特征映射到 RNN 的输入维度
        # RNN 的输入维度可以与隐藏层维度不同，这里为了简化设为相同
        self.pre_rnn_mlp = nn.Linear(in_features=feature_dim, out_features=self.hidden_size)

        # 3. 将 Mamba 替换为 nn.RNN
        self.rnn = nn.RNN(
            input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True
        )  # batch_first -> (batch, sequence, features)

        # 4. 策略和价值网络头 (输入维度从 d_model 变为 rnn_hidden_size)
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
        # 6. 实现 get_specification 来告诉 skrl RNN 隐藏状态的形状
        # 对于 nn.RNN，只有一个隐藏状态 h
        # 形状为 (D * num_layers, N, Hout)
        # D=1 (单向), N=num_envs, Hout=self.hidden_size
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [(self.num_layers, self.num_envs, self.hidden_size)],
            }
        }

    # Mamba 特有的 _pack_cache 和 _unpack_cache 方法不再需要，可以删除

    def _extract_and_combine_features(self, space):
        # 这个方法的功能保持不变，只是最后输出的特征将送入 pre_rnn_mlp
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

        # 返回映射到 RNN 输入维度的特征
        return self.pre_rnn_mlp(combined_features)

    def act(self, inputs, role):
        # 7. 实现 act 路由 (保持不变)
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        # 8. 实现核心的 compute 方法 (适配 RNN)

        # --- 步骤 1: 准备输入和缓存 ---
        states = self.tensor_to_space(inputs["states"], self.observation_space)
        terminated = inputs.get("terminated", None)

        # 从 inputs["rnn"] 解包 RNN 隐藏状态
        hidden_states = inputs["rnn"][0]  # 对于 RNN, 只有一个隐藏状态

        # --- 步骤 2: 特征提取 ---
        rnn_input = self._extract_and_combine_features(states)  # (N*L or N, rnn_hidden_size)

        # --- 步骤 3: 通过 RNN 运行序列 ---
        if self.training:
            # --- 修正开始 ---
            # batch_size_in_training 应该是指 mini-batch 中序列的数量，而不是总的 transition 数量
            num_transitions = rnn_input.shape[0]
            num_sequences = num_transitions // self.sequence_length

            # 断言确保可以完美切分
            if num_transitions % self.sequence_length != 0:
                raise ValueError(
                    f"The number of transitions ({num_transitions}) in the mini-batch "
                    f"is not divisible by the sequence length ({self.sequence_length})."
                )

            rnn_input = rnn_input.view(num_sequences, self.sequence_length, -1)

            # 同样，调整 hidden_states 的形状
            # skrl 传入的 hidden_states 是 (num_layers, total_transitions, hidden_size)
            # 我们需要的是 (num_layers, num_sequences, hidden_size) 作为初始状态
            # 这需要我们从 memory 中获取与序列开头对应的 hidden_states
            # skrl 的 PPO_RNN 已经帮我们做了这个工作，我们只需要正确地 reshape
            # initial hidden states from skrl have shape (D * num_layers, N, Hout)
            # where N is the number of sequences in the minibatch.

            # 让我们重新审视 skrl 传入的 hidden_states 形状
            # PPO_RNN.py L615
            # rnn_policy = {
            #     "rnn": [s.transpose(0, 1) for s in sampled_rnn_batches[i]], # s from memory is (batch_size, D*num_layers, Hout)
            #     "terminated": ...
            # }
            # 这意味着 transpose 后是 (D*num_layers, batch_size, Hout)
            # 这里的 batch_size 到底是 300 还是 6？
            # 让我们假设 skrl 已经正确处理了序列，那么 `hidden_states.shape[1]` 就应该是序列数 `num_sequences`

            # hidden_states 的形状是 (num_layers, total_transitions, hidden_size)
            # 我们需要的是对应每个序列开始的 hidden_states
            # skrl 的 `sample_all` 已经按序列排好了，所以我们只需要取每隔 `sequence_length` 的 hidden_states
            # 但 PPO_RNN 的实现更简单，它直接传递了所有 state，然后在模型中 reshape
            # 如 `torch_gymnasium_pendulumnovel_ppo_lstm.py` L80 所示

            batch_size_in_training = num_sequences  # 明确变量名

            # (D * num_layers, N*L, Hout) -> (D*num_layers, N, L, Hout)
            hidden_states = hidden_states.view(
                self.num_layers, batch_size_in_training, self.sequence_length, self.hidden_size
            )
            # 取每个序列的第一个 hidden state 作为初始状态
            hidden_states = hidden_states[:, :, 0, :].contiguous()

            if terminated is not None and torch.any(terminated):
                # 存在 episode 结束，需要分段处理
                rnn_outputs = []
                terminated = terminated.view(batch_size_in_training, self.sequence_length)
                # 找到所有需要重置状态的时间点
                indexes = (
                    [0]
                    + (terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist()
                    + [self.sequence_length]
                )

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    # 处理子序列
                    rnn_output_segment, hidden_states = self.rnn(rnn_input[:, i0:i1, :], hidden_states)
                    # 在子序列结束时，如果对应的 episode 结束了，就重置隐藏状态
                    finished_episodes_mask = terminated[:, i1 - 1]
                    hidden_states[:, finished_episodes_mask, :] = 0
                    rnn_outputs.append(rnn_output_segment)

                rnn_output = torch.cat(rnn_outputs, dim=1)
            else:
                # 序列中没有 episode 结束，直接处理
                rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)

            # 将输出展平以匹配后续 MLP
            rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)
        else:
            # Rollout 模式：只处理一个时间步 (L=1)
            rnn_input = rnn_input.unsqueeze(1)  # (N, d_model) -> (N, 1, d_model)
            rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)
            rnn_output = rnn_output.squeeze(1)

        # --- 步骤 4: 打包最终缓存 ---
        # 对于 nn.RNN, 最终缓存就是更新后的 hidden_states
        final_hidden_states = [hidden_states]

        # --- 步骤 5: 根据 role 计算并返回结果 ---
        if role == "policy":
            mean = self.policy_head(rnn_output)
            return mean, self.log_std_parameter, {"rnn": final_hidden_states}
        elif role == "value":
            value = self.value_head(rnn_output)
            return value, {"rnn": final_hidden_states}
