# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed


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
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
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


# define shared model (stochastic and deterministic models) using mixins
# --- Actor (Policy) Model for SAC ---
class StochasticActor(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        # 定义网络结构
        # 请确保这里的输入维度与你的环境观测空间匹配
        # 假设低维状态拼接后是37维
        self.state_net = nn.Sequential(nn.Linear(37, 64), nn.ELU(), nn.Linear(64, 16), nn.ELU())
        self.rgb_net = CustomFeatureExtractor(in_channels=16, output_dim=16)

        # 融合后的共享主干
        self.net = nn.Sequential(nn.Linear(16 + 16, 64), nn.ELU(), nn.Linear(64, 16), nn.ELU())

        # 策略头部
        self.mean_layer = nn.Linear(16, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        space = self.tensor_to_space(inputs["states"], self.observation_space)

        # 特征提取与融合 (与PPO模型类似)
        rgb_history = space["rgb_obs_history"].reshape(
            space["rgb_obs"].shape[0], space["rgb_obs"].shape[1], space["rgb_obs"].shape[2], -1
        )
        depth_history = space["depth_obs_history"].reshape(
            space["rgb_obs"].shape[0], space["rgb_obs"].shape[1], space["rgb_obs"].shape[2], -1
        )
        rgb_data = torch.cat([space["rgb_obs"], rgb_history, space["depth_obs"], depth_history], dim=-1).permute(
            0, 3, 1, 2
        )

        state_data = torch.cat(
            [
                space["joint_pos"],
                space["joint_vel"],
                space["ee_position"],
                space["target_object_position"],
                space["actions"],
                space["contact_force_left_finger"],
                space["contact_force_right_finger"],
            ],
            dim=-1,
        )

        state_features = self.state_net(state_data)
        image_features = self.rgb_net(rgb_data)

        fused_features = self.net(torch.cat([state_features, image_features], dim=-1))

        return self.mean_layer(fused_features), self.log_std_parameter, {}

    def act(self, inputs, role):
        return GaussianMixin.act(self, inputs, role)


# --- Critic (Q-function) Model for SAC ---
class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        # 状态特征提取器 (可以与Actor共享，但为了简单，这里独立定义)
        self.state_net = nn.Sequential(nn.Linear(37, 64), nn.ELU(), nn.Linear(64, 16), nn.ELU())
        self.rgb_net = CustomFeatureExtractor(in_channels=16, output_dim=16)

        # 融合网络，输入是 state_features + image_features + action
        self.net = nn.Sequential(
            nn.Linear(16 + 16, 64), nn.ELU(), nn.Linear(64, 16), nn.ELU(), nn.Linear(16, 1)
        )  # 输出单个 Q 值

    def compute(self, inputs, role):
        # Critic 的输入包含 states 和 taken_actions
        space = self.tensor_to_space(inputs["states"], self.observation_space)
        actions = inputs["taken_actions"]

        # 特征提取
        rgb_history = space["rgb_obs_history"].reshape(
            space["rgb_obs"].shape[0], space["rgb_obs"].shape[1], space["rgb_obs"].shape[2], -1
        )
        depth_history = space["depth_obs_history"].reshape(
            space["rgb_obs"].shape[0], space["rgb_obs"].shape[1], space["rgb_obs"].shape[2], -1
        )
        rgb_data = torch.cat([space["rgb_obs"], rgb_history, space["depth_obs"], depth_history], dim=-1).permute(
            0, 3, 1, 2
        )

        state_data = torch.cat(
            [
                space["joint_pos"],
                space["joint_vel"],
                space["ee_position"],
                space["target_object_position"],
                space["actions"],
                space["contact_force_left_finger"],
                space["contact_force_right_finger"],
            ],
            dim=-1,
        )

        state_features = self.state_net(state_data)
        image_features = self.rgb_net(rgb_data)

        # 融合状态特征和动作
        fused_input = torch.cat([state_features, image_features, actions], dim=-1)

        # 计算 Q 值
        return self.net(fused_input), {}

    def act(self, inputs, role):
        return DeterministicMixin.act(self, inputs, role)


# load and wrap the Isaac Lab environment
task_name = "Isaac-my_Lift-Cube-Franka-IK-Rel-v1"
env = load_isaaclab_env(task_name=task_name, num_envs=8)
env = wrap_env(env)

device = env.device


# --- 3. 修改 Memory 配置 ---
# SAC is off-policy, so it needs a larger replay buffer
memory_size = 1000  # e.g., 100k, 1M, etc.
memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)


# --- 4. 实例化 SAC 的模型 ---
models = {}
models["policy"] = StochasticActor(env.observation_space, env.action_space, device)
models["critic_1"] = Critic(env.observation_space, env.action_space, device)
models["critic_2"] = Critic(env.observation_space, env.action_space, device)


# --- 5. 修改 Agent 配置为 SAC ---
cfg = SAC_DEFAULT_CONFIG.copy()
# 这些是需要为你的任务仔细调整的超参数
cfg["gradient_steps"] = 1
cfg["batch_size"] = 256  # Off-policy 算法通常使用较大的 batch size
cfg["discount_factor"] = 0.99
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 5e-4  # SAC 学习率通常比PPO小
cfg["critic_learning_rate"] = 5e-4
cfg["entropy_learning_rate"] = 5e-4
cfg["learn_entropy"] = True
cfg["random_timesteps"] = 1000  # 在开始学习前进行一些随机探索
cfg["learning_starts"] = 1000  # 在收集到足够多的经验后再开始学习
cfg["grad_norm_clip"] = 5.0  # SAC有时会用稍大的梯度裁剪
cfg["rewards_shaper"] = None
# `state_preprocessor` 和 `value_preprocessor` 对于 off-policy 的 SAC 通常不使用，
# 因为价值目标 (target_values) 是在训练循环中动态计算的，
# 而不是像 PPO 的 GAE 那样预先计算。

# logging
cfg["experiment"]["write_interval"] = 250
cfg["experiment"]["checkpoint_interval"] = 2500
cfg["experiment"]["directory"] = f"runs/torch/sac_{task_name}"  # 使用不同的目录名

# --- 6. 实例化 SAC Agent ---
agent = SAC(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 67200, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
# path = '/home/linhai/code/IsaacLab/runs/torch/Isaac-my_Lift-Cube-Franka-v1/25-06-30_17-22-25-289545_PPO/checkpoints/best_agent.pt'
# agent.load(path)
# start training
trainer.train()


# # ---------------------------------------------------------
# # comment the code above: `trainer.train()`, and...
# # uncomment the following lines to evaluate a trained agent
# # ---------------------------------------------------------
# from skrl.utils.huggingface import download_model_from_huggingface

# # download the trained agent's checkpoint from Hugging Face Hub and load it
# path = download_model_from_huggingface("skrl/IsaacOrbit-Isaac-Lift-Franka-v0-PPO", filename="agent.pt")
# agent.load(path)

# # start evaluation
# trainer.eval()
