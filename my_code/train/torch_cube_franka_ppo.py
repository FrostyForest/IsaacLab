# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
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
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.state_net = nn.Sequential(
            nn.Linear(37, 64),  # ik rel input 37，原38
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
        )
        self.rgb_net = CustomFeatureExtractor(in_channels=16, output_dim=16)
        self.net = nn.Sequential(nn.Linear(48, 64), nn.ELU(), nn.Linear(64, 32), nn.ELU())

        self.mean_layer = nn.Sequential(
            nn.Linear(32, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions))

        self.value_layer = nn.Sequential(
            nn.Linear(32, 1),
        )

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):

        space = self.tensor_to_space(inputs["states"], self.observation_space)
        if role == "policy":
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
                    space["contact_force_left_finger"],
                ],
                dim=-1,
            )

            feature1 = self.state_net(state_data)
            feature2 = self.rgb_net(rgb_data)

            self._shared_output = self.net(torch.cat([feature1, feature2], dim=-1))
            return self.mean_layer(self._shared_output), self.log_std_parameter, {}
        elif role == "value":
            # shared_output = self.net(inputs["states"]) if self._shared_output is None else self._shared_output
            if self._shared_output is None:
                rgb_history = space["rgb_obs_history"].reshape(
                    space["rgb_obs"].shape[0], space["rgb_obs"].shape[1], space["rgb_obs"].shape[2], -1
                )
                depth_history = space["depth_obs_history"].reshape(
                    space["rgb_obs"].shape[0], space["rgb_obs"].shape[1], space["rgb_obs"].shape[2], -1
                )
                rgb_data = torch.cat(
                    [space["rgb_obs"], rgb_history, space["depth_obs"], depth_history], dim=-1
                ).permute(0, 3, 1, 2)
                state_data = torch.cat(
                    [
                        space["joint_pos"],
                        space["joint_vel"],
                        space["ee_position"],
                        space["target_object_position"],
                        space["actions"],
                        space["contact_force_left_finger"],
                        space["contact_force_left_finger"],
                    ],
                    dim=-1,
                )

                feature1 = self.state_net(state_data)
                feature2 = self.rgb_net(rgb_data)
                shared_output = self.net(torch.cat([feature1, feature2], dim=-1))
            else:
                shared_output = self._shared_output
            self._shared_output = None
            return self.value_layer(shared_output), {}


# load and wrap the Isaac Lab environment
task_name = "Isaac-my_Lift-Cube-Franka-IK-Rel-v1"  # "Isaac-my_Lift-Cube-Franka-v1"
env = load_isaaclab_env(task_name=task_name, num_envs=96)  # 设置环境数量
env = wrap_env(env)

device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=40, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = Shared(env.observation_space, env.action_space, device)
models["value"] = models["policy"]  # same instance: shared model


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 40  # memory_size
cfg["learning_epochs"] = 18
cfg["mini_batches"] = 6  # 96 * 4096 / 98304
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 1e-3
cfg["learning_rate_scheduler"] = KLAdaptiveLR
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01, "min_lr": 1e-5}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.01
cfg["value_loss_scale"] = 2.0
cfg["kl_threshold"] = 0
cfg["rewards_shaper"] = None
cfg["time_limit_bootstrap"] = False
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 100
cfg["experiment"]["checkpoint_interval"] = 2500
cfg["experiment"]["directory"] = f"runs/torch/{task_name}"

agent = PPO(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 67200, "headless": True, "environment_info": "log"}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
# path = '/home/linhai/code/IsaacLab/runs/torch/Isaac-my_Lift-Cube-Franka-IK-Rel-v1/25-07-01_14-41-03-440280_PPO/checkpoints/best_agent.pt'
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
