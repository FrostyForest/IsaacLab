# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# 用来测试实验不同因素对训练效果的影响
import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG, my_PPO
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist

import sys
import os

# 获取当前文件 (torch_cube_franka_ppo2.py) 的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件所在的目录 (train/)
current_dir = os.path.dirname(current_file_path)
# 获取项目根目录 (my_code/)，即 train/ 的父目录
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到 sys.path 的最前面
sys.path.insert(0, project_root)

# 引入模型结构
# from models.model_without_image import Shared
from models.model_with_image import Shared

# from models.model_without_image_distilling import Shared

# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed


# load and wrap the Isaac Lab environment
task_name = "Isaac-my_Lift-Cube-Franka-v1"  #
# task_name = "Isaac-my_Lift-Cube-Franka-IK-Rel-v1"
env = load_isaaclab_env(task_name=task_name, num_envs=52)  # 设置环境数量
env = wrap_env(env)

device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=30, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = Shared(env.observation_space, env.action_space, device)
models["value"] = models["policy"]  # same instance: shared model


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 30  # memory_size
cfg["learning_epochs"] = 10
cfg["mini_batches"] = 4  # 96 * 4096 / 98304
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 1e-3
cfg["learning_rate_scheduler"] = KLAdaptiveLR
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01, "min_lr": 2e-6}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["policy_loss_scale"] = 2
cfg["entropy_loss_scale"] = 0.001
cfg["value_loss_scale"] = 1.0
cfg["kl_threshold"] = 0
cfg["rewards_shaper"] = None
cfg["time_limit_bootstrap"] = False
# cfg["state_preprocessor"] = RunningStandardScaler
# cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 100
cfg["experiment"]["checkpoint_interval"] = 2500
cfg["experiment"]["directory"] = f"runs/torch/{task_name}"
cfg["optimizer"] = "muon"  # 设置优化器

agent = my_PPO(
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
# path = "runs/torch/Isaac-my_Lift-Cube-Franka-v1/25-07-11_17-18-21-810651_my_PPO/checkpoints/best_agent.pt"
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
#
