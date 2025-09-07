# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# 用来测试实验不同因素对训练效果的影响
import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG, my_PPO, my_PPO_rank, PPO_ADAPTIVE_CONFIG
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
import wandb
from dotenv import load_dotenv

# 获取当前文件 (torch_cube_franka_ppo2.py) 的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件所在的目录 (train/)
current_dir = os.path.dirname(current_file_path)
# 获取项目根目录 (my_code/)，即 train/ 的父目录
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到 sys.path 的最前面
sys.path.insert(0, project_root)

from tools.utils import send_email_notification

# 引入模型结构
# from models.model_without_image import Shared
# from models.model_with_image import Shared
# from models.value_model_with_image import Shared as value_model

# from models.model_without_image_distilling import Shared

# seed for reproducibility
set_seed(21)  # e.g. `set_seed(42)` for fixed seed
train_with_img = "--enable_cameras" in sys.argv

# load and wrap the Isaac Lab environment
task_name = "Isaac-my_Lift-Cube-Franka-v1"  #
# task_name = "Isaac-my_Lift-Cube-Franka-IK-Rel-v1"
env = load_isaaclab_env(task_name=task_name, num_envs=1)  # 设置环境数量
env = wrap_env(env)
device = env.device

memory_size = 30
# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
cfg = PPO_ADAPTIVE_CONFIG.copy()
cfg["init_log_std"] = 0
if train_with_img == True:
    # from models.model_with_image import Shared
    # from models.value_model_with_image import Shared as value_model
    # models["policy"] = Shared(env.observation_space, env.action_space, device,from_scratch=False)
    # models["value"] = value_model(env.observation_space, env.action_space, device,from_scratch=False)  # same instance: shared model
    from models.model_with_image_shared_weight import Shared

    models["policy"] = Shared(
        env.observation_space, env.action_space, device, perfect_position=True, no_object_position=True
    )

    models["value"] = models["policy"]
else:
    from models.model_without_image import Shared

    models["policy"] = Shared(env.observation_space, env.action_space, device, cfg["init_log_std"])
    models["value"] = Shared(env.observation_space, env.action_space, device, cfg["init_log_std"])
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg["rollouts"] = memory_size  # memory_size
cfg["learning_epochs"] = 8
cfg["mini_batches"] = 4  # 96 * 4096 / 98304
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 1e-4
cfg["learning_rate_scheduler"] = KLAdaptiveLR
cfg["learning_rate_scheduler_kwargs"] = {
    "kl_threshold": 0.011,
    "min_lr": 1e-6,
    "max_lr": 1e-3,
}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.3
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["policy_loss_scale"] = 1
cfg["entropy_loss_scale"] = 0.001
cfg["value_loss_scale"] = 1.0
cfg["rankingloss_scale"] = 1.0
cfg["kl_threshold"] = 0.055
cfg["rewards_shaper"] = None
cfg["time_limit_bootstrap"] = False
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 100
cfg["experiment"]["checkpoint_interval"] = 200
cfg["experiment"]["directory"] = f"runs/torch/{task_name}_cameras_{train_with_img}"
cfg["optimizer"] = "muon"  # 设置优化器
cfg["activate_ranknet_loss"] = True
cfg["value_clip_ratio"] = 0.02
cfg["target_ranking_saturation_ratio"] = 0.2
cfg["timesteps"] = 32000
cfg["activate_cos_weight_grad_norm"] = False
cfg["cos_weight_cycle"] = 600
cfg["experiment"]["wandb"] = True
cfg["enable_adaptive_policy_ratio_clip"] = True
cfg["enable_adaptive_value_ratio_clip"] = True
cfg["enable_adaptive_policy_norm_clip"] = False

# 2. (可选但推荐) 提供传递给 wandb.init() 的参数
cfg["experiment"]["wandb_kwargs"] = {
    "project": "franka_lift_ppo_experiments",  # 你的 W&B 项目名称
    # "name": "run_with_lr_1e-3", # (可选) 为这次运行命名
    "entity": None,  # (可选) 如果是团队项目，填写你的团队名
    "config": {
        # W&B 会自动跟踪这些参数，方便后续比较实验
        "task_name": task_name,
        "num_envs": env.num_envs,
        "begin_learning_rate": cfg["learning_rate"],
        "learning_rate_scheduler": str(cfg["learning_rate_scheduler"]),
        "KLAdaptiveLR_kl_threshold": 0.011,
        "value_clip_ratio": cfg["value_clip_ratio"],
        "kl_threshold": cfg["kl_threshold"],
        "policy_clip_ratio_upband": cfg["policy_clip_ratio_upband"],
        "policy_clip_ratio_downband": cfg["policy_clip_ratio_downband"],
        "activate_ranknet_loss": cfg["activate_ranknet_loss"],
        "target_policy_norm": cfg["target_policy_norm"],
        "target_ranking_saturation_ratio": cfg["target_ranking_saturation_ratio"],
        "memory_size": memory_size,
        "model_class": models["policy"].__class__,
        "train_with_img": train_with_img,
        "activate_cos_weight_grad_norm": cfg["activate_cos_weight_grad_norm"],
        "init_log_std": cfg["init_log_std"],
        "enable_adaptive_policy_ratio_clip": cfg["enable_adaptive_policy_ratio_clip"],
    },
}

# agent = my_PPO(
#     models=models,
#     memory=memory,
#     cfg=cfg,
#     observation_space=env.observation_space,
#     action_space=env.action_space,
#     device=device,
#     only_value_update=False
# )
agent = my_PPO_rank(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
    only_value_update=False,
)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": cfg["timesteps"], "headless": True, "environment_info": "log"}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
if train_with_img:
    path = "runs/torch/Isaac-my_Lift-Cube-Franka-v1_cameras_True/25-08-02_02-01-49-623231_my_PPO_rank/checkpoints/best_agent.pt"
else:
    path = "runs/torch/Isaac-my_Lift-Cube-Franka-v1_cameras_False/25-07-30_02-09-48-130521_my_PPO_rank/checkpoints/best_agent.pt"
# # #仅加载模型权重
# checkpoint = torch.load(path, map_location=device)
# agent.models["policy"].load_state_dict(checkpoint["policy"])
# #---如有需要可单独加载价值模型
# value_model_path='runs/torch/Isaac-my_Lift-Cube-Franka-v1/25-07-18_10-37-56-949491_my_PPO_rank/checkpoints/best_agent.pt'
# check_point_value=torch.load(value_model_path, map_location=device)
# agent.models['value'].load_state_dict(checkpoint['value'])
# agent._value_preprocessor.load_state_dict(checkpoint['value_preprocessor'])

# #同时加载模型和优化器的状态
agent.load(path)

# start training
# trainer.train()
# email_subject = f"训练完成: {task_name}"
# email_content = (
#     f"The reinforcement learning training for the task '{task_name}' has successfully completed.\n\n"
#     f"Experiment Directory: {cfg['experiment']['directory']}\n"
#     f"Total Timesteps: {cfg_trainer['timesteps']}\n\n"
#     "Please check the logs and TensorBoard for detailed results."
# )
# recipient_email = "748773880@qq.com"
# send_email_notification(email_subject, email_content, recipient_email)

# # ---------------------------------------------------------
# # comment the code above: `trainer.train()`, and...
# # uncomment the following lines to evaluate a trained agent
# # ---------------------------------------------------------
# from skrl.utils.huggingface import download_model_from_huggingface

# # download the trained agent's checkpoint from Hugging Face Hub and load it
# path = download_model_from_huggingface("skrl/IsaacOrbit-Isaac-Lift-Franka-v0-PPO", filename="agent.pt")
# agent.load(path)

# # start evaluation
trainer.eval()
