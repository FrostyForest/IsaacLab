import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
from skrl.agents.torch.ppo.my_ppo_rnn import PPO_RNN
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


# seed for reproducibility
set_seed(21)  # e.g. `set_seed(42)` for fixed seed
import sys
import os

current_working_directory = os.path.join(os.getcwd(), "my_code")
sys.path.insert(0, current_working_directory)

from models.Isaac_my_Lift_Cube_Franka_v1.mamba_skrl_Isaac_my_Lift_Cube_Franka_v1 import SharedMamba
from models.Isaac_my_Lift_Cube_Franka_v1.rnn_skrl_Isaac_my_Lift_Cube_Franka_v1 import SharedRNN

# load and wrap the Isaac Lab environment
task_name = "Isaac-my_Lift-Cube-Franka-v1"  #
# task_name = "Isaac-my_Lift-Cube-Franka-IK-Rel-v1"
env = load_isaaclab_env(task_name=task_name)  # 设置环境数量
env = wrap_env(env)
device = env.device
bptt_length = 15
train = True

load_pretrain_weight = False
load_distill_model = False  # 只有第一次加载预训练蒸馏模型采用该方式
# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=bptt_length * 2, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = SharedMamba(
    env.observation_space,
    env.action_space,
    device,
    num_envs=env.num_envs,
    sequence_length=bptt_length,
    perfect_position=True,
    no_object_position=False,
    single_step=False,
).to(device)
if train == True:
    models["policy"].train()
else:
    models["policy"].eval()
# models["policy"] = SharedRNN(env.observation_space, env.action_space, device, num_envs=env.num_envs,sequence_length=50,perfect_position=True,no_object_position=True,rnn_hidden_size=256,rnn_num_layers=4)
models["value"] = models["policy"]


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = bptt_length * 2  # memory_size
cfg["learning_epochs"] = 8
cfg["mini_batches"] = 5
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 1e-4
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.015}
cfg["grad_norm_clip"] = 1
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.001
cfg["value_loss_scale"] = 0.5
cfg["kl_threshold"] = 0.1
# cfg["state_preprocessor"] = RunningStandardScaler
# cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 50
cfg["experiment"]["checkpoint_interval"] = 1000
cfg["experiment"]["directory"] = f"runs/torch/{task_name}_rnn"


agent = PPO_RNN(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)
if load_pretrain_weight:
    if load_distill_model:
        model_path = "runs/distillation_learning/bptt_distill_Isaac-my_Lift-Cube-Franka-v1_Aug06_00-47-18/checkpoints/best_model.pt"
        value_preprocessor_path = "runs/distillation_learning/distillation_Isaac-my_Lift-Cube-Franka-v1_Jul30_15-43-46/checkpoints/best_model.pt"
        checkpoint_value_preprocessor = torch.load(value_preprocessor_path, map_location=device)
        # 首次加载模型权重
        checkpoint = torch.load(model_path, map_location=device)
        agent.models["policy"].load_state_dict(checkpoint["policy"])
        agent._value_preprocessor.load_state_dict(checkpoint_value_preprocessor["value_preprocessor"])
    else:
        model_path = (
            "runs/torch/Isaac-my_Lift-Cube-Franka-v1_rnn/25-08-28_00-56-03-008881_PPO_RNN/checkpoints/best_agent.pt"
        )
        agent.load(model_path)

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 100000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

# # start training
if train == True:
    trainer.train()

# start evaluation
else:
    trainer.eval()
