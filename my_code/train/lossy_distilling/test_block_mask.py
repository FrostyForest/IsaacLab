import sys
import os

os.environ["TRITON_F32_DEFAULT"] = "ieee"
# 设置项目根目录路径
project_root = os.path.abspath("/home/linhai/code/IsaacLab")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import time
from tqdm import tqdm
import torch
from isaaclab.app import AppLauncher

# --- 命令行参数解析 ---
parser = argparse.ArgumentParser(description="Distillation learning for Isaac Lab environments.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--num_envs", type=int, default=30, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-my_Lift-Cube-Franka-v1", help="Name of the task.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
args_cli = parser.parse_args()
from my_code.models.Isaac_my_Lift_Cube_Franka_v1.mamba_restrict_skrl_Isaac_my_Lift_Cube_Franka_v1 import (
    SharedMamba as mask_mamba,
)
from isaaclab_tasks.utils import parse_env_cfg
import gymnasium as gym
from skrl.envs.wrappers.torch import wrap_env

env_cfg = parse_env_cfg(
    args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
)
env = gym.make(args_cli.task, cfg=env_cfg)
env = wrap_env(env)
device = env.device
bptt_length = 15
model = mask_mamba(
    env.observation_space,
    env.action_space,
    device,
    num_envs=env.num_envs,
    sequence_length=bptt_length,
    perfect_position=True,
    no_object_position=True,
    n_layers=6,
    single_step=True,
).to(device)

path = "/home/linhai/code/IsaacLab/runs/distillation_learning_mask/bptt_distill_Isaac-my_Lift-Cube-Franka-v1_Aug25_11-12-59/checkpoints/best_model.pt"
state_dict = torch.load(path, map_location=device)
model.load_state_dict(state_dict["policy"])
print("mask形状与数值", model.masking_logits.shape, model.masking_logits)
