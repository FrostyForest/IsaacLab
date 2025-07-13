# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""
import sys
import os

# # 获取当前文件 (torch_cube_franka_ppo2.py) 的绝对路径
# current_file_path = os.path.abspath()
# # 获取当前文件所在的目录 (train/)
# current_dir = os.path.dirname(current_file_path)
# 获取项目根目录 (my_code/)，即 train/ 的父目录
project_root = os.path.abspath("/home/linhai/code/IsaacLab/my_code")
# 将项目根目录添加到 sys.path 的最前面
sys.path.insert(0, project_root)

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import pprint
from models.model_without_image_distilling import Shared as model_without_img
from models.model_without_image import Shared as model_without_img_student
from skrl.envs.wrappers.torch import wrap_env
from skrl.resources.preprocessors.torch import RunningStandardScaler
import torch.distributions as D
from torch.utils.tensorboard import SummaryWriter
import time

torch.autograd.set_detect_anomaly(True)


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    cfg = {}

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = wrap_env(env)
    device = env.device

    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg["state_preprocessor"] = RunningStandardScaler

    state_preprocessor = cfg["state_preprocessor"](**cfg["state_preprocessor_kwargs"])

    teacher_model_path = (
        "runs/torch/Isaac-my_Lift-Cube-Franka-v1/25-07-11_17-18-21-810651_my_PPO/checkpoints/best_agent.pt"
    )
    teacher_model = model_without_img(env.observation_space, env.action_space, device)
    state_dict = torch.load(teacher_model_path, map_location=device)
    teacher_model.load_state_dict(state_dict["policy"])
    teacher_model = teacher_model.eval().to(device)
    state_preprocessor.load_state_dict(state_dict["state_preprocessor"])

    student_model = model_without_img_student(env.observation_space, env.action_space, device).to(device)

    # 1. 初始化 TensorBoard SummaryWriter
    log_dir = os.path.join("runs", "distillation_" + args_cli.task + "_" + time.strftime("%b%d_%H-%M-%S"))
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[INFO] TensorBoard logs will be saved to: {log_dir}")

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    student_optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4, weight_decay=0.01)
    # reset environment
    observations, _ = env.reset()
    global_step = 0

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        student_actions, _, student_metadata = student_model.act({"states": observations}, role="policy")
        # mean_actions, log_std, outputs = student_model.compute({"states": observations}, role="policy")
        student_mean = student_metadata["mean_actions"]
        student_log_std = student_metadata["log_std"]
        # 创建学生分布对象
        student_dist = D.Normal(student_mean, student_log_std.exp())

        with torch.no_grad():
            if state_preprocessor is not None:
                observations_processed = state_preprocessor(observations, train=False)
            # b. 模型输入需要是字典形式
            # 教师模型（继承自skrl.Model）的 act 方法期望一个字典
            # 并且我们通常取确定性动作（均值）进行评估
            _, _, teacher_metadata = teacher_model.act({"states": observations_processed}, role="policy")
            # teacher_mean,teacher_std,output=teacher_model.compute({"states": observations_processed}, role="policy")
            teacher_mean = teacher_metadata.get("mean_actions").detach()
            teacher_log_std = teacher_metadata.get("log_std").detach()
            teacher_log_std = teacher_metadata.get("log_std")
            teacher_dist = D.Normal(teacher_mean, teacher_log_std.exp().detach())

        distillation_loss = D.kl.kl_divergence(teacher_dist, student_dist).mean()
        # 优化学生模型
        student_optimizer.zero_grad()
        distillation_loss.backward()
        student_optimizer.step()

        if global_step % 100 == 0:  # 每 10 步记录一次，避免日志文件过大
            writer.add_scalar("Loss/Distillation", distillation_loss.item(), global_step)

        with torch.no_grad():
            next_observations, reward, terminated, truncated, info = env.step(student_actions.detach())
            writer.add_scalar("task metrics/lift ratio", info["lift_ratio"].item(), global_step)
            observations = next_observations
            if terminated.any() or truncated.any():
                observations, _ = env.reset()
                # observations = observations.clone()

        global_step += 1

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
