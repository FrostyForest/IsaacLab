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

# from models.model_without_image_distilling import Shared as model_without_img
from models.model_without_image import Shared as model_without_img
from models.model_without_image_with_preprocessor import Shared as model_without_img_with_preprocessor

# from models.model_with_image import Shared as model_with_img
from models.model_with_image import Shared as policy_model_teacher
from models.value_model_with_image import Shared as value_model_teacher
from models.model_with_image_shared_weight import Shared as Model
from skrl.envs.wrappers.torch import wrap_env
from skrl.resources.preprocessors.torch import RunningStandardScaler
import torch.distributions as D
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm

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
    is_teacher_model_without_image = True
    teacher_model_path = (
        "runs/distillation_learning/distillation_Isaac-my_Lift-Cube-Franka-v1_Jul30_15-43-46/checkpoints/best_model.pt"
    )

    state_dict = torch.load(teacher_model_path, map_location=device)

    # cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}

    # if is_teacher_model_without_image:#新模型架构不需要外置的预处理器
    #     cfg["state_preprocessor"] = RunningStandardScaler
    #     cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    #     state_preprocessor = cfg["state_preprocessor"](**cfg["state_preprocessor_kwargs"])
    #     state_preprocessor.load_state_dict(state_dict["state_preprocessor"])
    # else:
    #     state_preprocessor = None

    teacher_model_policy = model_without_img_with_preprocessor(env.observation_space, env.action_space, device).to(
        device
    )
    state_preprocessor = None  # 新模型架构不需要外置的预处理器
    teacher_model_policy.load_state_dict(state_dict["policy"])
    teacher_model_policy = teacher_model_policy.eval().to(device)

    teacher_model_value = teacher_model_policy
    teacher_model_value = teacher_model_value.eval().to(device)
    # state_preprocessor.load_state_dict(state_dict["state_preprocessor"])

    teacher_value_preprocessor = RunningStandardScaler(size=1, device=device)
    teacher_value_preprocessor.load_state_dict(state_dict["value_preprocessor"])
    teacher_value_preprocessor = teacher_value_preprocessor.eval()

    # student_model_path='runs/distillation_learning/distillation_Isaac-my_Lift-Cube-Franka-v1_Jul13_16-23-00/checkpoints/best_model.pt'
    student_model = Model(
        env.observation_space, env.action_space, device, perfect_position=True, no_object_position=True
    ).to(
        device
    )  # 当启用perfect_position的时候需要启用no_object_position
    # state_dict = torch.load(student_model_path, map_location=device)
    # student_model.load_state_dict(state_dict["policy"]).to(device)
    student_value_preprocessor = RunningStandardScaler(size=1, device=device)
    # **(可选但推荐)**: 将教师的统计数据复制给学生的预处理器
    # 这确保了如果我们在后续微调中加载这个模型，预处理器已经处于“热启动”状态。
    student_value_preprocessor.load_state_dict(teacher_value_preprocessor.state_dict())

    # 1. 初始化 TensorBoard SummaryWriter
    log_dir = os.path.join(
        "runs", "distillation_learning", "distillation_" + args_cli.task + "_" + time.strftime("%b%d_%H-%M-%S")
    )
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[INFO] TensorBoard logs will be saved to: {log_dir}")

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    student_optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4, weight_decay=0.01)
    # reset environment
    observations, _ = env.reset()
    global_step = 0
    save_interval = 20
    lift_ratios_over_interval = []
    best_avg_lift_ratio = 0

    optimization_batch_size = 5
    # Lists to store batch data
    batch_student_means = []
    batch_student_log_stds = []
    batch_teacher_means = []
    batch_teacher_log_stds = []
    batch_student_values = []  # <--- 新增
    batch_teacher_values = []  # <--- 新增
    batch_observations = []  # <--- 新增: 我们需要保存观测值来进行价值预测
    huber_loss_fn = torch.nn.HuberLoss(delta=0.05)
    pbar = tqdm(total=12000, desc="Distillation Training")
    # simulate environment
    while simulation_app.is_running():
        for _ in range(optimization_batch_size):
            batch_observations.append(observations)  # <--- 新增
            # run everything in inference mode
            student_actions, _, student_metadata = student_model.act({"states": observations}, role="policy")
            student_value_outputs, _, _ = student_model.act({"states": observations}, role="value")
            # mean_actions, log_std, outputs = student_model.compute({"states": observations}, role="policy")
            student_mean = student_metadata["mean_actions"]
            student_log_std = student_metadata["log_std"].unsqueeze(0).expand(student_mean.shape[0], -1)
            student_value = student_value_outputs  # <--- 新增
            # 创建学生分布对象
            # student_dist = D.Normal(student_mean, student_log_std.exp())

            with torch.no_grad():
                if state_preprocessor is not None:
                    observations_processed = state_preprocessor(observations, train=False)
                else:
                    observations_processed = observations
                # b. 模型输入需要是字典形式
                # 教师模型（继承自skrl.Model）的 act 方法期望一个字典
                # 并且我们通常取确定性动作（均值）进行评估
                _, _, teacher_metadata = teacher_model_policy.act({"states": observations_processed}, role="policy")
                teacher_value_outputs, _, _ = teacher_model_value.act({"states": observations}, role="value")
                # teacher_mean,teacher_std,output=teacher_model.compute({"states": observations_processed}, role="policy")
                teacher_mean = teacher_metadata.get("mean_actions").detach()
                teacher_log_std = (
                    teacher_metadata.get("log_std").unsqueeze(0).expand(teacher_mean.shape[0], -1).detach()
                )
                # teacher_dist = D.Normal(teacher_mean, teacher_log_std.exp().detach())
                teacher_value = teacher_value_outputs.detach()
            batch_student_means.append(student_mean)
            batch_student_log_stds.append(student_log_std)
            batch_teacher_means.append(teacher_mean)
            batch_teacher_log_stds.append(teacher_log_std)
            batch_student_values.append(student_value)  # <--- 新增
            batch_teacher_values.append(teacher_value)  # <--- 新增

            # distillation_loss = D.kl.kl_divergence(teacher_dist, student_dist).mean()
            # # 优化学生模型
            # student_optimizer.zero_grad()
            # distillation_loss.backward()
            # student_optimizer.step()

            # if global_step % 100 == 0:  # 每 10 步记录一次，避免日志文件过大
            #     writer.add_scalar("Loss/Distillation", distillation_loss.item(), global_step)

            with torch.no_grad():
                next_observations, reward, terminated, truncated, info = env.step(student_actions.detach())
                if global_step % 10 == 0:
                    writer.add_scalar("task metrics/lift ratio", info["lift_ratio"], global_step)
                lift_ratios_over_interval.append(info["lift_ratio"])
                observations = next_observations
                if terminated.any() or truncated.any():
                    observations, _ = env.reset()
                    # observations = observations.clone()

            # Check and save the best model every 1000 steps
            if global_step > 0 and global_step % save_interval == 0:
                if len(lift_ratios_over_interval) > 0:
                    avg_lift_ratio = sum(lift_ratios_over_interval) / len(lift_ratios_over_interval)
                    print(
                        f"Step {global_step}: Average lift ratio over last {save_interval} steps: {avg_lift_ratio:.4f}"
                    )

                    if avg_lift_ratio > best_avg_lift_ratio:
                        best_avg_lift_ratio = avg_lift_ratio
                        print(f"New best model found! Average lift ratio: {best_avg_lift_ratio:.4f}. Saving...")

                        # Define save path
                        save_dir = os.path.join(log_dir, "checkpoints")
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, "best_model.pt")

                        # Prepare the checkpoint dictionary
                        checkpoint = {
                            "policy": student_model.state_dict(),
                            "optimizer": student_optimizer.state_dict(),
                            "value_preprocessor": student_value_preprocessor.state_dict(),
                        }

                        # Save the checkpoint
                        torch.save(checkpoint, save_path)
                        print(f"Model saved to {save_path}")
                    lift_ratios_over_interval.clear()
            global_step += 1
            pbar.update(1)
        # --- New: Optimization Step ---
        # Concatenate the collected data into tensors
        student_means_tensor = torch.cat(batch_student_means, dim=0)
        student_log_stds_tensor = torch.cat(batch_student_log_stds, dim=0)
        teacher_means_tensor = torch.cat(batch_teacher_means, dim=0)
        teacher_log_stds_tensor = torch.cat(batch_teacher_log_stds, dim=0)
        student_values_tensor = torch.cat(batch_student_values, dim=0)  # <--- 新增
        teacher_values_tensor = torch.cat(batch_teacher_values, dim=0)  # <--- 新增
        observations_tensor = torch.cat(batch_observations, dim=0)  # <--- 新增

        num_optimizations = 3
        total_size = observations_tensor.shape[0]
        for i in range(num_optimizations):
            permuted_indices = torch.randperm(total_size, device=device)
            # 在每个优化步骤中，重新对整个批次进行前向传播以获取最新的梯度
            # 这是一个更标准的做法，而不是使用收集时的旧输出

            # 学生模型对整个批次进行前向传播
            student_policy_metadata = student_model.compute(
                {"states": observations_tensor[permuted_indices]}, role="policy"
            )
            student_value_outputs = student_model.compute(
                {"states": observations_tensor[permuted_indices]}, role="value"
            )
            student_position_outputs = student_model.compute(
                {"states": observations_tensor[permuted_indices]}, role="position"
            )

            current_student_mean = student_policy_metadata[0]
            current_student_log_std = student_policy_metadata[1].unsqueeze(0).expand(current_student_mean.shape[0], -1)
            current_student_value = student_value_outputs[0]
            predict_positions = student_position_outputs[0]
            predict_pos_dif = student_position_outputs[1]
            real_object_positions = student_position_outputs[2]
            real_ee_positions = student_position_outputs[3]

            # 创建分布
            student_dist = D.Normal(current_student_mean, current_student_log_std.exp())
            # 注意：教师的输出是固定的，不需要重新计算
            # print(teacher_means_tensor[permuted_indices].shape,teacher_log_stds_tensor[permuted_indices].shape)
            teacher_dist = D.Normal(
                teacher_means_tensor[permuted_indices], teacher_log_stds_tensor[permuted_indices].exp()
            )

            # **修改**: 计算总损失
            # 1. 策略蒸馏损失 (KL 散度)
            policy_distillation_loss = D.kl.kl_divergence(teacher_dist, student_dist).mean()

            # 2. **新增**: 价值蒸馏损失 (均方误差 MSE)
            # **重要**: 学生模型的价值输出也需要经过教师的预处理器进行标准化，
            #         才能和教师的原始输出在同一尺度上比较。或者，就像我们上面做的，
            #         将教师的输出反标准化，与学生模型的原始输出比较。后者更直接。
            value_distillation_loss = F.mse_loss(current_student_value, teacher_values_tensor[permuted_indices])

            position_loss = F.mse_loss(predict_positions, real_object_positions)
            pos_dif_loss = F.mse_loss(predict_pos_dif, real_ee_positions - real_object_positions)

            # 3. **新增**: 加权求和得到总损失
            # 你可以引入权重来平衡两个损失
            policy_loss_weight = 1.0
            value_loss_weight = 0.5  # 价值损失的权重通常设置得比策略损失小
            position_loss_weight = 1.0
            pos_dif_loss_weight = 1.0
            total_loss = (
                (policy_loss_weight * policy_distillation_loss)
                + (value_loss_weight * value_distillation_loss)
                + position_loss * position_loss_weight
                + pos_dif_loss * pos_dif_loss_weight
            )

            # 优化学生模型
            student_optimizer.zero_grad()
            total_loss.backward()
            student_optimizer.step()

            # **修改**: 记录所有损失
            writer.add_scalar("Loss/Total", total_loss.item(), global_step + i)
            writer.add_scalar("Loss/Policy_Distillation (KL)", policy_distillation_loss.item(), global_step + i)
            writer.add_scalar("Loss/Value_Distillation (MSE)", value_distillation_loss.item(), global_step + i)
            writer.add_scalar("Loss/Position_Distillation (MSE)", position_loss.item(), global_step + i)
            writer.add_scalar("Loss/Position_Dif_Loss (MSE)", pos_dif_loss.item(), global_step + i)

        # Clear the batch data lists for the next collection phase
        batch_student_means.clear()
        batch_student_log_stds.clear()
        batch_teacher_means.clear()
        batch_teacher_log_stds.clear()
        batch_student_values.clear()  # <--- 新增
        batch_teacher_values.clear()  # <--- 新增
        batch_observations.clear()  # <--- 新增

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
