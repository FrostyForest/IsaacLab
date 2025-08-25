"""
Script to train a student RNN/Mamba model via distillation from a pre-trained teacher model,
using Truncated Backpropagation Through Time (TBPTT) to learn temporal dependencies.
"""

"""Launch Isaac Sim Simulator first."""
import sys
import os

# 设置项目根目录路径
project_root = os.path.abspath("/home/linhai/code/IsaacLab")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import time
from tqdm import tqdm
from isaaclab.app import AppLauncher

# --- 命令行参数解析 ---
parser = argparse.ArgumentParser(description="Distillation learning for Isaac Lab environments.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--num_envs", type=int, default=30, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-my_Lift-Cube-Franka-v1", help="Name of the task.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# --- 启动 Isaac Sim ---
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.distributions as D
from torch.utils.tensorboard import SummaryWriter

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# 导入模型
from my_code.models.Isaac_my_Lift_Cube_Franka_v1.mamba_skrl_Isaac_my_Lift_Cube_Franka_v1 import SharedMamba
from my_code.models.Isaac_my_Lift_Cube_Franka_v1.rnn_skrl_Isaac_my_Lift_Cube_Franka_v1 import SharedRNN
from my_code.models.Isaac_my_Lift_Cube_Franka_v1.lstm_skrl_Isaac_my_Lift_Cube_Franka_v1 import SharedLSTM
from my_code.models.model_without_image_with_preprocessor import Shared as TeacherModel  # 假设教师模型路径
from skrl.envs.wrappers.torch import wrap_env


def main():
    """Distillation learning with Truncated BPTT."""

    # --- 环境配置和加载 ---
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = wrap_env(env)
    device = env.device
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Observation space: {env.observation_space}")
    print(f"[INFO] Action space: {env.action_space}")

    # --- 教师模型加载 ---
    print("[INFO] Loading teacher model...")
    teacher_model_path = (
        "runs/distillation_learning/distillation_Isaac-my_Lift-Cube-Franka-v1_Jul30_15-43-46/checkpoints/best_model.pt"
    )
    state_dict = torch.load(teacher_model_path, map_location=device)
    teacher_model_policy = TeacherModel(env.observation_space, env.action_space, device).to(device)
    teacher_model_policy.load_state_dict(state_dict["policy"])
    teacher_model_policy = teacher_model_policy.eval()
    teacher_model_value = teacher_model_policy
    print("[INFO] Teacher model loaded successfully.")

    # --- 学生模型实例化 ---
    bptt_length = 15  # 定义 BPTT 的序列长度
    print(f"[INFO] Initializing student Mamba model with BPTT length: {bptt_length}")

    student_model = SharedLSTM(
        env.observation_space,
        env.action_space,
        device,
        num_envs=env.num_envs,
        sequence_length=bptt_length,
        perfect_position=True,
        no_object_position=True,
        lstm_num_layers=6,
        lstm_hidden_size=256,
        single_step=True,
    ).to(device)

    student_model_path = (
        "runs/distillation_learning/bptt_distill_Isaac-my_Lift-Cube-Franka-v1_Aug05_18-11-02/checkpoints/best_model.pt"
    )
    # state_dict = torch.load(student_model_path, map_location=device)
    # student_model.load_state_dict(state_dict['policy'])
    student_optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4, weight_decay=0.01)
    print("[INFO] Student model and optimizer initialized.")

    # --- 日志和保存设置 ---
    log_dir = os.path.join(
        "runs", "distillation_learning", "bptt_distill_" + args_cli.task + "_" + time.strftime("%b%d_%H-%M-%S")
    )
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[INFO] TensorBoard logs will be saved to: {log_dir}")

    # --- 训练循环变量初始化 ---
    observations, _ = env.reset()
    global_step = 0
    save_interval = 200
    best_avg_lift_ratio = -1.0
    lift_ratios_over_interval = []

    # 初始化学生模型的隐藏状态
    spec = student_model.get_specification()["rnn"]
    # spec["sizes"] 是 [(h_shape), (c_shape)]
    hidden_state_0 = torch.zeros(spec["sizes"][0], device=device)
    cell_state_0 = torch.zeros(spec["sizes"][1], device=device)
    student_caches = [hidden_state_0, cell_state_0]  # <--- 现在是一个列表

    # 权重因子
    policy_loss_weight = 1.0
    value_loss_weight = 0.5

    pbar = tqdm(total=200000, desc="TBPTT Distillation Training")

    while simulation_app.is_running():

        # --- 步骤 1: 收集数据和累积损失 (BPTT 序列) ---
        student_model.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_lift_ratio = 0.0

        # 将初始缓存从计算图中分离，作为本序列的起点
        initial_caches_for_bptt = [h.detach() for h in student_caches]

        for step_in_bptt in range(bptt_length):
            # 前向传播 (保持在计算图中)
            student_input = {"states": observations, "rnn": student_caches}
            policy_meta = student_model.compute(student_input, role="policy")
            value_meta = student_model.compute(student_input, role="value")

            student_mean, student_log_std, new_cache_meta = policy_meta
            student_value, _ = value_meta
            student_dist = D.Normal(student_mean, student_log_std.exp())

            # 获取教师模型的输出 (不跟踪梯度)
            with torch.no_grad():
                teacher_input = {"states": observations}
                _, _, teacher_meta = teacher_model_policy.act(teacher_input, role="policy")
                teacher_value_out, _, _ = teacher_model_value.act(teacher_input, role="value")
                teacher_mean = teacher_meta.get("mean_actions")
                teacher_log_std = teacher_meta.get("log_std").unsqueeze(0).expand_as(teacher_mean)
                teacher_value = teacher_value_out
                teacher_dist = D.Normal(teacher_mean, teacher_log_std.exp())

            # 计算并累积当前步的损失
            total_policy_loss += D.kl.kl_divergence(teacher_dist, student_dist).mean()
            total_value_loss += F.mse_loss(student_value, teacher_value)

            # --- 与环境交互 ---
            with torch.no_grad():
                actions_to_step = student_mean.detach()
                next_observations, reward, terminated, truncated, info = env.step(actions_to_step)
                observations = next_observations
                total_lift_ratio += info["lift_ratio"]
                # 更新隐藏状态 (保持在计算图中)
                student_caches = new_cache_meta["rnn"]

                # 如果 episode 结束，重置状态 (乘以 (1-done) 以保持在计算图中)
                done_mask = (terminated | truncated).view(1, -1, 1).float()
                student_caches[0] = student_caches[0] * (1.0 - done_mask)
                student_caches[1] = student_caches[1] * (1.0 - done_mask)

                # --- 日志和进度条 ---
                if "lift_ratio" in info:
                    lift_ratios_over_interval.append(info["lift_ratio"])
                global_step += 1
                pbar.update(1)

        # --- 步骤 2: 优化 ---

        # 对累加的损失进行平均
        avg_policy_loss = total_policy_loss / bptt_length
        avg_value_loss = total_value_loss / bptt_length
        total_loss = (policy_loss_weight * avg_policy_loss) + (value_loss_weight * avg_value_loss)
        avg_lift_ratio = total_lift_ratio / bptt_length

        # 一次性反向传播和优化
        student_optimizer.zero_grad()
        total_loss.backward()
        # (可选) 梯度裁剪
        # torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
        student_optimizer.step()

        # 将最终的缓存 detach，作为下一次 BPTT 序列的起点
        student_caches = [h.detach() for h in student_caches]

        # --- 步骤 3: 记录和保存 ---
        if global_step % 20 == 0:  # 每 10 次 BPTT 记录一次
            writer.add_scalar("Loss/Total", total_loss.item(), global_step)
            writer.add_scalar("Loss/Policy_Distillation (KL)", avg_policy_loss.item(), global_step)
            writer.add_scalar("Loss/Value_Distillation (MSE)", avg_value_loss.item(), global_step)
            writer.add_scalar("task metrics/lift ratio", info["lift_ratio"], global_step)

        if global_step > 0 and global_step % save_interval == 0:
            if len(lift_ratios_over_interval) > 0:
                avg_lift_ratio = torch.mean(torch.tensor(lift_ratios_over_interval)).item()
                writer.add_scalar("Metrics/Avg_Lift_Ratio", avg_lift_ratio, global_step)
                print(
                    f"\nStep {global_step}: Avg lift ratio over last {len(lift_ratios_over_interval)} steps: {avg_lift_ratio:.4f}"
                )

                if avg_lift_ratio > best_avg_lift_ratio:
                    best_avg_lift_ratio = avg_lift_ratio
                    print(f"New best model! Avg lift ratio: {best_avg_lift_ratio:.4f}. Saving...")
                    save_dir = os.path.join(log_dir, "checkpoints")
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, "best_model.pt")
                    checkpoint = {"policy": student_model.state_dict(), "optimizer": student_optimizer.state_dict()}
                    torch.save(checkpoint, save_path)
                    print(f"Model saved to {save_path}")
                lift_ratios_over_interval.clear()

    # --- 清理 ---
    pbar.close()
    env.close()
    writer.close()
    print("[INFO] Training finished.")


if __name__ == "__main__":
    main()
    simulation_app.close()
