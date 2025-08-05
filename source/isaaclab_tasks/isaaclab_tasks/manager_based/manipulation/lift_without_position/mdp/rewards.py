# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, FrameTransformer
from isaaclab.utils.math import combine_frame_transforms
from .string_target import ID_TO_TARGET, NUM_TARGETS, PREDEFINED_TARGETS, TARGET_TO_ID

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from isaaclab_tasks.manager_based.manipulation.my_lift.lift_env import LiftEnv


def object_is_lifted(env: LiftEnv, minimal_height: float) -> torch.Tensor:
    """Reward the agent for lifting the TARGET object above the minimal height."""
    # 1. 获取所有物体的引用
    objects = [env.scene[name] for name in PREDEFINED_TARGETS]

    # 2. 计算每个物体的“是否被举起”的布尔张量
    #    这里我们用一个列表来存储每个物体的奖励
    per_object_lifted_rewards = [torch.where(obj.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0) for obj in objects]

    # 3. 将所有物体的奖励堆叠成一个张量
    #    形状: (num_envs, num_objects) -> (num_envs, 3)
    all_rewards = torch.stack(per_object_lifted_rewards, dim=1)

    # 4. 获取当前每个环境的目标ID
    #    形状: (num_envs,)
    #    确保它是 long 类型以便用作索引
    target_ids = env.current_target_ids_per_env.long()

    # 5. 【核心】使用 gather 根据目标ID选择正确的奖励
    #    - target_ids.unsqueeze(-1) 将形状变为 (num_envs, 1)
    #    - torch.gather 会沿着 dim=1，根据 target_ids 中的索引，从 all_rewards 中取出对应的奖励
    #    - 例如，如果 env 0 的 target_id 是 1 (green)，它就会从 all_rewards[0] 中取出索引为 1 的那个奖励值
    target_specific_reward = torch.gather(all_rewards, 1, target_ids.unsqueeze(-1)).squeeze(-1)

    # 6. 更新 extras 用于监控（可选，但推荐）
    #    这里我们监控所有物体的平均举起比例，以及目标物体的举起比例
    # env.extras["lift_ratio/yellow"] = torch.mean(per_object_lifted_rewards[0])
    # env.extras["lift_ratio/green"] = torch.mean(per_object_lifted_rewards[1])
    # env.extras["lift_ratio/red"] = torch.mean(per_object_lifted_rewards[2])
    env.extras["lift_ratio"] = torch.mean(target_specific_reward).item()  # 这是最重要的监控指标

    return target_specific_reward


def object_ee_distance(
    env: LiftEnv, std: float, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    """Reward the agent for reaching the TARGET object using a tanh-kernel."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]

    # 1. 获取所有物体的引用
    objects = [env.scene[name] for name in PREDEFINED_TARGETS]
    # 2. 计算末端到每个物体的距离奖励
    per_object_dist_rewards = [
        1 - torch.tanh(torch.norm(obj.data.root_pos_w - ee_pos_w, dim=1) / std) for obj in objects
    ]

    # 3. 堆叠
    all_rewards = torch.stack(per_object_dist_rewards, dim=1)

    # 4. 获取目标ID
    target_ids = env.current_target_ids_per_env.long()

    # 5. 【核心】使用 gather 选择目标物体的奖励
    target_specific_reward = torch.gather(all_rewards, 1, target_ids.unsqueeze(-1)).squeeze(-1)

    # 6. 监控（可选）
    env.extras["ee_object_distance"] = torch.mean(
        torch.norm(objects[target_ids[0].item()].data.root_pos_w - ee_pos_w, dim=1)
    ).item()

    return target_specific_reward


def object_goal_distance(
    env: LiftEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose with the TARGET object."""
    robot: Articulation = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)

    # 计算目标位置
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)

    # 1. 获取所有物体的引用
    objects = [env.scene[name] for name in PREDEFINED_TARGETS]
    # 2. 计算每个物体到目标位置的距离奖励，并乘以“是否被举起”的条件
    per_object_goal_rewards = [
        (obj.data.root_pos_w[:, 2] > minimal_height)
        * (1 - torch.tanh(torch.norm(des_pos_w - obj.data.root_pos_w[:, :3], dim=1) / std))
        for obj in objects
    ]

    # 3. 堆叠
    all_rewards = torch.stack(per_object_goal_rewards, dim=1)

    # 4. 获取目标ID
    target_ids = env.current_target_ids_per_env.long()

    # 5. 【核心】使用 gather 选择目标物体的奖励
    target_specific_reward = torch.gather(all_rewards, 1, target_ids.unsqueeze(-1)).squeeze(-1)

    # 6. 监控（可选）
    # 使用 all_rewards 可以监控智能体是否在错误地移动非目标物体
    # non_target_mask = torch.ones_like(all_rewards, dtype=torch.bool)
    # non_target_mask.scatter_(1, target_ids.unsqueeze(-1), False)
    # env.extras["goal_distance_reward/non_target_mean"] = all_rewards[non_target_mask].mean()

    return target_specific_reward


def my_action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    penalty = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    penalty = torch.clamp(penalty, max=30)
    return penalty


def my_joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    p = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    p = torch.clamp(p, max=60)
    return p


def touch_object(
    env: LiftEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    # 1. 获取 ContactSensor 实例
    sensor1: ContactSensor = env.scene.sensors[SceneEntityCfg("left_finger_contactsensor").name]
    sensor2: ContactSensor = env.scene.sensors[SceneEntityCfg("right_finger_contactsensor").name]

    # 2. 获取净法向接触力数据
    # sensor.data.net_forces_w 的形状是 (num_envs, num_sensor_bodies, 3)
    # 其中 num_sensor_bodies 是由 sensor_cfg.prim_path 匹配到的物体数量。
    # 如果 prim_path 只匹配一个物体 (例如一个手指), 则 num_sensor_bodies = 1.
    def get_combine_force(contact_sensor):
        net_forces_w = contact_sensor.data.net_forces_w
        if net_forces_w is None:
            # print(f"Warning: net_forces_w for sensor '{sensor_cfg.name}' is None. Returning zeros.")
            # 需要确定在传感器未正确读取数据时，返回的观测的维度。
            # 假设一个手指对应一个传感器，我们返回一个标量（力的大小）或一个3D向量。
            # 为了简单，我们返回力的大小。如果 prim_path 匹配多个，你需要调整。
            # 假设我们期望每个传感器只关联一个身体，所以取 [:, 0, :]
            # 如果传感器可能未初始化，返回一个形状为 (num_envs, 1) 的零张量
            # 或者，如果你的传感器肯定会附着到单个物体上，那么可以预期 (num_envs, 1, 3)
            # 更好的做法是在env的 __init__ 中就确定好这个观测的维度。
            # 为了简单起见，如果传感器数据无效，我们返回0。
            # 但在 _prepare_terms 阶段，这需要一个固定形状。

            # 在 _prepare_terms 阶段，我们需要知道单个环境的输出维度
            # 假设我们返回单个手指的接触力大小（标量）
            # print(f"[ContactObs Debug] Sensor '{sensor_cfg.name}' net_forces_w is None. Returning scalar zero shape for prepare_terms.")
            return torch.zeros(1, device=env.device if hasattr(env, "device") else "cpu")  # 单个环境，单个标量

        # 假设 prim_path 只匹配一个物体（例如一个手指）
        # net_forces_w 形状是 (num_envs, 1, 3)
        # 我们取这个单一身体上的力
        force_vector_on_finger = net_forces_w[:, :, :]  # Shape: (num_envs,num_object, 3)
        # 3. 计算力的大小 (L2 范数)
        force_magnitude = torch.norm(force_vector_on_finger, dim=-1, keepdim=False)  # Shape: (num_envs, num_object)
        return force_magnitude

    force1 = get_combine_force(sensor1)
    force2 = get_combine_force(sensor2)

    force_combine = torch.mul(force1, force2)
    condition_for_where = torch.any(force_combine > 1, dim=1, keepdim=True)
    # 如果条件为 True，则值为 0.2，否则为 0.0 (或者你希望的其他默认值)
    value_if_true = torch.tensor(1)
    value_if_false = torch.tensor(0.0)  # 假设条件不满足时为0
    reward = torch.where(condition_for_where, value_if_true, value_if_false).squeeze(-1)

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]

    object1: RigidObject = env.scene["yellow_cube"]
    object2: RigidObject = env.scene["green_cube"]
    object3: RigidObject = env.scene["red_cube"]
    # Target object position: (num_envs, 3)
    object1_pos_w = object1.data.root_pos_w
    object2_pos_w = object2.data.root_pos_w
    object3_pos_w = object3.data.root_pos_w
    # Distance of the end-effector to the object: (num_envs,)
    object1_ee_distance = torch.norm(object1_pos_w - ee_w, dim=1)
    object2_ee_distance = torch.norm(object2_pos_w - ee_w, dim=1)
    object3_ee_distance = torch.norm(object3_pos_w - ee_w, dim=1)

    r1 = 1 - torch.tanh((object1_ee_distance - 0.02) / 0.1)
    r2 = 1 - torch.tanh((object2_ee_distance - 0.02) / 0.1)
    r3 = 1 - torch.tanh((object3_ee_distance - 0.02) / 0.1)

    # current_target_idx=env.current_target_ids_per_env#torch.Size([num_env])

    # mapping_vectors = torch.tensor([
    # [1.0, 0.0, 0.0],  # 对应输入 0
    # [0.0, 1.0, 0.0],  # 对应输入 1
    # [0.0, 0.0, 1.0]   # 对应输入 2
    # ],device=env.device)
    # if current_target_idx.dtype != torch.long:
    #    current_target_idx = current_target_idx.long()
    # else:
    #     current_target_idx = current_target_idx
    # weight_tensor=mapping_vectors[current_target_idx]
    r_div = torch.stack([r1, r2, r3], dim=1)

    # r=torch.sum(torch.mul(weight_tensor,r_div),dim=-1).squeeze(-1)
    r, i = r_div.max(dim=1, keepdim=False)

    reward = torch.mul(r1, reward)  # 最大值为1,最小值为0

    return reward


@torch.no_grad()
def lift_ratio_obs(env: LiftEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""

    object1: RigidObject = env.scene["yellow_cube"]
    object2: RigidObject = env.scene["green_cube"]
    object3: RigidObject = env.scene["red_cube"]
    minimal_height = 0.05

    r1 = torch.where(object1.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)
    r2 = torch.where(object2.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)
    r3 = torch.where(object3.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)

    current_target_idx = env.current_target_ids_per_env  # torch.Size([num_env])

    mapping_vectors = torch.tensor(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], device=env.device  # 对应输入 0  # 对应输入 1  # 对应输入 2
    )
    if current_target_idx.dtype != torch.long:
        current_target_idx = current_target_idx.long()
    else:
        current_target_idx = current_target_idx
    weight_tensor = mapping_vectors[current_target_idx]

    r_div = torch.stack([r1, r2, r3], dim=1)

    r = torch.sum(torch.mul(weight_tensor, r_div), dim=-1).squeeze(-1)
    # print("lift reward", r)
    env.extras["lift_ratio"] = torch.mean(r)
    print(env.extras)
    return r
