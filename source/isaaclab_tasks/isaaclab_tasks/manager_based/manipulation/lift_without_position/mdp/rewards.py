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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from isaaclab_tasks.manager_based.manipulation.my_lift.lift_env import LiftEnv


def object_is_lifted(
    env: LiftEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""

    object1: RigidObject = env.scene["yellow_object"]
    object2: RigidObject = env.scene["green_object"]
    object3: RigidObject = env.scene["red_object"]

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

    r = torch.sum(torch.mul(weight_tensor, r_div), dim=-1).squeeze(-1)  # 最大值为1,最小值为0
    # print("lift reward", r)
    r = r1
    env.extras["lift_ratio"] = torch.mean(r1)

    return r


def object_ee_distance(
    env: LiftEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # 物体与末端执行器之间的距离奖励
    # extract the used quantities (to enable type-hinting)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]

    object1: RigidObject = env.scene["yellow_object"]
    object2: RigidObject = env.scene["green_object"]
    object3: RigidObject = env.scene["red_object"]
    # Target object position: (num_envs, 3)
    object1_pos_w = object1.data.root_pos_w
    object2_pos_w = object2.data.root_pos_w
    object3_pos_w = object3.data.root_pos_w
    # Distance of the end-effector to the object: (num_envs,)
    object1_ee_distance = torch.norm(object1_pos_w - ee_w, dim=1)
    object2_ee_distance = torch.norm(object2_pos_w - ee_w, dim=1)
    object3_ee_distance = torch.norm(object3_pos_w - ee_w, dim=1)

    r1 = 1 - torch.tanh(object1_ee_distance / std)
    r2 = 1 - torch.tanh(object2_ee_distance / std)
    r3 = 1 - torch.tanh(object3_ee_distance / std)

    current_target_idx = env.current_target_ids_per_env  # torch.Size([num_env])

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
    r = r1  # 只需要黄色物体到末端的距离

    return r


def object_goal_distance(
    env: LiftEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # 物体与目标位置之间的距离奖励
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)

    object1: RigidObject = env.scene["yellow_object"]
    object2: RigidObject = env.scene["green_object"]
    object3: RigidObject = env.scene["red_object"]
    distance1 = torch.norm(des_pos_w - object1.data.root_pos_w[:, :3], dim=1)
    distance2 = torch.norm(des_pos_w - object2.data.root_pos_w[:, :3], dim=1)
    distance3 = torch.norm(des_pos_w - object3.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    r1 = (object1.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance1 / std))  # torch.Size([num_env])
    r2 = (object2.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance2 / std))
    r3 = (object3.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance3 / std))
    current_target_idx = env.current_target_ids_per_env  # torch.Size([num_env])

    mapping_vectors = torch.tensor(
        [  # target序号为0，代表目标为yellow cube，。。。
            [1.0, 1.0, 1.0],  # 对应输入 0
            [1.0, 1.0, 1.0],  # 对应输入 1
            [1.0, 1.0, 1.0],  # 对应输入 2
        ],
        device=env.device,
    )
    if current_target_idx.dtype != torch.long:
        current_target_idx = current_target_idx.long()
    else:
        current_target_idx = current_target_idx
    weight_tensor = mapping_vectors[current_target_idx]
    r_div = torch.stack([r1, r2, r3], dim=1)
    r_div = torch.mul(weight_tensor, r_div)
    r, i = r_div.max(dim=1, keepdim=False)
    r = r1  # 只需要黄色物体

    return r  # shape:(n)


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

    object1: RigidObject = env.scene["yellow_object"]
    object2: RigidObject = env.scene["green_object"]
    object3: RigidObject = env.scene["red_object"]
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

    object1: RigidObject = env.scene["yellow_object"]
    object2: RigidObject = env.scene["green_object"]
    object3: RigidObject = env.scene["red_object"]
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
