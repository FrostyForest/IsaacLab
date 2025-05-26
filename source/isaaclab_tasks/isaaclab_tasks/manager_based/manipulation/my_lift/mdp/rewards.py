# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.manager_based.manipulation.my_lift.lift_env import LiftEnv


def object_is_lifted(
    env: LiftEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""

    object1: RigidObject = env.scene['yellow_object']
    object2: RigidObject = env.scene['green_object']
    object3: RigidObject = env.scene['red_object']

    r1=torch.where(object1.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)
    r2=torch.where(object2.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)
    r3=torch.where(object3.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)

    current_target_idx=env.current_target_ids_per_env#torch.Size([num_env])

    mapping_vectors = torch.tensor([
    [1.0, 0.2, 0.2],  # 对应输入 0
    [0.2, 1.0, 0.2],  # 对应输入 1
    [0.2, 0.2, 1.0]   # 对应输入 2
    ],device=env.device)
    if current_target_idx.dtype != torch.long:
       current_target_idx = current_target_idx.long()
    else:
        current_target_idx = current_target_idx
    weight_tensor=mapping_vectors[current_target_idx]

    r_div=torch.stack([r1,r2,r3],dim=1)

    r=torch.sum(torch.mul(weight_tensor,r_div),dim=-1).squeeze(-1)

    return r


def object_ee_distance(
    env: LiftEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    #物体与末端执行器之间的距离奖励
    # extract the used quantities (to enable type-hinting)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]

    object1: RigidObject = env.scene['yellow_object']
    object2: RigidObject = env.scene['green_object']
    object3: RigidObject = env.scene['red_object']
    # Target object position: (num_envs, 3)
    object1_pos_w = object1.data.root_pos_w
    object2_pos_w = object2.data.root_pos_w
    object3_pos_w = object3.data.root_pos_w
    # Distance of the end-effector to the object: (num_envs,)
    object1_ee_distance = torch.norm(object1_pos_w - ee_w, dim=1)
    object2_ee_distance = torch.norm(object2_pos_w - ee_w, dim=1)
    object3_ee_distance = torch.norm(object3_pos_w - ee_w, dim=1)

    r1=1 - torch.tanh(object1_ee_distance / std)
    r2=1 - torch.tanh(object2_ee_distance / std)
    r3=1 - torch.tanh(object3_ee_distance / std)

    current_target_idx=env.current_target_ids_per_env#torch.Size([num_env])

    mapping_vectors = torch.tensor([
    [1.0, 0.2, 0.2],  # 对应输入 0
    [0.2, 1.0, 0.2],  # 对应输入 1
    [0.2, 0.2, 1.0]   # 对应输入 2
    ],device=env.device)
    if current_target_idx.dtype != torch.long:
       current_target_idx = current_target_idx.long()
    else:
        current_target_idx = current_target_idx
    weight_tensor=mapping_vectors[current_target_idx]
    r_div=torch.stack([r1,r2,r3],dim=1)

    r=torch.sum(torch.mul(weight_tensor,r_div),dim=-1).squeeze(-1)

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
    #物体与目标位置之间的距离奖励
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)

    object1: RigidObject = env.scene['yellow_object']
    object2: RigidObject = env.scene['green_object']
    object3: RigidObject = env.scene['red_object']
    distance1 = torch.norm(des_pos_w - object1.data.root_pos_w[:, :3], dim=1)
    distance2 = torch.norm(des_pos_w - object2.data.root_pos_w[:, :3], dim=1)
    distance3 = torch.norm(des_pos_w - object3.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    r1=(object1.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance1 / std))#torch.Size([num_env])
    r2=(object2.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance2 / std))
    r3=(object3.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance3 / std))
    current_target_idx=env.current_target_ids_per_env#torch.Size([num_env])

    mapping_vectors = torch.tensor([
    [1.0, 0.25, 0.25],  # 对应输入 0
    [0.25, 1.0, 0.25],  # 对应输入 1
    [0.25, 0.25, 1.0]   # 对应输入 2
    ],device=env.device)
    if current_target_idx.dtype != torch.long:
       current_target_idx = current_target_idx.long()
    else:
        current_target_idx = current_target_idx
    weight_tensor=mapping_vectors[current_target_idx]
    r_div=torch.stack([r1,r2,r3],dim=1)
    r=torch.sum(torch.mul(weight_tensor,r_div),dim=-1).squeeze(-1)

    return r #shape:(n)

