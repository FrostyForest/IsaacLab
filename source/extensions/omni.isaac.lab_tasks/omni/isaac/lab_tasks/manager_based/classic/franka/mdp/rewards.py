# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)

def distance_robot2cube(
    env: ManagerBasedRLEnv
) -> torch.Tensor:
    robot=env.scene["robot"]
    cube=env.scene["cube"]

    robot_world_pos_default = robot.data.default_root_state[:, :3]
    cube_world_pos_default = cube.data.default_root_state[:, :3]
    distance0 = torch.linalg.norm(robot_world_pos_default[:, :3] - cube_world_pos_default[:, :3])

    robot_world_pos=robot.data.body_pos_w[:,8,:3]
    cube_world_pos = cube.data.root_pos_w

    distance=torch.linalg.norm(robot_world_pos[:, :3] - cube_world_pos[:, :3])
    return (distance0-distance)/(distance0+0.3)
