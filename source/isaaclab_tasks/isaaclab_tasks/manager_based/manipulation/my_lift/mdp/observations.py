# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING,Tuple
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.manager_based.manipulation.my_lift.lift_env import LiftEnv
from .string_target import PREDEFINED_TARGETS, TARGET_TO_ID, ID_TO_TARGET, NUM_TARGETS
from transformers import AutoProcessor
from . import image
import torch.nn.functional as F
from isaaclab.sensors import FrameTransformer


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b

def ee_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
)-> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], ee_pos_w
    )
    return ee_pos_b

def image_feature_obs(env: LiftEnv) -> torch.Tensor:
    if hasattr(env, 'current_target_state_per_env') and env.encoded_task_goal_per_env is not None:
        #text_list=env.current_target_strings_per_env
        raw_image_data = image(env, sensor_cfg=SceneEntityCfg("camera_1"), data_type="rgb", normalize=False)
        inputs = env.image_processor(images=raw_image_data, max_num_patches=64, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = env.siglip_image_model(**inputs)
        image_feature = outputs.pooler_output
        #similirity=F.cosine_similarity(text_embeddings, image_embeddings, dim=-1)
        #print("image feature shape",image_feature.shape)
        # print("similarity",similirity.shape)
        return image_feature
    else:
    # 在 ObservationManager 的 _prepare_terms 阶段，如果属性尚未创建，
    # 返回一个具有正确“形状”的占位符张量，但只包含一个样本（或使用 env.cfg 中的 num_envs）。
    # _prepare_terms 只需要知道单个环境的观测维度。
    # 注意：env.num_envs 可能在 _prepare_terms 第一次调用时还不可用，
    # 所以最好是返回一个代表单个环境的形状。
    # ObservationManager 会处理批处理。
    # 或者，如果能安全地访问 cfg.scene.num_envs:
    # num_envs_for_shape = env.cfg.scene.num_envs
    # device_for_shape = env.device # device 也可能稍后才完全设置好

    # 最安全的方式是返回一个表示“单个环境的预期形状”的张量
    # Manager 会处理 num_envs 的批处理
    # print("[Debug current_env_target_encoded_obs] encoded_task_goal_per_env not found, returning placeholder shape.")
        return torch.zeros((env.num_envs,env.feature_dim), dtype=torch.float32)
    

def text_feature_obs(env: LiftEnv) -> torch.Tensor:
    if hasattr(env, 'current_target_state_per_env') and env.encoded_task_goal_per_env is not None:
        #text_list=env.current_target_strings_per_env
        #return env.current_target_state_per_env
        return env.current_target_ids_per_env
    else:
    # 在 ObservationManager 的 _prepare_terms 阶段，如果属性尚未创建，
    # 返回一个具有正确“形状”的占位符张量，但只包含一个样本（或使用 env.cfg 中的 num_envs）。
    # _prepare_terms 只需要知道单个环境的观测维度。
    # 注意：env.num_envs 可能在 _prepare_terms 第一次调用时还不可用，
    # 所以最好是返回一个代表单个环境的形状。
    # ObservationManager 会处理批处理。
    # 或者，如果能安全地访问 cfg.scene.num_envs:
    # num_envs_for_shape = env.cfg.scene.num_envs
    # device_for_shape = env.device # device 也可能稍后才完全设置好

    # 最安全的方式是返回一个表示“单个环境的预期形状”的张量
    # Manager 会处理 num_envs 的批处理
    # print("[Debug current_env_target_encoded_obs] encoded_task_goal_per_env not found, returning placeholder shape.")
        #return torch.zeros((env.num_envs,env.feature_dim), dtype=torch.float32)
        return torch.zeros((env.num_envs,), dtype=torch.float32)

def get_cubes_position(env: LiftEnv)-> torch.Tensor:
    yellow_cube_pos=object_position_in_robot_root_frame(env,object_cfg=SceneEntityCfg("yellow_object")).squeeze(-1)
    green_cube_pos=object_position_in_robot_root_frame(env,object_cfg=SceneEntityCfg("green_object")).squeeze(-1)
    red_cube_pos=object_position_in_robot_root_frame(env,object_cfg=SceneEntityCfg("red_object")).squeeze(-1)

    pos_tensor=torch.cat([yellow_cube_pos,green_cube_pos,red_cube_pos],dim=1)#(n,9)
    return pos_tensor
