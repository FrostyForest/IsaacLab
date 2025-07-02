# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Tuple

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.manager_based.manipulation.my_lift.lift_env import LiftEnv

import torch.nn.functional as F
from transformers import AutoProcessor

from isaaclab.sensors import ContactSensor, ContactSensorCfg, FrameTransformer

from . import image
from .string_target import ID_TO_TARGET, NUM_TARGETS, PREDEFINED_TARGETS, TARGET_TO_ID


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
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_pos_b, _ = subtract_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], ee_pos_w)
    return ee_pos_b


def image_feature_obs(env: LiftEnv) -> torch.Tensor:
    if hasattr(env, "current_target_state_per_env") and env.encoded_task_goal_per_env is not None:
        # text_list=env.current_target_strings_per_env
        raw_image_data = image(env, sensor_cfg=SceneEntityCfg("camera_1"), data_type="rgb", normalize=False)
        inputs = env.image_processor(images=raw_image_data, max_num_patches=64, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = env.siglip_image_model(**inputs)
        image_feature = outputs.pooler_output
        # similirity=F.cosine_similarity(text_embeddings, image_embeddings, dim=-1)
        # print("image feature shape",image_feature.shape)
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
        return torch.zeros((env.num_envs, env.feature_dim), dtype=torch.float32)


def rgb_obs(env: LiftEnv) -> torch.Tensor:
    raw_image_data = image(
        env, sensor_cfg=SceneEntityCfg("camera_1"), data_type="rgb", normalize=True
    )  # torch.Size([1, 256, 256, 3])
    downsampled_tensor_first = F.interpolate(
        raw_image_data.permute(0, 3, 1, 2),
        scale_factor=0.5,
        mode="bilinear",
        align_corners=False,  # 推荐在 scale_factor < 1 时设为 False
    ).permute(0, 2, 3, 1)
    return downsampled_tensor_first


def depth_obs(env: LiftEnv) -> torch.Tensor:
    raw_image_data = image(
        env, sensor_cfg=SceneEntityCfg("camera_1"), data_type="depth", normalize=True
    )  # torch.Size([1, 256, 256, 3])
    downsampled_nearest = F.interpolate(
        raw_image_data.permute(0, 3, 1, 2),  # 改成channel first
        scale_factor=0.5,
        mode="nearest",  # 关键点：使用 'nearest'
    ).permute(
        0, 2, 3, 1
    )  # 该会channel last

    return downsampled_nearest


def text_feature_obs(env: LiftEnv) -> torch.Tensor:
    if hasattr(env, "current_target_state_per_env") and env.encoded_task_goal_per_env is not None:
        # text_list=env.current_target_strings_per_env
        # return env.current_target_state_per_env
        return env.current_target_ids_per_env.unsqueeze(-1)
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
        # return torch.zeros((env.num_envs,env.feature_dim), dtype=torch.float32)
        return torch.zeros((env.num_envs, 1), dtype=torch.float32)


def get_cubes_position(env: LiftEnv) -> torch.Tensor:
    yellow_cube_pos = object_position_in_robot_root_frame(env, object_cfg=SceneEntityCfg("yellow_object")).squeeze(-1)
    green_cube_pos = object_position_in_robot_root_frame(env, object_cfg=SceneEntityCfg("green_object")).squeeze(-1)
    red_cube_pos = object_position_in_robot_root_frame(env, object_cfg=SceneEntityCfg("red_object")).squeeze(-1)

    pos_tensor = torch.cat([yellow_cube_pos, green_cube_pos, red_cube_pos], dim=1)  # (n,9)

    bottle_pos = object_position_in_robot_root_frame(env, object_cfg=SceneEntityCfg("bottle")).squeeze(-1)
    return bottle_pos


def get_finger_contact_forces(env: LiftEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    获取指定接触传感器上检测到的净法向接触力 (大小)。
    sensor_cfg.name 应该是你在 SceneCfg 中为 ContactSensor 定义的名称。
    """
    # 1. 获取 ContactSensor 实例
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # 2. 获取净法向接触力数据
    # sensor.data.net_forces_w 的形状是 (num_envs, num_sensor_bodies, 3)
    # 其中 num_sensor_bodies 是由 sensor_cfg.prim_path 匹配到的物体数量。
    # 如果 prim_path 只匹配一个物体 (例如一个手指), 则 num_sensor_bodies = 1.
    net_forces_w = sensor.data.net_forces_w

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
    force_vector_on_finger = net_forces_w[:, :, :]  # Shape: (num_envs, num_object,3)

    # 3. 计算力的大小 (L2 范数)
    force_magnitude = torch.norm(force_vector_on_finger, dim=-1, keepdim=False)  # Shape: (num_envs, 1)

    return force_magnitude
