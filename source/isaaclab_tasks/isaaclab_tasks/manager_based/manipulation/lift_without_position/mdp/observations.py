# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from typing import TYPE_CHECKING, Tuple

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms, quat_apply

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
    print(object_cfg.name, object_pos_b)
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


def calcualte_object_pos_from_depth(env: LiftEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # 首先要计算camera的位姿
    robot: RigidObject = env.scene[robot_cfg.name]
    camera = env.scene["camera_1"]
    camera_pos_w = camera.data.pos_w[:, :3]
    camera_rot_w = camera.data.quat_w_world
    camera_pos_b, camera_rot_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], camera_pos_w, camera_rot_w
    )
    # 获取深度图
    raw_depth_data = image(env, sensor_cfg=SceneEntityCfg("camera_1"), data_type="depth", normalize=True)  # 单位米

    # 获取中点的深度
    # 3. 计算中点坐标
    batch_size, img_height, img_width, _ = raw_depth_data.shape
    center_y = img_height // 2  # 结果是 128
    center_x = img_width // 2  # 结果是 128
    midpoint_coords = torch.tensor([center_x, center_y]).to("cuda").repeat(batch_size, 1)
    # 注意：PyTorch/NumPy 的索引通常是 [row, column]，对应 [height, width]

    # 4. 进行索引以获取中点的深度值
    #    选择所有环境 (:)
    #    在 height 维度上选择 center_y
    #    在 width 维度上选择 center_x
    #    在 channel 维度上选择 0
    center_depth_values = raw_depth_data[:, center_y, center_x, 0]
    # print(center_depth_values,center_depth_values.shape)

    def get_point_in_robot_frame_from_pixel(
        pixel_coords: torch.Tensor,  # shape: (N, 2) or (2,)
        pixel_depth: torch.Tensor,  # shape: (N,) or scalar
        # camera_intrinsics: torch.Tensor,# shape: (N, 3, 3) or (3, 3)
        camera_pos_in_robot_frame: torch.Tensor,  # shape: (N, 3)
        camera_rot_in_robot_frame: torch.Tensor,  # shape: (N, 4)
        image_height,
        image_width,
    ) -> torch.Tensor:
        """
        Converts a pixel coordinate with its depth value to a 3D point in the robot's root frame.
        """
        # 确保输入是批处理的张量
        if pixel_coords.ndim == 1:
            pixel_coords = pixel_coords.unsqueeze(0)
        if pixel_depth.ndim == 0:
            pixel_depth = pixel_depth.unsqueeze(0)
        if camera_pos_in_robot_frame.ndim == 1:
            camera_pos_in_robot_frame = camera_pos_in_robot_frame.unsqueeze(0)
        if camera_rot_in_robot_frame.ndim == 1:
            camera_rot_in_robot_frame = camera_rot_in_robot_frame.unsqueeze(0)

        # 1. 反投影到相机坐标系
        fx = 1462.857142857143
        fy = 1462.857142857143
        cx = image_width // 2
        cy = image_height // 2

        u = pixel_coords[:, 0]
        v = pixel_coords[:, 1]
        d = pixel_depth

        z_c = d
        x_c = (u - cx) * z_c / fx
        y_c = (v - cy) * z_c / fy

        point_pos_c = torch.stack([x_c, y_c, z_c], dim=1)  # 物体在camera坐标系的位置
        print("物体在camera坐标系的位置", point_pos_c)

        # 2. 变换到机器人基座坐标系
        rotated_point = quat_apply(camera_rot_in_robot_frame, point_pos_c)
        point_pos_b = rotated_point + camera_pos_in_robot_frame

        return point_pos_b, point_pos_c

    def transform_points_from_camera_to_world_batch(
        pts_cam: torch.Tensor, cam_pos_in_world: torch.Tensor, cam_quat_in_world: torch.Tensor
    ) -> torch.Tensor:
        """
        批量将相机坐标系下的点，变换到世界坐标系。

        参数:
        pts_cam (torch.Tensor): 形状为 [N, 3] 的点坐标张量。
        cam_pos_in_world (torch.Tensor): 形状为 [N, 3] 的相机位置张量。
        cam_quat_in_world (torch.Tensor): 形状为 [N, 4] 的相机姿态四元数 (w, x, y, z) 张量。

        返回:
        torch.Tensor: 形状为 [N, 3] 的点在世界坐标系下的坐标张量。
        """
        N = pts_cam.shape[0]
        device = pts_cam.device

        # --- Step 1: 批量从位姿构建 c2w 矩阵 ---

        # 1. 批量归一化四元数
        # 使用 F.normalize 来确保所有四元数都是单位的
        q = torch.nn.functional.normalize(cam_quat_in_world, p=2, dim=1)

        # 提取 w, x, y, z 分量，每个都是 [N] 的向量
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        # 2. 批量计算旋转矩阵 R_c2w
        # 创建一个 [N, 3, 3] 的零矩阵来存储所有旋转矩阵
        R_c2w = torch.zeros((N, 3, 3), device=device)

        # 利用张量操作，一次性计算所有矩阵的元素
        R_c2w[:, 0, 0] = 1 - 2 * (y**2 + z**2)
        R_c2w[:, 0, 1] = 2 * (x * y - w * z)
        R_c2w[:, 0, 2] = 2 * (x * z + w * y)

        R_c2w[:, 1, 0] = 2 * (x * y + w * z)
        R_c2w[:, 1, 1] = 1 - 2 * (x**2 + z**2)
        R_c2w[:, 1, 2] = 2 * (y * z - w * x)

        R_c2w[:, 2, 0] = 2 * (x * z - w * y)
        R_c2w[:, 2, 1] = 2 * (y * z + w * x)
        R_c2w[:, 2, 2] = 1 - 2 * (x**2 + y**2)

        # 3. 批量组装 c2w 矩阵
        # 创建一个 [N, 4, 4] 的单位矩阵批次
        c2w = torch.eye(4, device=device).unsqueeze(0).repeat(N, 1, 1)

        c2w[:, :3, :3] = R_c2w
        # .unsqueeze(-1) 是为了将 [N, 3] 的平移向量变成 [N, 3, 1] 以便赋值
        c2w[:, :3, 3] = cam_pos_in_world

        # --- Step 2: 批量变换点 ---

        # 1. 批量将点转换为齐次坐标
        # 创建一个 [N, 1] 的1向量
        ones = torch.ones((N, 1), device=device)
        # 拼接后得到 [N, 4] 的齐次坐标点
        pts_cam_h = torch.cat([pts_cam, ones], dim=1)

        # 2. 应用 c2w 变换
        # 为了使用批量矩阵乘法，我们需要将 pts_cam_h 的形状调整为 [N, 4, 1]
        # c2w 的形状是 [N, 4, 4]
        # 结果 pts_world_h 的形状将是 [N, 4, 1]
        pts_world_h = torch.bmm(c2w, pts_cam_h.unsqueeze(-1))

        # 3. 返回世界坐标系下的三维坐标
        # .squeeze(-1) 去掉最后的维度，然后取前三维
        return pts_world_h.squeeze(-1)[:, :3]

    object_pos, pixel_pos_camera = get_point_in_robot_frame_from_pixel(
        pixel_coords=midpoint_coords,
        pixel_depth=center_depth_values,
        camera_pos_in_robot_frame=camera_pos_b,
        camera_rot_in_robot_frame=camera_rot_b,
        image_height=img_height,
        image_width=img_width,
    )
    # 应该有问题
    point_pos_world = transform_points_from_camera_to_world_batch(
        pts_cam=pixel_pos_camera, cam_pos_in_world=camera_pos_w, cam_quat_in_world=camera_rot_w
    )
    print("point_pos_world", point_pos_world)
    print("robot_pos_world", robot.data.root_state_w[:, :3])
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], point_pos_world
    )
    print("predict object pose in root", object_pos.shape, object_pos)  # 相比point_pos_world没变化
    print("predict object pose2 in root", object_pos_b.shape, object_pos_b)
    print("camera pose in root", camera_pos_b)  # 应该正确
    return object_pos
