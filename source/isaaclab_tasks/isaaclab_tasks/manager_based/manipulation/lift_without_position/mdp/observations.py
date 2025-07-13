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
    from isaaclab_tasks.manager_based.manipulation.lift_without_position.lift_env import LiftEnv

import torch.nn.functional as F
from transformers import AutoProcessor

from isaaclab.sensors import ContactSensor, ContactSensorCfg, FrameTransformer
from isaaclab.utils.math import transform_points, unproject_depth

from . import image
from .string_target import ID_TO_TARGET, NUM_TARGETS, PREDEFINED_TARGETS, TARGET_TO_ID
import cv2
from fastsam import FastSAM, FastSAMPrompt
import numpy as np
import json


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
    # print(object_cfg.name, object_pos_b)
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


@torch.no_grad()
def image_feature_obs(env: LiftEnv) -> torch.Tensor:
    if hasattr(env, "current_target_state_per_env") and env.encoded_task_goal_per_env is not None:
        # text_list=env.current_target_strings_per_env
        raw_image_data1 = image(env, sensor_cfg=SceneEntityCfg("camera_1"), data_type="rgb", normalize=False)
        raw_image_data2 = image(env, sensor_cfg=SceneEntityCfg("camera_2"), data_type="rgb", normalize=False)
        inputs = env.image_processor(
            images=[raw_image_data1, raw_image_data2], max_num_patches=64, return_tensors="pt"
        ).to("cuda")
        with torch.no_grad():
            outputs = env.siglip_image_model(**inputs)
        image_feature = outputs.pooler_output
        image_feature = image_feature.view(2, -1, 768)
        image_feature = image_feature.permute(1, 0, 2)
        # similirity=F.cosine_similarity(text_embeddings, image_embeddings, dim=-1)
        # print("image feature shape",image_feature.shape)
        # print("similarity",similirity.shape)
        return image_feature  # (n,2,768)
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


@torch.no_grad()
def rgb_obs(env: LiftEnv) -> torch.Tensor:
    raw_image_data = image(
        env, sensor_cfg=SceneEntityCfg("camera_1"), data_type="rgb", normalize=True
    )  # torch.Size([n, 256, 256, 3])
    downsampled_tensor_first = F.interpolate(
        raw_image_data.permute(0, 3, 1, 2),
        scale_factor=0.5,
        mode="bilinear",
        align_corners=False,  # 推荐在 scale_factor < 1 时设为 False
    ).permute(0, 2, 3, 1)
    return downsampled_tensor_first


@torch.no_grad()
def depth_obs(env: LiftEnv) -> torch.Tensor:
    raw_image_data1 = image(
        env, sensor_cfg=SceneEntityCfg("camera_1"), data_type="depth", normalize=True
    )  # torch.Size([1, 256, 256, 1])
    raw_image_data2 = image(
        env, sensor_cfg=SceneEntityCfg("camera_2"), data_type="depth", normalize=True
    )  # torch.Size([1, 256, 256, 1])
    # downsampled_nearest = F.interpolate(
    #     raw_image_data.permute(0, 3, 1, 2),  # 改成channel first
    #     scale_factor=0.5,
    #     mode="nearest",  # 关键点：使用 'nearest'
    # ).permute(
    #     0, 2, 3, 1
    # )  # 该会channel last

    return torch.cat([raw_image_data1, raw_image_data2], dim=-1)


@torch.no_grad()
def text_feature_obs(env: LiftEnv) -> torch.Tensor:
    if hasattr(env, "current_target_state_per_env") and env.encoded_task_goal_per_env is not None:
        text_list = env.current_target_strings_per_env
        return env.current_target_state_per_env
        # return env.current_target_ids_per_env.unsqueeze(-1)
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
        return torch.zeros((env.num_envs, 768), dtype=torch.float32)


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


@torch.no_grad()
def calcualte_object_pos_from_depth(env: LiftEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # 首先要计算camera的位姿
    robot: RigidObject = env.scene[robot_cfg.name]
    camera = env.scene["camera_1"]
    camera_pos_w = camera.data.pos_w[:, :3]
    camera_rot_w = camera.data.quat_w_ros
    camera_pos_b, camera_rot_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], camera_pos_w, camera_rot_w
    )
    # 获取深度图
    raw_depth_data = image(env, sensor_cfg=SceneEntityCfg("camera_1"), data_type="depth", normalize=True)  # 单位米
    raw_rgb_data = image(
        env, sensor_cfg=SceneEntityCfg("camera_1"), data_type="rgb", normalize=False
    )  # torch.Size([n, 256, 256, 3])
    # 获取中点的深度
    # 3. 计算中点坐标
    batch_size, img_height, img_width, _ = raw_depth_data.shape

    rgb_tensor_chw_float = raw_rgb_data.permute(0, 3, 1, 2).float()
    list_of_processed_tensors = []
    for i in range(batch_size):
        tensor_padded, _, _ = letterbox_tensor(rgb_tensor_chw_float[i], new_shape=480, stride=32, auto=True)
        list_of_processed_tensors.append(tensor_padded / 255.0)
    final_batch_tensor = torch.stack(list_of_processed_tensors, dim=0)

    list_of_orig_imgs_bgr = [cv2.cvtColor(raw_rgb_data[i].cpu().numpy(), cv2.COLOR_RGB2BGR) for i in range(batch_size)]

    everything_results = env.sam_model.predict_tensor(
        final_batch_tensor, orig_imgs=list_of_orig_imgs_bgr, retina_masks=True, conf=0.65, iou=0.9
    )
    # --- 3. 核心处理：提取、抠图、编码、定位 ---
    # 准备一个大的批次来存储所有环境中所有需要被编码的抠图
    images_to_encode_in_batch = []
    # 映射表，用于在编码后将 embedding 重新分配给正确的环境和物体
    encoding_map = []  # 存储 (env_id, item_idx_in_env)
    all_midpoints = []  # 列表的列表，外层对应环境，内层对应每个物体
    all_depths = []
    depth_in_batch = []
    midpoints_in_batch = []
    if everything_results:
        for env_id, single_result in enumerate(everything_results):
            if len(single_result) == 0:
                continue
            env_midpoints = []
            env_depths = []
            masks = single_result.masks.data  # (num_objects, H, W)
            # 获取当前环境的原始RGB图像 (N, H, W, 3) -> (H, W, 3)
            current_rgb_image = raw_rgb_data[env_id]

            for item_idx, mask in enumerate(masks):
                # a. 抠图: 应用掩码
                # 确保掩码是 (H, W, 1) 形状以便广播
                mask_expanded = mask.unsqueeze(-1).bool()
                # 使用掩码，背景部分设为0 (黑色)
                # where(condition, x, y) 如果条件为真取x，否则取y
                cropped_object_tensor = torch.where(
                    mask_expanded, current_rgb_image, torch.zeros_like(current_rgb_image)
                )

                # b. 收集抠图和映射信息
                images_to_encode_in_batch.append(cropped_object_tensor)
                encoding_map.append({"env_id": env_id, "item_idx": item_idx})  # 每一个图片对应一个标签

                # c. 收集各个图片的质心位置及其深度
                # 掩码的尺寸可能与原始图像不同，需要缩放
                # 但 FastSAM 的 `predict_tensor` 返回的 mask 已经被缩放回原始尺寸了
                # (如果 retina_masks=True 且传入了 orig_imgs)
                # 确保掩码是布尔类型
                mask_bool = mask > 0.5
                # 计算中点坐标 (x, y) 或 (col, row)
                # `nonzero()` 返回非零元素的索引
                coords = mask_bool.nonzero(as_tuple=False)  # (num_points, 2), 格式是 (row, col)

                # 计算平均坐标 (质心)
                midpoint = coords.float().mean(dim=0)  # (2,)
                # 四舍五入到最近的像素
                midpoint_px = torch.round(midpoint).long()
                # PyTorch 索引是 [row, col]，所以 midpoint_px[0] 是 y, midpoint_px[1] 是 x
                u, v = midpoint_px[1], midpoint_px[0]  # (x, y)

                # 确保坐标在图像范围内
                u = torch.clamp(u, 0, img_width - 1)
                v = torch.clamp(v, 0, img_height - 1)

                # 获取该中点的深度值
                depth_at_midpoint = raw_depth_data[i, v, u, 0]

                midpoints_in_batch.append(torch.stack([u, v]))
                depth_in_batch.append(depth_at_midpoint)

        # --- 4. 批量编码所有抠图 (SigLIP) ---
        flat_embeddings = []
        if images_to_encode_in_batch:
            # 使用 image_processor 进行批量预处理
            # 注意：image_processor 需要一个 PIL Image 或 NumPy 数组的列表，或者 (H, W, C) 的 Tensor 列表
            # 我们的 cropped_object_tensor 是 (H, W, C) 的，可以直接用
            inputs = env.image_processor(images=images_to_encode_in_batch, return_tensors="pt").to(env.device)
            # 批量推理
            outputs = env.siglip_image_model(**inputs)
            flat_embeddings = outputs.pooler_output  # (Total_Objects, embedding_dim)

        all_embeddings_in_batch = []  # 列表的列表，外层对应环境，内层对应每个物体
        all_positions_in_batch = []  # (n_env,n_obj,3)

        # --- 5. 重新组合 Embedding 并计算位置 ---
        # 根据 encoding_map 将扁平化的 embedding 重新分配回每个环境
        # 同时计算位置
        camera_pos_w = camera.data.pos_w[:, :3]
        camera_rot_w = camera.data.quat_w_ros
        camera_pos_b, camera_rot_b = subtract_frame_transforms(
            robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], camera_pos_w, camera_rot_w
        )
        for env_id in range(batch_size):
            env_embeddings = []
            env_positions = []
            env_coords = []
            env_depths = []

            # 找到属于当前环境的所有物体
            indices = [i for i, item in enumerate(encoding_map) if item["env_id"] == env_id]

            if not indices:
                all_embeddings_in_batch.append(torch.empty((0, env.feature_dim), device=env.device))
                all_positions_in_batch.append(torch.empty((0, 3), device=env.device))
                continue

            for i in indices:
                item_map = encoding_map[i]
                item_idx = item_map["item_idx"]  # 这个物体是该环境中的第几个

                # a. 获取 embedding
                embedding = flat_embeddings[i]
                depth = depth_in_batch[i]
                midpoint = midpoints_in_batch[i]
                env_embeddings.append(embedding)
                env_coords.append(midpoint)
                env_depths.append(depth)

            # 计算物体的位置
            pixel_coords = torch.stack(env_coords)  # (num_objects, 2)
            pixel_depths = torch.stack(env_depths)  # (num_objects,)

            # 1. 反投影到相机坐标系
            # 相机内参，通常从 camera.data.intrinsic_matrices 获取
            # 为了简化，我们使用硬编码值，但最好是从相机获取
            fx = camera.cfg.intrinsic_matrix[0, 0] if hasattr(camera.cfg, "intrinsic_matrix") else 1462.8
            fy = camera.cfg.intrinsic_matrix[1, 1] if hasattr(camera.cfg, "intrinsic_matrix") else 1462.8
            cx = camera.cfg.intrinsic_matrix[0, 2] if hasattr(camera.cfg, "intrinsic_matrix") else img_width / 2.0
            cy = camera.cfg.intrinsic_matrix[1, 2] if hasattr(camera.cfg, "intrinsic_matrix") else img_height / 2.0

            u = pixel_coords[:, 0]
            v = pixel_coords[:, 1]
            d = pixel_depths

            z_c = d
            x_c = (u - cx) * z_c / fx
            y_c = (v - cy) * z_c / fy

            # (num_objects, 3)
            points_pos_c = torch.stack([x_c, y_c, z_c], dim=1)

            # 2. 变换到机器人基座坐标系
            # quat_apply 需要被应用到批次中的每个点上
            # 我们需要将相机的姿态扩展以匹配点的数量
            # cam_rot_b[i] shape is (4,), points_pos_c shape is (num_objects, 3)
            # quat_apply can handle this broadcasting
            rotated_points = quat_apply(camera_rot_b[env_id].unsqueeze(0), points_pos_c)
            points_pos_b = rotated_points + camera_pos_b[env_id].unsqueeze(0)

            all_embeddings_in_batch.append(torch.stack(env_embeddings))
            all_positions_in_batch.append(points_pos_b)

        # print(all_embeddings_in_batch[0].shape)
        # print(all_positions_in_batch[0].shape)
        update_object_database_with_chromadb(
            env, all_embeddings_in_batch=all_embeddings_in_batch, all_positions_in_batch=all_positions_in_batch
        )

    pos_list = []
    for env_id in range(batch_size):
        text_embedding = env.current_target_state_per_env[env_id].cpu().numpy()
        pos = torch.from_numpy(find_item_by_text(env, text_embedding=text_embedding, env_id=env_id))
        pos_list.append(pos)

    object_position = torch.stack(pos_list).to(env.device)
    # print('object position shape',object_position.shape)
    # print(object_position)

    # center_y = img_height // 2
    # center_x = img_width // 2
    # midpoint_coords = torch.tensor([center_x, center_y]).to("cuda").repeat(batch_size, 1)
    # # 注意：PyTorch/NumPy 的索引通常是 [row, column]，对应 [height, width]

    # # 4. 进行索引以获取中点的深度值
    # #    选择所有环境 (:)
    # #    在 height 维度上选择 center_y
    # #    在 width 维度上选择 center_x
    # #    在 channel 维度上选择 0
    # center_depth_values = raw_depth_data[:, center_y, center_x, 0]
    # # print(center_depth_values,center_depth_values.shape)

    # object_pos, pixel_pos_camera = get_point_in_robot_frame_from_pixel(
    #     pixel_coords=midpoint_coords,
    #     pixel_depth=center_depth_values,
    #     camera_pos_in_robot_frame=camera_pos_b,
    #     camera_rot_in_robot_frame=camera_rot_b,
    #     image_height=img_height,
    #     image_width=img_width,
    # )

    # 第二种方法算点的坐标
    # point_pos_world = transform_points_from_camera_to_world_batch(
    #     pts_cam=pixel_pos_camera, cam_pos_in_world=camera_pos_w, cam_quat_in_world=camera_rot_w
    # )
    # object_pos_b, _ = subtract_frame_transforms(
    #     robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], point_pos_world
    # )

    # 第三种方法算点的坐标
    # points_3d_cam = unproject_depth(camera.data.output["depth"], camera.data.intrinsic_matrices)
    # points_3d_world = transform_points(points_3d_cam, camera.data.pos_w, camera.data.quat_w_ros)

    # print("predict object pose in root", object_pos.shape, object_pos)  # 相比point_pos_world没变化
    # print("predict object pose2 in root", object_pos_b.shape, object_pos_b)
    # print("camera pose in root", camera_pos_b)  # 应该正确
    # print("camera pose in world", camera_pos_w)
    # print("camera rot in root", camera_rot_b)
    # print("camera rot in world", camera_rot_w)
    # print("robot rot in world", robot.data.root_state_w[:, 3:7])
    # print("point in world", points_3d_world[:, :3])
    return object_position


@torch.no_grad()
def rgb_feature(env: LiftEnv):
    raw_image_data1 = image(env, sensor_cfg=SceneEntityCfg("camera_1"), data_type="rgb", normalize=False).permute(
        0, 3, 1, 2
    )
    raw_image_data2 = image(env, sensor_cfg=SceneEntityCfg("camera_2"), data_type="rgb", normalize=False).permute(
        0, 3, 1, 2
    )
    input_tensor = env.rgb_processor(torch.cat([raw_image_data1, raw_image_data2], dim=0))

    # ----------------- 3. 提取特征 -----------------
    # 使用 torch.no_grad() 来禁用梯度计算，节省内存和计算资源
    with torch.no_grad():
        features = env.rgb_extractor(input_tensor)
    features = features.reshape(2, -1, 1280)
    features = features.permute(1, 0, 2)
    # features = features.contiguous().reshape(-1, 2 * 1280)
    # print(features.shape)
    return features


##--------------以下是工具函数-----------------
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


def letterbox_tensor(
    im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32
):
    # (此处省略函数代码，与之前相同)
    was_3d = False
    if im.dim() == 3:
        was_3d = True
        im = im.unsqueeze(0)
    shape = im.shape[2:]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
    dw, dh = new_shape[1] - new_unpad[1], new_shape[0] - new_unpad[0]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = F.interpolate(im, size=new_unpad, mode="bilinear", align_corners=False)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    pad_color_value = color[0] / 255.0 if im.is_floating_point() else color[0]
    im = F.pad(im, (left, right, top, bottom), "constant", value=pad_color_value)
    if was_3d:
        im = im.squeeze(0)
    return im, r, (dw, dh)


# In a new file, e.g., my_lift/chromadb_updater.py
# Or directly in observations.py if you prefer

import torch
import chromadb
import uuid
import numpy as np


def process_new_observation(env, new_embedding, new_position, env_id):

    results = env.collection.query(
        query_embeddings=[new_embedding],
        n_results=1,
        include=["metadatas", "distances", "embeddings"],
        where={"env_id": env_id},
    )

    # 检查数据库是否为空或最相似的物品是否满足阈值
    is_new_item = True
    if results["ids"][0]:  # 如果查询有返回结果
        # ChromaDB的距离是L2距离，值越小越相似。我们需要转换成相似度或直接比较距离。
        # 为简单起见，我们假设距离越小越好，设定一个距离阈值
        # 这个值需要实验确定,THRESHOLD越大，不同物品被合并的可能性越小，同时物品位置信息被更新的可能性也越小，参考：red cube和yellow cube的相似度为0.82
        cosine_sim_manual_THRESHOLD = 0.87
        distance = results["distances"][0][0]
        old_embedding = results["embeddings"][0][0]
        dot_product = np.dot(new_embedding, old_embedding)
        # 2. 计算每个向量的 L2 范数 (模长)
        norm_a = np.linalg.norm(new_embedding)
        norm_b = np.linalg.norm(old_embedding)
        # 3. 根据公式计算余弦相似度
        cosine_sim_manual = dot_product / (norm_a * norm_b)
        # print(cosine_sim_manual, distance)
        # 0.18,0.9

        if cosine_sim_manual > cosine_sim_manual_THRESHOLD:

            is_new_item = False
            existing_id = results["ids"][0][0]
            # print(f"找到相似物品 (ID: {existing_id}), 距离: {distance:.4f}。正在更新...")
            position_str = json.dumps(new_position.tolist())
            # 4b. 覆盖原来的值
            env.collection.update(
                ids=[existing_id], embeddings=[new_embedding], metadatas={"env_id": env_id, "position": position_str}
            )

    if is_new_item:
        new_id = str(uuid.uuid4())
        # print(f"未找到相似物品，新建索引 (ID: {new_id})")
        # 4a. 新建一个索引
        position_str = json.dumps(new_position.tolist())
        env.collection.add(
            ids=[new_id], embeddings=[new_embedding], metadatas={"env_id": env_id, "position": position_str}
        )


@torch.no_grad()
def find_item_by_text(env, text_embedding, env_id):
    # 将文本查询编码
    # 5. 在 collection 中查询

    results = env.collection.query(
        query_embeddings=[text_embedding], n_results=1, where={"env_id": env_id}, include=["metadatas", "embeddings"]
    )

    if results["ids"][0]:
        embedding = results["embeddings"][0][0]
        norm_a = np.linalg.norm(text_embedding)
        norm_b = np.linalg.norm(embedding)
        # 3. 根据公式计算余弦相似度
        dot_product = np.dot(norm_a, norm_b)
        cosine_sim_manual = dot_product / (norm_a * norm_b)
        if cosine_sim_manual > 0.7:
            found_id = results["ids"][0][0]
            found_metadata = results["metadatas"][0][0]
            found_position = found_metadata.get("position")
            found_position = np.array(json.loads(found_position))
            # print(f"找到最相似的物品 (ID: {found_id}), 位置是: {found_position}")
            return found_position
        else:
            print("未在数据库中找到匹配的物品。")
            return np.array([0, 0, 0])
    else:
        print("未在数据库中找到匹配的物品。")
        return np.array([0, 0, 0])


@torch.no_grad()
def update_object_database_with_chromadb(
    env: LiftEnv,
    all_embeddings_in_batch: list[torch.Tensor],
    all_positions_in_batch: list[torch.Tensor],
):
    """
    批量更新 ChromaDB 中的物体数据库。

    Args:
        env (LiftEnv): The environment instance, used to access the ChromaDB collection.
        all_embeddings_in_batch (list[torch.Tensor]):
            一个列表，长度为 N (环境数)。每个元素是 (num_objects, embedding_dim) 的张量，
            包含了该环境中所有检测到的物体的 embedding。
        all_positions_in_batch (list[torch.Tensor]):
            一个列表，长度为 N。每个元素是 (num_objects, 3) 的张量，
            包含了该环境中所有检测到的物体的位置 (point_pos_b)。
    """
    # 假设 collection 在 env 实例中
    # collection = env.collection
    batch_size = len(all_embeddings_in_batch)
    device = env.device

    # --- 2. 准备批量查询数据 ---
    # 将来自所有环境的 embedding 和元数据展平到一个大列表中
    # flat_query_embeddings = []
    # 记录每个 embedding 原始的 env_id 和 item_idx_in_env
    # 这对于后续将查询结果重新映射回原环境至关重要

    for env_id in range(batch_size):
        embeddings_in_env = all_embeddings_in_batch[env_id]
        positions_in_env = all_positions_in_batch[env_id]
        num_objects_in_env = embeddings_in_env.shape[0]

        for item_idx in range(num_objects_in_env):
            embedding = embeddings_in_env[item_idx].cpu().numpy()  # 某个物品对应的embedding
            position = positions_in_env[item_idx].cpu().numpy()
            process_new_observation(env, new_embedding=embedding, new_position=position, env_id=env_id)
