# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg,RigidObject,RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns,Camera
from torchvision import transforms
from collections import OrderedDict

@configclass
class FrankaCubeEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 5.5  # 500 timesteps
    decimation = 10
    action_space = 9
    observation_space = {
        'rgb':[64,64,3],
        'rgb2':[64,64,3],
        'actions':9,
        'to_target':3
    }
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=5,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8, env_spacing=3.0, replicate_physics=True)

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469,
                "panda_finger_joint.*": 0.035,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    cube=RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube",
        spawn=sim_utils.CuboidCfg(
                size=(0.1, 0.1, 0.1),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=4, solver_velocity_iteration_count=0
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.5, 0.25),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    # cabinet
    cabinet = ArticulationCfg(
        prim_path="/World/envs/env_.*/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-1, -1, 0.4),
            rot=(0.1, 0.0, 0.0, 0.0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=10.0,
                damping=1.0,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=10.0,
                damping=2.5,
            ),
        },
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    camera = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link7/front_cam",
        update_period=0.1,
        height=256,
        width=256,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(

            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)

        ),

        offset=CameraCfg.OffsetCfg(pos=(0, 0.0, 0.15), rot=(1, 0, 0, 0), convention="ros"),#xyzw

    )

    camera2 = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link1/base_cam",
        update_period=0.1,
        height=256,
        width=256,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(

            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)

        ),

        offset=CameraCfg.OffsetCfg(pos=(0,-0.06, -0.18), rot=(0, 0,-0.70711, 0.70711), convention="ros"),#wxyz
        #为什么我设置(0, -0.70711, -0.70711, 0)到了环境中发生了变化，改成world试一下
        #offset=CameraCfg.OffsetCfg(pos=(0,-0.13, 0.0), rot=(0, 0,0, 1), convention="ros"),#wxyz
    )

    action_scale = 7.5
    dof_velocity_scale = 0.1

    # reward scales
    dist_reward_scale = 1.5
    rot_reward_scale = 1.5
    open_reward_scale = 10.0
    action_penalty_scale = 0.01#0.05
    finger_reward_scale = 2.0


class FrankaCubeEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: FrankaCubeEnvCfg

    def __init__(self, cfg: FrankaCubeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint2")[0]] = 0.1

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        stage = get_current_stage()
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_link7")),
            self.device,
        )
        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_leftfinger")),
            self.device,
        )
        rfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_rightfinger")),
            self.device,
        )

        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])#转换到hand坐标系所需的旋转和平移

        robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]#将环境坐标系下的finger_pose转换到hand坐标系
        )
        robot_local_pose_pos += torch.tensor([0, 0.04, 0], device=self.device)
        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

        drawer_local_grasp_pose = torch.tensor([0.3, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0], device=self.device)
        self.drawer_local_grasp_pos = drawer_local_grasp_pose[0:3].repeat((self.num_envs, 1))
        self.drawer_local_grasp_rot = drawer_local_grasp_pose[3:7].repeat((self.num_envs, 1))

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.drawer_inward_axis = torch.tensor([-1, 0, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.drawer_up_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )

        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]
        self.drawer_link_idx = self._cabinet.find_bodies("drawer_top")[0][0]

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.drawer_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.drawer_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

        self.panda_hand_link_idx=self._robot.find_bodies("panda_hand")[0][0]
        self.storage_beginning=torch.zeros((self.num_envs), device=self.device,dtype=torch.bool)

        self.panda_hand_pose=torch.zeros((self.num_envs, 3), device=self.device)
        self.cube_pose=torch.zeros((self.num_envs, 3), device=self.device)
        self.initial_distance=torch.norm(self.panda_hand_pose-self.cube_pose, p=2, dim=-1)#shape:(num_envs)
        self.current_distance=torch.norm(self.panda_hand_pose-self.cube_pose, p=2, dim=-1)#shape:(num_envs)

        self.camera_data=torch.zeros((self.num_envs, 64,64,3), device=self.device)
        self.camera_data2=torch.zeros((self.num_envs, 64,64,3), device=self.device)
        
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._cabinet = Articulation(self.cfg.cabinet)
        self._cube = RigidObject(self.cfg.cube)
        self._camera = Camera(self.cfg.camera)
        self._camera2 = Camera(self.cfg.camera2)
        self.scene.articulations["robot"] = self._robot
        # self.scene.sensors['camera']=self._camera
        # self.scene.sensors['camera2']=self._camera2


        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)




    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):#对动作进行裁剪
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self):#对动作进行应用
        self._robot.set_joint_position_target(self.robot_dof_targets)
        #print(self.initial_distance.shape,self.initial_distance)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        reach_in=torch.any(self.current_distance/self.initial_distance<0.13,dim=-1)
        reach_in = reach_in | torch.any(self.current_distance<0.12,dim=-1)
        terminated = reach_in
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()
        robot_left_finger_pos = self._robot.data.body_pos_w[:, self.left_finger_link_idx]
        robot_right_finger_pos = self._robot.data.body_pos_w[:, self.right_finger_link_idx]

        return self._compute_rewards(
            self.actions,
            self._cabinet.data.joint_pos,
            self.robot_grasp_pos,
            self.drawer_grasp_pos,
            self.robot_grasp_rot,
            self.drawer_grasp_rot,
            robot_left_finger_pos,
            robot_right_finger_pos,
            self.gripper_forward_axis,
            self.drawer_inward_axis,
            self.gripper_up_axis,
            self.drawer_up_axis,
            self.num_envs,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.open_reward_scale,
            self.cfg.action_penalty_scale,
            self.cfg.finger_reward_scale,
            self._robot.data.joint_pos,
            self.current_distance,
            self.initial_distance,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # cabinet state
        zeros = torch.zeros((len(env_ids), self._cabinet.num_joints), device=self.device)
        self._cabinet.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)

        # initial_cube_state=torch.tensor([0.5, 0.5, 0.25,1.0, 0.0, 0.0, 0.0],device=self.device)
        # initial_cube_state=initial_cube_state.unsqueeze(0).expand(len(env_ids), -1) 
        default_root_state = self._cube.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]


        random_pos_dif=sample_uniform(
            -0.25,
            0.25,
            (len(env_ids), 2),
            self.device,
        )
        pos=default_root_state[:, :7]
        pos[:,:2]+=random_pos_dif
        self._cube.write_root_pose_to_sim(pos, env_ids)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)

        self.storage_beginning[env_ids]=False

    def _get_observations(self) -> dict:
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        #to_target = self.drawer_grasp_pos - self.robot_grasp_pos
        to_target = self._cube.data.body_pos_w[:, 0] - self._robot.data.body_pos_w[:, self.hand_link_idx]#torch.Size([n, 3])

        # obs = torch.cat(
        #     (
        #         dof_pos_scaled,
        #         self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
        #         to_target,
        #         self._cabinet.data.joint_pos[:, 3].unsqueeze(-1),
        #         self._cabinet.data.joint_vel[:, 3].unsqueeze(-1),
        #     ),
        #     dim=-1,
        # )
        
        obs= {}
        obs['actions']=self.actions
        obs['rgb']=self.camera_data
        obs['rgb2']=self.camera_data2
        obs['to_target']=to_target
        env_ids=(~self.storage_beginning).nonzero(as_tuple=False).squeeze(-1)#哪些环境是已经重置了的
        if env_ids is not None:
            self.panda_hand_pose[env_ids]=self._robot.data.body_pos_w[env_ids, self.panda_hand_link_idx].squeeze(1)
            self.cube_pose[env_ids]=self._cube.data.body_pos_w[env_ids].squeeze(1)
            self.initial_distance[env_ids]=torch.norm(self.panda_hand_pose[env_ids]-self.cube_pose[env_ids], p=2, dim=-1)
            self.storage_beginning[env_ids]=True

        
        #print(self.scene['camera'].data.output['rgb'].shape)
        return {"policy": obs}

    # auxiliary methods

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        hand_pos = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]#世界坐标系
        hand_rot = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]
        drawer_pos = self._cabinet.data.body_pos_w[env_ids, self.drawer_link_idx]
        drawer_rot = self._cabinet.data.body_quat_w[env_ids, self.drawer_link_idx]
        (
            self.robot_grasp_rot[env_ids],
            self.robot_grasp_pos[env_ids],
            self.drawer_grasp_rot[env_ids],
            self.drawer_grasp_pos[env_ids],
        ) = self._compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.robot_local_grasp_rot[env_ids],
            self.robot_local_grasp_pos[env_ids],
            drawer_rot,
            drawer_pos,
            self.drawer_local_grasp_rot[env_ids],
            self.drawer_local_grasp_pos[env_ids],
        )

        self.panda_hand_pose[env_ids]=self._robot.data.body_pos_w[env_ids, self.panda_hand_link_idx].squeeze(1)
        self.cube_pose[env_ids]=self._cube.data.body_pos_w[env_ids].squeeze(1)
        self.current_distance[env_ids]=torch.norm(self.panda_hand_pose[env_ids]-self.cube_pose[env_ids], p=2, dim=-1)

        # import matplotlib.pyplot as plt#测试相机信息
        # img_tensor_hwc = self._camera.data.output["rgb"].squeeze(0) # 或者 tensor_image[0]
        # print(f"移除 Batch 维度后形状: {img_tensor_hwc.shape}")

        # # 2. 确保在 CPU 上 (如果你的 tensor 在 GPU 上)
        # if img_tensor_hwc.is_cuda:
        #     img_tensor_hwc = img_tensor_hwc.cpu()

        # # 3. 转换为 NumPy 数组
        # #    注意：如果原始 tensor 需要梯度，先 .detach()
        # img_numpy = img_tensor_hwc.detach().numpy()
        # print(f"NumPy 数组形状: {img_numpy.shape}, 数据类型: {img_numpy.dtype}")

        # # 4. 数据类型和范围处理 (根据实际情况调整)
        # #    - 如果是 float 类型，确保值在 [0, 1]
        # #      如果值超出范围 (例如 -1 到 1, 或者 0 到 255 但仍是 float)，需要调整
        # #      例如，如果值是0-255的float，可以 img_numpy = img_numpy / 255.0
        # #      例如，如果值是-1到1的float，可以 img_numpy = (img_numpy + 1.0) / 2.0
        # #    - 如果希望转换为 uint8 类型 (0-255)
        # #      img_numpy_uint8 = (img_numpy * 255).astype(np.uint8)

        # # 在这个示例中，我们的 tensor_image 已经是 float 且值在 [0, 1]

        # # --- 方法 1: 使用 Matplotlib ---
        # print("\n--- 使用 Matplotlib 显示 ---")
        # plt.figure("Matplotlib Display")
        # plt.imshow(img_numpy) # Matplotlib 的 imshow 可以处理 float [0,1] 或 uint8 [0,255]
        # plt.title("Image (Matplotlib)")
        # plt.axis('off') # 关闭坐标轴
        # plt.show()
        # plt.imsave("saved_image_matplotlib_uint8.png", img_numpy)
        # breakpoint()

        self.camera_data=(self._camera.data.output["rgb"]/255.0-0.5)*5#uint8->fp32
        print(self.camera_data.shape)
        resize_transform_bilinear = transforms.Resize(size=(64, 64))
        self.camera_data=resize_transform_bilinear(self.camera_data.permute(0,3,1,2)).permute(0,2,3,1)

        self.camera_data2=(self._camera2.data.output["rgb"]/255.0-0.5)*5#uint8->fp32
        self.camera_data2=resize_transform_bilinear(self.camera_data2.permute(0,3,1,2)).permute(0,2,3,1)




    def _compute_rewards(
        self,
        actions,
        cabinet_dof_pos,
        franka_grasp_pos,
        drawer_grasp_pos,
        franka_grasp_rot,
        drawer_grasp_rot,
        franka_lfinger_pos,
        franka_rfinger_pos,
        gripper_forward_axis,
        drawer_inward_axis,
        gripper_up_axis,
        drawer_up_axis,
        num_envs,
        dist_reward_scale,
        rot_reward_scale,
        open_reward_scale,
        action_penalty_scale,
        finger_reward_scale,
        joint_positions,
        current_distance,
        initial_distance,
    ):
        # distance from hand to the drawer
        d = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + d**2)
        dist_reward *= dist_reward
        dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

        axis1 = tf_vector(franka_grasp_rot, gripper_forward_axis)
        axis2 = tf_vector(drawer_grasp_rot, drawer_inward_axis)
        axis3 = tf_vector(franka_grasp_rot, gripper_up_axis)
        axis4 = tf_vector(drawer_grasp_rot, drawer_up_axis)

        dot1 = (
            torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of forward axis for gripper
        dot2 = (
            torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of up axis for gripper
        # reward for matching the orientation of the hand to the drawer (fingers wrapped)
        rot_reward = 0.5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2)

        # regularization on the actions (summed for each environment)
        action_penalty = torch.sum(actions**2, dim=-1)

        # how far the cabinet has been opened out
        open_reward = cabinet_dof_pos[:, 3]  # drawer_top_joint
        # penalty for distance of each finger from the drawer handle
        lfinger_dist = franka_lfinger_pos[:, 2] - drawer_grasp_pos[:, 2]
        rfinger_dist = drawer_grasp_pos[:, 2] - franka_rfinger_pos[:, 2]
        finger_dist_penalty = torch.zeros_like(lfinger_dist)
        finger_dist_penalty += torch.where(lfinger_dist < 0, lfinger_dist, torch.zeros_like(lfinger_dist))
        finger_dist_penalty += torch.where(rfinger_dist < 0, rfinger_dist, torch.zeros_like(rfinger_dist))

        distance_penalty=current_distance/initial_distance-1#current_distance越小越好
        distance_penalty.clamp_(max=0.0)
        rewards = (
            - action_penalty_scale * action_penalty
            - dist_reward_scale * distance_penalty
        )#shape:(num_envs)

        self.extras["log"] = {
            # "dist_reward": (dist_reward_scale * dist_reward).mean(),
            # "rot_reward": (rot_reward_scale * rot_reward).mean(),
            # "open_reward": (open_reward_scale * open_reward).mean(),
            "action_penalty": (-action_penalty_scale * action_penalty).mean(),
            # "left_finger_distance_reward": (finger_reward_scale * lfinger_dist).mean(),
            # "right_finger_distance_reward": (finger_reward_scale * rfinger_dist).mean(),
            # "finger_dist_penalty": (finger_reward_scale * finger_dist_penalty).mean(),
            'distance_penalty': (- dist_reward_scale * distance_penalty).mean()
        }

        # bonus for opening drawer properly
        # rewards = torch.where(cabinet_dof_pos[:, 3] > 0.01, rewards + 0.25, rewards)
        # rewards = torch.where(cabinet_dof_pos[:, 3] > 0.2, rewards + 0.25, rewards)
        # rewards = torch.where(cabinet_dof_pos[:, 3] > 0.35, rewards + 0.25, rewards)
        #print(- action_penalty_scale * action_penalty,- dist_reward_scale * distance_penalty)
        #print(- action_penalty_scale * action_penalty,- dist_reward_scale * distance_penalty)
        return rewards

    def _compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,#世界坐标系
        franka_local_grasp_rot,
        franka_local_grasp_pos,
        drawer_rot,
        drawer_pos,
        drawer_local_grasp_rot,
        drawer_local_grasp_pos,
    ):
        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )
        global_drawer_rot, global_drawer_pos = tf_combine(
            drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
        )

        return global_franka_rot, global_franka_pos, global_drawer_rot, global_drawer_pos
    

