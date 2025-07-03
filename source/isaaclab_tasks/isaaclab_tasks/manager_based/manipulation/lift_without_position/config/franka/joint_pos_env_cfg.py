# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift_without_position import mdp
from isaaclab_tasks.manager_based.manipulation.lift_without_position.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class FrankaCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        franka_cfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # franka_cfg.spawn.activate_contact_sensors=True
        # print(franka_cfg)
        self.scene.robot = franka_cfg

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_hand"

        # Set Cube as object
        # self.scene.object = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Object",
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
        #     spawn=UsdFileCfg(
        #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        #         scale=(0.8, 0.8, 0.8),
        #         rigid_props=RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=16,
        #             solver_velocity_iteration_count=1,
        #             max_angular_velocity=1000.0,
        #             max_linear_velocity=1000.0,
        #             max_depenetration_velocity=5.0,
        #             disable_gravity=False,
        #         ),
        #     ),
        # )

        self.scene.green_object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/GreenCube",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.45, 0.1, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
                scale=(0.85, 0.85, 0.85),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=MassPropertiesCfg(mass=0.3),
                activate_contact_sensors=True,
            ),
        )

        self.scene.red_object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/RedCube",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, -0.05, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                scale=(0.85, 0.85, 0.85),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=MassPropertiesCfg(mass=0.3),
            ),
        )

        self.scene.yellow_object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/YellowCube",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.52, -0.1, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/yellow_block.usd",
                scale=(0.85, 0.85, 0.85),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=MassPropertiesCfg(mass=0.3),
            ),
        )
        # self.scene.bottle = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/bottle",
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.45, 0, 0], rot=[1, 0, 0, 0]),
        #     spawn=UsdFileCfg(
        #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
        #         scale=(0.85, 0.85, 0.85),
        #         rigid_props=RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=16,
        #             solver_velocity_iteration_count=1,
        #             max_angular_velocity=1000.0,
        #             max_linear_velocity=1000.0,
        #             max_depenetration_velocity=5.0,
        #             disable_gravity=False,
        #         ),
        #         mass_props=MassPropertiesCfg(mass=0.4)
        #     ),
        # )

        self.scene.camera_1 = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link7/front_cam",
            update_period=0.1,
            height=256,  # 480,
            width=256,  # 640,
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0, 0.0, 0.15), rot=(1, 0, 0, 0), convention="ros"),  # xyzw
        )

        self.scene.left_finger_contactsensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",  # 传感器附着在左手指上
            filter_prim_paths_expr=[
                "{ENV_REGEX_NS}/GreenCube/Cube",
                "{ENV_REGEX_NS}/YellowCube/Cube",
                "{ENV_REGEX_NS}/RedCube/Cube",
            ],  # 过滤目标是绿色立方体
            update_period=0.0,  # 每步更新
            force_threshold=0.1,
            # track_pose=False, # 通常不需要追踪传感器的位姿，除非有特殊需求
            # track_air_time=False, # 如果不需要接触/分离时间信息，可以关闭
            # history_length=0, # 如果不需要历史数据
            debug_vis=False,  # 在调试时可以打开可视化
        )
        self.scene.right_finger_contactsensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",  # 传感器附着在左手指上
            filter_prim_paths_expr=[
                "{ENV_REGEX_NS}/GreenCube/Cube",
                "{ENV_REGEX_NS}/YellowCube/Cube",
                "{ENV_REGEX_NS}/RedCube/Cube",
            ],  # 过滤目标是绿色立方体
            update_period=0.0,  # 每步更新
            force_threshold=0.1,
            # track_pose=False, # 通常不需要追踪传感器的位姿，除非有特殊需求
            # track_air_time=False, # 如果不需要接触/分离时间信息，可以关闭
            # history_length=0, # 如果不需要历史数据
            debug_vis=False,  # 在调试时可以打开可视化
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )

        # self.observation_space = {
        #     "joint_pos": 9,
        #     "joint_vel": 9,
        #     "ee_position": 3,
        #     #'object_position':9,
        #     "target_object_position": 7,
        #     "actions": 8,
        # }


@configclass
class FrankaCubeLiftEnvCfg_PLAY(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
