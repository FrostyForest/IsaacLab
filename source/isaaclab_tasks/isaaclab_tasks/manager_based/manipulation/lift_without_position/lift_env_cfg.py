# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.sensors import CameraCfg, ContactSensorCfg
from . import mdp
import string

from .mdp import PREDEFINED_TARGETS, TARGET_TO_ID, ID_TO_TARGET, NUM_TARGETS

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    # object: RigidObjectCfg | DeformableObjectCfg = MISSING

    green_cube: RigidObjectCfg | DeformableObjectCfg = MISSING
    red_cube: RigidObjectCfg | DeformableObjectCfg = MISSING
    yellow_cube: RigidObjectCfg | DeformableObjectCfg = MISSING

    # bottle: RigidObjectCfg | DeformableObjectCfg = MISSING

    camera_1: CameraCfg = MISSING
    camera_2: CameraCfg = MISSING
    left_finger_contactsensor: ContactSensorCfg = MISSING
    right_finger_contactsensor: ContactSensorCfg = MISSING
    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame)
        # object_position = ObsTerm(func=mdp.get_cubes_position)#各个cube的位置
        target_object_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "object_pose"}
        )  # target position
        actions = ObsTerm(func=mdp.last_action)
        text_clip_feature = ObsTerm(func=mdp.text_feature_obs, noise=None, params={"debug": False})

        ee_camera_orientation = ObsTerm(func=mdp.ee_camera_orientation_in_robot_root_frame)
        image_clip_feature = ObsTerm(func=mdp.image_feature_obs, noise=None)  # 使用新的观测函数名
        depth_obs = ObsTerm(func=mdp.depth_obs, noise=None)
        rgb_feature = ObsTerm(func=mdp.rgb_feature)
        # object_position = ObsTerm(func=mdp.calcualte_object_pos_from_depth)

        contact_force_left_finger = ObsTerm(
            func=mdp.get_finger_contact_forces,
            noise=None,
            params={"sensor_cfg": SceneEntityCfg("left_finger_contactsensor")},
        )
        contact_force_right_finger = ObsTerm(
            func=mdp.get_finger_contact_forces,
            noise=None,
            params={"sensor_cfg": SceneEntityCfg("right_finger_contactsensor")},
        )
        object_position_perfect = ObsTerm(func=mdp.object_position_in_robot_root_frame)

        # ------- obs shape
        """
        joint_pos torch.Size([2, 9])
        joint_vel torch.Size([2, 9])
        ee_position torch.Size([2, 3])
        ee_camera_orientation ([n,4])
        object_position torch.Size([2, 9])
        target_object_position torch.Size([2, 7])
        actions torch.Size([2, 8])
        image_feature torch.Size([2, 768])
        text_feature torch.Size([2, 768])
        """

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.15, 0.15), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("yellow_cube", body_names="Cube"),
        },
    )

    reset_object_position2 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.15, 0.15), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("green_cube", body_names="Cube"),
        },
    )

    reset_object_position3 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.15, 0.15), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("red_cube", body_names="Cube"),
        },
    )

    # 设定目标
    randomize_task_goal_event = EventTerm(
        func=mdp.randomize_string_task_goal,
        mode="reset",  # 确保它在环境重置时被调用
        # params: {} # 如果 randomize_string_task_goal 需要额外参数，可以在这里提供
        # 但在这个设计中，它直接从 env 实例和预定义列表获取信息
    )
    startup_randomize_task_goal = EventTerm(
        func=mdp.randomize_string_task_goal,
        mode="startup",  # 这个事件只会在环境第一次启动时运行一次
        # params: {} # 如果需要，可以传递参数，但 randomize_string_task_goal
        # 通常会作用于所有环境 (env_ids=None 效果)
    )
    # camera模式用到
    # reset_database = EventTerm(
    #     func=mdp.reset_database,
    #     mode="reset",  # 确保它在环境重置时被调用
    #     # params: {} # 如果 randomize_string_task_goal 需要额外参数，可以在这里提供
    #     # 但在这个设计中，它直接从 env 实例和预定义列表获取信息
    # )

    # reset_bottle_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.15, 0.15), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("bottle"),
    #     },
    # )


@configclass
class RewardsCfg:  # 记录的结果是乘了weight之后
    """Reward terms for the MDP."""

    # reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=0.075)
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1)

    # lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=20)
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=15)

    # object_goal_tracking = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
    #     weight=16.0,
    # )
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16,
    )

    # object_goal_tracking_fine_grained = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
    #     weight=5.0,
    # )
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.075, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=10,
    )

    action_rate = RewTerm(func=mdp.my_action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.my_joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # touching_object = RewTerm(
    #     func=mdp.touch_object,
    #     weight=0.5,
    # )
    # touching_object = RewTerm(
    #     func=mdp.touch_object,
    #     weight=1,
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    yellow_object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("yellow_cube")},
    )

    green_object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("green_cube")}
    )

    red_object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("red_cube")}
    )


# @configclass
# class CurriculumCfg:
#     """Curriculum terms for the MDP."""

#     action_rate = CurrTerm(
#         func=mdp.my_modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-2, "num_steps": 10000}
#     )

#     joint_vel = CurrTerm(
#         func=mdp.my_modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-2, "num_steps": 10000}
#     )


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 10
        self.episode_length_s = 5.5
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = 2

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
