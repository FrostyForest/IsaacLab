

import math
import torch

import omni.isaac.lab.sim as sim_utils
#import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab_tasks.manager_based.classic.franka.mdp as mdp
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg,ManagerBasedRLEnv,ManagerBasedRLEnvCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg,RigidObjectCfg
from omni.isaac.lab.sensors import CameraCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.controllers import DifferentialIKControllerCfg

##
# Pre-defined configs
##
from omni.isaac.lab_assets.franka import FRANKA_PANDA_CFG  # isort:skip

##
# Scene definition
##


@configclass
class FrankaSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # franka
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    # distant_light = AssetBaseCfg(
    #     prim_path="/World/DistantLight",
    #     spawn=sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
    #     init_state=AssetBaseCfg.InitialStateCfg(rot=(0.738, 0.477, 0.477, 0.0)),
    # )

    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/cube",
        spawn=sim_utils.CuboidCfg(size=(0.25,0.25,0.25),
                                  rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                                  mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                                  collision_props=sim_utils.CollisionPropertiesCfg(),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                                  ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0,2.0,0.4)),
    )

    camera_bottom =  CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link1/camera_bottom",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0., -0.14, 0.), rot=(0.5, 0.5, -0.5, -0.5), convention="opengl"),
    )
    #
    camera_hand =  CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/camera_hand",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.01259, 0.01059, 0.09526), rot=(0., -0.70711, -0.70711, -0.), convention="opengl"),
    )


@configclass
class ActionsCfg:
    """Action specifications for the environment."""
    #
    # joint_efforts = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=5.0)

    # left_wheel_efforts = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["left_wheel"], scale=500.0)
    # right_wheel_efforts = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["right_wheel"], scale=500.0)

    # left_wheel_velocity = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["left_wheel"], scale=5)
    # right_wheel_velocity = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["right_wheel"], scale=5)

    # carter_action = mdp.AssetActionCfg(asset_name="robot", joint_names=["right_wheel","left_wheel"])


    franka_action = mdp.DifferentialInverseKinematicsActionCfg(asset_name="robot",joint_names=["panda_joint.*"], body_name=["panda_hand"],
                                                               controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),scale=0.1)


@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        # joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        #joint_vel = ObsTerm(func=mdp.joint_vel)
        #joint_names = ObsTerm(func=mdp.joint_names)
        #camera = ObsTerm(func=mdp.camera_data)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # # on startup
    # add_pole_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["pole"]),
    #         "mass_distribution_params": (0.1, 0.5),
    #         "operation": "add",
    #     },
    # )

    # # on reset
    # reset_left_wheel_position = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["left_wheel"]),
    #         "position_range": (-0 * math.pi, 0 * math.pi),
    #         "velocity_range": (0.8 * math.pi, 1 * math.pi),
    #     },
    # )
    #
    # reset_right_wheel_position = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["right_wheel"]),
    #         # "position_range": (-0.125 * math.pi, 0.125 * math.pi),
    #         # "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
    #         "position_range": (-0 * math.pi, 0 * math.pi),
    #         "velocity_range": (-1 * math.pi, -0.8 * math.pi),
    #     },
    # )

    set_cube_root_state=EventTerm(
        func=mdp.set_default_root_state,
        mode="reset",
        params={
            "pose_range": {"x": (-2.5, 2.5), "y": (-2.5, 2.5), "z": (0.05, 0.1)},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )

    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )

    # reset_cube_position = EventTerm(
    #     func=mdp.reset_root_state_with_random_orientation,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-2, 2), "y": (-2, 2), "z": (0.1, 0.2)},
    #         "velocity_range": {"x": (0, 0), "y": (0, 0), "z": (0, 0), "roll": (0, 0), "pitch": (0, 0), "yaw": (0, 0)},
    #         "asset_cfg": SceneEntityCfg("cube"),
    #     },
    # )

    # reset_carter_position = EventTerm(
    #     func=mdp.reset_root_state_with_random_orientation,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.2, 0.3)},
    #         "velocity_range": {"x": (0, 0), "y": (0, 0), "z": (0, 0), "roll": (0, 0), "pitch": (0, 0), "yaw": (0, 0)},
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # # (1) Constant running reward
    # alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # # (2) Failure penalty
    # terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # # (3) Primary task: keep pole upright
    # pole_pos = RewTerm(
    #     func=mdp.joint_vel_l1,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_wheel"]), "target": 0.0},
    # )
    # (4) Shaping tasks: lower cart velocity
    # cart_vel = RewTerm(
    #     func=mdp.joint_vel_l1,
    #     weight=-0.01,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_wheel"])},
    # )
    # # (5) Shaping tasks: lower pole angular velocity
    # pole_vel = RewTerm(
    #     func=mdp.joint_vel_l1,
    #     weight=-0.005,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_wheel"])},
    # )

    distance = RewTerm(
        func=mdp.distance_robot2cube,
        weight=1,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    # cart_out_of_bounds = DoneTerm(
    #     func=mdp.joint_pos_out_of_manual_limit,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    # )
    #too_close = DoneTerm(func=mdp.distance_in, time_out=True)


@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""
    pass

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # no commands for this MDP
    null = mdp.NullCommandCfg()


@configclass
class FrankaEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""

    # Scene settings
    scene = FrankaSceneCfg(num_envs=1, env_spacing=2.5)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # No command generator
    commands: CommandsCfg = CommandsCfg()

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        self.decimation = 5  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.01  # sim step every 5ms: 200Hz
        self.episode_length_s = 30
        self.sim.render_interval = self.decimation

