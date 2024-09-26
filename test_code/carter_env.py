

"""
This script demonstrates how to create a simple environment with a cartpole. It combines the concepts of
scene, action, observation and event managers to create an environment.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import torch

import omni.isaac.lab.envs.mdp as mdp
#import omni.isaac.lab_tasks.manager_based.classic.cartpole.mdp as mdp
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg,ManagerBasedRLEnv,ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab_tasks.manager_based.classic.carter.carter_env_cfg import CarterSceneCfg


@configclass
class ActionsCfg:
    """Action specifications for the environment."""
    #
    # joint_efforts = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=5.0)

    # left_wheel_efforts = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["left_wheel"], scale=500.0)
    # right_wheel_efforts = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["right_wheel"], scale=500.0)

    left_wheel_velocity = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["left_wheel"], scale=1.0)
    right_wheel_velocity = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["right_wheel"], scale=1.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        # joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        #joint_vel = ObsTerm(func=mdp.joint_vel)
        #joint_names = ObsTerm(func=mdp.joint_names)

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

    # on reset
    reset_left_wheel_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["left_wheel"]),
            "position_range": (-0 * math.pi, 0 * math.pi),
            "velocity_range": (0.8 * math.pi, 1 * math.pi),
        },
    )

    reset_right_wheel_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["right_wheel"]),
            # "position_range": (-0.125 * math.pi, 0.125 * math.pi),
            # "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
            "position_range": (-0 * math.pi, 0 * math.pi),
            "velocity_range": (-1 * math.pi, -0.8 * math.pi),
        },
    )

    reset_cube_position = EventTerm(
        func=mdp.reset_root_state_with_random_orientation,
        mode="reset",
        params={
            "pose_range": {"x": (-2, 2), "y": (-2, 2), "z": (0.5, 1)},
            "velocity_range": {"x": (0, 0), "y": (0, 0), "z": (0, 0), "roll": (0, 0), "pitch": (0, 0), "yaw": (0, 0)},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )


@configclass
class CarterEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""

    # Scene settings
    scene = CarterSceneCfg(num_envs=1, env_spacing=2.5)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: keep pole upright
    pole_pos = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_wheel"]), "target": 0.0},
    )
    # (4) Shaping tasks: lower cart velocity
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_wheel"])},
    )
    # (5) Shaping tasks: lower pole angular velocity
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_wheel"])},
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    )
def main():
    """Main function."""
    # parse the arguments
    env_cfg = CarterEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup base environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    #joint_efforts = torch.randn_like(env.action_manager.action)
    joint_efforts = torch.tensor([[-5.0, 5.0]])
    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            if count % 100 == 0:
                #joint_efforts = torch.randn_like(env.action_manager.action)
                joint_efforts = torch.tensor([[-5.0,5.0]])
            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_efforts)
            # print current orientation of pole
            print("[Env 0]: Pole joint: ", obs["policy"][0])
            # update counter
            count += 1

            asset: Articulation = env.scene[SceneEntityCfg("robot").name]
            cube : RigidObject =env.scene["cube"]
            #print(asset.joint_names)
            print(cube.data.root_state_w)


    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()