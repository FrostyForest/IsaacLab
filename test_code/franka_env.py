

"""
This script demonstrates how to create a simple environment with a cartpole. It combines the concepts of
scene, action, observation and event managers to create an environment.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

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

#import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab_tasks.manager_based.classic.carter.mdp as mdp
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg,ManagerBasedRLEnv,ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab_tasks.manager_based.classic.franka.franka_env_cfg import FrankaEnvCfg






def main():
    """Main function."""
    # parse the arguments
    env_cfg = FrankaEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup base environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    #joint_efforts = torch.randn_like(env.action_manager.action)
    #joint_efforts = torch.tensor([[-2.0, 2.0]])
    # simulate physics
    count = 0
    action0 = torch.randn_like(env.action_manager.action)
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")

            # sample random actions
            if count % 1 == 0:
                action=torch.randn_like(env.action_manager.action)
                joint_efforts = action*0.5+action0*0.5
                action0=action

            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_efforts)
            # print current orientation of pole
            #
            #print("[Env 0]: Pole joint: ", obs["policy"][0])
            # update counter
            count += 1

            asset: Articulation = env.scene[SceneEntityCfg("robot").name]
            cube : RigidObject =env.scene["cube"]
            #camera = env.scene["carter_camera_first_person"]
            #print(asset.joint_names)

            robot = env.scene["robot"]
            cube = env.scene["cube"]

            robot_world_pos_default = robot.data.default_root_state[:, :3]
            cube_world_pos_default = cube.data.default_root_state[:, :3]
            distance0 = torch.linalg.norm(robot_world_pos_default[:, :3] - cube_world_pos_default[:, :3])
            robot_world_pos = robot.data.body_pos_w[:, 8, :3]
            cube_world_pos = cube.data.root_pos_w
            distance = torch.linalg.norm(robot_world_pos[:, :3] - cube_world_pos[:, :3])
            print(distance)
            #print(env.action_space.shape)

            # raw_actions = torch.tensor([[1]])
            # # 将原始动作输入给 JointPositionAction 对象
            # left_wheel_velocity_action.process_actions(raw_actions)
            # # 应用处理后的动作到机器人关节
            # left_wheel_velocity_action.apply_actions()

            # 输出机械臂的各个body的名字
            # body_names = asset.data.body_names
            # for i, body_name in enumerate(body_names):
            #     print(f"Body {i}: {body_name}")

            #print(asset.data.body_pos_w)




    # close the environment
    env.close()



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()