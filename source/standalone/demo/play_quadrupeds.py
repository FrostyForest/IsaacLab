# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to simulate a legged robot.

We currently support the following robots:

* ANYmal-B (from ANYbotics)
* ANYmal-C (from ANYbotics)
* A1 (from Unitree Robotics)
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate a legged robot.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import traceback

import carb

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.assets.config.anymal import ANYMAL_B_CFG, ANYMAL_C_CFG
from omni.isaac.orbit.assets.config.unitree import UNITREE_A1_CFG
from omni.isaac.orbit.sim import SimulationContext


def main():
    """Main function."""

    # Load kit helper
    sim = SimulationContext(
        sim_utils.SimulationCfg(device="cpu", use_gpu_pipeline=False, dt=0.005, physx=sim_utils.PhysxCfg(use_gpu=False))
    )
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])

    # Spawn things into stage
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Robots
    # -- anymal-b
    robot_b_cfg = ANYMAL_B_CFG
    robot_b_cfg.spawn.func("/World/Anymal_b/Robot_1", robot_b_cfg.spawn, translation=(0.0, -1.5, 0.65))
    robot_b_cfg.spawn.func("/World/Anymal_b/Robot_2", robot_b_cfg.spawn, translation=(0.0, -0.5, 0.65))
    # -- anymal-c
    robot_c_cfg = ANYMAL_C_CFG
    robot_c_cfg.spawn.func("/World/Anymal_c/Robot_1", robot_c_cfg.spawn, translation=(1.5, -1.5, 0.65))
    robot_c_cfg.spawn.func("/World/Anymal_c/Robot_2", robot_c_cfg.spawn, translation=(1.5, -0.5, 0.65))
    # -- unitree a1
    robot_a_cfg = UNITREE_A1_CFG
    robot_a_cfg.spawn.func("/World/Unitree_A1/Robot_1", robot_a_cfg.spawn, translation=(1.5, 0.5, 0.42))
    robot_a_cfg.spawn.func("/World/Unitree_A1/Robot_2", robot_a_cfg.spawn, translation=(1.5, 1.5, 0.42))

    # create handles for the robots
    robot_b = Articulation(robot_b_cfg.replace(prim_path="/World/Anymal_b/Robot.*"))
    robot_c = Articulation(robot_c_cfg.replace(prim_path="/World/Anymal_c/Robot.*"))
    robot_a = Articulation(robot_a_cfg.replace(prim_path="/World/Unitree_A1/Robot.*"))

    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 1000 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset dof state
            for robot in [robot_a, robot_b, robot_c]:
                joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                robot.reset()
            # reset command
            print(">>>>>>>> Reset!")
        # apply action to the robot
        for robot in [robot_a, robot_b, robot_c]:
            robot.set_joint_position_target(robot.data.default_joint_pos.clone())
            robot.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        for robot in [robot_a, robot_b, robot_c]:
            robot.update(sim_dt)


if __name__ == "__main__":
    try:
        # run the main execution
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
