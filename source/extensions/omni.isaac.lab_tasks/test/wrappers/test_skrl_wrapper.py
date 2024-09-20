# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from omni.isaac.lab.app import AppLauncher, run_tests

# launch the simulator
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import torch
import unittest

import omni.usd

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper


class TestSKRLVecEnvWrapper(unittest.TestCase):
    """Test that SKRL VecEnv wrapper works as expected."""

    @classmethod
    def setUpClass(cls):
        # acquire all Isaac environments names
        cls.registered_tasks = list()
        for task_spec in gym.registry.values():
            if "Isaac" in task_spec.id:
                cfg_entry_point = gym.spec(task_spec.id).kwargs.get("skrl_cfg_entry_point")
                if cfg_entry_point is not None:
                    cls.registered_tasks.append(task_spec.id)
        # sort environments by name
        cls.registered_tasks.sort()
        cls.registered_tasks = cls.registered_tasks[:4]
        # print all existing task names
        print(">>> All registered environments:", cls.registered_tasks)

    def setUp(self) -> None:
        # common parameters
        self.num_envs = 64
        self.device = "cuda"

    def test_random_actions(self):
        """Run random actions and check environments return valid signals."""
        for task_name in self.registered_tasks:
            with self.subTest(task_name=task_name):
                print(f">>> Running test for environment: {task_name}")
                # create a new stage
                omni.usd.get_context().new_stage()
                # parse configuration
                env_cfg = parse_env_cfg(task_name, device=self.device, num_envs=self.num_envs)
                agent_cfg = load_cfg_from_registry(task_name, "skrl_cfg_entry_point")  # noqa: F841
                # create environment
                env = gym.make(task_name, cfg=env_cfg)
                if isinstance(env.unwrapped, DirectMARLEnv):
                    env = multi_agent_to_single_agent(env)
                # wrap environment
                env = SkrlVecEnvWrapper(env)

                # reset environment
                obs, extras = env.reset()
                # check signal
                self.assertTrue(self._check_valid_tensor(obs))
                self.assertTrue(self._check_valid_tensor(extras))

                # simulate environment for 1000 steps
                with torch.inference_mode():
                    for _ in range(10):
                        # sample actions from -1 to 1
                        actions = (
                            2 * torch.rand(self.num_envs, *env.action_space.shape, device=env.unwrapped.device) - 1
                        )
                        # apply actions
                        transition = env.step(actions)
                        # check signals
                        for data in transition:
                            self.assertTrue(self._check_valid_tensor(data), msg=f"Invalid data: {data}")

                # close the environment
                print(f">>> Closing environment: {task_name}")
                env.close()

    """
    Helper functions.
    """

    @staticmethod
    def _check_valid_tensor(data: torch.Tensor | dict) -> bool:
        """Checks if given data does not have corrupted values.

        Args:
            data: Data buffer.

        Returns:
            True if the data is valid.
        """
        if isinstance(data, torch.Tensor):
            return not torch.any(torch.isnan(data))
        elif isinstance(data, dict):
            valid_tensor = True
            for value in data.values():
                if isinstance(value, dict):
                    valid_tensor &= TestSKRLVecEnvWrapper._check_valid_tensor(value)
                elif isinstance(value, torch.Tensor):
                    valid_tensor &= not torch.any(torch.isnan(value))
            return valid_tensor
        else:
            raise ValueError(f"Input data of invalid type: {type(data)}.")


if __name__ == "__main__":
    run_tests()
