# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from omni.isaac.orbit.app import AppLauncher

# Can set this to False to see the GUI for debugging
HEADLESS = True

# launch omniverse app
app_launcher = AppLauncher(headless=HEADLESS)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import unittest

import omni.usd

from omni.isaac.orbit.envs import BaseEnv, BaseEnvCfg
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.utils import configclass


@configclass
class EmptyActionsCfg:
    """Action specifications for the environment."""

    pass


@configclass
class EmptySceneCfg(InteractiveSceneCfg):
    """Configuration for an empty scene."""

    pass


def get_empty_base_env_cfg(device: str = "cuda:0", num_envs: int = 1, env_spacing: float = 1.0):
    """Generate base environment config based on device"""

    @configclass
    class EmptyEnvCfg(BaseEnvCfg):
        """Configuration for the empty test environment."""

        # Scene settings
        scene: EmptySceneCfg = EmptySceneCfg(num_envs=num_envs, env_spacing=env_spacing)
        # Basic settings
        actions: EmptyActionsCfg = EmptyActionsCfg()

        def __post_init__(self):
            """Post initialization."""
            # step settings
            self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
            # simulation settings
            self.sim.dt = 0.005  # sim step every 5ms: 200Hz
            # pass device down from test
            self.sim.device = device

    return EmptyEnvCfg()


class TestBaseEnv(unittest.TestCase):
    """Test for base env class"""

    """
    Tests
    """

    def test_initialization(self):
        for device in ("cuda:0", "cpu"):
            with self.subTest(device=device):
                # create a new stage
                omni.usd.get_context().new_stage()
                # create environment
                env = BaseEnv(cfg=get_empty_base_env_cfg(device=device))
                # check size of action manager terms
                self.assertEqual(env.action_manager.total_action_dim, 0)
                self.assertEqual(len(env.action_manager.active_terms), 0)
                self.assertEqual(len(env.action_manager.action_term_dim), 0)
                # check size of observation manager terms
                self.assertEqual(len(env.observation_manager.active_terms), 0)
                self.assertEqual(len(env.observation_manager.group_obs_dim), 0)
                self.assertEqual(len(env.observation_manager.group_obs_term_dim), 0)
                self.assertEqual(len(env.observation_manager.group_obs_concatenate), 0)
                # create actions of correct size (1,0)
                act = torch.randn_like(env.action_manager.action)
                # step environment to verify setup
                for _ in range(2):
                    obs, ext = env.step(action=act)
                # close the environment
                env.close()
