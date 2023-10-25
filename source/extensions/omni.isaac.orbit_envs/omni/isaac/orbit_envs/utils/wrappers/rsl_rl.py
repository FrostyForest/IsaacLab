# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure an :class:`RLEnv` instance to RSL-RL vectorized environment.

The following example shows how to wrap an environment for RSL-RL:

.. code-block:: python

    from omni.isaac.orbit_envs.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

    env = RslRlVecEnvWrapper(env)

"""

from __future__ import annotations

import copy
import gym
import gym.spaces
import os
import torch
from typing import Dict, Optional, Tuple

from omni.isaac.orbit.envs import RLEnv

__all__ = ["RslRlVecEnvWrapper", "export_policy_as_jit", "export_policy_as_onnx"]


"""
Vectorized environment wrapper.
"""

# VecEnvObs is what is returned by the reset() method
# it contains the observation for each env
VecEnvObs = Tuple[torch.Tensor, Optional[torch.Tensor]]
# VecEnvStepReturn is what is returned by the step() method
# it contains the observation (actor and critic), reward, done, info for each env
VecEnvStepReturn = Tuple[VecEnvObs, VecEnvObs, torch.Tensor, torch.Tensor, Dict]


class RslRlVecEnvWrapper(gym.Wrapper):
    """Wraps around Isaac Orbit environment for RSL-RL.

    To use asymmetric actor-critic, the environment instance must have the attributes :attr:`num_states` (int)
    and :attr:`state_space` (:obj:`gym.spaces.Box`). These are used by the learning agent to allocate buffers in
    the trajectory memory. Additionally, the method :meth:`_get_observations()` should have the key "critic"
    which corresponds to the privileged observations. Since this is optional for some environments, the wrapper
    checks if these attributes exist. If they don't then the wrapper defaults to zero as number of privileged
    observations.

    Reference:
        https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/env/vec_env.py
    """

    def __init__(self, env: RLEnv):
        """Initializes the wrapper.

        Args:
            env: The environment to wrap around.

        Raises:
            ValueError: When the environment is not an instance of :class:`RLEnv`.
            ValueError: When the observation space is not a :obj:`gym.spaces.Box`.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, RLEnv):
            raise ValueError(f"The environment must be inherited from RLEnv. Environment type: {type(env)}")
        # initialize the wrapper
        gym.Wrapper.__init__(self, env)
        # store information required by wrapper
        orbit_env: RLEnv = self.env.unwrapped
        self.num_envs = orbit_env.num_envs
        self.num_actions = orbit_env.action_manager.total_action_dim
        self.num_obs = orbit_env.observation_manager.group_obs_dim["policy"][0]
        # reset at the start since the RSL-RL runner does not call reset
        self.env.reset()

    """
    Properties
    """

    def get_observations(self) -> torch.Tensor:
        """Returns the current observations of the environment."""
        obs_dict = self.env.unwrapped.observation_manager.compute()
        return obs_dict["policy"], {"observations": obs_dict}

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.env.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note: This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        self.env.unwrapped.episode_length_buf = value

    """
    Operations - MDP
    """

    def reset(self) -> VecEnvObs:  # noqa: D102
        # reset the environment
        obs_dict = self.env.reset()
        # return observations
        return obs_dict["policy"], {"observations": obs_dict}

    def step(self, actions: torch.Tensor) -> VecEnvStepReturn:  # noqa: D102
        # record step information
        obs_dict, rew, dones, extras = self.env.step(actions)
        # return step information
        obs = obs_dict["policy"]
        extras["observations"] = obs_dict
        return obs, rew, dones.to(torch.long), extras


"""
Helper functions.
"""


def export_policy_as_jit(actor_critic: object, path: str, filename="policy.pt"):
    """Export policy into a Torch JIT file.

    Args:
        actor_critic: The actor-critic torch module.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".

    Reference:
        https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/utils/helpers.py#L180
    """
    policy_exporter = _TorchPolicyExporter(actor_critic)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(actor_critic: object, path: str, filename="policy.onnx", verbose=False):
    """Export policy into a Torch ONNX file.

    Args:
        actor_critic: The actor-critic torch module.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(actor_critic, verbose)
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file.

    Reference:
        https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/utils/helpers.py#L193
    """

    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        if self.is_recurrent:
            self.rnn = copy.deepcopy(actor_critic.memory_a.rnn)
            self.rnn.cpu()
            self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            self.forward = self.forward_lstm
            self.reset = self.reset_memory

    def forward_lstm(self, x):
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return self.actor(x)

    def forward(self, x):
        return self.actor(x)

    @torch.jit.export
    def reset(self):
        pass

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, actor_critic, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        if self.is_recurrent:
            self.rnn = copy.deepcopy(actor_critic.memory_a.rnn)
            self.rnn.cpu()
            self.forward = self.forward_lstm

    def forward_lstm(self, x_in, h_in, c_in):
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h, c

    def forward(self, x):
        return self.actor(x)

    def export(self, path, filename):
        self.to("cpu")
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            actions, h_out, c_out = self(obs, h_in, c_in)
            torch.onnx.export(
                self,
                (obs, h_in, c_in),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs", "h_in", "c_in"],
                output_names=["actions", "h_out", "c_out"],
                dynamic_axes={},
            )
        else:
            obs = torch.zeros(1, self.actor[0].in_features)
            torch.onnx.export(
                self,
                obs,
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )
