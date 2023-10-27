# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import asyncio
import gym
import math
import numpy as np
import torch
import weakref
from typing import Any, ClassVar, Dict, Sequence, Tuple, Union

import omni.usd

from omni.isaac.orbit.command_generators import CommandGeneratorBase
from omni.isaac.orbit.managers import CurriculumManager, RewardManager, TerminationManager

from .base_env import BaseEnv
from .rl_env_cfg import RLEnvCfg

VecEnvObs = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
"""Observation returned by the environment.

The observations are stored in a dictionary. The keys are the group to which the observations belong.
This is useful for various learning setups beyond vanilla reinforcement learning, such as asymmetric
actor-critic, multi-agent, or hierarchical reinforcement learning.

For example, for asymmetric actor-critic, the observation for the actor and the critic can be accessed
using the keys ``"policy"`` and ``"critic"`` respectively.

Within each group, the observations can be stored either as a dictionary with keys as the names of each
observation term in the group, or a single tensor obtained from concatenating all the observation terms.

Note:
    By default, most learning frameworks deal with default and privileged observations in different ways.
    This handling must be taken care of by the wrapper around the :class:`RLEnv` instance.

    For included frameworks (RSL-RL, RL-Games, skrl), the observations must have the key "policy". In case,
    the key "critic" is also present, then the critic observations are taken from the "critic" group.
    Otherwise, they are the same as the "policy" group.

"""


VecEnvStepReturn = Tuple[VecEnvObs, torch.Tensor, torch.Tensor, Dict]
"""The environment signals processed at the end of each step.

It contains the observation, reward, termination signal and additional information for each sub-environment.
"""


class RLEnv(BaseEnv, gym.Env):
    """The superclass for reinforcement learning-based environments.

    This class inherits from :class:`BaseEnv` and implements the core functionality for
    reinforcement learning-based environments. It is designed to be used with any RL
    library. The class is designed to be used with vectorized environments, i.e., the
    environment is expected to be run in parallel with multiple sub-environments. The
    number of sub-environments is specified using the ``num_envs``.

    Each observation from the environment is a batch of observations for each sub-
    environments. The method :meth:`step` is also expected to receive a batch of actions
    for each sub-environment.

    While the environment itself is implemented as a vectorized environment, we do not
    inherit from :class:`gym.vector.VectorEnv`. This is mainly because the class adds
    various methods (for wait and asynchronous updates) which are not required.
    Additionally, each RL library typically has its own definition for a vectorized
    environment. Thus, to reduce complexity, we directly use the :class:`gym.Env` over
    here and leave it up to library-defined wrappers to take care of wrapping this
    environment for their agents.
    """

    is_vector_env: ClassVar[bool] = True
    """Whether the environment is a vectorized environment."""
    metadata: ClassVar[dict[str, Any]] = {"render.modes": ["human", "rgb_array"]}
    """Metadata for the environment."""

    cfg: RLEnvCfg
    """Configuration for the environment."""

    def __init__(self, cfg: RLEnvCfg):
        # initialize the base class to setup the scene.
        super().__init__(cfg=cfg)

        # initialize data and constants
        # -- counter for curriculum
        self.common_step_counter = 0
        # -- init buffers
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.reward_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # -- allocate dictionary to store metrics
        self.extras = {}
        # print the environment information
        print("[INFO]: Completed setting up the environment...")

        # setup the action and observation spaces for Gym
        # -- observation space
        self.observation_space = gym.spaces.Dict()
        for group_name, group_dim in self.observation_manager.group_obs_dim.items():
            self.observation_space[group_name] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=group_dim)
        # -- action space (unbounded since we don't impose any limits)
        action_dim = sum(self.action_manager.action_term_dim)
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(action_dim,))

        # perform randomization at the start of the simulation
        if "startup" in self.randomization_manager.available_modes:
            self.randomization_manager.randomize(mode="startup")
        # extend UI elements
        # we need to do this here after all the managers are initialized
        # this is because they dictate the sensors and commands right now
        if self.sim.has_gui():
            self._build_ui()
        else:
            # if no window, then we don't need to store the window
            self._orbit_window = None
            self._orbit_window_elements = dict()

    """
    Properties.
    """

    @property
    def max_episode_length_s(self) -> float:
        """Maximum episode length in seconds."""
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self) -> int:
        """Maximum episode length in environment steps."""
        return math.ceil(self.max_episode_length_s / self.step_dt)

    """
    Operations - Setup.
    """

    def load_managers(self):
        # note: this order is important since observation manager needs to know the command and action managers
        # -- command manager
        self.command_manager: CommandGeneratorBase = self.cfg.commands.class_type(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)
        # call the parent class to load the managers for observations and actions.
        super().load_managers()
        # prepare the managers
        # -- reward manager
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        # -- termination manager
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        # -- curriculum manager
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

    """
    Operations - MDP
    """

    def reset(self) -> VecEnvObs:
        """Resets all the environments and returns observations.

        Note:
            This function (if called) must **only** be called before the first call to :meth:`step`, i.e.
            after the environment is created. After that, the :meth:`step` function handles the reset
            of terminated sub-environments.

        Returns:
            Observations from the environment.
        """
        # reset state of scene
        indices = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        self._reset_idx(indices)
        # return observations
        return self.observation_manager.compute()

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Apply actions on the environment and reset terminated environments.

        This function deals with various timeline events (play, pause and stop) for clean execution.
        When the simulation is stopped all the physics handles expire and we cannot perform any read or
        write operations. The timeline event is only detected after every `sim.step()` call. Hence, at
        every call we need to check the status of the simulator. The logic is as follows:

        1. If the simulation is stopped, the environment is closed and the simulator is shutdown.
        2. If the simulation is paused, we step the simulator until it is playing.
        3. If the simulation is playing, we set the actions and step the simulator.

        Args:
            action: Actions to apply on the simulator.

        Returns:
            VecEnvStepReturn: A tuple containing:
                - (VecEnvObs) observations from the environment
                - (torch.Tensor) reward from the environment
                - (torch.Tensor) whether the current episode is completed or not
                - (dict) misc information
        """
        # process actions
        self.action_manager.process_action(action)
        # perform physics stepping
        for _ in range(self.cfg.decimation):
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)
        # perform rendering if gui is enabled
        if self.sim.has_gui():
            self.sim.render()

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        # compute MDP signals
        # -- check terminations
        self.reset_buf = self.termination_manager.compute().to(torch.long)
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        # -- reset envs that terminated and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval randomization
        if "interval" in self.randomization_manager.available_modes:
            self.randomization_manager.randomize(mode="interval", dt=self.step_dt)

        # return observations, rewards, resets and extras
        return self.observation_manager.compute(), self.reward_buf, self.reset_buf, self.extras

    def render(self, mode: str = "human") -> np.ndarray | None:
        """Run rendering without stepping through the physics.

        By convention, if mode is:

        - **human**: render to the current display and return nothing. Usually for human consumption.
        - **rgb_array**: Return an numpy.ndarray with shape (x, y, 3), representing RGB values for an
          x-by-y pixel image, suitable for turning into a video.

        Args:
            mode: The mode to render with. Defaults to "human".

        Returns:
            The rendered image as a numpy array if mode is "rgb_array".

        Raises:
            RuntimeError: If mode is set to "rgb_data" and simulation render mode does not support it.
                In this case, the simulation render mode must be set to ``RenderMode.PARTIAL_RENDERING``
                or ``RenderMode.FULL_RENDERING``.
            NotImplementedError: If an unsupported rendering mode is specified.
        """
        # run a rendering step of the simulator
        self.sim.render()
        # decide the rendering mode
        if mode == "human":
            return None
        elif mode == "rgb_array":
            # check that if any render could have happened
            if self.sim.render_mode.value < self.sim.RenderMode.PARTIAL_RENDERING.value:
                raise RuntimeError(
                    f"Cannot render '{mode}' when the simulation render mode is '{self.sim.render_mode.name}'."
                    f" Please set the simulation render mode to '{self.sim.RenderMode.PARTIAL_RENDERING.name}' or "
                    f" '{self.sim.RenderMode.FULL_RENDERING.name}'."
                )
            # create the annotator if it does not exist
            if not hasattr(self, "_rgb_annotator"):
                import omni.replicator.core as rep

                # create render product
                self._render_product = rep.create.render_product(
                    self.cfg.viewer.cam_prim_path, self.cfg.viewer.resolution
                )
                # create rgb annotator -- used to read data from the render product
                self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
                self._rgb_annotator.attach([self._render_product])
            # obtain the rgb data
            rgb_data = self._rgb_annotator.get_data()
            # convert to numpy array
            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            # return the rgb data
            # note: initially the renerer is warming up and returns empty data
            if rgb_data.size == 0:
                return np.zeros((self.cfg.viewer.resolution[1], self.cfg.viewer.resolution[0], 3), dtype=np.uint8)
            else:
                return rgb_data[:, :, :3]
        else:
            raise NotImplementedError(
                f"Render mode '{mode}' is not supported. Please use: {self.metadata['render.modes']}."
            )

    def close(self):
        if not self._is_closed:
            # destroy the window
            if self._orbit_window is not None:
                self._orbit_window.visible = False
                self._orbit_window.destroy()
                self._orbit_window = None
            # update closing status
            super().close()

    """
    Implementation specifics.
    """

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        # update the curriculum for environments that need a reset
        self.curriculum_manager.compute(env_ids=env_ids)
        # reset the internal buffers of the scene elements
        self.scene.reset(env_ids)
        # randomize the MDP for environments that need a reset
        if "reset" in self.randomization_manager.available_modes:
            self.randomization_manager.randomize(env_ids=env_ids, mode="reset")

        # iterate over all managers and reset them
        # this returns a dictionary of information which is stored in the extras
        # note: This is order-sensitive! Certain things need be reset before others.
        self.extras["log"] = dict()
        # -- observation manager
        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- action manager
        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- rewards manager
        info = self.reward_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- curriculum manager
        info = self.curriculum_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- command manager
        info = self.command_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- randomization manager
        info = self.randomization_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- termination manager
        info = self.termination_manager.reset(env_ids)
        self.extras["log"].update(info)

        # reset the episode length buffer
        self.episode_length_buf[env_ids] = 0
        #  -- add information to extra if timeout occurred due to episode length
        # Note: this is used by algorithms like PPO where time-outs are handled differently
        self.extras["time_outs"] = self.termination_manager.time_outs

    """
    Helper functions - GUI.
    """

    def _build_ui(self):
        """Constructs the GUI for the environment."""
        # need to import here to wait for the GUI extension to be loaded
        import omni.isaac.ui.ui_utils as ui_utils
        import omni.ui as ui
        from omni.kit.window.extensions import SimpleCheckBox

        # create window for UI
        self._orbit_window = omni.ui.Window(
            "Orbit", width=400, height=500, visible=True, dock_preference=ui.DockPreference.RIGHT_TOP
        )
        # dock next to properties window
        asyncio.ensure_future(self._dock_window(window_title=self._orbit_window.title))

        # keep a dictionary of stacks so that child environments can add their own UI elements
        # this can be done by using the `with` context manager
        self._orbit_window_elements = dict()
        # create main frame
        self._orbit_window_elements["main_frame"] = self._orbit_window.frame
        with self._orbit_window_elements["main_frame"]:
            # create main stack
            self._orbit_window_elements["main_vstack"] = ui.VStack(spacing=5, height=0)
            with self._orbit_window_elements["main_vstack"]:
                # create collapsable frame for controls
                self._orbit_window_elements["control_frame"] = ui.CollapsableFrame(
                    title="Controls",
                    width=ui.Fraction(1),
                    height=0,
                    collapsed=False,
                    style=ui_utils.get_style(),
                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                )
                with self._orbit_window_elements["control_frame"]:
                    # create stack for controls
                    self._orbit_window_elements["controls_vstack"] = ui.VStack(spacing=5, height=0)
                    with self._orbit_window_elements["controls_vstack"]:
                        # create rendering mode dropdown
                        render_mode_cfg = {
                            "label": "Rendering Mode",
                            "type": "dropdown",
                            "default_val": self.sim.render_mode.value,
                            "items": [member.name for member in self.sim.RenderMode if member.value >= 0],
                            "tooltip": "Select a rendering mode\n" + self.sim.RenderMode.__doc__,
                            "on_clicked_fn": lambda value: self.sim.set_render_mode(self.sim.RenderMode[value]),
                        }
                        self._orbit_window_elements["render_dropdown"] = ui_utils.dropdown_builder(**render_mode_cfg)

                        # create a number slider to move to environment origin
                        def viewport_camera_origin_fn(model: ui.SimpleIntModel):
                            """Moves the viewport to the origin of the environment."""
                            # obtain the origin of the environment
                            origin = self.scene.env_origins[model.as_int - 1].detach().cpu().numpy()
                            cam_eye = origin + np.asarray(self.cfg.viewer.eye)
                            cam_target = origin + np.asarray(self.cfg.viewer.lookat)
                            # set the camera view
                            self.sim.set_camera_view(eye=cam_eye, target=cam_target)

                        viewport_origin_cfg = {
                            "label": "View Environment",
                            "type": "button",
                            "default_val": 1,
                            "min": 1,
                            "max": self.num_envs,
                            "tooltip": "Move the viewport to the origin of the environment",
                        }
                        self._orbit_window_elements["viewport_btn"] = ui_utils.int_builder(**viewport_origin_cfg)
                        # create a number slider to move to environment origin
                        self._orbit_window_elements["viewport_btn"].add_value_changed_fn(viewport_camera_origin_fn)

                # create collapsable frame for debug visualization
                self._orbit_window_elements["debug_frame"] = ui.CollapsableFrame(
                    title="Debug Visualization",
                    width=ui.Fraction(1),
                    height=0,
                    collapsed=False,
                    style=ui_utils.get_style(),
                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                )
                with self._orbit_window_elements["debug_frame"]:
                    # create stack for debug visualization
                    self._orbit_window_elements["debug_vstack"] = ui.VStack(spacing=5, height=0)
                    with self._orbit_window_elements["debug_vstack"]:
                        elements = [
                            self.scene.terrain,
                            self.command_manager,
                            *self.scene.rigid_objects.values(),
                            *self.scene.articulations.values(),
                            *self.scene.sensors.values(),
                        ]
                        names = [
                            "terrain",
                            "commands",
                            *self.scene.rigid_objects.keys(),
                            *self.scene.articulations.keys(),
                            *self.scene.sensors.keys(),
                        ]
                        # create one for the terrain
                        for elem, name in zip(elements, names):
                            if elem is not None:
                                with ui.HStack():
                                    # create the UI element
                                    text = (
                                        "Toggle debug visualization."
                                        if elem.has_debug_vis_implementation
                                        else "Debug visualization not implemented."
                                    )
                                    ui.Label(
                                        name.replace("_", " ").title(),
                                        width=ui_utils.LABEL_WIDTH - 12,
                                        alignment=ui.Alignment.LEFT_CENTER,
                                        tooltip=text,
                                    )
                                    self._orbit_window_elements[f"{name}_cb"] = SimpleCheckBox(
                                        model=ui.SimpleBoolModel(),
                                        enabled=elem.has_debug_vis_implementation,
                                        checked=elem.cfg.debug_vis,
                                        on_checked_fn=lambda value, e=weakref.proxy(elem): e.set_debug_vis(value),
                                    )
                                    ui_utils.add_line_rect_flourish()

    async def _dock_window(self, window_title: str):
        """Docks the orbit window to the property window."""
        # need to import here to wait for the GUI extension to be loaded
        import omni.ui as ui

        for _ in range(5):
            if ui.Workspace.get_window(window_title):
                break
            await self.sim.app.next_update_async()

        # dock next to properties window
        orbit_window = ui.Workspace.get_window(window_title)
        property_window = ui.Workspace.get_window("Property")

        if orbit_window and property_window:
            orbit_window.dock_in(property_window, ui.DockPosition.SAME, 1.0)
            orbit_window.focus()
