# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import carb

import omni.isaac.lab.utils.string as string_utils
from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

    from . import actions_cfg


class AssetAction(ActionTerm):
    r"""
    魔改成对应整个资产的动作组合
    """

    cfg: actions_cfg.JointActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _offset: torch.Tensor | float
    """The offset applied to the input action."""

    def __init__(self, cfg: actions_cfg.JointActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        # log the resolved joint names for debugging
        carb.log_info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")
        # parse offset
        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)
        elif isinstance(cfg.offset, dict):
            self._offset = torch.zeros_like(self._raw_actions)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.offset, self._joint_names)
            self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}. Supported types are float and dict.")

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def fix_actions(self):
        action1 = torch.tensor([-1, -1], dtype=torch.float,device=self.device)
        action1 = action1.unsqueeze(0).repeat(self.num_envs, 1)

        action2 = torch.tensor([1, -1], dtype=torch.float,device=self.device)
        action2 = action2.unsqueeze(0).repeat(self.num_envs, 1)

        action3 = torch.tensor([1, 1], dtype=torch.float,device=self.device)
        action3 = action3.unsqueeze(0).repeat(self.num_envs, 1)

        action4 = torch.tensor([-1, 1], dtype=torch.float,device=self.device)
        action4 = action4.unsqueeze(0).repeat(self.num_envs, 1)

        actions = torch.stack([action1, action2,action3,action4], dim=1)

        return actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions

        softmax_actions = torch.softmax(self._raw_actions, dim=-1)
        # apply the affine transformations
        self._processed_actions = torch.einsum("ij,ijk->ik", softmax_actions, self.fix_actions)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = torch.tensor([1,1,1,1],dtype=torch.float,device=self.device)#stop

    def apply_actions(self):
        # add current joint positions to the processed actions
        current_actions = self.processed_actions + self._asset.data.joint_pos[:, self._joint_ids]
        # set position targets
        self._asset.set_joint_velocity_target(current_actions, joint_ids=self._joint_ids)


# class JointPositionAction(JointAction):
#     """Joint action term that applies the processed actions to the articulation's joints as position commands."""
#
#     cfg: actions_cfg.JointPositionActionCfg
#     """The configuration of the action term."""
#
#     def __init__(self, cfg: actions_cfg.JointPositionActionCfg, env: ManagerBasedEnv):
#         # initialize the action term
#         super().__init__(cfg, env)
#         # use default joint positions as offset
#         if cfg.use_default_offset:
#             self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
#
#     def apply_actions(self):
#         # set position targets
#         self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)
#
#
# class RelativeJointPositionAction(JointAction):
#     r"""Joint action term that applies the processed actions to the articulation's joints as relative position commands.
#
#     Unlike :class:`JointPositionAction`, this action term applies the processed actions as relative position commands.
#     This means that the processed actions are added to the current joint positions of the articulation's joints
#     before being sent as position commands.
#
#     This means that the action applied at every step is:
#
#     .. math::
#
#          \text{applied action} = \text{current joint positions} + \text{processed actions}
#
#     where :math:`\text{current joint positions}` are the current joint positions of the articulation's joints.
#     """
#
#     cfg: actions_cfg.RelativeJointPositionActionCfg
#     """The configuration of the action term."""
#
#     def __init__(self, cfg: actions_cfg.RelativeJointPositionActionCfg, env: ManagerBasedEnv):
#         # initialize the action term
#         super().__init__(cfg, env)
#         # use zero offset for relative position
#         if cfg.use_zero_offset:
#             self._offset = 0.0
#
#     def apply_actions(self):
#         # add current joint positions to the processed actions
#         current_actions = self.processed_actions + self._asset.data.joint_pos[:, self._joint_ids]
#         # set position targets
#         self._asset.set_joint_position_target(current_actions, joint_ids=self._joint_ids)
#
#
# class JointVelocityAction(JointAction):
#     """Joint action term that applies the processed actions to the articulation's joints as velocity commands."""
#
#     cfg: actions_cfg.JointVelocityActionCfg
#     """The configuration of the action term."""
#
#     def __init__(self, cfg: actions_cfg.JointVelocityActionCfg, env: ManagerBasedEnv):
#         # initialize the action term
#         super().__init__(cfg, env)
#         # use default joint velocity as offset
#         if cfg.use_default_offset:
#             self._offset = self._asset.data.default_joint_vel[:, self._joint_ids].clone()
#
#     def apply_actions(self):
#         # set joint velocity targets
#         self._asset.set_joint_velocity_target(self.processed_actions, joint_ids=self._joint_ids)
#
#
# class JointEffortAction(JointAction):
#     """Joint action term that applies the processed actions to the articulation's joints as effort commands."""
#
#     cfg: actions_cfg.JointEffortActionCfg
#     """The configuration of the action term."""
#
#     def __init__(self, cfg: actions_cfg.JointEffortActionCfg, env: ManagerBasedEnv):
#         super().__init__(cfg, env)
#
#     def apply_actions(self):
#         # set joint effort targets
#         self._asset.set_joint_effort_target(self.processed_actions, joint_ids=self._joint_ids)
