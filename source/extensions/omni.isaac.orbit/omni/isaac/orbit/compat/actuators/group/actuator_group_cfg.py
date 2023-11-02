# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This submodule contains the configuration for the ActuatorGroup classes.

Currently, the following configuration classes are supported:

* ActuatorGroupCfg: standard actuator group with joint-level controllers
* GripperActuatorGroupCfg: actuator group for grippers (mimics transmission across joints)
* NonHolonomicKinematicsGroupCfg: actuator group for a non-holonomic kinematic constraint

"""

from __future__ import annotations

from dataclasses import MISSING
from typing import ClassVar

from omni.isaac.orbit.actuators.model.actuator_cfg import BaseActuatorCfg
from omni.isaac.orbit.utils import configclass

from .actuator_control_cfg import ActuatorControlCfg


@configclass
class ActuatorGroupCfg:
    """Configuration for default group of actuators in an articulation."""

    class_type: ClassVar[str] = "ActuatorGroup"
    """Name of the associated actuator group class. Used to construct the actuator group."""

    dof_names: list[str] = MISSING
    """Articulation's DOF names that are part of the group."""

    model_cfg: BaseActuatorCfg = MISSING
    """Actuator model configuration used by the group."""

    control_cfg: ActuatorControlCfg = MISSING
    """Actuator control configuration used by the group."""


@configclass
class GripperActuatorGroupCfg(ActuatorGroupCfg):
    """Configuration for mimicking actuators in a gripper."""

    class_type: ClassVar[str] = "GripperActuatorGroup"

    speed: float = MISSING
    """The speed at which gripper should close. (used with velocity command type.)"""

    mimic_multiplier: dict[str, float] = MISSING
    """
    Mapping of command from DOF names to command axis [-1, 1].

    The first joint in the dictionary is considered the joint to mimic.

    For convention purposes:

    - :obj:`+1` -> opening direction
    - :obj:`-1` -> closing direction
    """

    open_dof_pos: float = MISSING
    """The DOF position at *open* configuration. (used with position command type.)"""

    close_dof_pos: float = MISSING
    """The DOF position at *close* configuration. (used with position command type.)"""


@configclass
class NonHolonomicKinematicsGroupCfg(ActuatorGroupCfg):
    """Configuration for formulating non-holonomic base constraint."""

    class_type: ClassVar[str] = "NonHolonomicKinematicsGroup"
