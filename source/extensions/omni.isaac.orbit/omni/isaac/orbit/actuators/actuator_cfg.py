# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Iterable

from omni.isaac.orbit.utils import configclass

from . import actuator_net, actuator_pd
from .actuator_base import ActuatorBase


@configclass
class ActuatorBaseCfg:
    """Configuration for default actuators in an articulation."""

    class_type: type[ActuatorBase] = MISSING
    """The associated actuator class.

    The class should inherit from :class:`omni.isaac.orbit.actuators.actuator_base.ActuatorBase`.
    """

    joint_names_expr: list[str] = MISSING
    """Articulation's joint names that are part of the group.

    Note:
        This can be a list of joint names or a list of regex expressions (e.g. ".*").
    """

    effort_limit: float | None = None
    """Force/Torque limit of the joints in the group. Defaults to :obj:`None`.

    If :obj:`None`, the limit is set to infinity.
    """

    velocity_limit: float | None = None
    """Velocity limit of the joints in the group. Defaults to :obj:`None`.

    If :obj:`None`, the limit is set to infinity.
    """

    stiffness: dict[str, float] | float | None = MISSING
    """Stiffness gains (also known as p-gain) of the joints in the group.

    If :obj:`None`, the stiffness is set to 0.
    """

    damping: dict[str, float] | float | None = MISSING
    """Damping gains (also known as d-gain) of the joints in the group.

    If :obj:`None`, the damping is set to 0.
    """


"""
Implicit Actuator Models.
"""


@configclass
class ImplicitActuatorCfg(ActuatorBaseCfg):
    """Configuration for an implicit actuator.

    Note:
        The PD control is handled implicitly by the simulation.
    """

    class_type: type = actuator_pd.ImplicitActuator


"""
Explicit Actuator Models.
"""


@configclass
class IdealPDActuatorCfg(ActuatorBaseCfg):
    """Configuration for an ideal PD actuator."""

    class_type: type = actuator_pd.IdealPDActuator


@configclass
class DCMotorCfg(IdealPDActuatorCfg):
    """Configuration for direct control (DC) motor actuator model."""

    class_type: type = actuator_pd.DCMotor

    saturation_effort: float = MISSING
    """Peak motor force/torque of the electric DC motor (in N-m)."""


@configclass
class ActuatorNetLSTMCfg(DCMotorCfg):
    """Configuration for LSTM-based actuator model."""

    class_type: type = actuator_net.ActuatorNetLSTM
    # we don't use stiffness and damping for actuator net
    stiffness = None
    damping = None

    network_file: str = MISSING
    """Path to the file containing network weights."""


@configclass
class ActuatorNetMLPCfg(DCMotorCfg):
    """Configuration for MLP-based actuator model."""

    class_type: type = actuator_net.ActuatorNetMLP
    # we don't use stiffness and damping for actuator net
    stiffness = None
    damping = None

    network_file: str = MISSING
    """Path to the file containing network weights."""

    pos_scale: float = MISSING
    """Scaling of the joint position errors input to the network."""
    vel_scale: float = MISSING
    """Scaling of the joint velocities input to the network."""
    torque_scale: float = MISSING
    """Scaling of the joint efforts output from the network."""
    input_idx: Iterable[int] = MISSING
    """
    Indices of the actuator history buffer passed as inputs to the network.

    The index *0* corresponds to current time-step, while *n* corresponds to n-th
    time-step in the past. The allocated history length is `max(input_idx) + 1`.
    """
