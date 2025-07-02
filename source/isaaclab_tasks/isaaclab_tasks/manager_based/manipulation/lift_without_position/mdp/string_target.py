# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

PREDEFINED_TARGETS = ["yellow_cube", "green_cube", "red_cube"]
TARGET_TO_ID = {name: i for i, name in enumerate(PREDEFINED_TARGETS)}
ID_TO_TARGET = {i: name for i, name in enumerate(PREDEFINED_TARGETS)}
NUM_TARGETS = len(PREDEFINED_TARGETS)
