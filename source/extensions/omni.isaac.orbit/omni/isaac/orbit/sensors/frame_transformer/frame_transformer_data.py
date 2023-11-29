# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import dataclass


@dataclass
class FrameTransformerData:
    """Data container for the frame transformer sensor."""

    target_frame_names: list[str] = None
    """Target frame names (this denotes the order that frame data will be ordered).

    The frame names are resolved from the :attr:`FrameTransformerCfg.FrameCfg.name` field.
    This usually follows the order in which the frames are defined in the config. However, in the case of
    regex matching, the order may be different.
    """

    target_pos_source: torch.Tensor = None
    """Position of the target frame(s) relative to source frame.

    Shape is (N, M, 3), where N is the number of environments, and M is the number of target frames.
    """
    target_rot_source: torch.Tensor = None
    """Orientation of the target frame(s) relative to source frame quaternion (w, x, y, z).

    Shape is (N, M, 4), where N is the number of environments, and M is the number of target frames.
    """
    target_pos_w: torch.Tensor = None
    """Position of the target frame(s) after offset (in world frame).

    Shape is (N, M, 3), where N is the number of environments, and M is the number of target frames.
    """
    target_rot_w: torch.Tensor = None
    """Orientation of the target frame(s) after offset (in world frame) quaternion (w, x, y, z).

    Shape is (N, M, 4), where N is the number of environments, and M is the number of target frames.
    """
    source_pos_w: torch.Tensor = None
    """Position of the source frame after offset (in world frame).

    Shape is (N, 3), where N is the number of environments.
    """
    source_rot_w: torch.Tensor = None
    """Orientation of the source frame after offset (in world frame) quaternion (w, x, y, z).

    Shape is (N, 4), where N is the number of environments.
    """
