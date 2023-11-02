# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This subpackage contains the sensor classes that are compatible with Isaac Sim. We include both
USD-based and custom sensors. The USD-based sensors are the ones that are available in Omniverse and
require creating a USD prim for them. Custom sensors, on the other hand, are the ones that are
implemented in Python and do not require creating a USD prim for them.

A prim path (or expression) is still set for each sensor based on the following schema:

+-------------------+--------------------------+---------------------------------------------------------------+
| Sensor Type       | Example Prim Path        | Pre-check                                                     |
+===================+==========================+===============================================================+
| Camera            | /World/robot/base/camera | Leaf is available, and it will spawn a USD camera             |
| Contact Sensor    | /World/robot/feet_*      | Leaf is available and checks if the schema exists             |
| Ray Casters       | /World/robot/base        | Leaf exists and is a physics body (Articulation / Rigid Body) |
| Frame Transformer | /World/robot/base        | Leaf exists and is a physics body (Articulation / Rigid Body) |
+-------------------+--------------------------+---------------------------------------------------------------+

"""

from __future__ import annotations

from .camera import *  # noqa: F401, F403
from .contact_sensor import *  # noqa: F401, F403
from .frame_transformer import *  # noqa: F401
from .ray_caster import *  # noqa: F401, F403
from .sensor_base import SensorBase  # noqa: F401
from .sensor_base_cfg import SensorBaseCfg  # noqa: F401
