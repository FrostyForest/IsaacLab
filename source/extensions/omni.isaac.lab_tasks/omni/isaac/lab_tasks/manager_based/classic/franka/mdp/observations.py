from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import RayCaster,camera
import torch.nn.functional as F
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def hand_camera_data(env: ManagerBasedRLEnv)-> torch.Tensor:

    sensor: camera = env.scene.sensors["camera_hand"]
    data=sensor.data.output["rgb"]
    output_tensor = F.interpolate(data.float().permute(0, 3, 1, 2), size=(120, 160), mode='bilinear', align_corners=False)
    return output_tensor/255

def bottom_camera_data(env: ManagerBasedRLEnv)-> torch.Tensor:

    sensor: camera = env.scene.sensors["camera_bottom"]
    data=sensor.data.output["rgb"]
    output_tensor = F.interpolate(data.float().permute(0, 3, 1, 2), size=(120, 160), mode='bilinear', align_corners=False)
    return output_tensor/255