# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from omni.isaac.orbit.app import AppLauncher, run_tests

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import numpy as np
import torch
import unittest

import omni.isaac.core.utils.prims as prim_utils
import omni.kit.commands
from omni.isaac.cloner import GridCloner
from omni.isaac.core.materials import PhysicsMaterial, PreviewSurface
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.prims import GeometryPrim, RigidPrim, RigidPrimView

import omni.isaac.orbit.terrains as terrain_gen
from omni.isaac.orbit.sim import SimulationContext, build_simulation_context
from omni.isaac.orbit.terrains import TerrainImporter, TerrainImporterCfg
from omni.isaac.orbit.terrains.config.rough import ROUGH_TERRAINS_CFG
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR


class TestTerrainImporter(unittest.TestCase):
    """Test the terrain importer for different ground and procedural terrains."""

    def test_grid_clone_env_origins(self):
        """Tests that env origins are consistent when computed using the TerrainImporter and IsaacSim GridCloner."""
        # iterate over different number of environments and environment spacing
        for device in ("cuda:0", "cpu"):
            for env_spacing in [1.0, 4.325, 8.0]:
                for num_envs in [1, 4, 125, 379, 1024]:
                    with self.subTest(num_envs=num_envs, env_spacing=env_spacing):
                        with build_simulation_context(device=device, auto_add_lighting=True) as sim:
                            # create terrain importer
                            terrain_importer_cfg = TerrainImporterCfg(
                                num_envs=num_envs,
                                env_spacing=env_spacing,
                                prim_path="/World/ground",
                                terrain_type="plane",  # for flat ground, origins are in grid
                                terrain_generator=None,
                            )
                            terrain_importer = TerrainImporter(terrain_importer_cfg)
                            # obtain env origins using terrain importer
                            terrain_importer_origins = terrain_importer.env_origins

                            # obtain env origins using grid cloner
                            grid_cloner_origins = self._obtain_grid_cloner_env_origins(
                                num_envs, env_spacing, device=sim.device
                            )

                            # check if the env origins are the same
                            torch.testing.assert_close(
                                terrain_importer_origins, grid_cloner_origins, rtol=1e-5, atol=1e-5
                            )

    def test_terrain_generation(self) -> None:
        """Generates assorted terrains and tests that the resulting mesh has the correct size."""
        for device in ("cuda:0", "cpu"):
            with build_simulation_context(device=device, auto_add_lighting=True) as _:
                # Handler for terrains importing
                terrain_importer_cfg = terrain_gen.TerrainImporterCfg(
                    prim_path="/World/ground",
                    max_init_terrain_level=None,
                    terrain_type="generator",
                    terrain_generator=ROUGH_TERRAINS_CFG.replace(curriculum=True),
                    num_envs=1,
                )
                terrain_importer = TerrainImporter(terrain_importer_cfg)

                # check mesh exists
                mesh = terrain_importer.meshes["terrain"]
                self.assertIsNotNone(mesh)

                # calculate expected size from config
                cfg = terrain_importer.cfg.terrain_generator
                self.assertIsNotNone(cfg)
                expectedSizeX = cfg.size[0] * cfg.num_rows + 2 * cfg.border_width
                expectedSizeY = cfg.size[1] * cfg.num_cols + 2 * cfg.border_width

                # get size from mesh bounds
                bounds = mesh.bounds
                actualSize = abs(bounds[1] - bounds[0])

                self.assertAlmostEqual(actualSize[0], expectedSizeX)
                self.assertAlmostEqual(actualSize[1], expectedSizeY)

    def test_plane(self) -> None:
        """Generates a plane and tests that the resulting mesh has the correct size."""
        for device in ("cuda:0", "cpu"):
            with build_simulation_context(device=device, auto_add_lighting=True) as _:

                expectedSizeX = 2.0e6
                expectedSizeY = 2.0e6

                # Handler for terrains importing
                terrain_importer_cfg = terrain_gen.TerrainImporterCfg(
                    prim_path="/World/ground",
                    terrain_type="plane",
                    num_envs=1,
                    env_spacing=1.0,
                )
                terrain_importer = TerrainImporter(terrain_importer_cfg)

                # check mesh exists
                mesh = terrain_importer.meshes["terrain"]
                self.assertIsNotNone(mesh)

                # get size from mesh bounds
                bounds = mesh.bounds
                actualSize = abs(bounds[1] - bounds[0])

                self.assertAlmostEqual(actualSize[0], expectedSizeX)
                self.assertAlmostEqual(actualSize[1], expectedSizeY)

    def test_usd(self) -> None:
        """Imports terrain from a usd and tests that the resulting mesh has the correct size."""
        for device in ("cuda:0", "cpu"):
            with build_simulation_context(device=device, auto_add_lighting=True) as _:
                # Handler for terrains importing
                terrain_importer_cfg = terrain_gen.TerrainImporterCfg(
                    prim_path="/World/ground",
                    terrain_type="usd",
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd",
                    num_envs=1,
                    env_spacing=1.0,
                )
                terrain_importer = TerrainImporter(terrain_importer_cfg)

                # check mesh exists
                mesh = terrain_importer.meshes["terrain"]
                self.assertIsNotNone(mesh)

                # expect values from USD file
                expectedSizeX = 96
                expectedSizeY = 96

                # get size from mesh bounds
                bounds = mesh.bounds
                actualSize = abs(bounds[1] - bounds[0])

                self.assertAlmostEqual(actualSize[0], expectedSizeX)
                self.assertAlmostEqual(actualSize[1], expectedSizeY)

    def test_ball_drop(self) -> None:
        """Generates assorted terrains and spheres. Tests that spheres fall onto terrain and do not pass through it"""
        for device in ("cuda:0", "cpu"):
            with build_simulation_context(device=device, auto_add_lighting=True) as sim:
                self._populate_scene(geom_sphere=False, sim=sim)
                ball_view = RigidPrimView("/World/envs/env_.*/ball", reset_xform_properties=False)
                sim.reset()

                # Initialize the ball views for physics simulation
                ball_view.initialize()

                # Play simulator
                for _ in range(500):
                    sim.step(render=False)

                # Ball may have some small non-zero velocity if the roll on terrain <~.2
                # If balls fall through terrain velocity is much higher ~82.0
                max_velocity_z = torch.max(torch.abs(ball_view.get_linear_velocities()[:, 2]))
                self.assertLessEqual(max_velocity_z.item(), 0.5)

    def test_ball_drop_geom_sphere(self) -> None:
        """Generates assorted terrains and geom sepheres. Tests that spheres fall onto terrain and do not pass through it"""
        for device in ("cuda:0", "cpu"):
            with build_simulation_context(device=device, auto_add_lighting=True) as sim:
                self._populate_scene(geom_sphere=False, sim=sim)
                ball_view = RigidPrimView("/World/envs/env_.*/ball", reset_xform_properties=False)
                sim.reset()

                # Initialize the ball views for physics simulation
                ball_view.initialize()

                # Play simulator
                for _ in range(500):
                    sim.step(render=False)

                # Ball may have some small non-zero velocity if the roll on terrain <~.2
                # If balls fall through terrain velocity is much higher ~82.0
                max_velocity_z = torch.max(torch.abs(ball_view.get_linear_velocities()[:, 2]))
                self.assertLessEqual(max_velocity_z.item(), 0.5)

    """
    Helper functions.
    """

    @staticmethod
    def _obtain_grid_cloner_env_origins(num_envs: int, env_spacing: float, device: str) -> torch.Tensor:
        """Obtain the env origins generated by IsaacSim GridCloner (grid_cloner.py)."""
        # create grid cloner
        cloner = GridCloner(spacing=env_spacing)
        cloner.define_base_env("/World/envs")
        envs_prim_paths = cloner.generate_paths("/World/envs/env", num_paths=num_envs)
        prim_utils.define_prim("/World/envs/env_0")
        # clone envs using grid cloner
        env_origins = cloner.clone(
            source_prim_path="/World/envs/env_0", prim_paths=envs_prim_paths, replicate_physics=True
        )
        # return as tensor
        return torch.tensor(env_origins, dtype=torch.float32, device=device)

    def _populate_scene(self, sim: SimulationContext, num_balls: int = 2048, geom_sphere: bool = False):
        """Create a scene with terrain and randomly spawned balls.

        The spawned balls are either USD Geom Spheres or are USD Meshes. We check against both these to make sure
        both USD-shape and USD-mesh collisions work as expected.
        """
        # Handler for terrains importing
        terrain_importer_cfg = terrain_gen.TerrainImporterCfg(
            prim_path="/World/ground",
            max_init_terrain_level=None,
            terrain_type="generator",
            terrain_generator=ROUGH_TERRAINS_CFG.replace(curriculum=True),
            num_envs=num_balls,
        )
        terrain_importer = TerrainImporter(terrain_importer_cfg)

        # Create interface to clone the scene
        cloner = GridCloner(spacing=2.0)
        cloner.define_base_env("/World/envs")
        # Everything under the namespace "/World/envs/env_0" will be cloned
        prim_utils.define_prim(prim_path="/World/envs/env_0", prim_type="Xform")

        # Define the scene
        # -- Ball
        if geom_sphere:
            # -- Ball physics
            _ = DynamicSphere(
                prim_path="/World/envs/env_0/ball", translation=np.array([0.0, 0.0, 5.0]), mass=0.5, radius=0.25
            )
        else:
            # -- Ball geometry
            cube_prim_path = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Sphere")[1]
            prim_utils.move_prim(cube_prim_path, "/World/envs/env_0/ball")
            # -- Ball physics
            RigidPrim(prim_path="/World/envs/env_0/ball", mass=0.5, scale=(0.5, 0.5, 0.5), translation=(0.0, 0.0, 0.5))
            GeometryPrim(prim_path="/World/envs/env_0/ball", collision=True)

        # -- Ball material
        sphere_geom = GeometryPrim(prim_path="/World/envs/env_0/ball", collision=True)
        visual_material = PreviewSurface(prim_path="/World/Looks/ballColorMaterial", color=np.asarray([0.0, 0.0, 1.0]))
        physics_material = PhysicsMaterial(
            prim_path="/World/Looks/ballPhysicsMaterial",
            dynamic_friction=1.0,
            static_friction=0.2,
            restitution=0.0,
        )
        sphere_geom.set_collision_approximation("convexHull")
        sphere_geom.apply_visual_material(visual_material)
        sphere_geom.apply_physics_material(physics_material)

        # Clone the scene
        cloner.define_base_env("/World/envs")
        envs_prim_paths = cloner.generate_paths("/World/envs/env", num_paths=num_balls)
        cloner.clone(source_prim_path="/World/envs/env_0", prim_paths=envs_prim_paths, replicate_physics=True)
        physics_scene_path = sim.get_physics_context().prim_path
        cloner.filter_collisions(
            physics_scene_path, "/World/collisions", prim_paths=envs_prim_paths, global_paths=["/World/ground"]
        )

        # Set ball positions over terrain origins
        # Create a view over all the balls
        ball_view = RigidPrimView("/World/envs/env_.*/ball", reset_xform_properties=False)
        # cache initial state of the balls
        ball_initial_positions = terrain_importer.env_origins
        ball_initial_positions[:, 2] += 5.0
        # set initial poses
        # note: setting here writes to USD :)
        ball_view.set_world_poses(positions=ball_initial_positions)


if __name__ == "__main__":
    run_tests()
