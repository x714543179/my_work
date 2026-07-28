from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from isaacgym import terrain_utils

from legged_gym.terrains.generators.mgdp import _height_units, _length_units
from legged_gym.terrains.generator import TerrainGenerator
from legged_gym.terrains.terrain_data import SubTerrainResult


SLOPE = 0
DISCRETE_STONES = 1
STAIRS = 2
GAP = 3
HIGH_PLATFORM = 4


class SFTIMTerrainGenerator(TerrainGenerator):
    """Five-terrain curriculum used by SF-TIM.

    The default ranges follow the Lite3 row of Table IV in the paper.  They can
    be overridden from config for larger robots.
    """

    def _build_sub_terrains(self):
        sub_terrains = super()._build_sub_terrains()
        if sub_terrains:
            return sub_terrains
        module = "legged_gym.terrains.generators.sf_tim"
        return [
            ("slope", {"class_name": f"{module}:SFTIMSlopeTerrain"}, 1.0),
            ("discrete_stones", {"class_name": f"{module}:SFTIMDiscreteStonesTerrain"}, 1.0),
            ("stairs", {"class_name": f"{module}:SFTIMStairsTerrain"}, 1.0),
            ("gap", {"class_name": f"{module}:SFTIMMGDPGapTerrain"}, 1.0),
            ("high_platform", {"class_name": f"{module}:SFTIMMGDPHighPlatformTerrain"}, 1.0),
        ]


@dataclass
class SFTIMSlopeTerrain:
    height_difference_range: tuple[float, float] = (0.0, 0.45)
    terrain_type: int = SLOPE

    def generate(self, terrain, difficulty: float, row: int, col: int, rng) -> SubTerrainResult:
        height_difference = self.height_difference_range[0] + (
            self.height_difference_range[1] - self.height_difference_range[0]
        ) * difficulty
        if rng.random() < 0.5:
            height_difference *= -1.0
        x = np.linspace(0.0, height_difference, terrain.width, dtype=np.float32)
        terrain.height_field_raw[:] = (x[:, None] / terrain.vertical_scale).astype(np.int16)
        return SubTerrainResult(terrain, self.terrain_type)


@dataclass
class SFTIMDiscreteStonesTerrain:
    height_range: tuple[float, float] = (0.05, 0.275)
    rectangle_min_size: float = 0.35
    rectangle_max_size: float = 1.0
    num_rectangles: int = 28
    platform_size: float = 2.0
    terrain_type: int = DISCRETE_STONES

    def generate(self, terrain, difficulty: float, row: int, col: int, rng) -> SubTerrainResult:
        height = self.height_range[0] + (self.height_range[1] - self.height_range[0]) * difficulty
        terrain_utils.discrete_obstacles_terrain(
            terrain,
            height,
            self.rectangle_min_size,
            self.rectangle_max_size,
            self.num_rectangles,
            platform_size=self.platform_size,
        )
        return SubTerrainResult(terrain, self.terrain_type)


@dataclass
class SFTIMStairsTerrain:
    step_height_range: tuple[float, float] = (0.05, 0.167)
    step_width: float = 0.35
    platform_size: float = 2.0
    terrain_type: int = STAIRS

    def generate(self, terrain, difficulty: float, row: int, col: int, rng) -> SubTerrainResult:
        step_height = self.step_height_range[0] + (self.step_height_range[1] - self.step_height_range[0]) * difficulty
        if rng.random() < 0.5:
            step_height *= -1.0
        terrain_utils.pyramid_stairs_terrain(
            terrain,
            step_width=self.step_width,
            step_height=step_height,
            platform_size=self.platform_size,
        )
        return SubTerrainResult(terrain, self.terrain_type)


@dataclass
class SFTIMGapTerrain:
    gap_width_range: tuple[float, float] = (0.2, 0.515)
    gap_start: float = 1.0
    terrain_type: int = GAP

    def generate(self, terrain, difficulty: float, row: int, col: int, rng) -> SubTerrainResult:
        del rng
        gap_width = self.gap_width_range[0] + (self.gap_width_range[1] - self.gap_width_range[0]) * difficulty
        center_x = terrain.width // 2
        start = center_x + int(self.gap_start / terrain.horizontal_scale)
        end = start + max(1, int(gap_width / terrain.horizontal_scale))
        start = min(max(start, 0), terrain.width)
        end = min(max(end, start), terrain.width)
        terrain.height_field_raw[start:end, :] = -1000
        return SubTerrainResult(terrain, self.terrain_type)


@dataclass
class SFTIMHighPlatformTerrain:
    platform_height_range: tuple[float, float] = (0.1, 0.55)
    platform_start: float = 1.25
    terrain_type: int = HIGH_PLATFORM

    def generate(self, terrain, difficulty: float, row: int, col: int, rng) -> SubTerrainResult:
        del rng
        platform_height = self.platform_height_range[0] + (
            self.platform_height_range[1] - self.platform_height_range[0]
        ) * difficulty
        center_x = terrain.width // 2
        start = center_x + int(self.platform_start / terrain.horizontal_scale)
        start = min(max(start, 0), terrain.width)
        terrain.height_field_raw[start:, :] = int(platform_height / terrain.vertical_scale)
        return SubTerrainResult(terrain, self.terrain_type)


@dataclass
class SFTIMMGDPGapTerrain:
    gap_width_range: tuple[float, float] = (0.35, 0.95)
    gap_start: float = 0.9
    depth: float = 0.6
    terrain_type: int = GAP

    def generate(self, terrain, difficulty: float, row: int, col: int, rng) -> SubTerrainResult:
        del rng
        alpha = np.clip(difficulty, 0.0, 1.0)
        gap_width = self.gap_width_range[0] + (self.gap_width_range[1] - self.gap_width_range[0]) * alpha
        center_x = terrain.width // 2
        start = center_x + _length_units(self.gap_start, terrain)
        end = start + max(1, _length_units(gap_width, terrain))
        start = min(max(start, 0), terrain.width)
        end = min(max(end, start + 1), terrain.width)
        terrain.height_field_raw[start:end, :] = -_height_units(self.depth, terrain)
        return SubTerrainResult(terrain, self.terrain_type)


@dataclass
class SFTIMMGDPHighPlatformTerrain:
    platform_height_range: tuple[float, float] = (0.25, 0.70)
    platform_start: float = 1.0
    terrain_type: int = HIGH_PLATFORM

    def generate(self, terrain, difficulty: float, row: int, col: int, rng) -> SubTerrainResult:
        del rng
        alpha = np.clip(difficulty, 0.0, 1.0)
        platform_height = self.platform_height_range[0] + (
            self.platform_height_range[1] - self.platform_height_range[0]
        ) * alpha
        center_x = terrain.width // 2
        start = center_x + _length_units(self.platform_start, terrain)
        start = min(max(start, 0), terrain.width)
        terrain.height_field_raw[start:, :] = _height_units(platform_height, terrain)
        return SubTerrainResult(terrain, self.terrain_type)
