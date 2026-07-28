from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from isaacgym import terrain_utils

from legged_gym.terrains.terrain_data import SubTerrainResult


def _height_units(value, terrain):
    return int(value / terrain.vertical_scale)


def _length_units(value, terrain):
    return int(value / terrain.horizontal_scale)


def _height_range_units(max_height, terrain, step=4, start=1, clip=None):
    max_height = _height_units(max_height, terrain)
    if clip is not None:
        max_height = int(np.clip(max_height, clip[0], clip[1]))
    values = np.arange(start, max_height, step=max(1, step), dtype=np.int16)
    if values.size == 0:
        values = np.array([max(start, max_height)], dtype=np.int16)
    return values


def _choice(rng, values):
    values = np.asarray(values)
    if values.size == 0:
        return 0
    return int(rng.choice(values))


def _add_roughness(terrain, rng, enabled=True, height_range=(0.01, 0.04), downsampled_scale=0.5):
    if not enabled:
        return
    max_height = float(height_range[1])
    min_height = float(height_range[0])
    height = rng.uniform(min_height, max_height)
    terrain_utils.random_uniform_terrain(
        terrain,
        min_height=-height,
        max_height=height,
        step=0.005,
        downsampled_scale=downsampled_scale,
    )


def _platform_y(terrain, platform_size):
    return (terrain.length - platform_size) // 2


def _fill_initial_platform(terrain, platform_size):
    y = _platform_y(terrain, platform_size)
    terrain.height_field_raw[0:platform_size, y : y + platform_size] = 0


@dataclass
class ParkourStepTerrain:
    platform_size: float = 2.0
    num_stones: int = 8
    depth: float = 0.6
    add_roughness: bool = True
    roughness_height_range: tuple[float, float] = (0.01, 0.04)
    downsampled_scale: float = 0.5
    terrain_type: int = 5

    def generate(self, terrain, difficulty: float, row: int, col: int, rng) -> SubTerrainResult:
        step_height = 0.05 + 0.18 * difficulty
        if self.terrain_type == 4:
            stone_size = 0.7 if difficulty < 0.2 else -0.5 * difficulty * difficulty + 0.7
            stone_distance = 0.05 if difficulty < 0.2 else 0.4 * int(10 * difficulty) / 10
            self._stepping_stones(terrain, rng, stone_size, stone_distance, step_height)
        elif self.terrain_type == 7 or self.num_stones == 1:
            stone_size = 0.8 if difficulty < 0.2 else -0.5 * difficulty + 0.8
            stone_distance = 0.1 if difficulty < 0.2 else 0.4 * int(10 * difficulty) / 10
            self._stepping_one_stones(terrain, rng, difficulty, stone_size, stone_distance, step_height)
        elif self.terrain_type == 6:
            stone_size = 0.8 if difficulty < 0.2 else -0.5 * difficulty + 0.8
            stone_distance = 0.0 if difficulty < 0.2 else 0.2 * int(10 * difficulty) / 10
            self._stepping_two_discrete_stones(terrain, rng, difficulty, stone_size, stone_distance, step_height)
        else:
            stone_size = 0.8 if difficulty < 0.2 else -0.5 * difficulty + 0.8
            stone_distance = 0.1 if difficulty < 0.2 else 0.4 * int(10 * difficulty) / 10
            self._stepping_two_stones(terrain, rng, stone_size, stone_distance, step_height)
        _add_roughness(terrain, rng, self.add_roughness, self.roughness_height_range, self.downsampled_scale)
        return SubTerrainResult(terrain, self.terrain_type)

    def _stepping_stones(self, terrain, rng, stone_size, stone_distance, max_height):
        stone_size = _length_units(stone_size, terrain)
        stone_distance_x = _length_units(stone_distance, terrain)
        stone_distance_y = min(_length_units(stone_distance, terrain), 1)
        height_range = _height_range_units(max_height, terrain, step=4, start=1, clip=(1, 40))
        platform_size = _length_units(self.platform_size, terrain)
        terrain.height_field_raw[:, :] = -_height_units(self.depth, terrain)

        start_x = 0
        if terrain.width > terrain.length:
            while start_x < terrain.width:
                stop_x = min(terrain.width, start_x + stone_size)
                start_y = int(rng.integers(0, max(1, stone_size)))
                stop_y = max(0, start_y - stone_distance_y)
                terrain.height_field_raw[start_x:stop_x, 0:stop_y] = _choice(rng, height_range)
                while start_y < terrain.length - 10:
                    stop_y = min(terrain.length - 10, start_y + stone_size)
                    terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = _choice(rng, height_range)
                    start_y += stone_size + stone_distance_y
                start_x += stone_size + stone_distance_x
        _fill_initial_platform(terrain, platform_size)

    def _stepping_two_stones(self, terrain, rng, stone_size, stone_distance, max_height):
        stone_size = np.clip(_length_units(stone_size, terrain), 6, 16)
        height_range = _height_range_units(max_height, terrain, step=3, start=1, clip=(0, 30))
        platform_size = _length_units(self.platform_size, terrain)
        platform_y = _platform_y(terrain, platform_size)
        row1_y = int(platform_y + platform_size / 2 - stone_size - 0.1)
        row2_y = int(platform_y + platform_size / 2 + 0.1)
        stone_distance_gap = np.clip(_length_units(stone_distance, terrain), 1, 6)
        stone_distance_range = np.arange(0, stone_distance_gap, step=1) + 1

        terrain.height_field_raw[:, :] = -_height_units(self.depth, terrain)
        start_x = 0
        while start_x < terrain.width:
            stop_x = min(terrain.width, start_x + stone_size)
            terrain.height_field_raw[start_x:stop_x, row1_y : row1_y + stone_size] = _choice(rng, height_range)
            terrain.height_field_raw[start_x:stop_x, row2_y : row2_y + stone_size] = _choice(rng, height_range)
            start_x += stone_size + _choice(rng, stone_distance_range)
        _fill_initial_platform(terrain, platform_size)

    def _stepping_two_discrete_stones(self, terrain, rng, difficulty, stone_size, stone_distance, max_height):
        stone_size = np.clip(_length_units(stone_size, terrain), 6, 16)
        height_range = _height_range_units(max_height, terrain, step=3, start=0, clip=(0, 30))
        platform_size = _length_units(self.platform_size, terrain)
        platform_y = _platform_y(terrain, platform_size)
        row1_y = int(platform_y + platform_size / 2 - stone_size - 0.01)
        row2_y = int(platform_y + platform_size / 2 + 0.01)
        random_gap = int(rng.integers(1, 3 if difficulty < 0.4 else 4))
        stone_distance = int(stone_size / 2) + random_gap

        terrain.height_field_raw[:, :] = -_height_units(self.depth, terrain)
        start_x = 0
        is_left = bool(rng.choice([True, False]))
        while start_x < terrain.width:
            stop_x = min(terrain.width, start_x + stone_size)
            height = _choice(rng, height_range)
            if is_left:
                terrain.height_field_raw[start_x:stop_x, row1_y : row1_y + stone_size] = height
            else:
                terrain.height_field_raw[start_x:stop_x, row2_y : row2_y + stone_size] = height
            start_x += stone_distance
            is_left = not is_left
        _fill_initial_platform(terrain, platform_size)

    def _stepping_one_stones(self, terrain, rng, difficulty, stone_size, stone_distance, max_height):
        stone_size = _length_units(stone_size, terrain)
        max_height_units = int(np.clip(_height_units(max_height, terrain), 0, 20))
        if difficulty < 0.3:
            height_range = np.arange(1, max_height_units, step=3, dtype=np.int16)
        elif difficulty < 0.7:
            height_range = np.arange(1, max_height_units, step=4, dtype=np.int16)
        else:
            height_range = np.arange(1, max_height_units, step=12, dtype=np.int16)
        if height_range.size == 0:
            height_range = np.array([max(1, max_height_units)], dtype=np.int16)

        platform_size = _length_units(self.platform_size, terrain)
        platform_y = _platform_y(terrain, platform_size)
        stone_distance_gap = np.clip(_length_units(difficulty, terrain), 4, 16)
        terrain.height_field_raw[:, :] = -_height_units(self.depth, terrain)

        start_x = 0
        while start_x < terrain.width:
            stone_size_x = int(np.clip(stone_size, 12, 30))
            stone_size_y = int(rng.choice(np.arange(12, 30, step=1)))
            row_y = int(platform_y + platform_size / 2 - stone_size_y / 2)
            stop_x = min(terrain.width, start_x + stone_size_x)
            terrain.height_field_raw[start_x:stop_x, row_y : row_y + stone_size_y] = _choice(rng, height_range)
            start_x += stone_size + stone_distance_gap
        _fill_initial_platform(terrain, platform_size)


@dataclass
class ParkourGapTerrain:
    depth: float = 0.6
    platform_size: float = 2.0
    add_roughness: bool = True
    roughness_height_range: tuple[float, float] = (0.01, 0.04)
    downsampled_scale: float = 0.5
    terrain_type: int = 6

    def generate(self, terrain, difficulty: float, row: int, col: int, rng) -> SubTerrainResult:
        gap_size = np.clip(_length_units(difficulty, terrain), 2, 12)
        depth = _height_units(self.depth, terrain)
        platform_size = _length_units(self.platform_size, terrain)
        start_y = 0
        end_y = int(terrain.length - platform_size / 8)
        start_x = platform_size
        center_x = terrain.width // 2
        terrain.height_field_raw[start_x:center_x, start_y:end_y] = -depth
        terrain.height_field_raw[start_x + gap_size : center_x - gap_size, start_y + gap_size : end_y - gap_size] = 0

        start_x = center_x + int(platform_size / 2)
        end_x = terrain.width
        terrain.height_field_raw[start_x:end_x, start_y:end_y] = -depth
        terrain.height_field_raw[start_x + gap_size : end_x - gap_size, start_y + gap_size : end_y - gap_size] = 0
        _add_roughness(terrain, rng, self.add_roughness, self.roughness_height_range, self.downsampled_scale)
        return SubTerrainResult(terrain, self.terrain_type)


@dataclass
class RampTerrain:
    platform_size: float = 2.0
    depth: float = 0.6
    final_platform_length: int = 5
    add_roughness: bool = True
    roughness_height_range: tuple[float, float] = (0.01, 0.04)
    downsampled_scale: float = 0.5
    terrain_type: int = 7

    def generate(self, terrain, difficulty: float, row: int, col: int, rng) -> SubTerrainResult:
        platform_size = _length_units(self.platform_size, terrain)
        slope_strength = 2.0 + 2.5 * difficulty * 2.5
        terrain.height_field_raw[:, :] = -_height_units(self.depth, terrain)
        start_y, end_y = 20, terrain.length - 20
        terrain.height_field_raw[:, start_y:end_y] = 0

        up_start, up_end = platform_size, 2 * platform_size
        mid_start, mid_end = 2 * platform_size, 3 * platform_size
        down_start = 3 * platform_size
        down_end = min(down_start + (up_end - up_start), 4 * platform_size)
        final_end = min(down_end + self.final_platform_length, terrain.width)

        xs = np.arange(up_start, up_end)
        max_height = slope_strength * max(1, up_end - up_start)
        up_heights = (slope_strength * (xs - up_start)).astype(np.int16)
        terrain.height_field_raw[up_start:up_end, start_y:end_y] = up_heights[:, None]
        terrain.height_field_raw[mid_start:mid_end, start_y:end_y] = int(max_height)
        xs = np.arange(down_start, down_end)
        down_heights = (max_height - slope_strength * (xs - down_start)).astype(np.int16)
        terrain.height_field_raw[down_start:down_end, start_y:end_y] = down_heights[:, None]
        terrain.height_field_raw[down_end:final_end, start_y:end_y] = 0
        _add_roughness(terrain, rng, self.add_roughness, self.roughness_height_range, self.downsampled_scale)
        return SubTerrainResult(terrain, self.terrain_type)


@dataclass
class BeamTerrain:
    stone_size_range: tuple[float, float] = (0.35, 0.25)
    platform_size: float = 2.0
    depth: float = 0.6
    add_roughness: bool = True
    roughness_height_range: tuple[float, float] = (0.01, 0.04)
    downsampled_scale: float = 0.5
    terrain_type: int = 8

    def generate(self, terrain, difficulty: float, row: int, col: int, rng) -> SubTerrainResult:
        platform_size = _length_units(self.platform_size, terrain)
        terrain.height_field_raw[:, :] = -_height_units(self.depth, terrain)
        _fill_initial_platform(terrain, platform_size)

        stone_size = 0.35 if difficulty < 0.2 else -0.1 * difficulty + 0.35
        stone_distance = 0.1 if difficulty < 0.2 else 0.4 * int(10 * difficulty) / 10
        beam_length = max(1, _length_units(stone_size, terrain))
        beam_distance = max(1, _length_units(stone_distance, terrain))
        step_height = 0.04 if difficulty < 0.2 else 0.05 + 0.18 * difficulty
        max_height = int(step_height / terrain.vertical_scale) / 100
        height_values = np.arange(0.0, max_height, step=0.04, dtype=np.float32)
        if height_values.size == 0:
            height_values = np.array([0.0], dtype=np.float32)

        platform_y = _platform_y(terrain, platform_size)
        start_x = platform_size
        while start_x < terrain.width:
            beam_width = int(2 * rng.integers(7, 16))
            row_y = int(platform_y + platform_size / 2 - beam_width / 2)
            stop_x = min(terrain.width, start_x + beam_length)
            height = _height_units(float(rng.choice(height_values)), terrain)
            terrain.height_field_raw[start_x:stop_x, row_y : row_y + beam_width] = height
            start_x += beam_length + beam_distance
        _add_roughness(terrain, rng, self.add_roughness, self.roughness_height_range, self.downsampled_scale)
        return SubTerrainResult(terrain, self.terrain_type)


@dataclass
class HurdleTerrain:
    platform_size: float = 2.0
    depth: float = 0.6
    hurdle_height_range: tuple[float, float] = (0.1, 0.5)
    terrain_type: int = 15

    def generate(self, terrain, difficulty: float, row: int, col: int, rng) -> SubTerrainResult:
        platform_size = _length_units(self.platform_size, terrain)
        platform_y = (terrain.length - platform_size) // 2
        terrain.height_field_raw[:, :] = -_height_units(self.depth, terrain)
        row_y = int(platform_y + platform_size / 2 - 17 / 2)
        terrain.height_field_raw[:, row_y : row_y + 18] = 0
        h0, h1 = self.hurdle_height_range
        height_min = _height_units(h0, terrain)
        height_max = max(height_min + 1, _height_units(h1, terrain))
        step_height = rng.integers(height_min, height_max)
        start_x = platform_size + 20
        while start_x < terrain.width - 20:
            size_x = int(np.clip(12 + 18 * (1.0 - difficulty), 12, 30))
            stop_x = min(terrain.width, start_x + size_x)
            size_y = int(rng.integers(17, 30))
            y = int(platform_y + platform_size / 2 - size_y / 2)
            terrain.height_field_raw[start_x:stop_x, y : y + size_y] = step_height
            start_x += size_x + int(rng.integers(17, 30))
        return SubTerrainResult(terrain, self.terrain_type)


@dataclass
class AirStoneTerrain:
    depth: float = 0.6
    stone_length: float = 4.5
    min_stone_width: float = 1.8
    max_stone_width: float = 2.4
    min_height: float = 0.18
    max_height: float = 0.55
    terrain_type: int = 14

    def generate(self, terrain, difficulty: float, row: int, col: int, rng) -> SubTerrainResult:
        terrain.height_field_raw[:] = -_height_units(self.depth, terrain)
        start_y, end_y = 20, terrain.length - 20
        terrain.height_field_raw[:, start_y:end_y] = 0
        stone_length = min(_length_units(self.stone_length, terrain), terrain.width - 1)
        stone_width = _length_units(rng.uniform(self.min_stone_width, self.max_stone_width), terrain)
        stone_height = self.max_height - (self.max_height - self.min_height) * difficulty
        stone_height += rng.uniform(0.05, 0.2)
        x2 = terrain.width - 5
        x1 = max(0, x2 - stone_length)
        center_y = terrain.length // 2
        y1 = int(np.clip(center_y - stone_width // 2, 0, terrain.length - 1))
        y2 = int(np.clip(y1 + stone_width, y1 + 1, terrain.length))
        terrain.height_field_raw[x1:x2, y1:y2] = _height_units(stone_height, terrain)
        center = np.array([0.5, 0.5, stone_height], dtype=np.float32)
        return SubTerrainResult(terrain, self.terrain_type, {"goals_stone": center})


@dataclass
class NarrowCorridorTerrain:
    depth: float = 0.6
    platform_size: float = 2.0
    wall_height: float = 0.5
    terrain_type: int = 17

    def generate(self, terrain, difficulty: float, row: int, col: int, rng) -> SubTerrainResult:
        platform_size = _length_units(self.platform_size, terrain)
        terrain.height_field_raw[:] = -_height_units(self.depth, terrain)
        start_y, end_y = 20, terrain.length - 20
        terrain.height_field_raw[:, start_y:end_y] = 0
        wall_abs_height = _height_units(self.wall_height, terrain)
        center_y = terrain.length // 2
        narrow_gap = int(np.clip(8 - (8 - 2) * difficulty, 2, 8))
        terrain.height_field_raw[platform_size + 20 :, center_y + narrow_gap : end_y] = wall_abs_height
        terrain.height_field_raw[platform_size + 20 :, start_y : center_y - narrow_gap] = wall_abs_height
        center = np.array([1.0, 0.5, 0.32], dtype=np.float32)
        return SubTerrainResult(terrain, self.terrain_type, {"goals_narrow": center})
