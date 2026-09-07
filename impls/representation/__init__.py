"""Input representations used by computation bodies."""

from .interfaces import StructuredRepresentation
from .manipulation import (
    CubeTokenAdapter,
    SceneTokenAdapter,
    parse_cube_observation,
    parse_scene_observation,
)
from .puzzle import PuzzleTokenAdapter, parse_puzzle_observation

__all__ = (
    'StructuredRepresentation',
    'PuzzleTokenAdapter',
    'CubeTokenAdapter',
    'SceneTokenAdapter',
    'parse_puzzle_observation',
    'parse_cube_observation',
    'parse_scene_observation',
)
