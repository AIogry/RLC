"""Input representations used by computation bodies."""

from .interfaces import StructuredRepresentation
from .manipulation import (
    CubeTokenAdapter,
    SceneTokenAdapter,
    parse_cube_observation,
    parse_scene_observation,
)
from .puzzle import PuzzleTokenAdapter, parse_puzzle_observation
from .puzzle_effects import (
    detect_board_change_events,
    extract_binary_task_board,
    pack_raw_effect_signature,
    raw_effect_signature,
    unpack_raw_effect_signature,
)

__all__ = (
    'StructuredRepresentation',
    'PuzzleTokenAdapter',
    'CubeTokenAdapter',
    'SceneTokenAdapter',
    'parse_puzzle_observation',
    'parse_cube_observation',
    'parse_scene_observation',
    'detect_board_change_events',
    'extract_binary_task_board',
    'pack_raw_effect_signature',
    'raw_effect_signature',
    'unpack_raw_effect_signature',
)
