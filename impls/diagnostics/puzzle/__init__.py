"""Reusable, observation-only diagnostics for canonical OGBench Puzzle states."""

from .algebra import (
    DStarError,
    GF2SolveResult,
    build_toggle_matrix,
    compute_dstar,
    enumerate_affine_solutions,
    gf2_nullspace,
    gf2_rank,
    gf2_solve,
    minimum_weight_solution,
)
from .events import PressEvent, PuzzleEventError, identify_press_events
from .layout import (
    BUTTON_FEATURE_DIM,
    BUTTON_STATE_DIM,
    POSITION_SCALE,
    PRESS_THRESHOLD_RAW,
    ROBOT_DIM,
    PuzzleLayoutError,
    decode_board_bits,
    extract_button_joint_positions,
    extract_button_joint_velocities,
)
from .metrics import PuzzleMetricError, analyze_puzzle_episode

__all__ = [
    'BUTTON_FEATURE_DIM',
    'BUTTON_STATE_DIM',
    'DStarError',
    'GF2SolveResult',
    'POSITION_SCALE',
    'PRESS_THRESHOLD_RAW',
    'PressEvent',
    'PuzzleEventError',
    'PuzzleLayoutError',
    'PuzzleMetricError',
    'ROBOT_DIM',
    'analyze_puzzle_episode',
    'build_toggle_matrix',
    'compute_dstar',
    'decode_board_bits',
    'enumerate_affine_solutions',
    'extract_button_joint_positions',
    'extract_button_joint_velocities',
    'gf2_nullspace',
    'gf2_rank',
    'gf2_solve',
    'identify_press_events',
    'minimum_weight_solution',
]
