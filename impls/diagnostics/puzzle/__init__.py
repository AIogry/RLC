"""Lazy observation-only Puzzle diagnostic API.

Lazy exports keep the ground-truth Puzzle algebra module out of M25's
training import graph while preserving the existing package-level API.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    'DStarError': ('.algebra', 'DStarError'),
    'GF2SolveResult': ('.algebra', 'GF2SolveResult'),
    'build_toggle_matrix': ('.algebra', 'build_toggle_matrix'),
    'compute_dstar': ('.algebra', 'compute_dstar'),
    'enumerate_affine_solutions': ('.algebra', 'enumerate_affine_solutions'),
    'gf2_nullspace': ('.algebra', 'gf2_nullspace'),
    'gf2_rank': ('.algebra', 'gf2_rank'),
    'gf2_solve': ('.algebra', 'gf2_solve'),
    'minimum_weight_solution': ('.algebra', 'minimum_weight_solution'),
    'PressEvent': ('.events', 'PressEvent'),
    'PuzzleEventError': ('.events', 'PuzzleEventError'),
    'identify_press_events': ('.events', 'identify_press_events'),
    'BUTTON_FEATURE_DIM': ('.layout', 'BUTTON_FEATURE_DIM'),
    'BUTTON_STATE_DIM': ('.layout', 'BUTTON_STATE_DIM'),
    'POSITION_SCALE': ('.layout', 'POSITION_SCALE'),
    'PRESS_THRESHOLD_RAW': ('.layout', 'PRESS_THRESHOLD_RAW'),
    'ROBOT_DIM': ('.layout', 'ROBOT_DIM'),
    'PuzzleLayoutError': ('.layout', 'PuzzleLayoutError'),
    'decode_board_bits': ('.layout', 'decode_board_bits'),
    'extract_button_joint_positions': ('.layout', 'extract_button_joint_positions'),
    'extract_button_joint_velocities': ('.layout', 'extract_button_joint_velocities'),
    'PuzzleMetricError': ('.metrics', 'PuzzleMetricError'),
    'analyze_puzzle_episode': ('.metrics', 'analyze_puzzle_episode'),
    'CONTROLLED_GOAL_REPLAY_SCHEMA_VERSION': (
        '.replay', 'CONTROLLED_GOAL_REPLAY_SCHEMA_VERSION'
    ),
    'ControlledGoalRecord': ('.replay', 'ControlledGoalRecord'),
    'ControlledGoalReplay': ('.replay', 'ControlledGoalReplay'),
    'ControlledGoalReplayEnv': ('.replay', 'ControlledGoalReplayEnv'),
    'ControlledGoalReplayError': ('.replay', 'ControlledGoalReplayError'),
    'array_fingerprint': ('.replay', 'array_fingerprint'),
    'paired_episode_id': ('.replay', 'paired_episode_id'),
    'verify_paired_fingerprint_groups': ('.replay', 'verify_paired_fingerprint_groups'),
}


def __getattr__(name):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = tuple(_EXPORTS)
