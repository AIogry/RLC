"""Strict layout decoding for canonical state-based OGBench Puzzle observations."""

from __future__ import annotations

from numbers import Integral

import numpy as np

from ...representation.puzzle import parse_puzzle_observation


ROBOT_DIM = 19
BUTTON_FEATURE_DIM = 4
BUTTON_STATE_DIM = 2
# Source: ogbench/manipspace/envs/puzzle_env.py:PuzzleEnv.compute_observation.
POSITION_SCALE = 120.0
# Source: ogbench/manipspace/envs/puzzle_env.py:PuzzleEnv.post_step.
PRESS_THRESHOLD_RAW = -0.02
_ONE_HOT_ATOL = 1e-6


class PuzzleLayoutError(ValueError):
    """Raised when an input is not a canonical Puzzle state observation."""


def _validate_grid(rows, cols):
    for name, value in (('rows', rows), ('cols', cols)):
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
            raise PuzzleLayoutError(f'{name} must be a positive integer, got {value!r}')
    rows, cols = int(rows), int(cols)
    return rows, cols, rows * cols


def _parse_buttons(observation, *, rows, cols):
    rows, cols, num_buttons = _validate_grid(rows, cols)
    try:
        _, buttons = parse_puzzle_observation(
            observation,
            num_buttons=num_buttons,
            robot_dim=ROBOT_DIM,
            button_feature_dim=BUTTON_FEATURE_DIM,
        )
    except (TypeError, ValueError) as error:
        raise PuzzleLayoutError(str(error)) from error
    buttons = np.asarray(buttons)
    if not np.issubdtype(buttons.dtype, np.number):
        raise PuzzleLayoutError(f'Puzzle button features must be numeric, got {buttons.dtype}')
    return buttons


def _finite(values, *, name):
    try:
        finite = np.isfinite(values)
    except TypeError as error:
        raise PuzzleLayoutError(f'{name} must be numeric') from error
    if not np.all(finite):
        raise PuzzleLayoutError(f'{name} contains non-finite values')


def decode_board_bits(observation, *, rows, cols):
    """Decode strict button one-hot blocks into ``[..., N]`` binary board bits.

    The parser is intentionally delegated to
    :func:`impls.representation.puzzle.parse_puzzle_observation`; this module
    only validates the audited button-feature semantics.
    """

    buttons = _parse_buttons(observation, rows=rows, cols=cols)
    one_hot = buttons[..., :BUTTON_STATE_DIM]
    _finite(one_hot, name='Puzzle button one-hot features')
    if not np.all(
        np.isclose(one_hot, 0.0, rtol=0.0, atol=_ONE_HOT_ATOL)
        | np.isclose(one_hot, 1.0, rtol=0.0, atol=_ONE_HOT_ATOL)
    ):
        raise PuzzleLayoutError('Puzzle button state features are not approximately binary')
    if not np.all(np.isclose(one_hot.sum(axis=-1), 1.0, rtol=0.0, atol=_ONE_HOT_ATOL)):
        raise PuzzleLayoutError('Puzzle button state features do not sum to one')
    decoded = np.argmax(one_hot, axis=-1).astype(np.int8)
    reconstructed = np.eye(BUTTON_STATE_DIM, dtype=np.float64)[decoded]
    if not np.allclose(one_hot, reconstructed, rtol=0.0, atol=_ONE_HOT_ATOL):
        raise PuzzleLayoutError('Puzzle button state features are not strict one-hot vectors')
    return decoded


def extract_button_joint_positions(observation, *, rows, cols, raw=True):
    """Extract ``[..., N]`` button joint positions.

    Canonical Puzzle stores ``raw_q * 120`` in the observation.  ``raw=True``
    returns raw joint coordinates; ``raw=False`` returns stored features.
    """

    if not isinstance(raw, (bool, np.bool_)):
        raise PuzzleLayoutError(f'raw must be boolean, got {raw!r}')
    buttons = _parse_buttons(observation, rows=rows, cols=cols)
    stored = buttons[..., :, 2]
    _finite(stored, name='Puzzle button joint positions')
    return stored / POSITION_SCALE if raw else stored.copy()


def extract_button_joint_velocities(observation, *, rows, cols):
    """Extract unscaled ``[..., N]`` button joint velocities."""

    buttons = _parse_buttons(observation, rows=rows, cols=cols)
    velocity = buttons[..., :, 3]
    _finite(velocity, name='Puzzle button joint velocities')
    return velocity.copy()


__all__ = [
    'BUTTON_FEATURE_DIM',
    'BUTTON_STATE_DIM',
    'POSITION_SCALE',
    'PRESS_THRESHOLD_RAW',
    'ROBOT_DIM',
    'PuzzleLayoutError',
    'decode_board_bits',
    'extract_button_joint_positions',
    'extract_button_joint_velocities',
]
