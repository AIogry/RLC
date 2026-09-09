"""Observation-only physical button press event extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .layout import PRESS_THRESHOLD_RAW, decode_board_bits, extract_button_joint_positions


class PuzzleEventError(ValueError):
    """Raised when a transition cannot be interpreted as canonical Puzzle data."""


@dataclass(frozen=True)
class PressEvent:
    """One transition-level press event, including zero-source transitions."""

    source_mask: np.ndarray
    source_indices: tuple[int, ...]
    num_sources: int
    has_press: bool
    is_single_press: bool
    is_multi_press: bool
    event_type: str
    board_before: np.ndarray
    board_after: np.ndarray
    board_delta: np.ndarray


def identify_press_events(observation, next_observation, *, rows, cols):
    """Identify threshold-crossing sources for one canonical transition.

    The source mask is *only* computed from the authoritative
    ``prev_joint_pos > -0.02 and cur_joint_pos <= -0.02`` rule from
    ``ogbench/manipspace/envs/puzzle_env.py:PuzzleEnv.post_step``.  Board
    differences are returned for validation and are never used to infer a
    source.
    """

    observation = np.asarray(observation)
    next_observation = np.asarray(next_observation)
    if observation.ndim != 1 or next_observation.ndim != 1:
        raise PuzzleEventError(
            'identify_press_events expects one unbatched transition; '
            'iterate over leading batch dimensions before calling it'
        )
    board_before = decode_board_bits(observation, rows=rows, cols=cols)
    board_after = decode_board_bits(next_observation, rows=rows, cols=cols)
    previous_position = extract_button_joint_positions(observation, rows=rows, cols=cols)
    current_position = extract_button_joint_positions(next_observation, rows=rows, cols=cols)
    if previous_position.shape != current_position.shape:
        raise PuzzleEventError('Transition observations have incompatible button shapes')

    source_mask = np.asarray(
        (previous_position > PRESS_THRESHOLD_RAW)
        & (current_position <= PRESS_THRESHOLD_RAW),
        dtype=np.uint8,
    )
    source_indices = tuple(int(index) for index in np.flatnonzero(source_mask))
    num_sources = len(source_indices)
    board_delta = np.bitwise_xor(board_before, board_after).astype(np.uint8)
    return PressEvent(
        source_mask=source_mask,
        source_indices=source_indices,
        num_sources=num_sources,
        has_press=num_sources > 0,
        is_single_press=num_sources == 1,
        is_multi_press=num_sources > 1,
        event_type=(
            'single_press' if num_sources == 1
            else 'multi_press' if num_sources > 1
            else 'no_press'
        ),
        board_before=np.asarray(board_before, dtype=np.uint8),
        board_after=np.asarray(board_after, dtype=np.uint8),
        board_delta=board_delta,
    )


__all__ = ['PressEvent', 'PuzzleEventError', 'identify_press_events']
