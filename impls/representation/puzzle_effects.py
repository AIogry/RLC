"""Observation-only Puzzle board and event-effect semantics for M25.

This module deliberately has no access to Puzzle's operation matrix, inverse,
physical button labels, solver state, actions, or privileged dataset fields.
"""

from __future__ import annotations

from numbers import Integral

import jax
import numpy as np

from .puzzle import parse_puzzle_observation


def extract_binary_task_board(
    observation,
    *,
    num_buttons,
    robot_dim=19,
    button_feature_dim=4,
):
    """Decode binary task state through the canonical Puzzle parser."""

    if isinstance(num_buttons, bool) or not isinstance(num_buttons, Integral):
        raise ValueError(f'num_buttons must be an integer, got {num_buttons!r}')
    _, buttons = parse_puzzle_observation(
        observation,
        num_buttons=int(num_buttons),
        robot_dim=robot_dim,
        button_feature_dim=button_feature_dim,
    )
    if button_feature_dim < 2:
        raise ValueError(
            'Puzzle button features must include a two-class task-state onehot'
        )
    onehot = buttons[..., :2]
    if any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(onehot)):
        raise ValueError('Offline Puzzle board extraction requires concrete observations')
    onehot = np.asarray(onehot)
    if not np.all(np.isfinite(onehot)):
        raise ValueError('Puzzle task-state onehot contains non-finite values')
    if not np.all((onehot == 0) | (onehot == 1)):
        raise ValueError('Puzzle task-state fields must contain exact 0/1 values')
    if not np.all(onehot.sum(axis=-1) == 1):
        raise ValueError('Puzzle task-state fields must be valid two-class onehots')
    return np.argmax(onehot, axis=-1).astype(np.uint8)


def extract_dataset_boards(
    observations,
    *,
    num_buttons,
    robot_dim=19,
    button_feature_dim=4,
    chunk_size=65_536,
):
    """Decode a large standard observation array without staging it all in JAX."""

    observations = np.asarray(observations)
    if observations.ndim != 2:
        raise ValueError(
            f'Standard state observations must have shape [steps, features], got {observations.shape}'
        )
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral) or chunk_size <= 0:
        raise ValueError(f'chunk_size must be positive, got {chunk_size!r}')
    chunks = []
    for start in range(0, len(observations), int(chunk_size)):
        chunks.append(
            extract_binary_task_board(
                observations[start:start + int(chunk_size)],
                num_buttons=num_buttons,
                robot_dim=robot_dim,
                button_feature_dim=button_feature_dim,
            )
        )
    if not chunks:
        return np.empty((0, int(num_buttons)), dtype=np.uint8)
    return np.concatenate(chunks, axis=0)


def detect_board_change_events(boards):
    """Return a mask for exactly the adjacent transitions whose board changes."""

    boards = _binary_boards(boards, name='boards')
    if len(boards) < 2:
        return np.zeros((0,), dtype=bool)
    return np.any(boards[:-1] != boards[1:], axis=-1)


def raw_transition_xor(start_board, end_board):
    """Return the diagnostic raw XOR effect of one or more transitions."""

    start_board = _binary_boards(start_board, name='start_board')
    end_board = _binary_boards(end_board, name='end_board')
    if start_board.shape != end_board.shape:
        raise ValueError(
            f'Board endpoint shapes differ: {start_board.shape} != {end_board.shape}'
        )
    return np.bitwise_xor(start_board, end_board).astype(np.uint8)


def pack_raw_effect_signature(effect):
    """Pack one diagnostic effect vector into a dimension-tagged stable key."""

    effect = np.asarray(effect)
    if effect.ndim != 1:
        raise ValueError(f'An effect signature requires one bit vector, got {effect.shape}')
    effect = _binary_boards(effect[None], name='effect')[0]
    payload = np.packbits(effect, bitorder='little').tobytes().hex()
    return f'{effect.size}:{payload}'


def unpack_raw_effect_signature(signature):
    """Invert :func:`pack_raw_effect_signature` exactly."""

    if not isinstance(signature, str) or ':' not in signature:
        raise ValueError(f'Malformed raw effect signature: {signature!r}')
    dimension_text, payload = signature.split(':', 1)
    try:
        dimension = int(dimension_text)
        packed = bytes.fromhex(payload)
    except (TypeError, ValueError) as error:
        raise ValueError(f'Malformed raw effect signature: {signature!r}') from error
    if dimension <= 0 or len(packed) != (dimension + 7) // 8:
        raise ValueError(f'Malformed raw effect signature: {signature!r}')
    bits = np.unpackbits(
        np.frombuffer(packed, dtype=np.uint8), bitorder='little'
    )[:dimension].astype(np.uint8)
    # Reject alternate encodings with nonzero padding bits.
    if pack_raw_effect_signature(bits) != signature.lower():
        raise ValueError(f'Non-canonical raw effect signature: {signature!r}')
    return bits


def raw_effect_signature(start_board, end_board):
    """Return a packed diagnostic key for one nonzero board transition."""

    effect = raw_transition_xor(start_board, end_board)
    if effect.ndim != 1:
        raise ValueError('raw_effect_signature expects one transition')
    if not np.any(effect):
        raise ValueError('A zero-change transition has no event-effect signature')
    return pack_raw_effect_signature(effect)


def _binary_boards(value, *, name):
    array = np.asarray(value)
    if array.ndim < 1:
        raise ValueError(f'{name} requires a final bit axis, got {array.shape}')
    if not np.issubdtype(array.dtype, np.number) and not np.issubdtype(
        array.dtype, np.bool_
    ):
        raise ValueError(f'{name} must be numeric or Boolean, got {array.dtype}')
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{name} contains non-finite values')
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f'{name} must contain exact 0/1 values')
    return array.astype(np.uint8, copy=False)


__all__ = [
    'detect_board_change_events',
    'extract_binary_task_board',
    'extract_dataset_boards',
    'pack_raw_effect_signature',
    'raw_effect_signature',
    'raw_transition_xor',
    'unpack_raw_effect_signature',
]
