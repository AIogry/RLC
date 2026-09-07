"""Task-grounded, parameter-free relation builders for M20A.

The builders in this module deliberately own no learned parameters.  They
turn the *actual* ``(state, goal)`` pair supplied to a network call into a
binary relation tensor with the shared direction convention

``R[..., source, target, relation_type] == 1``.

This keeps relation construction call-specific: the actor, value, critic,
target critic, and evaluation policy each receive a relation built from the
goal passed to that particular forward call.  ``legacy_none`` remains outside
this module and means that no relation tensor/augmenter path exists at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import jax.numpy as jnp
import numpy as np


RELATION_MODES = frozenset({'zero', 'correct', 'shuffled'})
PUZZLE_SHUFFLE_DESIGN_SEED = 20020
PUZZLE_4X4_SHUFFLE_PERMUTATION = (
    2, 10, 15, 12, 11, 8, 7, 3, 6, 0, 5, 14, 9, 4, 1, 13,
)
CUBE_TRIPLE_SHUFFLE_DERANGEMENT = (1, 2, 0)


def validate_relation_mode(mode: str) -> str:
    if mode not in RELATION_MODES:
        raise ValueError(
            f'Unsupported relation mode {mode!r}; expected one of {sorted(RELATION_MODES)!r}'
        )
    return mode


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f'{name} must be a positive integer, got {value!r}')
    return int(value)


def _permutation(permutation: Sequence[int], num_entities: int, *, label: str):
    value = tuple(int(item) for item in permutation)
    if len(value) != num_entities or set(value) != set(range(num_entities)):
        raise ValueError(
            f'{label} must be a permutation of 0..{num_entities - 1}, got {value!r}'
        )
    return value


def _shuffle_endpoints(relations, permutation):
    """Return ``P R P^T`` using a fixed explicit endpoint permutation.

    Indexing both endpoint axes deliberately leaves token features untouched;
    only the relation-to-token assignment is corrupted.
    """

    relations = jnp.asarray(relations)
    permutation = jnp.asarray(permutation, dtype=jnp.int32)
    if relations.ndim == 3:
        return relations[permutation, :, :][:, permutation, :]
    if relations.ndim == 4:
        return relations[:, permutation, :, :][:, :, permutation, :]
    raise ValueError(
        'Relation tensor must be [T, T, K] or [B, T, T, K] before endpoint shuffling; '
        f'got {relations.shape}'
    )


def puzzle_toggle_relation(rows: int, cols: int, *, dtype=jnp.float32):
    """Build the exact static Lights-Out press-to-toggle adjacency.

    This mirrors ``PuzzleEnv.post_step``: a press toggles self plus every
    in-bounds up/down/left/right neighbor.  The source axis is the pressed
    button and the target axis is the toggled button.
    """

    rows = _positive_int('rows', rows)
    cols = _positive_int('cols', cols)
    num_buttons = rows * cols
    relation = np.zeros((num_buttons, num_buttons, 1), dtype=np.float32)
    for index in range(num_buttons):
        row, col = divmod(index, cols)
        for delta_row, delta_col in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            target_row = row + delta_row
            target_col = col + delta_col
            if 0 <= target_row < rows and 0 <= target_col < cols:
                target = target_row * cols + target_col
                relation[index, target, 0] = 1.0
    return jnp.asarray(relation, dtype=dtype)


def build_puzzle_relations(
    state,
    goal,
    *,
    mode: str,
    rows: int,
    cols: int,
    shuffle_permutation: Sequence[int] | None = PUZZLE_4X4_SHUFFLE_PERMUTATION,
):
    """Return static Puzzle relations with a state/goal-compatible batch axis.

    ``state`` and ``goal`` are intentionally accepted even though the Puzzle
    graph itself is static.  This gives every builder the same public
    ``R(state, goal)`` interface and keeps the call site semantically honest.
    """

    del goal
    mode = validate_relation_mode(mode)
    state = jnp.asarray(state)
    rows = _positive_int('rows', rows)
    cols = _positive_int('cols', cols)
    num_buttons = rows * cols
    if shuffle_permutation is None:
        shuffle_permutation = PUZZLE_4X4_SHUFFLE_PERMUTATION
    permutation = _permutation(
        shuffle_permutation, num_buttons, label='Puzzle shuffle permutation'
    )
    correct = puzzle_toggle_relation(rows, cols, dtype=state.dtype)
    if mode == 'zero':
        base = jnp.zeros_like(correct)
    elif mode == 'correct':
        base = correct
    else:
        base = _shuffle_endpoints(correct, permutation)
    if state.ndim == 1:
        return base
    if state.ndim < 2:
        raise ValueError(f'Puzzle state must be [D] or [B, D], got {state.shape}')
    return jnp.broadcast_to(base, (*state.shape[:-1], *base.shape))


def puzzle_shuffle_diagnostics(rows: int, cols: int, permutation=PUZZLE_4X4_SHUFFLE_PERMUTATION):
    """Return deterministic validity facts for the fixed Puzzle control."""

    correct = np.asarray(puzzle_toggle_relation(rows, cols))
    shuffled = np.asarray(_shuffle_endpoints(correct, _permutation(
        permutation, rows * cols, label='Puzzle shuffle permutation'
    )))
    differing_entries = int(np.count_nonzero(correct != shuffled))
    return {
        'rows': int(rows),
        'cols': int(cols),
        'permutation': [int(item) for item in permutation],
        'differing_adjacency_entries': differing_entries,
        'is_non_identity_control': bool(differing_entries > 0),
        'correct_edge_count': int(correct.sum()),
        'shuffled_edge_count': int(shuffled.sum()),
    }


SCENE_ENTITY_ORDER = ('cube', 'button_0', 'button_1', 'drawer', 'window')


def scene_controls_relation(*, mode: str, dtype=jnp.float32):
    """Build the source-audited Scene V0 control topology.

    SceneEnv._apply_button_states maps button_0 to drawer locking and
    button_1 to window locking.  The shuffled control swaps only the two
    target endpoints, preserving the number and types of active edges.
    """

    mode = validate_relation_mode(mode)
    correct = np.zeros((5, 5, 1), dtype=np.float32)
    correct[1, 3, 0] = 1.0  # button_0 -> drawer
    correct[2, 4, 0] = 1.0  # button_1 -> window
    if mode == 'zero':
        relation = np.zeros_like(correct)
    elif mode == 'correct':
        relation = correct
    else:
        relation = np.zeros_like(correct)
        relation[1, 4, 0] = 1.0  # button_0 -> window
        relation[2, 3, 0] = 1.0  # button_1 -> drawer
    return jnp.asarray(relation, dtype=dtype)


def build_scene_relations(state, goal, *, mode: str):
    """Return Scene V0 relations with a batch axis matching ``state``."""

    del goal
    state = jnp.asarray(state)
    base = scene_controls_relation(mode=mode, dtype=state.dtype)
    if state.ndim == 1:
        return base
    if state.ndim < 2:
        raise ValueError(f'Scene state must be [D] or [B, D], got {state.shape}')
    return jnp.broadcast_to(base, (*state.shape[:-1], *base.shape))


def _cube_positions(cubes, *, xyz_scaler: float):
    cubes = jnp.asarray(cubes)
    if cubes.ndim not in (2, 3) or cubes.shape[-1] < 3:
        raise ValueError(
            'Cube entity features must be [T, F] or [B, T, F] with F >= 3; '
            f'got {cubes.shape}'
        )
    if xyz_scaler <= 0:
        raise ValueError(f'xyz_scaler must be positive, got {xyz_scaler!r}')
    # OGBench stores canonical cube XYZ as (xyz - center) * 10.  Center
    # cancels in all pairwise comparisons, so divide only by the scaler.
    return cubes[..., :3] / float(xyz_scaler)


def cube_correct_relations(
    state_cubes,
    goal_cubes,
    *,
    current_support_epsilon_xy: float,
    current_support_epsilon_z: float,
    goal_support_epsilon_xy: float,
    goal_support_epsilon_z: float,
    conflict_radius: float,
    cube_height: float = 0.04,
    xyz_scaler: float = 10.0,
):
    """Build dynamic Cube V0 channels from the received state/goal tensors.

    Channels are ordered as ``current_support``, ``goal_support``, and
    ``goal_conflict``.  A support edge points from the lower cube to the
    directly supported upper cube.  A conflict edge points from a current
    occupant to the distinct cube whose goal-success region it occupies.
    """

    for name, value in (
        ('current_support_epsilon_xy', current_support_epsilon_xy),
        ('current_support_epsilon_z', current_support_epsilon_z),
        ('goal_support_epsilon_xy', goal_support_epsilon_xy),
        ('goal_support_epsilon_z', goal_support_epsilon_z),
        ('conflict_radius', conflict_radius),
        ('cube_height', cube_height),
        ('xyz_scaler', xyz_scaler),
    ):
        if value is None or float(value) <= 0:
            raise ValueError(f'{name} must be a positive resolved threshold, got {value!r}')
    state_cubes = jnp.asarray(state_cubes)
    goal_cubes = jnp.asarray(goal_cubes)
    if state_cubes.shape != goal_cubes.shape:
        raise ValueError(
            'Cube state and goal entity tensors must have identical shape; '
            f'got {state_cubes.shape} and {goal_cubes.shape}'
        )
    current_xyz = _cube_positions(state_cubes, xyz_scaler=xyz_scaler)
    goal_xyz = _cube_positions(goal_cubes, xyz_scaler=xyz_scaler)
    num_cubes = current_xyz.shape[-2]
    if num_cubes <= 1:
        raise ValueError(f'Cube relation builder requires at least two cubes, got {num_cubes}')
    eye = jnp.eye(num_cubes, dtype=bool)

    def support_edges(xyz, *, epsilon_xy, epsilon_z):
        source = xyz[..., :, None, :]
        target = xyz[..., None, :, :]
        delta_xy = jnp.linalg.norm(source[..., :2] - target[..., :2], axis=-1)
        vertical_delta = target[..., 2] - source[..., 2]
        return (
            (delta_xy <= float(epsilon_xy))
            & (jnp.abs(vertical_delta - float(cube_height)) <= float(epsilon_z))
            & (vertical_delta > 0.0)
            & (~eye)
        )

    current_support = support_edges(
        current_xyz,
        epsilon_xy=current_support_epsilon_xy,
        epsilon_z=current_support_epsilon_z,
    )
    goal_support = support_edges(
        goal_xyz,
        epsilon_xy=goal_support_epsilon_xy,
        epsilon_z=goal_support_epsilon_z,
    )
    current = current_xyz[..., :, None, :]
    goal = goal_xyz[..., None, :, :]
    goal_conflict = (
        (jnp.linalg.norm(current - goal, axis=-1) <= float(conflict_radius))
        & (~eye)
    )
    return jnp.stack((current_support, goal_support, goal_conflict), axis=-1).astype(
        state_cubes.dtype
    )


def build_cube_relations(
    state_cubes,
    goal_cubes,
    *,
    mode: str,
    current_support_epsilon_xy: float | None,
    current_support_epsilon_z: float | None,
    goal_support_epsilon_xy: float | None,
    goal_support_epsilon_z: float | None,
    conflict_radius: float | None,
    shuffle_derangement: Sequence[int] = CUBE_TRIPLE_SHUFFLE_DERANGEMENT,
    cube_height: float = 0.04,
    xyz_scaler: float = 10.0,
):
    """Return Zero, Correct, or fixed-endpoint-Shuffled Cube relations."""

    mode = validate_relation_mode(mode)
    state_cubes = jnp.asarray(state_cubes)
    goal_cubes = jnp.asarray(goal_cubes)
    if state_cubes.shape != goal_cubes.shape:
        raise ValueError(
            'Cube state and goal entity tensors must have identical shape; '
            f'got {state_cubes.shape} and {goal_cubes.shape}'
        )
    num_cubes = int(state_cubes.shape[-2])
    permutation = _permutation(
        shuffle_derangement, num_cubes, label='Cube shuffle derangement'
    )
    if mode == 'zero':
        return jnp.zeros((*state_cubes.shape[:-1], num_cubes, 3), dtype=state_cubes.dtype)
    correct = cube_correct_relations(
        state_cubes,
        goal_cubes,
        current_support_epsilon_xy=current_support_epsilon_xy,
        current_support_epsilon_z=current_support_epsilon_z,
        goal_support_epsilon_xy=goal_support_epsilon_xy,
        goal_support_epsilon_z=goal_support_epsilon_z,
        conflict_radius=conflict_radius,
        cube_height=cube_height,
        xyz_scaler=xyz_scaler,
    )
    return correct if mode == 'correct' else _shuffle_endpoints(correct, permutation)


def relation_edge_counts(relations):
    """Return per-sample, per-channel edge counts for audit/tests."""

    relations = jnp.asarray(relations)
    if relations.ndim == 3:
        return jnp.sum(relations, axis=(0, 1))
    if relations.ndim == 4:
        return jnp.sum(relations, axis=(1, 2))
    raise ValueError(
        'Relation tensor must be [T, T, K] or [B, T, T, K]; '
        f'got {relations.shape}'
    )
