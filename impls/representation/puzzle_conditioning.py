"""Parameter-free Puzzle goal re-expression on canonical observations."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .puzzle import parse_puzzle_observation
from .puzzle_algebra import apply_gf2_map, operator_metadata
from .interfaces import GoalConditioningOutput


PUZZLE_GOAL_CONDITIONING_MODES = frozenset({
    'canonical',
    'board',
    'residual',
    'oracle_operation',
})


def _validate_eager_binary(value, *, name: str) -> None:
    leaves = jax.tree_util.tree_leaves(value)
    if any(isinstance(leaf, jax.core.Tracer) for leaf in leaves):
        return
    array = np.asarray(value)
    if not (
        np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError(f'{name} must be numeric binary data, got {array.dtype}')
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{name} contains non-finite values')
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f'{name} must contain only 0/1 values')


def extract_button_bits(
    observation,
    *,
    num_buttons: int,
    robot_dim: int = 19,
    button_feature_dim: int = 4,
):
    """Extract board bits through the one canonical Puzzle parser."""

    _, buttons = parse_puzzle_observation(
        observation,
        num_buttons=num_buttons,
        robot_dim=robot_dim,
        button_feature_dim=button_feature_dim,
    )
    if button_feature_dim < 2:
        raise ValueError(
            'Puzzle button features must contain the two state-onehot fields; '
            f'got button_feature_dim={button_feature_dim}'
        )
    onehot = buttons[..., :2]
    _validate_eager_binary(onehot, name='Puzzle button state onehot')
    if not any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(onehot)):
        sums = np.asarray(onehot).sum(axis=-1)
        if not np.all(sums == 1):
            raise ValueError('Puzzle button state fields must be valid two-class onehots')
    return jnp.argmax(onehot, axis=-1).astype(jnp.uint8)


def encode_binary_button_condition(bits, *, dtype=jnp.float32):
    """Encode bits as canonical ``[onehot(bit), q=0, qdot=0]`` features."""

    bits = jnp.asarray(bits)
    if bits.ndim < 1:
        raise ValueError(f'Puzzle condition bits require a button axis, got {bits.shape}')
    _validate_eager_binary(bits, name='Puzzle condition bits')
    bits = bits.astype(dtype)
    zeros = jnp.zeros_like(bits)
    return jnp.stack((1 - bits, bits, zeros, zeros), axis=-1)


def condition_puzzle_goal(
    observations,
    goals,
    *,
    mode: str,
    rows: int,
    cols: int,
    num_buttons: int,
    robot_dim: int = 19,
    button_feature_dim: int = 4,
    operation_inverse=None,
):
    """Return one call-specific Puzzle goal in canonical observation shape.

    ``residual`` and ``oracle_operation`` are recomputed from the observation
    supplied to this call.  No state-dependent result is cached.
    """

    if mode not in PUZZLE_GOAL_CONDITIONING_MODES:
        raise ValueError(
            f'Unsupported Puzzle goal-conditioning mode {mode!r}; '
            f'expected one of {sorted(PUZZLE_GOAL_CONDITIONING_MODES)!r}'
        )
    if rows * cols != num_buttons:
        raise ValueError(
            'Puzzle grid must match num_buttons: '
            f'rows*cols={rows * cols}, num_buttons={num_buttons}'
        )
    observations = jnp.asarray(observations)
    goals = jnp.asarray(goals)
    if observations.shape != goals.shape:
        raise ValueError(
            'Puzzle observations and goals must have identical canonical shape; '
            f'got {observations.shape} and {goals.shape}'
        )
    robot_goal, _ = parse_puzzle_observation(
        goals,
        num_buttons=num_buttons,
        robot_dim=robot_dim,
        button_feature_dim=button_feature_dim,
    )
    # Validate the current observation through the same parser even for board
    # and canonical modes; there is no second 19 + 4N layout implementation.
    parse_puzzle_observation(
        observations,
        num_buttons=num_buttons,
        robot_dim=robot_dim,
        button_feature_dim=button_feature_dim,
    )
    if mode == 'canonical':
        return goals

    current_bits = extract_button_bits(
        observations,
        num_buttons=num_buttons,
        robot_dim=robot_dim,
        button_feature_dim=button_feature_dim,
    )
    goal_bits = extract_button_bits(
        goals,
        num_buttons=num_buttons,
        robot_dim=robot_dim,
        button_feature_dim=button_feature_dim,
    )
    if mode == 'board':
        condition_bits = goal_bits
    else:
        residual = jnp.bitwise_xor(current_bits, goal_bits)
        if mode == 'residual':
            condition_bits = residual
        else:
            if operation_inverse is None:
                metadata = operator_metadata(rows, cols)
                operation_inverse = metadata['inverse']
            if operation_inverse is None:
                rank = operator_metadata(rows, cols)['rank']
                raise ValueError(
                    'oracle_operation requires a unique full-rank GF(2) operator; '
                    f'layout={rows}x{cols}, rank={rank}, num_buttons={num_buttons}'
                )
            condition_bits = apply_gf2_map(operation_inverse, residual)

    conditioned_buttons = encode_binary_button_condition(
        condition_bits, dtype=goals.dtype
    )
    if conditioned_buttons.shape[-2:] != (num_buttons, button_feature_dim):
        raise ValueError(
            'Conditioned Puzzle button layout must remain canonical 4D; '
            f'got {conditioned_buttons.shape[-2:]}, expected '
            f'{(num_buttons, button_feature_dim)}'
        )
    return jnp.concatenate(
        [jnp.zeros_like(robot_goal), conditioned_buttons.reshape(*goals.shape[:-1], -1)],
        axis=-1,
    )


def prepare_puzzle_goal(
    observations,
    goals,
    *,
    coordinate,
    distance_feature,
    transform,
    num_buttons,
    robot_dim=19,
    button_feature_dim=4,
    operation_inverse=None,
):
    """V2 preparation using only this call's state and the external raw goal.

    The factory has already validated/frozen the operator and transform.
    Residual+zero needs no operation coordinates, including at setup.
    """
    observations, goals = jnp.asarray(observations), jnp.asarray(goals)
    if observations.shape != goals.shape:
        raise ValueError('Puzzle observations and raw goals must have identical shape')
    for name, value in (('observations', observations), ('goals', goals)):
        if not isinstance(value, jax.core.Tracer) and not np.all(np.isfinite(np.asarray(value))):
            raise ValueError(f'Puzzle {name} contains non-finite values')
    layout = dict(num_buttons=num_buttons, robot_dim=robot_dim,
                  button_feature_dim=button_feature_dim)
    current_bits = extract_button_bits(observations, **layout)
    goal_bits = extract_button_bits(goals, **layout)
    residual = jnp.bitwise_xor(current_bits, goal_bits)
    needs_operation = coordinate == 'operation' or distance_feature == 'exact_press_fraction'
    if coordinate not in ('residual', 'operation'):
        raise ValueError(f'Unsupported goal coordinate {coordinate!r}')
    if distance_feature not in ('zero', 'exact_press_fraction'):
        raise ValueError(f'Unsupported distance_feature {distance_feature!r}')
    operation = None
    if needs_operation:
        if operation_inverse is None:
            raise ValueError('Operation/distance preparation requires a setup-time inverse')
        operation = apply_gf2_map(operation_inverse, residual)
    condition = transform(operation) if coordinate == 'operation' else residual
    buttons = encode_binary_button_condition(condition, dtype=goals.dtype)
    robot_goal, _ = parse_puzzle_observation(goals, **layout)
    conditioned_goal = jnp.concatenate(
        [jnp.zeros_like(robot_goal), buttons.reshape(*goals.shape[:-1], -1)], axis=-1,
    )
    shape = (*goals.shape[:-1], num_buttons, 1)
    if distance_feature == 'zero':
        aux = jnp.zeros(shape, dtype=jnp.float32)
    else:
        fraction = jnp.sum(operation, axis=-1, dtype=jnp.float32) / num_buttons
        aux = jnp.broadcast_to(fraction[..., None, None], shape)
    return GoalConditioningOutput(conditioned_goal, aux)


__all__ = [
    'PUZZLE_GOAL_CONDITIONING_MODES',
    'condition_puzzle_goal',
    'encode_binary_button_condition',
    'extract_button_bits',
    'prepare_puzzle_goal',
]
