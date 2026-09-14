"""Network-level wrapper for the M25 linear Boolean coordinate map."""

from __future__ import annotations

from numbers import Integral, Real

import flax.linen as nn
import jax.numpy as jnp

from ..computation.blocks.linear_boolean_flow import (
    LinearBooleanFlow,
    binary_xor,
    make_permutation_schedule,
    permutation_schedule_metadata,
    validate_binary_array,
)


REQUIRED_FLOW_CONFIG_FIELDS = (
    'num_bits',
    'num_layers',
    'permutation_seed',
    'coupling_logit_init_mean',
    'coupling_logit_init_std',
    'binary_temperature',
)


def resolve_control_coordinate_config(config):
    """Validate flow configuration and materialize its fixed permutations."""

    missing = [field for field in REQUIRED_FLOW_CONFIG_FIELDS if field not in config]
    if missing:
        raise ValueError(f'Control-coordinate config is missing fields: {missing}')
    result = {field: config[field] for field in REQUIRED_FLOW_CONFIG_FIELDS}
    for field in ('num_bits', 'num_layers'):
        value = result[field]
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f'{field} must be a positive integer, got {value!r}')
        result[field] = int(value)
    seed = result['permutation_seed']
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError(f'permutation_seed must be an integer, got {seed!r}')
    result['permutation_seed'] = int(seed)
    for field in (
        'coupling_logit_init_mean',
        'coupling_logit_init_std',
        'binary_temperature',
    ):
        value = result[field]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f'{field} must be a real scalar, got {value!r}')
        result[field] = float(value)
    if result['coupling_logit_init_mean'] >= 0:
        raise ValueError('coupling_logit_init_mean must be negative')
    if result['coupling_logit_init_std'] < 0:
        raise ValueError('coupling_logit_init_std must be non-negative')
    if result['binary_temperature'] <= 0:
        raise ValueError('binary_temperature must be positive')

    schedule = make_permutation_schedule(
        result['num_bits'], result['num_layers'], result['permutation_seed']
    )
    result['permutations'] = schedule
    result['permutation_schedule'] = permutation_schedule_metadata(schedule)
    return result


class ControlCoordinateFlow(nn.Module):
    """The information-preserving linear map H(b)=Ab over GF(2)."""

    num_bits: int
    permutations: tuple[tuple[int, ...], ...]
    coupling_logit_init_mean: float
    coupling_logit_init_std: float
    binary_temperature: float

    def setup(self):
        self.boolean_flow = LinearBooleanFlow(
            num_bits=self.num_bits,
            permutations=self.permutations,
            coupling_logit_init_mean=self.coupling_logit_init_mean,
            coupling_logit_init_std=self.coupling_logit_init_std,
            binary_temperature=self.binary_temperature,
        )

    def encode(self, board):
        board = validate_binary_array(
            board, name='control-coordinate board', final_dim=self.num_bits
        )
        return self.boolean_flow.forward(board)

    def inverse_encode(self, coordinates):
        coordinates = validate_binary_array(
            coordinates,
            name='control-coordinate inverse input',
            final_dim=self.num_bits,
        )
        return self.boolean_flow.inverse(coordinates)

    def effective_matrix(self):
        """Return A[out_bit, input_bit] from encoded standard basis rows."""

        encoded_basis = self.encode(jnp.eye(self.num_bits, dtype=jnp.float32))
        return encoded_basis.T

    def goal_mask(self, state_board, goal_board):
        state = self.encode(state_board)
        goal = self.encode(goal_board)
        return binary_xor(state, goal)

    def hard_coupling_matrices(self):
        return self.boolean_flow.hard_coupling_matrices()

    def __call__(self, board):
        return self.encode(board)


__all__ = [
    'ControlCoordinateFlow',
    'REQUIRED_FLOW_CONFIG_FIELDS',
    'resolve_control_coordinate_config',
]
