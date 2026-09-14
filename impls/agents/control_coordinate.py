"""Standalone Stage-1 optimizer for M25 control-coordinate discovery."""

from __future__ import annotations

from numbers import Integral, Real

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from ..computation.blocks.linear_boolean_flow import binary_xor, validate_binary_array
from ..networks.control_coordinates import (
    ControlCoordinateFlow,
    resolve_control_coordinate_config,
)
from ..utils.flax_utils import TrainState, nonpytree_field


def hard_axis_statistics(encoded_start, encoded_end):
    """Compute scientific metrics from actual hard binary flow outputs."""

    encoded_start = validate_binary_array(
        encoded_start, name='hard encoded event starts'
    )
    encoded_end = validate_binary_array(
        encoded_end, name='hard encoded event ends'
    )
    if encoded_start.shape != encoded_end.shape or encoded_start.ndim < 2:
        raise ValueError(
            'Encoded event endpoints must have identical [batch, bits] shapes, '
            f'got {encoded_start.shape} and {encoded_end.shape}'
        )
    hard_delta = jnp.bitwise_xor(
        encoded_start.astype(jnp.uint8), encoded_end.astype(jnp.uint8)
    )
    distance = hard_delta.sum(axis=-1).astype(jnp.float32)
    return {
        'metric/hard_axis_success': jnp.mean(distance == 1),
        'metric/hard_axis_distance_mean': jnp.mean(distance),
        'metric/hard_axis_distance_median': jnp.median(distance),
        'metric/hard_axis_distance_gt_one': jnp.mean(distance > 1),
        'metric/hard_axis_distance_max': jnp.max(distance),
    }


def axis_locality_objective(encoded_start, encoded_end):
    """Return the STE axis-locality surrogate and separate hard metrics."""

    encoded_start = jnp.asarray(encoded_start)
    encoded_end = jnp.asarray(encoded_end)
    if encoded_start.shape != encoded_end.shape:
        raise ValueError(
            f'Encoded endpoints differ in shape: {encoded_start.shape} != {encoded_end.shape}'
        )
    delta_surrogate = binary_xor(encoded_start, encoded_end)
    surrogate_distance = delta_surrogate.sum(axis=-1)
    loss = jnp.mean((surrogate_distance - 1.0) ** 2)
    info = {'loss/axis_surrogate': loss}
    info.update(hard_axis_statistics(encoded_start, encoded_end))
    return loss, info


def _validate_training_config(config):
    resolved_flow = resolve_control_coordinate_config(config)
    required = ('learning_rate', 'batch_size', 'train_steps', 'seed', 'event_sampling_mode')
    missing = [field for field in required if field not in config]
    if missing:
        raise ValueError(f'Control-coordinate training config is missing fields: {missing}')
    if config['event_sampling_mode'] != 'uniform':
        raise ValueError(
            'M25 Stage-1 primary training requires event_sampling_mode="uniform"'
        )
    for field in ('batch_size', 'train_steps'):
        value = config[field]
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f'{field} must be a positive integer, got {value!r}')
    seed = config['seed']
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError(f'seed must be a non-negative integer, got {seed!r}')
    learning_rate = config['learning_rate']
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, Real) or learning_rate <= 0:
        raise ValueError(f'learning_rate must be positive, got {learning_rate!r}')
    return resolved_flow


def _validate_event_batch(batch, num_bits):
    allowed = {'start_board', 'end_board', 'record_indices'}
    unexpected = set(batch) - allowed
    if unexpected:
        raise ValueError(
            'Stage-1 batches may contain only board endpoints and optional record indices; '
            f'got unexpected fields {sorted(unexpected)}'
        )
    if 'start_board' not in batch or 'end_board' not in batch:
        raise ValueError('Stage-1 batch requires start_board and end_board')
    start = np.asarray(batch['start_board'])
    end = np.asarray(batch['end_board'])
    if start.shape != end.shape or start.ndim != 2 or start.shape[-1] != num_bits:
        raise ValueError(
            f'Expected matching [batch, {num_bits}] boards, got {start.shape} and {end.shape}'
        )
    if not np.all((start == 0) | (start == 1)) or not np.all((end == 0) | (end == 1)):
        raise ValueError('Stage-1 event boards must be exactly binary')
    if np.any(np.all(start == end, axis=-1)):
        raise ValueError('Zero-change transitions are not valid Stage-1 events')


class ControlCoordinateAgent(flax.struct.PyTreeNode):
    """Optimizer state for the standalone reversible Boolean H model."""

    rng: object
    network: TrainState
    config: object = nonpytree_field()

    @jax.jit
    def encode(self, boards):
        return self.network(boards)

    @jax.jit
    def inverse_encode(self, coordinates):
        return self.network(coordinates, method='inverse_encode')

    @jax.jit
    def effective_matrix(self):
        return self.network(method='effective_matrix')

    @jax.jit
    def goal_mask(self, state_boards, goal_boards):
        return self.network(state_boards, goal_boards, method='goal_mask')

    @jax.jit
    def hard_coupling_matrices(self):
        return self.network(method='hard_coupling_matrices')

    def total_loss(self, batch, grad_params):
        encoded_start = self.network(batch['start_board'], params=grad_params)
        encoded_end = self.network(batch['end_board'], params=grad_params)
        return axis_locality_objective(encoded_start, encoded_end)

    @jax.jit
    def _update(self, batch):
        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params)

        network, info = self.network.apply_loss_fn(loss_fn)
        rng, _ = jax.random.split(self.rng)
        return self.replace(network=network, rng=rng), info

    def update(self, batch):
        _validate_event_batch(batch, int(self.config['num_bits']))
        model_batch = {
            'start_board': jnp.asarray(batch['start_board'], dtype=jnp.float32),
            'end_board': jnp.asarray(batch['end_board'], dtype=jnp.float32),
        }
        return self._update(model_batch)

    def supervised_binary_loss(self, inputs, targets, grad_params):
        predictions = self.network(inputs, params=grad_params)
        loss = jnp.mean((predictions - targets) ** 2)
        exact = jnp.all(
            predictions.astype(jnp.uint8) == targets.astype(jnp.uint8), axis=-1
        )
        return loss, {
            'loss/supervised_binary': loss,
            'metric/hard_exact_state_accuracy': jnp.mean(exact),
            'metric/hard_bit_accuracy': jnp.mean(predictions == targets),
        }

    @jax.jit
    def _supervised_update(self, inputs, targets):
        def loss_fn(grad_params):
            return self.supervised_binary_loss(inputs, targets, grad_params)

        network, info = self.network.apply_loss_fn(loss_fn)
        rng, _ = jax.random.split(self.rng)
        return self.replace(network=network, rng=rng), info

    def supervised_update(self, inputs, targets):
        """Architecture-capacity-only update; never used by Puzzle training."""

        inputs = np.asarray(inputs)
        targets = np.asarray(targets)
        num_bits = int(self.config['num_bits'])
        if inputs.shape != targets.shape or inputs.ndim != 2 or inputs.shape[-1] != num_bits:
            raise ValueError(
                f'Expected matching supervised [batch, {num_bits}] arrays, '
                f'got {inputs.shape} and {targets.shape}'
            )
        if not np.all((inputs == 0) | (inputs == 1)) or not np.all(
            (targets == 0) | (targets == 1)
        ):
            raise ValueError('Supervised capacity pairs must be binary')
        return self._supervised_update(
            jnp.asarray(inputs, dtype=jnp.float32),
            jnp.asarray(targets, dtype=jnp.float32),
        )

    @classmethod
    def create(cls, config, example_board=None):
        resolved = _validate_training_config(config)
        seed = int(config['seed'])
        num_bits = int(resolved['num_bits'])
        if example_board is None:
            example_board = jnp.zeros((1, num_bits), dtype=jnp.float32)
        example_board = jnp.asarray(example_board, dtype=jnp.float32)
        if example_board.ndim != 2 or example_board.shape[-1] != num_bits:
            raise ValueError(
                f'example_board must have shape [batch, {num_bits}], got {example_board.shape}'
            )

        model = ControlCoordinateFlow(
            num_bits=num_bits,
            permutations=resolved['permutations'],
            coupling_logit_init_mean=resolved['coupling_logit_init_mean'],
            coupling_logit_init_std=resolved['coupling_logit_init_std'],
            binary_temperature=resolved['binary_temperature'],
        )
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        variables = model.init(init_rng, example_board)
        network = TrainState.create(
            model,
            variables['params'],
            tx=optax.adam(float(config['learning_rate'])),
        )
        stored_config = dict(config)
        stored_config['permutations'] = resolved['permutations']
        stored_config['permutation_schedule'] = resolved['permutation_schedule']
        return cls(
            rng=rng,
            network=network,
            config=flax.core.freeze(stored_config),
        )


def get_config():
    """Return explicit engineering defaults, not a frozen scientific depth."""

    return ml_collections.ConfigDict({
        'agent_name': 'control_coordinate',
        'num_bits': 20,
        'num_layers': 32,
        'permutation_seed': 25001,
        'coupling_logit_init_mean': -2.0,
        'coupling_logit_init_std': 0.1,
        'binary_temperature': 1.0,
        'learning_rate': 3e-3,
        'batch_size': 256,
        'train_steps': 5_000,
        'seed': 0,
        'event_sampling_mode': 'uniform',
    })


__all__ = [
    'ControlCoordinateAgent',
    'axis_locality_objective',
    'get_config',
    'hard_axis_statistics',
]
