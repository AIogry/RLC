from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import flax
import jax.numpy as jnp
import numpy as np

from impls.agents.control_coordinate import (
    ControlCoordinateAgent,
    axis_locality_objective,
    hard_axis_statistics,
)
from impls.utils.flax_utils import (
    restore_agent_from_checkpoint,
    save_agent,
)


def _config(seed=0):
    return {
        'num_bits': 4,
        'num_layers': 4,
        'permutation_seed': 31,
        'coupling_logit_init_mean': -2.0,
        'coupling_logit_init_std': 0.0,
        'binary_temperature': 1.0,
        'learning_rate': 3e-3,
        'batch_size': 8,
        'train_steps': 10,
        'seed': seed,
        'event_sampling_mode': 'uniform',
    }


def _make_nontrivial(agent):
    params = flax.core.unfreeze(agent.network.params)
    logits = params['boolean_flow']['coupling_000']['coupling_logits']
    logits = np.full_like(np.asarray(logits), -2.0)
    logits[0, 0] = 2.0
    params['boolean_flow']['coupling_000']['coupling_logits'] = jnp.asarray(logits)
    return agent.replace(
        network=agent.network.replace(params=flax.core.freeze(params))
    )


class ControlCoordinateObjectiveTest(unittest.TestCase):
    def test_identity_has_expected_raw_distances_and_surrogate(self):
        agent = ControlCoordinateAgent.create(_config())
        start = np.zeros((2, 4), dtype=np.float32)
        end = np.asarray([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=np.float32)
        loss, info = axis_locality_objective(agent.encode(start), agent.encode(end))
        self.assertAlmostEqual(float(loss), 2.5)
        self.assertAlmostEqual(float(info['metric/hard_axis_distance_mean']), 2.5)
        self.assertEqual(float(info['metric/hard_axis_distance_max']), 3.0)
        self.assertEqual(float(info['metric/hard_axis_success']), 0.0)

    def test_exact_synthetic_inverse_transform_has_zero_axis_loss(self):
        agent = _make_nontrivial(ControlCoordinateAgent.create(_config()))
        axes = np.eye(4, dtype=np.float32)
        operation_effects = np.asarray(agent.inverse_encode(axes))
        self.assertFalse(np.array_equal(operation_effects, axes))
        contexts = np.asarray([
            [0, 0, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 1],
            [1, 1, 0, 1],
        ], dtype=np.uint8)
        endpoints = np.bitwise_xor(contexts, operation_effects.astype(np.uint8))
        loss, info = axis_locality_objective(
            agent.encode(contexts.astype(np.float32)),
            agent.encode(endpoints.astype(np.float32)),
        )
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(info['metric/hard_axis_success']), 1.0)

    def test_hard_metric_refuses_relaxed_values(self):
        with self.assertRaisesRegex(ValueError, 'exactly 0/1'):
            hard_axis_statistics(
                jnp.asarray([[0.1, 0.9]]), jnp.asarray([[0.9, 0.1]])
            )

    def test_axis_surrogate_has_finite_nonzero_coupling_gradients(self):
        agent = ControlCoordinateAgent.create(_config())
        batch = {
            'start_board': np.asarray([
                [0, 0, 0, 0], [1, 0, 0, 1], [0, 1, 1, 0], [1, 1, 0, 0]
            ], dtype=np.uint8),
            'end_board': np.asarray([
                [1, 1, 0, 0], [0, 1, 1, 1], [1, 0, 0, 1], [0, 0, 1, 1]
            ], dtype=np.uint8),
        }
        _, info = agent.update(batch)
        self.assertTrue(np.isfinite(float(info['loss/axis_surrogate'])))
        self.assertTrue(np.isfinite(float(info['grad/norm'])))
        self.assertGreater(float(info['grad/norm']), 0.0)

    def test_agent_rejects_zero_change_and_any_diagnostic_or_identity_field(self):
        agent = ControlCoordinateAgent.create(_config())
        zero = np.zeros((2, 4), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, 'Zero-change'):
            agent.update({'start_board': zero, 'end_board': zero})
        valid = {'start_board': zero, 'end_board': zero.copy()}
        valid['end_board'][:, 0] = 1
        valid['raw_effect_signature'] = np.asarray(['x', 'y'])
        with self.assertRaisesRegex(ValueError, 'unexpected fields'):
            agent.update(valid)


class ControlCoordinateCheckpointTest(unittest.TestCase):
    def test_save_load_preserves_exact_boolean_outputs(self):
        config = _config(seed=9)
        agent = _make_nontrivial(ControlCoordinateAgent.create(config))
        states = np.asarray([
            [0, 0, 0, 0], [1, 0, 1, 1], [0, 1, 1, 0], [1, 1, 1, 1]
        ], dtype=np.float32)
        expected = np.asarray(agent.encode(states))
        expected_inverse = np.asarray(agent.inverse_encode(expected))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = save_agent(agent, Path(directory), 7)
            restored = restore_agent_from_checkpoint(
                ControlCoordinateAgent.create(config), checkpoint
            )
            np.testing.assert_array_equal(restored.encode(states), expected)
            np.testing.assert_array_equal(
                restored.inverse_encode(expected), expected_inverse
            )
            np.testing.assert_array_equal(restored.effective_matrix(), agent.effective_matrix())


if __name__ == '__main__':
    unittest.main()
