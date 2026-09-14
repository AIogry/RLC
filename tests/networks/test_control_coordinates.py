from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict

from impls.computation.blocks.linear_boolean_flow import make_permutation_schedule
from impls.networks.control_coordinates import (
    ControlCoordinateFlow,
    resolve_control_coordinate_config,
)


class ControlCoordinateNetworkTest(unittest.TestCase):
    def test_configuration_is_explicit_and_near_identity(self):
        config = {
            'num_bits': 4,
            'num_layers': 4,
            'permutation_seed': 11,
            'coupling_logit_init_mean': -3.0,
            'coupling_logit_init_std': 0.0,
            'binary_temperature': 0.7,
        }
        resolved = resolve_control_coordinate_config(config)
        self.assertEqual(len(resolved['permutations']), 4)
        self.assertEqual(resolved['permutation_schedule']['num_bits'], 4)
        model = ControlCoordinateFlow(
            num_bits=4,
            permutations=resolved['permutations'],
            coupling_logit_init_mean=-3.0,
            coupling_logit_init_std=0.0,
            binary_temperature=0.7,
        )
        states = jnp.asarray([[1, 0, 1, 1], [0, 1, 0, 1]], dtype=jnp.float32)
        variables = model.init(jax.random.PRNGKey(0), states)
        np.testing.assert_array_equal(model.apply(variables, states), states)

    def test_goal_mask_is_exact_H_state_xor_H_goal(self):
        schedule = make_permutation_schedule(4, 4, 5)
        model = ControlCoordinateFlow(4, schedule, -2.0, 0.0, 1.0)
        state = jnp.asarray([[0, 1, 1, 0]], dtype=jnp.float32)
        goal = jnp.asarray([[1, 1, 0, 1]], dtype=jnp.float32)
        variables = model.init(jax.random.PRNGKey(0), state)
        mask = model.apply(variables, state, goal, method=model.goal_mask)
        expected = np.bitwise_xor(
            np.asarray(model.apply(variables, state)).astype(np.uint8),
            np.asarray(model.apply(variables, goal)).astype(np.uint8),
        )
        np.testing.assert_array_equal(mask, expected)

    def test_only_coupling_logits_are_trainable_and_output_capacity_is_exactly_n(self):
        schedule = make_permutation_schedule(6, 5, 13)
        model = ControlCoordinateFlow(6, schedule, -2.0, 0.1, 1.0)
        states = jnp.zeros((3, 6), dtype=jnp.float32)
        variables = model.init(jax.random.PRNGKey(4), states)
        paths = flatten_dict(variables['params'])
        self.assertEqual(len(paths), 5)
        self.assertTrue(
            all(path[-1] == 'coupling_logits' for path in paths)
        )
        self.assertFalse(
            any('bias' in component or 'permutation' in component for path in paths for component in path)
        )
        self.assertEqual(model.apply(variables, states).shape, (3, 6))

    def test_configuration_rejects_missing_or_non_identity_initialization(self):
        with self.assertRaisesRegex(ValueError, 'missing fields'):
            resolve_control_coordinate_config({'num_bits': 4})
        config = {
            'num_bits': 4,
            'num_layers': 2,
            'permutation_seed': 0,
            'coupling_logit_init_mean': 0.0,
            'coupling_logit_init_std': 0.1,
            'binary_temperature': 1.0,
        }
        with self.assertRaisesRegex(ValueError, 'must be negative'):
            resolve_control_coordinate_config(config)


if __name__ == '__main__':
    unittest.main()
