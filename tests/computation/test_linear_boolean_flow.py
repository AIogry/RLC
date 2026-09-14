from __future__ import annotations

import itertools
import unittest

import flax
import jax
import jax.numpy as jnp
import numpy as np

from impls.computation.blocks.linear_boolean_flow import (
    binary_threshold_ste,
    binary_xor,
    gf2_parity,
    make_permutation_schedule,
    permutation_schedule_metadata,
)
from impls.diagnostics.puzzle.control_coordinates import gf2_apply, gf2_rank
from impls.networks.control_coordinates import ControlCoordinateFlow


def _all_states(num_bits):
    return np.asarray(list(itertools.product((0, 1), repeat=num_bits)), dtype=np.float32)


def _nontrivial_model(num_bits=4, num_layers=4, seed=7):
    schedule = make_permutation_schedule(num_bits, num_layers, seed)
    model = ControlCoordinateFlow(
        num_bits=num_bits,
        permutations=schedule,
        coupling_logit_init_mean=-2.0,
        coupling_logit_init_std=0.0,
        binary_temperature=0.75,
    )
    variables = model.init(jax.random.PRNGKey(0), jnp.zeros((1, num_bits)))
    params = flax.core.unfreeze(variables['params'])
    for layer_index, layer in enumerate(params['boolean_flow'].values()):
        logits = np.full_like(np.asarray(layer['coupling_logits']), -2.0)
        logits.flat[layer_index % logits.size] = 2.0
        layer['coupling_logits'] = jnp.asarray(logits)
    return model, {'params': flax.core.freeze(params)}


class LinearBooleanPrimitiveTest(unittest.TestCase):
    def test_binary_ste_has_exact_hard_primal_and_smooth_nonzero_gradient(self):
        logits = jnp.asarray([-1.0, 0.0, 1.0])
        hard = binary_threshold_ste(logits, temperature=0.5)
        np.testing.assert_array_equal(np.asarray(hard), [0.0, 1.0, 1.0])
        gradient = jax.grad(
            lambda value: binary_threshold_ste(value, temperature=0.5).sum()
        )(logits)
        self.assertTrue(np.all(np.isfinite(np.asarray(gradient))))
        self.assertTrue(np.all(np.asarray(gradient) > 0))

    def test_xor_and_parity_have_exact_gf2_forward_values(self):
        left = jnp.asarray([0.0, 0.0, 1.0, 1.0])
        right = jnp.asarray([0.0, 1.0, 0.0, 1.0])
        np.testing.assert_array_equal(binary_xor(left, right), [0, 1, 1, 0])
        source = jnp.asarray([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
        matrix = jnp.asarray([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        np.testing.assert_array_equal(
            gf2_parity(source, matrix), np.asarray([[0, 0], [1, 0]])
        )

    def test_permutation_schedule_is_seeded_distinct_and_exposes_both_halves(self):
        first = make_permutation_schedule(8, 6, 91)
        second = make_permutation_schedule(8, 6, 91)
        other = make_permutation_schedule(8, 6, 92)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        partitions = {tuple(sorted(permutation[:4])) for permutation in first}
        self.assertEqual(len(partitions), 6)
        metadata = permutation_schedule_metadata(first)
        self.assertTrue(metadata['all_coordinates_exposed_as_source'])
        self.assertTrue(metadata['all_coordinates_exposed_as_target'])
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            make_permutation_schedule(2, 3, 0)


class LinearBooleanFlowTest(unittest.TestCase):
    def setUp(self):
        self.model, self.variables = _nontrivial_model()
        self.states = _all_states(4)

    def test_forward_is_binary_and_both_inverse_compositions_are_exact_exhaustively(self):
        encoded = self.model.apply(self.variables, self.states)
        recovered = self.model.apply(
            self.variables, encoded, method=self.model.inverse_encode
        )
        inverse_first = self.model.apply(
            self.variables, self.states, method=self.model.inverse_encode
        )
        round_trip = self.model.apply(self.variables, inverse_first)
        self.assertTrue(set(np.unique(np.asarray(encoded))) <= {0.0, 1.0})
        np.testing.assert_array_equal(recovered, self.states)
        np.testing.assert_array_equal(round_trip, self.states)

    def test_zero_and_gf2_linearity_hold_exactly_exhaustively(self):
        zero = np.zeros((1, 4), dtype=np.float32)
        np.testing.assert_array_equal(self.model.apply(self.variables, zero), zero)
        encoded = np.asarray(self.model.apply(self.variables, self.states)).astype(np.uint8)
        for left_index, left in enumerate(self.states):
            xor_inputs = np.bitwise_xor(
                left.astype(np.uint8), self.states.astype(np.uint8)
            ).astype(np.float32)
            encoded_xor = np.asarray(self.model.apply(self.variables, xor_inputs)).astype(
                np.uint8
            )
            expected = np.bitwise_xor(encoded[left_index], encoded)
            np.testing.assert_array_equal(encoded_xor, expected)

    def test_effective_matrix_exactly_reproduces_forward_and_stays_full_rank(self):
        matrix = np.asarray(
            self.model.apply(self.variables, method=self.model.effective_matrix)
        ).astype(np.uint8)
        self.assertEqual(gf2_rank(matrix), 4)
        predicted = gf2_apply(matrix, self.states.astype(np.uint8))
        actual = np.asarray(self.model.apply(self.variables, self.states)).astype(np.uint8)
        np.testing.assert_array_equal(predicted, actual)


if __name__ == '__main__':
    unittest.main()
