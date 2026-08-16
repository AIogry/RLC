"""M8 computation interface and standalone block parity tests."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from impls.computation.accounting import count_parameters
from impls.computation.blocks import MLPMixerBlock
from impls.computation.factory import ComputationSpec, make_computation_core
from impls.computation.interfaces import ComputationOutput
from impls.networks.coghp import MixerBlock


class ComputationFoundationTest(unittest.TestCase):
    def test_core_delegates_none_state_and_feedforward_rejects_non_none_state(self):
        x = jnp.ones((2, 3, 5), dtype=jnp.float32)
        core = make_computation_core(
            ComputationSpec(), hidden_dims=(7, 5), activate_final=True
        )
        variables = core.init(jax.random.PRNGKey(0), x, state=None)
        output = core.apply(variables, x, state=None)
        self.assertIsInstance(output, ComputationOutput)
        self.assertIsNone(output.state)
        self.assertEqual(output.representation.shape, x.shape)

        with self.assertRaisesRegex(ValueError, 'FeedForward topology'):
            core.apply(variables, x, state=jnp.zeros((2, 5)))

    def test_mixer_block_matches_vanilla_reference_for_multiple_shapes(self):
        cases = (
            (2, 4, 3, 5, 7, 6),
            (1, 3, 4, 6, 5, 8),
            (3, 6, 5, 2, 9, 4),
        )
        for case in cases:
            batch_size, num_tokens, embed_dim, token_hidden, channel_hidden, seed = case
            x = jnp.arange(batch_size * num_tokens * embed_dim, dtype=jnp.float32)
            x = x.reshape(batch_size, num_tokens, embed_dim) / 17.0
            reference = MixerBlock(
                num_tokens=num_tokens,
                embed_dim=embed_dim,
                hidden_dim_tokens=token_hidden,
                hidden_dim_channels=channel_hidden,
            )
            computation = MLPMixerBlock(
                num_tokens=num_tokens,
                embed_dim=embed_dim,
                hidden_dim_tokens=token_hidden,
                hidden_dim_channels=channel_hidden,
            )
            key = jax.random.PRNGKey(seed)
            reference_params = reference.init(key, x)['params']
            computation_params = computation.init(key, x)['params']

            self.assertEqual(
                count_parameters(reference_params), count_parameters(computation_params)
            )
            for expected, actual in zip(
                jax.tree_util.tree_leaves(reference_params),
                jax.tree_util.tree_leaves(computation_params),
            ):
                np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))

            expected_output = reference.apply({'params': reference_params}, x)
            actual_output = computation.apply({'params': computation_params}, x)
            np.testing.assert_array_equal(np.asarray(expected_output), np.asarray(actual_output))

            def reference_loss(params):
                output = reference.apply({'params': params}, x)
                return jnp.sum(output * output)

            def computation_loss(params):
                output = computation.apply({'params': params}, x)
                return jnp.sum(output * output)

            reference_grads = jax.grad(reference_loss)(reference_params)
            computation_grads = jax.grad(computation_loss)(computation_params)
            for expected, actual in zip(
                jax.tree_util.tree_leaves(reference_grads),
                jax.tree_util.tree_leaves(computation_grads),
            ):
                np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))


if __name__ == '__main__':
    unittest.main()
