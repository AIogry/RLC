"""Architecture regression tests for the M12B ontology cleanup."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from impls.computation.accounting import actor_slot_accounting
from impls.computation.blocks import ResidualMLPStack
from impls.computation.factory import ComputationSpec, make_computation_core
from impls.utils.checkpointing import tree_fingerprint


def _single_spec(iterations=4, sharing=None, state_init='normal_buffer'):
    mapping = {
        'primitive': 'mlp',
        'topology': 'single_state',
        'credit': 'direct',
        'topology_kwargs': {
            'iterations': iterations,
            'residual': False,
            'input_injection': 'z_plus_x',
            'state_dim': 8,
            'state_init': state_init,
            'state_init_std': 1.0,
            'update_depth': 2,
            'layer_norm': False,
            'update_activate_final': True,
        },
    }
    if sharing is not None:
        mapping['parameter_sharing'] = sharing
    return ComputationSpec.from_mapping(mapping)


class M12BArchitectureTest(unittest.TestCase):
    def test_legacy_single_state_defaults_to_explicit_shared_bitwise(self):
        x = jnp.arange(10, dtype=jnp.float32).reshape(2, 5) / 10.0
        legacy = make_computation_core(
            _single_spec(sharing=None), hidden_dims=(8, 8, 8), activate_final=True,
        )
        explicit = make_computation_core(
            _single_spec(sharing='shared'), hidden_dims=(8, 8, 8), activate_final=True,
        )
        rngs = {'params': jax.random.PRNGKey(3), 'buffers': jax.random.PRNGKey(4)}
        legacy_vars = legacy.init(rngs, x)
        explicit_vars = explicit.init(rngs, x)
        self.assertEqual(tree_fingerprint(legacy_vars['params']), tree_fingerprint(explicit_vars['params']))
        self.assertEqual(tree_fingerprint(legacy_vars['buffers']), tree_fingerprint(explicit_vars['buffers']))
        np.testing.assert_array_equal(
            np.asarray(legacy.apply(legacy_vars, x).representation),
            np.asarray(explicit.apply(explicit_vars, x).representation),
        )
        self.assertEqual(set(legacy_vars['params']['topology']), {'input_mapping', 'update_module'})

    def test_single_state_shared_and_untied_share_topology_but_not_parameters(self):
        x = jnp.ones((3, 5), dtype=jnp.float32)
        shared = make_computation_core(
            _single_spec(sharing='shared'), hidden_dims=(8, 8, 8), activate_final=True,
        )
        untied = make_computation_core(
            _single_spec(sharing='untied'), hidden_dims=(8, 8, 8), activate_final=True,
        )
        shared_vars = shared.init(
            {'params': jax.random.PRNGKey(5), 'buffers': jax.random.PRNGKey(6)}, x,
        )
        untied_vars = untied.init(
            {'params': jax.random.PRNGKey(5), 'buffers': jax.random.PRNGKey(6)}, x,
        )
        self.assertIn('update_module', shared_vars['params']['topology'])
        self.assertNotIn('update_modules_0', shared_vars['params']['topology'])
        update_keys = [
            key for key in untied_vars['params']['topology']
            if str(key).startswith('update_modules_')
        ]
        self.assertEqual(len(update_keys), 4)
        self.assertNotIn('update_module', untied_vars['params']['topology'])
        output = untied.apply(untied_vars, x)
        self.assertEqual(output.representation.shape, (3, 8))
        self.assertIsNotNone(output.state)

        def loss(params):
            return untied.apply(
                {'params': params, 'buffers': untied_vars['buffers']}, x,
            ).representation.sum()

        gradients = jax.grad(loss)(untied_vars['params'])['topology']
        for key in update_keys:
            self.assertTrue(np.all(np.isfinite(np.asarray(gradients[key]['Dense_0']['kernel']))))

    def test_single_state_init_pairing_and_k_invariance(self):
        x = jnp.ones((2, 5), dtype=jnp.float32)
        normal_k1 = make_computation_core(
            _single_spec(iterations=1, sharing='shared', state_init='normal_buffer'),
            hidden_dims=(8, 8, 8), activate_final=True,
        )
        normal_k4 = make_computation_core(
            _single_spec(iterations=4, sharing='shared', state_init='normal_buffer'),
            hidden_dims=(8, 8, 8), activate_final=True,
        )
        zero_k1 = make_computation_core(
            _single_spec(iterations=1, sharing='shared', state_init='zero_buffer'),
            hidden_dims=(8, 8, 8), activate_final=True,
        )
        zero_k4 = make_computation_core(
            _single_spec(iterations=4, sharing='shared', state_init='zero_buffer'),
            hidden_dims=(8, 8, 8), activate_final=True,
        )
        rngs = {'params': jax.random.PRNGKey(7), 'buffers': jax.random.PRNGKey(8)}
        nk1 = normal_k1.init(rngs, x)
        nk4 = normal_k4.init(rngs, x)
        zk1 = zero_k1.init(rngs, x)
        zk4 = zero_k4.init(rngs, x)
        self.assertEqual(tree_fingerprint(nk1['params']), tree_fingerprint(nk4['params']))
        self.assertEqual(tree_fingerprint(zk1['params']), tree_fingerprint(zk4['params']))
        self.assertEqual(tree_fingerprint(zk1['buffers']), tree_fingerprint(zk4['buffers']))
        self.assertNotEqual(tree_fingerprint(nk1['buffers']), tree_fingerprint(zk1['buffers']))
        self.assertTrue(np.all(np.asarray(zk1['buffers']['topology']['z_init']) == 0))

    def test_residual_is_a_feedforward_body_with_independent_blocks(self):
        x = jnp.ones((3, 5), dtype=jnp.float32)
        spec = ComputationSpec.from_mapping({
            'primitive': 'mlp',
            'block': 'residual',
            'topology': 'feedforward',
            'credit': 'direct',
            'block_kwargs': {
                'state_dim': 8,
                'blocks': 4,
                'block_depth': 2,
                'layer_norm': False,
                'block_activate_final': True,
            },
        })
        core = make_computation_core(spec, hidden_dims=(8, 8, 8), activate_final=True)
        variables = core.init(jax.random.PRNGKey(9), x)
        output = core.apply(variables, x)
        self.assertEqual(output.representation.shape, (3, 8))
        self.assertIsNone(output.state)
        self.assertNotIn('buffers', variables)
        body = variables['params']['topology']['primitive']
        block_keys = [key for key in body if str(key).startswith('residual_blocks_')]
        self.assertEqual(len(block_keys), 4)
        self.assertEqual(
            actor_slot_accounting(
                {'actor_net': {'topology': variables['params']['topology']}},
                {}, topology='feedforward', block='residual',
            )['executed_dense_layers'],
            9,
        )

        def loss(params):
            return core.apply({'params': params}, x).representation.sum()

        gradients = jax.grad(loss)(variables['params'])
        self.assertTrue(all(
            np.all(np.isfinite(np.asarray(leaf)))
            for leaf in jax.tree_util.tree_leaves(gradients)
        ))

    def test_residual_body_direct_equation_has_no_state_lifecycle(self):
        x = jnp.ones((2, 5), dtype=jnp.float32)
        body = ResidualMLPStack(state_dim=8, blocks=4, block_depth=2)
        variables = body.init(jax.random.PRNGKey(10), x)
        self.assertNotIn('buffers', variables)
        output = body.apply(variables, x)
        self.assertEqual(output.shape, (2, 8))


if __name__ == '__main__':
    unittest.main()
