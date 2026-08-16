import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze, unfreeze

from impls.computation.factory import ComputationSpec, make_computation_core
from impls.computation.primitives.mlp import MLP


def _spec(iterations=1, residual=False, state_dim=8):
    return ComputationSpec.from_mapping({
        'primitive': 'mlp',
        'topology': 'single_state',
        'credit': 'direct',
        'topology_kwargs': {
            'iterations': iterations,
            'residual': residual,
            'input_injection': 'z_plus_x',
            'state_dim': state_dim,
            'state_init': 'normal_buffer',
            'state_init_std': 1.0,
        },
    })


def _init_core(iterations=1, residual=False, input_dim=5, state_dim=8, seed=0, buffer_seed=1):
    core = make_computation_core(
        _spec(iterations, residual, state_dim),
        hidden_dims=(state_dim, state_dim, state_dim),
    )
    x = jnp.arange(2 * input_dim, dtype=jnp.float32).reshape(2, input_dim) / 10.0
    variables = core.init(
        {'params': jax.random.PRNGKey(seed), 'buffers': jax.random.PRNGKey(buffer_seed)},
        x,
    )
    return core, variables, x


def _manual_unroll(variables, x, iterations, residual):
    topology = variables['params']['topology']
    x_hidden = MLP(hidden_dims=(8,), activate_final=True).apply(
        {'params': topology['input_mapping']}, x,
    )
    z = jnp.broadcast_to(variables['buffers']['topology']['z_init'], x_hidden.shape)
    update_params = topology['update_module']
    for _ in range(iterations):
        update = MLP(hidden_dims=(8, 8), activate_final=True).apply(
            {'params': update_params}, z + x_hidden,
        )
        z = z + update if residual else update
    return z


class SingleStateTopologyTest(unittest.TestCase):
    def test_dimensions_buffer_and_forward_immutability(self):
        core, variables, x = _init_core(iterations=2, residual=True)
        self.assertIn('buffers', variables)
        self.assertNotIn('z_init', variables['params'])
        self.assertEqual(variables['buffers']['topology']['z_init'].shape, (8,))
        output = core.apply(variables, x)
        np.testing.assert_equal(output.representation.shape, (2, 8))
        np.testing.assert_equal(output.state.shape, (2, 8))
        np.testing.assert_array_equal(
            np.asarray(variables['buffers']['topology']['z_init']),
            np.asarray(variables['buffers']['topology']['z_init']),
        )
        np.testing.assert_allclose(output.representation, core.apply(variables, x).representation)

    def test_same_and_different_buffer_seeds(self):
        core, variables_a, x = _init_core(seed=4, buffer_seed=5)
        _, variables_b, _ = _init_core(seed=4, buffer_seed=5)
        _, variables_c, _ = _init_core(seed=4, buffer_seed=6)
        np.testing.assert_array_equal(
            variables_a['buffers']['topology']['z_init'],
            variables_b['buffers']['topology']['z_init'],
        )
        self.assertFalse(np.array_equal(
            np.asarray(variables_a['buffers']['topology']['z_init']),
            np.asarray(variables_c['buffers']['topology']['z_init']),
        ))
        # The buffer RNG is independent of parameter initialization.
        np.testing.assert_equal(
            jax.tree_util.tree_map(
                lambda a, b: np.array_equal(np.asarray(a), np.asarray(b)),
                variables_a['params'], variables_c['params'],
            ),
            jax.tree_util.tree_map(lambda _: True, variables_a['params']),
        )

    def test_residual_and_non_residual_equations(self):
        for residual in (False, True):
            core, variables, x = _init_core(iterations=2, residual=residual)
            actual = core.apply(variables, x).representation
            expected = _manual_unroll(variables, x, iterations=2, residual=residual)
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)

    def test_one_shared_update_module_for_k1_k2_k4(self):
        counts = []
        for iterations in (1, 2, 4):
            _, variables, _ = _init_core(iterations=iterations)
            params = variables['params']['topology']['update_module']
            counts.append(sum(np.asarray(value).size for value in jax.tree_util.tree_leaves(params)))
            self.assertEqual(set(params.keys()), {'Dense_0', 'Dense_1'})
        self.assertEqual(counts[0], counts[1])
        self.assertEqual(counts[1], counts[2])

    def test_gradient_matches_manual_unroll_and_does_not_include_buffer(self):
        for iterations in (1, 2, 4):
            core, variables, x = _init_core(iterations=iterations, residual=True)

            def loss(params):
                return core.apply({'params': params, 'buffers': variables['buffers']}, x).representation.sum()

            def manual_loss(params):
                manual_variables = {'params': params, 'buffers': variables['buffers']}
                return _manual_unroll(manual_variables, x, iterations=iterations, residual=True).sum()

            actual_grad = jax.grad(loss)(variables['params'])
            expected_grad = jax.grad(manual_loss)(variables['params'])
            leaves = jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda a, b: jnp.max(jnp.abs(a - b)), actual_grad, expected_grad))
            self.assertLessEqual(float(max(leaves)), 1e-6)

    def test_k1_zero_state_decomposes_vanilla_mlp(self):
        input_dim = 5
        state_dim = 8
        x = jnp.arange(2 * input_dim, dtype=jnp.float32).reshape(2, input_dim) / 10.0
        vanilla = MLP(hidden_dims=(state_dim, state_dim, state_dim), activate_final=True)
        vanilla_vars = vanilla.init(jax.random.PRNGKey(10), x)
        core = make_computation_core(_spec(iterations=1, residual=False, state_dim=state_dim), hidden_dims=(state_dim,) * 3)
        variables = core.init({'params': jax.random.PRNGKey(11), 'buffers': jax.random.PRNGKey(12)}, x)
        variables = unfreeze(variables)
        variables['params']['topology']['input_mapping']['Dense_0'] = vanilla_vars['params']['Dense_0']
        variables['params']['topology']['update_module']['Dense_0'] = vanilla_vars['params']['Dense_1']
        variables['params']['topology']['update_module']['Dense_1'] = vanilla_vars['params']['Dense_2']
        variables['buffers']['topology']['z_init'] = jnp.zeros((state_dim,))
        variables = freeze(variables)
        actual = core.apply(variables, x).representation
        expected = vanilla.apply(vanilla_vars, x)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
