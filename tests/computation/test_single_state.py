import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze, unfreeze

from impls.computation.accounting import count_parameters
from impls.computation.factory import ComputationSpec, make_computation_core
from impls.computation.primitives.mlp import MLP


def _spec(
    iterations=1,
    residual=False,
    state_dim=8,
    state_init='normal_buffer',
    state_init_std=1.0,
    update_depth=2,
    layer_norm=False,
    update_activate_final=True,
):
    return ComputationSpec.from_mapping({
        'primitive': 'mlp',
        'topology': 'single_state',
        'credit': 'direct',
        'topology_kwargs': {
            'iterations': iterations,
            'residual': residual,
            'input_injection': 'z_plus_x',
            'state_dim': state_dim,
            'state_init': state_init,
            'state_init_std': state_init_std,
            'update_depth': update_depth,
            'layer_norm': layer_norm,
            'update_activate_final': update_activate_final,
        },
    })


def _init_core(
    iterations=1,
    residual=False,
    input_dim=5,
    state_dim=8,
    seed=0,
    buffer_seed=1,
    state_init='normal_buffer',
    state_init_std=1.0,
    update_depth=2,
    layer_norm=False,
    update_activate_final=True,
):
    core = make_computation_core(
        _spec(
            iterations, residual, state_dim, state_init, state_init_std,
            update_depth, layer_norm, update_activate_final,
        ),
        hidden_dims=(state_dim, state_dim, state_dim),
    )
    x = jnp.arange(2 * input_dim, dtype=jnp.float32).reshape(2, input_dim) / 10.0
    variables = core.init(
        {'params': jax.random.PRNGKey(seed), 'buffers': jax.random.PRNGKey(buffer_seed)},
        x,
    )
    return core, variables, x


def _manual_unroll(
    variables, x, iterations, residual, state_dim=8, update_depth=2,
    layer_norm=False, update_activate_final=True,
):
    topology = variables['params']['topology']
    x_hidden = MLP(hidden_dims=(state_dim,), activate_final=True).apply(
        {'params': topology['input_mapping']}, x,
    )
    z = jnp.broadcast_to(variables['buffers']['topology']['z_init'], x_hidden.shape)
    update_params = topology['update_module']
    for _ in range(iterations):
        update = MLP(
            hidden_dims=(state_dim,) * update_depth,
            activate_final=update_activate_final,
            layer_norm=layer_norm,
        ).apply(
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
            expected = _manual_unroll(variables, x, iterations=2, residual=residual, state_dim=8)
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)

    def test_one_shared_update_module_for_k1_k2_k4(self):
        counts = []
        for iterations in (1, 2, 3, 4, 5, 8, 11, 15):
            _, variables, _ = _init_core(iterations=iterations)
            params = variables['params']['topology']['update_module']
            counts.append(sum(np.asarray(value).size for value in jax.tree_util.tree_leaves(params)))
            self.assertEqual(set(params.keys()), {'Dense_0', 'Dense_1'})
        self.assertTrue(all(count == counts[0] for count in counts))

    def test_positive_integer_validation(self):
        for iterations in (0, -1, 1.5, True):
            with self.subTest(iterations=iterations):
                with self.assertRaises(ValueError):
                    _init_core(iterations=iterations)

    def test_generalized_update_depth_three(self):
        core, variables, x = _init_core(update_depth=3)
        update_params = variables['params']['topology']['update_module']
        self.assertEqual(set(update_params), {'Dense_0', 'Dense_1', 'Dense_2'})
        output = core.apply(variables, x).representation
        self.assertEqual(output.shape, (2, 8))
        self.assertTrue(np.all(np.isfinite(np.asarray(output))))

        def loss(params):
            return core.apply({'params': params, 'buffers': variables['buffers']}, x).representation.sum()

        gradients = jax.grad(loss)(variables['params'])
        self.assertTrue(all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree_util.tree_leaves(gradients)))

    def test_update_depth_validation(self):
        for update_depth in (0, -1, 1.5, True):
            with self.subTest(update_depth=update_depth):
                with self.assertRaises(ValueError):
                    _init_core(update_depth=update_depth)

    def test_zero_buffer_is_zero_non_trainable_deterministic_and_immutable(self):
        core, variables_a, x = _init_core(state_init='zero_buffer', state_init_std=0.0, buffer_seed=5)
        _, variables_b, _ = _init_core(state_init='zero_buffer', state_init_std=-100.0, buffer_seed=6)
        z_a = variables_a['buffers']['topology']['z_init']
        z_b = variables_b['buffers']['topology']['z_init']
        np.testing.assert_array_equal(np.asarray(z_a), np.zeros((8,), dtype=np.float32))
        np.testing.assert_array_equal(z_a, z_b)
        self.assertNotIn('z_init', variables_a['params']['topology'])
        before = np.array(z_a, copy=True)
        output = core.apply(variables_a, x)
        self.assertTrue(np.all(np.isfinite(np.asarray(output.representation))))
        np.testing.assert_array_equal(np.asarray(variables_a['buffers']['topology']['z_init']), before)

    def test_zero_buffer_k1_parameter_and_gradient_parity(self):
        input_dim = 5
        state_dim = 8
        x = jnp.arange(2 * input_dim, dtype=jnp.float32).reshape(2, input_dim) / 10.0
        vanilla = MLP(hidden_dims=(state_dim, state_dim, state_dim), activate_final=True)
        vanilla_vars = vanilla.init(jax.random.PRNGKey(10), x)
        core = make_computation_core(
            _spec(iterations=1, residual=False, state_dim=state_dim, state_init='zero_buffer'),
            hidden_dims=(state_dim,) * 3,
        )
        variables = core.init(
            {'params': jax.random.PRNGKey(11), 'buffers': jax.random.PRNGKey(12)}, x,
        )

        def mapped_params(vanilla_params):
            params = unfreeze(variables['params'])
            params['topology']['input_mapping']['Dense_0'] = vanilla_params['Dense_0']
            params['topology']['update_module']['Dense_0'] = vanilla_params['Dense_1']
            params['topology']['update_module']['Dense_1'] = vanilla_params['Dense_2']
            return freeze(params)

        def actual_loss(vanilla_params):
            return core.apply(
                {'params': mapped_params(vanilla_params), 'buffers': variables['buffers']}, x,
            ).representation.sum()

        def expected_loss(vanilla_params):
            return vanilla.apply({'params': vanilla_params}, x).sum()

        np.testing.assert_allclose(
            np.asarray(actual_loss(vanilla_vars['params'])),
            np.asarray(expected_loss(vanilla_vars['params'])),
            rtol=0.0,
            atol=1e-6,
        )
        actual_grad = jax.grad(actual_loss)(vanilla_vars['params'])
        expected_grad = jax.grad(expected_loss)(vanilla_vars['params'])
        for actual, expected in zip(
            jax.tree_util.tree_leaves(actual_grad),
            jax.tree_util.tree_leaves(expected_grad),
        ):
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)
        self.assertEqual(
            count_parameters(variables['params']['topology']),
            count_parameters(vanilla_vars['params']),
        )

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

    def test_k15_gradient_is_finite(self):
        core, variables, x = _init_core(iterations=15, residual=False)

        def loss(params):
            output = core.apply({'params': params, 'buffers': variables['buffers']}, x)
            return jnp.mean(output.representation ** 2)

        gradients = jax.grad(loss)(variables['params'])
        self.assertTrue(all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree_util.tree_leaves(gradients)))

    def test_k1_zero_state_decomposes_vanilla_mlp(self):
        input_dim = 5
        state_dim = 8
        x = jnp.arange(2 * input_dim, dtype=jnp.float32).reshape(2, input_dim) / 10.0
        vanilla = MLP(hidden_dims=(state_dim, state_dim, state_dim), activate_final=True)
        vanilla_vars = vanilla.init(jax.random.PRNGKey(10), x)
        core = make_computation_core(
            _spec(iterations=1, residual=False, state_dim=state_dim, state_init='zero_buffer'),
            hidden_dims=(state_dim,) * 3,
        )
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

    def test_critic_k1_zero_state_vanilla_primitive_parity(self):
        input_dim = 5
        state_dim = 8
        x = jnp.arange(2 * input_dim, dtype=jnp.float32).reshape(2, input_dim) / 10.0
        vanilla = MLP(
            hidden_dims=(state_dim,) * 4,
            activate_final=False,
            layer_norm=True,
        )
        vanilla_vars = vanilla.init(jax.random.PRNGKey(20), x)
        core = make_computation_core(
            _spec(
                iterations=1,
                residual=False,
                state_dim=state_dim,
                state_init='zero_buffer',
                update_depth=3,
                layer_norm=True,
                update_activate_final=False,
            ),
            hidden_dims=(state_dim,) * 4,
            activate_final=False,
            layer_norm=True,
        )
        variables = unfreeze(core.init(
            {'params': jax.random.PRNGKey(21), 'buffers': jax.random.PRNGKey(22)}, x,
        ))
        vanilla_params = vanilla_vars['params']
        topology = variables['params']['topology']
        topology['input_mapping'] = {
            'Dense_0': vanilla_params['Dense_0'],
            'LayerNorm_0': vanilla_params['LayerNorm_0'],
        }
        topology['update_module'] = {
            f'Dense_{index}': vanilla_params[f'Dense_{index + 1}']
            for index in range(3)
        }
        topology['update_module'].update({
            f'LayerNorm_{index}': vanilla_params[f'LayerNorm_{index + 1}']
            for index in range(2)
        })
        variables['buffers']['topology']['z_init'] = jnp.zeros((state_dim,))
        variables = freeze(variables)

        self.assertEqual(
            set(topology['input_mapping']), {'Dense_0', 'LayerNorm_0'},
        )
        self.assertEqual(
            set(topology['update_module']),
            {'Dense_0', 'Dense_1', 'Dense_2', 'LayerNorm_0', 'LayerNorm_1'},
        )

        def actual_loss(vanilla_params):
            mapped = unfreeze(variables)
            mapped['params']['topology']['input_mapping']['Dense_0'] = vanilla_params['Dense_0']
            mapped['params']['topology']['input_mapping']['LayerNorm_0'] = vanilla_params['LayerNorm_0']
            mapped['params']['topology']['update_module'] = {
                **{
                    f'Dense_{index}': vanilla_params[f'Dense_{index + 1}']
                    for index in range(3)
                },
                **{
                    f'LayerNorm_{index}': vanilla_params[f'LayerNorm_{index + 1}']
                    for index in range(2)
                },
            }
            return core.apply(freeze(mapped), x).representation.sum()

        def expected_loss(vanilla_params):
            return vanilla.apply({'params': vanilla_params}, x).sum()

        actual = core.apply(variables, x).representation
        expected = vanilla.apply(vanilla_vars, x)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)
        self.assertEqual(count_parameters(variables['params']['topology']), count_parameters(vanilla_params))
        actual_grad = jax.grad(actual_loss)(vanilla_params)
        expected_grad = jax.grad(expected_loss)(vanilla_params)
        for actual_leaf, expected_leaf in zip(
            jax.tree_util.tree_leaves(actual_grad),
            jax.tree_util.tree_leaves(expected_grad),
        ):
            np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=0.0, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
