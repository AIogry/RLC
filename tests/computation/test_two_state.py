import unittest

import jax
import jax.numpy as jnp
import numpy as np

from impls.computation.factory import ComputationSpec, make_computation_core
from impls.computation.accounting import actor_slot_accounting
from impls.computation.primitives.mlp import MLP
from impls.computation.topologies.two_state import execution_trace


STATE_DIM = 8
INPUT_DIM = 5


def _spec(h_cycles=2, l_cycles=1, credit='full_bptt'):
    return ComputationSpec.from_mapping({
        'primitive': 'mlp',
        'topology': 'two_state',
        'credit': credit,
        'topology_kwargs': {
            'h_cycles': h_cycles,
            'l_cycles': l_cycles,
            'input_injection': 'l_receives_x',
            'state_dim': STATE_DIM,
            'state_init': 'normal_buffer',
            'state_init_std': 1.0,
        },
    })


def _init_core(h_cycles=2, l_cycles=1, credit='full_bptt', seed=0, buffer_seed=1):
    core = make_computation_core(
        _spec(h_cycles, l_cycles, credit),
        hidden_dims=(STATE_DIM, STATE_DIM, STATE_DIM),
    )
    x = jnp.arange(2 * INPUT_DIM, dtype=jnp.float32).reshape(2, INPUT_DIM) / 10.0
    variables = core.init(
        {'params': jax.random.PRNGKey(seed), 'buffers': jax.random.PRNGKey(buffer_seed)},
        x,
    )
    return core, variables, x


def _manual_unroll(variables, x, h_cycles, l_cycles, one_step=False):
    topology = variables['params']['topology']
    input_mapping = MLP(hidden_dims=(STATE_DIM,), activate_final=True)
    update = MLP(hidden_dims=(STATE_DIM, STATE_DIM), activate_final=True)
    x_hidden = input_mapping.apply({'params': topology['input_mapping']}, x)
    z_h = jnp.broadcast_to(variables['buffers']['topology']['z_h_init'], x_hidden.shape)
    z_l = jnp.broadcast_to(variables['buffers']['topology']['z_l_init'], x_hidden.shape)
    trace = execution_trace(h_cycles, l_cycles)
    for index, level in enumerate(trace):
        if one_step and index == len(trace) - 2:
            z_h = jax.lax.stop_gradient(z_h)
            z_l = jax.lax.stop_gradient(z_l)
        if level == 'L':
            z_l = update.apply({'params': topology['l_update']}, z_l + z_h + x_hidden)
        else:
            z_h = update.apply({'params': topology['h_update']}, z_h + z_l)
    return z_h, z_l


def _max_tree_difference(first, second):
    differences = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda a, b: jnp.max(jnp.abs(a - b)), first, second)
    )
    return float(max(differences))


class TwoStateTopologyTest(unittest.TestCase):
    def test_schedule_trace_and_counts(self):
        self.assertEqual(execution_trace(2, 1), ('L', 'H', 'L', 'H'))
        self.assertEqual(execution_trace(2, 6), ('L',) * 6 + ('H',) + ('L',) * 6 + ('H',))
        for h_cycles, l_cycles in ((2, 1), (2, 6)):
            trace = execution_trace(h_cycles, l_cycles)
            self.assertEqual(trace.count('L'), h_cycles * l_cycles)
            self.assertEqual(trace.count('H'), h_cycles)
            self.assertEqual(len(trace), h_cycles * (l_cycles + 1))

    def test_accounting_matches_execution_trace(self):
        for h_cycles, l_cycles in ((2, 1), (2, 6)):
            _, variables, _ = _init_core(h_cycles, l_cycles)
            report = actor_slot_accounting(
                {'actor_net': {'topology': variables['params']['topology']}},
                {'actor_net': {'topology': variables['buffers']['topology']}},
                topology='two_state',
                iterations=(h_cycles, l_cycles),
            )
            self.assertEqual(report['h_update_executions'], h_cycles)
            self.assertEqual(report['l_update_executions'], h_cycles * l_cycles)
            self.assertEqual(report['total_update_executions'], len(execution_trace(h_cycles, l_cycles)))
            self.assertEqual(report['update_executions'], len(execution_trace(h_cycles, l_cycles)))

    def test_dimensions_buffers_and_decision_local_reset(self):
        core, variables, x = _init_core(h_cycles=2, l_cycles=6)
        self.assertIn('buffers', variables)
        self.assertNotIn('z_h_init', variables['params']['topology'])
        self.assertNotIn('z_l_init', variables['params']['topology'])
        buffers = variables['buffers']['topology']
        self.assertEqual(buffers['z_h_init'].shape, (STATE_DIM,))
        self.assertEqual(buffers['z_l_init'].shape, (STATE_DIM,))
        self.assertFalse(np.array_equal(buffers['z_h_init'], buffers['z_l_init']))
        before_h = np.array(buffers['z_h_init'])
        before_l = np.array(buffers['z_l_init'])
        first = core.apply(variables, x)
        second = core.apply(variables, x)
        self.assertEqual(first.representation.shape, (2, STATE_DIM))
        self.assertEqual(first.state['z_h'].shape, (2, STATE_DIM))
        self.assertEqual(first.state['z_l'].shape, (2, STATE_DIM))
        np.testing.assert_array_equal(before_h, variables['buffers']['topology']['z_h_init'])
        np.testing.assert_array_equal(before_l, variables['buffers']['topology']['z_l_init'])
        np.testing.assert_array_equal(first.representation, second.representation)

    def test_buffer_seed_reproducibility_and_independence(self):
        _, variables_a, _ = _init_core(seed=3, buffer_seed=4)
        _, variables_b, _ = _init_core(seed=3, buffer_seed=4)
        _, variables_c, _ = _init_core(seed=3, buffer_seed=5)
        for name in ('z_h_init', 'z_l_init'):
            np.testing.assert_array_equal(
                variables_a['buffers']['topology'][name],
                variables_b['buffers']['topology'][name],
            )
        self.assertFalse(np.array_equal(
            variables_a['buffers']['topology']['z_h_init'],
            variables_c['buffers']['topology']['z_h_init'],
        ))
        self.assertFalse(np.array_equal(
            variables_a['buffers']['topology']['z_l_init'],
            variables_c['buffers']['topology']['z_l_init'],
        ))

    def test_equations_and_final_representation(self):
        for h_cycles, l_cycles in ((2, 1), (2, 6)):
            for credit in ('full_bptt', 'one_step'):
                core, variables, x = _init_core(h_cycles, l_cycles, credit)
                actual = core.apply(variables, x)
                expected_h, expected_l = _manual_unroll(
                    variables, x, h_cycles, l_cycles, one_step=credit == 'one_step'
                )
                np.testing.assert_allclose(actual.representation, expected_h, rtol=0.0, atol=1e-6)
                np.testing.assert_allclose(actual.state['z_l'], expected_l, rtol=0.0, atol=1e-6)

    def test_h_l_parameter_subtrees_are_independent_and_shared_by_schedule(self):
        parameter_counts = {}
        for h_cycles, l_cycles in ((2, 1), (2, 6)):
            for credit in ('full_bptt', 'one_step'):
                _, variables, _ = _init_core(h_cycles, l_cycles, credit)
                topology = variables['params']['topology']
                self.assertEqual(set(topology), {'input_mapping', 'h_update', 'l_update'})
                self.assertGreater(_max_tree_difference(topology['h_update'], topology['l_update']), 0.0)
                parameter_counts[(h_cycles, l_cycles, credit)] = sum(
                    np.asarray(value).size for value in jax.tree_util.tree_leaves(topology)
                )
        self.assertEqual(parameter_counts[(2, 1, 'full_bptt')], parameter_counts[(2, 6, 'full_bptt')])
        self.assertEqual(parameter_counts[(2, 1, 'full_bptt')], parameter_counts[(2, 1, 'one_step')])

    def test_full_bptt_gradients_match_manual_reference(self):
        for h_cycles, l_cycles in ((2, 1), (2, 6)):
            core, variables, x = _init_core(h_cycles, l_cycles, 'full_bptt')

            def production_loss(params, input_x):
                return core.apply({'params': params, 'buffers': variables['buffers']}, input_x).representation.sum()

            def manual_loss(params, input_x):
                manual_variables = {'params': params, 'buffers': variables['buffers']}
                return _manual_unroll(manual_variables, input_x, h_cycles, l_cycles)[0].sum()

            actual_params, actual_x = jax.grad(production_loss, argnums=(0, 1))(variables['params'], x)
            expected_params, expected_x = jax.grad(manual_loss, argnums=(0, 1))(variables['params'], x)
            self.assertLessEqual(_max_tree_difference(actual_params, expected_params), 1e-6)
            np.testing.assert_allclose(actual_x, expected_x, rtol=0.0, atol=1e-6)

    def test_one_step_gradients_match_manual_reference(self):
        h_cycles, l_cycles = 2, 6
        core, variables, x = _init_core(h_cycles, l_cycles, 'one_step')

        def production_loss(params, input_x):
            return core.apply({'params': params, 'buffers': variables['buffers']}, input_x).representation.sum()

        def manual_loss(params, input_x):
            manual_variables = {'params': params, 'buffers': variables['buffers']}
            return _manual_unroll(manual_variables, input_x, h_cycles, l_cycles, one_step=True)[0].sum()

        actual_params, actual_x = jax.grad(production_loss, argnums=(0, 1))(variables['params'], x)
        expected_params, expected_x = jax.grad(manual_loss, argnums=(0, 1))(variables['params'], x)
        self.assertLessEqual(_max_tree_difference(actual_params, expected_params), 1e-6)
        np.testing.assert_allclose(actual_x, expected_x, rtol=0.0, atol=1e-6)

    def test_credit_changes_backward_only(self):
        _, full_variables, x = _init_core(2, 6, 'full_bptt')
        core_full, _, _ = _init_core(2, 6, 'full_bptt')
        core_one, _, _ = _init_core(2, 6, 'one_step')
        one_variables = {'params': full_variables['params'], 'buffers': full_variables['buffers']}

        full_output = core_full.apply(one_variables, x).representation
        one_output = core_one.apply(one_variables, x).representation
        np.testing.assert_array_equal(full_output, one_output)

        def loss(core, params):
            return core.apply({'params': params, 'buffers': full_variables['buffers']}, x).representation.sum()

        full_grad = jax.grad(lambda params: loss(core_full, params))(full_variables['params'])
        one_grad = jax.grad(lambda params: loss(core_one, params))(full_variables['params'])
        self.assertGreater(_max_tree_difference(full_grad, one_grad), 1e-6)


if __name__ == '__main__':
    unittest.main()
