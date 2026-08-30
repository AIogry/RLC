"""Correctness gates for M17 modular structured recurrent computation."""

import copy
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze, unfreeze

from impls.agents import agent_configs, agents
from impls.computation.accounting import (
    count_non_trainable,
    count_parameters,
    modular_structured_body_accounting,
)
from impls.computation.factory import ComputationSpec, make_computation_core
from impls.computation.structured import PuzzleStructuredBody
from impls.main import _computation_runtime_extras
from impls.representation.interfaces import StructuredRepresentation
from impls.representation.puzzle import PuzzleTokenAdapter


def _structure_kwargs(num_buttons=9, depth=2):
    return {
        'num_buttons': num_buttons,
        'robot_dim': 19,
        'button_feature_dim': 4,
        'token_dim': 7,
        'robot_hidden_dim': 8,
        'token_mlp_hidden_dim': 5,
        'channel_mlp_hidden_dim': 11,
        'num_mixer_blocks': depth,
        'index_embedding': True,
        'readout': 'mean',
        'tm_mode': 'none',
    }


def _adapter_kwargs(num_buttons=9):
    kwargs = _structure_kwargs(num_buttons=num_buttons)
    return {
        key: kwargs[key]
        for key in (
            'num_buttons', 'robot_dim', 'button_feature_dim', 'token_dim',
            'robot_hidden_dim', 'index_embedding',
        )
    }


def _spec(depth=2, *, topology='feedforward', iterations=1, state_init='zero_buffer',
          input_semantics='goal_pair', action_semantics='none'):
    mapping = {
        'primitive': 'mlp',
        'structure': 'puzzle_tokens',
        'block': 'mlp_mixer',
        'topology': topology,
        'credit': 'direct',
        'parameter_sharing': 'shared',
        'input_semantics': input_semantics,
        'action_semantics': action_semantics,
        'structure_kwargs': _structure_kwargs(depth=depth),
    }
    if topology == 'single_state':
        mapping['topology_kwargs'] = {
            'iterations': iterations,
            'input_mapping': 'identity',
            'state_init': state_init,
            'state_init_std': 0.5,
            'input_injection': 'z_plus_x',
            'residual': False,
        }
    return ComputationSpec.from_mapping(mapping)


def _input(action_dim=0, batch_size=3):
    obs_dim = 19 + 9 * 4
    width = 2 * obs_dim + action_dim
    return jnp.arange(batch_size * width, dtype=jnp.float32).reshape(batch_size, width) / 97.0


def _copy_optional(source, target, name):
    if name in source:
        target[name] = source[name]


def _transplant_legacy_to_modular(legacy_params, modular_variables, depth):
    """Map legacy semantic components into the M17 ownership layout."""

    variables = unfreeze(modular_variables)
    params = variables['params']
    adapter = params['adapter']
    for name in ('button_projection', 'index_embedding', 'robot_projection', 'robot_layer_norm'):
        _copy_optional(legacy_params, adapter, name)
    stack = params['core']['topology']['primitive']
    for index in range(depth):
        stack[f'blocks_{index}'] = legacy_params[f'mixer_blocks_{index}']
    readout = params['readout']
    for name in ('fusion', 'fusion_layer_norm'):
        _copy_optional(legacy_params, readout, name)
    return freeze(variables)


def _transplant_ff_to_ss(ff_variables, ss_variables):
    """Use one shared FF block unit as the K=1 recurrent update block."""

    variables = unfreeze(ss_variables)
    source = ff_variables['params']
    target = variables['params']
    target['adapter'] = source['adapter']
    target['readout'] = source['readout']
    target['core']['topology']['update_module'] = source['core']['topology']['primitive']
    return freeze(variables)


def _tree_norm(tree):
    return sum(float(jnp.sum(jnp.abs(leaf))) for leaf in jax.tree_util.tree_leaves(tree))


class M17ModularStructuredTest(unittest.TestCase):
    def assertTreeAllClose(self, left, right, atol=2e-6):
        # Semantic transplant may cross dict/FrozenDict boundaries.  The
        # ownership mapping, leaf order, shapes, and values are the parity
        # contract; container implementation is not.
        actual_leaves = jax.tree_util.tree_leaves(left)
        expected_leaves = jax.tree_util.tree_leaves(right)
        self.assertEqual(len(actual_leaves), len(expected_leaves))
        for actual, expected in zip(actual_leaves, expected_leaves):
            self.assertEqual(np.asarray(actual).shape, np.asarray(expected).shape)
            np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=0.0, atol=atol)

    def test_structured_representation_is_parameter_free_and_adapter_owns_only_representation(self):
        self.assertEqual(
            StructuredRepresentation._fields,
            ('tokens', 'context', 'mask', 'auxiliary'),
        )
        adapter = PuzzleTokenAdapter(**_adapter_kwargs())
        variables = adapter.init(jax.random.PRNGKey(0), _input())
        params = variables['params']
        self.assertEqual(
            set(params),
            {'button_projection', 'index_embedding', 'robot_projection'},
        )
        self.assertNotIn('mixer_blocks_0', params)
        self.assertNotIn('fusion', params)
        output = adapter.apply(variables, _input())
        self.assertIsInstance(output, StructuredRepresentation)
        self.assertEqual(output.tokens.shape, (3, 9, 7))
        self.assertEqual(output.context.shape, (3, 8))

    def test_legacy_to_modular_ff_semantic_parameter_forward_and_gradient_parity(self):
        for action_semantics in ('none', 'robot_context'):
            with self.subTest(action_semantics=action_semantics):
                x = _input(action_dim=2 if action_semantics == 'robot_context' else 0)
                kwargs = _structure_kwargs(depth=2)
                legacy = PuzzleStructuredBody(
                    output_dim=8,
                    input_semantics='goal_pair',
                    action_semantics=action_semantics,
                    layer_norm=True,
                    activate_final=True,
                    **kwargs,
                )
                legacy_vars = legacy.init(jax.random.PRNGKey(1), x)
                modular = make_computation_core(
                    _spec(depth=2, action_semantics=action_semantics),
                    hidden_dims=(8,),
                    activate_final=True,
                    layer_norm=True,
                )
                modular_vars = modular.init(jax.random.PRNGKey(2), x)
                modular_vars = _transplant_legacy_to_modular(
                    legacy_vars['params'], modular_vars, depth=2,
                )
                self.assertEqual(
                    count_parameters(legacy_vars['params']),
                    count_parameters(modular_vars['params']),
                )
                expected = legacy.apply(legacy_vars, x)
                actual = modular.apply(modular_vars, x).representation
                np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=0.0, atol=2e-6)

                legacy_grads = jax.grad(
                    lambda p: jnp.sum(legacy.apply({'params': p}, x) ** 2)
                )(legacy_vars['params'])
                modular_grads = jax.grad(
                    lambda p: jnp.sum(modular.apply({'params': p}, x).representation ** 2)
                )(modular_vars['params'])
                self.assertTreeAllClose(
                    legacy_grads['button_projection'], modular_grads['adapter']['button_projection']
                )
                self.assertTreeAllClose(
                    legacy_grads['robot_projection'], modular_grads['adapter']['robot_projection']
                )
                self.assertTreeAllClose(
                    legacy_grads['index_embedding'], modular_grads['adapter']['index_embedding']
                )
                for index in range(2):
                    self.assertTreeAllClose(
                        legacy_grads[f'mixer_blocks_{index}'],
                        modular_grads['core']['topology']['primitive'][f'blocks_{index}'],
                    )
                self.assertTreeAllClose(legacy_grads['fusion'], modular_grads['readout']['fusion'])

    def test_batched_unbatched_parity_for_actor_value_and_critic_semantics(self):
        for role, action_semantics in (
            ('actor', 'none'), ('value', 'none'), ('critic', 'robot_context')
        ):
            with self.subTest(role=role):
                x = _input(action_dim=2 if action_semantics == 'robot_context' else 0)
                body = make_computation_core(
                    _spec(action_semantics=action_semantics), hidden_dims=(8,), activate_final=True
                )
                variables = body.init(jax.random.PRNGKey(3), x)
                batched = body.apply(variables, x).representation
                unbatched = body.apply(variables, x[0]).representation
                np.testing.assert_allclose(np.asarray(unbatched), np.asarray(batched[0]), rtol=0.0, atol=2e-6)

    def test_critic_action_enters_context_only(self):
        adapter = PuzzleTokenAdapter(
            **_adapter_kwargs(), input_semantics='goal_pair', action_semantics='robot_context'
        )
        state_goal = _input(action_dim=0, batch_size=2)
        first = jnp.concatenate([state_goal, jnp.zeros((2, 2), dtype=jnp.float32)], axis=-1)
        second = first.at[:, -2:].set(7.0)
        variables = adapter.init(jax.random.PRNGKey(4), first)
        left = adapter.apply(variables, first)
        right = adapter.apply(variables, second)
        np.testing.assert_array_equal(np.asarray(left.tokens), np.asarray(right.tokens))
        self.assertFalse(np.array_equal(np.asarray(left.context), np.asarray(right.context)))

    def test_ff_equals_ss_k1_for_l1_l2_l4_forward_gradient_params_macs_and_depth(self):
        x = _input()
        for depth in (1, 2, 4):
            with self.subTest(depth=depth):
                ff = make_computation_core(_spec(depth), hidden_dims=(8,), activate_final=True)
                ss = make_computation_core(
                    _spec(depth, topology='single_state', iterations=1),
                    hidden_dims=(8,), activate_final=True,
                )
                ff_vars = ff.init(jax.random.PRNGKey(10 + depth), x)
                ss_vars = ss.init(
                    {'params': jax.random.PRNGKey(20 + depth), 'buffers': jax.random.PRNGKey(30 + depth)},
                    x,
                )
                ss_vars = _transplant_ff_to_ss(ff_vars, ss_vars)
                ff_output = ff.apply(ff_vars, x)
                ss_output = ss.apply(ss_vars, x)
                np.testing.assert_allclose(
                    np.asarray(ff_output.auxiliary['computed_tokens']),
                    np.asarray(ss_output.auxiliary['computed_tokens']),
                    rtol=0.0,
                    atol=2e-6,
                )
                np.testing.assert_allclose(
                    np.asarray(ff_output.representation), np.asarray(ss_output.representation),
                    rtol=0.0, atol=2e-6,
                )
                self.assertEqual(count_parameters(ff_vars['params']), count_parameters(ss_vars['params']))
                self.assertEqual(count_non_trainable(ff_vars.get('buffers', {})), 0)
                self.assertEqual(count_non_trainable(ss_vars['buffers']), 7)

                ff_grads = jax.grad(
                    lambda p: jnp.sum(ff.apply({'params': p}, x).representation ** 2)
                )(ff_vars['params'])
                ss_grads = jax.grad(
                    lambda p: jnp.sum(
                        ss.apply({'params': p, 'buffers': ss_vars['buffers']}, x).representation ** 2
                    )
                )(ss_vars['params'])
                self.assertTreeAllClose(ff_grads['adapter'], ss_grads['adapter'])
                self.assertTreeAllClose(ff_grads['readout'], ss_grads['readout'])
                self.assertTreeAllClose(
                    ff_grads['core']['topology']['primitive'],
                    ss_grads['core']['topology']['update_module'],
                )
                ff_report = modular_structured_body_accounting(
                    ff_vars['params'], _structure_kwargs(depth=depth), topology='feedforward'
                )
                ss_report = modular_structured_body_accounting(
                    ss_vars['params'], _structure_kwargs(depth=depth), topology='single_state',
                    topology_kwargs={'iterations': 1},
                )
                for field in ('mixer_dense_macs_per_execution', 'executed_mixer_dense_macs',
                              'executed_mixer_layers', 'executed_sequential_depth'):
                    self.assertEqual(ff_report[field], ss_report[field])
                self.assertEqual(ff_report['block_depth_L'], depth)
                self.assertEqual(ss_report['iterations_K'], 1)

    def test_k_invariance_state_initialization_and_direct_recurrent_gradient(self):
        x = _input()
        specs = {
            k: make_computation_core(
                _spec(2, topology='single_state', iterations=k), hidden_dims=(8,), activate_final=True
            )
            for k in (1, 2, 4)
        }
        variables = {
            k: core.init(
                {'params': jax.random.PRNGKey(40), 'buffers': jax.random.PRNGKey(41)}, x
            )
            for k, core in specs.items()
        }
        counts = [count_parameters(variables[k]['params']) for k in (1, 2, 4)]
        self.assertEqual(counts, [counts[0]] * 3)
        for k in (1, 2, 4):
            topology_params = variables[k]['params']['core']['topology']
            self.assertIn('update_module', topology_params)
            self.assertFalse(any(str(key).startswith('update_block_') for key in topology_params))
            self.assertFalse(any(str(key).startswith('update_modules_') for key in topology_params))
        reports = {
            k: modular_structured_body_accounting(
                variables[k]['params'], _structure_kwargs(depth=2), topology='single_state',
                topology_kwargs={'iterations': k},
            )
            for k in (1, 2, 4)
        }
        self.assertEqual(reports[1]['executed_mixer_dense_macs'] * 2, reports[2]['executed_mixer_dense_macs'])
        self.assertEqual(reports[1]['executed_mixer_dense_macs'] * 4, reports[4]['executed_mixer_dense_macs'])
        self.assertEqual(reports[1]['unique_mixer_layers'], reports[4]['unique_mixer_layers'])

        zero = variables[4]
        z_init = zero['buffers']['core']['topology']['z_init']
        np.testing.assert_array_equal(np.asarray(z_init), np.zeros((7,), dtype=np.float32))
        first_output = specs[4].apply(zero, x)
        second_output = specs[4].apply(zero, x)
        np.testing.assert_array_equal(
            np.asarray(first_output.representation), np.asarray(second_output.representation)
        )
        self.assertEqual(first_output.state.shape, (3, 9, 7))
        self.assertNotIn('z_init', zero['params']['core']['topology'])

        normal = make_computation_core(
            _spec(2, topology='single_state', iterations=2, state_init='normal_buffer'),
            hidden_dims=(8,), activate_final=True,
        )
        normal_vars = normal.init(
            {'params': jax.random.PRNGKey(42), 'buffers': jax.random.PRNGKey(43)}, x
        )
        normal_z = normal_vars['buffers']['core']['topology']['z_init']
        self.assertEqual(normal_z.shape, (7,))
        self.assertTrue(np.all(np.isfinite(np.asarray(normal_z))))
        self.assertNotIn('z_init', normal_vars['params']['core']['topology'])
        normal_first = normal.apply(normal_vars, x)
        normal_second = normal.apply(normal_vars, x)
        self.assertEqual(normal_first.state.shape, (3, 9, 7))
        np.testing.assert_array_equal(
            np.asarray(normal_first.representation), np.asarray(normal_second.representation)
        )

        def loss(params):
            output = specs[4].apply({'params': params, 'buffers': zero['buffers']}, x)
            return jnp.mean(output.representation ** 2)

        gradients = jax.grad(loss)(zero['params'])
        self.assertTrue(all(
            np.all(np.isfinite(np.asarray(leaf)))
            for leaf in jax.tree_util.tree_leaves(gradients)
        ))
        self.assertGreater(_tree_norm(gradients['core']['topology']['update_module']), 0.0)
        self.assertGreater(_tree_norm(gradients['adapter']), 0.0)
        self.assertGreater(_tree_norm(gradients['readout']), 0.0)

    def test_gciql_actor_value_critic_initialize_update_and_action_for_ff_and_ss(self):
        observations = _input(batch_size=3)[:, :55]
        batch = {
            'observations': observations,
            'next_observations': observations + 0.01,
            'actions': jnp.ones((3, 2), dtype=jnp.float32),
            'value_goals': observations + 0.02,
            'actor_goals': observations + 0.03,
            'rewards': jnp.array([-1.0, 0.0, -1.0]),
            'masks': jnp.array([1.0, 0.0, 1.0]),
        }
        for topology in ('feedforward', 'single_state'):
            with self.subTest(topology=topology):
                config = copy.deepcopy(agent_configs['gciql']())
                config.actor_hidden_dims = (8,)
                config.value_hidden_dims = (8,)
                config.batch_size = 3
                for slot_name in ('actor', 'value', 'critic'):
                    slot = config.compute[slot_name]
                    slot.enabled = True
                    slot.primitive = 'mlp'
                    slot.structure = 'puzzle_tokens'
                    slot.block = 'mlp_mixer'
                    slot.topology = topology
                    slot.credit = 'direct'
                    slot.parameter_sharing = 'shared'
                    slot.structure_kwargs = copy.deepcopy(_structure_kwargs())
                    if topology == 'single_state':
                        slot.topology_kwargs = {
                            'iterations': 2,
                            'input_mapping': 'identity',
                            'state_init': 'zero_buffer',
                            'input_injection': 'z_plus_x',
                            'residual': False,
                        }
                runtime = _computation_runtime_extras(config)
                if topology == 'feedforward':
                    record = runtime['feedforward']['actor']
                    self.assertEqual(record['block_depth_L'], 2)
                    self.assertEqual(record['iterations_K'], 1)
                    self.assertEqual(record['readout'], 'mean')
                else:
                    record = runtime['single_state']['actor']
                    self.assertEqual(record['block_depth_L'], 2)
                    self.assertEqual(record['iterations_K'], 2)
                    self.assertEqual(record['input_mapping'], 'identity')
                    self.assertEqual(record['state_init'], 'zero_buffer')
                agent = agents['gciql'].create(50, observations, batch['actions'], config)
                updated, info = agent.update(batch)
                self.assertTrue(all(
                    np.all(np.isfinite(np.asarray(value))) for value in info.values()
                ))
                action = updated.sample_actions(
                    observations[:1], batch['actor_goals'][:1], seed=jax.random.PRNGKey(51)
                )
                self.assertEqual(action.shape, (1, 2))
                if topology == 'single_state':
                    before = agent.network.params['modules_actor']['actor_net']['core']['topology']['update_module']
                    after = updated.network.params['modules_actor']['actor_net']['core']['topology']['update_module']
                    differences = [
                        float(jnp.max(jnp.abs(left - right)))
                        for left, right in zip(
                            jax.tree_util.tree_leaves(before),
                            jax.tree_util.tree_leaves(after),
                        )
                    ]
                    self.assertGreater(max(differences), 0.0)


if __name__ == '__main__':
    unittest.main()
