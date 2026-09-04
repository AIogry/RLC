"""Hard correctness tests for the M19A EntityMLP control."""

import copy
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze, unfreeze

from impls.agents import agent_configs, agents
from impls.computation.accounting import (
    count_parameters,
    gciql_architecture_accounting,
)
from impls.computation.blocks import EntityMLPBlock, EntityMLPStack, MLPMixerBlock
from impls.computation.factory import ComputationSpec, make_computation_core
from impls.main import _computation_slot_accounting


def _entity_spec(*, num_buttons=9, input_semantics='goal_pair', action_semantics='none'):
    return ComputationSpec.from_mapping({
        'primitive': 'mlp',
        'structure': 'puzzle_tokens',
        'block': 'entity_mlp',
        'topology': 'feedforward',
        'credit': 'direct',
        'input_semantics': input_semantics,
        'action_semantics': action_semantics,
        'structure_kwargs': {
            'num_buttons': num_buttons,
            'robot_dim': 19,
            'button_feature_dim': 4,
            'token_dim': 7,
            'robot_hidden_dim': 8,
            'index_embedding': True,
        },
        'block_kwargs': {
            'num_blocks': 2,
            'channel_hidden_dim': 11,
        },
        'readout': 'mean_context',
        'readout_kwargs': {'output_dim': 8},
    })


def _raw_goal_pair(num_buttons=9, action_dim=0, batch_size=3):
    observation_dim = 19 + 4 * num_buttons
    width = 2 * observation_dim + action_dim
    return jnp.arange(batch_size * width, dtype=jnp.float32).reshape(batch_size, width) / 101.0


def _entity_gciql_config(num_buttons=9):
    config = copy.deepcopy(agent_configs['gciql']())
    config.actor_hidden_dims = (8,)
    config.value_hidden_dims = (8,)
    config.batch_size = 3
    structure_kwargs = {
        'num_buttons': num_buttons,
        'robot_dim': 19,
        'button_feature_dim': 4,
        'token_dim': 7,
        'robot_hidden_dim': 8,
        'index_embedding': True,
    }
    for slot_name in ('actor', 'value', 'critic'):
        slot = config.compute[slot_name]
        slot.enabled = True
        slot.primitive = 'mlp'
        slot.structure = 'puzzle_tokens'
        slot.block = 'entity_mlp'
        slot.topology = 'feedforward'
        slot.credit = 'direct'
        slot.structure_kwargs = copy.deepcopy(structure_kwargs)
        slot.block_kwargs = {'num_blocks': 2, 'channel_hidden_dim': 11}
        slot.readout = 'mean_context'
        slot.readout_kwargs = {'output_dim': 8}
    return config


def _batch(num_buttons=9):
    observations = _raw_goal_pair(num_buttons=num_buttons, batch_size=3)[:, :19 + 4 * num_buttons]
    return {
        'observations': observations,
        'next_observations': observations + 0.01,
        'actions': jnp.ones((3, 2), dtype=jnp.float32),
        'value_goals': observations + 0.02,
        'actor_goals': observations + 0.03,
        'rewards': jnp.array([-1.0, 0.0, -1.0]),
        'masks': jnp.array([1.0, 0.0, 1.0]),
    }


def _tree_any_difference(left, right):
    return any(
        not np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right))
    )


class EntityMLPBlockTest(unittest.TestCase):
    def assertTreeAllClose(self, left, right, atol=2e-6):
        left_leaves = jax.tree_util.tree_leaves(left)
        right_leaves = jax.tree_util.tree_leaves(right)
        self.assertEqual(len(left_leaves), len(right_leaves))
        for actual, expected in zip(left_leaves, right_leaves):
            self.assertEqual(np.asarray(actual).shape, np.asarray(expected).shape)
            np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=0.0, atol=atol)

    def test_shapes_parameter_sharing_and_untied_depth(self):
        x = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4) / 23.0
        block = EntityMLPBlock(embed_dim=4, hidden_dim_channels=6)
        block_variables = block.init(jax.random.PRNGKey(1), x)
        self.assertEqual(block.apply(block_variables, x).shape, x.shape)
        self.assertEqual(block_variables['params']['channel_dense1']['kernel'].shape, (4, 6))
        self.assertEqual(block_variables['params']['channel_dense2']['kernel'].shape, (6, 4))
        self.assertFalse(any(
            'token' in str(key) or 'tm_weights' in str(key)
            for key in block_variables['params']
        ))

        stack = EntityMLPStack(num_blocks=2, embed_dim=4, hidden_dim_channels=6)
        stack_variables = stack.init(jax.random.PRNGKey(2), x)
        self.assertEqual(stack.apply(stack_variables, x).shape, x.shape)
        self.assertEqual(set(stack_variables['params']), {'blocks_0', 'blocks_1'})
        self.assertFalse(np.array_equal(
            np.asarray(stack_variables['params']['blocks_0']['channel_dense1']['kernel']),
            np.asarray(stack_variables['params']['blocks_1']['channel_dense1']['kernel']),
        ))
        seven_tokens = jnp.ones((2, 7, 4), dtype=jnp.float32)
        seven_variables = stack.init(jax.random.PRNGKey(3), seven_tokens)
        self.assertEqual(count_parameters(stack_variables['params']), count_parameters(seven_variables['params']))

    def test_cross_token_perturbation_and_jacobian_are_block_diagonal(self):
        stack = EntityMLPStack(num_blocks=2, embed_dim=4, hidden_dim_channels=6)
        x = jnp.arange(1 * 3 * 4, dtype=jnp.float32).reshape(1, 3, 4) / 19.0
        variables = stack.init(jax.random.PRNGKey(4), x)
        changed = x.at[:, 1, :].add(3.0)
        baseline = stack.apply(variables, x)
        perturbed = stack.apply(variables, changed)
        np.testing.assert_allclose(
            np.asarray(baseline[:, 0, :]), np.asarray(perturbed[:, 0, :]), rtol=0.0, atol=2e-6
        )
        np.testing.assert_allclose(
            np.asarray(baseline[:, 2, :]), np.asarray(perturbed[:, 2, :]), rtol=0.0, atol=2e-6
        )
        self.assertGreater(
            float(jnp.max(jnp.abs(baseline[:, 1, :] - perturbed[:, 1, :]))), 0.0
        )

        jacobian = jax.jacrev(lambda value: stack.apply(variables, value))(x)
        for output_token in range(3):
            for input_token in range(3):
                block_jacobian = np.asarray(jacobian[0, output_token, :, 0, input_token, :])
                if output_token == input_token:
                    self.assertGreater(float(np.max(np.abs(block_jacobian))), 0.0)
                else:
                    np.testing.assert_allclose(block_jacobian, 0.0, rtol=0.0, atol=2e-6)

    def test_mixer_positive_control_detects_cross_token_interaction(self):
        x = jnp.ones((1, 3, 2), dtype=jnp.float32)
        mixer = MLPMixerBlock(
            num_tokens=3,
            embed_dim=2,
            hidden_dim_tokens=2,
            hidden_dim_channels=3,
            tm_mode='none',
        )
        variables = unfreeze(mixer.init(jax.random.PRNGKey(5), x))
        params = variables['params']
        params['token_dense1']['kernel'] = jnp.ones_like(params['token_dense1']['kernel'])
        params['token_dense1']['bias'] = jnp.zeros_like(params['token_dense1']['bias'])
        params['token_dense2']['kernel'] = jnp.ones_like(params['token_dense2']['kernel'])
        params['token_dense2']['bias'] = jnp.zeros_like(params['token_dense2']['bias'])
        params['channel_dense1']['kernel'] = jnp.zeros_like(params['channel_dense1']['kernel'])
        params['channel_dense1']['bias'] = jnp.zeros_like(params['channel_dense1']['bias'])
        params['channel_dense2']['kernel'] = jnp.zeros_like(params['channel_dense2']['kernel'])
        params['channel_dense2']['bias'] = jnp.zeros_like(params['channel_dense2']['bias'])
        variables = freeze(variables)
        jacobian = jax.jacrev(lambda value: mixer.apply(variables, value))(x)
        cross_token = np.asarray(jacobian[0, 0, 0, 0, 1, 0])
        self.assertGreater(abs(float(cross_token)), 1e-5)

    def test_zero_token_branch_mixer_forward_and_channel_gradient_parity(self):
        x = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4) / 17.0
        mixer = MLPMixerBlock(
            num_tokens=3,
            embed_dim=4,
            hidden_dim_tokens=5,
            hidden_dim_channels=6,
            tm_mode='none',
        )
        entity = EntityMLPBlock(embed_dim=4, hidden_dim_channels=6)
        mixer_variables = unfreeze(mixer.init(jax.random.PRNGKey(6), x))
        entity_variables = unfreeze(entity.init(jax.random.PRNGKey(7), x))
        mixer_params = mixer_variables['params']
        for dense_name in ('token_dense1', 'token_dense2'):
            mixer_params[dense_name]['kernel'] = jnp.zeros_like(mixer_params[dense_name]['kernel'])
            mixer_params[dense_name]['bias'] = jnp.zeros_like(mixer_params[dense_name]['bias'])
        entity_params = entity_variables['params']
        for dense_name in ('channel_dense1', 'channel_dense2'):
            entity_params[dense_name] = mixer_params[dense_name]
        mixer_variables = freeze(mixer_variables)
        entity_variables = freeze(entity_variables)

        np.testing.assert_allclose(
            np.asarray(mixer.apply(mixer_variables, x)),
            np.asarray(entity.apply(entity_variables, x)),
            rtol=0.0,
            atol=2e-6,
        )
        mixer_grads = jax.grad(
            lambda params: jnp.sum(mixer.apply({'params': params}, x) ** 2)
        )(mixer_variables['params'])
        entity_grads = jax.grad(
            lambda params: jnp.sum(entity.apply({'params': params}, x) ** 2)
        )(entity_variables['params'])
        for dense_name in ('channel_dense1', 'channel_dense2'):
            self.assertTreeAllClose(mixer_grads[dense_name], entity_grads[dense_name])


class EntityMLPFactoryAndGCIQLTest(unittest.TestCase):
    def test_structured_body_batching_and_critic_action_context(self):
        actor_body = make_computation_core(
            _entity_spec(), hidden_dims=(8,), activate_final=True,
        )
        critic_body = make_computation_core(
            _entity_spec(action_semantics='robot_context'),
            hidden_dims=(8,), activate_final=True,
        )
        actor_input = _raw_goal_pair(batch_size=3)
        actor_variables = actor_body.init(jax.random.PRNGKey(8), actor_input)
        batched = actor_body.apply(actor_variables, actor_input)
        unbatched = actor_body.apply(actor_variables, actor_input[0])
        self.assertEqual(batched.representation.shape, (3, 8))
        self.assertEqual(unbatched.representation.shape, (8,))
        np.testing.assert_allclose(
            np.asarray(unbatched.representation), np.asarray(batched.representation[0]),
            rtol=0.0,
            # GPU Dense kernels may choose a batched versus unbatched GEMM
            # implementation. This is a public-boundary batching check, not
            # a bitwise parameter-transplant check.
            atol=1e-3,
        )

        critic_input = _raw_goal_pair(action_dim=2, batch_size=2)
        changed_action = critic_input.at[:, -2:].set(9.0)
        critic_variables = critic_body.init(jax.random.PRNGKey(9), critic_input)
        first = critic_body.apply(critic_variables, critic_input)
        second = critic_body.apply(critic_variables, changed_action)
        np.testing.assert_array_equal(
            np.asarray(first.auxiliary['computed_tokens']),
            np.asarray(second.auxiliary['computed_tokens']),
        )
        self.assertFalse(np.array_equal(
            np.asarray(first.representation), np.asarray(second.representation),
        ))

    def test_invalid_entity_variants_fail_fast(self):
        invalid = {
            'single_state': {'topology': 'single_state', 'topology_kwargs': {'iterations': 1}},
            'one_step': {'credit': 'one_step'},
            'mean_alias': {'readout': 'mean'},
            'token_kwargs': {'block_kwargs': {
                'num_blocks': 2, 'channel_hidden_dim': 11, 'token_hidden_dim': 5,
            }},
        }
        for name, overrides in invalid.items():
            with self.subTest(name=name):
                mapping = {
                    'primitive': 'mlp',
                    'structure': 'puzzle_tokens',
                    'block': 'entity_mlp',
                    'topology': 'feedforward',
                    'credit': 'direct',
                    'input_semantics': 'goal_pair',
                    'structure_kwargs': {
                        'num_buttons': 9, 'token_dim': 7, 'robot_hidden_dim': 8,
                    },
                    'block_kwargs': {'num_blocks': 2, 'channel_hidden_dim': 11},
                    'readout': 'mean_context',
                    'readout_kwargs': {'output_dim': 8},
                }
                mapping.update(overrides)
                spec = ComputationSpec.from_mapping(mapping)
                with self.assertRaises(ValueError):
                    make_computation_core(spec, hidden_dims=(8,), activate_final=True)

    def test_gciql_actor_value_critic_update_target_and_accounting(self):
        config = _entity_gciql_config()
        batch = _batch()
        agent = agents['gciql'].create(10, batch['observations'], batch['actions'], config)
        before_online = agent.network.params['modules_critic']
        before_target = agent.network.params['modules_target_critic']
        updated, info = agent.update(batch)
        self.assertTrue(all(
            np.all(np.isfinite(np.asarray(value))) for value in info.values()
        ))
        self.assertTrue(_tree_any_difference(before_online, updated.network.params['modules_critic']))
        self.assertTrue(_tree_any_difference(before_target, updated.network.params['modules_target_critic']))
        action = updated.sample_actions(
            batch['observations'][:1], batch['actor_goals'][:1], seed=jax.random.PRNGKey(11),
        )
        self.assertEqual(action.shape, (1, 2))
        report = _computation_slot_accounting(updated, config)
        self.assertEqual(set(report), {'actor', 'value', 'critic'})
        for slot_name, slot_report in report.items():
            self.assertEqual(slot_report['block_type'], 'entity_mlp')
            self.assertFalse(slot_report['token_interaction'])
            self.assertEqual(slot_report['token_mixing_params'], 0)
            self.assertEqual(slot_report['token_mixing_dense_macs'], 0)
            self.assertEqual(slot_report['block_depth_L'], 2)
            self.assertEqual(slot_report['iterations_K'], 1)
            self.assertEqual(slot_report['structured_sequential_depth'], 6)
            self.assertEqual(slot_report['buffer_elements'], 0)
            self.assertGreater(slot_report['channel_mixing_params'], 0, slot_name)
        architecture = gciql_architecture_accounting(updated.network.params, config, report)
        self.assertEqual(architecture['slots']['actor']['block'], 'entity_mlp')
        self.assertEqual(architecture['slots']['actor']['computation_body_sequential_depth'], 6)


if __name__ == '__main__':
    unittest.main()
