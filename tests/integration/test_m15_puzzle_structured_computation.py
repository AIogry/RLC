"""M15 Puzzle structured-token computation tests."""

import copy
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from impls.agents import agent_configs, agents
from impls.computation.accounting import structured_body_accounting
from impls.computation.blocks import MLPMixerBlock
from impls.computation.factory import ComputationSpec, make_computation_core
from impls.computation.structured import PuzzleStructuredBody
from impls.main import _computation_slot_accounting
from impls.representation.puzzle import parse_puzzle_observation


PUZZLE_SIZES = ((3, 3, 9, 55), (4, 4, 16, 83), (4, 5, 20, 99), (4, 6, 24, 115))


def _structure_kwargs(num_buttons, **overrides):
    result = {
        'num_buttons': num_buttons,
        'robot_dim': 19,
        'button_feature_dim': 4,
        'token_dim': 7,
        'robot_hidden_dim': 8,
        'token_mlp_hidden_dim': 5,
        'channel_mlp_hidden_dim': 11,
        'num_mixer_blocks': 2,
        'index_embedding': True,
        'readout': 'mean',
        'tm_mode': 'none',
    }
    result.update(overrides)
    return result


def _spec(num_buttons, *, input_semantics='goal_pair', action_semantics='none', **kwargs):
    return ComputationSpec.from_mapping({
        'structure': 'puzzle_tokens',
        'topology': 'feedforward',
        'credit': 'direct',
        'block': 'mlp_mixer',
        'input_semantics': input_semantics,
        'action_semantics': action_semantics,
        'structure_kwargs': _structure_kwargs(num_buttons, **kwargs),
    })


def _small_config(name):
    config = copy.deepcopy(agent_configs[name]())
    config.actor_hidden_dims = (8,)
    if 'value_hidden_dims' in config:
        config.value_hidden_dims = (8,)
    config.batch_size = 3
    if name == 'qrl':
        config.latent_dim = 6
        config.dim_per_component = 3
    return config


def _enable_structured(config, slots, num_buttons=9):
    kwargs = _structure_kwargs(num_buttons)
    for slot_name in slots:
        slot = config.compute[slot_name]
        slot.enabled = True
        slot.structure = 'puzzle_tokens'
        slot.block = 'mlp_mixer'
        slot.topology = 'feedforward'
        slot.credit = 'direct'
        slot.structure_kwargs = copy.deepcopy(kwargs)


def _batch(observation_dim=55):
    observations = jnp.arange(3 * observation_dim, dtype=jnp.float32).reshape(3, observation_dim) / 10
    actions = jnp.arange(6, dtype=jnp.float32).reshape(3, 2) / 10
    return {
        'observations': observations,
        'next_observations': observations + 0.1,
        'actions': actions,
        'value_goals': observations + 0.2,
        'actor_goals': observations + 0.3,
        'rewards': jnp.array([-1.0, 0.0, -1.0]),
        'masks': jnp.array([1.0, 0.0, 1.0]),
    }


class M15PuzzleRepresentationTest(unittest.TestCase):
    def test_parser_all_standard_sizes_and_rejects_wrong_dimension(self):
        for _, _, num_buttons, observation_dim in PUZZLE_SIZES:
            x = jnp.zeros((2, observation_dim), dtype=jnp.float32)
            robot, buttons = parse_puzzle_observation(x, num_buttons=num_buttons)
            self.assertEqual(robot.shape, (2, 19))
            self.assertEqual(buttons.shape, (2, num_buttons, 4))
        with self.assertRaisesRegex(ValueError, 'expected final dimension 55'):
            parse_puzzle_observation(jnp.zeros((2, 54)), num_buttons=9)

    def test_structured_shape_and_index_embedding_for_all_sizes(self):
        for _, _, num_buttons, observation_dim in PUZZLE_SIZES:
            with self.subTest(num_buttons=num_buttons):
                body = PuzzleStructuredBody(
                    output_dim=6,
                    input_semantics='single_observation',
                    **_structure_kwargs(num_buttons),
                )
                variables = body.init(
                    {'params': jax.random.PRNGKey(num_buttons)},
                    jnp.ones((2, observation_dim), dtype=jnp.float32),
                )
                output = body.apply(variables, jnp.ones((2, observation_dim)))
                self.assertEqual(output.shape, (2, 6))
                self.assertEqual(
                    variables['params']['index_embedding'].shape,
                    (num_buttons, 7),
                )
                self.assertNotIn('readout', variables['params'])

    def test_shared_projection_and_optional_index_embedding(self):
        body = PuzzleStructuredBody(
            output_dim=6,
            input_semantics='single_observation',
            **_structure_kwargs(9),
        )
        variables = body.init(jax.random.PRNGKey(0), jnp.ones((2, 55)))
        params = variables['params']
        self.assertEqual(params['button_projection']['kernel'].shape, (4, 7))
        self.assertEqual(params['index_embedding'].shape, (9, 7))

        no_index = PuzzleStructuredBody(
            output_dim=6,
            input_semantics='single_observation',
            **_structure_kwargs(9, index_embedding=False),
        )
        no_index_vars = no_index.init(jax.random.PRNGKey(0), jnp.ones((2, 55)))
        self.assertNotIn('index_embedding', no_index_vars['params'])

    def test_mixer_none_has_no_tm_parameter_and_has_finite_gradients(self):
        mixer = MLPMixerBlock(
            num_tokens=9,
            embed_dim=7,
            hidden_dim_tokens=5,
            hidden_dim_channels=11,
            tm_mode='none',
        )
        x = jnp.ones((2, 9, 7))
        variables = mixer.init(jax.random.PRNGKey(1), x)
        self.assertNotIn('tm_weights', variables['params'])
        output = mixer.apply(variables, x)
        self.assertEqual(output.shape, x.shape)
        grads = jax.grad(lambda p: jnp.sum(mixer.apply({'params': p}, x) ** 2))(variables['params'])
        self.assertTrue(all(np.all(np.isfinite(np.asarray(v))) for v in jax.tree_util.tree_leaves(grads)))

    def test_mixer_stack_blocks_are_untied_and_depth_scales(self):
        one = PuzzleStructuredBody(output_dim=6, input_semantics='single_observation', **_structure_kwargs(9, num_mixer_blocks=1))
        four = PuzzleStructuredBody(output_dim=6, input_semantics='single_observation', **_structure_kwargs(9, num_mixer_blocks=4))
        x = jnp.ones((2, 55))
        one_vars = one.init(jax.random.PRNGKey(2), x)
        four_vars = four.init(jax.random.PRNGKey(2), x)
        one_blocks = [k for k in one_vars['params'] if str(k).startswith('mixer_blocks_')]
        four_blocks = [k for k in four_vars['params'] if str(k).startswith('mixer_blocks_')]
        self.assertEqual(len(one_blocks), 1)
        self.assertEqual(len(four_blocks), 4)
        self.assertGreater(
            sum(np.asarray(v).size for v in jax.tree_util.tree_leaves(four_vars['params'])),
            sum(np.asarray(v).size for v in jax.tree_util.tree_leaves(one_vars['params'])),
        )

    def test_robot_fusion_and_critic_context_dimensions(self):
        pair = PuzzleStructuredBody(
            output_dim=8,
            input_semantics='goal_pair',
            action_semantics='none',
            **_structure_kwargs(9),
        )
        pair_vars = pair.init(jax.random.PRNGKey(3), jnp.ones((2, 110)))
        self.assertEqual(pair_vars['params']['robot_projection']['kernel'].shape, (38, 8))
        critic = PuzzleStructuredBody(
            output_dim=8,
            input_semantics='goal_pair',
            action_semantics='robot_context',
            **_structure_kwargs(9),
        )
        critic_vars = critic.init(jax.random.PRNGKey(4), jnp.ones((2, 112)))
        self.assertEqual(critic_vars['params']['robot_projection']['kernel'].shape, (40, 8))

    def test_structured_body_gradient_reaches_all_required_components(self):
        body = PuzzleStructuredBody(output_dim=6, input_semantics='single_observation', **_structure_kwargs(9))
        x = jnp.ones((2, 55))
        variables = body.init(jax.random.PRNGKey(5), x)
        grads = jax.grad(lambda p: jnp.sum(body.apply({'params': p}, x) ** 2))(variables['params'])
        for key in ('button_projection', 'index_embedding', 'robot_projection', 'fusion', 'mixer_blocks_0'):
            self.assertIn(key, grads)
            self.assertTrue(all(np.all(np.isfinite(np.asarray(v))) for v in jax.tree_util.tree_leaves(grads[key])))


class M15AlgorithmIntegrationTest(unittest.TestCase):
    def test_gciql_structured_initializes_all_puzzle_sizes(self):
        for _, _, num_buttons, observation_dim in PUZZLE_SIZES:
            with self.subTest(num_buttons=num_buttons):
                config = _small_config('gciql')
                _enable_structured(config, ('actor', 'value', 'critic'), num_buttons=num_buttons)
                observations = jnp.ones((2, observation_dim), dtype=jnp.float32)
                actions = jnp.ones((2, 2), dtype=jnp.float32)
                agent = agents['gciql'].create(14, observations, actions, config)
                actor_dist = agent.network.select('actor')(observations, observations)
                q1, q2 = agent.network.select('critic')(observations, observations, actions)
                self.assertEqual(actor_dist.mode().shape, (2, 2))
                self.assertEqual(q1.shape, (2,))
                self.assertEqual(q2.shape, (2,))

    def test_gciql_all_three_slots_and_target_structured_copy(self):
        config = _small_config('gciql')
        _enable_structured(config, ('actor', 'value', 'critic'))
        batch = _batch()
        agent = agents['gciql'].create(10, batch['observations'], batch['actions'], config)
        critic = agent.network.params['modules_critic']['value_net']['core']['topology']['primitive']
        target = agent.network.params['modules_target_critic']['value_net']['core']['topology']['primitive']
        self.assertEqual(critic['index_embedding'].shape, (2, 9, 7))
        for left, right in zip(jax.tree_util.tree_leaves(critic), jax.tree_util.tree_leaves(target)):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
        updated, info = agent.update(batch)
        self.assertTrue(all(np.all(np.isfinite(np.asarray(v))) for v in info.values()))
        report = _computation_slot_accounting(updated, config)
        self.assertEqual(set(report), {'actor', 'value', 'critic'})
        self.assertEqual(report['actor']['robot_input_dim'], 38)
        self.assertEqual(report['critic']['robot_input_dim'], 40)
        self.assertTrue(all(item['structured_body_dense_macs'] > 0 for item in report.values()))

    def test_gcbc_gcivl_and_qrl_structured_smokes(self):
        batch = _batch()
        for name, slots in (
            ('gcbc', ('actor',)),
            ('gcivl', ('actor', 'value')),
        ):
            with self.subTest(agent=name):
                config = _small_config(name)
                _enable_structured(config, slots)
                agent = agents[name].create(11, batch['observations'], batch['actions'], config)
                updated, info = agent.update(batch)
                self.assertTrue(all(np.all(np.isfinite(np.asarray(v))) for v in info.values()))

        config = _small_config('qrl')
        _enable_structured(config, ('actor', 'value'))
        config.compute.dynamics.enabled = True
        config.compute.dynamics.topology = 'feedforward'
        config.compute.dynamics.credit = 'direct'
        config.compute.dynamics.structure = 'vector'
        config.compute.dynamics.primitive = 'mlp'
        agent = agents['qrl'].create(12, batch['observations'], batch['actions'], config)
        value = agent.network.select('value')(batch['observations'], batch['value_goals'])
        bypass = agent.network.select('value')(jnp.ones((3, 6)), jnp.zeros((3, 6)), is_phi=True)
        self.assertEqual(value.shape, (3,))
        self.assertEqual(bypass.shape, (3,))
        updated, info = agent.update(batch)
        self.assertTrue(all(np.all(np.isfinite(np.asarray(v))) for v in info.values()))
        self.assertIn('modules_dynamics', updated.network.params)

    def test_invalid_puzzle_topology_fails_loudly(self):
        spec = ComputationSpec.from_mapping({
            'structure': 'puzzle_tokens',
            'topology': 'single_state',
            'credit': 'direct',
            'block': 'mlp_mixer',
            'structure_kwargs': _structure_kwargs(9),
            'input_semantics': 'goal_pair',
        })
        with self.assertRaisesRegex(ValueError, 'topology=feedforward'):
            make_computation_core(spec, hidden_dims=(8,), activate_final=True)

    def test_structured_accounting_contains_token_multiplicity(self):
        body = PuzzleStructuredBody(output_dim=6, input_semantics='single_observation', **_structure_kwargs(9))
        variables = body.init(jax.random.PRNGKey(13), jnp.ones((2, 55)))
        report = structured_body_accounting(variables['params'], _structure_kwargs(9))
        self.assertEqual(report['num_tokens'], 9)
        self.assertEqual(report['index_embedding_params'], 63)
        self.assertEqual(report['structured_sequential_depth'], 10)
        self.assertGreater(report['structured_body_dense_macs'], report['mixer_dense_macs'])


if __name__ == '__main__':
    unittest.main()
