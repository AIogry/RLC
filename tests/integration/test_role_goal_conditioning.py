"""CPU engineering gates for role conditioning, gradients and fresh restore."""

import copy
import gc
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from impls.agents import agents
from impls.computation.factory import make_computation_core, resolve_slot_spec
from impls.computation.interfaces import ComputationOutput
from impls.diagnostics.goal_conditioning import audit_goal_conditioning, capture_goal_conditioning_inputs
from impls.experiment.reevaluation import _make_restored_agent, _restore_agent_for_reevaluation
from impls.main import _computation_runtime_extras, _computation_slot_accounting, _make_config, _parse_args
from impls.networks.goal_conditioning import resolve_goal_conditioning
from impls.representation.interfaces import StructuredNetworkInput
from impls.representation.puzzle import PuzzleTokenAdapter
from impls.representation.puzzle_conditioning import extract_button_bits
from impls.utils.checkpointing import GoalConditioningMismatch, goal_conditioning_checkpoint_semantics
from impls.utils.flax_utils import restore_agent_from_checkpoint, restore_module_from_checkpoint, save_agent, save_semantic_checkpoint
from impls.utils.puzzle_datasets import PuzzleBoardGCDataset
from tests.integration.test_gciql_puzzle_goal_conditioning import _config as legacy_config
from tests.integration.test_puzzle_board_dataset import _dataset_config, _puzzle_raw
from tests.reference.goal_conditioning import agent_config, batch, goal_config, observations, role


def tree_equal(testcase, a, b):
    testcase.assertEqual(jax.tree_util.tree_structure(a), jax.tree_util.tree_structure(b))
    for left, right in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        np.testing.assert_array_equal(left, right)


def norm(tree):
    return sum(float(jnp.sum(jnp.abs(leaf))) for leaf in jax.tree_util.tree_leaves(tree))


def create(config=None, seed=26019):
    data = batch()
    return agents['gciql'].create(seed, data['observations'], data['actions'],
                                 agent_config() if config is None else config)


def output_snapshot(agent, data):
    return jax.device_get({
        'state': flax.serialization.to_state_dict(agent),
        'action': agent.sample_actions(data['observations'], data['actor_goals'],
                                       seed=jax.random.PRNGKey(83), temperature=0.0),
        'value': agent.network.select('value')(data['observations'], data['value_goals']),
        'critic': agent.network.select('critic')(data['observations'], data['value_goals'], data['actions']),
        'target': agent.network.select('target_critic')(data['observations'], data['value_goals'], data['actions']),
    })


class RoleGoalConditioningTest(unittest.TestCase):
    def tearDown(self):
        jax.clear_caches()
        gc.collect()

    def test_same_input_schema_same_initial_parameters_and_optimizer(self):
        reference = create()
        treatments = (
            (role('operation'), role()),
            (role(), role('operation')),
            (role('operation', 'P'), role('operation', 'B')),
            (role('residual', distance_feature='exact_press_fraction'), role()),
            (role('operation', 'B', 'exact_press_fraction'), role('operation', 'P', 'exact_press_fraction')),
        )
        for actor, value in treatments:
            with self.subTest(actor=actor, value=value):
                candidate = create(agent_config(actor, value))
                tree_equal(self, reference.network.params, candidate.network.params)
                tree_equal(self, reference.network.opt_state, candidate.network.opt_state)
                tree_equal(self, reference.network.model_state, candidate.network.model_state)
                tree_equal(self, reference.rng, candidate.rng)
        actor_params = reference.network.params['modules_actor']['actor_net']['adapter']
        self.assertEqual(actor_params['button_projection']['kernel'].shape, (9, 7))
        self.assertEqual(actor_params['robot_projection']['kernel'].shape, (38, 8))
        critic_params = reference.network.params['modules_critic']['value_net']['core']['adapter']
        self.assertEqual(critic_params['button_projection']['kernel'].shape, (2, 9, 7))
        self.assertEqual(critic_params['robot_projection']['kernel'].shape, (2, 40, 8))
        self.assertFalse(np.array_equal(critic_params['button_projection']['kernel'][0],
                                       critic_params['button_projection']['kernel'][1]))

    def test_actual_inputs_roles_next_state_and_manual_body_equivalence(self):
        config = agent_config(role('operation', 'P'), role('operation', 'B', 'exact_press_fraction'))
        agent = create(config)
        plan = resolve_goal_conditioning(config.goal_conditioning)
        data = batch()
        for unbatched in (False, True):
            s, g, a = data['observations'], data['actor_goals'], data['actions']
            if unbatched:
                s, g, a = s[0], g[0], a[0]
            captured = capture_goal_conditioning_inputs(agent, s, g, a)
            for name in ('actor', 'value', 'critic', 'target_critic'):
                prepared = (plan.actor if name == 'actor' else plan.value_side).prepare(s, g)
                pieces = [s, prepared.goals] + ([a] if name in ('critic', 'target_critic') else [])
                manual = StructuredNetworkInput(jnp.concatenate(pieces, axis=-1), prepared.token_aux)
                tree_equal(self, captured[name], manual)
                def body_only(module_dict, inputs):
                    module = module_dict.modules[name]
                    if name == 'actor':
                        output = module.actor_net(inputs)
                        if isinstance(output, ComputationOutput):
                            output = output.representation
                        return module.mean_net(output)
                    return module.value_readout(module.value_net(inputs)).squeeze(-1)
                actual = agent.network.select(name)(*([s, g, a] if name in ('critic', 'target_critic') else [s, g]))
                if name == 'actor':
                    actual = actual.mode()
                expected = agent.network.model_def.apply({'params': agent.network.params}, manual, method=body_only)
                np.testing.assert_array_equal(actual, expected)
                if name == 'critic':
                    self.assertEqual(actual.shape, (2,) if unbatched else (2, 3))
        current = capture_goal_conditioning_inputs(agent, data['observations'], data['value_goals'], data['actions'])
        future = capture_goal_conditioning_inputs(agent, data['next_observations'], data['value_goals'], data['actions'])
        self.assertFalse(np.array_equal(current['value'].flat_inputs[..., 55:110], future['value'].flat_inputs[..., 55:110]))
        actor_goal_inputs = capture_goal_conditioning_inputs(agent, data['observations'], data['actor_goals'], data['actions'])
        self.assertFalse(np.array_equal(current['critic'].flat_inputs, actor_goal_inputs['critic'].flat_inputs))
        report = audit_goal_conditioning(agent, data['observations'], data['actor_goals'], data['actions'])
        self.assertEqual(report['source'], 'production_forward_intermediate_capture')
        self.assertEqual(report['modules']['critic']['flat_shape'], [3, 112])

    def test_adapter_aux_shapes_physical_identity_and_no_context_leak(self):
        data = batch()
        plan = resolve_goal_conditioning(goal_config(role('operation', 'P', 'exact_press_fraction')))
        prepared = plan.actor.prepare(data['observations'], data['actor_goals'])
        flat = jnp.concatenate([data['observations'], prepared.goals, data['actions']], axis=-1)
        inputs = StructuredNetworkInput(flat, prepared.token_aux)
        adapter = PuzzleTokenAdapter(num_buttons=9, token_dim=7, robot_hidden_dim=8,
                                     action_semantics='robot_context', token_aux_dim=1)
        variables = adapter.init(jax.random.PRNGKey(13), inputs)
        representation = adapter.apply(variables, inputs)
        robot, buttons = adapter.apply(variables, flat, method=adapter._split_input)
        np.testing.assert_array_equal(buttons[..., :4], data['observations'][..., 19:].reshape(3, 9, 4))
        np.testing.assert_array_equal(robot[..., -2:], data['actions'])
        np.testing.assert_array_equal(robot[..., 19:38], 0)
        params = variables['params']
        explicit = jnp.concatenate([buttons, prepared.token_aux], axis=-1) @ params['button_projection']['kernel']
        explicit = explicit + params['button_projection']['bias'] + params['index_embedding']
        np.testing.assert_array_equal(representation.tokens, explicit)
        zero_aux = adapter.apply(variables, StructuredNetworkInput(flat, jnp.zeros_like(prepared.token_aux)))
        np.testing.assert_array_equal(representation.context, zero_aux.context)
        self.assertFalse(np.array_equal(representation.tokens, zero_aux.tokens))
        for malformed in (None, jnp.zeros((9, 1)), jnp.zeros((3, 9, 2)), jnp.zeros((1, 9, 1))):
            with self.subTest(shape=getattr(malformed, 'shape', None)), self.assertRaises(ValueError):
                adapter.apply(variables, StructuredNetworkInput(flat, malformed))
        with self.assertRaisesRegex(ValueError, 'StructuredNetworkInput'):
            adapter.apply(variables, flat)
        with self.assertRaisesRegex(ValueError, 'Legacy'):
            adapter.clone(token_aux_dim=0).init(jax.random.PRNGKey(13), inputs)

    def test_isolated_gradient_routes_and_post_gradient_polyak(self):
        config = agent_config(role('operation', 'P'), role('operation', 'B', 'exact_press_fraction'))
        config.alpha = 0.0  # Engineering-only Q-path isolation, not a Study alpha.
        agent = create(config)
        data = batch()
        for method, active in ((agent.actor_loss, 'actor'), (agent.value_loss, 'value'), (agent.critic_loss, 'critic')):
            gradients = jax.grad(lambda params: method(data, params)[0])(agent.network.params)
            for name, subtree in gradients.items():
                if name == f'modules_{active}':
                    self.assertGreater(norm(subtree), 0)
                else:
                    self.assertEqual(norm(subtree), 0, (active, name))
        action_gradient = jax.grad(lambda actions: jnp.mean(agent.network.select('critic')(
            data['observations'], data['actor_goals'], actions)))(data['actions'])
        self.assertGreater(norm(action_gradient), 0)
        tree_equal(self, agent.network.params['modules_critic'], agent.network.params['modules_target_critic'])
        updated, _ = agent.update(data)
        expected = jax.tree_util.tree_map(
            lambda p, tp: config.tau * p + (1 - config.tau) * tp,
            updated.network.params['modules_critic'], agent.network.params['modules_target_critic'],
        )
        for actual, reference in zip(jax.tree_util.tree_leaves(updated.network.params['modules_target_critic']),
                                     jax.tree_util.tree_leaves(expected)):
            np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=1e-7)
        delta = jax.tree_util.tree_map(lambda a, b: a - b, updated.network.params['modules_target_critic'],
                                      agent.network.params['modules_target_critic'])
        self.assertGreater(norm(delta), 0)

    def test_losses_use_each_roles_raw_goals_and_current_call_state(self):
        config = agent_config(role('operation', 'P'), role('residual', distance_feature='exact_press_fraction'))
        agent, data = create(config), batch()
        dist = agent.network.select('actor')(data['observations'], data['actor_goals'])
        q_actions = jnp.clip(dist.mode(), -1, 1)
        q1, q2 = agent.network.select('critic')(data['observations'], data['actor_goals'], q_actions)
        q = jnp.minimum(q1, q2)
        expected_actor = -q.mean() / (jnp.abs(q).mean() + 1e-6) - (config.alpha * dist.log_prob(data['actions'])).mean()
        actual_actor, _ = agent.actor_loss(data, agent.network.params)
        np.testing.assert_array_equal(actual_actor, expected_actor)
        plan = resolve_goal_conditioning(config.goal_conditioning)
        already_conditioned = plan.actor.prepare(data['observations'], data['actor_goals']).goals
        incorrect_q = agent.network.select('critic')(data['observations'], already_conditioned, q_actions)
        self.assertFalse(np.array_equal(jnp.stack((q1, q2)), incorrect_q))

        t1, t2 = agent.network.select('target_critic')(data['observations'], data['value_goals'], data['actions'])
        value = agent.network.select('value')(data['observations'], data['value_goals'])
        diff = jnp.minimum(t1, t2) - value
        expected_value = (jnp.where(diff >= 0, config.expectile, 1 - config.expectile) * diff ** 2).mean()
        np.testing.assert_array_equal(agent.value_loss(data, agent.network.params)[0], expected_value)
        next_value = agent.network.select('value')(data['next_observations'], data['value_goals'])
        target = data['rewards'] + config.discount * data['masks'] * next_value
        q1, q2 = agent.network.select('critic')(data['observations'], data['value_goals'], data['actions'])
        expected_critic = ((q1 - target) ** 2 + (q2 - target) ** 2).mean()
        np.testing.assert_array_equal(agent.critic_loss(data, agent.network.params)[0], expected_critic)

    def test_constructed_network_eager_and_jit_do_not_reenter_setup_algebra(self):
        agent = create(agent_config(role('operation', 'B'), role('operation', 'P', 'exact_press_fraction')))
        data = batch()
        with mock.patch('impls.networks.goal_conditioning.operator_metadata', side_effect=AssertionError('operator')), \
             mock.patch('impls.representation.puzzle_conditioning.operator_metadata', side_effect=AssertionError('operator')), \
             mock.patch('impls.representation.goal_coordinate_transforms.gf2_inverse', side_effect=AssertionError('inverse')), \
             mock.patch('impls.representation.goal_coordinate_transforms.gf2_rank', side_effect=AssertionError('rank')), \
             mock.patch('impls.networks.goal_conditioning.semantic_hash', side_effect=AssertionError('hash')), \
             mock.patch('numpy.random.default_rng', side_effect=AssertionError('transform RNG')):
            actor_forward = lambda s, g: agent.network.select('actor')(s, g).mode()
            np.testing.assert_allclose(
                actor_forward(data['observations'], data['actor_goals']),
                jax.jit(actor_forward)(data['observations'], data['actor_goals']),
                rtol=2e-5, atol=2e-6,
            )
            for name in ('value', 'critic', 'target_critic'):
                args = (data['observations'], data['value_goals'])
                if name != 'value':
                    args += (data['actions'],)
                eager = agent.network.select(name)(*args)
                compiled = jax.jit(agent.network.select(name))(*args)
                np.testing.assert_allclose(eager, compiled, rtol=2e-5, atol=2e-6)

    def test_synthetic_production_widths_4x5_4x6_without_environment_or_data(self):
        rng = np.random.default_rng(26)
        for cols in (5, 6):
            config = agent_config()
            config.goal_conditioning = goal_config(
                role('operation', 'P'), role('operation', 'B', 'exact_press_fraction'), rows=4, cols=cols,
            )
            for slot in config.compute.values():
                slot.structure_kwargs.num_buttons = 4 * cols
            states = observations(rng.integers(0, 2, (2, 4 * cols), dtype=np.uint8))
            goals = observations(rng.integers(0, 2, (2, 4 * cols), dtype=np.uint8))
            actions = jnp.zeros((2, 5))
            agent = agents['gciql'].create(26, states, actions, config)
            report = audit_goal_conditioning(agent, states, goals, actions)
            self.assertEqual(report['modules']['actor']['token_aux_shape'], [2, 4 * cols, 1])
            self.assertEqual(report['modules']['critic']['flat_shape'], [2, 2 * (19 + 16 * cols) + 5])
            self.assertEqual(agent.sample_actions(states, goals, seed=jax.random.PRNGKey(2)).shape, (2, 5))

    def test_explicit_v1_keeps_legacy_metadata_and_array_interface(self):
        original = legacy_config('residual')
        explicit = copy.deepcopy(original)
        explicit.goal_conditioning.schema_version = 1
        left, right = create(original), create(explicit)
        tree_equal(self, output_snapshot(left, batch()), output_snapshot(right, batch()))
        self.assertEqual(_computation_runtime_extras(original), _computation_runtime_extras(explicit))
        captured = capture_goal_conditioning_inputs(right, batch()['observations'], batch()['actor_goals'], batch()['actions'])
        self.assertFalse(any(isinstance(value, StructuredNetworkInput) for value in captured.values()))

    def test_actor_only_intervention_preserves_value_parameters_and_adam_multistep(self):
        for value_role in (role(), role('operation', 'B', 'exact_press_fraction')):
            with self.subTest(value_role=value_role):
                left = create(agent_config(role(), value_role))
                right = create(agent_config(role('operation', 'P', 'exact_press_fraction'), value_role))
                data = batch()
                for step in range(4):
                    current = {**data, 'actor_goals': jnp.roll(data['actor_goals'], step, axis=0),
                               'value_goals': jnp.roll(data['value_goals'], step, axis=0)}
                    left, _ = left.update(current)
                    right, _ = right.update(current)
                    for name in ('modules_value', 'modules_critic', 'modules_target_critic'):
                        tree_equal(self, left.network.params[name], right.network.params[name])
                        tree_equal(self, left.network.opt_state[0].mu[name], right.network.opt_state[0].mu[name])
                        tree_equal(self, left.network.opt_state[0].nu[name], right.network.opt_state[0].nu[name])
                    tree_equal(self, left.rng, right.rng)
                    tree_equal(self, left.network.opt_state[0].count, right.network.opt_state[0].count)
                actor_delta = jax.tree_util.tree_map(lambda a, b: a - b, left.network.params['modules_actor'],
                                                    right.network.params['modules_actor'])
                self.assertGreater(norm(actor_delta), 0)

    def test_fresh_agent_json_restore_outputs_next_update_and_semantic_metadata(self):
        config = agent_config(role('operation', 'P', 'exact_press_fraction'), role('operation', 'B'))
        data = batch()
        with tempfile.TemporaryDirectory(prefix='role_goal_restore_') as root:
            root = Path(root)
            original = create(config)
            original, _ = original.update(data)
            resolved = config.to_dict()
            resolved['goal_conditioning'] = resolve_goal_conditioning(config.goal_conditioning).to_config()
            (root / 'agent.json').write_text(json.dumps(resolved))
            expected_semantics = goal_conditioning_checkpoint_semantics(original)
            expected_output = output_snapshot(original, data)
            checkpoint = save_agent(original, root, 1)
            semantic = save_semantic_checkpoint(original, root / 'semantic', 'last', 1)
            self.assertEqual(semantic['goal_conditioning_semantics'], expected_semantics)
            next_agent, next_info = original.update(data)
            expected_next = output_snapshot(next_agent, data)
            expected_next_info = jax.device_get(next_info)
            del original, config, resolved, next_agent, next_info
            gc.collect()
            saved_config = json.loads((root / 'agent.json').read_text())
            fresh = create(saved_config, seed=991)
            restored = restore_agent_from_checkpoint(fresh, checkpoint)
            tree_equal(self, output_snapshot(restored, data), expected_output)
            self.assertEqual(goal_conditioning_checkpoint_semantics(restored), expected_semantics)
            next_agent, next_info = restored.update(data)
            tree_equal(self, output_snapshot(next_agent, data), expected_next)
            tree_equal(self, next_info, expected_next_info)

    def test_same_shape_semantic_tampering_and_missing_metadata_rejected(self):
        original_config = agent_config(role('operation', 'P'), role('operation', 'B'))
        with tempfile.TemporaryDirectory(prefix='role_goal_tamper_') as root:
            path = save_agent(create(original_config), root, 1)
            changed = []
            p = copy.deepcopy(original_config)
            p.goal_conditioning.transforms.P.permutation = list(reversed(range(9)))
            changed.append(p)
            b = copy.deepcopy(original_config)
            matrix = np.eye(9, dtype=int)
            matrix[3, 4] = 1
            b.goal_conditioning.transforms.B.matrix = matrix.tolist()
            changed.append(b)
            for field, value in (('coordinate', 'residual'), ('distance_feature', 'exact_press_fraction')):
                config = copy.deepcopy(original_config)
                config.goal_conditioning.roles.actor[field] = value
                if field == 'coordinate':
                    config.goal_conditioning.roles.actor.transform_id = 'identity'
                changed.append(config)
            swapped = copy.deepcopy(original_config)
            swapped.goal_conditioning.roles.actor = role('operation', 'B')
            swapped.goal_conditioning.roles.value_side = role('operation', 'P')
            changed.append(swapped)
            changed.append(legacy_config('oracle_operation'))
            for config in changed:
                with self.subTest(goal=config.goal_conditioning), self.assertRaises(GoalConditioningMismatch):
                    restore_agent_from_checkpoint(create(config), path)
            with self.assertRaises(GoalConditioningMismatch):
                restore_module_from_checkpoint(create(changed[0]), path, 'actor')
            with mock.patch('impls.experiment.reevaluation._restore_legacy_mixer_agent_for_reevaluation',
                            side_effect=AssertionError('semantic mismatch entered legacy adapter')):
                with self.assertRaises(GoalConditioningMismatch):
                    _restore_agent_for_reevaluation(create(changed[0]), path)
            with open(path, 'rb') as file:
                payload = pickle.load(file)
            payload['checkpoint_metadata'] = None
            missing = Path(root) / 'missing.pkl'
            with missing.open('wb') as file:
                pickle.dump(payload, file)
            with self.assertRaises(GoalConditioningMismatch):
                restore_agent_from_checkpoint(create(original_config), missing)
            old_path = save_agent(create(legacy_config('residual')), root, 2)
            with self.assertRaises(GoalConditioningMismatch):
                restore_agent_from_checkpoint(create(original_config), old_path)

    def test_resolved_main_metadata_and_source_reconstruction(self):
        source = agent_config(role('operation', 'P'), role('operation', 'B', 'exact_press_fraction'))
        configuration = type('FixtureConfiguration', (), {'data': {
            'algorithm': 'gciql', 'allow_noncanonical_actor_hidden_dims': True,
            'agent_overrides': source.to_dict(),
        }})()
        resolved = _make_config(_parse_args(['--agent', 'gciql']), configuration=configuration)
        plan = resolve_goal_conditioning(resolved.goal_conditioning)
        self.assertIn('content_sha256', resolved.goal_conditioning.transforms.P)
        metadata = _computation_runtime_extras(resolved)['goal_conditioning']
        self.assertEqual(metadata['semantic_fingerprint'], plan.fingerprint)
        with tempfile.TemporaryDirectory(prefix='role_source_reconstruction_') as root:
            original = create(resolved)
            checkpoint = save_agent(original, root, 1)
            provenance = {
                'source_metadata': {'algorithm': 'gciql', 'dataset_dir': 'unused-synthetic'},
                'resolved_config': {'algorithm_config': {'agent': json.loads(json.dumps(resolved.to_dict()))}},
                'source_environment': 'puzzle-3x3-play-v0', 'source_training_seed': 26019,
                'checkpoint_path': checkpoint,
            }
            with mock.patch('impls.utils.env_utils.make_env_and_datasets', return_value=(mock.Mock(), _puzzle_raw(), None)):
                restored, _, config, example = _make_restored_agent(provenance)
            tree_equal(self, original.network.params, restored.network.params)
            self.assertEqual(goal_conditioning_checkpoint_semantics(restored), goal_conditioning_checkpoint_semantics(original))
            self.assertEqual(config['goal_conditioning']['roles']['actor']['transform_id'], 'P')
            self.assertEqual(example['observations'].shape, (1, 55))

    def test_raw_sampling_reward_mask_and_dataset_rng_unchanged(self):
        old_config = _dataset_config('residual')
        new_config = {**old_config, 'goal_conditioning': goal_config(role('operation', 'P'), role('operation', 'B'))}
        legacy = PuzzleBoardGCDataset(_puzzle_raw(), old_config, rng=26)
        with mock.patch('impls.networks.goal_conditioning.operator_metadata', side_effect=AssertionError('dataset oracle')):
            candidate = PuzzleBoardGCDataset(_puzzle_raw(), new_config, rng=26)
            for _ in range(4):
                tree_equal(self, legacy.sample(5, return_sampling_trace=True), candidate.sample(5, return_sampling_trace=True))

    def test_actual_dense_accounting_includes_only_added_kernel_rows(self):
        legacy = create(legacy_config('residual'))
        candidate = create()
        old = _computation_slot_accounting(legacy, legacy.config)
        new = _computation_slot_accounting(candidate, candidate.config)
        for name, multiplier in (('actor', 1), ('value', 1), ('critic', 2)):
            self.assertEqual(new[name]['structured_body_params'] - old[name]['structured_body_params'], multiplier * 7)
            self.assertEqual(new[name]['structured_body_dense_macs'] - old[name]['structured_body_dense_macs'], multiplier * 9 * 7)
        self.assertNotIn('target_critic', new)

    def test_reject_unsupported_and_slot_local_scientific_overrides(self):
        mutations = (
            lambda c: setattr(c, 'discrete', True),
            lambda c: setattr(c, 'encoder', 'impala'),
            lambda c: setattr(c, 'frame_stack', 2),
            lambda c: setattr(c, 'actor_loss', 'awr'),
            lambda c: setattr(c, 'dataset_class', None),
            lambda c: setattr(c.compute.actor, 'enabled', False),
            lambda c: setattr(c.compute.actor, 'topology', 'single_state'),
            lambda c: setattr(c.compute.actor, 'block', 'entity_mlp'),
            lambda c: setattr(c.compute.actor, 'relation_mode', 'zero'),
            lambda c: setattr(c.compute.actor, 'readout', 'hybrid_context_query'),
            lambda c: setattr(c.compute.actor.structure_kwargs, 'token_aux_dim', 1),
            lambda c: setattr(c.compute.critic, 'goal_conditioning', goal_config()),
        )
        for mutate in mutations:
            config = agent_config()
            mutate(config)
            with self.subTest(mutation=mutate), self.assertRaises((ValueError, NotImplementedError)):
                create(config)
        config = legacy_config('residual', structured=False)
        config.compute.actor.enabled = True
        spec = resolve_slot_spec(config, 'actor')
        with self.assertRaisesRegex(ValueError, 'token_aux_v1'):
            make_computation_core(spec, hidden_dims=(8,), token_aux_dim=1)


if __name__ == '__main__':
    unittest.main()
