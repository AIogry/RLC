import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from impls.agents.crl import CRLAgent, get_config
from impls.analysis.crl_interaction import (
    _bootstrap_ratio,
    _candidate_source_map,
    _critic_q,
    _identity_component,
    _extraction_sample_fields,
    _extraction_rows,
    _load_bank,
    _mode_actions,
    _temporal_rows,
    load_interaction_spec,
    validate_source_set,
)
from impls.computation.accounting import count_parameters
from impls.experiment import load_study, prepare_run_design
from impls.main import _computation_slot_accounting, _make_config, _parse_args
from impls.utils.flax_utils import restore_agent, save_agent


ROOT = Path(__file__).resolve().parents[2]
STUDY_PATH = ROOT / 'experiments' / 'M11A_crl_computation_interaction' / 'study.yaml'
SPEC_PATH = ROOT / 'experiments' / 'M11A_crl_computation_interaction' / 'diagnostic.yaml'
M9A_PATH = ROOT / 'experiments' / 'M9A_single_state_iteration' / 'study.yaml'
M9B_PATH = ROOT / 'experiments' / 'M9B_two_state' / 'study.yaml'


def _small_crl_config(actor_topology='feedforward', critic_topology='feedforward'):
    config = get_config()
    config['actor_hidden_dims'] = (6, 6)
    config['value_hidden_dims'] = (6, 6)
    config['latent_dim'] = 3
    config['batch_size'] = 4
    if actor_topology != 'feedforward':
        config['compute']['actor'].update({
            'enabled': True,
            'topology': actor_topology,
            'credit': 'direct' if actor_topology == 'single_state' else 'full_bptt',
            'topology_kwargs': (
                {
                    'iterations': 4,
                    'residual': False,
                    'input_injection': 'z_plus_x',
                    'state_dim': 6,
                    'state_init': 'normal_buffer',
                    'state_init_std': 1.0,
                    'update_depth': 2,
                }
                if actor_topology == 'single_state' else
                {
                    'h_cycles': 2,
                    'l_cycles': 1,
                    'input_injection': 'l_receives_x',
                    'state_dim': 6,
                    'state_init': 'normal_buffer',
                    'state_init_std': 1.0,
                    'update_depth': 2,
                }
            ),
        })
    if critic_topology != 'feedforward':
        kwargs = (
            {
                'iterations': 4,
                'residual': False,
                'input_injection': 'z_plus_x',
                'state_dim': 3,
                'state_init': 'normal_buffer',
                'state_init_std': 1.0,
                'update_depth': 3,
            }
            if critic_topology == 'single_state' else
            {
                'h_cycles': 2,
                'l_cycles': 1,
                'input_injection': 'l_receives_x',
                'state_dim': 3,
                'state_init': 'normal_buffer',
                'state_init_std': 1.0,
                'update_depth': 3,
            }
        )
        for slot in ('critic_state', 'critic_goal'):
            config['compute'][slot].update({
                'enabled': True,
                'topology': critic_topology,
                'credit': 'direct' if critic_topology == 'single_state' else 'full_bptt',
                'topology_kwargs': kwargs,
            })
    return config


def _batch():
    observations = jnp.arange(4 * 4, dtype=jnp.float32).reshape(4, 4) / 10.0
    goals = observations + 0.1
    actions = jnp.zeros((4, 2), dtype=jnp.float32)
    return {
        'observations': observations,
        'value_goals': goals,
        'actor_goals': goals,
        'actions': actions,
    }


class M11AInteractionStudyTest(unittest.TestCase):
    def test_legacy_m9_crl_actor_configs_keep_primitive_and_checkpoint_behavior(self):
        cases = (
            (M9A_PATH, 'M9A-C007', 'single_state', {'iterations': 4}),
            (M9B_PATH, 'M9B-C001', 'two_state', {'h_cycles': 2, 'l_cycles': 1}),
        )
        observations = jnp.zeros((1, 29), dtype=jnp.float32)
        actions = jnp.zeros((1, 8), dtype=jnp.float32)
        for study_path, config_id, topology_name, expected_schedule in cases:
            with self.subTest(config_id=config_id):
                _, configuration = prepare_run_design(study_path, config_id)
                config = _make_config(_parse_args(['--agent', 'crl']), configuration=configuration)
                slot = config['compute']['actor']
                self.assertNotIn('update_depth', slot['topology_kwargs'])
                self.assertNotIn('layer_norm', slot['topology_kwargs'])
                self.assertNotIn('update_activate_final', slot['topology_kwargs'])
                self.assertEqual(slot['topology'], topology_name)
                for key, value in expected_schedule.items():
                    self.assertEqual(slot['topology_kwargs'][key], value)
                agent = CRLAgent.create(0, observations, actions, config)
                topology = agent.network.params['modules_actor']['actor_net']['topology']
                self.assertEqual(set(topology['update_module'] if topology_name == 'single_state' else topology['h_update']), {'Dense_0', 'Dense_1'})
                if topology_name == 'two_state':
                    self.assertEqual(set(topology['l_update']), {'Dense_0', 'Dense_1'})
                self.assertNotIn('LayerNorm_0', topology['input_mapping'])
                action_before = np.asarray(
                    agent.sample_actions(observations, observations, seed=jax.random.PRNGKey(91))
                )
                self.assertTrue(np.all(np.isfinite(action_before)))
                with tempfile.TemporaryDirectory(prefix=f'{config_id}_restore_') as temp_dir:
                    save_agent(agent, temp_dir, 1)
                    restored = restore_agent(agent, temp_dir, 1)
                    action_after = np.asarray(
                        restored.sample_actions(observations, observations, seed=jax.random.PRNGKey(91))
                    )
                    np.testing.assert_array_equal(action_before, action_after)

    def test_exact_seven_factorial_conditions_and_protocol(self):
        study = load_study(STUDY_PATH)
        configs = sorted((STUDY_PATH.parent / 'configs').glob('M11A-C*.yaml'))
        self.assertEqual(len(configs), 7)
        self.assertEqual(study.data['environments'], ['antmaze-large-navigate-v0'])
        self.assertEqual(study.data['seeds'], [0])
        self.assertEqual(study.data['protocol']['train_steps'], 1_000_000)
        self.assertEqual(study.data['protocol']['primary_checkpoint'], 'last@1M')
        self.assertEqual(study.data['fixed_design']['critic_branches'], ['critic_state', 'critic_goal'])
        self.assertEqual(study.data['fixed_design']['critic_branch_parameter_sharing'], False)

    def test_config_slots_and_resolved_depths(self):
        expected = {
            'M11A-C001': (False, set()),
            'M11A-C002': (False, {'single_state'}),
            'M11A-C003': (True, {'single_state'}),
            'M11A-C004': (True, {'single_state'}),
            'M11A-C005': (False, {'two_state'}),
            'M11A-C006': (True, {'two_state'}),
            'M11A-C007': (True, {'two_state'}),
        }
        for config_id, (actor, expected_topologies) in expected.items():
            _, configuration = prepare_run_design(STUDY_PATH, config_id)
            config = _make_config(_parse_args(['--agent', 'crl']), configuration=configuration)
            self.assertEqual(bool(config['compute']['actor']['enabled']), actor)
            enabled_topologies = {
                slot.get('topology')
                for slot_name in ('actor', 'critic_state', 'critic_goal')
                for slot in [config['compute'][slot_name]]
                if slot.get('enabled', False)
            }
            self.assertEqual(enabled_topologies, expected_topologies)
            for slot_name in ('value_state', 'value_goal'):
                self.assertFalse(config['compute'][slot_name]['enabled'])

    def test_both_recurrent_critic_branches_update_and_restore(self):
        config = _small_crl_config(actor_topology='two_state', critic_topology='single_state')
        batch = _batch()
        agent = CRLAgent.create(7, batch['observations'][:1], batch['actions'][:1], config)
        updated, info = agent.update(batch)
        self.assertTrue(all(np.all(np.isfinite(np.asarray(value))) for value in info.values()))
        critic = updated.network.params['modules_critic']
        for branch in ('phi', 'psi'):
            update = critic[branch]['core']['topology']['update_module']
            self.assertEqual(
                set(update),
                {'Dense_0', 'Dense_1', 'Dense_2', 'LayerNorm_0', 'LayerNorm_1'},
            )
            buffers = updated.network.model_state['buffers']['modules_critic'][branch]['core']['topology']
            self.assertEqual(buffers['z_init'].shape, (2, 3))
        self.assertNotEqual(
            np.asarray(critic['phi']['core']['topology']['input_mapping']['Dense_0']['kernel']).tobytes(),
            np.asarray(critic['psi']['core']['topology']['input_mapping']['Dense_0']['kernel']).tobytes(),
        )
        report = _computation_slot_accounting(updated, config)
        self.assertEqual(report['critic_state']['update_depth'], 3)
        self.assertEqual(report['critic_goal']['update_depth'], 3)
        self.assertEqual(report['critic_state']['total_update_executions'], 4)
        self.assertTrue(report['critic_state']['layer_norm'])
        self.assertFalse(report['critic_state']['update_activate_final'])
        self.assertFalse(report['actor']['layer_norm'])
        self.assertTrue(report['actor']['update_activate_final'])
        for branch in ('phi', 'psi'):
            topology = critic[branch]['core']['topology']
            self.assertIn('LayerNorm_0', topology['input_mapping'])
            self.assertIn('LayerNorm_0', topology['update_module'])
        with tempfile.TemporaryDirectory(prefix='m11a_crl_restore_') as temp_dir:
            save_agent(updated, temp_dir, 1)
            restored = restore_agent(updated, temp_dir, 1)
            before = np.asarray(updated.sample_actions(batch['observations'][:1], batch['actor_goals'][:1], seed=jax.random.PRNGKey(4)))
            after = np.asarray(restored.sample_actions(batch['observations'][:1], batch['actor_goals'][:1], seed=jax.random.PRNGKey(4)))
            np.testing.assert_array_equal(before, after)

    def test_diagnostic_spec_is_declarative_and_uses_common_seeds(self):
        spec = load_interaction_spec(SPEC_PATH)
        self.assertEqual(spec['protocol']['seed_scheme'], 'common_task_episode_v1')
        self.assertEqual(spec['protocol']['episodes_per_task'], 20)
        self.assertEqual(spec['anchor_stride'], 25)
        self.assertEqual(spec['max_goal_offset'], 200)
        self.assertEqual(spec['checkpoint'], {'selector': 'last'})

    def test_diagnostic_formulas_on_synthetic_scores(self):
        bank = {
            'pair_anchor_indices': np.asarray([0, 0, 0]),
            'pair_task_ids': np.asarray([1, 1, 1]),
            'pair_episode_indices': np.asarray([0, 0, 0]),
            'pair_h': np.asarray([25, 50, 75]),
            'anchor_task_ids': np.asarray([1]),
        }
        spec = {
            'bootstrap_seed': 11,
            'bootstrap_replicates': 20,
            'epsilon': 1e-6,
        }
        temporal = _temporal_rows(bank, np.asarray([3.0, 2.0, 1.0]), 'critic', spec)
        overall = next(row for row in temporal if row['scope'] == 'overall')
        self.assertEqual(overall['value'], 0.0)
        extraction_bank = {
            'pair_anchor_indices': np.asarray([0]),
            'pair_task_ids': np.asarray([1]),
            'pair_episode_indices': np.asarray([0]),
        }
        extraction = _extraction_rows(
            extraction_bank,
            np.asarray([[5.0, 3.0, 1.0, 4.0, 2.0]]),
            'critic', 'actor', ['exec', 'A', 'S-C', 'S-A', 'S-CA'], 3, spec,
            candidate_actions=np.asarray([[
                [0.0, 0.0],
                [0.1, 0.0],
                [0.2, 0.0],
                [0.3, 0.0],
                [0.4, 0.0],
            ]]),
        )
        gap = next(row for row in extraction if row['metric'] == 'E_ext_gap' and row['scope'] == 'overall')
        rank = next(row for row in extraction if row['metric'] == 'E_ext_rank' and row['scope'] == 'overall')
        self.assertAlmostEqual(gap['value'], 0.25, places=5)
        self.assertAlmostEqual(rank['value'], 0.25, places=5)

    def test_duplicate_candidate_metric_uses_action_vectors_not_q_scores(self):
        bank = {
            'pair_anchor_indices': np.asarray([0, 0]),
            'pair_task_ids': np.asarray([1, 1]),
            'pair_episode_indices': np.asarray([0, 0]),
        }
        spec = {'bootstrap_seed': 13, 'bootstrap_replicates': 10, 'epsilon': 1e-6}
        rows = _extraction_rows(
            bank,
            np.asarray([
                [1.0, 1.0, 2.0],
                [1.0, 2.0, 3.0],
            ]),
            'critic', 'actor', ['exec', 'A', 'S-C'], 1, spec,
            candidate_actions=np.asarray([
                [[0.0], [0.0], [1.0]],
                [[0.0], [1.0], [2.0]],
            ]),
        )
        duplicate = next(
            row for row in rows
            if row['metric'] == 'duplicate_candidate_pool' and row['scope'] == 'overall'
        )
        self.assertAlmostEqual(duplicate['value'], 0.5, places=6)

    def test_temporal_tie_semantics_and_episode_cluster(self):
        bank = {
            'pair_anchor_indices': np.asarray([0, 0, 0]),
            'pair_task_ids': np.asarray([1, 1, 1]),
            'pair_episode_indices': np.asarray([0, 0, 0]),
            'pair_h': np.asarray([25, 50, 75]),
            'anchor_task_ids': np.asarray([1]),
        }
        rows = _temporal_rows(
            bank, np.asarray([1.0, 1.0, 0.0]), 'critic',
            {'bootstrap_seed': 7, 'bootstrap_replicates': 20},
        )
        overall = next(row for row in rows if row['scope'] == 'overall')
        self.assertAlmostEqual(overall['value'], 1 / 3)
        self.assertEqual(overall['ties'], 1)
        self.assertEqual(overall['n_episodes'], 1)

    def test_extraction_raw_fields_keep_degenerate_and_duplicate_flags(self):
        q_values = np.asarray([[2.0, 2.0, 2.0], [3.0, 1.0, 2.0]])
        actions = np.asarray([
            [[0.0], [0.0], [1.0]],
            [[0.0], [1.0], [2.0]],
        ])
        fields = _extraction_sample_fields(
            q_values, ['exec', 'actor', 'other'], actions, actor_index=1, epsilon=1e-6,
        )
        self.assertEqual(fields['q_max'].shape, (2,))
        self.assertTrue(fields['degenerate_pool'][0])
        self.assertFalse(fields['degenerate_pool'][1])
        self.assertTrue(fields['exact_duplicate_action_pool'][0])
        self.assertFalse(fields['exact_duplicate_action_pool'][1])
        np.testing.assert_array_equal(fields['actor_rank'], np.asarray([0, 2]))
        self.assertEqual(fields['best_candidate_identity'][1], 'exec')

    def test_conservative_critic_score_uses_ensemble_min(self):
        class Network:
            def select(self, name):
                self.assert_name = name
                return lambda observations, goals, actions: jnp.asarray(
                    [[3.0, 1.0], [2.0, 4.0]], dtype=jnp.float32
                )

        class Agent:
            network = Network()

        scores = _critic_q(
            Agent(), np.zeros((2, 3), dtype=np.float32),
            np.zeros((2, 3), dtype=np.float32), np.zeros((2, 1), dtype=np.float32),
        )
        np.testing.assert_array_equal(scores, np.asarray([2.0, 1.0]))

    def test_mode_actions_are_deterministic_and_clipped(self):
        class Distribution:
            def __init__(self, values):
                self.values = values

            def mode(self):
                return self.values

        class Network:
            def select(self, name):
                return lambda observations, goals, temperature: Distribution(
                    jnp.full((len(observations), 2), 2.0)
                )

        class Agent:
            network = Network()

        observations = np.zeros((3, 4), dtype=np.float32)
        goals = np.zeros((3, 4), dtype=np.float32)
        first = _mode_actions(Agent(), observations, goals, chunk_size=2)
        second = _mode_actions(Agent(), observations, goals, chunk_size=2)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, np.ones((3, 2), dtype=np.float32))

    def test_declarative_candidate_order_and_source_mapping(self):
        spec = load_interaction_spec(SPEC_PATH)
        mapping = _candidate_source_map(spec, 'single_state')
        self.assertEqual(list(mapping), [
            'a_exec', 'a_M11A-C001', 'a_M11A-C002', 'a_M11A-C003', 'a_M11A-C004',
        ])
        self.assertIsNone(mapping['a_exec'])
        self.assertEqual(mapping['a_M11A-C004'], 'M11A-C004')

    def test_episode_bootstrap_is_clustered_and_deterministic(self):
        clusters = {('task1', 0): (1.0, 1.0), ('task1', 1): (0.0, 1.0)}
        first = _bootstrap_ratio(clusters, seed=123, replicates=100)
        second = _bootstrap_ratio(clusters, seed=123, replicates=100)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first[0], 0.0)
        self.assertLessEqual(first[1], 1.0)

    def test_source_commit_and_checkpoint_mismatch_fail_loudly(self):
        spec = load_interaction_spec(SPEC_PATH)
        bad_commit = dict(spec)
        bad_commit['source_git_commit'] = '0' * 40
        with self.assertRaises(ValueError):
            validate_source_set(bad_commit)
        bad_checkpoint = dict(spec)
        bad_checkpoint['checkpoint'] = {'selector': 'best'}
        with self.assertRaises(ValueError):
            validate_source_set(bad_checkpoint)

    def test_bank_hash_mismatch_fails_loudly(self):
        spec = load_interaction_spec(SPEC_PATH)
        with tempfile.TemporaryDirectory(prefix='m11a_bad_bank_') as temp_dir:
            bank_dir = Path(temp_dir) / spec['diagnostic_id'] / 'bank'
            bank_dir.mkdir(parents=True)
            np.savez_compressed(bank_dir / 'diagnostic_bank.npz', pair_ids=np.asarray([0]))
            (bank_dir / 'bank_metadata.json').write_text(
                '{"diagnostic_id": "M11A-D001", "bank_sha256": "bad"}\n'
            )
            with self.assertRaises(ValueError):
                _load_bank(temp_dir, spec)

    def test_critic_identity_component_requires_params_and_buffers(self):
        params = {'Dense_0': {'kernel': np.ones((2, 2), dtype=np.float32)}}
        buffers_a = {'z_init': np.zeros((2,), dtype=np.float32)}
        buffers_b = {'z_init': np.ones((2,), dtype=np.float32)}
        params_audit = _identity_component(params, params, tolerance=1e-6)
        buffers_audit = _identity_component(buffers_a, buffers_b, tolerance=1e-6)
        self.assertTrue(params_audit['exact_array_equal'])
        self.assertFalse(buffers_audit['exact_array_equal'])
        self.assertFalse(
            params_audit['exact_array_equal'] and buffers_audit['exact_array_equal']
        )
        self.assertEqual(buffers_audit['element_count_reference'], 2)
        self.assertEqual(buffers_audit['element_count_compared'], 2)

    def test_ddpgbc_actor_q_branch_does_not_update_critic_params(self):
        config = _small_crl_config(actor_topology='single_state', critic_topology='feedforward')
        batch = _batch()
        agent = CRLAgent.create(11, batch['observations'][:1], batch['actions'][:1], config)

        def actor_loss(params):
            return agent.actor_loss(batch, params, rng=jax.random.PRNGKey(17))[0]

        def q_loss(params):
            return agent.actor_loss(batch, params, rng=jax.random.PRNGKey(17))[1]['q_loss']

        actor_grad = jax.grad(actor_loss)(agent.network.params)
        q_grad = jax.grad(q_loss)(agent.network.params)
        for branch in ('phi', 'psi'):
            actor_critic_leaves = jax.tree_util.tree_leaves(actor_grad['modules_critic'][branch])
            q_critic_leaves = jax.tree_util.tree_leaves(q_grad['modules_critic'][branch])
            for actor_leaf, q_leaf in zip(actor_critic_leaves, q_critic_leaves):
                np.testing.assert_array_equal(actor_leaf, np.zeros_like(actor_leaf))
                np.testing.assert_array_equal(q_leaf, np.zeros_like(q_leaf))

        actor_q_leaves = jax.tree_util.tree_leaves(q_grad['modules_actor'])
        self.assertTrue(any(np.any(np.asarray(leaf) != 0) for leaf in actor_q_leaves))


if __name__ == '__main__':
    unittest.main()
