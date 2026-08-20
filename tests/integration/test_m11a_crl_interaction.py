import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from impls.agents.crl import CRLAgent, get_config
from impls.analysis.crl_interaction import (
    _extraction_rows,
    _temporal_rows,
    load_interaction_spec,
)
from impls.computation.accounting import count_parameters
from impls.experiment import load_study, prepare_run_design
from impls.main import _computation_slot_accounting, _make_config, _parse_args
from impls.utils.flax_utils import restore_agent, save_agent


ROOT = Path(__file__).resolve().parents[2]
STUDY_PATH = ROOT / 'experiments' / 'M11A_crl_computation_interaction' / 'study.yaml'
SPEC_PATH = ROOT / 'experiments' / 'M11A_crl_computation_interaction' / 'diagnostic.yaml'


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
            self.assertEqual(set(update), {'Dense_0', 'Dense_1', 'Dense_2'})
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
        )
        gap = next(row for row in extraction if row['metric'] == 'E_ext_gap' and row['scope'] == 'overall')
        rank = next(row for row in extraction if row['metric'] == 'E_ext_rank' and row['scope'] == 'overall')
        self.assertAlmostEqual(gap['value'], 0.25, places=5)
        self.assertAlmostEqual(rank['value'], 0.25, places=5)


if __name__ == '__main__':
    unittest.main()
