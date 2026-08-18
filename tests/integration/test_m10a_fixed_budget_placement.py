import unittest
from pathlib import Path

import jax
import jax.numpy as jnp

from impls.agents.hiql import HIQLAgent
from impls.computation.accounting import hiql_policy_accounting
from impls.experiment import load_study, prepare_run_design
from impls.main import _make_config, _parse_args


ROOT = Path(__file__).resolve().parents[2]
STUDY_PATH = ROOT / 'experiments' / 'M10A_fixed_budget_placement' / 'study.yaml'


def _m10a_config(config_id):
    _, configuration = prepare_run_design(STUDY_PATH, config_id)
    return _make_config(_parse_args(['--agent', 'hiql']), configuration=configuration)


def _make_agent(config_id):
    config = _m10a_config(config_id)
    observations = jnp.zeros((1, 29), dtype=jnp.float32)
    actions = jnp.zeros((1, 8), dtype=jnp.float32)
    return HIQLAgent.create(0, observations, actions, config), config, observations


class M10AFixedBudgetPlacementTest(unittest.TestCase):
    def test_matrix_has_exactly_11_configurations_and_33_runs(self):
        study = load_study(STUDY_PATH)
        configs = sorted((STUDY_PATH.parent / 'configs').glob('M10A-C*.yaml'))
        self.assertEqual(len(configs), 11)
        self.assertEqual(len(configs) * len(study.data['environments']) * len(study.data['seeds']), 33)
        self.assertEqual(study.data['environments'], ['antmaze-large-navigate-v0'])
        self.assertEqual(study.data['seeds'], [0, 1, 2])
        self.assertEqual(
            [path.stem for path in configs],
            [f'M10A-C{index:03d}' for index in range(1, 12)],
        )

    def test_configs_record_exact_budget_factors_and_zero_buffer(self):
        expected = {
            'M10A-C002': (1, 1, 2),
            'M10A-C003': (4, 1, 5),
            'M10A-C004': (3, 2, 5),
            'M10A-C005': (2, 3, 5),
            'M10A-C006': (1, 4, 5),
            'M10A-C007': (15, 1, 16),
            'M10A-C008': (11, 5, 16),
            'M10A-C009': (8, 8, 16),
            'M10A-C010': (5, 11, 16),
            'M10A-C011': (1, 15, 16),
        }
        for config_id, (high_k, low_k, budget) in expected.items():
            _, configuration = prepare_run_design(STUDY_PATH, config_id)
            factors = configuration.data['factors']
            self.assertEqual(
                (factors['high_iterations_K'], factors['low_iterations_K'], factors['body_compute_budget']),
                (high_k, low_k, budget),
            )
            config = _m10a_config(config_id)
            for slot_name, iterations in (('high_actor', high_k), ('low_actor', low_k)):
                slot = config['compute'][slot_name]
                self.assertTrue(slot['enabled'])
                self.assertEqual(slot['topology'], 'single_state')
                self.assertEqual(slot['credit'], 'direct')
                self.assertEqual(slot['topology_kwargs']['iterations'], iterations)
                self.assertFalse(slot['topology_kwargs']['residual'])
                self.assertEqual(slot['topology_kwargs']['state_init'], 'zero_buffer')
        baseline = _m10a_config('M10A-C001')
        self.assertFalse(baseline['compute']['high_actor']['enabled'])
        self.assertFalse(baseline['compute']['low_actor']['enabled'])

    def test_high_low_accept_long_and_independent_allocations_and_value_stays_vanilla(self):
        for config_id, high_k, low_k in (
            ('M10A-C007', 15, 1),
            ('M10A-C009', 8, 8),
            ('M10A-C011', 1, 15),
        ):
            with self.subTest(config_id=config_id):
                agent, config, observations = _make_agent(config_id)
                self.assertEqual(config['compute']['high_actor']['topology_kwargs']['iterations'], high_k)
                self.assertEqual(config['compute']['low_actor']['topology_kwargs']['iterations'], low_k)
                high_output = agent.network.select('high_actor')(observations, observations)
                low_goal = jnp.zeros((1, config['rep_dim']), dtype=jnp.float32)
                low_output = agent.network.select('low_actor')(
                    observations, low_goal, goal_encoded=True,
                )
                self.assertEqual(high_output.loc.shape, (1, config['rep_dim']))
                self.assertEqual(low_output.loc.shape, (1, 8))
                actions = agent.sample_actions(observations, observations, seed=jax.random.PRNGKey(7))
                self.assertEqual(actions.shape, (1, 8))
                self.assertFalse(config['compute']['value']['enabled'])
                self.assertNotIn('topology', agent.network.params['modules_value']['value_net'])
                self.assertNotIn('topology', agent.network.params['modules_target_value']['value_net'])

    def test_matched_budget_mac_and_parameter_invariants(self):
        audits = {}
        for config_id in ('M10A-C002', 'M10A-C003', 'M10A-C004', 'M10A-C005', 'M10A-C006',
                          'M10A-C007', 'M10A-C008', 'M10A-C009', 'M10A-C010', 'M10A-C011'):
            agent, config, _ = _make_agent(config_id)
            audits[config_id] = hiql_policy_accounting(
                agent.network.params,
                agent.network.model_state.get('buffers', {}),
                config.get('compute', {}),
            )
        moderate = [audits[f'M10A-C{index:03d}']['combined_high_low_computation_core_dense_macs'] for index in range(3, 7)]
        large = [audits[f'M10A-C{index:03d}']['combined_high_low_computation_core_dense_macs'] for index in range(7, 12)]
        self.assertEqual(len(set(moderate)), 1)
        self.assertEqual(len(set(large)), 1)
        parameter_counts = [audits[f'M10A-C{index:03d}']['combined_high_low_trainable_params'] for index in range(2, 12)]
        self.assertEqual(len(set(parameter_counts)), 1)
        input_mapping_macs = (
            audits['M10A-C003']['slots']['high_actor']['input_mapping_dense_macs']
            + audits['M10A-C003']['slots']['low_actor']['input_mapping_dense_macs']
        )
        self.assertEqual(
            audits['M10A-C003']['combined_high_low_computation_core_dense_macs'],
            input_mapping_macs + 5 * 2 * 512 * 512,
        )
        self.assertEqual(
            audits['M10A-C007']['combined_high_low_computation_core_dense_macs'],
            input_mapping_macs + 16 * 2 * 512 * 512,
        )


if __name__ == '__main__':
    unittest.main()
