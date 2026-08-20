import unittest
from pathlib import Path

from impls.experiment import load_configuration, load_study, prepare_run_design
from impls.main import _make_config, _parse_args, _resolved_compute_snapshot


ROOT = Path(__file__).resolve().parents[2]
STUDY_PATH = ROOT / 'experiments' / 'M9B_two_state_1m' / 'study.yaml'
HISTORICAL_STUDY_PATH = ROOT / 'experiments' / 'M9B_two_state' / 'study.yaml'


EXPECTED = {
    'M9B1M-C001': {
        'algorithm': 'crl',
        'placement': 'actor',
        'h_cycles': 2,
        'l_cycles': 1,
        'enabled_slots': ('actor',),
        'historical_config_id': 'M9B-C001',
    },
    'M9B1M-C002': {
        'algorithm': 'crl',
        'placement': 'actor',
        'h_cycles': 2,
        'l_cycles': 6,
        'enabled_slots': ('actor',),
        'historical_config_id': 'M9B-C003',
    },
    'M9B1M-C003': {
        'algorithm': 'hiql',
        'placement': 'high_actor+low_actor',
        'h_cycles': 2,
        'l_cycles': 1,
        'enabled_slots': ('high_actor', 'low_actor'),
        'historical_config_id': 'M9B-C013',
    },
    'M9B1M-C004': {
        'algorithm': 'hiql',
        'placement': 'high_actor+low_actor',
        'h_cycles': 2,
        'l_cycles': 6,
        'enabled_slots': ('high_actor', 'low_actor'),
        'historical_config_id': 'M9B-C015',
    },
}


def _resolved_compute(study_path, config_id, algorithm):
    _, configuration = prepare_run_design(study_path, config_id)
    config = _make_config(_parse_args(['--agent', algorithm]), configuration=configuration)
    return _resolved_compute_snapshot(config)


class M9B1MStudyTest(unittest.TestCase):
    def test_exact_matrix_protocol_and_run_count(self):
        study = load_study(STUDY_PATH)
        configs = sorted((STUDY_PATH.parent / 'configs').glob('M9B1M-C*.yaml'))
        self.assertEqual([path.stem for path in configs], sorted(EXPECTED))
        self.assertEqual(len(configs), 4)
        self.assertEqual(study.data['environments'], ['antmaze-large-navigate-v0'])
        self.assertEqual(study.data['seeds'], [0])
        self.assertEqual(len(configs) * len(study.data['environments']) * len(study.data['seeds']), 4)

        self.assertEqual(study.data['protocol'], {
            'train_steps': 1000000,
            'batch_size': 1024,
            'log_interval': 5000,
            'eval_interval': 100000,
            'eval_tasks': 'all',
            'eval_task_count': 5,
            'eval_episodes': 20,
            'eval_temperature': 0.0,
            'eval_gaussian': None,
            'video_episodes': 0,
            'save_interval': 100000,
            'save_best_checkpoint': True,
            'save_last_checkpoint': True,
            'selection_metric': 'evaluation/overall_success',
            'selection_rule': 'strict_greater_than_keep_earlier_tie',
            'numeric_checkpoint_steps': [
                100000, 200000, 300000, 400000, 500000,
                600000, 700000, 800000, 900000, 1000000,
            ],
        })

    def test_configs_have_only_allowed_slots_and_no_omitted_conditions(self):
        study = load_study(STUDY_PATH)
        for config_id, expected in EXPECTED.items():
            _, configuration = prepare_run_design(STUDY_PATH, config_id)
            data = configuration.data
            self.assertEqual(data['algorithm'], expected['algorithm'])
            self.assertEqual(data['placement'], expected['placement'])
            self.assertEqual(
                data['historical_counterpart']['config_id'],
                expected['historical_config_id'],
            )
            self.assertEqual(data['factors']['training_horizon'], 1000000)
            self.assertEqual(data['factors']['topology'], 'two_state')
            self.assertEqual(data['factors']['block'], 'mlp')
            self.assertEqual(data['factors']['credit'], 'full_bptt')
            self.assertEqual(data['factors']['h_cycles'], expected['h_cycles'])
            self.assertEqual(data['factors']['l_cycles'], expected['l_cycles'])
            self.assertNotIn('one_step', data['slug'])
            compute = data['agent_overrides']['compute']
            enabled_slots = tuple(
                slot_name for slot_name, slot in compute.items()
                if slot.get('enabled', False)
            )
            self.assertEqual(enabled_slots, expected['enabled_slots'])
            for slot_name in compute:
                if slot_name not in expected['enabled_slots']:
                    self.assertFalse(compute[slot_name].get('enabled', False))

            if expected['algorithm'] == 'hiql':
                self.assertIn('value', compute)
                self.assertFalse(compute['value']['enabled'])
            else:
                self.assertTrue(compute['actor']['enabled'])
                self.assertFalse(compute['critic_state']['enabled'])
                self.assertFalse(compute['critic_goal']['enabled'])
                self.assertFalse(compute['value_state']['enabled'])
                self.assertFalse(compute['value_goal']['enabled'])

        self.assertEqual(study.data['environments'], ['antmaze-large-navigate-v0'])
        self.assertNotIn('antmaze-medium-navigate-v0', study.data['environments'])
        placements = [
            load_configuration(study, path).data['placement']
            for path in (STUDY_PATH.parent / 'configs').glob('*.yaml')
        ]
        self.assertNotIn('baseline', placements)

    def test_historical_compute_configuration_parity(self):
        for config_id, expected in EXPECTED.items():
            historical_id = expected['historical_config_id']
            new_snapshot = _resolved_compute(STUDY_PATH, config_id, expected['algorithm'])
            historical_snapshot = _resolved_compute(
                HISTORICAL_STUDY_PATH,
                historical_id,
                expected['algorithm'],
            )
            self.assertEqual(new_snapshot, historical_snapshot)

            _, new_configuration = prepare_run_design(STUDY_PATH, config_id)
            _, historical_configuration = prepare_run_design(HISTORICAL_STUDY_PATH, historical_id)
            for factor in ('topology', 'block', 'schedule', 'h_cycles', 'l_cycles', 'credit', 'placement'):
                self.assertEqual(
                    new_configuration.data['factors'][factor],
                    historical_configuration.data['factors'][factor],
                    msg=f'{config_id} factor {factor}',
                )


if __name__ == '__main__':
    unittest.main()
