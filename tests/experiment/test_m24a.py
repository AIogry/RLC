"""Declarative and resolved-runtime gates for the M24A Study."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from impls.experiment import load_study, make_run_path, prepare_run_design
from impls.experiment.management import jsonable
from impls.main import _computation_runtime_extras, _make_config, _parse_args
from tools import m24a_doctor


ROOT = Path(__file__).resolve().parents[2]
STUDY = (
    ROOT
    / 'experiments/M24A_puzzle_goal_conditioning_intervention/study.yaml'
)
CONFIG_DIR = STUDY.parent / 'configs'


def _resolved_configs():
    args = _parse_args(['--agent', 'gciql'])
    result = {}
    for path in sorted(CONFIG_DIR.glob('*.yaml')):
        _, configuration = prepare_run_design(STUDY, path)
        result[configuration.config_id] = (
            configuration,
            _make_config(args, configuration=configuration),
        )
    return result


class M24AStudyTest(unittest.TestCase):
    def test_exact_six_cell_matrix_and_frozen_scientific_contract(self):
        study = load_study(STUDY)
        m24a_doctor._check_study(study)
        resolved = _resolved_configs()

        self.assertEqual(set(resolved), set(m24a_doctor.EXPECTED_CONFIGS))
        self.assertEqual(study.data['seeds'], [0])
        self.assertEqual(study.data['conditions'], ['G1', 'G2', 'G4'])
        self.assertFalse(study.data['protocol']['formal_training_started'])
        self.assertIsNone(study.data['execution']['physical_gpu'])

        observed = {
            config_id: (
                configuration.data['environment'],
                configuration.data['condition_id'],
                config['goal_conditioning']['mode'],
            )
            for config_id, (configuration, config) in resolved.items()
        }
        expected = {
            config_id: (
                environment,
                condition,
                m24a_doctor.CONDITIONS[condition]['mode'],
            )
            for config_id, (environment, condition) in
            m24a_doctor.EXPECTED_CONFIGS.items()
        }
        self.assertEqual(observed, expected)
        self.assertTrue(all(
            'seed' not in item[0].data for item in resolved.values()
        ))

    def test_only_conditioning_mode_changes_within_each_environment(self):
        resolved = _resolved_configs()
        for environment in m24a_doctor.ENVIRONMENTS:
            common_by_condition = {}
            for condition in m24a_doctor.CONDITIONS:
                config_id = next(
                    config_id
                    for config_id, cell in m24a_doctor.EXPECTED_CONFIGS.items()
                    if cell == (environment, condition)
                )
                configuration, config = resolved[config_id]
                common_configuration = copy.deepcopy(configuration.data)
                for field in (
                    'condition_id',
                    'semantic_condition',
                    'description',
                    'slug',
                    'config_id',
                ):
                    common_configuration[field] = '<condition>'
                factors = common_configuration['factors']
                for field in (
                    'condition',
                    'goal_conditioning_mode',
                    'goal_coordinate',
                ):
                    factors[field] = '<condition>'
                common_configuration['agent_overrides'][
                    'goal_conditioning'
                ]['mode'] = '<condition>'
                common_by_condition[condition] = (
                    jsonable(common_configuration),
                    config,
                )

            reference_configuration, reference_config = common_by_condition['G1']
            for condition in ('G2', 'G4'):
                candidate_configuration, candidate_config = (
                    common_by_condition[condition]
                )
                self.assertEqual(
                    candidate_configuration, reference_configuration
                )
                reference_common = copy.deepcopy(jsonable(reference_config))
                candidate_common = copy.deepcopy(jsonable(candidate_config))
                reference_common['goal_conditioning']['mode'] = '<condition>'
                candidate_common['goal_conditioning']['mode'] = '<condition>'
                self.assertEqual(candidate_common, reference_common)

    def test_runtime_metadata_exposes_board_success_and_full_rank_coordinates(self):
        resolved = _resolved_configs()
        for config_id, (configuration, config) in resolved.items():
            environment, condition = m24a_doctor.EXPECTED_CONFIGS[config_id]
            runtime = _computation_runtime_extras(config)
            self.assertEqual(
                runtime['goal_success_semantics'], 'board_equality'
            )
            self.assertEqual(
                runtime['goal_conditioning']['mode'],
                m24a_doctor.CONDITIONS[condition]['mode'],
            )
            self.assertTrue(
                runtime['goal_conditioning']['operator_full_rank']
            )
            self.assertEqual(
                runtime['goal_conditioning']['operator_rank'],
                m24a_doctor.ENVIRONMENTS[environment]['num_buttons'],
            )
            self.assertEqual(
                configuration.data['agent_overrides']['dataset_class'],
                'PuzzleBoardGCDataset',
            )

    def test_run_identities_are_unique_and_do_not_assign_a_physical_gpu(self):
        study = load_study(STUDY)
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / 'runs'
            paths = []
            for config_path in sorted(CONFIG_DIR.glob('*.yaml')):
                _, configuration = prepare_run_design(STUDY, config_path)
                paths.append(make_run_path(
                    run_root,
                    study.study_id,
                    configuration.config_id,
                    configuration.slug,
                    configuration.data['environment'],
                    seed=0,
                    run_attempt=0,
                ))

            self.assertEqual(len(paths), 6)
            self.assertEqual(len(set(paths)), 6)
            self.assertFalse(run_root.exists())
        self.assertIsNone(study.data['execution']['physical_gpu'])
        self.assertEqual(study.data['execution']['concurrency'], {
            'policy_role': 'operational_not_scientific',
            'planned_jobs_per_gpu': 2,
            'gate': (
                'exact_workload_concurrent_gpu_smoke_required_before_formal_launch'
            ),
            'suggested_heaviest_pair': ['M24A-C004', 'M24A-C006'],
            'wave_order': [
                ['M24A-C001', 'M24A-C002'],
                ['M24A-C003', 'M24A-C004'],
                ['M24A-C005', 'M24A-C006'],
            ],
        })

    def test_doctor_requires_the_entire_formal_namespace_to_be_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            study_root = Path(directory) / 'runs' / 'M24A'
            study_root.mkdir(parents=True)
            with self.assertRaisesRegex(
                ValueError, 'formal output namespace must be absent'
            ):
                m24a_doctor.validate(
                    STUDY,
                    Path(directory) / 'datasets-not-reached',
                    Path(directory) / 'runs',
                )

    def test_historical_g0_is_descriptive_and_primary_contrasts_are_clean(self):
        study = load_study(STUDY)
        comparison = study.data['comparison_policy']
        self.assertEqual(comparison['historical_G0'], {
            'role': 'descriptive_reference_only',
            'semantics': 'canonical_full_goal_plus_index_equality_success',
            'plotting': 'dashed_gray_reference',
            'prohibited_interpretation': (
                'G0_to_G1_is_not_a_clean_representation_effect'
            ),
        })
        self.assertEqual(
            [
                item['contrast']
                for item in comparison['clean_primary_comparisons']
            ],
            ['G2_minus_G1', 'G4_minus_G2'],
        )
        endpoint = study.data['scientific_endpoints']
        self.assertEqual(
            endpoint['primary_endpoint'], 'final@1M_task_wise_success'
        )
        self.assertEqual(
            endpoint['primary_derived_metric']['equation'],
            'mean(Success_task2, Success_task3, Success_task4, Success_task5)',
        )

    def test_final_diagnosis_is_deferred_and_paired(self):
        diagnosis = load_study(STUDY).data['final_diagnosis']
        self.assertEqual(
            diagnosis['status'],
            'required_after_all_six_last_at_1m_checkpoints_complete',
        )
        self.assertEqual(diagnosis['task_ids'], [1, 2, 3, 4, 5])
        self.assertEqual(diagnosis['episodes_per_task'], 50)
        self.assertEqual(diagnosis['evaluation_seed'], 20260909)
        self.assertEqual(
            diagnosis['pairing_contract'],
            'identical_raw_full_goal_and_initial_condition_within_environment_task_episode',
        )
        self.assertEqual(
            diagnosis['operation_selection_advantage']['symbol'], 'A_useful'
        )
        self.assertEqual(
            diagnosis['launch_authority'], 'user_manual_only_after_training'
        )


if __name__ == '__main__':
    unittest.main()
