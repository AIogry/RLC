"""Declarative and resolved-runtime tests for the M16D Study."""

import tempfile
import unittest
from pathlib import Path

from impls.experiment import load_study, prepare_run_design
from impls.main import _make_config, _parse_args
from tools import m16d_doctor


REPO_ROOT = Path(__file__).parents[2]
STUDY = REPO_ROOT / 'experiments/M16D_puzzle_mixer_alpha04_completion/study.yaml'
ANCHOR_STUDY = REPO_ROOT / 'experiments/M16B_puzzle_alpha_correction/study.yaml'
CONFIG_DIR = STUDY.parent / 'configs'


class M16DStudyTest(unittest.TestCase):
    def test_exact_four_cell_matrix_and_alpha(self):
        study = load_study(STUDY)
        self.assertEqual(study.study_id, 'M16D')
        self.assertEqual(study.data['conditions'], ['S002'])
        self.assertEqual(study.data['seeds'], [0])
        self.assertEqual(study.data['fixed_design']['alpha'], 0.4)
        self.assertEqual(study.data['alpha_policy']['value'], 0.4)
        self.assertFalse(study.data['protocol']['formal_training_started'])
        self.assertEqual(len(list(CONFIG_DIR.glob('*.yaml'))), 4)

        expected = {
            ('puzzle-3x3-play-v0', 'M16D-P3X3-S002-A04', 9, 55),
            ('puzzle-4x4-play-v0', 'M16D-P4X4-S002-A04', 16, 83),
            ('puzzle-4x5-play-v0', 'M16D-P4X5-S002-A04', 20, 99),
            ('puzzle-4x6-play-v0', 'M16D-P4X6-S002-A04', 24, 115),
        }
        observed = set()
        args = _parse_args(['--agent', 'gciql'])
        for path in sorted(CONFIG_DIR.glob('*.yaml')):
            _, configuration = prepare_run_design(STUDY, path)
            data = configuration.data
            self.assertEqual(data['study_id'], 'M16D')
            self.assertEqual(data['condition_id'], 'S002')
            self.assertEqual(data['factors']['alpha'], 0.4)
            self.assertEqual(data['agent_overrides']['alpha'], 0.4)
            config = _make_config(args, configuration=configuration)
            self.assertEqual(config['alpha'], 0.4)
            observed.add((
                data['environment'],
                data['config_id'],
                data['factors']['num_buttons'],
                data['factors']['num_buttons'] * 4 + 19,
            ))
        self.assertEqual(observed, expected)

    def test_m16b_protocol_and_resolved_s002_parity(self):
        m16d = load_study(STUDY)
        m16b = load_study(ANCHOR_STUDY)
        self.assertEqual(m16d.data['protocol'], m16b.data['protocol'])
        self.assertEqual(
            {key: value for key, value in m16d.data['fixed_design'].items() if key != 'alpha'},
            {key: value for key, value in m16b.data['fixed_design'].items() if key != 'alpha'},
        )
        args = _parse_args(['--agent', 'gciql'])
        for environment, anchor_id in m16d_doctor.ANCHOR_CONFIGS.items():
            candidate_id = m16d_doctor.EXPECTED_CONFIGS[environment]
            _, candidate = prepare_run_design(STUDY, CONFIG_DIR / f'{candidate_id}.yaml')
            _, anchor = prepare_run_design(
                ANCHOR_STUDY,
                ANCHOR_STUDY.parent / 'configs' / f'{anchor_id}.yaml',
            )
            candidate_runtime = _make_config(args, configuration=candidate)
            anchor_runtime = _make_config(args, configuration=anchor)
            self.assertEqual(candidate_runtime['alpha'], 0.4)
            self.assertEqual(anchor_runtime['alpha'], 1.0)
            candidate_json = m16d_doctor.jsonable(candidate_runtime)
            anchor_json = m16d_doctor.jsonable(anchor_runtime)
            candidate_json['alpha'] = anchor_json['alpha']
            self.assertEqual(candidate_json, anchor_json, msg=environment)

    def test_doctor_preflight_is_non_mutating_and_has_no_conflicting_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / 'dataset'
            dataset_root.mkdir()
            for environment in m16d_doctor.ENVIRONMENTS:
                (dataset_root / f'{environment}.npz').touch()
                (dataset_root / f'{environment}-val.npz').touch()
            run_root = root / 'runs'
            report = m16d_doctor.validate(
                STUDY,
                ANCHOR_STUDY,
                dataset_root,
                run_root,
                gpu='0',
            )
            self.assertEqual(report['status'], 'PASS')
            self.assertEqual(report['expected_formal_runs'], 4)
            self.assertEqual(report['configs'], sorted(m16d_doctor.EXPECTED_CONFIGS.values()))
            self.assertEqual(report['seeds'], [0])
            self.assertEqual(report['gpu_policy'], {'physical_gpu': '0', 'jobs_per_gpu': 2})
            self.assertFalse(run_root.exists())


if __name__ == '__main__':
    unittest.main()
