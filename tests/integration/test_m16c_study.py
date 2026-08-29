"""Declarative tests for the M16C Puzzle-4x4 S002 alpha sweep."""

import unittest
from pathlib import Path

from impls.experiment import load_study, prepare_run_design
from impls.main import _make_config, _parse_args


STUDY = Path(__file__).parents[2] / 'experiments/M16C_puzzle_4x4_mixer_alpha_sweep/study.yaml'
CONFIG_DIR = STUDY.parent / 'configs'
EXPECTED_ALPHAS = (0.1, 0.2, 0.5, 0.7)


class M16CStudyTest(unittest.TestCase):
    def test_exact_s002_alpha_matrix_and_runtime_resolution(self):
        study = load_study(STUDY)
        self.assertEqual(study.study_id, 'M16C')
        self.assertEqual(study.data['environments'], ['puzzle-4x4-play-v0'])
        self.assertEqual(study.data['conditions'], ['S002'])
        self.assertEqual(study.data['seeds'], [0])
        self.assertEqual(tuple(study.data['alpha_policy']['new_scanned_values']), EXPECTED_ALPHAS)
        self.assertEqual(len(list(CONFIG_DIR.glob('*.yaml'))), len(EXPECTED_ALPHAS))

        args = _parse_args(['--agent', 'gciql'])
        observed = []
        fixed_reference = None
        for path in sorted(CONFIG_DIR.glob('*.yaml')):
            _, configuration = prepare_run_design(STUDY, path)
            data = configuration.data
            self.assertEqual(data['environment'], 'puzzle-4x4-play-v0')
            self.assertEqual(data['condition_id'], 'S002')
            factor_alpha = float(data['factors']['alpha'])
            self.assertEqual(float(data['agent_overrides']['alpha']), factor_alpha)
            config = _make_config(args, configuration=configuration)
            self.assertEqual(float(config['alpha']), factor_alpha)
            observed.append(factor_alpha)
            fixed = {key: value for key, value in config.items() if key != 'alpha'}
            if fixed_reference is None:
                fixed_reference = fixed
            else:
                self.assertEqual(fixed, fixed_reference)

        self.assertEqual(tuple(sorted(observed)), EXPECTED_ALPHAS)


if __name__ == '__main__':
    unittest.main()
