"""Declarative tests for the M16B Puzzle alpha correction Study."""

import unittest
from pathlib import Path

from impls.experiment import load_study, prepare_run_design
from impls.main import _make_config, _parse_args


STUDY = Path(__file__).parents[2] / 'experiments/M16B_puzzle_alpha_correction/study.yaml'
CONFIG_DIR = STUDY.parent / 'configs'


class M16BStudyTest(unittest.TestCase):
    def test_exact_matrix_and_explicit_alpha(self):
        study = load_study(STUDY)
        self.assertEqual(study.study_id, 'M16B')
        self.assertEqual(study.data['conditions'], ['B000', 'S002'])
        self.assertEqual(study.data['seeds'], [0])
        self.assertEqual(study.data['fixed_design']['alpha'], 1.0)
        self.assertEqual(study.data['alpha_policy']['value'], 1.0)
        self.assertEqual(len(list(CONFIG_DIR.glob('*.yaml'))), 8)

        cells = set()
        args = _parse_args(['--agent', 'gciql'])
        for path in sorted(CONFIG_DIR.glob('*.yaml')):
            _, configuration = prepare_run_design(STUDY, path)
            cell = (configuration.data['environment'], configuration.data['condition_id'])
            self.assertNotIn(cell, cells)
            cells.add(cell)
            self.assertEqual(configuration.data['factors']['alpha'], 1.0)
            self.assertEqual(configuration.data['agent_overrides']['alpha'], 1.0)
            self.assertEqual(_make_config(args, configuration=configuration)['alpha'], 1.0)

        expected = {
            (environment, condition)
            for environment in study.data['environments']
            for condition in study.data['conditions']
        }
        self.assertEqual(cells, expected)


if __name__ == '__main__':
    unittest.main()
