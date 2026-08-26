"""Declarative and metric-contract tests for the M16A Study."""

import unittest
from pathlib import Path

from impls.experiment import load_study, prepare_run_design
from tools.analyze_m16a import EXPECTED_STEPS, _curve_summary


STUDY = Path(__file__).parents[2] / 'experiments/M16A_puzzle_mixer_depth_scaling/study.yaml'
CONFIG_DIR = STUDY.parent / 'configs'


class M16AStudyTest(unittest.TestCase):
    def test_exact_matrix_and_unique_cells(self):
        study = load_study(STUDY)
        self.assertEqual(study.study_id, 'M16A')
        self.assertEqual(study.data['seeds'], [0])
        self.assertEqual(len(list(CONFIG_DIR.glob('*.yaml'))), 16)
        cells = set()
        for path in sorted(CONFIG_DIR.glob('*.yaml')):
            configuration = prepare_run_design(STUDY, path)[1]
            cell = (configuration.data['environment'], configuration.data['condition_id'])
            self.assertNotIn(cell, cells)
            cells.add(cell)
        self.assertEqual(len(cells), 16)

    def test_auc_and_endpoint_contract(self):
        rows = [(step, step / 1_000_000.0) for step in EXPECTED_STEPS]
        summary = _curve_summary(rows)
        self.assertEqual(summary['status'], 'complete')
        self.assertEqual(summary['final_success'], 1.0)
        self.assertEqual(summary['best_step'], 1_000_000)
        self.assertAlmostEqual(summary['normalized_eval_auc'], 0.55)
        partial = _curve_summary(rows[:-1])
        self.assertEqual(partial['status'], 'partial')
        self.assertIsNone(partial['final_success'])
        self.assertIsNone(partial['normalized_eval_auc'])


if __name__ == '__main__':
    unittest.main()
