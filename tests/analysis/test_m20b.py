"""Unit tests for the M20B within-environment analyzer."""

import csv
import tempfile
import unittest
from pathlib import Path

from tools import analyze_m20b
from impls.experiment import load_configuration, load_study, make_run_path


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'experiments/M20B_cube_entity_mixer_scaling/study.yaml'


class M20BAnalyzerTest(unittest.TestCase):
    def _write_curve(self, run_root, config_id, values):
        study = load_study(STUDY)
        configuration = load_configuration(study, config_id)
        run_dir = make_run_path(
            run_root,
            study.study_id,
            config_id,
            configuration.slug,
            configuration.data['environment'],
            0,
        )
        run_dir.mkdir(parents=True)
        task_names = analyze_m20b.ENV_SPECS[configuration.data['environment']]['tasks']
        fields = ['step', analyze_m20b.METRIC] + [
            f'evaluation/{name}_success' for name in task_names
        ]
        with (run_dir / 'eval.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for index, value in enumerate(values, start=1):
                row = {'step': index * 100_000, analyze_m20b.METRIC: value}
                row.update({field: value / 2 for field in fields[2:]})
                writer.writerow(row)

    def test_complete_curve_and_earliest_best_tie(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / 'runs'
            values = [0.1, 0.9, 0.9, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 0.7]
            self._write_curve(run_root, 'M20B-double-B000', values)
            self._write_curve(run_root, 'M20B-double-S002', [value + 0.1 for value in values])
            report = analyze_m20b.collect(STUDY, run_root)
            flat = next(row for row in report['cells'] if row['config_id'] == 'M20B-double-B000')
            contrast = report['within_environment_contrasts']['cube-double-play-v0']
            self.assertEqual(flat['curve_status'], 'complete')
            self.assertEqual(flat['best_step'], 200_000)
            self.assertAlmostEqual(flat['last3_mean'], (0.8 + 0.8 + 0.7) / 3)
            self.assertAlmostEqual(flat['normalized_auc'], 0.6666666667, places=6)
            self.assertEqual(len(flat['per_task_final_at_1m']), 5)
            self.assertAlmostEqual(contrast['raw_delta'], 0.1)
            self.assertAlmostEqual(report['delta_double'], 0.1)

    def test_missing_auc_point_is_incomplete_without_interpolation(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / 'runs'
            self._write_curve(run_root, 'M20B-single-B000', [0.1] * 10)
            self._write_curve(run_root, 'M20B-single-S002', [0.2] * 10)
            path = next(
                run_root.rglob('M20B-single-S002*/cube-single-play-v0/seed_000/eval.csv')
            )
            lines = path.read_text().splitlines()
            path.write_text('\n'.join(lines[:-2]) + '\n')
            report = analyze_m20b.collect(STUDY, run_root)
            mixer = next(row for row in report['cells'] if row['config_id'] == 'M20B-single-S002')
            self.assertEqual(mixer['curve_status'], 'incomplete_curve')
            self.assertIsNone(mixer['normalized_auc'])
            self.assertEqual(mixer['auc_status'], 'incomplete_curve')


if __name__ == '__main__':
    unittest.main()
