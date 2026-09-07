"""Study readiness and sampling-regression tests for frozen M20A Phase 2."""

import contextlib
import io
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from impls.experiment import load_study
from impls.utils.datasets import Dataset, GCDataset
from tools import analyze_m20a, m20a_doctor, sweep


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'experiments/M20A_cross_task_oracle_relation_diagnosis/study.yaml'


def _dataset_config():
    return {
        'frame_stack': None,
        'value_p_curgoal': 0.2,
        'value_p_trajgoal': 0.5,
        'value_p_randomgoal': 0.3,
        'value_geom_sample': True,
        'actor_p_curgoal': 0.0,
        'actor_p_trajgoal': 1.0,
        'actor_p_randomgoal': 0.0,
        'actor_geom_sample': False,
        'discount': 0.99,
        'gc_negative': True,
        'p_aug': 0.0,
    }


class M20APhase2StudyTest(unittest.TestCase):
    def test_exact_executable_factorial_schema(self):
        report = m20a_doctor.check_study_schema(STUDY)
        self.assertEqual(report['config_count'], 18)
        self.assertEqual(len(report['config_ids']), 18)
        study = load_study(STUDY)
        self.assertEqual(study.data['seeds'], [0])
        self.assertEqual(study.data['alpha_policy']['value'], 1.0)
        self.assertEqual(study.data['phase2_readiness']['formal_executable_runs'], 18)
        self.assertEqual(study.data['phase2_readiness']['blocked_phase2_skeletons'], 0)

    def test_sweep_marks_every_frozen_cell_planned_in_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                result = sweep.main([
                    '--study', str(STUDY), '--run-root', str(Path(directory) / 'runs'),
                    '--gpus', '0', '--dry-run',
                ])
        output = stream.getvalue()
        self.assertEqual(result, 0)
        self.assertIn('total=18 planned=18', output)
        self.assertIn('remaining=18', output)
        self.assertEqual(output.count('[PLANNED] M20A-'), 18)
        self.assertNotIn('blocked phase2 skeletons', output)

    def test_analyzer_skeleton_has_no_cross_task_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            report = analyze_m20a.collect(STUDY, Path(directory) / 'runs')
        self.assertFalse(report['formal_results_present'])
        self.assertEqual(len(report['cells']), 18)
        self.assertEqual(report['analysis_scope']['cross_task_raw_success_pooling'], 'prohibited')
        self.assertEqual(set(report['within_task_contrasts']), {'puzzle', 'cube', 'scene'})
        self.assertTrue(all(row['curve_status'] == 'missing' for row in report['cells']))

    def test_gcdataset_sampling_trace_is_opt_in_and_rng_neutral(self):
        observations = np.arange(24, dtype=np.float32).reshape(6, 4)
        actions = np.zeros((6, 2), dtype=np.float32)
        terminals = np.asarray([0, 0, 1, 0, 0, 1], dtype=np.float32)
        raw = Dataset.create(observations=observations, actions=actions, terminals=terminals)
        first = GCDataset(raw, _dataset_config(), rng=np.random.default_rng(20020))
        second = GCDataset(raw, _dataset_config(), rng=np.random.default_rng(20020))
        default_batch = first.sample(3)
        self.assertIsInstance(default_batch, Mapping)
        batch, trace = second.sample(3, return_sampling_trace=True)
        self.assertEqual(set(trace), {'transition_indices', 'value_goal_indices', 'actor_goal_indices'})
        self.assertEqual(batch['actor_goals'].shape, (3, 4))
        # A separate pair with identical seed verifies trace IDs and goals.
        left = GCDataset(raw, _dataset_config(), rng=np.random.default_rng(7))
        right = GCDataset(raw, _dataset_config(), rng=np.random.default_rng(7))
        left_batch, left_trace = left.sample(4, return_sampling_trace=True)
        right_batch, right_trace = right.sample(4, return_sampling_trace=True)
        for key in left_trace:
            np.testing.assert_array_equal(left_trace[key], right_trace[key])
        np.testing.assert_array_equal(left_batch['actor_goals'], right_batch['actor_goals'])
        np.testing.assert_array_equal(left_batch['value_goals'], right_batch['value_goals'])


if __name__ == '__main__':
    unittest.main()
