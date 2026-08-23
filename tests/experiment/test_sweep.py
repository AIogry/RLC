import threading
import time
import tempfile
import unittest
from pathlib import Path

from tools import sweep


STUDY_YAML = """
study_id: TEST
name: Test study
question: Does the test factor matter?
primary_factors: [variant]
fixed_design: {}
deferred_factors: [future]
algorithms: [hiql]
placements: [baseline]
environments: [toy-a, toy-b]
seeds: [0]
primary_metric: evaluation/overall_success
protocol:
  stage1:
    save_best_checkpoint: false
    save_last_checkpoint: true
"""

CONFIG_TEMPLATE = """
study_id: TEST
config_id: {config_id}
slug: {slug}
algorithm: hiql
stage: stage1
protocol_stage: stage1
factors:
  variant: {variant}
"""


class SweepInfrastructureTest(unittest.TestCase):
    def _fixture(self):
        root = Path(tempfile.mkdtemp())
        study_dir = root / 'experiments' / 'toy'
        config_dir = study_dir / 'configs'
        config_dir.mkdir(parents=True)
        (study_dir / 'study.yaml').write_text(STUDY_YAML)
        (config_dir / 'TEST-C001.yaml').write_text(
            CONFIG_TEMPLATE.format(config_id='TEST-C001', slug='control', variant=0)
        )
        (config_dir / 'TEST-C002.yaml').write_text(
            CONFIG_TEMPLATE.format(config_id='TEST-C002', slug='variant', variant=1)
        )
        dataset_dir = root / 'data'
        dataset_dir.mkdir()
        for environment in ('toy-a', 'toy-b'):
            (dataset_dir / f'{environment}.npz').touch()
            (dataset_dir / f'{environment}-val.npz').touch()
        return root, study_dir / 'study.yaml', dataset_dir

    def test_include_and_exclude_filtering(self):
        _, study_path, _ = self._fixture()
        included = sweep._jobs(
            study_path,
            '/tmp/test-sweep',
            include_configs={'TEST-C001'},
        )
        excluded = sweep._jobs(
            study_path,
            '/tmp/test-sweep',
            exclude_configs={'TEST-C001'},
        )
        self.assertEqual(len(included), 2)
        self.assertEqual({job['configuration'].config_id for job in included}, {'TEST-C001'})
        self.assertEqual(len(excluded), 2)
        self.assertEqual({job['configuration'].config_id for job in excluded}, {'TEST-C002'})

    def test_invalid_and_mutually_exclusive_filters_fail(self):
        _, study_path, _ = self._fixture()
        with self.assertRaises(SystemExit):
            sweep._jobs(study_path, '/tmp/test-sweep', include_configs={'NOPE'})
        with self.assertRaises(SystemExit):
            sweep._jobs(
                study_path,
                '/tmp/test-sweep',
                include_configs={'TEST-C001'},
                exclude_configs={'TEST-C002'},
            )

    def test_dataset_preflight_requires_train_and_validation_files(self):
        _, study_path, dataset_dir = self._fixture()
        required = sweep._validate_dataset(study_path, dataset_dir)
        self.assertEqual(len(required), 4)
        (dataset_dir / 'toy-b-val.npz').unlink()
        with self.assertRaises(SystemExit):
            sweep._validate_dataset(study_path, dataset_dir)

    def test_dynamic_scheduler_has_one_active_run_per_gpu(self):
        jobs = list(range(6))
        lock = threading.Lock()
        active = {'0': 0, '1': 0}
        max_active = {'global': 0, '0': 0, '1': 0}
        assignments = []

        def runner(job, gpu, run_root, extra_args):
            with lock:
                active[gpu] += 1
                max_active['global'] = max(max_active['global'], sum(active.values()))
                max_active[gpu] = max(max_active[gpu], active[gpu])
                assignments.append((job, gpu))
            time.sleep(0.01 if gpu == '0' else 0.08)
            with lock:
                active[gpu] -= 1
            return 0

        self.assertEqual(
            sweep._dispatch_jobs(jobs, ['0', '1'], '/tmp/test-sweep', [], runner=runner),
            0,
        )
        self.assertEqual(len(assignments), len(jobs))
        self.assertLessEqual(max_active['global'], 2)
        self.assertLessEqual(max_active['0'], 1)
        self.assertLessEqual(max_active['1'], 1)
        # The fast worker should take work after its first job while GPU 1 is
        # still occupied, demonstrating a dynamic free-GPU queue.
        self.assertGreater(sum(gpu == '0' for _, gpu in assignments), 1)

    def test_study_protocol_fills_omitted_checkpoint_flags(self):
        _, study_path, _ = self._fixture()
        job = sweep._jobs(study_path, '/tmp/test-sweep')[0]
        command = sweep._command(job, '/tmp/test-sweep', [])
        self.assertIn('--no-save-best-checkpoint', command)
        self.assertIn('--save-last-checkpoint', command)
        explicit = sweep._command(
            job,
            '/tmp/test-sweep',
            ['--save-best-checkpoint', '--no-save-last-checkpoint'],
        )
        self.assertEqual(explicit.count('--save-best-checkpoint'), 1)
        self.assertEqual(explicit.count('--no-save-last-checkpoint'), 1)


if __name__ == '__main__':
    unittest.main()
