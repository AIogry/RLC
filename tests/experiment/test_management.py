import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from impls.experiment import (
    aggregate_manifest,
    create_run_context,
    finalize_run,
    load_configuration,
    load_study,
    make_run_path,
    prepare_run_design,
    summarize_eval_csv,
    write_manifest,
)
from impls.experiment.management import config_fingerprint, git_metadata, jsonable


STUDY_YAML = """
study_id: TEST
name: Test study
question: Does the test factor matter?
primary_factors: [iterations, residual]
fixed_design:
  state_scope: decision_local
deferred_factors: [gating]
algorithms: [hiql]
placements: [low_actor]
environments: [toy-v0]
seeds: [0, 1]
primary_metric: evaluation/overall_success
"""

CONFIG_YAML = """
study_id: TEST
config_id: TEST-C001
slug: k1_control
factors:
  internal_iterations_K: 1
  residual: false
"""


class ExperimentManagementTest(unittest.TestCase):
    def _fixture(self):
        root = Path(tempfile.mkdtemp())
        study_dir = root / 'experiments' / 'toy'
        config_dir = study_dir / 'configs'
        config_dir.mkdir(parents=True)
        (study_dir / 'study.yaml').write_text(STUDY_YAML)
        (config_dir / 'TEST-C001.yaml').write_text(CONFIG_YAML)
        return root, study_dir / 'study.yaml', config_dir / 'TEST-C001.yaml'

    def test_study_and_configuration_parsing(self):
        _, study_path, config_path = self._fixture()
        study, configuration = prepare_run_design(study_path, config_path)
        self.assertEqual(study.study_id, 'TEST')
        self.assertEqual(configuration.config_id, 'TEST-C001')
        self.assertFalse('seed' in configuration.data)

    def test_stable_run_identity(self):
        path = make_run_path('runs', 'TEST', 'TEST-C001', 'k1_control', 'toy-v0', 7)
        self.assertEqual(
            path.as_posix(),
            'runs/TEST/TEST-C001__k1_control/toy-v0/seed_007',
        )

    def test_resolved_config_and_runtime_metadata(self):
        root, study_path, config_path = self._fixture()
        study = load_study(study_path)
        configuration = load_configuration(study, config_path)
        context = create_run_context(
            study=study,
            configuration=configuration,
            run_root=root / 'runs',
            algorithm='hiql',
            environment='toy-v0',
            seed=0,
            dataset_dir=root / 'data',
            computation=False,
            compute_slots={'low_actor': {'enabled': False}},
            resolved_config={'tuple': (1, 2), 'array': np.asarray([3])},
            repo_root=root,
        )
        resolved = json.loads((context.run_dir / 'resolved_config.json').read_text())
        metadata = json.loads((context.run_dir / 'runtime_metadata.json').read_text())
        self.assertEqual(resolved['configuration']['config_id'], 'TEST-C001')
        self.assertEqual(metadata['study_id'], 'TEST')
        self.assertEqual(metadata['config_id'], 'TEST-C001')
        self.assertEqual(metadata['compute_slots']['low_actor']['enabled'], False)
        self.assertIn('git_commit', metadata)
        self.assertIn('jax_backend', metadata)
        self.assertEqual(
            resolved['resolved_config_fingerprint'],
            metadata['resolved_config_fingerprint'],
        )
        self.assertIn('training_protocol', metadata)
        self.assertEqual(jsonable(np.int32(3)), 3)

    def test_config_fingerprint_ignores_mapping_order(self):
        self.assertEqual(
            config_fingerprint({'b': 2, 'a': [1, 3]}),
            config_fingerprint({'a': [1, 3], 'b': 2}),
        )
        self.assertNotEqual(
            config_fingerprint({'a': 1}),
            config_fingerprint({'a': 2}),
        )

    def test_duplicate_run_identity_fails_fast(self):
        root, study_path, config_path = self._fixture()
        study, configuration = prepare_run_design(study_path, config_path)
        kwargs = {
            'study': study,
            'configuration': configuration,
            'run_root': root / 'runs',
            'algorithm': 'hiql',
            'environment': 'toy-v0',
            'seed': 0,
            'repo_root': root,
        }
        create_run_context(**kwargs)
        with self.assertRaises(FileExistsError):
            create_run_context(**kwargs)

    def test_git_helper_shape(self):
        metadata = git_metadata(Path(__file__).resolve().parents[2])
        self.assertIn('git_commit', metadata)
        self.assertIn('git_dirty', metadata)

    def test_summary_and_failed_run_are_retained_in_manifest(self):
        root, study_path, config_path = self._fixture()
        study, configuration = prepare_run_design(study_path, config_path)
        context = create_run_context(
            study=study,
            configuration=configuration,
            run_root=root / 'runs',
            algorithm='hiql',
            environment='toy-v0',
            seed=0,
            repo_root=root,
        )
        with (context.run_dir / 'eval.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['step', 'evaluation/overall_success'])
            writer.writeheader()
            writer.writerow({'step': 1, 'evaluation/overall_success': 0.25})
            writer.writerow({'step': 2, 'evaluation/overall_success': 0.5})
        summary = finalize_run(context.run_dir, 'failed', 'synthetic failure')
        self.assertEqual(summary['final_success'], 0.5)
        self.assertEqual(summary['best_step'], 2)
        self.assertTrue((context.run_dir / 'failure.json').exists())
        manifest = write_manifest(study_path, root / 'runs', repo_root=root)
        with manifest.open(newline='') as file:
            rows = list(csv.DictReader(file))
        observed = next(row for row in rows if row['seed'] == '0')
        self.assertEqual(observed['status'], 'failed')
        self.assertEqual(observed['best_success'], '0.5')
        self.assertEqual(sum(row['status'] == 'planned' for row in rows), 1)

    def test_aggregation_mean_std(self):
        root = Path(tempfile.mkdtemp())
        manifest = root / 'manifest.csv'
        with manifest.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['config_id', 'slug', 'environment', 'final_success'])
            writer.writeheader()
            writer.writerow({'config_id': 'TEST-C001', 'slug': 'control', 'environment': 'toy-v0', 'final_success': 0.25})
            writer.writerow({'config_id': 'TEST-C001', 'slug': 'control', 'environment': 'toy-v0', 'final_success': 0.75})
        output = aggregate_manifest(manifest)
        with output.open(newline='') as file:
            row = next(csv.DictReader(file))
        self.assertEqual(row['count'], '2')
        self.assertAlmostEqual(float(row['mean']), 0.5)
        self.assertAlmostEqual(float(row['std']), 0.25)

    def test_missing_success_column_is_transparent(self):
        root = Path(tempfile.mkdtemp())
        eval_path = root / 'eval.csv'
        eval_path.write_text('step,other_metric\n1,3\n')
        summary = summarize_eval_csv(eval_path)
        self.assertIsNone(summary['final_success'])
        self.assertIsNone(summary['best_success'])
        self.assertIsNone(summary['best_step'])


if __name__ == '__main__':
    unittest.main()
