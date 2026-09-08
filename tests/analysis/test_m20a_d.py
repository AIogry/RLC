"""Post-hoc-only contract tests for the M20A-D analyzer."""

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.analyze_m20a_d import M20ADAnalysisError, analyze
from tools.m20a_relation_utilization import (
    discover_numeric_checkpoints,
    resolve_authoritative_source_run,
)


class M20ADAnalyzerTest(unittest.TestCase):
    def test_reads_npz_json_only_and_writes_separate_report_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'diagnostics'
            job = root / 'M20AD' / 'relation_utilization' / 'M20A-CUBE-C-Q' / 'checkpoint_100000'
            job.mkdir(parents=True)
            sample_id = np.arange(2, dtype=np.int64)
            arrays = {'sample_id': sample_id}
            for input_name in ('actor', 'value'):
                arrays[f'{input_name}_relation_counts'] = np.asarray([[1, 0, 0], [0, 0, 0]], dtype=np.int32)
                for group_name in (
                    'inactive', 'any_relation_active', 'current_support_active',
                    'goal_support_active', 'goal_conflict_active',
                ):
                    arrays[f'{input_name}_group_{group_name}'] = np.asarray([1, 0], dtype=np.uint8)
                for metric in (
                    'delta_rel', 'relative_relation_update', 'delta_mix',
                    'relative_core_difference', 'propagation_ratio', 'delta_readout',
                    'relative_readout_difference', 'attention_tv',
                ):
                    arrays[f'{input_name}_{metric}'] = np.asarray([0.25, 0.0], dtype=np.float32)
            arrays['actor_delta_action'] = np.asarray([0.25, 0.0], dtype=np.float32)
            arrays['value_delta_value'] = np.asarray([0.25, 0.0], dtype=np.float32)
            npz_path = job / 'per_sample_metrics.npz'
            np.savez_compressed(npz_path, **arrays)

            identity = {
                'source_run_path': str(Path(directory) / 'source-run'),
                'source_resolved_config_fingerprint': 'config-fingerprint',
                'fixed_batch_fingerprint_sha256': 'batch-fingerprint',
                'source_checkpoint_step': 100000,
                'source_checkpoint_sha256': 'checkpoint-fingerprint',
            }
            metadata = {
                'status': 'completed',
                'diagnostic_id': 'M20AD',
                **{key: identity[key] for key in (
                    'source_run_path', 'source_resolved_config_fingerprint',
                    'fixed_batch_fingerprint_sha256',
                )},
            }
            summary = {
                'status': 'completed',
                'diagnostic_id': 'M20AD',
                **identity,
                'normal_forward_parity': {'status': 'pass'},
                'state_integrity': {'unchanged': True},
                'gradient_analysis': {'metrics': {}},
                'parameter_learning': {},
                'artifact_paths': {'per_sample_metrics': str(npz_path)},
            }
            (job / 'm20ad_metadata.json').write_text(json.dumps(metadata))
            (job / 'diagnostic_summary.json').write_text(json.dumps(summary))

            output = Path(directory) / 'posthoc-report'
            report = analyze(root, output)
            self.assertEqual(report['analysis_mode'], 'posthoc_npz_json_only_no_forward')
            self.assertEqual(len(report['outputs']['figures']), 11)
            self.assertTrue(all(Path(path).is_file() for path in report['outputs']['figures']))
            self.assertTrue((output / 'decision_table.csv').is_file())
            with self.assertRaises(FileExistsError):
                analyze(root, output)

    def test_missing_completed_artifact_is_insufficient_for_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(M20ADAnalysisError):
                analyze(Path(directory) / 'empty-diagnostics', Path(directory) / 'report')

    def test_mean_campaign_uses_separate_namespace_and_marks_attention_inapplicable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'diagnostics'
            job = (
                root / 'M20AD' / 'campaigns' / 'M20A-CUBE-C-M'
                / 'relation_utilization' / 'checkpoint_100000'
            )
            job.mkdir(parents=True)
            arrays = {
                'sample_id': np.arange(2, dtype=np.int64),
                'actor_relation_counts': np.asarray([[1, 1, 1], [0, 0, 0]], dtype=np.int32),
                'value_relation_counts': np.asarray([[1, 1, 1], [0, 0, 0]], dtype=np.int32),
                'actor_group_inactive': np.asarray([0, 1], dtype=np.uint8),
                'actor_group_any_relation_active': np.asarray([1, 0], dtype=np.uint8),
                'value_group_inactive': np.asarray([0, 1], dtype=np.uint8),
                'value_group_any_relation_active': np.asarray([1, 0], dtype=np.uint8),
                'actor_delta_rel': np.asarray([0.2, 0.0], dtype=np.float32),
                'actor_delta_mix': np.asarray([0.3, 0.0], dtype=np.float32),
                'actor_propagation_ratio': np.asarray([1.5, np.nan], dtype=np.float32),
                'actor_delta_readout': np.asarray([0.1, 0.0], dtype=np.float32),
                'actor_delta_action': np.asarray([0.05, 0.0], dtype=np.float32),
                'value_delta_rel': np.asarray([0.2, 0.0], dtype=np.float32),
                'value_delta_mix': np.asarray([0.3, 0.0], dtype=np.float32),
                'value_propagation_ratio': np.asarray([1.5, np.nan], dtype=np.float32),
                'value_delta_readout': np.asarray([0.1, 0.0], dtype=np.float32),
                'value_delta_value': np.asarray([0.4, 0.0], dtype=np.float32),
            }
            for input_name in ('actor', 'value'):
                for channel in ('current_support', 'goal_support', 'goal_conflict'):
                    for metric in (
                        'channel_delta_rel', 'channel_delta_mix',
                        'channel_delta_readout', 'channel_output_effect',
                    ):
                        arrays[
                            f'{input_name}_channel_{channel}_{metric}'
                        ] = np.asarray([0.1, 0.0], dtype=np.float32)
            npz_path = job / 'per_sample_metrics.npz'
            np.savez_compressed(npz_path, **arrays)
            identity = {
                'source_run_path': str(Path(directory) / 'source-run'),
                'source_resolved_config_fingerprint': 'config-fingerprint',
                'fixed_batch_fingerprint_sha256': 'batch-fingerprint',
                'source_checkpoint_step': 100000,
                'source_checkpoint_sha256': 'checkpoint-fingerprint',
                'source_config': 'M20A-CUBE-C-M',
                'source_readout': 'mean_context',
            }
            metadata = {
                'status': 'completed',
                'diagnostic_id': 'M20A-D-M',
                'source_run_path': identity['source_run_path'],
                'source_resolved_config_fingerprint': identity['source_resolved_config_fingerprint'],
                'fixed_batch_fingerprint_sha256': identity['fixed_batch_fingerprint_sha256'],
                'source_config': identity['source_config'],
                'source_readout': identity['source_readout'],
                'attention_status': 'not_applicable_mean_readout',
            }
            summary = {
                'status': 'completed',
                'diagnostic_id': 'M20A-D-M',
                **identity,
                'attention_status': 'not_applicable_mean_readout',
                'normal_forward_parity': {'status': 'pass'},
                'state_integrity': {'unchanged': True},
                'gradient_analysis': {'metrics': {}},
                'parameter_learning': {},
                'artifact_paths': {'per_sample_metrics': str(npz_path)},
            }
            (job / 'm20ad_metadata.json').write_text(json.dumps(metadata))
            (job / 'diagnostic_summary.json').write_text(json.dumps(summary))

            output = Path(directory) / 'mean-posthoc'
            report = analyze(root, output, source_config='M20A-CUBE-C-M')
            self.assertEqual(report['provenance']['source_readout'], 'mean_context')
            self.assertEqual(
                report['provenance']['attention_status'],
                'not_applicable_mean_readout',
            )
            self.assertEqual(len(report['outputs']['figures']), 10)
            self.assertNotIn('attention_tv_trajectory.png', report['outputs']['figures'])
            self.assertTrue(Path(report['outputs']['trajectory_table_csv']).is_file())


class M20ADCheckpointDiscoveryTest(unittest.TestCase):
    def test_resolver_includes_configuration_slug(self):
        path = resolve_authoritative_source_run()
        self.assertIn('M20A-CUBE-C-Q__cube_c_q_phase2', str(path))
        self.assertTrue(str(path).endswith('/cube-triple-play-v0/seed_000'))

    def test_resolver_accepts_mean_configuration_explicitly(self):
        path = resolve_authoritative_source_run(config_id='M20A-CUBE-C-M')
        self.assertIn('M20A-CUBE-C-M__cube_c_m_phase2', str(path))
        self.assertTrue(str(path).endswith('/cube-triple-play-v0/seed_000'))

    def test_recursive_numeric_checkpoint_audit_locks_metadata_and_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            source = (
                Path(directory) / 'runs' / 'M20A'
                / 'M20A-CUBE-C-Q__cube_c_q_phase2'
                / 'cube-triple-play-v0' / 'seed_000'
            )
            for relative, role, step in (
                ('checkpoints/params_100000.pkl', 'numeric', 100000),
                ('checkpoints/best/params_100000.pkl', 'best', 100000),
            ):
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('wb') as file:
                    pickle.dump({
                        'agent': {},
                        'checkpoint_metadata': {
                            'checkpoint_role': role,
                            'checkpoint_step': step,
                        },
                    }, file)
            (source / 'checkpoints/best/checkpoint.json').write_text('{}')

            records = discover_numeric_checkpoints(source)
            self.assertEqual(len(records), 2)
            self.assertTrue(all(row['status'] == 'stable_numeric_checkpoint' for row in records))
            self.assertEqual(
                {row['relative_path'] for row in records},
                {'params_100000.pkl', 'best/params_100000.pkl'},
            )
            self.assertTrue(all(len(row['sha256']) == 64 for row in records))
            self.assertEqual(
                {row['embedded_checkpoint_step'] for row in records},
                {100000},
            )

    def test_filename_metadata_step_mismatch_is_reported_with_exact_path(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source'
            path = source / 'checkpoints' / 'params_200000.pkl'
            path.parent.mkdir(parents=True)
            with path.open('wb') as file:
                pickle.dump({
                    'agent': {},
                    'checkpoint_metadata': {
                        'checkpoint_role': 'numeric',
                        'checkpoint_step': 100000,
                    },
                }, file)

            records = discover_numeric_checkpoints(source)
            self.assertEqual(records[0]['status'], 'invalid_numeric_checkpoint')
            self.assertIn(str(path), records[0]['error'])
            self.assertIn('filename/metadata checkpoint step', records[0]['error'])


if __name__ == '__main__':
    unittest.main()
