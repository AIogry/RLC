"""M18 Study, preflight, aggregation, and cross-K restore tests."""

import contextlib
import copy
import csv
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from impls.agents import agents
from impls.computation.accounting import count_parameters, gciql_architecture_accounting
from impls.experiment import load_study, make_run_path, prepare_run_design
from impls.experiment.management import config_fingerprint, jsonable
from impls.main import _computation_slot_accounting, _make_config, _parse_args
from impls.utils.checkpointing import sha256_file, write_checkpoint_index
from impls.utils.flax_utils import (
    restore_agent_from_checkpoint,
    save_agent,
    save_semantic_checkpoint,
)
from tools import analyze_m18, m18_cross_k_eval, m18_doctor, sweep


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'experiments/M18_puzzle_recurrent_compute_scaling/study.yaml'
CONFIG_DIR = STUDY.parent / 'configs'
DATASET_ROOT = Path('/data/qijunrong/06-RL/offline-rl/data/raw_ogbench')
DATASET_AVAILABLE = all(
    (DATASET_ROOT / name).is_file()
    for name in ('puzzle-4x4-play-v0.npz', 'puzzle-4x4-play-v0-val.npz')
)
K_VALUES = (1, 2, 4, 8)


def _configuration_for_k(k):
    path = CONFIG_DIR / f'M18-4x4-L2-K{k}.yaml'
    return prepare_run_design(STUDY, path)[1]


def _resolved_config(k):
    return _make_config(_parse_args(['--agent', 'gciql']), configuration=_configuration_for_k(k))


class M18RecurrentComputeScalingTest(unittest.TestCase):
    def test_exact_matrix_and_runtime_semantics(self):
        study = load_study(STUDY)
        self.assertEqual(study.study_id, 'M18')
        self.assertEqual(study.data['primary_factors'], ['recurrent_compute_budget_K'])
        self.assertEqual(study.data['environments'], ['puzzle-4x4-play-v0'])
        self.assertEqual(study.data['algorithms'], ['gciql'])
        self.assertEqual(study.data['seeds'], [0])
        self.assertEqual(study.data['matrix']['block_depth_L'], 2)
        self.assertEqual(tuple(study.data['matrix']['recurrent_compute_budget_K']), K_VALUES)
        self.assertEqual(study.data['alpha_provenance']['runtime_authority'], 'configuration.agent_overrides.alpha')
        self.assertEqual(len(list(CONFIG_DIR.glob('*.yaml'))), len(K_VALUES))

        observed = []
        reference = None
        for path in sorted(CONFIG_DIR.glob('*.yaml')):
            _, configuration = prepare_run_design(STUDY, path)
            data = configuration.data
            k = data['factors']['recurrent_compute_budget_K']
            self.assertIn(k, K_VALUES)
            self.assertEqual(data['factors']['K'], k)
            self.assertEqual(data['factors']['block_depth_L'], 2)
            self.assertEqual(data['factors']['alpha'], 0.4)
            self.assertEqual(data['agent_overrides']['alpha'], 0.4)
            config = _make_config(_parse_args(['--agent', 'gciql']), configuration=configuration)
            self.assertEqual(float(config['alpha']), 0.4)
            frozen = {key: value for key, value in config.items() if key != 'compute'}
            if reference is None:
                reference = frozen
            else:
                self.assertEqual(frozen, reference)
            for slot_name in ('actor', 'value', 'critic'):
                slot = config['compute'][slot_name]
                self.assertTrue(slot['enabled'])
                self.assertEqual(slot['structure'], 'puzzle_tokens')
                self.assertEqual(slot['block'], 'mlp_mixer')
                self.assertEqual(slot['block_kwargs']['num_blocks'], 2)
                self.assertEqual(slot['topology'], 'single_state')
                self.assertEqual(slot['topology_kwargs']['iterations'], k)
                self.assertEqual(slot['topology_kwargs']['input_mapping'], 'identity')
                self.assertEqual(slot['topology_kwargs']['state_init'], 'zero_buffer')
                self.assertEqual(slot['topology_kwargs']['input_injection'], 'z_plus_x')
                self.assertFalse(slot['topology_kwargs']['residual'])
                self.assertEqual(slot['parameter_sharing'], 'shared')
                self.assertEqual(slot['readout'], 'mean_context')
            observed.append(k)
        self.assertEqual(tuple(sorted(observed)), K_VALUES)

    @unittest.skipUnless(DATASET_AVAILABLE, 'real Puzzle-4x4 train/validation data unavailable')
    def test_doctor_checks_params_compute_and_output_collisions(self):
        with tempfile.TemporaryDirectory() as directory:
            report = m18_doctor.validate(STUDY, DATASET_ROOT, Path(directory) / 'runs', ['0', '1'])
            self.assertEqual(report['status'], 'PASS')
            self.assertEqual(report['expected_runs'], 4)
            jobs = report['jobs']
            self.assertEqual([job['K'] for job in jobs], list(K_VALUES))
            self.assertTrue(all(job['alpha'] == 0.4 for job in jobs))
            self.assertEqual(
                {job['total_trainable_params'] for job in jobs},
                {jobs[0]['total_trainable_params']},
            )
            self.assertEqual(
                {job['actor_trainable_params'] for job in jobs},
                {jobs[0]['actor_trainable_params']},
            )
            self.assertEqual([job['actor_unique_mixer_layers'] for job in jobs], [2, 2, 2, 2])
            self.assertEqual([job['actor_executed_mixer_layers'] for job in jobs], [2, 4, 8, 16])
            self.assertTrue(all(
                right['actor_body_dense_macs'] > left['actor_body_dense_macs']
                for left, right in zip(jobs, jobs[1:])
            ))
            collision = Path(jobs[0]['run_dir'])
            collision.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'Output path already exists'):
                m18_doctor.validate(STUDY, DATASET_ROOT, Path(directory) / 'runs', ['0'])

    @unittest.skipUnless(DATASET_AVAILABLE, 'real Puzzle-4x4 train/validation data unavailable')
    def test_generic_sweep_dry_run_emits_exactly_four_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                result = sweep.main([
                    '--study', str(STUDY),
                    '--gpus', '0,1',
                    '--run-root', str(Path(directory) / 'runs'),
                    '--dataset-root', str(DATASET_ROOT),
                    '--dry-run',
                ])
            output = stream.getvalue()
        self.assertEqual(result, 0)
        self.assertIn('total=4 planned=4', output)
        self.assertEqual(output.count('[PLANNED]'), 4)

    def test_cross_k_override_restores_same_parameter_tree_without_update(self):
        observations = np.zeros((2, 83), dtype=np.float32)
        actions = np.zeros((2, 5), dtype=np.float32)
        source_config = jsonable(_resolved_config(1))
        target_config = m18_cross_k_eval.prepare_test_time_config(source_config, 8)
        self.assertEqual(
            [target_config['compute'][slot]['topology_kwargs']['iterations'] for slot in ('actor', 'value', 'critic')],
            [8, 8, 8],
        )
        for slot in ('actor', 'value', 'critic'):
            self.assertEqual(source_config['compute'][slot]['block_kwargs'], target_config['compute'][slot]['block_kwargs'])
            self.assertEqual(source_config['compute'][slot]['structure_kwargs'], target_config['compute'][slot]['structure_kwargs'])
        source = agents['gciql'].create(181, observations, actions, source_config)
        target = agents['gciql'].create(181, observations, actions, target_config)
        self.assertEqual(count_parameters(source.network.params), count_parameters(target.network.params))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = save_agent(source, directory, 1)
            restored = restore_agent_from_checkpoint(target, checkpoint)
        self.assertEqual(count_parameters(restored.network.params), count_parameters(source.network.params))
        report = _computation_slot_accounting(restored, target_config)
        self.assertEqual(report['actor']['unique_mixer_layers'], 2)
        self.assertEqual(report['actor']['executed_mixer_layers'], 16)
        self.assertEqual(report['actor']['trainable_params'], _computation_slot_accounting(source, source_config)['actor']['trainable_params'])

    @unittest.skipUnless(DATASET_AVAILABLE, 'real Puzzle-4x4 train/validation data unavailable')
    def test_cross_k_tiny_checkpoint_only_lifecycle_preserves_source_checkpoint(self):
        """Exercise the M18-D source validation, restore, and all-task evaluation path."""

        configuration = _configuration_for_k(1)
        source_config = jsonable(_resolved_config(1))
        observations = np.zeros((2, 83), dtype=np.float32)
        actions = np.zeros((2, 5), dtype=np.float32)
        source_agent = agents['gciql'].create(191, observations, actions, source_config)
        source_slots = _computation_slot_accounting(source_agent, source_config)
        source_architecture = gciql_architecture_accounting(
            source_agent.network.params, source_config, source_slots,
        )
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / 'runs'
            output_root = Path(directory) / 'diagnostics'
            source_run = make_run_path(
                run_root, 'M18', configuration.config_id, configuration.slug,
                'puzzle-4x4-play-v0', 0,
            )
            (source_run / 'checkpoints').mkdir(parents=True)
            resolved_payload = {
                'study': load_study(STUDY).data,
                'configuration': configuration.data,
                'algorithm_config': {'agent': source_config},
            }
            fingerprint = config_fingerprint(resolved_payload)
            metadata = {
                'status': 'completed',
                'study_id': 'M18',
                'config_id': configuration.config_id,
                'config_slug': configuration.slug,
                'environment': 'puzzle-4x4-play-v0',
                'seed': 0,
                'algorithm': 'gciql',
                'git_commit': 'm18-test-commit',
                'git_dirty': False,
                'dataset_dir': str(DATASET_ROOT),
                'run_dir': str(source_run.resolve()),
                'resolved_config_fingerprint': fingerprint,
                'computation_slot_accounting': jsonable(source_slots),
                'architecture_accounting': jsonable(source_architecture),
            }
            (source_run / 'runtime_metadata.json').write_text(json.dumps(metadata))
            (source_run / 'resolved_config.json').write_text(json.dumps(
                resolved_payload | {'resolved_config_fingerprint': fingerprint}
            ))
            record = save_semantic_checkpoint(
                source_agent,
                source_run,
                'last',
                1,
                {
                    'environment': 'puzzle-4x4-play-v0',
                    'study_id': 'M18',
                    'config_id': configuration.config_id,
                    'config_slug': configuration.slug,
                    'git_commit': 'm18-test-commit',
                    'seed': 0,
                },
            )
            write_checkpoint_index(source_run, best=None, last=record)
            checkpoint = source_run / record['path']
            before_hash = sha256_file(checkpoint)
            jobs = m18_cross_k_eval.plan_jobs(
                STUDY,
                run_root,
                output_root,
                'last',
                config_filter={configuration.config_id},
                test_ks=(8,),
            )
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]['status'], 'planned')
            summary = m18_cross_k_eval.run_one(
                jobs[0],
                episodes_per_task=1,
                evaluation_seed=18191,
                eval_temperature=0.0,
                eval_gaussian=None,
            )
            self.assertEqual(summary['status'], 'completed')
            self.assertEqual((summary['K_train'], summary['K_test']), (1, 8))
            self.assertEqual(summary['total_episodes'], 5)
            self.assertTrue((jobs[0]['output_dir'] / 'task_results.csv').is_file())
            self.assertEqual(before_hash, sha256_file(checkpoint))

    def test_aggregation_exposes_complete_curves_task_endpoints_and_deltas(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / 'runs'
            for k in K_VALUES:
                configuration = _configuration_for_k(k)
                config = jsonable(_resolved_config(k))
                run_dir = make_run_path(
                    run_root, 'M18', configuration.config_id, configuration.slug,
                    'puzzle-4x4-play-v0', 0,
                )
                run_dir.mkdir(parents=True)
                metadata = {
                    'study_id': 'M18',
                    'config_id': configuration.config_id,
                    'environment': 'puzzle-4x4-play-v0',
                    'seed': 0,
                    'status': 'completed',
                    'architecture_accounting': {'total_trainable_params': 1000 + k, 'total_dense_macs': 2000 + k, 'slots': {}},
                    'computation_slot_accounting': {
                        'actor': {
                            'trainable_params': 100,
                            'structured_body_dense_macs': 200 + k,
                            'unique_mixer_layers': 2,
                            'executed_mixer_layers': 2 * k,
                            'executed_sequential_depth': 2 + 8 * k,
                            'buffer_elements': 128,
                        },
                    },
                }
                (run_dir / 'runtime_metadata.json').write_text(json.dumps(metadata))
                (run_dir / 'resolved_config.json').write_text(json.dumps({'algorithm_config': {'agent': config}}))
                fields = ['step', 'evaluation/overall_success'] + [f'evaluation/task{task}_success' for task in range(1, 6)]
                with (run_dir / 'eval.csv').open('w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=fields)
                    writer.writeheader()
                    for step in analyze_m18.EXPECTED_STEPS:
                        value = k / 10.0 + step / 10_000_000.0
                        writer.writerow({
                            'step': step,
                            'evaluation/overall_success': value,
                            **{f'evaluation/task{task}_success': value for task in range(1, 6)},
                        })
            rows = analyze_m18.collect(STUDY, run_root)
            effects = analyze_m18.descriptive_effects(rows)
        self.assertEqual([row['curve_status'] for row in rows], ['complete'] * 4)
        self.assertEqual([row['runtime_validation'] for row in rows], ['valid'] * 4)
        self.assertEqual(rows[-1]['task_final_success']['evaluation/task5_success'], 0.9)
        k2_minus_k1 = next(row for row in effects if row['contrast'] == 'K2-K1')
        self.assertAlmostEqual(k2_minus_k1['delta_final_success'], 0.1)


if __name__ == '__main__':
    unittest.main()
