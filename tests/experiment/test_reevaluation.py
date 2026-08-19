import csv
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from impls.experiment import (
    EPISODE_FIELDS,
    TASK_SUMMARY_FIELDS,
    aggregate_campaign,
    config_fingerprint,
    load_reevaluation_spec,
    protocol_fingerprint,
    sha256_file,
    validate_source_run,
)
from impls.experiment.reevaluation import (
    _read_episode_rows,
    _restore_probe,
    _write_task_and_overall_summaries,
)
from impls.utils.checkpointing import sha256_file as checkpoint_sha256
from impls.utils.evaluation import (
    COMMON_EPISODE_SEED_SCHEME,
    common_episode_seeds,
    evaluate,
    evaluate_episodes,
    extract_episode_success,
)
from impls.utils.reproducibility import derive_seed


class _FakeEnv:
    def __init__(self, task_count=2):
        self.unwrapped = self
        self.task_infos = [{'task_name': f'task-{i}'} for i in range(task_count)]
        self.reset_seeds = []
        self.step_count = 0

    def reset(self, seed=None, options=None):
        self.reset_seeds.append((int(seed), dict(options or {})))
        self.step_count = 0
        return np.asarray([0.0], dtype=np.float32), {'goal': np.asarray([0.0], dtype=np.float32)}

    def step(self, action):
        del action
        self.step_count += 1
        done = self.step_count == 2
        info = {'success': float(done), 'scalar': 3.0} if done else {'scalar': 0.0}
        return np.asarray([self.step_count], dtype=np.float32), 1.0, done, False, info

    def render(self):
        return np.zeros((2, 2, 3), dtype=np.uint8)


class _FakeAgent:
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        del observations, goals, seed, temperature
        return np.asarray([0.0], dtype=np.float32)


class _ProbeAgent:
    def __init__(self):
        self.seed = None

    def sample_actions(self, observations, goals=None, seed=None):
        del observations, goals
        self.seed = seed
        return np.asarray([0.0], dtype=np.float32)


class ReevaluationTest(unittest.TestCase):
    def test_seed_scheme_is_versioned_and_policy_independent(self):
        first = common_episode_seeds(20260819, 2, 37)
        second = common_episode_seeds(20260819, 2, 37)
        self.assertEqual(first, second)
        self.assertEqual(first['task_seed'], common_episode_seeds(20260819, 2, 0)['task_seed'])
        self.assertNotEqual(first['episode_seed'], common_episode_seeds(20260819, 2, 38)['episode_seed'])
        self.assertEqual(COMMON_EPISODE_SEED_SCHEME, 'common_task_episode_v1')

    def test_streaming_evaluation_is_deterministic_and_does_not_retain_trajectory(self):
        agent = _FakeAgent()
        first_env = _FakeEnv()
        second_env = _FakeEnv()
        kwargs = dict(
            task_id=2,
            task_name='task-1',
            config={},
            evaluation_seed=20260819,
            episode_indices=[0, 1, 2],
        )
        first = evaluate_episodes(agent, first_env, **kwargs)
        second = evaluate_episodes(agent, second_env, **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual([row['paired_episode_id'] for row in first], ['task02_ep000', 'task02_ep001', 'task02_ep002'])
        self.assertEqual([row['episode_length'] for row in first], [2, 2, 2])
        self.assertTrue(all('trajectory' not in row for row in first))
        self.assertEqual([seed for seed, _ in first_env.reset_seeds], [row['episode_seed'] for row in first])

    def test_legacy_evaluate_return_api_and_trajectory_remain_available(self):
        stats, trajectories, renders = evaluate(
            _FakeAgent(),
            _FakeEnv(),
            task_id=1,
            config={},
            num_eval_episodes=2,
            num_video_episodes=0,
            seed=123,
        )
        self.assertAlmostEqual(float(stats['success']), 1.0)
        self.assertEqual(len(trajectories), 2)
        self.assertEqual(len(trajectories[0]['reward']), 2)
        self.assertEqual(renders, [])

    def test_success_extraction_fails_on_ambiguity(self):
        self.assertEqual(extract_episode_success({'success': True}), 1.0)
        with self.assertRaises(ValueError):
            extract_episode_success({'a_success': 1.0, 'b_success': 0.0})

    def test_restore_probe_uses_deterministic_derived_seed(self):
        import jax

        agent = _ProbeAgent()
        _restore_probe(
            agent,
            {'observations': np.asarray([[0.0]], dtype=np.float32)},
            'hiql',
            training_seed=7,
        )
        expected = jax.random.PRNGKey(derive_seed(7, 0xA11CE))
        np.testing.assert_array_equal(agent.seed, expected)

        second = _ProbeAgent()
        _restore_probe(
            second,
            {'observations': np.asarray([[0.0]], dtype=np.float32)},
            'hiql',
            training_seed=7,
        )
        np.testing.assert_array_equal(second.seed, agent.seed)

    def test_task_and_overall_accounting(self):
        root = Path(tempfile.mkdtemp())
        rows = []
        for task_id in (1, 2):
            for episode_index in range(3):
                rows.append({
                    'task_id': task_id,
                    'task_name': f'task-{task_id}',
                    'episode_index': episode_index,
                    'success': float(task_id == 1 or episode_index == 0),
                    'episode_return': float(episode_index),
                    'episode_length': 2 + episode_index,
                })
        summary = _write_task_and_overall_summaries(
            root,
            rows,
            task_names={1: 'task-1', 2: 'task-2'},
            episodes_per_task=3,
            checkpoint_step=500000,
            evaluation_seed=20260819,
        )
        self.assertAlmostEqual(summary['task_success']['task1'], 1.0)
        self.assertAlmostEqual(summary['task_success']['task2'], 1 / 3)
        self.assertAlmostEqual(summary['evaluation/overall_success'], (1 + 1 / 3) / 2)
        with (root / 'task_summary.csv').open(newline='') as file:
            task_rows = list(csv.DictReader(file))
        self.assertEqual(len(task_rows), 2)
        self.assertEqual(tuple(task_rows[0]), TASK_SUMMARY_FIELDS)

    def test_duplicate_episode_rows_are_rejected(self):
        root = Path(tempfile.mkdtemp())
        path = root / 'episode_results.csv'
        with path.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=EPISODE_FIELDS)
            writer.writeheader()
            row = {field: '' for field in EPISODE_FIELDS}
            row.update({'task_id': 1, 'episode_index': 0, 'paired_episode_id': 'task01_ep000'})
            writer.writerow(row)
            writer.writerow(row)
        with self.assertRaises(ValueError):
            _read_episode_rows(path)

    def test_source_provenance_and_checkpoint_hash(self):
        root = Path(tempfile.mkdtemp())
        source = root / 'runs' / 'TEST' / 'TEST-C001__control' / 'toy-v0' / 'seed_000'
        (source / 'checkpoints').mkdir(parents=True)
        resolved_payload = {
            'study': {'study_id': 'TEST'},
            'configuration': {'config_id': 'TEST-C001', 'slug': 'control'},
            'algorithm_config': {'agent': {'agent_name': 'hiql'}},
        }
        fingerprint = config_fingerprint(resolved_payload)
        metadata = {
            'status': 'completed',
            'git_dirty': False,
            'run_dir': str(source.resolve()),
            'study_id': 'TEST',
            'config_id': 'TEST-C001',
            'config_slug': 'control',
            'environment': 'toy-v0',
            'seed': 0,
            'git_commit': 'abc',
            'algorithm': 'hiql',
            'dataset_dir': str(root / 'data'),
            'resolved_config_fingerprint': fingerprint,
        }
        (source / 'runtime_metadata.json').write_text(json.dumps(metadata))
        (source / 'resolved_config.json').write_text(json.dumps(resolved_payload | {'resolved_config_fingerprint': fingerprint}))
        checkpoint = source / 'checkpoints' / 'params_7.pkl'
        with checkpoint.open('wb') as file:
            pickle.dump({'agent': {}, 'checkpoint_metadata': {
                'environment': 'toy-v0', 'study_id': 'TEST', 'config_id': 'TEST-C001',
                'config_slug': 'control', 'git_commit': 'abc', 'seed': 0,
            }}, file)
        provenance = validate_source_run(source, checkpoint_step=7, expected_study_id='TEST', expected_environment='toy-v0')
        self.assertEqual(provenance['checkpoint_sha256'], sha256_file(checkpoint))
        self.assertEqual(provenance['source_resolved_config_fingerprint'], fingerprint)
        self.assertEqual(provenance['source_training_seed'], 0)

    def test_best_last_selectors_use_index_not_eval_csv(self):
        root = Path(tempfile.mkdtemp())
        source = root / 'runs' / 'TEST' / 'TEST-C001__control' / 'toy-v0' / 'seed_000'
        (source / 'checkpoints' / 'best').mkdir(parents=True)
        (source / 'checkpoints' / 'last').mkdir(parents=True)
        resolved_payload = {
            'study': {'study_id': 'TEST'},
            'configuration': {'config_id': 'TEST-C001', 'slug': 'control'},
            'algorithm_config': {'agent': {'agent_name': 'hiql'}},
        }
        fingerprint = config_fingerprint(resolved_payload)
        runtime_metadata = {
            'status': 'completed',
            'git_dirty': False,
            'run_dir': str(source.resolve()),
            'study_id': 'TEST',
            'config_id': 'TEST-C001',
            'config_slug': 'control',
            'environment': 'toy-v0',
            'seed': 0,
            'git_commit': 'abc',
            'algorithm': 'hiql',
            'dataset_dir': str(root / 'data'),
            'resolved_config_fingerprint': fingerprint,
        }
        (source / 'runtime_metadata.json').write_text(json.dumps(runtime_metadata))
        (source / 'resolved_config.json').write_text(
            json.dumps(resolved_payload | {'resolved_config_fingerprint': fingerprint})
        )
        (source / 'eval.csv').write_text(
            'step,evaluation/overall_success\n7,0.2\n99,1.0\n'
        )

        entries = {}
        for role, step in (('best', 7), ('last', 8)):
            path = source / 'checkpoints' / role / f'params_{step}.pkl'
            metadata = {
                'checkpoint_role': role,
                'checkpoint_step': step,
                'environment': 'toy-v0',
                'study_id': 'TEST',
                'config_id': 'TEST-C001',
                'config_slug': 'control',
                'git_commit': 'abc',
                'seed': 0,
            }
            with path.open('wb') as file:
                pickle.dump({'agent': {}, 'checkpoint_metadata': metadata}, file)
            sha = checkpoint_sha256(path)
            metadata |= {
                'checkpoint_sha256': sha,
                'path': str(path.relative_to(source)),
                'metadata_path': str((path.parent / 'checkpoint.json').relative_to(source)),
            }
            (path.parent / 'checkpoint.json').write_text(json.dumps(metadata))
            entries[role] = {
                'step': step,
                'metric': 0.2 if role == 'best' else 0.1,
                'path': str(path.relative_to(source)),
                'sha256': sha,
                'metadata_path': str((path.parent / 'checkpoint.json').relative_to(source)),
            }
        (source / 'checkpoints' / 'index.json').write_text(json.dumps({
            'schema_version': 1,
            'selection_metric': 'evaluation/overall_success',
            'best': entries['best'],
            'last': entries['last'],
        }))

        best = validate_source_run(
            source,
            checkpoint_selector={'selector': 'best'},
            expected_study_id='TEST',
            expected_environment='toy-v0',
        )
        last = validate_source_run(
            source,
            checkpoint_selector={'selector': 'last'},
            expected_study_id='TEST',
            expected_environment='toy-v0',
        )
        self.assertEqual(best['resolved_checkpoint_role'], 'best')
        self.assertEqual(best['checkpoint_step'], 7)
        self.assertEqual(last['resolved_checkpoint_role'], 'last')
        self.assertEqual(last['checkpoint_step'], 8)
        self.assertEqual(best['checkpoint_sha256'], entries['best']['sha256'])

    def test_reevaluation_spec_normalizes_semantic_and_legacy_selectors(self):
        root = Path(tempfile.mkdtemp())
        common = (
            'reevaluation_id: TEST-R001\n'
            'source_study_id: TEST\n'
            'source_run_root: /tmp/runs\n'
            'environments: [toy-v0]\n'
            'configs: all\n'
            'training_seeds: [0]\n'
            'protocol:\n'
            '  task_selection: all\n'
            '  episodes_per_task: 2\n'
            '  evaluation_seed: 20260819\n'
            '  seed_scheme: common_task_episode_v1\n'
        )
        for selector in ('best', 'last'):
            path = root / f'{selector}.yaml'
            path.write_text(common + f'checkpoint:\n  selector: {selector}\n')
            spec = load_reevaluation_spec(path)
            self.assertEqual(spec['checkpoint'], {'selector': selector})
        legacy = load_reevaluation_spec(
            Path(__file__).resolve().parents[2]
            / 'experiments/M10A_fixed_budget_placement/reevaluations/M10A-R001.yaml'
        )
        self.assertEqual(legacy['checkpoint'], {'selector': 'step', 'step': 500000})

    def test_campaign_aggregation_keeps_seed_and_episode_variability_separate(self):
        root = Path(tempfile.mkdtemp())
        spec = {
            'reevaluation_id': 'TEST-R001',
            'source_study_id': 'TEST',
            'checkpoint_step': 7,
            'protocol': {
                'task_selection': 'all', 'episodes_per_task': 2,
                'evaluation_seed': 1, 'seed_scheme': COMMON_EPISODE_SEED_SCHEME,
                'eval_temperature': 0.0, 'eval_gaussian': None, 'video_episodes': 0,
            },
        }
        provenances = []
        for seed, success in ((0, 0.5), (1, 1.0), (2, 0.0)):
            source = root / 'source' / f'seed_{seed:03d}'
            source.mkdir(parents=True)
            checkpoint = source / 'params.pkl'
            checkpoint.write_bytes(f'checkpoint-{seed}'.encode())
            provenance = {
                'source_run_dir': str(source), 'source_study_id': 'TEST',
                'source_config_id': 'TEST-C001', 'source_config_slug': 'control',
                'source_environment': 'toy-v0', 'source_training_seed': seed,
                'checkpoint_step': 7, 'checkpoint_sha256': sha256_file(checkpoint),
            }
            provenances.append(provenance)
            output = root / 'reevaluations' / 'TEST' / 'TEST-R001' / 'TEST-C001__control' / 'toy-v0' / f'seed_{seed:03d}'
            output.mkdir(parents=True)
            (output / 'reevaluation_metadata.json').write_text(json.dumps({
                'status': 'completed', 'checkpoint_sha256': provenance['checkpoint_sha256'],
            }))
            (output / 'summary.json').write_text(json.dumps({'evaluation/overall_success': success}))
            with (output / 'task_summary.csv').open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=TASK_SUMMARY_FIELDS)
                writer.writeheader()
                for task_id in (1, 2):
                    writer.writerow({'task_id': task_id, 'task_name': f'task-{task_id}', 'success_rate': success})
        campaign = aggregate_campaign(spec, reeval_root=root / 'reevaluations', source_runs=provenances)
        with (campaign / 'config_summary.csv').open(newline='') as file:
            row = next(csv.DictReader(file))
        self.assertEqual(row['number_training_seeds'], '3')
        self.assertAlmostEqual(float(row['overall_success_mean']), 0.5)
        self.assertAlmostEqual(float(row['overall_success_population_sd']), np.sqrt(1 / 6))
        self.assertAlmostEqual(float(row['overall_success_sample_sd']), 0.5)

    @unittest.skipUnless(
        Path('/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M10A').is_dir(),
        'formal M10A source runs are not available in this environment',
    )
    def test_m10a_inventory_has_33_valid_checkpoints(self):
        from tools.reevaluate_study import _source_candidates

        spec = load_reevaluation_spec(
            Path(__file__).resolve().parents[2]
            / 'experiments/M10A_fixed_budget_placement/reevaluations/M10A-R001.yaml'
        )
        items = _source_candidates(
            spec,
            spec['source_run_root'],
            config_filter=None,
            seed_filter=set(spec['training_seeds']),
        )
        self.assertEqual(len(items), 33)
        self.assertTrue(all(item.get('provenance') for item in items))


if __name__ == '__main__':
    unittest.main()
