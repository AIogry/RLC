"""Targeted M18-D diagnostic parity, provenance, and aggregation tests."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import jax
import numpy as np

from impls.agents import agents
from impls.computation.topologies.single_state import SingleState
from impls.diagnostics.puzzle_logic import Puzzle4x4LogicalOracle, audit_real_puzzle_environment
from impls.experiment import load_study, make_run_path, prepare_run_design
from impls.experiment.management import config_fingerprint, jsonable
from impls.experiment.reevaluation import ReevaluationError, validate_source_run
from impls.main import _make_config, _parse_args
from impls.utils.checkpointing import sha256_file, write_checkpoint_index
from impls.utils.flax_utils import save_semantic_checkpoint
from tools import (
    analyze_m18_d,
    m18_cross_actor_critic,
    m18_cross_k_eval,
    m18_paired_rollout_diagnostics,
    m18_trace_diagnostics,
)


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'experiments/M18_puzzle_recurrent_compute_scaling/study.yaml'
CONFIG_DIR = STUDY.parent / 'configs'
DATASET_ROOT = Path('/data/qijunrong/06-RL/offline-rl/data/raw_ogbench')
DATASET_AVAILABLE = all(
    (DATASET_ROOT / name).is_file()
    for name in ('puzzle-4x4-play-v0.npz', 'puzzle-4x4-play-v0-val.npz')
)
REAL_RUN_ROOT = Path('/data/qijunrong/06-RL/offline-rl/exp/RLC/runs')


def _configuration(k):
    return prepare_run_design(STUDY, CONFIG_DIR / f'M18-4x4-L2-K{k}.yaml')[1]


def _config(k):
    return jsonable(_make_config(_parse_args(['--agent', 'gciql']), configuration=_configuration(k)))


def _tree_paths(tree, path=()):
    if hasattr(tree, 'items'):
        return tuple(
            item
            for key, value in tree.items()
            for item in _tree_paths(value, path + (str(key),))
        )
    return (path,)


def _synthetic_source(root, *, k=1, status='running'):
    """Create a checksum-valid M18 semantic-best source entirely under tmp."""

    configuration = _configuration(k)
    config = _config(k)
    observations = np.zeros((2, 83), dtype=np.float32)
    actions = np.zeros((2, 5), dtype=np.float32)
    agent = agents['gciql'].create(981, observations, actions, config)
    run_root = Path(root) / 'runs'
    source_run = make_run_path(
        run_root, 'M18', configuration.config_id, configuration.slug,
        'puzzle-4x4-play-v0', 0,
    )
    source_run.mkdir(parents=True)
    resolved_payload = {
        'study': load_study(STUDY).data,
        'configuration': configuration.data,
        'algorithm_config': {'agent': config},
    }
    fingerprint = config_fingerprint(resolved_payload)
    metadata = {
        'status': status,
        'study_id': 'M18',
        'config_id': configuration.config_id,
        'config_slug': configuration.slug,
        'environment': 'puzzle-4x4-play-v0',
        'seed': 0,
        'algorithm': 'gciql',
        'git_commit': 'm18-d-test-commit',
        'git_dirty': False,
        'dataset_dir': str(DATASET_ROOT),
        'run_dir': str(source_run.resolve()),
        'resolved_config_fingerprint': fingerprint,
    }
    (source_run / 'runtime_metadata.json').write_text(json.dumps(metadata))
    (source_run / 'resolved_config.json').write_text(json.dumps(
        resolved_payload | {'resolved_config_fingerprint': fingerprint}
    ))
    record = save_semantic_checkpoint(
        agent,
        source_run,
        'best',
        1,
        {
            'environment': 'puzzle-4x4-play-v0',
            'study_id': 'M18',
            'config_id': configuration.config_id,
            'config_slug': configuration.slug,
            'git_commit': 'm18-d-test-commit',
            'seed': 0,
            'selection_metric': 'evaluation/overall_success',
            'selection_metric_value': 0.5,
        },
    )
    write_checkpoint_index(source_run, best=record, last=None)
    return source_run, configuration, agent, record


class M18DDiagnosticCoreTest(unittest.TestCase):
    def test_single_state_trace_is_additive_and_matches_normal_forward(self):
        x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        module = SingleState(
            state_dim=4,
            iterations=4,
            input_mapping_mode='identity',
            state_init='zero_buffer',
            parameter_sharing='shared',
        )
        variables = module.init(jax.random.PRNGKey(11), x)
        params_before = _tree_paths(variables['params'])
        buffers_before = _tree_paths(variables['buffers'])
        def loss_fn(params):
            output = module.apply({'params': params, 'buffers': variables['buffers']}, x)
            return jax.numpy.sum(output.representation ** 2)
        gradients_before = jax.grad(loss_fn)(variables['params'])
        normal = module.apply(variables, x)
        trace = module.apply(variables, x, method=module.trace_states, max_iterations=8)
        gradients_after = jax.grad(loss_fn)(variables['params'])
        self.assertEqual(len(trace), 9)
        self.assertEqual(trace[0].shape, x.shape)
        np.testing.assert_allclose(np.asarray(trace[0]), 0.0, atol=0.0)
        np.testing.assert_allclose(np.asarray(trace[4]), np.asarray(normal.representation), rtol=1e-6, atol=1e-6)
        self.assertEqual(params_before, _tree_paths(variables['params']))
        self.assertEqual(buffers_before, _tree_paths(variables['buffers']))
        for before, after in zip(
            jax.tree_util.tree_leaves(gradients_before),
            jax.tree_util.tree_leaves(gradients_after),
        ):
            np.testing.assert_allclose(np.asarray(before), np.asarray(after), rtol=0.0, atol=0.0)

    def test_actor_value_critic_traces_preserve_normal_actor_and_source_depths(self):
        config = _config(4)
        observations = np.zeros((3, 83), dtype=np.float32)
        goals = np.ones((3, 83), dtype=np.float32) * 0.1
        actions = np.zeros((3, 5), dtype=np.float32)
        agent = agents['gciql'].create(982, observations, actions, config)
        params_before = _tree_paths(agent.network.params)
        buffers_before = _tree_paths(agent.network.model_state)
        network_step_before = agent.network.step

        actor_trace = agent.network(
            observations, goals, name='actor', method='diagnostic_trace', max_iterations=8,
        )
        value_trace = agent.network(
            observations, goals, name='value', method='diagnostic_trace', max_iterations=8,
        )
        critic_trace = agent.network(
            observations, goals, actions, name='critic', method='diagnostic_trace', max_iterations=8,
        )
        actor_tokens = np.asarray(actor_trace['token_states'])
        self.assertEqual(actor_tokens.shape, (3, 9, 16, 128))
        self.assertEqual(np.asarray(value_trace['token_states']).shape, (3, 9, 16, 128))
        self.assertEqual(np.asarray(critic_trace['token_states']).shape, (2, 3, 9, 16, 128))
        np.testing.assert_allclose(actor_tokens[:, 0], 0.0, atol=0.0)
        normal_mean = np.asarray(agent.network.select('actor')(observations, goals, temperature=0.0).mode())
        np.testing.assert_allclose(
            np.asarray(actor_trace['action_means'])[:, 4], normal_mean, rtol=1e-6, atol=1e-6,
        )
        target = m18_cross_k_eval.prepare_actor_test_time_config(config, 8)
        self.assertEqual(target['compute']['actor']['topology_kwargs']['iterations'], 8)
        self.assertEqual(target['compute']['value']['topology_kwargs']['iterations'], 4)
        self.assertEqual(target['compute']['critic']['topology_kwargs']['iterations'], 4)
        self.assertEqual(params_before, _tree_paths(agent.network.params))
        self.assertEqual(buffers_before, _tree_paths(agent.network.model_state))
        self.assertEqual(network_step_before, agent.network.step)

        clipped, q_metrics = m18_trace_diagnostics._action_metrics(
            np.asarray(actor_trace['action_means']), actions,
        )
        d4_metrics, _ = m18_trace_diagnostics._critic_action_metrics(
            agent,
            {'observations': observations, 'actor_goals': goals, 'dataset_actions': actions},
            clipped,
        )
        ordinary_q = np.asarray(agent.network.select('critic')(
            observations, goals, clipped[:, 4],
        ))
        np.testing.assert_allclose(d4_metrics['q1'][:, 4], ordinary_q[0], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(d4_metrics['q2'][:, 4], ordinary_q[1], rtol=1e-6, atol=1e-6)

    def test_running_best_source_is_allowed_but_last_is_not(self):
        with tempfile.TemporaryDirectory() as directory:
            source_run, _, _, record = _synthetic_source(directory, status='running')
            before = sha256_file(source_run / record['path'])
            provenance = m18_cross_k_eval.validate_m18_best_source(source_run)
            self.assertEqual(provenance['source_run_status_at_validation'], 'running')
            self.assertEqual(provenance['resolved_checkpoint_role'], 'best')
            self.assertEqual(before, provenance['checkpoint_sha256'])
            with self.assertRaises(ReevaluationError):
                validate_source_run(
                    source_run,
                    checkpoint_selector='last',
                    expected_study_id='M18',
                    expected_environment='puzzle-4x4-play-v0',
                )
            self.assertEqual(before, sha256_file(source_run / record['path']))

    def test_trace_dry_run_writes_no_artifacts_and_supports_running_best(self):
        with tempfile.TemporaryDirectory() as directory:
            source_run, _, _, _ = _synthetic_source(directory, status='running')
            output_root = Path(directory) / 'diagnostics'
            result = m18_trace_diagnostics.main([
                '--study', str(STUDY),
                '--source-run-root', str(Path(directory) / 'runs'),
                '--output-root', str(output_root),
                '--checkpoint', 'best',
                '--train-ks', '1',
                '--batch-size', '16',
                '--max-trace-k', '8',
                '--dry-run',
            ])
            self.assertEqual(result, 0)
            self.assertFalse(output_root.exists())
            self.assertTrue(source_run.is_dir())


class M18DAggregationTest(unittest.TestCase):
    def _fake_artifacts(self, diagnostics_root):
        diagnostics_root = Path(diagnostics_root)
        cross_root = diagnostics_root / 'M18D' / 'cross_k' / 'checkpoint_best'
        for train_k in (1, 2, 4, 8):
            for test_k in (1, 2, 4, 8):
                job_dir = cross_root / f'K{train_k}' / f'Kactor{test_k}'
                job_dir.mkdir(parents=True)
                success = 0.2 + 0.03 * train_k - 0.01 * abs(test_k - train_k)
                (job_dir / 'summary.json').write_text(json.dumps({
                    'status': 'completed',
                    'diagnostic_id': 'M18-D1',
                    'K_train': train_k,
                    'K_actor_test': test_k,
                    'overall_success': success,
                    'checkpoint_role': 'best',
                    'checkpoint_step': 100 * train_k,
                    'checkpoint_sha256': f'hash-{train_k}',
                }))
        metrics = (
            'state_rms', 'relative_update_from_previous', 'state_cosine_from_previous',
            'token_variance', 'pairwise_token_cosine', 'action_delta_from_previous',
            'action_drift_from_k1', 'action_mean_saturation_fraction',
            'dataset_action_mse', 'qmin', 'qgap_vs_dataset_action',
        )
        for train_k in (4, 8):
            trace_dir = (
                diagnostics_root / 'M18D' / 'trace' / 'checkpoint_best'
                / 'fixed_batch_N16_seed1' / f'trainK{train_k}' / 'maxTraceK8'
            )
            trace_dir.mkdir(parents=True)
            (trace_dir / 'm18d_metadata.json').write_text(json.dumps({
                'status': 'completed',
                'diagnostic_id': 'M18-D234',
                'source_checkpoint_role': 'best',
                'source_checkpoint_step': 100 * train_k,
                'source_checkpoint_sha256': f'hash-{train_k}',
            }))
            fields = (
                'K_train', 'checkpoint_role', 'slot', 'ensemble_member', 'iteration_k',
                'is_depth_extrapolation', 'metric', 'count', 'mean', 'std', 'median',
                'p10', 'p25', 'p75', 'p90',
            )
            with (trace_dir / 'trace_summary.csv').open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                for metric in metrics:
                    for iteration_k in range(9):
                        if iteration_k == 0 and metric in (
                            'relative_update_from_previous', 'state_cosine_from_previous',
                            'action_delta_from_previous', 'action_drift_from_k1',
                            'action_mean_saturation_fraction', 'dataset_action_mse',
                            'qmin', 'qgap_vs_dataset_action',
                        ):
                            continue
                        writer.writerow({
                            'K_train': train_k,
                            'checkpoint_role': 'best',
                            'slot': 'actor',
                            'ensemble_member': '',
                            'iteration_k': iteration_k,
                            'is_depth_extrapolation': iteration_k > train_k,
                            'metric': metric,
                            'count': 16,
                            'mean': 0.1 * (iteration_k + 1),
                            'std': 0.01,
                            'median': 0.1 * (iteration_k + 1),
                            'p10': 0.09 * (iteration_k + 1),
                            'p25': 0.095 * (iteration_k + 1),
                            'p75': 0.105 * (iteration_k + 1),
                            'p90': 0.11 * (iteration_k + 1),
                        })

    def test_aggregation_dry_run_execute_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            diagnostics_root = Path(directory) / 'diagnostics'
            self._fake_artifacts(diagnostics_root)
            output_dir = Path(directory) / 'report'
            dry = analyze_m18_d.analyze(diagnostics_root, output_dir, dry_run=True)
            self.assertEqual(dry['status'], 'dry-run')
            self.assertFalse(output_dir.exists())
            completed = analyze_m18_d.analyze(diagnostics_root, output_dir, dry_run=False)
            self.assertEqual(completed['status'], 'completed')
            self.assertTrue((output_dir / 'm18d_summary.csv').is_file())
            self.assertTrue((output_dir / 'm18d_summary.json').is_file())
            self.assertTrue((output_dir / 'M18D_report.md').is_file())
            self.assertEqual(len(list(output_dir.glob('D*.png'))), 11)
            with self.assertRaises(FileExistsError):
                analyze_m18_d.analyze(diagnostics_root, output_dir, dry_run=False)


class M18DSupplementUnitTest(unittest.TestCase):
    def test_d2_plus_is_per_sample_and_validates_both_energy_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = (
                Path(directory) / 'diagnostics' / 'M18D' / 'trace' / 'checkpoint_best'
                / 'fixed_batch_N16_seed1' / 'trainK4' / 'maxTraceK8'
            )
            artifact_dir.mkdir(parents=True)
            (artifact_dir / 'm18d_metadata.json').write_text(json.dumps({
                'status': 'completed', 'diagnostic_id': 'M18-D234', 'K_train': 4,
                'source_checkpoint_step': 900000, 'source_checkpoint_sha256': 'test-sha',
            }))
            # state_rms^2 = mean_token_rms^2 + token_variance for every
            # sample/iteration.  k=0 is zero and must remain NaN, not 0/1.
            state_energy = np.asarray([
                [0.0, 4.0, 9.0, 16.0, 25.0, 36.0],
                [0.0, 16.0, 25.0, 36.0, 49.0, 64.0],
            ])
            mean_energy = np.asarray([
                [0.0, 1.0, 4.0, 4.0, 9.0, 9.0],
                [0.0, 4.0, 9.0, 9.0, 16.0, 16.0],
            ])
            np.savez_compressed(
                artifact_dir / 'actor_metrics.npz',
                sample_id=np.asarray([7, 8], dtype=np.int64),
                iteration_k=np.arange(6, dtype=np.int64),
                slot=np.asarray('actor'), K_train=np.asarray(4, dtype=np.int64),
                checkpoint_role=np.asarray('best'), ensemble_member=np.asarray(''),
                state_rms=np.sqrt(state_energy), mean_token_rms=np.sqrt(mean_energy),
                token_variance=state_energy - mean_energy,
            )
            rows, arrays, _ = analyze_m18_d._collect_retained_energy(Path(directory) / 'diagnostics')
            retained = [
                row for row in rows
                if row['metric'] == 'mean_pooling_retained_energy_fraction' and row['iteration_k'] == 1
            ]
            self.assertEqual(len(retained), 1)
            self.assertAlmostEqual(retained[0]['mean'], (1.0 / 4.0 + 4.0 / 16.0) / 2.0, places=7)
            zero = [
                row for row in rows
                if row['metric'] == 'mean_pooling_retained_energy_fraction' and row['iteration_k'] == 0
            ]
            self.assertEqual(zero[0]['count'], 0)
            self.assertTrue(zero[0]['mean'] is None)
            self.assertTrue(np.all(np.isnan(arrays['mean_pooling_retained_energy_fraction'][arrays['iteration_k'] == 0])))
            self.assertTrue(np.allclose(
                arrays['mean_pooling_retained_energy_fraction'][np.isfinite(arrays['mean_pooling_retained_energy_fraction'])],
                arrays['rho_from_token_variance_identity'][np.isfinite(arrays['mean_pooling_retained_energy_fraction'])],
                rtol=0.0,
                atol=2e-6,
            ))
            extrapolated = arrays['is_depth_extrapolation'][arrays['iteration_k'] == 5]
            self.assertTrue(np.all(extrapolated == 1))

    def test_d6_within_critic_margins_joint_rate_and_dataset_controls(self):
        q4 = {
            'data': np.asarray([[0.0, 0.0], [0.0, 0.0]]),
            'a4': np.asarray([[2.0, -1.0], [1.0, -2.0]]),
            'a8': np.asarray([[1.0, 1.0], [0.0, 0.0]]),
        }
        q8 = {
            'data': np.asarray([[0.0, 0.0], [0.0, 0.0]]),
            'a4': np.asarray([[1.0, 2.0], [0.0, 1.0]]),
            'a8': np.asarray([[2.0, 0.0], [1.0, -1.0]]),
        }
        result = m18_cross_actor_critic.d6_per_sample(q4, q8, tolerance=1e-6)
        np.testing.assert_allclose(result['Q4_a4'], np.asarray([1.0, -2.0]))
        np.testing.assert_allclose(result['Delta_Q4_self'], np.asarray([1.0, -2.0]))
        np.testing.assert_allclose(result['Delta_Q8_self'], np.asarray([1.0, -2.0]))
        np.testing.assert_array_equal(result['self_preference_4'], np.asarray([1, 0], dtype=np.int8))
        np.testing.assert_array_equal(result['self_preference_8'], np.asarray([1, 0], dtype=np.int8))
        np.testing.assert_array_equal(result['joint_self_preference'], np.asarray([1, 0], dtype=np.int8))
        np.testing.assert_allclose(result['Delta_Q4_own_vs_data'], np.asarray([1.0, -2.0]))
        rows = {row['metric']: row for row in m18_cross_actor_critic.d6_summary_rows(result)}
        self.assertAlmostEqual(rows['joint_self_preference']['mean'], 0.5)
        self.assertAlmostEqual(rows['Delta_Q4_self']['positive_fraction'], 0.5)

    def test_d3_native_final_action_loader_reuses_exact_saved_action(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'actor_metrics.npz'
            clipped = np.zeros((3, 5, 2), dtype=np.float64)
            clipped[:, 4] = np.asarray([[0.1, -0.2], [0.3, -0.4], [0.5, -0.6]])
            np.savez_compressed(
                path,
                sample_id=np.asarray([10, 11, 12], dtype=np.int64),
                iteration_k=np.arange(5, dtype=np.int64), slot=np.asarray('actor'),
                K_train=np.asarray(4, dtype=np.int64), checkpoint_role=np.asarray('best'),
                checkpoint_step=np.asarray(9, dtype=np.int64), ensemble_member=np.asarray(''),
                clipped_action=clipped,
                normal_actor_mode_at_train_k=clipped[:, 4].copy(),
            )
            action, metadata = m18_cross_actor_critic._load_d3_final_action(
                path,
                train_k=4,
                expected_sample_id=np.asarray([10, 11, 12]),
                max_samples=3,
                expected_checkpoint_step=9,
            )
            np.testing.assert_array_equal(action, clipped[:, 4])
            self.assertEqual(metadata['iteration_k'], 4)
            self.assertEqual(metadata['checkpoint_step'], 9)
            self.assertEqual(metadata['normal_mode_clipped_parity_max_abs_error'], 0.0)

    def test_puzzle_oracle_extraction_transition_distance_and_pair_plan(self):
        oracle = Puzzle4x4LogicalOracle()
        self.assertEqual(oracle.state_count, 2 ** 16)
        zero = np.zeros(16, dtype=np.int8)
        next_state = oracle.transition_states(zero, 5)
        self.assertEqual(oracle.distance(zero, zero), 0)
        self.assertEqual(oracle.distance(next_state, zero), 1)
        event = oracle.classify_observed_transition(zero, next_state)
        self.assertTrue(event['verified_single_press_event'])
        self.assertEqual(event['pressed_button_id'], 5)
        optimal = oracle.optimal_pressed_buttons(next_state, zero)
        self.assertIn(5, optimal)
        self.assertEqual(oracle.distance(oracle.transition_states(next_state, 5), zero), 0)
        invariant = oracle.validate_distance_invariants(zero)
        self.assertEqual(invariant['goal_distance'], 0)
        self.assertGreater(invariant['reachable_state_count'], 0)
        forward = m18_paired_rollout_diagnostics.paired_episode_plan((1, 2), 2, 18018)
        reverse = m18_paired_rollout_diagnostics.paired_episode_plan((2, 1), 2, 18018)
        self.assertEqual(forward, reverse)

    def test_paired_goal_manifest_is_byte_identical_for_both_model_consumers(self):
        """D5's raw policy goal is shared, despite env-local goal rendering noise."""

        import ogbench

        plan = m18_paired_rollout_diagnostics.paired_episode_plan((1, 2), 1, 18018)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrays, created = m18_paired_rollout_diagnostics._create_paired_goal_manifest(root, plan)
            loaded, recovered = m18_paired_rollout_diagnostics._load_paired_goal_manifest(root, plan)
            self.assertEqual(created['fingerprint'], recovered['fingerprint'])
            self.assertEqual(len(loaded), len(plan))
            oracle = Puzzle4x4LogicalOracle()
            env = ogbench.make_env_and_datasets('puzzle-4x4-play-v0', env_only=True)
            try:
                for index, paired in enumerate(plan):
                    shared_goal = loaded[paired['paired_episode_id']]['policy_goal']
                    # These two copies model the K4 and K8 worker inputs.  The
                    # requirement is byte identity, not merely equal logical
                    # button targets.
                    k4_goal = np.asarray(shared_goal).copy()
                    k8_goal = np.asarray(shared_goal).copy()
                    np.testing.assert_array_equal(k4_goal, k8_goal)
                    np.testing.assert_array_equal(k4_goal, arrays['policy_goal'][index])
                    _, info = env.reset(
                        seed=int(paired['episode_seed']),
                        options={'task_id': int(paired['task_id']), 'render_goal': False},
                    )
                    self.assertEqual(
                        oracle.encode(oracle.extract_logical_state(k4_goal)),
                        oracle.encode(oracle.extract_logical_state(np.asarray(info['goal']))),
                    )
                    self.assertEqual(
                        loaded[paired['paired_episode_id']]['policy_goal_logical_configuration'],
                        int(arrays['policy_goal_logical_configuration'][index]),
                    )
            finally:
                env.close()

    def test_real_puzzle_environment_transition_oracle_parity(self):
        import ogbench

        env = ogbench.make_env_and_datasets('puzzle-4x4-play-v0', env_only=True)
        try:
            audit = audit_real_puzzle_environment(env, validation_seed=18018, transition_cases=8)
        finally:
            env.close()
        self.assertTrue(audit['environment_semantics_audit_passed'], audit['errors'])
        self.assertTrue(audit['exact_shortest_distance_available'], audit['errors'])
        self.assertEqual(audit['transition_cases_passed'], 8)


class M18DRealPuzzleSmokeTest(unittest.TestCase):
    @unittest.skipUnless(DATASET_AVAILABLE, 'real Puzzle-4x4 train/validation data unavailable')
    def test_fixed_batch_is_reproducible_and_actor_metrics_are_finite(self):
        """A tiny real-data D2/D3/D4 probe; never writes outside TemporaryDirectory."""

        configuration = _configuration(4)
        source_run = make_run_path(
            REAL_RUN_ROOT, 'M18', configuration.config_id, configuration.slug,
            'puzzle-4x4-play-v0', 0,
        )
        try:
            provenance = m18_cross_k_eval.validate_m18_best_source(source_run)
        except (FileNotFoundError, ReevaluationError) as error:
            self.skipTest(f'real M18 K4 best checkpoint unavailable: {error}')
        before_hash = sha256_file(provenance['checkpoint_path'])
        with tempfile.TemporaryDirectory() as directory:
            batch_root = Path(directory) / 'fixed_batch'
            batch_a, metadata_a = m18_trace_diagnostics._create_fixed_batch(
                batch_root, provenance, 16, 181801,
            )
            batch_b, metadata_b = m18_trace_diagnostics._create_fixed_batch(
                batch_root, provenance, 16, 181801,
            )
            self.assertEqual(metadata_a['batch_fingerprint_sha256'], metadata_b['batch_fingerprint_sha256'])
            for key in batch_a:
                np.testing.assert_array_equal(batch_a[key], batch_b[key])
            agent, env, _, _ = m18_trace_diagnostics._build_restored_agent(provenance)
            try:
                actor_trace = m18_trace_diagnostics._trace_slot(agent, 'actor', batch_a, 8)
                metrics = m18_trace_diagnostics._state_metrics(
                    actor_trace['token_states'], actor_trace['readout_states'],
                )
                clipped, action_metrics = m18_trace_diagnostics._action_metrics(
                    actor_trace['action_means'], batch_a['dataset_actions'],
                )
                q_metrics, _ = m18_trace_diagnostics._critic_action_metrics(agent, batch_a, clipped)
                m18_trace_diagnostics._finite_or_nan(metrics | action_metrics | q_metrics)
                normal_mean = np.asarray(agent.network.select('actor')(
                    batch_a['observations'], batch_a['actor_goals'], temperature=0.0,
                ).mode())
                np.testing.assert_allclose(
                    actor_trace['action_means'][:, 4], normal_mean, rtol=1e-6, atol=1e-6,
                )
            finally:
                env.close()
        self.assertEqual(before_hash, sha256_file(provenance['checkpoint_path']))


if __name__ == '__main__':
    unittest.main()
