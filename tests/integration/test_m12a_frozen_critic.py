"""CPU-sized mechanism and provenance tests for M12A."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze, unfreeze

from experiments.M12A_frozen_critic_policy_extraction.preflight import (
    print_dry_run,
    validate_design,
)
from impls.agents.crl import CRLAgent, get_config
from impls.agents.crl_policy_extractor import CRLPolicyExtractorAgent
from impls.experiment import (
    Configuration,
    make_run_path,
    prepare_run_design,
    validate_source_run_dependency,
)
from impls.main import (
    _assert_frozen_dependencies,
    _evaluate_tasks,
    _make_config,
    _parse_args,
)
from impls.utils.datasets import Dataset, GCDataset
from impls.utils.checkpointing import (
    parameter_module_key,
    tree_fingerprint,
    write_checkpoint_index,
)
from impls.utils.flax_utils import restore_module_from_checkpoint, save_semantic_checkpoint


def _small_config():
    config = get_config()
    config['batch_size'] = 4
    config['actor_hidden_dims'] = (8, 8)
    config['value_hidden_dims'] = (8, 8)
    config['latent_dim'] = 4
    config['actor_loss'] = 'ddpgbc'
    config['const_std'] = True
    config['layer_norm'] = False
    config['p_aug'] = 0.0
    return config


def _small_extraction_config(single_state=False):
    config = _small_config()
    config['training_mode'] = 'policy_extraction'
    config['runtime_variant'] = 'policy_extractor'
    if single_state:
        actor = config['compute']['actor']
        actor['enabled'] = True
        actor['primitive'] = 'mlp'
        actor['topology'] = 'single_state'
        actor['credit'] = 'direct'
        actor['topology_kwargs'] = {
            'iterations': 4,
            'residual': False,
            'input_injection': 'z_plus_x',
            'state_dim': 8,
            'state_init': 'normal_buffer',
            'state_init_std': 1.0,
            'update_depth': 2,
            'layer_norm': False,
            'update_activate_final': True,
        }
    return config


def _batch(index=0, batch_size=4, obs_dim=4, action_dim=2):
    observations = (
        jnp.arange(batch_size * obs_dim, dtype=jnp.float32)
        .reshape(batch_size, obs_dim)
        / 17.0
    )
    observations = observations + index * 0.001
    goals = jnp.roll(observations, index % batch_size, axis=0) + 0.13
    actions = (
        jnp.arange(batch_size * action_dim, dtype=jnp.float32)
        .reshape(batch_size, action_dim)
        / 19.0
        - 0.2
    )
    return {
        'observations': observations,
        'value_goals': goals,
        'actor_goals': goals * 0.7,
        'actions': actions,
    }


def _module_params(agent, module):
    return agent.network.params[parameter_module_key(agent.network.params, module)]


class M12AMechanismTest(unittest.TestCase):
    def test_critic_only_loss_is_canonical_critic_loss(self):
        config = _small_config()
        batch = _batch()
        agent = CRLAgent.create(0, batch['observations'], batch['actions'], config)
        expected_loss, expected_info = agent.contrastive_loss(
            batch, agent.network.params, module_name='critic'
        )
        actual_loss, actual_info = agent.critic_only_loss(batch, agent.network.params)
        np.testing.assert_array_equal(np.asarray(actual_loss), np.asarray(expected_loss))
        self.assertEqual(set(actual_info), set(expected_info))
        for key in actual_info:
            np.testing.assert_array_equal(np.asarray(actual_info[key]), np.asarray(expected_info[key]))

    def test_joint_and_critic_only_critic_trajectories_match_for_100_steps(self):
        config = _small_config()
        first = _batch()
        joint = CRLAgent.create(1, first['observations'], first['actions'], config)
        critic_only = CRLAgent.create(1, first['observations'], first['actions'], config)
        for index in range(100):
            batch = _batch(index)
            joint, _ = joint.update(batch)
            critic_only, _ = critic_only.critic_only_update(batch)
        self.assertEqual(
            tree_fingerprint(_module_params(joint, 'critic')),
            tree_fingerprint(_module_params(critic_only, 'critic')),
        )
        probe = _batch(101)
        joint_q = joint.network.select('critic')(
            probe['observations'], probe['value_goals'], probe['actions'],
        )
        critic_only_q = critic_only.network.select('critic')(
            probe['observations'], probe['value_goals'], probe['actions'],
        )
        for joint_value, critic_only_value in zip(joint_q, critic_only_q):
            np.testing.assert_array_equal(
                np.asarray(joint_value), np.asarray(critic_only_value),
            )

    def test_critic_only_keeps_actor_bitwise_unchanged_for_100_steps(self):
        config = _small_config()
        first = _batch()
        agent = CRLAgent.create(2, first['observations'], first['actions'], config)
        before = tree_fingerprint(_module_params(agent, 'actor'))
        for index in range(100):
            agent, _ = agent.critic_only_update(_batch(index))
        self.assertEqual(before, tree_fingerprint(_module_params(agent, 'actor')))

    def test_policy_extractor_freezes_critic_and_changes_actor(self):
        config = _small_config()
        config['training_mode'] = 'policy_extraction'
        config['runtime_variant'] = 'policy_extractor'
        batch = _batch()
        agent = CRLPolicyExtractorAgent.create(
            3, batch['observations'], batch['actions'], config
        )
        critic_before = tree_fingerprint(_module_params(agent, 'critic'))
        actor_before = tree_fingerprint(_module_params(agent, 'actor'))
        agent, info = agent.update(batch)
        self.assertIn('frozen/q_delta', info)
        self.assertEqual(critic_before, tree_fingerprint(_module_params(agent, 'critic')))
        self.assertNotEqual(actor_before, tree_fingerprint(_module_params(agent, 'actor')))

    def test_policy_extractor_reuses_canonical_ddpgbc_actor_loss(self):
        config = _small_config()
        batch = _batch()
        base = CRLAgent.create(4, batch['observations'], batch['actions'], config)
        config['training_mode'] = 'policy_extraction'
        config['runtime_variant'] = 'policy_extractor'
        extracted = CRLPolicyExtractorAgent.create(
            4, batch['observations'], batch['actions'], config
        )
        base_loss, base_info = base.actor_loss(
            batch, base.network.params, rng=jax.random.PRNGKey(9)
        )
        extracted_loss, extracted_info = extracted.actor_loss(
            batch, extracted.network.params, rng=jax.random.PRNGKey(9)
        )
        np.testing.assert_array_equal(np.asarray(base_loss), np.asarray(extracted_loss))
        self.assertEqual(set(base_info), set(extracted_info))
        for key in base_info:
            np.testing.assert_array_equal(np.asarray(base_info[key]), np.asarray(extracted_info[key]))

    def test_module_restore_uses_source_critic_without_source_optimizer_state(self):
        config = _small_config()
        batch = _batch()
        source = CRLAgent.create(5, batch['observations'], batch['actions'], config)
        target_config = copy.deepcopy(config)
        target_config['training_mode'] = 'policy_extraction'
        target_config['runtime_variant'] = 'policy_extractor'
        target = CRLPolicyExtractorAgent.create(
            5, batch['observations'], batch['actions'], target_config
        )
        target_opt_state_before = [
            np.asarray(leaf).copy()
            for leaf in jax.tree_util.tree_leaves(target.network.opt_state)
        ]
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / 'source'
            (run_dir / 'checkpoints').mkdir(parents=True)
            last = save_semantic_checkpoint(
                source,
                run_dir,
                'last',
                1_000_000,
                {'checkpoint_role': 'last', 'checkpoint_step': 1_000_000},
            )
            write_checkpoint_index(run_dir, best=None, last=last)
            restored = restore_module_from_checkpoint(
                target, run_dir / last['path'], 'critic'
            )
            self.assertEqual(
                tree_fingerprint(_module_params(source, 'critic')),
                tree_fingerprint(_module_params(restored, 'critic')),
            )
            self.assertEqual(
                tree_fingerprint(_module_params(source, 'actor')),
                tree_fingerprint(_module_params(restored, 'actor')),
            )
            for before, after in zip(
                target_opt_state_before,
                jax.tree_util.tree_leaves(restored.network.opt_state),
            ):
                np.testing.assert_array_equal(before, np.asarray(after))


class M12AStudyAndDependencyTest(unittest.TestCase):
    def test_design_has_three_configs_and_stage1_eval_is_none(self):
        study, configs, errors = validate_design()
        self.assertFalse(errors, errors)
        self.assertEqual(study.data['seeds'], [0, 1, 2])
        self.assertEqual([item[0].config_id for item in configs], [
            'M12A-C001', 'M12A-C002', 'M12A-C003'
        ])
        self.assertFalse(Path('impls/experiment/m12a.py').exists())
        self.assertEqual(_parse_args(['--eval_tasks', 'none']).eval_tasks, 'none')
        self.assertEqual(_parse_args(['--eval_tasks', 'all']).eval_tasks, 'all')
        self.assertEqual(
            _evaluate_tasks(None, None, None, _parse_args(['--eval_tasks', 'none']), 0),
            {},
        )

    def test_paired_c002_c003_data_stream_matches_for_first_10_batches(self):
        study, c002 = prepare_run_design(
            'experiments/M12A_frozen_critic_policy_extraction/study.yaml',
            'M12A-C002',
        )
        _, c003 = prepare_run_design(
            'experiments/M12A_frozen_critic_policy_extraction/study.yaml',
            'M12A-C003',
        )
        config_ff = _make_config(_parse_args(['--agent', 'crl']), configuration=c002)
        config_ss = _make_config(_parse_args(['--agent', 'crl']), configuration=c003)
        del study
        size = 60
        observations = np.arange(size * 4, dtype=np.float32).reshape(size, 4)
        actions = np.arange(size * 2, dtype=np.float32).reshape(size, 2) / 31.0
        terminals = np.zeros(size, dtype=np.float32)
        terminals[9::10] = 1.0
        raw = {
            'observations': observations,
            'actions': actions,
            'terminals': terminals,
        }
        stream_ff = GCDataset(
            Dataset.create(**raw), config_ff, rng=12345,
        )
        stream_ss = GCDataset(
            Dataset.create(**raw), config_ss, rng=12345,
        )
        for _ in range(10):
            batch_ff = stream_ff.sample(4)
            batch_ss = stream_ss.sample(4)
            self.assertEqual(tree_fingerprint(batch_ff), tree_fingerprint(batch_ss))

    def test_stage2_evaluation_all_uses_20_episodes_and_zero_temperature(self):
        class FakeEnv:
            unwrapped = None

        env = FakeEnv()
        env.unwrapped = env
        env.task_infos = [{'task_name': 'task_a'}, {'task_name': 'task_b'}]
        args = _parse_args([
            '--eval_tasks', 'all', '--eval_episodes', '20',
            '--eval_temperature', '0.0',
        ])
        with mock.patch('impls.main.evaluate', return_value=({'success': 1.0}, [], [])) as evaluate_mock:
            metrics = _evaluate_tasks(None, env, {'discrete': False}, args, 77)
        self.assertEqual(metrics['evaluation/overall_success'], 1.0)
        self.assertEqual(evaluate_mock.call_count, 2)
        for call in evaluate_mock.call_args_list:
            self.assertEqual(call.kwargs['num_eval_episodes'], 20)
            self.assertEqual(call.kwargs['eval_temperature'], 0.0)
            self.assertIsNone(call.kwargs['eval_gaussian'])

    def test_restore_then_real_update_preserves_ff_and_ss_optimizer_pytrees(self):
        study, c001 = prepare_run_design(
            'experiments/M12A_frozen_critic_policy_extraction/study.yaml',
            'M12A-C001',
        )
        _, c002 = prepare_run_design(
            'experiments/M12A_frozen_critic_policy_extraction/study.yaml',
            'M12A-C002',
        )
        _, c003 = prepare_run_design(
            'experiments/M12A_frozen_critic_policy_extraction/study.yaml',
            'M12A-C003',
        )
        batch = _batch()
        source_agent = CRLAgent.create(
            19, batch['observations'], batch['actions'], _small_config(),
        )
        for label, configuration, extraction_config in (
            ('FF', c002, _small_extraction_config(False)),
            ('SS-K4', c003, _small_extraction_config(True)),
        ):
            with self.subTest(actor=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self._make_completed_source_run(
                    root, study, c001, source_agent, seed=0,
                )
                dependency = validate_source_run_dependency(
                    study,
                    configuration,
                    'frozen_critic',
                    seed=0,
                    run_root=root,
                    resolved_agent=None,
                )
                target = CRLPolicyExtractorAgent.create(
                    19,
                    batch['observations'],
                    batch['actions'],
                    extraction_config,
                )
                params_before = target.network.params
                structure_before = jax.tree_util.tree_structure(params_before)
                optimizer_before = [
                    np.asarray(leaf).copy()
                    for leaf in jax.tree_util.tree_leaves(target.network.opt_state)
                ]
                actor_before = tree_fingerprint(_module_params(target, 'actor'))
                target = restore_module_from_checkpoint(
                    target, dependency['checkpoint_path'], dependency['module'],
                )
                self.assertEqual(
                    structure_before,
                    jax.tree_util.tree_structure(target.network.params),
                )
                for before, after in zip(
                    optimizer_before,
                    jax.tree_util.tree_leaves(target.network.opt_state),
                ):
                    np.testing.assert_array_equal(before, np.asarray(after))
                critic_before = tree_fingerprint(_module_params(target, 'critic'))
                for index in range(10):
                    target, info = target.update(_batch(index))
                    self.assertTrue(
                        all(np.all(np.isfinite(np.asarray(value))) for value in info.values()),
                        msg=f'non-finite {label} metrics at update {index}',
                    )
                self.assertNotEqual(actor_before, tree_fingerprint(_module_params(target, 'actor')))
                self.assertEqual(critic_before, tree_fingerprint(_module_params(target, 'critic')))
                self.assertEqual(target.network.step, 11)

    def _make_completed_source_run(
        self, root, study, source_configuration, source_agent, seed=0,
    ):
        source_run = make_run_path(
            root,
            study.study_id,
            source_configuration.config_id,
            source_configuration.slug,
            source_configuration.data['environment'],
            seed,
        )
        (source_run / 'checkpoints').mkdir(parents=True)
        last = save_semantic_checkpoint(
            source_agent,
            source_run,
            'last',
            1_000_000,
            {'checkpoint_role': 'last', 'checkpoint_step': 1_000_000},
        )
        write_checkpoint_index(source_run, best=None, last=last)
        (source_run / 'runtime_metadata.json').write_text(json.dumps({
            'status': 'completed',
            'config_id': source_configuration.config_id,
            'environment': source_configuration.data['environment'],
            'algorithm': source_configuration.data['algorithm'],
            'seed': seed,
            'run_attempt': 0,
        }))
        source_config = _make_config(
            _parse_args(['--agent', 'crl']), configuration=source_configuration
        )
        (source_run / 'resolved_config.json').write_text(json.dumps({
            'algorithm_config': {'agent': _jsonable(source_config)},
        }))
        return source_run

    def test_stage2_preflight_plans_six_runs_after_temporary_sources_exist(self):
        study, c001 = prepare_run_design(
            'experiments/M12A_frozen_critic_policy_extraction/study.yaml',
            'M12A-C001',
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in study.data['seeds']:
                batch = _batch(seed)
                source_agent = CRLAgent.create(
                    seed,
                    batch['observations'],
                    batch['actions'],
                    _small_config(),
                )
                self._make_completed_source_run(
                    root, study, c001, source_agent, seed=seed,
                )
            self.assertEqual(print_dry_run(2, root), 0)

    def test_dependency_accepts_only_completed_same_seed_last_at_1m_source(self):
        study, c001 = prepare_run_design(
            'experiments/M12A_frozen_critic_policy_extraction/study.yaml',
            'M12A-C001',
        )
        _, c002 = prepare_run_design(
            'experiments/M12A_frozen_critic_policy_extraction/study.yaml',
            'M12A-C002',
        )
        batch = _batch()
        source_agent = CRLAgent.create(6, batch['observations'], batch['actions'], _small_config())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_run = self._make_completed_source_run(root, study, c001, source_agent)
            target_config = _make_config(
                _parse_args(['--agent', 'crl']), configuration=c002
            )
            record = validate_source_run_dependency(
                study,
                c002,
                'frozen_critic',
                seed=0,
                run_root=root,
                resolved_agent=target_config,
            )
            self.assertEqual(Path(record['source_run_dir']), source_run.resolve())
            self.assertEqual(record['checkpoint_role'], 'last')
            self.assertEqual(record['checkpoint_step'], 1_000_000)
            self.assertTrue(record['module_fingerprint'])

    def test_dependency_rejects_best_selector_before_touching_source(self):
        study, c002 = prepare_run_design(
            'experiments/M12A_frozen_critic_policy_extraction/study.yaml',
            'M12A-C002',
        )
        data = copy.deepcopy(c002.data)
        data['dependencies']['frozen_critic']['checkpoint_role'] = 'best'
        invalid = Configuration(path=c002.path, data=data)
        with self.assertRaisesRegex(ValueError, 'checkpoint_role=last'):
            validate_source_run_dependency(
                study,
                invalid,
                'frozen_critic',
                seed=0,
                run_root=Path(tempfile.mkdtemp()),
            )

    def test_dependency_rejects_source_checkpoint_sha_mismatch(self):
        study, c001 = prepare_run_design(
            'experiments/M12A_frozen_critic_policy_extraction/study.yaml',
            'M12A-C001',
        )
        _, c002 = prepare_run_design(
            'experiments/M12A_frozen_critic_policy_extraction/study.yaml',
            'M12A-C002',
        )
        batch = _batch()
        source_agent = CRLAgent.create(7, batch['observations'], batch['actions'], _small_config())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_run = self._make_completed_source_run(root, study, c001, source_agent)
            checkpoint_path = next((source_run / 'checkpoints' / 'last').glob('params_*.pkl'))
            with checkpoint_path.open('ab') as file:
                file.write(b'corrupted-after-index')
            target_config = _make_config(
                _parse_args(['--agent', 'crl']), configuration=c002
            )
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                validate_source_run_dependency(
                    study,
                    c002,
                    'frozen_critic',
                    seed=0,
                    run_root=root,
                    resolved_agent=target_config,
                )

    def test_frozen_invariant_fails_loudly_if_critic_tree_changes(self):
        config = _small_config()
        batch = _batch()
        agent = CRLPolicyExtractorAgent.create(
            8,
            batch['observations'],
            batch['actions'],
            config,
        )
        module_key = parameter_module_key(agent.network.params, 'critic')
        expected = tree_fingerprint(agent.network.params[module_key])
        params = unfreeze(agent.network.params)
        params[module_key] = jax.tree_util.tree_map(
            lambda value: value + 1,
            params[module_key],
        )
        changed = agent.replace(network=agent.network.replace(params=freeze(params)))
        with self.assertRaisesRegex(ValueError, 'changed in memory'):
            _assert_frozen_dependencies(
                changed,
                {'frozen_critic': {
                    'module': 'critic',
                    'module_fingerprint': expected,
                }},
            )


def _jsonable(value):
    if isinstance(value, dict) or hasattr(value, 'items'):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, 'item'):
        return value.item()
    return value


if __name__ == '__main__':
    unittest.main()
