"""M11B design, canonical-override, shape, and aggregation tests."""

import json
import os
import unittest
from pathlib import Path

import gymnasium
import jax.numpy as jnp
import numpy as np

from impls.agents.crl import get_config as crl_get_config
from impls.agents.hiql import get_config as hiql_get_config
from impls.experiment import config_fingerprint, load_study, prepare_run_design
from impls.experiment.m11b import (
    ALL_ENVIRONMENTS,
    ANCHOR_ENVIRONMENT,
    CANONICAL_SOURCE,
    CRL_CONDITIONS,
    HIQL_CONDITIONS,
    PROTOCOL,
    aggregate_factorial_rows,
    canonical_hyperparameters,
    computation_slots,
    config_specs,
    m11b_agent_overrides,
    normalized_eval_auc,
    single_state_spec,
)
from impls.main import _make_config, _parse_args, _resolved_compute_snapshot
from tools.m11b_doctor import _underlying_env_id


ROOT = Path(__file__).resolve().parents[2]
STUDY_PATH = ROOT / 'experiments' / 'M11B_cross_task_computation' / 'study.yaml'
DATASET_ROOT = Path(os.environ.get('OGBENCH_DATASET_DIR', '/data/qijunrong/06-RL/offline-rl/data/raw_ogbench'))


class M11BStudyTest(unittest.TestCase):
    def test_exact_34_configuration_order_and_fixed_environment_runs(self):
        study = load_study(STUDY_PATH)
        specs = config_specs()
        config_paths = sorted((STUDY_PATH.parent / 'configs').glob('M11B-C*.yaml'))
        self.assertEqual(len(config_paths), 34)
        self.assertEqual([spec['config_id'] for spec in specs], [path.stem for path in config_paths])
        self.assertEqual(study.data['environments'], list(ALL_ENVIRONMENTS))
        self.assertEqual(study.data['seeds'], [0])
        labels = []
        for spec in specs:
            _, configuration = prepare_run_design(STUDY_PATH, spec['config_id'])
            self.assertEqual(configuration.data['environment'], spec['environment'])
            self.assertEqual(configuration.data['semantic_label'], spec['semantic_label'])
            labels.append(configuration.data['semantic_label'])
        self.assertEqual(len(labels), len(set(labels)))

    def test_crl_and_hiql_factorials_have_only_declared_slots(self):
        for condition in CRL_CONDITIONS:
            slots = computation_slots('crl', condition)
            enabled = {name for name, spec in slots.items() if spec['enabled']}
            expected = {
                'baseline': set(),
                'critic_ss': {'critic_state', 'critic_goal'},
                'actor_ss': {'actor'},
                'actor_critic_ss': {'actor', 'critic_state', 'critic_goal'},
            }[condition]
            self.assertEqual(enabled, expected)
            self.assertFalse(any(slots[name]['enabled'] for name in ('value_state', 'value_goal')))
        for condition in HIQL_CONDITIONS:
            slots = computation_slots('hiql', condition)
            enabled = {name for name, spec in slots.items() if spec['enabled']}
            expected = {
                'baseline': set(),
                'high_ss': {'high_actor'},
                'low_ss': {'low_actor'},
                'high_low_ss': {'high_actor', 'low_actor'},
            }[condition]
            self.assertEqual(enabled, expected)
            self.assertFalse(slots['value']['enabled'])

    def test_single_state_semantics_are_frozen_by_role(self):
        actor = single_state_spec('actor')['topology_kwargs']
        critic = single_state_spec('critic')['topology_kwargs']
        for spec in (actor, critic):
            self.assertEqual(spec['iterations'], 4)
            self.assertFalse(spec['residual'])
            self.assertEqual(spec['input_injection'], 'z_plus_x')
            self.assertEqual(spec['state_dim'], 512)
            self.assertEqual(spec['state_init'], 'normal_buffer')
            self.assertEqual(spec['state_init_std'], 1.0)
        self.assertEqual(actor['update_depth'], 2)
        self.assertFalse(actor['layer_norm'])
        self.assertTrue(actor['update_activate_final'])
        self.assertEqual(critic['update_depth'], 3)
        self.assertTrue(critic['layer_norm'])
        self.assertFalse(critic['update_activate_final'])

    def test_canonical_giant_humanoid_and_stitch_overrides(self):
        crl_giant = canonical_hyperparameters('crl', 'antmaze-giant-navigate-v0')
        self.assertEqual(crl_giant['discount'], 0.995)
        humanoid = canonical_hyperparameters('hiql', 'humanoidmaze-large-navigate-v0')
        self.assertEqual(humanoid['discount'], 0.995)
        self.assertEqual(humanoid['subgoal_steps'], 100)
        stitch = canonical_hyperparameters('hiql', 'antmaze-large-stitch-v0')
        self.assertEqual(stitch['discount'], 0.99)
        self.assertEqual(stitch['actor_p_trajgoal'], 0.5)
        self.assertEqual(stitch['actor_p_randomgoal'], 0.5)
        self.assertEqual(stitch['dataset_class'], 'HGCDataset')
        self.assertEqual(CANONICAL_SOURCE, '/home/eai/Research/offline-rl/docs/ALGORITHM_HYPERPARAMETERS.md')

    def test_m11b_baselines_match_current_canonical_agent_defaults(self):
        for algorithm, get_config in (('crl', crl_get_config), ('hiql', hiql_get_config)):
            _, configuration = prepare_run_design(
                STUDY_PATH,
                'M11B-C001' if algorithm == 'crl' else 'M11B-C002',
            )
            resolved = _make_config(_parse_args(['--agent', algorithm]), configuration=configuration)
            defaults = get_config()
            expected = canonical_hyperparameters(algorithm, ANCHOR_ENVIRONMENT)
            for key, value in expected.items():
                actual = tuple(resolved[key]) if key.endswith('_dims') else resolved[key]
                expected_value = tuple(value) if key.endswith('_dims') else value
                self.assertEqual(actual, expected_value, msg=f'{algorithm}:{key}')
            self.assertEqual(
                _resolved_compute_snapshot(resolved),
                _resolved_compute_snapshot(defaults),
            )

    def test_humanoid_hiql_body_changes_do_not_change_hierarchy_fields(self):
        _, configuration = prepare_run_design(STUDY_PATH, 'M11B-C016')
        resolved = _make_config(_parse_args(['--agent', 'hiql']), configuration=configuration)
        defaults = hiql_get_config()
        for key in ('subgoal_steps', 'high_alpha', 'low_alpha', 'rep_dim', 'discount', 'actor_p_trajgoal', 'actor_p_randomgoal'):
            self.assertEqual(resolved[key], canonical_hyperparameters('hiql', configuration.data['environment'])[key])
            if key != 'subgoal_steps':
                self.assertEqual(resolved[key], defaults[key] if key not in {'discount'} else 0.995)

    def test_stitch_targeted_sampling_semantics_do_not_leak_navigate_mix(self):
        navigate = m11b_agent_overrides('crl', 'antmaze-large-navigate-v0', 'baseline')
        stitch = m11b_agent_overrides('crl', 'antmaze-large-stitch-v0', 'baseline')
        self.assertEqual((navigate['actor_p_trajgoal'], navigate['actor_p_randomgoal']), (1.0, 0.0))
        self.assertEqual((stitch['actor_p_trajgoal'], stitch['actor_p_randomgoal']), (0.5, 0.5))
        self.assertNotEqual(stitch['actor_p_trajgoal'], navigate['actor_p_trajgoal'])

    def test_real_environment_and_dataset_shapes_for_available_references(self):
        import ogbench  # noqa: F401

        for environment in ALL_ENVIRONMENTS[:-1]:
            train_path = DATASET_ROOT / f'{environment}.npz'
            val_path = DATASET_ROOT / f'{environment}-val.npz'
            self.assertTrue(train_path.is_file(), environment)
            self.assertTrue(val_path.is_file(), environment)
            with np.load(train_path, mmap_mode='r', allow_pickle=False) as data:
                observation_shape = data['observations'].shape[1:]
                action_shape = data['actions'].shape[1:]
            env = gymnasium.make(_underlying_env_id(environment))
            try:
                self.assertEqual(tuple(observation_shape), env.observation_space.shape)
                self.assertEqual(tuple(action_shape), env.action_space.shape)
            finally:
                env.close()

    def test_fingerprint_and_secondary_aggregation_are_deterministic(self):
        spec = config_specs()[3]
        resolved = m11b_agent_overrides(spec['algorithm'], spec['environment'], spec['condition'])
        payload = {
            'spec': spec,
            'resolved_agent': resolved,
            'dataset_root': str(DATASET_ROOT),
            'seed': 0,
            'training_protocol': PROTOCOL,
        }
        self.assertEqual(config_fingerprint(payload), config_fingerprint(json.loads(json.dumps(payload))))
        self.assertNotEqual(config_fingerprint(payload), config_fingerprint({**payload, 'seed': 1}))
        points = {step: step / 1_000_000 for step in PROTOCOL['auc_checkpoints']}
        self.assertAlmostEqual(normalized_eval_auc(points), 0.55)
        rows = []
        for algorithm, conditions in (('crl', CRL_CONDITIONS), ('hiql', HIQL_CONDITIONS)):
            for condition, value in zip(conditions, (0.2, 0.4, 0.5, 0.8)):
                rows.append({
                    'algorithm': algorithm,
                    'environment': ANCHOR_ENVIRONMENT,
                    'condition': condition,
                    'final_success': value,
                })
        result = aggregate_factorial_rows(rows)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(np.isclose(row['interaction'], 0.1) for row in result))


if __name__ == '__main__':
    unittest.main()
