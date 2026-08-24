"""Contract tests for M12-D, independent of formal experiment artifacts."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from impls.agents.crl import get_config
from impls.agents.crl_policy_extractor import CRLPolicyExtractorAgent
from impls.diagnostics.banks import (
    arrays_hash,
    build_eval_goal_bank,
    build_training_support_bank,
    load_bank,
    save_bank,
)
from impls.diagnostics.checkpoints import actor_run_dir, actor_sources
from impls.diagnostics.metrics import evaluate_training_batch, pairwise_contrasts
from impls.diagnostics.rollout import select_progress_balanced_states
from impls.diagnostics.support import build_support_reference, support_distance
from impls.utils.checkpointing import tree_fingerprint
from impls.utils.datasets import Dataset

from experiments.M12D_fixed_critic_policy_realization_diagnosis.common import protocol_from_arg, source_config


def small_config():
    config = get_config()
    config['actor_hidden_dims'] = (8, 8)
    config['value_hidden_dims'] = (8, 8)
    config['latent_dim'] = 4
    config['layer_norm'] = False
    config['p_aug'] = 0.0
    return config


def small_agent():
    config = small_config()
    obs = jnp.ones((4, 3), dtype=jnp.float32)
    actions = jnp.zeros((4, 2), dtype=jnp.float32)
    return SimpleNamespace(
        agent=CRLPolicyExtractorAgent.create(0, obs, actions, config), config=config
    )


def batch(size=4):
    observations = jnp.arange(size * 3, dtype=jnp.float32).reshape(size, 3) / 7
    goals = observations + 0.1
    return {
        'observations': observations,
        'actions': jnp.zeros((size, 2), dtype=jnp.float32),
        'actor_goals': goals,
        'value_goals': goals,
    }


class M12DContractTest(unittest.TestCase):
    def test_A_primary_protocol(self):
        protocol = protocol_from_arg()
        self.assertEqual(tuple(protocol['primary_actor_names']), ('K1SN', 'K4SN', 'K4SZ', 'D9', 'Residual'))
        self.assertEqual(protocol['critic']['config'], 'M12A-C001')

    def test_B_bank_hash_and_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'bank'
            save_bank(root, {'x': np.arange(4)}, {'bank_type': 'test'})
            loaded = load_bank(root)
            self.assertEqual(loaded.manifest['bank_hash'], arrays_hash({'x': np.arange(4)}))
            with self.assertRaises(FileExistsError):
                save_bank(root, {'x': np.arange(4)}, {'bank_type': 'test'})

    def test_C_exact_gcdataset_sampler_deterministic(self):
        config = small_config()
        dataset = Dataset.create(
            seed=1,
            observations=np.arange(60, dtype=np.float32).reshape(20, 3),
            actions=np.zeros((20, 2), dtype=np.float32),
            terminals=np.array([0] * 9 + [1] + [0] * 9 + [1], dtype=np.float32),
        )
        first, _, _ = build_training_support_bank(
            dataset, config, seed=0, batches=2, batch_size=4, environment='env',
            dataset_root='/data', source_commit='sha',
        )
        second, _, _ = build_training_support_bank(
            dataset, config, seed=0, batches=2, batch_size=4, environment='env',
            dataset_root='/data', source_commit='sha',
        )
        self.assertEqual(arrays_hash(first), arrays_hash(second))

    def test_D_eval_bank_reuses_states(self):
        arrays = {
            'observations': np.arange(12, dtype=np.float32).reshape(4, 3),
            'actions': np.zeros((4, 2), dtype=np.float32),
            'dataset_indices': np.array([2, 5, 9, 11]),
        }
        parent = SimpleNamespace(arrays=arrays, manifest={'bank_hash': 'parent'})
        result, _, _ = build_eval_goal_bank(
            parent, eval_goals={1: np.ones(3), 2: np.zeros(3)},
            task_names={1: 'task1', 2: 'task2'}, environment='env', source_commit='sha',
            dataset_root='/data', evaluation_seed=1,
        )
        np.testing.assert_array_equal(result['dataset_indices'], np.tile(arrays['dataset_indices'], 2))

    def test_E_progress_balancing(self):
        records = [{
            'actor_name': 'A', 'task_id': 1, 'episode_index': 0, 'timestep': t,
            'progress': t / 4, 'observation': np.array([t]), 'eval_goal': np.array([1]),
            'actor_seed': 1, 'critic_seed': 0, 'episode_length': 5, 'episode_success': 1,
        } for t in range(5)]
        self.assertEqual(
            [row['timestep'] for row in select_progress_balanced_states(records)],
            [row['timestep'] for row in select_progress_balanced_states(records)],
        )

    def test_F_support_proxy_deterministic(self):
        observations = np.arange(30, dtype=np.float32).reshape(10, 3)
        arrays, manifest = build_support_reference(
            observations, max_states=5, environment='env', dataset_root='/data', source_commit='sha'
        )
        bank = SimpleNamespace(arrays=arrays, manifest=manifest)
        np.testing.assert_array_equal(support_distance(observations[:3], bank), support_distance(observations[:3], bank))
        self.assertIn('proxy', manifest['interpretation'])

    def test_G_qmin_pairwise(self):
        results = {
            'a': {'action_clipped': np.zeros((2, 1)), 'q_min': np.array([1., 2.])},
            'b': {'action_clipped': np.ones((2, 1)), 'q_min': np.array([2., 1.])},
        }
        row = pairwise_contrasts(results, ['a', 'b'])[0]
        self.assertAlmostEqual(row['action_l2_squared_mean'], 1.0)
        self.assertAlmostEqual(row['q_right_win_rate'], 0.5)

    def test_H_k4sn_attempt2(self):
        protocol = protocol_from_arg()
        source = actor_sources(protocol)['K4SN']
        path = actor_run_dir(source, run_root=protocol['run_root'], environment=protocol['environment'], seed=0)
        self.assertIn('M12A-C003', str(path))
        self.assertIn('__attempt_002', str(path))

    def test_I_d9_residual_identity(self):
        protocol = protocol_from_arg()
        for name, config_id in [('D9', 'M12B-C006'), ('Residual', 'M12B-C007')]:
            _, source, configuration = source_config(protocol, name)
            self.assertEqual(source.config, config_id)
            self.assertEqual(configuration.config_id, config_id)

    def test_J_exact_objective_and_no_mutation(self):
        actor = small_agent()
        data = batch()
        before = tree_fingerprint(actor.agent.network.params)
        loss, _ = actor.agent.policy_extraction_loss(data, actor.agent.network.params, rng=actor.agent.rng)
        row = evaluate_training_batch(actor, data)
        after = tree_fingerprint(actor.agent.network.params)
        np.testing.assert_allclose(row['actor_loss_return'], np.asarray(loss), rtol=1e-5, atol=1e-6)
        self.assertEqual(before, after)

    def test_K_no_internal_trace_api(self):
        text = '\\n'.join(path.read_text() for path in Path('impls/diagnostics').glob('*.py'))
        self.assertNotIn('z1', text)
        self.assertNotIn('dynamics_trace', text)

    def test_L_temporal_protocol(self):
        protocol = protocol_from_arg()
        self.assertEqual(protocol['temporal_checkpoints'], [800000, 900000, 1000000])
        self.assertTrue(protocol['formal_constraints']['no_training'])


if __name__ == '__main__':
    unittest.main()

