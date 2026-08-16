"""Real OGBench Dataset and HIQL parity checks for the first runtime slice."""

import copy
import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from impls.agents.hiql import HIQLAgent, get_config
from impls.utils.datasets import Dataset, HGCDataset
from impls.utils.env_utils import make_env_and_datasets, resolve_dataset_dir
from tests.computation.test_mlp_parity import (
    _assert_full_semantic_allclose,
    _assert_info_allclose,
    _graft_semantic_params,
)


DATASET_NAME = 'antmaze-medium-navigate-v0'
REQUIRED_BATCH_KEYS = (
    'observations',
    'next_observations',
    'actions',
    'rewards',
    'masks',
    'value_goals',
    'high_actor_goals',
    'high_actor_targets',
    'low_actor_goals',
)


def _runtime_config():
    config = get_config()
    config['batch_size'] = 8
    config['actor_hidden_dims'] = (6, 6)
    config['value_hidden_dims'] = (6, 6)
    config['rep_dim'] = 3
    config['subgoal_steps'] = 7
    config['p_aug'] = 0.0
    return config


def _set_slots(config, enabled):
    for name in ('low_actor', 'high_actor', 'value'):
        config['compute'][name]['enabled'] = enabled


def _copy_dataset(raw_dataset, seed):
    return Dataset.create(seed=seed, **dict(raw_dataset))


class RealHIQLRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dataset_path = os.path.join(resolve_dataset_dir(), f'{DATASET_NAME}.npz')
        if not os.path.exists(dataset_path):
            raise unittest.SkipTest(f'real OGBench dataset is unavailable: {dataset_path}')
        cls.env, cls.raw_train, cls.raw_val = make_env_and_datasets(
            DATASET_NAME,
            seed=1234,
            dataset_seed=5678,
        )
        cls.config = _runtime_config()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def _make_dataset_pair(self, config=None):
        config = self.config if config is None else config
        return (
            HGCDataset(_copy_dataset(self.raw_train, 2026), copy.deepcopy(config), rng=2026),
            HGCDataset(_copy_dataset(self.raw_train, 2026), copy.deepcopy(config), rng=2026),
        )

    def test_real_dataset_same_seed_is_bitwise_deterministic_for_twenty_batches(self):
        first, second = self._make_dataset_pair()
        max_error = 0.0
        for step in range(20):
            left = first.sample(self.config['batch_size'])
            right = second.sample(self.config['batch_size'])
            self.assertEqual(set(left), set(right), msg=f'batch {step} keys')
            for key in REQUIRED_BATCH_KEYS:
                self.assertIn(key, left)
                self.assertEqual(left[key].shape, right[key].shape, msg=f'batch {step} {key} shape')
                error = float(np.max(np.abs(np.asarray(left[key]) - np.asarray(right[key]))))
                max_error = max(max_error, error)
                np.testing.assert_array_equal(left[key], right[key], err_msg=f'batch {step} {key}')
        self.assertEqual(max_error, 0.0)

    def test_real_metadata_sampling_stays_inside_trajectories_and_is_finite(self):
        config = copy.deepcopy(self.config)
        config['value_p_curgoal'] = 1.0
        config['value_p_trajgoal'] = 0.0
        config['value_p_randomgoal'] = 0.0
        config['actor_p_curgoal'] = 0.0
        config['actor_p_trajgoal'] = 1.0
        config['actor_p_randomgoal'] = 0.0
        config['actor_geom_sample'] = False
        dataset, _ = self._make_dataset_pair(config)
        sampled_idxs = dataset.dataset.get_random_idxs(1024, rng=np.random.default_rng(91))
        self.assertGreaterEqual(int(np.min(sampled_idxs)), 0)
        self.assertLess(int(np.max(sampled_idxs)), dataset.size)
        if 'valids' in dataset.dataset:
            self.assertTrue(np.all(np.asarray(dataset.dataset['valids'])[sampled_idxs] > 0))
        idxs = np.asarray([1, 2, 17, 999, 1000, 1001, 2001, 3001], dtype=np.int64)
        batch = dataset.sample(len(idxs), idxs=idxs)

        terminals = np.asarray(dataset.dataset['terminals']) > 0
        terminal_locs = np.flatnonzero(terminals)
        final_idxs = terminal_locs[np.searchsorted(terminal_locs, idxs)]
        observations = np.asarray(dataset.dataset['observations'])

        def observation_index(observation):
            matches = np.flatnonzero(np.all(observations == observation, axis=1))
            self.assertGreater(len(matches), 0)
            return int(matches[0])

        for row, idx in enumerate(idxs):
            low_idx = observation_index(batch['low_actor_goals'][row])
            high_idx = observation_index(batch['high_actor_goals'][row])
            target_idx = observation_index(batch['high_actor_targets'][row])
            final_idx = int(final_idxs[row])
            self.assertGreaterEqual(low_idx, idx)
            self.assertLessEqual(low_idx, min(idx + config['subgoal_steps'], final_idx))
            self.assertGreaterEqual(high_idx, min(idx + 1, final_idx))
            self.assertLessEqual(high_idx, final_idx)
            self.assertEqual(target_idx, min(idx + config['subgoal_steps'], high_idx))
            self.assertLessEqual(target_idx, final_idx)
            self.assertEqual(observation_index(batch['value_goals'][row]), idx)

        self.assertEqual(batch['observations'].shape, (len(idxs), observations.shape[-1]))
        self.assertEqual(batch['actions'].shape[0], len(idxs))
        for key in REQUIRED_BATCH_KEYS:
            self.assertTrue(np.all(np.isfinite(np.asarray(batch[key]))), msg=key)
        # OGBench's reference equations use Python ``float`` for these two
        # derived arrays, which is float64 on the host; JAX applies its normal
        # x64-disabled promotion when the batch enters the loss.
        self.assertTrue(np.issubdtype(np.asarray(batch['rewards']).dtype, np.floating))
        self.assertTrue(np.issubdtype(np.asarray(batch['masks']).dtype, np.floating))

    def test_real_batches_have_strict_legacy_computationized_hiql_parity_for_twenty_updates(self):
        old_config = copy.deepcopy(self.config)
        new_config = copy.deepcopy(self.config)
        _set_slots(old_config, False)
        _set_slots(new_config, True)
        old_dataset, new_dataset = self._make_dataset_pair(old_config)
        old_dataset.config = old_config
        new_dataset.config = new_config

        # Keep the two training wrappers at the same RNG position.  Sampling
        # an initialization example from a third, identically seeded wrapper
        # avoids consuming only the legacy stream.
        init_dataset, _ = self._make_dataset_pair(old_config)
        initial_batch = init_dataset.sample(2)
        old_agent = HIQLAgent.create(44, initial_batch['observations'], initial_batch['actions'], old_config)
        new_agent = HIQLAgent.create(44, initial_batch['observations'], initial_batch['actions'], new_config)
        slots = {'low_actor': True, 'high_actor': True, 'value': True}
        grafted = _graft_semantic_params(
            old_agent.network.params,
            new_agent.network.params,
            slots,
            value_enabled=True,
        )
        new_agent = new_agent.replace(network=new_agent.network.replace(params=grafted))

        max_loss_error = 0.0
        max_gradient_error = 0.0
        first_divergence = None
        for step in range(20):
            old_batch = old_dataset.sample(old_config['batch_size'])
            new_batch = new_dataset.sample(new_config['batch_size'])
            for key in REQUIRED_BATCH_KEYS:
                np.testing.assert_array_equal(old_batch[key], new_batch[key], err_msg=f'batch {step} {key}')

            old_loss, old_info = old_agent.total_loss(old_batch, old_agent.network.params)
            new_loss, new_info = new_agent.total_loss(new_batch, new_agent.network.params)
            loss_error = float(np.max(np.abs(np.asarray(old_loss) - np.asarray(new_loss))))
            max_loss_error = max(max_loss_error, loss_error)
            if first_divergence is None and loss_error > 1e-6:
                first_divergence = (step, 'total_loss', loss_error)
            np.testing.assert_allclose(np.asarray(old_loss), np.asarray(new_loss), rtol=0, atol=1e-6)
            _assert_info_allclose(self, old_info, new_info, f'real N=20 step={step} loss')

            old_grads = jax.grad(lambda params: old_agent.total_loss(old_batch, params)[0])(old_agent.network.params)
            new_grads = jax.grad(lambda params: new_agent.total_loss(new_batch, params)[0])(new_agent.network.params)
            gradient_error = _assert_full_semantic_allclose(
                self,
                old_grads,
                new_grads,
                slots,
                True,
                f'real N=20 step={step} gradients',
            )
            max_gradient_error = max(max_gradient_error, gradient_error)

            old_agent, old_update_info = old_agent.update(old_batch)
            new_agent, new_update_info = new_agent.update(new_batch)
            _assert_info_allclose(
                self,
                old_update_info,
                new_update_info,
                f'real N=20 step={step} update',
                skip_keys=('grad/norm',),
            )
            _assert_full_semantic_allclose(
                self,
                old_agent.network.params,
                new_agent.network.params,
                slots,
                True,
                f'real N=20 step={step} params and target',
            )
            np.testing.assert_array_equal(np.asarray(old_agent.rng), np.asarray(new_agent.rng))

        self.assertIsNone(first_divergence, msg=f'first divergence: {first_divergence}')
        self.assertEqual(max_loss_error, 0.0)
        self.assertEqual(max_gradient_error, 0.0)


if __name__ == '__main__':
    unittest.main()
