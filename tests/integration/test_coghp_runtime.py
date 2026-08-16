"""Regression tests for the official vanilla CoGHP migration."""

import copy
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from impls.agents.coghp import CoGHPAgent, get_config
from impls.utils.datasets import Dataset, MultiHGCDataset


def _tiny_config():
    config = copy.deepcopy(get_config())
    config.actor_hidden_dims = (6,)
    config.value_hidden_dims = (6,)
    config.enc_hidden_dims = (5,)
    config.feature_dim = 4
    config.mixer_hidden = 3
    config.num_mixer_blocks = 2
    config.num_subgoals = 2
    config.batch_size = 3
    return config


def _count(tree):
    return sum(np.asarray(leaf).size for leaf in jax.tree_util.tree_leaves(tree))


class CoGHPRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.config = _tiny_config()
        self.observations = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 10
        self.actions = jnp.arange(6, dtype=jnp.float32).reshape(3, 2) / 10
        self.agent = CoGHPAgent.create(0, self.observations, self.actions, self.config)

    def test_official_mixer_sharing_and_parameter_accounting(self):
        actor_params = self.agent.network.params['modules_actor_mixer']
        self.assertEqual(
            sorted(actor_params['mixer_blocks_0'].keys()),
            ['channel_dense1', 'channel_dense2', 'tm_weights', 'token_dense1', 'token_dense2'],
        )
        self.assertEqual(len([key for key in actor_params if key.startswith('mixer_blocks_')]), 2)
        self.assertEqual(_count(self.agent.network.params['modules_goal_rep']), 79)
        self.assertEqual(_count(self.agent.network.params['modules_value']), 146)
        self.assertEqual(_count(self.agent.network.params['modules_target_value']), 146)
        self.assertEqual(_count(actor_params), 361)
        self.assertEqual(_count(self.agent.network.params), 732)

        high = actor_params['high_actor_head']
        low = actor_params['low_actor_head']
        self.assertIn('mean_net', high)
        self.assertIn('mean_net', low)
        self.assertNotEqual(high['mean_net']['kernel'].shape, low['mean_net']['kernel'].shape)

    def test_multihgc_sampling_is_deterministic_and_has_official_fields(self):
        observations = np.arange(30 * 4, dtype=np.float32).reshape(30, 4)
        actions = np.zeros((30, 2), dtype=np.float32)
        terminals = np.zeros(30, dtype=np.float32)
        terminals[9::10] = 1
        fields = dict(observations=observations, actions=actions, terminals=terminals)
        first = MultiHGCDataset(
            Dataset.create(freeze=False, seed=10, **fields), self.config, rng=2026
        )
        second = MultiHGCDataset(
            Dataset.create(freeze=False, seed=10, **fields), self.config, rng=2026
        )
        required = {
            'observations', 'next_observations', 'actions', 'value_goals',
            'low_actor_goals', 'high_actor_goals', 'high_actor_targets',
            'masks', 'rewards',
        }
        for _ in range(20):
            left = first.sample(3)
            right = second.sample(3)
            self.assertEqual(set(left), set(right))
            self.assertTrue(required.issubset(left))
            for key in required:
                np.testing.assert_array_equal(np.asarray(left[key]), np.asarray(right[key]))
        self.assertEqual(first.sample(3)['high_actor_targets'].shape, (3, 2, 4))

    def test_update_and_autoregressive_policy_are_finite(self):
        batch = {
            'observations': self.observations,
            'next_observations': self.observations + 0.1,
            'actions': self.actions,
            'value_goals': self.observations + 0.2,
            'high_actor_goals': self.observations + 0.2,
            'high_actor_targets': jnp.stack(
                [self.observations + 0.3, self.observations + 0.4], axis=1
            ),
            'rewards': jnp.array([-1.0, 0.0, -1.0]),
            'masks': jnp.array([1.0, 0.0, 1.0]),
        }
        key = jax.random.PRNGKey(5)
        loss, info = self.agent.total_loss(batch, self.agent.network.params, rng=key)
        self.assertTrue(np.isfinite(float(loss)))
        self.assertTrue(all(np.isfinite(float(value)) for value in info.values()))
        actions = self.agent.sample_actions(self.observations[0], self.observations[0], seed=key)
        self.assertEqual(actions.shape, (2,))
        self.assertTrue(np.all(np.isfinite(np.asarray(actions))))
        updated, update_info = self.agent.update(batch)
        self.assertTrue(all(np.isfinite(float(value)) for value in update_info.values()))
        self.assertTrue(
            all(np.all(np.isfinite(np.asarray(value))) for value in jax.tree_util.tree_leaves(updated.network.params))
        )


if __name__ == '__main__':
    unittest.main()
