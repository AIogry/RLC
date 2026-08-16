"""Minimal runtime smoke for the currently available RLC HIQL surface."""

import os
import pickle
import tempfile
import unittest

import flax
import jax
import numpy as np
import jax.numpy as jnp

from impls.agents.hiql import HIQLAgent
from tests.computation.test_mlp_parity import _small_config, _synthetic_batch


class HIQLSyntheticSmokeTest(unittest.TestCase):
    def test_three_slot_jit_update_action_path_and_checkpoint(self):
        config = _small_config(low_enabled=True, high_enabled=True, value_enabled=True)
        example_observations = jnp.arange(8, dtype=jnp.float32).reshape(2, 4) / 5.0
        example_actions = jnp.arange(4, dtype=jnp.float32).reshape(2, 2) / 7.0
        agent = HIQLAgent.create(51, example_observations, example_actions, config)

        for step in range(3):
            agent, info = agent.update(_synthetic_batch(step=step))
            self.assertTrue(all(np.isfinite(np.asarray(value)) for value in info.values()))

        batch = _synthetic_batch()
        actions = agent.sample_actions(
            batch['observations'], batch['high_actor_goals'], seed=jax.random.PRNGKey(52),
        )
        self.assertEqual(actions.shape, (3, 2))
        self.assertTrue(np.all(np.isfinite(np.asarray(actions))))

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'params_3.pkl')
            with open(path, 'wb') as handle:
                pickle.dump({'agent': flax.serialization.to_state_dict(agent)}, handle)
            with open(path, 'rb') as handle:
                state = pickle.load(handle)['agent']
            restored = flax.serialization.from_state_dict(agent, state)
            restored_actions = restored.sample_actions(
                batch['observations'], batch['high_actor_goals'], seed=jax.random.PRNGKey(52),
            )
            np.testing.assert_array_equal(np.asarray(actions), np.asarray(restored_actions))


if __name__ == '__main__':
    unittest.main()
