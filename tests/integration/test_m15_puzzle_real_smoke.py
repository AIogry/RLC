"""Optional M15 real OGBench Puzzle tiny smoke.

This is deliberately an infrastructure test: two updates and one evaluation
episode are not a scientific run and must not be interpreted as performance.
"""

import copy
import os
import tempfile
import unittest

import jax.numpy as jnp
import numpy as np

from impls.agents import agent_configs, agents
from impls.utils.datasets import GCDataset
from impls.utils.env_utils import make_env_and_datasets
from impls.utils.evaluation import evaluate
from impls.utils.flax_utils import restore_agent_from_checkpoint, save_agent


def _config(name):
    config = copy.deepcopy(agent_configs[name]())
    config.actor_hidden_dims = (8,)
    if 'value_hidden_dims' in config:
        config.value_hidden_dims = (8,)
    config.batch_size = 4
    if name == 'qrl':
        config.latent_dim = 6
        config.dim_per_component = 3
    kwargs = {
        'num_buttons': 9,
        'robot_dim': 19,
        'button_feature_dim': 4,
        'token_dim': 7,
        'robot_hidden_dim': 8,
        'token_mlp_hidden_dim': 5,
        'channel_mlp_hidden_dim': 11,
        'num_mixer_blocks': 1,
        'index_embedding': True,
        'readout': 'mean',
        'tm_mode': 'none',
    }
    slots = {
        'gcbc': ('actor',),
        'gciql': ('actor', 'value', 'critic'),
        'gcivl': ('actor', 'value'),
        'qrl': ('actor', 'value'),
    }[name]
    for slot_name in slots:
        slot = config.compute[slot_name]
        slot.enabled = True
        slot.structure = 'puzzle_tokens'
        slot.topology = 'feedforward'
        slot.block = 'mlp_mixer'
        slot.credit = 'direct'
        slot.structure_kwargs = copy.deepcopy(kwargs)
    return config


@unittest.skipUnless(
    os.path.exists('/data/qijunrong/06-RL/offline-rl/data/raw_ogbench/puzzle-3x3-play-v0.npz'),
    'real OGBench Puzzle data is unavailable',
)
class M15PuzzleRealSmokeTest(unittest.TestCase):
    def test_all_four_algorithms_real_tiny_lifecycle(self):
        env, raw_train, _ = make_env_and_datasets(
            'puzzle-3x3-play-v0',
            seed=123,
            dataset_seed=456,
            dataset_dir='/data/qijunrong/06-RL/offline-rl/data/raw_ogbench',
        )
        try:
            for name in ('gciql', 'gcbc', 'gcivl', 'qrl'):
                with self.subTest(agent=name):
                    config = _config(name)
                    dataset = GCDataset(raw_train, config, rng=789)
                    batch = dataset.sample(config.batch_size)
                    agent = agents[name].create(
                        17,
                        batch['observations'],
                        batch['actions'],
                        config,
                    )
                    for _ in range(2):
                        agent, info = agent.update(batch)
                        self.assertTrue(all(
                            np.all(np.isfinite(np.asarray(value)))
                            for value in info.values()
                        ))
                    with tempfile.TemporaryDirectory() as directory:
                        path = save_agent(agent, directory, 2)
                        restored = restore_agent_from_checkpoint(agent, path)
                        actions = restored.sample_actions(
                            batch['observations'][:1],
                            batch['actor_goals'][:1],
                            seed=jnp.array([0, 17], dtype=jnp.uint32),
                        )
                        self.assertEqual(actions.shape[-1], 5)
                        stats, _, renders = evaluate(
                            restored,
                            env,
                            task_id=1,
                            config=config,
                            num_eval_episodes=1,
                            num_video_episodes=0,
                            eval_temperature=0.0,
                            seed=901,
                        )
                        self.assertTrue(any(
                            key == 'success' or key.endswith('_success')
                            for key in stats
                        ))
                        self.assertEqual(renders, [])
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
