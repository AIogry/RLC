"""Tiny real-data lifecycle smoke for M17 modular Puzzle FF and SingleState."""

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


DATASET_ROOT = '/data/qijunrong/06-RL/offline-rl/data/raw_ogbench'
DATASET_FILE = os.path.join(DATASET_ROOT, 'puzzle-4x4-play-v0.npz')


def _config(topology):
    config = copy.deepcopy(agent_configs['gciql']())
    config.actor_hidden_dims = (8,)
    config.value_hidden_dims = (8,)
    config.batch_size = 4
    structure_kwargs = {
        'num_buttons': 16,
        'robot_dim': 19,
        'button_feature_dim': 4,
        'token_dim': 7,
        'robot_hidden_dim': 8,
        'token_mlp_hidden_dim': 5,
        'channel_mlp_hidden_dim': 11,
        'num_mixer_blocks': 2,
        'index_embedding': True,
        'readout': 'mean',
        'tm_mode': 'none',
    }
    for slot_name in ('actor', 'value', 'critic'):
        slot = config.compute[slot_name]
        slot.enabled = True
        slot.primitive = 'mlp'
        slot.structure = 'puzzle_tokens'
        slot.block = 'mlp_mixer'
        slot.topology = topology
        slot.credit = 'direct'
        slot.parameter_sharing = 'shared'
        slot.structure_kwargs = copy.deepcopy(structure_kwargs)
        if topology == 'single_state':
            slot.topology_kwargs = {
                'iterations': 2,
                'input_mapping': 'identity',
                'state_init': 'zero_buffer',
                'input_injection': 'z_plus_x',
                'residual': False,
            }
    return config


@unittest.skipUnless(os.path.exists(DATASET_FILE), 'real OGBench Puzzle-4x4 data is unavailable')
class M17PuzzleRealSmokeTest(unittest.TestCase):
    def test_gciql_ff_and_single_state_tiny_lifecycle(self):
        env, raw_train, _ = make_env_and_datasets(
            'puzzle-4x4-play-v0',
            seed=170,
            dataset_seed=171,
            dataset_dir=DATASET_ROOT,
        )
        try:
            for topology in ('feedforward', 'single_state'):
                with self.subTest(topology=topology):
                    config = _config(topology)
                    dataset = GCDataset(raw_train, config, rng=172)
                    batch = dataset.sample(config.batch_size)
                    agent = agents['gciql'].create(
                        173, batch['observations'], batch['actions'], config
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
                        action = restored.sample_actions(
                            batch['observations'][:1],
                            batch['actor_goals'][:1],
                            seed=jnp.array([17, 4], dtype=jnp.uint32),
                        )
                        self.assertEqual(action.shape[-1], 5)
                        stats, _, renders = evaluate(
                            restored,
                            env,
                            task_id=1,
                            config=config,
                            num_eval_episodes=1,
                            num_video_episodes=0,
                            eval_temperature=0.0,
                            seed=174,
                        )
                        self.assertTrue(any(
                            key == 'success' or key.endswith('_success') for key in stats
                        ))
                        self.assertEqual(renders, [])
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
