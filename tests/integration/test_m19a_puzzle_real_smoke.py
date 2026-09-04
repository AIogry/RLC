"""Real-data M19A EntityMLP lifecycle smoke; this is not a formal run."""

import os
import tempfile
import unittest
from pathlib import Path

import jax
import numpy as np

from impls.agents import agents
from impls.experiment import prepare_run_design
from impls.main import _make_config, _parse_args
from impls.utils.datasets import GCDataset
from impls.utils.env_utils import make_env_and_datasets
from impls.utils.evaluation import evaluate
from impls.utils.flax_utils import restore_agent_from_checkpoint, save_agent


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'experiments/M19A_puzzle_entity_factorization_isolation/study.yaml'
DATASET_ROOT = '/data/qijunrong/06-RL/offline-rl/data/raw_ogbench'
DATASET_AVAILABLE = all(
    os.path.exists(os.path.join(DATASET_ROOT, name))
    for name in ('puzzle-4x4-play-v0.npz', 'puzzle-4x4-play-v0-val.npz')
)


@unittest.skipUnless(DATASET_AVAILABLE, 'real Puzzle-4x4 data is unavailable')
class M19APuzzleRealSmokeTest(unittest.TestCase):
    def test_m19a_entity_agent_two_updates_checkpoint_and_one_episode(self):
        _, configuration = prepare_run_design(STUDY, 'M19A-4x4-E001')
        config = _make_config(_parse_args(['--agent', 'gciql']), configuration=configuration)
        # Keep the formal architecture and all algorithm semantics exactly as
        # declared; reducing only this sampled tiny-batch size does not create
        # a training run or alter the EntityMLP control definition.
        config.batch_size = 4
        env, raw_train, _ = make_env_and_datasets(
            'puzzle-4x4-play-v0',
            seed=19010,
            dataset_seed=19011,
            dataset_dir=DATASET_ROOT,
        )
        try:
            dataset = GCDataset(raw_train, config, rng=19012)
            batch = dataset.sample(config.batch_size)
            agent = agents['gciql'].create(
                19013, batch['observations'], batch['actions'], config,
            )
            for _ in range(2):
                agent, info = agent.update(batch)
                self.assertTrue(all(
                    np.all(np.isfinite(np.asarray(value))) for value in info.values()
                ))
                for key in ('actor/actor_loss', 'value/value_loss', 'critic/critic_loss'):
                    self.assertIn(key, info)
            deterministic = agent.sample_actions(
                batch['observations'][:1], batch['actor_goals'][:1],
                seed=jax.random.PRNGKey(19014), temperature=0.0,
            )
            stochastic = agent.sample_actions(
                batch['observations'][:1], batch['actor_goals'][:1],
                seed=jax.random.PRNGKey(19015), temperature=1.0,
            )
            self.assertEqual(deterministic.shape, (1, 5))
            self.assertEqual(stochastic.shape, (1, 5))
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = save_agent(agent, directory, 2)
                restored = restore_agent_from_checkpoint(agent, checkpoint)
                stats, _, renders = evaluate(
                    restored,
                    env,
                    task_id=1,
                    config=config,
                    num_eval_episodes=1,
                    num_video_episodes=0,
                    eval_temperature=0.0,
                    seed=19016,
                )
                self.assertTrue(any(
                    key == 'success' or key.endswith('_success') for key in stats
                ))
                self.assertEqual(renders, [])
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
