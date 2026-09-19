"""Explicit opt-in, CPU-only real-data smoke. Never enabled by data presence.

Requires RLC_RUN_GOAL_INPUT_REAL_SMOKE=1 and RLC_PUZZLE_DATASET_ROOT.
Missing data after opt-in is an error, not a passed/automatically skipped gate.
Two small updates/layout; no evaluation rollout and no formal output namespace.
"""

import json
import os
from pathlib import Path
import tempfile
import unittest

import jax
import numpy as np

from impls.agents import agents
from impls.diagnostics.goal_conditioning import audit_goal_conditioning
from impls.networks.goal_conditioning import resolve_goal_conditioning
from impls.utils.env_utils import make_env_and_datasets
from impls.utils.flax_utils import restore_agent_from_checkpoint, save_agent
from impls.utils.puzzle_datasets import PuzzleBoardGCDataset
from tests.reference.goal_conditioning import agent_config, goal_config, role


@unittest.skipUnless(os.environ.get('RLC_RUN_GOAL_INPUT_REAL_SMOKE') == '1',
                     'real-data smoke not run: explicit opt-in required')
class RoleGoalConditioningRealSmokeTest(unittest.TestCase):
    def test_small_updates_and_fresh_restore_4x5_4x6(self):
        self.assertEqual(jax.default_backend(), 'cpu', 'This entry point never authorizes GPU use')
        data_root = Path(os.environ['RLC_PUZZLE_DATASET_ROOT'])
        for cols in (5, 6):
            if not (data_root / f'puzzle-4x{cols}-play-v0.npz').is_file():
                raise FileNotFoundError(f'Opted-in smoke data unavailable: {data_root}, 4x{cols}')
        with tempfile.TemporaryDirectory(prefix='role_goal_real_smoke_') as root:
            for cols in (5, 6):
                environment = f'puzzle-4x{cols}-play-v0'
                config = agent_config()
                config.goal_conditioning = goal_config(
                    role('operation', 'P'), role('operation', 'B', 'exact_press_fraction'), rows=4, cols=cols,
                )
                for slot in config.compute.values():
                    slot.structure_kwargs.num_buttons = 4 * cols
                env, raw, _ = make_env_and_datasets(environment, seed=31, dataset_seed=32, dataset_dir=str(data_root))
                try:
                    dataset = PuzzleBoardGCDataset(raw, config, rng=33)
                    batch = dataset.sample(2)
                    agent = agents['gciql'].create(34, batch['observations'], batch['actions'], config)
                    for _ in range(2):
                        agent, info = agent.update(batch)
                        self.assertTrue(all(np.all(np.isfinite(v)) for v in jax.tree_util.tree_leaves(info)))
                    audit_goal_conditioning(agent, batch['observations'], batch['actor_goals'], batch['actions'])
                    expected = jax.device_get(agent.network.params)
                    path = save_agent(agent, Path(root) / environment, 2)
                    serialized = config.to_dict()
                    serialized['goal_conditioning'] = resolve_goal_conditioning(config.goal_conditioning).to_config()
                    serialized = json.dumps(serialized)
                    del agent
                    fresh = agents['gciql'].create(34, batch['observations'], batch['actions'], json.loads(serialized))
                    restored = restore_agent_from_checkpoint(fresh, path)
                    for a, b in zip(jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(restored.network.params)):
                        np.testing.assert_array_equal(a, b)
                finally:
                    env.close()


if __name__ == '__main__':
    unittest.main()
