"""Six-cell real-data lifecycle smoke for Puzzle goal conditioning.

This is an infrastructure check with one optimizer update and one evaluation
episode per cell.  It is not a training run and its success rates are not
scientific results.
"""

import copy
import os
import tempfile
import unittest

import jax
import numpy as np
from flax.traverse_util import flatten_dict

from impls.agents import agent_configs, agents
from impls.utils.env_utils import make_env_and_datasets
from impls.utils.evaluation import evaluate
from impls.utils.flax_utils import restore_agent_from_checkpoint, save_agent
from impls.utils.puzzle_datasets import PuzzleBoardGCDataset


DATASET_ROOT = '/data/qijunrong/06-RL/offline-rl/data/raw_ogbench'
LAYOUTS = {
    'puzzle-4x5-play-v0': (4, 5),
    'puzzle-4x6-play-v0': (4, 6),
}
DATA_AVAILABLE = all(
    os.path.exists(os.path.join(DATASET_ROOT, f'{environment}.npz'))
    for environment in LAYOUTS
)


def _config(mode, rows, cols):
    num_buttons = rows * cols
    config = copy.deepcopy(agent_configs['gciql']())
    config.alpha = 0.4
    # Keep the planned network architecture intact.  Only the sampled smoke
    # batch is small; this does not define a formal training protocol.
    config.batch_size = 4
    config.dataset_class = 'PuzzleBoardGCDataset'
    config.goal_conditioning = {
        'domain': 'puzzle',
        'mode': mode,
        'rows': rows,
        'cols': cols,
        'num_buttons': num_buttons,
        'robot_dim': 19,
        'button_feature_dim': 4,
        'robot_goal_policy': 'zero',
        'button_goal_transient_policy': 'zero',
    }
    structure_kwargs = {
        'num_buttons': num_buttons,
        'robot_dim': 19,
        'button_feature_dim': 4,
        'token_dim': 128,
        'robot_hidden_dim': 128,
        'token_mlp_hidden_dim': 64,
        'channel_mlp_hidden_dim': 256,
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
        slot.topology = 'feedforward'
        slot.credit = 'direct'
        slot.structure_kwargs = copy.deepcopy(structure_kwargs)
    return config


@unittest.skipUnless(DATA_AVAILABLE, 'real Puzzle-4x5/4x6 datasets are unavailable')
class PuzzleGoalConditioningRealSmokeTest(unittest.TestCase):
    def test_six_real_cells_one_update_evaluation_and_checkpoint(self):
        modes = ('board', 'residual', 'oracle_operation')
        with tempfile.TemporaryDirectory(prefix='puzzle_goal_conditioning_smoke_') as root:
            for environment_index, (environment, layout) in enumerate(LAYOUTS.items()):
                rows, cols = layout
                env, raw_train, _ = make_env_and_datasets(
                    environment,
                    seed=24100 + environment_index,
                    dataset_seed=24200 + environment_index,
                    dataset_dir=DATASET_ROOT,
                )
                try:
                    datasets = {
                        mode: PuzzleBoardGCDataset(
                            raw_train,
                            _config(mode, rows, cols),
                            rng=24300 + environment_index,
                        )
                        for mode in modes
                    }
                    sampled = {
                        mode: dataset.sample(4, return_sampling_trace=True)
                        for mode, dataset in datasets.items()
                    }
                    reference_trace = sampled['board'][1]
                    for mode in modes[1:]:
                        for key in reference_trace:
                            np.testing.assert_array_equal(
                                sampled[mode][1][key], reference_trace[key]
                            )

                    reference_parameter_schema = None
                    reference_parameter_count = None
                    for mode_index, mode in enumerate(modes):
                        with self.subTest(environment=environment, mode=mode):
                            config = _config(mode, rows, cols)
                            batch, trace = sampled[mode]
                            self.assertEqual(
                                set(trace),
                                {
                                    'transition_indices',
                                    'value_goal_indices',
                                    'actor_goal_indices',
                                },
                            )
                            agent = agents['gciql'].create(
                                24400 + environment_index,
                                batch['observations'],
                                batch['actions'],
                                config,
                            )
                            flat_params = flatten_dict(agent.network.params)
                            parameter_schema = {
                                key: value.shape for key, value in flat_params.items()
                            }
                            parameter_count = sum(
                                value.size for value in flat_params.values()
                            )
                            if reference_parameter_schema is None:
                                reference_parameter_schema = parameter_schema
                                reference_parameter_count = parameter_count
                            else:
                                self.assertEqual(
                                    parameter_schema, reference_parameter_schema
                                )
                                self.assertEqual(
                                    parameter_count, reference_parameter_count
                                )
                            value = agent.network.select('value')(
                                batch['observations'], batch['value_goals']
                            )
                            q1, q2 = agent.network.select('critic')(
                                batch['observations'],
                                batch['value_goals'],
                                batch['actions'],
                            )
                            self.assertTrue(np.all(np.isfinite(np.asarray(value))))
                            self.assertTrue(np.all(np.isfinite(np.asarray(q1))))
                            self.assertTrue(np.all(np.isfinite(np.asarray(q2))))

                            agent, info = agent.update(batch)
                            self.assertTrue(all(
                                np.all(np.isfinite(np.asarray(item)))
                                for item in info.values()
                            ))
                            cell_dir = os.path.join(root, environment, mode)
                            checkpoint = save_agent(agent, cell_dir, 1)
                            restored = restore_agent_from_checkpoint(agent, checkpoint)
                            action = restored.sample_actions(
                                batch['observations'][:1],
                                batch['actor_goals'][:1],
                                seed=jax.random.PRNGKey(
                                    24500 + 10 * environment_index + mode_index
                                ),
                                temperature=0.0,
                            )
                            self.assertEqual(action.shape, (1, 5))
                            self.assertTrue(np.all(np.isfinite(np.asarray(action))))
                            stats, _, renders = evaluate(
                                restored,
                                env,
                                task_id=1,
                                config=config,
                                num_eval_episodes=1,
                                num_video_episodes=0,
                                eval_temperature=0.0,
                                seed=24600 + 10 * environment_index + mode_index,
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
