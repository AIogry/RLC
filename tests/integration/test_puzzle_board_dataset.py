"""Board-equality success and RNG-parity tests for Puzzle GC sampling."""

import copy
import hashlib
import unittest
from unittest import mock

import numpy as np

from impls.utils.datasets import Dataset, GCDataset
from impls.utils.puzzle_datasets import PuzzleBoardGCDataset


def _goal_config(mode='board'):
    return {
        'domain': 'puzzle',
        'mode': mode,
        'rows': 3,
        'cols': 3,
        'num_buttons': 9,
        'robot_dim': 19,
        'button_feature_dim': 4,
        'robot_goal_policy': 'zero',
        'button_goal_transient_policy': 'zero',
    }


def _dataset_config(mode='board'):
    return {
        'frame_stack': None,
        'value_p_curgoal': 0.2,
        'value_p_trajgoal': 0.5,
        'value_p_randomgoal': 0.3,
        'value_geom_sample': True,
        'actor_p_curgoal': 0.0,
        'actor_p_trajgoal': 1.0,
        'actor_p_randomgoal': 0.0,
        'actor_geom_sample': False,
        'discount': 0.99,
        'gc_negative': True,
        'p_aug': 0.0,
        'compute': {},
        'dataset_class': 'PuzzleBoardGCDataset',
        'goal_conditioning': _goal_config(mode),
    }


def _observation(bits, offset):
    bits = np.asarray(bits, dtype=np.uint8)
    buttons = np.zeros((9, 4), dtype=np.float32)
    buttons[:, 0] = 1 - bits
    buttons[:, 1] = bits
    buttons[:, 2] = np.arange(9, dtype=np.float32) + offset
    buttons[:, 3] = -np.arange(9, dtype=np.float32) - offset
    robot = np.arange(19, dtype=np.float32) + offset
    return np.concatenate([robot, buttons.reshape(-1)])


def _puzzle_raw():
    boards = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0],
    ]
    observations = np.stack([
        _observation(board, index + 1) for index, board in enumerate(boards)
    ])
    terminals = np.asarray([0, 0, 0, 0, 0, 1], dtype=np.float32)
    actions = np.arange(12, dtype=np.float32).reshape(6, 2)
    return Dataset.create(
        freeze=False,
        observations=observations,
        actions=actions,
        terminals=terminals,
    )


def _fingerprint(array):
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(array)).tobytes()
    ).hexdigest()


class PuzzleBoardDatasetTest(unittest.TestCase):
    def test_same_board_different_index_is_success_and_different_board_is_failure(self):
        dataset = PuzzleBoardGCDataset(_puzzle_raw(), _dataset_config(), rng=10)
        successes = dataset._compute_value_successes(
            np.asarray([0, 0, 3]), np.asarray([1, 2, 4])
        )
        np.testing.assert_array_equal(successes, np.asarray([1.0, 0.0, 1.0]))

    def test_parent_reward_and_mask_construction_uses_board_hook(self):
        dataset = PuzzleBoardGCDataset(_puzzle_raw(), _dataset_config(), rng=11)
        transition_idxs = np.asarray([0, 0, 3])
        value_goal_idxs = np.asarray([1, 2, 4])
        actor_goal_idxs = np.asarray([2, 3, 5])
        with mock.patch.object(
            dataset,
            'sample_goals',
            side_effect=[value_goal_idxs, actor_goal_idxs],
        ):
            batch = dataset.sample(3, idxs=transition_idxs, evaluation=True)
        np.testing.assert_array_equal(batch['rewards'], np.asarray([0.0, -1.0, 0.0]))
        np.testing.assert_array_equal(batch['masks'], np.asarray([0.0, 1.0, 0.0]))

    def test_pre_refactor_canonical_golden_replays_exactly(self):
        observations = np.arange(12 * 7, dtype=np.float32).reshape(12, 7) / 13
        actions = np.arange(12 * 2, dtype=np.float32).reshape(12, 2) / 17
        terminals = np.zeros(12, dtype=np.float32)
        terminals[[3, 7, 11]] = 1
        config = _dataset_config()
        config.pop('goal_conditioning')
        config['dataset_class'] = 'GCDataset'
        raw = Dataset.create(
            freeze=False,
            observations=observations,
            actions=actions,
            terminals=terminals,
        )
        dataset = GCDataset(raw, config, rng=240910)
        batch, trace = dataset.sample(6, return_sampling_trace=True)
        np.testing.assert_array_equal(
            trace['transition_indices'], [1, 2, 3, 3, 3, 4]
        )
        np.testing.assert_array_equal(
            trace['value_goal_indices'], [7, 2, 2, 3, 3, 4]
        )
        np.testing.assert_array_equal(
            trace['actor_goal_indices'], [3, 3, 3, 3, 3, 6]
        )
        expected_hashes = {
            'actions': 'b2f58903b58a81317de622bb1aef04196635aae1173d2b8b318890c7f5d36149',
            'actor_goals': 'f234f4e70d68d984aab9274dc32994eddd98fe0ad73ccc5bee7d1187355978c5',
            'masks': '11a6663169df446f417c561069221d5211974e46cda9eb1cc7b901e49ee4c7da',
            'next_observations': 'ba8fbe53950a7dcb20d9783b69e50f9d83f5ca7c9d126e2f35dfbd5484f67d63',
            'observations': '97e75fc03fea146954edfc9c3a3c1a340995077e8770622acd00d86b7474e87c',
            'rewards': '53b1a027895feac2d816e43cd9937956be5c799769b3013ec6567cbb373755a5',
            'terminals': 'f7b6a6a04197a92f0cf49980ef33c9778c7545b49c635aa763ff929d5f4f96ca',
            'value_goals': '0b1f463e3c3b37db37268a80398d2f7073ec26681302cc5a54c93d81cb9265db',
        }
        self.assertEqual(
            {key: _fingerprint(value) for key, value in sorted(batch.items())},
            expected_hashes,
        )

    def test_three_modes_have_identical_raw_sampling_pairs_and_rng_state(self):
        datasets = {
            mode: PuzzleBoardGCDataset(
                _puzzle_raw(), _dataset_config(mode), rng=24003
            )
            for mode in ('board', 'residual', 'oracle_operation')
        }
        for _ in range(3):
            sampled = {
                mode: dataset.sample(5, return_sampling_trace=True)
                for mode, dataset in datasets.items()
            }
            reference_batch, reference_trace = sampled['board']
            for mode in ('residual', 'oracle_operation'):
                batch, trace = sampled[mode]
                for key in reference_trace:
                    np.testing.assert_array_equal(trace[key], reference_trace[key])
                for key in reference_batch:
                    np.testing.assert_array_equal(batch[key], reference_batch[key])
        states = [copy.deepcopy(dataset.rng.bit_generator.state) for dataset in datasets.values()]
        self.assertEqual(states[0], states[1])
        self.assertEqual(states[0], states[2])

    def test_board_success_hook_draws_no_additional_rng(self):
        config = _dataset_config('residual')
        board_dataset = PuzzleBoardGCDataset(_puzzle_raw(), config, rng=24004)
        canonical_dataset = GCDataset(_puzzle_raw(), config, rng=24004)
        board_dataset.sample(4)
        canonical_dataset.sample(4)
        self.assertEqual(
            board_dataset.rng.bit_generator.state,
            canonical_dataset.rng.bit_generator.state,
        )

    def test_dataset_construction_never_builds_operator_or_inverse(self):
        with mock.patch(
            'impls.networks.goal_conditioning.operator_metadata',
            side_effect=AssertionError('Dataset requested Puzzle operator algebra'),
        ):
            dataset = PuzzleBoardGCDataset(
                _puzzle_raw(), _dataset_config('oracle_operation'), rng=24005
            )
            np.testing.assert_array_equal(
                dataset._compute_value_successes(
                    np.asarray([0, 2]), np.asarray([1, 3])
                ),
                np.asarray([1.0, 0.0]),
            )


if __name__ == '__main__':
    unittest.main()
