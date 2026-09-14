from __future__ import annotations

import dataclasses
import unittest

import numpy as np

from impls.representation.puzzle_effects import (
    detect_board_change_events,
    raw_effect_signature,
)
from impls.utils.effect_datasets import (
    EffectEventDataset,
    build_effect_event_index,
    episode_bounds_from_terminals,
    episode_terminals_from_compact_valids,
)


class EffectDatasetTest(unittest.TestCase):
    def setUp(self):
        # The change from row 2 to row 3 is across an episode boundary and must
        # never become an event, even though the raw adjacent arrays differ.
        self.boards = np.asarray([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ], dtype=np.uint8)
        self.terminals = np.asarray([0, 0, 1, 0, 0, 1], dtype=np.uint8)
        self.index = build_effect_event_index(
            self.boards,
            self.terminals,
            event_mask=detect_board_change_events(self.boards),
            effect_signature_fn=raw_effect_signature,
        )

    def test_episode_bounds_and_exact_event_endpoints(self):
        self.assertEqual(episode_bounds_from_terminals(self.terminals), ((0, 2), (3, 5)))
        self.assertEqual([record.transition_idx for record in self.index.records], [0, 1, 3, 4])
        self.assertNotIn(2, [record.transition_idx for record in self.index.records])
        for record in self.index.records:
            self.assertEqual(record.start_idx, record.transition_idx)
            self.assertEqual(record.end_idx, record.transition_idx + 1)
            self.assertEqual(record.start_board, tuple(self.boards[record.start_idx]))
            self.assertEqual(record.end_board, tuple(self.boards[record.end_idx]))

    def test_compact_valids_restore_observation_bounds_and_final_transition(self):
        # Canonical compact OGBench terminals would be [0, 1, 1, 0, 1, 1]:
        # one marker for the last valid transition and another for the final
        # observation.  ``valids`` retains the original observation boundary.
        valids = np.asarray([1, 1, 0, 1, 1, 0], dtype=np.float32)
        restored = episode_terminals_from_compact_valids(valids)
        np.testing.assert_array_equal(restored, self.terminals)
        index = build_effect_event_index(
            self.boards,
            restored,
            event_mask=detect_board_change_events(self.boards),
            effect_signature_fn=raw_effect_signature,
        )
        self.assertEqual(
            [record.transition_idx for record in index.records], [0, 1, 3, 4]
        )
        self.assertIn(1, [record.transition_idx for record in index.records])
        self.assertIn(4, [record.transition_idx for record in index.records])

        with self.assertRaisesRegex(ValueError, 'episode boundary'):
            episode_terminals_from_compact_valids(np.asarray([1, 1]))

    def test_future_window_metadata_and_event_gaps_are_retained(self):
        first, second, third, fourth = self.index.records
        self.assertIsNone(first.previous_event_gap)
        self.assertEqual(first.next_event_gap, 1)
        self.assertEqual(second.previous_event_gap, 1)
        self.assertIsNone(second.next_event_gap)
        self.assertEqual((first.pre_event_steps, first.post_event_steps), (0, 1))
        self.assertEqual((fourth.pre_event_steps, fourth.post_event_steps), (1, 0))
        self.assertEqual(third.episode_start_idx, 3)
        self.assertEqual(third.episode_end_idx, 5)

    def test_event_index_is_immutable_and_rejects_zero_change_records(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            self.index.num_bits = 99
        zero_boards = np.asarray([[0, 0], [0, 0]], dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, 'zero-change'):
            build_effect_event_index(
                zero_boards,
                np.asarray([0, 1]),
                event_mask=np.asarray([True]),
            )

    def test_uniform_sampling_is_deterministic_with_explicit_rng(self):
        first = EffectEventDataset(self.index, seed=17)
        second = EffectEventDataset(self.index, seed=17)
        for _ in range(3):
            batch_a = first.sample(9)
            batch_b = second.sample(9)
            np.testing.assert_array_equal(batch_a['start_board'], batch_b['start_board'])
            np.testing.assert_array_equal(batch_a['end_board'], batch_b['end_board'])

    def test_default_batch_contains_no_identity_action_or_effect_metadata(self):
        batch = EffectEventDataset(self.index, seed=0).sample(5)
        self.assertEqual(set(batch), {'start_board', 'end_board'})
        forbidden = {
            'action', 'actions', 'button_id', 'physical_button_id',
            'target_button', 'raw_effect_signature', 'effect_id',
        }
        self.assertFalse(set(batch) & forbidden)
        debug = EffectEventDataset(
            self.index, seed=0, include_record_indices=True
        ).sample(2)
        self.assertEqual(set(debug), {'start_board', 'end_board', 'record_indices'})

    def test_effect_balanced_sampling_is_not_a_silent_option(self):
        with self.assertRaisesRegex(ValueError, 'uniform'):
            EffectEventDataset(
                self.index, seed=0, event_sampling_mode='effect_balanced'
            )


if __name__ == '__main__':
    unittest.main()
