from __future__ import annotations

import unittest

import numpy as np

from impls.representation.puzzle_effects import (
    detect_board_change_events,
    extract_binary_task_board,
    extract_dataset_boards,
    pack_raw_effect_signature,
    raw_effect_signature,
    raw_transition_xor,
    unpack_raw_effect_signature,
)


def _observation(bits):
    bits = np.asarray(bits, dtype=np.uint8)
    robot = np.arange(19, dtype=np.float32)
    buttons = np.zeros((len(bits), 4), dtype=np.float32)
    buttons[:, 0] = 1 - bits
    buttons[:, 1] = bits
    buttons[:, 2] = np.arange(len(bits), dtype=np.float32) + 0.25
    buttons[:, 3] = -np.arange(len(bits), dtype=np.float32)
    return np.concatenate([robot, buttons.reshape(-1)])


class PuzzleEffectsTest(unittest.TestCase):
    def test_binary_board_extraction_reuses_standard_layout(self):
        bits = np.asarray([0, 1, 1, 0, 1], dtype=np.uint8)
        decoded = extract_binary_task_board(_observation(bits), num_buttons=5)
        np.testing.assert_array_equal(decoded, bits)
        batch = np.stack([_observation(bits), _observation(1 - bits)])
        np.testing.assert_array_equal(
            extract_binary_task_board(batch, num_buttons=5),
            np.stack([bits, 1 - bits]),
        )

    def test_dataset_extraction_is_chunk_independent(self):
        observations = np.stack([
            _observation([0, 0, 0]),
            _observation([1, 0, 1]),
            _observation([1, 1, 1]),
        ])
        expected = np.asarray([[0, 0, 0], [1, 0, 1], [1, 1, 1]])
        np.testing.assert_array_equal(
            extract_dataset_boards(observations, num_buttons=3, chunk_size=1),
            expected,
        )

    def test_malformed_standard_observation_and_onehot_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, 'expected final dimension'):
            extract_binary_task_board(np.zeros(12), num_buttons=3)
        malformed = _observation([0, 1, 0])
        malformed[19:21] = (1, 1)
        with self.assertRaisesRegex(ValueError, 'valid two-class onehots'):
            extract_binary_task_board(malformed, num_buttons=3)

    def test_event_detection_is_exactly_adjacent_board_change(self):
        boards = np.asarray([
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
        ], dtype=np.uint8)
        np.testing.assert_array_equal(
            detect_board_change_events(boards),
            np.asarray([False, True, False, True]),
        )

    def test_raw_xor_signature_round_trip_is_diagnostic_and_dimension_tagged(self):
        start = np.asarray([0, 1, 0, 1, 1], dtype=np.uint8)
        end = np.asarray([1, 1, 1, 0, 1], dtype=np.uint8)
        effect = raw_transition_xor(start, end)
        signature = raw_effect_signature(start, end)
        self.assertEqual(signature, pack_raw_effect_signature(effect))
        np.testing.assert_array_equal(unpack_raw_effect_signature(signature), effect)
        self.assertTrue(signature.startswith('5:'))
        with self.assertRaisesRegex(ValueError, 'zero-change'):
            raw_effect_signature(start, start)


if __name__ == '__main__':
    unittest.main()
