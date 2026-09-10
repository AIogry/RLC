"""Semantic tests for canonical-shape Puzzle goal conditioning."""

import unittest

import jax.numpy as jnp
import numpy as np

from impls.representation.puzzle import parse_puzzle_observation
from impls.representation.puzzle_algebra import apply_gf2_map, build_toggle_matrix
from impls.representation.puzzle_conditioning import (
    condition_puzzle_goal,
    extract_button_bits,
)


def _observation(bits, *, robot_offset=0.0, transient_offset=0.0):
    bits = np.asarray(bits, dtype=np.uint8)
    leading = bits.shape[:-1]
    num_buttons = bits.shape[-1]
    robot = np.broadcast_to(
        np.arange(19, dtype=np.float32) + robot_offset,
        (*leading, 19),
    ).copy()
    buttons = np.zeros((*leading, num_buttons, 4), dtype=np.float32)
    buttons[..., 0] = 1 - bits
    buttons[..., 1] = bits
    transient = np.arange(num_buttons, dtype=np.float32) + transient_offset
    buttons[..., 2] = np.broadcast_to(transient, leading + (num_buttons,))
    buttons[..., 3] = np.broadcast_to(-transient - 1, leading + (num_buttons,))
    return np.concatenate([robot, buttons.reshape(*leading, -1)], axis=-1)


class PuzzleConditioningTest(unittest.TestCase):
    def _condition(self, observations, goals, mode, rows=3, cols=3):
        return condition_puzzle_goal(
            observations,
            goals,
            mode=mode,
            rows=rows,
            cols=cols,
            num_buttons=rows * cols,
            robot_dim=19,
            button_feature_dim=4,
        )

    def test_canonical_is_exact_raw_goal_identity(self):
        observations = _observation(np.arange(9) % 2)
        goals = _observation((np.arange(9) + 1) % 2, robot_offset=8, transient_offset=3)
        conditioned = self._condition(observations, goals, 'canonical')
        np.testing.assert_array_equal(np.asarray(conditioned), goals)

    def test_board_and_residual_keep_shape_and_zero_nonboard_goal_fields(self):
        current_bits = np.asarray([
            np.arange(9) % 2,
            np.asarray([1, 1, 0, 0, 1, 0, 1, 0, 1]),
        ])
        goal_bits = np.asarray([
            (np.arange(9) + 1) % 2,
            np.asarray([1, 0, 0, 1, 1, 1, 0, 0, 1]),
        ])
        observations = _observation(current_bits, robot_offset=2, transient_offset=4)
        goals = _observation(goal_bits, robot_offset=9, transient_offset=7)
        observations_before = observations.copy()
        for mode, expected_bits in (
            ('board', goal_bits),
            ('residual', np.bitwise_xor(current_bits, goal_bits)),
        ):
            with self.subTest(mode=mode):
                conditioned = self._condition(observations, goals, mode)
                robot, buttons = parse_puzzle_observation(conditioned, num_buttons=9)
                self.assertEqual(conditioned.shape, goals.shape)
                np.testing.assert_array_equal(np.asarray(robot), np.zeros_like(robot))
                np.testing.assert_array_equal(
                    np.asarray(extract_button_bits(conditioned, num_buttons=9)),
                    expected_bits,
                )
                np.testing.assert_array_equal(np.asarray(buttons[..., 2:]), 0)
        np.testing.assert_array_equal(observations, observations_before)

    def test_oracle_operation_is_remaining_press_parity(self):
        rows, cols = 4, 5
        rng = np.random.default_rng(24002)
        current_bits = rng.integers(0, 2, size=(4, rows * cols), dtype=np.uint8)
        goal_bits = rng.integers(0, 2, size=(4, rows * cols), dtype=np.uint8)
        observations = _observation(current_bits)
        goals = _observation(goal_bits, robot_offset=3, transient_offset=5)
        observations_before = observations.copy()
        conditioned = self._condition(
            observations, goals, 'oracle_operation', rows=rows, cols=cols
        )
        robot, buttons = parse_puzzle_observation(
            conditioned, num_buttons=rows * cols
        )
        operations = extract_button_bits(conditioned, num_buttons=rows * cols)
        reconstructed = apply_gf2_map(build_toggle_matrix(rows, cols), operations)
        np.testing.assert_array_equal(
            np.asarray(reconstructed), np.bitwise_xor(current_bits, goal_bits)
        )
        np.testing.assert_array_equal(np.asarray(robot), np.zeros_like(robot))
        np.testing.assert_array_equal(np.asarray(buttons[..., 2:]), 0)
        np.testing.assert_array_equal(observations, observations_before)

    def test_state_dependent_modes_recompute_for_each_call(self):
        goal_bits = np.asarray([1, 0, 1, 1, 0, 0, 1, 0, 1], dtype=np.uint8)
        state_bits = np.zeros(9, dtype=np.uint8)
        next_bits = state_bits.copy()
        next_bits[[0, 1, 3]] = 1
        goal = _observation(goal_bits)
        for mode in ('residual', 'oracle_operation'):
            first = self._condition(_observation(state_bits), goal, mode)
            second = self._condition(_observation(next_bits), goal, mode)
            first_bits = np.asarray(extract_button_bits(first, num_buttons=9))
            second_bits = np.asarray(extract_button_bits(second, num_buttons=9))
            self.assertFalse(np.array_equal(first_bits, second_bits), msg=mode)

    def test_unbatched_and_batched_results_match(self):
        current = np.asarray([0, 1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        goal = 1 - current
        unbatched = self._condition(
            _observation(current), _observation(goal), 'residual'
        )
        batched = self._condition(
            _observation(current[None]), _observation(goal[None]), 'residual'
        )
        np.testing.assert_array_equal(np.asarray(unbatched), np.asarray(batched[0]))

    def test_malformed_inputs_and_singular_oracle_fail_loudly(self):
        valid = _observation(np.zeros(9, dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, 'identical canonical shape'):
            self._condition(valid, valid[:-1], 'residual')
        malformed = valid.copy()
        malformed[19:21] = 0
        with self.assertRaisesRegex(ValueError, 'valid two-class onehots'):
            self._condition(malformed, valid, 'residual')
        with self.assertRaisesRegex(ValueError, 'full-rank'):
            self._condition(
                _observation(np.zeros(16, dtype=np.uint8)),
                _observation(np.ones(16, dtype=np.uint8)),
                'oracle_operation',
                rows=4,
                cols=4,
            )
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            self._condition(valid, valid, 'unknown')


if __name__ == '__main__':
    unittest.main()
