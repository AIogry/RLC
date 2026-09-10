"""Targeted tests for generic controlled Puzzle policy-goal replay."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from impls.diagnostics.puzzle import (
    ControlledGoalReplay,
    ControlledGoalReplayEnv,
    ControlledGoalReplayError,
    array_fingerprint,
    verify_paired_fingerprint_groups,
)


def _puzzle_vector(board, *, marker=0.0):
    """Return a valid synthetic 3x3 Puzzle observation/goal vector."""

    board = np.asarray(board, dtype=np.uint8)
    result = np.zeros(19 + 4 * board.size, dtype=np.float32)
    result[0] = float(marker)
    blocks = result[19:].reshape(board.size, 4)
    blocks[:, 0] = 1 - board
    blocks[:, 1] = board
    return result


class _SyntheticPuzzleEnv:
    """Mimics the reset-facing contract required by the replay wrapper."""

    def __init__(self):
        self.unwrapped = self
        self._num_rows = 3
        self._num_cols = 3
        self.task_infos = [{
            'task_name': 'synthetic-task-1',
            'goal_button_states': np.asarray([1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8),
        }]
        self.reset_count = 0
        self.mismatch_initial = False
        self.last_native_goal = None

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        if options.get('task_id') != 1:
            raise ValueError('Synthetic environment only supports task 1')
        self.reset_count += 1
        marker = int(seed) % 997
        if self.mismatch_initial:
            marker += 1
        observation = _puzzle_vector(np.zeros(9, dtype=np.uint8), marker=marker)
        native_goal = _puzzle_vector(
            self.task_infos[0]['goal_button_states'], marker=float(self.reset_count),
        )
        self.last_native_goal = native_goal.copy()
        return observation, {'goal': native_goal}

    def close(self):
        return None


class ControlledGoalReplayTest(unittest.TestCase):
    def _plan(self):
        root = Path(tempfile.mkdtemp())
        env = _SyntheticPuzzleEnv()
        replay = ControlledGoalReplay.create(
            root / 'plan',
            env,
            environment='puzzle-3x3-play-v0',
            evaluation_seed=20260909,
            task_ids=[1],
            episodes_per_task=2,
        )
        return env, replay

    def test_replays_complete_goal_but_leaves_environment_target_native(self):
        env, replay = self._plan()
        loaded = ControlledGoalReplay.load(replay.root)
        self.assertEqual(loaded.record_count, 2)
        record = loaded.record_for(1, 0)
        wrapper = ControlledGoalReplayEnv(env, loaded)
        with wrapper.paired_episode(1, 0):
            observation, info = wrapper.reset(
                seed=record.episode_seed,
                options={'task_id': 1, 'render_goal': False},
            )
            np.testing.assert_array_equal(info['goal'], record.goal)
            np.testing.assert_array_equal(observation, record.initial_observation)
        self.assertNotEqual(array_fingerprint(env.last_native_goal), record.goal_fingerprint)
        self.assertEqual(wrapper.last_pairing['paired_episode_id'], 'task01_ep000')
        self.assertEqual(
            wrapper.last_pairing['initial_observation_fingerprint'],
            record.initial_observation_fingerprint,
        )

    def test_initial_observation_mismatch_fails_before_a_paired_rollout(self):
        env, replay = self._plan()
        env.mismatch_initial = True
        record = replay.record_for(1, 0)
        wrapper = ControlledGoalReplayEnv(env, replay)
        with self.assertRaisesRegex(ControlledGoalReplayError, 'Initial observation fingerprint mismatch'):
            with wrapper.paired_episode(1, 0):
                wrapper.reset(
                    seed=record.episode_seed,
                    options={'task_id': 1, 'render_goal': False},
                )

    def test_cross_policy_fingerprints_must_be_exactly_equal(self):
        _, replay = self._plan()
        record = replay.record_for(1, 0)
        rows = [{
            'environment': 'puzzle-3x3-play-v0',
            'paired_episode_id': record.paired_episode_id,
            'goal_fingerprint': record.goal_fingerprint,
            'board_goal_fingerprint': record.board_goal_fingerprint,
            'initial_observation_fingerprint': record.initial_observation_fingerprint,
        } for _ in range(3)]
        summary = verify_paired_fingerprint_groups(rows, expected_members=3)
        self.assertEqual(summary['status'], 'passed')
        rows[-1] = dict(rows[-1], goal_fingerprint='different-goal')
        with self.assertRaisesRegex(ControlledGoalReplayError, 'fingerprint mismatch'):
            verify_paired_fingerprint_groups(rows, expected_members=3)
