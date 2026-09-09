"""Targeted tests for observation-only Puzzle event and GF(2) diagnostics."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

from impls.diagnostics.puzzle import (
    PRESS_THRESHOLD_RAW,
    POSITION_SCALE,
    PuzzleLayoutError,
    analyze_puzzle_episode,
    build_toggle_matrix,
    compute_dstar,
    decode_board_bits,
    enumerate_affine_solutions,
    extract_button_joint_positions,
    extract_button_joint_velocities,
    gf2_rank,
    gf2_solve,
    identify_press_events,
    minimum_weight_solution,
)


def _observation(states, positions=None, velocities=None):
    states = np.asarray(states, dtype=np.uint8)
    if states.ndim < 1:
        raise ValueError('states must have a button axis')
    num_buttons = states.shape[-1]
    positions = np.zeros_like(states, dtype=np.float32) if positions is None else np.asarray(positions)
    velocities = np.zeros_like(states, dtype=np.float32) if velocities is None else np.asarray(velocities)
    if positions.shape != states.shape or velocities.shape != states.shape:
        raise ValueError('synthetic feature shapes must match states')
    observation = np.zeros(states.shape[:-1] + (19 + 4 * num_buttons,), dtype=np.float32)
    blocks = observation[..., 19:].reshape(states.shape[:-1] + (num_buttons, 4))
    blocks[..., 0] = 1 - states
    blocks[..., 1] = states
    blocks[..., 2] = positions * POSITION_SCALE
    blocks[..., 3] = velocities
    return observation


def _transition(states, source_mask, *, positions_before=None, positions_after=None):
    matrix = build_toggle_matrix(*states)
    before = np.zeros(states[0] * states[1], dtype=np.uint8)
    after = before ^ ((matrix @ np.asarray(source_mask, dtype=np.uint8)) % 2).astype(np.uint8)
    previous_position = np.zeros_like(before, dtype=np.float32) if positions_before is None else positions_before
    current_position = np.zeros_like(before, dtype=np.float32) if positions_after is None else positions_after
    return _observation(before, previous_position), _observation(after, current_position)


class PuzzleLayoutAndEventTest(unittest.TestCase):
    def test_canonical_dimensions_batching_and_strict_features(self):
        for rows, cols, dimension in ((3, 3, 55), (4, 4, 83), (4, 5, 99), (4, 6, 115)):
            num_buttons = rows * cols
            states = np.arange(num_buttons, dtype=np.uint8) % 2
            positions = np.linspace(-0.01, 0.02, num_buttons, dtype=np.float32)
            velocities = np.linspace(-0.2, 0.2, num_buttons, dtype=np.float32)
            observation = _observation(states, positions, velocities)
            self.assertEqual(observation.shape, (dimension,))
            np.testing.assert_array_equal(decode_board_bits(observation, rows=rows, cols=cols), states)
            np.testing.assert_allclose(
                extract_button_joint_positions(observation, rows=rows, cols=cols), positions,
                rtol=0.0, atol=1e-7,
            )
            np.testing.assert_allclose(
                extract_button_joint_positions(observation, rows=rows, cols=cols, raw=False),
                positions * POSITION_SCALE, rtol=0.0, atol=1e-6,
            )
            np.testing.assert_allclose(
                extract_button_joint_velocities(observation, rows=rows, cols=cols), velocities,
                rtol=0.0, atol=1e-7,
            )
            batched = np.stack([observation, observation])
            self.assertEqual(decode_board_bits(batched, rows=rows, cols=cols).shape, (2, num_buttons))
            self.assertEqual(
                extract_button_joint_positions(batched, rows=rows, cols=cols).shape,
                (2, num_buttons),
            )
            with self.assertRaises(PuzzleLayoutError):
                decode_board_bits(observation[:-1], rows=rows, cols=cols)
            malformed = observation.copy()
            malformed[19] = 0.5
            with self.assertRaises(PuzzleLayoutError):
                decode_board_bits(malformed, rows=rows, cols=cols)

    def test_threshold_crossing_boundaries_and_multiple_sources(self):
        states = np.zeros(4, dtype=np.uint8)
        previous = np.zeros(4, dtype=np.float32)
        current = np.zeros(4, dtype=np.float32)
        previous[1] = PRESS_THRESHOLD_RAW
        current[0] = PRESS_THRESHOLD_RAW
        current[1] = PRESS_THRESHOLD_RAW - 0.01
        current[2] = PRESS_THRESHOLD_RAW - 0.01
        event = identify_press_events(
            _observation(states, previous),
            _observation(states, current),
            rows=2,
            cols=2,
        )
        np.testing.assert_array_equal(event.source_mask, np.asarray([1, 0, 1, 0], dtype=np.uint8))
        self.assertEqual(event.source_indices, (0, 2))
        self.assertEqual(event.num_sources, 2)
        self.assertTrue(event.has_press)
        self.assertTrue(event.is_multi_press)
        self.assertEqual(event.event_type, 'multi_press')

        no_crossing = identify_press_events(
            _observation(states, np.full(4, PRESS_THRESHOLD_RAW)),
            _observation(states, np.full(4, PRESS_THRESHOLD_RAW)),
            rows=2,
            cols=2,
        )
        np.testing.assert_array_equal(no_crossing.source_mask, np.zeros(4, dtype=np.uint8))
        self.assertFalse(no_crossing.has_press)


class PuzzleAlgebraTest(unittest.TestCase):
    def test_rank_and_nullity_hard_cases(self):
        for rows, cols, rank, nullity in (
            (3, 3, 9, 0), (4, 4, 12, 4), (4, 5, 20, 0), (4, 6, 24, 0),
        ):
            matrix = build_toggle_matrix(rows, cols)
            self.assertEqual(matrix.shape, (rows * cols, rows * cols))
            self.assertEqual(gf2_rank(matrix), rank)
            solved = gf2_solve(matrix, np.zeros(rows * cols, dtype=np.uint8))
            self.assertEqual(solved.nullity, nullity)
            self.assertEqual(solved.kernel_basis.shape, (nullity, rows * cols))
            for vector in solved.kernel_basis:
                np.testing.assert_array_equal((matrix @ vector) % 2, np.zeros(rows * cols, dtype=np.uint8))

    def test_affine_solutions_minimum_weight_and_unreachable_rhs(self):
        matrix = build_toggle_matrix(4, 4)
        source = np.zeros(16, dtype=np.uint8)
        source[[0, 5, 10]] = 1
        rhs = ((matrix @ source) % 2).astype(np.uint8)
        solved = gf2_solve(matrix, rhs)
        self.assertEqual(solved.status, 'affine')
        solutions = enumerate_affine_solutions(
            solved.particular_solution, solved.kernel_basis, max_solutions=16,
        )
        self.assertEqual(len(solutions), 16)
        self.assertEqual(len({tuple(vector) for vector in solutions}), 16)
        for vector in solutions:
            np.testing.assert_array_equal((matrix @ vector) % 2, rhs)
        minimum = minimum_weight_solution(matrix, rhs)
        self.assertIsNotNone(minimum)
        self.assertEqual(int(minimum.sum()), min(int(vector.sum()) for vector in solutions))
        self.assertEqual(compute_dstar(rhs, np.zeros(16, dtype=np.uint8), matrix), int(minimum.sum()))
        with self.assertRaises(ValueError):
            enumerate_affine_solutions(solved.particular_solution, solved.kernel_basis, max_solutions=15)

        unreachable_rhs = None
        for index in range(16):
            candidate = np.eye(16, dtype=np.uint8)[index]
            if gf2_solve(matrix, candidate).status == 'inconsistent':
                unreachable_rhs = candidate
                break
        self.assertIsNotNone(unreachable_rhs)
        self.assertEqual(gf2_solve(matrix, unreachable_rhs).status, 'inconsistent')
        self.assertIsNone(compute_dstar(unreachable_rhs, np.zeros(16, dtype=np.uint8), matrix))

    def test_full_rank_solution_and_monotonic_identity(self):
        rng = np.random.default_rng(23041)
        for rows, cols in ((3, 3), (4, 5), (4, 6)):
            matrix = build_toggle_matrix(rows, cols)
            for _ in range(5):
                goal = rng.integers(0, 2, size=rows * cols, dtype=np.uint8)
                source = rng.integers(0, 2, size=rows * cols, dtype=np.uint8)
                board = goal ^ ((matrix @ source) % 2).astype(np.uint8)
                solved = gf2_solve(matrix, board ^ goal)
                self.assertEqual(solved.status, 'unique')
                np.testing.assert_array_equal(solved.particular_solution, source)
                dstar = compute_dstar(board, goal, matrix)
                self.assertEqual(dstar, int(source.sum()))
                for button in range(rows * cols):
                    next_board = board ^ matrix[:, button]
                    next_dstar = compute_dstar(next_board, goal, matrix)
                    expected = dstar - 1 if source[button] else dstar + 1
                    self.assertEqual(next_dstar, expected)

    def test_every_source_column_matches_observation_event_effect(self):
        rng = np.random.default_rng(23042)
        for rows, cols in ((3, 3), (4, 4), (4, 5), (4, 6)):
            num_buttons = rows * cols
            matrix = build_toggle_matrix(rows, cols)
            before = rng.integers(0, 2, size=num_buttons, dtype=np.uint8)
            for source_index in range(num_buttons):
                previous_position = np.zeros(num_buttons, dtype=np.float32)
                current_position = np.zeros(num_buttons, dtype=np.float32)
                current_position[source_index] = -0.03
                after = before ^ matrix[:, source_index]
                event = identify_press_events(
                    _observation(before, previous_position),
                    _observation(after, current_position),
                    rows=rows,
                    cols=cols,
                )
                self.assertEqual(event.source_indices, (source_index,))
                np.testing.assert_array_equal(event.board_delta, matrix[:, source_index])
            source_mask = np.zeros(num_buttons, dtype=np.uint8)
            source_mask[[0, num_buttons - 1]] = 1
            previous_position = np.zeros(num_buttons, dtype=np.float32)
            current_position = np.zeros(num_buttons, dtype=np.float32)
            current_position[source_mask.astype(bool)] = -0.03
            after = before ^ ((matrix @ source_mask) % 2).astype(np.uint8)
            event = identify_press_events(
                _observation(before, previous_position),
                _observation(after, current_position),
                rows=rows,
                cols=cols,
            )
            self.assertEqual(event.source_indices, (0, num_buttons - 1))
            np.testing.assert_array_equal(
                event.board_delta, ((matrix @ source_mask) % 2).astype(np.uint8)
            )

    def test_official_task_distances_are_read_from_canonical_task_infos(self):
        import ogbench

        expected = {
            'puzzle-3x3-play-v0': [2, 5, 8, 9, 7],
            'puzzle-4x4-play-v0': [4, 6, 6, 6, 7],
            'puzzle-4x5-play-v0': [4, 10, 14, 16, 20],
            'puzzle-4x6-play-v0': [6, 8, 12, 16, 24],
        }
        for environment, distances in expected.items():
            env = ogbench.make_env_and_datasets(environment, env_only=True)
            try:
                base = env.unwrapped
                matrix = build_toggle_matrix(base._num_rows, base._num_cols)
                observed = [
                    compute_dstar(task['init_button_states'], task['goal_button_states'], matrix)
                    for task in base.task_infos
                ]
                self.assertEqual(observed, distances)
            finally:
                env.close()


class PuzzleMetricTest(unittest.TestCase):
    def test_effect_consistency_and_episode_accounting(self):
        rows, cols = 3, 3
        matrix = build_toggle_matrix(rows, cols)
        before_board = np.zeros(9, dtype=np.uint8)
        source = np.zeros(9, dtype=np.uint8)
        source[4] = 1
        after_board = before_board ^ matrix[:, 4]
        previous_position = np.zeros(9, dtype=np.float32)
        current_position = np.zeros(9, dtype=np.float32)
        current_position[4] = -0.03
        trajectory = [{
            'observation': _observation(before_board, previous_position),
            'next_observation': _observation(after_board, current_position),
            'action': np.zeros(5), 'reward': 0.0, 'done': True,
            'info': {'success': 1.0}, 'terminated': True, 'truncated': False,
        }]
        result = analyze_puzzle_episode(
            trajectory,
            goal_board=after_board,
            rows=rows,
            cols=cols,
            max_episode_steps=500,
        )
        episode = result['episode']
        self.assertEqual(episode['num_press_events'], 1)
        self.assertEqual(episode['num_single_press_events'], 1)
        self.assertEqual(episode['num_source_presses'], 1)
        self.assertEqual(episode['first_press_step'], 1)
        self.assertEqual(episode['initial_Dstar'], 1)
        self.assertEqual(episode['final_Dstar'], 0)
        self.assertEqual(episode['progress_rate'], 1.0)
        self.assertTrue(result['press_events'][0]['effect_consistent'])
        self.assertEqual(result['invariants']['event_effect_consistency_failures'], 0)
        self.assertEqual(result['invariants']['unreachable_residuals'], 0)

    def test_four_by_four_random_reference_is_null_and_multisource_is_separate(self):
        rows, cols = 4, 4
        matrix = build_toggle_matrix(rows, cols)
        before = np.zeros(16, dtype=np.uint8)
        source = np.zeros(16, dtype=np.uint8)
        source[[0, 15]] = 1
        after = before ^ ((matrix @ source) % 2).astype(np.uint8)
        previous_position = np.zeros(16, dtype=np.float32)
        current_position = np.zeros(16, dtype=np.float32)
        current_position[source.astype(bool)] = -0.03
        result = analyze_puzzle_episode(
            [{
                'observation': _observation(before, previous_position),
                'next_observation': _observation(after, current_position),
                'action': np.zeros(5), 'reward': 0.0, 'done': True,
                'info': {'success': 0.0}, 'terminated': False, 'truncated': True,
            }],
            goal_board=after,
            rows=rows,
            cols=cols,
            max_episode_steps=1,
        )
        episode = result['episode']
        event = result['press_events'][0]
        self.assertEqual(event['event_type'], 'multi_press')
        self.assertEqual(episode['num_multi_press_events'], 1)
        self.assertEqual(episode['num_single_press_events'], 0)
        self.assertIsNone(episode['progress_rate'])
        self.assertIsNone(event['random_progress_probability'])
        self.assertIsNone(event['progress_advantage'])
        self.assertTrue(episode['horizon_exhausted'])


class PuzzleRealEnvironmentTest(unittest.TestCase):
    def test_observation_positions_and_detector_match_post_step_timing(self):
        import mujoco
        import ogbench

        environments = (
            ('puzzle-3x3-play-v0', 3, 3),
            ('puzzle-4x4-play-v0', 4, 4),
            ('puzzle-4x5-play-v0', 4, 5),
            ('puzzle-4x6-play-v0', 4, 6),
        )
        for environment, rows, cols in environments:
            env = ogbench.make_env_and_datasets(environment, env_only=True)
            try:
                observation, info = env.reset(seed=230422, options={'task_id': 1, 'render_goal': False})
                base = env.unwrapped
                np.testing.assert_array_equal(
                    decode_board_bits(info['goal'], rows=rows, cols=cols),
                    np.asarray(base.task_infos[0]['goal_button_states'], dtype=np.uint8),
                )
                raw_previous = np.asarray([
                    base._data.joint(f'buttonbox_joint_{i}').qpos.copy()[0]
                    for i in range(rows * cols)
                ])
                base.pre_step()
                base._data.joint('buttonbox_joint_0').qpos[0] = -0.03
                mujoco.mj_forward(base._model, base._data)
                base.post_step()
                next_observation = base.compute_observation()
                raw_current = np.asarray([
                    base._data.joint(f'buttonbox_joint_{i}').qpos.copy()[0]
                    for i in range(rows * cols)
                ])
                np.testing.assert_allclose(
                    extract_button_joint_positions(observation, rows=rows, cols=cols),
                    raw_previous, rtol=0.0, atol=1e-7,
                )
                np.testing.assert_allclose(
                    extract_button_joint_positions(next_observation, rows=rows, cols=cols),
                    raw_current, rtol=0.0, atol=1e-7,
                )
                event = identify_press_events(
                    observation, next_observation, rows=rows, cols=cols,
                )
                expected_source = ((raw_previous > PRESS_THRESHOLD_RAW)
                                   & (raw_current <= PRESS_THRESHOLD_RAW)).astype(np.uint8)
                np.testing.assert_array_equal(event.source_mask, expected_source)
                matrix = build_toggle_matrix(rows, cols)
                predicted = ((matrix @ expected_source) % 2).astype(np.uint8)
                np.testing.assert_array_equal(event.board_delta, predicted)
            finally:
                env.close()


REAL_SOURCE = Path(os.environ.get(
    'M23_DIAGNOSTIC_SOURCE_3X3',
    '/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M22/'
    'M22-CRL-P3X3__crl-p3x3/puzzle-3x3-play-v0/seed_000',
))


class _ReplayFirstGoal:
    """Test-only goal replay to isolate post-processing from reset noise.

    The pinned Puzzle reset creates the goal arm pose using unseeded Box
    sampling.  Replaying the first returned goal keeps the policy input fixed
    while exercising the same canonical rollout code; production diagnosis
    does not use this wrapper.
    """

    def __init__(self, env):
        self.env = env
        self.goal = None

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def reset(self, *args, **kwargs):
        observation, info = self.env.reset(*args, **kwargs)
        info = dict(info)
        if self.goal is None:
            self.goal = np.asarray(info['goal']).copy()
        else:
            info['goal'] = self.goal.copy()
        return observation, info

    def step(self, action):
        return self.env.step(action)


@unittest.skipUnless(REAL_SOURCE.is_dir(), 'completed immutable Puzzle checkpoint is unavailable')
class PuzzleRolloutParityTest(unittest.TestCase):
    def test_diagnosis_postprocessing_does_not_change_rollout_or_checkpoint(self):
        from impls.diagnostics.puzzle import analyze_puzzle_episode
        from impls.experiment.reevaluation import _make_restored_agent, _restore_probe, validate_source_run
        from impls.utils.checkpointing import sha256_file, tree_fingerprint
        from impls.utils.evaluation import _rollout_episode, common_episode_seeds, extract_episode_success

        provenance = validate_source_run(
            REAL_SOURCE,
            checkpoint_selector={'selector': 'step', 'step': 1_000_000},
            check_checkpoint_metadata=True,
        )
        restored, env, config, example_batch = _make_restored_agent(provenance)
        checkpoint_before = sha256_file(provenance['checkpoint_path'])
        fingerprint_before = tree_fingerprint(restored.network.params)
        try:
            _restore_probe(restored, example_batch, provenance['source_metadata']['algorithm'], provenance['source_training_seed'])
            seeds = common_episode_seeds(230423, 1, 0)
            parity_env = _ReplayFirstGoal(env)
            kwargs = dict(
                task_id=1,
                config=config,
                episode_seed=seeds['episode_seed'],
                actor_seed=seeds['actor_seed'],
                noise_seed=seeds['noise_seed'],
                eval_temperature=0.0,
                eval_gaussian=None,
                retain_trajectory=True,
                render=False,
                video_frame_skip=1,
            )
            canonical = _rollout_episode(restored, parity_env, **kwargs)
            goal = np.asarray(parity_env.unwrapped.task_infos[0]['goal_button_states'], dtype=np.uint8)
            analysis = analyze_puzzle_episode(
                canonical['trajectory'], goal_board=goal, rows=3, cols=3,
                max_episode_steps=env.spec.max_episode_steps,
            )
            diagnosis = _rollout_episode(restored, parity_env, **kwargs)
            self.assertEqual(canonical['episode_length'], diagnosis['episode_length'])
            self.assertEqual(
                extract_episode_success(canonical['final_info']),
                extract_episode_success(diagnosis['final_info']),
            )
            self.assertGreaterEqual(analysis['episode']['episode_length'], 1)
            for key in ('observation', 'next_observation', 'action'):
                self.assertEqual(len(canonical['trajectory'][key]), len(diagnosis['trajectory'][key]))
                for first, second in zip(canonical['trajectory'][key], diagnosis['trajectory'][key]):
                    np.testing.assert_allclose(np.asarray(first), np.asarray(second), rtol=0.0, atol=0.0)
            self.assertEqual(checkpoint_before, sha256_file(provenance['checkpoint_path']))
            self.assertEqual(fingerprint_before, tree_fingerprint(restored.network.params))
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
