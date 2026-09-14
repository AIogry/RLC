from __future__ import annotations

import unittest

import numpy as np

from impls.diagnostics.puzzle.control_coordinates import (
    audit_effect_event_index,
    axis_assignment_diagnostics,
    effective_matrix_diagnostics,
    evaluate_control_coordinates,
    gf2_apply,
    gf2_rank,
    goal_mask_diagnostics,
    same_effect_consistency_diagnostics,
)
from impls.representation.puzzle_effects import (
    detect_board_change_events,
    raw_effect_signature,
)
from impls.utils.effect_datasets import build_effect_event_index


def _event_index(effects, *, episodes=1):
    effects = np.asarray(effects, dtype=np.uint8)
    boards = []
    terminals = []
    for episode in range(episodes):
        current = np.asarray(
            [(episode + bit) % 2 for bit in range(effects.shape[1])],
            dtype=np.uint8,
        )
        boards.append(current.copy())
        terminals.append(0)
        for effect in effects:
            current = np.bitwise_xor(current, effect)
            boards.append(current.copy())
            terminals.append(0)
        terminals[-1] = 1
    boards = np.asarray(boards, dtype=np.uint8)
    return build_effect_event_index(
        boards,
        np.asarray(terminals, dtype=np.uint8),
        event_mask=detect_board_change_events(boards),
        effect_signature_fn=raw_effect_signature,
    )


class GF2DiagnosticTest(unittest.TestCase):
    def test_rank_is_correct_for_full_and_rank_deficient_matrices(self):
        full = np.asarray([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=np.uint8)
        deficient = np.asarray([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.uint8)
        self.assertEqual(gf2_rank(full), 3)
        self.assertEqual(gf2_rank(deficient), 2)

    def test_gf2_apply_uses_output_by_input_matrix_orientation(self):
        matrix = np.asarray([[1, 1], [0, 1]], dtype=np.uint8)
        vectors = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(
            gf2_apply(matrix, vectors),
            np.asarray([[1, 0], [1, 1], [0, 1]], dtype=np.uint8),
        )


class EventAuditTest(unittest.TestCase):
    def test_full_rank_audit_reports_identifiability_and_context_coverage(self):
        index = _event_index(np.eye(3, dtype=np.uint8), episodes=2)
        audit = audit_effect_event_index(index, window_scales=(1, 2))
        self.assertEqual(audit['number_of_episodes'], 2)
        self.assertEqual(audit['number_of_task_state_events'], 6)
        self.assertEqual(audit['number_of_unique_raw_xor_effects'], 3)
        self.assertEqual(audit['observed_effect_gf2_rank'], 3)
        self.assertTrue(audit['observed_effects_span_full_task_space'])
        self.assertEqual(audit['effects_with_cross_episode_coverage'], 3)
        self.assertTrue(audit['all_effects_have_cross_episode_coverage'])
        self.assertEqual(audit['events_per_episode'], [3, 3])
        self.assertIn('1', audit['candidate_future_window_scale_coverage'])

    def test_rank_deficient_effect_set_is_reported_without_oracle_reference(self):
        effects = np.asarray([[1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.uint8)
        audit = audit_effect_event_index(_event_index(effects))
        self.assertEqual(audit['observed_effect_gf2_rank'], 2)
        self.assertEqual(audit['number_of_independent_effect_vectors'], 2)
        self.assertFalse(audit['observed_effects_span_full_task_space'])


class CoordinateDiagnosticTest(unittest.TestCase):
    def setUp(self):
        # A is a nontrivial invertible upper-triangular map.  The three effect
        # vectors are columns of A^-1, so A maps them to a permutation basis.
        self.matrix = np.asarray([
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
        ], dtype=np.uint8)
        self.inverse = np.asarray([
            [1, 1, 1],
            [0, 1, 1],
            [0, 0, 1],
        ], dtype=np.uint8)
        effects = np.asarray([
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
        ], dtype=np.uint8)
        self.index = _event_index(effects, episodes=2)

    def encode(self, value):
        return gf2_apply(self.matrix, np.asarray(value, dtype=np.uint8))

    def inverse_encode(self, value):
        return gf2_apply(self.inverse, np.asarray(value, dtype=np.uint8))

    def goal_mask(self, state, goal):
        return np.bitwise_xor(self.encode(state), self.encode(goal))

    def test_independent_one_hot_effects_form_a_collision_free_axis_permutation(self):
        result = axis_assignment_diagnostics(self.index, self.matrix)
        self.assertEqual(result['number_mapped_to_exactly_one_latent_axis'], 3)
        self.assertEqual(result['number_of_distinct_latent_axes_used'], 3)
        self.assertFalse(result['one_hot_axis_collision'])
        self.assertEqual(result['latent_delta_collision_count'], 0)
        self.assertEqual(
            sorted(
                assignment['learned_axis']
                for assignment in result['effect_axis_assignments'].values()
            ),
            [0, 1, 2],
        )

    def test_same_effect_delta_is_identical_across_contexts_and_episodes(self):
        deltas = np.stack([
            self.encode(np.bitwise_xor(record.start_board, record.end_board))
            for record in self.index.records
        ])
        result = same_effect_consistency_diagnostics(self.index, deltas)
        self.assertTrue(result['same_effect_latent_delta_consistency'])
        self.assertTrue(result['cross_episode_consistency'])
        self.assertEqual(result['cross_episode_effect_group_count'], 3)

    def test_effective_matrix_and_goal_mask_diagnostics_are_exact(self):
        states = np.asarray([[0, 0, 0], [1, 0, 1], [1, 1, 0]], dtype=np.uint8)
        matrix = effective_matrix_diagnostics(
            self.matrix, encode_fn=self.encode, test_inputs=states
        )
        self.assertEqual(matrix['effective_matrix_rank'], 3)
        self.assertTrue(matrix['matrix_forward_exact'])
        goals = np.asarray([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=np.uint8)
        masks = goal_mask_diagnostics(states, goals, self.goal_mask)
        self.assertTrue(masks['goal_mask_exact_reproducibility'])
        self.assertEqual(masks['goal_mask_count'], 3)

    def test_complete_evaluation_reports_hard_and_structural_invariants(self):
        result = evaluate_control_coordinates(
            self.index,
            encode_fn=self.encode,
            inverse_fn=self.inverse_encode,
            goal_mask_fn=self.goal_mask,
            effective_matrix=self.matrix,
            batch_size=2,
            diagnostic_seed=19,
            matrix_test_states=32,
            goal_pair_count=6,
        )
        self.assertEqual(result['hard_event_metrics']['hard_axis_success_rate'], 1.0)
        self.assertEqual(result['hard_event_metrics']['mean_hard_latent_event_distance'], 1.0)
        self.assertEqual(result['effective_matrix']['effective_matrix_rank'], 3)
        self.assertEqual(result['event_audit']['observed_effect_gf2_rank'], 3)
        self.assertEqual(result['structural_invariants'], {
            'forward_inverse_exact': True,
            'gf2_linearity_exact': True,
            'zero_maps_to_zero': True,
            'forward_outputs_exactly_binary': True,
        })


if __name__ == '__main__':
    unittest.main()
