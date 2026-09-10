"""Exactness and orientation gates for production Puzzle GF(2) algebra."""

import unittest

import numpy as np

from impls.diagnostics.puzzle.algebra import (
    build_toggle_matrix as diagnostic_toggle_matrix,
)
from impls.representation.puzzle_algebra import (
    apply_gf2_map,
    build_toggle_matrix,
    gf2_inverse,
    gf2_rank,
    operator_metadata,
)
from impls.representation.relations import puzzle_toggle_relation


class PuzzleAlgebraTest(unittest.TestCase):
    def test_exact_ranks_and_singular_inverse_failure(self):
        expected = {(3, 3): 9, (4, 4): 12, (4, 5): 20, (4, 6): 24}
        for (rows, cols), rank in expected.items():
            with self.subTest(layout=(rows, cols)):
                matrix = build_toggle_matrix(rows, cols)
                self.assertEqual(gf2_rank(matrix), rank)
                metadata = operator_metadata(rows, cols)
                self.assertEqual(metadata['rank'], rank)
                self.assertEqual(metadata['full_rank'], rank == rows * cols)
        with self.assertRaisesRegex(ValueError, 'singular'):
            gf2_inverse(build_toggle_matrix(4, 4))

    def test_operator_metadata_is_cached_and_immutable(self):
        first = operator_metadata(4, 5)
        second = operator_metadata(4, 5)
        self.assertIs(first, second)
        with self.assertRaises(ValueError):
            first['matrix'][0, 0] = 0
        with self.assertRaises(ValueError):
            first['inverse'][0, 0] = 0

    def test_inverse_and_batched_map_are_exact_over_gf2(self):
        rng = np.random.default_rng(24001)
        for rows, cols in ((3, 3), (4, 5), (4, 6)):
            with self.subTest(layout=(rows, cols)):
                matrix = build_toggle_matrix(rows, cols)
                inverse = gf2_inverse(matrix)
                identity = np.eye(rows * cols, dtype=np.uint8)
                np.testing.assert_array_equal(
                    np.asarray(apply_gf2_map(matrix, inverse.T)).T,
                    identity,
                )
                residuals = rng.integers(
                    0, 2, size=(7, rows * cols), dtype=np.uint8
                )
                operations = apply_gf2_map(inverse, residuals)
                reconstructed = apply_gf2_map(matrix, operations)
                np.testing.assert_array_equal(np.asarray(reconstructed), residuals)

    def test_every_source_column_matches_environment_toggle_support(self):
        for rows, cols in ((3, 3), (4, 4), (4, 5), (4, 6)):
            matrix = build_toggle_matrix(rows, cols)
            for source in range(rows * cols):
                source_row, source_col = divmod(source, cols)
                expected = set()
                for delta_row, delta_col in (
                    (0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)
                ):
                    target_row = source_row + delta_row
                    target_col = source_col + delta_col
                    if 0 <= target_row < rows and 0 <= target_col < cols:
                        expected.add(target_row * cols + target_col)
                actual = set(np.flatnonzero(matrix[:, source]).tolist())
                self.assertEqual(actual, expected, msg=(rows, cols, source))

    def test_production_matches_diagnostic_and_relation_conventions(self):
        for rows, cols in ((3, 3), (4, 4), (4, 5), (4, 6)):
            production = build_toggle_matrix(rows, cols)
            np.testing.assert_array_equal(
                production, diagnostic_toggle_matrix(rows, cols)
            )
            relation = np.asarray(puzzle_toggle_relation(rows, cols))[..., 0]
            np.testing.assert_array_equal(production.T, relation)

    def test_malformed_binary_inputs_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, 'only 0/1'):
            gf2_rank(np.asarray([[1, 2], [0, 1]]))
        with self.assertRaisesRegex(ValueError, 'input width'):
            apply_gf2_map(np.eye(3, dtype=np.uint8), np.zeros(2, dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, 'only 0/1'):
            apply_gf2_map(np.eye(2, dtype=np.uint8), np.asarray([0.0, 0.5]))


if __name__ == '__main__':
    unittest.main()
