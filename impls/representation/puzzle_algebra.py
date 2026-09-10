"""Exact production GF(2) operators for canonical Puzzle boards.

The matrix convention is ``M[target, source] == 1``: a column is the board
effect of pressing one source button.  Matrix construction, rank, and inverse
are NumPy setup-time operations.  Network calls only use ``apply_gf2_map``,
which is JAX-compatible integer arithmetic modulo two.
"""

from __future__ import annotations

from functools import lru_cache
from numbers import Integral
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np


PUZZLE_OPERATOR_MATRIX_SOURCE = 'canonical_puzzle_toggle_rule'
PUZZLE_OPERATOR_ORIENTATION = 'M[target,source]'


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f'{name} must be a positive integer, got {value!r}')
    return int(value)


def _grid(rows: int, cols: int) -> tuple[int, int, int]:
    rows = _positive_int('rows', rows)
    cols = _positive_int('cols', cols)
    return rows, cols, rows * cols


def _binary_matrix(matrix) -> np.ndarray:
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError(f'GF(2) matrix must be two-dimensional, got {matrix.shape}')
    if not (
        np.issubdtype(matrix.dtype, np.number) or np.issubdtype(matrix.dtype, np.bool_)
    ):
        raise ValueError(f'GF(2) matrix must be numeric binary data, got {matrix.dtype}')
    if not np.all(np.isfinite(matrix)):
        raise ValueError('GF(2) matrix contains non-finite values')
    if not np.all((matrix == 0) | (matrix == 1)):
        raise ValueError('GF(2) matrix must contain only 0/1 values')
    return np.asarray(matrix, dtype=np.uint8)


def _rref(matrix: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    reduced = np.array(matrix, dtype=np.uint8, copy=True)
    num_rows, num_cols = reduced.shape
    pivot_columns = []
    pivot_row = 0
    for column in range(num_cols):
        if pivot_row == num_rows:
            break
        candidates = np.flatnonzero(reduced[pivot_row:, column])
        if not len(candidates):
            continue
        selected = pivot_row + int(candidates[0])
        if selected != pivot_row:
            reduced[[pivot_row, selected]] = reduced[[selected, pivot_row]]
        for row in range(num_rows):
            if row != pivot_row and reduced[row, column]:
                reduced[row] ^= reduced[pivot_row]
        pivot_columns.append(column)
        pivot_row += 1
    return reduced, tuple(pivot_columns)


def build_toggle_matrix(rows: int, cols: int) -> np.ndarray:
    """Build the canonical Lights-Out toggle operator over GF(2).

    A press toggles itself and every in-bounds up/down/left/right neighbour.
    Buttons use the environment's row-major ordering.
    """

    rows, cols, num_buttons = _grid(rows, cols)
    matrix = np.zeros((num_buttons, num_buttons), dtype=np.uint8)
    for source in range(num_buttons):
        source_row, source_col = divmod(source, cols)
        for row_delta, col_delta in (
            (0, 0),
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ):
            target_row = source_row + row_delta
            target_col = source_col + col_delta
            if 0 <= target_row < rows and 0 <= target_col < cols:
                target = target_row * cols + target_col
                matrix[target, source] = 1
    return matrix


def gf2_rank(matrix) -> int:
    """Return the exact matrix rank over GF(2)."""

    _, pivots = _rref(_binary_matrix(matrix))
    return len(pivots)


def gf2_inverse(matrix) -> np.ndarray:
    """Return the exact inverse of a full-rank square binary matrix.

    Singular and non-square matrices fail explicitly; this function never
    substitutes a float inverse, pseudo-inverse, or arbitrary affine solution.
    """

    matrix = _binary_matrix(matrix)
    num_rows, num_cols = matrix.shape
    if num_rows != num_cols:
        raise ValueError(
            f'GF(2) inverse requires a square matrix, got {matrix.shape}'
        )
    augmented = np.concatenate(
        [matrix.copy(), np.eye(num_rows, dtype=np.uint8)], axis=1
    )
    pivot_row = 0
    for column in range(num_cols):
        candidates = np.flatnonzero(augmented[pivot_row:, column])
        if not len(candidates):
            rank = gf2_rank(matrix)
            raise ValueError(
                'GF(2) matrix is singular and has no unique inverse: '
                f'shape={matrix.shape}, rank={rank}'
            )
        selected = pivot_row + int(candidates[0])
        if selected != pivot_row:
            augmented[[pivot_row, selected]] = augmented[[selected, pivot_row]]
        for row in range(num_rows):
            if row != pivot_row and augmented[row, column]:
                augmented[row] ^= augmented[pivot_row]
        pivot_row += 1
    if not np.array_equal(augmented[:, :num_cols], np.eye(num_rows, dtype=np.uint8)):
        raise ValueError('GF(2) elimination did not produce an identity left block')
    return np.asarray(augmented[:, num_cols:], dtype=np.uint8)


def _validate_concrete_binary(value, *, name: str) -> None:
    """Validate values when eager, while remaining usable under JAX tracing."""

    leaves = jax.tree_util.tree_leaves(value)
    if any(isinstance(leaf, jax.core.Tracer) for leaf in leaves):
        return
    array = np.asarray(value)
    if not (
        np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError(f'{name} must be numeric binary data, got {array.dtype}')
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{name} contains non-finite values')
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f'{name} must contain only 0/1 values')


def apply_gf2_map(matrix, bits):
    """Apply ``matrix @ bits`` over GF(2), with arbitrary leading batch axes."""

    matrix = jnp.asarray(matrix)
    bits = jnp.asarray(bits)
    if matrix.ndim != 2:
        raise ValueError(f'GF(2) matrix must be two-dimensional, got {matrix.shape}')
    if bits.ndim < 1:
        raise ValueError(f'GF(2) bits require a final feature axis, got {bits.shape}')
    if bits.shape[-1] != matrix.shape[1]:
        raise ValueError(
            'GF(2) input width must match matrix columns: '
            f'bits={bits.shape}, matrix={matrix.shape}'
        )
    _validate_concrete_binary(matrix, name='GF(2) matrix')
    _validate_concrete_binary(bits, name='GF(2) bits')
    products = matrix.astype(jnp.uint8) * bits.astype(jnp.uint8)[..., None, :]
    return jnp.bitwise_and(jnp.sum(products, axis=-1), 1).astype(jnp.uint8)


@lru_cache(maxsize=None)
def operator_metadata(rows: int, cols: int):
    """Return immutable cached static operator facts for one Puzzle layout."""

    rows, cols, num_buttons = _grid(rows, cols)
    matrix = build_toggle_matrix(rows, cols)
    rank = gf2_rank(matrix)
    full_rank = rank == num_buttons
    inverse = gf2_inverse(matrix) if full_rank else None
    matrix.setflags(write=False)
    if inverse is not None:
        inverse.setflags(write=False)
    return MappingProxyType({
        'rows': rows,
        'cols': cols,
        'num_buttons': num_buttons,
        'matrix': matrix,
        'rank': rank,
        'full_rank': full_rank,
        'inverse': inverse,
        'operator_matrix_source': PUZZLE_OPERATOR_MATRIX_SOURCE,
        'operator_orientation': PUZZLE_OPERATOR_ORIENTATION,
    })


__all__ = [
    'PUZZLE_OPERATOR_MATRIX_SOURCE',
    'PUZZLE_OPERATOR_ORIENTATION',
    'apply_gf2_map',
    'build_toggle_matrix',
    'gf2_inverse',
    'gf2_rank',
    'operator_metadata',
]
