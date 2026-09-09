"""Exact GF(2) algebra for canonical Lights-Out Puzzle boards."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np


class DStarError(ValueError):
    """Raised for invalid GF(2) boards or algebraic dimensions."""


@dataclass(frozen=True)
class GF2SolveResult:
    """Structured result for ``Mx = rhs`` over GF(2).

    ``kernel_basis`` has shape ``[nullity, n_columns]``; each row is one
    independent binary nullspace vector.  An inconsistent system has no
    ``particular_solution`` and must not be treated as solved.
    """

    status: str
    particular_solution: np.ndarray | None
    kernel_basis: np.ndarray
    rank: int
    nullity: int


def _validate_grid(rows, cols):
    for name, value in (('rows', rows), ('cols', cols)):
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
            raise DStarError(f'{name} must be a positive integer, got {value!r}')
    rows, cols = int(rows), int(cols)
    return rows, cols, rows * cols


def _binary_array(value, *, name, ndim=None):
    array = np.asarray(value)
    if ndim is not None and array.ndim != ndim:
        raise DStarError(f'{name} must have ndim={ndim}, got shape {array.shape}')
    if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
        raise DStarError(f'{name} must be numeric binary data, got {array.dtype}')
    if not np.all(np.isfinite(array)):
        raise DStarError(f'{name} contains non-finite values')
    if not np.all((array == 0) | (array == 1)):
        raise DStarError(f'{name} must contain only 0/1 values')
    return np.asarray(array, dtype=np.uint8)


def _matrix(matrix):
    matrix = _binary_array(matrix, name='GF(2) matrix', ndim=2)
    return matrix


def _rref(matrix, rhs=None):
    matrix = np.array(matrix, dtype=np.uint8, copy=True)
    rhs = None if rhs is None else np.array(rhs, dtype=np.uint8, copy=True)
    rows, cols = matrix.shape
    pivot_columns = []
    pivot_row = 0
    for column in range(cols):
        if pivot_row == rows:
            break
        candidates = np.flatnonzero(matrix[pivot_row:, column])
        if not len(candidates):
            continue
        selected_row = pivot_row + int(candidates[0])
        if selected_row != pivot_row:
            matrix[[pivot_row, selected_row]] = matrix[[selected_row, pivot_row]]
            if rhs is not None:
                rhs[[pivot_row, selected_row]] = rhs[[selected_row, pivot_row]]
        for row in range(rows):
            if row != pivot_row and matrix[row, column]:
                matrix[row] ^= matrix[pivot_row]
                if rhs is not None:
                    rhs[row] ^= rhs[pivot_row]
        pivot_columns.append(column)
        pivot_row += 1
    return matrix, rhs, tuple(pivot_columns)


def build_toggle_matrix(rows, cols):
    """Build ``M`` with press-source effects as columns in row-major order.

    The support is sourced from the local pinned
    ``PuzzleEnv.post_step`` implementation: a press toggles self and each
    in-bounds cardinal neighbour.  The matrix convention is therefore
    ``board_after = board_before XOR (M @ source_mask mod 2)``.
    """

    rows, cols, num_buttons = _validate_grid(rows, cols)
    matrix = np.zeros((num_buttons, num_buttons), dtype=np.uint8)
    for source in range(num_buttons):
        row, column = divmod(source, cols)
        for delta_row, delta_col in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            next_row, next_col = row + delta_row, column + delta_col
            if 0 <= next_row < rows and 0 <= next_col < cols:
                matrix[next_row * cols + next_col, source] = 1
    return matrix


def gf2_rank(matrix):
    """Return the exact rank of a binary matrix over GF(2)."""

    matrix = _matrix(matrix)
    _, _, pivots = _rref(matrix)
    return int(len(pivots))


def gf2_nullspace(matrix):
    """Return a row-oriented independent basis for the GF(2) nullspace."""

    matrix = _matrix(matrix)
    reduced, _, pivot_columns = _rref(matrix)
    columns = matrix.shape[1]
    free_columns = [column for column in range(columns) if column not in pivot_columns]
    basis = np.zeros((len(free_columns), columns), dtype=np.uint8)
    for basis_index, free_column in enumerate(free_columns):
        vector = basis[basis_index]
        vector[free_column] = 1
        for row, pivot_column in enumerate(pivot_columns):
            if reduced[row, free_column]:
                vector[pivot_column] = 1
    return basis


def gf2_solve(matrix, rhs):
    """Solve ``matrix @ x = rhs`` over GF(2) with explicit status semantics."""

    matrix = _matrix(matrix)
    rhs = _binary_array(rhs, name='GF(2) rhs', ndim=1)
    if rhs.shape[0] != matrix.shape[0]:
        raise DStarError(
            f'GF(2) rhs length {rhs.shape[0]} does not match matrix rows {matrix.shape[0]}'
        )
    reduced, reduced_rhs, pivot_columns = _rref(matrix, rhs)
    rank = len(pivot_columns)
    nullity = matrix.shape[1] - rank
    kernel_basis = gf2_nullspace(matrix)
    inconsistent = np.any((~np.any(reduced, axis=1)) & (reduced_rhs == 1))
    if inconsistent:
        return GF2SolveResult(
            status='inconsistent',
            particular_solution=None,
            kernel_basis=kernel_basis,
            rank=int(rank),
            nullity=int(nullity),
        )
    particular = np.zeros(matrix.shape[1], dtype=np.uint8)
    for row, pivot_column in enumerate(pivot_columns):
        particular[pivot_column] = reduced_rhs[row]
    return GF2SolveResult(
        status='unique' if nullity == 0 else 'affine',
        particular_solution=particular,
        kernel_basis=kernel_basis,
        rank=int(rank),
        nullity=int(nullity),
    )


def enumerate_affine_solutions(particular, kernel_basis, *, max_solutions=65536):
    """Enumerate ``particular + span(kernel_basis)`` with an exponential guard."""

    particular = _binary_array(particular, name='particular solution', ndim=1)
    kernel_basis = _binary_array(kernel_basis, name='kernel basis', ndim=2)
    if kernel_basis.shape[1] != particular.shape[0]:
        raise DStarError(
            f'Kernel width {kernel_basis.shape[1]} does not match solution width {particular.shape[0]}'
        )
    if isinstance(max_solutions, bool) or not isinstance(max_solutions, Integral) or int(max_solutions) <= 0:
        raise DStarError(f'max_solutions must be a positive integer, got {max_solutions!r}')
    max_solutions = int(max_solutions)
    nullity = kernel_basis.shape[0]
    solution_count = 1 << nullity
    if solution_count > max_solutions:
        raise DStarError(
            f'Affine solution space has {solution_count} solutions, exceeding max_solutions={max_solutions}'
        )
    solutions = []
    for mask in range(solution_count):
        solution = particular.copy()
        for basis_index in range(nullity):
            if mask & (1 << basis_index):
                solution ^= kernel_basis[basis_index]
        solutions.append(solution)
    return solutions


def minimum_weight_solution(matrix, rhs, *, max_solutions=65536):
    """Return the deterministic minimum-Hamming-weight solution or ``None``.

    ``None`` is the only result for an inconsistent system; callers must not
    interpret it as a zero vector or as a valid distance.
    """

    solved = gf2_solve(matrix, rhs)
    if solved.status == 'inconsistent':
        return None
    candidates = enumerate_affine_solutions(
        solved.particular_solution,
        solved.kernel_basis,
        max_solutions=max_solutions,
    )
    return min(candidates, key=lambda value: (int(np.sum(value)), tuple(int(x) for x in value)))


def compute_dstar(board, goal, matrix):
    """Return algebraic minimum press count ``D*`` or ``None`` if unreachable.

    This is a button-press count in the GF(2) model, not robot motion
    distance, environment-step distance, reward, policy value, or Q value.
    """

    board = _binary_array(board, name='board', ndim=1)
    goal = _binary_array(goal, name='goal', ndim=1)
    matrix = _matrix(matrix)
    if matrix.shape != (board.shape[0], board.shape[0]) or goal.shape != board.shape:
        raise DStarError(
            f'Expected square matrix and board/goal width {board.shape[0]}, '
            f'got matrix={matrix.shape}, board={board.shape}, goal={goal.shape}'
        )
    residual = np.bitwise_xor(board, goal)
    solution = minimum_weight_solution(matrix, residual)
    return None if solution is None else int(np.sum(solution))


__all__ = [
    'DStarError',
    'GF2SolveResult',
    'build_toggle_matrix',
    'compute_dstar',
    'enumerate_affine_solutions',
    'gf2_nullspace',
    'gf2_rank',
    'gf2_solve',
    'minimum_weight_solution',
]
