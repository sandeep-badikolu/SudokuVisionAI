"""Legal digits for empty cells.

Candidates are the digits that do not yet appear in the cell's row, column,
or box. Filled cells have no remaining choices. This module does not decide
whether the board is solvable.
"""

from collections.abc import Sequence

from app.core.grid import GridSpec

Board = Sequence[Sequence[int]]


def peer_coordinates(spec: GridSpec, row: int, col: int) -> tuple[tuple[int, int], ...]:
    """Return every other cell that shares a row, column, or box."""
    spec.require_coordinates(row, col)
    peers: set[tuple[int, int]] = set()
    for other_col in range(spec.size):
        if other_col != col:
            peers.add((row, other_col))
    for other_row in range(spec.size):
        if other_row != row:
            peers.add((other_row, col))
    box = spec.box_index(row, col)
    for other_row, other_col in spec.cells_in_box(box):
        if (other_row, other_col) != (row, col):
            peers.add((other_row, other_col))
    return tuple(sorted(peers))


def cell_candidates(spec: GridSpec, board: Board, row: int, col: int) -> frozenset[int]:
    """Return the digits that can still be placed in an empty cell.

    A filled cell returns an empty set because it has no placement choice.
    """
    spec.require_coordinates(row, col)
    _require_board(spec, board)
    value = board[row][col]
    if value != 0:
        return frozenset()
    used = {
        board[other_row][other_col]
        for other_row, other_col in peer_coordinates(spec, row, col)
        if board[other_row][other_col] != 0
    }
    digits = range(1, spec.max_digit + 1)
    return frozenset(digit for digit in digits if digit not in used)


def candidate_grid(
    spec: GridSpec, board: Board
) -> tuple[tuple[frozenset[int], ...], ...]:
    """Return candidate sets in row-major order."""
    _require_board(spec, board)
    return tuple(
        tuple(cell_candidates(spec, board, row, col) for col in range(spec.size))
        for row in range(spec.size)
    )


def _require_board(spec: GridSpec, board: Board) -> None:
    if len(board) != spec.size or any(len(row) != spec.size for row in board):
        raise ValueError(f"board must be {spec.size}×{spec.size}")
    for row in range(spec.size):
        for col in range(spec.size):
            value = board[row][col]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"cell ({row}, {col}) must be an int "
                    f"from 0 through {spec.max_digit}"
                )
            if value < 0 or value > spec.max_digit:
                raise ValueError(
                    f"cell ({row}, {col}) value {value} is outside 0..{spec.max_digit}"
                )
