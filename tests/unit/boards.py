"""Completed grids and small puzzle builders shared by unit tests."""

from app.core import GridSpec, Puzzle

COMPLETED = {
    4: (
        (1, 2, 3, 4),
        (3, 4, 1, 2),
        (2, 1, 4, 3),
        (4, 3, 2, 1),
    ),
    6: (
        (1, 2, 3, 4, 5, 6),
        (4, 5, 6, 1, 2, 3),
        (2, 3, 1, 5, 6, 4),
        (5, 6, 4, 2, 3, 1),
        (3, 1, 2, 6, 4, 5),
        (6, 4, 5, 3, 1, 2),
    ),
    9: (
        (5, 3, 4, 6, 7, 8, 9, 1, 2),
        (6, 7, 2, 1, 9, 5, 3, 4, 8),
        (1, 9, 8, 3, 4, 2, 5, 6, 7),
        (8, 5, 9, 7, 6, 1, 4, 2, 3),
        (4, 2, 6, 8, 5, 3, 7, 9, 1),
        (7, 1, 3, 9, 2, 4, 8, 5, 6),
        (9, 6, 1, 5, 3, 7, 2, 8, 4),
        (2, 8, 7, 4, 1, 9, 6, 3, 5),
        (3, 4, 5, 2, 8, 6, 1, 7, 9),
    ),
}


def puzzle_from(rows: tuple[tuple[int, ...], ...]) -> Puzzle:
    """Build a puzzle whose order matches the given rows."""
    return Puzzle.from_rows(GridSpec.for_size(len(rows)), rows)


def with_cells(
    rows: tuple[tuple[int, ...], ...], edits: dict[tuple[int, int], int]
) -> Puzzle:
    """Copy ``rows`` and replace the cells named in ``edits``."""
    mutable = [list(row) for row in rows]
    for (row, col), value in edits.items():
        mutable[row][col] = value
    return Puzzle.from_rows(GridSpec.for_size(len(rows)), mutable)
