"""A Sudoku puzzle whose given clues cannot be overwritten."""

from collections.abc import Iterable, Sequence

from app.core.grid import Cell, GridSpec


class Puzzle:
    """Given clues for one grid, stored separately from any working board.

    The clue grid is fixed at construction. ``working_board`` returns a new
    mutable copy so a solver can fill cells without changing the puzzle that
    was recognized or entered.
    """

    def __init__(self, spec: GridSpec, clues: Iterable[Iterable[int]]) -> None:
        rows = _freeze_clues(spec, clues)
        self._spec = spec
        self._clues = rows

    @classmethod
    def from_rows(cls, spec: GridSpec, rows: Iterable[Iterable[int]]) -> "Puzzle":
        """Build a puzzle from row-major values. ``0`` means empty."""
        return cls(spec, rows)

    @property
    def spec(self) -> GridSpec:
        """Geometry of this puzzle."""
        return self._spec

    @property
    def clues(self) -> tuple[tuple[int, ...], ...]:
        """Immutable clue grid. Empty cells are 0."""
        return self._clues

    def clue(self, row: int, col: int) -> int:
        """Return the given value at ``(row, col)``, or 0 when the cell was empty."""
        self._spec.require_coordinates(row, col)
        return self._clues[row][col]

    def is_given(self, row: int, col: int) -> bool:
        """Return whether the cell was filled in the original puzzle."""
        return self.clue(row, col) != 0

    def cell(self, row: int, col: int) -> Cell:
        """Return the original cell at ``(row, col)``."""
        return Cell(row=row, col=col, value=self.clue(row, col))

    def working_board(self) -> list[list[int]]:
        """Return a mutable copy of the clues.

        Assignments on the returned board do not change ``clues``. Each call
        returns a new copy.
        """
        return [list(row) for row in self._clues]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Puzzle):
            return NotImplemented
        return self._spec == other._spec and self._clues == other._clues

    def __hash__(self) -> int:
        return hash((self._spec, self._clues))

    def __repr__(self) -> str:
        return f"Puzzle(spec={self._spec!r}, clues={self._clues!r})"


def _freeze_clues(
    spec: GridSpec, clues: Iterable[Iterable[int]]
) -> tuple[tuple[int, ...], ...]:
    if isinstance(clues, (str, bytes)):
        raise TypeError("clues must be a sequence of rows, not a string")
    rows = list(clues)
    if len(rows) != spec.size:
        raise ValueError(f"expected {spec.size} rows, got {len(rows)}")
    frozen: list[tuple[int, ...]] = []
    for row_index, row in enumerate(rows):
        if isinstance(row, (str, bytes)):
            raise TypeError(
                f"row {row_index} must be a sequence of integers, not a string"
            )
        if not isinstance(row, Sequence):
            row = list(row)
        if len(row) != spec.size:
            raise ValueError(
                f"row {row_index} must contain {spec.size} values, got {len(row)}"
            )
        frozen.append(tuple(_require_symbol(spec, row_index, col_index, value)
                            for col_index, value in enumerate(row)))
    return tuple(frozen)


def _require_symbol(spec: GridSpec, row: int, col: int, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"cell ({row}, {col}) must be an int from 0 through {spec.max_digit}, "
            f"got {value!r}"
        )
    if value < 0 or value > spec.max_digit:
        raise ValueError(
            f"cell ({row}, {col}) value {value} is outside 0..{spec.max_digit}"
        )
    return value
