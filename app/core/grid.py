"""Supported Sudoku geometries.

A grid is defined only by its size and box shape. Digit recognition and
solving are outside this module.
"""

from dataclasses import dataclass

_CANONICAL_BOXES: dict[int, tuple[int, int]] = {
    4: (2, 2),
    6: (2, 3),
    9: (3, 3),
}


def _unsupported_size_message(size: int) -> str:
    supported = ", ".join(str(item) for item in sorted(_CANONICAL_BOXES))
    return f"unsupported grid size {size}; expected one of {supported}"


@dataclass(frozen=True, slots=True)
class GridSpec:
    """Size and box shape of one supported Sudoku variant.

    The only accepted geometries are 4×4 with 2×2 boxes, 6×6 with 2×3 boxes,
    and 9×9 with 3×3 boxes. Digits run from 1 through ``size``. Empty cells
    use 0, which is not a digit.
    """

    size: int
    box_rows: int
    box_cols: int

    def __post_init__(self) -> None:
        expected = _CANONICAL_BOXES.get(self.size)
        if expected is None:
            raise ValueError(_unsupported_size_message(self.size))
        if (self.box_rows, self.box_cols) != expected:
            box_rows, box_cols = expected
            raise ValueError(
                f"{self.size}×{self.size} grids use {box_rows}×{box_cols} boxes, "
                f"not {self.box_rows}×{self.box_cols}"
            )

    @classmethod
    def for_size(cls, size: int) -> "GridSpec":
        """Return the canonical spec for a supported grid size."""
        expected = _CANONICAL_BOXES.get(size)
        if expected is None:
            raise ValueError(_unsupported_size_message(size))
        box_rows, box_cols = expected
        return cls(size=size, box_rows=box_rows, box_cols=box_cols)

    @property
    def max_digit(self) -> int:
        """Largest legal digit. Empty cells are 0 and are not included."""
        return self.size

    @property
    def cell_count(self) -> int:
        """Number of cells in the full grid."""
        return self.size * self.size

    @property
    def box_count(self) -> int:
        """Number of boxes. For these geometries it equals ``size``."""
        return (self.size // self.box_rows) * (self.size // self.box_cols)

    def contains(self, row: int, col: int) -> bool:
        """Return whether ``(row, col)`` lies inside the grid."""
        return 0 <= row < self.size and 0 <= col < self.size

    def require_coordinates(self, row: int, col: int) -> None:
        """Raise ``ValueError`` when ``(row, col)`` is outside the grid."""
        if not self.contains(row, col):
            raise ValueError(
                f"cell ({row}, {col}) is outside a {self.size}×{self.size} grid"
            )

    def box_index(self, row: int, col: int) -> int:
        """Return the box number for a cell, numbered left to right, top to bottom."""
        self.require_coordinates(row, col)
        box_row = row // self.box_rows
        box_col = col // self.box_cols
        boxes_across = self.size // self.box_cols
        return box_row * boxes_across + box_col

    def box_origin(self, row: int, col: int) -> tuple[int, int]:
        """Return the top-left coordinates of the box that contains ``(row, col)``."""
        self.require_coordinates(row, col)
        origin_row = (row // self.box_rows) * self.box_rows
        origin_col = (col // self.box_cols) * self.box_cols
        return origin_row, origin_col

    def cells_in_box(self, box_index: int) -> tuple[tuple[int, int], ...]:
        """Return every cell coordinate that belongs to ``box_index``."""
        if not 0 <= box_index < self.box_count:
            raise ValueError(
                f"box {box_index} is outside a {self.size}×{self.size} grid"
            )
        boxes_across = self.size // self.box_cols
        box_row, box_col = divmod(box_index, boxes_across)
        origin_row = box_row * self.box_rows
        origin_col = box_col * self.box_cols
        return tuple(
            (origin_row + offset_row, origin_col + offset_col)
            for offset_row in range(self.box_rows)
            for offset_col in range(self.box_cols)
        )


@dataclass(frozen=True, slots=True)
class Cell:
    """One grid position and the symbol stored there.

    ``value`` is 0 when the cell is empty, otherwise a digit from 1 through
    the grid size. This type does not decide whether that digit is legal in
    context; that check belongs to puzzle validation.
    """

    row: int
    col: int
    value: int
