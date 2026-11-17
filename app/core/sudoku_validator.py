"""Board shape, cell values, and row, column, and box duplicates.

Box shape comes from ``GridSpec`` (2×2, 2×3, or 3×3). A board that fails
these checks is invalid. A board that passes can still be unsolvable.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from app.core.grid import GridSpec
from app.core.puzzle import Puzzle

_SUPPORTED_SIZES = (4, 6, 9)


class UnitKind(Enum):
    """Why a board was rejected, or which constraint unit contains a duplicate."""

    DIMENSION = "dimension"
    VALUE = "value"
    ROW = "row"
    COLUMN = "column"
    BOX = "box"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One problem found while checking a board."""

    kind: UnitKind
    index: int
    value: int
    cells: tuple[tuple[int, int], ...]
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Every problem found in one board."""

    issues: tuple[ValidationIssue, ...]

    @property
    def valid(self) -> bool:
        """True when the board has no dimension, value, or duplicate issues."""
        return not self.issues


def validate_board(
    board: object, *, spec: GridSpec | None = None
) -> ValidationResult:
    """Check shape, allowed values, and duplicate clues.

    ``0`` is an empty cell. Digits must be ``1`` through the grid size.
    Duplicate checks use the box shape on ``spec`` when it is given, otherwise
    the canonical shape for a 4×4, 6×6, or 9×9 square. The input is not modified.
    """
    rows = _rows(board)
    if rows is None:
        return ValidationResult(
            issues=(
                ValidationIssue(
                    kind=UnitKind.DIMENSION,
                    index=0,
                    value=0,
                    cells=(),
                    detail="board must be a sequence of rows",
                ),
            )
        )

    dimension = _dimension_issue(rows, spec)
    if dimension is not None:
        return ValidationResult(issues=(dimension,))

    resolved = spec if spec is not None else GridSpec.for_size(len(rows))
    value_issues = _value_issues(resolved, rows)
    duplicate_issues = _duplicate_issues(resolved, rows)
    return ValidationResult(issues=tuple(value_issues + duplicate_issues))


def validate_puzzle(puzzle: Puzzle) -> ValidationResult:
    """Check a puzzle's original clues. Empty cells are ignored."""
    return validate_board(puzzle.clues, spec=puzzle.spec)


def _rows(board: object) -> tuple[tuple[object, ...], ...] | None:
    if isinstance(board, (str, bytes)) or not isinstance(board, Sequence):
        return None
    rows: list[tuple[object, ...]] = []
    for row in board:
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            return None
        rows.append(tuple(row))
    return tuple(rows)


def _dimension_issue(
    rows: tuple[tuple[object, ...], ...], spec: GridSpec | None
) -> ValidationIssue | None:
    height = len(rows)
    widths = {len(row) for row in rows}
    if spec is None:
        if height in _SUPPORTED_SIZES and widths == {height}:
            return None
        supported = ", ".join(str(size) for size in _SUPPORTED_SIZES)
        return ValidationIssue(
            kind=UnitKind.DIMENSION,
            index=height,
            value=next(iter(widths)) if len(widths) == 1 else 0,
            cells=(),
            detail=f"expected a {supported} square",
        )
    if height == spec.size and widths == {spec.size}:
        return None
    return ValidationIssue(
        kind=UnitKind.DIMENSION,
        index=height,
        value=next(iter(widths)) if len(widths) == 1 else 0,
        cells=(),
        detail=f"expected a {spec.size}×{spec.size} board",
    )


def _value_issues(
    spec: GridSpec, rows: tuple[tuple[object, ...], ...]
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for row in range(spec.size):
        for col in range(spec.size):
            value = rows[row][col]
            if isinstance(value, bool) or not isinstance(value, int):
                issues.append(
                    ValidationIssue(
                        kind=UnitKind.VALUE,
                        index=row,
                        value=0,
                        cells=((row, col),),
                        detail=f"cell ({row}, {col}) is {value!r}",
                    )
                )
                continue
            if value < 0 or value > spec.max_digit:
                issues.append(
                    ValidationIssue(
                        kind=UnitKind.VALUE,
                        index=row,
                        value=value,
                        cells=((row, col),),
                        detail=f"cell ({row}, {col}) is outside 0..{spec.max_digit}",
                    )
                )
    return issues


def _duplicate_issues(
    spec: GridSpec, rows: tuple[tuple[object, ...], ...]
) -> list[ValidationIssue]:
    clues = tuple(
        tuple(value if _is_clue(spec, value) else 0 for value in row) for row in rows
    )
    issues: list[ValidationIssue] = []
    issues.extend(_row_issues(spec, clues))
    issues.extend(_column_issues(spec, clues))
    issues.extend(_box_issues(spec, clues))
    return issues


def _is_clue(spec: GridSpec, value: object) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 1 <= value <= spec.max_digit
    )


def _row_issues(
    spec: GridSpec, clues: tuple[tuple[int, ...], ...]
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for row in range(spec.size):
        cells = tuple((row, col) for col in range(spec.size))
        issues.extend(_issues_for_unit(UnitKind.ROW, row, cells, clues))
    return issues


def _column_issues(
    spec: GridSpec, clues: tuple[tuple[int, ...], ...]
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for col in range(spec.size):
        cells = tuple((row, col) for row in range(spec.size))
        issues.extend(_issues_for_unit(UnitKind.COLUMN, col, cells, clues))
    return issues


def _box_issues(
    spec: GridSpec, clues: tuple[tuple[int, ...], ...]
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for index in range(spec.box_count):
        issues.extend(
            _issues_for_unit(UnitKind.BOX, index, spec.cells_in_box(index), clues)
        )
    return issues


def _issues_for_unit(
    kind: UnitKind,
    index: int,
    cells: tuple[tuple[int, int], ...],
    clues: tuple[tuple[int, ...], ...],
) -> list[ValidationIssue]:
    seen: dict[int, list[tuple[int, int]]] = {}
    for row, col in cells:
        value = clues[row][col]
        if value == 0:
            continue
        seen.setdefault(value, []).append((row, col))
    issues: list[ValidationIssue] = []
    for value in sorted(seen):
        coordinates = seen[value]
        if len(coordinates) > 1:
            issues.append(
                ValidationIssue(
                    kind=kind,
                    index=index,
                    value=value,
                    cells=tuple(coordinates),
                )
            )
    return issues
