"""Row, column, and box checks for a puzzle's given clues.

A puzzle is invalid when a digit is repeated inside one of those units.
Consistent clues can still be unsolvable; that decision belongs to the solver.
"""

from dataclasses import dataclass
from enum import Enum

from app.core.grid import GridSpec
from app.core.puzzle import Puzzle


class UnitKind(Enum):
    """A Sudoku constraint unit."""

    ROW = "row"
    COLUMN = "column"
    BOX = "box"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One repeated digit inside a single row, column, or box."""

    kind: UnitKind
    index: int
    value: int
    cells: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Every duplicate found in the given clues."""

    issues: tuple[ValidationIssue, ...]

    @property
    def valid(self) -> bool:
        """True when the clues contain no repeated digit in a unit."""
        return not self.issues


def validate_puzzle(puzzle: Puzzle) -> ValidationResult:
    """Check the original clues. Empty cells (0) are ignored."""
    spec = puzzle.spec
    issues: list[ValidationIssue] = []
    issues.extend(_row_issues(spec, puzzle.clues))
    issues.extend(_column_issues(spec, puzzle.clues))
    issues.extend(_box_issues(spec, puzzle.clues))
    return ValidationResult(issues=tuple(issues))


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
