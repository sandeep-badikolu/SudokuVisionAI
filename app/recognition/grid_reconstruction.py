"""Turn recognized cells into a Sudoku grid and, when it is complete, solve it.

Uncertain cells stay unresolved. They are not written as 0. The solver runs
only after every cell is a confirmed blank or a confirmed digit and the
existing row, column, and box checks pass.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from app.core.grid import GridSpec
from app.core.puzzle import Puzzle
from app.core.sudoku_solver import solve
from app.core.sudoku_validator import UnitKind, validate_board
from app.recognition.digit_recognizer import (
    RecognitionResult,
    RecognitionStatus,
    recognize_cells,
)
from app.vision.grid_detector import extract_cells

_NOT_ATTEMPTED = "not_attempted"
_VALID = "valid"
_INVALID = "invalid"
_NEEDS_CORRECTION = "needs_correction"
_SOURCE = "recognizer"


@dataclass(frozen=True, slots=True)
class CellRecord:
    """Provenance for one reconstructed cell."""

    row: int
    col: int
    value: int | None
    source: str
    status: str
    confidence: float | None
    blank_confidence: float | None

    def to_report(self) -> dict[str, object]:
        """JSON fields for this cell."""
        return {
            "row": self.row,
            "col": self.col,
            "value": self.value,
            "source": self.source,
            "status": self.status,
            "confidence": _round(self.confidence),
            "blank_confidence": _round(self.blank_confidence),
        }


@dataclass(frozen=True, slots=True)
class Reconstruction:
    """A recognized puzzle, its validation, and a solution only when one exists.

    ``grid`` and ``original_clues`` are the recognized clues. ``0`` is a
    confirmed empty cell. ``None`` is unresolved and is never a blank.
    ``solution`` is the solver's completion and is present only when the
    search proves a unique solution.
    """

    puzzle_order: int
    grid: tuple[tuple[int | None, ...], ...]
    original_clues: tuple[tuple[int | None, ...], ...]
    unresolved_cells: tuple[CellRecord, ...]
    validation_status: str
    validation_errors: tuple[dict[str, object], ...]
    recognition_metadata: tuple[CellRecord, ...]
    solution_status: str
    solution: tuple[tuple[int, ...], ...] | None

    def to_report(self) -> dict[str, object]:
        """JSON-ready reconstruction. Unresolved cells are null, not 0."""
        return {
            "puzzle_order": self.puzzle_order,
            "grid": _matrix(self.grid),
            "original_clues": _matrix(self.original_clues),
            "unresolved_cells": [cell.to_report() for cell in self.unresolved_cells],
            "validation_status": self.validation_status,
            "validation_errors": [dict(error) for error in self.validation_errors],
            "recognition_metadata": [
                cell.to_report() for cell in self.recognition_metadata
            ],
            "solution_status": self.solution_status,
            "solution": None if self.solution is None else _matrix(self.solution),
        }


def reconstruct(
    order: int, cells: Sequence[RecognitionResult]
) -> Reconstruction:
    """Build a grid from recognition results for a 4×4, 6×6, or 9×9 puzzle."""
    spec = GridSpec.for_size(order)
    placed, metadata, structure_errors = _place(spec, cells)
    if structure_errors:
        return _result(
            spec,
            placed,
            metadata,
            _INVALID,
            tuple(structure_errors),
            _NOT_ATTEMPTED,
            None,
        )
    unresolved_at = {
        (cell.row, cell.col)
        for cell in metadata
        if cell.status == RecognitionStatus.UNCERTAIN
    }
    confirmed = _confirmed_board(placed, unresolved_at)
    issues = validate_board(confirmed, spec=spec).issues
    errors = tuple(_issue_report(issue) for issue in issues)
    unresolved = tuple(cell for cell in metadata if cell.value is None)
    if errors:
        return _result(spec, placed, metadata, _INVALID, errors, _NOT_ATTEMPTED, None)
    if unresolved:
        return _result(
            spec,
            placed,
            metadata,
            _NEEDS_CORRECTION,
            (),
            _NOT_ATTEMPTED,
            None,
        )
    puzzle = Puzzle.from_rows(spec, confirmed)
    clues_before = puzzle.clues
    outcome = solve(puzzle)
    if puzzle.clues != clues_before:
        raise RuntimeError("solver changed the recognized clues")
    solution = outcome.solution
    return _result(
        spec,
        placed,
        metadata,
        _VALID,
        (),
        outcome.status.value,
        solution,
    )


def reconstruct_image(image: object, *, order: int | None = None) -> Reconstruction:
    """Run detection, recognition, and reconstruction on one image.

    ``image`` is a BGR array already loaded by the vision pipeline. Digit
    recognition is the existing template matcher.
    """
    detection, patches = extract_cells(image, order=order)  # type: ignore[arg-type]
    recognized = recognize_cells(patches, order=detection.order)
    return reconstruct(detection.order, recognized)


def _place(
    spec: GridSpec, cells: Sequence[RecognitionResult]
) -> tuple[
    list[list[int | None]],
    tuple[CellRecord, ...],
    list[dict[str, object]],
]:
    grid = [[None for _ in range(spec.size)] for _ in range(spec.size)]
    seen: dict[tuple[int, int], int] = {}
    metadata: list[CellRecord] = []
    errors: list[dict[str, object]] = []
    for result in cells:
        record = _record(spec, result)
        metadata.append(record)
        key = (result.row, result.col)
        if not spec.contains(result.row, result.col):
            errors.append(
                _error(
                    "dimension",
                    result.row,
                    result.col,
                    record.value,
                    ((result.row, result.col),),
                    f"cell ({result.row}, {result.col}) is outside the grid",
                )
            )
            continue
        if key in seen:
            errors.append(
                _error(
                    "dimension",
                    result.row,
                    result.col,
                    record.value if record.value is not None else 0,
                    ((result.row, result.col),),
                    f"cell ({result.row}, {result.col}) was recognized more than once",
                )
            )
            grid[result.row][result.col] = None
            continue
        seen[key] = 1
        grid[result.row][result.col] = record.value
    expected = spec.cell_count
    if len(cells) != expected or len(seen) != expected:
        errors.append(
            _error(
                "dimension",
                None,
                None,
                0,
                (),
                f"expected {expected} cells, received {len(cells)}",
            )
        )
    return grid, tuple(metadata), errors


def _record(spec: GridSpec, result: RecognitionResult) -> CellRecord:
    value, status = _interpret(spec, result)
    return CellRecord(
        row=result.row,
        col=result.col,
        value=value,
        source=_SOURCE,
        status=status,
        confidence=result.confidence,
        blank_confidence=result.blank_confidence,
    )


def _interpret(spec: GridSpec, result: RecognitionResult) -> tuple[int | None, str]:
    """Map a recognition result to a clue value without filling uncertain cells."""
    if result.status == RecognitionStatus.UNCERTAIN:
        return None, RecognitionStatus.UNCERTAIN
    if result.status == RecognitionStatus.BLANK:
        if result.digit == 0:
            return 0, RecognitionStatus.BLANK
        return result.digit if isinstance(result.digit, int) else None, result.status
    if result.status == RecognitionStatus.RECOGNIZED and isinstance(result.digit, int):
        if isinstance(result.digit, bool) or not 1 <= result.digit <= spec.max_digit:
            return result.digit, result.status
        return result.digit, RecognitionStatus.RECOGNIZED
    return None, result.status


def _confirmed_board(
    grid: list[list[int | None]], unresolved: set[tuple[int, int]]
) -> tuple[tuple[int, ...], ...]:
    """Copy confirmed clues for the existing validator.

    Unresolved cells are passed as 0 so they are not duplicate clues. That
    copy is not the reconstructed grid. A missing recognized digit is passed
    as -1 so it fails the value check instead of becoming a blank.
    """
    rows: list[tuple[int, ...]] = []
    for row_index, row in enumerate(grid):
        values: list[int] = []
        for col_index, value in enumerate(row):
            if (row_index, col_index) in unresolved:
                values.append(0)
            elif value is None:
                values.append(-1)
            else:
                values.append(value)
        rows.append(tuple(values))
    return tuple(rows)


def _result(
    spec: GridSpec,
    grid: list[list[int | None]],
    metadata: tuple[CellRecord, ...],
    status: str,
    errors: tuple[dict[str, object], ...],
    solution_status: str,
    solution: tuple[tuple[int, ...], ...] | None,
) -> Reconstruction:
    frozen = tuple(tuple(row) for row in grid)
    unresolved = tuple(
        cell
        for cell in metadata
        if cell.value is None and cell.status == RecognitionStatus.UNCERTAIN
    )
    return Reconstruction(
        puzzle_order=spec.size,
        grid=frozen,
        original_clues=frozen,
        unresolved_cells=unresolved,
        validation_status=status,
        validation_errors=errors,
        recognition_metadata=metadata,
        solution_status=solution_status,
        solution=None if solution is None else tuple(tuple(row) for row in solution),
    )


def _issue_report(issue: object) -> dict[str, object]:
    kind = issue.kind
    cells = tuple(issue.cells)
    row, column = _anchor(kind, issue.index, cells)
    return _error(
        kind.value,
        row,
        column,
        issue.value,
        cells,
        issue.detail or _duplicate_detail(kind.value, issue.value, cells),
    )


def _anchor(
    kind: UnitKind, index: int, cells: tuple[tuple[int, int], ...]
) -> tuple[int | None, int | None]:
    if kind is UnitKind.VALUE and cells:
        return cells[0]
    if kind is UnitKind.ROW:
        return index, None
    if kind is UnitKind.COLUMN:
        return None, index
    return None, None


def _duplicate_detail(
    kind: str, value: int, cells: tuple[tuple[int, int], ...]
) -> str:
    places = ", ".join(f"({row}, {col})" for row, col in cells)
    return f"value {value} is repeated in {kind} at {places}"


def _error(
    kind: str,
    row: int | None,
    column: int | None,
    value: int | None,
    cells: tuple[tuple[int, int], ...],
    detail: str,
) -> dict[str, object]:
    return {
        "kind": kind,
        "row": row,
        "column": column,
        "value": value,
        "cells": [list(cell) for cell in cells],
        "detail": detail,
    }


def _matrix(
    rows: tuple[tuple[int | None, ...], ...],
) -> list[list[int | None]]:
    return [list(row) for row in rows]


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)
