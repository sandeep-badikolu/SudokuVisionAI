"""Review a recognized puzzle without calling the image pipeline again.

OCR values stay on ``ocr_grid``. Corrections are stored separately. An
uncertain cell remains ``None`` until a person enters a digit or a blank.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.core.grid import GridSpec
from app.core.puzzle import Puzzle
from app.core.sudoku_solver import SolveStatus, solve
from app.core.sudoku_validator import UnitKind, ValidationIssue, validate_board
from app.pipeline import run_image
from app.vision.errors import ImageInputError
from app.vision.image_loader import LoadedImage, load_image

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
_SUFFIXES = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg"}


class UploadError(ValueError):
    """The uploaded file is not a usable PNG or JPEG."""


class ProcessingError(RuntimeError):
    """The existing pipeline could not build a puzzle from the image."""


class CorrectionError(ValueError):
    """A manual edit is outside the grid or the allowed digits."""


@dataclass(frozen=True, slots=True)
class CellMeta:
    """Recognition confidence for one cell. It is not a corrected value."""

    status: str
    confidence: float | None
    blank_confidence: float | None


@dataclass(frozen=True, slots=True)
class DisplayCell:
    """One cell prepared for the review grid."""

    row: int
    col: int
    value: int | None
    ocr_value: int | None
    source: str
    uncertain: bool
    confidence: float | None
    blank_confidence: float | None


def confidence_caption(cell: DisplayCell) -> str:
    """Describe digit and occupancy confidence without turning either into a digit.

    A missing digit confidence means the matcher did not accept a digit.
    The cell is not treated as blank, and the missing score is not a puzzle error.
    """
    if cell.uncertain:
        state = "Uncertain recognition. The cell stays unresolved."
    elif cell.value == 0:
        state = "Blank."
    else:
        state = f"Recognized digit {cell.value}."
    if cell.confidence is None:
        digit = "Digit confidence is not available."
    else:
        digit = f"Digit confidence {cell.confidence:.2f}."
    if cell.blank_confidence is None:
        occupancy = "Occupancy confidence is not available."
    else:
        occupancy = f"Occupancy confidence {cell.blank_confidence:.2f}."
    return f"{state} {digit} {occupancy}"


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Validation of the corrected grid. This does not solve the puzzle."""

    status: str
    errors: tuple[str, ...]
    unresolved: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class SolveView:
    """Solver outcome for a corrected grid that was eligible to be solved."""

    status: str
    solution: tuple[tuple[int, ...], ...] | None
    message: str


@dataclass(frozen=True, slots=True)
class SolutionCell:
    """A solved cell, marked as a clue or a value the solver filled."""

    row: int
    col: int
    value: int
    role: str


@dataclass
class PuzzleReview:
    """OCR reconstruction plus edits made in the review screen."""

    filename: str
    order: int
    ocr_grid: tuple[tuple[int | None, ...], ...]
    metadata: tuple[tuple[CellMeta, ...], ...]
    images: dict[str, np.ndarray] = field(default_factory=dict)
    corrections: dict[tuple[int, int], int] = field(default_factory=dict)
    check: CheckResult | None = None
    solve_result: SolveView | None = None

    def display_cells(self) -> tuple[DisplayCell, ...]:
        """Return every cell, leaving uncertain OCR values as ``None``."""
        cells: list[DisplayCell] = []
        for row in range(self.order):
            for col in range(self.order):
                ocr_value = self.ocr_grid[row][col]
                source = "ocr"
                value = ocr_value
                if (row, col) in self.corrections:
                    source = "correction"
                    value = self.corrections[(row, col)]
                meta = self.metadata[row][col]
                cells.append(
                    DisplayCell(
                        row=row,
                        col=col,
                        value=value,
                        ocr_value=ocr_value,
                        source=source,
                        uncertain=value is None,
                        confidence=meta.confidence,
                        blank_confidence=meta.blank_confidence,
                    )
                )
        return tuple(cells)

    def set_cell(self, row: int, col: int, digit: int) -> None:
        """Store a digit or a confirmed blank. ``0`` means blank."""
        self._require_cell(row, col)
        if isinstance(digit, bool) or not isinstance(digit, int):
            raise CorrectionError("a cell value must be an integer")
        if digit < 0 or digit > self.order:
            raise CorrectionError(
                f"a {self.order}×{self.order} cell accepts 0 through {self.order}"
            )
        if digit == self.ocr_grid[row][col]:
            self.corrections.pop((row, col), None)
        else:
            self.corrections[(row, col)] = digit
        self.check = None
        self.solve_result = None

    def clear_cell(self, row: int, col: int) -> None:
        """Drop the correction at one cell and restore its OCR value."""
        self._require_cell(row, col)
        self.corrections.pop((row, col), None)
        self.check = None
        self.solve_result = None

    def reset(self) -> None:
        """Drop every correction and any validation or solution."""
        self.corrections.clear()
        self.check = None
        self.solve_result = None

    def validate(self) -> CheckResult:
        """Check the corrected grid. Uncertain cells are not treated as blank."""
        unresolved = tuple(
            (cell.row, cell.col) for cell in self.display_cells() if cell.uncertain
        )
        if unresolved:
            result = CheckResult(
                status="needs_correction",
                errors=tuple(
                    f"Row {row + 1}, column {col + 1} is uncertain."
                    for row, col in unresolved
                ),
                unresolved=unresolved,
            )
            self.check = result
            return result
        board = self.resolved_board()
        issues = validate_board(board, spec=GridSpec.for_size(self.order)).issues
        result = CheckResult(
            status="valid" if not issues else "invalid",
            errors=tuple(_format_issue(issue) for issue in issues),
            unresolved=(),
        )
        self.check = result
        return result

    def solve_puzzle(self) -> SolveView:
        """Solve only when the corrected grid is complete and valid."""
        check = self.validate()
        if check.status == "needs_correction":
            view = SolveView(
                status="not_attempted",
                solution=None,
                message=(
                    "Uncertain cells are still unresolved, "
                    "so the puzzle was not solved."
                ),
            )
            self.solve_result = view
            return view
        if check.status != "valid":
            view = SolveView(
                status="not_attempted",
                solution=None,
                message="The puzzle has validation errors, so it was not solved.",
            )
            self.solve_result = view
            return view
        outcome = solve(
            Puzzle.from_rows(GridSpec.for_size(self.order), self.resolved_board())
        )
        solution = outcome.solution if outcome.status is SolveStatus.UNIQUE else None
        view = SolveView(
            status=outcome.status.value,
            solution=solution,
            message=_solve_message(outcome.status.value),
        )
        self.solve_result = view
        return view

    def resolved_board(self) -> tuple[tuple[int, ...], ...]:
        """Return the corrected grid. Every cell must already be resolved."""
        rows: list[tuple[int, ...]] = []
        for row in range(self.order):
            values: list[int] = []
            for col in range(self.order):
                value = self._value(row, col)
                if value is None:
                    raise CorrectionError("the grid still has an uncertain cell")
                values.append(value)
            rows.append(tuple(values))
        return tuple(rows)

    def solution_cells(self) -> tuple[SolutionCell, ...] | None:
        """Mark solver-filled cells separately from the clues that were given."""
        result = self.solve_result
        if result is None or result.solution is None:
            return None
        cells: list[SolutionCell] = []
        for row in range(self.order):
            for col in range(self.order):
                given = self._value(row, col)
                role = "generated" if given in (0, None) else "clue"
                cells.append(
                    SolutionCell(
                        row=row,
                        col=col,
                        value=result.solution[row][col],
                        role=role,
                    )
                )
        return tuple(cells)

    def _value(self, row: int, col: int) -> int | None:
        if (row, col) in self.corrections:
            return self.corrections[(row, col)]
        return self.ocr_grid[row][col]

    def _require_cell(self, row: int, col: int) -> None:
        if not GridSpec.for_size(self.order).contains(row, col):
            raise CorrectionError(
                f"cell ({row}, {col}) is outside a {self.order}×{self.order} grid"
            )


def inspect_upload(
    filename: str,
    payload: bytes,
    *,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> LoadedImage:
    """Check the file name, size, and image bytes. The source file is not written."""
    name = Path(filename).name
    suffix = Path(name).suffix.lower()
    if suffix not in _SUFFIXES:
        raise UploadError("upload a PNG, JPG, or JPEG file")
    if not payload:
        raise UploadError("the uploaded file is empty")
    if len(payload) > max_bytes:
        raise UploadError("the uploaded file is larger than 10 MB")
    with tempfile.TemporaryDirectory(prefix="sudokuvision-upload-") as directory:
        path = Path(directory) / f"upload{suffix}"
        path.write_bytes(payload)
        try:
            return load_image(path)
        except ImageInputError as exc:
            raise UploadError(str(exc)) from exc


def process_upload(
    filename: str,
    payload: bytes,
    *,
    order: int | None = None,
) -> PuzzleReview:
    """Run the existing pipeline in a temporary directory and return a review."""
    inspect_upload(filename, payload)
    suffix = Path(filename).suffix.lower()
    with tempfile.TemporaryDirectory(prefix="sudokuvision-run-") as directory:
        root = Path(directory)
        image_path = root / f"upload{suffix}"
        image_path.write_bytes(payload)
        record = run_image(image_path, root / "out", order=order)
        if not record["success"]:
            reason = record["failure_reason"] or "the image could not be processed"
            raise ProcessingError(str(reason))
        destination = root / "out"
        report = json.loads(
            (destination / "reconstruction.json").read_text(encoding="utf-8")
        )
        images = {
            "original": _read_rgb(destination / f"original{suffix}"),
            "detected_grid": _read_rgb(destination / "detected_grid.png"),
            "perspective_grid": _read_rgb(destination / "perspective_grid.png"),
            "contact_sheet": _read_rgb(destination / "contact_sheet.png"),
        }
    return review_from_report(report, filename=Path(filename).name, images=images)


def review_from_report(
    report: dict[str, object],
    *,
    filename: str,
    images: dict[str, np.ndarray] | None = None,
) -> PuzzleReview:
    """Build a review from a reconstruction report. Corrections start empty."""
    order = int(report["puzzle_order"])  # type: ignore[arg-type]
    raw_grid = report["grid"]
    if not isinstance(raw_grid, list) or len(raw_grid) != order:
        raise ProcessingError("reconstruction did not contain a complete grid")
    ocr_rows: list[tuple[int | None, ...]] = []
    for row in raw_grid:
        if not isinstance(row, list) or len(row) != order:
            raise ProcessingError("reconstruction did not contain a complete grid")
        ocr_rows.append(tuple(_cell_value(value) for value in row))
    metadata = _metadata(report, order)
    return PuzzleReview(
        filename=filename,
        order=order,
        ocr_grid=tuple(ocr_rows),
        metadata=metadata,
        images={} if images is None else images,
    )


def _metadata(
    report: dict[str, object], order: int
) -> tuple[tuple[CellMeta, ...], ...]:
    found: dict[tuple[int, int], CellMeta] = {}
    raw = report.get("recognition_metadata")
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            key = (int(item["row"]), int(item["col"]))
            found[key] = CellMeta(
                status=str(item.get("status") or ""),
                confidence=_optional_float(item.get("confidence")),
                blank_confidence=_optional_float(item.get("blank_confidence")),
            )
    rows: list[tuple[CellMeta, ...]] = []
    for row in range(order):
        rows.append(
            tuple(
                found.get((row, col), CellMeta("", None, None))
                for col in range(order)
            )
        )
    return tuple(rows)


def _cell_value(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProcessingError("reconstruction contained a value that is not a digit")
    return value


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _format_issue(issue: ValidationIssue) -> str:
    number = issue.index + 1
    if issue.kind is UnitKind.ROW:
        return f"Row {number} repeats {issue.value}."
    if issue.kind is UnitKind.COLUMN:
        return f"Column {number} repeats {issue.value}."
    if issue.kind is UnitKind.BOX:
        return f"Box {number} repeats {issue.value}."
    if issue.detail:
        return issue.detail
    return issue.kind.value


def _solve_message(status: str) -> str:
    messages = {
        "unique": "One solution was found.",
        "unsolvable": "The puzzle is valid and has no solution.",
        "multiple": "The puzzle has more than one solution. No single grid is shown.",
        "invalid": "The puzzle is invalid, so it was not solved.",
        "limit_reached": "The search stopped before it could decide.",
    }
    return messages.get(status, "The puzzle was not solved.")


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ProcessingError(f"could not read {path.name}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
