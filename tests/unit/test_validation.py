"""Clue validation for every supported grid size."""

import pytest

from app.core import GridSpec, UnitKind, validate_puzzle
from tests.unit.boards import COMPLETED, puzzle_from, with_cells


@pytest.mark.parametrize("size", [4, 6, 9])
def test_completed_and_empty_grids_are_valid(size: int) -> None:
    completed = COMPLETED[size]
    empty = tuple(tuple(0 for _ in range(size)) for _ in range(size))
    assert validate_puzzle(puzzle_from(completed)).valid
    assert validate_puzzle(puzzle_from(empty)).valid


@pytest.mark.parametrize("size", [4, 6, 9])
def test_duplicate_in_a_row_is_invalid(size: int) -> None:
    value = COMPLETED[size][0][0]
    result = validate_puzzle(with_cells(COMPLETED[size], {(0, size - 1): value}))
    assert not result.valid
    row_issues = [issue for issue in result.issues if issue.kind is UnitKind.ROW]
    assert row_issues[0].index == 0
    assert row_issues[0].value == value
    assert (0, 0) in row_issues[0].cells
    assert (0, size - 1) in row_issues[0].cells


@pytest.mark.parametrize("size", [4, 6, 9])
def test_duplicate_in_a_column_is_invalid(size: int) -> None:
    value = COMPLETED[size][0][0]
    result = validate_puzzle(with_cells(COMPLETED[size], {(size - 1, 0): value}))
    assert not result.valid
    column_issues = [issue for issue in result.issues if issue.kind is UnitKind.COLUMN]
    assert column_issues[0].index == 0
    assert (0, 0) in column_issues[0].cells
    assert (size - 1, 0) in column_issues[0].cells


@pytest.mark.parametrize("size", [4, 6, 9])
def test_duplicate_inside_a_box_is_invalid(size: int) -> None:
    """Repeat a digit that shares a box but not a row or a column."""
    spec = GridSpec.for_size(size)
    origin_row, origin_col = spec.box_origin(0, 0)
    other = (origin_row + 1, origin_col + 1)
    value = COMPLETED[size][0][0]
    result = validate_puzzle(with_cells(COMPLETED[size], {other: value}))
    assert (other[0], other[1]) != (0, 0)
    assert not result.valid
    box_issues = [issue for issue in result.issues if issue.kind is UnitKind.BOX]
    assert box_issues[0].index == 0
    assert box_issues[0].value == value
    assert (0, 0) in box_issues[0].cells
    assert other in box_issues[0].cells


def test_validation_reports_every_broken_unit() -> None:
    rows = (
        (1, 1, 0, 0),
        (0, 0, 0, 0),
        (1, 0, 0, 0),
        (0, 0, 0, 0),
    )
    result = validate_puzzle(puzzle_from(rows))
    kinds = {(issue.kind, issue.index) for issue in result.issues}
    assert (UnitKind.ROW, 0) in kinds
    assert (UnitKind.COLUMN, 0) in kinds
    assert (UnitKind.BOX, 0) in kinds
