"""Validator, candidate generator, and solver used together."""

import copy

import pytest

from app.core.candidate_generator import candidate_grid, cell_candidates
from app.core.grid import GridSpec
from app.core.puzzle import Puzzle
from app.core.sudoku_solver import SolveStatus, solve
from app.core.sudoku_validator import UnitKind, validate_board
from tests.unit.boards import COMPLETED, with_cells


def _assert_complete(spec: GridSpec, solution: tuple[tuple[int, ...], ...]) -> None:
    for row in solution:
        assert sorted(row) == list(range(1, spec.size + 1))
    for col in range(spec.size):
        assert len({solution[row][col] for row in range(spec.size)}) == spec.size
    for box in range(spec.box_count):
        values = [solution[row][col] for row, col in spec.cells_in_box(box)]
        assert sorted(values) == list(range(1, spec.size + 1))


@pytest.mark.parametrize("size", [4, 6, 9])
def test_valid_puzzle_has_one_correct_solution(size: int) -> None:
    puzzle = with_cells(COMPLETED[size], {(0, 1): 0})
    original = puzzle.clues
    result = solve(puzzle, solution_limit=2)

    assert result.status is SolveStatus.UNIQUE
    assert result.solution is not None
    assert result.solution_count == 1
    assert result.solution[0][1] == COMPLETED[size][0][1]
    _assert_complete(puzzle.spec, result.solution)
    assert puzzle.clues is original


def test_invalid_dimensions_stop_before_solving() -> None:
    board = [[0, 0], [0, 0]]
    result = validate_board(board)
    assert result.issues[0].kind is UnitKind.DIMENSION
    assert board == [[0, 0], [0, 0]]


def test_invalid_values_stop_before_solving() -> None:
    board = [[0, 0, 0, 9], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    result = validate_board(board)
    assert any(issue.kind is UnitKind.VALUE for issue in result.issues)


@pytest.mark.parametrize("size", [4, 6, 9])
def test_duplicate_row_column_and_box_are_invalid(size: int) -> None:
    spec = GridSpec.for_size(size)
    value = COMPLETED[size][0][0]
    row_board = [list(row) for row in COMPLETED[size]]
    row_board[0][size - 1] = value
    column_board = [list(row) for row in COMPLETED[size]]
    column_board[size - 1][0] = value
    box_board = [list(row) for row in COMPLETED[size]]
    box_board[1][1] = value

    row_result = validate_board(row_board, spec=spec)
    column_result = validate_board(column_board, spec=spec)
    box_result = validate_board(box_board, spec=spec)

    assert any(issue.kind is UnitKind.ROW for issue in row_result.issues)
    assert any(issue.kind is UnitKind.COLUMN for issue in column_result.issues)
    box_hits = [
        issue
        for issue in box_result.issues
        if issue.kind is UnitKind.BOX and issue.index == 0
    ]
    assert box_hits


def test_unsolvable_puzzle_is_distinct_from_invalid() -> None:
    rows = (
        (0, 1, 2, 0),
        (3, 0, 0, 0),
        (4, 0, 0, 0),
        (0, 0, 0, 0),
    )
    validation = validate_board(rows)
    result = solve(Puzzle.from_rows(GridSpec.for_size(4), rows))

    assert validation.valid
    assert result.status is SolveStatus.UNSOLVABLE
    assert result.solution is None
    assert result.solution_count == 0


def test_multiple_solutions_do_not_select_a_grid() -> None:
    rows = (
        (1, 2, 3, 4),
        (3, 4, 0, 0),
        (2, 1, 4, 3),
        (4, 3, 0, 0),
    )
    supplied = [list(row) for row in rows]
    snapshot = copy.deepcopy(supplied)
    puzzle = Puzzle.from_rows(GridSpec.for_size(4), supplied)
    result = solve(puzzle, solution_limit=10)

    assert result.status is SolveStatus.MULTIPLE
    assert result.solution is None
    assert result.solution_count == 2
    assert supplied == snapshot
    assert puzzle.clues == rows


def test_candidate_generation_uses_the_configured_box() -> None:
    spec = GridSpec.for_size(6)
    board = [[0 for _ in range(6)] for _ in range(6)]
    board[1][2] = 5
    before = copy.deepcopy(board)
    candidates = cell_candidates(spec, board, 0, 0)
    grid = candidate_grid(spec, board)

    assert spec.box_rows == 2 and spec.box_cols == 3
    assert 5 not in candidates
    assert grid[0][0] == candidates
    assert board == before


def test_solver_modules_do_not_import_vision_or_ui_libraries() -> None:
    import app.core.candidate_generator as candidates
    import app.core.sudoku_solver as solver
    import app.core.sudoku_validator as validator

    for module in (validator, candidates, solver):
        imported = set(module.__dict__)
        assert "cv2" not in imported
        assert "torch" not in imported
        assert "streamlit" not in imported
