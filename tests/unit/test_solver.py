"""Search outcomes for 4×4, 6×6, and 9×9 puzzles."""

import pytest

from app.core import GridSpec, Puzzle, SolveStatus, StepKind, solve, validate_puzzle
from tests.unit.boards import COMPLETED, puzzle_from, with_cells

NINE_CLUES = (
    (5, 3, 0, 0, 7, 0, 0, 0, 0),
    (6, 0, 0, 1, 9, 5, 0, 0, 0),
    (0, 9, 8, 0, 0, 0, 0, 6, 0),
    (8, 0, 0, 0, 6, 0, 0, 0, 3),
    (4, 0, 0, 8, 0, 3, 0, 0, 1),
    (7, 0, 0, 0, 2, 0, 0, 0, 6),
    (0, 6, 0, 0, 0, 0, 2, 8, 0),
    (0, 0, 0, 4, 1, 9, 0, 0, 5),
    (0, 0, 0, 0, 8, 0, 0, 7, 9),
)

TWO_SOLUTION_FOUR = (
    (1, 2, 3, 4),
    (3, 4, 0, 0),
    (2, 1, 4, 3),
    (4, 3, 0, 0),
)

BLOCKED = {
    4: (
        (0, 1, 2, 0),
        (3, 0, 0, 0),
        (4, 0, 0, 0),
        (0, 0, 0, 0),
    ),
    6: (
        (0, 1, 2, 3, 0, 0),
        (4, 0, 0, 0, 0, 0),
        (5, 0, 0, 0, 0, 0),
        (6, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
    ),
    9: (
        (0, 1, 2, 3, 4, 5, 6, 7, 0),
        (8, 0, 0, 0, 0, 0, 0, 0, 0),
        (9, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
    ),
}


def _blank_one(size: int) -> Puzzle:
    rows = COMPLETED[size]
    return with_cells(rows, {(0, size - 1): 0})


def _assert_solution_respects_clues(
    puzzle: Puzzle, solution: tuple[tuple[int, ...], ...]
) -> None:
    spec = puzzle.spec
    assert validate_puzzle(puzzle_from(solution)).valid
    for row in range(spec.size):
        assert sorted(solution[row]) == list(range(1, spec.size + 1))
        for col in range(spec.size):
            if puzzle.is_given(row, col):
                assert solution[row][col] == puzzle.clue(row, col)
            assert solution[row][col] != 0
    for col in range(spec.size):
        assert len({solution[row][col] for row in range(spec.size)}) == spec.size
    for box in range(spec.box_count):
        values = [solution[row][col] for row, col in spec.cells_in_box(box)]
        assert sorted(values) == list(range(1, spec.size + 1))


def _brute_force_count(rows: tuple[tuple[int, ...], ...], limit: int) -> int:
    """Independent counter used to check the 4×4 search, not the production solver."""
    spec = GridSpec.for_size(4)
    board = [list(row) for row in rows]
    empties = [
        (row, col)
        for row in range(spec.size)
        for col in range(spec.size)
        if board[row][col] == 0
    ]
    found = 0

    def legal(row: int, col: int, value: int) -> bool:
        if any(board[row][other] == value for other in range(spec.size)):
            return False
        if any(board[other][col] == value for other in range(spec.size)):
            return False
        origin_row, origin_col = spec.box_origin(row, col)
        for other_row in range(origin_row, origin_row + spec.box_rows):
            for other_col in range(origin_col, origin_col + spec.box_cols):
                if board[other_row][other_col] == value:
                    return False
        return True

    def walk(index: int) -> None:
        nonlocal found
        if found >= limit:
            return
        if index == len(empties):
            found += 1
            return
        row, col = empties[index]
        for value in range(1, spec.size + 1):
            if not legal(row, col, value):
                continue
            board[row][col] = value
            walk(index + 1)
            board[row][col] = 0

    walk(0)
    return found


@pytest.mark.parametrize("size", [4, 6, 9])
def test_completed_grid_is_the_only_solution(size: int) -> None:
    puzzle = puzzle_from(COMPLETED[size])
    clues = puzzle.clues
    result = solve(puzzle, solution_limit=5)

    assert result.status is SolveStatus.UNIQUE
    assert result.solution_count == 1
    assert result.solution == COMPLETED[size]
    assert result.statistics.assignments == 0
    assert puzzle.clues == clues
    assert puzzle.clues is clues


@pytest.mark.parametrize("size", [4, 6, 9])
def test_one_blank_has_one_solution_and_keeps_the_clues(size: int) -> None:
    puzzle = _blank_one(size)
    clues = puzzle.clues
    result = solve(puzzle, solution_limit=5)

    assert result.status is SolveStatus.UNIQUE
    assert result.solution_count == 1
    assert result.solution is not None
    _assert_solution_respects_clues(puzzle, result.solution)
    assert result.solution[0][size - 1] == COMPLETED[size][0][size - 1]
    assert puzzle.clues == clues
    assert puzzle.working_board() == [list(row) for row in clues]


@pytest.mark.parametrize("size", [4, 6, 9])
def test_duplicate_clues_are_invalid_and_are_not_searched(size: int) -> None:
    puzzle = with_cells(COMPLETED[size], {(0, 1): COMPLETED[size][0][0]})
    result = solve(puzzle)

    assert result.status is SolveStatus.INVALID
    assert result.solution is None
    assert result.solution_count == 0
    assert result.issues
    assert result.statistics.assignments == 0
    assert result.steps == ()


@pytest.mark.parametrize("size", [4, 6, 9])
def test_consistent_blocked_cell_is_unsolvable(size: int) -> None:
    puzzle = puzzle_from(BLOCKED[size])
    validation = validate_puzzle(puzzle)
    result = solve(puzzle)

    assert validation.valid
    assert result.status is SolveStatus.UNSOLVABLE
    assert result.solution is None
    assert result.solution_count == 0
    assert result.statistics.empty_domains >= 1
    assert result.statistics.assignments == 0


@pytest.mark.parametrize("size", [4, 6, 9])
def test_underconstrained_puzzle_does_not_return_a_chosen_grid(size: int) -> None:
    rows = tuple(tuple(0 for _ in range(size)) for _ in range(size))
    puzzle = puzzle_from(rows)
    clues = puzzle.clues
    result = solve(puzzle, solution_limit=2)

    assert result.status is SolveStatus.MULTIPLE
    assert result.solution is None
    assert result.solution_count == 2
    assert result.statistics.assignments > 0
    assert result.statistics.backtracks > 0
    assert result.statistics.mrv_selections > 0
    assert result.statistics.candidate_eliminations > 0
    assert puzzle.clues is clues


def test_solution_limit_caps_how_many_completions_are_counted() -> None:
    puzzle = puzzle_from(tuple(tuple(0 for _ in range(4)) for _ in range(4)))
    assert solve(puzzle, solution_limit=2).solution_count == 2
    assert solve(puzzle, solution_limit=4).solution_count == 4
    assert _brute_force_count(puzzle.clues, limit=4) == 4


def test_four_by_four_with_exactly_two_solutions_is_not_unique() -> None:
    puzzle = puzzle_from(TWO_SOLUTION_FOUR)
    result = solve(puzzle, solution_limit=10)

    assert _brute_force_count(TWO_SOLUTION_FOUR, limit=10) == 2
    assert result.status is SolveStatus.MULTIPLE
    assert result.solution_count == 2
    assert result.solution is None
    assert result.statistics.backtracks >= 1
    assert any(step.kind is StepKind.BACKTRACK for step in result.steps)
    assert any(step.kind is StepKind.ASSIGN for step in result.steps)


def test_limit_of_one_does_not_claim_a_unique_solution() -> None:
    puzzle = puzzle_from(tuple(tuple(0 for _ in range(4)) for _ in range(4)))
    result = solve(puzzle, solution_limit=1)

    assert result.status is SolveStatus.LIMIT_REACHED
    assert result.solution_count == 1
    assert result.solution is None


def test_limit_of_one_on_a_finished_grid_is_unique() -> None:
    result = solve(puzzle_from(COMPLETED[4]), solution_limit=1)
    assert result.status is SolveStatus.UNIQUE
    assert result.solution == COMPLETED[4]


def test_nine_by_nine_partial_puzzle_matches_its_only_completion() -> None:
    puzzle = puzzle_from(NINE_CLUES)
    result = solve(puzzle, solution_limit=2)

    assert result.status is SolveStatus.UNIQUE
    assert result.solution == COMPLETED[9]
    assert result.solution is not None
    _assert_solution_respects_clues(puzzle, result.solution)
    assert result.statistics.candidate_eliminations > 0


def test_forward_checking_rejects_a_value_that_empties_a_peer() -> None:
    """Two peer cells can both be left with the same single digit.

    The clues do not repeat a digit, so the puzzle is not invalid. Placing
    that digit in the first cell leaves the other with no candidates.
    """
    rows = (
        (0, 0, 2, 3),
        (4, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
    )
    puzzle = puzzle_from(rows)
    result = solve(puzzle)

    assert validate_puzzle(puzzle).valid
    assert result.status is SolveStatus.UNSOLVABLE
    assert result.solution is None
    assert result.statistics.forward_check_rejections > 0
    assert any(step.kind is StepKind.REJECT for step in result.steps)


@pytest.mark.parametrize("limit", [0, -1])
def test_solution_limit_must_be_positive(limit: int) -> None:
    puzzle = puzzle_from(COMPLETED[4])
    with pytest.raises(ValueError, match="at least 1"):
        solve(puzzle, solution_limit=limit)


def test_solution_limit_rejects_a_boolean() -> None:
    puzzle = puzzle_from(COMPLETED[4])
    with pytest.raises(TypeError, match="int"):
        solve(puzzle, solution_limit=True)  # type: ignore[arg-type]
