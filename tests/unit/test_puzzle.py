"""The original clue grid stays unchanged."""

import pytest

from app.core import Cell, GridSpec, Puzzle


def _empty(size: int) -> list[list[int]]:
    return [[0 for _ in range(size)] for _ in range(size)]


def test_clues_are_stored_as_an_immutable_grid() -> None:
    spec = GridSpec.for_size(4)
    rows = _empty(4)
    rows[0][0] = 1
    rows[3][2] = 4
    puzzle = Puzzle.from_rows(spec, rows)

    assert puzzle.spec is spec
    assert puzzle.clues[0][0] == 1
    assert puzzle.clue(3, 2) == 4
    assert puzzle.is_given(0, 0) is True
    assert puzzle.is_given(0, 1) is False
    assert puzzle.cell(3, 2) == Cell(row=3, col=2, value=4)
    with pytest.raises(TypeError):
        puzzle.clues[0][0] = 2  # type: ignore[index]


def test_mutating_the_input_does_not_change_clues() -> None:
    rows = _empty(4)
    rows[1][1] = 3
    puzzle = Puzzle.from_rows(GridSpec.for_size(4), rows)

    rows[1][1] = 9
    rows.append([1, 1, 1, 1])

    assert puzzle.clue(1, 1) == 3
    assert len(puzzle.clues) == 4


def test_working_board_is_a_fresh_copy() -> None:
    rows = _empty(9)
    rows[0][0] = 5
    puzzle = Puzzle.from_rows(GridSpec.for_size(9), rows)

    first = puzzle.working_board()
    second = puzzle.working_board()
    first[0][0] = 9
    first[8][8] = 1
    second[0][1] = 2

    assert puzzle.clue(0, 0) == 5
    assert puzzle.clue(8, 8) == 0
    assert puzzle.clue(0, 1) == 0
    assert second[0][0] == 5
    assert first is not second


def test_equal_puzzles_compare_by_spec_and_clues() -> None:
    spec = GridSpec.for_size(4)
    rows = _empty(4)
    rows[0][1] = 2
    left = Puzzle.from_rows(spec, rows)
    right = Puzzle.from_rows(GridSpec.for_size(4), [list(row) for row in rows])

    assert left == right
    assert hash(left) == hash(right)
    assert left != Puzzle.from_rows(spec, _empty(4))


@pytest.mark.parametrize("size", [4, 6, 9])
def test_each_supported_size_accepts_an_empty_grid(size: int) -> None:
    puzzle = Puzzle.from_rows(GridSpec.for_size(size), _empty(size))
    assert puzzle.spec.size == size
    assert all(value == 0 for row in puzzle.clues for value in row)


def test_wrong_shape_is_rejected() -> None:
    spec = GridSpec.for_size(4)
    with pytest.raises(ValueError, match="expected 4 rows"):
        Puzzle.from_rows(spec, _empty(4)[:3])
    short_row = _empty(4)
    short_row[2] = [0, 0, 0]
    with pytest.raises(ValueError, match="row 2"):
        Puzzle.from_rows(spec, short_row)


def test_symbols_outside_the_digit_range_are_rejected() -> None:
    spec = GridSpec.for_size(6)
    rows = _empty(6)
    rows[2][4] = 7
    with pytest.raises(ValueError, match="outside 0..6"):
        Puzzle.from_rows(spec, rows)

    rows[2][4] = -1
    with pytest.raises(ValueError, match="outside 0..6"):
        Puzzle.from_rows(spec, rows)


def test_non_integer_symbols_are_rejected() -> None:
    spec = GridSpec.for_size(4)
    rows: list[list[object]] = _empty(4)
    rows[0][0] = True
    with pytest.raises(TypeError, match="int"):
        Puzzle.from_rows(spec, rows)  # type: ignore[arg-type]

    rows[0][0] = "1"
    with pytest.raises(TypeError, match="int"):
        Puzzle.from_rows(spec, rows)  # type: ignore[arg-type]


def test_coordinates_outside_the_puzzle_are_rejected() -> None:
    puzzle = Puzzle.from_rows(GridSpec.for_size(4), _empty(4))
    with pytest.raises(ValueError, match="outside"):
        puzzle.clue(-1, 0)
