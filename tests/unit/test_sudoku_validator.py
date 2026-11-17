"""Shape and value checks that do not require a Puzzle object."""

import pytest

from app.core.sudoku_validator import UnitKind, validate_board


def test_unsupported_shape_is_a_dimension_error() -> None:
    board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    result = validate_board(board)

    assert not result.valid
    assert result.issues[0].kind is UnitKind.DIMENSION
    assert board == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]


def test_ragged_board_is_a_dimension_error() -> None:
    result = validate_board([[0, 0, 0, 0], [0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert result.issues[0].kind is UnitKind.DIMENSION


@pytest.mark.parametrize(
    "value",
    [5, -1, True, "1"],
)
def test_disallowed_cell_values_are_reported(value: object) -> None:
    board = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, value, 0, 0]]
    result = validate_board(board)

    assert not result.valid
    assert result.issues[0].kind is UnitKind.VALUE
    assert result.issues[0].cells == ((3, 1),)
    assert board[3][1] is value or board[3][1] == value
