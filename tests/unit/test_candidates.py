"""Candidate sets follow row, column, and box geometry."""

import pytest

from app.core import GridSpec, cell_candidates, peer_coordinates
from tests.unit.boards import COMPLETED


@pytest.mark.parametrize("size", [4, 6, 9])
def test_empty_grid_offers_every_digit(size: int) -> None:
    spec = GridSpec.for_size(size)
    board = [[0 for _ in range(size)] for _ in range(size)]
    assert cell_candidates(spec, board, 0, 0) == frozenset(range(1, size + 1))


@pytest.mark.parametrize("size", [4, 6, 9])
def test_filled_cell_has_no_candidates(size: int) -> None:
    spec = GridSpec.for_size(size)
    assert cell_candidates(spec, COMPLETED[size], 0, 0) == frozenset()


@pytest.mark.parametrize("size", [4, 6, 9])
def test_peers_include_the_rest_of_the_box(size: int) -> None:
    spec = GridSpec.for_size(size)
    peers = set(peer_coordinates(spec, 0, 0))
    box = set(spec.cells_in_box(0)) - {(0, 0)}
    assert box <= peers
    assert (0, 0) not in peers
    box_only = (spec.box_rows - 1) * (spec.box_cols - 1)
    assert len(peers) == 2 * (size - 1) + box_only


def test_six_by_six_box_removes_a_digit_the_row_and_column_do_not() -> None:
    spec = GridSpec.for_size(6)
    board = [[0 for _ in range(6)] for _ in range(6)]
    board[1][2] = 5
    assert spec.box_index(0, 0) == spec.box_index(1, 2)
    assert 5 not in cell_candidates(spec, board, 0, 0)
    assert cell_candidates(spec, board, 0, 3) == frozenset(range(1, 7))


def test_nine_by_nine_box_removes_a_digit_the_row_and_column_do_not() -> None:
    spec = GridSpec.for_size(9)
    board = [[0 for _ in range(9)] for _ in range(9)]
    board[1][1] = 5
    assert 5 not in cell_candidates(spec, board, 0, 2)
    assert 5 in cell_candidates(spec, board, 0, 3)
