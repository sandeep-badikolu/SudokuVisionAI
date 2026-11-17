"""Geometry of the supported Sudoku sizes."""

import pytest

from app.core import GridSpec


def test_canonical_sizes() -> None:
    four = GridSpec.for_size(4)
    six = GridSpec.for_size(6)
    nine = GridSpec.for_size(9)

    assert (four.size, four.box_rows, four.box_cols, four.max_digit) == (4, 2, 2, 4)
    assert (six.size, six.box_rows, six.box_cols, six.max_digit) == (6, 2, 3, 6)
    assert (nine.size, nine.box_rows, nine.box_cols, nine.max_digit) == (9, 3, 3, 9)
    assert four.cell_count == 16
    assert six.cell_count == 36
    assert nine.cell_count == 81
    assert four.box_count == 4
    assert six.box_count == 6
    assert nine.box_count == 9


def test_direct_construction_matches_factory() -> None:
    assert GridSpec(size=9, box_rows=3, box_cols=3) == GridSpec.for_size(9)


@pytest.mark.parametrize("size", [0, 1, 2, 3, 5, 8, 16])
def test_unsupported_size_is_rejected(size: int) -> None:
    with pytest.raises(ValueError, match="unsupported grid size"):
        GridSpec.for_size(size)


def test_non_canonical_box_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="2×3"):
        GridSpec(size=6, box_rows=3, box_cols=2)
    with pytest.raises(ValueError, match="3×3"):
        GridSpec(size=9, box_rows=9, box_cols=1)


def test_box_index_for_each_geometry() -> None:
    four = GridSpec.for_size(4)
    assert four.box_index(0, 0) == 0
    assert four.box_index(1, 1) == 0
    assert four.box_index(0, 2) == 1
    assert four.box_index(3, 3) == 3

    six = GridSpec.for_size(6)
    assert six.box_index(0, 0) == 0
    assert six.box_index(1, 2) == 0
    assert six.box_index(0, 3) == 1
    assert six.box_index(2, 0) == 2
    assert six.box_index(5, 5) == 5

    nine = GridSpec.for_size(9)
    assert nine.box_index(0, 0) == 0
    assert nine.box_index(4, 7) == 5
    assert nine.box_index(8, 8) == 8


def test_cells_in_box_cover_the_box_and_nothing_else() -> None:
    spec = GridSpec.for_size(6)
    cells = spec.cells_in_box(3)
    assert cells == (
        (2, 3),
        (2, 4),
        (2, 5),
        (3, 3),
        (3, 4),
        (3, 5),
    )
    assert spec.box_origin(3, 5) == (2, 3)
    assert all(spec.box_index(row, col) == 3 for row, col in cells)


def test_coordinates_outside_the_grid_are_rejected() -> None:
    spec = GridSpec.for_size(4)
    with pytest.raises(ValueError, match="outside"):
        spec.box_index(4, 0)
    with pytest.raises(ValueError, match="outside"):
        spec.cells_in_box(-1)
    with pytest.raises(ValueError, match="outside"):
        spec.cells_in_box(4)
