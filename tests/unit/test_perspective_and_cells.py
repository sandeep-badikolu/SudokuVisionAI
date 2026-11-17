"""Perspective warping and cell crops for 4, 6, and 9."""

import cv2
import numpy as np
import pytest

from app.vision import CellSegmentationError, segment_cells, warp_quad
from app.vision.cell_segmenter import NORMALIZED_SIDE


def test_warp_maps_a_known_square_to_the_requested_side() -> None:
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image[15:85, 20:100] = (240, 240, 240)
    corners = np.float32([[20, 15], [99, 15], [99, 84], [20, 84]])
    warped = warp_quad(image, corners, 80)
    original = image.copy()

    assert warped.shape == (80, 80, 3)
    assert int(warped[40, 40, 0]) > 200
    assert np.array_equal(image, original)


@pytest.mark.parametrize("order", [4, 6, 9])
def test_cell_segmentation_covers_every_coordinate(order: int) -> None:
    image = np.full((NORMALIZED_SIDE, NORMALIZED_SIDE, 3), 255, dtype=np.uint8)
    cell = NORMALIZED_SIDE // order
    for index in range(order + 1):
        position = index * cell
        cv2.line(image, (position, 0), (position, NORMALIZED_SIDE - 1), (0, 0, 0), 2)
        cv2.line(image, (0, position), (NORMALIZED_SIDE - 1, position), (0, 0, 0), 2)
    patches = segment_cells(image, order)
    located = {(patch.row, patch.col): patch for patch in patches}

    assert len(patches) == order * order
    assert set(located) == {(row, col) for row in range(order) for col in range(order)}
    first = located[(0, 0)]
    assert first.origin[0] > 0
    assert first.origin[1] > 0
    assert first.image.shape[0] == first.size[1]
    assert first.image.shape[1] == first.size[0]
    assert float(first.image.mean()) > 240
    next_col = located[(0, 1)]
    next_row = located[(1, 0)]
    assert next_col.origin[0] > first.origin[0]
    assert next_row.origin[1] > first.origin[1]


def test_invalid_order_and_shape_are_rejected() -> None:
    image = np.full((NORMALIZED_SIDE, NORMALIZED_SIDE, 3), 255, dtype=np.uint8)
    with pytest.raises(CellSegmentationError, match="4, 6, or 9"):
        segment_cells(image, 5)
    wide = np.full((NORMALIZED_SIDE, NORMALIZED_SIDE + 12, 3), 255, dtype=np.uint8)
    with pytest.raises(CellSegmentationError, match="square"):
        segment_cells(wide, 4)
