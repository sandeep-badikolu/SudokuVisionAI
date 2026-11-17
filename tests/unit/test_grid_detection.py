"""Grid detection on synthetic drawings, including rejection cases."""

import cv2
import numpy as np
import pytest

from app.vision import GridDetectionError, detect_grid, extract_cells
from tests.unit.vision_samples import draw_grid, paint_cell, shade, skew


@pytest.mark.parametrize("order", [4, 6, 9])
def test_axis_aligned_grid_is_detected_and_segmented(order: int) -> None:
    drawn = draw_grid(order)
    paint_cell(drawn, 1, 2, (20, 20, 20))
    before = drawn.image.copy()
    _detection, patches = extract_cells(drawn.image, order=order)
    located = {(patch.row, patch.col): patch for patch in patches}

    assert np.array_equal(drawn.image, before)
    assert len(patches) == order * order
    marked = float(located[(1, 2)].image.mean())
    empty = float(located[(0, 0)].image.mean())
    assert marked < empty - 30


def test_perspective_and_uneven_light_still_keep_cell_coordinates() -> None:
    drawn = draw_grid(9)
    paint_cell(drawn, 3, 5, (15, 15, 15))
    photographed = shade(skew(drawn.image))
    detection, patches = extract_cells(photographed)
    located = {(patch.row, patch.col): patch for patch in patches}

    assert detection.order == 9
    assert detection.confidence >= 0.6
    marked_mean = float(located[(3, 5)].image.mean())
    empty_mean = float(located[(0, 0)].image.mean())
    assert marked_mean < empty_mean - 20


def test_detected_order_must_match_the_requested_order() -> None:
    drawn = draw_grid(4)
    with pytest.raises(GridDetectionError, match="4×4"):
        extract_cells(drawn.image, order=9)


@pytest.mark.parametrize("builder", ["blank", "triangle", "wide_bar", "circle"])
def test_invalid_shapes_are_rejected(builder: str) -> None:
    image = _invalid_image(builder)
    with pytest.raises(GridDetectionError):
        detect_grid(image)


def _invalid_image(kind: str) -> np.ndarray:
    image = np.full((240, 240, 3), 255, dtype=np.uint8)
    if kind == "blank":
        return image
    if kind == "triangle":
        points = np.array([[20, 200], [120, 20], [220, 200]], dtype=np.int32)
        cv2.fillPoly(image, [points], (0, 0, 0))
        return image
    if kind == "wide_bar":
        cv2.rectangle(image, (10, 90), (230, 140), (0, 0, 0), thickness=-1)
        return image
    cv2.circle(image, (120, 120), 80, (0, 0, 0), thickness=3)
    return image
