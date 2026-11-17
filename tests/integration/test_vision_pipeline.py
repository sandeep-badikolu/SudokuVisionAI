"""Load a synthetic PNG, detect the grid, and read cell coordinates.

These images are drawn in memory. They are not photographs of paper puzzles.
"""

import cv2
import numpy as np
import pytest

from app.vision import detect_grid, load_image, preprocess, segment_cells
from tests.unit.vision_samples import draw_grid, paint_cell


@pytest.mark.parametrize("order", [4, 6, 9])
def test_png_round_trip_preserves_detection(order: int, tmp_path) -> None:
    drawn = draw_grid(order)
    paint_cell(drawn, 0, order - 1, (0, 0, 0))
    path = tmp_path / f"grid-{order}.png"
    assert cv2.imwrite(str(path), drawn.image)
    loaded = load_image(path)
    prepared = preprocess(loaded.original)
    detection = detect_grid(loaded.original)
    patches = segment_cells(detection.warped, detection.order)
    marked = next(
        patch for patch in patches if patch.row == 0 and patch.col == order - 1
    )
    empty = next(
        patch for patch in patches if patch.row == order - 1 and patch.col == 0
    )

    assert prepared.source.shape == loaded.original.shape
    assert np.array_equal(loaded.original, drawn.image)
    assert detection.order == order
    assert detection.warped.shape[0] == detection.warped.shape[1]
    assert "overlay" in detection.diagnostics
    assert marked.origin[0] > marked.origin[1]
    assert float(marked.image.mean()) < float(empty.image.mean()) - 40
