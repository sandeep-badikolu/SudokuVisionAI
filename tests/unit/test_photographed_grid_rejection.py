"""Newspaper 4×4 photos fail after a valid warp, not before it.

The line threshold stays where it is. These photographs must not be
imported by lowering that threshold or by skipping the line check.
"""

from pathlib import Path

import pytest

from app.vision.errors import GridDetectionError
from app.vision.grid_detector import extract_cells
from app.vision.grid_lines import _LINE_RELATIVE
from app.vision.image_loader import load_image

_INCOMING = Path("incoming_photographed/4x4")


def test_the_line_threshold_is_unchanged() -> None:
    assert _LINE_RELATIVE == 0.45


@pytest.mark.parametrize("name", ["4*4 1.jpeg", "4*4 7.jpeg"])
def test_bold_box_photos_fail_only_the_line_check(name: str) -> None:
    path = _INCOMING / name
    if not path.is_file():
        pytest.skip("incoming 4×4 photograph is not present")
    image = load_image(path).original
    with pytest.raises(GridDetectionError, match="regular 4, 6, or 9 grid") as caught:
        extract_cells(image)
    boundary = caught.value.boundary
    assert boundary is not None
    assert boundary.quad_confidence >= 0.6
    assert boundary.warped.shape[0] == boundary.warped.shape[1]
    assert boundary.lines.order is None
    assert boundary.lines.horizontal.merged_peak_count == 3
    assert boundary.lines.vertical.merged_peak_count == 3


def test_the_soft_page_does_not_show_five_grid_lines() -> None:
    path = _INCOMING / "4*4 8.jpeg"
    if not path.is_file():
        pytest.skip("incoming 4×4 photograph is not present")
    image = load_image(path).original
    with pytest.raises(GridDetectionError, match="regular 4, 6, or 9 grid") as caught:
        extract_cells(image)
    boundary = caught.value.boundary
    assert boundary is not None
    assert boundary.quad_confidence >= 0.6
    assert boundary.lines.horizontal.merged_peak_count < 5
    assert boundary.lines.vertical.merged_peak_count < 5
