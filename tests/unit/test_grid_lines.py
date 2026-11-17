"""Line grouping and grid-order inference, independent of the solver."""

import cv2
import numpy as np
import pytest

from app.vision.errors import GridDetectionError
from app.vision.grid_detector import detect_grid, estimate_grid_order
from app.vision.grid_lines import LinePeak, analyze_grid_lines, collapse_thick_lines
from tests.unit.vision_samples import draw_grid


def _peak(center: float, weight: float = 100.0) -> LinePeak:
    return LinePeak(center=center, weight=weight)


def _positions(order: int, pitch: float, origin: float) -> list[int]:
    return [int(round(origin + index * pitch)) for index in range(order + 1)]


def _ruled(
    horizontal: list[int],
    vertical: list[int],
    *,
    side: int = 360,
    thickness: int = 2,
    pair_gap: int = 0,
) -> np.ndarray:
    """Draw grid lines. ``pair_gap`` adds a second edge beside each line."""
    image = np.full((side, side), 255, dtype=np.uint8)
    for position in horizontal:
        _stroke(
            image, position, horizontal=True, thickness=thickness, pair_gap=pair_gap
        )
    for position in vertical:
        _stroke(
            image, position, horizontal=False, thickness=thickness, pair_gap=pair_gap
        )
    return image


def _stroke(
    image: np.ndarray,
    position: int,
    *,
    horizontal: bool,
    thickness: int,
    pair_gap: int,
) -> None:
    side = image.shape[0]
    offsets = (0,) if pair_gap == 0 else (-pair_gap // 2, pair_gap // 2)
    for offset in offsets:
        located = int(np.clip(position + offset, 0, side - 1))
        if horizontal:
            cv2.line(image, (0, located), (side - 1, located), 0, thickness)
        else:
            cv2.line(image, (located, 0), (located, side - 1), 0, thickness)


@pytest.mark.parametrize("order", [4, 6, 9])
def test_thin_lines_keep_their_order(order: int) -> None:
    drawn = draw_grid(order, line=1)
    detection = detect_grid(drawn.image)
    assert detection.order == order
    assert detection.lines.horizontal.raw_peak_count == order + 1
    assert detection.lines.horizontal.merged_peak_count == order + 1
    assert detection.grid_confidence >= 0.6
    assert detection.quad_confidence == pytest.approx(detection.confidence)


@pytest.mark.parametrize("order", [4, 9])
def test_solid_thick_lines_stay_one_peak_per_line(order: int) -> None:
    drawn = draw_grid(order, cell=40, margin=36, line=10)
    assert estimate_grid_order(drawn.image) == order


def test_double_edge_peaks_group_into_one_line() -> None:
    peaks = []
    for center in (20, 100, 180, 260, 340):
        peaks.append(_peak(center - 8, weight=80))
        peaks.append(_peak(center + 8, weight=80))
    grouped = collapse_thick_lines(peaks, 360)

    assert len(grouped) == 5
    assert grouped[0].center == pytest.approx(20)
    assert grouped[-1].center == pytest.approx(340)
    image = _ruled(
        _positions(4, 80, 20),
        _positions(4, 80, 20),
        pair_gap=14,
    )
    analysis = analyze_grid_lines(image)
    assert analysis.order == 4
    assert analysis.horizontal.raw_peak_count > analysis.horizontal.merged_peak_count
    assert analysis.horizontal.merged_peak_count == 5


def test_a_legal_narrow_cell_gap_is_not_merged() -> None:
    """A gap of 0.62 of the pitch still passes the 1.65 regularity check."""
    peaks = [_peak(center) for center in (0, 100, 162, 262, 362)]
    grouped = collapse_thick_lines(peaks, 400)
    assert len(grouped) == 5


def test_light_border_spike_groups_with_the_boundary_line() -> None:
    centers = [0, 25, *range(59, 360, 34)]
    peaks = [_peak(0, weight=40), _peak(25, weight=200)]
    peaks.extend(_peak(center) for center in centers[2:])
    grouped = collapse_thick_lines(peaks, 360)
    assert grouped[0].center == pytest.approx(25)
    assert len(grouped) < len(peaks)


def test_equal_border_line_is_not_absorbed_across_a_wide_gap() -> None:
    peaks = [_peak(0, weight=200), _peak(25, weight=200), _peak(59), _peak(93)]
    grouped = collapse_thick_lines(peaks, 360)
    assert [peak.center for peak in grouped[:2]] == pytest.approx([0, 25])


def test_blurry_axis_with_one_extra_peak_recovers_the_supported_order() -> None:
    horizontal = _positions(4, 60, 10)
    vertical = [*horizontal, 355]
    image = cv2.GaussianBlur(_ruled(horizontal, vertical), (5, 5), 0)
    analysis = analyze_grid_lines(image)
    assert analysis.order == 4
    assert analysis.recovery == "vertical_extra_peak"
    assert analysis.grid_confidence is not None
    assert analysis.grid_confidence < 1


def test_one_missing_line_recovers_only_when_the_other_axis_is_regular() -> None:
    horizontal = _positions(4, 70, 10)
    vertical = [10, 80, 150, 290]
    analysis = analyze_grid_lines(_ruled(horizontal, vertical))
    assert analysis.order == 4
    assert analysis.recovery == "vertical_missing_line"


def test_two_missing_lines_are_rejected() -> None:
    horizontal = _positions(4, 70, 10)
    vertical = [10, 80, 290]
    assert estimate_grid_order(_ruled(horizontal, vertical)) is None


def test_irregular_spacing_is_rejected() -> None:
    positions = [12, 28, 140, 155, 260, 340]
    analysis = analyze_grid_lines(_ruled(positions, positions))
    assert analysis.order is None
    assert analysis.grid_confidence is None
    assert "regular 4, 6, or 9" in (analysis.rejection_reason or "")


def test_axes_that_support_different_orders_are_rejected() -> None:
    analysis = analyze_grid_lines(
        _ruled(_positions(4, 70, 20), _positions(6, 50, 20))
    )
    assert analysis.order is None
    assert "disagrees" in (analysis.rejection_reason or "")


def test_an_extra_peak_with_two_explanations_is_rejected() -> None:
    horizontal = _positions(4, 70, 10)
    vertical = [0, 70, 140, 210, 280, 350]
    analysis = analyze_grid_lines(_ruled(horizontal, vertical))
    assert analysis.order is None
    assert analysis.horizontal.inferred_order == 4
    assert "extra peak" in (analysis.rejection_reason or "")


def test_two_extra_peaks_are_not_silently_removed() -> None:
    horizontal = _positions(4, 70, 10)
    vertical = [*horizontal, 330, 350]
    assert estimate_grid_order(_ruled(horizontal, vertical)) is None


def test_rejected_grid_keeps_the_boundary_separate_from_a_detected_order() -> None:
    image = np.full((320, 320, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (30, 30), (290, 290), (0, 0, 0), 4)
    for position in (50, 70, 210, 250):
        cv2.line(image, (30, position), (290, position), (0, 0, 0), 3)
        cv2.line(image, (position, 30), (position, 290), (0, 0, 0), 3)
    with pytest.raises(GridDetectionError, match="regular 4, 6, or 9") as raised:
        detect_grid(image)
    boundary = raised.value.boundary
    assert boundary is not None
    assert boundary.quad_confidence >= 0.6
    assert boundary.lines.order is None
    assert boundary.lines.grid_confidence is None
