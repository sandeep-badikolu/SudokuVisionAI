"""Infer a 4, 6, or 9 grid from the lines in a warped Sudoku image.

Thick strokes often produce two edge peaks. Those peaks are grouped into one
line when the gap is too small to be a cell. The regularity limit that rejects
uneven grids is unchanged: a gap large enough to pass that limit is never
treated as a second edge of the same stroke.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from app.vision.cell_segmenter import SUPPORTED_ORDERS
from app.vision.image_preprocessor import adaptive_threshold

_LINE_RELATIVE = 0.45
_MAX_GAP_RATIO = 1.65
# 1 / 1.65 ≈ 0.606. Staying under that keeps a real cell gap, even an uneven
# one that the regularity check would still accept, from being merged.
_THICK_GAP_FRACTION = 0.55
# A lighter spike on the warped border can be the outer edge of the boundary
# stroke. It may sit farther in than an interior double edge, but not a full
# cell away from the boundary line.
_BORDER_GAP_FRACTION = 0.85
_BORDER_PX = 2.0
_MISSING_GAP_LOW = 1.7
_MISSING_GAP_HIGH = 2.3
_PROFILE_HEIGHT = 96


@dataclass(frozen=True, slots=True)
class LinePeak:
    """One projection peak.

    ``center`` is the intensity-weighted position of the run. ``weight`` is
    the sum of the projection inside that run, so a faint border spike can be
    told apart from the grid stroke.
    """

    center: float
    weight: float


@dataclass(frozen=True, slots=True)
class AxisLineEvidence:
    """Line evidence along one axis, after thick-stroke grouping."""

    raw_peak_count: int
    merged_peak_count: int
    inferred_order: int | None
    pitch: float | None
    gap_min: float | None
    gap_max: float | None
    gap_ratio: float | None
    regular: bool
    raw_centers: tuple[float, ...]
    merged_centers: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class GridLineAnalysis:
    """Horizontal and vertical line evidence for one warped grid.

    ``order`` is set only when both axes support the same 4, 6, or 9 grid.
    ``grid_confidence`` scores that spacing. It is not the quadrilateral score.
    ``recovery`` names a one-peak repair when one axis carried it; it stays
    ``None`` when both axes already agreed.
    """

    horizontal: AxisLineEvidence
    vertical: AxisLineEvidence
    order: int | None
    grid_confidence: float | None
    recovery: str | None
    rejection_reason: str | None
    horizontal_profile: np.ndarray
    vertical_profile: np.ndarray

    def to_report(self) -> dict[str, object]:
        """JSON-ready spacing summary. Peak positions stay in the profiles."""
        return {
            "horizontal": _axis_report(self.horizontal),
            "vertical": _axis_report(self.vertical),
            "order": self.order,
            "grid_confidence": _round(self.grid_confidence),
            "recovery": self.recovery,
            "rejection_reason": self.rejection_reason,
        }


def analyze_grid_lines(warped: np.ndarray) -> GridLineAnalysis:
    """Measure both axes and accept only a supported, regular order.

    Returns an analysis either way. ``order`` is ``None`` when the lines do
    not support one 4, 6, or 9 grid. Callers that require a grid should treat
    that as a rejection and keep the quadrilateral evidence separate.
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if warped.ndim == 3 else warped
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    ink = cv2.bitwise_not(adaptive_threshold(blurred))
    horizontal_projection = ink.mean(axis=1)
    vertical_projection = ink.mean(axis=0)
    horizontal = _analyze_axis(horizontal_projection)
    vertical = _analyze_axis(vertical_projection)
    order, confidence, recovery, reason = _decide(horizontal, vertical)
    return GridLineAnalysis(
        horizontal=horizontal,
        vertical=vertical,
        order=order,
        grid_confidence=confidence,
        recovery=recovery,
        rejection_reason=reason,
        horizontal_profile=render_projection(
            horizontal_projection,
            horizontal.raw_centers,
            horizontal.merged_centers,
        ),
        vertical_profile=render_projection(
            vertical_projection,
            vertical.raw_centers,
            vertical.merged_centers,
        ),
    )


def collapse_thick_lines(
    peaks: tuple[LinePeak, ...] | list[LinePeak], length: int
) -> tuple[LinePeak, ...]:
    """Group edge peaks that belong to one physical grid line.

    The cell pitch is the upper quartile of the gaps. A doubled stroke can
    produce as many narrow gaps as there are lines, and those must not pull
    the pitch down to the stroke width. Each pass merges only the narrowest
    eligible gap. A gap at least ``1 / 1.65`` of the pitch is left alone,
    because that is already a legal cell gap.
    """
    grouped = list(peaks)
    while len(grouped) >= 2:
        gaps = [
            grouped[index + 1].center - grouped[index].center
            for index in range(len(grouped) - 1)
        ]
        pitch = _robust_pitch(gaps)
        if pitch is None or pitch <= 0:
            break
        choice = _narrowest_group(grouped, length, pitch)
        if choice is None:
            break
        index, border = choice
        grouped = (
            grouped[:index]
            + [_combine(grouped[index], grouped[index + 1], border)]
            + grouped[index + 2 :]
        )
    return tuple(grouped)


def render_projection(
    projection: np.ndarray,
    raw_centers: tuple[float, ...],
    merged_centers: tuple[float, ...],
) -> np.ndarray:
    """Draw a projection, its raw peaks, and the grouped line centers."""
    width = max(int(projection.size), 1)
    canvas = np.full((_PROFILE_HEIGHT, width, 3), 255, dtype=np.uint8)
    peak = float(projection.max()) if projection.size else 0.0
    scale = (_PROFILE_HEIGHT - 8) / peak if peak > 0 else 0.0
    for x, value in enumerate(projection.tolist()):
        bar = int(round(float(value) * scale))
        if bar <= 0:
            continue
        cv2.line(
            canvas,
            (x, _PROFILE_HEIGHT - 1),
            (x, _PROFILE_HEIGHT - 1 - bar),
            (180, 180, 180),
            1,
        )
    for center in raw_centers:
        x = int(round(center))
        if 0 <= x < width:
            cv2.line(canvas, (x, 0), (x, _PROFILE_HEIGHT - 1), (40, 40, 220), 1)
    for center in merged_centers:
        x = int(round(center))
        if 0 <= x < width:
            cv2.circle(canvas, (x, 8), 3, (0, 140, 0), -1)
    return canvas


def _analyze_axis(projection: np.ndarray) -> AxisLineEvidence:
    raw = _peaks(projection)
    merged = collapse_thick_lines(raw, int(projection.size))
    pitch, gap_min, gap_max, gap_ratio, regular = _spacing(merged)
    return AxisLineEvidence(
        raw_peak_count=len(raw),
        merged_peak_count=len(merged),
        inferred_order=_supported_order(len(merged), regular),
        pitch=pitch,
        gap_min=gap_min,
        gap_max=gap_max,
        gap_ratio=gap_ratio,
        regular=regular,
        raw_centers=tuple(peak.center for peak in raw),
        merged_centers=tuple(peak.center for peak in merged),
    )


def _peaks(projection: np.ndarray) -> tuple[LinePeak, ...]:
    if projection.size == 0 or float(projection.max()) == 0:
        return ()
    active = projection >= (_LINE_RELATIVE * float(projection.max()))
    peaks: list[LinePeak] = []
    start: int | None = None
    flags = active.tolist()
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            peaks.append(_peak(projection, start, index))
            start = None
    if start is not None:
        peaks.append(_peak(projection, start, len(flags)))
    return tuple(peaks)


def _peak(projection: np.ndarray, start: int, end: int) -> LinePeak:
    weights = projection[start:end].astype(np.float64)
    total = float(weights.sum())
    if total <= 0:
        return LinePeak(center=(start + end - 1) / 2, weight=0.0)
    positions = np.arange(start, end, dtype=np.float64)
    return LinePeak(center=float((positions * weights).sum() / total), weight=total)


def _narrowest_group(
    peaks: list[LinePeak], length: int, pitch: float
) -> tuple[int, bool] | None:
    thick_limit = _THICK_GAP_FRACTION * pitch
    border_limit = _BORDER_GAP_FRACTION * pitch
    best: tuple[float, int, bool] | None = None
    for index in range(len(peaks) - 1):
        gap = peaks[index + 1].center - peaks[index].center
        if gap <= 0:
            continue
        border = _border_pair(peaks[index], peaks[index + 1], length, gap, border_limit)
        if gap < thick_limit or border:
            if best is None or gap < best[0]:
                best = (gap, index, border and gap >= thick_limit)
    if best is None:
        return None
    return best[1], best[2]


def _border_pair(
    left: LinePeak, right: LinePeak, length: int, gap: float, limit: float
) -> bool:
    """A light spike on the image edge, closer than one cell to a stronger line."""
    if gap >= limit:
        return False
    if _on_border(left.center, length) and left.weight < right.weight:
        return True
    return _on_border(right.center, length) and right.weight < left.weight


def _on_border(center: float, length: int) -> bool:
    return center <= _BORDER_PX or center >= (length - 1) - _BORDER_PX


def _combine(left: LinePeak, right: LinePeak, border: bool) -> LinePeak:
    """Center a thick stroke, or keep the stronger line beside a border spike."""
    if border:
        return right if left.weight < right.weight else left
    total = left.weight + right.weight
    if total <= 0:
        return LinePeak(center=(left.center + right.center) / 2, weight=0.0)
    center = (left.center * left.weight + right.center * right.weight) / total
    return LinePeak(center=center, weight=total)


def _decide(
    horizontal: AxisLineEvidence, vertical: AxisLineEvidence
) -> tuple[int | None, float | None, str | None, str | None]:
    h_order = horizontal.inferred_order
    v_order = vertical.inferred_order
    if h_order is not None and h_order == v_order:
        confidence = _spacing_confidence(
            (horizontal.gap_ratio, vertical.gap_ratio), recovered=False
        )
        return h_order, confidence, None, None
    if h_order is not None and v_order is not None:
        return (
            None,
            None,
            None,
            (
                f"horizontal order {h_order} disagrees with vertical order {v_order}; "
                "warped region does not contain a regular 4, 6, or 9 grid"
            ),
        )
    if h_order is not None:
        return _recover("vertical", horizontal, vertical)
    if v_order is not None:
        return _recover("horizontal", vertical, horizontal)
    return (
        None,
        None,
        None,
        "warped region does not contain a regular 4, 6, or 9 grid",
    )


def _recover(
    weak_name: str, strong: AxisLineEvidence, weak: AxisLineEvidence
) -> tuple[int | None, float | None, str | None, str | None]:
    order = strong.inferred_order
    if order is None:
        return (
            None,
            None,
            None,
            "warped region does not contain a regular 4, 6, or 9 grid",
        )
    needed = order + 1
    centers = weak.merged_centers
    if len(centers) == needed + 1:
        return _recover_extra(weak_name, order, strong, centers)
    if len(centers) == needed - 1:
        return _recover_missing(weak_name, order, strong, centers)
    return (
        None,
        None,
        None,
        (
            f"{weak_name} line evidence does not support the "
            f"{order}×{order} grid on the other axis; "
            "warped region does not contain a regular 4, 6, or 9 grid"
        ),
    )


def _recover_extra(
    weak_name: str,
    order: int,
    strong: AxisLineEvidence,
    centers: tuple[float, ...],
) -> tuple[int | None, float | None, str | None, str | None]:
    fits: list[float] = []
    for index in range(len(centers)):
        trial = centers[:index] + centers[index + 1 :]
        ratio = _ratio_if_order(trial, order)
        trial_pitch = _robust_pitch(np.diff(np.asarray(trial, dtype=np.float64)))
        if ratio is None or not _pitches_agree(strong.pitch, trial_pitch):
            continue
        fits.append(ratio)
    if len(fits) != 1:
        return (
            None,
            None,
            None,
            (
                f"{weak_name} axis has one extra peak but it does not identify "
                "a single regular 4, 6, or 9 grid"
            ),
        )
    confidence = _spacing_confidence((strong.gap_ratio, fits[0]), recovered=True)
    return order, confidence, f"{weak_name}_extra_peak", None


def _recover_missing(
    weak_name: str,
    order: int,
    strong: AxisLineEvidence,
    centers: tuple[float, ...],
) -> tuple[int | None, float | None, str | None, str | None]:
    gaps = np.diff(np.asarray(centers, dtype=np.float64))
    if gaps.size < 2:
        return _unsupported(weak_name, order)
    # Remove only the one widest gap. A second wide gap stays in the remainder
    # and fails the regularity check, so two missing lines are not repaired.
    widest = int(np.argmax(gaps))
    others = np.concatenate((gaps[:widest], gaps[widest + 1 :]))
    largest = float(gaps[widest])
    typical = float(np.median(others))
    missing_span = _MISSING_GAP_LOW * typical <= largest <= _MISSING_GAP_HIGH * typical
    if typical <= 0 or not missing_span:
        return _unsupported(weak_name, order)
    other_min = float(others.min())
    other_ratio = float(others.max() / other_min) if other_min > 0 else None
    if other_ratio is None or other_ratio > _MAX_GAP_RATIO:
        return _unsupported(weak_name, order)
    if not _pitches_agree(strong.pitch, typical):
        return _unsupported(weak_name, order)
    confidence = _spacing_confidence((strong.gap_ratio, other_ratio), recovered=True)
    return order, confidence, f"{weak_name}_missing_line", None


def _unsupported(
    weak_name: str, order: int
) -> tuple[int | None, float | None, str | None, str | None]:
    return (
        None,
        None,
        None,
        (
            f"{weak_name} line evidence does not support the "
            f"{order}×{order} grid on the other axis; "
            "warped region does not contain a regular 4, 6, or 9 grid"
        ),
    )


def _ratio_if_order(centers: tuple[float, ...], order: int) -> float | None:
    _pitch, _gap_min, _gap_max, ratio, regular = _spacing_values(centers)
    if not regular or len(centers) - 1 != order or ratio is None:
        return None
    return ratio


def _spacing(
    peaks: tuple[LinePeak, ...],
) -> tuple[float | None, float | None, float | None, float | None, bool]:
    pitch, gap_min, gap_max, ratio, regular = _spacing_values(
        tuple(peak.center for peak in peaks)
    )
    return pitch, gap_min, gap_max, ratio, regular


def _spacing_values(
    centers: tuple[float, ...],
) -> tuple[float | None, float | None, float | None, float | None, bool]:
    if len(centers) < 2:
        return None, None, None, None, False
    gaps = np.diff(np.asarray(centers, dtype=np.float64))
    gap_min = float(gaps.min())
    gap_max = float(gaps.max())
    pitch = _robust_pitch(gaps)
    if gap_min <= 0:
        return pitch, gap_min, gap_max, None, False
    ratio = gap_max / gap_min
    return pitch, gap_min, gap_max, ratio, ratio <= _MAX_GAP_RATIO


def _robust_pitch(gaps: np.ndarray | list[float]) -> float | None:
    values = np.asarray(gaps, dtype=np.float64)
    positive = values[values > 0]
    if positive.size == 0:
        return None
    return float(np.percentile(positive, 75))


def _supported_order(line_count: int, regular: bool) -> int | None:
    order = line_count - 1
    if regular and order in SUPPORTED_ORDERS:
        return order
    return None


def _pitches_agree(strong: float | None, weak: float | None) -> bool:
    if strong is None or weak is None or min(strong, weak) <= 0:
        return False
    return max(strong, weak) / min(strong, weak) <= _MAX_GAP_RATIO


def _spacing_confidence(
    ratios: tuple[float | None, ...], recovered: bool
) -> float:
    usable = [ratio for ratio in ratios if ratio is not None]
    worst = max(usable) if usable else _MAX_GAP_RATIO
    worst = max(worst, 1.0)
    score = 1.0 - 0.4 * (worst - 1.0) / (_MAX_GAP_RATIO - 1.0)
    if recovered:
        score *= 0.85
    return float(min(1.0, max(0.0, score)))


def _axis_report(axis: AxisLineEvidence) -> dict[str, object]:
    return {
        "raw_peaks": axis.raw_peak_count,
        "merged_peaks": axis.merged_peak_count,
        "inferred_order": axis.inferred_order,
        "pitch": _round(axis.pitch),
        "gap_min": _round(axis.gap_min),
        "gap_max": _round(axis.gap_max),
        "gap_ratio": _round(axis.gap_ratio),
    }


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)
