"""Find a Sudoku boundary and warp it to a square grid.

A large contour is not accepted on area alone. It must be a convex
quadrilateral with plausible side ratios and angles, and the warped image
must contain a regular 4, 6, or 9 line grid.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from app.vision.cell_segmenter import (
    NORMALIZED_SIDE,
    CellPatch,
    segment_cells,
    segment_cells_at_lines,
)
from app.vision.errors import GridDetectionError
from app.vision.grid_lines import AxisLineEvidence, GridLineAnalysis, analyze_grid_lines
from app.vision.image_preprocessor import preprocess
from app.vision.perspective_transform import order_corners, warp_quad

_MIN_AREA_FRACTION = 0.08
_MAX_AREA_FRACTION = 0.95
_MIN_SIDE_RATIO = 0.5
_MIN_ANGLE = 40.0
_MAX_ANGLE = 150.0
_MIN_CONFIDENCE = 0.6
# A faint inner line can miss the production peak threshold. Equal slicing is
# allowed only when the outer border and every detected peak already sit on
# the same 9×9 lattice. Production recognition does not use this path.
_EQUAL_NINE_BORDER = 2.0
_EQUAL_NINE_TOLERANCE = 6.0
_EQUAL_NINE_MIN_PEAKS = 4


@dataclass(frozen=True, slots=True)
class GridQuad:
    """One quadrilateral that survived the geometric checks."""

    corners: np.ndarray
    confidence: float
    area: float


@dataclass(frozen=True, slots=True)
class BoundaryEvidence:
    """A warped quadrilateral whose grid lines have not been accepted.

    This is the evidence saved when the boundary is valid and the line check
    is not. It does not mean a grid order was detected.
    """

    corners: np.ndarray
    quad_confidence: float
    warped: np.ndarray
    overlay: np.ndarray
    lines: GridLineAnalysis


@dataclass(frozen=True, slots=True)
class GridDetection:
    """A warped Sudoku grid and the evidence used to accept it.

    ``confidence`` and ``quad_confidence`` are the quadrilateral geometry
    score. ``grid_confidence`` is the line-spacing score and is independent
    of that geometry. ``order`` is present only after the lines validate a
    4, 6, or 9 grid. ``diagnostics`` holds copies of the threshold, edges,
    corner overlay, and projection profiles.
    """

    corners: np.ndarray
    confidence: float
    warped: np.ndarray
    order: int
    diagnostics: dict[str, np.ndarray]
    quad_confidence: float
    grid_confidence: float
    lines: GridLineAnalysis


def detect_grid(image: np.ndarray) -> GridDetection:
    """Detect the outer Sudoku boundary and return a top-down square.

    Raises ``GridDetectionError`` when no candidate passes geometry,
    confidence, and grid-line checks.
    """
    prepared = preprocess(image)
    quads = _candidate_quads(prepared.threshold, prepared.source.shape)
    if not quads:
        raise GridDetectionError("no quadrilateral passed the geometry checks")
    best = max(quads, key=lambda quad: (quad.confidence, quad.area))
    if best.confidence < _MIN_CONFIDENCE:
        raise GridDetectionError(
            f"best quadrilateral confidence {best.confidence:.2f} "
            f"is below {_MIN_CONFIDENCE:.2f}"
        )
    warped = warp_quad(prepared.source, best.corners, NORMALIZED_SIDE)
    analysis = analyze_grid_lines(warped)
    overlay = _overlay(prepared.source, best.corners)
    if analysis.order is None:
        error = GridDetectionError(
            analysis.rejection_reason
            or "warped region does not contain a regular 4, 6, or 9 grid"
        )
        error.boundary = BoundaryEvidence(
            corners=best.corners.copy(),
            quad_confidence=best.confidence,
            warped=warped,
            overlay=overlay,
            lines=analysis,
        )
        raise error
    return GridDetection(
        corners=best.corners.copy(),
        confidence=best.confidence,
        warped=warped,
        order=analysis.order,
        diagnostics={
            "threshold": prepared.threshold.copy(),
            "edges": prepared.edges.copy(),
            "overlay": overlay,
            "horizontal_profile": analysis.horizontal_profile,
            "vertical_profile": analysis.vertical_profile,
        },
        quad_confidence=best.confidence,
        grid_confidence=float(analysis.grid_confidence or 0.0),
        lines=analysis,
    )


def extract_cells(
    image: np.ndarray, order: int | None = None
) -> tuple[GridDetection, tuple[CellPatch, ...]]:
    """Detect a grid and split it into cells.

    When ``order`` is given it must match the order inferred from the grid
    lines. A mismatch is rejected instead of slicing the wrong number of cells.
    """
    detection = detect_grid(image)
    if order is not None and order != detection.order:
        raise GridDetectionError(
            f"detected a {detection.order}×{detection.order} grid, not {order}×{order}"
        )
    chosen = detection.order if order is None else order
    horizontal = detection.lines.horizontal.merged_centers
    vertical = detection.lines.vertical.merged_centers
    if len(horizontal) == chosen + 1 and len(vertical) == chosen + 1:
        return detection, segment_cells_at_lines(
            detection.warped, horizontal, vertical
        )
    return detection, segment_cells(detection.warped, chosen)


def extract_cells_faint_nine(
    image: np.ndarray,
) -> tuple[GridDetection, tuple[CellPatch, ...]]:
    """Extract a 9×9 grid, including one whose inner lines are faint.

    ``extract_cells`` is tried first and its result is returned unchanged.
    When that check rejects a warped square, the cells are divided equally
    only if both axes already show an outer border and every detected peak
    lies on a 9×9 lattice. Recognition and the ordinary importer do not
    call this function.
    """
    try:
        return extract_cells(image)
    except GridDetectionError as exc:
        boundary = exc.boundary
        side = 0 if boundary is None else int(boundary.warped.shape[0])
        if boundary is None or not _equal_nine_lattice(boundary.lines, side):
            raise
        lines = _equal_nine_lines(side, boundary.lines)
        detection = GridDetection(
            corners=boundary.corners.copy(),
            confidence=boundary.quad_confidence,
            warped=boundary.warped,
            order=9,
            diagnostics={
                "threshold": boundary.overlay,
                "edges": boundary.overlay,
                "overlay": boundary.overlay,
                "horizontal_profile": boundary.lines.horizontal_profile,
                "vertical_profile": boundary.lines.vertical_profile,
            },
            quad_confidence=boundary.quad_confidence,
            grid_confidence=0.0,
            lines=lines,
        )
        return detection, segment_cells(boundary.warped, 9)


def _equal_nine_lattice(lines: GridLineAnalysis, side: int) -> bool:
    if side < 2:
        return False
    ideal = tuple(index * (side - 1) / 9 for index in range(10))
    horizontal = _axis_on_lattice(lines.horizontal.merged_centers, ideal)
    vertical = _axis_on_lattice(lines.vertical.merged_centers, ideal)
    return horizontal and vertical


def _axis_on_lattice(centers: tuple[float, ...], ideal: tuple[float, ...]) -> bool:
    if len(centers) < _EQUAL_NINE_MIN_PEAKS:
        return False
    if abs(centers[0] - ideal[0]) > _EQUAL_NINE_BORDER:
        return False
    if abs(centers[-1] - ideal[-1]) > _EQUAL_NINE_BORDER:
        return False
    return all(
        min(abs(center - line) for line in ideal) <= _EQUAL_NINE_TOLERANCE
        for center in centers
    )


def _equal_nine_lines(side: int, observed: GridLineAnalysis) -> GridLineAnalysis:
    centers = tuple(index * (side - 1) / 9 for index in range(10))
    pitch = (side - 1) / 9
    axis = AxisLineEvidence(
        raw_peak_count=len(centers),
        merged_peak_count=len(centers),
        inferred_order=9,
        pitch=pitch,
        gap_min=pitch,
        gap_max=pitch,
        gap_ratio=1.0,
        regular=True,
        raw_centers=centers,
        merged_centers=centers,
    )
    return GridLineAnalysis(
        horizontal=axis,
        vertical=axis,
        order=9,
        grid_confidence=None,
        recovery="equal_nine_faint_lines",
        rejection_reason=None,
        horizontal_profile=observed.horizontal_profile,
        vertical_profile=observed.vertical_profile,
    )


def estimate_grid_order(warped: np.ndarray) -> int | None:
    """Infer 4, 6, or 9 from regular horizontal and vertical line spacing.

    Nearby edge peaks from one thick stroke are grouped first. Returns
    ``None`` when the directions disagree, the spacing is uneven, or no
    supported order has a single consistent explanation.
    """
    return analyze_grid_lines(warped).order


def _candidate_quads(threshold: np.ndarray, shape: tuple[int, ...]) -> list[GridQuad]:
    inverted = cv2.bitwise_not(threshold)
    kernel = np.ones((3, 3), dtype=np.uint8)
    closed = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel)
    contours, _hierarchy = cv2.findContours(
        closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    image_area = float(shape[0] * shape[1])
    quads: list[GridQuad] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < _MIN_AREA_FRACTION * image_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        corners = order_corners(approx.reshape(4, 2))
        confidence = _confidence(corners, image_area)
        if confidence <= 0:
            continue
        quads.append(GridQuad(corners=corners, confidence=confidence, area=area))
    return quads


def _confidence(corners: np.ndarray, image_area: float) -> float:
    area = float(cv2.contourArea(corners.reshape(-1, 1, 2)))
    fraction = area / image_area
    if fraction < _MIN_AREA_FRACTION or fraction > _MAX_AREA_FRACTION:
        return 0.0
    sides = [
        float(np.linalg.norm(corners[(index + 1) % 4] - corners[index]))
        for index in range(4)
    ]
    if max(sides) == 0:
        return 0.0
    side_ratio = min(sides) / max(sides)
    if side_ratio < _MIN_SIDE_RATIO:
        return 0.0
    angles = _interior_angles(corners)
    if min(angles) < _MIN_ANGLE or max(angles) > _MAX_ANGLE:
        return 0.0
    diagonals = (
        float(np.linalg.norm(corners[2] - corners[0])),
        float(np.linalg.norm(corners[3] - corners[1])),
    )
    if max(diagonals) == 0:
        return 0.0
    diagonal_ratio = min(diagonals) / max(diagonals)
    area_score = min(fraction / 0.4, 1.0)
    return 0.5 * side_ratio + 0.3 * diagonal_ratio + 0.2 * area_score


def _interior_angles(corners: np.ndarray) -> list[float]:
    angles: list[float] = []
    for index in range(4):
        previous = corners[(index - 1) % 4]
        current = corners[index]
        nxt = corners[(index + 1) % 4]
        first = previous - current
        second = nxt - current
        denom = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denom == 0:
            return [0.0]
        cosine = float(np.clip(np.dot(first, second) / denom, -1.0, 1.0))
        angles.append(math.degrees(math.acos(cosine)))
    return angles


def _overlay(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    canvas = image.copy()
    points = np.round(corners).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [points], isClosed=True, color=(0, 180, 0), thickness=2)
    return canvas
