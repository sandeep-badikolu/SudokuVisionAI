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
    SUPPORTED_ORDERS,
    CellPatch,
    segment_cells,
)
from app.vision.errors import GridDetectionError
from app.vision.image_preprocessor import adaptive_threshold, preprocess
from app.vision.perspective_transform import order_corners, warp_quad

_MIN_AREA_FRACTION = 0.08
_MAX_AREA_FRACTION = 0.95
_MIN_SIDE_RATIO = 0.5
_MIN_ANGLE = 40.0
_MAX_ANGLE = 150.0
_MIN_CONFIDENCE = 0.6
_LINE_RELATIVE = 0.45
_MAX_GAP_RATIO = 1.65


@dataclass(frozen=True, slots=True)
class GridQuad:
    """One quadrilateral that survived the geometric checks."""

    corners: np.ndarray
    confidence: float
    area: float


@dataclass(frozen=True, slots=True)
class GridDetection:
    """A warped Sudoku grid and the evidence used to accept it.

    ``order`` is the detected cell count along one side: 4, 6, or 9.
    ``diagnostics`` holds copies of the threshold, edges, and an overlay of
    the accepted corners. None of these arrays alias the caller's image.
    """

    corners: np.ndarray
    confidence: float
    warped: np.ndarray
    order: int
    diagnostics: dict[str, np.ndarray]


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
    order = estimate_grid_order(warped)
    if order is None:
        raise GridDetectionError(
            "warped region does not contain a regular 4, 6, or 9 grid"
        )
    return GridDetection(
        corners=best.corners.copy(),
        confidence=best.confidence,
        warped=warped,
        order=order,
        diagnostics={
            "threshold": prepared.threshold.copy(),
            "edges": prepared.edges.copy(),
            "overlay": _overlay(prepared.source, best.corners),
        },
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
    return detection, segment_cells(detection.warped, chosen)


def estimate_grid_order(warped: np.ndarray) -> int | None:
    """Infer 4, 6, or 9 from regular horizontal and vertical line spacing.

    Returns ``None`` when the two directions disagree or the spacing is uneven.
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if warped.ndim == 3 else warped
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    lines = cv2.bitwise_not(adaptive_threshold(blurred))
    horizontal = _line_centers(lines.mean(axis=1))
    vertical = _line_centers(lines.mean(axis=0))
    horizontal_order = _order_from_centers(horizontal)
    vertical_order = _order_from_centers(vertical)
    if horizontal_order is None or horizontal_order != vertical_order:
        return None
    if horizontal_order not in SUPPORTED_ORDERS:
        return None
    return horizontal_order


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


def _line_centers(projection: np.ndarray) -> list[int]:
    if projection.size == 0 or float(projection.max()) == 0:
        return []
    active = projection >= (_LINE_RELATIVE * float(projection.max()))
    centers: list[int] = []
    start: int | None = None
    for index, flag in enumerate(active.tolist()):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            centers.append((start + index - 1) // 2)
            start = None
    if start is not None:
        centers.append((start + len(active) - 1) // 2)
    return centers


def _order_from_centers(centers: list[int]) -> int | None:
    if len(centers) < 5:
        return None
    gaps = np.diff(np.asarray(centers, dtype=np.float32))
    if float(gaps.min()) <= 0:
        return None
    if float(gaps.max()) > _MAX_GAP_RATIO * float(gaps.min()):
        return None
    return len(centers) - 1


def _overlay(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    canvas = image.copy()
    points = np.round(corners).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [points], isClosed=True, color=(0, 180, 0), thickness=2)
    return canvas
