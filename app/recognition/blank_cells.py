"""Decide whether a Sudoku cell is empty, occupied, or uncertain.

Dark pixels alone are not a digit. Grid-line remnants and small specks are
ignored. A faint but coherent interior mark is left uncertain instead of
being called empty.
"""

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from app.vision.cell_segmenter import CellPatch


class CellStatus(Enum):
    """Occupancy of one cell. ``UNCERTAIN`` is not a blank cell."""

    BLANK = "blank"
    OCCUPIED = "occupied"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class Component:
    """One connected dark region inside a cell."""

    area: int
    width: int
    height: int
    center_x: float
    center_y: float
    touches_border: bool
    border_fraction: float
    stroke_strength: float


@dataclass(frozen=True, slots=True)
class CellClassification:
    """Blank-cell decision for one crop.

    ``blank`` is true only for ``BLANK``. It is false for ``OCCUPIED`` and
    ``None`` for ``UNCERTAIN``. ``confidence`` is confidence in ``status``,
    from 0 to 1. An uncertain cell stays at or below 0.6.
    """

    row: int
    col: int
    status: CellStatus
    blank: bool | None
    confidence: float

    def to_report(self) -> dict[str, object]:
        """JSON fields for this cell. The crop itself is stored separately."""
        return {
            "row": self.row,
            "col": self.col,
            "blank": self.blank,
            "status": self.status.value,
            "confidence": round(float(self.confidence), 4),
        }


_BACKGROUND_PERCENTILE = 90.0
_STRICT_DELTA = 40.0
_FAINT_DELTA = 18.0
_NOISE_AREA_FRACTION = 0.008
_DIGIT_AREA_FRACTION = 0.02
_MIN_SHORT_FRACTION = 0.16
_MIN_LONG_FRACTION = 0.28
_LINE_ASPECT = 4.0
_LINE_THICKNESS_FRACTION = 0.22
_BORDER_BAND_FRACTION = 0.18
_BORDER_INK_FRACTION = 0.75
# A printed stroke has a sharper local gradient than a smooth shadow.
_MIN_STROKE_STRENGTH = 18.0
_CENTER_MARGIN = 0.16
_UNCERTAIN_CONFIDENCE_CAP = 0.6


def classify_cell(image: np.ndarray, row: int = 0, col: int = 0) -> CellClassification:
    """Classify one cell image as blank, occupied, or uncertain."""
    gray = _gray(image)
    if gray.size == 0:
        return _result(row, col, CellStatus.UNCERTAIN, 0.0)
    background = float(np.percentile(gray, _BACKGROUND_PERCENTILE))
    strength = _gradient(gray)
    strict = _components(gray, background, _STRICT_DELTA, strength)
    faint = _components(gray, background, _FAINT_DELTA, strength)
    strict_digits = [item for item in strict if _is_digit(item, gray.shape)]
    faint_digits = [item for item in faint if _is_digit(item, gray.shape)]
    if strict_digits:
        confidence = _occupied_confidence(strict_digits, gray, background)
        return _result(row, col, CellStatus.OCCUPIED, confidence)
    if faint_digits:
        confidence = min(
            _UNCERTAIN_CONFIDENCE_CAP,
            0.35 + 0.25 * _area_fraction(faint_digits, gray.shape),
        )
        return _result(row, col, CellStatus.UNCERTAIN, confidence)
    confidence = 0.96 if not faint else 0.86
    return _result(row, col, CellStatus.BLANK, confidence)


def classify_cells(
    patches: tuple[CellPatch, ...] | list[CellPatch],
) -> tuple[CellClassification, ...]:
    """Classify every patch. The result follows the patch order."""
    return tuple(
        classify_cell(patch.image, row=patch.row, col=patch.col) for patch in patches
    )


def render_contact_sheet(
    patches: tuple[CellPatch, ...] | list[CellPatch],
    classes: tuple[CellClassification, ...] | list[CellClassification],
    *,
    tile: int = 48,
) -> np.ndarray:
    """Draw every cell in row-major order with a border for its status.

    Green is blank, blue is occupied, and amber is uncertain. Tiles are
    resized so grids of different orders share one sheet.
    """
    if len(patches) != len(classes):
        raise ValueError("each cell needs one classification")
    if not patches:
        raise ValueError("contact sheet needs at least one cell")
    order = int(round(len(patches) ** 0.5))
    if order * order != len(patches):
        raise ValueError("contact sheet expects a square grid of cells")
    border = 3
    stride = tile + border
    canvas = np.full(
        (order * stride + border, order * stride + border, 3),
        255,
        dtype=np.uint8,
    )
    for patch, classification in zip(patches, classes, strict=True):
        swatch = _tile(patch.image, tile)
        y = border + patch.row * stride
        x = border + patch.col * stride
        color = _STATUS_COLOR[classification.status]
        cv2.rectangle(
            canvas,
            (x - border, y - border),
            (x + tile + border - 1, y + tile + border - 1),
            color,
            thickness=border,
        )
        canvas[y : y + tile, x : x + tile] = swatch
    return canvas


_STATUS_COLOR = {
    CellStatus.BLANK: (80, 160, 60),
    CellStatus.OCCUPIED: (180, 90, 30),
    CellStatus.UNCERTAIN: (40, 180, 220),
}


def _result(
    row: int, col: int, status: CellStatus, confidence: float
) -> CellClassification:
    blank: bool | None
    if status is CellStatus.BLANK:
        blank = True
    elif status is CellStatus.OCCUPIED:
        blank = False
    else:
        blank = None
    return CellClassification(
        row=row,
        col=col,
        status=status,
        blank=blank,
        confidence=float(min(0.99, max(0.0, confidence))),
    )


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _gradient(gray: np.ndarray) -> np.ndarray:
    horizontal = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    vertical = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(horizontal, vertical)


def _components(
    gray: np.ndarray, background: float, delta: float, strength: np.ndarray
) -> tuple[Component, ...]:
    ink = gray < (background - delta)
    mask = ink.astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    height, width = gray.shape[:2]
    band = max(2, int(_BORDER_BAND_FRACTION * min(height, width)))
    found: list[Component] = []
    for index in range(1, count):
        x, y, box_width, box_height, area = (int(value) for value in stats[index])
        ys, xs = np.where(labels == index)
        near_border = (
            (xs < band)
            | (ys < band)
            | (xs >= width - band)
            | (ys >= height - band)
        )
        border_fraction = float(near_border.mean()) if area else 1.0
        stroke = float(strength[labels == index].mean()) if area else 0.0
        found.append(
            Component(
                area=area,
                width=box_width,
                height=box_height,
                center_x=float(centroids[index, 0]),
                center_y=float(centroids[index, 1]),
                touches_border=(
                    x <= 1
                    or y <= 1
                    or x + box_width >= width - 1
                    or y + box_height >= height - 1
                ),
                border_fraction=border_fraction,
                stroke_strength=stroke,
            )
        )
    return tuple(found)


def _is_digit(component: Component, shape: tuple[int, ...]) -> bool:
    height, width = shape[:2]
    side = float(min(height, width))
    cell_area = float(height * width)
    if component.area < _NOISE_AREA_FRACTION * cell_area:
        return False
    if component.border_fraction >= _BORDER_INK_FRACTION:
        return False
    if component.stroke_strength < _MIN_STROKE_STRENGTH:
        return False
    short = float(min(component.width, component.height))
    long = float(max(component.width, component.height))
    if short <= 0:
        return False
    thin = short <= _LINE_THICKNESS_FRACTION * side
    line_like = long / short >= _LINE_ASPECT and thin
    if line_like:
        return False
    if component.area < _DIGIT_AREA_FRACTION * cell_area:
        return False
    if short < _MIN_SHORT_FRACTION * side or long < _MIN_LONG_FRACTION * side:
        return False
    margin_x = _CENTER_MARGIN * width
    margin_y = _CENTER_MARGIN * height
    centered = (
        margin_x <= component.center_x <= width - margin_x
        and margin_y <= component.center_y <= height - margin_y
    )
    return centered


def _occupied_confidence(
    digits: list[Component], gray: np.ndarray, background: float
) -> float:
    largest = max(digits, key=lambda item: item.area)
    fraction = largest.area / float(gray.size)
    coverage = min(fraction / 0.08, 1.0)
    contrast = min(max(background - float(np.percentile(gray, 10)), 0.0) / 80.0, 1.0)
    return 0.64 + 0.22 * coverage + 0.12 * contrast


def _area_fraction(components: list[Component], shape: tuple[int, ...]) -> float:
    area = float(shape[0] * shape[1])
    if area <= 0:
        return 0.0
    return min(sum(item.area for item in components) / area, 1.0)


def _tile(image: np.ndarray, tile: int) -> np.ndarray:
    color = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.resize(color, (tile, tile), interpolation=cv2.INTER_AREA)
