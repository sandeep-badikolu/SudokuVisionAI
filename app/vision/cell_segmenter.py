"""Split a normalized square grid into padded cells.

The crop for each cell is inset from the grid lines so the border stroke is
not part of the cell image. When detected line centers are available, the
cells are cut between those lines so a margin around the grid is not divided
as if it were part of the puzzle.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from app.vision.errors import CellSegmentationError

SUPPORTED_ORDERS = (4, 6, 9)
NORMALIZED_SIDE = 360
_BORDER_FRACTION = 0.12
# Inset from a detected line center. This is wider than the equal-grid inset
# because the center sits in the middle of the stroke, not on its inner edge.
_LINE_INSET_FRACTION = 0.18


@dataclass(frozen=True, slots=True)
class CellPatch:
    """One cell from a normalized grid.

    ``origin`` is the top-left ``(x, y)`` of this crop inside the normalized
    grid. ``row`` and ``col`` are zero-based grid coordinates.
    """

    row: int
    col: int
    image: np.ndarray
    origin: tuple[int, int]
    size: tuple[int, int]


def segment_cells(warped: np.ndarray, order: int) -> tuple[CellPatch, ...]:
    """Divide ``warped`` into ``order`` by ``order`` cells.

    ``order`` must be 4, 6, or 9. The normalized image must be square and
    evenly divisible by ``order``. Each returned image is a copy.
    """
    if isinstance(order, bool) or not isinstance(order, int):
        raise CellSegmentationError("cell order must be 4, 6, or 9")
    if order not in SUPPORTED_ORDERS:
        raise CellSegmentationError("cell order must be 4, 6, or 9")
    if not isinstance(warped, np.ndarray) or warped.size == 0:
        raise CellSegmentationError("normalized grid is empty")
    height, width = warped.shape[:2]
    if height != width:
        raise CellSegmentationError("normalized grid must be square")
    if height % order != 0:
        raise CellSegmentationError(
            f"normalized side {height} is not divisible by order {order}"
        )

    cell_side = height // order
    inset = max(2, int(cell_side * _BORDER_FRACTION))
    if inset * 2 >= cell_side:
        raise CellSegmentationError("cell inset removed the entire cell")

    patches: list[CellPatch] = []
    for row in range(order):
        for col in range(order):
            x0 = col * cell_side + inset
            y0 = row * cell_side + inset
            x1 = (col + 1) * cell_side - inset
            y1 = (row + 1) * cell_side - inset
            crop = warped[y0:y1, x0:x1]
            patches.append(
                CellPatch(
                    row=row,
                    col=col,
                    image=np.ascontiguousarray(crop.copy()),
                    origin=(x0, y0),
                    size=(x1 - x0, y1 - y0),
                )
            )
    return tuple(patches)


def segment_cells_at_lines(
    warped: np.ndarray,
    horizontal_centers: Sequence[float],
    vertical_centers: Sequence[float],
) -> tuple[CellPatch, ...]:
    """Cut one cell from each gap between detected grid lines.

    Both center lists must contain ``order + 1`` positions for the same
    supported order. The crop is inset from each line center so the stroke
    itself is left outside the cell. Each returned image is a copy.
    """
    _require_image(warped)
    horizontal = _require_centers(horizontal_centers, "horizontal")
    vertical = _require_centers(vertical_centers, "vertical")
    if len(horizontal) != len(vertical):
        raise CellSegmentationError("horizontal and vertical line counts differ")
    order = len(horizontal) - 1
    if order not in SUPPORTED_ORDERS:
        raise CellSegmentationError("cell order must be 4, 6, or 9")

    height, width = warped.shape[:2]
    patches: list[CellPatch] = []
    for row in range(order):
        y0, y1 = _inset_span(horizontal[row], horizontal[row + 1], height)
        for col in range(order):
            x0, x1 = _inset_span(vertical[col], vertical[col + 1], width)
            crop = warped[y0:y1, x0:x1]
            patches.append(
                CellPatch(
                    row=row,
                    col=col,
                    image=np.ascontiguousarray(crop.copy()),
                    origin=(x0, y0),
                    size=(x1 - x0, y1 - y0),
                )
            )
    return tuple(patches)


def _require_image(warped: np.ndarray) -> None:
    if not isinstance(warped, np.ndarray) or warped.size == 0:
        raise CellSegmentationError("normalized grid is empty")
    height, width = warped.shape[:2]
    if height != width:
        raise CellSegmentationError("normalized grid must be square")


def _require_centers(centers: Sequence[float], axis: str) -> tuple[float, ...]:
    if isinstance(centers, (str, bytes)) or len(centers) < 2:
        raise CellSegmentationError(f"{axis} line centers are not a grid")
    values = tuple(float(center) for center in centers)
    pairs = zip(values, values[1:], strict=False)
    if any(right <= left for left, right in pairs):
        raise CellSegmentationError(f"{axis} line centers are not increasing")
    return values


def _inset_span(start: float, end: float, limit: int) -> tuple[int, int]:
    gap = end - start
    inset = max(2, int(gap * _LINE_INSET_FRACTION))
    if inset * 2 >= gap:
        raise CellSegmentationError("cell inset removed the entire cell")
    left = int(np.clip(int(round(start)) + inset, 0, limit - 1))
    right = int(np.clip(int(round(end)) - inset, left + 1, limit))
    if right - left < 2:
        raise CellSegmentationError("cell inset removed the entire cell")
    return left, right
