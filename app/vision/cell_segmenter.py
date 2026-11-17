"""Split a normalized square grid into padded cells.

The crop for each cell is inset from the grid lines so the border stroke is
not part of the cell image.
"""

from dataclasses import dataclass

import numpy as np

from app.vision.errors import CellSegmentationError

SUPPORTED_ORDERS = (4, 6, 9)
NORMALIZED_SIDE = 360
_BORDER_FRACTION = 0.12


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
