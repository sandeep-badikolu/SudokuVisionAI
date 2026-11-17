"""Synthetic grids for vision tests. These are drawings, not photographs."""

from dataclasses import dataclass

import cv2
import numpy as np

_BOXES = {4: (2, 2), 6: (2, 3), 9: (3, 3)}


@dataclass(frozen=True, slots=True)
class DrawnGrid:
    """An axis-aligned grid drawn on a white canvas."""

    image: np.ndarray
    order: int
    cell: int
    margin: int


def draw_grid(
    order: int, *, cell: int = 48, margin: int = 40, line: int = 2
) -> DrawnGrid:
    """Draw a square Sudoku frame with box boundaries thicker than cell lines."""
    box_rows, box_cols = _BOXES[order]
    span = order * cell
    canvas = span + 2 * margin
    image = np.full((canvas, canvas, 3), 255, dtype=np.uint8)
    for index in range(order + 1):
        position = margin + index * cell
        horizontal = line + 2 if index % box_rows == 0 else line
        vertical = line + 2 if index % box_cols == 0 else line
        cv2.line(
            image,
            (margin, position),
            (margin + span, position),
            (0, 0, 0),
            horizontal,
        )
        cv2.line(
            image,
            (position, margin),
            (position, margin + span),
            (0, 0, 0),
            vertical,
        )
    return DrawnGrid(image=image, order=order, cell=cell, margin=margin)


def paint_cell(
    grid: DrawnGrid, row: int, col: int, color: tuple[int, int, int]
) -> None:
    """Fill the interior of one cell. The grid lines are left in place."""
    inset = max(4, grid.cell // 5)
    y0 = grid.margin + row * grid.cell + inset
    y1 = grid.margin + (row + 1) * grid.cell - inset
    x0 = grid.margin + col * grid.cell + inset
    x1 = grid.margin + (col + 1) * grid.cell - inset
    grid.image[y0:y1, x0:x1] = color


def skew(image: np.ndarray) -> np.ndarray:
    """Apply a mild perspective change that keeps the grid near the same orientation."""
    height, width = image.shape[:2]
    source = np.float32(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    )
    destination = np.float32(
        [
            [28, 18],
            [width - 36, 8],
            [width - 12, height - 20],
            [16, height - 28],
        ]
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        borderValue=(255, 255, 255),
    )


def shade(image: np.ndarray) -> np.ndarray:
    """Darken one side of the image to imitate uneven lighting."""
    gradient = np.linspace(0.55, 1.0, image.shape[1], dtype=np.float32)
    shaded = image.astype(np.float32) * gradient[None, :, None]
    return np.clip(shaded, 0, 255).astype(np.uint8)
