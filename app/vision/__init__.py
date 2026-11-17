"""Image loading, preprocessing, and Sudoku grid geometry.

This package does not recognize digits and does not solve puzzles.
"""

from app.vision.cell_segmenter import (
    SUPPORTED_ORDERS,
    CellPatch,
    segment_cells,
    segment_cells_at_lines,
)
from app.vision.errors import (
    CellSegmentationError,
    GridDetectionError,
    ImageInputError,
    VisionError,
)
from app.vision.grid_detector import GridDetection, detect_grid, extract_cells
from app.vision.image_loader import LoadedImage, load_image
from app.vision.image_preprocessor import PreprocessedImage, preprocess
from app.vision.perspective_transform import order_corners, warp_quad

__all__ = [
    "SUPPORTED_ORDERS",
    "CellPatch",
    "CellSegmentationError",
    "GridDetection",
    "GridDetectionError",
    "ImageInputError",
    "LoadedImage",
    "PreprocessedImage",
    "VisionError",
    "detect_grid",
    "extract_cells",
    "load_image",
    "order_corners",
    "preprocess",
    "segment_cells",
    "segment_cells_at_lines",
    "warp_quad",
]
