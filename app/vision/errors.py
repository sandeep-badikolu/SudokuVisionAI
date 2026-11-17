"""Failures from image loading and grid detection.

These errors stay inside the vision package. They are not solver failures.
"""


class VisionError(Exception):
    """Base class for computer-vision failures."""


class ImageInputError(VisionError):
    """The image is missing, empty, corrupt, or not a supported type."""


class GridDetectionError(VisionError):
    """No quadrilateral passed the geometry, confidence, and grid-line checks.

    ``boundary`` is set when a quadrilateral was accepted and warped, but the
    line evidence did not validate a 4, 6, or 9 grid. It stays ``None`` when
    the boundary itself was rejected. A set ``boundary`` is not a detected grid.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.boundary = None


class CellSegmentationError(VisionError):
    """The normalized grid could not be split into the requested cells."""
