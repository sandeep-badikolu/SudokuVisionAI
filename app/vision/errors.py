"""Failures from image loading and grid detection.

These errors stay inside the vision package. They are not solver failures.
"""


class VisionError(Exception):
    """Base class for computer-vision failures."""


class ImageInputError(VisionError):
    """The image is missing, empty, corrupt, or not a supported type."""


class GridDetectionError(VisionError):
    """No quadrilateral passed the geometry, confidence, and grid-line checks."""


class CellSegmentationError(VisionError):
    """The normalized grid could not be split into the requested cells."""
