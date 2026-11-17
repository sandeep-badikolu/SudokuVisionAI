"""Grayscale, denoise, blur, threshold, and edge stages.

Every function returns a new array. The input image is not written.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from app.vision.errors import ImageInputError

_GAUSSIAN_KERNEL = (5, 5)
_MEDIAN_KERNEL = 3
_ADAPTIVE_BLOCK = 31
_ADAPTIVE_C = 8
_CANNY_LOW = 40
_CANNY_HIGH = 120


@dataclass(frozen=True, slots=True)
class PreprocessedImage:
    """Diagnostic images from one preprocessing pass.

    ``source`` is a copy of the BGR image that was read. Later stages are
    derived from that copy.
    """

    source: np.ndarray
    gray: np.ndarray
    denoised: np.ndarray
    blurred: np.ndarray
    threshold: np.ndarray
    edges: np.ndarray


def preprocess(image: np.ndarray) -> PreprocessedImage:
    """Run the full preprocessing chain on a copy of ``image``."""
    source = _require_bgr(image)
    gray = to_grayscale(source)
    denoised = reduce_noise(gray)
    blurred = gaussian_blur(denoised)
    threshold = adaptive_threshold(blurred)
    edges = detect_edges(blurred)
    return PreprocessedImage(
        source=source,
        gray=gray,
        denoised=denoised,
        blurred=blurred,
        threshold=threshold,
        edges=edges,
    )


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Return a single-channel 8-bit image."""
    if image.ndim == 2:
        _require_uint8(image)
        return image.copy()
    bgr = _require_bgr(image)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def reduce_noise(gray: np.ndarray) -> np.ndarray:
    """Apply a small median filter to suppress isolated speckle."""
    _require_gray(gray)
    return cv2.medianBlur(gray, _MEDIAN_KERNEL)


def gaussian_blur(gray: np.ndarray) -> np.ndarray:
    """Smooth the grayscale image before thresholding and edges."""
    _require_gray(gray)
    return cv2.GaussianBlur(gray, _GAUSSIAN_KERNEL, 0)


def adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    """Threshold under uneven lighting. Pixels are 0 or 255.

    Dark grid lines stay dark. The block size shrinks for small images so
    OpenCV's odd-block requirement still holds.
    """
    _require_gray(gray)
    block = _block_size(gray)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        _ADAPTIVE_C,
    )


def detect_edges(gray: np.ndarray) -> np.ndarray:
    """Return a Canny edge map. Pixels are 0 or 255."""
    _require_gray(gray)
    return cv2.Canny(gray, _CANNY_LOW, _CANNY_HIGH)


def _require_bgr(image: np.ndarray) -> np.ndarray:
    _require_uint8(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ImageInputError("expected a BGR image with shape (height, width, 3)")
    if min(image.shape[:2]) < 3:
        raise ImageInputError("image is too small to preprocess")
    return np.ascontiguousarray(image.copy())


def _require_gray(image: np.ndarray) -> None:
    _require_uint8(image)
    if image.ndim != 2:
        raise ImageInputError("expected a single-channel grayscale image")
    if min(image.shape) < 3:
        raise ImageInputError("image is too small to preprocess")


def _require_uint8(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.size == 0 or image.dtype != np.uint8:
        raise ImageInputError("expected a non-empty uint8 image")


def _block_size(gray: np.ndarray) -> int:
    limit = min(gray.shape) // 2
    block = min(_ADAPTIVE_BLOCK, limit if limit % 2 == 1 else limit - 1)
    return max(3, block)
