"""Isolate a printed digit and place it on a fixed canvas.

The same steps are applied to cell crops and to template images. Thin strokes
are kept: there is no morphological erosion.
"""

import cv2
import numpy as np

from app.recognition.errors import RecognitionError

CANVAS_SIDE = 32
_CONTENT_SIDE = 24
_BACKGROUND_PERCENTILE = 90.0
_INK_DELTA = 28.0
_BORDER_BAND_FRACTION = 0.16
_BORDER_INK_FRACTION = 0.72


def preprocess_digit(image: np.ndarray) -> np.ndarray:
    """Return a ``32×32`` float image with the digit ink in ``[0, 1]``.

    The digit keeps its aspect ratio and sits in the middle of the canvas.
    An empty result means no separable stroke was found.
    """
    gray = _as_gray(image)
    ink = _digit_mask(gray)
    if not np.any(ink):
        return np.zeros((CANVAS_SIDE, CANVAS_SIDE), dtype=np.float32)
    crop = _tight_crop(ink)
    fitted = _fit(crop)
    return _center(fitted)


def _as_gray(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise RecognitionError("digit image is empty")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] in {3, 4}:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise RecognitionError("digit image must be grayscale or BGR")


def _digit_mask(gray: np.ndarray) -> np.ndarray:
    background = float(np.percentile(gray, _BACKGROUND_PERCENTILE))
    ink = gray < (background - _INK_DELTA)
    mask = ink.astype(np.uint8)
    count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return np.zeros(gray.shape, dtype=bool)
    height, width = gray.shape[:2]
    band = max(2, int(_BORDER_BAND_FRACTION * min(height, width)))
    kept = np.zeros(gray.shape, dtype=bool)
    best_area = 0
    best_label = 0
    for label in range(1, count):
        ys, xs = np.where(labels == label)
        area = int(ys.size)
        if area == 0:
            continue
        near = (
            (xs < band)
            | (ys < band)
            | (xs >= width - band)
            | (ys >= height - band)
        )
        if float(near.mean()) >= _BORDER_INK_FRACTION:
            continue
        if area > best_area:
            best_area = area
            best_label = label
    if best_label == 0:
        return kept
    kept[labels == best_label] = True
    return kept


def _tight_crop(ink: np.ndarray) -> np.ndarray:
    ys, xs = np.where(ink)
    crop = ink[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    return crop.astype(np.float32)


def _fit(crop: np.ndarray) -> np.ndarray:
    height, width = crop.shape[:2]
    longest = max(height, width)
    if longest == _CONTENT_SIDE:
        return crop
    scale = _CONTENT_SIDE / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_LINEAR if scale >= 1 else cv2.INTER_AREA
    return cv2.resize(crop, (new_width, new_height), interpolation=interpolation)


def _center(fitted: np.ndarray) -> np.ndarray:
    canvas = np.zeros((CANVAS_SIDE, CANVAS_SIDE), dtype=np.float32)
    height, width = fitted.shape[:2]
    y = (CANVAS_SIDE - height) // 2
    x = (CANVAS_SIDE - width) // 2
    canvas[y : y + height, x : x + width] = np.clip(fitted, 0.0, 1.0)
    return canvas
