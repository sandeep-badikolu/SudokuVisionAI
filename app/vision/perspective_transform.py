"""Map a detected quadrilateral onto a square top-down view."""

import cv2
import numpy as np

from app.vision.errors import GridDetectionError


def order_corners(points: np.ndarray) -> np.ndarray:
    """Return corners as top-left, top-right, bottom-right, bottom-left.

    Ordering uses the image axes. It does not know which side of a photographed
    puzzle is the logical top; that needs digit recognition.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    total = pts.sum(axis=1)
    diff = pts[:, 0] - pts[:, 1]
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[int(np.argmin(total))]
    ordered[2] = pts[int(np.argmax(total))]
    ordered[1] = pts[int(np.argmax(diff))]
    ordered[3] = pts[int(np.argmin(diff))]
    return ordered


def warp_quad(image: np.ndarray, corners: np.ndarray, side: int) -> np.ndarray:
    """Warp ``corners`` to a ``side`` by ``side`` square.

    ``image`` is not modified. ``side`` must be at least 2.
    """
    if side < 2:
        raise GridDetectionError("normalized grid side must be at least 2 pixels")
    ordered = order_corners(corners)
    destination = np.array(
        [
            [0, 0],
            [side - 1, 0],
            [side - 1, side - 1],
            [0, side - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(
        image,
        matrix,
        (side, side),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
