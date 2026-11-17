"""Load JPG, JPEG, and PNG images without changing the source file."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.vision.errors import ImageInputError

_JPEG_SUFFIXES = {".jpg", ".jpeg"}
_PNG_SUFFIX = ".png"
_SUPPORTED_SUFFIXES = _JPEG_SUFFIXES | {_PNG_SUFFIX}
_MIN_SIDE = 32


@dataclass(frozen=True, slots=True)
class LoadedImage:
    """A decoded BGR image and the name of the file it came from, if any.

    ``original`` is a copy. Preprocessing must not write into it, and loading
    does not write the source file.
    """

    original: np.ndarray
    source_name: str | None = None


def load_image(source: str | Path | bytes) -> LoadedImage:
    """Decode a PNG or JPEG from a path or from encoded bytes.

    The returned array is writable and independent of the source. GIF and
    other formats are rejected. An empty or undecodable file is rejected.
    """
    if isinstance(source, bytes):
        image = _decode(source, expected_format=None)
        return LoadedImage(original=image, source_name=None)

    path = Path(source)
    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise ImageInputError(
            f"unsupported image type {path.suffix!r}; expected JPG, JPEG, or PNG"
        )
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ImageInputError(f"could not read image {path.name}") from exc
    expected = "jpeg" if suffix in _JPEG_SUFFIXES else "png"
    image = _decode(payload, expected_format=expected)
    return LoadedImage(original=image, source_name=path.name)


def _decode(payload: bytes, *, expected_format: str | None) -> np.ndarray:
    if not payload:
        raise ImageInputError("image is empty")
    found = _sniff_format(payload)
    if found is None:
        raise ImageInputError("image is not a JPEG or PNG, or the file is corrupt")
    if expected_format is not None and found != expected_format:
        raise ImageInputError(
            f"file contents are {found}, which does not match the extension"
        )
    buffer = np.frombuffer(payload, dtype=np.uint8)
    decoded = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.size == 0:
        raise ImageInputError("image could not be decoded")
    image = _to_bgr(decoded)
    height, width = image.shape[:2]
    if height < _MIN_SIDE or width < _MIN_SIDE:
        raise ImageInputError(
            f"image is {width}×{height}; both sides must be at least {_MIN_SIDE} pixels"
        )
    return np.ascontiguousarray(image.copy())


def _sniff_format(payload: bytes) -> str | None:
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if payload.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    return None


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    raise ImageInputError("decoded image does not have 1, 3, or 4 channels")
