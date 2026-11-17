"""JPEG and PNG loading rejects empty and corrupt input."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.vision import ImageInputError, load_image


def _sample() -> np.ndarray:
    image = np.full((48, 64, 3), 200, dtype=np.uint8)
    image[4, 4] = (10, 20, 30)
    return image


def test_png_path_loads_without_changing_the_file(tmp_path: Path) -> None:
    image = _sample()
    path = tmp_path / "grid.png"
    assert cv2.imwrite(str(path), image)
    loaded = load_image(path)
    loaded.original[0, 0] = 0

    assert loaded.source_name == "grid.png"
    again = load_image(path)
    assert np.array_equal(again.original, image)


def test_jpeg_bytes_load() -> None:
    image = _sample()
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    loaded = load_image(encoded.tobytes())
    assert loaded.source_name is None
    assert loaded.original.shape == image.shape
    assert loaded.original.dtype == np.uint8


def test_jpeg_suffix_is_accepted(tmp_path: Path) -> None:
    image = _sample()
    path = tmp_path / "photo.JPEG"
    assert cv2.imwrite(str(path), image)
    loaded = load_image(path)
    assert loaded.original.shape == image.shape


def test_extension_must_match_the_encoded_bytes(tmp_path: Path) -> None:
    image = _sample()
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path = tmp_path / "mislabeled.jpg"
    path.write_bytes(encoded.tobytes())
    with pytest.raises(ImageInputError, match="does not match"):
        load_image(path)


@pytest.mark.parametrize(
    "payload",
    [b"", b"GIF89a not-an-image", b"\xff\xd8\xff not-a-jpeg"],
)
def test_empty_and_corrupt_bytes_are_rejected(payload: bytes) -> None:
    with pytest.raises(ImageInputError):
        load_image(payload)


def test_unsupported_suffix_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "notes.gif"
    path.write_bytes(b"GIF89a")
    with pytest.raises(ImageInputError, match="unsupported image type"):
        load_image(path)


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ImageInputError, match="could not read"):
        load_image(tmp_path / "missing.png")
