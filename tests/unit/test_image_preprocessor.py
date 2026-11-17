"""Preprocessing returns new arrays and does not write the input."""

import numpy as np

from app.vision.image_preprocessor import (
    adaptive_threshold,
    detect_edges,
    gaussian_blur,
    preprocess,
    reduce_noise,
    to_grayscale,
)


def test_grayscale_conversion_drops_color_channels() -> None:
    image = np.zeros((30, 40, 3), dtype=np.uint8)
    image[..., 1] = 255
    gray = to_grayscale(image)
    assert gray.shape == (30, 40)
    assert gray.ndim == 2
    assert int(gray[0, 0]) > 0


def test_preprocess_keeps_the_callers_image_intact() -> None:
    image = np.full((48, 48, 3), 255, dtype=np.uint8)
    image[10:30, 24] = 0
    before = image.copy()
    prepared = preprocess(image)

    assert np.array_equal(image, before)
    assert prepared.gray.ndim == 2
    assert prepared.denoised.shape == prepared.gray.shape
    assert prepared.blurred.shape == prepared.gray.shape
    assert set(np.unique(prepared.threshold)).issubset({0, 255})
    assert set(np.unique(prepared.edges)).issubset({0, 255})


def test_gaussian_blur_softens_a_hard_edge() -> None:
    gray = np.full((21, 21), 255, dtype=np.uint8)
    gray[:, :10] = 0
    blurred = gaussian_blur(gray)
    assert blurred.shape == gray.shape
    assert 0 < int(blurred[10, 10]) < 255


def test_threshold_and_edges_are_binary() -> None:
    gray = np.full((48, 48), 255, dtype=np.uint8)
    gray[:, :24] = 0
    denoised = reduce_noise(gray)
    threshold = adaptive_threshold(gaussian_blur(denoised))
    edges = detect_edges(gaussian_blur(denoised))
    assert set(np.unique(threshold)).issubset({0, 255})
    assert set(np.unique(edges)).issubset({0, 255})
    assert int(edges.sum()) > 0
