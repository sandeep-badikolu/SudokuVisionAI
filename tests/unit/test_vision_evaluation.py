"""The real-image evaluator records pipeline outputs without inventing scores."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.vision import detect_grid
from scripts.evaluate_vision import evaluate_directory, load_ground_truth, main
from tests.unit.vision_samples import draw_grid


def _write_png(directory: Path, name: str, order: int) -> Path:
    drawn = draw_grid(order)
    path = directory / name
    assert cv2.imwrite(str(path), drawn.image)
    return path


def test_successful_detection_saves_intermediates_and_measured_confidence(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    source = _write_png(images, "grid.png", 4)
    output = tmp_path / "out"
    report = evaluate_directory(images, output)
    result = report["results"][0]
    detection = detect_grid(cv2.imread(str(source)))
    saved = output / "grid.png"

    assert result["filename"] == "grid.png"
    assert result["success"] is True
    assert result["detected"] is True
    assert result["detected_order"] == 4
    assert result["confidence"] == pytest.approx(float(detection.confidence))
    assert result["failure_reason"] is None
    assert result["elapsed_seconds"] >= 0
    assert result["expected_order"] is None
    assert result["order_matches"] is None
    assert (saved / "original.png").is_file()
    assert (saved / "grayscale.png").is_file()
    assert (saved / "threshold.png").is_file()
    assert (saved / "edges.png").is_file()
    assert (saved / "grid_boundary.png").is_file()
    assert (saved / "perspective_grid.png").is_file()
    assert len(list((saved / "cells").glob("*.png"))) == 16
    assert "accuracy" not in report
    assert (output / "report.json").is_file()
    assert (output / "report.csv").is_file()


def test_rejected_image_has_a_reason_and_no_confidence(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    blank = images / "blank.png"
    blank_image = np.full((80, 80, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(blank), blank_image)
    report = evaluate_directory(images, tmp_path / "out")
    result = report["results"][0]

    assert result["success"] is False
    assert result["detected"] is False
    assert result["detected_order"] is None
    assert result["confidence"] is None
    assert result["failure_reason"]
    assert not (tmp_path / "out" / "blank.png" / "grid_boundary.png").exists()
    assert (tmp_path / "out" / "blank.png" / "grayscale.png").is_file()


def test_expected_order_is_compared_and_not_turned_into_accuracy(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    _write_png(images, "four.png", 4)
    truth = {"four.png": 9}
    report = evaluate_directory(images, tmp_path / "out", truth)
    result = report["results"][0]

    assert result["detected"] is True
    assert result["detected_order"] == 4
    assert result["expected_order"] == 9
    assert result["order_matches"] is False
    assert result["success"] is False
    assert "expected 9×9" in result["failure_reason"]
    assert result["confidence"] is not None
    assert report["images_with_expected_order"] == 1
    assert report["expected_order_match_count"] == 0
    assert "accuracy" not in report


def test_matching_expected_order_is_counted(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    _write_png(images, "nine.png", 9)
    report = evaluate_directory(images, tmp_path / "out", {"nine.png": 9})
    result = report["results"][0]

    assert result["success"] is True
    assert result["order_matches"] is True
    assert report["expected_order_match_count"] == 1
    assert len(list((tmp_path / "out" / "nine.png" / "cells").glob("*.png"))) == 81


def test_non_images_are_skipped_and_missing_ground_truth_is_reported(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "notes.txt").write_text("not an image", encoding="utf-8")
    report = evaluate_directory(images, tmp_path / "out", {"missing.png": 9})

    assert report["skipped"] == ["notes.txt"]
    assert report["results"][0]["filename"] == "missing.png"
    assert report["results"][0]["confidence"] is None
    assert "not found" in report["results"][0]["failure_reason"]


def test_ground_truth_csv_rejects_an_unsupported_order(tmp_path: Path) -> None:
    path = tmp_path / "truth.csv"
    path.write_text("filename,order\npuzzle.jpg,5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="4, 6, or 9"):
        load_ground_truth(path)


def test_command_writes_a_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    _write_png(images, "grid.jpg", 6)
    output = tmp_path / "out"
    code = main([str(images), "--output", str(output)])
    captured = capsys.readouterr()
    saved = json.loads((output / "report.json").read_text(encoding="utf-8"))

    assert code == 0
    assert "report.json" in captured.out
    assert saved["results"][0]["detected_order"] == 6
    assert saved["results"][0]["success"] is True
