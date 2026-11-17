"""Run the vision pipeline on a directory of real Sudoku photographs.

The report records detection outcomes returned by the pipeline. It does not
score digits and it does not invent an accuracy percentage.

Example:
    python scripts/evaluate_vision.py sample_images/real --output evaluation_output
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from app.vision import (
    SUPPORTED_ORDERS,
    VisionError,
    detect_grid,
    load_image,
    preprocess,
    segment_cells,
)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_NOTE = (
    "This report records grid detection only. A match between the detected "
    "order and a supplied expected order is not digit accuracy and is not a "
    "photograph benchmark."
)


@dataclass(frozen=True, slots=True)
class ImageResult:
    """One image's measured detection outcome."""

    filename: str
    success: bool
    detected: bool
    detected_order: int | None
    expected_order: int | None
    order_matches: bool | None
    confidence: float | None
    elapsed_seconds: float
    failure_reason: str | None


def evaluate_directory(
    image_dir: Path,
    output_dir: Path,
    ground_truth: dict[str, int] | None = None,
) -> dict[str, object]:
    """Evaluate every JPEG and PNG in ``image_dir`` and write a report.

    Other files are listed as skipped. ``ground_truth`` maps a filename to
    an expected grid order of 4, 6, or 9. Images without an entry are still
    detected; ``order_matches`` stays null for them.
    """
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {image_dir}")
    expected = {} if ground_truth is None else dict(ground_truth)
    _validate_expected_orders(expected)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = _list_images(image_dir)
    skipped = _list_skipped(image_dir)
    present = {path.name for path in images}
    missing = sorted(name for name in expected if name not in present)
    results: list[ImageResult] = []
    for path in images:
        results.append(
            _evaluate_one(path, output_dir / path.name, expected.get(path.name))
        )
    for name in missing:
        results.append(
            ImageResult(
                filename=name,
                success=False,
                detected=False,
                detected_order=None,
                expected_order=expected[name],
                order_matches=None,
                confidence=None,
                elapsed_seconds=0.0,
                failure_reason=(
                    "listed in ground truth but not found in the image directory"
                ),
            )
        )

    report = _report(image_dir, results, skipped)
    _write_report(output_dir, report)
    return report


def load_ground_truth(path: Path) -> dict[str, int]:
    """Read ``filename,order`` rows. ``order`` must be 4, 6, or 9."""
    if not path.is_file():
        raise FileNotFoundError(f"ground-truth file does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "filename" not in fieldnames or "order" not in fieldnames:
            raise ValueError("ground truth CSV must have filename and order columns")
        expected: dict[str, int] = {}
        for row_number, row in enumerate(reader, start=2):
            filename = (row.get("filename") or "").strip()
            raw_order = (row.get("order") or "").strip()
            if not filename:
                raise ValueError(f"ground truth row {row_number} has an empty filename")
            if filename in expected:
                raise ValueError(f"ground truth lists {filename} more than once")
            expected[filename] = _parse_order(raw_order, row_number)
    return expected


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, evaluate the directory, and print the report path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "images",
        type=Path,
        help="directory of JPG, JPEG, and PNG files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="directory for intermediate images and report.json",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        help="CSV with filename,order columns; order is 4, 6, or 9",
    )
    args = parser.parse_args(argv)
    try:
        if args.ground_truth is None:
            expected = None
        else:
            expected = load_ground_truth(args.ground_truth)
        report = evaluate_directory(args.images, args.output, expected)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(args.output / "report.json")
    print(
        f"images={report['image_count']} "
        f"detected={report['detected_count']} "
        f"failed={report['failure_count']}"
    )
    return 0


def _evaluate_one(
    path: Path, destination: Path, expected_order: int | None
) -> ImageResult:
    started = time.perf_counter()
    detected = False
    detected_order: int | None = None
    confidence: float | None = None
    failure_reason: str | None = None
    success = False
    try:
        loaded = load_image(path)
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination / f"original{path.suffix.lower()}")
        prepared = preprocess(loaded.original)
        _write_image(destination / "grayscale.png", prepared.gray)
        _write_image(destination / "threshold.png", prepared.threshold)
        _write_image(destination / "edges.png", prepared.edges)
        detection = detect_grid(loaded.original)
        detected = True
        detected_order = detection.order
        confidence = float(detection.confidence)
        overlay = detection.diagnostics["overlay"]
        _write_image(destination / "grid_boundary.png", overlay)
        _write_image(destination / "perspective_grid.png", detection.warped)
        cells = segment_cells(detection.warped, detection.order)
        cell_dir = destination / "cells"
        cell_dir.mkdir(parents=True, exist_ok=True)
        for patch in cells:
            name = f"r{patch.row:02d}_c{patch.col:02d}.png"
            _write_image(cell_dir / name, patch.image)
        if expected_order is not None and expected_order != detection.order:
            failure_reason = (
                f"detected {detection.order}×{detection.order}, "
                f"expected {expected_order}×{expected_order}"
            )
        else:
            success = True
    except VisionError as exc:
        failure_reason = str(exc)
    elapsed = time.perf_counter() - started
    return ImageResult(
        filename=path.name,
        success=success,
        detected=detected,
        detected_order=detected_order,
        expected_order=expected_order,
        order_matches=_order_matches(expected_order, detected_order),
        confidence=confidence,
        elapsed_seconds=elapsed,
        failure_reason=failure_reason,
    )


def _order_matches(
    expected_order: int | None, detected_order: int | None
) -> bool | None:
    if expected_order is None or detected_order is None:
        return None
    return expected_order == detected_order


def _report(
    image_dir: Path, results: list[ImageResult], skipped: list[str]
) -> dict[str, object]:
    detected_count = sum(1 for result in results if result.detected)
    failure_count = sum(1 for result in results if not result.success)
    compared = [result for result in results if result.expected_order is not None]
    matched = sum(1 for result in compared if result.order_matches is True)
    return {
        "note": _NOTE,
        "image_directory": str(image_dir),
        "image_count": len(results),
        "detected_count": detected_count,
        "failure_count": failure_count,
        "images_with_expected_order": len(compared),
        "expected_order_match_count": matched,
        "skipped": skipped,
        "results": [asdict(result) for result in results],
    }


def _write_report(output_dir: Path, report: dict[str, object]) -> None:
    payload = json.dumps(report, indent=2) + "\n"
    (output_dir / "report.json").write_text(payload, encoding="utf-8")
    fieldnames = [
        "filename",
        "success",
        "detected",
        "detected_order",
        "expected_order",
        "order_matches",
        "confidence",
        "elapsed_seconds",
        "failure_reason",
    ]
    with (output_dir / "report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["results"]:
            writer.writerow({name: row[name] for name in fieldnames})  # type: ignore[index]


def _write_image(path: Path, image: object) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"could not write {path.name}")


def _list_images(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )


def _list_skipped(image_dir: Path) -> list[str]:
    return sorted(
        path.name
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() not in _IMAGE_SUFFIXES
    )


def _validate_expected_orders(expected: dict[str, int]) -> None:
    for filename, order in expected.items():
        if order not in SUPPORTED_ORDERS:
            raise ValueError(
                f"ground truth order for {filename} must be 4, 6, or 9, got {order}"
            )


def _parse_order(raw_order: str, row_number: int) -> int:
    if raw_order.lower() in {"true", "false"}:
        raise ValueError(f"ground truth row {row_number} has a non-integer order")
    try:
        order = int(raw_order)
    except ValueError as exc:
        raise ValueError(
            f"ground truth row {row_number} has a non-integer order"
        ) from exc
    if order not in SUPPORTED_ORDERS:
        raise ValueError(f"ground truth row {row_number} order must be 4, 6, or 9")
    return order


if __name__ == "__main__":
    raise SystemExit(main())
