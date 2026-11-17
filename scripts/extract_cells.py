"""Extract Sudoku cells and record which ones look empty.

The report counts blank, occupied, and uncertain cells. It does not read
digits and it does not claim accuracy.

Example:
    python scripts/extract_cells.py sample_images/real \\
        --output evaluation_output/cell_extraction
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from app.recognition import classify_cells, render_contact_sheet
from app.vision import (
    VisionError,
    detect_grid,
    load_image,
    segment_cells,
    segment_cells_at_lines,
)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_NOTE = (
    "Blank, occupied, and uncertain are baseline cell labels. They are not "
    "digit identities and they are not an accuracy score."
)


def extract_directory(image_dir: Path, output_dir: Path) -> dict[str, object]:
    """Extract cells for every JPEG and PNG and write a JSON report."""
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {image_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for path in _list_images(image_dir):
        results.append(_extract_one(path, output_dir / path.name))
    report = _summary(image_dir, results)
    payload = json.dumps(report, indent=2) + "\n"
    (output_dir / "report.json").write_text(payload, encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, extract the directory, and print the report path."""
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
        help="directory for cell images, contact sheets, and report.json",
    )
    args = parser.parse_args(argv)
    try:
        report = extract_directory(args.images, args.output)
    except (FileNotFoundError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(args.output / "report.json")
    print(
        f"images={report['image_count']} "
        f"cells={report['cells_extracted']} "
        f"blank={report['blank_count']} "
        f"occupied={report['occupied_count']} "
        f"uncertain={report['uncertain_count']} "
        f"failed={report['failure_count']}"
    )
    return 0


def _extract_one(path: Path, destination: Path) -> dict[str, object]:
    started = time.perf_counter()
    try:
        loaded = load_image(path)
        detection = detect_grid(loaded.original)
        patches = _patches(detection)
        classes = classify_cells(patches)
    except VisionError as exc:
        return {
            "filename": path.name,
            "success": False,
            "order": None,
            "cell_count": 0,
            "blank_count": 0,
            "occupied_count": 0,
            "uncertain_count": 0,
            "elapsed_seconds": time.perf_counter() - started,
            "failure_reason": str(exc),
            "cells": [],
        }
    destination.mkdir(parents=True, exist_ok=True)
    _write_image(destination / "perspective_grid.png", detection.warped)
    cell_dir = destination / "cells"
    cell_dir.mkdir(parents=True, exist_ok=True)
    for patch in patches:
        name = f"r{patch.row:02d}_c{patch.col:02d}.png"
        _write_image(cell_dir / name, patch.image)
    sheet = render_contact_sheet(patches, classes)
    _write_image(destination / "contact_sheet.png", sheet)
    cell_rows = [item.to_report() for item in classes]
    image_report = {
        "filename": path.name,
        "order": detection.order,
        "cell_count": len(cell_rows),
        "cells": cell_rows,
    }
    (destination / "cells.json").write_text(
        json.dumps(image_report, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "filename": path.name,
        "success": True,
        "order": detection.order,
        "cell_count": len(cell_rows),
        "blank_count": sum(1 for item in classes if item.blank is True),
        "occupied_count": sum(1 for item in classes if item.blank is False),
        "uncertain_count": sum(1 for item in classes if item.blank is None),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "failure_reason": None,
        "cells": cell_rows,
    }


def _patches(detection: object) -> tuple:
    order = detection.order
    horizontal = detection.lines.horizontal.merged_centers
    vertical = detection.lines.vertical.merged_centers
    if len(horizontal) == order + 1 and len(vertical) == order + 1:
        return segment_cells_at_lines(detection.warped, horizontal, vertical)
    return segment_cells(detection.warped, order)


def _summary(image_dir: Path, results: list[dict[str, object]]) -> dict[str, object]:
    return {
        "note": _NOTE,
        "image_directory": str(image_dir),
        "image_count": len(results),
        "processed_count": sum(1 for result in results if result["success"]),
        "failure_count": sum(1 for result in results if not result["success"]),
        "cells_extracted": sum(int(result["cell_count"]) for result in results),
        "blank_count": sum(int(result["blank_count"]) for result in results),
        "occupied_count": sum(int(result["occupied_count"]) for result in results),
        "uncertain_count": sum(int(result["uncertain_count"]) for result in results),
        "results": results,
    }


def _write_image(path: Path, image: object) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"could not write {path.name}")


def _list_images(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )


if __name__ == "__main__":
    raise SystemExit(main())
