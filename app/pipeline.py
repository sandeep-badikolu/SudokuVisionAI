"""Run the Sudoku photograph pipeline and score it when labels exist.

Detection, extraction, recognition, reconstruction, and solving stay in their
own modules. This module only connects them and writes reports. A missing
label file does not become an accuracy number.
"""

from __future__ import annotations

import csv
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.recognition.blank_cells import classify_cells, render_contact_sheet
from app.recognition.digit_recognizer import RecognitionStatus, recognize_cells
from app.recognition.digit_templates import (
    TemplateBank,
    builtin_template_bank,
    load_template_manifest,
    merge_banks,
)
from app.recognition.errors import RecognitionError
from app.recognition.grid_reconstruction import reconstruct
from app.vision.cell_segmenter import SUPPORTED_ORDERS
from app.vision.errors import VisionError
from app.vision.grid_detector import extract_cells
from app.vision.image_loader import load_image

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_NOTE = (
    "Each rate measures one stage. They are not combined into a single "
    "accuracy. Digit, classification, and reconstruction scores are present "
    "only for rows whose ground-truth label is 0–9."
)
_LABEL_FIELDS = ("image", "cell", "row", "column", "order", "digit")


@dataclass(frozen=True, slots=True)
class CellLabel:
    """One manually entered cell label. An empty digit is unlabelled."""

    image: str
    cell: str
    row: int
    column: int
    order: int | None
    digit: int | None
    marker: str | None
    source: Path | None


def run_directory(
    image_dir: Path,
    output_dir: Path,
    *,
    order: int | None = None,
    template_manifest: Path | None = None,
    use_builtin: bool = True,
    labels: Path | None = None,
) -> dict[str, object]:
    """Process every JPEG and PNG in ``image_dir``.

    One failed image is recorded and the rest continue. ``labels`` is an
    optional ground-truth CSV. It is never filled from the recognizer.
    """
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {image_dir}")
    _require_order(order)
    bank = _bank(template_manifest, use_builtin)
    parsed = None if labels is None else load_labels(labels)
    return _finish(
        image_dir,
        output_dir,
        _list_images(image_dir),
        order=order,
        bank=bank,
        labels=labels,
        parsed=parsed,
    )


def run_paths(
    paths: list[Path],
    output_dir: Path,
    *,
    order: int | None = None,
    template_manifest: Path | None = None,
    use_builtin: bool = True,
    labels: Path | None = None,
) -> dict[str, object]:
    """Process an explicit list of images. Missing files are recorded as failures."""
    _require_order(order)
    bank = _bank(template_manifest, use_builtin)
    parsed = None if labels is None else load_labels(labels)
    return _finish(
        paths[0].parent if paths else output_dir,
        output_dir,
        paths,
        order=order,
        bank=bank,
        labels=labels,
        parsed=parsed,
    )


def _finish(
    image_dir: Path,
    output_dir: Path,
    paths: list[Path],
    *,
    order: int | None,
    bank: TemplateBank,
    labels: Path | None,
    parsed: tuple[CellLabel, ...] | None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for path in paths:
        if not path.is_file():
            record = _empty_record(path.name)
            record["failure_reason"] = f"image does not exist: {path.name}"
            results.append(record)
            continue
        results.append(run_image(path, output_dir / path.name, order=order, bank=bank))
    template_paths = _template_paths(bank)
    metrics = build_metrics(
        results,
        parsed,
        template_paths=template_paths,
        label_root=None if labels is None else labels.parent,
        output_dir=output_dir,
    )
    report = {
        "note": _NOTE,
        "image_directory": str(image_dir),
        "image_count": len(results),
        "success_count": sum(1 for row in results if row["success"]),
        "failure_count": sum(1 for row in results if not row["success"]),
        "labels": None if labels is None else str(labels),
        "metrics": metrics,
        "failed_images": [
            {"filename": row["filename"], "failure_reason": row["failure_reason"]}
            for row in results
            if not row["success"]
        ],
        "uncertain_cells": _uncertain_index(results),
        "incorrect_predictions": metrics["incorrect_predictions"],
        "results": results,
    }
    _write_report(output_dir, report)
    write_label_template(results, output_dir / "labels_template.csv")
    return report


def run_image(
    path: Path,
    destination: Path,
    *,
    order: int | None = None,
    bank: TemplateBank | None = None,
) -> dict[str, object]:
    """Run one image through the full pipeline and save its artifacts."""
    started = time.perf_counter()
    record = _empty_record(path.name)
    try:
        loaded = load_image(path)
        destination.mkdir(parents=True, exist_ok=True)
        original = destination / f"original{path.suffix.lower()}"
        shutil.copy2(path, original)
        record["paths"]["original"] = original.name
        detection, patches = extract_cells(loaded.original, order=order)
        record["detected"] = True
        record["order"] = detection.order
        overlay = detection.diagnostics["overlay"]
        _write_image(destination / "detected_grid.png", overlay)
        _write_image(destination / "perspective_grid.png", detection.warped)
        record["paths"]["detected_grid"] = "detected_grid.png"
        record["paths"]["perspective_grid"] = "perspective_grid.png"
        cell_dir = destination / "cells"
        cell_dir.mkdir(parents=True, exist_ok=True)
        for patch in patches:
            name = f"r{patch.row:02d}_c{patch.col:02d}.png"
            _write_image(cell_dir / name, patch.image)
        record["cells_extracted"] = len(patches)
        record["paths"]["cells"] = "cells"
        classes = classify_cells(patches)
        sheet = render_contact_sheet(patches, classes)
        _write_image(destination / "contact_sheet.png", sheet)
        record["paths"]["contact_sheet"] = "contact_sheet.png"
        chosen = builtin_template_bank() if bank is None else bank
        recognized = recognize_cells(patches, order=detection.order, bank=chosen)
        built = reconstruct(detection.order, recognized)
        recognition_payload = [item.to_report() for item in recognized]
        (destination / "recognition.json").write_text(
            json.dumps(recognition_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        (destination / "reconstruction.json").write_text(
            json.dumps(built.to_report(), indent=2) + "\n",
            encoding="utf-8",
        )
        record["paths"]["recognition"] = "recognition.json"
        record["paths"]["reconstruction"] = "reconstruction.json"
        record["validation_status"] = built.validation_status
        record["solution_status"] = built.solution_status
        located = {(cell.row, cell.col): cell for cell in classes}
        record["cells"] = [_cell_row(item, located) for item in recognized]
        record["success"] = True
    except (VisionError, RecognitionError, ValueError) as exc:
        record["failure_reason"] = str(exc)
    record["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    return record


def write_label_template(results: list[dict[str, object]], path: Path) -> None:
    """Write a blank ground-truth CSV. Predictions are not copied into it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_LABEL_FIELDS))
        writer.writeheader()
        for result in results:
            if not result["success"]:
                continue
            image = str(result["filename"])
            order = result["order"]
            for cell in result["cells"]:  # type: ignore[union-attr]
                row = int(cell["row"])
                column = int(cell["col"])
                writer.writerow(
                    {
                        "image": image,
                        "cell": f"{image}/cells/r{row:02d}_c{column:02d}.png",
                        "row": row,
                        "column": column,
                        "order": order,
                        "digit": "",
                    }
                )


def load_labels(path: Path) -> tuple[CellLabel, ...]:
    """Read ground truth. Empty ``digit`` stays unlabelled, not blank."""
    if not path.is_file():
        raise FileNotFoundError(f"ground-truth file does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = {"image", "row", "column", "digit"}
        if not required.issubset(set(fieldnames)):
            raise ValueError("ground truth CSV must have image, row, column, and digit")
        labels: list[CellLabel] = []
        for row_number, row in enumerate(reader, start=2):
            labels.append(_parse_label(row, row_number))
    return tuple(labels)


def build_metrics(
    results: list[dict[str, object]],
    labels: tuple[CellLabel, ...] | None,
    *,
    template_paths: set[Path],
    label_root: Path | None,
    output_dir: Path,
) -> dict[str, object]:
    """Stage rates, plus digit scores when labelled cells are present."""
    detected = [row for row in results if row["detected"]]
    extracted = [
        row
        for row in detected
        if row["order"] and row["cells_extracted"] == int(row["order"]) ** 2
    ]
    reconstructed = [row for row in results if row["success"]]
    valid = [row for row in reconstructed if row["validation_status"] == "valid"]
    eligible = [
        row for row in reconstructed if row["solution_status"] != "not_attempted"
    ]
    unique = [row for row in eligible if row["solution_status"] == "unique"]
    recognized_cells = [cell for row in reconstructed for cell in row["cells"]]  # type: ignore[union-attr]
    uncertain = [cell for cell in recognized_cells if cell["status"] == "uncertain"]
    metrics: dict[str, object] = {
        "note": _NOTE,
        "ground_truth": labels is not None,
        "grid_detection": {
            "images": len(results),
            "detected": len(detected),
            "rate": _rate(len(detected), len(results)),
        },
        "cell_extraction": {
            "detected": len(detected),
            "extracted": len(extracted),
            "rate": _rate(len(extracted), len(detected)),
        },
        "recognition_uncertain": {
            "cells": len(recognized_cells),
            "uncertain": len(uncertain),
            "rate": _rate(len(uncertain), len(recognized_cells)),
        },
        "validation": {
            "reconstructed": len(reconstructed),
            "valid": len(valid),
            "invalid": sum(
                row["validation_status"] == "invalid" for row in reconstructed
            ),
            "needs_correction": sum(
                row["validation_status"] == "needs_correction" for row in reconstructed
            ),
            "pass_rate": _rate(len(valid), len(reconstructed)),
        },
        "solver": {
            "eligible": len(eligible),
            "unique": len(unique),
            "unsolvable": sum(
                row["solution_status"] == "unsolvable" for row in eligible
            ),
            "multiple": sum(row["solution_status"] == "multiple" for row in eligible),
            "completion_rate": _rate(len(unique), len(eligible)),
        },
        "blank_occupied": None,
        "digit_recognition": None,
        "reconstruction": None,
        "leakage_count": 0,
        "incorrect_predictions": [],
    }
    if labels is None:
        return metrics
    comparisons, leakage = _comparisons(
        results, labels, template_paths, label_root, output_dir
    )
    metrics["leakage_count"] = leakage
    metrics["blank_occupied"] = _occupancy_metrics(comparisons)
    metrics["digit_recognition"] = _digit_metrics(comparisons)
    metrics["reconstruction"] = _reconstruction_metrics(comparisons, results)
    metrics["incorrect_predictions"] = [
        {
            "image": item["image"],
            "row": item["row"],
            "column": item["column"],
            "expected": item["expected"],
            "predicted": item["predicted"],
            "status": item["status"],
        }
        for item in comparisons
        if (
            item["digit_label"]
            and item["status"] != RecognitionStatus.UNCERTAIN
            and item["predicted"] != str(item["expected"])
        )
    ]
    return metrics


def _cell_row(
    item: object, classes: dict[tuple[int, int], object]
) -> dict[str, object]:
    occupancy = classes[(item.row, item.col)].status.value  # type: ignore[attr-defined]
    return {
        "row": item.row,  # type: ignore[attr-defined]
        "col": item.col,  # type: ignore[attr-defined]
        "occupancy": occupancy,
        "status": item.status,  # type: ignore[attr-defined]
        "digit": item.digit,  # type: ignore[attr-defined]
        "confidence": item.confidence,  # type: ignore[attr-defined]
        "blank_confidence": item.blank_confidence,  # type: ignore[attr-defined]
    }


def _comparisons(
    results: list[dict[str, object]],
    labels: tuple[CellLabel, ...],
    template_paths: set[Path],
    label_root: Path | None,
    output_dir: Path,
) -> tuple[list[dict[str, object]], int]:
    by_image = {str(row["filename"]): row for row in results}
    compared: list[dict[str, object]] = []
    leakage = 0
    for label in labels:
        if label.digit is None and label.marker is None:
            continue
        located = _resolve_label_cell(label, label_root, output_dir)
        if located is not None and located in template_paths:
            leakage += 1
            continue
        image = by_image.get(label.image)
        if image is None or not image["success"]:
            continue
        cell = next(
            (
                item
                for item in image["cells"]  # type: ignore[union-attr]
                if item["row"] == label.row and item["col"] == label.column
            ),
            None,
        )
        if cell is None:
            continue
        predicted = _predicted_name(cell)
        compared.append(
            {
                "image": label.image,
                "row": label.row,
                "column": label.column,
                "expected": label.digit,
                "marker": label.marker,
                "digit_label": label.digit is not None,
                "predicted": predicted,
                "status": cell["status"],
                "occupancy": cell["occupancy"],
                "value": (
                    None
                    if cell["status"] == RecognitionStatus.UNCERTAIN
                    else cell["digit"]
                ),
            }
        )
    return compared, leakage


def _occupancy_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    scored = [row for row in rows if row["digit_label"]]
    counts = {
        "blank": {"tp": 0, "fp": 0, "fn": 0},
        "occupied": {"tp": 0, "fp": 0, "fn": 0},
    }
    uncertain = 0
    for row in scored:
        expected = "blank" if row["expected"] == 0 else "occupied"
        predicted = str(row["occupancy"])
        if predicted == "uncertain":
            uncertain += 1
            counts[expected]["fn"] += 1
            continue
        if predicted == expected:
            counts[expected]["tp"] += 1
        else:
            counts[expected]["fn"] += 1
            if predicted in counts:
                counts[predicted]["fp"] += 1
    return {
        "compared": len(scored),
        "occupancy_uncertain": uncertain,
        "blank": _class_rates(counts["blank"]),
        "occupied": _class_rates(counts["occupied"]),
    }


def _digit_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    scored = [row for row in rows if row["digit_label"]]
    if not scored:
        return {
            "compared": 0,
            "correct": 0,
            "accuracy": None,
            "uncertain": 0,
            "per_digit": {},
            "confusion": {},
        }
    confusion: dict[str, dict[str, int]] = {}
    correct = 0
    uncertain = 0
    for row in scored:
        expected = str(row["expected"])
        predicted = str(row["predicted"])
        if predicted == "uncertain":
            uncertain += 1
        confusion.setdefault(expected, {})
        confusion[expected][predicted] = confusion[expected].get(predicted, 0) + 1
        if predicted == expected:
            correct += 1
    per_digit = {}
    for digit in range(10):
        key = str(digit)
        support = sum(confusion.get(key, {}).values())
        tp = confusion.get(key, {}).get(key, 0)
        predicted_count = sum(bucket.get(key, 0) for bucket in confusion.values())
        fp = predicted_count - tp
        fn = support - tp
        per_digit[key] = {
            "support": support,
            "correct": tp,
            "precision": _divide(tp, tp + fp),
            "recall": _divide(tp, tp + fn),
            "accuracy": _divide(tp, support),
        }
    return {
        "compared": len(scored),
        "correct": correct,
        "accuracy": _divide(correct, len(scored)),
        "uncertain": uncertain,
        "per_digit": per_digit,
        "confusion": confusion,
    }


def _reconstruction_metrics(
    rows: list[dict[str, object]], results: list[dict[str, object]]
) -> dict[str, object]:
    scored = [row for row in rows if row["digit_label"]]
    matches = 0
    for row in scored:
        value = row["value"]
        if row["status"] == RecognitionStatus.BLANK:
            value = 0
        if value == row["expected"]:
            matches += 1
    by_image: dict[str, list[dict[str, object]]] = {}
    for row in scored:
        by_image.setdefault(str(row["image"]), []).append(row)
    full = 0
    exact = 0
    orders = {str(row["filename"]): row["order"] for row in results}
    for image, group in by_image.items():
        order = orders.get(image)
        if not isinstance(order, int) or len(group) != order * order:
            continue
        full += 1
        if all(_reconstructed_value(item) == item["expected"] for item in group):
            exact += 1
    return {
        "cells_compared": len(scored),
        "cells_correct": matches,
        "cell_accuracy": _divide(matches, len(scored)),
        "fully_labelled_puzzles": full,
        "puzzles_exact": exact,
        "puzzle_exact_rate": _divide(exact, full),
    }


def _reconstructed_value(row: dict[str, object]) -> object:
    if row["status"] == RecognitionStatus.BLANK:
        return 0
    if row["status"] == RecognitionStatus.UNCERTAIN:
        return None
    return row["value"]


def _predicted_name(cell: dict[str, object]) -> str:
    if cell["status"] == RecognitionStatus.UNCERTAIN or cell["digit"] is None:
        return "uncertain"
    return str(cell["digit"])


def _class_rates(counts: dict[str, int]) -> dict[str, object]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _divide(tp, tp + fp),
        "recall": _divide(tp, tp + fn),
    }


def _parse_label(row: dict[str, str], row_number: int) -> CellLabel:
    image = (row.get("image") or "").strip()
    if not image:
        raise ValueError(f"ground truth row {row_number} has an empty image")
    try:
        grid_row = int((row.get("row") or "").strip())
        column = int((row.get("column") or "").strip())
    except ValueError as exc:
        raise ValueError(
            f"ground truth row {row_number} has a non-integer coordinate"
        ) from exc
    order = _optional_order(row.get("order") or "", row_number)
    digit, marker = _parse_digit(row.get("digit") or "", row_number, order)
    return CellLabel(
        image=image,
        cell=(row.get("cell") or "").strip(),
        row=grid_row,
        column=column,
        order=order,
        digit=digit,
        marker=marker,
        source=None,
    )


def _parse_digit(
    raw: str, row_number: int, order: int | None
) -> tuple[int | None, str | None]:
    text = raw.strip().lower()
    if not text:
        return None, None
    if text in {"uncertain", "invalid"}:
        return None, text
    if text in {"true", "false"}:
        raise ValueError(f"ground truth row {row_number} digit is invalid")
    try:
        digit = int(text)
    except ValueError as exc:
        raise ValueError(f"ground truth row {row_number} digit is invalid") from exc
    if digit < 0 or digit > 9:
        raise ValueError(f"ground truth row {row_number} digit must be 0–9")
    if order is not None and digit > order:
        raise ValueError(
            f"ground truth row {row_number} digit {digit} "
            f"is outside a {order}×{order} grid"
        )
    return digit, None


def _optional_order(raw: str, row_number: int) -> int | None:
    text = raw.strip()
    if not text:
        return None
    try:
        order = int(text)
    except ValueError as exc:
        raise ValueError(f"ground truth row {row_number} order is invalid") from exc
    if order not in SUPPORTED_ORDERS:
        raise ValueError(f"ground truth row {row_number} order must be 4, 6, or 9")
    return order


def _resolve_label_cell(
    label: CellLabel, label_root: Path | None, output_dir: Path
) -> Path | None:
    if not label.cell:
        return None
    raw = Path(label.cell)
    if raw.is_absolute():
        return raw.resolve()
    roots = [output_dir] if label_root is None else [label_root, output_dir]
    for root in roots:
        candidate = (root / raw).resolve()
        if candidate.is_file():
            return candidate
    return (roots[0] / raw).resolve()


def _bank(template_manifest: Path | None, use_builtin: bool) -> TemplateBank:
    banks: list[TemplateBank] = []
    if use_builtin:
        banks.append(builtin_template_bank())
    if template_manifest is not None:
        banks.append(load_template_manifest(template_manifest))
    if not banks:
        raise RecognitionError("no templates were selected")
    return merge_banks(*banks)


def _template_paths(bank: TemplateBank) -> set[Path]:
    return {Path(source) for source in bank.sources if Path(source).is_absolute()}


def _uncertain_index(results: list[dict[str, object]]) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    for result in results:
        for cell in result["cells"]:  # type: ignore[union-attr]
            if cell["status"] == RecognitionStatus.UNCERTAIN:
                found.append(
                    {
                        "image": result["filename"],
                        "row": cell["row"],
                        "column": cell["col"],
                    }
                )
    return found


def _empty_record(filename: str) -> dict[str, object]:
    return {
        "filename": filename,
        "success": False,
        "detected": False,
        "order": None,
        "cells_extracted": 0,
        "validation_status": None,
        "solution_status": None,
        "failure_reason": None,
        "elapsed_seconds": 0.0,
        "paths": {},
        "cells": [],
    }


def _require_order(order: int | None) -> None:
    if order is None:
        return
    if (
        isinstance(order, bool)
        or not isinstance(order, int)
        or order not in SUPPORTED_ORDERS
    ):
        raise ValueError("order must be 4, 6, or 9")


def _write_report(output_dir: Path, report: dict[str, object]) -> None:
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    fieldnames = [
        "filename",
        "success",
        "detected",
        "order",
        "cells_extracted",
        "validation_status",
        "solution_status",
        "failure_reason",
        "elapsed_seconds",
    ]
    with (output_dir / "report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["results"]:  # type: ignore[union-attr]
            writer.writerow({name: row.get(name) for name in fieldnames})


def _write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"could not write {path.name}")


def _list_images(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return _divide(numerator, denominator)


def _divide(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)
