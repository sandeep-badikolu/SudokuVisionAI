"""Score template recognition against a labelled cell manifest.

Accuracy is computed only from the rows in the manifest. Cells that were
also used as templates are excluded.

Example:
    python scripts/evaluate_digits.py \\
        --labels labels.csv \\
        --output evaluation_output/digit_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

from app.recognition.digit_recognizer import RecognitionStatus, recognize_cell
from app.recognition.digit_templates import (
    builtin_template_bank,
    load_template_manifest,
    merge_banks,
)
from app.recognition.errors import RecognitionError
from app.vision.cell_segmenter import SUPPORTED_ORDERS

_NOTE = (
    "Accuracy counts only rows in the label manifest. An uncertain prediction "
    "is not counted as correct. This is not a score for unlabelled photographs."
)


def evaluate_labels(
    labels_path: Path,
    *,
    template_manifest: Path | None = None,
    use_builtin: bool = True,
) -> dict[str, object]:
    """Compare each labelled cell with the recognizer."""
    if not labels_path.is_file():
        raise FileNotFoundError(f"label manifest does not exist: {labels_path}")
    rows = _read_labels(labels_path)
    bank = _bank(template_manifest, use_builtin)
    template_paths = {
        Path(source) for source in bank.sources if Path(source).is_absolute()
    }
    root = labels_path.parent
    predictions: list[dict[str, object]] = []
    for row_number, row in rows:
        predictions.append(_score_row(row_number, row, root, bank, template_paths))
    return _summarize(labels_path, predictions)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, score the manifest, and write the report when asked."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="CSV of labelled cells",
    )
    parser.add_argument(
        "--templates",
        type=Path,
        help="CSV of extra templates; evaluation rows in it are not matched",
    )
    parser.add_argument(
        "--no-builtin",
        action="store_true",
        help="use only the templates listed in --templates",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="directory for report.json; omitted results are printed only",
    )
    args = parser.parse_args(argv)
    try:
        report = evaluate_labels(
            args.labels,
            template_manifest=args.templates,
            use_builtin=not args.no_builtin,
        )
    except (FileNotFoundError, RecognitionError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(report, indent=2) + "\n"
        (args.output / "report.json").write_text(payload, encoding="utf-8")
        print(args.output / "report.json")
    else:
        print(json.dumps(report, indent=2))
    scored = report["scored_count"]
    correct = report["correct_count"]
    accuracy = report["accuracy"]
    shown = "n/a" if accuracy is None else f"{accuracy:.3f}"
    print(
        f"scored={scored} correct={correct} accuracy={shown} "
        f"uncertain={report['uncertain_count']} "
        f"incorrect={report['incorrect_count']}"
    )
    return 0


def _bank(template_manifest: Path | None, use_builtin: bool):
    banks = []
    if use_builtin:
        banks.append(builtin_template_bank())
    if template_manifest is not None:
        banks.append(load_template_manifest(template_manifest))
    if not banks:
        raise RecognitionError("no templates were selected")
    return merge_banks(*banks)


def _score_row(
    row_number: int,
    row: dict[str, str],
    root: Path,
    bank: object,
    template_paths: set[Path],
) -> dict[str, object]:
    filename = (row.get("filename") or "").strip()
    if not filename:
        raise ValueError(f"label row {row_number} has an empty filename")
    expected, marker = _parse_expected(row.get("digit") or "", row_number)
    order = _parse_order(row.get("order") or "", row_number, 0 if marker else expected)
    image_path = (root / filename).resolve()
    source = (row.get("image") or "").strip()
    record: dict[str, object] = {
        "filename": filename,
        "path": str(image_path),
        "image": source or None,
        "expected": expected,
        "marker": marker,
        "order": order,
        "predicted": None,
        "status": None,
        "occupancy": None,
        "correct": False,
        "leakage": False,
        "failure_reason": None,
    }
    if image_path in template_paths:
        record["leakage"] = True
        record["failure_reason"] = "image is also a template"
        return record
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        record["failure_reason"] = "could not read image"
        return record
    try:
        result = recognize_cell(image, order=order, bank=bank)  # type: ignore[arg-type]
    except RecognitionError as exc:
        record["failure_reason"] = str(exc)
        return record
    record["predicted"] = result.digit
    record["status"] = result.status
    record["occupancy"] = result.diagnostics.get("occupancy")
    record["confidence"] = result.confidence
    record["blank_confidence"] = result.blank_confidence
    if marker is None:
        record["correct"] = (
            result.status == RecognitionStatus.RECOGNIZED and result.digit == expected
        ) or (result.status == RecognitionStatus.BLANK and expected == 0)
    return record


def _summarize(labels_path: Path, rows: list[dict[str, object]]) -> dict[str, object]:
    scored = [
        row
        for row in rows
        if (
            row["marker"] is None
            and not row["leakage"]
            and row["failure_reason"] is None
        )
    ]
    correct = [row for row in scored if row["correct"] is True]
    uncertain = [row for row in scored if row["status"] == RecognitionStatus.UNCERTAIN]
    incorrect = [
        row
        for row in scored
        if row["correct"] is False and row["status"] != RecognitionStatus.UNCERTAIN
    ]
    accuracy = None if not scored else round(len(correct) / len(scored), 4)
    return {
        "note": _NOTE,
        "labels": str(labels_path),
        "labelled_count": len(rows),
        "scored_count": len(scored),
        "correct_count": len(correct),
        "incorrect_count": len(incorrect),
        "uncertain_count": len(uncertain),
        "uncertain_rate": _divide(len(uncertain), len(scored)),
        "excluded_uncertain": sum(row["marker"] == "uncertain" for row in rows),
        "excluded_invalid": sum(row["marker"] == "invalid" for row in rows),
        "leakage_count": sum(1 for row in rows if row["leakage"]),
        "failure_count": sum(1 for row in rows if row["failure_reason"]),
        "accuracy": accuracy,
        "per_digit": _per_digit(scored),
        "blank_classification": _blank_classification(scored),
        "by_order": _by_order(scored),
        "by_image": _by_image(scored),
        "confusion": _confusion(scored),
        "incorrect": [_prediction_brief(row) for row in incorrect],
        "uncertain": [
            {
                "filename": row["filename"],
                "path": row["path"],
                "expected": row["expected"],
            }
            for row in uncertain
        ],
        "predictions": [_prediction_brief(row) for row in rows],
    }


def _per_digit(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    report: dict[str, dict[str, object]] = {}
    for digit in range(0, 10):
        group = [row for row in rows if row["expected"] == digit]
        if not group:
            continue
        hits = sum(1 for row in group if row["correct"] is True)
        predicted = sum(1 for row in rows if _predicted_name(row) == str(digit))
        report[str(digit)] = {
            "labelled": len(group),
            "correct": hits,
            "accuracy": round(hits / len(group), 4),
            "precision": _divide(hits, predicted),
            "recall": _divide(hits, len(group)),
        }
    return report


def _by_order(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    report: dict[str, dict[str, object]] = {}
    orders = {row["order"] for row in rows if row["order"] is not None}
    for order in sorted(orders, key=lambda value: int(value)):  # type: ignore[arg-type]
        group = [row for row in rows if row["order"] == order]
        hits = sum(1 for row in group if row["correct"] is True)
        report[str(order)] = {
            "labelled": len(group),
            "correct": hits,
            "uncertain": sum(
                1 for row in group if row["status"] == RecognitionStatus.UNCERTAIN
            ),
            "accuracy": round(hits / len(group), 4) if group else None,
        }
    return report


def _by_image(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    report: dict[str, dict[str, object]] = {}
    images = {row["image"] for row in rows if row["image"]}
    for image in sorted(str(item) for item in images):
        group = [row for row in rows if row["image"] == image]
        hits = sum(1 for row in group if row["correct"] is True)
        report[image] = {
            "labelled": len(group),
            "correct": hits,
            "uncertain": sum(
                1 for row in group if row["status"] == RecognitionStatus.UNCERTAIN
            ),
            "accuracy": _divide(hits, len(group)),
        }
    return report


def _blank_classification(rows: list[dict[str, object]]) -> dict[str, object]:
    counts = {
        "blank": {"tp": 0, "fp": 0, "fn": 0},
        "occupied": {"tp": 0, "fp": 0, "fn": 0},
    }
    uncertain = 0
    for row in rows:
        expected = "blank" if row["expected"] == 0 else "occupied"
        predicted = row["occupancy"]
        if predicted not in {"blank", "occupied"}:
            uncertain += 1
            counts[expected]["fn"] += 1
            continue
        if predicted == expected:
            counts[expected]["tp"] += 1
        else:
            counts[expected]["fn"] += 1
            counts[str(predicted)]["fp"] += 1
    return {
        "compared": len(rows),
        "occupancy_uncertain": uncertain,
        "blank": _class_rates(counts["blank"]),
        "occupied": _class_rates(counts["occupied"]),
    }


def _confusion(rows: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for row in rows:
        expected = str(row["expected"])
        predicted = _predicted_name(row)
        matrix.setdefault(expected, {})
        matrix[expected][predicted] = matrix[expected].get(predicted, 0) + 1
    return matrix


def _prediction_brief(row: dict[str, object]) -> dict[str, object]:
    return {
        "filename": row["filename"],
        "path": row["path"],
        "image": row["image"],
        "expected": row["expected"],
        "marker": row["marker"],
        "predicted": row["predicted"],
        "status": row["status"],
        "correct": row["correct"],
    }


def _predicted_name(row: dict[str, object]) -> str:
    if row["status"] == RecognitionStatus.UNCERTAIN or row["predicted"] is None:
        return "uncertain"
    return str(row["predicted"])


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


def _divide(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def _read_labels(path: Path) -> list[tuple[int, dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "filename" not in fieldnames or "digit" not in fieldnames:
            raise ValueError("label manifest must have filename and digit columns")
        return [(number, row) for number, row in enumerate(reader, start=2)]


def _parse_expected(raw: str, row_number: int) -> tuple[int | None, str | None]:
    text = raw.strip().lower()
    if text in {"uncertain", "invalid"}:
        return None, text
    if text in {"true", "false"} or not text:
        raise ValueError(f"label row {row_number} digit is invalid")
    try:
        digit = int(text)
    except ValueError as exc:
        raise ValueError(f"label row {row_number} digit is invalid") from exc
    if digit < 0 or digit > 9:
        raise ValueError(f"label row {row_number} digit must be 0–9")
    return digit, None


def _parse_order(raw: str, row_number: int, expected: int) -> int:
    text = raw.strip()
    order = 9 if not text else _parse_order_value(text, row_number)
    if expected != 0 and expected > order:
        raise ValueError(
            f"label row {row_number} digit {expected} is outside a {order}×{order} grid"
        )
    return order


def _parse_order_value(raw: str, row_number: int) -> int:
    try:
        order = int(raw)
    except ValueError as exc:
        raise ValueError(f"label row {row_number} order is invalid") from exc
    if order not in SUPPORTED_ORDERS:
        raise ValueError(f"label row {row_number} order must be 4, 6, or 9")
    return order


if __name__ == "__main__":
    raise SystemExit(main())
