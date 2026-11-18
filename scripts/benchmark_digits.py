"""Score the template recognizer on the held-out portion of a labelled dataset.

Accuracy is omitted when any evaluation cell is still unlabelled. Template
cells are not scored, and they are loaded only when a person has labelled a
printed digit.

Example:
    python scripts/benchmark_digits.py \\
        --manifest evaluation_output/dataset/manifest.csv \\
        --output evaluation_output/benchmark
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from app.dataset import (
    ManifestRow,
    evaluation_rows,
    load_manifest,
    template_rows,
)
from app.recognition.errors import RecognitionError
from scripts.evaluate_digits import evaluate_labels

_NOTE = (
    "Accuracy counts manually labelled evaluation cells only. "
    "Uncertain and invalid labels are excluded. "
    "An incomplete label file does not produce an accuracy."
)
_CSV_FIELDS = (
    "image",
    "cell",
    "row",
    "column",
    "order",
    "expected",
    "marker",
    "predicted",
    "status",
    "correct",
)


def benchmark_dataset(
    manifest_path: Path,
    output_dir: Path,
    *,
    use_builtin: bool = True,
) -> dict[str, object]:
    """Evaluate the held-out split and write the configuration and reports."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
    rows = load_manifest(manifest_path)
    held_out = evaluation_rows(rows)
    if not held_out:
        raise ValueError("manifest has no evaluation images")
    output_dir.mkdir(parents=True, exist_ok=True)
    templates = _write_templates(
        output_dir / "templates.csv", rows, manifest_path.parent
    )
    config = {
        "note": _NOTE,
        "manifest": str(manifest_path),
        "use_builtin": use_builtin,
        "template_manifest": None if templates is None else str(templates),
        "template_image_count": len(
            {row.image for row in rows if row.split == "template"}
        ),
        "evaluation_image_count": len({row.image for row in held_out}),
    }
    missing = [row for row in held_out if row.digit is None and row.marker is None]
    if missing:
        report = _unverified_report(config, held_out, missing)
    else:
        labels = _write_evaluation_labels(
            output_dir / "evaluation_labels.csv", held_out, manifest_path.parent
        )
        config["evaluation_labels"] = str(labels)
        scored = evaluate_labels(
            labels,
            template_manifest=templates,
            use_builtin=use_builtin,
        )
        report = {"verified": True, "configuration": config, **scored}
        _attach_cells(report, held_out, manifest_path.parent)
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report_csv(output_dir / "report.csv", report)
    return report


def main(argv: list[str] | None = None) -> int:
    """Score a manifest. Incomplete evaluation labels do not yield an accuracy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--no-builtin",
        action="store_true",
        help="use only manually labelled template cells",
    )
    args = parser.parse_args(argv)
    try:
        report = benchmark_dataset(
            args.manifest,
            args.output,
            use_builtin=not args.no_builtin,
        )
    except (FileNotFoundError, RecognitionError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    accuracy = report.get("accuracy")
    shown = "n/a" if accuracy is None else f"{accuracy:.3f}"
    print(args.output / "report.json")
    print(f"verified={report['verified']} accuracy={shown}")
    return 0


def _unverified_report(
    config: dict[str, object],
    held_out: tuple[ManifestRow, ...],
    missing: list[ManifestRow],
) -> dict[str, object]:
    return {
        "note": _NOTE,
        "verified": False,
        "configuration": config,
        "accuracy": None,
        "scored_count": 0,
        "unlabelled_count": len(missing),
        "evaluation_cells": len(held_out),
        "predictions": [_labelled_prediction(row) for row in held_out],
    }


def _attach_cells(
    report: dict[str, object],
    held_out: tuple[ManifestRow, ...],
    root: Path,
) -> None:
    located = {str((root / row.cell).resolve()): row for row in held_out}
    predictions = report.get("predictions")
    if not isinstance(predictions, list):
        return
    for item in predictions:
        source = located.get(str(item.get("filename")))
        if source is None:
            continue
        item["cell"] = source.cell
        item["row"] = source.row
        item["column"] = source.column
        item["order"] = source.order
        item["image"] = source.image


def _labelled_prediction(row: ManifestRow) -> dict[str, object]:
    expected = row.marker if row.marker is not None else row.digit
    return {
        "image": row.image,
        "cell": row.cell,
        "row": row.row,
        "column": row.column,
        "order": row.order,
        "expected": expected,
        "marker": row.marker,
        "predicted": None,
        "status": None,
        "correct": False,
    }


def _write_templates(
    path: Path, rows: tuple[ManifestRow, ...], root: Path
) -> Path | None:
    chosen = template_rows(rows)
    if not chosen:
        return None
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "digit", "role"])
        writer.writeheader()
        for row in chosen:
            writer.writerow(
                {
                    "filename": str((root / row.cell).resolve()),
                    "digit": row.digit,
                    "role": "template",
                }
            )
    return path


def _write_evaluation_labels(
    path: Path, rows: tuple[ManifestRow, ...], root: Path
) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "digit", "order", "image"],
        )
        writer.writeheader()
        for row in rows:
            digit = row.marker if row.marker is not None else row.digit
            writer.writerow(
                {
                    "filename": str((root / row.cell).resolve()),
                    "digit": digit,
                    "order": row.order,
                    "image": row.image,
                }
            )
    return path


def _write_report_csv(path: Path, report: dict[str, object]) -> None:
    predictions = report.get("predictions") or []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_CSV_FIELDS))
        writer.writeheader()
        for row in predictions:  # type: ignore[union-attr]
            writer.writerow({name: row.get(name, "") for name in _CSV_FIELDS})


if __name__ == "__main__":
    raise SystemExit(main())
