"""Prepare a labelled cell dataset from Sudoku photographs.

The manifest records where each cell came from. The digit column is left
empty. Predictions are not copied into it, and every cell from one source
image stays in a single split.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from app.pipeline import CellLabel, load_labels, run_directory

SPLITS = ("template", "evaluation")
SOURCE_TYPES = ("synthetic", "photographed", "unknown")
MANIFEST_FIELDS = (
    "image",
    "cell_id",
    "cell",
    "row",
    "column",
    "order",
    "split",
    "digit",
    "development_split",
)
_NOTE = (
    "The digit column is empty until a person fills it. "
    "Recognizer output is not ground truth."
)


@dataclass(frozen=True, slots=True)
class ManifestRow:
    """One extracted cell and its manual label, if a person has entered one."""

    image: str
    cell_id: str
    cell: str
    row: int
    column: int
    order: int
    split: str
    digit: int | None
    marker: str | None
    development_split: str | None = None
    source_type: str | None = None


def prepare_dataset(
    image_dir: Path,
    output_dir: Path,
    *,
    splits: Path | None = None,
) -> dict[str, object]:
    """Extract cells and write an unlabelled manifest.

    ``splits`` maps each source image to ``template`` or ``evaluation``.
    When it is omitted, the split column is left blank for later assignment.
    Source photographs are copied into the output and are not modified.
    """
    report = run_directory(image_dir, output_dir)
    assignments = {} if splits is None else load_split_file(splits)
    _require_known_images(report, assignments, required=splits is not None)
    rows = _rows_from_report(report, assignments)
    manifest_path = output_dir / "manifest.csv"
    _write_manifest(manifest_path, rows)
    _write_split_table(output_dir / "splits.csv", report, assignments)
    payload = {
        "note": _NOTE,
        "source_directory": str(image_dir),
        "manifest": "manifest.csv",
        "splits": "splits.csv",
        "labels_are_blank": True,
        "split_counts": _split_counts(rows, report),
        "images": [
            {
                "filename": result["filename"],
                "order": result["order"],
                "split": assignments.get(str(result["filename"]), ""),
                "cells": result["cells_extracted"],
                "contact_sheet": f"{result['filename']}/contact_sheet.png",
            }
            for result in report["results"]
            if result["success"]
        ],
        "failed_images": report["failed_images"],
    }
    (output_dir / "dataset.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def load_split_file(path: Path) -> dict[str, str]:
    """Read one split per source image."""
    if not path.is_file():
        raise FileNotFoundError(f"split file does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "image" not in fieldnames or "split" not in fieldnames:
            raise ValueError("split file must have image and split columns")
        assignments: dict[str, str] = {}
        for row_number, row in enumerate(reader, start=2):
            image = (row.get("image") or "").strip()
            split = (row.get("split") or "").strip().lower()
            if not image:
                raise ValueError(f"split row {row_number} has an empty image")
            if split not in SPLITS:
                raise ValueError(
                    f"split row {row_number} must be template or evaluation"
                )
            previous = assignments.get(image)
            if previous is not None and previous != split:
                raise ValueError(f"{image} is assigned to more than one split")
            assignments[image] = split
    if not assignments:
        raise ValueError(f"split file {path} has no images")
    return assignments


def load_manifest(path: Path) -> tuple[ManifestRow, ...]:
    """Read a dataset manifest and reject inconsistent image splits."""
    labels = load_labels(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {"image", "cell", "row", "column", "order", "split", "digit"}
        if not required.issubset(fieldnames):
            raise ValueError(
                "manifest must have image, cell, row, column, order, split, and digit"
            )
        raw_rows = list(reader)
    if len(raw_rows) != len(labels):
        raise ValueError("manifest row count does not match its labels")
    rows = tuple(
        _manifest_row(label, raw)
        for label, raw in zip(labels, raw_rows, strict=True)
    )
    _ensure_one_split_per_image(rows)
    _ensure_one_source_per_image(rows)
    return rows


ManifestRows = tuple[ManifestRow, ...] | list[ManifestRow]


def evaluation_rows(rows: ManifestRows) -> tuple[ManifestRow, ...]:
    """Return cells whose source image belongs to the evaluation split."""
    chosen = tuple(row for row in rows if row.split == "evaluation")
    template_images = {row.image for row in rows if row.split == "template"}
    leaked = [row.image for row in chosen if row.image in template_images]
    if leaked:
        raise ValueError(f"{leaked[0]} is in both the template and evaluation splits")
    return chosen


def template_rows(rows: ManifestRows) -> tuple[ManifestRow, ...]:
    """Return manually labelled digits that may be used as templates.

    Evaluation photographs are never templates, even when they contain a digit.
    Blank labels are not templates. Empty cells are not labels.
    """
    evaluation = {row.image for row in rows if row.split == "evaluation"}
    chosen = tuple(
        row
        for row in rows
        if row.split == "template" and row.digit is not None and row.digit != 0
    )
    leaked = [row.image for row in chosen if row.image in evaluation]
    if leaked:
        raise ValueError(f"{leaked[0]} is an evaluation image and cannot be a template")
    return chosen


def validate_ground_truth(rows: ManifestRows) -> dict[str, object]:
    """Check split, order, and label rules without changing the rows.

    Empty digits are unlabelled. Uncertain and invalid markers are reported
    separately because they are excluded from digit accuracy.
    """
    materialized = tuple(rows)
    _ensure_one_split_per_image(materialized)
    seen: set[tuple[str, str]] = set()
    orders: dict[str, int] = {}
    for row in materialized:
        if row.order not in {4, 6, 9}:
            raise ValueError(f"{row.image} has order {row.order}")
        if row.marker is not None and row.digit is not None:
            raise ValueError(f"{row.image} {row.cell_id} has both a digit and a marker")
        if row.marker not in {None, "uncertain", "invalid"}:
            raise ValueError(f"{row.image} {row.cell_id} has marker {row.marker!r}")
        if row.digit is not None and (row.digit < 0 or row.digit > row.order):
            raise ValueError(
                f"{row.image} {row.cell_id} digit {row.digit} "
                f"is outside a {row.order}×{row.order} grid"
            )
        key = (row.image, row.cell_id)
        if key in seen:
            raise ValueError(f"{row.image} {row.cell_id} is duplicated")
        seen.add(key)
        previous = orders.get(row.image)
        if previous is not None and previous != row.order:
            raise ValueError(f"{row.image} has more than one grid order")
        orders[row.image] = row.order
    template_rows(materialized)
    evaluation_rows(materialized)
    return {
        "images": len(orders),
        "cells": len(materialized),
        "one_split_per_image": True,
        "evaluation_images_excluded_from_templates": True,
    }


def _manifest_row(label: CellLabel, raw: dict[str, str]) -> ManifestRow:
    if label.order is None:
        raise ValueError(f"{label.image} row {label.row} is missing its order")
    split = (raw.get("split") or "").strip().lower()
    if split not in SPLITS:
        raise ValueError(f"{label.image} must be template or evaluation")
    fallback = f"r{label.row:02d}_c{label.column:02d}"
    cell_id = (raw.get("cell_id") or "").strip() or fallback
    role = (raw.get("development_split") or "").strip().lower()
    if role == "":
        development_split = None
    elif role in {"train", "validation", "holdout"}:
        development_split = role
    else:
        raise ValueError(
            f"{label.image} has invalid development assignment {role!r}"
        )
    source_type = _source_type(label.image, raw)
    return ManifestRow(
        image=label.image,
        cell_id=cell_id,
        cell=label.cell,
        row=label.row,
        column=label.column,
        order=label.order,
        split=split,
        digit=label.digit,
        marker=label.marker,
        development_split=development_split,
        source_type=source_type,
    )


def _source_type(image: str, raw: dict[str, str]) -> str | None:
    """Return a stored source, or None when the column is absent or empty.

    An empty value is not treated as ``unknown``. That word is written only
    when provenance was reviewed and is unavailable.
    """
    if "source_type" not in raw:
        return None
    token = (raw.get("source_type") or "").strip().lower()
    if token == "":
        return None
    if token not in SOURCE_TYPES:
        raise ValueError(f"{image} has invalid source type {token!r}")
    return token


def _ensure_one_split_per_image(rows: tuple[ManifestRow, ...]) -> None:
    seen: dict[str, str] = {}
    for row in rows:
        previous = seen.get(row.image)
        if previous is not None and previous != row.split:
            raise ValueError(f"{row.image} is assigned to more than one split")
        seen[row.image] = row.split


def _ensure_one_source_per_image(rows: tuple[ManifestRow, ...]) -> None:
    seen: dict[str, str | None] = {}
    for row in rows:
        if row.image not in seen:
            seen[row.image] = row.source_type
            continue
        if seen[row.image] != row.source_type:
            raise ValueError(f"{row.image} has more than one source type")


def _require_known_images(
    report: dict[str, object],
    assignments: dict[str, str],
    *,
    required: bool,
) -> None:
    names = {
        str(result["filename"])
        for result in report["results"]  # type: ignore[union-attr]
        if result["success"]
    }
    unknown = sorted(set(assignments) - names)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"split file names images that were not extracted: {joined}")
    missing = sorted(names - set(assignments))
    if required and missing:
        joined = ", ".join(missing)
        raise ValueError(f"split file is missing images: {joined}")


def _rows_from_report(
    report: dict[str, object], assignments: dict[str, str]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in report["results"]:  # type: ignore[union-attr]
        if not result["success"]:
            continue
        image = str(result["filename"])
        for cell in result["cells"]:
            row = int(cell["row"])
            column = int(cell["col"])
            rows.append(
                {
                    "image": image,
                    "cell_id": f"r{row:02d}_c{column:02d}",
                    "cell": f"{image}/cells/r{row:02d}_c{column:02d}.png",
                    "row": row,
                    "column": column,
                    "order": result["order"],
                    "split": assignments.get(image, ""),
                    "digit": "",
                    "development_split": "",
                }
            )
    return rows


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def _write_split_table(
    path: Path, report: dict[str, object], assignments: dict[str, str]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "split"])
        writer.writeheader()
        for result in report["results"]:  # type: ignore[union-attr]
            if result["success"]:
                image = str(result["filename"])
                writer.writerow({"image": image, "split": assignments.get(image, "")})


def _split_counts(
    rows: list[dict[str, object]], report: dict[str, object]
) -> dict[str, int]:
    images = {
        str(row["image"]): str(row["split"])
        for row in rows
    }
    return {
        "template": sum(split == "template" for split in images.values()),
        "evaluation": sum(split == "evaluation" for split in images.values()),
        "unassigned": sum(split == "" for split in images.values()),
        "failed": len(report["failed_images"]),  # type: ignore[arg-type]
    }
