"""Export manually labelled template cells for a future digit CNN.

Evaluation photographs are counted and left out of the export. Uncertain and
invalid markers are not digit classes. Blank cells stay in their own task.
Recognizer predictions are not read and are not written as labels.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from app.dataset import ManifestRow, load_manifest, validate_ground_truth
from app.recognition.digit_preprocess import CANVAS_SIDE, preprocess_digit

VALIDATION_FRACTION = 0.2
MIN_TRAIN_PER_DIGIT = 100
MIN_TRAIN_BLANKS = 100
MIN_PHOTOGRAPHS_PER_DIGIT = 3
_HASH_SIDE = 16
_SIMILAR_HAMMING = 4
_CELL_FIELDS = (
    "image",
    "cell_id",
    "row",
    "column",
    "order",
    "development_split",
    "task",
    "label",
    "normalized",
)
_EXCLUDED_FIELDS = ("image", "cell_id", "row", "column", "order", "reason")


def audit_dataset(manifest: Path) -> dict[str, object]:
    """Summarize splits, labels, crops, and similar photographs.

    The manifest is only read. Empty digits stay unlabelled.
    """
    rows = load_manifest(manifest)
    validate_ground_truth(rows)
    missing = _missing_crops(manifest, rows)
    photographs = _photograph_records(rows, missing)
    return {
        "manifest": str(manifest),
        "photographs": len(photographs),
        "cells": len(rows),
        "by_split": _split_summary(photographs),
        "photographs_detail": photographs,
        "missing_crops": missing,
        "exact_duplicate_groups": _exact_duplicates(manifest, photographs),
        "similar_photographs": _similar_photographs(manifest, photographs),
        "cell_geometry": _cell_geometry(manifest, rows),
        "one_split_per_image": True,
    }


def assign_development_split(rows: tuple[ManifestRow, ...]) -> dict[str, str]:
    """Return the stored train or validation role of each template photograph.

    The role is the ``development_split`` value on the manifest. It is not
    derived from filenames. Evaluation photographs are omitted. A missing,
    invalid, or inconsistent role raises ``ValueError``.
    """
    roles: dict[str, set[str | None]] = defaultdict(set)
    splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        roles[row.image].add(row.development_split)
        splits[row.image].add(row.split)
    assignments: dict[str, str] = {}
    for image in sorted(roles):
        if len(splits[image]) != 1:
            raise ValueError(f"{image} is assigned to more than one split")
        split = next(iter(splits[image]))
        if len(roles[image]) != 1:
            raise ValueError(
                f"{image} has more than one development assignment"
            )
        role = next(iter(roles[image]))
        if split == "evaluation":
            if role is not None:
                raise ValueError(
                    f"{image} is an evaluation photograph with "
                    f"development assignment {role}"
                )
            continue
        if role == "holdout":
            continue
        if role not in {"train", "validation"}:
            raise ValueError(
                f"{image} has no persistent development assignment"
            )
        assignments[image] = role
    return assignments


def freeze_development_assignments(
    manifest: Path,
    *,
    backup: Path | None = None,
    expected_validation: frozenset[str] | None = None,
) -> dict[str, str]:
    """Store a development role on each template row without changing labels.

    When every template photograph already has one role, the file is left
    unchanged. When none do, the historical filename rule is written once.
    A mixture of stored and missing roles is refused, because filling the
    gaps would recalculate the photographs that are already assigned.
    """
    raw_rows, fieldnames = _read_manifest_rows(manifest)
    loaded = load_manifest(manifest)
    stored = _stored_roles(loaded)
    if stored is not None:
        if expected_validation is not None:
            validation = {
                image for image, role in stored.items() if role == "validation"
            }
            if validation != expected_validation:
                raise ValueError(
                    "stored validation photographs do not match the frozen set"
                )
        return stored
    proposed = legacy_filename_development_split(loaded)
    if expected_validation is not None:
        validation = {
            image for image, role in proposed.items() if role == "validation"
        }
        if validation != set(expected_validation):
            raise ValueError(
                "filename rule does not match the frozen validation photographs"
            )
    if backup is not None:
        if backup.exists():
            raise ValueError(f"backup already exists: {backup}")
        backup.write_bytes(manifest.read_bytes())
    if "development_split" not in fieldnames:
        fieldnames = [*fieldnames, "development_split"]
    label_snapshot = [
        (
            row.get("image") or "",
            row.get("cell_id") or "",
            row.get("split") or "",
            row.get("digit") or "",
        )
        for row in raw_rows
    ]
    for row in raw_rows:
        image = (row.get("image") or "").strip()
        split = (row.get("split") or "").strip().lower()
        if split == "evaluation":
            row["development_split"] = ""
        else:
            row["development_split"] = proposed[image]
    _replace_manifest(manifest, fieldnames, raw_rows)
    rewritten, _ = _read_manifest_rows(manifest)
    _require_same_labels(label_snapshot, rewritten)
    return proposed


def _stored_roles(rows: tuple[ManifestRow, ...]) -> dict[str, str] | None:
    """Return stored roles, or None when no template photograph has one."""
    template = [row for row in rows if row.split == "template"]
    filled = {row.image for row in template if row.development_split is not None}
    if not filled:
        for row in rows:
            if row.split == "evaluation" and row.development_split is not None:
                raise ValueError(
                    f"{row.image} is an evaluation photograph with "
                    f"development assignment {row.development_split}"
                )
        return None
    return assign_development_split(rows)


def _read_manifest_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        return list(reader), fieldnames


def _replace_manifest(
    path: Path, fieldnames: list[str], rows: list[dict[str, str]]
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    os.replace(temporary, path)


def _require_same_labels(
    before: list[tuple[str, str, str, str]], after: list[dict[str, str]]
) -> None:
    if len(before) != len(after):
        raise ValueError("migration changed the manifest row count")
    for original, saved in zip(before, after, strict=True):
        saved_key = (
            saved.get("image") or "",
            saved.get("cell_id") or "",
            saved.get("split") or "",
            saved.get("digit") or "",
        )
        if original != saved_key:
            raise ValueError(
                f"migration changed {original[0]} {original[1]}"
            )


def legacy_filename_development_split(
    rows: tuple[ManifestRow, ...],
) -> dict[str, str]:
    """Return the historical filename rule. It does not assign new photographs.

    This is only the one-time source for freezing the assignments that
    already existed. Readiness and import do not call it.
    """
    by_order: dict[int, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        if row.split != "template" or row.image in seen:
            continue
        seen.add(row.image)
        by_order[row.order].append(row.image)
    assignments: dict[str, str] = {}
    for order in sorted(by_order):
        names = tuple(sorted(by_order[order]))
        validation = _validation_names(names)
        for name in names:
            assignments[name] = "validation" if name in validation else "train"
    return assignments


def export_cnn_dataset(manifest: Path, output_dir: Path) -> dict[str, object]:
    """Write the template-split CNN package and its metadata.

    Only cells with a manual digit are exported. ``0`` is the blank task.
    Digits 1–9 are the digit task. Uncertain, invalid, and empty cells are
    listed as excluded and are not given a class. Evaluation cells are not
    written.
    """
    _require_separate_output(manifest, output_dir)
    rows = load_manifest(manifest)
    validate_ground_truth(rows)
    missing = _missing_crops(manifest, rows)
    if missing:
        raise ValueError(f"{missing[0]['cell']} is missing")
    audit = audit_dataset(manifest)
    assignments = assign_development_split(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_root = output_dir / "normalized"
    if normalized_root.exists():
        _clear_tree(normalized_root)
    exported: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    for row in rows:
        if row.split != "template":
            continue
        if row.development_split == "holdout":
            excluded.append(
                {
                    "image": row.image,
                    "cell_id": row.cell_id,
                    "row": row.row,
                    "column": row.column,
                    "order": row.order,
                    "reason": "holdout",
                }
            )
            continue
        reason = _exclusion_reason(row)
        if reason is not None:
            excluded.append(
                {
                    "image": row.image,
                    "cell_id": row.cell_id,
                    "row": row.row,
                    "column": row.column,
                    "order": row.order,
                    "reason": reason,
                }
            )
            continue
        assert row.digit is not None
        task = "blank" if row.digit == 0 else "digit"
        relative = Path("normalized") / assignments[row.image] / task
        relative = relative / row.image / f"{row.cell_id}.png"
        _write_normalized(manifest.parent / row.cell, output_dir / relative)
        exported.append(
            {
                "image": row.image,
                "cell_id": row.cell_id,
                "row": row.row,
                "column": row.column,
                "order": row.order,
                "development_split": assignments[row.image],
                "task": task,
                "label": row.digit,
                "normalized": relative.as_posix(),
            }
        )
    _write_csv(output_dir / "cells.csv", _CELL_FIELDS, exported)
    _write_csv(output_dir / "excluded_template.csv", _EXCLUDED_FIELDS, excluded)
    _write_development_split(output_dir / "development_split.csv", rows, assignments)
    distribution = _distribution(exported, excluded, rows)
    readiness = _readiness(exported, assignments, rows)
    payload = {
        "note": (
            "Labels come from the manifest digit column. "
            "Evaluation photographs are held out and were not exported. "
            "Augmentation was not applied."
        ),
        "augmentation_applied": False,
        "evaluation_used_for_export": False,
        "development_split": {
            "train_photographs": sum(role == "train" for role in assignments.values()),
            "validation_photographs": sum(
                role == "validation" for role in assignments.values()
            ),
            "rule": (
                "Each template photograph keeps the development_split stored "
                "on its manifest rows. New template photographs are training. "
                "Evaluation photographs are not assigned."
            ),
        },
        "class_distribution": distribution,
        "readiness": readiness,
        "audit": audit,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "class_distribution.json").write_text(
        json.dumps(distribution, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "class_distribution.md").write_text(
        _distribution_markdown(distribution, readiness),
        encoding="utf-8",
    )
    return payload


def _validation_names(names: tuple[str, ...]) -> set[str]:
    count = len(names)
    if count < 2:
        return set()
    wanted = round(count * VALIDATION_FRACTION)
    wanted = min(max(wanted, 1), count - 1)
    chosen: set[int] = set()
    for slot in range(1, wanted + 1):
        index = round(slot * (count - 1) / (wanted + 1))
        index = min(max(index, 0), count - 1)
        while index in chosen and index + 1 < count:
            index += 1
        chosen.add(index)
    cursor = 0
    while len(chosen) < wanted and cursor < count:
        chosen.add(cursor)
        cursor += 1
    return {names[index] for index in chosen}


def _exclusion_reason(row: ManifestRow) -> str | None:
    if row.marker == "uncertain":
        return "uncertain"
    if row.marker == "invalid":
        return "invalid"
    if row.digit is None:
        return "unlabelled"
    return None


def _write_normalized(source: Path, destination: Path) -> None:
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cell image could not be read: {source}")
    normalized = preprocess_digit(image)
    canvas = np.clip(np.round(normalized * 255.0), 0, 255).astype(np.uint8)
    if canvas.shape != (CANVAS_SIDE, CANVAS_SIDE):
        raise ValueError(f"normalized cell is {canvas.shape}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), canvas):
        raise ValueError(f"normalized cell could not be written: {destination}")


def _distribution(
    exported: list[dict[str, object]],
    excluded: list[dict[str, object]],
    rows: tuple[ManifestRow, ...],
) -> dict[str, object]:
    buckets = {
        "train": _empty_class_bucket(),
        "validation": _empty_class_bucket(),
    }
    for row in exported:
        bucket = buckets[str(row["development_split"])]
        label = str(row["label"])
        bucket["labels"][label] = int(bucket["labels"][label]) + 1
        if row["task"] == "blank":
            bucket["blank"] = int(bucket["blank"]) + 1
        else:
            bucket["digits"] = int(bucket["digits"]) + 1
    held = [row for row in rows if row.split == "evaluation"]
    return {
        "train": buckets["train"],
        "validation": buckets["validation"],
        "excluded_template": {
            "unlabelled": sum(row["reason"] == "unlabelled" for row in excluded),
            "uncertain": sum(row["reason"] == "uncertain" for row in excluded),
            "invalid": sum(row["reason"] == "invalid" for row in excluded),
        },
        "held_out_evaluation": {
            "images": len({row.image for row in held}),
            "cells": len(held),
            "exported": False,
        },
    }


def _empty_class_bucket() -> dict[str, object]:
    return {
        "blank": 0,
        "digits": 0,
        "labels": {str(digit): 0 for digit in range(10)},
    }


def _readiness(
    exported: list[dict[str, object]],
    assignments: dict[str, str],
    rows: tuple[ManifestRow, ...],
) -> dict[str, object]:
    train = [row for row in exported if row["development_split"] == "train"]
    photos_by_digit: dict[str, set[str]] = {str(digit): set() for digit in range(1, 10)}
    counts = {str(digit): 0 for digit in range(1, 10)}
    blanks = 0
    for row in train:
        label = int(row["label"])
        if label == 0:
            blanks += 1
            continue
        counts[str(label)] += 1
        photos_by_digit[str(label)].add(str(row["image"]))
    orders = {
        order: {"train": 0, "validation": 0}
        for order in (4, 6, 9)
    }
    image_order = {row.image: row.order for row in rows if row.split == "template"}
    for image, role in assignments.items():
        orders[image_order[image]][role] += 1
    reasons: list[str] = []
    if not exported:
        reasons.append("The template split has no manual blank or digit labels.")
    for digit, count in counts.items():
        if count < MIN_TRAIN_PER_DIGIT:
            reasons.append(
                f"Digit {digit} has {count} training cells; "
                f"{MIN_TRAIN_PER_DIGIT} are required."
            )
        if len(photos_by_digit[digit]) < MIN_PHOTOGRAPHS_PER_DIGIT:
            reasons.append(
                f"Digit {digit} comes from {len(photos_by_digit[digit])} "
                f"training photographs; {MIN_PHOTOGRAPHS_PER_DIGIT} are required."
            )
    if blanks < MIN_TRAIN_BLANKS:
        reasons.append(
            f"Blank cells in training: {blanks}. {MIN_TRAIN_BLANKS} are required."
        )
    for order, roles in orders.items():
        if roles["train"] == 0 or roles["validation"] == 0:
            reasons.append(
                f"Order {order} needs photographs in both train and validation."
            )
    return {
        "ready_for_training": not reasons,
        "reasons": reasons,
        "minimum": {
            "training_cells_per_digit": MIN_TRAIN_PER_DIGIT,
            "training_blank_cells": MIN_TRAIN_BLANKS,
            "training_photographs_per_digit": MIN_PHOTOGRAPHS_PER_DIGIT,
            "source": "template photographs only",
        },
        "training_counts": {"blank": blanks, "digits": counts},
        "orders": {str(order): roles for order, roles in orders.items()},
    }


def training_decision(
    rows: tuple[ManifestRow, ...],
    *,
    manifest: Path | None = None,
) -> dict[str, object]:
    """Decide whether CNN training is allowed.

    The minimum counts are unchanged. Evaluation photographs are not training
    or validation data. Uncertain and invalid markers are not digit classes.
    """
    missing: list[str] = []
    try:
        validate_ground_truth(rows)
    except ValueError as exc:
        return _decision([str(exc)], blocked=True)
    assignments = assign_development_split(rows)
    evaluation = {row.image for row in rows if row.split == "evaluation"}
    leaked = sorted(set(assignments) & evaluation)
    if leaked:
        missing.append(
            f"{leaked[0]} is an evaluation photograph in the development split."
        )
    for image, role in assignments.items():
        roles = {row.split for row in rows if row.image == image}
        if roles != {"template"}:
            missing.append(f"{image} is not kept entirely in the template split.")
        if role not in {"train", "validation"}:
            missing.append(f"{image} has development split {role}.")
    exported: list[dict[str, object]] = []
    for row in rows:
        if row.split != "template" or row.digit is None:
            continue
        if row.marker in {"uncertain", "invalid"}:
            missing.append(f"{row.image} {row.cell_id} is both a marker and a digit.")
            continue
        exported.append(
            {
                "image": row.image,
                "label": row.digit,
                "development_split": assignments[row.image],
            }
        )
    if manifest is not None:
        absent = _missing_crops(manifest, rows)
        if absent:
            missing.append(f"{len(absent)} cell crops are missing or empty.")
        for group in _exact_duplicates(manifest, _photograph_records(rows, absent)):
            splits = {row.split for row in rows if row.image in group}
            if len(splits) > 1:
                missing.append(
                    f"{group[0]} duplicates a photograph in another split."
                )
    readiness = _readiness(exported, assignments, rows)
    reasons = readiness["reasons"]
    assert isinstance(reasons, list)
    missing.extend(str(reason) for reason in reasons)
    blocked = bool(missing) or not bool(readiness["ready_for_training"])
    payload = _decision(missing, blocked=blocked)
    payload["training_counts"] = readiness["training_counts"]
    payload["orders"] = readiness["orders"]
    payload["minimum"] = readiness["minimum"]
    return payload


def dataset_validation_report(manifest: Path) -> dict[str, object]:
    """Build the photograph-level validation report. The manifest is not written."""
    rows = load_manifest(manifest)
    decision = training_decision(rows, manifest=manifest)
    audit = audit_dataset(manifest)
    return {
        "decision": decision["decision"],
        "ready_for_training": decision["ready_for_training"],
        "missing": decision["missing"],
        "training_counts": decision["training_counts"],
        "orders": decision["orders"],
        "coverage": {
            "template": _coverage(rows, "template"),
            "evaluation_held_out": _coverage(rows, "evaluation"),
        },
        "completeness": _completeness(rows),
        "by_photograph": audit["photographs_detail"],
        "exact_duplicate_groups": audit["exact_duplicate_groups"],
        "similar_photographs": audit["similar_photographs"],
        "missing_crops": audit["missing_crops"],
        "cell_geometry": audit["cell_geometry"],
        "source_distribution": source_class_distribution(rows),
    }


def write_validation_report(report: dict[str, object], output_dir: Path) -> Path:
    """Write the validation JSON and a short markdown decision."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    path = output_dir / "report.md"
    path.write_text(_validation_markdown(report), encoding="utf-8")
    return path


def _decision(missing: list[str], *, blocked: bool) -> dict[str, object]:
    return {
        "decision": "NO-GO" if blocked else "GO",
        "ready_for_training": not blocked,
        "missing": missing,
    }


def _coverage(rows: tuple[ManifestRow, ...], split: str) -> dict[str, object]:
    chosen = [row for row in rows if row.split == split]
    classes: dict[str, dict[str, object]] = {}
    for digit in range(10):
        matched = [row for row in chosen if row.digit == digit]
        photos = sorted({row.image for row in matched})
        classes[str(digit)] = {"cells": len(matched), "photographs": photos}
    by_order: dict[str, dict[str, int]] = {}
    for row in chosen:
        bucket = by_order.setdefault(
            str(row.order),
            {"cells": 0, "labelled": 0, "blank": 0, "digits": 0},
        )
        bucket["cells"] += 1
        if row.digit is not None:
            bucket["labelled"] += 1
            if row.digit == 0:
                bucket["blank"] += 1
            else:
                bucket["digits"] += 1
    return {"classes": classes, "by_order": by_order}


def _completeness(rows: tuple[ManifestRow, ...]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for split in ("template", "evaluation"):
        chosen = [row for row in rows if row.split == split]
        summary[split] = {
            "cells": len(chosen),
            "blank": sum(row.digit == 0 for row in chosen),
            "digits": sum(
                row.digit is not None and row.digit != 0 for row in chosen
            ),
            "unlabelled": sum(
                row.digit is None and row.marker is None for row in chosen
            ),
            "uncertain": sum(row.marker == "uncertain" for row in chosen),
            "invalid": sum(row.marker == "invalid" for row in chosen),
        }
    return summary


def _validation_markdown(report: dict[str, object]) -> str:
    decision = str(report["decision"])
    lines = [
        "# Dataset validation",
        "",
        f"Decision: **{decision}**.",
        "",
        "CNN training is allowed only when every requirement below is met.",
        "",
    ]
    missing = report["missing"]
    assert isinstance(missing, list)
    if missing:
        lines.extend(f"- {item}" for item in missing)
    else:
        lines.append("Every training requirement is met.")
    lines.extend(["", "## Template coverage", ""])
    coverage = report["coverage"]
    assert isinstance(coverage, dict)
    template = coverage["template"]
    assert isinstance(template, dict)
    classes = template["classes"]
    assert isinstance(classes, dict)
    lines.extend(
        [
            "| Class | Cells | Photographs |",
            "| --- | ---: | ---: |",
        ]
    )
    for name, bucket in classes.items():
        assert isinstance(bucket, dict)
        photos = bucket["photographs"]
        assert isinstance(photos, list)
        lines.append(f"| {name} | {bucket['cells']} | {len(photos)} |")
    warnings = report["similar_photographs"]
    assert isinstance(warnings, list)
    lines.extend(["", f"Similar-photograph warnings: {len(warnings)}.", ""])
    lines.extend(_source_markdown(report))
    return "\n".join(lines)


def source_class_distribution(
    rows: tuple[ManifestRow, ...],
) -> dict[str, dict[str, dict[str, object]]]:
    """Count labelled cells by development role and source type.

    The result is informational. It is not an input to the training gate.
    A missing source type is reported as ``unset`` and is not called unknown.
    """
    assignments = assign_development_split(rows)
    names = ("synthetic", "photographed", "unknown", "unset")
    tables: dict[str, dict[str, dict[str, object]]] = {
        role: {
            name: {
                "blank": 0,
                "digits": {str(digit): 0 for digit in range(1, 10)},
                "photographs": set(),
            }
            for name in names
        }
        for role in ("train", "validation", "evaluation")
    }
    for row in rows:
        role = "evaluation" if row.split == "evaluation" else assignments[row.image]
        source = row.source_type if row.source_type is not None else "unset"
        bucket = tables[role][source]
        photographs = bucket["photographs"]
        assert isinstance(photographs, set)
        photographs.add(row.image)
        if row.digit is None or row.marker is not None:
            continue
        if row.digit == 0:
            bucket["blank"] = int(bucket["blank"]) + 1
        elif 1 <= row.digit <= 9:
            digits = bucket["digits"]
            assert isinstance(digits, dict)
            digits[str(row.digit)] = int(digits[str(row.digit)]) + 1
    rendered: dict[str, dict[str, dict[str, object]]] = {}
    for role, sources in tables.items():
        rendered[role] = {}
        for name, bucket in sources.items():
            photographs = bucket["photographs"]
            assert isinstance(photographs, set)
            rendered[role][name] = {
                "blank": bucket["blank"],
                "digits": bucket["digits"],
                "photographs": len(photographs),
            }
    return rendered


def _source_markdown(report: dict[str, object]) -> list[str]:
    source = report.get("source_distribution")
    if not isinstance(source, dict):
        return []
    training = source.get("train")
    if not isinstance(training, dict):
        return []
    names = ("synthetic", "photographed", "unknown", "unset")
    lines = [
        "## Training cells by source",
        "",
        "These counts split the same training cells. "
        "They do not change the minimums above.",
        "",
        "| Class | " + " | ".join(names) + " |",
        "| --- | " + " | ".join("---:" for _ in names) + " |",
    ]
    for digit in range(10):
        values: list[str] = []
        for name in names:
            bucket = training[name]
            assert isinstance(bucket, dict)
            if digit == 0:
                values.append(str(bucket["blank"]))
            else:
                digits = bucket["digits"]
                assert isinstance(digits, dict)
                values.append(str(digits[str(digit)]))
        lines.append(f"| {digit} | " + " | ".join(values) + " |")
    counts = []
    for name in names:
        bucket = training[name]
        assert isinstance(bucket, dict)
        counts.append(f"{name} {bucket['photographs']}")
    joined = ", ".join(counts)
    lines.extend(["", f"Training photographs by source: {joined}.", ""])
    return lines


def _split_summary(photographs: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for split in ("template", "evaluation"):
        chosen = [row for row in photographs if row["split"] == split]
        labels: dict[str, int] = defaultdict(int)
        orders: dict[str, dict[str, int]] = {}
        for row in chosen:
            for name, count in row["labels"].items():  # type: ignore[union-attr]
                labels[str(name)] += int(count)
            bucket = orders.setdefault(
                str(row["order"]),
                {"photographs": 0, "cells": 0},
            )
            bucket["photographs"] += 1
            bucket["cells"] += int(row["cells"])
        summary[split] = {
            "photographs": len(chosen),
            "cells": sum(int(row["cells"]) for row in chosen),
            "labels": dict(sorted(labels.items())),
            "by_order": orders,
        }
    return summary


def _photograph_records(
    rows: tuple[ManifestRow, ...], missing: list[dict[str, str]]
) -> list[dict[str, object]]:
    missing_keys = {(item["image"], item["cell_id"]) for item in missing}
    grouped: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in rows:
        grouped[row.image].append(row)
    records: list[dict[str, object]] = []
    for image in sorted(grouped):
        cells = grouped[image]
        labels: dict[str, int] = defaultdict(int)
        for cell in cells:
            if cell.marker is not None:
                labels[cell.marker] += 1
            elif cell.digit is None:
                labels["unlabelled"] += 1
            else:
                labels[str(cell.digit)] += 1
        records.append(
            {
                "image": image,
                "split": cells[0].split,
                "order": cells[0].order,
                "cells": len(cells),
                "missing_crops": sum(
                    (image, cell.cell_id) in missing_keys for cell in cells
                ),
                "labels": dict(sorted(labels.items())),
            }
        )
    return records


def _missing_crops(
    manifest: Path, rows: tuple[ManifestRow, ...]
) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    for row in rows:
        path = manifest.parent / row.cell
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(
                {"image": row.image, "cell_id": row.cell_id, "cell": row.cell}
            )
    return missing


def _exact_duplicates(
    manifest: Path, photographs: list[dict[str, object]]
) -> list[list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for record in photographs:
        path = _original_path(manifest, str(record["image"]))
        if path is None:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        groups[digest].append(str(record["image"]))
    return [sorted(names) for names in groups.values() if len(names) > 1]


def _similar_photographs(
    manifest: Path, photographs: list[dict[str, object]]
) -> list[dict[str, object]]:
    prepared: list[tuple[str, np.ndarray, np.ndarray]] = []
    for record in photographs:
        image = str(record["image"])
        path = _original_path(manifest, image)
        if path is None:
            continue
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        small = cv2.resize(gray, (_HASH_SIDE, _HASH_SIDE), interpolation=cv2.INTER_AREA)
        bits = (small > float(small.mean())).astype(np.uint8).reshape(-1)
        prepared.append((image, bits, gray))
    details = {str(row["image"]): row for row in photographs}
    pairs: list[dict[str, object]] = []
    for index, (left, left_bits, left_gray) in enumerate(prepared):
        for right, right_bits, right_gray in prepared[index + 1 :]:
            hamming = int(np.count_nonzero(left_bits != right_bits))
            if hamming > _SIMILAR_HAMMING:
                continue
            left_view = cv2.resize(
                left_gray, (128, 128), interpolation=cv2.INTER_AREA
            )
            right_view = cv2.resize(
                right_gray, (128, 128), interpolation=cv2.INTER_AREA
            )
            gap = np.abs(left_view.astype(np.int16) - right_view.astype(np.int16))
            difference = float(np.mean(gap))
            pairs.append(
                {
                    "hamming": hamming,
                    "mean_absolute_difference": round(difference, 2),
                    "images": [left, right],
                    "splits": [details[left]["split"], details[right]["split"]],
                    "orders": [details[left]["order"], details[right]["order"]],
                }
            )
    return sorted(pairs, key=lambda item: int(item["hamming"]))


def _cell_geometry(manifest: Path, rows: tuple[ManifestRow, ...]) -> dict[str, object]:
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    low_contrast = {"template": 0, "evaluation": 0}
    for row in rows:
        image = cv2.imread(str(manifest.parent / row.cell), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        grouped[(row.split, row.order)].append(int(min(image.shape[:2])))
        if float(image.std()) < 8:
            low_contrast[row.split] += 1
    summary: dict[str, object] = {}
    for (split, order), sides in sorted(grouped.items()):
        values = np.array(sides)
        summary[f"{split}_{order}"] = {
            "cells": int(values.size),
            "min_side": int(values.min()),
            "median_side": float(np.median(values)),
            "max_side": int(values.max()),
        }
    return {"by_split_and_order": summary, "low_contrast_cells": low_contrast}


def _original_path(manifest: Path, image: str) -> Path | None:
    folder = manifest.parent / image
    copied = folder / f"original{Path(image).suffix.lower()}"
    if copied.is_file():
        return copied
    direct = manifest.parent / image
    if direct.is_file():
        return direct
    return None


def _write_development_split(
    path: Path,
    rows: tuple[ManifestRow, ...],
    assignments: dict[str, str],
) -> None:
    orders = {row.image: row.order for row in rows if row.image in assignments}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image", "order", "development_split"],
        )
        writer.writeheader()
        for image in sorted(assignments):
            writer.writerow(
                {
                    "image": image,
                    "order": orders[image],
                    "development_split": assignments[image],
                }
            )


def _write_csv(
    path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _distribution_markdown(
    distribution: dict[str, object], readiness: dict[str, object]
) -> str:
    lines = [
        "# CNN class distribution",
        "",
        "Blank cells are separate from digits 1–9. "
        "Uncertain and invalid cells are excluded. "
        "Evaluation cells were not exported.",
        "",
        "| Split | Blank | Digits 1–9 |",
        "| --- | ---: | ---: |",
    ]
    for name in ("train", "validation"):
        bucket = distribution[name]
        assert isinstance(bucket, dict)
        lines.append(f"| {name} | {bucket['blank']} | {bucket['digits']} |")
    excluded = distribution["excluded_template"]
    assert isinstance(excluded, dict)
    lines.extend(
        [
            "",
            f"Template cells left out: {excluded['unlabelled']} unlabelled, "
            f"{excluded['uncertain']} uncertain, {excluded['invalid']} invalid.",
            "",
            "Ready for training: "
            + ("yes" if readiness["ready_for_training"] else "no"),
            "",
        ]
    )
    reasons = readiness["reasons"]
    assert isinstance(reasons, list)
    lines.extend(f"- {reason}" for reason in reasons)
    if reasons:
        lines.append("")
    return "\n".join(lines)


def _require_separate_output(manifest: Path, output_dir: Path) -> None:
    if output_dir.resolve() == manifest.resolve().parent:
        raise ValueError("the export directory must not be the dataset directory")
    if (output_dir / "manifest.csv").is_file():
        raise ValueError("the export directory looks like the labelled dataset")


def _clear_tree(path: Path) -> None:
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    path.rmdir()
