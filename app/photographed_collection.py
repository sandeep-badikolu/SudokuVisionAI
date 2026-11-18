"""Coverage of genuine camera photographs. Missing facts stay missing.

Condition notes are used only when a person recorded them. An empty note
is not a guess about paper, print, light, or perspective.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from app.dataset import ManifestRow

COLLECTION_TARGETS = {4: 8, 6: 8, 9: 12}
CONDITION_FIELDS = ("paper", "print_style", "lighting", "perspective")


def collection_report(
    rows: Sequence[ManifestRow],
    conditions: dict[str, dict[str, str]] | None = None,
) -> dict[str, object]:
    """Summarise photographed coverage and name what is still absent."""
    notes = conditions or {}
    photographed = [row for row in rows if row.source_type == "photographed"]
    by_source = _photographs_by_source(rows)
    by_order = [_order_coverage(photographed, order) for order in (4, 6, 9)]
    digits = [_digit_coverage(photographed, digit) for digit in range(10)]
    recorded = _condition_rows(photographed, notes)
    missing = _missing(by_order, digits, recorded)
    holdout = [row for row in photographed if row.development_split == "holdout"]
    holdout_orders = {row.order for row in holdout}
    holdout_labelled = bool(holdout) and all(
        row.digit is not None or row.marker is not None for row in holdout
    )
    return {
        "targets_are_collection_goals": True,
        "targets_are_model_criteria": False,
        "photographs_by_source": by_source,
        "by_order": by_order,
        "digits": digits,
        "conditions": recorded,
        "unlabelled_cells": sum(
            row.digit is None and row.marker is None for row in photographed
        ),
        "missing": missing,
        "holdout_covers_every_grid_size": holdout_orders >= {4, 6, 9},
        "holdout_labelled": holdout_labelled,
        "generalisation_evaluated": False,
    }


def load_conditions(path: Path | None) -> dict[str, dict[str, str]]:
    """Read manual condition notes. A missing file means nothing was recorded."""
    if path is None or not path.is_file():
        return {}
    notes: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "image" not in reader.fieldnames:
            raise ValueError("condition notes need an image column")
        for row in reader:
            image = (row.get("image") or "").strip()
            if not image:
                continue
            notes[image] = {
                field: (row.get(field) or "").strip() for field in CONDITION_FIELDS
            }
    return notes


def _photographs_by_source(rows: Sequence[ManifestRow]) -> dict[str, int]:
    counts = {"synthetic": 0, "photographed": 0, "unknown": 0, "unset": 0}
    seen: set[str] = set()
    for row in rows:
        if row.image in seen:
            continue
        seen.add(row.image)
        source = row.source_type if row.source_type in counts else "unset"
        counts[source] += 1
    return counts


def _order_coverage(rows: Sequence[ManifestRow], order: int) -> dict[str, object]:
    chosen = [row for row in rows if row.order == order]
    images = {row.image for row in chosen}
    target = COLLECTION_TARGETS[order]
    return {
        "order": order,
        "photographs": len(images),
        "target": target,
        "shortfall": max(0, target - len(images)),
        "train": len({row.image for row in chosen if row.development_split == "train"}),
        "holdout": len(
            {row.image for row in chosen if row.development_split == "holdout"}
        ),
        "cells": len(chosen),
    }


def _digit_coverage(rows: Sequence[ManifestRow], digit: int) -> dict[str, object]:
    chosen = [row for row in rows if row.digit == digit]
    return {
        "class": digit,
        "cells": len(chosen),
        "photographs": len({row.image for row in chosen}),
        "present": bool(chosen),
    }


def _condition_rows(
    rows: Sequence[ManifestRow],
    notes: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    images = sorted({row.image for row in rows})
    report: list[dict[str, object]] = []
    for image in images:
        recorded = notes.get(image, {})
        fields = {
            field: recorded.get(field) or "not recorded" for field in CONDITION_FIELDS
        }
        report.append({"image": image, **fields})
    return report


def _missing(
    orders: list[dict[str, object]],
    digits: list[dict[str, object]],
    conditions: list[dict[str, object]],
) -> list[str]:
    missing: list[str] = []
    for order in orders:
        shortfall = int(order["shortfall"])
        if shortfall:
            missing.append(
                f"{order['order']}×{order['order']} has {order['photographs']} "
                f"photographed photographs; the collection target is {order['target']}."
            )
        if int(order["holdout"]) == 0:
            missing.append(
                f"{order['order']}×{order['order']} has no photographed holdout."
            )
    for digit in digits:
        if not digit["present"]:
            missing.append(
                f"Digit {digit['class']} has no labelled photographed cells."
            )
    for row in conditions:
        for field in CONDITION_FIELDS:
            if row[field] == "not recorded":
                missing.append(f"{row['image']} has no recorded {field}.")
    if not conditions:
        missing.append("No photographed photographs are in the dataset.")
    return missing
