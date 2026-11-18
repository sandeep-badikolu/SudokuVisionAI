"""Add photographs to the template split without touching evaluation data.

New cells are written with an empty digit. Recognizer output is not read and
is not stored as ground truth. A file that matches an evaluation photograph
is refused.
"""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from app.dataset import MANIFEST_FIELDS, SOURCE_TYPES, load_manifest
from app.vision.errors import VisionError
from app.vision.grid_detector import extract_cells, extract_cells_faint_nine
from app.vision.image_loader import load_image

_HASH_SIDE = 16
_SIMILAR_HAMMING = 4


class ImportBatchError(ValueError):
    """A batch was refused and the dataset was put back."""

    def __init__(self, message: str, report: dict[str, object]) -> None:
        super().__init__(message)
        self.report = report


def import_template_photographs(
    dataset_dir: Path,
    sources: list[Path],
    *,
    source_type: str | None = None,
    faint_grid: bool = False,
    development_split: str = "train",
) -> dict[str, object]:
    """Extract new photographs into ``dataset_dir`` as unlabelled template cells.

    Every cell from one photograph gets the same split. Exact duplicates are
    refused. A photograph that is the same file as, or suspiciously similar
    to, an evaluation photograph is refused. Similar template photographs are
    imported and reported as warnings. ``source_type`` is written only when
    the caller sets it. Existing rows keep the source they already have.
    """
    if development_split not in {"train", "holdout"}:
        raise ValueError("new photographs must be train or holdout")
    if source_type is not None and source_type not in SOURCE_TYPES:
        raise ValueError(
            "source type must be synthetic, photographed, or unknown"
        )
    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
    rows = load_manifest(manifest_path)
    known = _known_photographs(dataset_dir, rows)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        raw_rows = list(reader)
    if list(MANIFEST_FIELDS) != fieldnames[: len(MANIFEST_FIELDS)]:
        raise ValueError(
            "manifest columns do not match the dataset schema. "
            "A development_split column is required so existing "
            "assignments are not recalculated."
        )
    if source_type is not None and "source_type" not in fieldnames:
        fieldnames.append("source_type")
    added: list[dict[str, object]] = []
    rejected: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    for source in sources:
        outcome = _consider_source(dataset_dir, source, known)
        if outcome["status"] == "rejected":
            rejected.append(
                {"image": source.name, "reason": str(outcome["reason"])}
            )
            continue
        warning = outcome.get("warning")
        if warning:
            warnings.append({"image": source.name, "reason": str(warning)})
        extracted = _extract_template(
            dataset_dir,
            source,
            source_type=source_type,
            faint_grid=faint_grid,
            development_split=development_split,
        )
        if extracted["status"] == "rejected":
            rejected.append(
                {"image": source.name, "reason": str(extracted["reason"])}
            )
            continue
        new_rows = extracted["rows"]
        assert isinstance(new_rows, list)
        raw_rows.extend(new_rows)
        known[source.name] = outcome["fingerprint"]
        added.append(
            {
                "image": source.name,
                "order": extracted["order"],
                "cells": len(new_rows),
                "split": "template",
                "development_split": development_split,
                "source_type": source_type or "",
            }
        )
    if added:
        _write_manifest(manifest_path, fieldnames, raw_rows)
        names = [str(item["image"]) for item in added]
        _append_splits(dataset_dir / "splits.csv", names)
    return {
        "added": added,
        "rejected": rejected,
        "warnings": warnings,
        "labels_written": False,
    }


_BATCH_SIZE = 5


def import_nine_by_nine_batch(
    dataset_dir: Path,
    sources: list[Path],
) -> dict[str, object]:
    """Import exactly five new 9×9 photographs as unlabelled training cells.

    The existing importer extracts the cells and leaves every digit empty.
    A batch that is not five distinct 9×9 grids is removed again, including
    any photographs that were extracted before the failure was known.
    Existing labels and development roles are not rewritten.
    """
    if len(sources) != _BATCH_SIZE:
        raise ValueError(
            f"the batch must contain exactly {_BATCH_SIZE} photographs"
        )
    names = [source.name for source in sources]
    if len(set(names)) != len(names):
        raise ValueError("the batch photographs must have distinct filenames")
    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
    before_rows = load_manifest(manifest_path)
    before_manifest = manifest_path.read_bytes()
    splits_path = dataset_dir / "splits.csv"
    before_splits = splits_path.read_bytes() if splits_path.is_file() else None
    prior = _cell_snapshot(before_rows)
    report = import_template_photographs(dataset_dir, list(sources))
    try:
        _require_new_training_batch(
            manifest_path,
            report,
            prior,
            count=_BATCH_SIZE,
            description="five new 9×9 photographs",
            source_type=None,
        )
    except ValueError:
        _restore_batch(
            dataset_dir,
            before_manifest,
            splits_path,
            before_splits,
            report["added"],
        )
        raise
    return report


_PRINTED_BATCH_SIZE = 12
SYNTHETIC_PHOTOGRAPHS = frozenset(
    {
        "sudoku_01.jpg",
        "sudoku_02.jpg",
        "sudoku_03.jpg",
        "sudoku_04.jpg",
        "sudoku_05.jpg",
    }
)
_CellKey = tuple[str, str]
_CellState = tuple[int | None, str | None, str, str | None, str | None]


def import_printed_nine_by_nine(
    dataset_dir: Path,
    sources: list[Path],
) -> dict[str, object]:
    """Import exactly twelve real 9×9 photographs as unlabelled training cells.

    Each new photograph is marked ``photographed`` and ``development_split``
    train. Digits stay empty. Existing labels, roles, and source types are
    restored if the batch is not twelve distinct 9×9 grids.
    """
    if len(sources) != _PRINTED_BATCH_SIZE:
        raise ValueError(
            f"the batch must contain exactly {_PRINTED_BATCH_SIZE} photographs"
        )
    names = [source.name for source in sources]
    if len(set(names)) != len(names):
        raise ValueError("the batch photographs must have distinct filenames")
    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
    before_rows = load_manifest(manifest_path)
    before_manifest = manifest_path.read_bytes()
    splits_path = dataset_dir / "splits.csv"
    before_splits = splits_path.read_bytes() if splits_path.is_file() else None
    prior = _cell_snapshot(before_rows)
    report = import_template_photographs(
        dataset_dir,
        list(sources),
        source_type="photographed",
    )
    try:
        _require_new_training_batch(
            manifest_path,
            report,
            prior,
            count=_PRINTED_BATCH_SIZE,
            description="twelve new 9×9 photographs",
            source_type="photographed",
        )
    except ValueError:
        _restore_batch(
            dataset_dir,
            before_manifest,
            splits_path,
            before_splits,
            report["added"],
        )
        raise
    return report


_SYNTHETIC_BATCH_SIZE = 20


def import_synthetic_nine_by_nine(
    dataset_dir: Path,
    sources: list[Path],
) -> dict[str, object]:
    """Import exactly twenty synthetic 9×9 images as unlabelled training cells.

    Each new photograph is marked ``synthetic`` and ``development_split``
    train. Digits stay empty. A faint inner grid may be equal-sliced when
    the ordinary line check rejects it. Existing labels, roles, and source
    types are restored if any image in the batch fails.
    """
    if len(sources) != _SYNTHETIC_BATCH_SIZE:
        raise ValueError(
            f"the batch must contain exactly {_SYNTHETIC_BATCH_SIZE} photographs"
        )
    names = [source.name for source in sources]
    if len(set(names)) != len(names):
        raise ValueError("the batch photographs must have distinct filenames")
    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
    before_rows = load_manifest(manifest_path)
    before_manifest = manifest_path.read_bytes()
    splits_path = dataset_dir / "splits.csv"
    before_splits = splits_path.read_bytes() if splits_path.is_file() else None
    prior = _cell_snapshot(before_rows)
    report = import_template_photographs(
        dataset_dir,
        list(sources),
        source_type="synthetic",
        faint_grid=True,
    )
    try:
        _require_new_training_batch(
            manifest_path,
            report,
            prior,
            count=_SYNTHETIC_BATCH_SIZE,
            description="twenty new 9×9 photographs",
            source_type="synthetic",
        )
    except ValueError as exc:
        _restore_batch(
            dataset_dir,
            before_manifest,
            splits_path,
            before_splits,
            report["added"],
        )
        report["restored"] = True
        raise ImportBatchError(str(exc), report) from exc
    report["restored"] = False
    return report


def import_photographed_photographs(
    dataset_dir: Path,
    sources: list[Path],
    *,
    provenance: str,
    pool: str,
    confirm: bool,
) -> dict[str, object]:
    """Import genuine camera photographs as unlabelled template cells.

    ``provenance`` must be the word ``photographed`` and ``confirm`` must be
    true. A folder name is not provenance. ``pool`` is ``train`` or
    ``holdout``. Holdout cells are not the frozen evaluation split and are
    not a training role. Digits are left empty. The batch is restored if
    any photograph is rejected.
    """
    if not confirm or provenance != "photographed":
        raise ValueError(
            "provenance must be explicitly confirmed as photographed"
        )
    if pool not in {"train", "holdout"}:
        raise ValueError("pool must be train or holdout")
    if not sources:
        raise ValueError("at least one photograph is required")
    names = [source.name for source in sources]
    if len(set(names)) != len(names):
        raise ValueError("the batch photographs must have distinct filenames")
    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
    before_rows = load_manifest(manifest_path)
    before_manifest = manifest_path.read_bytes()
    splits_path = dataset_dir / "splits.csv"
    before_splits = splits_path.read_bytes() if splits_path.is_file() else None
    prior = _cell_snapshot(before_rows)
    frozen = {
        row.image for row in before_rows if row.split == "evaluation"
    }
    report = import_template_photographs(
        dataset_dir,
        list(sources),
        source_type="photographed",
        development_split=pool,
    )
    try:
        _require_photographed_batch(manifest_path, report, prior, pool, frozen)
    except ValueError as exc:
        _restore_batch(
            dataset_dir,
            before_manifest,
            splits_path,
            before_splits,
            report["added"],
        )
        report["restored"] = True
        raise ImportBatchError(str(exc), report) from exc
    report["restored"] = False
    report["pool"] = pool
    return report


def format_import_report(report: dict[str, object]) -> str:
    """Describe imported photographs and any failures."""
    added = report.get("added")
    rejected = report.get("rejected")
    warnings = report.get("warnings")
    if not isinstance(added, list):
        added = []
    if not isinstance(rejected, list):
        rejected = []
    if not isinstance(warnings, list):
        warnings = []
    cells = sum(int(item["cells"]) for item in added)
    lines = [
        "# Import report",
        "",
        f"Imported photographs: {len(added)}.",
        f"Extracted cells: {cells}.",
        f"Labels written: {report.get('labels_written', False)}.",
        f"Restored: {bool(report.get('restored'))}.",
        "",
        "| File | Cells | Order | Split | Development | Source |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for item in added:
        lines.append(
            f"| {item['image']} | {item['cells']} | {item['order']} | "
            f"{item['split']} | {item['development_split']} | {item['source_type']} |"
        )
    if not added:
        lines.append("| — | 0 | — | — | — | — |")
    lines.extend(["", "## Failures", ""])
    if rejected:
        for item in rejected:
            lines.append(f"- {item['image']}: {item['reason']}")
    else:
        lines.append("None.")
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for item in warnings:
            lines.append(f"- {item['image']}: {item['reason']}")
    error = report.get("error")
    if error:
        lines.extend(["", f"Error: {error}"])
    return "\n".join(lines) + "\n"


def record_source_types(
    manifest: Path,
    *,
    synthetic: frozenset[str] = SYNTHETIC_PHOTOGRAPHS,
    backup: Path | None = None,
    require_synthetic: bool = False,
) -> dict[str, str]:
    """Fill missing source types without changing labels or roles.

    Names in ``synthetic`` are marked synthetic. A photograph that already
    has another source type is left unchanged, except a known synthetic
    photograph marked as something else, which is refused. Every other
    photograph with no source type is marked unknown. Photographed is never
    inferred.
    """
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest}")
    raw_rows, fieldnames = _read_rows(manifest)
    loaded = load_manifest(manifest)
    present = {row.image for row in loaded}
    missing_names = sorted(synthetic - present)
    if require_synthetic and missing_names:
        joined = ", ".join(missing_names)
        raise ValueError(f"known synthetic photographs are missing: {joined}")
    desired = _desired_sources(loaded, synthetic)
    if all(row.source_type == desired[row.image] for row in loaded):
        return desired
    if backup is not None:
        if backup.exists():
            raise ValueError(f"backup already exists: {backup}")
        backup.write_bytes(manifest.read_bytes())
    original = manifest.read_bytes()
    if "source_type" not in fieldnames:
        fieldnames = [*fieldnames, "source_type"]
    snapshot = [_raw_identity(row) for row in raw_rows]
    for row in raw_rows:
        image = (row.get("image") or "").strip()
        row["source_type"] = desired[image]
    _write_manifest(manifest, fieldnames, raw_rows)
    rewritten, _ = _read_rows(manifest)
    stored = load_manifest(manifest)
    unchanged = [_raw_identity(row) for row in rewritten] == snapshot
    stored_ok = all(row.source_type == desired[row.image] for row in stored)
    if not unchanged or not stored_ok:
        manifest.write_bytes(original)
        raise ValueError("recording source types changed a label or assignment")
    return desired


def _desired_sources(
    rows: tuple[object, ...],
    synthetic: frozenset[str],
) -> dict[str, str]:
    desired: dict[str, str] = {}
    for row in rows:
        image = str(row.image)  # type: ignore[attr-defined]
        current = row.source_type  # type: ignore[attr-defined]
        if image in synthetic:
            if current not in {None, "synthetic"}:
                raise ValueError(
                    f"{image} is a known synthetic photograph "
                    f"but is marked {current}"
                )
            chosen = "synthetic"
        elif current is None:
            chosen = "unknown"
        else:
            chosen = str(current)
        previous = desired.get(image)
        if previous is not None and previous != chosen:
            raise ValueError(f"{image} has more than one source type")
        desired[image] = chosen
    return desired


def _raw_identity(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("image") or "",
        row.get("cell_id") or "",
        row.get("split") or "",
        row.get("digit") or "",
        row.get("development_split") or "",
    )


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _cell_snapshot(
    rows: tuple[object, ...],
) -> dict[_CellKey, _CellState]:
    return {
        (row.image, row.cell_id): (  # type: ignore[attr-defined]
            row.digit,  # type: ignore[attr-defined]
            row.marker,  # type: ignore[attr-defined]
            row.split,  # type: ignore[attr-defined]
            row.development_split,  # type: ignore[attr-defined]
            row.source_type,  # type: ignore[attr-defined]
        )
        for row in rows
    }


def _require_new_training_batch(
    manifest_path: Path,
    report: dict[str, object],
    prior: dict[_CellKey, _CellState],
    *,
    count: int,
    description: str,
    source_type: str | None,
) -> None:
    added = report["added"]
    rejected = report["rejected"]
    assert isinstance(added, list)
    assert isinstance(rejected, list)
    if rejected or len(added) != count:
        reason = rejected[0]["reason"] if rejected else "not every photograph was added"
        raise ValueError(f"the batch was not {description}: {reason}")
    rows = load_manifest(manifest_path)
    for item in added:
        image = str(item["image"])
        cells = [row for row in rows if row.image == image]
        if len(cells) != 81 or any(row.order != 9 for row in cells):
            raise ValueError(f"{image} did not produce 81 cells of a 9×9 grid")
        if any(
            row.split != "template" or row.development_split != "train"
            for row in cells
        ):
            raise ValueError(f"{image} was not assigned to training")
        if source_type is not None and any(
            row.source_type != source_type for row in cells
        ):
            raise ValueError(f"{image} was not marked {source_type}")
        if any(row.digit is not None or row.marker is not None for row in cells):
            raise ValueError(f"{image} received a ground-truth label")
    for row in rows:
        key = (row.image, row.cell_id)
        if key not in prior:
            continue
        current = (
            row.digit,
            row.marker,
            row.split,
            row.development_split,
            row.source_type,
        )
        if current != prior[key]:
            raise ValueError(f"{row.image} {row.cell_id} changed during import")


def _require_photographed_batch(
    manifest_path: Path,
    report: dict[str, object],
    prior: dict[_CellKey, _CellState],
    pool: str,
    frozen_evaluation: set[str],
) -> None:
    added = report["added"]
    rejected = report["rejected"]
    assert isinstance(added, list)
    assert isinstance(rejected, list)
    if rejected:
        raise ValueError(
            f"the photographed batch was refused: {rejected[0]['reason']}"
        )
    if not added:
        raise ValueError("the photographed batch was refused: nothing was added")
    rows = load_manifest(manifest_path)
    evaluation = {row.image for row in rows if row.split == "evaluation"}
    if evaluation != frozen_evaluation:
        raise ValueError("the frozen evaluation photographs changed")
    for item in added:
        image = str(item["image"])
        if image in frozen_evaluation or item["split"] != "template":
            raise ValueError(f"{image} was assigned to the frozen evaluation split")
        cells = [row for row in rows if row.image == image]
        order = int(item["order"])
        if order not in {4, 6, 9} or len(cells) != order * order:
            raise ValueError(f"{image} did not produce a 4, 6, or 9 grid")
        if any(row.order != order for row in cells):
            raise ValueError(f"{image} has more than one grid order")
        if any(
            row.split != "template"
            or row.development_split != pool
            or row.source_type != "photographed"
            for row in cells
        ):
            raise ValueError(f"{image} was not stored as photographed {pool}")
        if any(row.digit is not None or row.marker is not None for row in cells):
            raise ValueError(f"{image} received a ground-truth label")
    for row in rows:
        key = (row.image, row.cell_id)
        if key not in prior:
            continue
        current = (
            row.digit,
            row.marker,
            row.split,
            row.development_split,
            row.source_type,
        )
        if current != prior[key]:
            raise ValueError(f"{row.image} {row.cell_id} changed during import")


def _restore_batch(
    dataset_dir: Path,
    manifest_bytes: bytes,
    splits_path: Path,
    splits_bytes: bytes | None,
    added: object,
) -> None:
    (dataset_dir / "manifest.csv").write_bytes(manifest_bytes)
    if splits_bytes is None:
        if splits_path.is_file():
            splits_path.unlink()
    else:
        splits_path.write_bytes(splits_bytes)
    if not isinstance(added, list):
        return
    for item in added:
        folder = dataset_dir / str(item["image"])
        if folder.is_dir():
            shutil.rmtree(folder)


def _consider_source(
    dataset_dir: Path,
    source: Path,
    known: dict[str, dict[str, object]],
) -> dict[str, object]:
    if not source.is_file():
        return {"status": "rejected", "reason": f"image does not exist: {source}"}
    name = source.name
    if name in known:
        split = known[name]["split"]
        if split == "evaluation":
            return {
                "status": "rejected",
                "reason": f"{name} is already an evaluation photograph",
            }
        return {"status": "rejected", "reason": f"{name} is already in the dataset"}
    digest = _sha256(source)
    bits = _average_hash(source)
    for image, record in known.items():
        if record["sha256"] == digest:
            if record["split"] == "evaluation":
                reason = f"{name} is a duplicate of evaluation photograph {image}"
            else:
                reason = f"{name} is a duplicate of {image}"
            return {"status": "rejected", "reason": reason}
    warning = ""
    stored_bits = bits
    for image, record in known.items():
        other = record["bits"]
        assert isinstance(other, np.ndarray)
        distance = int(np.count_nonzero(stored_bits != other))
        if distance > _SIMILAR_HAMMING:
            continue
        if record["split"] == "evaluation":
            return {
                "status": "rejected",
                "reason": (
                    f"{name} looks like evaluation photograph {image} "
                    f"(hash distance {distance})"
                ),
            }
        warning = (
            f"{name} looks like template photograph {image} "
            f"(hash distance {distance})"
        )
    return {
        "status": "accepted",
        "warning": warning,
        "fingerprint": {"split": "template", "sha256": digest, "bits": bits},
    }


def _extract_template(
    dataset_dir: Path,
    source: Path,
    *,
    source_type: str | None,
    faint_grid: bool = False,
    development_split: str = "train",
) -> dict[str, object]:
    destination = dataset_dir / source.name
    if destination.exists():
        return {
            "status": "rejected",
            "reason": f"{source.name} already has a dataset folder",
        }
    destination.mkdir(parents=True)
    try:
        loaded = load_image(source)
        suffix = source.suffix.lower()
        shutil.copy2(source, destination / f"original{suffix}")
        if faint_grid:
            detection, patches = extract_cells_faint_nine(loaded.original)
        else:
            detection, patches = extract_cells(loaded.original)
        cell_dir = destination / "cells"
        cell_dir.mkdir()
        rows: list[dict[str, str]] = []
        for patch in patches:
            cell_id = f"r{patch.row:02d}_c{patch.col:02d}"
            relative = f"{source.name}/cells/{cell_id}.png"
            if not cv2.imwrite(str(cell_dir / f"{cell_id}.png"), patch.image):
                raise ValueError(f"could not write {relative}")
            payload = {
                "image": source.name,
                "cell_id": cell_id,
                "cell": relative,
                "row": str(patch.row),
                "column": str(patch.col),
                "order": str(detection.order),
                "split": "template",
                "digit": "",
                "development_split": development_split,
            }
            if source_type is not None:
                payload["source_type"] = source_type
            rows.append(payload)
        if len(rows) != detection.order ** 2:
            raise ValueError(
                f"{source.name} produced {len(rows)} cells for order {detection.order}"
            )
        orders = {row["order"] for row in rows}
        if len(orders) != 1:
            raise ValueError(f"{source.name} has more than one grid order")
    except (VisionError, ValueError, OSError) as exc:
        shutil.rmtree(destination, ignore_errors=True)
        return {"status": "rejected", "reason": str(exc)}
    return {"status": "added", "order": detection.order, "rows": rows}


def _known_photographs(
    dataset_dir: Path, rows: tuple[object, ...]
) -> dict[str, dict[str, object]]:
    known: dict[str, dict[str, object]] = {}
    for row in rows:
        image = str(row.image)  # type: ignore[attr-defined]
        if image in known:
            continue
        path = _original_path(dataset_dir, image)
        record: dict[str, object] = {
            "split": row.split,  # type: ignore[attr-defined]
            "sha256": "",
            "bits": np.zeros(_HASH_SIDE * _HASH_SIDE, dtype=np.uint8),
        }
        if path is not None:
            record["sha256"] = _sha256(path)
            record["bits"] = _average_hash(path)
        known[image] = record
    return known


def _original_path(dataset_dir: Path, image: str) -> Path | None:
    folder = dataset_dir / image
    copied = folder / f"original{Path(image).suffix.lower()}"
    if copied.is_file():
        return copied
    direct = dataset_dir / image
    if direct.is_file():
        return direct
    return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _average_hash(path: Path) -> np.ndarray:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return np.zeros(_HASH_SIDE * _HASH_SIDE, dtype=np.uint8)
    small = cv2.resize(gray, (_HASH_SIDE, _HASH_SIDE), interpolation=cv2.INTER_AREA)
    return (small > float(small.mean())).astype(np.uint8).reshape(-1)


def _append_splits(path: Path, images: list[str]) -> None:
    if not path.is_file():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if "image" not in fieldnames or "split" not in fieldnames:
        return
    present = {(row.get("image") or "").strip() for row in rows}
    for image in images:
        if image not in present:
            rows.append({"image": image, "split": "template"})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_manifest(
    path: Path, fieldnames: list[str], rows: list[dict[str, str]]
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            payload = {name: row.get(name, "") for name in fieldnames}
            writer.writerow(payload)
    os.replace(temporary, path)
