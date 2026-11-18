"""Camera import requires confirmation and leaves the frozen split alone."""

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.cnn_dataset import assign_development_split
from app.dataset import ManifestRow, load_manifest
from app.dataset_import import ImportBatchError, import_photographed_photographs
from app.photographed_collection import collection_report
from app.recognition import digit_preprocess
from app.recognition.digit_recognizer import _MIN_MARGIN, _MIN_SIMILARITY
from ml.cnn.data import partition_examples
from scripts.import_photographed import main
from tests.unit.vision_samples import draw_grid


def test_provenance_is_not_taken_from_the_folder(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    source_dir = tmp_path / "photographed"
    source_dir.mkdir()
    source = _grid(source_dir / "puzzle.png", 4)
    before = (dataset / "manifest.csv").read_bytes()
    with pytest.raises(ValueError, match="explicitly confirmed"):
        import_photographed_photographs(
            dataset,
            [source],
            provenance="photographed",
            pool="holdout",
            confirm=False,
        )
    assert (dataset / "manifest.csv").read_bytes() == before
    assert not (dataset / "puzzle.png").exists()
    refused = main(
        [
            "--dataset",
            str(dataset),
            "--provenance",
            "folder",
            "--pool",
            "holdout",
            str(source),
        ]
    )
    assert refused == 2


def test_holdout_import_preserves_labels_and_skips_training(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"frozen-checkpoint")
    source = _grid(tmp_path / "camera-four.png", 4)
    report = import_photographed_photographs(
        dataset,
        [source],
        provenance="photographed",
        pool="holdout",
        confirm=True,
    )
    added = report["added"]
    assert isinstance(added, list)
    assert added[0]["order"] == 4
    assert added[0]["cells"] == 16
    assert added[0]["split"] == "template"
    assert added[0]["development_split"] == "holdout"
    assert added[0]["source_type"] == "photographed"
    rows = load_manifest(dataset / "manifest.csv")
    new_rows = [row for row in rows if row.image == "camera-four.png"]
    assert len(new_rows) == 16
    assert all(row.digit is None and row.marker is None for row in new_rows)
    kept = next(row for row in rows if row.image == "kept.png")
    assert kept.digit == 6
    assert kept.development_split == "validation"
    assert kept.source_type == "unknown"
    held = next(row for row in rows if row.image == "held.png")
    assert held.digit == 8
    assert held.split == "evaluation"
    assert held.development_split is None
    assignments = assign_development_split(rows)
    assert "camera-four.png" not in assignments
    assert "held.png" not in assignments
    partitioned = partition_examples(rows, dataset)
    selected = partitioned["train"]
    assert isinstance(selected, tuple)
    assert all(item.image != "camera-four.png" for item in selected)
    assert partitioned["skipped_holdout_photographs"] == 1
    assert checkpoint.read_bytes() == b"frozen-checkpoint"
    ink_cutoff = getattr(digit_preprocess, "DEFAULT_INK_DELTA", None)
    if ink_cutoff is None:
        ink_cutoff = digit_preprocess._INK_DELTA
    assert ink_cutoff == 28.0
    assert _MIN_SIMILARITY == 0.55
    assert _MIN_MARGIN == 0.04
    pipeline = Path("app/pipeline.py").read_text(encoding="utf-8")
    assert "import_photographed_photographs" not in pipeline


def test_a_failed_batch_rolls_back_and_duplicates_are_refused(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    before = (dataset / "manifest.csv").read_bytes()
    good = _grid(tmp_path / "good.png", 6)
    flat = tmp_path / "flat.png"
    cv2.imwrite(str(flat), np.full((64, 64), 200, dtype=np.uint8))
    with pytest.raises(ImportBatchError):
        import_photographed_photographs(
            dataset,
            [good, flat],
            provenance="photographed",
            pool="train",
            confirm=True,
        )
    assert (dataset / "manifest.csv").read_bytes() == before
    assert not (dataset / "good.png").exists()
    duplicate = tmp_path / "copy.png"
    original = dataset / "held.png" / "original.png"
    duplicate.write_bytes(original.read_bytes())
    with pytest.raises(ImportBatchError, match="duplicate of evaluation"):
        import_photographed_photographs(
            dataset,
            [duplicate],
            provenance="photographed",
            pool="holdout",
            confirm=True,
        )
    assert (dataset / "manifest.csv").read_bytes() == before


def test_coverage_names_gaps_without_inventing_conditions() -> None:
    rows = (
        _row("held.jpg", "evaluation", None, "unknown", 9, 8),
        _row("camera.jpg", "template", "holdout", "photographed", 9, None),
    )
    report = collection_report(rows, {})
    orders = report["by_order"]
    assert isinstance(orders, list)
    assert orders[2]["photographs"] == 1
    assert orders[2]["shortfall"] == 11
    assert orders[2]["holdout"] == 1
    assert orders[0]["photographs"] == 0
    digits = report["digits"]
    assert isinstance(digits, list)
    assert digits[8]["cells"] == 0
    conditions = report["conditions"]
    assert isinstance(conditions, list)
    assert conditions[0]["paper"] == "not recorded"
    assert report["generalisation_evaluated"] is False
    missing = report["missing"]
    assert isinstance(missing, list)
    assert any("4×4" in item for item in missing)
    assert any("Digit 7" in item for item in missing)
    assert "plain paper" not in " ".join(missing)


def _dataset(root: Path) -> Path:
    dataset = root / "dataset"
    manifest = dataset / "manifest.csv"
    dataset.mkdir()
    fields = [
        "image",
        "cell_id",
        "cell",
        "row",
        "column",
        "order",
        "split",
        "digit",
        "development_split",
        "source_type",
    ]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for image, split, role, source, digit in (
            ("held.png", "evaluation", "", "unknown", "8"),
            ("kept.png", "template", "validation", "unknown", "6"),
        ):
            folder = dataset / image / "cells"
            folder.mkdir(parents=True)
            blank = np.full((8, 8), 255, dtype=np.uint8)
            cv2.imwrite(str(folder / "r00_c00.png"), blank)
            original = dataset / image / "original.png"
            if split == "evaluation":
                page = np.full((96, 96), 30, dtype=np.uint8)
                page[:, :48] = 220
                cv2.imwrite(str(original), page)
            writer.writerow(
                {
                    "image": image,
                    "cell_id": "r00_c00",
                    "cell": f"{image}/cells/r00_c00.png",
                    "row": "0",
                    "column": "0",
                    "order": "9",
                    "split": split,
                    "digit": digit,
                    "development_split": role,
                    "source_type": source,
                }
            )
    splits = dataset / "splits.csv"
    with splits.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "split"])
        writer.writerow(["held.png", "evaluation"])
        writer.writerow(["kept.png", "template"])
    return dataset


def _grid(path: Path, order: int) -> Path:
    image = draw_grid(order, cell=32, margin=16, line=2).image
    cv2.imwrite(str(path), image)
    return path


def _row(
    image: str,
    split: str,
    role: str | None,
    source: str,
    order: int,
    digit: int | None,
) -> ManifestRow:
    return ManifestRow(
        image=image,
        cell_id="r00_c00",
        cell=f"{image}/cells/r00_c00.png",
        row=0,
        column=0,
        order=order,
        split=split,
        digit=digit,
        marker=None,
        development_split=role,
        source_type=source,
    )
