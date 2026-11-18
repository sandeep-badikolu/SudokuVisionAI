"""Ground-truth label checks, filters, and single-cell manifest updates."""

import csv
from pathlib import Path

import pytest

from app.dataset import ManifestRow, load_manifest
from app.ui.annotate import (
    AnnotationError,
    allowed_labels,
    filter_rows,
    move,
    progress,
    replace_row,
    resolve_cell,
    resolve_original,
    save_cell_label,
    with_label,
)


def test_allowed_labels_follow_the_grid_order() -> None:
    assert allowed_labels(4) == ("0", "1", "2", "3", "4", "uncertain", "invalid")
    assert allowed_labels(6)[-3:] == ("6", "uncertain", "invalid")
    assert allowed_labels(9)[0] == "0"
    assert "9" in allowed_labels(9)
    with pytest.raises(AnnotationError):
        allowed_labels(5)


def test_labels_outside_the_order_are_rejected() -> None:
    row = _row(order=4)
    with pytest.raises(AnnotationError, match="0 through 4"):
        with_label(row, "5")
    with pytest.raises(AnnotationError, match="0 through 4"):
        with_label(row, "true")
    assert with_label(row, "0").digit == 0
    assert with_label(row, "0").marker is None
    assert with_label(row, "uncertain").marker == "uncertain"
    assert with_label(row, "uncertain").digit is None
    assert with_label(row, "invalid").marker == "invalid"


def test_filters_and_progress_keep_splits_apart() -> None:
    rows = (
        _row("a.jpg", "r00_c00", "template", digit=None),
        _row("a.jpg", "r00_c01", "template", digit=2),
        _row("a.jpg", "r00_c02", "template", marker="uncertain"),
        _row("b.jpg", "r00_c00", "evaluation", marker="invalid"),
        _row("b.jpg", "r00_c01", "evaluation", digit=None),
    )
    template = filter_rows(rows, split="template", label_filter="unlabelled")
    assert [row.cell_id for row in template] == ["r00_c00"]
    labelled = filter_rows(rows, split="template", label_filter="labelled")
    assert [row.cell_id for row in labelled] == ["r00_c01"]
    uncertain = filter_rows(rows, split="template", label_filter="uncertain")
    assert uncertain[0].marker == "uncertain"
    invalid = filter_rows(rows, split="evaluation", label_filter="invalid")
    assert [row.image for row in invalid] == ["b.jpg"]
    assert progress(rows, "template") == {
        "total": 3,
        "labelled": 2,
        "unlabelled": 1,
    }
    assert progress(rows, "evaluation")["unlabelled"] == 1
    assert move(0, 3, -1) == 0
    assert move(2, 3, 1) == 2
    assert move(0, 0, 1) == 0


def test_save_updates_one_cell_and_preserves_the_others(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write(
        manifest,
        [
            _row("a.jpg", "r00_c00", "template"),
            _row("a.jpg", "r00_c01", "template", digit=1),
            _row("b.jpg", "r00_c00", "evaluation"),
        ],
    )
    original = (tmp_path / "a.jpg")
    original.mkdir()
    (original / "original.jpg").write_bytes(b"photo")
    cell = original / "cells"
    cell.mkdir()
    (cell / "r00_c00.png").write_bytes(b"cell")
    row = load_manifest(manifest)[0]
    assert resolve_cell(manifest, row) == (cell / "r00_c00.png").resolve()
    assert resolve_original(manifest, row) == (original / "original.jpg").resolve()

    save_cell_label(
        manifest,
        image="a.jpg",
        cell_id="r00_c00",
        order=9,
        label="uncertain",
    )
    saved = load_manifest(manifest)
    assert saved[0].marker == "uncertain"
    assert saved[0].digit is None
    assert saved[1].digit == 1
    assert saved[2].digit is None
    assert saved[2].split == "evaluation"
    updated = replace_row(saved, with_label(saved[0], "3"))
    assert updated[0].digit == 3
    assert updated[1].digit == 1
    with pytest.raises(AnnotationError, match="0 through 9"):
        save_cell_label(
            manifest,
            image="b.jpg",
            cell_id="r00_c00",
            order=9,
            label="invalid-digit",
        )
    reread = list(csv.DictReader(manifest.open()))
    assert reread[2]["digit"] == ""


def _row(
    image: str = "a.jpg",
    cell_id: str = "r00_c00",
    split: str = "template",
    *,
    order: int = 9,
    digit: int | None = None,
    marker: str | None = None,
) -> ManifestRow:
    row_index = int(cell_id[1:3])
    column = int(cell_id[5:7])
    return ManifestRow(
        image=image,
        cell_id=cell_id,
        cell=f"{image}/cells/{cell_id}.png",
        row=row_index,
        column=column,
        order=order,
        split=split,
        digit=digit,
        marker=marker,
    )


def _write(path: Path, rows: list[ManifestRow]) -> None:
    fields = [
        "image",
        "cell_id",
        "cell",
        "row",
        "column",
        "order",
        "split",
        "digit",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if row.marker is not None:
                digit = row.marker
            elif row.digit is None:
                digit = ""
            else:
                digit = str(row.digit)
            writer.writerow(
                {
                    "image": row.image,
                    "cell_id": row.cell_id,
                    "cell": row.cell,
                    "row": row.row,
                    "column": row.column,
                    "order": row.order,
                    "split": row.split,
                    "digit": digit,
                }
            )
