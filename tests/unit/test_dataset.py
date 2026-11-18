"""Dataset manifests and the held-out digit benchmark.

Synthetic glyphs check the workflow. They are not a photograph accuracy.
"""

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from app.dataset import evaluation_rows, load_manifest, load_split_file, prepare_dataset
from scripts.benchmark_digits import benchmark_dataset, main
from scripts.prepare_dataset import main as prepare_main
from tests.unit.vision_samples import draw_grid


def test_prepare_writes_blank_manifest_and_contact_sheet(tmp_path: Path) -> None:
    source = tmp_path / "photos"
    original = _save_grid(source / "puzzle.png", digits=((0, 1, "2"),))
    before = original.read_bytes()
    payload = prepare_dataset(source, tmp_path / "dataset")
    manifest = list(csv.DictReader((tmp_path / "dataset" / "manifest.csv").open()))
    recognition = json.loads(
        (tmp_path / "dataset" / "puzzle.png" / "recognition.json").read_text()
    )

    assert original.read_bytes() == before
    assert payload["labels_are_blank"] is True
    assert len(manifest) == 16
    assert manifest[1]["cell_id"] == "r00_c01"
    assert manifest[1]["cell"] == "puzzle.png/cells/r00_c01.png"
    assert manifest[1]["order"] == "4"
    assert manifest[1]["digit"] == ""
    assert all(row["digit"] == "" for row in manifest)
    assert any(item["digit"] == 2 for item in recognition)
    assert (tmp_path / "dataset" / "puzzle.png" / "contact_sheet.png").is_file()
    assert (tmp_path / "dataset" / "dataset.json").is_file()
    assert (tmp_path / "dataset" / "splits.csv").is_file()


def test_splits_follow_the_source_image(tmp_path: Path) -> None:
    source = tmp_path / "photos"
    _save_grid(source / "train.png")
    _save_grid(source / "held.png")
    splits = tmp_path / "splits.csv"
    splits.write_text(
        "image,split\ntrain.png,template\nheld.png,evaluation\n",
        encoding="utf-8",
    )
    prepare_dataset(source, tmp_path / "dataset", splits=splits)
    manifest = list(csv.DictReader((tmp_path / "dataset" / "manifest.csv").open()))
    by_image = {row["image"] for row in manifest}
    assert by_image == {"train.png", "held.png"}
    assert {row["split"] for row in manifest if row["image"] == "train.png"} == {
        "template"
    }
    assert {row["split"] for row in manifest if row["image"] == "held.png"} == {
        "evaluation"
    }
    counts = json.loads((tmp_path / "dataset" / "dataset.json").read_text())
    assert counts["split_counts"]["template"] == 1
    assert counts["split_counts"]["evaluation"] == 1


def test_split_file_rejects_conflicts_and_unknown_names(tmp_path: Path) -> None:
    both = tmp_path / "both.csv"
    both.write_text(
        "image,split\npuzzle.png,template\npuzzle.png,evaluation\n",
        encoding="utf-8",
    )
    try:
        load_split_file(both)
    except ValueError as exc:
        assert "more than one split" in str(exc)
    else:
        raise AssertionError("one image cannot be in both splits")

    source = tmp_path / "photos"
    _save_grid(source / "puzzle.png")
    missing = tmp_path / "missing.csv"
    missing.write_text("image,split\nother.png,evaluation\n", encoding="utf-8")
    try:
        prepare_dataset(source, tmp_path / "dataset", splits=missing)
    except ValueError as exc:
        assert "were not extracted" in str(exc)
    else:
        raise AssertionError("unknown split images should fail")
    code = prepare_main(
        [
            str(source),
            "--output",
            str(tmp_path / "cli"),
            "--splits",
            str(tmp_path / "no.csv"),
        ]
    )
    assert code == 2


def test_invalid_annotations_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    _write_manifest(
        path,
        [("puzzle.png", "r00_c00", "cell.png", 0, 0, 4, "template", "8")],
    )
    try:
        load_manifest(path)
    except ValueError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("digit 8 is outside a 4×4 grid")

    _write_manifest(
        path,
        [("puzzle.png", "r00_c00", "cell.png", 0, 0, 9, "template", "true")],
    )
    try:
        load_manifest(path)
    except ValueError as exc:
        assert "invalid" in str(exc)
    else:
        raise AssertionError("boolean labels should fail")

    _write_manifest(
        path,
        [
            ("puzzle.png", "r00_c00", "a.png", 0, 0, 9, "template", "1"),
            ("puzzle.png", "r00_c01", "b.png", 0, 1, 9, "evaluation", "2"),
        ],
    )
    try:
        load_manifest(path)
    except ValueError as exc:
        assert "more than one split" in str(exc)
    else:
        raise AssertionError("cells from one image must share a split")


def test_evaluation_rows_exclude_template_images(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    _write_manifest(
        path,
        [
            (
                "train.png",
                "r00_c00",
                "train.png/cells/r00_c00.png",
                0,
                0,
                9,
                "template",
                "5",
            ),
            (
                "held.png",
                "r00_c00",
                "held.png/cells/r00_c00.png",
                0,
                0,
                9,
                "evaluation",
                "2",
            ),
        ],
    )
    rows = load_manifest(path)
    held = evaluation_rows(rows)
    assert [row.image for row in held] == ["held.png"]
    assert all(row.split == "evaluation" for row in held)


def test_missing_labels_do_not_produce_accuracy(tmp_path: Path) -> None:
    source = tmp_path / "photos"
    _save_grid(source / "held.png")
    splits = tmp_path / "splits.csv"
    splits.write_text("image,split\nheld.png,evaluation\n", encoding="utf-8")
    prepare_dataset(source, tmp_path / "dataset", splits=splits)
    report = benchmark_dataset(
        tmp_path / "dataset" / "manifest.csv",
        tmp_path / "benchmark",
    )
    saved = json.loads((tmp_path / "benchmark" / "report.json").read_text())
    config = json.loads((tmp_path / "benchmark" / "config.json").read_text())

    assert report["verified"] is False
    assert report["accuracy"] is None
    assert saved["accuracy"] is None
    assert all(row["predicted"] is None for row in report["predictions"])
    assert config["evaluation_image_count"] == 1
    assert (tmp_path / "benchmark" / "report.csv").is_file()
    code = main(
        [
            "--manifest",
            str(tmp_path / "missing.csv"),
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert code == 2


def test_benchmark_scores_held_out_cells_and_keeps_templates_out(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    cv2.imwrite(str(root / "five.png"), _glyph(5))
    cv2.imwrite(str(root / "two.png"), _glyph(2))
    cv2.imwrite(str(root / "empty.png"), np.full((48, 48), 255, dtype=np.uint8))
    cv2.imwrite(str(root / "odd.png"), _glyph(1))
    manifest = root / "manifest.csv"
    _write_manifest(
        manifest,
        [
            ("train.png", "r00_c00", "five.png", 0, 0, 9, "template", "5"),
            ("held.png", "r00_c00", "two.png", 0, 0, 9, "evaluation", "2"),
            ("held.png", "r00_c01", "empty.png", 0, 1, 4, "evaluation", "0"),
            ("held.png", "r01_c00", "odd.png", 1, 0, 9, "evaluation", "9"),
            ("held.png", "r01_c01", "five.png", 1, 1, 9, "evaluation", "uncertain"),
            ("held.png", "r02_c00", "two.png", 2, 0, 9, "evaluation", "invalid"),
        ],
    )
    report = benchmark_dataset(manifest, tmp_path / "benchmark")
    templates = (tmp_path / "benchmark" / "templates.csv").read_text(encoding="utf-8")
    config = json.loads((tmp_path / "benchmark" / "config.json").read_text())

    assert report["verified"] is True
    assert report["excluded_uncertain"] == 1
    assert report["excluded_invalid"] == 1
    assert report["scored_count"] == 3
    assert report["accuracy"] == round(2 / 3, 4)
    assert report["per_digit"]["2"]["precision"] == 1.0
    assert report["per_digit"]["2"]["recall"] == 1.0
    assert report["per_digit"]["9"]["recall"] == 0.0
    assert report["confusion"]["9"]
    assert report["blank_classification"]["blank"]["recall"] == 1.0
    assert report["by_order"]["9"]["labelled"] == 2
    assert report["by_order"]["4"]["labelled"] == 1
    assert report["by_image"]["held.png"]["labelled"] == 3
    assert "train.png" not in report["by_image"]
    assert "five.png" in templates
    assert "two.png" not in templates
    assert "empty.png" not in templates
    assert report["leakage_count"] == 1
    assert config["use_builtin"] is True
    incorrect = report["incorrect"]
    assert incorrect[0]["expected"] == 9
    assert incorrect[0]["path"]
    assert (tmp_path / "benchmark" / "report.csv").is_file()


def _save_grid(
    path: Path, digits: tuple[tuple[int, int, str], ...] = ()
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    drawn = draw_grid(4, cell=64, margin=32, line=2)
    for row, col, text in digits:
        origin_x = drawn.margin + col * drawn.cell + drawn.cell // 5
        origin_y = drawn.margin + row * drawn.cell + int(drawn.cell * 0.75)
        cv2.putText(
            drawn.image,
            text,
            (origin_x, origin_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            drawn.cell / 55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(path), drawn.image)
    return path


def _glyph(digit: int) -> np.ndarray:
    canvas = np.full((96, 96), 255, dtype=np.uint8)
    text = str(digit)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), _baseline = cv2.getTextSize(text, font, 2.1, 2)
    x = max(0, (96 - width) // 2)
    y = max(height + 2, (96 + height) // 2)
    cv2.putText(canvas, text, (x, y), font, 2.1, 0, 2, cv2.LINE_AA)
    return canvas


def _write_manifest(path: Path, rows: list[tuple[object, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        for image, cell_id, cell, row, column, order, split, digit in rows:
            writer.writerow(
                {
                    "image": image,
                    "cell_id": cell_id,
                    "cell": cell,
                    "row": row,
                    "column": column,
                    "order": order,
                    "split": split,
                    "digit": digit,
                }
            )
