"""End-to-end pipeline reports. Synthetic drawings are not photograph accuracy."""

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from app.pipeline import (
    CellLabel,
    build_metrics,
    load_labels,
    run_directory,
    run_paths,
    write_label_template,
)
from app.recognition.digit_templates import load_template_manifest
from app.recognition.errors import RecognitionError
from scripts.run_pipeline import main
from tests.unit.vision_samples import draw_grid


def test_single_synthetic_image_writes_each_stage(tmp_path: Path) -> None:
    image = _save_grid(tmp_path / "images" / "puzzle.png")
    report = run_paths([image], tmp_path / "out")
    folder = tmp_path / "out" / "puzzle.png"

    assert report["success_count"] == 1
    assert (folder / "original.png").is_file()
    assert (folder / "detected_grid.png").is_file()
    assert (folder / "perspective_grid.png").is_file()
    assert (folder / "contact_sheet.png").is_file()
    assert (folder / "cells" / "r00_c00.png").is_file()
    assert (folder / "recognition.json").is_file()
    assert (folder / "reconstruction.json").is_file()
    assert (tmp_path / "out" / "report.json").is_file()
    assert (tmp_path / "out" / "report.csv").is_file()
    template = list(csv.DictReader((tmp_path / "out" / "labels_template.csv").open()))
    assert len(template) == 16
    assert template[0]["digit"] == ""
    assert template[0]["image"] == "puzzle.png"
    saved = json.loads((folder / "recognition.json").read_text())
    assert len(saved) == 16
    assert all(row["digit"] == "" for row in template)


def test_directory_continues_after_one_bad_image(tmp_path: Path) -> None:
    images = tmp_path / "images"
    good = _save_grid(images / "good.png")
    white = np.full((64, 64, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(images / "blank.png"), white)
    (images / "notes.txt").write_text("not an image", encoding="utf-8")
    (images / "broken.png").write_bytes(b"not a png")

    report = run_directory(images, tmp_path / "out")

    by_name = {row["filename"]: row for row in report["results"]}
    assert set(by_name) == {"good.png", "blank.png", "broken.png"}
    assert by_name["good.png"]["success"] is True
    assert by_name["blank.png"]["success"] is False
    assert by_name["blank.png"]["failure_reason"]
    assert by_name["broken.png"]["success"] is False
    assert report["failure_count"] == 2
    assert good.is_file()


def test_missing_directory_templates_and_labels_stop_before_scoring(
    tmp_path: Path,
) -> None:
    try:
        run_directory(tmp_path / "missing", tmp_path / "out")
    except FileNotFoundError as exc:
        assert "image directory" in str(exc)
    else:
        raise AssertionError("missing directory should fail")

    images = tmp_path / "images"
    _save_grid(images / "puzzle.png")
    try:
        run_directory(images, tmp_path / "out", template_manifest=tmp_path / "no.csv")
    except RecognitionError as exc:
        assert "template manifest" in str(exc)
    else:
        raise AssertionError("missing templates should fail")

    try:
        run_directory(images, tmp_path / "out", labels=tmp_path / "no-labels.csv")
    except FileNotFoundError as exc:
        assert "ground-truth" in str(exc)
    else:
        raise AssertionError("missing labels should fail")

    assert main(["missing.png", "--output", str(tmp_path / "cli")]) == 2
    missing_templates = [
        str(images),
        "--output",
        str(tmp_path / "cli"),
        "--templates",
        str(tmp_path / "no.csv"),
    ]
    assert main(missing_templates) == 2


def test_unlabelled_template_does_not_invent_accuracy(tmp_path: Path) -> None:
    image = _save_grid(tmp_path / "images" / "puzzle.png")
    report = run_paths([image], tmp_path / "out")

    assert report["metrics"]["ground_truth"] is False
    assert report["metrics"]["digit_recognition"] is None
    assert report["metrics"]["blank_occupied"] is None
    assert report["metrics"]["reconstruction"] is None
    assert "accuracy" not in report["metrics"]

    labels = tmp_path / "out" / "labels_template.csv"
    scored = run_paths([image], tmp_path / "scored", labels=labels)
    assert scored["metrics"]["ground_truth"] is True
    assert scored["metrics"]["digit_recognition"]["accuracy"] is None
    assert scored["metrics"]["digit_recognition"]["compared"] == 0


def test_metrics_keep_each_stage_separate() -> None:
    cells = [
        _cell(0, 0, "blank", "blank", 0),
        _cell(0, 1, "occupied", "recognized", 2),
        _cell(0, 2, "uncertain", "uncertain", None),
        _cell(0, 3, "occupied", "recognized", 3),
    ]
    results = [
        _image("a.png", cells, validation="valid", solution="unique"),
        _image("b.png", [], success=False, detected=False),
    ]
    labels = (
        CellLabel("a.png", "", 0, 0, 4, 0, None, None),
        CellLabel("a.png", "", 0, 1, 4, 2, None, None),
        CellLabel("a.png", "", 0, 2, 4, 2, None, None),
        CellLabel("a.png", "", 0, 3, 4, 5, None, None),
        CellLabel("a.png", "", 1, 0, 4, None, "uncertain", None),
        CellLabel("a.png", "", 1, 1, 4, None, "invalid", None),
    )
    metrics = build_metrics(
        results,
        labels,
        template_paths=set(),
        label_root=None,
        output_dir=Path("."),
    )

    assert metrics["grid_detection"]["rate"] == 0.5
    assert metrics["cell_extraction"]["rate"] == 1.0
    assert metrics["validation"]["pass_rate"] == 1.0
    assert metrics["solver"]["completion_rate"] == 1.0
    digits = metrics["digit_recognition"]
    assert digits["compared"] == 4
    assert digits["correct"] == 2
    assert digits["accuracy"] == 0.5
    assert digits["uncertain"] == 1
    assert digits["confusion"]["2"] == {"2": 1, "uncertain": 1}
    assert digits["confusion"]["5"] == {"3": 1}
    assert digits["per_digit"]["2"]["precision"] == 1.0
    assert digits["per_digit"]["2"]["recall"] == 0.5
    assert digits["per_digit"]["5"]["recall"] == 0.0
    assert metrics["blank_occupied"]["blank"]["precision"] == 1.0
    assert metrics["blank_occupied"]["occupied"]["recall"] == round(2 / 3, 4)
    assert metrics["reconstruction"]["cell_accuracy"] == 0.5
    assert metrics["reconstruction"]["puzzle_exact_rate"] is None
    assert [row["expected"] for row in metrics["incorrect_predictions"]] == [5]
    assert set(metrics) >= {
        "grid_detection",
        "cell_extraction",
        "digit_recognition",
        "validation",
        "solver",
    }


def test_template_paths_are_excluded_and_evaluation_rows_are_not_loaded(
    tmp_path: Path,
) -> None:
    template = _ink(tmp_path / "bank" / "two.png")
    held_out = _ink(tmp_path / "bank" / "held.png")
    manifest = tmp_path / "bank" / "templates.csv"
    manifest.write_text(
        "filename,digit,role\ntwo.png,2,template\nheld.png,2,evaluation\n",
        encoding="utf-8",
    )
    bank = load_template_manifest(manifest)
    assert str(template.resolve()) in bank.sources
    assert str(held_out.resolve()) not in bank.sources

    cells = [_cell(0, 0, "occupied", "recognized", 2)]
    labels = (
        CellLabel("a.png", str(template.resolve()), 0, 0, 4, 2, None, None),
        CellLabel("b.png", str(held_out.resolve()), 0, 0, 4, 2, None, None),
    )
    metrics = build_metrics(
        [
            _image("a.png", cells),
            _image("b.png", cells),
        ],
        labels,
        template_paths={template.resolve()},
        label_root=tmp_path,
        output_dir=tmp_path,
    )
    assert metrics["leakage_count"] == 1
    assert metrics["digit_recognition"]["compared"] == 1
    assert metrics["digit_recognition"]["correct"] == 1


def test_full_pipeline_matches_hand_labels_on_a_drawing(tmp_path: Path) -> None:
    image = _save_grid(
        tmp_path / "images" / "grid.png",
        digits=((0, 1, "2"), (2, 3, "4")),
    )
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "\n".join(
            [
                "image,cell,row,column,order,digit",
                "grid.png,grid.png/cells/r00_c01.png,0,1,4,2",
                "grid.png,grid.png/cells/r02_c03.png,2,3,4,4",
                "grid.png,grid.png/cells/r01_c01.png,1,1,4,0",
                "grid.png,grid.png/cells/r00_c00.png,0,0,4,",
                "grid.png,grid.png/cells/r03_c03.png,3,3,4,uncertain",
                "grid.png,grid.png/cells/r03_c02.png,3,2,4,invalid",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report = run_directory(image.parent, tmp_path / "out", labels=labels)
    result = report["results"][0]
    digits = report["metrics"]["digit_recognition"]

    assert result["order"] == 4
    assert result["validation_status"] == "valid"
    assert result["solution_status"] != "not_attempted"
    assert digits["compared"] == 3
    assert digits["accuracy"] == 1.0
    assert report["metrics"]["reconstruction"]["cells_correct"] == 3
    assert report["metrics"]["leakage_count"] == 0
    assert load_labels(labels)[3].digit is None
    assert load_labels(labels)[4].marker == "uncertain"
    written = list(csv.DictReader((tmp_path / "out" / "labels_template.csv").open()))
    assert all(row["digit"] == "" for row in written)
    assert (
        main(
            [
                str(image.parent),
                "--output",
                str(tmp_path / "cli-out"),
                "--order",
                "4",
            ]
        )
        == 0
    )


def test_bad_label_tokens_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text("image,row,column,digit\na.png,0,0,true\n", encoding="utf-8")
    try:
        load_labels(path)
    except ValueError as exc:
        assert "invalid" in str(exc)
    else:
        raise AssertionError("boolean labels should fail")


def test_label_template_uses_stable_cell_names(tmp_path: Path) -> None:
    destination = tmp_path / "labels.csv"
    write_label_template(
        [_image("photo.jpg", [_cell(0, 2, "blank", "blank", 0)], order=9)],
        destination,
    )
    row = next(csv.DictReader(destination.open()))
    assert row["cell"] == "photo.jpg/cells/r00_c02.png"
    assert row["order"] == "9"
    assert row["digit"] == ""


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


def _ink(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = np.full((96, 96), 255, dtype=np.uint8)
    cv2.putText(canvas, "2", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 2.1, 0, 2, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)
    return path


def _cell(
    row: int,
    col: int,
    occupancy: str,
    status: str,
    digit: int | None,
) -> dict[str, object]:
    return {
        "row": row,
        "col": col,
        "occupancy": occupancy,
        "status": status,
        "digit": digit,
        "confidence": None,
        "blank_confidence": None,
    }


def _image(
    name: str,
    cells: list[dict[str, object]],
    *,
    success: bool = True,
    detected: bool = True,
    order: int = 4,
    validation: str = "needs_correction",
    solution: str = "not_attempted",
) -> dict[str, object]:
    return {
        "filename": name,
        "success": success,
        "detected": detected,
        "order": order if detected else None,
        "cells_extracted": order * order if success else 0,
        "validation_status": validation if success else None,
        "solution_status": solution if success else None,
        "failure_reason": None if success else "failed",
        "cells": cells,
    }
