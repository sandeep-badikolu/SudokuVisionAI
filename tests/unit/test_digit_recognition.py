"""Template OCR for printed digits, including blank and uncertain cells."""

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.recognition import (
    CellClassification,
    CellStatus,
    RecognitionError,
    RecognitionStatus,
    allowed_digits,
    builtin_template_bank,
    load_template_manifest,
    match_normalized,
    preprocess_digit,
    recognize_cell,
    recognize_cells,
)
from app.vision import extract_cells
from scripts.evaluate_digits import evaluate_labels, main
from tests.unit.vision_samples import draw_grid


def _glyph(
    digit: int, *, font: int = cv2.FONT_HERSHEY_SIMPLEX, thickness: int = 2
) -> np.ndarray:
    canvas = np.full((96, 96), 255, dtype=np.uint8)
    text = str(digit)
    (width, height), _baseline = cv2.getTextSize(text, font, 2.1, thickness)
    x = max(0, (96 - width) // 2)
    y = max(height + 2, (96 + height) // 2)
    cv2.putText(canvas, text, (x, y), font, 2.1, 0, thickness, cv2.LINE_AA)
    return canvas


def _occupied() -> CellClassification:
    return CellClassification(
        row=0,
        col=0,
        status=CellStatus.OCCUPIED,
        blank=False,
        confidence=0.9,
    )


def test_preprocessing_centers_a_digit_and_keeps_aspect() -> None:
    wide = np.full((64, 64), 255, dtype=np.uint8)
    cv2.line(wide, (6, 32), (58, 32), 0, 3)
    tall = np.full((64, 64), 255, dtype=np.uint8)
    cv2.line(tall, (32, 6), (32, 58), 0, 3)
    wide_norm = preprocess_digit(wide)
    tall_norm = preprocess_digit(tall)

    assert wide_norm.shape == (32, 32)
    assert tall_norm.dtype == np.float32
    wide_span = _span(wide_norm)
    tall_span = _span(tall_norm)
    assert wide_span[0] > wide_span[1]
    assert tall_span[1] > tall_span[0]
    assert _centroid(wide_norm) == pytest.approx((16, 16), abs=2)


def test_thin_stroke_survives_preprocessing() -> None:
    image = np.full((64, 64), 255, dtype=np.uint8)
    cv2.line(image, (32, 10), (32, 54), 0, 1)
    normalized = preprocess_digit(image)
    assert float(normalized.sum()) > 5


def test_builtin_bank_has_several_templates_for_every_digit() -> None:
    bank = builtin_template_bank()
    for digit in range(1, 10):
        assert len(bank.for_digit(digit)) >= 2


def test_template_manifest_loads_templates_and_skips_evaluation_rows(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "one.png"
    assert cv2.imwrite(str(image_path), _glyph(1))
    manifest = tmp_path / "templates.csv"
    manifest.write_text(
        "filename,digit,role\none.png,1,template\nheld.png,1,evaluation\n",
        encoding="utf-8",
    )
    bank = load_template_manifest(manifest)
    assert len(bank.templates) == 1
    assert bank.templates[0].digit == 1
    assert bank.templates[0].image.shape == (32, 32)


def test_template_matching_prefers_the_closer_of_two_templates() -> None:
    bank = builtin_template_bank()
    normalized = preprocess_digit(_glyph(5))
    ranked, margin = match_normalized(normalized, bank, allowed_digits(9))
    assert ranked[0].digit == 5
    assert margin is not None
    assert margin > 0
    similarities = [item.similarity for item in ranked]
    assert similarities == sorted(similarities, reverse=True)
    assert len(ranked[:3]) == 3


def test_blank_cell_is_zero_without_an_ocr_confidence() -> None:
    result = recognize_cell(np.full((48, 48, 3), 255, dtype=np.uint8), order=9)
    assert result.status == RecognitionStatus.BLANK
    assert result.digit == 0
    assert result.confidence is None
    assert result.blank_confidence is not None
    assert result.blank_confidence >= 0.8


def test_clear_digit_is_recognized_for_each_grid_size() -> None:
    for order, digit in ((4, 3), (6, 6), (9, 8)):
        result = recognize_cell(_glyph(digit), order=order, occupancy=_occupied())
        assert result.status == RecognitionStatus.RECOGNIZED
        assert result.digit == digit
        assert result.confidence is not None
        assert result.confidence >= 0.5
        assert len(result.candidates) == 3
        assert result.blank_confidence == pytest.approx(0.9)


def test_uncertain_occupancy_does_not_force_a_digit() -> None:
    occupancy = CellClassification(
        row=1,
        col=2,
        status=CellStatus.UNCERTAIN,
        blank=None,
        confidence=0.45,
    )
    result = recognize_cell(
        _glyph(5),
        order=9,
        occupancy=occupancy,
        row=1,
        col=2,
    )
    assert result.status == RecognitionStatus.UNCERTAIN
    assert result.digit is None
    assert result.confidence is None
    assert result.blank_confidence == pytest.approx(0.45)
    assert result.candidates == ()


def test_low_similarity_stays_uncertain() -> None:
    result = recognize_cell(
        np.full((40, 40), 255, dtype=np.uint8),
        order=9,
        occupancy=_occupied(),
    )
    assert result.status == RecognitionStatus.UNCERTAIN
    assert result.digit is None
    assert result.to_report()["digit"] is None


def test_blended_templates_are_ambiguous() -> None:
    bank = builtin_template_bank()
    three = bank.for_digit(3)[0].image
    eight = bank.for_digit(8)[0].image
    blend = np.clip((three + eight) / 2, 0.0, 1.0).astype(np.float32)
    ranked, margin = match_normalized(blend, bank, allowed_digits(9))
    assert margin is not None
    assert margin < 0.2
    assert len(ranked) >= 3
    top = {item.digit for item in ranked[:3]}
    assert 3 in top or 8 in top


def test_four_by_four_cannot_emit_a_digit_above_four() -> None:
    result = recognize_cell(_glyph(8), order=4, occupancy=_occupied())
    assert result.digit is None or result.digit <= 4
    assert all(item.digit <= 4 for item in result.candidates)
    assert 8 not in {item.digit for item in result.candidates}


@pytest.mark.parametrize("order", [4, 6, 9])
def test_allowed_digits_follow_the_grid_order(order: int) -> None:
    assert allowed_digits(order) == tuple(range(1, order + 1))


def test_invalid_image_and_order_are_rejected() -> None:
    with pytest.raises(RecognitionError, match="empty"):
        recognize_cell(np.array([]), order=9)
    with pytest.raises(RecognitionError, match="4, 6, or 9"):
        recognize_cell(np.full((16, 16), 255, dtype=np.uint8), order=5)
    with pytest.raises(RecognitionError, match="does not exist"):
        load_template_manifest(Path("/nonexistent/templates.csv"))


def test_drawn_grid_reads_printed_cells_and_leaves_empty_cells_blank() -> None:
    drawn = draw_grid(4, cell=64, margin=32, line=2)
    _write(drawn, 0, 1, "2")
    _write(drawn, 2, 3, "4")
    _detection, patches = extract_cells(drawn.image, order=4)
    results = recognize_cells(patches, order=4)
    located = {(item.row, item.col): item for item in results}

    assert located[(0, 1)].digit == 2
    assert located[(2, 3)].digit == 4
    assert located[(1, 1)].status == RecognitionStatus.BLANK
    assert located[(1, 1)].digit == 0
    assert all(item.digit is None or item.digit <= 4 for item in results)


def test_evaluation_reports_accuracy_only_from_labels(tmp_path: Path) -> None:
    root = tmp_path / "cells"
    root.mkdir()
    assert cv2.imwrite(str(root / "five.png"), _glyph(5))
    assert cv2.imwrite(str(root / "empty.png"), np.full((48, 48), 255, dtype=np.uint8))
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "filename,digit,order\ncells/five.png,5,9\ncells/empty.png,0,4\n",
        encoding="utf-8",
    )
    report = evaluate_labels(labels)
    assert report["scored_count"] == 2
    assert report["correct_count"] == 2
    assert report["accuracy"] == 1.0
    assert report["by_order"]["9"]["correct"] == 1
    assert report["by_order"]["4"]["correct"] == 1
    assert report["per_digit"]["5"]["accuracy"] == 1.0
    assert report["uncertain_count"] == 0
    assert "accuracy" in report


def test_evaluation_excludes_template_leakage_and_lists_mistakes(
    tmp_path: Path,
) -> None:
    assert cv2.imwrite(str(tmp_path / "five.png"), _glyph(5))
    assert cv2.imwrite(str(tmp_path / "noise.png"), _glyph(1))
    (tmp_path / "templates.csv").write_text(
        "filename,digit,role\nfive.png,5,template\n",
        encoding="utf-8",
    )
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "filename,digit,order\nfive.png,5,9\nnoise.png,9,9\n",
        encoding="utf-8",
    )
    report = evaluate_labels(labels, template_manifest=tmp_path / "templates.csv")
    assert report["leakage_count"] == 1
    assert report["scored_count"] == 1
    assert report["incorrect_count"] == 1
    assert report["incorrect"][0]["expected"] == 9
    assert report["accuracy"] == 0.0


def test_evaluation_command_writes_a_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    image = tmp_path / "two.png"
    assert cv2.imwrite(str(image), _glyph(2))
    labels = tmp_path / "labels.csv"
    labels.write_text("filename,digit,order\ntwo.png,2,6\n", encoding="utf-8")
    output = tmp_path / "out"
    code = main(["--labels", str(labels), "--output", str(output)])
    captured = capsys.readouterr()
    assert code == 0
    assert "accuracy=1.000" in captured.out
    saved = (output / "report.json").read_text(encoding="utf-8")
    assert "confusion" in saved


def test_sample_manifest_has_the_documented_columns() -> None:
    path = Path("tests/fixtures/digit_labels_sample.csv")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["filename", "digit", "order"]
        rows = list(reader)
    assert rows[0]["digit"] == "0"
    assert {row["order"] for row in rows} == {"9", "4"}


def _write(drawn, row: int, col: int, text: str) -> None:
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


def _span(image: np.ndarray) -> tuple[int, int]:
    ys, xs = np.where(image > 0.3)
    return int(xs.max() - xs.min()), int(ys.max() - ys.min())


def _centroid(image: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(image > 0.3)
    return float(xs.mean()), float(ys.mean())
