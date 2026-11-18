"""Review edits, upload checks, and the decision to call the solver."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.ui.review import (
    CorrectionError,
    DisplayCell,
    ProcessingError,
    UploadError,
    confidence_caption,
    inspect_upload,
    process_upload,
    review_from_report,
)
from tests.unit.boards import COMPLETED
from tests.unit.vision_samples import draw_grid


def test_upload_rejects_the_wrong_type_size_and_corrupt_bytes(tmp_path: Path) -> None:
    png = _png(np.full((40, 40, 3), 255, dtype=np.uint8))
    loaded = inspect_upload("photo.png", png)
    assert loaded.original.shape[0] == 40

    with pytest.raises(UploadError, match="PNG, JPG, or JPEG"):
        inspect_upload("notes.gif", png)
    with pytest.raises(UploadError, match="empty"):
        inspect_upload("photo.jpg", b"")
    with pytest.raises(UploadError, match="10 MB"):
        inspect_upload("photo.jpg", b"x" * (10 * 1024 * 1024 + 1))
    with pytest.raises(UploadError, match="corrupt"):
        inspect_upload("photo.png", b"not a png")
    assert not (tmp_path / "photo.png").exists()


def test_confidence_captions_keep_digit_and_occupancy_scores_separate() -> None:
    recognized = _display(3, confidence=0.81, blank_confidence=0.66)
    missing = _display(3, confidence=None, blank_confidence=0.66)
    uncertain = _display(None, confidence=None, blank_confidence=0.42)
    blank = _display(0, confidence=None, blank_confidence=0.93)
    neither = _display(None, confidence=None, blank_confidence=None)

    assert confidence_caption(recognized) == (
        "Recognized digit 3. Digit confidence 0.81. Occupancy confidence 0.66."
    )
    assert "Digit confidence is not available." in confidence_caption(missing)
    assert "Recognized digit 3." in confidence_caption(missing)
    assert "Occupancy confidence 0.66." in confidence_caption(missing)
    uncertain_text = confidence_caption(uncertain)
    assert uncertain_text.startswith("Uncertain recognition.")
    assert "Digit confidence is not available." in uncertain_text
    assert "Occupancy confidence 0.42." in uncertain_text
    assert "Blank." not in uncertain_text
    assert confidence_caption(blank).startswith("Blank.")
    assert "Occupancy confidence 0.93." in confidence_caption(blank)
    assert "Occupancy confidence is not available." in confidence_caption(neither)


def test_missing_digit_confidence_does_not_reject_a_complete_puzzle() -> None:
    rows = [list(line) for line in COMPLETED[4]]
    metadata = []
    for row in range(4):
        for col in range(4):
            metadata.append(
                {
                    "row": row,
                    "col": col,
                    "status": "recognized" if rows[row][col] else "blank",
                    "confidence": None,
                    "blank_confidence": 0.5,
                }
            )
    review = _review(
        {"puzzle_order": 4, "grid": rows, "recognition_metadata": metadata}
    )
    cell = review.display_cells()[0]

    assert cell.value == COMPLETED[4][0][0]
    assert cell.confidence is None
    assert cell.blank_confidence == 0.5
    assert "Digit confidence is not available." in confidence_caption(cell)
    assert review.validate().status == "valid"
    assert review.solve_puzzle().status == "unique"


def test_uncertain_cell_with_no_digit_confidence_can_still_be_corrected() -> None:
    review = _review(_grid(uncertain=(0, 0), digit=((0, 1), 2)))
    cell = review.display_cells()[0]
    check = review.validate()

    assert cell.value is None
    assert cell.confidence is None
    assert cell.blank_confidence == 0.8
    assert review.ocr_grid[0][0] is None
    assert check.status == "needs_correction"
    assert check.errors == ("Row 1, column 1 is uncertain.",)
    assert "confidence" not in check.errors[0]
    review.set_cell(0, 0, 3)
    corrected = review.display_cells()[0]
    assert corrected.value == 3
    assert corrected.uncertain is False
    assert review.ocr_grid[0][0] is None


def _display(
    value: int | None,
    *,
    confidence: float | None,
    blank_confidence: float | None,
):
    return DisplayCell(
        row=0,
        col=0,
        value=value,
        ocr_value=value,
        source="ocr",
        uncertain=value is None,
        confidence=confidence,
        blank_confidence=blank_confidence,
    )


def test_display_keeps_uncertain_cells_distinct_from_blanks() -> None:
    review = _review(_grid(blank=(0, 1), uncertain=(0, 0), digit=((0, 2), 3)))
    cells = {(cell.row, cell.col): cell for cell in review.display_cells()}

    assert cells[(0, 0)].value is None
    assert cells[(0, 0)].uncertain is True
    assert cells[(0, 1)].value == 0
    assert cells[(0, 1)].uncertain is False
    assert cells[(0, 2)].value == 3
    assert cells[(0, 2)].confidence == 0.91
    assert cells[(0, 1)].blank_confidence == 0.8
    assert review.ocr_grid[0][0] is None


def test_corrections_do_not_replace_the_ocr_grid_and_reset_restores_it() -> None:
    review = _review(_grid(digit=((0, 0), 1), uncertain=(1, 1)))
    review.set_cell(0, 0, 2)
    review.set_cell(1, 1, 0)

    assert review.ocr_grid[0][0] == 1
    assert review.ocr_grid[1][1] is None
    assert review.display_cells()[0].value == 2
    assert review.display_cells()[0].source == "correction"
    assert {(cell.row, cell.col): cell.value for cell in review.display_cells()}[
        (1, 1)
    ] == 0
    review.clear_cell(0, 0)
    assert review.display_cells()[0].value == 1
    assert review.display_cells()[0].source == "ocr"
    review.reset()
    assert review.corrections == {}
    assert review.display_cells()[5].value is None


def test_correction_rejects_digits_outside_the_order() -> None:
    review = _review(_grid())
    with pytest.raises(CorrectionError, match="0 through 4"):
        review.set_cell(0, 0, 5)
    with pytest.raises(CorrectionError, match="integer"):
        review.set_cell(0, 0, True)  # type: ignore[arg-type]
    with pytest.raises(CorrectionError, match="outside"):
        review.set_cell(4, 0, 1)


def test_validate_reports_duplicates_and_does_not_treat_uncertain_as_blank() -> None:
    duplicate = [list(row) for row in COMPLETED[4]]
    duplicate[0][1] = duplicate[0][0]
    review = _review({"puzzle_order": 4, "grid": duplicate, "recognition_metadata": []})
    check = review.validate()
    assert check.status == "invalid"
    assert any(error.startswith("Row 1") for error in check.errors)
    assert review.solve_result is None

    uncertain = _review(_grid(uncertain=(0, 0)))
    check = uncertain.validate()
    assert check.status == "needs_correction"
    assert check.unresolved == ((0, 0),)
    with pytest.raises(CorrectionError, match="uncertain"):
        uncertain.resolved_board()


def test_solver_runs_only_for_a_complete_valid_grid() -> None:
    rows = [list(row) for row in COMPLETED[4]]
    rows[3][3] = 0
    review = _review({"puzzle_order": 4, "grid": rows, "recognition_metadata": []})
    solved = review.solve_puzzle()
    assert solved.status == "unique"
    assert solved.solution is not None
    assert solved.solution[3][3] == COMPLETED[4][3][3]
    overlay = review.solution_cells() or ()
    generated = [cell for cell in overlay if cell.role == "generated"]
    assert [(cell.row, cell.col) for cell in generated] == [(3, 3)]

    uncertain = _review(_grid(uncertain=(0, 0)))
    blocked = uncertain.solve_puzzle()
    assert blocked.status == "not_attempted"
    assert blocked.solution is None
    assert uncertain.ocr_grid[0][0] is None

    invalid = _review(
        {
            "puzzle_order": 4,
            "grid": [[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            "recognition_metadata": [],
        }
    )
    refused = invalid.solve_puzzle()
    assert refused.status == "not_attempted"
    assert refused.solution is None

    multiple = _review(
        {
            "puzzle_order": 4,
            "grid": [
                [1, 2, 3, 4],
                [3, 4, 0, 0],
                [2, 1, 4, 3],
                [4, 3, 0, 0],
            ],
            "recognition_metadata": [],
        }
    )
    many = multiple.solve_puzzle()
    assert many.status == "multiple"
    assert many.solution is None

    unsolvable = _review(
        {
            "puzzle_order": 4,
            "grid": [
                [0, 1, 2, 0],
                [3, 0, 0, 0],
                [4, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            "recognition_metadata": [],
        }
    )
    none = unsolvable.solve_puzzle()
    assert none.status == "unsolvable"
    assert none.solution is None


def test_process_upload_uses_the_pipeline_and_keeps_ocr_values(tmp_path: Path) -> None:
    drawn = draw_grid(4, cell=64, margin=32, line=2)
    origin_x = drawn.margin + drawn.cell + drawn.cell // 5
    origin_y = drawn.margin + int(drawn.cell * 0.75)
    cv2.putText(
        drawn.image,
        "2",
        (origin_x, origin_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        drawn.cell / 55,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    payload = _png(drawn.image)
    review = process_upload("puzzle.png", payload)

    assert review.order == 4
    assert review.ocr_grid[0][1] == 2
    assert review.ocr_grid[1][1] == 0
    assert set(review.images) == {
        "original",
        "detected_grid",
        "perspective_grid",
        "contact_sheet",
    }
    assert not (tmp_path / "puzzle.png").exists()
    with pytest.raises(ProcessingError):
        process_upload("blank.png", _png(np.full((64, 64, 3), 255, dtype=np.uint8)))


def _grid(
    *,
    blank: tuple[int, int] | None = None,
    uncertain: tuple[int, int] | None = None,
    digit: tuple[tuple[int, int], int] | None = None,
) -> dict[str, object]:
    grid = [[0 for _ in range(4)] for _ in range(4)]
    metadata = []
    for row in range(4):
        for col in range(4):
            metadata.append(
                {
                    "row": row,
                    "col": col,
                    "value": 0,
                    "source": "recognizer",
                    "status": "blank",
                    "confidence": None,
                    "blank_confidence": 0.8,
                }
            )
    if uncertain is not None:
        grid[uncertain[0]][uncertain[1]] = None
        metadata[uncertain[0] * 4 + uncertain[1]]["status"] = "uncertain"
        metadata[uncertain[0] * 4 + uncertain[1]]["value"] = None
    if digit is not None:
        (row, col), value = digit
        grid[row][col] = value
        item = metadata[row * 4 + col]
        item["status"] = "recognized"
        item["value"] = value
        item["confidence"] = 0.91
        item["blank_confidence"] = 0.2
    if blank is not None:
        grid[blank[0]][blank[1]] = 0
    return {"puzzle_order": 4, "grid": grid, "recognition_metadata": metadata}


def _review(report: dict[str, object]):
    return review_from_report(report, filename="puzzle.png")


def _png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()
