"""Reconstruction of a recognized grid and the decision to call the solver."""

import cv2
import pytest

from app.recognition import (
    RecognitionResult,
    RecognitionStatus,
    reconstruct,
    reconstruct_image,
)
from app.vision.cell_segmenter import SUPPORTED_ORDERS
from tests.unit.boards import COMPLETED
from tests.unit.vision_samples import DrawnGrid, draw_grid


def _cell(
    row: int,
    col: int,
    digit: int | None,
    status: str,
    *,
    confidence: float | None = 0.9,
) -> RecognitionResult:
    return RecognitionResult(
        row=row,
        col=col,
        digit=digit,
        status=status,
        confidence=confidence if status == RecognitionStatus.RECOGNIZED else None,
        blank_confidence=0.95 if status == RecognitionStatus.BLANK else 0.4,
        candidates=(),
        margin=None,
        diagnostics={},
    )


def _from_rows(
    rows: tuple[tuple[int | None, ...], ...],
    uncertain: set[tuple[int, int]] | None = None,
) -> list[RecognitionResult]:
    unresolved = set() if uncertain is None else uncertain
    results: list[RecognitionResult] = []
    for row_index, row in enumerate(rows):
        for col_index, value in enumerate(row):
            if (row_index, col_index) in unresolved:
                results.append(
                    _cell(row_index, col_index, None, RecognitionStatus.UNCERTAIN)
                )
                continue
            if value == 0:
                results.append(_cell(row_index, col_index, 0, RecognitionStatus.BLANK))
            else:
                results.append(
                    _cell(row_index, col_index, value, RecognitionStatus.RECOGNIZED)
                )
    return results


@pytest.mark.parametrize("order", [4, 6, 9])
def test_completed_grid_is_valid_and_solved_without_changing_clues(order: int) -> None:
    rows = COMPLETED[order]
    built = reconstruct(order, _from_rows(rows))

    assert built.validation_status == "valid"
    assert built.validation_errors == ()
    assert built.unresolved_cells == ()
    assert built.solution_status == "unique"
    assert built.solution == rows
    assert built.original_clues == rows
    assert built.grid == rows
    assert built.puzzle_order == order
    assert len(built.recognition_metadata) == order * order
    assert built.recognition_metadata[0].source == "recognizer"


@pytest.mark.parametrize("order", [4, 6, 9])
def test_blank_in_a_unique_puzzle_is_zero_and_is_filled_only_in_the_solution(
    order: int,
) -> None:
    rows = tuple(
        tuple(0 if (row, col) == (0, 0) else value for col, value in enumerate(line))
        for row, line in enumerate(COMPLETED[order])
    )
    built = reconstruct(order, _from_rows(rows))

    assert built.validation_status == "valid"
    assert built.original_clues[0][0] == 0
    assert built.grid[0][0] == 0
    assert built.solution is not None
    assert built.solution[0][0] == COMPLETED[order][0][0]
    assert built.original_clues[0][0] == 0
    assert built.solution_status == "unique"


def test_invalid_dimensions_are_reported_and_not_solved() -> None:
    built = reconstruct(4, _from_rows(COMPLETED[4])[:-1])

    assert built.validation_status == "invalid"
    assert built.solution is None
    assert built.solution_status == "not_attempted"
    assert built.validation_errors[0]["kind"] == "dimension"
    assert "16" in str(built.validation_errors[0]["detail"])


def test_unsupported_order_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported grid size"):
        reconstruct(5, [])


def test_out_of_range_digit_names_the_cell() -> None:
    rows = _from_rows(COMPLETED[4])
    rows[0] = _cell(0, 0, 5, RecognitionStatus.RECOGNIZED)
    built = reconstruct(4, rows)
    value_errors = [
        error for error in built.validation_errors if error["kind"] == "value"
    ]

    assert built.validation_status == "invalid"
    assert built.solution_status == "not_attempted"
    assert value_errors[0]["row"] == 0
    assert value_errors[0]["column"] == 0
    assert value_errors[0]["value"] == 5


def test_duplicate_row_reports_both_cells_and_the_value() -> None:
    rows = [list(line) for line in COMPLETED[4]]
    rows[0][1] = rows[0][0]
    built = reconstruct(4, _from_rows(tuple(tuple(line) for line in rows)))
    hit = next(error for error in built.validation_errors if error["kind"] == "row")

    assert built.validation_status == "invalid"
    assert hit["row"] == 0
    assert hit["value"] == rows[0][0]
    assert hit["cells"] == [[0, 0], [0, 1]]
    assert built.solution is None


def test_duplicate_column_reports_the_column_and_both_rows() -> None:
    rows = [list(line) for line in COMPLETED[4]]
    rows[2][0] = rows[0][0]
    built = reconstruct(4, _from_rows(tuple(tuple(line) for line in rows)))
    hit = next(error for error in built.validation_errors if error["kind"] == "column")

    assert hit["column"] == 0
    assert hit["value"] == rows[0][0]
    assert [0, 0] in hit["cells"]
    assert [2, 0] in hit["cells"]


def test_duplicate_subgrid_is_separate_from_row_and_column() -> None:
    rows = [list(line) for line in COMPLETED[4]]
    rows[1][1] = rows[0][0]
    built = reconstruct(4, _from_rows(tuple(tuple(line) for line in rows)))
    hit = next(error for error in built.validation_errors if error["kind"] == "box")

    assert hit["value"] == rows[0][0]
    assert [0, 0] in hit["cells"]
    assert [1, 1] in hit["cells"]


def test_uncertain_cell_stays_unresolved_and_blocks_the_solver() -> None:
    rows = _from_rows(COMPLETED[4], uncertain={(1, 2)})
    built = reconstruct(4, rows)
    report = built.to_report()

    assert built.validation_status == "needs_correction"
    assert built.solution_status == "not_attempted"
    assert built.solution is None
    assert built.grid[1][2] is None
    assert built.original_clues[1][2] is None
    assert built.grid[0][0] == COMPLETED[4][0][0]
    assert len(built.unresolved_cells) == 1
    assert built.unresolved_cells[0].row == 1
    assert built.unresolved_cells[0].col == 2
    assert built.unresolved_cells[0].status == RecognitionStatus.UNCERTAIN
    assert report["grid"][1][2] is None
    assert report["solution"] is None


def test_uncertain_cell_is_not_stored_as_a_blank() -> None:
    cells = _from_rows(
        tuple(tuple(0 for _ in range(4)) for _ in range(4)),
        uncertain={(0, 0)},
    )
    built = reconstruct(4, cells)

    assert built.grid[0][0] is None
    assert built.grid[0][1] == 0
    assert built.validation_status == "needs_correction"


def test_unsolvable_clues_are_valid_and_have_no_solution() -> None:
    rows = (
        (0, 1, 2, 0),
        (3, 0, 0, 0),
        (4, 0, 0, 0),
        (0, 0, 0, 0),
    )
    built = reconstruct(4, _from_rows(rows))

    assert built.validation_status == "valid"
    assert built.validation_errors == ()
    assert built.solution_status == "unsolvable"
    assert built.solution is None
    assert built.original_clues == rows


def test_multiple_solutions_do_not_choose_a_grid() -> None:
    rows = (
        (1, 2, 3, 4),
        (3, 4, 0, 0),
        (2, 1, 4, 3),
        (4, 3, 0, 0),
    )
    built = reconstruct(4, _from_rows(rows))

    assert built.validation_status == "valid"
    assert built.solution_status == "multiple"
    assert built.solution is None
    assert built.original_clues == rows


def test_duplicates_remain_invalid_when_another_cell_is_uncertain() -> None:
    rows = [list(line) for line in COMPLETED[4]]
    rows[0][1] = rows[0][0]
    frozen = tuple(tuple(line) for line in rows)
    cells = _from_rows(frozen, uncertain={(3, 3)})
    built = reconstruct(4, cells)

    assert built.validation_status == "invalid"
    assert built.grid[3][3] is None
    assert built.solution_status == "not_attempted"
    assert any(error["kind"] == "row" for error in built.validation_errors)


def test_pipeline_reconstructs_a_drawn_grid() -> None:
    drawn = draw_grid(4, cell=64, margin=32, line=2)
    _write(drawn, 0, 1, "2")
    _write(drawn, 2, 3, "4")
    built = reconstruct_image(drawn.image)

    assert built.puzzle_order == 4
    assert built.grid[0][1] == 2
    assert built.grid[2][3] == 4
    assert built.grid[1][1] == 0
    assert built.recognition_metadata[0].source == "recognizer"
    assert built.solution_status in {"multiple", "unique", "unsolvable"}
    assert SUPPORTED_ORDERS == (4, 6, 9)


def _write(drawn: DrawnGrid, row: int, col: int, text: str) -> None:
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


def test_reconstruction_json_uses_null_for_unresolved_cells() -> None:
    built = reconstruct(4, _from_rows(COMPLETED[4], uncertain={(2, 2)}))
    payload = built.to_report()

    assert payload["puzzle_order"] == 4
    assert payload["validation_status"] == "needs_correction"
    assert payload["grid"][2][2] is None
    assert payload["original_clues"][2][2] is None
    assert payload["unresolved_cells"][0]["confidence"] is None
    assert payload["recognition_metadata"][0]["source"] == "recognizer"
    assert isinstance(payload["grid"], list)
