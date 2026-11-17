"""Cell crops and the baseline blank-cell decision."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.recognition import (
    CellStatus,
    classify_cell,
    classify_cells,
    render_contact_sheet,
)
from app.vision import (
    CellSegmentationError,
    extract_cells,
    segment_cells,
    segment_cells_at_lines,
)
from app.vision.cell_segmenter import NORMALIZED_SIDE
from scripts.extract_cells import extract_directory, main
from tests.unit.vision_samples import DrawnGrid, draw_grid


def _cell(side: int = 64, value: int = 255) -> np.ndarray:
    return np.full((side, side), value, dtype=np.uint8)


def _digit(side: int = 64, color: int = 0, blur: int = 0) -> np.ndarray:
    image = _cell(side)
    cv2.putText(
        image,
        "5",
        (side // 5, int(side * 0.78)),
        cv2.FONT_HERSHEY_SIMPLEX,
        side / 70,
        color,
        max(1, side // 28),
        cv2.LINE_AA,
    )
    if blur:
        image = cv2.GaussianBlur(image, (blur, blur), 0)
    return image


@pytest.mark.parametrize("order", [4, 6, 9])
def test_every_cell_keeps_row_column_and_a_usable_crop(order: int) -> None:
    drawn = draw_grid(order)
    _detection, patches = extract_cells(drawn.image, order=order)

    assert [patch.row * order + patch.col for patch in patches] == list(
        range(order * order)
    )
    assert [(patch.row, patch.col) for patch in patches] == [
        (row, col) for row in range(order) for col in range(order)
    ]
    for patch in patches:
        assert patch.image.shape[0] == patch.size[1]
        assert patch.image.shape[1] == patch.size[0]
        assert min(patch.image.shape[:2]) >= 8
        assert patch.origin[0] >= 0
        assert patch.origin[1] >= 0


@pytest.mark.parametrize("side", [160, 320])
def test_equal_cells_scale_with_the_image(side: int) -> None:
    image = np.full((side, side, 3), 255, dtype=np.uint8)
    patches = segment_cells(image, 4)
    cell = side // 4
    inset = max(2, int(cell * 0.12))
    expected = cell - 2 * inset

    assert len(patches) == 16
    assert patches[0].image.shape[:2] == (expected, expected)
    assert patches[-1].row == 3
    assert patches[-1].col == 3


def test_line_centers_crop_the_grid_and_skip_the_margin() -> None:
    image = np.full((NORMALIZED_SIDE, NORMALIZED_SIDE), 255, dtype=np.uint8)
    centers = [30, 70, 110, 150, 190]
    for position in centers:
        cv2.line(image, (0, position), (NORMALIZED_SIDE - 1, position), 0, 3)
        cv2.line(image, (position, 0), (position, NORMALIZED_SIDE - 1), 0, 3)
    cv2.putText(image, "8", (78, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    patches = segment_cells_at_lines(image, centers, centers)
    marked = next(patch for patch in patches if patch.row == 1 and patch.col == 1)

    assert len(patches) == 16
    assert marked.origin[0] > 70
    assert marked.origin[1] > 70
    assert float(marked.image.min()) < 40
    assert float(patches[0].image.mean()) > 200


def test_empty_cell_is_blank_with_high_confidence() -> None:
    result = classify_cell(_cell())
    assert result.status is CellStatus.BLANK
    assert result.blank is True
    assert result.confidence >= 0.9


def test_printed_digit_is_occupied() -> None:
    result = classify_cell(_digit())
    assert result.status is CellStatus.OCCUPIED
    assert result.blank is False
    assert result.confidence >= 0.7


def test_thick_grid_lines_do_not_count_as_a_digit() -> None:
    image = _cell(72)
    cv2.rectangle(image, (0, 0), (71, 71), 0, thickness=8)
    result = classify_cell(image)
    assert result.status is CellStatus.BLANK
    assert result.blank is True


def test_faint_digit_is_not_called_blank() -> None:
    result = classify_cell(_digit(color=232))
    assert result.blank is not True
    assert result.status in {CellStatus.OCCUPIED, CellStatus.UNCERTAIN}
    if result.status is CellStatus.UNCERTAIN:
        assert result.blank is None
        assert result.confidence <= 0.6


def test_smooth_shadow_is_not_called_a_digit() -> None:
    ramp = np.linspace(140, 220, 64, dtype=np.float32)
    image = np.tile(ramp, (64, 1)).astype(np.uint8)
    result = classify_cell(image)
    assert result.status is CellStatus.BLANK
    assert result.blank is True


def test_noise_and_blur_keep_empty_and_digit_cells_apart() -> None:
    noisy = _cell()
    rng = np.random.default_rng(4)
    ys = rng.integers(0, 64, size=25)
    xs = rng.integers(0, 64, size=25)
    noisy[ys, xs] = 0
    blurred = classify_cell(_digit(blur=5))
    noise = classify_cell(noisy)

    assert blurred.status is CellStatus.OCCUPIED
    assert blurred.blank is False
    assert noise.status is CellStatus.BLANK
    assert noise.blank is True


@pytest.mark.parametrize("order", [4, 6, 9])
def test_drawn_grid_classifies_empty_and_printed_cells(order: int) -> None:
    drawn = draw_grid(order, cell=56, margin=28, line=2)
    _write(drawn, 0, 0, "3")
    _write(drawn, order - 1, 1, "7")
    _detection, patches = extract_cells(drawn.image)
    classes = classify_cells(patches)
    located = {(item.row, item.col): item for item in classes}

    assert len(classes) == order * order
    assert located[(0, 0)].status is CellStatus.OCCUPIED
    assert located[(order - 1, 1)].status is CellStatus.OCCUPIED
    assert located[(0, order - 1)].status is CellStatus.BLANK
    sheet = render_contact_sheet(patches, classes)
    assert sheet.shape[0] > 40
    assert sheet.shape[2] == 3


def test_line_segmentation_rejects_a_non_grid() -> None:
    image = np.full((80, 80), 255, dtype=np.uint8)
    with pytest.raises(CellSegmentationError, match="4, 6, or 9"):
        segment_cells_at_lines(image, [0, 40, 80], [0, 40, 80])


def test_extractor_writes_cells_sheet_and_json(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    drawn = draw_grid(4, cell=56, margin=28)
    _write(drawn, 1, 2, "4")
    assert cv2.imwrite(str(images / "puzzle.png"), drawn.image)
    output = tmp_path / "out"
    report = extract_directory(images, output)
    saved = output / "puzzle.png"
    cells = json_cells(saved / "cells.json")

    assert report["image_count"] == 1
    assert report["cells_extracted"] == 16
    assert report["failure_count"] == 0
    assert report["occupied_count"] >= 1
    assert report["uncertain_count"] >= 0
    assert "accuracy" not in report
    assert (saved / "perspective_grid.png").is_file()
    assert (saved / "contact_sheet.png").is_file()
    assert len(list((saved / "cells").glob("*.png"))) == 16
    marked = next(cell for cell in cells if cell["row"] == 1 and cell["col"] == 2)
    assert marked["blank"] is False
    assert marked["status"] == "occupied"
    assert 0 <= marked["confidence"] <= 1


def test_command_prints_the_cell_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    drawn = draw_grid(6, cell=40, margin=24)
    assert cv2.imwrite(str(images / "grid.png"), drawn.image)
    output = tmp_path / "out"
    code = main([str(images), "--output", str(output)])
    captured = capsys.readouterr()

    assert code == 0
    assert "report.json" in captured.out
    assert "cells=36" in captured.out


def _write(drawn: DrawnGrid, row: int, col: int, text: str) -> None:
    origin_x = drawn.margin + col * drawn.cell + drawn.cell // 5
    origin_y = drawn.margin + row * drawn.cell + int(drawn.cell * 0.75)
    cv2.putText(
        drawn.image,
        text,
        (origin_x, origin_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        drawn.cell / 70,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def json_cells(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["cells"]
