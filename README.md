# SudokuVision AI

Recognize a Sudoku from a photograph or screenshot, validate it, and solve it.

The repository has the puzzle solver and the first computer-vision stage: load a JPEG or PNG, find a grid boundary, correct perspective, and split the grid into cells. Digit recognition and the Streamlit UI are not implemented.

## Requirements

- Python 3.12 or newer

Dependencies are listed in `requirements.txt` and are not installed by the scaffold. When you choose to install them:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Tests

```bash
pytest
ruff check app tests
```

## Layout

| Path | Role |
| --- | --- |
| `app/main.py` | Process entry point. It does not load images or solve. |
| `app/core/` | Grid geometry, `sudoku_validator`, `candidate_generator`, and `sudoku_solver`. |
| `app/vision/` | Image loading, preprocessing, grid detection, perspective warp, and cell crops. |
| `app/recognition/` | Future digit recognition and confidence scores. |
| `app/models/` | Future runtime model files. Training stays out of this package. |
| `app/ui/` | Future Streamlit screens. |
| `ml/datasets/` | Labeled data used for training and evaluation. |
| `ml/notebooks/` | Exploratory notebooks. |
| `ml/training/` | Future training scripts. |
| `ml/evaluation/` | Future metric runs on real labeled data. |
| `tests/unit/` | Tests that do not need the UI or image files. |
| `tests/integration/` | Future tests that cross module boundaries. |
| `tests/fixtures/` | Future shared boards and images. |
| `sample_images/` | Example photographs and screenshots. |
| `docs/` | Design notes that are not part of the README. |

Supported grids are 4×4 with 2×2 boxes, 6×6 with 2×3 boxes, and 9×9 with 3×3 boxes. `0` is an empty cell. `Puzzle.working_board()` returns a copy, and `solve` searches that copy so the given clues stay unchanged.

`solve` returns a status of invalid, unsolvable, unique, multiple, or limit reached. The solved grid is included only when the search proves there is exactly one completion. The default solution limit is 2.

## Computer vision

`app.vision` loads JPG, JPEG, and PNG images, preprocesses them, and tries to extract a top-down grid. Supported cell counts are 4×4, 6×6, and 9×9. A contour is accepted only when it is a convex quadrilateral with a passing geometry score and the warped image shows a regular grid of one of those sizes. Cell crops are inset so the grid stroke is not the cell border. The loader returns a copy; preprocessing does not write the caller's array or the source file.

This package does not read digits, does not call the solver, and does not import Streamlit or PyTorch.

The automated vision tests draw grids in memory. Those results show that the pipeline works on those drawings, including a mild perspective change and a simple shading gradient. They are not an evaluation on photographs or screenshots. No accuracy figure is claimed for real pictures.

Known limits of this milestone:

- Which side of the puzzle is "up" is not decided. Corner order follows the image axes.
- Two nearby edge peaks from one thick stroke are grouped into that stroke. A lighter spike on the warped border can be grouped with the boundary line when it is closer than one cell. The check that rejects uneven spacing is unchanged.
- One axis may be one peak high or one peak low only when the other axis is already a regular 4, 6, or 9 grid and that repair has one explanation. Disagreeing orders and irregular spacing are still rejected.
- Faint, broken, or decorative grid lines can still fail the line-spacing check and be rejected.
- Extra frames, page borders, or app chrome can outrank the puzzle or cause a rejection.
- Extreme perspective, blur, or low contrast is outside what the synthetic tests cover.

## Real photographs

Synthetic tests do not measure photographs. To inspect the OpenCV pipeline on your own pictures, put JPG, JPEG, or PNG files in a directory. Do not commit those photographs unless you mean to. Optional ground truth is a CSV named with `filename` and `order` columns. `order` is the grid size you know by looking at the puzzle: 4, 6, or 9. That comparison checks the detected grid size only. It is not digit accuracy.

```csv
filename,order
puzzle.jpg,9
small-grid.png,4
```

From the repository root, with the virtual environment active:

```bash
python scripts/evaluate_vision.py sample_images/real \
  --output evaluation_output \
  --ground-truth sample_images/real/ground_truth.csv
```

Omit `--ground-truth` when you have not labeled sizes. The command writes `evaluation_output/report.json`, `evaluation_output/report.csv`, and one folder per image with the original file, grayscale, threshold, edges, detected boundary, perspective-corrected grid, and cell crops. A failed detection leaves the boundary and cell images out and records the failure reason. The report does not contain an accuracy percentage. `evaluation_output/` is gitignored.

## Cell extraction

`segment_cells` divides a square warp into equal cells. `extract_cells` uses the detected line centers when both axes have one line per boundary, so a margin around the puzzle is not split as if it were a cell. Each crop is inset from those lines.

`app.recognition` classifies each crop as blank, occupied, or uncertain. A grid-line remnant or a speck is not treated as a digit. A faint coherent mark stays uncertain instead of being called empty. The confidence is confidence in that label. This step does not read which digit is printed.

```bash
python scripts/extract_cells.py sample_images/real \
  --output evaluation_output/cell_extraction
```

The output has the warped grid, one image per cell, a contact sheet, `cells.json` for that image, and `report.json` for the directory. Counts of blank, occupied, and uncertain cells are not digit accuracy.

## Digit recognition

The baseline reader is classical template matching. It is not a neural network. A CNN is a later milestone, once labelled cell photographs show where the templates fail.

### Workflow

1. `extract_cells` warps the puzzle and returns one crop per cell.
2. The blank-cell classifier marks each crop blank, occupied, or uncertain. Its confidence stays on `blank_confidence`.
3. A blank cell is digit `0`. OCR does not run.
4. An uncertain cell stays uncertain. No digit is stored.
5. An occupied cell is cleaned, centered on a 32×32 canvas without stretching, and compared with every template for the digits that grid size allows. 4×4 uses 1–4, 6×6 uses 1–6, and 9×9 uses 1–9.
6. The best similarity per digit is kept. The result includes the top three digits, their similarities, and the gap between the best and second-best.
7. The digit is accepted only when the best similarity and that gap are both strong enough. Otherwise the status is `uncertain` and `digit` is null. A weak match is not stored as a guess.

`confidence` on a recognized digit is the match confidence. It is null for blank and uncertain cells. Do not treat it as the blank-cell confidence.

### Templates

Built-in templates are rendered from several Hershey fonts and stroke weights the first time the recognizer runs. Real photographs need extra templates. Put those images in their own directory, not under `evaluation_output/`, and describe them with a CSV next to the files:

```csv
filename,digit,role
one_printed.png,1,template
one_held_out.png,1,evaluation
```

`filename` is relative to the CSV. `digit` is 1–9. `role` is `template` or `evaluation`. Evaluation rows are not loaded into the matcher, so those cells can be scored later without training on them.

```python
from app.recognition import (
    builtin_template_bank,
    load_template_manifest,
    merge_banks,
    recognize_cell,
)

bank = merge_banks(builtin_template_bank(), load_template_manifest(path))
result = recognize_cell(cell_image, order=9, bank=bank)
```

`builtin_template_bank` is exported from `app.recognition`.

### Commands

Recognize through the library, as above, or score a labelled set:

```bash
python scripts/evaluate_digits.py \
  --labels tests/fixtures/digit_labels_sample.csv \
  --templates path/to/templates.csv \
  --output evaluation_output/digit_eval
```

`--templates` is optional. Omit `--no-builtin` to keep the font templates as well. The label CSV lives beside the images it names:

```csv
filename,digit,order
cells/r00_c00.png,0,9
cells/r00_c01.png,5,9
cells/r00_c02.png,2,4
```

`digit` is 0 for an empty cell and 1–9 for a printed digit. `order` is 4, 6, or 9. A missing `order` is treated as 9. A copy of this header is `tests/fixtures/digit_labels_sample.csv`. The sample paths are documentation, not labelled photographs.

The report gives overall accuracy, per-digit accuracy, a confusion matrix, the uncertain rows, and the incorrect rows. Accuracy is `correct / scored` on that manifest only. Uncertain predictions are scored and are not correct. Rows whose image is also a template are listed as leakage and are not scored. No accuracy is produced for photographs that do not appear in the manifest.

### Limits

Template matching follows the fonts and the labelled examples it was given. A new typeface, a heavy blur, or a digit that touches the grid line can stay uncertain or match a similar glyph such as 3 and 8 or 1 and 7. Pencil marks that the occupancy step calls occupied are passed to OCR. This baseline does not read handwriting and does not fine-tune itself. Measure it on a labelled hold-out set before describing how it behaves on real puzzles. A CNN is the following milestone because it can learn the variation that a fixed template set misses.
