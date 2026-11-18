# SudokuVision AI

Recognize a Sudoku from a photograph or screenshot, review the digits, validate the puzzle, and solve it.

The processing engine detects the grid, reads printed digits with template matching, and solves a complete valid puzzle. The Streamlit application is the review screen on top of that engine.

## Requirements

- Python 3.12 or newer

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Application

The app accepts a PNG or JPEG up to 10 MB, shows the detected grid and extracted cells, and lets you correct the reconstruction before validation or solving. Uncertain cells stay unresolved until you enter a digit or a blank. Original OCR values are kept separate from those edits. Reset restores the OCR grid. Solve runs only after every cell is resolved and the puzzle is valid.

```bash
python -m streamlit run app/ui/streamlit_app.py
```

1. Upload a photograph and choose Process image. Leave the grid size on automatic detection unless you already know it.
2. Compare the original, the detected boundary, the straightened grid, and the contact sheet.
3. Select a cell, enter a digit or Blank, and apply the correction. Use Restore OCR value to drop that one edit.
4. Choose Validate puzzle to see duplicate rows, columns, and boxes.
5. Choose Solve puzzle. A unique solution marks clues in bold and solver-filled cells in blue. An unsolvable puzzle and a puzzle with several solutions do not show a chosen grid.

Photographs in this repository are not a labelled accuracy benchmark. Correct the grid before trusting a solution. A CNN is a later milestone.

## Tests

```bash
pytest
ruff check app tests
```

## Layout

| Path | Role |
| --- | --- |
| `app/main.py` | Prints the Streamlit launch command. |
| `app/core/` | Grid geometry, `sudoku_validator`, `candidate_generator`, and `sudoku_solver`. |
| `app/vision/` | Image loading, preprocessing, grid detection, perspective warp, and cell crops. |
| `app/recognition/` | Blank-cell classification, template digit recognition, and grid reconstruction. |
| `app/models/` | Future runtime model files. Training stays out of this package. |
| `app/ui/` | Streamlit screen and the review logic it calls. |
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

## Grid reconstruction

`reconstruct` places recognition results on a 4×4, 6×6, or 9×9 grid. Confirmed blanks stay `0`. Uncertain cells stay null and are listed in `unresolved_cells`; they are not stored as empty clues. Each cell keeps its source, recognition status, and the separate blank and digit confidences.

Confirmed digits are checked with the existing validator: dimensions, the digit range for that order, and duplicates in rows, columns, and boxes. Errors include the row, column, conflicting value, and the cells involved.

The solver runs only when that check passes and every cell is resolved. `original_clues` remains the recognized grid. `solution` is set only when the solver proves a single completion. An invalid grid, a grid that still has uncertain cells, an unsolvable puzzle, and a puzzle with several solutions do not receive a chosen solution.

```python
from app.recognition import reconstruct, reconstruct_image

built = reconstruct(order, recognition_results)
report = built.to_report()
```

`reconstruct_image` runs the existing detector, cell extractor, and template recognizer, then reconstructs. Its output on photographs is not a measured OCR accuracy.

## End-to-end pipeline

`scripts/run_pipeline.py` connects the existing stages for one image or a directory:

image loading → grid detection → perspective correction → cell extraction → blank classification → digit recognition → grid reconstruction → validation → solver.

A failed image is recorded and the directory continues. The solver still runs only when reconstruction is valid and every cell is resolved.

```bash
python scripts/run_pipeline.py sample_images/real \
  --output evaluation_output/pipeline

python scripts/run_pipeline.py sample_images/real/one.jpg \
  --output evaluation_output/pipeline \
  --order 9 \
  --templates path/to/templates.csv \
  --labels path/to/labels.csv
```

`--order`, `--templates`, and `--labels` are optional. `--no-builtin` keeps only the CSV templates. Evaluation rows in that CSV are not loaded. Generated files stay under the output directory, which should remain outside tracked source folders. `evaluation_output/` is gitignored.

Each successful image directory contains:

```text
puzzle.jpg/
  original.jpg
  detected_grid.png
  perspective_grid.png
  contact_sheet.png
  cells/r00_c00.png
  recognition.json
  reconstruction.json
```

The directory also contains `report.json`, `report.csv`, and `labels_template.csv`.

### Ground truth

The pipeline writes `labels_template.csv` with an empty `digit` column. Those cells are unlabelled. Fill the column by hand. Do not copy recognizer output into it.

```csv
image,cell,row,column,order,digit
puzzle.jpg,puzzle.jpg/cells/r00_c00.png,0,0,9,0
puzzle.jpg,puzzle.jpg/cells/r00_c01.png,0,1,9,5
puzzle.jpg,puzzle.jpg/cells/r00_c02.png,0,2,9,uncertain
puzzle.jpg,puzzle.jpg/cells/r00_c03.png,0,3,9,invalid
puzzle.jpg,puzzle.jpg/cells/r00_c04.png,0,4,9,
```

`0` is a blank cell. `1`–`9` are printed digits and must fit the order. `uncertain` and `invalid` mark labels that should not be scored as a digit. An empty digit stays unlabelled. `cell` is the stable crop name. A row whose resolved cell path is also a template image is leakage and is not scored.

Keep template photographs and held-out photographs in separate files. The same crop must not appear as a template and as an evaluation label.

### Metrics

`report.json` reports each stage on its own. There is no combined accuracy.

| Stage | What it counts |
| --- | --- |
| Grid detection | Images with a detected grid, over images attempted |
| Cell extraction | Detected images whose cell count is order² |
| Blank versus occupied | Precision and recall against labels 0 and 1–9. Occupancy-uncertain is not a true positive |
| Digit recognition | Correct 0–9 labels over labelled 0–9 cells, plus a confusion matrix and per-digit precision, recall, and accuracy |
| Uncertain recognition | Recognition status `uncertain` over extracted cells |
| Reconstruction | Labelled cells whose reconstructed value matches, and fully labelled puzzles that match every cell |
| Validation | Reconstructed images whose puzzle is valid |
| Solver completion | Unique solutions over puzzles the solver was allowed to attempt |

Digit, classification, and reconstruction scores are `null` when `--labels` is omitted or every digit cell is still empty. Detection and the other operational rates can still be reported. They are not real-world digit accuracy. A score on synthetic drawings is not a score on photographs.

`report.csv` has one row per image: status, order, cell count, validation, solver status, and the failure reason. Incorrect digit predictions and uncertain cell locations are in `report.json`.

### Limits

This milestone does not train a model. Template matching still fails on unseen typefaces, blur, and digits that touch a grid line. Photographs in `sample_images/real` do not have a committed ground-truth file, so this repository does not state an accuracy for them. Label a held-out set of real cells, keep those cells out of the template manifest, and only then judge the recognizer. A CNN is the following milestone.

## Real-world OCR benchmark

The benchmark dataset is built from the photographs already used by the pipeline. Preparation extracts every cell, writes a contact sheet, and creates a manifest whose `digit` column is empty. Fill that column by hand. The recognizer output next to each image is a diagnostic, not a label.

```bash
python scripts/prepare_dataset.py sample_images/real \
  --output evaluation_output/dataset \
  --splits path/to/splits.csv
```

`--splits` is optional at preparation time. When it is supplied, every extracted photograph must appear once:

```csv
image,split
one.jpg,template
two.jpg,evaluation
```

`split` is `template` or `evaluation`. All cells from one photograph take that split. A photograph cannot be in both. The command writes `manifest.csv`, `splits.csv`, and `dataset.json` beside the extracted cells and contact sheets. It does not modify the files in `sample_images/real`.

### Manifest

```csv
image,cell_id,cell,row,column,order,split,digit
one.jpg,r00_c00,one.jpg/cells/r00_c00.png,0,0,9,template,0
one.jpg,r00_c01,one.jpg/cells/r00_c01.png,0,1,9,template,5
two.jpg,r00_c00,two.jpg/cells/r00_c00.png,0,0,9,evaluation,2
two.jpg,r00_c01,two.jpg/cells/r00_c01.png,0,1,9,evaluation,uncertain
two.jpg,r00_c02,two.jpg/cells/r00_c02.png,0,2,9,evaluation,invalid
two.jpg,r00_c03,two.jpg/cells/r00_c03.png,0,3,9,evaluation,
```

`cell_id` is the stable `rNN_cNN` name. `0` is blank. `1`–`9` must fit the order. `uncertain` is unreadable. `invalid` is an unusable extraction. An empty digit is still unlabelled. Leave template blanks as `0`; only labelled digits `1`–`9` are loaded as templates.

### Scoring the held-out split

```bash
python scripts/benchmark_digits.py \
  --manifest evaluation_output/dataset/manifest.csv \
  --output evaluation_output/benchmark
```

The scorer is the existing `evaluate_labels` routine. It reads evaluation cells only. Template cells are not scored. `--no-builtin` uses the manually labelled template digits and no font templates. If any evaluation digit is empty, the report sets `verified` to false and `accuracy` to null. It does not fill those cells from a prediction.

When every evaluation cell has a manual label, `report.json` includes overall accuracy, per-digit precision, recall, and accuracy, a confusion matrix, the uncertain prediction rate, blank-versus-occupied precision and recall, and the same scores grouped by order and by source photograph. Incorrect rows include the cell path. `report.csv` lists each evaluation cell. `config.json` records the manifest, the split sizes, and whether the built-in templates were used.

Generated files belong under `evaluation_output/`, which is gitignored. Do not commit the photographs or these outputs. A result on drawings or font glyphs is not an accuracy for the photographs. No accuracy is claimed until the evaluation split has been labelled by hand. A CNN is still a later milestone.
