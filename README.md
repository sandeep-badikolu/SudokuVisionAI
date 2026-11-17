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
- Faint, broken, or decorative grid lines can fail the line-spacing check and be rejected.
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
