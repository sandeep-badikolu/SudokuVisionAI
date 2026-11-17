# SudokuVision AI

Recognize a Sudoku from a photograph or screenshot, validate it, and solve it.

The repository has the project layout, the puzzle domain model, and the solving engine. Computer vision, digit recognition, and the Streamlit UI are not implemented.

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
| `app/core/` | Grid geometry, clue validation, candidates, and the solver. |
| `app/vision/` | Future boundary detection, warp, and cell extraction. |
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
