"""Digit recognition.

Blank-cell classification is the baseline step. A later model will read the
occupied cells. This package does not validate Sudoku rules and does not
solve puzzles.
"""

from app.recognition.blank_cells import (
    CellClassification,
    CellStatus,
    classify_cell,
    classify_cells,
    render_contact_sheet,
)

__all__ = [
    "CellClassification",
    "CellStatus",
    "classify_cell",
    "classify_cells",
    "render_contact_sheet",
]
