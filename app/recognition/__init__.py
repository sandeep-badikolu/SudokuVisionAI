"""Digit recognition.

Blank-cell classification decides whether a crop is empty. Occupied crops are
read with classical template matching. A neural recognizer is a later step.
This package does not validate Sudoku rules and does not solve puzzles.
"""

from app.recognition.blank_cells import (
    CellClassification,
    CellStatus,
    classify_cell,
    classify_cells,
    render_contact_sheet,
)
from app.recognition.digit_preprocess import CANVAS_SIDE, preprocess_digit
from app.recognition.digit_recognizer import (
    DigitCandidate,
    RecognitionResult,
    RecognitionStatus,
    allowed_digits,
    match_normalized,
    recognize_cell,
    recognize_cells,
)
from app.recognition.digit_templates import (
    DigitTemplate,
    TemplateBank,
    builtin_template_bank,
    load_template_manifest,
    merge_banks,
)
from app.recognition.errors import RecognitionError
from app.recognition.grid_reconstruction import (
    Reconstruction,
    reconstruct,
    reconstruct_image,
)

__all__ = [
    "CANVAS_SIDE",
    "CellClassification",
    "CellStatus",
    "DigitCandidate",
    "DigitTemplate",
    "RecognitionError",
    "RecognitionResult",
    "RecognitionStatus",
    "Reconstruction",
    "reconstruct",
    "reconstruct_image",
    "TemplateBank",
    "allowed_digits",
    "builtin_template_bank",
    "classify_cell",
    "classify_cells",
    "load_template_manifest",
    "match_normalized",
    "merge_banks",
    "preprocess_digit",
    "recognize_cell",
    "recognize_cells",
    "render_contact_sheet",
]
