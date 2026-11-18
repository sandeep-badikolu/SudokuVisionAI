"""Recognize a printed Sudoku digit by template similarity.

Blank and uncertain occupancy come from the existing cell classifier. OCR
runs only for an occupied cell. A weak or ambiguous match stays uncertain
and does not invent a digit.
"""

from dataclasses import dataclass

import numpy as np

from app.recognition.blank_cells import CellClassification, CellStatus, classify_cell
from app.recognition.digit_preprocess import preprocess_digit
from app.recognition.digit_templates import (
    DigitTemplate,
    TemplateBank,
    builtin_template_bank,
)
from app.recognition.errors import RecognitionError
from app.vision.cell_segmenter import SUPPORTED_ORDERS, CellPatch

_MIN_SIMILARITY = 0.55
_MIN_MARGIN = 0.04


@dataclass(frozen=True, slots=True)
class DigitCandidate:
    """One digit and its best template similarity, from -1 to 1."""

    digit: int
    similarity: float


class RecognitionStatus:
    """How to read ``digit`` on a recognition result.

    ``BLANK`` uses digit 0. ``RECOGNIZED`` uses a digit from 1 to 9.
    ``UNCERTAIN`` leaves ``digit`` unset.
    """

    BLANK = "blank"
    RECOGNIZED = "recognized"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class RecognitionResult:
    """One cell after blank classification and, when needed, template matching.

    ``confidence`` is the digit-match confidence. It is ``None`` when OCR did
    not accept a digit. ``blank_confidence`` is the occupancy classifier's
    own score and is never reused as a digit confidence.
    """

    row: int
    col: int
    digit: int | None
    status: str
    confidence: float | None
    blank_confidence: float | None
    candidates: tuple[DigitCandidate, ...]
    margin: float | None
    diagnostics: dict[str, object]

    def to_report(self) -> dict[str, object]:
        """JSON-ready recognition fields. Uncertain cells have a null digit."""
        return {
            "row": self.row,
            "col": self.col,
            "digit": self.digit,
            "status": self.status,
            "confidence": _round(self.confidence),
            "blank_confidence": _round(self.blank_confidence),
            "margin": _round(self.margin),
            "candidates": [
                {"digit": item.digit, "similarity": round(item.similarity, 4)}
                for item in self.candidates
            ],
            "diagnostics": self.diagnostics,
        }


def allowed_digits(order: int) -> tuple[int, ...]:
    """Digits that may appear in a grid of this order. 0 is empty, not a glyph."""
    if isinstance(order, bool) or not isinstance(order, int):
        raise RecognitionError("grid order must be 4, 6, or 9")
    if order not in SUPPORTED_ORDERS:
        raise RecognitionError("grid order must be 4, 6, or 9")
    return tuple(range(1, order + 1))


def recognize_cell(
    image: np.ndarray,
    *,
    order: int,
    bank: TemplateBank | None = None,
    occupancy: CellClassification | None = None,
    row: int = 0,
    col: int = 0,
) -> RecognitionResult:
    """Recognize one cell, reusing an occupancy decision when one is supplied."""
    _require_image(image)
    allowed = allowed_digits(order)
    chosen_bank = builtin_template_bank() if bank is None else bank
    if not chosen_bank.templates:
        raise RecognitionError("template bank is empty")
    decision = occupancy if occupancy is not None else classify_cell(image, row, col)
    if decision.status is CellStatus.BLANK:
        return _blank(row, col, decision)
    if decision.status is CellStatus.UNCERTAIN:
        return _uncertain(
            row,
            col,
            decision.confidence,
            (),
            None,
            {"occupancy": decision.status.value, "ocr": "not run"},
        )
    normalized = preprocess_digit(image)
    return _from_match(row, col, decision, normalized, chosen_bank, allowed)


def recognize_cells(
    patches: tuple[CellPatch, ...] | list[CellPatch],
    *,
    order: int,
    bank: TemplateBank | None = None,
) -> tuple[RecognitionResult, ...]:
    """Classify and recognize every patch in the order given."""
    chosen = builtin_template_bank() if bank is None else bank
    return tuple(
        recognize_cell(
            patch.image,
            order=order,
            bank=chosen,
            row=patch.row,
            col=patch.col,
        )
        for patch in patches
    )


def match_normalized(
    normalized: np.ndarray,
    bank: TemplateBank,
    allowed: tuple[int, ...],
) -> tuple[tuple[DigitCandidate, ...], float | None]:
    """Score an already normalized digit. The best candidate is first.

    The second value is the gap between the best and second-best similarities.
    """
    if normalized.shape != (32, 32):
        raise RecognitionError("normalized digit must be 32×32")
    by_digit: dict[int, float] = {}
    for template in bank.templates:
        if template.digit not in allowed:
            continue
        score = _similarity(normalized, template.image)
        previous = by_digit.get(template.digit)
        if previous is None or score > previous:
            by_digit[template.digit] = score
    if not by_digit:
        raise RecognitionError("template bank has none of the allowed digits")
    ordered = sorted(by_digit.items(), key=lambda item: (-item[1], item[0]))
    ranked = tuple(
        DigitCandidate(digit=digit, similarity=score) for digit, score in ordered
    )
    margin = None
    if len(ranked) >= 2:
        margin = ranked[0].similarity - ranked[1].similarity
    return ranked, margin


def _from_match(
    row: int,
    col: int,
    decision: CellClassification,
    normalized: np.ndarray,
    bank: TemplateBank,
    allowed: tuple[int, ...],
) -> RecognitionResult:
    ranked, margin = match_normalized(normalized, bank, allowed)
    top = ranked[:3]
    best = ranked[0]
    winner = _winning_template(bank, best.digit, normalized)
    diagnostics: dict[str, object] = {
        "occupancy": decision.status.value,
        "ink_fraction": round(float(normalized.mean()), 4),
        "best_similarity": round(best.similarity, 4),
        "matched_template": None if winner is None else winner.source,
    }
    weak = best.similarity < _MIN_SIMILARITY or (
        margin is not None and margin < _MIN_MARGIN
    )
    if float(normalized.sum()) <= 0 or weak:
        return _uncertain(row, col, decision.confidence, top, margin, diagnostics)
    confidence = _match_confidence(best.similarity, margin)
    return RecognitionResult(
        row=row,
        col=col,
        digit=best.digit,
        status=RecognitionStatus.RECOGNIZED,
        confidence=confidence,
        blank_confidence=decision.confidence,
        candidates=top,
        margin=margin,
        diagnostics=diagnostics,
    )


def _require_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise RecognitionError("digit image is empty")
    if image.ndim == 2:
        return
    if image.ndim == 3 and image.shape[2] in {3, 4}:
        return
    raise RecognitionError("digit image must be grayscale or BGR")


def _blank(row: int, col: int, decision: CellClassification) -> RecognitionResult:
    return RecognitionResult(
        row=row,
        col=col,
        digit=0,
        status=RecognitionStatus.BLANK,
        confidence=None,
        blank_confidence=decision.confidence,
        candidates=(),
        margin=None,
        diagnostics={"occupancy": decision.status.value, "ocr": "not run"},
    )


def _uncertain(
    row: int,
    col: int,
    blank_confidence: float | None,
    candidates: tuple[DigitCandidate, ...],
    margin: float | None,
    diagnostics: dict[str, object],
) -> RecognitionResult:
    return RecognitionResult(
        row=row,
        col=col,
        digit=None,
        status=RecognitionStatus.UNCERTAIN,
        confidence=None,
        blank_confidence=blank_confidence,
        candidates=candidates,
        margin=margin,
        diagnostics=diagnostics,
    )


def _similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_values = left.astype(np.float64).ravel()
    right_values = right.astype(np.float64).ravel()
    left_values = left_values - left_values.mean()
    right_values = right_values - right_values.mean()
    denom = float(np.linalg.norm(left_values) * np.linalg.norm(right_values))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(left_values, right_values) / denom)


def _winning_template(
    bank: TemplateBank, digit: int, normalized: np.ndarray
) -> DigitTemplate | None:
    best: DigitTemplate | None = None
    best_score = -2.0
    for template in bank.for_digit(digit):
        score = _similarity(normalized, template.image)
        if score > best_score:
            best_score = score
            best = template
    return best


def _match_confidence(similarity: float, margin: float | None) -> float:
    span = 1.0 - _MIN_SIMILARITY
    strength = min(max((similarity - _MIN_SIMILARITY) / span, 0.0), 1.0)
    gap = 1.0 if margin is None else min(max(margin / 0.2, 0.0), 1.0)
    return float(min(0.99, 0.45 + 0.35 * strength + 0.19 * gap))


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)
