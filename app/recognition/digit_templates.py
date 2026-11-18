"""Template images for digits 1–9.

Built-in templates are drawn from several Hershey fonts and stroke weights.
Labelled photographs are loaded from a CSV manifest and are kept out of the
evaluation set when their role is ``evaluation``.
"""

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from app.recognition.digit_preprocess import preprocess_digit
from app.recognition.errors import RecognitionError

_FONTS = (
    ("simplex", cv2.FONT_HERSHEY_SIMPLEX, 1),
    ("simplex", cv2.FONT_HERSHEY_SIMPLEX, 2),
    ("plain", cv2.FONT_HERSHEY_PLAIN, 1),
    ("duplex", cv2.FONT_HERSHEY_DUPLEX, 1),
    ("duplex", cv2.FONT_HERSHEY_DUPLEX, 2),
    ("complex", cv2.FONT_HERSHEY_COMPLEX, 1),
    ("triplex", cv2.FONT_HERSHEY_TRIPLEX, 1),
    ("complex_small", cv2.FONT_HERSHEY_COMPLEX_SMALL, 1),
)


@dataclass(frozen=True, slots=True)
class DigitTemplate:
    """One normalized example of a digit."""

    digit: int
    source: str
    image: np.ndarray


@dataclass(frozen=True, slots=True)
class TemplateBank:
    """The examples a matcher is allowed to compare against."""

    templates: tuple[DigitTemplate, ...]

    def for_digit(self, digit: int) -> tuple[DigitTemplate, ...]:
        """Return every template whose label is ``digit``."""
        return tuple(item for item in self.templates if item.digit == digit)

    @property
    def sources(self) -> frozenset[str]:
        """Source names, including resolved paths of labelled files."""
        return frozenset(item.source for item in self.templates)


def builtin_template_bank() -> TemplateBank:
    """Return the font templates shipped with the baseline recognizer."""
    return _builtin_bank()


def load_template_manifest(path: Path) -> TemplateBank:
    """Load rows whose role is ``template``.

    The CSV columns are ``filename``, ``digit``, and optional ``role``.
    ``filename`` is relative to the manifest's directory. ``digit`` is 1–9.
    ``role`` is ``template`` or ``evaluation``. Evaluation rows are skipped
    so those images are not used as templates.
    """
    if not path.is_file():
        raise RecognitionError(f"template manifest does not exist: {path}")
    rows = _read_manifest(path)
    root = path.parent
    templates: list[DigitTemplate] = []
    for row_number, row in rows:
        role = (row.get("role") or "template").strip().lower()
        if role == "evaluation":
            continue
        if role != "template":
            raise RecognitionError(
                f"template manifest row {row_number} has role {role!r}"
            )
        digit = _parse_digit(row.get("digit") or "", row_number)
        filename = (row.get("filename") or "").strip()
        if not filename:
            raise RecognitionError(
                f"template manifest row {row_number} has an empty filename"
            )
        image_path = (root / filename).resolve()
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RecognitionError(f"could not read template image {image_path}")
        normalized = preprocess_digit(image)
        if float(normalized.sum()) <= 0:
            raise RecognitionError(f"template image {image_path} contains no digit")
        templates.append(
            DigitTemplate(digit=digit, source=str(image_path), image=normalized)
        )
    if not templates:
        raise RecognitionError(f"template manifest {path} has no template rows")
    return TemplateBank(templates=tuple(templates))


def merge_banks(*banks: TemplateBank) -> TemplateBank:
    """Concatenate template banks. Later banks do not replace earlier digits."""
    templates: list[DigitTemplate] = []
    for bank in banks:
        templates.extend(bank.templates)
    if not templates:
        raise RecognitionError("template bank is empty")
    return TemplateBank(templates=tuple(templates))


@lru_cache(maxsize=1)
def _builtin_bank() -> TemplateBank:
    templates: list[DigitTemplate] = []
    for digit in range(1, 10):
        for name, font, thickness in _FONTS:
            rendered = _render_hershey(digit, font, thickness)
            normalized = preprocess_digit(rendered)
            if float(normalized.sum()) <= 0:
                continue
            source = f"hershey:{name}:t{thickness}"
            templates.append(
                DigitTemplate(digit=digit, source=source, image=normalized)
            )
    if not templates:
        raise RecognitionError("built-in digit templates could not be rendered")
    return TemplateBank(templates=tuple(templates))


def _render_hershey(digit: int, font: int, thickness: int) -> np.ndarray:
    side = 96
    canvas = np.full((side, side), 255, dtype=np.uint8)
    text = str(digit)
    scale = 2.1
    (width, height), _baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, (side - width) // 2)
    y = max(height + 2, (side + height) // 2)
    if y >= side:
        y = side - 2
    cv2.putText(
        canvas,
        text,
        (x, y),
        font,
        scale,
        0,
        thickness,
        cv2.LINE_AA,
    )
    return canvas


def _read_manifest(path: Path) -> list[tuple[int, dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "filename" not in fieldnames or "digit" not in fieldnames:
            raise RecognitionError(
                "template manifest must have filename and digit columns"
            )
        return [(number, row) for number, row in enumerate(reader, start=2)]


def _parse_digit(raw: str, row_number: int) -> int:
    if raw.strip().lower() in {"true", "false"}:
        raise RecognitionError(f"template manifest row {row_number} digit is invalid")
    try:
        digit = int(raw.strip())
    except ValueError as exc:
        raise RecognitionError(
            f"template manifest row {row_number} digit is invalid"
        ) from exc
    if digit < 1 or digit > 9:
        raise RecognitionError(
            f"template manifest row {row_number} digit must be 1–9"
        )
    return digit
