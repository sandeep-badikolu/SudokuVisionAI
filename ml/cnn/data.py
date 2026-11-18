"""Select training and validation cells without opening evaluation files."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from app.dataset import ManifestRow, load_manifest
from app.recognition.digit_preprocess import CANVAS_SIDE, preprocess_digit

SOURCE_TYPES = ("synthetic", "unknown", "photographed")


@dataclass(frozen=True, slots=True)
class DigitExample:
    """One labelled cell that belongs to training or validation."""

    image: str
    cell_id: str
    path: Path
    label: int
    role: str
    source_type: str
    order: int


def partition_examples(
    rows: tuple[ManifestRow, ...] | list[ManifestRow],
    dataset_dir: Path,
) -> dict[str, object]:
    """Split labelled template cells into train and validation.

    Evaluation rows are counted and then ignored. Their cell files are not
    opened and their paths are not returned. Uncertain, invalid, and empty
    labels are excluded. Every cell of one photograph must share one role.
    """
    roles: dict[str, set[str | None]] = {}
    skipped_images: set[str] = set()
    holdout_images: set[str] = set()
    skipped_cells = 0
    holdout_cells = 0
    for row in rows:
        if row.split == "evaluation":
            if row.development_split in {"train", "validation", "holdout"}:
                raise ValueError(
                    f"{row.image} is an evaluation photograph with a development role"
                )
            skipped_images.add(row.image)
            skipped_cells += 1
            continue
        if row.development_split == "holdout":
            holdout_images.add(row.image)
            holdout_cells += 1
            continue
        roles.setdefault(row.image, set()).add(row.development_split)
    for image, found in roles.items():
        if len(found) != 1 or None in found:
            raise ValueError(f"{image} does not have one development role")
    train: list[DigitExample] = []
    validation: list[DigitExample] = []
    for row in rows:
        if (
            row.split == "evaluation"
            or row.development_split == "holdout"
            or row.digit is None
            or row.marker is not None
        ):
            continue
        role = row.development_split
        if role not in {"train", "validation"}:
            raise ValueError(f"{row.image} has development role {role}")
        source = row.source_type if row.source_type in SOURCE_TYPES else "unset"
        example = DigitExample(
            image=row.image,
            cell_id=row.cell_id,
            path=dataset_dir / row.cell,
            label=row.digit,
            role=role,
            source_type=source,
            order=row.order,
        )
        if role == "train":
            train.append(example)
        else:
            validation.append(example)
    _require_disjoint(train, validation)
    return {
        "train": tuple(train),
        "validation": tuple(validation),
        "skipped_evaluation_photographs": len(skipped_images),
        "skipped_evaluation_cells": skipped_cells,
        "skipped_holdout_photographs": len(holdout_images),
        "skipped_holdout_cells": holdout_cells,
    }


def class_counts(
    examples: tuple[DigitExample, ...] | list[DigitExample],
) -> dict[str, int]:
    """Count labels 0–9. Missing classes stay at zero."""
    counts = {str(digit): 0 for digit in range(10)}
    for example in examples:
        counts[str(example.label)] += 1
    return counts


def counts_by_source(
    examples: tuple[DigitExample, ...] | list[DigitExample],
) -> dict[str, dict[str, int]]:
    """Count labels separately for each source. Sources are not pooled."""
    names = (*SOURCE_TYPES, "unset")
    grouped: dict[str, dict[str, int]] = {
        source: {str(digit): 0 for digit in range(10)} for source in names
    }
    for example in examples:
        grouped.setdefault(
            example.source_type, {str(digit): 0 for digit in range(10)}
        )
        grouped[example.source_type][str(example.label)] += 1
    return grouped


class DigitTensors(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """In-memory 32×32 tensors and their class indexes."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, CANVAS_SIDE, CANVAS_SIDE):
            raise ValueError("digit tensors must have shape (N, 1, 32, 32)")
        if images.shape[0] != labels.shape[0]:
            raise ValueError("each digit tensor needs one label")
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


def load_split(
    examples: tuple[DigitExample, ...] | list[DigitExample],
    reader: Callable[[Path], np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
    """Read only the given cell paths and normalize them to 32×32 ink."""
    images: list[np.ndarray] = []
    labels: list[int] = []
    sources: list[str] = []
    for example in examples:
        if example.role not in {"train", "validation"}:
            raise ValueError(f"{example.image} is not a training or validation cell")
        array = reader(example.path)
        canvas = _as_canvas(array)
        images.append(canvas)
        labels.append(example.label)
        sources.append(example.source_type)
    if not images:
        raise ValueError("the split has no labelled cells")
    stacked = torch.from_numpy(np.stack(images)[:, None, :, :].astype(np.float32))
    targets = torch.tensor(labels, dtype=torch.long)
    return stacked, targets, tuple(sources)


def read_training_cell(path: Path) -> np.ndarray:
    """Open one training or validation crop and isolate its ink.

    This is the only image reader the baseline uses. It is not given
    evaluation paths.
    """
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cell image does not exist: {path}")
    return preprocess_digit(image)


def load_manifest_examples(manifest: Path) -> dict[str, object]:
    """Read the manifest and partition it. Evaluation files stay closed."""
    rows = load_manifest(manifest)
    return partition_examples(rows, manifest.parent)


def _as_canvas(array: np.ndarray) -> np.ndarray:
    if array.shape != (CANVAS_SIDE, CANVAS_SIDE):
        raise ValueError(f"normalized digit must be {CANVAS_SIDE}×{CANVAS_SIDE}")
    canvas = np.asarray(array, dtype=np.float32)
    if canvas.min() < 0 or canvas.max() > 1:
        raise ValueError("normalized digit values must be in [0, 1]")
    return canvas


def _require_disjoint(
    train: list[DigitExample], validation: list[DigitExample]
) -> None:
    train_images = {example.image for example in train}
    validation_images = {example.image for example in validation}
    shared = sorted(train_images & validation_images)
    if shared:
        raise ValueError(f"{shared[0]} is in both training and validation")
