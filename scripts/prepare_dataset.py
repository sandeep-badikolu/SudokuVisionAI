"""Extract cells from Sudoku photographs and write an unlabelled manifest.

Example:
    python scripts/prepare_dataset.py sample_images/real \\
        --output evaluation_output/dataset \\
        --splits path/to/splits.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.dataset import prepare_dataset
from app.recognition.errors import RecognitionError
from app.vision.errors import VisionError


def main(argv: list[str] | None = None) -> int:
    """Prepare a dataset. Digit labels are not filled from the recognizer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, help="directory of Sudoku photographs")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="directory for cells, contact sheets, and the manifest",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        help="CSV with image,split columns; split is template or evaluation",
    )
    args = parser.parse_args(argv)
    try:
        payload = prepare_dataset(args.images, args.output, splits=args.splits)
    except (
        FileNotFoundError,
        RecognitionError,
        ValueError,
        VisionError,
        OSError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    counts = payload["split_counts"]
    print(args.output / "manifest.csv")
    print(
        f"template={counts['template']} "
        f"evaluation={counts['evaluation']} "
        f"unassigned={counts['unassigned']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
