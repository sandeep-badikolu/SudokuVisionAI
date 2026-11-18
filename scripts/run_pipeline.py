"""Run the photograph pipeline and, when labels exist, score each stage.

Example:
    python scripts/run_pipeline.py sample_images/real \\
        --output evaluation_output/pipeline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.pipeline import run_directory, run_paths
from app.recognition.errors import RecognitionError
from app.vision.errors import VisionError


def main(argv: list[str] | None = None) -> int:
    """Process one image or a directory. A bad label or template file stops the run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, help="one image, or a directory of images")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="directory for artifacts",
    )
    parser.add_argument("--order", type=int, help="expected grid order: 4, 6, or 9")
    parser.add_argument(
        "--templates",
        type=Path,
        help="CSV of labelled templates; evaluation rows are not loaded",
    )
    parser.add_argument(
        "--no-builtin",
        action="store_true",
        help="use only the templates listed in --templates",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help="ground-truth CSV; empty digit cells stay unlabelled",
    )
    args = parser.parse_args(argv)
    try:
        if args.images.is_dir():
            report = run_directory(
                args.images,
                args.output,
                order=args.order,
                template_manifest=args.templates,
                use_builtin=not args.no_builtin,
                labels=args.labels,
            )
        elif args.images.is_file():
            report = run_paths(
                [args.images],
                args.output,
                order=args.order,
                template_manifest=args.templates,
                use_builtin=not args.no_builtin,
                labels=args.labels,
            )
        else:
            raise FileNotFoundError(f"image does not exist: {args.images}")
    except (
        FileNotFoundError,
        RecognitionError,
        ValueError,
        VisionError,
        OSError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(args.output / "report.json")
    print(
        f"images={report['image_count']} "
        f"succeeded={report['success_count']} "
        f"failed={report['failure_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
