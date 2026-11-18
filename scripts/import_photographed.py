"""Import genuine camera photographs after an explicit provenance check.

Example:
    python -m scripts.import_photographed \\
        --dataset evaluation_output/dataset \\
        --provenance photographed \\
        --pool holdout \\
        --confirm \\
        one.jpg two.jpg

``--pool holdout`` reserves the new photographed evaluation pool.
``--pool train`` is only for photographs that are not part of that pool.
Neither pool is added to the existing 20 evaluation photographs.
Digits are left empty for the annotation page.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.dataset_import import ImportBatchError, import_photographed_photographs


def main(argv: list[str] | None = None) -> int:
    """Import confirmed camera photographs, or leave the dataset unchanged."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--pool", required=True, choices=("train", "holdout"))
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("images", type=Path, nargs="*")
    args = parser.parse_args(argv)
    if not args.confirm or args.provenance != "photographed":
        print(
            "Refused. Pass --provenance photographed and --confirm. "
            "A folder name is not provenance. Nothing was imported.",
            file=sys.stderr,
        )
        return 2
    if not args.images:
        print(
            "At least one photograph is required. Nothing was imported.",
            file=sys.stderr,
        )
        return 2
    try:
        report = import_photographed_photographs(
            args.dataset,
            list(args.images),
            provenance=args.provenance,
            pool=args.pool,
            confirm=args.confirm,
        )
    except ImportBatchError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    added = report["added"]
    warnings = report["warnings"]
    assert isinstance(added, list)
    assert isinstance(warnings, list)
    print(f"added={len(added)} pool={args.pool} source=photographed")
    for item in added:
        print(
            f"{item['development_split']} {item['image']} "
            f"order={item['order']} cells={item['cells']}"
        )
    for item in warnings:
        print(f"warning {item['image']}: {item['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
