"""Report photographed collection coverage without inferring missing notes.

Example:
    python -m scripts.report_photographed_coverage \\
        --manifest evaluation_output/dataset/manifest.csv \\
        --conditions evaluation_output/dataset/photograph_conditions.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.dataset import load_manifest
from app.photographed_collection import collection_report, load_conditions


def main(argv: list[str] | None = None) -> int:
    """Print coverage gaps. This does not import images or train a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--conditions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        rows = load_manifest(args.manifest)
        notes = load_conditions(args.conditions)
        report = collection_report(rows, notes)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    text = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    missing = report["missing"]
    assert isinstance(missing, list)
    print(f"missing={len(missing)}")
    for item in missing:
        print(item)
    if args.output is not None:
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
