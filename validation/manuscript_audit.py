"""Check the manuscript's finite displays against the exact computations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .manuscript_agreement import (
    ManuscriptAgreementError,
    audit_paths,
    default_source_paths,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-source", type=Path, help="main.tex input")
    parser.add_argument("--sharp-four-source", type=Path, help="sharp_four.tex input")
    args = parser.parse_args(argv)
    if (args.main_source is None) != (args.sharp_four_source is None):
        parser.error("--main-source and --sharp-four-source must be supplied together")
    sources = (
        default_source_paths()
        if args.main_source is None
        else (args.main_source, args.sharp_four_source)
    )
    try:
        report = audit_paths(*sources)
    except (OSError, ManuscriptAgreementError) as error:
        print(f"MANUSCRIPT_AGREEMENT_FAIL: {error}", file=sys.stderr)
        return 1
    print(report.text(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
