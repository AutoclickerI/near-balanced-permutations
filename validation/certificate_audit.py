"""CLI for the solver-free exact prescribed-aspect certificate audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .lp_certificates import (
    CertificateError,
    audit_certificate_path,
    default_certificate_path,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificates", type=Path, help="exact_lp_certificates.json input")
    args = parser.parse_args(argv)
    path = args.certificates or default_certificate_path()
    try:
        report = audit_certificate_path(path)
        text = report.text()
    except (OSError, CertificateError) as error:
        print(f"LP_CERTIFICATE_AUDIT_FAIL: {error}", file=sys.stderr)
        return 1
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
