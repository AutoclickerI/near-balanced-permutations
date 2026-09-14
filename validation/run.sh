#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
python3 -B -m unittest discover -s validation -t . -v
python3 -B -m validation.manuscript_audit
python3 -B -m validation.certificate_audit
python3 -B -m validation.oriented_four
python3 -B -m validation.growing_patterns
python3 -B -m validation.bll_coefficient_audit
