#!/usr/bin/env bash
set -euo pipefail
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export TZ=UTC
SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH:-1788307200}
export SOURCE_DATE_EPOCH FORCE_SOURCE_DATE=1
BUILD=$(mktemp -d "${TMPDIR:-/tmp}/near-balanced-permutations.XXXXXX")
cp "$ROOT/main.tex" "$ROOT/sharp_four.tex" "$BUILD/"
(
  cd "$BUILD"
  if ! latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >latexmk-console.txt 2>&1; then
    echo "LaTeX build failed; inspect $BUILD/latexmk-console.txt" >&2
    exit 1
  fi
)
if grep -Eiq 'undefined references|undefined citations|There were undefined' "$BUILD/main.log"; then
  echo 'unresolved references or citations' >&2
  exit 1
fi
if grep -Eiq 'Overfull \\[hv]box|multiply defined labels|Package hyperref Warning: Difference' "$BUILD/main.log"; then
  echo 'layout overflow or duplicate labels detected' >&2
  exit 1
fi
pages=$(pdfinfo "$BUILD/main.pdf" | python3 -c 'import sys; print(next(line.split(":",1)[1].strip() for line in sys.stdin if line.startswith("Pages:")))')
if [ -z "$pages" ] || [ "$pages" -gt 30 ]; then
  echo "Combinatorica page limit failed: ${pages:-unknown}" >&2
  exit 1
fi
page_size=$(pdfinfo "$BUILD/main.pdf" | python3 -c 'import sys; print(next(line.split(":",1)[1].strip().split(" pts",1)[0] for line in sys.stdin if line.startswith("Page size:")))')
if [ "$page_size" != '612 x 792' ]; then
  echo "US-letter page-size check failed: $page_size" >&2
  exit 1
fi
if ! pdffonts "$BUILD/main.pdf" | python3 -c 'import sys; lines=sys.stdin.read().splitlines()[2:]; bad=[line for line in lines if line.split()[-5] != "yes"]; raise SystemExit(1 if bad else 0)'; then
  echo 'unembedded PDF font detected' >&2
  exit 1
fi
cp "$BUILD/main.pdf" "$ROOT/main.pdf"
rm -rf "$BUILD"
echo "BUILD_OK pages=$pages pdf=$ROOT/main.pdf"
