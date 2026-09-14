# Near-Balanced Permutations at Every Size

Companion code and exact certificates for the paper by Seonghyeon Bak,
Jihoon Kim, Geonwoo Kim, and Woojun Jeong.

The manuscript is in [paper/main.pdf](paper/main.pdf); its LaTeX sources
are in `paper/`. Version `v1.1.0` is the package cited in the manuscript.

## Run the checks

Python 3.10 or later and Bash are required. The code uses only the Python
standard library. Tested with Python 3.12.12.

From the repository root:

```sh
bash validation/run.sh
```

This runs the tests and all five audits: manuscript agreement, primal/dual
certificates, oriented four-pattern formulas, growing-pattern identities,
and the square-coefficient comparison. The tests include independent
enumerations and deliberately incorrect inputs that the checkers must reject.
These are finite checks, not substitutes for the analytic proofs.

The construction and profile computations are in `validation/construction.py`,
`validation/profiles.py`, and `validation/sharp_four.py`. The separate
`validation/oracle.py` and `validation/independent.py` provide independent
construction checks. The oriented and growing-pattern checks are in
`validation/oriented_four.py` and `validation/growing_patterns.py`.

To check only the LP certificates:

```sh
python3 -B -m validation.certificate_audit
```

`validation/exact_lp_certificates.json` contains the primal matrices, dual
supports, aspects, and objectives. The checker recomputes the residuals,
assignment marginals, and normalization using exact arithmetic over Q and
Q(sqrt(3)); the file contains no cached pass/fail results.

## Build the paper

The PDF is included. To rebuild it, install a LaTeX distribution with
`latexmk` and the packages used by `paper/main.tex`, plus Poppler's
`pdfinfo` and `pdffonts`, then run:

```sh
bash paper/build.sh
```

The script writes `paper/main.pdf` and checks the 30-page limit, US-letter
page size, embedded fonts, unresolved references, and layout overflows.

`SHA256SUMS` records the files in this version. Check a fresh checkout with:

```sh
sha256sum -c SHA256SUMS
```
