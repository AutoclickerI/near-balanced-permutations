"""Finite agreement checks against the active manuscript's displayed data.

The readers support the labelled displays in ``paper/main.tex`` and
``paper/sharp_four.tex``, including whitespace and simple layout variations.
They are not general TeX parsers. Missing, ambiguous, or unsupported displays
raise :class:`ManuscriptAgreementError`.

The checks connect displayed data to calculations already present in the
validation package:

* the displayed ``D`` matrix is recomputed from the corrected cubic
  Newton/Bernstein moment calculation, and ``C`` is recomputed from the
  projected 12-point moments;
* the displayed 24-pattern oriented ledger and ten-pattern centred dual are
  checked from the parsed ``C`` matrix and the independent oriented local
  formula;
* the displayed square ``k=4`` polynomial table is checked against the exact
  composition formula and a direct signed-grid enumerator for small sides;
* the displayed ``eq:k6-aspect-optimum`` aliases and ``eq:fixed-k-optima``
  six-row ``(k, theta)`` table are read in narrow supported forms and compared
  with the solver-free exact rational/algebraic primal/dual certificates.

The finite checks establish source-to-validation agreement for these displays
only.  They do not verify the manuscript's general proofs, asymptotic claims,
or arbitrary TeX.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from math import comb
from pathlib import Path
from typing import Mapping

from . import sharp_four
from .independent import enumerate_profiles
from .lp_certificates import (
    Q_STAR,
    RHO,
    CertificateAuditReport,
    CertificateError,
    CertificateSummary,
    audit_certificate_path,
    default_certificate_path,
)
from .profiles import all_patterns, square_profile_formula
from .quadratic import Qsqrt3


class ManuscriptAgreementError(ValueError):
    """Raised when a required manuscript display cannot be checked exactly."""


@dataclass(frozen=True)
class ParsedMatrix:
    """One narrow integer matrix display and its optional scalar factor."""

    scalar: Fraction
    entries: tuple[tuple[int, ...], ...]

    @property
    def values(self) -> tuple[tuple[Fraction, ...], ...]:
        return tuple(tuple(self.scalar * value for value in row) for row in self.entries)


@dataclass(frozen=True)
class ParsedLedger:
    """The displayed ``2304 L_tau`` oriented ledger."""

    scale: int
    levels: Mapping[str, int]

    @property
    def values(self) -> dict[str, Fraction]:
        return {pattern: Fraction(level, self.scale) for pattern, level in self.levels.items()}


@dataclass(frozen=True)
class ParsedContrast:
    """The exact rational weights in the displayed centred dual."""

    weights: Mapping[str, Fraction]
    denominator: int


@dataclass(frozen=True)
class ParsedSharpFour:
    """All corrected oriented displays read from ``sharp_four.tex``."""

    d_matrix: ParsedMatrix
    c_matrix: ParsedMatrix
    ledger: ParsedLedger
    contrast: ParsedContrast


@dataclass(frozen=True)
class SquarePolynomialRow:
    """One class in the displayed normalized square polynomial table."""

    coefficient_m2: int
    constant: int
    multiplicity: int
    patterns: tuple[str, ...]

    def evaluate(self, side: int) -> int:
        q = side * side
        return self.coefficient_m2 * q + self.constant


@dataclass(frozen=True)
class SquarePolynomialTable:
    """The exact displayed normalization and its six polynomial classes."""

    normalization_numerator: int
    rows: tuple[SquarePolynomialRow, ...]


@dataclass(frozen=True)
class FixedKOptimaRow:
    """One exact ``(k, theta)`` row of the displayed optimum table."""

    k: int
    theta: Qsqrt3
    q: Qsqrt3
    normalized: Qsqrt3


@dataclass(frozen=True)
class K6AspectAliases:
    """Numeric aliases read from the labelled algebraic definition equation."""

    rho: Qsqrt3
    q_star: Qsqrt3


@dataclass(frozen=True)
class FixedKOptimaTable:
    """The supported six-row ``q_k(theta)`` table and denominator header."""

    denominator_header: str
    rows: tuple[FixedKOptimaRow, ...]
    aliases: K6AspectAliases


@dataclass(frozen=True)
class ManuscriptAgreementReport:
    """Successful finite audit result and deterministic report data."""

    main_sha256: str
    sharp_four_sha256: str
    certificate_sha256: str
    table: SquarePolynomialTable
    fixed_k_optima: FixedKOptimaTable
    certificates: tuple[CertificateSummary, ...]
    sharp: ParsedSharpFour

    def text(self) -> str:
        """Render a path-independent, deterministic source-hashed report."""

        table_patterns = sum(len(row.patterns) for row in self.table.rows)
        fixed_rows = len(self.fixed_k_optima.rows)
        fixed_patterns = sum(item.patterns for item in self.certificates)
        lines = [
            "MANUSCRIPT_AGREEMENT_REPORT_V3",
            f"main.tex sha256={self.main_sha256}",
            f"sharp_four.tex sha256={self.sharp_four_sha256}",
            f"exact_lp_certificates.json sha256={self.certificate_sha256}",
            "coverage=finite displayed-data agreement only",
            f"D_matrix=OK scale=1/{self.sharp.d_matrix.scalar.denominator}",
            "C_matrix=OK projected_scale=1/64; AD_scale=1/32",
            f"ledger=OK patterns={len(self.sharp.ledger.levels)} scale={self.sharp.ledger.scale}",
            f"dual=OK terms={len(self.sharp.contrast.weights)} value=1/192 l1=1 marginal=1/8",
            f"square_table=OK rows={len(self.table.rows)} patterns={table_patterns} normalization={self.table.normalization_numerator}/[m^2(m^2-1)]",
            f"fixed_k_optima=OK rows={fixed_rows} certificate_patterns={fixed_patterns} (=46224+720)",
            f"k6_aliases=OK rho={self.fixed_k_optima.aliases.rho} q_star={self.fixed_k_optima.aliases.q_star}",
        ]
        lines.extend(
            [
                "scope=This is finite source-to-validation agreement, not universal proof verification.",
                "uncovered=General TeX, prose, proof steps, IFT proof verification, execution of the nonconstructive family, and claims outside the listed displays.",
                "MANUSCRIPT_AGREEMENT_OK",
            ]
        )
        return "\n".join(lines) + "\n"


# Narrow TeX readers.


_INT = r"[+-]?\d+"


def _unsupported(label: str, detail: str) -> ManuscriptAgreementError:
    return ManuscriptAgreementError(f"{label}: unsupported or missing TeX form ({detail})")


def _unique_match(source: str, pattern: str, label: str, *, flags: int = 0) -> re.Match[str]:
    matches = list(re.finditer(pattern, source, flags))
    if len(matches) != 1:
        if not matches:
            raise _unsupported(label, "required display was not found")
        raise _unsupported(label, f"expected one display, found {len(matches)}")
    return matches[0]


def _parse_integer_matrix(body: str, label: str) -> tuple[tuple[int, ...], ...]:
    # Only the canonical pmatrix row syntax is supported.  In particular,
    # bmatrix, \smallmatrix, macros, and symbolic entries fail loudly.
    rows = [part.strip() for part in re.split(r"\\\\", body) if part.strip()]
    if not rows:
        raise _unsupported(label, "empty pmatrix")
    parsed: list[tuple[int, ...]] = []
    for row_number, row in enumerate(rows, 1):
        cells = [cell.strip() for cell in row.split("&")]
        if any(not re.fullmatch(_INT, cell) for cell in cells):
            raise _unsupported(label, f"row {row_number} is not an integer '&'-row")
        parsed.append(tuple(int(cell) for cell in cells))
    width = len(parsed[0])
    if width == 0 or any(len(row) != width for row in parsed):
        raise _unsupported(label, "pmatrix rows have inconsistent widths")
    return tuple(parsed)


def _matrix_match(source: str, name: str, scalar_pattern: str, label: str) -> ParsedMatrix:
    pattern = (
        rf"(?<![A-Za-z]){re.escape(name)}\s*=\s*"
        rf"{scalar_pattern}"
        rf"\s*\\begin\s*\{{\s*pmatrix\s*\}}(?P<body>.*?)"
        rf"\\end\s*\{{\s*pmatrix\s*\}}"
    )
    match = _unique_match(source, pattern, label, flags=re.DOTALL)
    scalar_text = match.group("scalar") if "scalar" in match.groupdict() else None
    if scalar_text is None:
        scalar = Fraction(1)
    else:
        scalar = Fraction(1, int(scalar_text))
    entries = _parse_integer_matrix(match.group("body"), label)
    if len(entries) != 4 or any(len(row) != 4 for row in entries):
        raise _unsupported(label, "expected a 4-by-4 pmatrix")
    return ParsedMatrix(scalar, entries)


def _parse_sharp_matrix(source: str, name: str) -> ParsedMatrix:
    if name == "D":
        # The active source uses \frac1{192}; the brace-wrapped numerator is
        # accepted as the same narrow canonical form for clearer diagnostics.
        scalar_pattern = (
            r"\\frac\s*(?:\{\s*1\s*\}|1)\s*"
            r"\{\s*(?P<scalar>\d+)\s*\}"
        )
        return _matrix_match(source, name, scalar_pattern, "D matrix")
    if name == "C":
        return _matrix_match(source, name, r"", "C matrix")
    raise ValueError(f"unsupported matrix name {name!r}")


def _split_tex_rows(body: str) -> list[str]:
    """Split canonical array rows after nested gathered rows are flattened."""

    def flatten(match: re.Match[str]) -> str:
        return match.group(1).replace("\\\\", " ")

    body = re.sub(
        r"\\begin\s*\{\s*gathered\s*\}(.*?)\\end\s*\{\s*gathered\s*\}",
        flatten,
        body,
        flags=re.DOTALL,
    )
    return [part.strip() for part in re.split(r"\\\\", body) if part.strip()]


def _pattern_names(text: str, label: str) -> tuple[str, ...]:
    compact = re.sub(r"\s+", "", text)
    if "\\" in compact or "{" in compact or "}" in compact:
        raise _unsupported(label, "pattern cell contains an unsupported TeX command")
    if not re.fullmatch(r"[1-4]{4}(?:,[1-4]{4})*", compact):
        raise _unsupported(label, f"invalid pattern list {text!r}")
    names = tuple(compact.split(","))
    if any(len(name) != 4 or len(set(name)) != 4 for name in names):
        raise _unsupported(label, "pattern list contains a non-permutation")
    return names


def _parse_contrast(source: str) -> ParsedContrast:
    match = _unique_match(
        source,
        r"(?P<denominator>16)\s*\\mathcal\s*(?:\{\s*W\s*\}|W)\s*\(\s*z\s*\)\s*=\s*"
        r"(?:\{\s*\}\s*&)?\s*(?P<expr>.*?)(?:\\end\s*\{\s*split\s*\}\s*)?\\\]",
        "ten-pattern centred dual",
        flags=re.DOTALL,
    )
    expression = re.sub(r"\s+", "", match.group("expr"))
    # Alignment and row breaks are layout tokens, not mathematical terms.
    expression = expression.replace("&", "").replace(r"\\", "")
    if expression.endswith("."):
        expression = expression[:-1]
    position = 0
    weights: dict[str, Fraction] = {}
    first = True
    term_pattern = re.compile(r"(?P<coefficient>\d*)z_\{(?P<pattern>[1-4]{4})\}")
    while position < len(expression):
        sign = 1
        if expression[position] == "+":
            position += 1
        elif expression[position] == "-":
            sign = -1
            position += 1
        elif not first:
            raise _unsupported("ten-pattern centred dual", "term is missing a sign")
        term = term_pattern.match(expression, position)
        if term is None:
            raise _unsupported(
                "ten-pattern centred dual",
                "only signed integer multiples of z_{1234} are supported",
            )
        coefficient = int(term.group("coefficient") or "1")
        if coefficient == 0:
            raise _unsupported("ten-pattern centred dual", "zero coefficient")
        pattern = term.group("pattern")
        if pattern in weights:
            raise _unsupported("ten-pattern centred dual", f"duplicate pattern {pattern}")
        weights[pattern] = Fraction(sign * coefficient, int(match.group("denominator")))
        position = term.end()
        first = False
    if len(weights) != 10:
        raise _unsupported(
            "ten-pattern centred dual",
            f"expected 10 terms, found {len(weights)}",
        )
    return ParsedContrast(weights, int(match.group("denominator")))


def _parse_ledger(source: str) -> ParsedLedger:
    candidates: list[tuple[re.Match[str], re.Match[str]]] = []
    pattern = (
        r"\\begin\s*\{\s*array\s*\}\s*\{\s*c\s*\|\s*l\s*\}"
        r"(?P<body>.*?)\\end\s*\{\s*array\s*\}"
    )
    for match in re.finditer(pattern, source, flags=re.DOTALL):
        scale_match = re.search(r"(?P<scale>\d+)L_\\tau", match.group("body"))
        if scale_match is not None:
            candidates.append((match, scale_match))
    if len(candidates) != 1:
        if not candidates:
            raise _unsupported(
                "24-pattern oriented ledger",
                "the 2304L_tau array was not found",
            )
        raise _unsupported(
            "24-pattern oriented ledger",
            f"expected one 2304L_tau array, found {len(candidates)}",
        )
    array_match, scale_match = candidates[0]
    body = array_match.group("body")
    if r"\hline" not in body:
        raise _unsupported("24-pattern oriented ledger", "ledger has no canonical hline")
    rows_body = body.split(r"\hline", 1)[1]
    rows = _split_tex_rows(rows_body)
    levels: dict[str, int] = {}
    for row_number, row in enumerate(rows, 1):
        cells = [cell.strip() for cell in row.split("&")]
        if len(cells) != 2:
            raise _unsupported(
                "24-pattern oriented ledger", f"row {row_number} is not a two-cell row"
            )
        level_text = re.sub(r"\s+", "", cells[0])
        if not re.fullmatch(_INT, level_text):
            raise _unsupported(
                "24-pattern oriented ledger", f"row {row_number} has a noninteger level"
            )
        names = _pattern_names(cells[1], "24-pattern oriented ledger")
        level = int(level_text)
        for name in names:
            if name in levels:
                raise _unsupported("24-pattern oriented ledger", f"duplicate pattern {name}")
            levels[name] = level
    expected_names = {"".join(str(value + 1) for value in pattern) for pattern in all_patterns(4)}
    if set(levels) != expected_names:
        missing = sorted(expected_names - set(levels))
        extra = sorted(set(levels) - expected_names)
        raise _unsupported(
            "24-pattern oriented ledger",
            f"pattern set mismatch missing={missing} extra={extra}",
        )
    return ParsedLedger(int(scale_match.group("scale")), levels)


def parse_sharp_four_source(source: str) -> ParsedSharpFour:
    """Read exactly the four supported displays from active ``sharp_four.tex``."""

    if not isinstance(source, str):
        raise TypeError("sharp_four source must be text")
    d_matrix = _parse_sharp_matrix(source, "D")
    c_matrix = _parse_sharp_matrix(source, "C")
    ledger = _parse_ledger(source)
    contrast = _parse_contrast(source)
    return ParsedSharpFour(d_matrix, c_matrix, ledger, contrast)


# Square polynomial table reader.


def _parse_polynomial(text: str) -> tuple[int, int]:
    compact = re.sub(r"\s+", "", text)
    if compact.startswith(r"\displaystyle"):
        compact = compact[len(r"\displaystyle") :]
    # Supported forms are exactly affine polynomials in q=m^2, optionally
    # written as an integer multiple of a parenthesized affine polynomial.
    factored = re.fullmatch(
        r"(?P<outer>[+-]?\d*?)\((?P<inner_a>[+-]?\d*)m\^2(?P<inner_b>[+-]\d+)\)",
        compact,
    )
    if factored:
        outer_text = factored.group("outer")
        if outer_text in ("", "+"):
            outer = 1
        elif outer_text == "-":
            outer = -1
        else:
            outer = int(outer_text)
        inner_a_text = factored.group("inner_a")
        if inner_a_text in ("", "+"):
            inner_a = 1
        elif inner_a_text == "-":
            inner_a = -1
        else:
            inner_a = int(inner_a_text)
        return outer * inner_a, outer * int(factored.group("inner_b"))
    affine = re.fullmatch(
        r"(?P<a>[+-]?\d*?)m\^2(?P<b>[+-]\d+)",
        compact,
    )
    if affine:
        a_text = affine.group("a")
        if a_text in ("", "+"):
            coefficient = 1
        elif a_text == "-":
            coefficient = -1
        else:
            coefficient = int(a_text)
        return coefficient, int(affine.group("b"))
    constant = re.fullmatch(r"(?P<b>[+-]?\d+)", compact)
    if constant:
        return 0, int(constant.group("b"))
    raise _unsupported("square polynomial table", f"unsupported polynomial {text!r}")


def _parse_table_header(header: str) -> int:
    cells = [cell.strip() for cell in header.split("&")]
    if len(cells) != 3:
        raise _unsupported("square polynomial table", "header does not have three cells")
    first = re.sub(r"\s+", "", cells[0])
    if first.startswith(r"\displaystyle"):
        first = first[len(r"\displaystyle") :]
    # This intentionally does not implement a general TeX expression parser.
    # It recognizes only the displayed normalization and centering expression.
    fraction_numerator = r"(?:\{(?P<num1>\d+)\}|(?P<num2>\d+))"
    expected = re.fullmatch(
        rf"\\frac{fraction_numerator}"
        r"\{m\^2\(m\^2-1\)\}"
        r"\\left\(\\#_\\tau-\\frac(?:\{1\}|1)\{24\}"
        r"\\binom\{2m\^2\}\{4\}\\right\)",
        first,
    )
    if expected is None:
        raise _unsupported(
            "square polynomial table",
            "expected 144/[m^2(m^2-1)] times the displayed centered count",
        )
    numerator = expected.group("num1") or expected.group("num2")
    assert numerator is not None
    if cells[1].strip() != r"\#":
        raise _unsupported("square polynomial table", "unexpected multiplicity header")
    if cells[2].strip().rstrip("\\") != r"\tau":
        raise _unsupported("square polynomial table", "unexpected pattern header")
    return int(numerator)


def parse_square_polynomial_table(source: str) -> SquarePolynomialTable:
    """Read the one exact normalized ``k=4`` square table from ``main.tex``."""

    if not isinstance(source, str):
        raise TypeError("main source must be text")
    candidates: list[re.Match[str]] = []
    pattern = (
        r"\\begin\s*\{array\}\s*\{\s*c\s*\|\s*c\s*\|\s*l\s*\}"
        r"(?P<body>.*?)\\end\s*\{\s*array\s*\}"
    )
    for match in re.finditer(pattern, source, flags=re.DOTALL):
        body = match.group("body")
        if r"#_\tau" in body and r"m^2(m^2-1)" in re.sub(r"\s+", "", body):
            candidates.append(match)
    if len(candidates) != 1:
        if not candidates:
            raise _unsupported("square polynomial table", "required c|c|l table was not found")
        raise _unsupported(
            "square polynomial table", f"expected one table, found {len(candidates)}"
        )
    body = candidates[0].group("body")
    if r"\hline" not in body:
        raise _unsupported("square polynomial table", "table has no canonical hline")
    header, rows_body = body.split(r"\hline", 1)
    normalization = _parse_table_header(header)
    rows = _split_tex_rows(rows_body)
    parsed_rows: list[SquarePolynomialRow] = []
    for row_number, row in enumerate(rows, 1):
        cells = [cell.strip() for cell in row.split("&")]
        if len(cells) != 3:
            raise _unsupported(
                "square polynomial table", f"row {row_number} is not a three-cell row"
            )
        coefficient, constant = _parse_polynomial(cells[0])
        multiplicity_text = re.sub(r"\s+", "", cells[1])
        if not re.fullmatch(r"\d+", multiplicity_text):
            raise _unsupported(
                "square polynomial table", f"row {row_number} has a noninteger multiplicity"
            )
        patterns = _pattern_names(cells[2], "square polynomial table")
        multiplicity = int(multiplicity_text)
        if multiplicity != len(patterns):
            raise _unsupported(
                "square polynomial table",
                f"row {row_number} multiplicity {multiplicity} != {len(patterns)} listed patterns",
            )
        parsed_rows.append(SquarePolynomialRow(coefficient, constant, multiplicity, patterns))
    if len(parsed_rows) != 6:
        raise _unsupported(
            "square polynomial table", f"expected six rows, found {len(parsed_rows)}"
        )
    all_names = tuple(name for row in parsed_rows for name in row.patterns)
    expected_names = tuple(
        "".join(str(value + 1) for value in pattern) for pattern in all_patterns(4)
    )
    if len(all_names) != 24 or set(all_names) != set(expected_names):
        raise _unsupported(
            "square polynomial table", "the six rows do not partition all 24 patterns"
        )
    if len(set(all_names)) != 24:
        raise _unsupported("square polynomial table", "pattern occurs in more than one row")
    return SquarePolynomialTable(normalization, tuple(parsed_rows))


# Fixed-k optimum table reader.


def _compact_tex(text: str) -> str:
    compact = re.sub(r"\s+", "", text)
    if compact.startswith(r"\displaystyle"):
        compact = compact[len(r"\displaystyle") :]
    return compact


def _balanced_group(text: str, position: int, label: str) -> tuple[str, int]:
    if position >= len(text) or text[position] != "{":
        raise _unsupported(label, "expected a braced group")
    depth = 0
    for index in range(position, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[position + 1 : index], index + 1
    raise _unsupported(label, "unclosed braced group")


def _parse_qsqrt3_tex(text: str, label: str) -> Qsqrt3:
    """Parse the narrow linear ``Q(sqrt(3))`` TeX form used by the aliases."""

    compact = _compact_tex(text).rstrip(".,;:")

    def parse_expression(expression: str) -> Qsqrt3:
        position = 0
        total = Qsqrt3.zero()
        first = True
        while position < len(expression):
            sign = 1
            if expression[position] == "+":
                position += 1
            elif expression[position] == "-":
                sign = -1
                position += 1
            elif not first:
                raise _unsupported(label, "alias expression term is missing a sign")
            if position >= len(expression):
                raise _unsupported(label, "alias expression ends after a sign")

            if expression.startswith(r"\frac", position):
                position += len(r"\frac")
                numerator, position = _balanced_group(expression, position, label)
                denominator, position = _balanced_group(expression, position, label)
                if not re.fullmatch(r"\d+", denominator):
                    raise _unsupported(
                        label, "alias fraction denominator is not a positive integer"
                    )
                atom = parse_expression(numerator) / Fraction(int(denominator))
            else:
                number_match = re.match(r"\d+", expression[position:])
                if number_match is not None:
                    number = Fraction(int(number_match.group(0)))
                    position += len(number_match.group(0))
                else:
                    number = Fraction(1)
                if expression.startswith(r"\sqrt3", position):
                    atom = Qsqrt3(Fraction(0), number)
                    position += len(r"\sqrt3")
                elif expression.startswith(r"\sqrt{3}", position):
                    atom = Qsqrt3(Fraction(0), number)
                    position += len(r"\sqrt{3}")
                elif number_match is not None:
                    atom = Qsqrt3(number)
                else:
                    raise _unsupported(label, "alias expression contains unsupported syntax")
            total += sign * atom
            first = False
        if first:
            raise _unsupported(label, "empty alias expression")
        return total

    try:
        return parse_expression(compact)
    except ManuscriptAgreementError:
        raise
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise _unsupported(label, f"malformed exact alias expression {text!r}") from error


def _labelled_equation_body(source: str, label_name: str, label: str) -> str:
    labels = list(re.finditer(rf"\\label\s*\{{\s*{re.escape(label_name)}\s*\}}", source))
    if len(labels) != 1:
        if not labels:
            raise _unsupported(label, f"label {label_name} was not found")
        raise _unsupported(label, f"label {label_name} is ambiguous")
    equation_pattern = (
        r"\\begin\s*\{\s*equation\*?\s*\}(?P<body>.*?)"
        r"\\end\s*\{\s*equation\*?\s*\}"
    )
    equations = [
        match
        for match in re.finditer(equation_pattern, source, flags=re.DOTALL)
        if match.start() <= labels[0].start() <= match.end()
    ]
    if len(equations) != 1:
        raise _unsupported(label, "label is not inside one supported equation environment")
    return equations[0].group("body")


def _parse_k6_aspect_aliases(source: str) -> K6AspectAliases:
    body = _labelled_equation_body(source, "eq:k6-aspect-optimum", "k6 aspect aliases")
    assignment_pattern = re.compile(r"(?P<name>\\rho|q_\s*(?:\*|\{\s*\*\s*\}))\s*=")
    assignments = list(assignment_pattern.finditer(body))
    if len(assignments) != 2 or {match.group("name").replace(" ", "") for match in assignments} != {
        r"\rho",
        "q_*",
    }:
        raise _unsupported("k6 aspect aliases", "expected exactly one rho= and one q_*= definition")
    prefix = body[: assignments[0].start()]
    prefix = re.sub(
        r"\\label\s*\{\s*eq:k6-aspect-optimum\s*\}",
        "",
        prefix,
        count=1,
    )
    compact_prefix = re.sub(r"\s+", "", prefix)
    if compact_prefix and re.fullmatch(r"(?:,|;|\\quad|\\qquad)*", compact_prefix) is None:
        raise _unsupported("k6 aspect aliases", "unexpected syntax before the rho definition")
    parsed: dict[str, Qsqrt3] = {}
    for index, match in enumerate(assignments):
        end = assignments[index + 1].start() if index + 1 < len(assignments) else len(body)
        rhs = body[match.end() : end]
        rhs = re.sub(r"\\(?:quad|qquad)", "", rhs)
        rhs = rhs.strip(" ,;.")
        key = "rho" if match.group("name").replace(" ", "") == r"\rho" else "q_star"
        if key in parsed:
            raise _unsupported("k6 aspect aliases", f"duplicate {key} definition")
        parsed[key] = _parse_qsqrt3_tex(rhs, f"k6 aspect aliases {key}")
    aliases = K6AspectAliases(parsed["rho"], parsed["q_star"])
    if aliases.rho != RHO or aliases.q_star != Q_STAR:
        raise ManuscriptAgreementError(
            "k6 aspect aliases: parsed values "
            f"rho={aliases.rho}, q_star={aliases.q_star} differ from exact certificate values "
            f"rho={RHO}, q_star={Q_STAR}"
        )
    return aliases


def _parse_fixed_rational(text: str, label: str) -> Fraction:
    """Accept only an integer, slash rational, or braced ``\\dfrac``."""

    compact = _compact_tex(text)
    integer = re.fullmatch(r"[+-]?\d+", compact)
    if integer:
        return Fraction(int(compact))
    slash = re.fullmatch(r"(?P<num>[+-]?\d+)/(?P<den>\d+)", compact)
    if slash:
        return Fraction(int(slash.group("num")), int(slash.group("den")))
    frac = re.fullmatch(r"\\dfrac\{(?P<num>[+-]?\d+)\}\{(?P<den>\d+)\}", compact)
    if frac:
        return Fraction(int(frac.group("num")), int(frac.group("den")))
    raise _unsupported(
        label,
        "expected an integer, slash rational, or braced \\dfrac{numerator}{denominator}",
    )


def _parse_fixed_exact(
    text: str,
    label: str,
    aliases: K6AspectAliases,
    *,
    normalized: bool = False,
) -> Qsqrt3:
    compact = _compact_tex(text)
    if compact == r"\rho":
        if normalized:
            raise _unsupported(label, "normalization cell cannot be rho")
        return aliases.rho
    if compact == "q_*":
        if normalized:
            raise _unsupported(label, "normalization cell cannot be q_*")
        return aliases.q_star
    if compact == "q_*/34560":
        if not normalized:
            raise _unsupported(label, "theta or q cell cannot be q_*/34560")
        return aliases.q_star / 34560
    return Qsqrt3(_parse_fixed_rational(text, label))


def _parse_fixed_header(header: str) -> str:
    cells = [cell.strip() for cell in header.split("&")]
    if len(cells) != 4:
        raise _unsupported("fixed-k optimum table", "header does not have four cells")
    first = _compact_tex(cells[0])
    second = _compact_tex(cells[1])
    third = _compact_tex(cells[2].rstrip("\\"))
    fourth = _compact_tex(cells[3].rstrip("\\"))
    if first != "k":
        raise _unsupported("fixed-k optimum table", "first header cell is not k")
    if second != r"\theta":
        raise _unsupported("fixed-k optimum table", "second header cell is not theta")
    if third not in {r"q_k(\theta)", r"q_{k}(\theta)"}:
        raise _unsupported("fixed-k optimum table", "third header cell is not q_k(theta)")
    accepted_third = {
        r"q_k(\theta)/(2k!(k-2)!)",
        r"q_{k}(\theta)/(2k!(k-2)!)",
        r"\dfrac{q_k(\theta)}{2k!(k-2)!}",
        r"\dfrac{q_{k}(\theta)}{2k!(k-2)!}",
    }
    if fourth not in accepted_third:
        raise _unsupported(
            "fixed-k optimum table",
            "fourth header cell is not q_k(theta)/(2k!(k-2)!)",
        )
    return "2k!(k-2)!"


def parse_fixed_k_optima_table(source: str) -> FixedKOptimaTable:
    """Read the labelled six-row prescribed-aspect table in exact forms."""

    if not isinstance(source, str):
        raise TypeError("main source must be text")
    aliases = _parse_k6_aspect_aliases(source)
    labels = list(re.finditer(r"\\label\s*\{\s*eq:fixed-k-optima\s*\}", source))
    if len(labels) != 1:
        if not labels:
            raise _unsupported("fixed-k optimum table", "label eq:fixed-k-optima was not found")
        raise _unsupported("fixed-k optimum table", "label eq:fixed-k-optima is ambiguous")
    equation_pattern = (
        r"\\begin\s*\{\s*equation\*?\s*\}(?P<body>.*?)"
        r"\\end\s*\{\s*equation\*?\s*\}"
    )
    equations = [
        match
        for match in re.finditer(equation_pattern, source, flags=re.DOTALL)
        if match.start() <= labels[0].start() <= match.end()
    ]
    if len(equations) != 1:
        raise _unsupported(
            "fixed-k optimum table", "label is not inside one supported equation environment"
        )
    equation = equations[0]
    array_pattern = (
        r"\\begin\s*\{\s*array\s*\}\s*\{(?P<columns>[^}]*)\}"
        r"(?P<array_body>.*?)\\end\s*\{\s*array\s*\}"
    )
    candidates = []
    for match in re.finditer(array_pattern, equation.group("body"), flags=re.DOTALL):
        columns = re.sub(r"\s+", "", match.group("columns"))
        compact_body = re.sub(r"\s+", "", match.group("array_body"))
        if columns == "c|c|c|c" and (
            ("q_k" in compact_body or "q_{k}" in compact_body) and r"\theta" in compact_body
        ):
            candidates.append(match)
    if len(candidates) != 1:
        detail = (
            "supported four-column array was not found"
            if not candidates
            else f"expected one labelled fixed-k array, found {len(candidates)}"
        )
        raise _unsupported("fixed-k optimum table", detail)
    body = candidates[0].group("array_body")
    if r"\hline" not in body:
        raise _unsupported("fixed-k optimum table", "table has no canonical hline")
    header, rows_body = body.split(r"\hline", 1)
    denominator_header = _parse_fixed_header(header)
    rule_matches = list(re.finditer(r"\\hline", rows_body))
    if len(rule_matches) > 1:
        raise _unsupported("fixed-k optimum table", "more than one optional body hline")
    if rule_matches:
        rule = rule_matches[0]
        if not rows_body[: rule.start()].rstrip().endswith(r"\\"):
            raise _unsupported("fixed-k optimum table", "optional hline is not between rows")
        before = _split_tex_rows(rows_body[: rule.start()])
        after = _split_tex_rows(rows_body[rule.end() :])
        if not before or len(after) != 1:
            raise _unsupported("fixed-k optimum table", "optional hline is not before the last row")
        rows = before + after
    else:
        rows = _split_tex_rows(rows_body)
    parsed_rows: list[FixedKOptimaRow] = []
    for row_number, row in enumerate(rows, 1):
        cells = [cell.strip() for cell in row.split("&")]
        if len(cells) != 4:
            raise _unsupported("fixed-k optimum table", f"row {row_number} is not a four-cell row")
        k_text = _compact_tex(cells[0])
        if not re.fullmatch(r"[+-]?\d+", k_text) or int(k_text) not in range(4, 9):
            raise _unsupported(
                "fixed-k optimum table", f"row {row_number} has unsupported k {cells[0]!r}"
            )
        k = int(k_text)
        theta = _parse_fixed_exact(
            cells[1], f"fixed-k optimum table row {row_number} theta", aliases
        )
        if theta <= 0:
            raise _unsupported("fixed-k optimum table", f"row {row_number} theta is not positive")
        q = _parse_fixed_exact(
            cells[2], f"fixed-k optimum table row {row_number} q_k(theta)", aliases
        )
        normalized = _parse_fixed_exact(
            cells[3],
            f"fixed-k optimum table row {row_number} normalization",
            aliases,
            normalized=True,
        )
        if row_number == len(rows):
            final_cells = tuple(_compact_tex(cell) for cell in cells)
            if final_cells != ("6", r"\rho", "q_*", "q_*/34560"):
                raise _unsupported(
                    "fixed-k optimum table",
                    "final row must be exactly 6&\\rho&q_*&q_*/34560",
                )
        parsed_rows.append(FixedKOptimaRow(k, theta, q, normalized))
    expected_cases = (
        (4, Qsqrt3(Fraction(1))),
        (5, Qsqrt3(Fraction(1))),
        (6, Qsqrt3(Fraction(1))),
        (7, Qsqrt3(Fraction(1))),
        (8, Qsqrt3(Fraction(1))),
        (6, aliases.rho),
    )
    observed_cases = tuple((row.k, row.theta) for row in parsed_rows)
    if observed_cases != expected_cases:
        raise _unsupported(
            "fixed-k optimum table",
            f"rows must be exactly (k,theta)={expected_cases} in order, got {observed_cases}",
        )
    return FixedKOptimaTable(denominator_header, tuple(parsed_rows), aliases)


# Independent derivations and complete finite audit.


def _matrix_product(
    left: tuple[tuple[Fraction, ...], ...],
    right: tuple[tuple[Fraction, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    if len(left) != len(right[0]) or any(len(row) != len(right) for row in left):
        raise ValueError("incompatible matrices")
    return tuple(
        tuple(
            sum((left[i][k] * right[k][j] for k in range(len(right))), Fraction(0))
            for j in range(len(right[0]))
        )
        for i in range(len(left))
    )


def _node_matrix() -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(sharp_four.bernstein3(row, sharp_four.FOUR_U[group]) for group in range(4))
        for row in range(4)
    )


def _matrix_scale(
    matrix: tuple[tuple[int, ...], ...],
    denominator: int,
) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(tuple(Fraction(value, denominator) for value in row) for row in matrix)


def _direct_square_profile(side: int) -> Counter[tuple[int, ...]]:
    layers = enumerate_profiles(side, 4)
    result: Counter[tuple[int, ...]] = Counter()
    for layer in layers.values():
        for pattern, count in layer.items():
            result[tuple(value - 1 for value in pattern)] += count
    return result


def _check_displayed_root_intervals(source: str) -> None:
    """Bind the printed cubic sign-change intervals to exact polynomials."""

    match = _unique_match(
        source,
        r"Each \$f_i\$ changes sign(?P<body>.*?)These disjoint intervals",
        "cubic root intervals",
        flags=re.DOTALL,
    )
    interval_text = re.findall(r"\((\d+/\d+),(\d+/\d+)\)", match.group("body"))
    if len(interval_text) != 3:
        raise _unsupported("cubic root intervals", "expected three rational intervals")
    intervals = tuple((Fraction(a), Fraction(b)) for a, b in interval_text)
    previous = Fraction(0)
    for interval, signs in zip(intervals, sharp_four.ROOT_ENDPOINT_SIGNS):
        a, b = interval
        if not previous < a < b < 1:
            raise ManuscriptAgreementError("cubic root intervals: not disjoint inside (0,1)")
        previous = b
        for polynomial in sharp_four.CUBIC_POLYNOMIALS:

            def evaluate(x: Fraction) -> Fraction:
                value = Fraction(0)
                for coefficient in polynomial:
                    value = value * x + coefficient
                return value

            observed = tuple((evaluate(x) > 0) - (evaluate(x) < 0) for x in interval)
            if observed != signs:
                raise ManuscriptAgreementError(
                    f"cubic root intervals: exact endpoint signs {observed} != {signs}"
                )


def _check_sharp_displays(parsed: ParsedSharpFour) -> None:
    d_values = parsed.d_matrix.values
    if parsed.d_matrix.scalar != Fraction(1, 192):
        raise ManuscriptAgreementError(
            "D matrix: displayed scalar is not the supported 1/192 normalization"
        )
    derived_d = tuple(
        tuple(2 * (value - Fraction(3, 4)) for value in row)
        for row in sharp_four.root_bernstein_matrix()
    )
    if d_values != derived_d:
        raise ManuscriptAgreementError(
            f"D matrix: displayed values differ from corrected Newton/Bernstein derivation "
            f"(displayed={d_values!r}, derived={derived_d!r})"
        )

    if parsed.c_matrix.scalar != Fraction(1):
        raise ManuscriptAgreementError("C matrix: unexpected scalar before pmatrix")
    c_units = parsed.c_matrix.entries
    displayed_c_over_64 = _matrix_scale(c_units, 64)
    derived_c_over_64 = sharp_four.centered_moment_matrix(sharp_four.root_moment_matrix())
    if displayed_c_over_64 != derived_c_over_64:
        raise ManuscriptAgreementError(
            f"C matrix: displayed C/64 differs from projected twelve-point moments "
            f"(displayed={displayed_c_over_64!r}, derived={derived_c_over_64!r})"
        )

    a_matrix = _node_matrix()
    ad = _matrix_product(a_matrix, d_values)
    displayed_c_over_32 = _matrix_scale(c_units, 32)
    if ad != displayed_c_over_32:
        raise ManuscriptAgreementError(
            f"D/C matrix relation: displayed AD does not equal displayed C/32 "
            f"(AD={ad!r}, C/32={displayed_c_over_32!r})"
        )


def _check_ledger_and_contrast(parsed: ParsedSharpFour) -> None:
    if parsed.ledger.scale != 2304:
        raise ManuscriptAgreementError(
            f"24-pattern oriented ledger: displayed scale {parsed.ledger.scale}, expected 2304"
        )
    c_units = parsed.c_matrix.entries
    core = sharp_four.core_four_vector(sharp_four.ASPECT, sharp_four.ETA)
    expected: dict[str, Fraction] = {}
    for pattern in all_patterns(4):
        name = "".join(str(value + 1) for value in pattern)
        correction = Fraction(
            sum(c_units[row][pattern[row]] for row in range(4)),
            2304,
        )
        expected[name] = core[pattern] + correction
    observed = parsed.ledger.values
    if observed != expected:
        mismatches = [name for name in sorted(expected) if observed.get(name) != expected.get(name)]
        raise ManuscriptAgreementError(
            f"24-pattern oriented ledger: displayed values disagree with the parsed C matrix "
            f"and corrected local formula at {mismatches[:5]}"
        )

    expected_dual = dict(sharp_four.DUAL_WEIGHTS)
    observed_dual = dict(parsed.contrast.weights)
    if observed_dual != expected_dual or parsed.contrast.denominator != 16:
        raise ManuscriptAgreementError(
            f"ten-pattern centred dual: displayed weights differ from the checked dual "
            f"(displayed={observed_dual!r}, expected={expected_dual!r})"
        )
    row_column_balance = {(row, column): Fraction(0) for row in range(4) for column in range(4)}
    dual_h_value = Fraction(0)
    dual_ledger_value = Fraction(0)
    for name, weight in observed_dual.items():
        pattern = tuple(int(value) - 1 for value in name)
        dual_h_value += weight * core[pattern] * 96
        dual_ledger_value += weight * observed[name]
        for row, column in enumerate(pattern):
            row_column_balance[row, column] += weight
    if set(row_column_balance.values()) != {Fraction(1, 8)}:
        raise ManuscriptAgreementError(
            "ten-pattern centred dual: position-value marginals are not all 1/8"
        )
    if dual_h_value != Fraction(1, 2):
        raise ManuscriptAgreementError(
            f"ten-pattern centred dual: W(h) is {dual_h_value}, expected 1/2"
        )
    if dual_ledger_value != Fraction(1, 192):
        raise ManuscriptAgreementError(
            f"ten-pattern centred dual: displayed ledger gives {dual_ledger_value}, expected 1/192"
        )
    if sum(abs(value) for value in observed_dual.values()) != 1:
        raise ManuscriptAgreementError("ten-pattern centred dual: absolute weight sum is not 1")


def _check_square_table(table: SquarePolynomialTable) -> None:
    # The rows are not checked against a second hard-coded polynomial list.
    # Their reconstructed exact count must agree with the existing composition
    # formula at several sides, while sides 2,3,4 also use direct enumeration.
    if table.normalization_numerator <= 0:
        raise ManuscriptAgreementError("square polynomial table: normalization is not positive")
    names_to_row: dict[str, SquarePolynomialRow] = {}
    for row in table.rows:
        for name in row.patterns:
            names_to_row[name] = row

    for side in range(2, 7):
        formula = square_profile_formula(side, 4)
        direct = _direct_square_profile(side) if side <= 4 else None
        if direct is not None and direct != formula:
            raise ManuscriptAgreementError(
                f"square polynomial table: independent direct/formula profile mismatch at side {side}"
            )
        q = side * side
        factor = q * (q - 1)
        target = Fraction(comb(2 * q, 4), 24)
        for pattern in all_patterns(4):
            name = "".join(str(value + 1) for value in pattern)
            row = names_to_row[name]
            observed_difference = Fraction(
                row.evaluate(side) * factor,
                table.normalization_numerator,
            )
            expected_difference = Fraction(formula[pattern]) - target
            if observed_difference != expected_difference:
                raise ManuscriptAgreementError(
                    "square polynomial table: normalized polynomial mismatch "
                    f"for {name} at side {side} "
                    f"(displayed={observed_difference}, derived={expected_difference})"
                )


def _check_fixed_k_optima(
    table: FixedKOptimaTable,
    certificates: tuple[CertificateSummary, ...],
) -> None:
    if table.denominator_header != "2k!(k-2)!":
        raise ManuscriptAgreementError("fixed-k optimum table: unsupported normalization header")
    if table.aliases.rho != RHO or table.aliases.q_star != Q_STAR:
        raise ManuscriptAgreementError(
            "fixed-k optimum table: algebraic aliases disagree with exact certificate values"
        )
    by_case = {certificate.case_key: certificate for certificate in certificates}
    expected_cases = tuple(certificate.case_key for certificate in certificates)
    observed_cases = tuple((row.k, row.theta) for row in table.rows)
    if len(by_case) != len(certificates) or observed_cases != expected_cases:
        raise ManuscriptAgreementError(
            "fixed-k certificates/table: expected exactly the six approved (k,theta) cases "
            f"{expected_cases}, got certificates={tuple(by_case)} table={observed_cases}"
        )
    for row in table.rows:
        case = (row.k, row.theta)
        certificate = by_case.get(case)
        if certificate is None:
            raise ManuscriptAgreementError(
                f"fixed-k optimum table: unsupported or missing case ({row.k},{row.theta})"
            )
        if row.q != certificate.q:
            raise ManuscriptAgreementError(
                f"fixed-k optimum table: q_{row.k}({row.theta})={row.q} differs from exact certificate {certificate.q}"
            )
        if row.normalized != certificate.normalized_constant:
            raise ManuscriptAgreementError(
                "fixed-k optimum table: normalized "
                f"q_{row.k}({row.theta})={row.normalized} differs from exact certificate "
                f"{certificate.normalized_constant}"
            )


def audit_sources(
    main_source: str,
    sharp_source: str,
    *,
    certificate_audit: CertificateAuditReport | None = None,
) -> ManuscriptAgreementReport:
    """Run the finite manuscript agreement audit on two source strings."""

    if not isinstance(main_source, str) or not isinstance(sharp_source, str):
        raise TypeError("manuscript sources must be text")
    if certificate_audit is None:
        try:
            certificate_audit = audit_certificate_path(default_certificate_path())
        except CertificateError as error:
            raise ManuscriptAgreementError(f"fixed-k certificates: {error}") from error
    parsed_sharp = parse_sharp_four_source(sharp_source)
    table = parse_square_polynomial_table(main_source)
    fixed_k_optima = parse_fixed_k_optima_table(sharp_source)
    _check_sharp_displays(parsed_sharp)
    _check_displayed_root_intervals(sharp_source)
    _check_ledger_and_contrast(parsed_sharp)
    _check_square_table(table)
    _check_fixed_k_optima(fixed_k_optima, certificate_audit.certificates)
    return ManuscriptAgreementReport(
        main_sha256=sha256(main_source.encode("utf-8")).hexdigest(),
        sharp_four_sha256=sha256(sharp_source.encode("utf-8")).hexdigest(),
        certificate_sha256=certificate_audit.input_sha256,
        table=table,
        fixed_k_optima=fixed_k_optima,
        certificates=certificate_audit.certificates,
        sharp=parsed_sharp,
    )


def audit_paths(
    main_path: str | Path,
    sharp_four_path: str | Path,
) -> ManuscriptAgreementReport:
    """Read exact UTF-8 bytes from paths and audit them without modifying them."""

    main_bytes = Path(main_path).read_bytes()
    sharp_bytes = Path(sharp_four_path).read_bytes()
    try:
        main_source = main_bytes.decode("utf-8")
        sharp_source = sharp_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ManuscriptAgreementError(f"paper source is not UTF-8: {error}") from error
    return audit_sources(
        main_source,
        sharp_source,
    )


def default_source_paths() -> tuple[Path, Path]:
    """Return the two manuscript sources included in this package."""
    paper = Path(__file__).resolve().parents[1] / "paper"
    return paper / "main.tex", paper / "sharp_four.tex"
