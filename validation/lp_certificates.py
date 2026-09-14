"""Solver-free exact verification of the prescribed-aspect LP certificates.

The certificate table stores rationals and ``Q(sqrt(3))`` pairs. Verification
uses exact arithmetic and does not require an LP solver.

The finite scope is the five square cases and the algebraic ``(k,theta)=(6,
rho)`` additive LP case in the certificate table.  The latter is also checked
against the three global lower-dual forms and their exact intersections at
``rho`` and ``1/rho``.  This does not verify the manuscript's IFT argument,
the general realization proof, or execution of the nonconstructive family.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from itertools import permutations
from math import factorial
from pathlib import Path
from typing import Any, Mapping

from .profiles import _local_statistics
from .quadratic import Qsqrt3


class CertificateError(ValueError):
    """Raised when a certificate is malformed or fails an exact check."""


_RATIONAL = re.compile(r"[+-]?\d+(?:/\d+)?\Z")
_SUPPORTED_K = frozenset(range(4, 9))
RHO = Qsqrt3(Fraction(2), Fraction(-2, 3))
Q_STAR = Qsqrt3(Fraction(37, 20), Fraction(1, 12))
_APPROVED_CASES: dict[tuple[int, Qsqrt3], Qsqrt3] = {
    (4, Qsqrt3(Fraction(1))): Qsqrt3(Fraction(2, 3)),
    (5, Qsqrt3(Fraction(1))): Qsqrt3(Fraction(5, 4)),
    (6, Qsqrt3(Fraction(1))): Qsqrt3(Fraction(2)),
    (7, Qsqrt3(Fraction(1))): Qsqrt3(Fraction(62, 21)),
    (8, Qsqrt3(Fraction(1))): Qsqrt3(Fraction(109, 28)),
    (6, RHO): Q_STAR,
}
_APPROVED_SQUARE_ALPHA: dict[int, Fraction] = {
    4: Fraction(0),
    5: Fraction(0),
    6: Fraction(-1, 5),
    7: Fraction(13, 42),
    8: Fraction(3, 8),
}


@dataclass(frozen=True)
class CertificateSummary:
    """Exact verification summary for one ``(k, theta)`` case."""

    k: int
    theta: Qsqrt3
    q: Qsqrt3
    normalized_constant: Qsqrt3
    patterns: int
    dual_support: int
    dual_lambda: Qsqrt3
    square_alpha: Fraction | None

    @property
    def case_key(self) -> tuple[int, Qsqrt3]:
        return self.k, self.theta


@dataclass(frozen=True)
class GlobalAspectSummary:
    """Exact global lower-dual and intersection checks for ``k=6``."""

    k: int
    rho: Qsqrt3
    q_star: Qsqrt3
    form_names: tuple[str, ...]
    intersections: int


@dataclass(frozen=True)
class CertificateAuditReport:
    """Deterministic source-hashed report for the certificate table."""

    input_sha256: str
    formula_code_sha256: str
    certificates: tuple[CertificateSummary, ...]
    global_aspect: GlobalAspectSummary

    @property
    def total_patterns(self) -> int:
        return sum(item.patterns for item in self.certificates)

    def text(self) -> str:
        lines = [
            "LP_CERTIFICATE_AUDIT_V1",
            f"input_sha256={self.input_sha256}",
            f"formula_code_sha256={self.formula_code_sha256}",
            "scope=Exact finite prescribed-aspect additive LP certificates only.",
            "scope_excluded=IFT proof verification and execution of the nonconstructive general family.",
            f"certificates={len(self.certificates)}",
            f"patterns_checked={self.total_patterns} (=46224+720)",
            "square_alpha_consequence=nonnegative k=4,5,7,8 only; k=6 square alpha is retained but not used for global aspect optimality",
            "global_aspect=exact Q(sqrt(3)) k=6 primal at rho; lower-dual forms W_left,W_right,W_square",
        ]
        for item in self.certificates:
            lines.append(
                "case=({k},{theta}) q={q} normalized={normalized} patterns={patterns} "
                "dual_support={support} dual_lambda={lambda_} square_alpha={alpha}".format(
                    k=item.k,
                    theta=item.theta,
                    q=item.q,
                    normalized=item.normalized_constant,
                    patterns=item.patterns,
                    support=item.dual_support,
                    lambda_=item.dual_lambda,
                    alpha=(
                        item.square_alpha if item.square_alpha is not None else "not_applicable"
                    ),
                )
            )
        lines.append(
            "global=OK k={k} rho={rho} q_star={q} forms={forms} intersections={intersections}".format(
                k=self.global_aspect.k,
                rho=self.global_aspect.rho,
                q=self.global_aspect.q_star,
                forms=",".join(self.global_aspect.form_names),
                intersections=self.global_aspect.intersections,
            )
        )
        lines.extend(
            [
                "LP_CERTIFICATE_AUDIT_OK",
            ]
        )
        return "\n".join(lines) + "\n"


def _exact_fraction(value: Any, label: str) -> Fraction:
    """Parse only integer or slash-rational JSON values, never a float."""

    if isinstance(value, bool):
        raise CertificateError(f"{label}: boolean is not an exact rational")
    if isinstance(value, int):
        return Fraction(value)
    if not isinstance(value, str) or _RATIONAL.fullmatch(value) is None:
        raise CertificateError(
            f"{label}: expected an integer or slash rational string, got {value!r}"
        )
    try:
        return Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise CertificateError(f"{label}: invalid rational {value!r}") from error


def _exact_quadratic(value: Any, label: str) -> Qsqrt3:
    """Parse a rational or typed ``{"a": ..., "b": ...}`` pair.

    The pair is intentionally structural JSON data.  Algebraic expressions
    such as ``"2-2*sqrt(3)/3"`` are not accepted or evaluated at runtime.
    """

    if isinstance(value, Mapping):
        if set(value) != {"a", "b"}:
            raise CertificateError(f"{label}: Q(sqrt(3)) value must have exactly a and b")
        return Qsqrt3(
            _exact_fraction(value["a"], f"{label}.a"),
            _exact_fraction(value["b"], f"{label}.b"),
        )
    return Qsqrt3(_exact_fraction(value, label), Fraction(0))


def _exact_integer(value: Any, label: str) -> int:
    result = _exact_fraction(value, label)
    if result.denominator != 1:
        raise CertificateError(f"{label}: expected an integer, got {value!r}")
    return result.numerator


def _case_label(k: int, theta: Qsqrt3) -> str:
    return f"case=({k},{theta})"


def _normalized_profile(k: int, theta: Qsqrt3) -> dict[tuple[int, ...], Qsqrt3]:
    """Compute the exact normalized profile in ``Q(sqrt(3))``."""

    mean = Qsqrt3(Fraction(2 * (k - 2), 3))
    one = Qsqrt3.one()
    interaction_factor = Qsqrt3(Fraction(k, k - 1))
    values: dict[tuple[int, ...], Qsqrt3] = {}
    for pattern in permutations(range(k)):
        tx, ty, _, _, interaction = _local_statistics(pattern)
        h = (
            (Qsqrt3(Fraction(tx)) - mean) / theta
            + one
            - interaction_factor * interaction
            + theta * (Qsqrt3(Fraction(ty)) - mean)
        )
        values[pattern] = h / (2 * factorial(k) * factorial(k - 2))
    return values


def _permutation(text: Any, k: int, label: str) -> tuple[int, ...]:
    if not isinstance(text, str) or re.fullmatch(rf"[1-{k}]{{{k}}}", text) is None:
        raise CertificateError(f"{label}: invalid permutation encoding {text!r}")
    values = tuple(int(value) for value in text)
    if set(values) != set(range(1, k + 1)):
        raise CertificateError(f"{label}: encoding is not a permutation")
    return tuple(value - 1 for value in values)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CertificateError(f"{label}: expected a JSON object")
    return value


def _load_payload_object(payload_or_path: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(payload_or_path, (str, Path)):
        path = Path(payload_or_path)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CertificateError(f"certificate table could not be read: {path}") from error
    else:
        value = payload_or_path
    return _require_mapping(value, "certificate table")


def verify_certificate(record: Mapping[str, Any]) -> CertificateSummary:
    """Verify one primal/dual certificate keyed by exact ``(k, theta)``."""

    record = _require_mapping(record, "certificate")
    k = _exact_integer(record.get("k"), "certificate.k")
    # Keep this guard before factorials or profile enumeration.  Apart from
    # producing a clear error, it prevents hostile unsupported k values from
    # reaching a factorial-sized computation.
    if k not in _SUPPORTED_K:
        raise CertificateError(f"certificate.k: unsupported k={k}")
    theta = _exact_quadratic(record.get("theta"), f"certificate.k={k} theta")
    if theta <= 0:
        raise CertificateError(f"{_case_label(k, theta)}: theta must be positive")
    case = _case_label(k, theta)
    scale = 2 * factorial(k) * factorial(k - 2)
    expected_profile = _normalized_profile(k, theta)
    if len(expected_profile) != factorial(k):
        raise CertificateError(
            f"{case}: formula returned {len(expected_profile)} patterns, expected {factorial(k)}"
        )
    g = {pattern: value * scale for pattern, value in expected_profile.items()}

    q = _exact_quadratic(record.get("objective_q"), f"{case} objective_q")
    if q <= 0:
        raise CertificateError(f"{case}: objective q must be positive")

    raw_matrix = record.get("primal_D")
    if not isinstance(raw_matrix, list) or len(raw_matrix) != k:
        raise CertificateError(f"{case}: primal_D must have {k} rows")
    matrix: list[tuple[Qsqrt3, ...]] = []
    for row_number, raw_row in enumerate(raw_matrix, 1):
        if not isinstance(raw_row, list) or len(raw_row) != k:
            raise CertificateError(f"{case}: primal_D row {row_number} must have {k} entries")
        matrix.append(
            tuple(_exact_quadratic(value, f"{case} primal_D[{row_number}]") for value in raw_row)
        )
    if sum((sum(row) for row in matrix), Qsqrt3.zero()) != 0:
        raise CertificateError(f"{case}: primal_D is not centered")

    residuals = {
        pattern: value - sum(matrix[i][j] for i, j in enumerate(pattern))
        for pattern, value in g.items()
    }
    outside = {pattern: residual for pattern, residual in residuals.items() if abs(residual) > q}
    if outside:
        pattern, residual = next(iter(outside.items()))
        raise CertificateError(
            f"{case}: primal inequality violated at {pattern}: {residual} outside [-{q},{q}]"
        )
    if max((abs(value) for value in residuals.values()), default=Qsqrt3.zero()) != q:
        raise CertificateError(f"{case}: primal certificate does not attain q exactly")
    raw_support = record.get("dual_signed_support")
    if not isinstance(raw_support, list) or not raw_support:
        raise CertificateError(f"{case}: dual_signed_support must be a nonempty list")
    support: dict[tuple[int, ...], Qsqrt3] = {}
    for index, raw_entry in enumerate(raw_support, 1):
        entry = _require_mapping(raw_entry, f"{case} dual support {index}")
        pattern = _permutation(entry.get("tau"), k, f"{case} dual support {index}.tau")
        if pattern in support:
            raise CertificateError(f"{case}: duplicate dual support pattern {pattern}")
        if pattern not in g:
            raise CertificateError(f"{case}: dual support pattern is not in the formula")
        support[pattern] = _exact_quadratic(
            entry.get("weight"), f"{case} dual support {index}.weight"
        )
    if sum((abs(value) for value in support.values()), Qsqrt3.zero()) != 1:
        raise CertificateError(f"{case}: dual l1 norm is not exactly 1")

    marginals = tuple(
        sum(
            (weight for pattern, weight in support.items() if pattern[row] == column), Qsqrt3.zero()
        )
        for row in range(k)
        for column in range(k)
    )
    if not marginals or len(set(marginals)) != 1:
        raise CertificateError(f"{case}: dual assignment marginals are not constant")
    dual_lambda = marginals[0]
    dual_objective = sum(
        (weight * g[pattern] for pattern, weight in support.items()), Qsqrt3.zero()
    )
    if dual_objective != q:
        raise CertificateError(f"{case}: dual objective does not equal q exactly")

    square_alpha: Fraction | None = None
    if theta == 1:
        mu = Fraction(2 * (k - 2), 3)
        alpha_value = Qsqrt3(Fraction(1, 2)) * sum(
            (
                weight * (sum(_local_statistics(pattern)[:2]) - 2 * mu)
                for pattern, weight in support.items()
            ),
            Qsqrt3.zero(),
        )
        if not alpha_value.is_rational:
            raise CertificateError(f"{case}: square dual alpha is not rational")
        square_alpha = alpha_value.a
        expected_alpha = _APPROVED_SQUARE_ALPHA[k]
        if square_alpha != expected_alpha:
            raise CertificateError(
                f"{case}: square dual alpha={square_alpha} differs from approved exact value {expected_alpha}"
            )

    return CertificateSummary(
        k=k,
        theta=theta,
        q=q,
        normalized_constant=q / scale,
        patterns=len(g),
        dual_support=len(support),
        dual_lambda=dual_lambda,
        square_alpha=square_alpha,
    )


_GLOBAL_FORM_NAMES = ("W_left", "W_right", "W_square")
_GLOBAL_FORM_COEFFICIENTS: dict[str, tuple[Qsqrt3, Qsqrt3, Qsqrt3]] = {
    "W_left": (Qsqrt3(Fraction(1, 3)), Qsqrt3(Fraction(8, 5)), Qsqrt3.zero()),
    "W_right": (Qsqrt3.zero(), Qsqrt3(Fraction(8, 5)), Qsqrt3(Fraction(1, 3))),
    "W_square": (
        Qsqrt3(Fraction(-1, 5)),
        Qsqrt3(Fraction(12, 5)),
        Qsqrt3(Fraction(-1, 5)),
    ),
}


def _global_h_coefficients(pattern: tuple[int, ...]) -> tuple[Qsqrt3, Qsqrt3, Qsqrt3]:
    tx, ty, _, _, interaction = _local_statistics(pattern)
    mean = Fraction(8, 3)
    return (
        Qsqrt3(Fraction(tx) - mean),
        Qsqrt3.one() - Qsqrt3(Fraction(6, 5)) * interaction,
        Qsqrt3(Fraction(ty) - mean),
    )


def _verify_global_support(
    raw_support: Any,
    name: str,
) -> tuple[tuple[tuple[int, ...], Qsqrt3], ...]:
    if not isinstance(raw_support, list) or not raw_support:
        raise CertificateError(f"global {name}: support must be a nonempty list")
    support: dict[tuple[int, ...], Qsqrt3] = {}
    for index, raw_entry in enumerate(raw_support, 1):
        entry = _require_mapping(raw_entry, f"global {name} support {index}")
        pattern = _permutation(entry.get("tau"), 6, f"global {name} support {index}.tau")
        if pattern in support:
            raise CertificateError(f"global {name}: duplicate support pattern {pattern}")
        weight = _exact_quadratic(entry.get("weight"), f"global {name} support {index}.weight")
        if not weight.is_rational:
            raise CertificateError(f"global {name}: support weights must be rational")
        support[pattern] = weight
    if sum((abs(value) for value in support.values()), Qsqrt3.zero()) != 1:
        raise CertificateError(f"global {name}: dual l1 norm is not exactly 1")
    marginals = tuple(
        sum(
            (weight for pattern, weight in support.items() if pattern[row] == column),
            Qsqrt3.zero(),
        )
        for row in range(6)
        for column in range(6)
    )
    if len(set(marginals)) != 1:
        raise CertificateError(f"global {name}: assignment marginals are not constant")
    return tuple(support.items())


def _support_form(
    support: tuple[tuple[tuple[int, ...], Qsqrt3], ...],
) -> tuple[Qsqrt3, Qsqrt3, Qsqrt3]:
    return (
        sum(
            (weight * _global_h_coefficients(pattern)[0] for pattern, weight in support),
            Qsqrt3.zero(),
        ),
        sum(
            (weight * _global_h_coefficients(pattern)[1] for pattern, weight in support),
            Qsqrt3.zero(),
        ),
        sum(
            (weight * _global_h_coefficients(pattern)[2] for pattern, weight in support),
            Qsqrt3.zero(),
        ),
    )


def _evaluate_form(
    coefficients: tuple[Qsqrt3, Qsqrt3, Qsqrt3],
    theta: Qsqrt3,
) -> Qsqrt3:
    a_over_theta, constant, b_times_theta = coefficients
    return a_over_theta / theta + constant + b_times_theta * theta


def _verify_global_aspect(
    payload: Mapping[str, Any],
    summaries: tuple[CertificateSummary, ...],
) -> GlobalAspectSummary:
    raw = _require_mapping(payload.get("global_aspect"), "certificate table.global_aspect")
    if _exact_integer(raw.get("k"), "global_aspect.k") != 6:
        raise CertificateError("global_aspect.k: expected 6")
    rho = _exact_quadratic(raw.get("rho"), "global_aspect.rho")
    q_star = _exact_quadratic(raw.get("q_star"), "global_aspect.q_star")
    if rho != RHO:
        raise CertificateError(f"global_aspect.rho={rho} differs from exact rho {RHO}")
    if q_star != Q_STAR:
        raise CertificateError(f"global_aspect.q_star={q_star} differs from exact q {Q_STAR}")
    if rho <= 0 or rho >= 1:
        raise CertificateError("global_aspect.rho must satisfy 0<rho<1 exactly")
    by_case = {summary.case_key: summary for summary in summaries}
    rho_case = by_case.get((6, rho))
    if rho_case is None or rho_case.q != q_star:
        raise CertificateError("global_aspect: the (6,rho) certificate does not carry q_star")

    raw_coefficients = _require_mapping(
        raw.get("dual_form_coefficients"), "global_aspect.dual_form_coefficients"
    )
    raw_supports = _require_mapping(raw.get("dual_supports"), "global_aspect.dual_supports")
    if tuple(raw_coefficients) != _GLOBAL_FORM_NAMES or tuple(raw_supports) != _GLOBAL_FORM_NAMES:
        raise CertificateError("global_aspect: expected W_left, W_right, W_square in order")

    coefficients: dict[str, tuple[Qsqrt3, Qsqrt3, Qsqrt3]] = {}
    for name in _GLOBAL_FORM_NAMES:
        raw_values = raw_coefficients.get(name)
        if not isinstance(raw_values, list) or len(raw_values) != 3:
            raise CertificateError(f"global {name}: form must have three exact coefficients")
        observed = tuple(
            _exact_quadratic(value, f"global {name} form coefficient {index}")
            for index, value in enumerate(raw_values)
        )
        expected = _GLOBAL_FORM_COEFFICIENTS[name]
        if observed != expected:
            raise CertificateError(
                f"global {name}: coefficients {observed} differ from exact form {expected}"
            )
        support = _verify_global_support(raw_supports.get(name), name)
        derived = _support_form(support)
        if derived != observed:
            raise CertificateError(f"global {name}: support does not derive the displayed form")
        coefficients[name] = observed

    inverse_rho = rho.reciprocal()
    intersection_points = (
        ("W_left", rho),
        ("W_square", rho),
        ("W_right", inverse_rho),
        ("W_square", inverse_rho),
    )
    for name, aspect in intersection_points:
        observed = _evaluate_form(coefficients[name], aspect)
        if observed != q_star:
            raise CertificateError(
                f"global {name}: value at theta={aspect} is {observed}, expected q_star={q_star}"
            )
    return GlobalAspectSummary(6, rho, q_star, _GLOBAL_FORM_NAMES, len(intersection_points))


def verify_payload(
    payload_or_path: Mapping[str, Any] | str | Path,
    *,
    require_approved_values: bool = True,
) -> tuple[CertificateSummary, ...]:
    """Verify all five square and one algebraic ``(k, theta)`` cases."""

    payload = _load_payload_object(payload_or_path)

    raw_certificates = payload.get("certificates")
    if not isinstance(raw_certificates, list):
        raise CertificateError("certificate table.certificates must be a list")
    summaries = tuple(verify_certificate(record) for record in raw_certificates)
    by_case: dict[tuple[int, Qsqrt3], CertificateSummary] = {}
    for summary in summaries:
        if summary.case_key in by_case:
            raise CertificateError(
                f"certificate table: duplicate case=({summary.k},{summary.theta})"
            )
        by_case[summary.case_key] = summary
    expected_cases = tuple(_APPROVED_CASES)
    if set(by_case) != set(expected_cases):
        raise CertificateError(
            f"certificate table: expected cases={expected_cases}, got {tuple(by_case)}"
        )
    if require_approved_values:
        for case_key, expected_q in _APPROVED_CASES.items():
            if by_case[case_key].q != expected_q:
                raise CertificateError(
                    f"case=({case_key[0]},{case_key[1]}): q={by_case[case_key].q} does not match approved exact value {expected_q}"
                )
    _verify_global_aspect(payload, tuple(by_case[case_key] for case_key in expected_cases))
    return tuple(by_case[case_key] for case_key in expected_cases)


def audit_certificate_path(path: str | Path) -> CertificateAuditReport:
    """Verify a JSON file and build its deterministic source-hashed report."""

    source_path = Path(path)
    source_bytes = source_path.read_bytes()
    try:
        payload = json.loads(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CertificateError(
            f"certificate table is not valid UTF-8 JSON: {source_path}"
        ) from error
    summaries = verify_payload(payload)
    formula_path = Path(__file__).with_name("profiles.py")
    quadratic_path = Path(__file__).with_name("quadratic.py")
    formula_bytes = (
        formula_path.read_bytes() + b"\n--quadratic.py--\n" + quadratic_path.read_bytes()
    )
    global_aspect = _verify_global_aspect(payload, summaries)
    return CertificateAuditReport(
        input_sha256=sha256(source_bytes).hexdigest(),
        formula_code_sha256=sha256(formula_bytes).hexdigest(),
        certificates=summaries,
        global_aspect=global_aspect,
    )


def default_certificate_path() -> Path:
    return Path(__file__).with_name("exact_lp_certificates.json")
