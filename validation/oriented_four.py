"""Exact finite checks for the oriented ``k=4`` completion class.

Counts from block comparisons are checked against independently constructed
coordinate orders. Coordinates, word counts, polynomial coefficients,
residuals, and dual calculations use integers and fractions.

The row sign is attached to an old signed row.  Inserted colour-zero points do
not use this sign: their ordinary key is ``(y,0,0)`` and their coordinates are
nonintegral, so no inserted tie is resolved by a row sign.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations, product
from math import factorial
from typing import Sequence

Pattern = tuple[int, ...]
SignWord = tuple[int, ...]

# The fixed period used by the all-size construction.  The formula is kept in
# integer-floor form so this check does not depend on floating point.
PERIOD8_SIGNS: tuple[int, ...] = (-1, 1, -1, 1, 1, -1, 1, 1)
PERIOD8_MEAN = Fraction(1, 4)
ETA = PERIOD8_MEAN

C_MATRIX: tuple[tuple[int, ...], ...] = (
    (9, -1, -4, -4),
    (-1, -3, -2, 6),
    (-4, -2, 3, 3),
    (-4, 6, 3, -5),
)

# The primal assignment matrix for the corrected oriented LP certificate.
D_LP: tuple[tuple[Fraction, ...], ...] = (
    (Fraction(-3, 8), Fraction(1, 24), Fraction(1, 6), Fraction(1, 6)),
    (Fraction(1, 24), Fraction(1, 8), Fraction(1, 12), Fraction(-1, 4)),
    (Fraction(1, 6), Fraction(1, 12), Fraction(-1, 8), Fraction(-1, 8)),
    (Fraction(1, 6), Fraction(-1, 4), Fraction(-1, 8), Fraction(5, 24)),
)

DUAL_WEIGHT_INTS: dict[str, int] = {
    "1243": -2,
    "1342": 2,
    "1423": 2,
    "2134": 2,
    "2143": -1,
    "2413": 1,
    "3142": 1,
    "3241": 2,
    "3412": -1,
    "4213": 2,
}
DUAL_WEIGHT_DENOMINATOR = 16
DUAL_WEIGHTS: dict[str, Fraction] = {
    name: Fraction(weight, DUAL_WEIGHT_DENOMINATOR) for name, weight in DUAL_WEIGHT_INTS.items()
}
# Tuple-key form is convenient for exact statistic calculations.
CENTERED_DUAL_WEIGHTS: dict[Pattern, Fraction] = {
    tuple(int(value) - 1 for value in name): weight for name, weight in DUAL_WEIGHTS.items()
}

EXPECTED_RESIDUALS: dict[str, Fraction] = {
    "1234": Fraction(0),
    "1243": Fraction(-1, 2),
    "1324": Fraction(-1, 2),
    "1342": Fraction(1, 2),
    "1423": Fraction(1, 2),
    "1432": Fraction(0),
    "2134": Fraction(1, 2),
    "2143": Fraction(-1, 2),
    "2314": Fraction(0),
    "2341": Fraction(-1, 6),
    "2413": Fraction(1, 2),
    "2431": Fraction(0),
    "3124": Fraction(0),
    "3142": Fraction(1, 2),
    "3214": Fraction(0),
    "3241": Fraction(1, 2),
    "3412": Fraction(-1, 2),
    "3421": Fraction(-1, 6),
    "4123": Fraction(-1, 6),
    "4132": Fraction(0),
    "4213": Fraction(1, 2),
    "4231": Fraction(-1, 2),
    "4312": Fraction(-1, 6),
    "4321": Fraction(-1, 3),
}
EXPECTED_LEDGER_GROUPS: dict[int, frozenset[str]] = {
    -12: frozenset(("1243", "1324", "2143", "3412", "4231")),
    -8: frozenset(("4321",)),
    -4: frozenset(("2341", "3421", "4123", "4312")),
    0: frozenset(("1234", "1432", "2314", "2431", "3124", "3214", "4132")),
    12: frozenset(("1342", "1423", "2134", "2413", "3142", "3241", "4213")),
}


def _validate_pattern(pattern: Pattern, size: int = 4) -> None:
    if len(pattern) != size or set(pattern) != set(range(size)):
        raise ValueError(f"pattern must be a permutation of range({size})")


def pattern_name(pattern: Pattern) -> str:
    _validate_pattern(pattern)
    return "".join(str(value + 1) for value in pattern)


def all_patterns(size: int = 4) -> tuple[Pattern, ...]:
    if size < 0:
        raise ValueError("pattern size must be nonnegative")
    return tuple(permutations(range(size)))


def compositions(total: int) -> tuple[tuple[int, ...], ...]:
    """Return all positive compositions of ``total`` in deterministic order."""

    if not isinstance(total, int) or isinstance(total, bool) or total < 1:
        raise ValueError("composition total must be a positive integer")
    result: list[tuple[int, ...]] = []
    for cuts in product((0, 1), repeat=total - 1):
        parts: list[int] = []
        current = 1
        for cut in cuts:
            if cut:
                parts.append(current)
                current = 1
            else:
                current += 1
        parts.append(current)
        result.append(tuple(parts))
    return tuple(result)


def block_indices(parts: Sequence[int]) -> tuple[int, ...]:
    if not parts or any(type(part) is not int or part <= 0 for part in parts):
        raise ValueError("block sizes must be positive integers")
    return tuple(block for block, size in enumerate(parts) for _ in range(size))


def period8_sign(row: int) -> int:
    """Return the exact sign at positive integer row ``row``."""

    if not isinstance(row, int) or isinstance(row, bool) or row < 1:
        raise ValueError("row must be a positive integer")
    # 2(floor(5x/8)-floor(5(x-1)/8))-1.
    return 2 * ((5 * row) // 8 - (5 * (row - 1)) // 8) - 1


def validate_period8() -> dict[str, object]:
    signs = tuple(period8_sign(row) for row in range(1, 9))
    if signs != PERIOD8_SIGNS:
        raise AssertionError(f"period-eight signs differ: {signs}")
    if sum(signs, 0) != 2:
        raise AssertionError("period-eight sign sum is not 2")
    if Fraction(sum(signs), len(signs)) != PERIOD8_MEAN:
        raise AssertionError("period-eight mean is not 1/4")
    if tuple(period8_sign(row) for row in range(1, 17)) != signs + signs:
        raise AssertionError("period-eight signs do not repeat")
    return {"period": 8, "signs": signs, "mean": PERIOD8_MEAN}


def _check_sign_word(signs: SignWord, length: int) -> None:
    if len(signs) != length or any(sign not in (-1, 1) for sign in signs):
        raise ValueError("a sign word must have one +/-1 sign per X block")


def _coordinate_blocks(
    pattern: Pattern,
    alpha: tuple[int, ...],
    beta: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    _validate_pattern(pattern)
    if sum(alpha) != 4 or sum(beta) != 4:
        raise ValueError("both compositions must sum to four")
    x_block = block_indices(alpha)
    y_block_by_rank = block_indices(beta)
    inverse = [0] * 4
    for x_position, y_rank in enumerate(pattern):
        inverse[y_rank] = x_position
    return x_block, y_block_by_rank, tuple(inverse)


def block_word_count(
    pattern: Pattern,
    alpha: tuple[int, ...],
    beta: tuple[int, ...],
    signs: SignWord,
) -> int:
    """Count compatible colours from block inequalities only.

    This is intentionally separate from :func:`direct_key_word_count`.  The
    latter constructs actual keys and sorts them; agreement between the two is
    the finite direct-key independence check.
    """

    x_block, y_block_by_rank, inverse = _coordinate_blocks(pattern, alpha, beta)
    _check_sign_word(signs, len(alpha))
    count = 0
    for colours in product((-1, 1), repeat=4):
        x_keys = tuple(
            (
                x_block[index],
                -colours[index] * (y_block_by_rank[pattern[index]] + 1),
                colours[index],
            )
            for index in range(4)
        )
        if tuple(sorted(range(4), key=x_keys.__getitem__)) != tuple(range(4)):
            continue
        y_keys = tuple(
            (
                y_block_by_rank[pattern[index]],
                signs[x_block[index]] * colours[index] * (x_block[index] + 1),
                colours[index],
            )
            for index in range(4)
        )
        if len(set(y_keys)) != 4:
            continue
        y_order = tuple(sorted(range(4), key=y_keys.__getitem__))
        if y_order == inverse:
            count += 1
    return count


def _default_coordinates(count: int) -> tuple[Fraction, ...]:
    return tuple(Fraction(index + 1) for index in range(count))


def oriented_x_key(point: tuple[Fraction, Fraction, int]) -> tuple[Fraction, Fraction, int]:
    """The unchanged signed X key ``(x,-c*y,c)``."""

    x, y, colour = point
    return (x, -colour * y, colour)


def oriented_y_key(
    point: tuple[Fraction, Fraction, int],
    row_sign: int,
) -> tuple[Fraction, Fraction, int]:
    """The oriented old-point Y key ``(y,sigma*c*x,c)``."""

    x, y, colour = point
    if row_sign not in (-1, 1):
        raise ValueError("row_sign must be -1 or +1")
    return (y, row_sign * colour * x, colour)


def direct_key_word_count(
    pattern: Pattern,
    alpha: tuple[int, ...],
    beta: tuple[int, ...],
    signs: SignWord,
    *,
    x_values: Sequence[Fraction] | None = None,
    y_values: Sequence[Fraction] | None = None,
) -> int:
    """Count compatible colours by constructing and sorting actual keys."""

    x_block, y_block_by_rank, inverse = _coordinate_blocks(pattern, alpha, beta)
    _check_sign_word(signs, len(alpha))
    if x_values is None:
        x_values = _default_coordinates(len(alpha))
    if y_values is None:
        y_values = _default_coordinates(len(beta))
    if len(x_values) != len(alpha) or len(y_values) != len(beta):
        raise ValueError("coordinate arrays must match their composition lengths")
    x_values = tuple(Fraction(value) for value in x_values)
    y_values = tuple(Fraction(value) for value in y_values)

    count = 0
    for colours in product((-1, 1), repeat=4):
        points = tuple(
            (
                x_values[x_block[index]],
                y_values[y_block_by_rank[pattern[index]]],
                colours[index],
            )
            for index in range(4)
        )
        x_keys = tuple(oriented_x_key(point) for point in points)
        y_keys = tuple(
            oriented_y_key(point, signs[x_block[index]]) for index, point in enumerate(points)
        )
        if len(set(x_keys)) != 4 or len(set(y_keys)) != 4:
            continue
        if tuple(sorted(range(4), key=x_keys.__getitem__)) != tuple(range(4)):
            continue
        if tuple(sorted(range(4), key=y_keys.__getitem__)) == inverse:
            count += 1
    return count


def direct_key_independence_check() -> dict[str, int]:
    """Compare actual-key and block-word counts in all 5280 finite cases."""

    comparisons = 0
    mismatches = 0
    for defect in (0, 1, 2):
        for pattern in all_patterns(4):
            for alpha in compositions(4):
                for beta in compositions(4):
                    if 8 - len(alpha) - len(beta) != defect:
                        continue
                    for signs in product((-1, 1), repeat=len(alpha)):
                        direct = direct_key_word_count(pattern, alpha, beta, signs)
                        formula = block_word_count(pattern, alpha, beta, signs)
                        comparisons += 1
                        if direct != formula:
                            mismatches += 1
    if comparisons != 5280:
        raise AssertionError(f"unexpected direct-key comparison count {comparisons}")
    if mismatches:
        raise AssertionError(f"direct-key/block count mismatches: {mismatches}")
    return {"comparisons": comparisons, "mismatches": mismatches}


def low_defect_uniformity_check() -> dict[str, int]:
    """Check sign-word independence for every defect-zero/one case."""

    comparisons = 0
    nonuniform = 0
    for defect in (0, 1):
        for pattern in all_patterns(4):
            for alpha in compositions(4):
                for beta in compositions(4):
                    if 8 - len(alpha) - len(beta) != defect:
                        continue
                    values = []
                    for signs in product((-1, 1), repeat=len(alpha)):
                        values.append(block_word_count(pattern, alpha, beta, signs))
                        comparisons += 1
                    # Uniformity is both across row signs and across patterns.
                    if any(value != (16 if defect == 0 else 8) for value in values):
                        nonuniform += 1
    if nonuniform:
        raise AssertionError(f"low-defect sign-word nonuniformity in {nonuniform} cases")
    return {"sign_word_cases": comparisons, "nonuniform_cases": nonuniform}


def _polynomial_add(
    left: list[Fraction], right: Sequence[Fraction], factor: Fraction = Fraction(1)
) -> list[Fraction]:
    if len(left) < len(right):
        left.extend(Fraction(0) for _ in range(len(right) - len(left)))
    for index, value in enumerate(right):
        left[index] += factor * value
    return left


def _sign_word_probability_polynomial(signs: SignWord) -> tuple[Fraction, ...]:
    # Product_s (1+s*eta)/2, represented in ascending eta powers.
    polynomial = [Fraction(1)]
    for sign in signs:
        next_values = [Fraction(0)] * (len(polynomial) + 1)
        for degree, value in enumerate(polynomial):
            next_values[degree] += value / 2
            next_values[degree + 1] += value * sign / 2
        polynomial = next_values
    return tuple(polynomial)


def _raw_defect_two_polynomials(
    pattern: Pattern,
) -> dict[int, tuple[Fraction, ...]]:
    raw: dict[int, list[Fraction]] = {
        2: [Fraction(0)] * 5,
        3: [Fraction(0)] * 5,
        4: [Fraction(0)] * 5,
    }
    for alpha in compositions(4):
        for beta in compositions(4):
            if 8 - len(alpha) - len(beta) != 2:
                continue
            denominator = factorial(len(alpha)) * factorial(len(beta))
            for signs in product((-1, 1), repeat=len(alpha)):
                probability = _sign_word_probability_polynomial(signs)
                word_count = block_word_count(pattern, alpha, beta, signs)
                contribution = tuple(
                    Fraction(word_count, denominator) * coefficient for coefficient in probability
                )
                _polynomial_add(raw[len(alpha)], contribution)
    return {degree: tuple(values) for degree, values in raw.items()}


def _polynomial_subtract(
    left: Sequence[Fraction], right: Sequence[Fraction]
) -> tuple[Fraction, ...]:
    size = max(len(left), len(right))
    return tuple(
        (left[index] if index < len(left) else Fraction(0))
        - (right[index] if index < len(right) else Fraction(0))
        for index in range(size)
    )


def _local_statistics(pattern: Pattern) -> tuple[int, int, int, int]:
    _validate_pattern(pattern)
    inverse = [0] * 4
    for index, value in enumerate(pattern):
        inverse[value] = index

    def middle_not_peak(values: Sequence[int]) -> int:
        return sum(values[index + 1] != max(values[index : index + 3]) for index in range(2))

    horizontal = [
        int(index < 3 and pattern[index + 1] < value)
        - int(index > 0 and pattern[index - 1] < value)
        for index, value in enumerate(pattern)
    ]
    vertical = [
        int(value < 3 and inverse[value + 1] < index)
        - int(value > 0 and inverse[value - 1] < index)
        for index, value in enumerate(pattern)
    ]
    interaction = sum(left * right for left, right in zip(horizontal, vertical))
    bonds = sum(abs(pattern[index + 1] - pattern[index]) == 1 for index in range(3))
    return middle_not_peak(pattern), middle_not_peak(tuple(inverse)), interaction, bonds


def local_statistics(pattern: Pattern) -> tuple[int, int, int, int]:
    return _local_statistics(pattern)


def corrected_h(
    pattern: Pattern,
    theta: Fraction = Fraction(1),
    eta: Fraction = ETA,
) -> Fraction:
    """Return the corrected oriented local leading expression."""

    if not isinstance(theta, Fraction) or theta <= 0:
        raise ValueError("theta must be a positive Fraction")
    if not isinstance(eta, Fraction):
        eta = Fraction(eta)
    tx, ty, interaction, bonds = _local_statistics(pattern)
    return (
        (Fraction(tx) - Fraction(4, 3)) / theta
        + theta * (Fraction(ty) - Fraction(4, 3))
        + 1
        - Fraction(4, 3) * eta * interaction
        - Fraction(2, 3) * (1 - eta) * bonds
    )


def expected_centered_polynomial(
    pattern: Pattern,
    degree: int,
) -> tuple[Fraction, ...]:
    """Expected centred eta polynomial for one theta-Laurent component."""

    tx, ty, interaction, bonds = _local_statistics(pattern)
    if degree == 2:
        return (Fraction(tx) - Fraction(4, 3), Fraction(0), Fraction(0), Fraction(0), Fraction(0))
    if degree == 3:
        return (
            Fraction(1) - Fraction(2, 3) * bonds,
            Fraction(2, 3) * bonds - Fraction(4, 3) * interaction,
            Fraction(0),
            Fraction(0),
            Fraction(0),
        )
    if degree == 4:
        return (Fraction(ty) - Fraction(4, 3), Fraction(0), Fraction(0), Fraction(0), Fraction(0))
    raise ValueError("defect-two Laurent component must have degree 2, 3, or 4")


def finite_color_word_polynomial_check() -> dict[str, int]:
    """Verify the 360 coefficient identities in the corrected eta polynomial."""

    patterns = all_patterns(4)
    raw_by_pattern = {pattern: _raw_defect_two_polynomials(pattern) for pattern in patterns}
    means: dict[int, tuple[Fraction, ...]] = {}
    for degree in (2, 3, 4):
        means[degree] = tuple(
            sum((raw_by_pattern[pattern][degree][coefficient] for pattern in patterns), Fraction(0))
            / 24
            for coefficient in range(5)
        )
    identities = 0
    mismatches = 0
    for pattern in patterns:
        for degree in (2, 3, 4):
            centered = _polynomial_subtract(raw_by_pattern[pattern][degree], means[degree])
            expected = expected_centered_polynomial(pattern, degree)
            for coefficient in range(5):
                identities += 1
                observed = 12 * centered[coefficient]
                if observed != expected[coefficient]:
                    mismatches += 1
    if identities != 360:
        raise AssertionError(f"unexpected polynomial identity count {identities}")
    if mismatches:
        raise AssertionError(f"finite eta-polynomial mismatches: {mismatches}")
    return {
        "patterns": 24,
        "laurent_components": 3,
        "eta_coefficients": 5,
        "identities": identities,
        "mismatches": mismatches,
    }


@dataclass(frozen=True)
class FiniteColorCertificate:
    """Summary of the three exact finite colour-word checks."""

    direct_key_comparisons: int
    low_defect_sign_word_cases: int
    polynomial_identities: int

    def as_dict(self) -> dict[str, int]:
        return {
            "direct_key_comparisons": self.direct_key_comparisons,
            "low_defect_sign_word_cases": self.low_defect_sign_word_cases,
            "polynomial_identities": self.polynomial_identities,
        }


def validate_finite_color_certificate() -> FiniteColorCertificate:
    direct = direct_key_independence_check()
    low = low_defect_uniformity_check()
    polynomial = finite_color_word_polynomial_check()
    return FiniteColorCertificate(
        direct_key_comparisons=direct["comparisons"],
        low_defect_sign_word_cases=low["sign_word_cases"],
        polynomial_identities=polynomial["identities"],
    )


def assignment_value(
    matrix: Sequence[Sequence[Fraction]],
    pattern: Pattern,
) -> Fraction:
    _validate_pattern(pattern)
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError("assignment matrix must be four by four")
    return sum((Fraction(matrix[row][pattern[row]]) for row in range(4)), Fraction(0))


def dual_marginals() -> dict[tuple[int, int], Fraction]:
    return {
        (row, column): sum(
            (weight for pattern, weight in CENTERED_DUAL_WEIGHTS.items() if pattern[row] == column),
            Fraction(0),
        )
        for row in range(4)
        for column in range(4)
    }


def weighted_statistic(name: str) -> Fraction:
    index = {"Tx": 0, "Ty": 1, "S": 2, "U": 3}
    if name == "1":
        return sum(CENTERED_DUAL_WEIGHTS.values(), Fraction(0))
    if name not in index:
        raise ValueError(f"unknown local statistic {name!r}")
    return sum(
        weight * _local_statistics(pattern)[index[name]]
        for pattern, weight in CENTERED_DUAL_WEIGHTS.items()
    )


def dual_objective(theta: Fraction = Fraction(1), eta: Fraction = ETA) -> Fraction:
    return sum(
        weight * corrected_h(pattern, theta, eta)
        for pattern, weight in CENTERED_DUAL_WEIGHTS.items()
    )


def residual_ledger(
    theta: Fraction = Fraction(1),
    eta: Fraction = ETA,
) -> dict[str, Fraction]:
    """Return the exact 24 residuals ``h-A_D``."""

    return {
        pattern_name(pattern): corrected_h(pattern, theta, eta) - assignment_value(D_LP, pattern)
        for pattern in all_patterns(4)
    }


def oriented_ledger(theta: Fraction = Fraction(1), eta: Fraction = ETA) -> dict[str, Fraction]:
    """Return the corrected limiting ledger ``h/96 + C-assignment/2304``."""

    return {
        pattern_name(pattern): corrected_h(pattern, theta, eta) / 96
        + Fraction(
            sum(C_MATRIX[row][pattern[row]] for row in range(4)),
            2304,
        )
        for pattern in all_patterns(4)
    }


def validate_centered_dual() -> dict[str, object]:
    """Verify all primal residuals and the exact centred ten-pattern dual."""

    row_sums = tuple(sum(row, Fraction(0)) for row in D_LP)
    column_sums = tuple(
        sum((D_LP[row][column] for row in range(4)), Fraction(0)) for column in range(4)
    )
    if row_sums != (Fraction(0),) * 4 or column_sums != (Fraction(0),) * 4:
        raise AssertionError("oriented primal assignment matrix is not centred")
    residuals = residual_ledger()
    if residuals != EXPECTED_RESIDUALS:
        raise AssertionError(f"oriented residual ledger differs: {residuals!r}")
    if len(residuals) != 24 or max(map(abs, residuals.values())) != Fraction(1, 2):
        raise AssertionError("oriented residual maximum is not 1/2")
    marginals = dual_marginals()
    if set(marginals.values()) != {Fraction(1, 8)}:
        raise AssertionError("dual position-value marginals are not all 1/8")
    l1 = sum((abs(weight) for weight in CENTERED_DUAL_WEIGHTS.values()), Fraction(0))
    if l1 != 1:
        raise AssertionError(f"dual l1 norm is {l1}, not 1")
    if weighted_statistic("1") != Fraction(1, 2):
        raise AssertionError("dual W(1) is not 1/2")
    if weighted_statistic("Tx") != Fraction(3, 4) or weighted_statistic("Ty") != Fraction(3, 4):
        raise AssertionError("dual T_x/T_y statistic is wrong")
    if weighted_statistic("S") != Fraction(1, 8) or weighted_statistic("U") != Fraction(1, 4):
        raise AssertionError("dual interaction/bond statistic is wrong")
    objective = dual_objective()
    if objective != Fraction(1, 2):
        raise AssertionError(f"dual objective is {objective}, not 1/2")
    weighted_assignment = sum(
        weight * assignment_value(D_LP, pattern)
        for pattern, weight in CENTERED_DUAL_WEIGHTS.items()
    )
    if weighted_assignment != 0:
        raise AssertionError("dual does not annihilate the centred primal assignment")
    if (
        sum(
            CENTERED_DUAL_WEIGHTS[pattern] * residuals[pattern_name(pattern)]
            for pattern in CENTERED_DUAL_WEIGHTS
        )
        != objective
    ):
        raise AssertionError("dual residual objective does not equal primal objective")
    return {
        "patterns": len(residuals),
        "max_residual": max(map(abs, residuals.values())),
        "active_boundary_residuals": sum(
            abs(value) == Fraction(1, 2) for value in residuals.values()
        ),
        "dual_terms": len(CENTERED_DUAL_WEIGHTS),
        "dual_l1": l1,
        "dual_marginal": Fraction(1, 8),
        "dual_objective": objective,
        "weighted_assignment": weighted_assignment,
        "statistics": {
            "one": weighted_statistic("1"),
            "Tx": weighted_statistic("Tx"),
            "Ty": weighted_statistic("Ty"),
            "S": weighted_statistic("S"),
            "U": weighted_statistic("U"),
        },
    }


def validate_oriented_ledger() -> dict[str, object]:
    ledger = oriented_ledger()
    scaled = {name: value * 2304 for name, value in ledger.items()}
    if any(value.denominator != 1 for value in scaled.values()):
        raise AssertionError("oriented ledger does not have integral scale 2304")
    observed = {
        int(value): frozenset(name for name, candidate in scaled.items() if candidate == value)
        for value in sorted(set(scaled.values()))
    }
    expected = {level: patterns for level, patterns in EXPECTED_LEDGER_GROUPS.items()}
    if observed != expected:
        raise AssertionError(f"oriented ledger groups differ: {observed!r}")
    if max(map(abs, ledger.values())) != Fraction(1, 192):
        raise AssertionError("oriented ledger maximum is not 1/192")
    return {
        "patterns": len(ledger),
        "scale": 2304,
        "levels": {level: len(patterns) for level, patterns in expected.items()},
        "maximum": Fraction(1, 192),
    }


def validate_all_finite_oriented_checks() -> dict[str, object]:
    """Run every finite check owned by this module."""

    period = validate_period8()
    colours = validate_finite_color_certificate()
    dual = validate_centered_dual()
    ledger = validate_oriented_ledger()
    return {
        "period8": period,
        "colour_words": colours.as_dict(),
        "dual": dual,
        "ledger": ledger,
    }


if __name__ == "__main__":
    import json

    result = validate_all_finite_oriented_checks()
    print(json.dumps(result, default=str, sort_keys=True, indent=2))
    print("ORIENTED_FOUR_CERTIFICATE_OK")
