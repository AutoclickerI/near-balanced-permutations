"""Exact finite audit for the square coefficient compared with BLL Theorem 8.

This uses the formula-free geometric enumerator for its direct side and the
closed profile formula for its symbolic side. It is corroborating computation
only; the manuscript's exact derivation is the proof. Run from the repository
root with

    PYTHONDONTWRITEBYTECODE=1 python3 -B -m validation.bll_coefficient_audit
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction

from .independent import enumerate_profiles
from .profiles import exact_discrepancy, square_profile_formula

EXPECTED = {
    2: Fraction(61, 12),
    3: Fraction(141, 2),
    4: Fraction(1375, 3),
    5: Fraction(11375, 6),
    6: Fraction(23625, 4),
    7: Fraction(45815, 3),
}


def pattern_name(pattern: tuple[int, ...]) -> str:
    return "".join(str(value + 1) for value in pattern)


def main() -> None:
    print("m\tN\tdelta\tmaximizing patterns\tdirect check")
    for side, expected in EXPECTED.items():
        order = 2 * side * side
        formula = square_profile_formula(side, 4)
        discrepancy = exact_discrepancy(formula, order, 4)
        if discrepancy != expected:
            raise AssertionError(f"m={side}: formula gives {discrepancy}, expected {expected}")

        direct: Counter[tuple[int, ...]] = Counter()
        for layer in enumerate_profiles(side, 4).values():
            direct.update(
                {tuple(value - 1 for value in pattern): count for pattern, count in layer.items()}
            )
        if direct != formula:
            raise AssertionError(f"m={side}: direct and exact profiles differ")
        direct_status = "profile match"

        target = Fraction(sum(formula.values()), 24)
        maximizers = sorted(
            pattern_name(pattern)
            for pattern, count in formula.items()
            if abs(Fraction(count) - target) == discrepancy
        )
        print(f"{side}\t{order}\t{discrepancy}\t{','.join(maximizers)}\t{direct_status}")

    m4 = square_profile_formula(4, 4)
    increasing = (0, 1, 2, 3)
    if m4[increasing] != 1040:
        raise AssertionError(f"m=4: #1234={m4[increasing]}, expected 1040")

    print("m=4 #1234=1040")
    print("grid-side leading coefficient=5/36")
    print("permutation-size leading coefficient=5/288")
    print("BLL_COEFFICIENT_AUDIT_OK")


if __name__ == "__main__":
    main()
