"""Current exact-profile and local-statistic validation oracles.

This module intentionally excludes the retired transport, perturbation, fixed-offset,
and growing-pattern validators preserved in the dated archive.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, permutations, product
from math import comb, factorial
from typing import Iterable, Sequence

from .construction import EvenShape, even_shape, point_orders

Pattern = tuple[int, ...]


def all_patterns(k: int) -> tuple[Pattern, ...]:
    if k < 0:
        raise ValueError("pattern size must be nonnegative")
    return tuple(permutations(range(k)))


def standardise(values: Sequence[int]) -> Pattern:
    """Return the relative order of pairwise-distinct values."""

    if len(set(values)) != len(values):
        raise ValueError("pattern values must be distinct")
    rank = {value: index for index, value in enumerate(sorted(values))}
    return tuple(rank[value] for value in values)


def direct_profile(order: int, k: int) -> Counter[Pattern]:
    """Enumerate all k-subsets from independently constructed total orders."""

    _validate_profile_args(order, k)
    points, y_ranks, _ = point_orders(order)
    result: Counter[Pattern] = Counter({pattern: 0 for pattern in all_patterns(k)})
    for chosen in combinations(points, k):
        result[standardise(tuple(y_ranks[point] for point in chosen))] += 1
    return result


def exact_discrepancy(profile: Counter[Pattern], order: int, k: int) -> Fraction:
    """Return max_tau |#tau-C(order,k)/k!| using exact arithmetic."""

    _validate_profile_args(order, k)
    expected = Fraction(comb(order, k), factorial(k))
    return max(
        (abs(Fraction(profile[pattern]) - expected) for pattern in all_patterns(k)),
        default=Fraction(0),
    )


def classify_even_subsets(order: int, k: int) -> dict[str, Counter[Pattern]]:
    """Partition all subsets into the five proof classes for even order."""

    _validate_profile_args(order, k)
    if order % 2:
        raise ValueError("order must be even")
    points, y_ranks, shape = point_orders(order)
    assert shape is not None
    categories = {
        name: Counter({pattern: 0 for pattern in all_patterns(k)})
        for name in (
            "core_good",
            "core_bad",
            "partial_at_least_two",
            "partial_one_collision",
            "partial_one_clean",
        )
    }
    partial_x = shape.rows + 1
    for chosen in combinations(points, k):
        partial_count = sum(point.x == partial_x for point in chosen)
        x_distinct = len({point.x for point in chosen})
        y_distinct = len({point.y for point in chosen})
        defect = 2 * k - x_distinct - y_distinct
        if partial_count == 0:
            category = "core_good" if defect <= 1 else "core_bad"
        elif partial_count >= 2:
            category = "partial_at_least_two"
        elif defect:
            category = "partial_one_collision"
        else:
            category = "partial_one_clean"
        pattern = standardise(tuple(y_ranks[point] for point in chosen))
        categories[category][pattern] += 1
    return categories


def clean_partial_formula(order: int, k: int) -> Counter[Pattern]:
    """Closed-form profile for exactly one partial-row point and no collisions."""

    _validate_profile_args(order, k)
    if order % 2:
        raise ValueError("order must be positive and even")
    shape = even_shape(order)
    result: Counter[Pattern] = Counter()
    choose_x = _comb0(shape.rows, k - 1)
    sums = {}
    for t in range(1, k + 1):
        sums[t] = sum(
            _comb0(y - 1, t - 1) * _comb0(shape.columns - y, k - t) for y in shape.partial_columns
        )
    for pattern in all_patterns(k):
        result[pattern] = (2**k) * choose_x * sums[pattern[-1] + 1]
    return result


def compositions(total: int) -> tuple[tuple[int, ...], ...]:
    """Return all positive compositions of ``total`` in a fixed order."""

    if total < 1:
        raise ValueError("composition total must be positive")
    result = []
    for separators in product((False, True), repeat=total - 1):
        parts = []
        current = 1
        for separator in separators:
            if separator:
                parts.append(current)
                current = 1
            else:
                current += 1
        parts.append(current)
        result.append(tuple(parts))
    return tuple(result)


def _block_indices(parts: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(block for block, size in enumerate(parts) for _ in range(size))


def _strictly_increasing(values: Sequence[tuple[int, int]]) -> bool:
    return all(left < right for left, right in zip(values, values[1:]))


def tie_colour_count(
    pattern: Pattern,
    x_multiplicities: tuple[int, ...],
    y_multiplicities: tuple[int, ...],
) -> int:
    """Count colour words compatible with prescribed X/Y equality blocks.

    The points are indexed in final X-order. ``x_multiplicities`` records
    consecutive equal-original-X blocks in that order, while
    ``y_multiplicities`` records consecutive equal-original-Y blocks in
    final Y-order. The count is independent of the chosen coordinate
    values and is the coefficient in ``square_profile_formula``.
    """

    k = len(pattern)
    if sorted(pattern) != list(range(k)):
        raise ValueError("pattern must be a permutation of range(k)")
    if (
        not x_multiplicities
        or not y_multiplicities
        or any(type(part) is not int or part <= 0 for part in x_multiplicities)
        or any(type(part) is not int or part <= 0 for part in y_multiplicities)
    ):
        raise ValueError("multiplicities must be nonempty positive integer compositions")
    if sum(x_multiplicities) != k or sum(y_multiplicities) != k:
        raise ValueError("multiplicities must both sum to the pattern size")
    return _tie_colour_count_cached(pattern, x_multiplicities, y_multiplicities)


@lru_cache(maxsize=None)
def _tie_colour_count_cached(
    pattern: Pattern,
    x_multiplicities: tuple[int, ...],
    y_multiplicities: tuple[int, ...],
) -> int:
    """Validated tie-colour count implementation with memoization."""

    k = len(pattern)
    x_block = _block_indices(x_multiplicities)
    y_rank_block = _block_indices(y_multiplicities)
    y_block = tuple(y_rank_block[pattern[index]] for index in range(k))
    inverse = [0] * k
    for index, rank in enumerate(pattern):
        inverse[rank] = index

    x_ranges = []
    start = 0
    for size in x_multiplicities:
        x_ranges.append(range(start, start + size))
        start += size
    y_ranges = []
    start = 0
    for size in y_multiplicities:
        y_ranges.append(range(start, start + size))
        start += size

    count = 0
    for colours in product((-1, 1), repeat=k):
        x_ok = all(
            _strictly_increasing(
                tuple(
                    (0, -y_block[index]) if colours[index] == 1 else (1, y_block[index])
                    for index in block_range
                )
            )
            for block_range in x_ranges
        )
        if not x_ok:
            continue
        y_ok = all(
            _strictly_increasing(
                tuple(
                    (0, -x_block[inverse[rank]])
                    if colours[inverse[rank]] == -1
                    else (1, x_block[inverse[rank]])
                    for rank in block_range
                )
            )
            for block_range in y_ranges
        )
        count += y_ok
    return count


def rectangular_profile_formula(rows: int, columns: int, k: int) -> Counter[Pattern]:
    """Exact binomial-basis profile of a complete two-sided rectangle.

    This formula is structurally independent of direct subset enumeration:
    it sums over the compositions that encode equality blocks on the two
    original coordinate axes and counts only compatible colour words.
    """

    if rows < 1 or columns < 1:
        raise ValueError("rectangle dimensions must be positive")
    _validate_profile_args(2 * rows * columns, k)
    patterns = all_patterns(k)
    parts = compositions(k)
    result: Counter[Pattern] = Counter({pattern: 0 for pattern in patterns})
    for pattern in patterns:
        total = 0
        for x_multiplicities in parts:
            choose_x = _comb0(rows, len(x_multiplicities))
            if not choose_x:
                continue
            for y_multiplicities in parts:
                choose_y = _comb0(columns, len(y_multiplicities))
                if not choose_y:
                    continue
                total += (
                    tie_colour_count(pattern, x_multiplicities, y_multiplicities)
                    * choose_x
                    * choose_y
                )
        result[pattern] = total
    return result


def square_profile_formula(side: int, k: int) -> Counter[Pattern]:
    """Exact profile of the two-sided ``side`` square."""

    return rectangular_profile_formula(side, side, k)


def square_leading_coefficients(k: int) -> dict[Pattern, Fraction]:
    """Return the centred coefficients of ``side**(2*k-2)``.

    Defect-zero and defect-one layers are exactly pattern-uniform. Hence
    only composition pairs whose total coordinate defect is two contribute
    to the first possibly nonuniform coefficient.
    """

    if k < 2:
        raise ValueError("pattern size must be at least two")
    patterns = all_patterns(k)
    parts = compositions(k)
    raw: dict[Pattern, Fraction] = {}
    for pattern in patterns:
        coefficient = Fraction(0)
        for x_multiplicities in parts:
            x_blocks = len(x_multiplicities)
            for y_multiplicities in parts:
                y_blocks = len(y_multiplicities)
                if 2 * k - x_blocks - y_blocks != 2:
                    continue
                coefficient += Fraction(
                    tie_colour_count(pattern, x_multiplicities, y_multiplicities),
                    factorial(x_blocks) * factorial(y_blocks),
                )
        raw[pattern] = coefficient
    average = sum(raw.values(), Fraction(0)) / factorial(k)
    return {pattern: raw[pattern] - average for pattern in patterns}


def square_leading_discrepancy(k: int) -> Fraction:
    """Return the exact leading square discrepancy coefficient ``Lambda_k``."""

    return max(abs(value) for value in square_leading_coefficients(k).values())


def _local_statistics(pattern: Pattern) -> tuple[int, int, int, int, int]:
    """Return ``(T_x,T_y,X,Y,S)`` for the local defect-two formula."""

    k = len(pattern)
    inverse = [0] * k
    for index, value in enumerate(pattern):
        inverse[value] = index

    def middle_not_peak_count(values: Sequence[int]) -> int:
        return sum(
            values[index + 1] != max(values[index], values[index + 1], values[index + 2])
            for index in range(k - 2)
        )

    horizontal: list[int] = []
    vertical: list[int] = []
    for index, value in enumerate(pattern):
        horizontal.append(
            int(index < k - 1 and pattern[index + 1] < value)
            - int(index > 0 and pattern[index - 1] < value)
        )
        vertical.append(
            int(value < k - 1 and inverse[value + 1] < index)
            - int(value > 0 and inverse[value - 1] < index)
        )
    interaction = sum(x_value * y_value for x_value, y_value in zip(horizontal, vertical))
    peaks_x = sum(
        pattern[index] > pattern[index - 1] and pattern[index] > pattern[index + 1]
        for index in range(1, k - 1)
    )
    peaks_y = sum(
        inverse[index] > inverse[index - 1] and inverse[index] > inverse[index + 1]
        for index in range(1, k - 1)
    )
    return (
        middle_not_peak_count(pattern),
        middle_not_peak_count(tuple(inverse)),
        peaks_x,
        peaks_y,
        interaction,
    )


def square_local_leading_coefficients(k: int) -> dict[Pattern, Fraction]:
    """Return the square leading vector from the local defect-two formula."""

    _validate_profile_args(k, k)
    result: dict[Pattern, Fraction] = {}
    prefactor = Fraction(2 ** (k - 2), factorial(k) * factorial(k - 2))
    for pattern in all_patterns(k):
        t_x, t_y, _, _, interaction = _local_statistics(pattern)
        g_value = t_x + t_y + Fraction(11 - 4 * k, 3) - Fraction(k, k - 1) * interaction
        result[pattern] = prefactor * g_value
    return result


def rectangular_normalized_leading_coefficients(
    k: int, aspect: Fraction
) -> dict[Pattern, Fraction]:
    """Return the fixed-aspect vector normalized by ``(2ab)**(k-1)``."""

    _validate_profile_args(k, k)
    if not isinstance(aspect, Fraction) or aspect <= 0:
        raise ValueError("aspect must be a positive Fraction")
    mean_t = Fraction(2 * (k - 2), 3)
    prefactor = Fraction(1, 2 * factorial(k) * factorial(k - 2))
    result: dict[Pattern, Fraction] = {}
    for pattern in all_patterns(k):
        t_x, t_y, _, _, interaction = _local_statistics(pattern)
        h_value = (
            Fraction(t_x, 1) / aspect
            - mean_t / aspect
            + 1
            - Fraction(k, k - 1) * interaction
            + aspect * (Fraction(t_y, 1) - mean_t)
        )
        result[pattern] = prefactor * h_value
    return result


def balanced_prefix_errors(shape: EvenShape) -> tuple[Fraction, ...]:
    """Return #R∩[z] - r*z/b for z=0,...,b."""

    partial = set(shape.partial_columns)
    count = 0
    errors = [Fraction(0)]
    for z in range(1, shape.columns + 1):
        count += z in partial
        errors.append(Fraction(count) - Fraction(shape.remainder * z, shape.columns))
    return tuple(errors)


def vandermonde_sums(columns: int, k: int) -> tuple[int, ...]:
    """Return sum_y C(y-1,t-1)C(b-y,k-t) for every t."""

    return tuple(
        sum(_comb0(y - 1, t - 1) * _comb0(columns - y, k - t) for y in range(1, columns + 1))
        for t in range(1, k + 1)
    )


def merge_profiles(profiles: Iterable[Counter[Pattern]]) -> Counter[Pattern]:
    result: Counter[Pattern] = Counter()
    for profile in profiles:
        result.update(profile)
    return result


def _comb0(n: int, k: int) -> int:
    if n < 0 or k < 0 or k > n:
        return 0
    return comb(n, k)


def _validate_profile_args(order: int, k: int) -> None:
    if order < 1 or not (2 <= k <= order):
        raise ValueError("require order >= k >= 2")
