"""Exact validation oracles for the oriented 12/17-point four-pattern family.

The insertion list has 12 points in the even case and 17 in the odd case.
Dimension selection, root isolation, floors, and moments use exact arithmetic.
The standard-orientation LP certificates are checked in ``lp_certificates``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb, floor, isqrt
from typing import Iterator, Mapping

from .construction import permutation as universal_permutation
from .oriented_four import (
    DUAL_WEIGHTS,
    ETA,
    corrected_h,
    oriented_x_key,
    oriented_y_key,
    period8_sign,
    validate_centered_dual,
    validate_oriented_ledger,
    validate_period8,
)
from .profiles import all_patterns


@dataclass(frozen=True)
class QuadraticFive:
    """An exact number of the form ``rational + radical * sqrt(5)``."""

    rational: Fraction = Fraction(0)
    radical: Fraction = Fraction(0)

    @classmethod
    def scalar(cls, value: int | Fraction) -> "QuadraticFive":
        return cls(Fraction(value), Fraction(0))

    def __sub__(self, other: "QuadraticFive") -> "QuadraticFive":
        return QuadraticFive(
            self.rational - other.rational,
            self.radical - other.radical,
        )

    def __neg__(self) -> "QuadraticFive":
        return QuadraticFive(-self.rational, -self.radical)

    def sign(self) -> int:
        """Return the exact sign without converting to floating point."""

        a, b = self.rational, self.radical
        if b == 0:
            return (a > 0) - (a < 0)
        if b < 0:
            return -(-self).sign()
        if a >= 0:
            return 1
        comparison = 5 * b * b - a * a
        return (comparison > 0) - (comparison < 0)

    def absolute(self) -> "QuadraticFive":
        return self if self.sign() >= 0 else -self

    def __lt__(self, other: "QuadraticFive") -> bool:
        return (self - other).sign() < 0


# Exact algebraic roots and floors.


@dataclass(frozen=True)
class IsolatedRoot:
    """A real algebraic root isolated by rational bounds.

    ``coefficients`` are in descending order.  The interval is expected to
    contain exactly the represented root.  The floor routine does not use a
    numerical approximation: it binary-searches an integer between the scaled
    isolating bounds and compares the polynomial at ``k/scale``.  The endpoint
    orientation identifies which polynomial sign is the lower side of the
    root, and an exact zero handles rational-root equality.
    """

    coefficients: tuple[int, ...]
    lower: Fraction
    upper: Fraction
    label: str = ""

    def __post_init__(self) -> None:
        coefficients = tuple(self.coefficients)
        if len(coefficients) < 2 or coefficients[0] == 0:
            raise ValueError("an isolated root needs a nonconstant polynomial")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in coefficients):
            raise TypeError("polynomial coefficients must be integers")
        try:
            lower = Fraction(self.lower)
            upper = Fraction(self.upper)
        except (TypeError, ValueError, ZeroDivisionError) as error:
            raise TypeError("isolating bounds must be rational") from error
        if not lower < upper:
            raise ValueError("isolating bounds must be strictly increasing")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def degree(self) -> int:
        return len(self.coefficients) - 1

    def evaluate(self, value: Fraction) -> Fraction:
        """Evaluate the defining polynomial at an exact rational value."""

        value = Fraction(value)
        result = Fraction(0)
        for coefficient in self.coefficients:
            result = result * value + coefficient
        return result

    def endpoint_signs(self) -> tuple[int, int]:
        """Return the exact signs at the two isolating endpoints."""

        return (_sign(self.evaluate(self.lower)), _sign(self.evaluate(self.upper)))

    def _strict_orientation(self) -> tuple[int, int, int]:
        lower_value = self.evaluate(self.lower)
        upper_value = self.evaluate(self.upper)
        lower_sign = _sign(lower_value)
        upper_sign = _sign(upper_value)
        if lower_sign == 0 or upper_sign == 0:
            raise AssertionError("the interval has a rational root endpoint")
        if lower_sign == upper_sign:
            raise AssertionError("isolating endpoints do not have opposite signs")
        # +1 means the polynomial changes from negative to positive as x grows.
        orientation = 1 if lower_sign < upper_sign else -1
        return lower_sign, upper_sign, orientation

    def compare_to_rational(self, value: Fraction) -> int:
        """Compare this root with ``value``: -1, 0, or +1."""

        value = Fraction(value)
        lower_value = self.evaluate(self.lower)
        upper_value = self.evaluate(self.upper)

        # A rational endpoint is itself the represented root.  This branch is
        # also what makes rational-root equality explicit rather than relying
        # on interval refinement eventually landing on it.
        if lower_value == 0:
            return 0 if value == self.lower else (-1 if value > self.lower else 1)
        if upper_value == 0:
            return 0 if value == self.upper else (-1 if value > self.upper else 1)

        lower_sign, upper_sign, orientation = self._strict_orientation()
        if value <= self.lower:
            return 1
        if value >= self.upper:
            return -1
        polynomial_sign = _sign(self.evaluate(value))
        if polynomial_sign == 0:
            return 0
        if polynomial_sign == -orientation:
            return 1
        if polynomial_sign == orientation:
            return -1
        # Keep the names in the local scope as a defensive consistency check:
        # an isolated interval can only expose its two endpoint signs.
        if polynomial_sign in (lower_sign, upper_sign):
            raise AssertionError("endpoint orientation is inconsistent")
        raise AssertionError("polynomial sign is inconsistent with isolation")

    def floor_scaled(self, scale: int) -> int:
        """Return ``floor(scale * root)`` with exact integer bisection."""

        if not isinstance(scale, int) or isinstance(scale, bool) or scale <= 0:
            raise ValueError("scale must be a positive integer")

        lower_value = self.evaluate(self.lower)
        upper_value = self.evaluate(self.upper)
        if lower_value == 0:
            return floor(self.lower * scale)
        if upper_value == 0:
            return floor(self.upper * scale)
        self._strict_orientation()

        # The root is at least lower_index/scale and strictly below
        # upper_index/scale.  Search for the largest integer k with
        # root >= k/scale; equality is accepted, so it is precisely the floor.
        lower_index = floor(self.lower * scale)
        upper_index = floor(self.upper * scale) + 1
        while lower_index + 1 < upper_index:
            middle = (lower_index + upper_index) // 2
            comparison = self.compare_to_rational(Fraction(middle, scale))
            if comparison >= 0:
                lower_index = middle
            else:
                upper_index = middle
        return lower_index


# Descriptive aliases used by callers that refer to a root rather than its
# isolating certificate.


def _sign(value: Fraction) -> int:
    return (value > 0) - (value < 0)


def algebraic_floor(value: Fraction | IsolatedRoot | int, scale: int) -> int:
    """Return ``floor(scale*value)`` for a rational or isolated root."""

    if not isinstance(scale, int) or isinstance(scale, bool) or scale <= 0:
        raise ValueError("scale must be a positive integer")
    if isinstance(value, IsolatedRoot):
        return value.floor_scaled(scale)
    return floor(Fraction(value) * scale)


# These primitive integer cubics are the exact Newton-root certificate for
# the corrected twelve-point design.  Each row has three roots, each used
# once (there is no old factor-of-two repetition).
CUBIC_POLYNOMIALS: tuple[tuple[int, ...], ...] = (
    (9172942848, -12525207552, 4569969024, -366861277),
    (4194304, -6586368, 2746496, -242611),
    (4194304, -6782976, 3256448, -401701),
    (9172942848, -13273694208, 5415500160, -602145179),
)

# These are the intervals displayed in the active source.  Their endpoint
# signs are checked exactly; the narrower rational intervals below are useful
# for replay and prove that the three roots in each row are distinct.
COMMON_ROOT_INTERVALS: tuple[tuple[Fraction, Fraction], ...] = (
    (Fraction(1, 10), Fraction(1, 5)),
    (Fraction(2, 5), Fraction(7, 10)),
    (Fraction(3, 4), Fraction(19, 20)),
)

ROOT_ENDPOINT_SIGNS: tuple[tuple[int, int], ...] = (
    (-1, 1),
    (1, -1),
    (-1, 1),
)

ROOT_ISOLATING_INTERVALS: tuple[tuple[tuple[Fraction, Fraction], ...], ...] = (
    (
        (Fraction(24, 215), Fraction(23, 206)),
        (Fraction(48, 109), Fraction(85, 193)),
        (Fraction(61, 75), Fraction(109, 134)),
    ),
    (
        (Fraction(43, 357), Fraction(10, 83)),
        (Fraction(43, 84), Fraction(64, 125)),
        (Fraction(151, 161), Fraction(136, 145)),
    ),
    (
        (Fraction(241, 1274), Fraction(7, 37)),
        (Fraction(53, 81), Fraction(89, 136)),
        (Fraction(41, 53), Fraction(147, 190)),
    ),
    (
        (Fraction(23, 126), Fraction(21, 115)),
        (Fraction(60, 139), Fraction(79, 183)),
        (Fraction(244, 293), Fraction(249, 299)),
    ),
)

# Short aliases make the reference data easy to inspect from a test or REPL.


def _make_cubic_root_groups() -> tuple[tuple[IsolatedRoot, ...], ...]:
    return tuple(
        tuple(
            IsolatedRoot(
                polynomial,
                lower,
                upper,
                label=f"cubic-{group}-{root}",
            )
            for root, (lower, upper) in enumerate(intervals)
        )
        for group, (polynomial, intervals) in enumerate(
            zip(CUBIC_POLYNOMIALS, ROOT_ISOLATING_INTERVALS)
        )
    )


CUBIC_ROOT_GROUPS = _make_cubic_root_groups()


def root_group(group: int) -> tuple[IsolatedRoot, ...]:
    """Return one zero-based cubic-root group."""

    if not isinstance(group, int) or isinstance(group, bool) or not 0 <= group < 4:
        raise ValueError("cubic root group must be in [0,3]")
    return CUBIC_ROOT_GROUPS[group]


def padding_polynomial(group: int) -> tuple[int, ...]:
    """Return the integer polynomial for one zero-based root group."""

    root_group(group)  # validate the index
    return CUBIC_POLYNOMIALS[group]


def validate_root_certificates() -> None:
    """Validate the exact endpoint signs and disjoint cubic isolation data."""

    for polynomial, intervals in zip(CUBIC_POLYNOMIALS, ROOT_ISOLATING_INTERVALS):
        if len(intervals) != 3:
            raise AssertionError("every cubic must have three root intervals")
        previous_upper: Fraction | None = None
        for interval, common, expected_signs in zip(
            intervals, COMMON_ROOT_INTERVALS, ROOT_ENDPOINT_SIGNS
        ):
            common_lower, common_upper = common
            common_root = IsolatedRoot(polynomial, common_lower, common_upper)
            if common_root.endpoint_signs() != expected_signs:
                raise AssertionError("common root endpoints have the wrong exact signs")
            lower, upper = interval
            if not common_lower < lower < upper < common_upper:
                raise AssertionError("root interval is outside its common isolating interval")
            if previous_upper is not None and not previous_upper < lower:
                raise AssertionError("root intervals are not disjoint and ordered")
            root = IsolatedRoot(polynomial, lower, upper)
            if root.endpoint_signs() != expected_signs:
                raise AssertionError(
                    f"unexpected endpoint signs {root.endpoint_signs()} != {expected_signs}"
                )
            previous_upper = upper

    for outer, expected in zip((G2_MINUS, G2_PLUS), ((1, -1), (-1, 1))):
        if outer.endpoint_signs() != expected:
            raise AssertionError("outer quadratic G2 root is not correctly isolated")
        if outer.degree != 2 or outer.coefficients != G2_POLYNOMIAL:
            raise AssertionError("outer G2 root has the wrong defining polynomial")

    for outer, expected in zip((G3_MINUS, G3_PLUS), ((1, -1), (-1, 1))):
        if outer.endpoint_signs() != expected:
            raise AssertionError("outer quadratic G3 root is not correctly isolated")
        if outer.degree != 2 or outer.coefficients != G3_POLYNOMIAL:
            raise AssertionError("outer G3 root has the wrong defining polynomial")


# Newton identities and exact Bernstein moments.


def newton_power_sums(
    coefficients: tuple[int, ...] | list[int],
    count: int | None = None,
) -> tuple[Fraction, ...]:
    """Return exact root power sums from Newton identities.

    Coefficients are descending and need not be monic.  When ``count`` exceeds
    the degree, the recurrence continues with the usual zero ``k*c_k`` term.
    """

    if len(coefficients) < 2 or coefficients[0] == 0:
        raise ValueError("Newton identities need a nonconstant polynomial")
    degree = len(coefficients) - 1
    if any(not isinstance(value, int) or isinstance(value, bool) for value in coefficients):
        raise TypeError("polynomial coefficients must be integers")
    if count is None:
        count = degree
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("power-sum count must be a nonnegative integer")

    monic = [Fraction(value, coefficients[0]) for value in coefficients]
    powers: list[Fraction] = []
    for k in range(1, count + 1):
        recurrence = Fraction(0)
        for j in range(1, min(k, degree) + 1):
            if j == k:
                recurrence += k * monic[j]
            else:
                recurrence += monic[j] * powers[k - j - 1]
        powers.append(-recurrence)
    return tuple(powers)


def bernstein3(index: int, value: Fraction) -> Fraction:
    """Evaluate the cubic Bernstein basis ``B_{3,index}`` exactly."""

    if not 0 <= index <= 3:
        raise ValueError("Bernstein index must lie in [0,3]")
    value = Fraction(value)
    return Fraction(comb(3, index)) * value**index * (1 - value) ** (3 - index)


def bernstein_sums_from_power_sums(
    powers: tuple[Fraction, Fraction, Fraction],
    root_count: int,
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    """Convert the first three power sums into four cubic Bernstein sums."""

    if root_count < 0:
        raise ValueError("root count must be nonnegative")
    p1, p2, p3 = powers
    return (
        Fraction(root_count) - 3 * p1 + 3 * p2 - p3,
        3 * (p1 - 2 * p2 + p3),
        3 * (p2 - p3),
        p3,
    )


def root_power_sums(group: int) -> tuple[Fraction, Fraction, Fraction]:
    """Return the first three power sums for one cubic root group."""

    powers = newton_power_sums(padding_polynomial(group), 3)
    return powers[0], powers[1], powers[2]


def root_bernstein_sums(group: int) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    """Return ``sum_root B_{3,s}(root)`` for one cubic group."""

    return bernstein_sums_from_power_sums(root_power_sums(group), 3)


def root_bernstein_matrix() -> tuple[tuple[Fraction, ...], ...]:
    """Return the four-by-four matrix of cubic-group Bernstein sums."""

    return tuple(root_bernstein_sums(group) for group in range(4))


# The displayed D matrix is the twice-centred Bernstein correction.  The
# cubic rows use beta = 3/4 + D/2, so their group deviation is D/2.
ROOT_DEVIATION: tuple[tuple[Fraction, ...], ...] = tuple(
    tuple(Fraction(value, 192) for value in row)
    for row in (
        (51, 5, -8, -48),
        (18, -36, -63, 81),
        (-63, -9, 72, 0),
        (-6, 40, -1, -33),
    )
)


def root_deviation_matrix() -> tuple[tuple[Fraction, ...], ...]:
    """Return ``2*(root_bernstein_matrix - 3/4)`` row by row."""

    return tuple(
        tuple(2 * (value - Fraction(3, 4)) for value in row) for row in root_bernstein_matrix()
    )


# The projected matrix of the corrected twelve-point root list is C/64.
PROJECTED_C: tuple[tuple[int, ...], ...] = (
    (9, -1, -4, -4),
    (-1, -3, -2, 6),
    (-4, -2, 3, 3),
    (-4, 6, 3, -5),
)
INTERIOR_CORRECTION_UNITS = PROJECTED_C
INTERIOR_TARGET_MOMENTS: tuple[tuple[Fraction, ...], ...] = tuple(
    tuple(Fraction(value, 64) for value in row) for row in PROJECTED_C
)


def node_bernstein_vector(value: Fraction) -> tuple[Fraction, ...]:
    return tuple(bernstein3(index, Fraction(value)) for index in range(4))


def root_moment_matrix() -> tuple[tuple[Fraction, ...], ...]:
    """Return the uncentered 4-by-4 moments of the 12-point root list."""

    return tuple(
        tuple(
            sum(
                (
                    node_bernstein_vector(FOUR_U[group])[row] * root_bernstein_sums(group)[column]
                    for group in range(4)
                ),
                Fraction(0),
            )
            for column in range(4)
        )
        for row in range(4)
    )


def centered_moment_matrix(
    matrix: tuple[tuple[Fraction, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    """Apply ``P=I-J/4`` on both sides of a four-by-four matrix."""

    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError("expected a 4-by-4 matrix")
    row_sums = [sum(matrix[row], Fraction(0)) for row in range(4)]
    column_sums = [
        sum((matrix[row][column] for row in range(4)), Fraction(0)) for column in range(4)
    ]
    total = sum(row_sums, Fraction(0))
    return tuple(
        tuple(
            matrix[row][column] - row_sums[row] / 4 - column_sums[column] / 4 + total / 16
            for column in range(4)
        )
        for row in range(4)
    )


def projected_root_moments() -> tuple[tuple[Fraction, ...], ...]:
    """Return the centered Bernstein moments of the 12-point root list."""

    return centered_moment_matrix(root_moment_matrix())


# The two-point rule $G_2$ and the three-point rule $G_3$ are used in the
# odd-order neutral tail.  Their outer roots are represented by exact
# quadratic certificates; the middle point of $G_3$ is 1/2.
G2_POLYNOMIAL: tuple[int, ...] = (6, -6, 1)
G2_MINUS = IsolatedRoot(G2_POLYNOMIAL, Fraction(0), Fraction(1, 2), "G2-minus")
G2_PLUS = IsolatedRoot(G2_POLYNOMIAL, Fraction(1, 2), Fraction(1), "G2-plus")
G2: tuple[Fraction | IsolatedRoot, ...] = (G2_MINUS, G2_PLUS)

G3_POLYNOMIAL: tuple[int, ...] = (8, -8, 1)
G3_MINUS = IsolatedRoot(G3_POLYNOMIAL, Fraction(0), Fraction(1, 2), "G3-minus")
G3_MIDDLE = Fraction(1, 2)
G3_PLUS = IsolatedRoot(G3_POLYNOMIAL, Fraction(1, 2), Fraction(1), "G3-plus")
G3: tuple[Fraction | IsolatedRoot, ...] = (G3_MINUS, G3_MIDDLE, G3_PLUS)


def g2_power_sums() -> tuple[Fraction, Fraction, Fraction]:
    powers = newton_power_sums(G2_POLYNOMIAL, 3)
    return (powers[0], powers[1], powers[2])


def g2_bernstein_sums() -> tuple[Fraction, Fraction, Fraction, Fraction]:
    return bernstein_sums_from_power_sums(g2_power_sums(), 2)


def g_power_sums() -> tuple[Fraction, Fraction, Fraction]:
    """Return the first three power sums of the three-point set G3."""

    outer = newton_power_sums(G3_POLYNOMIAL, 3)
    return (
        outer[0] + Fraction(1, 2),
        outer[1] + Fraction(1, 2) ** 2,
        outer[2] + Fraction(1, 2) ** 3,
    )


def g_bernstein_sums() -> tuple[Fraction, Fraction, Fraction, Fraction]:
    """Return the four exact Bernstein sums over G3."""

    return bernstein_sums_from_power_sums(g_power_sums(), 3)


def neutral_five_moment_matrix() -> tuple[tuple[Fraction, ...], ...]:
    """Return the uncentered Bernstein moments of the five-point H tail."""

    g2_sums = g2_bernstein_sums()
    g3_sums = g_bernstein_sums()
    u0 = node_bernstein_vector(Fraction(0))
    u1 = node_bernstein_vector(Fraction(1))
    return tuple(
        tuple(u0[row] * g2_sums[column] + u1[row] * g3_sums[column] for column in range(4))
        for row in range(4)
    )


def projected_neutral_moments() -> tuple[tuple[Fraction, ...], ...]:
    return centered_moment_matrix(neutral_five_moment_matrix())


# The 12/17-point insertion lists and their coordinates.


AlgebraicCoordinate = Fraction | IsolatedRoot
PaddingPair = tuple[AlgebraicCoordinate, AlgebraicCoordinate]


def _make_even_padding_pairs() -> tuple[PaddingPair, ...]:
    pairs: list[PaddingPair] = []
    for group, roots in enumerate(CUBIC_ROOT_GROUPS):
        u = Fraction(group, 3)
        for root in roots:
            pairs.append((u, root))
    return tuple(pairs)


def _make_odd_padding_pairs(
    even_pairs: tuple[PaddingPair, ...],
) -> tuple[PaddingPair, ...]:
    tail = tuple((Fraction(0), value) for value in G2)
    tail += tuple((Fraction(1), value) for value in G3)
    return even_pairs + tail


EVEN_PADDING_PAIRS = _make_even_padding_pairs()
ODD_PADDING_PAIRS = _make_odd_padding_pairs(EVEN_PADDING_PAIRS)
PADDING_PAIRS: dict[int, tuple[PaddingPair, ...]] = {
    0: EVEN_PADDING_PAIRS,
    1: ODD_PADDING_PAIRS,
}

# Active-family constants.
ASPECT = Fraction(1)
FOUR_U: tuple[Fraction, ...] = tuple(Fraction(index, 3) for index in range(4))
ROW_SLOPE = QuadraticFive(Fraction(3), Fraction(-1))  # 3 - sqrt(5)
PADDING_DELTA = Fraction(1, 2)
PADDING_COUNTS = {0: 12, 1: 17}


@dataclass(frozen=True)
class SharpDimensions:
    """The selected ``a x b`` core dimensions for ``E`` cells."""

    cells: int
    rows: int
    columns: int
    remainder: int
    search_start: int
    search_length: int

    @property
    def row_distance(self) -> QuadraticFive:
        return circular_distance(self.remainder, self.columns)


@dataclass(frozen=True)
class SharpPoint:
    x: Fraction
    y: Fraction
    colour: int


@dataclass(frozen=True)
class SharpFamilySpec:
    """The exact branch metadata for one ``Sigma_N`` order."""

    order: int
    parity: int
    padding_mass: int
    core_size: int
    dimensions: SharpDimensions | None
    fallback: bool


def _check_parity(parity: int) -> int:
    if not isinstance(parity, int) or isinstance(parity, bool) or parity not in (0, 1):
        raise ValueError("parity must be 0 (even) or 1 (odd)")
    return parity


def padding_pairs(parity: int) -> tuple[PaddingPair, ...]:
    """Return the ordered even/odd algebraic insertion list."""

    parity = _check_parity(parity)
    pairs = PADDING_PAIRS[parity]
    if len(pairs) != PADDING_COUNTS[parity]:
        raise AssertionError("padding list has the wrong cardinality")
    return pairs


def even_padding_pairs() -> tuple[PaddingPair, ...]:
    return padding_pairs(0)


def odd_padding_pairs() -> tuple[PaddingPair, ...]:
    return padding_pairs(1)


def validate_padding_lists() -> None:
    """Check twelve root pairs and the odd neutral five-point tail."""

    even = even_padding_pairs()
    if len(even) != 12:
        raise AssertionError("even padding must have 12 pairs")
    expected_even = tuple(
        (Fraction(group, 3), root)
        for group, roots in enumerate(CUBIC_ROOT_GROUPS)
        for root in roots
    )
    if even != expected_even:
        raise AssertionError("even padding is not in group/root order")
    if len(set(even)) != 12:
        raise AssertionError("even padding locations are not distinct")

    odd = odd_padding_pairs()
    expected_tail = tuple((Fraction(0), value) for value in G2)
    expected_tail += tuple((Fraction(1), value) for value in G3)
    if odd[:12] != even or len(odd[12:]) != 5:
        raise AssertionError("odd padding does not extend the even list by five pairs")
    if odd[12:] != expected_tail:
        raise AssertionError("odd G2/G3 tail is not lexicographic")


def padding_mass(parity: int) -> int:
    return len(padding_pairs(parity))


def padding_point_at(
    rows: int,
    columns: int,
    parity: int,
    index: int,
) -> SharpPoint:
    """Return the one-based indexed insertion point exactly."""

    if rows <= 0 or columns <= 0:
        raise ValueError("rows and columns must be positive")
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError("padding index must be an integer")
    pairs = padding_pairs(parity)
    mass = len(pairs)
    if not 1 <= index <= mass:
        raise IndexError("padding index must lie in [1,m]")
    u, v = pairs[index - 1]
    offset = Fraction(index, mass + 1)
    return SharpPoint(
        Fraction(algebraic_floor(u, rows)) + offset,
        Fraction(algebraic_floor(v, columns)) + offset,
        0,
    )


def iter_padding_points(
    rows: int,
    columns: int,
    parity: int,
) -> Iterator[SharpPoint]:
    """Yield the 12 or 17 insertion points in their global-index order."""

    for index in range(1, padding_mass(parity) + 1):
        yield padding_point_at(rows, columns, parity, index)


# Exact selectors, dimensions, and the all-size construction.


def circular_distance(
    remainder: int,
    columns: int,
    target: QuadraticFive = ROW_SLOPE,
) -> QuadraticFive:
    """Return the exact distance of ``remainder/columns`` to ``target`` modulo 1."""

    if columns <= 0 or not (0 <= remainder < columns):
        raise ValueError("remainder must lie in [0, columns)")
    difference = QuadraticFive.scalar(Fraction(remainder, columns)) - target
    direct = difference.absolute()
    wrapped = (QuadraticFive.scalar(1) - direct).absolute()
    return direct if direct < wrapped else wrapped


def shifted_partial_columns(
    columns: int,
    remainder: int,
    delta: Fraction,
) -> tuple[int, ...]:
    """Return the exact shifted mechanical set ``R_delta``."""

    if columns < 2 or not (0 < remainder < columns):
        raise ValueError("require 0 < remainder < columns")
    if not isinstance(delta, Fraction) or not (0 < delta < 1):
        raise ValueError("delta must be a Fraction in (0,1)")
    result = tuple(
        floor((Fraction(j) - delta) * columns / remainder) + 1 for j in range(1, remainder + 1)
    )
    if len(set(result)) != remainder or not all(1 <= y <= columns for y in result):
        raise AssertionError("shifted selector must contain distinct columns in [b]")
    return result


def shifted_prefix_errors(
    columns: int,
    remainder: int,
    delta: Fraction,
) -> tuple[Fraction, ...]:
    selected = set(shifted_partial_columns(columns, remainder, delta))
    count = 0
    errors = [Fraction(0)]
    for z in range(1, columns + 1):
        count += z in selected
        errors.append(Fraction(count) - Fraction(remainder * z, columns))
    return tuple(errors)


def boundary_quadrature(
    k: int,
    t: int,
    columns: int,
    remainder: int,
    delta: Fraction,
) -> Fraction:
    """Return the exact finite quantity ``q_{k,t}(b,R_delta)``."""

    if not (1 <= t <= k <= columns):
        raise ValueError("require 1 <= t <= k <= columns")
    selected = shifted_partial_columns(columns, remainder, delta)
    weighted = sum(comb(y - 1, t - 1) * comb(columns - y, k - t) for y in selected)
    density_average = Fraction(remainder, columns) * comb(columns, k)
    return Fraction(weighted - density_average, columns ** (k - 1))


def _integer_cube_root(value: int) -> int:
    """Return ``floor(value ** (1/3))`` without floating-point arithmetic."""

    if value < 0:
        raise ValueError("cube-root input must be nonnegative")
    low = 0
    high = 1 << ((value.bit_length() + 2) // 3)
    while low + 1 < high:
        middle = (low + high) // 2
        if middle**3 <= value:
            low = middle
        else:
            high = middle
    return low


def _ceil_cube_root(value: int) -> int:
    root = _integer_cube_root(value)
    return root if root**3 == value else root + 1


def dimension_search_base(cells: int) -> int:
    """Return the exact selected-family base ``B=isqrt(E)``."""

    if cells < 1:
        raise ValueError("the dimension search requires E >= 1")
    return isqrt(cells)


def dimension_search_window(base: int) -> int:
    """Return the exact public window ``ceil(base**(2/3))``."""

    if base < 1:
        raise ValueError("search base must be positive")
    return _ceil_cube_root(base * base)


def sharp_dimensions(cells: int) -> SharpDimensions:
    """Select ``a,b,r`` by the exact circular search in the public window."""

    base = dimension_search_base(cells)
    window = dimension_search_window(base)
    best: tuple[QuadraticFive, int, int, int] | None = None
    for columns in range(base, base + window + 1):
        rows, remainder = divmod(cells, columns)
        score = circular_distance(remainder, columns)
        candidate = (score, columns, rows, remainder)
        if best is None or score < best[0] or (score == best[0] and columns < best[1]):
            best = candidate
    assert best is not None
    _, columns, rows, remainder = best
    return SharpDimensions(cells, rows, columns, remainder, base, window)


def _fallback_point_orders(
    order: int,
) -> tuple[list[SharpPoint], dict[SharpPoint, int], None]:
    values = universal_permutation(order)
    points = [
        SharpPoint(Fraction(index + 1), Fraction(value + 1), 0)
        for index, value in enumerate(values)
    ]
    return points, {point: int(point.y) - 1 for point in points}, None


def sharp_family_spec(order: int) -> SharpFamilySpec:
    """Return branch metadata without changing the fallback convention."""

    if not isinstance(order, int) or isinstance(order, bool) or order <= 0:
        raise ValueError("order must be a positive integer")
    parity = order % 2
    mass = PADDING_COUNTS[parity]
    core_size = order - mass
    if core_size < 100:
        return SharpFamilySpec(order, parity, mass, core_size, None, True)
    if core_size % 2:
        raise AssertionError("parity-matched padding must leave an even core")
    dimensions = sharp_dimensions(core_size // 2)
    # The zero-remainder branch is intentionally retained as an exact Pi_N
    # fallback; it is a finite exceptional case, not part of the limit.
    return SharpFamilySpec(
        order,
        parity,
        mass,
        core_size,
        dimensions,
        dimensions.remainder == 0,
    )


def _oriented_row_sign(x: Fraction) -> int:
    """Return the period-eight sign used for an old signed point.

    Only integral old rows can participate in a Y tie.  Fractional insertion
    coordinates are colour zero and therefore use the ordinary ``(y,0,0)``
    key; assigning the floor-row sign there keeps the key helper total without
    changing any ordering.
    """

    row = x.numerator // x.denominator
    return period8_sign(max(1, row))


def _active_point_orders(
    spec: SharpFamilySpec,
) -> tuple[list[SharpPoint], dict[SharpPoint, int], SharpDimensions]:
    if spec.fallback or spec.dimensions is None:
        raise AssertionError("active point construction needs nonzero remainder dimensions")
    dimensions = spec.dimensions
    selected = shifted_partial_columns(
        dimensions.columns,
        dimensions.remainder,
        PADDING_DELTA,
    )
    points = [
        SharpPoint(Fraction(x), Fraction(y), colour)
        for x in range(1, dimensions.rows + 1)
        for y in range(1, dimensions.columns + 1)
        for colour in (-1, 1)
    ]
    points.extend(
        SharpPoint(Fraction(dimensions.rows + 1), Fraction(y), colour)
        for y in selected
        for colour in (-1, 1)
    )
    points.extend(iter_padding_points(dimensions.rows, dimensions.columns, spec.parity))

    if len(points) != spec.order or len(set(points)) != spec.order:
        raise AssertionError("Sigma_N point construction has the wrong cardinality")
    padding = [point for point in points if point.colour == 0]
    if len(padding) != spec.padding_mass:
        raise AssertionError("Sigma_N padding has the wrong cardinality")
    if len({point.x for point in padding}) != spec.padding_mass:
        raise AssertionError("padding X coordinates must be distinct")
    if len({point.y for point in padding}) != spec.padding_mass:
        raise AssertionError("padding Y coordinates must be distinct")
    if any(point.x.denominator == 1 or point.y.denominator == 1 for point in padding):
        raise AssertionError("padding coordinates must be nonintegral")

    x_keys = tuple(oriented_x_key((point.x, point.y, point.colour)) for point in points)
    y_keys = tuple(
        oriented_y_key(
            (point.x, point.y, point.colour),
            _oriented_row_sign(point.x),
        )
        for point in points
    )
    if len(set(x_keys)) != spec.order:
        raise AssertionError("oriented X keys are not injective")
    if len(set(y_keys)) != spec.order:
        raise AssertionError("oriented Y keys are not injective")
    x_order = [points[index] for index in sorted(range(spec.order), key=x_keys.__getitem__)]
    y_order = [points[index] for index in sorted(range(spec.order), key=y_keys.__getitem__)]
    ranks = {point: index for index, point in enumerate(y_order)}
    if len(x_order) != spec.order or len(ranks) != spec.order:
        raise AssertionError("Sigma_N orders have the wrong cardinality")
    if sorted(ranks.values()) != list(range(spec.order)):
        raise AssertionError("oriented Y order does not define all ranks")
    return x_order, ranks, dimensions


def sharp_point_orders(
    order: int,
) -> tuple[list[SharpPoint], dict[SharpPoint, int], SharpDimensions | None]:
    """Return the exact X-order, Y-ranks, and dimensions for ``Sigma_N``."""

    spec = sharp_family_spec(order)
    if spec.fallback:
        points, ranks, _ = _fallback_point_orders(order)
        return points, ranks, spec.dimensions
    return _active_point_orders(spec)


def sharp_permutation(order: int) -> tuple[int, ...]:
    """Return the zero-based permutation induced by the two exact orders."""

    points, ranks, _ = sharp_point_orders(order)
    result = tuple(ranks[point] for point in points)
    if sorted(result) != list(range(order)):
        raise AssertionError("Sigma_N does not define a permutation")
    return result


# Exact four-pattern limits and the centred ten-pattern dual.


def core_four_vector(
    aspect: Fraction = ASPECT,
    eta: Fraction = ETA,
) -> dict[tuple[int, ...], Fraction]:
    """Return ``h^eta_{4,tau}(aspect)/96`` for all 24 patterns."""

    if not isinstance(aspect, Fraction) or aspect <= 0:
        raise ValueError("aspect must be a positive Fraction")
    if not isinstance(eta, Fraction):
        eta = Fraction(eta)
    return {pattern: corrected_h(pattern, aspect, eta) / 96 for pattern in all_patterns(4)}


def insertion_correction(
    pattern: tuple[int, ...],
    parity: int = 0,
) -> Fraction:
    """Return the exact summed ``K_tau`` correction for one padding list."""

    if len(pattern) != 4 or set(pattern) != set(range(4)):
        raise ValueError("pattern must be a zero-based permutation of four values")
    matrix = projected_padding_moments(parity)
    return sum((matrix[row][pattern[row]] for row in range(4)), Fraction(0)) / 36


def projected_padding_moments(
    parity: int,
) -> tuple[tuple[Fraction, ...], ...]:
    """Return centred moments for the 12- or 17-point design."""

    parity = _check_parity(parity)
    matrix = root_moment_matrix()
    if parity:
        neutral = neutral_five_moment_matrix()
        matrix = tuple(
            tuple(matrix[row][column] + neutral[row][column] for column in range(4))
            for row in range(4)
        )
    return centered_moment_matrix(matrix)


def interior_limiting_four_vector(
    aspect: Fraction = ASPECT,
    parity: int = 0,
) -> dict[tuple[int, ...], Fraction]:
    """Return the exact limiting vector after the 12/17-point insertion."""

    _check_parity(parity)
    core = core_four_vector(aspect, ETA)
    return {
        pattern: value + insertion_correction(pattern, parity) for pattern, value in core.items()
    }


def maximum_absolute(
    values: Mapping[tuple[int, ...], Fraction],
) -> tuple[Fraction, tuple[tuple[int, ...], ...]]:
    """Return the exact maximum absolute value and all attaining patterns."""

    maximum = max(map(abs, values.values()))
    patterns = tuple(pattern for pattern, value in values.items() if abs(value) == maximum)
    return maximum, patterns


def grouped_limiting_table() -> dict[Fraction, tuple[tuple[int, ...], ...]]:
    """Group the active limiting vector by exact value."""

    groups: dict[Fraction, list[tuple[int, ...]]] = {}
    for pattern, value in interior_limiting_four_vector().items():
        groups.setdefault(value, []).append(pattern)
    return {value: tuple(patterns) for value, patterns in groups.items()}


def _contrast_pattern(text: str) -> tuple[int, ...]:
    return tuple(int(value) - 1 for value in text)


def centered_dual_value(
    values: Mapping[tuple[int, ...], Fraction] | None = None,
) -> Fraction:
    """Evaluate the active centred ten-pattern dual on a limiting vector."""

    vector = interior_limiting_four_vector() if values is None else values
    return sum(
        (weight * vector[_contrast_pattern(name)] for name, weight in DUAL_WEIGHTS.items()),
        Fraction(0),
    )


def interior_lower_bound() -> Fraction:
    """Return the active dual lower bound ``1/192``."""

    norm = sum(abs(weight) for weight in DUAL_WEIGHTS.values())
    return abs(centered_dual_value()) / norm


def validate_moment_certificate() -> None:
    """Validate roots, moments, neutral parity, ledger, and the centred dual."""

    if root_deviation_matrix() != ROOT_DEVIATION:
        raise AssertionError("Newton moments do not reproduce the recorded D matrix")
    if projected_root_moments() != INTERIOR_TARGET_MOMENTS:
        raise AssertionError("the root list does not give C/64 after projection")
    if g2_bernstein_sums() != (Fraction(1, 2),) * 4:
        raise AssertionError("G2 is not neutral in cubic Bernstein moments")
    if g_bernstein_sums() != (Fraction(3, 4),) * 4:
        raise AssertionError("G3 is not neutral in cubic Bernstein moments")
    if projected_neutral_moments() != tuple((Fraction(0),) * 4 for _ in range(4)):
        raise AssertionError("the five-point G2/G3 tail is not neutral")
    if projected_padding_moments(0) != INTERIOR_TARGET_MOMENTS:
        raise AssertionError("the even 12-point list has the wrong projection")
    if projected_padding_moments(1) != INTERIOR_TARGET_MOMENTS:
        raise AssertionError("the odd 17-point list has the wrong projection")
    if sum((sum(row, Fraction(0)) for row in root_moment_matrix()), Fraction(0)) != 12:
        raise AssertionError("the root list does not have mass 12")
    if sum((sum(row, Fraction(0)) for row in neutral_five_moment_matrix()), Fraction(0)) != 5:
        raise AssertionError("the neutral tail does not have mass 5")
    maximum, _ = maximum_absolute(interior_limiting_four_vector())
    if maximum != Fraction(1, 192):
        raise AssertionError("the 24-pattern oriented ledger does not have maximum 1/192")
    if interior_lower_bound() != Fraction(1, 192):
        raise AssertionError("the centred ten-pattern dual does not certify 1/192")
    validate_period8()
    validate_centered_dual()
    validate_oriented_ledger()
