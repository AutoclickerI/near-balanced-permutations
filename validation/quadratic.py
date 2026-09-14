"""Small exact arithmetic for the quadratic field ``Q(sqrt(3))``.

The validation runtime needs only a+b*sqrt(3), with rational a and b.  This
module deliberately has no numerical approximation or optional dependency.
Signs are decided by comparing rational squares, so primal inequalities and
``max`` checks remain exact at the algebraic aspect.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any


def _rational(value: Any) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError(f"expected an int or Fraction, got {value!r}")
    return Fraction(value)


def _coerce(value: Any) -> "Qsqrt3":
    if isinstance(value, Qsqrt3):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError(f"expected an exact rational or Qsqrt3, got {value!r}")
    return Qsqrt3(Fraction(value), Fraction(0))


@dataclass(frozen=True, eq=False)
class Qsqrt3:
    """An exact value ``a + b*sqrt(3)`` with ``a,b`` rational."""

    a: Fraction
    b: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "a", _rational(self.a))
        object.__setattr__(self, "b", _rational(self.b))

    @classmethod
    def zero(cls) -> "Qsqrt3":
        return cls(Fraction(0), Fraction(0))

    @classmethod
    def one(cls) -> "Qsqrt3":
        return cls(Fraction(1), Fraction(0))

    @property
    def is_rational(self) -> bool:
        return self.b == 0

    def sign(self) -> int:
        """Return the exact sign using only rational arithmetic.

        When the rational and irrational parts have opposite signs, compare
        ``a**2`` with ``3*b**2``.  This is the only nontrivial comparison and
        avoids converting either the value or ``sqrt(3)`` to a float.
        """

        if self.a == 0:
            return (self.b > 0) - (self.b < 0)
        if self.b == 0:
            return (self.a > 0) - (self.a < 0)
        if self.a > 0 and self.b > 0:
            return 1
        if self.a < 0 and self.b < 0:
            return -1
        square_difference = self.a * self.a - 3 * self.b * self.b
        if square_difference == 0:
            return 0
        if self.a > 0:
            return 1 if square_difference > 0 else -1
        return -1 if square_difference > 0 else 1

    def reciprocal(self) -> "Qsqrt3":
        denominator = self.a * self.a - 3 * self.b * self.b
        if denominator == 0:
            raise ZeroDivisionError("zero has no reciprocal in Q(sqrt(3))")
        return Qsqrt3(self.a / denominator, -self.b / denominator)

    def __add__(self, other: Any) -> "Qsqrt3":
        value = _coerce(other)
        return Qsqrt3(self.a + value.a, self.b + value.b)

    def __radd__(self, other: Any) -> "Qsqrt3":
        return self + other

    def __sub__(self, other: Any) -> "Qsqrt3":
        value = _coerce(other)
        return Qsqrt3(self.a - value.a, self.b - value.b)

    def __rsub__(self, other: Any) -> "Qsqrt3":
        return _coerce(other) - self

    def __neg__(self) -> "Qsqrt3":
        return Qsqrt3(-self.a, -self.b)

    def __mul__(self, other: Any) -> "Qsqrt3":
        value = _coerce(other)
        return Qsqrt3(
            self.a * value.a + 3 * self.b * value.b,
            self.a * value.b + self.b * value.a,
        )

    def __rmul__(self, other: Any) -> "Qsqrt3":
        return self * other

    def __truediv__(self, other: Any) -> "Qsqrt3":
        return self * _coerce(other).reciprocal()

    def __rtruediv__(self, other: Any) -> "Qsqrt3":
        return _coerce(other) * self.reciprocal()

    def __abs__(self) -> "Qsqrt3":
        return self if self.sign() >= 0 else -self

    def __bool__(self) -> bool:
        return self.sign() != 0

    def _compare(self, other: Any) -> int:
        difference = self - other
        return difference.sign()

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Qsqrt3):
            return self.a == other.a and self.b == other.b
        if isinstance(other, bool) or not isinstance(other, (int, Fraction)):
            return False
        return self.b == 0 and self.a == Fraction(other)

    def __hash__(self) -> int:
        # Match Fraction/int hashing for rational field elements because __eq__
        # intentionally treats Qsqrt3(a, 0) as equal to the corresponding
        # Python rational.
        return hash(self.a) if self.b == 0 else hash((self.a, self.b))

    def __lt__(self, other: Any) -> bool:
        return self._compare(other) < 0

    def __le__(self, other: Any) -> bool:
        return self._compare(other) <= 0

    def __gt__(self, other: Any) -> bool:
        return self._compare(other) > 0

    def __ge__(self, other: Any) -> bool:
        return self._compare(other) >= 0

    def __str__(self) -> str:
        if self.b == 0:
            return str(self.a)
        irrational = "sqrt(3)" if abs(self.b) == 1 else f"{abs(self.b)}*sqrt(3)"
        if self.a == 0:
            return irrational if self.b > 0 else f"-{irrational}"
        sign = "+" if self.b > 0 else "-"
        return f"{self.a}{sign}{irrational}"

    def __repr__(self) -> str:
        return f"Qsqrt3(a={self.a!r}, b={self.b!r})"
