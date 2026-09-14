"""Exact construction of the near-balanced permutation at every order."""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt


@dataclass(frozen=True)
class Point:
    """A coloured grid point; colour 0 is reserved for odd-order padding."""

    x: int
    y: int
    colour: int


@dataclass(frozen=True)
class EvenShape:
    cells: int
    rows: int
    columns: int
    remainder: int
    partial_columns: tuple[int, ...]


def balanced_partial_columns(columns: int, remainder: int) -> tuple[int, ...]:
    """Return R={ceil(j*columns/remainder): 1<=j<=remainder}."""

    if remainder < 0 or remainder >= columns:
        raise ValueError("remainder must lie in [0, columns)")
    if remainder == 0:
        return ()
    result = tuple((j * columns + remainder - 1) // remainder for j in range(1, remainder + 1))
    if len(set(result)) != remainder:
        raise AssertionError("balanced columns must be distinct")
    return result


def even_shape(order: int) -> EvenShape:
    """Return the near-square complete core and balanced partial row for even order."""

    if order <= 0 or order % 2:
        raise ValueError("order must be a positive even integer")
    cells = order // 2
    rows = isqrt(cells)
    columns = cells // rows
    remainder = cells - rows * columns
    partial = balanced_partial_columns(columns, remainder)
    if not (0 <= remainder < rows and rows <= columns <= rows + 2):
        raise AssertionError("near-square shape invariant failed")
    return EvenShape(cells, rows, columns, remainder, partial)


def even_points(order: int) -> tuple[list[Point], EvenShape]:
    """Return points in exact X-order and shape metadata."""

    shape = even_shape(order)
    cells = [(x, y) for x in range(1, shape.rows + 1) for y in range(1, shape.columns + 1)]
    cells.extend((shape.rows + 1, y) for y in shape.partial_columns)
    points = [Point(x, y, colour) for x, y in cells for colour in (-1, 1)]
    x_order = sorted(points, key=lambda p: (p.x, -p.colour * p.y, p.colour))
    if len(x_order) != order or len(set(x_order)) != order:
        raise AssertionError("point construction has the wrong cardinality")
    return x_order, shape


def point_orders(order: int) -> tuple[list[Point], dict[Point, int], EvenShape | None]:
    """Return exact X-order, zero-based Y-ranks, and even-core shape.

    For odd order, the final point is rightmost and highest; deleting it recovers
    the even construction of order ``order-1``.
    """

    if order <= 0:
        raise ValueError("order must be positive")
    if order == 1:
        point = Point(1, 1, 0)
        return [point], {point: 0}, None

    even_order = order if order % 2 == 0 else order - 1
    x_order, shape = even_points(even_order)
    y_order = sorted(x_order, key=lambda p: (p.y, p.colour * p.x, p.colour))

    if order % 2:
        pad = Point(shape.rows + 2, shape.columns + 1, 0)
        x_order.append(pad)
        y_order.append(pad)

    ranks = {point: index for index, point in enumerate(y_order)}
    if len(ranks) != order or sorted(ranks.values()) != list(range(order)):
        raise AssertionError("Y-order does not define a permutation")
    return x_order, ranks, shape


def permutation(order: int) -> tuple[int, ...]:
    """Return the one-line permutation in zero-based values."""

    points, ranks, _ = point_orders(order)
    result = tuple(ranks[point] for point in points)
    if sorted(result) != list(range(order)):
        raise AssertionError("constructed values are not a permutation")
    return result
