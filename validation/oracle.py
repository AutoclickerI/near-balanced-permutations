"""Independent geometric oracle used only by tests.

This module deliberately does not import the production construction. It derives the
partial row from prefix-count jumps and realizes the perturbation with integer
coordinates whose dominant coefficient preserves all untied grid orders.
"""

from __future__ import annotations

from math import isqrt


def geometric_oracle(order: int) -> tuple[int, ...]:
    """Return the same permutation through scaled opposite geometric rotations."""

    if order <= 0:
        raise ValueError("order must be positive")
    if order == 1:
        return (0,)

    even_order = order if order % 2 == 0 else order - 1
    cells_count = even_order // 2
    rows = isqrt(cells_count)
    columns = cells_count // rows
    remainder = cells_count - rows * columns

    # z enters R exactly when floor(r*z/b) increases. This is independent of
    # the ceil(j*b/r) implementation in construction.py.
    partial = tuple(
        z
        for z in range(1, columns + 1)
        if (remainder * z) // columns > (remainder * (z - 1)) // columns
    )
    cells = [(x, y) for x in range(1, rows + 1) for y in range(1, columns + 1)]
    cells.extend((rows + 1, y) for y in partial)
    points = [(x, y, colour) for x, y in cells for colour in (-1, 1)]

    # D exceeds every possible cross-colour perturbation. Sorting
    # (D*x-c*y, D*y+c*x) therefore realizes a sufficiently small pair of
    # opposite rotations without floating point arithmetic.
    dominant = 4 * max(rows + 1, columns) + 1
    x_order = sorted(points, key=lambda p: (dominant * p[0] - p[2] * p[1], p))
    y_order = sorted(points, key=lambda p: (dominant * p[1] + p[2] * p[0], p))

    if order % 2:
        pad = (rows + 2, columns + 1, 0)
        x_order.append(pad)
        y_order.append(pad)

    ranks = {point: index for index, point in enumerate(y_order)}
    result = tuple(ranks[point] for point in x_order)
    if len(result) != order or sorted(result) != list(range(order)):
        raise AssertionError("independent geometric oracle is not a permutation")
    return result
