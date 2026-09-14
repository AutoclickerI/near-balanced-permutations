"""Independent signed-grid verifier that does not import the profile formulas."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class Point:
    x: int
    y: int
    colour: int


def points_for(
    side: int,
    colours: tuple[int, ...] = (-1, 1),
    x_tie_mode: str = "signed",
    y_tie_mode: str = "signed",
) -> tuple[list[Point], dict[Point, int]]:
    """Return complete-square points in exact X-order and their Y-ranks."""

    points = [
        Point(x, y, colour)
        for x in range(1, side + 1)
        for y in range(1, side + 1)
        for colour in colours
    ]

    def x_key(point: Point) -> tuple[int, int, int]:
        tie = -point.colour * point.y if x_tie_mode == "signed" else point.y
        return point.x, tie, point.colour

    def y_key(point: Point) -> tuple[int, int, int]:
        tie = point.colour * point.x if y_tie_mode == "signed" else point.x
        return point.y, tie, point.colour

    x_order = sorted(points, key=x_key)
    y_order = sorted(points, key=y_key)
    return x_order, {point: index + 1 for index, point in enumerate(y_order)}


def build_permutation(
    side: int,
    *,
    colours: tuple[int, ...] = (-1, 1),
    x_tie_mode: str = "signed",
    y_tie_mode: str = "signed",
) -> tuple[list[Point], list[int]]:
    points, ranks = points_for(
        side,
        colours=colours,
        x_tie_mode=x_tie_mode,
        y_tie_mode=y_tie_mode,
    )
    return points, [ranks[point] for point in points]


def pattern(values: list[int]) -> tuple[int, ...]:
    rank = {value: index + 1 for index, value in enumerate(sorted(values))}
    return tuple(rank[value] for value in values)


def enumerate_profiles(
    side: int,
    k: int,
    *,
    colours: tuple[int, ...] = (-1, 1),
    x_tie_mode: str = "signed",
    y_tie_mode: str = "signed",
) -> dict[int, Counter[tuple[int, ...]]]:
    points, ranks = points_for(
        side,
        colours=colours,
        x_tie_mode=x_tie_mode,
        y_tie_mode=y_tie_mode,
    )
    profiles: dict[int, Counter[tuple[int, ...]]] = {
        0: Counter(),
        1: Counter(),
        2: Counter(),
    }
    for chosen in combinations(points, k):
        x_count = len({point.x for point in chosen})
        y_count = len({point.y for point in chosen})
        defect = 2 * k - x_count - y_count
        induced = pattern([ranks[point] for point in chosen])
        profiles[min(defect, 2)][induced] += 1
    return profiles


def mutation_variants() -> dict[str, dict[str, object]]:
    return {
        "remove_one_colour": {"colours": (1,)},
        "wrong_x_tie_sign": {"x_tie_mode": "same"},
        "wrong_y_tie_sign": {"y_tie_mode": "same"},
    }
