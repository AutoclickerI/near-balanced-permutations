from __future__ import annotations

import unittest
from collections import Counter
from fractions import Fraction
from itertools import combinations, permutations, product
from math import comb, factorial, isqrt
from typing import Any, cast
from unittest.mock import patch

from validation.construction import (
    balanced_partial_columns,
    even_shape,
    permutation,
    point_orders,
)
from validation.independent import (
    build_permutation,
    enumerate_profiles,
    mutation_variants,
)
from validation.oracle import geometric_oracle
from validation.oriented_four import DUAL_WEIGHTS
from validation.profiles import (
    _local_statistics,
    all_patterns,
    balanced_prefix_errors,
    classify_even_subsets,
    clean_partial_formula,
    compositions,
    direct_profile,
    exact_discrepancy,
    merge_profiles,
    rectangular_normalized_leading_coefficients,
    rectangular_profile_formula,
    square_leading_coefficients,
    square_leading_discrepancy,
    square_local_leading_coefficients,
    square_profile_formula,
    standardise,
    tie_colour_count,
    vandermonde_sums,
)
from validation.sharp_four import (
    ASPECT,
    COMMON_ROOT_INTERVALS,
    CUBIC_POLYNOMIALS,
    G2,
    G2_POLYNOMIAL,
    G3,
    G3_POLYNOMIAL,
    INTERIOR_CORRECTION_UNITS,
    INTERIOR_TARGET_MOMENTS,
    PADDING_COUNTS,
    PADDING_DELTA,
    ROOT_DEVIATION,
    ROOT_ENDPOINT_SIGNS,
    ROOT_ISOLATING_INTERVALS,
    ROW_SLOPE,
    IsolatedRoot,
    SharpDimensions,
    algebraic_floor,
    boundary_quadrature,
    centered_dual_value,
    circular_distance,
    dimension_search_base,
    dimension_search_window,
    even_padding_pairs,
    g2_bernstein_sums,
    g_bernstein_sums,
    grouped_limiting_table,
    insertion_correction,
    interior_limiting_four_vector,
    interior_lower_bound,
    maximum_absolute,
    neutral_five_moment_matrix,
    newton_power_sums,
    odd_padding_pairs,
    padding_mass,
    padding_pairs,
    padding_point_at,
    projected_neutral_moments,
    projected_padding_moments,
    projected_root_moments,
    root_deviation_matrix,
    root_group,
    root_moment_matrix,
    sharp_dimensions,
    sharp_family_spec,
    sharp_permutation,
    sharp_point_orders,
    shifted_partial_columns,
    shifted_prefix_errors,
    validate_moment_certificate,
    validate_padding_lists,
    validate_root_certificates,
)


def comb0(n: int, k: int) -> int:
    return 0 if n < 0 or k < 0 or k > n else comb(n, k)


def expected_core_good_per_pattern(order: int, k: int) -> int:
    shape = even_shape(order)
    rows, columns = shape.rows, shape.columns
    defect_zero = (2**k) * comb0(rows, k) * comb0(columns, k)
    repeated_x = rows * comb0(rows - 1, k - 2) * comb0(columns, k)
    repeated_y = columns * comb0(columns - 1, k - 2) * comb0(rows, k)
    return defect_zero + (2 ** (k - 1)) * (repeated_x + repeated_y)


class EveryOrderConstructionTests(unittest.TestCase):
    def test_nearby_grid_embedding_and_count_envelope(self) -> None:
        for order in range(2, 129):
            points, _, shape = point_orders(order)
            assert shape is not None
            a, b = shape.rows, shape.columns
            upper = {(x, y, c) for x in range(1, a + 3) for y in range(1, b + 2) for c in (-1, 1)}
            selected = {(p.x, p.y, p.colour or 1) for p in points}
            core = {(x, y, c) for x in range(1, a + 1) for y in range(1, b + 1) for c in (-1, 1)}
            self.assertTrue(core <= selected <= upper)
            xs = [
                p for p in sorted(upper, key=lambda p: (p[0], -p[2] * p[1], p[2])) if p in selected
            ]
            ys = [
                p for p in sorted(upper, key=lambda p: (p[1], p[2] * p[0], p[2])) if p in selected
            ]
            ranks = {p: j for j, p in enumerate(ys)}
            self.assertEqual(tuple(ranks[p] for p in xs), permutation(order))
            if order <= 20:
                for k in range(2, min(4, order) + 1):
                    lower = 2**k * comb0(a, k) * comb0(b, k)
                    upper_count = 2**k * comb(a + 2 + k - 1, k) * comb(b + 1 + k - 1, k)
                    counts = direct_profile(order, k).values()
                    self.assertTrue(all(lower <= count <= upper_count for count in counts))

    def test_ceiling_rank_weight_has_one_sided_bound(self) -> None:
        for b in range(2, 19):
            for r in range(b):
                selected = balanced_partial_columns(b, r)
                for k in range(2, min(6, b) + 1):
                    for t in range(1, k + 1):
                        total = sum(comb0(y - 1, t - 1) * comb0(b - y, k - t) for y in selected)
                        error = Fraction(total) - Fraction(r, b) * comb(b, k)
                        self.assertLessEqual(abs(error), comb(b - 1, k - 1))

    def test_tie_colours_are_uniform_at_zero_and_negative_coordinates(self) -> None:
        for axis_sign in (-1, 1):
            for first in range(-2, 3):
                for second in range(-2, 3):
                    if first == second:
                        continue
                    keys = [
                        ((0, axis_sign * c * first, c), (0, axis_sign * d * second, d))
                        for c in (-1, 1)
                        for d in (-1, 1)
                    ]
                    self.assertTrue(all(left != right for left, right in keys))
                    self.assertEqual(sum(left < right for left, right in keys), 2)

    def test_independent_geometric_oracle_matches_small_orders(self) -> None:
        for order in range(1, 101):
            with self.subTest(order=order):
                self.assertEqual(geometric_oracle(order), permutation(order))

    def test_every_small_order_is_a_permutation_and_odd_padding_is_northeast(self) -> None:
        for order in range(1, 81):
            with self.subTest(order=order):
                values = permutation(order)
                self.assertEqual(sorted(values), list(range(order)))
                if order > 1 and order % 2:
                    self.assertEqual(values[:-1], permutation(order - 1))
                    self.assertEqual(values[-1], order - 1)

    def test_near_square_shape_and_exact_size(self) -> None:
        for order in range(2, 1002, 2):
            with self.subTest(order=order):
                points, _, shape = point_orders(order)
                assert shape is not None
                self.assertEqual(len(points), order)
                self.assertEqual(
                    shape.rows * shape.columns + shape.remainder,
                    order // 2,
                )
                self.assertLess(shape.remainder, shape.rows)
                self.assertLessEqual(shape.columns, shape.rows + 2)

    def test_balanced_partial_row_has_claimed_prefix_counts(self) -> None:
        for cells in range(1, 1001):
            shape = even_shape(2 * cells)
            self.assertEqual(
                shape.partial_columns,
                balanced_partial_columns(shape.columns, shape.remainder),
            )
            errors = balanced_prefix_errors(shape)
            self.assertEqual(errors[0], 0)
            self.assertEqual(errors[-1], 0)
            self.assertTrue(all(Fraction(-1) < error <= 0 for error in errors))
            for endpoint in range(shape.columns + 1):
                expected = (shape.remainder * endpoint) // shape.columns
                observed = sum(y <= endpoint for y in shape.partial_columns)
                self.assertEqual(observed, expected)

    def test_vandermonde_identity_is_independent_of_final_rank(self) -> None:
        for columns in range(1, 16):
            for k in range(1, 8):
                with self.subTest(columns=columns, k=k):
                    self.assertEqual(set(vandermonde_sums(columns, k)), {comb(columns, k)})


class ExactProfileTests(unittest.TestCase):
    def test_tie_colour_count_rejects_nonpositive_compositions(self) -> None:
        self.assertEqual(tie_colour_count((0, 1), (2,), (1, 1)), 2)
        for invalid in ((0, 2), (3, -1), (True, 1), (1.0, 1), (1,)):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    tie_colour_count((0, 1), invalid, (2,))  # type: ignore[arg-type]

    def test_rectangular_formula_matches_direct_profiles(self) -> None:
        cases = (
            (1, 3, 2),
            (1, 3, 3),
            (1, 4, 4),
            (2, 3, 2),
            (2, 3, 3),
            (2, 4, 4),
            (3, 4, 5),
        )
        for rows, columns, k in cases:
            with self.subTest(rows=rows, columns=columns, k=k):
                points = tuple(
                    (x, y, colour)
                    for x in range(1, rows + 1)
                    for y in range(1, columns + 1)
                    for colour in (-1, 1)
                )
                x_order = sorted(
                    points, key=lambda point: (point[0], -point[2] * point[1], point[2])
                )
                y_order = sorted(
                    points, key=lambda point: (point[1], point[2] * point[0], point[2])
                )
                y_ranks = {point: rank for rank, point in enumerate(y_order)}
                direct = Counter({pattern: 0 for pattern in all_patterns(k)})
                for chosen in combinations(x_order, k):
                    direct[standardise(tuple(y_ranks[point] for point in chosen))] += 1
                formula = rectangular_profile_formula(rows, columns, k)
                self.assertEqual(formula, direct)
                self.assertEqual(sum(formula.values()), comb(len(points), k))

    def test_proof_classes_partition_the_direct_profile(self) -> None:
        cases = [(order, k) for order in range(4, 25, 2) for k in range(2, min(4, order) + 1)]
        cases += [(12, 5), (16, 5), (20, 5)]
        for order, k in cases:
            with self.subTest(order=order, k=k):
                direct = direct_profile(order, k)
                classes = classify_even_subsets(order, k)
                self.assertEqual(merge_profiles(classes.values()), direct)
                self.assertEqual(sum(direct.values()), comb(order, k))

    def test_complete_core_defect_zero_and_one_are_uniform(self) -> None:
        cases = [(order, k) for order in range(8, 27, 2) for k in range(2, min(5, order) + 1)]
        for order, k in cases:
            with self.subTest(order=order, k=k):
                good = classify_even_subsets(order, k)["core_good"]
                expected = expected_core_good_per_pattern(order, k)
                self.assertEqual(set(good), set(all_patterns(k)))
                self.assertEqual(set(good.values()), {expected})

    def test_clean_partial_class_matches_formula_pattern_by_pattern(self) -> None:
        cases = [(order, k) for order in range(4, 31, 2) for k in range(2, min(4, order) + 1)]
        cases += [(42, 5), (44, 5)]
        for order, k in cases:
            with self.subTest(order=order, k=k):
                direct = classify_even_subsets(order, k)["partial_one_clean"]
                self.assertEqual(direct, clean_partial_formula(order, k))

    def test_full_partial_row_composition_formula_matches_direct_profile(self) -> None:
        """Check the exact grouped-colour identity behind Proposition 8.2."""

        order = 28
        k = 4
        points, ranks, shape = point_orders(order)
        assert shape is not None
        self.assertEqual((shape.rows, shape.columns, shape.remainder), (3, 4, 2))
        partial_x = shape.rows + 1
        partial_columns = set(shape.partial_columns)

        direct: Counter[tuple[int, ...]] = Counter({pattern: 0 for pattern in all_patterns(k)})
        for indices in combinations(range(order), k):
            chosen = [points[index] for index in indices]
            if any(point.x == partial_x for point in chosen):
                pattern = standardise([ranks[point] for point in chosen])
                direct[pattern] += 1

        parts = compositions(k)
        formula: Counter[tuple[int, ...]] = Counter()
        for pattern in all_patterns(k):
            total = 0
            for alpha in parts:
                p = len(alpha)
                choose_x = comb0(shape.rows, p - 1)
                if not choose_x:
                    continue
                last_x_positions = range(k - alpha[-1], k)
                for beta in parts:
                    q = len(beta)
                    if q > shape.columns:
                        continue
                    beta_blocks = tuple(
                        block
                        for block, multiplicity in enumerate(beta)
                        for _ in range(multiplicity)
                    )
                    required_ranks = {beta_blocks[pattern[index]] for index in last_x_positions}
                    selector_sum = sum(
                        all(ordinates[rank] in partial_columns for rank in required_ranks)
                        for ordinates in combinations(range(1, shape.columns + 1), q)
                    )
                    total += choose_x * tie_colour_count(pattern, alpha, beta) * selector_sum
            formula[pattern] = total

        self.assertEqual(direct, formula)

    def test_odd_padding_count_change_is_bounded_pattern_by_pattern(self) -> None:
        for odd_order, k in ((9, 3), (11, 4), (15, 4), (17, 5)):
            with self.subTest(order=odd_order, k=k):
                old = direct_profile(odd_order - 1, k)
                new = direct_profile(odd_order, k)
                bound = comb(odd_order - 1, k - 1)
                self.assertEqual(sum(new.values()) - sum(old.values()), bound)
                for pattern_value in all_patterns(k):
                    difference = new[pattern_value] - old[pattern_value]
                    self.assertGreaterEqual(difference, 0)
                    self.assertLessEqual(difference, bound)
                    if pattern_value[-1] != k - 1:
                        self.assertEqual(difference, 0)

    def test_same_axis_disjoint_collision_pairs_are_uniform(self) -> None:
        order, k = 32, 4
        points, ranks, _ = point_orders(order)
        layers = [Counter({pattern: 0 for pattern in all_patterns(k)}) for _ in range(2)]
        for chosen in combinations(points, k):
            x_multiplicities = Counter(point.x for point in chosen)
            y_multiplicities = Counter(point.y for point in chosen)
            induced = standardise(tuple(ranks[point] for point in chosen))
            if (
                len(y_multiplicities) == k
                and max(x_multiplicities.values()) <= 2
                and len(x_multiplicities) < k - 1
            ):
                layers[0][induced] += 1
            if (
                len(x_multiplicities) == k
                and max(y_multiplicities.values()) <= 2
                and len(y_multiplicities) < k - 1
            ):
                layers[1][induced] += 1
        for layer in layers:
            self.assertGreater(sum(layer.values()), 0)
            self.assertEqual(len(set(layer.values())), 1)

    def test_complete_row_leading_defect_one_subclasses_are_uniform(self) -> None:
        """Mirror the grouped classes used in the repaired boundary-leading proof."""

        order, k = 32, 4
        points, ranks, shape = point_orders(order)
        assert shape is not None and shape.remainder == 0
        last_x = shape.rows
        groups = {
            ("x_collision", 1): Counter({pattern: 0 for pattern in all_patterns(k)}),
            ("x_collision", 2): Counter({pattern: 0 for pattern in all_patterns(k)}),
            ("y_collision", 1): Counter({pattern: 0 for pattern in all_patterns(k)}),
        }
        for chosen in combinations(points, k):
            if not any(point.x == last_x for point in chosen):
                continue
            x_counts = Counter(point.x for point in chosen)
            y_counts = Counter(point.y for point in chosen)
            key = None
            if len(x_counts) == k - 1 and len(y_counts) == k:
                key = ("x_collision", x_counts[last_x])
            elif len(x_counts) == k and len(y_counts) == k - 1:
                key = ("y_collision", 1)
            if key is not None:
                pattern = standardise(tuple(ranks[point] for point in chosen))
                groups[key][pattern] += 1

        for key, profile in groups.items():
            with self.subTest(key=key):
                self.assertGreater(sum(profile.values()), 0)
                self.assertEqual(len(set(profile.values())), 1)

    def test_exact_discrepancy_includes_zero_patterns(self) -> None:
        patterns = all_patterns(3)
        sparse = Counter({pattern: 4 for pattern in patterns[:-1]})
        self.assertEqual(exact_discrepancy(sparse, 6, 3), Fraction(10, 3))

    def test_square_profile_formula_matches_direct_profiles(self) -> None:
        for side, k in ((2, 2), (3, 3), (4, 4), (5, 4), (3, 5)):
            with self.subTest(side=side, k=k):
                formula = square_profile_formula(side, k)
                self.assertEqual(formula, direct_profile(2 * side * side, k))


class LeadingTermTests(unittest.TestCase):
    def test_square_leading_coefficients_have_known_values(self) -> None:
        expected = {
            2: Fraction(1, 2),
            3: Fraction(4, 9),
            4: Fraction(5, 36),
            5: Fraction(1, 40),
            6: Fraction(43, 16200),
        }
        for k, coefficient in expected.items():
            with self.subTest(k=k):
                leading = square_leading_coefficients(k)
                self.assertEqual(sum(leading.values(), Fraction(0)), 0)
                self.assertEqual(square_leading_discrepancy(k), coefficient)

    def test_square_growth_witness_and_upper_bracket(self) -> None:
        def witness(m: int, q: int) -> tuple[int, ...]:
            values = [6 * m - 3 - 6 * j for j in range(m)]
            values += [8 + 6 * j for j in range(m - 1)] + [6 * m - 2]
            values += [6 * m - 8 - 6 * j for j in range(m - 1)] + [2]
            values += [7 + 6 * j for j in range(m - 1)] + [6 * m - 1]
            values += [6 * m - 7 - 6 * j for j in range(m - 1)] + [1]
            values += [6 + 6 * j for j in range(m - 1)]
            values += list(range(6 * m, 6 * m + q))
            return tuple(value - 1 for value in values)

        for m in range(2, 21):
            for q in range(6):
                pattern = witness(m, q)
                k = 6 * m - 1 + q
                with self.subTest(m=m, q=q):
                    self.assertEqual(sorted(pattern), list(range(k)))
                    t_x, t_y, peaks_x, peaks_y, interaction = _local_statistics(pattern)
                    self.assertEqual(
                        (t_x, t_y, peaks_x, peaks_y, interaction),
                        (k - 4, k - m - 1, 2, m - 1, -4 * (m - 1) + q),
                    )
                    g_value = t_x + t_y + Fraction(11 - 4 * k, 3) - Fraction(k, k - 1) * interaction
                    expected = (
                        Fraction(k + 1, 3)
                        + m
                        - 2
                        + Fraction(q, 3)
                        + Fraction(k * (4 * m - 4 - q), k - 1)
                    )
                    self.assertEqual(g_value, expected)
                    self.assertGreater(g_value, Fraction(k + 1, 3))
                    self.assertLessEqual(g_value, Fraction(5 * k - 1, 3))

    def test_local_defect_two_formula_matches_composition_formula(self) -> None:
        for k in range(2, 7):
            with self.subTest(k=k):
                self.assertEqual(
                    square_local_leading_coefficients(k),
                    square_leading_coefficients(k),
                )

    def test_mixed_statistic_expectation(self) -> None:
        for k in range(2, 9):
            with self.subTest(k=k):
                total = sum(_local_statistics(pattern)[4] for pattern in all_patterns(k))
                self.assertEqual(Fraction(total, factorial(k)), Fraction(k - 1, k))

    def test_square_aspect_is_minimal_from_order_five(self) -> None:
        aspects = (
            Fraction(1, 4),
            Fraction(1, 2),
            Fraction(3, 4),
            Fraction(1),
            Fraction(4, 3),
            Fraction(2),
            Fraction(4),
        )
        for k in range(5, 10):
            square = max(
                abs(value)
                for value in rectangular_normalized_leading_coefficients(k, Fraction(1)).values()
            )
            for aspect in aspects:
                with self.subTest(k=k, aspect=aspect):
                    observed = max(
                        abs(value)
                        for value in rectangular_normalized_leading_coefficients(k, aspect).values()
                    )
                    self.assertGreaterEqual(observed, square)

    def test_four_point_nonsquare_aspect_strictly_improves_square(self) -> None:
        square = max(
            abs(value)
            for value in rectangular_normalized_leading_coefficients(4, Fraction(1)).values()
        )
        nonsquare = max(
            abs(value)
            for value in rectangular_normalized_leading_coefficients(4, Fraction(3, 4)).values()
        )
        self.assertLess(nonsquare, square)

    def test_square_four_exact_profile_classes_and_discrepancy(self) -> None:
        identity = tuple(range(4))
        for side in range(2, 11):
            with self.subTest(side=side):
                profile = square_profile_formula(side, 4)
                order = 2 * side * side
                target = Fraction(comb(order, 4), factorial(4))
                expected_identity = Fraction(
                    5 * side**2 * (side**2 - 1) * (4 * side**2 - 9),
                    144,
                )
                self.assertEqual(abs(Fraction(profile[identity]) - target), expected_identity)
                if side >= 4:
                    self.assertEqual(exact_discrepancy(profile, order, 4), expected_identity)

    def test_global_fourteen_thirds_normalization_inequalities(self) -> None:
        target = Fraction(14, 3)
        for k in range(2, 10001):
            complete_core_branch = Fraction(
                14 * k**3 - 22 * k**2 + k + 16,
                3 * k**3,
            )
            empty_core_branch = Fraction(2 * k**2 - 3 * k + 4, k**2)
            self.assertLess(complete_core_branch, target)
            self.assertLess(empty_core_branch, target)
        for cells in range(1, 1001):
            rows = isqrt(cells)
            columns = cells // rows
            self.assertLessEqual(columns - rows, 2)
            if rows * columns:
                aspect_excess = Fraction((columns - rows) ** 2, rows * columns)
                self.assertLessEqual(aspect_excess, Fraction(4, rows * columns))


class IndependentSignedGridTests(unittest.TestCase):
    def test_square_orders_and_good_layers_at_side_five(self) -> None:
        points, values = build_permutation(5)
        self.assertEqual(len(points), 50)
        self.assertEqual(sorted(values), list(range(1, 51)))
        profiles = enumerate_profiles(5, 5)
        self.assertEqual(set(profiles[0]), set(permutations(range(1, 6))))
        self.assertEqual(set(profiles[1]), set(permutations(range(1, 6))))
        self.assertEqual(set(profiles[0].values()), {32})
        self.assertEqual(set(profiles[1].values()), {640})

    def test_adversarial_mutations_break_defect_one_uniformity(self) -> None:
        normal = enumerate_profiles(4, 4)[1]
        self.assertEqual(len(set(normal.values())), 1)
        for name, config in mutation_variants().items():
            with self.subTest(name=name):
                profiles = enumerate_profiles(4, 4, **cast(Any, config))
                values = list(profiles[1].values())
                self.assertTrue(values)
                self.assertNotEqual(len(set(values)), 1)

    def test_total_discrepancy_is_bounded_by_bad_subsets(self) -> None:
        side, k = 3, 4
        profiles = enumerate_profiles(side, k)
        all_counts: Counter[tuple[int, ...]] = Counter()
        for layer in profiles.values():
            all_counts.update(layer)
        target = Fraction(comb(2 * side * side, k), factorial(k))
        discrepancy = max(
            abs(Fraction(all_counts[pattern]) - target) for pattern in permutations(range(1, k + 1))
        )
        self.assertLessEqual(discrepancy, sum(profiles[2].values()))

    def test_feasible_small_good_layers_are_uniform(self) -> None:
        for side, k in ((2, 2), (2, 3), (2, 4), (3, 2), (3, 3)):
            with self.subTest(side=side, k=k):
                profiles = enumerate_profiles(side, k)
                expected = set(permutations(range(1, k + 1)))
                for defect in (0, 1):
                    values = list(profiles[defect].values())
                    if not values:
                        continue
                    self.assertEqual(set(profiles[defect]), expected)
                    self.assertEqual(len(profiles[defect]), factorial(k))
                    self.assertEqual(len(set(values)), 1)


class SharpFourCompletionTests(unittest.TestCase):
    def test_shifted_selectors_have_exact_size_and_prefix_bound(self) -> None:
        for columns in range(2, 51):
            for remainder in range(1, columns):
                for delta in (Fraction(1, 2), Fraction(3, 10), PADDING_DELTA):
                    with self.subTest(columns=columns, remainder=remainder, delta=delta):
                        selected = shifted_partial_columns(columns, remainder, delta)
                        self.assertEqual(len(selected), remainder)
                        self.assertEqual(len(set(selected)), remainder)
                        self.assertTrue(all(1 <= y <= columns for y in selected))
                        self.assertLessEqual(
                            max(map(abs, shifted_prefix_errors(columns, remainder, delta))),
                            2,
                        )

    def test_boundary_quadrature_matches_manuscript_example(self) -> None:
        self.assertEqual(
            boundary_quadrature(4, 3, 5, 2, Fraction(1, 2)),
            Fraction(1, 125),
        )

    def test_dimension_search_is_exact_and_uses_the_public_window(self) -> None:
        for cells in (48, 49, 50, 100, 1_000, 10_000, 100_000):
            base = dimension_search_base(cells)
            window = dimension_search_window(base)
            self.assertEqual(base, isqrt(cells))
            self.assertLessEqual(base * base, cells)
            self.assertGreater((base + 1) ** 2, cells)
            self.assertTrue((window - 1) ** 3 < base * base <= window**3)
            dimensions = sharp_dimensions(cells)
            self.assertEqual(
                dimensions.rows * dimensions.columns + dimensions.remainder,
                cells,
            )
            self.assertTrue(base <= dimensions.columns <= base + window)
            self.assertEqual(dimensions.search_start, base)
            self.assertEqual(dimensions.search_length, window)
            self.assertTrue(0 <= dimensions.remainder < dimensions.columns)

            candidates = []
            for columns in range(base, base + window + 1):
                rows, remainder = divmod(cells, columns)
                candidates.append((circular_distance(remainder, columns), columns))
            expected_score, expected_columns = candidates[0]
            for score, columns in candidates[1:]:
                if score < expected_score or (
                    score == expected_score and columns < expected_columns
                ):
                    expected_score, expected_columns = score, columns
            self.assertEqual(
                (dimensions.row_distance, dimensions.columns),
                (expected_score, expected_columns),
            )

    def test_root_certificates_and_exact_algebraic_floors(self) -> None:
        validate_root_certificates()
        self.assertEqual(
            CUBIC_POLYNOMIALS,
            (
                (9172942848, -12525207552, 4569969024, -366861277),
                (4194304, -6586368, 2746496, -242611),
                (4194304, -6782976, 3256448, -401701),
                (9172942848, -13273694208, 5415500160, -602145179),
            ),
        )
        expected_floors = ((1, 3, 7), (1, 4, 8), (1, 5, 6), (1, 3, 7))
        for group in range(4):
            roots = root_group(group)
            self.assertEqual(len(roots), 3)
            self.assertEqual(
                tuple(root.endpoint_signs() for root in roots),
                (ROOT_ENDPOINT_SIGNS[0], ROOT_ENDPOINT_SIGNS[1], ROOT_ENDPOINT_SIGNS[2]),
            )
            for root, common, interval in zip(
                roots, COMMON_ROOT_INTERVALS, ROOT_ISOLATING_INTERVALS[group]
            ):
                self.assertTrue(common[0] < interval[0] < interval[1] < common[1])
                self.assertEqual(root.floor_scaled(9), expected_floors[group][roots.index(root)])

        # These are exact comparisons at rational cut points, including a
        # root that is itself rational and lands exactly on a scaled integer.
        rational_half = IsolatedRoot((2, -1), Fraction(0), Fraction(1))
        rational_endpoint = IsolatedRoot((1, -1), Fraction(0), Fraction(1))
        self.assertEqual(rational_half.floor_scaled(2), 1)
        self.assertEqual(rational_half.floor_scaled(4), 2)
        self.assertEqual(rational_endpoint.floor_scaled(1), 1)
        self.assertEqual(tuple(algebraic_floor(value, 9) for value in G3), (1, 4, 7))
        self.assertEqual(G2_POLYNOMIAL, (6, -6, 1))
        self.assertEqual(G3_POLYNOMIAL, (8, -8, 1))

    def test_newton_moments_match_the_recorded_D_and_C_over_64(self) -> None:
        validate_moment_certificate()
        self.assertEqual(
            newton_power_sums((2, -1), 3), (Fraction(1, 2), Fraction(1, 4), Fraction(1, 8))
        )
        self.assertEqual(root_deviation_matrix(), ROOT_DEVIATION)
        self.assertEqual(projected_root_moments(), INTERIOR_TARGET_MOMENTS)
        self.assertEqual(
            projected_root_moments(),
            tuple(tuple(Fraction(value, 64) for value in row) for row in INTERIOR_CORRECTION_UNITS),
        )
        root_raw = root_moment_matrix()
        self.assertEqual(sum((sum(row, Fraction(0)) for row in root_raw), Fraction(0)), 12)

        self.assertEqual(g2_bernstein_sums(), (Fraction(1, 2),) * 4)
        self.assertEqual(g_bernstein_sums(), (Fraction(3, 4),) * 4)
        self.assertEqual(
            neutral_five_moment_matrix(),
            (
                (Fraction(1, 2),) * 4,
                (Fraction(0),) * 4,
                (Fraction(0),) * 4,
                (Fraction(3, 4),) * 4,
            ),
        )
        self.assertEqual(projected_neutral_moments(), tuple((Fraction(0),) * 4 for _ in range(4)))
        self.assertEqual(projected_padding_moments(0), INTERIOR_TARGET_MOMENTS)
        self.assertEqual(projected_padding_moments(1), INTERIOR_TARGET_MOMENTS)

    def test_padding_lists_are_small_ordered_and_explicit(self) -> None:
        validate_padding_lists()
        even = even_padding_pairs()
        odd = odd_padding_pairs()
        self.assertEqual(PADDING_COUNTS, {0: 12, 1: 17})
        self.assertEqual(padding_mass(0), 12)
        self.assertEqual(padding_mass(1), 17)
        self.assertEqual(len(even), 12)
        self.assertEqual(len(odd), 17)
        self.assertEqual(odd[:12], even)
        self.assertEqual(
            odd[12:],
            tuple((Fraction(0), value) for value in G2)
            + tuple((Fraction(1), value) for value in G3),
        )
        self.assertEqual(len(set(even)), 12)
        self.assertEqual(
            tuple(u for u, _ in even),
            tuple(Fraction(group, 3) for group in range(4) for _ in range(3)),
        )

    def test_sigma_uses_the_exact_fallback_and_actual_112_117_branches(self) -> None:
        for order in range(1, 101):
            self.assertEqual(sharp_permutation(order), permutation(order))

        self.assertEqual(ASPECT, Fraction(1))
        self.assertEqual(ROW_SLOPE.rational, 3)
        self.assertEqual(ROW_SLOPE.radical, -1)
        self.assertEqual(PADDING_DELTA, Fraction(1, 2))
        for order, parity in ((112, 0), (117, 1)):
            with self.subTest(order=order):
                spec = sharp_family_spec(order)
                self.assertFalse(spec.fallback)
                self.assertEqual(spec.core_size, 100)
                self.assertEqual(spec.padding_mass, PADDING_COUNTS[parity])
                self.assertIsNotNone(spec.dimensions)
                assert spec.dimensions is not None
                self.assertEqual(
                    (spec.dimensions.rows, spec.dimensions.columns, spec.dimensions.remainder),
                    (5, 9, 5),
                )
                points, ranks, dimensions = sharp_point_orders(order)
                self.assertEqual(dimensions, spec.dimensions)
                self.assertEqual(len(points), order)
                self.assertEqual(len(ranks), order)
                self.assertEqual(sorted(ranks.values()), list(range(order)))
                self.assertEqual(sorted(sharp_permutation(order)), list(range(order)))
                self.assertEqual(
                    {point for point in points if point.colour == 0},
                    {
                        padding_point_at(5, 9, parity, index)
                        for index in range(1, PADDING_COUNTS[parity] + 1)
                    },
                )
                for index, (u, v) in enumerate(padding_pairs(parity), 1):
                    point = padding_point_at(5, 9, parity, index)
                    self.assertEqual(
                        point.x,
                        Fraction(algebraic_floor(u, 5))
                        + Fraction(index, PADDING_COUNTS[parity] + 1),
                    )
                    self.assertEqual(
                        point.y,
                        Fraction(algebraic_floor(v, 9))
                        + Fraction(index, PADDING_COUNTS[parity] + 1),
                    )

    def test_cutoff_and_zero_remainder_both_use_the_exact_fallback(self) -> None:
        for parity in (0, 1):
            mass = PADDING_COUNTS[parity]
            below = sharp_family_spec(mass + 98)
            at_cutoff = sharp_family_spec(mass + 100)
            self.assertTrue(below.fallback)
            self.assertEqual(below.core_size, 98)
            self.assertFalse(at_cutoff.fallback)
            self.assertEqual(at_cutoff.core_size, 100)
            self.assertIsNotNone(at_cutoff.dimensions)
            assert at_cutoff.dimensions is not None
            self.assertEqual(at_cutoff.dimensions.cells, 50)
            self.assertNotEqual(at_cutoff.dimensions.remainder, 0)

            # Exercise the second guard independently of the selected 50-cell
            # branch.  The real search is left untouched outside this context.
            zero_remainder = SharpDimensions(50, 5, 10, 0, 7, 4)
            with patch(
                "validation.sharp_four.sharp_dimensions",
                return_value=zero_remainder,
            ):
                fallback = sharp_family_spec(mass + 100)
                self.assertTrue(fallback.fallback)
                self.assertEqual(fallback.dimensions, zero_remainder)
                points, ranks, dimensions = sharp_point_orders(mass + 100)
                self.assertEqual(dimensions, zero_remainder)
                self.assertEqual(len(points), mass + 100)
                self.assertEqual(sorted(ranks.values()), list(range(mass + 100)))
                self.assertEqual(sharp_permutation(mass + 100), permutation(mass + 100))

    def test_independent_small_insertion_count_matches_the_binomial_formula(self) -> None:
        # Use one rational insertion and a 4-by-5 complete two-colour core.
        # This deliberately does not call padding_point_at or any production
        # moment routine; it checks the finite insertion count directly.
        rows, columns = 4, 5
        h, ell = 1, 2
        core = [
            (Fraction(x), Fraction(y), colour)
            for x in range(1, rows + 1)
            for y in range(1, columns + 1)
            for colour in (-1, 1)
        ]
        insertion = (Fraction(h) + Fraction(1, 2), Fraction(ell) + Fraction(1, 2), 0)
        points = core + [insertion]
        x_order = sorted(points, key=lambda p: (p[0], -p[2] * p[1], p[2]))
        y_order = sorted(points, key=lambda p: (p[1], p[2] * p[0], p[2]))
        ranks = {point: index for index, point in enumerate(y_order)}
        direct: Counter[tuple[int, ...]] = Counter()
        for chosen in combinations(core, 3):
            selected = chosen + (insertion,)
            if len({point[0] for point in selected}) != 4:
                continue
            if len({point[1] for point in selected}) != 4:
                continue
            ordered = sorted(selected, key=lambda p: x_order.index(p))
            direct[standardise(tuple(ranks[point] for point in ordered))] += 1

        def choose(n: int, k: int) -> int:
            return comb(n, k) if 0 <= k <= n else 0

        formula: Counter[tuple[int, ...]] = Counter()
        for pattern in all_patterns(4):
            count = 0
            for position in range(1, 5):
                value_rank = pattern[position - 1] + 1
                count += (
                    8
                    * choose(h, position - 1)
                    * choose(rows - h, 4 - position)
                    * choose(ell, value_rank - 1)
                    * choose(columns - ell, 4 - value_rank)
                )
            formula[pattern] = count
        self.assertEqual(direct, formula)
        self.assertEqual(sum(direct.values()), 1920)

    def test_exact_24_pattern_oriented_ledger_and_parity_independence(self) -> None:
        even = interior_limiting_four_vector(parity=0)
        odd = interior_limiting_four_vector(parity=1)
        self.assertEqual(even, odd)
        expected = {
            -12: {"1243", "1324", "2143", "3412", "4231"},
            -8: {"4321"},
            -4: {"2341", "3421", "4123", "4312"},
            0: {"1234", "1432", "2314", "2431", "3124", "3214", "4132"},
            12: {"1342", "1423", "2134", "2413", "3142", "3241", "4213"},
        }
        observed: dict[int, set[str]] = {}
        for pattern, value in even.items():
            observed.setdefault(int(value * 2304), set()).add(
                "".join(str(entry + 1) for entry in pattern)
            )
        self.assertEqual(observed, expected)
        groups = grouped_limiting_table()
        self.assertEqual(sum(len(patterns) for patterns in groups.values()), 24)
        self.assertIn(Fraction(1, 192), groups)

    def test_ten_pattern_dual_and_limit_are_exact(self) -> None:
        final = interior_limiting_four_vector()
        maximum, patterns = maximum_absolute(final)
        self.assertEqual(maximum, Fraction(1, 192))
        self.assertEqual(len(patterns), 12)
        self.assertEqual(centered_dual_value(final), Fraction(1, 192))
        self.assertEqual(centered_dual_value(), Fraction(1, 192))
        self.assertEqual(interior_lower_bound(), Fraction(1, 192))
        self.assertEqual(sum(abs(value) for value in DUAL_WEIGHTS.values()), 1)
        self.assertEqual(len(DUAL_WEIGHTS), 10)
        for parity in (0, 1):
            for pattern in all_patterns(4):
                expected = Fraction(
                    sum(INTERIOR_CORRECTION_UNITS[row][pattern[row]] for row in range(4)),
                    2304,
                )
                self.assertEqual(insertion_correction(pattern, parity), expected)

    def test_ten_pattern_dual_has_centred_position_value_marginals(self) -> None:
        counts = {(row, column): Fraction(0) for row in range(4) for column in range(4)}
        for name, weight in DUAL_WEIGHTS.items():
            pattern = tuple(int(value) - 1 for value in name)
            for row, column in enumerate(pattern):
                counts[(row, column)] += weight
        self.assertEqual(set(counts.values()), {Fraction(1, 8)})


ArbitraryCell = tuple[int, int]
ArbitraryPoint = tuple[int, int, int]
ArbitraryPattern = tuple[int, ...]


def _arbitrary_x_key(point: ArbitraryPoint) -> tuple[int, int, int]:
    x, y, colour = point
    return (x, -colour * y, colour)


def _arbitrary_y_key(point: ArbitraryPoint) -> tuple[int, int, int]:
    x, y, colour = point
    return (y, colour * x, colour)


def _arbitrary_pattern(points: tuple[ArbitraryPoint, ...]) -> ArbitraryPattern:
    """Compute a pattern from the two keys, without importing a grid oracle."""

    x_keys = tuple(_arbitrary_x_key(point) for point in points)
    y_keys = tuple(_arbitrary_y_key(point) for point in points)
    if len(set(x_keys)) != len(points) or len(set(y_keys)) != len(points):
        raise AssertionError("the direct pattern input has a repeated signed point")
    x_order = sorted(range(len(points)), key=lambda index: x_keys[index])
    y_order = sorted(range(len(points)), key=lambda index: y_keys[index])
    y_rank = {index: rank for rank, index in enumerate(y_order)}
    return tuple(y_rank[index] for index in x_order)


def _arbitrary_patterns(length: int) -> tuple[ArbitraryPattern, ...]:
    return tuple(permutations(range(length)))


def _arbitrary_axis_refinements(
    coordinates: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    """Enumerate independent uniform orders inside every equal-coordinate class."""

    classes = tuple(
        tuple(index for index, coordinate in enumerate(coordinates) if coordinate == value)
        for value in sorted(set(coordinates))
    )
    choices = tuple(tuple(permutations(indices)) for indices in classes)
    refinements: list[tuple[int, ...]] = []
    for class_orders in product(*choices):
        tie_rank: dict[int, int] = {}
        for class_order in class_orders:
            tie_rank.update({index: rank for rank, index in enumerate(class_order)})
        refinements.append(
            tuple(
                sorted(
                    range(len(coordinates)),
                    key=lambda index: (coordinates[index], tie_rank[index]),
                )
            )
        )
    return tuple(refinements)


def _arbitrary_colour_mass(
    cells: tuple[ArbitraryCell, ...],
) -> dict[ArbitraryPattern, int]:
    """Count direct signed colour words, rejecting repeated signed points."""

    length = len(cells)
    counts: Counter[ArbitraryPattern] = Counter()
    for colours in product((-1, 1), repeat=length):
        points = tuple((cell[0], cell[1], colour) for cell, colour in zip(cells, colours))
        if len(set(points)) != length:
            continue
        counts[_arbitrary_pattern(points)] += 1
    return {pattern: counts[pattern] for pattern in _arbitrary_patterns(length)}


def _arbitrary_jitter_mass(
    cells: tuple[ArbitraryCell, ...],
) -> dict[ArbitraryPattern, Fraction]:
    """Enumerate the independent label-order jitter distribution exactly."""

    length = len(cells)
    x_orders = _arbitrary_axis_refinements(tuple(cell[0] for cell in cells))
    y_orders = _arbitrary_axis_refinements(tuple(cell[1] for cell in cells))
    counts: Counter[ArbitraryPattern] = Counter()
    for x_order in x_orders:
        for y_order in y_orders:
            y_rank = {index: rank for rank, index in enumerate(y_order)}
            counts[tuple(y_rank[index] for index in x_order)] += 1
    denominator = len(x_orders) * len(y_orders)
    return {
        pattern: Fraction((1 << length) * counts[pattern], denominator)
        for pattern in _arbitrary_patterns(length)
    }


def _arbitrary_is_good(cells: tuple[ArbitraryCell, ...]) -> bool:
    """Return whether the equality graph is a matching with no repeated cell."""

    if len(set(cells)) != len(cells):
        return False
    degrees = [0] * len(cells)
    for left, right in combinations(range(len(cells)), 2):
        same_x = cells[left][0] == cells[right][0]
        same_y = cells[left][1] == cells[right][1]
        if same_x or same_y:
            degrees[left] += 1
            degrees[right] += 1
    return all(degree <= 1 for degree in degrees)


def _arbitrary_edge_count(cells: tuple[ArbitraryCell, ...]) -> int:
    return sum(
        cells[left][0] == cells[right][0] or cells[left][1] == cells[right][1]
        for left, right in combinations(range(len(cells)), 2)
    )


def _arbitrary_profile(
    cells: tuple[ArbitraryCell, ...],
    length: int,
) -> dict[ArbitraryPattern, int]:
    points = tuple((x, y, colour) for x, y in cells for colour in (-1, 1))
    counts: Counter[ArbitraryPattern] = Counter()
    for chosen in combinations(points, length):
        counts[_arbitrary_pattern(chosen)] += 1
    return {pattern: counts[pattern] for pattern in _arbitrary_patterns(length)}


def _arbitrary_a_sum(
    cells: tuple[ArbitraryCell, ...],
    length: int,
) -> dict[ArbitraryPattern, int]:
    total = {pattern: 0 for pattern in _arbitrary_patterns(length)}
    for ordered_cells in product(cells, repeat=length):
        mass = _arbitrary_colour_mass(tuple(ordered_cells))
        for pattern, value in mass.items():
            total[pattern] += value
    return total


def _arbitrary_b_sum(
    cells: tuple[ArbitraryCell, ...],
    length: int,
) -> dict[ArbitraryPattern, Fraction]:
    total = {pattern: Fraction(0) for pattern in _arbitrary_patterns(length)}
    for ordered_cells in product(cells, repeat=length):
        mass = _arbitrary_jitter_mass(tuple(ordered_cells))
        for pattern, value in mass.items():
            total[pattern] += value
    return total


def _arbitrary_interaction_moment(
    cells: tuple[ArbitraryCell, ...],
    length: int,
) -> int:
    size = len(cells)
    if length == 2:
        return size
    row_degrees = Counter(x for x, _ in cells)
    column_degrees = Counter(y for _, y in cells)
    row_cubes = sum(degree**3 for degree in row_degrees.values())
    column_cubes = sum(degree**3 for degree in column_degrees.values())
    mixed = sum(row_degrees[x] * column_degrees[y] for x, y in cells)
    return (
        comb(length, 3) * (row_cubes + column_cubes) * size ** (length - 3)
        + length * (length - 1) * (length - 2) * mixed * size ** (length - 3)
        + comb(length, 2) * size ** (length - 1)
    )


def _arbitrary_centered_norm(
    left: dict[ArbitraryPattern, int],
    right: dict[ArbitraryPattern, Fraction],
) -> Fraction:
    difference = {pattern: Fraction(left[pattern]) - right[pattern] for pattern in left}
    average = sum(difference.values(), Fraction(0)) / len(difference)
    return max(abs(value - average) for value in difference.values())


class ArbitraryCoordinateCancellationTests(unittest.TestCase):
    """Finite checks of the signed-grid cancellation mechanism on Z^2."""

    def test_arbitrary_signed_keys_are_injective_without_positive_coordinates(self) -> None:
        domains = (
            (
                (-2, -3),
                (-2, 1),
                (-1, 0),
                (0, -2),
                (0, 2),
                (3, 0),
            ),
            (
                (-3, -1),
                (-3, 2),
                (-1, -2),
                (0, 0),
                (2, -1),
                (3, 3),
            ),
        )
        for cells in domains:
            points = tuple((x, y, colour) for x, y in cells for colour in (-1, 1))
            self.assertEqual(len(set(_arbitrary_x_key(point) for point in points)), len(points))
            self.assertEqual(len(set(_arbitrary_y_key(point) for point in points)), len(points))

    def test_x_and_y_ties_have_two_to_two_sign_reversal_involutions(self) -> None:
        coordinates = (-3, -2, -1, 0, 1, 2, 3)
        for axis in ("x", "y"):
            for first, second in permutations(coordinates, 2):
                if axis == "x":
                    bases = ((0, first), (0, second))
                    key = _arbitrary_x_key
                else:
                    bases = ((first, 0), (second, 0))
                    key = _arbitrary_y_key

                all_points = tuple(
                    (base[0], base[1], colour) for base in bases for colour in (-1, 1)
                )
                self.assertEqual(len({key(point) for point in all_points}), 4)
                for base in bases:
                    for colour in (-1, 1):
                        point = (base[0], base[1], colour)
                        flipped = (base[0], base[1], -colour)
                        self.assertEqual(
                            key(flipped)[1:],
                            tuple(-value for value in key(point)[1:]),
                        )

                def before(colours: tuple[int, ...]) -> bool:
                    points = (
                        (bases[0][0], bases[0][1], colours[0]),
                        (bases[1][0], bases[1][1], colours[1]),
                    )
                    return key(points[0]) < key(points[1])

                self.assertEqual(
                    sum(before(colours) for colours in product((-1, 1), repeat=2)),
                    2,
                )
                for colours in product((-1, 1), repeat=2):
                    complement = (-colours[0], -colours[1])
                    self.assertEqual(before(colours), not before(complement))

    def test_negative_counterexample_breaks_the_old_positive_enumeration(self) -> None:
        first = (1, -2, 1)
        second = (1, -1, -1)
        direct = _arbitrary_x_key(first) < _arbitrary_x_key(second)

        # The former positive-ordinate table for y_first < y_second put both
        # (+,-) and (-,-) before the second point, i.e. it depended only on
        # the second colour.  This is the supplied negative-coordinate case.
        old_positive_table = second[2] == -1
        self.assertFalse(direct)
        self.assertTrue(old_positive_table)
        self.assertNotEqual(direct, old_positive_table)

        direct_count = sum(
            _arbitrary_x_key((1, -2, colours[0])) < _arbitrary_x_key((1, -1, colours[1]))
            for colours in product((-1, 1), repeat=2)
        )
        self.assertEqual(direct_count, 2)

    def test_disjoint_x_y_and_mixed_matchings_have_exact_colour_masses(self) -> None:
        matching_cases = (
            (
                "two_disjoint_x",
                ((-2, -3), (-2, 4), (1, -1), (1, 2)),
            ),
            (
                "two_disjoint_y",
                ((-3, -2), (4, -2), (-1, 1), (2, 1)),
            ),
            (
                "mixed_disjoint",
                ((0, -3), (0, 2), (-2, 1), (3, 1)),
            ),
        )
        for name, cells in matching_cases:
            with self.subTest(name=name):
                self.assertTrue(_arbitrary_is_good(cells))
                self.assertEqual(_arbitrary_edge_count(cells), 2)
                direct = _arbitrary_colour_mass(cells)
                jitter = _arbitrary_jitter_mass(cells)
                self.assertEqual(direct, jitter)
                self.assertEqual(sum(direct.values()), 2**4)
                self.assertEqual(sum(jitter.values()), Fraction(2**4))
                self.assertEqual(
                    sorted(value for value in direct.values() if value),
                    [2**2] * 2**2,
                )
                self.assertEqual(
                    sorted(value for value in jitter.values() if value),
                    [Fraction(2**2)] * 2**2,
                )

    def test_all_good_tuples_in_mixed_integer_domain_cancel_directly(self) -> None:
        cells = (
            (-2, -3),
            (-2, 1),
            (-1, 0),
            (0, -2),
            (0, 2),
            (3, 0),
        )
        good_counts: Counter[int] = Counter()
        bad_counts: Counter[int] = Counter()
        for length in (2, 3, 4):
            for ordered_cells in product(cells, repeat=length):
                direct = _arbitrary_colour_mass(tuple(ordered_cells))
                jitter = _arbitrary_jitter_mass(tuple(ordered_cells))
                self.assertLessEqual(sum(direct.values()), 2**length)
                self.assertEqual(sum(jitter.values()), Fraction(2**length))
                if _arbitrary_is_good(tuple(ordered_cells)):
                    self.assertEqual(direct, jitter)
                    good_counts[length] += 1
                else:
                    bad_counts[length] += 1
        self.assertTrue(all(good_counts[length] for length in (2, 3, 4)))
        self.assertTrue(all(bad_counts[length] for length in (2, 3, 4)))

    def test_repeated_cells_allow_only_opposite_colours(self) -> None:
        identity = (0, 1)
        reversal = (1, 0)
        for cell in ((0, 0), (-2, 3)):
            cells = (cell, cell)
            valid_colours = [
                colours
                for colours in product((-1, 1), repeat=2)
                if len({(cell[0], cell[1], colour) for colour in colours}) == 2
            ]
            direct = _arbitrary_colour_mass(cells)
            jitter = _arbitrary_jitter_mass(cells)
            self.assertEqual(valid_colours, [(-1, 1), (1, -1)])
            self.assertEqual(sum(direct.values()), 2)
            self.assertEqual(sum(jitter.values()), Fraction(4))
            if cell == (0, 0):
                self.assertEqual(direct[identity], 2)
                self.assertEqual(direct[reversal], 0)
                self.assertEqual(jitter[identity], Fraction(2))
                self.assertEqual(jitter[reversal], Fraction(2))

    def test_downstream_transfer_bound_on_small_arbitrary_finite_sets(self) -> None:
        domains = (
            (),
            ((0, 0),),
            ((-2, 3),),
            (
                (-2, -3),
                (-2, 1),
                (-1, 0),
                (0, -2),
                (0, 2),
                (3, 0),
            ),
            (
                (-3, -1),
                (-3, 2),
                (-1, -2),
                (0, 0),
                (2, -1),
                (3, 3),
            ),
            ((-1, -1), (-1, 1), (0, 0), (2, 0)),
        )
        nonzero_left = False
        for cells in domains:
            self.assertEqual(len(cells), len(set(cells)))
            for length in (2, 3, 4):
                with self.subTest(cells=cells, length=length):
                    profile = _arbitrary_profile(cells, length)
                    a_sum = _arbitrary_a_sum(cells, length)
                    b_sum = _arbitrary_b_sum(cells, length)
                    expected_a = {
                        pattern: factorial(length) * profile[pattern]
                        for pattern in _arbitrary_patterns(length)
                    }
                    self.assertEqual(a_sum, expected_a)
                    self.assertEqual(sum(profile.values()), comb(2 * len(cells), length))
                    self.assertEqual(
                        sum(a_sum.values()), factorial(length) * comb(2 * len(cells), length)
                    )
                    self.assertEqual(
                        sum(b_sum.values()), Fraction(2**length * len(cells) ** length)
                    )

                    centered = _arbitrary_centered_norm(a_sum, b_sum)
                    bound = Fraction(2**length * _arbitrary_interaction_moment(cells, length))
                    self.assertLessEqual(centered, bound)
                    nonzero_left = nonzero_left or centered > 0
        self.assertTrue(nonzero_left)


if __name__ == "__main__":
    unittest.main()
