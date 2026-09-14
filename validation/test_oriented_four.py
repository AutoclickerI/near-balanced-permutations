"""Independent finite regression checks for the oriented construction."""

import unittest
from fractions import Fraction as F
from itertools import permutations, product
from unittest.mock import patch

from . import oriented_four as oriented
from . import sharp_four as sharp


class OrientedFourTests(unittest.TestCase):
    def test_full_finite_certificate(self):
        result = oriented.validate_all_finite_oriented_checks()
        self.assertEqual(
            result["colour_words"],
            {
                "direct_key_comparisons": 5280,
                "low_defect_sign_word_cases": 2112,
                "polynomial_identities": 360,
            },
        )
        ledger = result["ledger"]
        self.assertIsInstance(ledger, dict)
        self.assertEqual(ledger["maximum"], F(1, 192))

    def test_nonconsecutive_positive_coordinates_preserve_counts(self):
        for alpha, beta in [((2, 1, 1), (1, 2, 1)), ((3, 1), (1, 1, 1, 1))]:
            for tau in permutations(range(4)):
                for signs in product((-1, 1), repeat=len(alpha)):
                    self.assertEqual(
                        oriented.block_word_count(tau, alpha, beta, signs),
                        oriented.direct_key_word_count(
                            tau,
                            alpha,
                            beta,
                            signs,
                            x_values=(F(1, 3), F(7), F(90), F(250))[: len(alpha)],
                            y_values=(F(2, 5), F(3), F(13), F(101))[: len(beta)],
                        ),
                    )

    def test_all_size_orders_and_actual_oriented_keys(self):
        for n in range(1, 301):
            pi = sharp.sharp_permutation(n)
            self.assertEqual(sorted(pi), list(range(n)))
        for n in (112, 117, 200, 201):
            points, ranks, _ = sharp.sharp_point_orders(n)
            ys = sorted(
                points,
                key=lambda p: (
                    p.y,
                    0
                    if p.colour == 0
                    else (-1, 1, -1, 1, 1, -1, 1, 1)[(int(p.x) - 1) % 8] * p.colour * p.x,
                    p.colour,
                ),
            )
            self.assertEqual({p: i for i, p in enumerate(ys)}, ranks)

    def test_dual_annihilates_centered_not_arbitrary_assignments(self):
        for i in range(4):
            for j in range(4):
                matrix = [[F(0) for _ in range(4)] for _ in range(4)]
                matrix[i][j] += 1
                matrix[0][0] -= 1
                self.assertEqual(
                    sum(
                        weight * oriented.assignment_value(matrix, tau)
                        for tau, weight in oriented.CENTERED_DUAL_WEIGHTS.items()
                    ),
                    0,
                )
        # Its nonzero action on constants must not be silently treated as a contrast.
        self.assertEqual(sum(oriented.CENTERED_DUAL_WEIGHTS.values()), F(1, 2))
        for theta in (F(1, 3), F(1), F(3, 2), F(4)):
            for eta in (F(-1), F(0), F(1, 4), F(1)):
                self.assertEqual(
                    oriented.dual_objective(theta, eta), F(1, 3) + (theta + 1 / theta) / 12
                )

    def test_old_mean_only_formula_is_rejected(self):
        tau = (0, 1, 3, 2)
        tx, ty, s, _ = oriented.local_statistics(tau)
        old = F(tx + ty) - F(8, 3) + F(1, 4) * (1 - F(4, 3) * s)
        self.assertNotEqual(old, oriented.corrected_h(tau))

    def test_negative_certificate_mutations_are_detected(self):
        with patch.object(oriented, "D_LP", tuple((F(0),) * 4 for _ in range(4))):
            with self.assertRaises(AssertionError):
                oriented.validate_centered_dual()
        changed = dict(oriented.CENTERED_DUAL_WEIGHTS)
        changed[(0, 1, 3, 2)] += F(1, 16)
        with patch.object(oriented, "CENTERED_DUAL_WEIGHTS", changed):
            with self.assertRaises(AssertionError):
                oriented.validate_centered_dual()
        stale = ((F(1, 10), F(1, 4)), (F(1, 3), F(3, 5)), (F(4, 5), F(9, 10)))
        with patch.object(sharp, "COMMON_ROOT_INTERVALS", stale):
            with self.assertRaises((AssertionError, ValueError)):
                sharp.validate_root_certificates()


if __name__ == "__main__":
    unittest.main()
