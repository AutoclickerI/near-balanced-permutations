"""Finite regressions for the new analytic growing-pattern arguments."""

import re
import unittest

from . import growing_patterns as g


class GrowingPatternTests(unittest.TestCase):
    def test_exact_expectation_normalization(self):
        self.assertEqual(g.normalization_audit()["exact_expectation_cases"], 258)

    def test_all_small_equality_components(self):
        data = g.equality_audit()["counts"]
        self.assertEqual(data["all_masks"], 32360)
        self.assertEqual(data["singleton_component_zero"], 6208)
        self.assertGreater(data["connected_coefficient_bounds"], 0)

    def test_joint_spacing_moments(self):
        self.assertEqual(g.spacing_audit()["moment_inequalities"], 310)

    def test_monotone_formula_and_zero_support(self):
        data = g.monotone_audit()
        self.assertEqual(data["rectangle_profile_entries"], 2592)
        self.assertEqual(data["square_product_identities"], 49)
        self.assertEqual(data["m4_k4"], 1040)

    def test_square_sandwich_and_canonical_odd_order(self):
        self.assertEqual(g.containment_audit()["cases"], 299)

    def test_source_formula_agreement(self):
        text = g.default_main_path().read_text()
        self.assertEqual(len(g.source_displays(text)["matched_display_labels"]), 5)

    def test_source_mutations_are_detected(self):
        text = g.default_main_path().read_text()
        mutations = [
            ("eq:increasing-exact", "=2\\sum", "=3\\sum"),
            ("eq:increasing-product", "1-\\frac{(J-r)^2}", "1+\\frac{(J-r)^2}"),
            ("eq:increasing-critical", "c^3/6", "c^3/12"),
            ("eq:connected-bound", "4096k^3", "4095k^3"),
            ("eq:collision-expectation", "(k!)^2", "k!"),
        ]
        for label, old, new in mutations:
            pattern = (
                r"(\\begin\{equation\}\s*\\label\{"
                + re.escape(label)
                + r"\})(.*?)(\\end\{equation\})"
            )
            match = re.search(pattern, text, re.S)
            assert match is not None
            self.assertEqual(match.group(2).count(old), 1)
            changed = (
                text[: match.start(2)] + match.group(2).replace(old, new) + text[match.end(2) :]
            )
            with self.subTest(label=label), self.assertRaises(ValueError):
                g.source_displays(changed)

    def test_missing_or_ambiguous_displays_fail(self):
        text = g.default_main_path().read_text()
        with self.assertRaises(ValueError):
            g.source_displays(text.replace("eq:increasing-exact", "eq:unknown"))
        block_match = re.search(
            r"\\begin\{equation\}\\label\{eq:increasing-exact\}.*?\\end\{equation\}", text, re.S
        )
        assert block_match is not None
        block = block_match.group()
        with self.assertRaises(ValueError):
            g.source_displays(text + "\n" + block)

    def test_repeated_cell_can_use_a_longer_path(self):
        # In the 1-by-2 rectangle the outer repeated cell joins vertices
        # 0 and 3 along an X path of length three and a single Y edge.
        tau = tuple(g.rectangle_permutation(1, 2))
        self.assertEqual(tau, (3, 1, 0, 2))
        inv = sorted(range(4), key=tau.__getitem__)
        xedges = [(i, i + 1) for i in range(3)]
        yedges = [(inv[i], inv[i + 1]) for i in range(3)]
        self.assertIn((3, 0), yedges)
        self.assertNotIn((0, 3), xedges)
        self.assertEqual(g.raw_w(tau, (4,), (2, 2)), 1)


if __name__ == "__main__":
    unittest.main()
