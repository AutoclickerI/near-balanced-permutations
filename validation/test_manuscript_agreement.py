"""Tests for finite source-to-validation agreement with the active displays."""

from __future__ import annotations

import unittest
from fractions import Fraction
from hashlib import sha256

from validation.lp_certificates import (
    Q_STAR,
    RHO,
    audit_certificate_path,
    default_certificate_path,
)
from validation.manuscript_agreement import (
    ManuscriptAgreementError,
    audit_sources,
    default_source_paths,
    parse_fixed_k_optima_table,
    parse_sharp_four_source,
    parse_square_polynomial_table,
)

MAIN, SHARP = default_source_paths()


class ManuscriptAgreementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.main_bytes = MAIN.read_bytes()
        cls.sharp_bytes = SHARP.read_bytes()
        cls.main_source = cls.main_bytes.decode("utf-8")
        cls.sharp_source = cls.sharp_bytes.decode("utf-8")

    def test_active_display_reads_centered_dual_and_fixed_k_table(self) -> None:
        parsed = parse_sharp_four_source(self.sharp_source)
        self.assertEqual(
            dict(parsed.contrast.weights),
            {
                name: Fraction(weight, 16)
                for name, weight in {
                    "1243": -2,
                    "1342": 2,
                    "1423": 2,
                    "2134": 2,
                    "2143": -1,
                    "2413": 1,
                    "3142": 1,
                    "3241": 2,
                    "3412": -1,
                    "4213": 2,
                }.items()
            },
        )
        fixed = parse_fixed_k_optima_table(self.sharp_source)
        self.assertEqual(
            tuple((row.k, str(row.theta)) for row in fixed.rows),
            ((4, "1"), (5, "1"), (6, "1"), (7, "1"), (8, "1"), (6, str(RHO))),
        )
        self.assertEqual(str(fixed.rows[0].q), "2/3")
        self.assertEqual(fixed.aliases.rho, RHO)
        self.assertEqual(fixed.aliases.q_star, Q_STAR)
        self.assertEqual(fixed.rows[-1].q, Q_STAR)
        self.assertEqual(fixed.rows[-1].normalized, Q_STAR / 34560)

    def test_source_audit_compares_fixed_k_table_to_exact_certificates(self) -> None:
        report = audit_sources(self.main_source, self.sharp_source)
        self.assertEqual(
            report.main_sha256,
            sha256(self.main_source.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            report.sharp_four_sha256,
            sha256(self.sharp_bytes).hexdigest(),
        )
        self.assertEqual(len(report.sharp.ledger.levels), 24)
        self.assertEqual(len(report.table.rows), 6)
        self.assertEqual(len(report.fixed_k_optima.rows), 6)
        self.assertEqual(sum(item.patterns for item in report.certificates), 46944)
        self.assertTrue(report.text().endswith("MANUSCRIPT_AGREEMENT_OK\n"))

    def test_source_mutations_are_rejected(self) -> None:
        cases = (
            ("D_entry", "sharp", "51&5", "52&5"),
            ("C_entry", "sharp", "9&-1", "10&-1"),
            ("ledger_entry", "sharp", "-12&1243,1324", "-11&1243,1324"),
            ("dual_coefficient", "sharp", "-2z_{1243}", "-3z_{1243}"),
            ("square_polynomial", "main", "-5(4m^2-9)", "-5(4m^2-8)"),
            ("square_normalization", "main", r"\frac{144}{m^2(m^2-1)}", r"\frac{145}{m^2(m^2-1)}"),
            ("fixed_k_case", "sharp", "4&1&", "5&1&"),
            ("fixed_k_theta", "sharp", r"6&\rho&", r"6&\dfrac{5}{7}&"),
            ("fixed_k_q", "sharp", r"\dfrac{2}{3}", "1"),
            ("fixed_k_normalization", "sharp", r"\dfrac{1}{144}", "1"),
            ("k6_surd_sign", "sharp", r"2-\frac{2\sqrt3}{3}", r"2+\frac{2\sqrt3}{3}"),
            ("k6_surd_coefficient", "sharp", r"\frac{111+5\sqrt3}{60}", r"\frac{111+6\sqrt3}{60}"),
            ("k6_surd_syntax", "sharp", r"2-\frac{2\sqrt3}{3}", r"2-\frac{2\sqrt[3]{3}}{3}"),
            ("k6_alias_prefix", "sharp", r"\rho=2-", r"1+\rho=2-"),
            ("ledger_scale", "sharp", r"2304L_\tau", r"2305L_\tau"),
            ("dual_scale", "sharp", r"16\mathcal W(z)", r"17\mathcal W(z)"),
            ("first_root_interval", "sharp", "(1/10,1/5)", "(1/100,1/10)"),
            ("second_root_interval", "sharp", "(2/5,7/10)", "(1/3,3/5)"),
            ("third_root_interval", "sharp", "(3/4,19/20)", "(4/5,9/10)"),
        )
        certificate = audit_certificate_path(default_certificate_path())
        for name, target, old, new in cases:
            with self.subTest(mutation=name):
                sources = {"main": self.main_source, "sharp": self.sharp_source}
                self.assertEqual(sources[target].count(old), 1)
                sources[target] = sources[target].replace(old, new, 1)
                with self.assertRaises(ManuscriptAgreementError):
                    audit_sources(sources["main"], sources["sharp"], certificate_audit=certificate)

        # A wider interval with the same certified root is still valid.
        widened = self.sharp_source.replace("(1/10,1/5)", "(1/10,1/4)", 1)
        audit_sources(self.main_source, widened, certificate_audit=certificate)

    def test_dual_parser_layout_and_invalid_math(self):
        source = self.sharp_source
        expected = parse_sharp_four_source(source).contrast.weights
        flattened = source.replace("={}&", "=").replace("\\\\\n &+z_{2413}", "+z_{2413}")
        self.assertEqual(parse_sharp_four_source(flattened).contrast.weights, expected)
        with self.assertRaises(ManuscriptAgreementError):
            parse_sharp_four_source(source.replace("-2z_{1243}", "-2z_{1243}+\\sin(0)"))

    def test_unsupported_fixed_k_forms_fail_loudly(self) -> None:
        mutated = self.sharp_source.replace(r"\dfrac{2}{3}", r"\tfrac{2}{3}", 1)
        with self.assertRaisesRegex(
            ManuscriptAgreementError,
            r"fixed-k optimum table row 1 q_k\(theta\): unsupported or missing",
        ):
            parse_fixed_k_optima_table(mutated)

        mutated = self.sharp_source.replace(r"\dfrac{2}{3}", r"\dfrac23", 1)
        with self.assertRaisesRegex(
            ManuscriptAgreementError,
            r"fixed-k optimum table row 1 q_k\(theta\): unsupported or missing",
        ):
            parse_fixed_k_optima_table(mutated)

        mutated = self.sharp_source.replace(r"\dfrac{2}{3}", r"\frac{2}{3}", 1)
        with self.assertRaisesRegex(
            ManuscriptAgreementError,
            r"fixed-k optimum table row 1 q_k\(theta\): unsupported or missing",
        ):
            parse_fixed_k_optima_table(mutated)

        mutated = self.sharp_source.replace(
            "q_k(\\theta)/(2k!(k-2)!)",
            "q_k(\\theta)/[2k!(k-2)!]",
            1,
        )
        with self.assertRaisesRegex(
            ManuscriptAgreementError,
            r"fixed-k optimum table: unsupported or missing",
        ):
            parse_fixed_k_optima_table(mutated)

    def test_unsupported_matrix_and_square_forms_fail_loudly(self) -> None:
        mutated_sharp = self.sharp_source.replace(r"\begin{pmatrix}", r"\begin{bmatrix}", 1)
        with self.assertRaisesRegex(ManuscriptAgreementError, r"D matrix: unsupported or missing"):
            parse_sharp_four_source(mutated_sharp)

        mutated_main = self.main_source.replace(
            r"\frac{144}{m^2(m^2-1)}",
            r"\frac{144}{m^{2}(m^2-1)}",
            1,
        )
        with self.assertRaisesRegex(
            ManuscriptAgreementError,
            r"square polynomial table: unsupported or missing",
        ):
            parse_square_polynomial_table(mutated_main)

    def test_unsupported_polynomial_form_fails_loudly(self) -> None:
        mutated = self.main_source.replace("-5(4m^2-9)", "-5(4m^2-9)+0", 1)
        with self.assertRaisesRegex(
            ManuscriptAgreementError,
            r"square polynomial table: unsupported or missing",
        ):
            parse_square_polynomial_table(mutated)

    def test_unsupported_k6_alias_forms_fail_loudly(self) -> None:
        mutated = self.sharp_source.replace(
            r"\frac{111+5\sqrt3}{60}",
            r"\dfrac{111+5\sqrt3}{60}",
            1,
        )
        with self.assertRaisesRegex(
            ManuscriptAgreementError,
            r"k6 aspect aliases q_star: unsupported or missing",
        ):
            parse_fixed_k_optima_table(mutated)

        mutated = self.sharp_source.replace(r"q_*/34560", r"\frac{q_*}{34560}", 1)
        with self.assertRaisesRegex(
            ManuscriptAgreementError,
            r"fixed-k optimum table",
        ):
            parse_fixed_k_optima_table(mutated)

        mutated = self.sharp_source.replace(r"\rho=2-", r"1+\rho=2-", 1)
        with self.assertRaisesRegex(
            ManuscriptAgreementError,
            r"k6 aspect aliases: unsupported or missing",
        ):
            parse_fixed_k_optima_table(mutated)


if __name__ == "__main__":
    unittest.main()
