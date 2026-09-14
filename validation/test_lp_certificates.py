"""Focused tests for exact fixed-k certificate verification."""

from __future__ import annotations

import json
import unittest
from copy import deepcopy
from fractions import Fraction
from unittest.mock import patch

from validation.lp_certificates import (
    Q_STAR,
    RHO,
    CertificateError,
    audit_certificate_path,
    default_certificate_path,
    verify_certificate,
    verify_payload,
)
from validation.quadratic import Qsqrt3


class ExactCertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = default_certificate_path()
        cls.payload = json.loads(cls.path.read_text(encoding="utf-8"))
        cls.report = audit_certificate_path(cls.path)

    def test_all_fixed_k_certificates_and_46944_patterns_are_exact(self) -> None:
        self.assertEqual(
            tuple((item.k, str(item.theta)) for item in self.report.certificates),
            ((4, "1"), (5, "1"), (6, "1"), (7, "1"), (8, "1"), (6, str(RHO))),
        )
        self.assertEqual(self.report.total_patterns, 46944)
        self.assertEqual(
            tuple(str(item.normalized_constant) for item in self.report.certificates),
            (
                "1/144",
                "1/1152",
                "1/17280",
                "31/12700800",
                "109/1625702400",
                str(Q_STAR / 34560),
            ),
        )
        self.assertTrue(self.report.text().endswith("LP_CERTIFICATE_AUDIT_OK\n"))
        self.assertEqual(self.report.global_aspect.rho, RHO)
        self.assertEqual(self.report.global_aspect.q_star, Q_STAR)
        self.assertEqual(
            self.report.global_aspect.form_names,
            ("W_left", "W_right", "W_square"),
        )
        self.assertEqual(self.report.global_aspect.intersections, 4)

    def test_square_dual_alpha_is_exact_and_does_not_claim_k6_optimality(self) -> None:
        self.assertEqual(
            tuple(item.square_alpha for item in self.report.certificates),
            (Fraction(0), Fraction(0), Fraction(-1, 5), Fraction(13, 42), Fraction(3, 8), None),
        )
        self.assertEqual(
            {
                item.k
                for item in self.report.certificates
                if item.square_alpha is not None and item.square_alpha >= 0
            },
            {4, 5, 7, 8},
        )
        self.assertIn("patterns_checked=46944 (=46224+720)", self.report.text())

    def test_deliberate_primal_dual_and_float_corruption_is_detected(self) -> None:
        primal, dual, floating, aspect = (
            deepcopy(self.payload["certificates"][0]) for _ in range(4)
        )
        primal["primal_D"][0][0] = str(Fraction(primal["primal_D"][0][0]) + 1)
        primal["primal_D"][0][1] = str(Fraction(primal["primal_D"][0][1]) - 1)
        dual["dual_signed_support"][0]["weight"] = str(
            Fraction(dual["dual_signed_support"][0]["weight"]) + Fraction(1, 997)
        )
        floating["objective_q"] = float(Fraction(floating["objective_q"]))
        aspect["theta"] = "2"

        surd_sign = deepcopy(self.payload["certificates"][-1])
        surd_sign["theta"]["b"] = str(-Fraction(surd_sign["theta"]["b"]))
        surd_coefficient = deepcopy(self.payload["certificates"][-1])
        entry = surd_coefficient["primal_D"][0][0]
        entry["b"] = str(Fraction(entry["b"]) + 1)

        for name, record in (
            ("primal_centered_entry", primal),
            ("dual_weight", dual),
            ("floating_objective", floating),
            ("aspect", aspect),
            ("surd_sign", surd_sign),
            ("surd_coefficient", surd_coefficient),
        ):
            with self.subTest(mutation=name), self.assertRaises(CertificateError):
                verify_certificate(record)

        global_form = deepcopy(self.payload)
        global_form["global_aspect"]["dual_form_coefficients"]["W_left"][0] = "2/3"
        with self.assertRaises(CertificateError):
            verify_payload(global_form)

    def test_float_and_duplicate_pattern_inputs_fail_closed(self) -> None:
        floating = deepcopy(self.payload["certificates"][0])
        floating["objective_q"] = 0.5
        with self.assertRaisesRegex(CertificateError, "integer or slash rational"):
            verify_certificate(floating)

        duplicate = deepcopy(self.payload["certificates"][0])
        duplicate["dual_signed_support"][1]["tau"] = duplicate["dual_signed_support"][0]["tau"]
        with self.assertRaisesRegex(CertificateError, "duplicate"):
            verify_certificate(duplicate)

        aspect = deepcopy(self.payload["certificates"][0])
        aspect["theta"] = "2"
        with self.assertRaises(CertificateError):
            verify_certificate(aspect)

    def test_unsupported_k_fails_before_profile_enumeration(self) -> None:
        unsupported = deepcopy(self.payload["certificates"][0])
        unsupported["k"] = 1000
        with patch(
            "validation.lp_certificates._normalized_profile",
            side_effect=AssertionError("unsupported certificate triggered enumeration"),
        ) as enumerate_profile:
            with self.assertRaisesRegex(CertificateError, "unsupported k=1000"):
                verify_certificate(unsupported)
            enumerate_profile.assert_not_called()

    def test_quadratic_runtime_parsing_rejects_float_and_bad_pairs(self) -> None:
        algebraic = deepcopy(self.payload["certificates"][-1])
        algebraic["objective_q"] = 1.5
        with self.assertRaisesRegex(CertificateError, "integer or slash rational"):
            verify_certificate(algebraic)

        algebraic = deepcopy(self.payload["certificates"][-1])
        algebraic["theta"] = {"a": "2", "b": "-2/3", "c": "0"}
        with self.assertRaisesRegex(CertificateError, "exactly a and b"):
            verify_certificate(algebraic)

        malformed = deepcopy(self.payload["certificates"][-1])
        malformed["primal_D"][0][0] = {"a": "-101/180", "b": "-5*sqrt(3)/108"}
        with self.assertRaisesRegex(CertificateError, "integer or slash rational"):
            verify_certificate(malformed)

    def test_quadratic_rational_equality_preserves_mapping_and_set_contract(self) -> None:
        rational = Qsqrt3(Fraction(1), Fraction(0))
        self.assertEqual(rational, Fraction(1))
        mapping: dict[object, str] = {rational: "quadratic"}
        self.assertEqual(mapping[Fraction(1)], "quadratic")
        self.assertEqual(len({rational, Fraction(1), 1}), 1)
        self.assertNotEqual(rational, True)

    def test_quadratic_sign_and_reciprocal_are_exact(self) -> None:
        self.assertEqual(Qsqrt3(Fraction(1), Fraction(-1, 2)).sign(), 1)
        self.assertEqual(Qsqrt3(Fraction(1), Fraction(-1)).sign(), -1)
        self.assertEqual(Qsqrt3(Fraction(-1), Fraction(1)).sign(), 1)
        self.assertEqual(RHO * RHO.reciprocal(), Qsqrt3.one())

    def test_certificate_set_rejects_missing_fixed_k_row(self) -> None:
        missing = deepcopy(self.payload)
        missing["certificates"].pop()
        with self.assertRaisesRegex(CertificateError, r"expected cases="):
            verify_payload(missing)


if __name__ == "__main__":
    unittest.main()
