"""Exact finite audits for the sharp growing-pattern theorem.

The component oracle uses full signed keys rather than the composition-code
oracle. Increasing counts are compared with a generic subsequence DP, and gap
moments with direct labelled-sample enumeration. These are finite challenges,
not proofs of the uniform bound, asymptotic limits or if-and-only-if theorem.
The deliberately narrow display checks bind the tested formulas to the paper;
they do not interpret arbitrary TeX or validate prose.
"""

import json
import re
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from itertools import combinations, permutations, product
from math import comb, factorial, isqrt, prod
from pathlib import Path

from .construction import permutation


def parts(k, mask):
    out = [1]
    for i in range(k - 1):
        if mask & (1 << i):
            out[-1] += 1
        else:
            out.append(1)
    return tuple(out)


def blocks(p):
    return tuple(j + 1 for j, n in enumerate(p) for _ in range(n))


def raw_w(tau, a, b):
    """Use full lexicographic signed keys, not compressed source formula."""
    k = len(tau)
    xs = blocks(a)
    ys0 = blocks(b)
    ys = tuple(ys0[t] for t in tau)
    inverse = sorted(range(k), key=tau.__getitem__)
    count = 0
    for colors in product((-1, 1), repeat=k):
        xkeys = [(xs[i], -colors[i] * ys[i], colors[i]) for i in range(k)]
        if not all(xkeys[i] < xkeys[i + 1] for i in range(k - 1)):
            continue
        ykeys = [(ys[i], colors[i] * xs[i], colors[i]) for i in inverse]
        count += all(ykeys[i] < ykeys[i + 1] for i in range(k - 1))
    return count


def components(k, edges, mask):
    parent = list(range(k))

    def root(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    for j, (u, v) in enumerate(edges):
        if mask & (1 << j):
            parent[root(u)] = root(v)
    groups = {}
    for j, (u, _) in enumerate(edges):
        if mask & (1 << j):
            r = root(u)
            groups[r] = groups.get(r, 0) | (1 << j)
    return list(groups.values())


def normalization_audit():
    """Direct subset counts versus an expectation over labelled samples."""
    cases = 0
    for k in range(2, 5):
        for a, b in product(range(1, 4), repeat=2):
            n = 2 * a * b
            if k > n:
                continue
            perm = rectangle_permutation(a, b)
            counts = Counter()
            for chosen in combinations(perm, k):
                ranks = {v: i for i, v in enumerate(sorted(chosen))}
                counts[tuple(ranks[v] for v in chosen)] += 1
            assert sum(counts.values()) == comb(n, k)
            masks = []
            for size in (a, b):
                hist = Counter()
                for sample in product(range(size), repeat=k):
                    s = sorted(sample)
                    mask = sum(1 << i for i in range(k - 1) if s[i] == s[i + 1])
                    hist[mask] += 1
                masks.append(hist)
            for tau in permutations(range(k)):
                ew = Fraction(0)
                for mx, nx in masks[0].items():
                    for my, ny in masks[1].items():
                        alpha, beta = parts(k, mx), parts(k, my)
                        weight = Fraction(
                            raw_w(tau, alpha, beta) * prod(factorial(r) for r in alpha + beta), 2**k
                        )
                        ew += weight * Fraction(nx * ny, (a * b) ** k)
                assert ew == Fraction(factorial(k) ** 2 * counts[tau], n**k)
                cases += 1
    return {"exact_expectation_cases": cases, "k_range": [2, 4], "dimension_range": [1, 3]}


def equality_audit():
    counts = Counter()
    nontrivial_examples = []
    for k in range(2, 6):
        n = 2 * (k - 1)
        smallmask = (1 << (k - 1)) - 1
        for tau in permutations(range(k)):
            inv = sorted(range(k), key=tau.__getitem__)
            edges = [(i, i + 1) for i in range(k - 1)] + [
                (inv[i], inv[i + 1]) for i in range(k - 1)
            ]
            ws = []
            factor = []
            for mask in range(1 << n):
                a, b = parts(k, mask & smallmask), parts(k, mask >> (k - 1))
                f = prod(factorial(r) for r in a + b)
                w = Fraction(raw_w(tau, a, b) * f, 2**k)
                ws.append(w)
                factor.append(f)
            assert ws[0] == 1
            cs = ws.copy()
            for j in range(n):
                for mask in range(1 << n):
                    if mask & (1 << j):
                        cs[mask] -= cs[mask ^ (1 << j)]
            for mask in range(1 << n):
                comps = components(k, edges, mask)
                assert ws[mask] == prod(ws[t] for t in comps)
                assert cs[mask] == prod(cs[t] for t in comps)
                if any(t.bit_count() == 1 for t in comps):
                    assert cs[mask] == 0
                    counts["singleton_component_zero"] += 1
                if len(comps) == 1:
                    assert abs(cs[mask]) <= 2 ** mask.bit_count() * factor[mask]
                    counts["connected_coefficient_bounds"] += 1
                if mask and cs[mask] and len(nontrivial_examples) < 10:
                    nontrivial_examples.append(
                        {"k": k, "tau": tau, "mask": mask, "c": str(cs[mask])}
                    )
                counts["all_masks"] += 1
            counts["patterns"] += 1
    return {"counts": dict(counts), "nonzero_examples": nontrivial_examples}


def spacing_audit():
    checks = 0
    for k in range(2, 7):
        for a in range(1, 6):
            hist = [0] * (1 << (k - 1))
            for seq in product(range(a), repeat=k):
                t = sorted(seq)
                mask = sum(1 << i for i in range(k - 1) if t[i] == t[i + 1])
                hist[mask] += 1
            moments = hist.copy()
            for j in range(k - 1):
                for mask in range(1 << (k - 1)):
                    if not mask & (1 << j):
                        moments[mask] += moments[mask | (1 << j)]
            for mask, count in enumerate(moments):
                d = mask.bit_count()
                run_d = [r - 1 for r in parts(k, mask) if r > 1]
                rhs = Fraction(
                    factorial(k), factorial(k - d) * a**d * prod(factorial(t) for t in run_d)
                )
                assert Fraction(count, a**k) <= rhs
                checks += 1
    return {"moment_inequalities": checks, "k_range": [2, 6], "alphabet_range": [1, 5]}


def rectangle_permutation(a, b):
    points = list(product(range(1, a + 1), range(1, b + 1), (-1, 1)))
    xorder = sorted(points, key=lambda p: (p[0], -p[2] * p[1], p[2]))
    yrank = {p: i for i, p in enumerate(sorted(points, key=lambda p: (p[1], p[2] * p[0], p[2])))}
    return [yrank[p] for p in xorder]


def increasing_dp(perm):
    """Generic subsequence DP, independent of the cell-chain formula."""
    end = []
    total = [0] * (len(perm) + 1)
    for i, y in enumerate(perm):
        row = [0, 1]
        for j in range(i):
            if perm[j] < y:
                prev = end[j]
                if len(row) < len(prev) + 1:
                    row.extend([0] * (len(prev) + 1 - len(row)))
                for k in range(1, len(prev)):
                    row[k + 1] += prev[k]
        end.append(row)
        for k in range(1, len(row)):
            total[k] += row[k]
    return total


def cb(n, k):
    return comb(n, k) if 0 <= k <= n else 0


def increasing_formula(a, b, k):
    return 2 * sum(cb(k - 1, j) * cb(a + k - 1 - j, k) * cb(b + j, k) for j in range(k))


def monotone_audit():
    checks = 0
    for a in range(1, 9):
        for b in range(1, 9):
            direct = increasing_dp(rectangle_permutation(a, b))
            for k in range(1, 2 * a * b + 1):
                assert direct[k] == increasing_formula(a, b, k), (a, b, k)
                checks += 1
            assert max(k for k, count in enumerate(direct) if count) == a + b - 1
    assert increasing_formula(4, 4, 4) == 1040
    exact_products = 0
    for m in range(1, 8):
        n = 2 * m * m
        for k in range(1, 2 * m):
            value = Fraction(
                sum(
                    comb(k - 1, j) * prod(m * m - (j - r) ** 2 for r in range(k)) for j in range(k)
                ),
                2 ** (k - 1),
            )
            expected = value * Fraction(2**k, factorial(k) ** 2)
            assert expected == increasing_formula(m, m, k)
            norm = Fraction(factorial(k) * increasing_formula(m, m, k), comb(n, k))
            assert 0 <= norm <= 1
            exact_products += 1
    return {
        "rectangle_profile_entries": checks,
        "square_product_identities": exact_products,
        "m4_k4": 1040,
    }


def containment_audit():
    """Independent signed-point representation versus the canonical Pi_N."""

    def keyx(p):
        return (p[0], -p[2] * p[1], p[2])

    def keyy(p):
        return (p[1], p[2] * p[0], p[2])

    def induced(points):
        ys = {p: i for i, p in enumerate(sorted(points, key=keyy))}
        return tuple(ys[p] for p in sorted(points, key=keyx))

    for N in range(2, 301):
        E = N // 2
        a = isqrt(E)
        b, r = divmod(E, a)
        cells = set(product(range(1, a + 1), range(1, b + 1)))
        if r:
            cells.update((a + 1, (j * b + r - 1) // r) for j in range(1, r + 1))
        pts = {(x, y, c) for x, y in cells for c in (-1, 1)}
        if N % 2:
            pts.add((a + 2, b + 1, 1))
        low = set(product(range(1, a + 1), range(1, a + 1), (-1, 1)))
        high = set(product(range(1, a + 4), range(1, a + 4), (-1, 1)))
        assert low <= pts <= high
        assert len(pts) == N and induced(pts) == permutation(N)
        # Restrict each actual ambient order, rather than calling the same
        # subset function on both sides of an equality.
        for sub, ambient in [(low, pts), (pts, high)]:
            for key in (keyx, keyy):
                assert [p for p in sorted(ambient, key=key) if p in sub] == sorted(sub, key=key)
    return {"N_range": [2, 300], "cases": 299, "odd_signed_representative_checked": True}


DISPLAYS = {
    "eq:collision-expectation": r"\mathbb EW(\mathsf S)=\frac{(k!)^2\#_\tau(\mathrm{ES}^{\pm}(a,b))}{n^k}.",
    "eq:increasing-exact": r"I_{a,b}(k)=2\sum_{j=0}^{k-1}\binom{k-1}{j}\binom{a+k-1-j}{k}\binom{b+j}{k},",
    "eq:increasing-product": r"\frac{k!I_{m,m}(k)}{\binom{2m^2}{k}}=\frac1{D_{m,k}}\mathbb E\prod_{r=0}^{k-1}\left(1-\frac{(J-r)^2}{m^2}\right).",
    "eq:increasing-critical": r"\frac{k!\,\#_{12\cdots k}(\Pi_N)}{\binom Nk}\longrightarrow e^{-c^3/6}.",
    "eq:connected-bound": r"\left|\frac{k!\,\#_\tau(\mathrm{ES}^{\pm}(a,b))}{\binom nk}-1\right|\le\exp\left(\frac{4096k^3}{s^2(1-64k/s)}+\frac{k(k-1)}{2(n-k+1)}\right)-1.",
}


def source_displays(source):
    """Fail closed on a missing, changed, ambiguous or unsupported display."""
    for label, expected in DISPLAYS.items():
        pat = r"\\begin\{equation\}\s*\\label\{" + re.escape(label) + r"\}(.*?)\\end\{equation\}"
        matches = re.findall(pat, source, re.S)
        if len(matches) != 1:
            raise ValueError("missing or ambiguous display: " + label)
        body = re.sub(r"%[^\n]*", "", matches[0])
        if re.sub(r"\s+", "", body) != re.sub(r"\s+", "", expected):
            raise ValueError("changed or unsupported display: " + label)
    return {
        "matched_display_labels": list(DISPLAYS),
        "scope": "Finite formula/source agreement only; not analytic proof verification.",
    }


def default_main_path() -> Path:
    return Path(__file__).resolve().parents[1] / "paper" / "main.tex"


def audit():
    source = default_main_path().read_bytes()
    return {
        "scope": "Exact finite regression evidence, not an asymptotic theorem certificate.",
        "source_sha256": sha256(source).hexdigest(),
        "source": source_displays(source.decode()),
        "normalization": normalization_audit(),
        "components": equality_audit(),
        "spacing": spacing_audit(),
        "monotone": monotone_audit(),
        "containment": containment_audit(),
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
