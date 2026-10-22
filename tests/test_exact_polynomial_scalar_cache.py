"""Exact coefficient caching preserves multiplication and cancellation."""
from collections import defaultdict
from copy import deepcopy
import sympy as sp
from ye3t.couplings.lifted_cauchy_scalar import _multiply_polynomials, _simplified_scalar_cached


def reference_product(left, right):
    terms = defaultdict(lambda: sp.Integer(0))
    for first, a in left.items():
        for second, b in right.items():
            terms[tuple(sorted(first+second))] += sp.simplify(a*b)
    return {key: sp.simplify(value) for key, value in terms.items() if sp.simplify(value) != 0}


def test_exact_product_preserves_radicals_complex_values_and_zero_cancellation():
    x, y, z = (0,0,0), (0,0,1), (1,1,2)
    left = {(x,): sp.sqrt(2), (y,): sp.I*sp.sqrt(3), (z,): sp.Rational(2,3)}
    right = {(x,): sp.sqrt(2), (y,): -sp.I*sp.sqrt(3), (z,): -sp.Rational(2,3)}
    expected = reference_product(left, right)
    assert _multiply_polynomials(left, right) == expected
    assert (x,y) not in expected
    assert _multiply_polynomials({}, right) == {}
    assert _multiply_polynomials({(): sp.Integer(1)}, left) == left


def test_repeated_coefficients_reuse_bounded_exact_cache(monkeypatch):
    _simplified_scalar_cached.cache_clear()
    simplify = sp.simplify
    calls = []
    def counted(value):
        calls.append(value)
        return simplify(value)
    monkeypatch.setattr(sp, "simplify", counted)
    left = {((i,0,0),): sp.sqrt(2)/3 for i in range(20)}
    right = {((i,1,0),): sp.sqrt(3)/5 for i in range(20)}
    first = _multiply_polynomials(left, right)
    count = len(calls)
    assert count < 10
    assert _multiply_polynomials(left, right) == first
    assert len(calls) == count
    assert _simplified_scalar_cached.cache_info().maxsize == 4096


def test_binary_residual_cache_preserves_every_entry_and_detects_changed_values():
    from ye3t.couplings.lifted_cauchy_scalar import (
        _binary64_residual_report, _binary64_scalar_residual, _exact_scalar_payload)
    values = [sp.sqrt(2)/3, sp.I*sp.sqrt(7)/5, sp.Rational(1,3)+sp.I*sp.sqrt(2), sp.Integer(0)]
    rows = [_exact_scalar_payload(value) for value in values]*4
    _binary64_scalar_residual.cache_clear()
    for changed in (False, True):
        payload = {"rows": deepcopy(rows)}
        if changed:
            payload["rows"][6]["binary64"] = [0.125, -0.25]
        errors = []
        for index, row in enumerate(payload["rows"]):
            exact = values[index % len(values)]
            real, imag = row["binary64"]
            delta = sp.simplify(exact-sp.Rational(real)-sp.I*sp.Rational(imag))
            absolute = float(sp.Abs(delta).evalf(40))
            magnitude = float(sp.Abs(exact).evalf(40))
            errors.append((absolute, absolute/magnitude if magnitude else absolute))
        actual = _binary64_residual_report(payload)
        assert actual["exact_scalar_count"] == len(rows)
        for index, name in enumerate(("absolute", "relative")):
            worst = max(range(len(rows)), key=lambda i: errors[i][index])
            assert actual[f"maximum_{name}_residual"] == errors[worst][index]
            assert actual[f"maximum_{name}_residual_path"] == ("rows", str(worst))
    info = _binary64_scalar_residual.cache_info()
    assert info.maxsize == 4096
    assert info.misses == 5
    assert info.hits == 2*len(rows)-5
