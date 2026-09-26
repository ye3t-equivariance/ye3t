from fractions import Fraction
import subprocess
import sys
from pathlib import Path

import pytest

from ye3t.exact_scalars import ExactRadical


def test_exact_radical_products_and_quotients_remain_canonical():
    root_six = ExactRadical.sqrt(6)
    root_fifteen = ExactRadical.sqrt(15)
    root_thirty = ExactRadical.sqrt(30)
    root_forty_two = ExactRadical.sqrt(42)

    assert root_six * root_fifteen == 3 * ExactRadical.sqrt(10)
    assert root_thirty * root_forty_two == 6 * ExactRadical.sqrt(35)
    assert root_six / root_fifteen == ExactRadical.sqrt(10) / 5
    assert (root_six + root_fifteen) - root_fifteen == root_six
    assert (root_six * root_fifteen).stable_key() == (
        ExactRadical.sqrt(10) * 3
    ).stable_key()


def test_exact_radical_arithmetic_matches_sympy_for_squarefree_inputs():
    sp = pytest.importorskip("sympy")
    values = (1, 2, 3, 5, 6, 7, 10, 14, 15, 21, 30, 42)
    for left in values:
        for right in values:
            actual_product = (ExactRadical.sqrt(left) * ExactRadical.sqrt(right))._sympy_()
            actual_quotient = (ExactRadical.sqrt(left) / ExactRadical.sqrt(right))._sympy_()
            assert sp.simplify(actual_product - sp.sqrt(left) * sp.sqrt(right)) == 0
            assert sp.simplify(actual_quotient - sp.sqrt(left) / sp.sqrt(right)) == 0


def test_exact_radical_public_construction_still_normalizes_untrusted_inputs():
    assert ExactRadical.sqrt(Fraction(12, 5)) == (
        2 * ExactRadical.sqrt(15) / 5
    )
    assert ExactRadical.sqrt(72) == 6 * ExactRadical.sqrt(2)

    large_prime = 1_000_000_007
    assert ExactRadical.sqrt(6 * large_prime * large_prime) == (
        large_prime * ExactRadical.sqrt(6)
    )
    denominator_prime = 1_000_000_009
    large_rational = Fraction(
        6 * large_prime * large_prime,
        35 * denominator_prime * denominator_prime,
    )
    assert ExactRadical.sqrt(large_rational) == (
        large_prime
        * ExactRadical.sqrt(210)
        / (35 * denominator_prime)
    )


def test_exact_radical_mutated_public_terms_reenter_normalization_boundary():
    value = ExactRadical.sqrt(2)
    value.terms.clear()
    value.terms[Fraction(8)] = Fraction(1)

    assert value * ExactRadical.sqrt(2) == ExactRadical.rational(4)


def test_exact_radical_payload_uses_the_existing_exact_schema_directly():
    from ye3t.couplings.lifted_cauchy_scalar import (
        _exact_scalar_from_payload,
        _exact_scalar_payload,
    )

    value = Fraction(2, 3) * ExactRadical.sqrt(5)
    payload = _exact_scalar_payload(value)
    assert payload["imag"] == ()
    expected_binary = complex(value._sympy_().evalf(17))
    assert payload["binary64"] == (
        float(expected_binary.real),
        float(expected_binary.imag),
    )
    assert _exact_scalar_from_payload(payload) == value._sympy_()


def test_exact_radical_payload_matches_legacy_sympy_for_signed_sum():
    from ye3t.couplings.lifted_cauchy_scalar import _exact_scalar_payload

    value = (
        ExactRadical.rational(Fraction(3, 7))
        - Fraction(5, 11) * ExactRadical.sqrt(2)
        + Fraction(13, 17) * ExactRadical.sqrt(15)
    )
    assert _exact_scalar_payload(value) == _exact_scalar_payload(
        value._sympy_()
    )


def test_large_shared_squarefree_factor_uses_bounded_canonical_arithmetic():
    package_root = Path(__file__).resolve().parents[1]
    code = r'''
from ye3t.exact_scalars import ExactRadical

prime = 1_000_000_007
left = ExactRadical.sqrt(2 * prime)
right = ExactRadical.sqrt(3 * prime)
assert left * right == prime * ExactRadical.sqrt(6)
assert left / right == ExactRadical.sqrt(6) / 3
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=package_root,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_large_generic_normalization_has_explicit_optional_sympy_boundary():
    package_root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] == "sympy":
        raise ImportError("blocked sympy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from ye3t.exact_scalars import ExactRadical

assert ExactRadical.sqrt(72) == 6 * ExactRadical.sqrt(2)
try:
    ExactRadical.sqrt(6 * 1_000_000_007 * 1_000_000_007)
except ImportError as error:
    assert "optional 'sympy' package" in str(error)
else:
    raise AssertionError("large generic normalization silently bypassed SymPy")
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=package_root,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
