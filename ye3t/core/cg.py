"""Small numeric Clebsch-Gordan helpers without symbolic runtime imports."""

from fractions import Fraction
from functools import lru_cache
from math import factorial, sqrt

from ye3t.exact_scalars import ExactRadical


def _half_integer_fraction(value):
    if isinstance(value, Fraction):
        frac = value
    elif isinstance(value, int):
        frac = Fraction(value, 1)
    elif isinstance(value, float):
        frac = Fraction(value).limit_denominator(2)
        if abs(float(frac) - value) > 1e-12:
            raise ValueError(f"CG arguments must be integer or half-integer, got {value!r}.")
    else:
        frac = Fraction(value)
    if frac.denominator not in {1, 2}:
        raise ValueError(f"CG arguments must be integer or half-integer, got {value!r}.")
    return frac


def _factorial_int(value):
    value = Fraction(value)
    if value.denominator != 1:
        raise ValueError(f"Expected an integer factorial argument, got {value!r}.")
    value = int(value)
    if value < 0:
        raise ValueError(f"Expected a nonnegative factorial argument, got {value!r}.")
    return factorial(value)


def _is_integral(value):
    return Fraction(value).denominator == 1


def _ceil_fraction(value):
    value = Fraction(value)
    return -((-value.numerator) // value.denominator)


def _floor_fraction(value):
    value = Fraction(value)
    return value.numerator // value.denominator


@lru_cache(maxsize=None)
def cg_exact_integer(j1, m1, j2, m2, j3, m3):
    """Return integer-angular-momentum CG coefficients as exact radical sums."""
    j1 = int(j1)
    m1 = int(m1)
    j2 = int(j2)
    m2 = int(m2)
    j3 = int(j3)
    m3 = int(m3)
    if m1 + m2 != m3:
        return ExactRadical.rational(0)
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return ExactRadical.rational(0)
    if j3 < abs(j1 - j2) or j3 > j1 + j2:
        return ExactRadical.rational(0)

    prefactor = Fraction(
        (2 * j3 + 1)
        * factorial(j1 + j2 - j3)
        * factorial(j1 - j2 + j3)
        * factorial(-j1 + j2 + j3)
        * factorial(j1 + m1)
        * factorial(j1 - m1)
        * factorial(j2 + m2)
        * factorial(j2 - m2)
        * factorial(j3 + m3)
        * factorial(j3 - m3),
        factorial(j1 + j2 + j3 + 1),
    )

    z_min = max(0, j2 - j3 - m1, j1 - j3 + m2)
    z_max = min(j1 + j2 - j3, j1 - m1, j2 + m2)
    total = Fraction(0)
    for z in range(int(z_min), int(z_max) + 1):
        denominator = (
            factorial(z)
            * factorial(j1 + j2 - j3 - z)
            * factorial(j1 - m1 - z)
            * factorial(j2 + m2 - z)
            * factorial(j3 - j2 + m1 + z)
            * factorial(j3 - j1 - m2 + z)
        )
        total += Fraction(-1 if z % 2 else 1, denominator)
    return ExactRadical.rational(total) * ExactRadical.sqrt(prefactor)


@lru_cache(maxsize=None)
def cg_exact_half_integer(j1, m1, j2, m2, j3, m3):
    """Return integer or half-integer SU(2) CG coefficients as exact radical sums."""
    j1 = _half_integer_fraction(j1)
    m1 = _half_integer_fraction(m1)
    j2 = _half_integer_fraction(j2)
    m2 = _half_integer_fraction(m2)
    j3 = _half_integer_fraction(j3)
    m3 = _half_integer_fraction(m3)

    if m1 + m2 != m3:
        return ExactRadical.rational(0)
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return ExactRadical.rational(0)
    if j3 < abs(j1 - j2) or j3 > j1 + j2:
        return ExactRadical.rational(0)

    parity_args = (
        j1 + m1,
        j1 - m1,
        j2 + m2,
        j2 - m2,
        j3 + m3,
        j3 - m3,
        j1 + j2 - j3,
        j1 - j2 + j3,
        -j1 + j2 + j3,
        j1 + j2 + j3 + 1,
        2 * j3 + 1,
    )
    if not all(_is_integral(value) for value in parity_args):
        return ExactRadical.rational(0)

    prefactor = Fraction(
        int(2 * j3 + 1)
        * _factorial_int(j1 + j2 - j3)
        * _factorial_int(j1 - j2 + j3)
        * _factorial_int(-j1 + j2 + j3)
        * _factorial_int(j1 + m1)
        * _factorial_int(j1 - m1)
        * _factorial_int(j2 + m2)
        * _factorial_int(j2 - m2)
        * _factorial_int(j3 + m3)
        * _factorial_int(j3 - m3),
        _factorial_int(j1 + j2 + j3 + 1),
    )

    z_min = max(Fraction(0), j2 - j3 - m1, j1 - j3 + m2)
    z_max = min(j1 + j2 - j3, j1 - m1, j2 + m2)
    total = Fraction(0)
    for z_value in range(_ceil_fraction(z_min), _floor_fraction(z_max) + 1):
        z = Fraction(z_value, 1)
        denominator = (
            _factorial_int(z)
            * _factorial_int(j1 + j2 - j3 - z)
            * _factorial_int(j1 - m1 - z)
            * _factorial_int(j2 + m2 - z)
            * _factorial_int(j3 - j2 + m1 + z)
            * _factorial_int(j3 - j1 - m2 + z)
        )
        total += Fraction(-1 if z_value % 2 else 1, denominator)
    return ExactRadical.rational(total) * ExactRadical.sqrt(prefactor)


def cg_exact(j1, m1, j2, m2, j3, m3):
    """Exact CG coefficient for integer and half-integer angular momenta."""
    values = (j1, m1, j2, m2, j3, m3)
    if all(int(x) == x for x in values):
        return cg_exact_integer(*(int(x) for x in values))
    return cg_exact_half_integer(*(_half_integer_fraction(value) for value in values))


@lru_cache(maxsize=None)
def cg_numeric_integer(j1, m1, j2, m2, j3, m3):
    """Return integer-angular-momentum CG coefficients as a Python complex."""
    j1 = int(j1)
    m1 = int(m1)
    j2 = int(j2)
    m2 = int(m2)
    j3 = int(j3)
    m3 = int(m3)
    if m1 + m2 != m3:
        return 0.0 + 0.0j
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0 + 0.0j
    if j3 < abs(j1 - j2) or j3 > j1 + j2:
        return 0.0 + 0.0j

    prefactor_num = (
        (2 * j3 + 1)
        * factorial(j1 + j2 - j3)
        * factorial(j1 - j2 + j3)
        * factorial(-j1 + j2 + j3)
        * factorial(j1 + m1)
        * factorial(j1 - m1)
        * factorial(j2 + m2)
        * factorial(j2 - m2)
        * factorial(j3 + m3)
        * factorial(j3 - m3)
    )
    prefactor_den = factorial(j1 + j2 + j3 + 1)

    z_min = max(0, j2 - j3 - m1, j1 - j3 + m2)
    z_max = min(j1 + j2 - j3, j1 - m1, j2 + m2)
    total = 0.0
    for z in range(int(z_min), int(z_max) + 1):
        denominator = (
            factorial(z)
            * factorial(j1 + j2 - j3 - z)
            * factorial(j1 - m1 - z)
            * factorial(j2 + m2 - z)
            * factorial(j3 - j2 + m1 + z)
            * factorial(j3 - j1 - m2 + z)
        )
        total += (-1.0 if z % 2 else 1.0) / denominator
    return complex(sqrt(prefactor_num / prefactor_den) * total)


@lru_cache(maxsize=None)
def cg_numeric_half_integer(j1, m1, j2, m2, j3, m3):
    """Return integer or half-integer SU(2) CG coefficients as a Python complex."""
    j1 = _half_integer_fraction(j1)
    m1 = _half_integer_fraction(m1)
    j2 = _half_integer_fraction(j2)
    m2 = _half_integer_fraction(m2)
    j3 = _half_integer_fraction(j3)
    m3 = _half_integer_fraction(m3)

    if m1 + m2 != m3:
        return 0.0 + 0.0j
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0 + 0.0j
    if j3 < abs(j1 - j2) or j3 > j1 + j2:
        return 0.0 + 0.0j
    parity_args = (
        j1 + m1,
        j1 - m1,
        j2 + m2,
        j2 - m2,
        j3 + m3,
        j3 - m3,
        j1 + j2 - j3,
        j1 - j2 + j3,
        -j1 + j2 + j3,
        j1 + j2 + j3 + 1,
    )
    if not all(_is_integral(value) for value in parity_args):
        return 0.0 + 0.0j

    prefactor_num = (
        (2 * j3 + 1)
        * _factorial_int(j1 + j2 - j3)
        * _factorial_int(j1 - j2 + j3)
        * _factorial_int(-j1 + j2 + j3)
        * _factorial_int(j1 + m1)
        * _factorial_int(j1 - m1)
        * _factorial_int(j2 + m2)
        * _factorial_int(j2 - m2)
        * _factorial_int(j3 + m3)
        * _factorial_int(j3 - m3)
    )
    prefactor_den = _factorial_int(j1 + j2 + j3 + 1)

    z_min = max(Fraction(0), j2 - j3 - m1, j1 - j3 + m2)
    z_max = min(j1 + j2 - j3, j1 - m1, j2 + m2)
    if not (_is_integral(z_min) and _is_integral(z_max)):
        return 0.0 + 0.0j

    total = 0.0
    for z in range(int(z_min), int(z_max) + 1):
        z = Fraction(z, 1)
        denominator = (
            _factorial_int(z)
            * _factorial_int(j1 + j2 - j3 - z)
            * _factorial_int(j1 - m1 - z)
            * _factorial_int(j2 + m2 - z)
            * _factorial_int(j3 - j2 + m1 + z)
            * _factorial_int(j3 - j1 - m2 + z)
        )
        total += (-1.0 if int(z) % 2 else 1.0) / denominator
    return complex(sqrt(float(prefactor_num / prefactor_den)) * total)


def cg_numeric(j1, m1, j2, m2, j3, m3):
    """Numeric CG coefficient for integer and half-integer angular momenta."""
    values = (j1, m1, j2, m2, j3, m3)
    if all(int(x) == x for x in values):
        return cg_numeric_integer(*(int(x) for x in values))
    return cg_numeric_half_integer(*(_half_integer_fraction(value) for value in values))


def wigner_3j_exact(j1, j2, j3, m1, m2, m3):
    """Exact Wigner-3j coefficient derived from the native exact CG kernel."""
    j1 = _half_integer_fraction(j1)
    j2 = _half_integer_fraction(j2)
    j3 = _half_integer_fraction(j3)
    m1 = _half_integer_fraction(m1)
    m2 = _half_integer_fraction(m2)
    m3 = _half_integer_fraction(m3)
    if m1 + m2 + m3 != 0:
        return ExactRadical.rational(0)
    exponent = j1 - j2 - m3
    if not _is_integral(exponent):
        return ExactRadical.rational(0)
    sign = -1 if int(exponent) % 2 else 1
    return (
        ExactRadical.rational(sign)
        * cg_exact(j1, m1, j2, m2, j3, -m3)
        / ExactRadical.sqrt(2 * j3 + 1)
    )


def wigner_3j_numeric(j1, j2, j3, m1, m2, m3):
    """Numeric Wigner-3j coefficient derived from the native numeric CG kernel."""
    j1 = _half_integer_fraction(j1)
    j2 = _half_integer_fraction(j2)
    j3 = _half_integer_fraction(j3)
    m1 = _half_integer_fraction(m1)
    m2 = _half_integer_fraction(m2)
    m3 = _half_integer_fraction(m3)
    if m1 + m2 + m3 != 0:
        return 0.0 + 0.0j
    exponent = j1 - j2 - m3
    if not _is_integral(exponent):
        return 0.0 + 0.0j
    sign = -1.0 if int(exponent) % 2 else 1.0
    return complex(
        sign
        * cg_numeric(j1, m1, j2, m2, j3, -m3)
        / sqrt(float(2 * j3 + 1))
    )


__all__ = [
    "cg_exact",
    "cg_exact_half_integer",
    "cg_exact_integer",
    "cg_numeric",
    "cg_numeric_half_integer",
    "cg_numeric_integer",
    "wigner_3j_exact",
    "wigner_3j_numeric",
]
