"""Exact source polynomials for R_ql(x)=x**l*(1-x)**3*T_q(2*x-1).

The source is zero for x>=1, with two continuous cutoff derivatives. Physical
evaluation excludes coincident atoms; Cartesian smoothness at x=0 is not an
assumption of this family. No orthogonality or normalization is imposed.

Chebyshev recurrence: NIST DLMF 18.9.1 and Table 18.9.1,
https://dlmf.nist.gov/18.9.T1 . Runtime derivatives differentiate that recurrence.
Exact same-occurrence products retain the additional envelope and origin
powers before triangular polynomial expansion; no radial truncation is used.
"""

from fractions import Fraction
from functools import lru_cache

from .orthogonal_shifted_jacobi import (
    _multiply_polynomials, _one_minus_x_power, _shift_polynomial,
)


CHEBYSHEV_C2_SOURCE_FAMILY = "chebyshev_c2_origin_regular_v1"
CHEBYSHEV_C2_SOURCE_METADATA = {
    "radial_family": CHEBYSHEV_C2_SOURCE_FAMILY,
    "radial_definition": "x^l*(1-x)^3*T_q(2*x-1)",
    "coordinate": "x=r/r_cut_pair",
    "radial_index": "q=0,1,...",
    "normalization": "none",
    "zero_extension": "C2",
    "cutoff_second_derivative": "continuous_zero",
    "physical_domain": "r>0; coincident atoms excluded",
    "product_algebra": "exact_untruncated_shifted_chebyshev_polynomials_v1",
}


@lru_cache(maxsize=512)
def shifted_chebyshev_power_coefficients(degree):
    """Integer power coefficients of T_degree(2*x-1), low power first."""
    degree = int(degree)
    if degree < 0:
        raise ValueError("Chebyshev degree must be nonnegative")
    previous = (1,)
    if degree == 0:
        return previous
    current = (-1, 2)
    for _ in range(1, degree):
        following = [0] * (len(current) + 1)
        for power, value in enumerate(current):
            following[power] -= 2 * value
            following[power + 1] += 4 * value
        for power, value in enumerate(previous):
            following[power] -= value
        previous, current = current, tuple(following)
    return current


def shifted_chebyshev_ladder_with_derivative(maximum_degree, coordinate):
    """Values and d/dx of every T_q(2*x-1) through maximum_degree."""
    maximum_degree = int(maximum_degree)
    if maximum_degree < 0:
        raise ValueError("Chebyshev degree must be nonnegative")
    one = coordinate * 0 + 1
    values, derivatives = [one], [coordinate * 0]
    if maximum_degree:
        scaled = 2 * coordinate - 1
        values.append(scaled)
        derivatives.append(2 * one)
        for _ in range(1, maximum_degree):
            derivatives.append(4 * values[-1] + 2 * scaled * derivatives[-1] - derivatives[-2])
            values.append(2 * scaled * values[-1] - values[-2])
    return tuple(values), tuple(derivatives)


def shifted_chebyshev_expansion(polynomial):
    """Exact triangular expansion of a polynomial in shifted Chebyshev rows."""
    remainder = [Fraction(value) for value in polynomial]
    result = [Fraction(0)] * len(remainder)
    for degree in range(len(remainder) - 1, -1, -1):
        basis = shifted_chebyshev_power_coefficients(degree)
        coefficient = remainder[degree] / basis[-1]
        result[degree] = coefficient
        for power, value in enumerate(basis):
            remainder[power] -= coefficient * value
    if any(remainder):
        raise ArithmeticError("Chebyshev product expansion failed exact reconstruction")
    return tuple(result)


@lru_cache(maxsize=4096)
def chebyshev_c2_radial_product(radial_components, output_l):
    """Expand products of R_ql into the enlarged output-L radial family."""
    if not radial_components:
        raise ValueError("Radial product needs at least one factor")
    output_l = int(output_l)
    origin_power = sum(int(ell) for _, ell in radial_components) - output_l
    if output_l < 0 or origin_power < 0 or any(int(ell) < 0 for _, ell in radial_components):
        raise ValueError("Product angular degree exceeds its source degrees")
    polynomial = (Fraction(1),)
    for degree, _ in radial_components:
        polynomial = _multiply_polynomials(polynomial, shifted_chebyshev_power_coefficients(degree))
    polynomial = _multiply_polynomials(polynomial, _one_minus_x_power(3 * (len(radial_components) - 1)))
    polynomial = _shift_polynomial(polynomial, origin_power)
    return tuple((degree, value) for degree, value in enumerate(shifted_chebyshev_expansion(polynomial)) if value)
