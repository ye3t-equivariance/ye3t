"""Exact product coordinates for the existing ChebExpCos radial evaluator.

Inside the common cutoff let x be its exponential warp and h its cosine
envelope. R_0=1, R_1=h and R_n=h*(1-T_n(x))/2 for n>1. Every nonempty
product has the common zero extension. These coordinates do not change the
radial family or add fitted features.

Chebyshev polynomials: DLMF 18.5.1 and 18.9.1. Polynomial products are exact.
For a real nonzero warp parameter, polynomials in x,h are independent after
radial evaluation: x=A+B*exp(-lambda*r/rc), B!=0, and
h=(2+exp(i*pi*r/rc)+exp(-i*pi*r/rc))/4. Expansion gives distinct exponential
rates indexed by the warp power and Fourier frequency. Their finite
Vandermonde derivative matrix is nonsingular. The affine substitution in x
and substitution h=(2+y+y**-1)/4 are injective polynomial maps. This proof
requires one common profile on an edge; it does not certify mixed profiles.
"""
from fractions import Fraction
from functools import lru_cache
import math

import sympy as sp

from .orthogonal_shifted_jacobi import _multiply_polynomials


def _chebexp_cos_radial_identity(n, cutoff, warp):
    """Canonical physical function identity for nonconstant source channels.

    R1=h is independent of the exponential warp. Higher n retain that warp.
    This is an identity of the existing radial specification, not a new basis.
    """
    if (type(n) is not int or n < 1 or type(cutoff) not in {int, float}
            or type(warp) not in {int, float} or not math.isfinite(cutoff)
            or cutoff <= 0 or not math.isfinite(warp)):
        raise ValueError("ChebExpCos channel identity needs n>=1, positive finite cutoff and finite warp")
    return n, float(cutoff), 0. if n == 1 else float(warp)


@lru_cache(maxsize=512)
def chebexp_cos_radial_product(indices):
    """Return (envelope power, warp power, coefficient) for a nonempty product."""
    if not indices or any(type(n) is not int or n < 0 for n in indices):
        raise ValueError("ChebExpCos products require nonnegative integer radial indices")
    x = sp.Symbol("x")
    polynomial, envelope = (Fraction(1),), 0
    for n in indices:
        if n == 0:
            continue
        envelope += 1
        if n == 1:
            continue
        basis = sp.Poly((1-sp.chebyshevt(n, x))/2, x)
        coefficients = tuple(Fraction(basis.nth(i)) for i in range(n+1))
        polynomial = _multiply_polynomials(polynomial, coefficients)
    return tuple((envelope, power, value) for power, value in enumerate(polynomial) if value)
