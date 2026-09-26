"""Small exact scalar kernels for runtime-free algebraic coefficients."""

from fractions import Fraction
from functools import lru_cache
from math import gcd


def _as_fraction(value):
    if isinstance(value, Fraction):
        return value
    return Fraction(value)


@lru_cache(maxsize=4096)
def _squarefree_factor(value):
    value = int(value)
    if value < 0:
        raise ValueError("squarefree factorization expects nonnegative integers")
    if value in {0, 1}:
        return 1, value
    if value.bit_length() > 32:
        from ye3t._optional_sympy import sp

        factors = sp.factorint(value)
        square = 1
        squarefree = 1
        for factor, exponent in factors.items():
            factor = int(factor)
            exponent = int(exponent)
            square *= factor ** (exponent // 2)
            if exponent % 2:
                squarefree *= factor
        if square * square * squarefree != value:
            raise ArithmeticError("exact square-free factorization failed")
        return int(square), int(squarefree)
    square = 1
    rest = value
    factor = 2
    while factor * factor <= rest:
        factor_square = factor * factor
        while rest % factor_square == 0:
            square *= factor
            rest //= factor_square
        factor += 1 if factor == 2 else 2
    return int(square), int(rest)


def _normalize_radicand(radicand):
    radicand = _as_fraction(radicand)
    if radicand < 0:
        raise ValueError("negative radical coefficients are not supported")
    if radicand == 0:
        return Fraction(0), Fraction(0)
    numerator = int(radicand.numerator)
    denominator = int(radicand.denominator)
    numerator_square, numerator_squarefree = _squarefree_factor(numerator)
    denominator_square, denominator_squarefree = _squarefree_factor(denominator)
    return (
        Fraction(
            numerator_square,
            denominator_square * denominator_squarefree,
        ),
        Fraction(numerator_squarefree * denominator_squarefree, 1),
    )


class ExactRadical:
    """Finite sums of rational multiples of square roots of squarefree integers."""

    def __init__(self, terms=None):
        cleaned = {}
        for radicand, coefficient in dict(terms or {}).items():
            coefficient = _as_fraction(coefficient)
            if coefficient == 0:
                continue
            scale, normalized_radicand = _normalize_radicand(radicand)
            coefficient *= scale
            if coefficient == 0:
                continue
            cleaned[normalized_radicand] = cleaned.get(normalized_radicand, Fraction(0)) + coefficient
        self.terms = {
            radicand: coefficient
            for radicand, coefficient in sorted(cleaned.items(), key=lambda item: (item[0].numerator, item[0].denominator))
            if coefficient != 0
        }
        self._canonical_snapshot = tuple(self.terms.items())

    @classmethod
    def _from_normalized_terms(cls, terms):
        """Build from an internally proved square-free term map."""

        cleaned = {
            _as_fraction(radicand): _as_fraction(coefficient)
            for radicand, coefficient in dict(terms or {}).items()
            if coefficient != 0
        }
        result = cls.__new__(cls)
        result.terms = {
            radicand: coefficient
            for radicand, coefficient in sorted(
                cleaned.items(),
                key=lambda item: (item[0].numerator, item[0].denominator),
            )
            if coefficient != 0
        }
        result._canonical_snapshot = tuple(result.terms.items())
        return result

    def _canonical_terms(self):
        current = tuple(self.terms.items())
        if current == getattr(self, "_canonical_snapshot", None):
            return self.terms
        return ExactRadical(self.terms).terms

    @classmethod
    def rational(cls, value):
        value = _as_fraction(value)
        if value == 0:
            return cls()
        return cls({Fraction(1): value})

    @classmethod
    def sqrt(cls, value):
        value = _as_fraction(value)
        if value == 0:
            return cls()
        return cls({value: Fraction(1)})

    def is_zero(self):
        return not self.terms

    def evalf(self):
        return float(self)

    def conjugate(self):
        return self

    def __float__(self):
        total = 0.0
        for radicand, coefficient in self.terms.items():
            total += float(coefficient) * float(radicand) ** 0.5
        return float(total)

    def __complex__(self):
        return complex(float(self), 0.0)

    def __bool__(self):
        return bool(self.terms)

    def __neg__(self):
        return self._from_normalized_terms(
            {
                radicand: -coefficient
                for radicand, coefficient in self._canonical_terms().items()
            }
        )

    def __add__(self, other):
        other = exact_scalar(other)
        out = dict(self._canonical_terms())
        for radicand, coefficient in other._canonical_terms().items():
            out[radicand] = out.get(radicand, Fraction(0)) + coefficient
        return self._from_normalized_terms(out)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-exact_scalar(other))

    def __rsub__(self, other):
        return exact_scalar(other) - self

    def __mul__(self, other):
        other = exact_scalar(other)
        out = {}
        for left_radicand, left_coefficient in self._canonical_terms().items():
            for right_radicand, right_coefficient in other._canonical_terms().items():
                left = int(left_radicand.numerator)
                right = int(right_radicand.numerator)
                common = gcd(left, right)
                radicand = Fraction(
                    (left // common) * (right // common), 1
                )
                coefficient = left_coefficient * right_coefficient * common
                out[radicand] = out.get(radicand, Fraction(0)) + coefficient
        return self._from_normalized_terms(out)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = exact_scalar(other)
        other_terms = other._canonical_terms()
        if not other_terms:
            raise ZeroDivisionError("division by zero ExactRadical")
        if len(other_terms) == 2 and Fraction(1) in other_terms:
            rational_coefficient = other_terms[Fraction(1)]
            radical_items = [
                (radicand, coefficient)
                for radicand, coefficient in other_terms.items()
                if radicand != 1
            ]
            radical_radicand, radical_coefficient = radical_items[0]
            conjugate = self._from_normalized_terms(
                {
                    Fraction(1): rational_coefficient,
                    radical_radicand: -radical_coefficient,
                }
            )
            norm = rational_coefficient * rational_coefficient - radical_coefficient * radical_coefficient * radical_radicand
            if norm == 0:
                raise ZeroDivisionError("division by zero ExactRadical")
            return (self * conjugate) / ExactRadical.rational(norm)
        if len(other_terms) != 1:
            raise ValueError("ExactRadical division currently supports single-radical divisors only")
        divisor_radicand, divisor_coefficient = next(iter(other_terms.items()))
        if divisor_coefficient == 0:
            raise ZeroDivisionError("division by zero ExactRadical")
        out = {}
        divisor = int(divisor_radicand.numerator)
        for radicand, coefficient in self._canonical_terms().items():
            numerator = int(radicand.numerator)
            common = gcd(numerator, divisor)
            remaining_divisor = divisor // common
            normalized_radicand = Fraction(
                (numerator // common) * remaining_divisor, 1
            )
            out[normalized_radicand] = out.get(
                normalized_radicand, Fraction(0)
            ) + coefficient / divisor_coefficient / remaining_divisor
        return self._from_normalized_terms(out)

    def __eq__(self, other):
        try:
            other = exact_scalar(other)
        except (TypeError, ValueError):
            return False
        return self.terms == other.terms

    def __hash__(self):
        return hash(tuple(self.terms.items()))

    def __repr__(self):
        if not self.terms:
            return "ExactRadical(0)"
        return f"ExactRadical({self.terms!r})"

    def stable_key(self):
        return tuple(
            (
                int(radicand.numerator),
                int(radicand.denominator),
                int(coefficient.numerator),
                int(coefficient.denominator),
            )
            for radicand, coefficient in self.terms.items()
        )

    def _sympy_(self):
        from ye3t._optional_sympy import sp

        total = sp.Integer(0)
        for radicand, coefficient in self.terms.items():
            total += sp.Rational(coefficient.numerator, coefficient.denominator) * sp.sqrt(
                sp.Rational(radicand.numerator, radicand.denominator)
            )
        return total


def exact_scalar(value):
    if isinstance(value, ExactRadical):
        return value
    if isinstance(value, int):
        return ExactRadical.rational(value)
    if isinstance(value, Fraction):
        return ExactRadical.rational(value)
    if isinstance(value, float):
        return ExactRadical.rational(Fraction(value).limit_denominator())
    if value == 0:
        return ExactRadical.rational(0)
    if getattr(value, "is_Rational", False):
        return ExactRadical.rational(Fraction(int(value.p), int(value.q)))
    if getattr(value, "is_Add", False):
        total = ExactRadical.rational(0)
        for arg in value.args:
            total += exact_scalar(arg)
        return total
    if getattr(value, "is_Mul", False):
        result = ExactRadical.rational(1)
        for arg in value.args:
            result *= exact_scalar(arg)
        return result
    if getattr(value, "is_Pow", False):
        base, exponent = value.as_base_exp()
        if getattr(exponent, "is_Rational", False) and int(exponent.p) == 1 and int(exponent.q) == 2:
            if getattr(base, "is_Rational", False):
                return ExactRadical.sqrt(Fraction(int(base.p), int(base.q)))
        if exponent == 1:
            return exact_scalar(base)
    raise TypeError(f"Unsupported exact scalar value: {value!r}")


__all__ = ["ExactRadical", "exact_scalar"]
