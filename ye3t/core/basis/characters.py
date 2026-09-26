
"""Low-level angular coupling-character helpers used by exact basis construction."""

from fractions import Fraction


def _angular_fraction(value):
    frac = Fraction(value)
    if frac.denominator not in {1, 2}:
        raise ValueError(f"Angular momentum must be integer or half-integer, got {value!r}.")
    if frac < 0:
        raise ValueError(f"Angular momentum must be nonnegative, got {value!r}.")
    return frac


def _angular_label(value):
    value = Fraction(value)
    if value.denominator == 1:
        return int(value)
    return value


def cg_allowed(L1, L2):
    """Allowed total angular momenta from coupling two SO(3)/SU(2) irreps."""
    left = _angular_fraction(L1)
    right = _angular_fraction(L2)
    start = abs(left - right)
    stop = left + right
    out = []
    current = start
    while current <= stop:
        out.append(_angular_label(current))
        current += 1
    return tuple(out)


def sym_square_allowed(J):
    """Allowed L in Sym^2(V_J) for integer J: only even L survive."""
    return tuple(range(0, 2 * J + 1, 2))
