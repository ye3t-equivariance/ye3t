"""Formula-based multiplicity utilities for ACE fixed-content sectors.

These functions implement the count side of the blockwise ACE decomposition:

``Sym^k(V_l) = direct_sum_L d_L V_L``

and the final SO(3) coupling of block multiplicity distributions. They do not
construct occupancy basis states, Young projectors, or coefficient tables.
Constructive materialization code may use the same counts as provenance, but
the counting path itself stays formula/recurrence based.
"""

from functools import lru_cache


def _m_values(l):
    return tuple(range(-int(l), int(l) + 1))


def _symmetric_power_weight_counts_l1_closed_form(k_b):
    k_b = int(k_b)
    out = {}
    for M in range(-k_b, k_b + 1):
        out[int(M)] = int((k_b - abs(int(M))) // 2 + 1)
    return out


def _symmetric_power_weight_counts_recurrence(l, k_b):
    weights = _m_values(l)
    polys = [{0: 1}] + [dict() for _ in range(int(k_b))]
    for weight in weights:
        for degree in range(1, int(k_b) + 1):
            source = polys[degree - 1]
            target = polys[degree]
            for exponent, coeff in source.items():
                new_exponent = int(exponent) + int(weight)
                target[new_exponent] = target.get(new_exponent, 0) + int(coeff)
    return {int(weight): int(count) for weight, count in polys[int(k_b)].items() if int(count) != 0}


@lru_cache(maxsize=512)
def symmetric_power_weight_counts(l, k_b):
    """Return exact weight multiplicities for ``Sym^k(V_l)``.

    This is a general integer-angular-momentum recurrence. The ``l=1`` case
    uses the same generating-function result in closed form as an acceleration;
    changing the benchmark or runtime request to ``l=2`` or ``l=3`` uses the
    general recurrence rather than failing or falling back to projector
    materialization.
    """
    l = int(l)
    k_b = int(k_b)
    if k_b < 0:
        raise ValueError("symmetric-power multiplicity must be nonnegative.")
    if k_b == 0:
        return {0: 1}
    if l == 0:
        return {0: 1}
    if l == 1:
        return _symmetric_power_weight_counts_l1_closed_form(k_b)
    return _symmetric_power_weight_counts_recurrence(l, k_b)


def so3_irrep_multiplicities_from_weight_counts(weight_counts, Lmax=None):
    """Recover SO(3) irrep multiplicities from weight multiplicities."""
    if Lmax is None:
        if not weight_counts:
            Lmax = 0
        else:
            Lmax = max(int(weight) for weight in weight_counts)
    d_by_L = {}
    for L in range(int(Lmax), -1, -1):
        multiplicity = int(weight_counts.get(int(L), 0)) - int(weight_counts.get(int(L) + 1, 0))
        if multiplicity > 0:
            d_by_L[int(L)] = int(multiplicity)
    return d_by_L


@lru_cache(maxsize=512)
def symmetric_power_irrep_multiplicities(l, k_b):
    """Return exact ``d_L`` for the decomposition of ``Sym^k(V_l)``."""
    l = int(l)
    k_b = int(k_b)
    weight_counts = symmetric_power_weight_counts(l, k_b)
    return so3_irrep_multiplicities_from_weight_counts(weight_counts, Lmax=l * k_b)


def couple_so3_multiplicity_distributions(left_counts, right_counts):
    """Couple two SO(3) multiplicity distributions by Clebsch-Gordan ranges."""
    if not left_counts:
        return {}
    if not right_counts:
        return {}
    max_l = max(int(left_L) for left_L in left_counts)
    max_r = max(int(right_L) for right_L in right_counts)
    delta = [0] * (max_l + max_r + 2)
    for left_L, left_mult in left_counts.items():
        left_L = int(left_L)
        left_mult = int(left_mult)
        if left_mult == 0:
            continue
        for right_L, right_mult in right_counts.items():
            right_L = int(right_L)
            right_mult = int(right_mult)
            if right_mult == 0:
                continue
            value = left_mult * right_mult
            low = abs(left_L - right_L)
            high = left_L + right_L
            delta[low] += value
            delta[high + 1] -= value
    out = {}
    running = 0
    for L_R, value in enumerate(delta[:-1]):
        running += int(value)
        if running:
            out[int(L_R)] = int(running)
    return out


def couple_block_irrep_multiplicities(block_multiplicities):
    """Return final ``alpha_{L_R}`` from block irrep multiplicity maps."""
    current = {0: 1}
    for d_by_L in block_multiplicities:
        current = couple_so3_multiplicity_distributions(current, d_by_L)
    return dict(sorted((int(L_R), int(value)) for L_R, value in current.items() if int(value) > 0))


def clear_multiplicity_caches():
    symmetric_power_weight_counts.cache_clear()
    symmetric_power_irrep_multiplicities.cache_clear()
