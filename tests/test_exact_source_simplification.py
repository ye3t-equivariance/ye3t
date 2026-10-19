"""Exact source rows against explicit tensor-product coordinate sums."""
from collections import defaultdict
from itertools import product

import pytest

from ye3t._optional_sympy import sp
from ye3t.couplings import covariant_cauchy as covariant
from ye3t.couplings import lifted_cauchy_scalar as scalar
from ye3t.couplings.tagged_cauchy_carriers import cg_exact_integer


def test_coalesced_terms_preserve_exact_cancellation_and_coordinate_order():
    x = sp.Symbol("x", real=True)
    terms = {
        ((1, 0), (0, 1)): sp.sin(x)**2,
        ((0, 1), (1, 0)): sp.cos(x)**2-1,
        ((2, 0), (0, 1)): sp.sqrt(8),
        ((0, 1), (2, 0)): -sp.sqrt(2),
        ((3, 0), (0, 1)): sp.sqrt(8)-sp.sqrt(2),
        ((4, 0), (0, 1)): sp.Integer(0),
    }
    assert scalar._coalesce_terms(terms) == {
        ((0, 1), (2, 0)): sp.sqrt(2),
        ((0, 1), (3, 0)): sp.sqrt(2),
    }
    assert list(scalar._coalesce_terms(terms)) == [((0, 1), (2, 0)), ((0, 1), (3, 0))]


@pytest.mark.parametrize("left_l,right_l,L", [(1, 1, 0), (1, 1, 1), (1, 1, 2),
                                              (1, 2, 1), (1, 2, 2), (1, 2, 3)])
def test_real_source_rows_match_full_tensor_product_sum(left_l, right_l, L):
    channels = [{"channel_index": 0, "l": left_l}, {"channel_index": 1, "l": right_l}]
    request = {"channels": channels, "target": {"L": L}}
    phase = (left_l+right_l-L) % 2
    canonical = {}
    for M in range(-L, L+1):
        canonical[M] = {((0, 0, m+left_l), (1, 1, n+right_l)): sp.sympify(cg_exact_integer(left_l, m, right_l, n, L, M))
            for m in range(-left_l, left_l+1) for n in range(-right_l, right_l+1)
            if m+n == M}
    left, right = scalar._real_form_matrix(left_l), scalar._real_form_matrix(right_l)
    parent = (-sp.I)**phase*scalar._real_form_matrix(L).conjugate().T
    expected = []
    for component in range(2*L+1):
        row = defaultdict(lambda: sp.Integer(0))
        for a, b in product(range(2*left_l+1), range(2*right_l+1)):
            value = sum(parent[component, M+L]*coefficient*left[coordinates[0][2], a]*right[coordinates[1][2], b]
                for M, terms in canonical.items() for coordinates, coefficient in terms.items())
            value = sp.simplify(sp.expand(value))
            if value != 0:
                assert sp.im(value) == 0
                row[((0, 0, a), (1, 1, b))] = value
        expected.append(dict(row))
    assert covariant._real_rows(canonical, request, phase) == tuple(expected)


def test_real_source_rows_reject_nonreal_scalar_coefficient():
    request = {"channels": [{"channel_index": 0, "l": 0}], "target": {"L": 0}}
    with pytest.raises(RuntimeError, match="not exactly real"):
        covariant._real_rows({0: {((0, 0, 0),): sp.I}}, request, 0)
