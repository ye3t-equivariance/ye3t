"""Exact real-form substitution, including repeated rank-eight inputs."""

from itertools import product
from collections import defaultdict

import pytest
import sympy as sp

from ye3t.couplings import covariant_cauchy as core


def symbolic_block_contraction(block_rows, outer_terms):
    # Previous exact implementation is retained only as a numerical oracle.
    terms = defaultdict(lambda: sp.Integer(0))
    for magnetic, outer in outer_terms.items():
        selected = [block_rows[b][m] for b, m in enumerate(magnetic)]
        for choices in product(*[tuple(row.items()) for row in selected]):
            coordinates = tuple(c for choice in choices for c in choice[0])
            terms[coordinates] += sp.simplify(outer*sp.prod(choice[1] for choice in choices))
    return terms


@pytest.mark.parametrize('value', [
    (sp.sqrt(2)+sp.sqrt(3))*(sp.sqrt(2)-sp.sqrt(3)),
    (sp.sqrt(5)+sp.sqrt(7))*sp.sqrt(6)/13,
    sp.Float('0.33333333333333331',50),
    sp.I*(sp.sqrt(2)+sp.sqrt(3)),
    sp.sin(sp.pi/7),
    (sp.Symbol('x',real=True)+1)**2,
])
def test_exact_scalar_simplification_preserves_general_and_floating_values(value):
    actual = core._scalar._simplified_scalar(value)
    expected = sp.simplify(value)
    assert sp.simplify(actual-expected) == 0
    if value.has(sp.Float):
        assert actual == expected and actual.has(sp.Float)


def test_complete_payload_equals_general_symbolic_simplification(monkeypatch):
    request = core.covariant_cauchy_request([
        {'neighbor_species':'X','radial_channel':0,'l':1,'source_family_id':'test'},
        {'neighbor_species':'X','radial_channel':1,'l':2,'source_family_id':'test'}],
        (2,1),target_L=1,role_dimension=2)
    actual = core.compile_covariant_cauchy(request)
    monkeypatch.setattr(core._scalar,'_simplified_scalar',sp.simplify)
    expected = core.compile_covariant_cauchy(request)
    assert actual == expected


def test_exact_block_products_preserve_radical_sums_cancellation_and_factor_order():
    left = {0: {((0, 0, 0),)*4: sp.sqrt(2)+sp.sqrt(3)},
            1: {((0, 0, 0),)*4: sp.sqrt(2)+sp.sqrt(3)}}
    right = {0: {((1, 1, 0),)*4: sp.sqrt(2)-sp.sqrt(3),
                 ((1, 1, 1),)*4: sp.sqrt(6)/5}}
    for outer in ({(0,0):sp.sqrt(7)}, {(0,0):sp.sqrt(7),(1,0):-sp.sqrt(7)}):
        actual = core._scalar._combine_block_rows([left,right],outer)
        expected = symbolic_block_contraction([left,right],outer)
        assert actual.keys() == expected.keys()
        assert all(sp.expand(actual[key]-expected[key]) == 0 for key in actual)
        assert all(len(key) == 8 and key[:4] == ((0,0,0),)*4 for key in actual)


@pytest.mark.parametrize('target', [0,1,2])
def test_compiled_block_contraction_payload_equals_symbolic_reference(target, monkeypatch):
    request = core.covariant_cauchy_request([
        {'neighbor_species':'X','radial_channel':0,'l':1,'source_family_id':'test'},
        {'neighbor_species':'X','radial_channel':1,'l':2,'source_family_id':'test'}],
        (2,1),target_L=target,role_dimension=2)
    actual = core.compile_covariant_cauchy(request)
    monkeypatch.setattr(core._scalar,'_combine_block_rows',symbolic_block_contraction)
    expected = core.compile_covariant_cauchy(request)
    assert actual == expected
    assert core.validate_covariant_cauchy(actual)


def cartesian_substitution(options):
    # Previous ordered expansion is deliberately retained only as a reference.
    return tuple((tuple(sorted(item[0] for item in choice)),
                  sp.prod(item[1] for item in choice))
                 for choice in product(*options))


@pytest.mark.parametrize("angular", [(1, 2), (2, 4)])
@pytest.mark.parametrize("magnetic", [(0, 0), (1, -1), (-1, 1)])
def test_repeated_rank_eight_substitution_matches_independent_polynomial(angular, magnetic):
    options = []
    variables = {}
    expression = sp.Integer(1)
    for channel, (ell, m) in enumerate(zip(angular, magnetic)):
        form = core._scalar._real_form_matrix(ell)
        for role in (0, 1):
            choices = tuple(((channel, role, j), form[m+ell, j])
                            for j in range(2*ell+1) if form[m+ell, j] != 0)
            factor = 0
            for coordinate, value in choices:
                variable = variables.setdefault(coordinate, sp.Symbol('x_'+'_'.join(map(str, coordinate))))
                factor += value*variable
            options.extend([choices, choices])
            expression *= factor**2
    options = tuple(options)
    actual = dict(core._real_monomial_substitution(options))
    actual_expression = sum(value*sp.prod(variables[c] for c in monomial)
                            for monomial, value in actual.items())
    assert sp.Poly(sp.expand(actual_expression-expression), *variables.values()).is_zero
    assert all(value != 0 for value in actual.values())
    assert all(len(monomial) == 8 for monomial in actual)
    # Repeated calls reuse immutable substitutions and cannot share mutable rows.
    actual.clear()
    assert core._real_monomial_substitution(options)


@pytest.mark.parametrize("target", [0, 1, 2])
def test_complete_compiled_payload_matches_previous_ordered_conversion(target, monkeypatch):
    request = core.covariant_cauchy_request([
        {"neighbor_species": "X", "radial_channel": 0, "l": 1, "source_family_id": "test"},
        {"neighbor_species": "X", "radial_channel": 1, "l": 2, "source_family_id": "test"}],
        (2, 1), target_L=target, role_dimension=2)
    actual = core.compile_covariant_cauchy(request)
    monkeypatch.setattr(core, '_real_monomial_substitution', cartesian_substitution)
    expected = core.compile_covariant_cauchy(request)
    assert actual == expected
    assert core.validate_covariant_cauchy(actual)


def test_opposite_magnetic_factors_cancel_exactly():
    form = core._scalar._real_form_matrix(1)
    options = tuple(tuple(((0, 0, j), form[m, j])
                          for j in range(3) if form[m, j] != 0)
                    for m in (0, 2))
    assert dict(core._real_monomial_substitution(options)) == {
        ((0, 0, 0), (0, 0, 0)): -sp.Rational(1, 2),
        ((0, 0, 2), (0, 0, 2)): -sp.Rational(1, 2)}
