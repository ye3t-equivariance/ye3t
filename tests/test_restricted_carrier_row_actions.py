"""Restricted factor actions agree exactly with raw permutation matrices."""

from itertools import product

import pytest

from ye3t._optional_sympy import sp
from ye3t.couplings import lifted_cauchy_scalar as scalar


@pytest.mark.parametrize("size,dimension", [(2, 3), (3, 2), (4, 2)])
def test_row_actions_match_full_matrix_reference(size, dimension):
    states = tuple(product(range(dimension), repeat=size))
    # A complete fixed-content permutation module, with a nonorthogonal
    # coordinate change, exercises every adjacent action and its orientation.
    selected = tuple(index for index, state in enumerate(states) if sum(state) == 1)
    basis = sp.zeros(len(states), len(selected))
    for column, row in enumerate(selected):
        basis[row, column] = column + 1
    for column in range(1, len(selected)):
        basis[selected[0], column] = 1
    actual = scalar._restricted_carrier_actions(states, tuple(basis.col(i) for i in range(basis.cols)), size)
    inverse = basis.extract(selected, tuple(range(basis.cols))).inv()
    for adjacent, action in enumerate(actual):
        permutation = list(range(size))
        permutation[adjacent], permutation[adjacent+1] = permutation[adjacent+1], permutation[adjacent]
        transformed = scalar._raw_permutation_matrix(states, tuple(permutation))*basis
        reference = inverse*transformed.extract(selected, tuple(range(basis.cols)))
        assert action == reference
        assert transformed == basis*action
    scalar._check_coxeter_actions(actual)


def test_row_actions_do_not_construct_raw_square_matrix(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Restricted actions must use basis-row permutation")

    monkeypatch.setattr(scalar, "_raw_permutation_matrix", forbidden)
    states = tuple(product(range(3), repeat=5))
    vector = sp.ones(len(states), 1)
    assert scalar._restricted_carrier_actions(states, (vector,), 5) == (sp.ones(1, 1),)*4


def test_row_actions_reject_dependent_basis():
    states = tuple(product(range(2), repeat=2))
    vector = sp.ones(4, 1)
    with pytest.raises(RuntimeError, match="not independent"):
        scalar._restricted_carrier_actions(states, (vector, vector), 2)


def test_row_actions_reject_nonclosed_basis():
    states = tuple(product(range(2), repeat=2))
    with pytest.raises(RuntimeError, match="not closed"):
        scalar._restricted_carrier_actions(states, (sp.Matrix([0, 1, 0, 0]),), 2)
