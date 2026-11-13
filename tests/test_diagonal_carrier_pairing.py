"""Exact pairing speed path retains its gauge and uniqueness requirements."""
import pytest

from ye3t.couplings import lifted_cauchy_scalar as scalar
from ye3t.representations.projectors import adjacent_transposition_representation_matrix


def _normalized(matrix):
    sp = scalar._sympy()
    matrix = (matrix / sp.sqrt((matrix.T * matrix).trace())).applyfunc(sp.simplify)
    first = next(value for value in matrix if value != 0)
    return -matrix if first.could_extract_minus_sign() else matrix


def _nullspace_reference(roles, angles):
    sp = scalar._sympy()
    dimension = roles[0].rows
    equations = sp.Matrix.vstack(*(sp.kronecker_product(left, right) - sp.eye(dimension**2)
                                  for left, right in zip(roles, angles)))
    vectors = equations.nullspace()
    assert len(vectors) == 1
    return _normalized(vectors[0].reshape(dimension, dimension))


@pytest.mark.parametrize('partition', [(2, 1), (2, 2), (3, 1), (3, 2)])
def test_diagonal_pairing_matches_independent_full_nullspace(partition):
    sp = scalar._sympy()
    actions = tuple(adjacent_transposition_representation_matrix(partition, i)
                    for i in range(sum(partition)-1))
    dimension = actions[0].rows
    role_gauge = sp.diag(*(sp.Integer(i+1) for i in range(dimension)))
    angle_gauge = sp.diag(*((-1)**i * sp.sqrt(i+2) for i in range(dimension)))
    roles = tuple(role_gauge * action * role_gauge.inv() for action in actions)
    angles = tuple(angle_gauge * action * angle_gauge.inv() for action in actions)
    candidate = scalar._diagonal_carrier_pairing(roles, angles)
    assert candidate is not None
    reference = _nullspace_reference(roles, angles)
    assert _normalized(candidate) == reference
    assert scalar._invariant_pairing(roles, angles) == reference


@pytest.mark.parametrize('partition', [(6, 2), (5, 2, 1), (4, 3, 1)])
def test_rank_eight_pairing_uses_complete_tableau_space_without_nullspace(partition, monkeypatch):
    from sympy.polys.matrices import DomainMatrix
    sp = scalar._sympy()
    actions = tuple(adjacent_transposition_representation_matrix(partition, i) for i in range(7))
    dimension = actions[0].rows
    role_gauge = sp.diag(*(sp.Integer(i+1) for i in range(dimension)))
    angle_gauge = sp.diag(*(sp.Rational(1, i+2) for i in range(dimension)))
    roles = tuple(role_gauge * action * role_gauge.inv() for action in actions)
    angles = tuple(angle_gauge * action * angle_gauge.inv() for action in actions)

    def forbidden(*args, **kwargs):
        raise AssertionError('The complete diagonal-gauge carrier must avoid the general nullspace')

    monkeypatch.setattr(DomainMatrix, 'nullspace', forbidden)
    scalar._invariant_pairing_cached.cache_clear()
    actual = scalar._invariant_pairing(roles, angles)
    assert actual == _normalized(role_gauge * angle_gauge.T)
    for left, right in zip(roles, angles):
        assert (left * actual * right.T - actual).applyfunc(sp.simplify) == sp.zeros(dimension)


def test_general_gauge_retains_exact_nullspace_path():
    sp = scalar._sympy()
    actions = tuple(adjacent_transposition_representation_matrix((2, 1), i) for i in range(2))
    gauge = sp.Matrix([[1, 1], [0, 2]])
    angles = tuple(gauge * action * gauge.inv() for action in actions)
    assert scalar._diagonal_carrier_pairing(actions, angles) is None
    assert scalar._invariant_pairing(actions, angles) == _nullspace_reference(actions, angles)


def test_nonunique_pairing_is_still_rejected():
    sp = scalar._sympy()
    actions = (sp.eye(2),)
    assert scalar._diagonal_carrier_pairing(actions, actions) is None
    with pytest.raises(RuntimeError, match='one invariant pairing'):
        scalar._invariant_pairing(actions, actions)


def test_incompatible_one_dimensional_pairing_is_still_rejected():
    sp = scalar._sympy()
    with pytest.raises(RuntimeError, match='one invariant pairing'):
        scalar._invariant_pairing((sp.Matrix([[1]]),), (sp.Matrix([[-1]]),))
