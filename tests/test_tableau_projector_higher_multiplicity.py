"""Higher-copy construction against complete component and group references."""

from itertools import permutations
from math import factorial

import numpy as np
import pytest
import sympy as sp

from ye3t.representations import projectors as p
from ye3t.representations.kronecker_intertwiners import _projector_kronecker_intertwiners
from ye3t.representations.generalized_irreps import Partition
from ye3t.representations.young_orthogonal import _projector_binary_split_matrices
from ye3t.exact_scalars import _exact_matrix_product

pytestmark = pytest.mark.filterwarnings(
    "once:bitcount function is deprecated:DeprecationWarning:mpmath.libmp.libintmath")


@pytest.mark.parametrize("parent,count", [((4,2), 3), ((3,2,1), 5)])
def test_higher_copies_have_complete_actions_and_independent_central_projector(parent, count):
    left = (3,2,1)
    result = _projector_kronecker_intertwiners(parent, left, left)
    copies = result["intertwiners"]
    assert len(copies) == result["multiplicity"] == count
    assert result["gauge"] == "tableau_lex_positions_root2_triangular_v2"
    dimension = len(p.standard_tableaux(parent))
    left_generators = [p.adjacent_transposition_representation_matrix(left, i) for i in range(5)]
    for i, matrix in enumerate(copies):
        assert matrix.shape == (256, dimension)
        for j, other in enumerate(copies):
            error = matrix.T * other - (sp.eye(dimension) if i == j else sp.zeros(dimension))
            assert all(sp.simplify(value) == 0 for value in error)
        for g, action in enumerate(left_generators):
            ambient = sp.kronecker_product(action, action)
            component = p.adjacent_transposition_representation_matrix(parent, g)
            assert all(sp.simplify(value) == 0 for value in ambient * matrix - matrix * component)

    # Independent bounded reference: central character averaging over S6.
    # This intentionally uses all permutations only in this reference test;
    # production uses selected tableau interpolation and no group table.
    numeric_generators = [np.asarray(a, dtype=float) for a in left_generators]
    parent_partition = Partition(parent)
    reference = np.zeros((256, 256))
    for permutation in permutations(range(6)):
        weight = p.symmetric_group_character(parent_partition, p.permutation_cycle_type(permutation))
        if not weight:
            continue
        ordering = list(range(6))
        action = np.eye(16)
        for destination, value in enumerate(permutation):
            position = ordering.index(value)
            while position > destination:
                ordering[position - 1], ordering[position] = ordering[position], ordering[position - 1]
                action = action @ numeric_generators[position - 1]
                position -= 1
        reference += int(weight) * np.kron(action, action)
    reference *= dimension / factorial(6)
    columns = np.concatenate([np.asarray(matrix, dtype=float) for matrix in copies], axis=1)
    np.testing.assert_allclose(columns @ columns.T, reference, atol=1e-12, rtol=1e-12)


def test_lr_three_copies_complete_components_and_subgroup_actions():
    left, right, parent = (3,2,1), (3,1), (4,3,2,1)
    copies = _projector_binary_split_matrices(left, right, parent)
    assert len(copies) == 3
    assert all(matrix.shape == (768, 48) for matrix in copies)
    for i, matrix in enumerate(copies):
        for j, other in enumerate(copies):
            error = matrix.T * other - (sp.eye(48) if i == j else sp.zeros(48))
            assert all(sp.simplify(value) == 0 for value in error.todok().values())
        for generator in tuple(range(5)) + tuple(range(6,9)):
            action = p.adjacent_transposition_representation_matrix(parent, generator)
            if generator < 5:
                component = sp.kronecker_product(
                    p.adjacent_transposition_representation_matrix(left, generator), sp.eye(3))
            else:
                component = sp.kronecker_product(sp.eye(16),
                    p.adjacent_transposition_representation_matrix(right, generator - 6))
            error = action * matrix - matrix * component
            assert all(sp.simplify(value) == 0 for value in error.todok().values())


def test_repeated_child_higher_copy_exchange_reuses_local_normalization(monkeypatch):
    from ye3t.couplings.carrier_products import _exchange_modes, _exchange_trace
    child = (3,2,1)
    maps = _projector_kronecker_intertwiners(child, child, child)["intertwiners"]
    trace = _exchange_trace(child, child)
    def forbidden(*args, **kwargs):
        raise AssertionError("Repeated-child normalization invoked a generic solve")
    for name in ("nullspace", "eigenvects", "QRdecomposition", "cholesky", "pinv", "inv", "rref"):
        monkeypatch.setattr(sp.matrices.matrixbase.MatrixBase, name, forbidden)
    modes = _exchange_modes(maps, 16, 16, trace)
    assert [sign for sign, _ in modes].count(1) == (5+trace)//2
    assert [sign for sign, _ in modes].count(-1) == (5-trace)//2
    for i, (sign, matrix) in enumerate(modes):
        swapped = matrix.extract([j*16+k for k in range(16) for j in range(16)], range(16))
        assert all(sp.simplify(value) == 0 for value in swapped-sign*matrix)
        for j, (_, other) in enumerate(modes):
            # Positive-root two-copy exchange frames can contain nested roots
            # beyond ExactRadical's quadratic domain. Polynomial expansion
            # evaluates the complete Gram exactly without heuristic simplify.
            error = (matrix.T*other).applyfunc(sp.expand)-(sp.eye(16) if i == j else sp.zeros(16))
            assert error == sp.zeros(16)
