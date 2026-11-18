"""Independent qualification for selected-column LR and Kronecker construction."""

import pytest
import sympy as sp

# Preserve one visible warning without retaining hundreds of thousands of
# identical third-party deprecations during exact reference simplification.
# This changes warning capture only, never the mathematical assertions.
pytestmark = pytest.mark.filterwarnings(
    "once:bitcount function is deprecated:DeprecationWarning:mpmath.libmp.libintmath")

from ye3t.representations import projectors as p
from ye3t.representations.kronecker_intertwiners import (
    _projector_kronecker_intertwiners, _exact_kronecker_intertwiners_general,
)
from ye3t.representations.young_orthogonal import (
    _reference_binary_split_matrices, _projector_binary_split_matrices,
)


def assert_exact(actual, expected):
    assert actual.shape == expected.shape
    assert all(sp.simplify(x) == 0 for x in actual - expected)


def check_copies(maps, ambient_generators, component_generators):
    for i, matrix in enumerate(maps):
        for j, other in enumerate(maps):
            assert_exact(matrix.T * other,
                         sp.eye(matrix.cols) if i == j else sp.zeros(matrix.cols))
        for ambient, component in zip(ambient_generators, component_generators, strict=True):
            assert_exact(ambient * matrix, matrix * component)


def check_old_conversion(new, old):
    dimension = new[0].cols
    change = sp.Matrix(len(new), len(old), lambda i, j:
                       sp.simplify(sp.trace(new[i].T * old[j]) / dimension))
    assert_exact(change.T * change, sp.eye(len(old)))
    for j, matrix in enumerate(old):
        assert_exact(sum((new[i] * change[i, j] for i in range(len(new))),
                         sp.zeros(*matrix.shape)), matrix)


def test_selected_columns_match_supplied_lr_projector_minor():
    parent = (3, 2, 1)
    priority = p._tableau_construction_schedule(parent)[3]
    def action(factor, generator, value):
        return p.adjacent_transposition_representation_matrix(parent, factor * 3 + generator) * value
    columns = p._tableau_projector_columns(
        action, ((2, 1), (2, 1)), sp.eye(16)[:, [priority[2], priority[3]]])
    gram = columns[[priority[2], priority[3]], :]
    assert_exact(gram, sp.Matrix([[7, -sp.sqrt(3)], [-sp.sqrt(3), 9]]) / 32)


def test_lr_two_copies_all_components_and_old_coordinate_conversion():
    left = right = (2, 1)
    parent = (3, 2, 1)
    maps = _projector_binary_split_matrices(left, right, parent)
    assert len(maps) == 2 and all(m.shape == (16, 4) for m in maps)
    ambient = [p.adjacent_transposition_representation_matrix(parent, i) for i in (0, 1, 3, 4)]
    child = [p.adjacent_transposition_representation_matrix(left, i) for i in (0, 1)]
    component = [sp.kronecker_product(a, sp.eye(2)) for a in child]
    component += [sp.kronecker_product(sp.eye(2), a) for a in child]
    check_copies(maps, ambient, component)
    check_old_conversion(maps, _reference_binary_split_matrices(left, right, parent))


def test_kronecker_two_copies_all_components_and_old_coordinate_conversion():
    parent, left = (3, 2), (3, 1, 1)
    result = _projector_kronecker_intertwiners(parent, left, left)
    assert result["multiplicity"] == 2
    maps = result["intertwiners"]
    assert all(m.shape == (36, 5) for m in maps)
    child = [p.adjacent_transposition_representation_matrix(left, i) for i in range(4)]
    ambient = [sp.kronecker_product(a, a) for a in child]
    component = [p.adjacent_transposition_representation_matrix(parent, i) for i in range(4)]
    check_copies(maps, ambient, component)
    check_old_conversion(maps, _exact_kronecker_intertwiners_general(parent, left, left)["intertwiners"])


@pytest.mark.parametrize("parent,left,right", [
    ((2, 2), (3, 1), (3, 1)), ((2, 1), (2, 1), (2, 1)),
    ((3,), (2, 1), (2, 1)), ((1, 1, 1), (2, 1), (2, 1)),
])
def test_projector_single_copy_agrees_with_old_span(parent, left, right):
    new = _projector_kronecker_intertwiners(parent, left, right)["intertwiners"]
    check_old_conversion(new, _exact_kronecker_intertwiners_general(parent, left, right)["intertwiners"])


def test_no_generic_solver_or_full_group_enumeration(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("generic solve or full-group construction entered")
    for name in ("nullspace", "eigenvects", "QRdecomposition", "pinv", "inv"):
        monkeypatch.setattr(sp.matrices.matrixbase.MatrixBase, name, forbidden)
    monkeypatch.setattr(p, "all_permutations", forbidden)
    monkeypatch.setattr(p, "canonical_irrep_matrices", forbidden)
    _projector_binary_split_matrices.cache_clear()
    _projector_kronecker_intertwiners.cache_clear()
    assert len(_projector_binary_split_matrices((2, 1), (2, 1), (3, 2, 1))) == 2
    assert _projector_kronecker_intertwiners((3, 2), (3, 1, 1), (3, 1, 1))["multiplicity"] == 2


def test_public_dispatch_uses_shared_projector_and_preserves_fast_paths(monkeypatch):
    from ye3t.representations import kronecker_intertwiners as kron
    from ye3t.representations import young_orthogonal as young
    def forbidden(*args, **kwargs):
        raise AssertionError("Production entered a historical generic constructor")
    kron.exact_kronecker_intertwiners.cache_clear()
    monkeypatch.setattr(kron, "_exact_kronecker_intertwiners_general", forbidden)
    monkeypatch.setattr(young, "_reference_binary_split_matrices", forbidden)
    monkeypatch.setattr(young, "_reference_subgroup_intertwiner_basis", forbidden)
    assert kron.exact_kronecker_intertwiners((3,2), (3,1,1), (3,1,1))["gauge"] == p.TABLEAU_PROJECTOR_COPY_CONVENTION
    assert young._binary_split_matrices is young._projector_binary_split_matrices
    assert len(young._binary_split_matrices((2,1), (2,1), (3,2,1))) == 2
    with monkeypatch.context() as fast:
        fast.setattr(kron, "_projector_kronecker_intertwiners", forbidden)
        kron.exact_kronecker_intertwiners.cache_clear()
        for target, left, right in (((3,2,1), (3,2,1), (6,)),
                                    ((3,2,1), (3,2,1), (1,)*6),
                                    ((6,), (3,2,1), (3,2,1))):
            assert kron.exact_kronecker_intertwiners(target, left, right)["multiplicity"] == 1


def test_approved_higher_copy_normalization_uses_no_generic_solver(monkeypatch):
    projector = sp.eye(4) - sp.ones(4) / 4
    def forbidden(*args, **kwargs):
        raise AssertionError("local normalization invoked a generic matrix solver")
    for name in ("nullspace", "eigenvects", "QRdecomposition", "cholesky", "pinv", "inv", "rref"):
        monkeypatch.setattr(sp.matrices.matrixbase.MatrixBase, name, forbidden)
    monkeypatch.setattr(p, "_tableau_projector_columns",
                        lambda action, partitions, columns: projector * columns)
    copies = p._tableau_projector_copies(None, (), 4, 3, range(4))
    coefficients = sp.Matrix.hstack(*copies)
    assert_exact(coefficients.T * coefficients, sp.eye(3))
    assert_exact(coefficients * coefficients.T, projector)


@pytest.mark.parametrize("size", range(2, 7))
def test_binary_complete_decomposition_through_rank_six(size):
    from ye3t.couplings import integer_partitions
    from ye3t.representations.generalized_irreps import Partition
    from ye3t.representations.young_orthogonal import littlewood_richardson_coefficient
    left_size = size // 2
    right_size = size - left_size
    for shape in integer_partitions(size):
        parent = tuple(shape)
        dimension = len(p.standard_tableaux(parent))
        total = sp.zeros(dimension)
        for left_shape in integer_partitions(left_size):
            for right_shape in integer_partitions(right_size):
                left, right = tuple(left_shape), tuple(right_shape)
                count = littlewood_richardson_coefficient(Partition(left), Partition(right), Partition(parent))
                maps = _projector_binary_split_matrices(left, right, parent)
                assert len(maps) == count
                if not maps:
                    continue
                dl, dr = len(p.standard_tableaux(left)), len(p.standard_tableaux(right))
                component = [sp.kronecker_product(p.adjacent_transposition_representation_matrix(left, i), sp.eye(dr))
                             for i in range(left_size - 1)]
                component += [sp.kronecker_product(sp.eye(dl), p.adjacent_transposition_representation_matrix(right, i))
                              for i in range(right_size - 1)]
                generators = list(range(left_size - 1)) + list(range(left_size, size - 1))
                ambient = [p.adjacent_transposition_representation_matrix(parent, i) for i in generators]
                check_copies(maps, ambient, component)
                for matrix in maps:
                    total += matrix * matrix.T
        assert_exact(total, sp.eye(dimension))


@pytest.mark.parametrize("left,right,parent", [
    ((), (2,1), (2,1)), ((2,1), (), (2,1)), ((), (), ()),
    ((3,), (1,1), (3,1,1)), ((1,1), (3,), (3,1,1)),
])
def test_empty_unequal_and_sign_children(left, right, parent):
    maps = _projector_binary_split_matrices(left, right, parent)
    assert len(maps) == 1
    assert_exact(maps[0].T * maps[0], sp.eye(maps[0].cols))


def test_generator_storage_is_sparse_immutable_and_matches_numeric():
    import numpy as np
    matrix = p.adjacent_transposition_representation_matrix((4, 2, 1), 3)
    assert isinstance(matrix, sp.ImmutableSparseMatrix)
    assert matrix.nnz() <= 2 * matrix.cols
    np.testing.assert_allclose(np.asarray(matrix, dtype=float),
                               p.adjacent_transposition_representation_matrix_numeric((4,2,1), 3),
                               atol=2e-16, rtol=0)
    with pytest.raises(TypeError):
        matrix[0,0] = 0
