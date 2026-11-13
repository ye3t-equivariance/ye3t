"""Exact requested-L coset construction against the full-projector oracle."""

import pytest
import sympy as sp

from ye3t.representations import builder
from ye3t.couplings.lifted_cauchy_scalar import (
    _angular_schur_vectors, _build_block_template, _validate_block_template,
)
from ye3t.representations.generalized_irreps import Partition
from ye3t.representations.projectors import permute_state_slots


@pytest.mark.parametrize("size,angular_l,partition,output_L", [
    (3, 1, (2, 1), 1), (4, 1, (2, 2), 0),
    (5, 2, (3, 2), 2), (6, 2, (4, 2), 0),
])
def test_packed_exact_columns_preserve_all_copies_and_components(size, angular_l, partition, output_L):
    from array import array
    from ye3t.couplings.lifted_cauchy_scalar import _sparse_carrier_columns

    options = dict(angular_basis_backend="exact_weight_space_v1")
    states, expected, count = _angular_schur_vectors(size, angular_l, partition, output_L, **options)
    actual_states, actual, actual_count = _angular_schur_vectors(
        size, angular_l, partition, output_L, compact_vectors=True, **options)
    assert actual_states == states
    assert actual_count == count
    assert isinstance(actual, builder._PackedExactColumns)
    assert tuple(actual) == tuple(expected)
    assert len(actual) == count * Partition(partition).dimension * (2 * output_L + 1)
    for key in actual:
        assert actual[key] == expected[key]
        assert all(isinstance(buffer, array) for buffer in actual._columns[key])
    assert _sparse_carrier_columns(actual[key] for key in actual) == sp.SparseMatrix.hstack(*expected.values())
    key = next(iter(actual))
    owned = actual[key]
    owned[0, 0] += 1
    assert actual[key] == expected[key]


def test_packed_columns_reject_coordinate_loss_and_invalid_shapes():
    packed = builder._PackedExactColumns(3)
    packed["a"] = sp.SparseMatrix([sp.sqrt(2), 0, sp.sqrt(2)])
    packed["b"] = sp.SparseMatrix([0, sp.sqrt(2), 0])
    assert packed._scalars == [sp.sqrt(2)]
    packed.reorder(("b", "a"))
    assert tuple(packed) == ("b", "a")
    with pytest.raises(ValueError, match="coordinates"):
        packed.reorder(("a", "a"))
    with pytest.raises(ValueError, match="shape"):
        packed["c"] = sp.ones(2, 1)


@pytest.mark.parametrize('size,partition,output_L', [(3, (2, 1), 1), (4, (3, 1), 2), (4, (2, 2), 0)])
def test_checked_packed_actions_match_raw_permutations_at_every_m(size, partition, output_L):
    states, vectors, count = _angular_schur_vectors(size, 1, partition, output_L,
        angular_basis_backend='exact_weight_space_v1', compact_vectors=True)
    assert len(vectors._checked_actions) == count*(2*output_L+1)
    index = {state: row for row, state in enumerate(states)}
    for keys, actions in vectors._checked_actions.items():
        basis = sp.SparseMatrix.hstack(*(vectors[key] for key in keys))
        for adjacent, action in enumerate(actions):
            permutation = list(range(size))
            permutation[adjacent], permutation[adjacent+1] = permutation[adjacent+1], permutation[adjacent]
            rows = tuple(index[permute_state_slots(state, tuple(range(size)), permutation)] for state in states)
            assert (basis*action-basis.extract(rows, range(basis.cols))).applyfunc(sp.simplify) == sp.zeros(*basis.shape)
    # Changing even one owned column discards the linked construction proof.
    first = next(iter(vectors))
    vectors[first] = vectors[first]
    assert vectors._checked_actions == {}


@pytest.mark.parametrize("size,angular_l,partition,output_L", [
    (2, 1, (1, 1), 1), (3, 1, (2, 1), 1),
    (3, 2, (2, 1), 2), (2, 3, (1, 1), 3),
])
def test_requested_weight_space_matches_exact_legacy_span(size, angular_l, partition, output_L, monkeypatch):
    states, reference, expected_count = _angular_schur_vectors(size, angular_l, partition, output_L)
    def forbidden(*args, **kwargs):
        raise AssertionError("Requested-weight construction invoked the full-projector backend.")
    monkeypatch.setattr(builder.GeneralizedExactSymbolicLabeler, "sector_for_partitions", forbidden)
    monkeypatch.setattr(builder, "combined_projector_matrix", forbidden)
    actual_states, actual, count = _angular_schur_vectors(size, angular_l, partition, output_L,
                                                        angular_basis_backend="exact_weight_space_v1")
    assert actual_states == states
    assert count == expected_count
    dimension = Partition(partition).dimension
    for magnetic in range(-output_L, output_L+1):
        columns = sp.Matrix.hstack(*(actual[copy, tableau, magnetic]
            for copy in range(count) for tableau in range(dimension)))
        oracle = sp.Matrix.hstack(*(reference[copy, tableau, magnetic]
            for copy in range(count) for tableau in range(dimension)))
        assert columns.rank() == count*dimension
        assert columns.row_join(oracle).rank() == columns.cols
        for copy in range(count):
            for tableau in range(dimension):
                vector = actual[copy, tableau, magnetic]
                assert isinstance(vector, sp.MutableSparseMatrix)
                top = actual[copy, tableau, output_L]
                assert sp.simplify((vector.T*vector)[0]-(top.T*top)[0]) == 0
        # Slot permutations commute with the ladder: action in the copy/Young
        # coordinates must agree at every magnetic component.
        if size > 1:
            permutation = (1, 0, *range(2, size))
            index = {state: i for i, state in enumerate(states)}
            permuted = sp.zeros(columns.rows, columns.cols)
            for i, state in enumerate(states):
                permuted[index[permute_state_slots(state, tuple(range(size)), permutation)], :] = columns[i, :]
            action = (columns.T*columns).inv()*columns.T*permuted
            assert columns*action == permuted
            if magnetic == -output_L:
                first_action = action
            else:
                assert (action-first_action).applyfunc(sp.simplify) == sp.zeros(action.rows)
    for magnetic in range(-output_L, output_L+1):
        source_states = tuple(state for state in states if sum(state) == magnetic)
        target_states = tuple(state for state in states if sum(state) == magnetic+1)
        raising = builder._weight_raising_matrix((angular_l,)*size, source_states, target_states)
        for copy in range(count):
            for tableau in range(dimension):
                source = sp.Matrix([actual[copy, tableau, magnetic][states.index(state)] for state in source_states])
                expected = sp.zeros(len(target_states), 1) if magnetic == output_L else sp.Matrix([
                    actual[copy, tableau, magnetic+1][states.index(state)] for state in target_states
                ])*sp.sqrt((output_L-magnetic)*(output_L+magnetic+1))
                assert (raising*source-expected).applyfunc(sp.simplify) == sp.zeros(len(target_states), 1)


def test_requested_weight_block_retains_exact_metric_adjoint():
    key = (2, 2, (1, 1), 1, 1)
    block = _build_block_template(key, angular_basis_backend="exact_weight_space_v1")
    _validate_block_template(block["payload"])
    assert block["payload"]["validation_report"]["angular_source"].endswith("requested_weight_space_coset")
    assert block["payload"]["template_id"] != _build_block_template(key)["payload"]["template_id"]


def test_requested_weight_raising_and_lowering_keep_sparse_operands(monkeypatch):
    reference = _angular_schur_vectors(
        3, 1, (2, 1), 1, angular_basis_backend="exact_weight_space_v1")
    original = builder._weight_raising_matrix
    exact_product = builder._exact_matrix_product
    multiplications = []
    exact_multiplications = []

    class SparseWeightOperator(sp.MutableSparseMatrix):
        def __mul__(self, other):
            if isinstance(other, sp.MatrixBase):
                assert isinstance(other, (sp.MutableSparseMatrix, sp.ImmutableSparseMatrix))
                multiplications.append((self.shape, other.shape))
            return super().__mul__(other)

        def _eval_transpose(self):
            return SparseWeightOperator(super()._eval_transpose())

    def sparse_operator(*args, **kwargs):
        return SparseWeightOperator(original(*args, **kwargs))

    def sparse_exact_product(matrix, other):
        assert isinstance(matrix, (sp.MutableSparseMatrix, sp.ImmutableSparseMatrix))
        assert isinstance(other, (sp.MutableSparseMatrix, sp.ImmutableSparseMatrix))
        exact_multiplications.append((matrix.shape, other.shape))
        return exact_product(matrix, other)

    monkeypatch.setattr(builder, "_weight_raising_matrix", sparse_operator)
    monkeypatch.setattr(builder, "_exact_matrix_product", sparse_exact_product)
    actual = _angular_schur_vectors(
        3, 1, (2, 1), 1, angular_basis_backend="exact_weight_space_v1")
    assert actual[0] == reference[0]
    assert actual[2] == reference[2]
    assert actual[1].keys() == reference[1].keys()
    for key, vector in actual[1].items():
        assert vector == reference[1][key]
    assert multiplications
    # Lowering from M=1 to M=0 increases the weight-space dimension.
    assert any(shape[0] > shape[1] for shape, _ in exact_multiplications)
    # One highest-weight check and two lowering steps each include both tableaux.
    assert len(exact_multiplications) == 3
    assert all(shape[1] == 2 for _, shape in exact_multiplications)
