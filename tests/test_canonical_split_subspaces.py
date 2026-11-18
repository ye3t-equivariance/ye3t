"""Coupling labels must identify actual intermediate subgroup subspaces.

These tests distinguish a valid basis of a whole isotypic space from a basis
adapted to the named tree. Whole-space orthogonality cannot detect this defect.
"""

import pytest
import sympy as sp
from ye3t.exact_scalars import _exact_matrix_product

from ye3t._record import record_replace
from ye3t.representations.projectors import adjacent_transposition_representation_matrix
from ye3t.representations.young_orthogonal import (
    young_orthogonal_nary_subduction,
    validate_young_orthogonal_nary_subduction,
)


@pytest.mark.parametrize("tree,generator", [("left", 0), ("right", 1), ("balanced", 0)])
def test_named_two_factor_subspace_is_the_actual_subspace(tree, generator):
    tensor = young_orthogonal_nary_subduction(((1,),) * 3, (2, 1), bracketing=tree)
    coefficients = tensor.coefficient_matrix()
    identity = tensor.coset_reps.index((0, 1, 2))
    action = adjacent_transposition_representation_matrix((2, 1), generator)
    for copy, label in enumerate(tensor.lr_chain_labels):
        intermediate, = [p.parts for p in label.intermediate_partitions if p.size == 2]
        sign = 1 if tuple(intermediate) == (2,) else -1
        vector = coefficients[identity, 2 * copy:2 * copy + 2].T
        assert vector != sp.zeros(2, 1)
        assert (action * vector - sign * vector).applyfunc(sp.simplify) == sp.zeros(2, 1)


def test_diagnostic_rejects_wrong_intermediate_labels_on_valid_columns():
    tensor = young_orthogonal_nary_subduction(((1,),) * 3, (2, 1), bracketing="left")
    wrong = record_replace(tensor, lr_chain_labels=tuple(reversed(tensor.lr_chain_labels)))
    report = validate_young_orthogonal_nary_subduction(wrong)
    assert report.orthonormal
    assert report.generator_equivariant
    assert not report.passed
    assert not report.intermediate_subspaces_match


@pytest.mark.parametrize("count,expected", [(3, 2), (4, 2), (5, 4), (6, 4), (7, 4), (8, 4), (9, 8)])
def test_canonical_adjacent_pair_topology_carries_odd_child(count, expected):
    from ye3t.core.basis.tree import coupling_tree_split
    assert coupling_tree_split(count) == expected


def test_lightweight_topology_matches_existing_tree_factory():
    from ye3t._coupling_tree import coupling_tree_split
    from ye3t.core.basis.tree import BalancedPairwiseTreeFactory
    for count in range(2, 258):
        assert coupling_tree_split(count) == BalancedPairwiseTreeFactory.build_hom_shape(count).left.size


def test_permutation_constructor_does_not_load_tensor_runtime():
    import subprocess
    import sys
    code = """
import builtins
import sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] == 'torch':
        raise AssertionError('Permutation-only construction imported PyTorch')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from ye3t.representations.young_orthogonal import young_orthogonal_nary_subduction
result = young_orthogonal_nary_subduction(((1,),)*3, (2,1))
assert result.multiplicity == 2
assert 'torch' not in sys.modules
"""
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_node_supports_include_unequal_child_ranks():
    from ye3t.representations.young_orthogonal import littlewood_richardson_chain_labels
    label, = littlewood_richardson_chain_labels(
        ((2,), (1,), (3,)), (6,), bracketing="balanced")
    assert tuple((support, p.parts) for support, p in label.intermediate_subgroups) == (
        ((0, 1, 2), (3,)), (tuple(range(6)), (6,)))


def test_compact_warm_reuse_does_not_reconstruct_coefficients(monkeypatch):
    from ye3t.couplings import young_coset_transport as compact
    from ye3t.representations import young_orthogonal as young
    compact._direct_restricted_young_values_cached.cache_clear()
    first = compact.direct_restricted_young_values(((2,), (1,)), (2, 1))
    monkeypatch.setattr(young, "_canonical_split_matrices",
                        lambda *args: pytest.fail("warm use rebuilt the split coefficients"))
    assert compact.direct_restricted_young_values(((2,), (1,)), (2, 1)) is first
    # Serialized inputs share the same immutable cache entry and conversion.
    assert compact.direct_restricted_young_values([[2], [1]], [2, 1]) is first


def test_warm_partition_records_are_shared_and_invalid_inputs_still_fail():
    from ye3t.representations.young_orthogonal import _coerce_partition
    first = _coerce_partition((3, 2, 1))
    assert _coerce_partition([3, 2, 1]) is first
    for invalid in ((1, 2), (2, 0), (2, -1)):
        with pytest.raises(ValueError):
            _coerce_partition(invalid)


def test_public_binary_and_general_split_share_coordinates_without_a_solve(monkeypatch):
    from ye3t.representations import young_orthogonal as young
    young._young_orthogonal_induced_coupling_cached.cache_clear()
    def forbidden(*args, **kwargs):
        pytest.fail("Public binary coupling used the superseded equation solver")
    assert not hasattr(young, "_exact_restricted_intertwiner_basis")
    monkeypatch.setattr(young, "_exact_nary_restricted_intertwiner_basis", forbidden)
    monkeypatch.setattr(young, "_reference_subgroup_intertwiner_basis", forbidden)
    binary = young.young_orthogonal_induced_coupling((2,1),(2,1),(3,2,1))
    general = young.young_orthogonal_nary_subduction(((2,1),(2,1)),(3,2,1))
    assert binary.multiplicity == general.multiplicity == 2
    assert binary.coset_reps == general.coset_reps
    assert binary.coefficient_matrix() == general.coefficient_matrix()


@pytest.mark.parametrize("children,target", [
    (((1,),) * 5, (4, 1)),
    (((2,), (1,), (2,), (1,)), (4, 2)),
    (((2, 1), (2, 1)), (3, 2, 1)),
])
def test_general_split_components_and_projector_match_independent_reference(children, target):
    actual = young_orthogonal_nary_subduction(children, target)
    reference = young_orthogonal_nary_subduction(
        children, target, coefficient_backend="intertwiner_oracle")
    candidate = actual.coefficient_matrix()
    expected = reference.coefficient_matrix()
    assert candidate.shape == expected.shape
    assert (_exact_matrix_product(candidate.T, candidate) - sp.eye(candidate.cols)).applyfunc(sp.simplify) == sp.zeros(candidate.cols)
    # Compare via overlaps rather than constructing the ambient projector.
    overlap = _exact_matrix_product(expected.T, candidate)
    assert (_exact_matrix_product(expected, overlap) - candidate).applyfunc(sp.simplify) == sp.zeros(*candidate.shape)
    assert (_exact_matrix_product(overlap.T, overlap) - sp.eye(candidate.cols)).applyfunc(sp.simplify) == sp.zeros(candidate.cols)
