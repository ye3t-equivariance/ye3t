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
