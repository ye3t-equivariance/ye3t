"""Exact coset transport needed by compact Young coefficient storage."""

import sympy as sym


def test_identity_coset_block_transports_in_current_young_copy_gauge():
    from ye3t.global_coupler import _build_young_subgroup_specht_coupling
    from ye3t.representations.young_orthogonal import _young_irrep_matrix

    cases = (
        (((1,), (1,), (1,)), (2, 1)),
        (((2, 1), (1,)), (3, 1)),
    )
    for children, parent in cases:
        coupling = _build_young_subgroup_specht_coupling(children, parent)
        matrix = coupling.coefficient_matrix()
        cosets = coupling.tensor.coset_reps
        assert cosets[0] == tuple(range(sum(map(sum, children))))
        rows_per_coset = matrix.rows // len(cosets)
        identity_block = matrix[:rows_per_coset, :]
        for index, representative in enumerate(cosets):
            actual = matrix[index * rows_per_coset:(index + 1) * rows_per_coset, :]
            irrep = _young_irrep_matrix(parent, representative)
            # Multiplicity copies are contiguous; each uses the same tableau
            # action. This comparison also tests coset orientation.
            blocks = tuple(identity_block[:, copy * irrep.rows:(copy + 1) * irrep.rows]
                           * irrep for copy in range(matrix.cols // irrep.rows))
            expected = sym.Matrix.hstack(*blocks)
            assert sym.simplify(actual - expected) == sym.zeros(*actual.shape)


def test_adjacent_generator_lookup_matches_full_irrep_table():
    from ye3t.representations.young_orthogonal import _young_irrep_matrix
    from ye3t.representations.projectors import (
        adjacent_transposition, canonical_irrep_matrices, inverse_permutation,
    )

    for partition in ((2, 1), (3, 1), (2, 2), (3, 2)):
        full = canonical_irrep_matrices(partition)
        for index in range(sum(partition) - 1):
            permutation = adjacent_transposition(sum(partition), index)
            lookup = _young_irrep_matrix(partition, inverse_permutation(permutation))
            assert sym.simplify(lookup - full[permutation]) == sym.zeros(*lookup.shape)


def test_stable_factor_grouping_is_the_canonical_young_coset():
    from ye3t.couplings.young_coset_transport import canonical_factor_coset
    from ye3t.representations.young_orthogonal import (
        _young_subgroup_shuffle_representatives,
    )

    for sizes in ((1, 1), (2, 1), (2, 2), (1, 2, 1), (2, 1, 2)):
        canonical = tuple(block for block, size in enumerate(sizes)
                          for _ in range(size))
        for representative in _young_subgroup_shuffle_representatives(sizes):
            word = [None] * len(canonical)
            for position, source in enumerate(representative):
                word[source] = canonical[position]
            assert canonical_factor_coset(word, canonical) == representative


def test_direct_restricted_graph_matches_full_subduction_gauge():
    from ye3t.global_coupler import _build_young_subgroup_specht_coupling
    from ye3t.couplings.young_coset_transport import direct_restricted_young_values

    cases = (
        (((1,), (1,), (1,)), (2, 1)),
        (((2, 1), (1,)), (3, 1)),
        (((2,), (1,)), (2, 1)),
        (((1,), (1,), (1,), (1,)), (2, 1, 1)),
    )
    for children, parent in cases:
        coupling = _build_young_subgroup_specht_coupling(children, parent)
        full = coupling.coefficient_matrix()
        rows = full.rows // len(coupling.tensor.coset_reps)
        restricted, copies = direct_restricted_young_values(children, parent)
        assert copies == coupling.tensor.multiplicity
        assert sym.Matrix(restricted).shape == (rows, full.cols)
        assert max(abs(float(left - right)) for left, right in zip(
            sym.Matrix(restricted), full[:rows, :]
        )) < 1e-12


def test_rank_eight_direct_young_map_is_isometric_without_full_compiler():
    import numpy as np

    from ye3t.couplings.young_coset_transport import (
        adjacent_generator_columns, direct_restricted_young_values,
        expanded_young_values,
    )
    from ye3t.representations.young_orthogonal import (
        _young_subgroup_shuffle_representatives,
    )

    children, target = ((4,), (4,)), (7, 1)
    identity, copies = direct_restricted_young_values(children, target)
    assert copies == 1
    cosets = _young_subgroup_shuffle_representatives((4, 4))
    assert len(cosets) == 70
    table = {"local_tableau_dim": 1, "gamma_count": copies,
             "target_tableau_dim": 7, "restricted_values": identity,
             "adjacent_generators": adjacent_generator_columns(target)}
    full = expanded_young_values(table, cosets)
    np.testing.assert_allclose(full.T @ full, np.eye(7), atol=1e-12, rtol=1e-12)


def test_sparse_young_transport_float32_matches_float64_on_long_shuffle():
    import torch

    from ye3t.couplings.young_coset_transport import (
        adjacent_generator_columns, direct_restricted_young_values,
        transported_young_slice,
    )

    identity, copies = direct_restricted_young_values(((4,), (4,)), (7, 1))
    table = {"local_tableau_dim": 1, "gamma_count": copies,
             "target_tableau_dim": 7, "restricted_values": identity,
             "adjacent_generators": adjacent_generator_columns((7, 1))}
    representative = (4, 5, 6, 7, 0, 1, 2, 3)
    double = transported_young_slice(table, representative, 0, torch.float64, "cpu")
    single = transported_young_slice(table, representative, 0, torch.float32, "cpu")
    torch.testing.assert_close(single.double(), double, atol=2e-6, rtol=2e-6)


def test_matrix_unit_young_storage_matches_direct_restriction():
    import math
    import torch

    from ye3t.couplings.young_coset_transport import (
        adjacent_generator_columns, direct_restricted_young_values,
        transported_young_slice,
    )

    target = (7, 1)
    dense, copies = direct_restricted_young_values(((1,),) * 8, target)
    assert copies == 7
    compact = {"local_tableau_dim": 1, "gamma_count": 7,
               "target_tableau_dim": 7, "restricted_kind": "matrix_units",
               "restricted_scale": math.sqrt(7 / math.factorial(8)),
               "adjacent_generators": adjacent_generator_columns(target)}
    representative = (7, 6, 5, 4, 3, 2, 1, 0)
    for gamma in range(copies):
        expected = {**compact, "restricted_kind": "explicit",
                    "restricted_values": dense}
        actual = transported_young_slice(compact, representative, gamma,
                                         torch.float64, "cpu")
        full = transported_young_slice(expected, representative, gamma,
                                       torch.float64, "cpu")
        torch.testing.assert_close(actual, full, atol=1e-12, rtol=1e-12)
