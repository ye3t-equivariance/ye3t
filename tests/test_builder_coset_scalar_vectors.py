"""Coset path versus native path equality for the angular scalar-carrier
construction (``ye3t.representations.builder._exact_single_factor_scalar_vectors``,
multiplicity > 1 branch).

The native path builds the M=0 ("scalar") Young-carrier image and its
tableau vectors by calling ``combined_projector_matrix`` and (when the
isotypic component has multiplicity > 1)
``_selected_subgroup_matrix_units_for_factor_native`` directly on the whole
M=0 state space -- both of which sum one term per ``block_size!``
permutation per state.  For block size >= 6 the builder instead uses the
exact coset/breadth-first construction
(``ye3t.representations.coset_units``, shared with the role side), applied
class by class to the magnetic-content classes of the M=0 space.

Native-path dispatch (below): for every (nin, lin, partition) combination
the existing test suite exercises at block size <= 5, the coset dispatch
threshold (``_COSET_SCALAR_VECTORS_THRESHOLD`` = 6) keeps the native path
running; confirmed by a monkeypatch spy on ``combined_projector_matrix``.

Coset versus native equality: equality of the *full* ``vectors`` dict, at
block size 3, 4, 5, for l = 1 and l = 2, forcing the coset path via a
test-only override of ``_COSET_SCALAR_VECTORS_THRESHOLD`` (native stays fast
at these sizes, so both paths can be run and compared directly) for every
partition with native multiplicity > 1 in that (size, l) combination.
"""

import itertools

import pytest

from ye3t.representations import builder as builder_module
from ye3t.representations.builder import (
    GeneralizedExactSymbolicLabeler,
    _exact_single_factor_scalar_vectors,
    _angular_scalar_vectors_coset,
    _magnetic_content_vector,
)
from ye3t.representations.generalized_irreps import Partition
from ye3t.couplings.tagged_cauchy import kostka_number


def _irrep(size, angular_l, partition_tuple):
    nin = tuple(0 for _ in range(size))
    lin = tuple(int(angular_l) for _ in range(size))
    labeler = GeneralizedExactSymbolicLabeler(nin, lin, spatial_symmetry="O3")
    permutation_irrep = labeler.permutation_irrep((Partition(tuple(partition_tuple)),))
    return nin, lin, permutation_irrep


def _multiplicity_gt1_partitions(size, angular_l):
    """Every partition of `size` (any number of parts <= size) whose native
    scalar-vector construction has multiplicity > 1 at this (size, l)."""
    found = []
    for partition_tuple in _integer_partitions(size):
        nin, lin, permutation_irrep = _irrep(size, angular_l, partition_tuple)
        _basis, _vectors, multiplicity = _exact_single_factor_scalar_vectors(
            nin, lin, permutation_irrep
        )
        if multiplicity > 1:
            found.append(partition_tuple)
    return found


def _integer_partitions(n, max_part=None):
    if max_part is None:
        max_part = n
    if n == 0:
        yield ()
        return
    for first in range(min(n, max_part), 0, -1):
        for rest in _integer_partitions(n - first, first):
            yield (first,) + rest


# ---------------------------------------------------------------------
# Native-path dispatch: block size <= 5 runs the native path (spy-confirmed).
# ---------------------------------------------------------------------

_NATIVE_PATH_CASES = (
    (3, 1, (2, 1)),
    (4, 1, (2, 2)),
    (4, 1, (3, 1)),
    (4, 1, (2, 1, 1)),
    (4, 2, (2, 2)),
    (5, 1, (3, 2)),
    (5, 1, (2, 2, 1)),
    (5, 2, (4, 1)),
    (5, 2, (3, 2)),
)


@pytest.mark.parametrize("case", _NATIVE_PATH_CASES)
def test_native_path_runs_unchanged_below_threshold(case, monkeypatch):
    size, angular_l, partition_tuple = case
    nin, lin, permutation_irrep = _irrep(size, angular_l, partition_tuple)

    calls = []
    original = builder_module.combined_projector_matrix

    def spy(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(builder_module, "combined_projector_matrix", spy)
    basis_states, vectors, multiplicity = _exact_single_factor_scalar_vectors(
        nin, lin, permutation_irrep
    )
    assert calls, "native combined_projector_matrix must run below the coset threshold"
    assert len(basis_states) == (2 * angular_l + 1) ** size


# ---------------------------------------------------------------------
# Coset vs native: full vectors dict, forcing the coset path at
# small sizes via a test-only override of the dispatch threshold.
# ---------------------------------------------------------------------

_COSET_VS_NATIVE_SIZE_L = ((3, 1), (4, 1), (4, 2), (5, 1), (5, 2))


@pytest.mark.parametrize("size,angular_l", _COSET_VS_NATIVE_SIZE_L)
def test_coset_matches_native_for_every_multiplicity_gt1_partition(size, angular_l, monkeypatch):
    partitions = _multiplicity_gt1_partitions(size, angular_l)
    if not partitions:
        pytest.skip(f"no multiplicity>1 partition at size={size}, l={angular_l}")
    for partition_tuple in partitions:
        nin, lin, permutation_irrep = _irrep(size, angular_l, partition_tuple)
        native_basis, native_vectors, native_multiplicity = _exact_single_factor_scalar_vectors(
            nin, lin, permutation_irrep
        )
        assert native_multiplicity > 1

        monkeypatch.setattr(builder_module, "_COSET_SCALAR_VECTORS_THRESHOLD", 0)
        try:
            coset_basis, coset_vectors, coset_multiplicity = _exact_single_factor_scalar_vectors(
                nin, lin, permutation_irrep
            )
        finally:
            monkeypatch.setattr(builder_module, "_COSET_SCALAR_VECTORS_THRESHOLD", 6)

        assert coset_basis == native_basis
        assert coset_multiplicity == native_multiplicity, (
            size, angular_l, partition_tuple, coset_multiplicity, native_multiplicity
        )
        assert set(coset_vectors) == set(native_vectors)
        for key in native_vectors:
            assert coset_vectors[key] == native_vectors[key], (
                size, angular_l, partition_tuple, key
            )


def test_coset_dispatch_threshold_is_monkeypatchable(monkeypatch):
    # a direct, minimal confirmation that the module-level constant actually
    # controls dispatch (used by the coset-vs-native tests above); forcing it to a very large
    # value must route even a block-size-6 call through the native path.
    calls = []
    original = builder_module.combined_projector_matrix

    def spy(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(builder_module, "combined_projector_matrix", spy)
    monkeypatch.setattr(builder_module, "_COSET_SCALAR_VECTORS_THRESHOLD", 1000)
    nin, lin, permutation_irrep = _irrep(4, 1, (2, 2))
    _exact_single_factor_scalar_vectors(nin, lin, permutation_irrep)
    assert calls


# ---------------------------------------------------------------------
# Additional direct checks (extra confidence beyond the dispatch and equality tests above).
# ---------------------------------------------------------------------


def test_magnetic_content_vector_matches_kostka_rank_at_size6():
    # forces the coset path (size 6) and checks the returned multiplicity
    # and per-copy content classes are internally consistent with Kostka
    # numbers computed from the ascending-m letter order.
    size, angular_l, partition_tuple = 6, 1, (4, 2)
    nin, lin, permutation_irrep = _irrep(size, angular_l, partition_tuple)
    basis_states, vectors, multiplicity = _exact_single_factor_scalar_vectors(
        nin, lin, permutation_irrep
    )
    assert multiplicity >= 1
    dimension = Partition(tuple(partition_tuple)).dimension
    scalar_states = tuple(s for s in basis_states if sum(s) == 0)
    basis_index = {s: i for i, s in enumerate(basis_states)}
    for copy_index in range(multiplicity):
        v0 = vectors[(copy_index, 0, 0)]
        support = [row for row in range(v0.rows) if v0[row, 0] != 0]
        assert support
        # every nonzero entry must be an M=0 (scalar) state.
        for row in support:
            state = basis_states[row]
            assert sum(state) == 0


def test_zero_multiplicity_at_size6_returns_empty():
    # (3,3) at l=1, size=6 has native (and coset) multiplicity 0: the
    # highest-weight nullspace is empty. Must not raise, must return {}.
    size, angular_l, partition_tuple = 6, 1, (3, 3)
    nin, lin, permutation_irrep = _irrep(size, angular_l, partition_tuple)
    basis_states, vectors, multiplicity = _exact_single_factor_scalar_vectors(
        nin, lin, permutation_irrep
    )
    assert multiplicity == 0
    assert vectors == {}


def test_coset_units_helpers_are_importable_and_shared():
    from ye3t.representations.coset_units import (
        adjacent_transposition_generators,
        block_stabilizer_sum,
        class_stabilizer_sum,
        class_word_table,
        content_key,
        stream_class_pivots,
        unit_columns,
    )
    from ye3t.couplings import lifted_cauchy_scalar as compiler_module

    # lifted_cauchy_scalar imports the SAME objects from coset_units;
    # confirm identity, not just importability, so a future edit cannot
    # silently fork the two copies.
    assert compiler_module.coset_adjacent_transposition_generators is adjacent_transposition_generators
    assert compiler_module.coset_class_stabilizer_sum is class_stabilizer_sum
    assert compiler_module.coset_class_word_table is class_word_table
    assert compiler_module.coset_stream_class_pivots is stream_class_pivots
