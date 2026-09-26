"""Small-case validation helpers for Schur-Weyl partition centralizers.

The routines in this module are intentionally finite and exact.  They are
designed to validate centralizer compression claims on hand-sized tensor powers
of the natural permutation representation of ``S_n``.  They are not runtime
contraction schedules.

Reference: Halverson and Ram, "Partition Algebras", arXiv:math/0401314,
survey partition-algebra Schur-Weyl duality for tensor powers of permutation
representations.  This file implements small validation matrices only; it does
not implement a partition-algebra runtime basis.
"""

from functools import lru_cache
from itertools import product

from ye3t._record import recordclass
from ye3t.exact_linalg import (
    exact_matrix_equal,
    exact_matrix_flatten,
    exact_matrix_from_entries,
    exact_matrix_matmul,
    exact_rational_rank,
)


@recordclass(
    (
        "n",
        "rank",
        "basis_dim",
        "diagram_count",
        "expected_orbit_count",
        "enumerated_orbit_count",
        "span_rank",
        "stable_range",
        "commutes_with_generators",
        "rank_matches_orbit_count",
        "passed",
        "detail",
    ),
    frozen=True,
)
class PartitionCentralizerValidationReport:
    """Exact small-case validation report for a partition centralizer image."""

    detail = ""


def _restricted_growth_partitions(size):
    """Yield set partitions of ``range(size)`` as restricted-growth blocks."""

    if size < 0:
        raise ValueError("size must be nonnegative")
    if size == 0:
        yield ()
        return

    labels = [0]

    def rec(index, max_label):
        if index == size:
            blocks = [[] for _ in range(max_label + 1)]
            for point, label in enumerate(labels):
                blocks[label].append(point)
            yield tuple(tuple(block) for block in blocks)
            return
        for label in range(max_label + 2):
            labels.append(label)
            yield from rec(index + 1, max(max_label, label))
            labels.pop()

    yield from rec(1, 0)


@lru_cache(maxsize=None)
def set_partitions_of_size(size):
    """Return all set partitions of ``range(size)`` in deterministic order."""

    return tuple(_restricted_growth_partitions(int(size)))


def partition_centralizer_orbit_count(n, rank):
    """Count equality-pattern orbits on pairs of ``rank`` tensor indices.

    For the natural permutation representation of ``S_n`` on an ``n`` element
    set, the diagonal action on ``V^{otimes rank} x V^{otimes rank}`` preserves
    exactly the equality pattern among the ``2 * rank`` indices.  Only patterns
    with at most ``n`` blocks can occur.
    """

    n = int(n)
    rank = int(rank)
    if n <= 0:
        raise ValueError("n must be positive")
    if rank < 0:
        raise ValueError("rank must be nonnegative")
    return sum(1 for blocks in set_partitions_of_size(2 * rank) if len(blocks) <= n)


def _equality_pattern(values):
    label_by_value = {}
    labels = []
    next_label = 0
    for value in values:
        if value not in label_by_value:
            label_by_value[value] = next_label
            next_label += 1
        labels.append(label_by_value[value])
    return tuple(labels)


def _tensor_basis(n, rank):
    return tuple(product(range(int(n)), repeat=int(rank)))


def _adjacent_value_transposition_matrix(n, rank, adjacent_index):
    n = int(n)
    rank = int(rank)
    adjacent_index = int(adjacent_index)
    basis = _tensor_basis(n, rank)
    index = {state: pos for pos, state in enumerate(basis)}
    entries = {}
    for col, state in enumerate(basis):
        image = tuple(
            adjacent_index + 1 if value == adjacent_index else adjacent_index if value == adjacent_index + 1 else value
            for value in state
        )
        entries[(index[image], col)] = 1
    return exact_matrix_from_entries(len(basis), len(basis), entries)


def _diagram_action_matrix(n, rank, blocks):
    """Matrix of one partition diagram on ``(C^n)^{otimes rank}``.

    Positions ``0..rank-1`` are output/top indices and positions
    ``rank..2*rank-1`` are input/bottom indices.  A matrix entry is one exactly
    when all positions in each diagram block carry equal values.
    """

    basis = _tensor_basis(n, rank)
    entries = {}
    for row, top_state in enumerate(basis):
        for col, bottom_state in enumerate(basis):
            values = top_state + bottom_state
            ok = True
            for block in blocks:
                first = values[block[0]]
                if any(values[pos] != first for pos in block[1:]):
                    ok = False
                    break
            if ok:
                entries[(row, col)] = 1
    return exact_matrix_from_entries(len(basis), len(basis), entries)


def _flatten_matrix(matrix):
    return exact_matrix_flatten(matrix)


@lru_cache(maxsize=None)
def validate_partition_centralizer(n, rank):
    """Validate the partition-diagram centralizer image for a small case.

    This routine constructs every partition diagram on ``2 * rank`` points,
    checks that each action matrix commutes with adjacent value transpositions
    generating ``S_n``, and checks that their span rank equals the enumerated
    orbit count on pairs of tensor basis states.
    """

    n = int(n)
    rank = int(rank)
    if n <= 0:
        raise ValueError("n must be positive")
    if rank < 0:
        raise ValueError("rank must be nonnegative")

    basis = _tensor_basis(n, rank)
    basis_dim = len(basis)
    pair_patterns = {
        _equality_pattern(top_state + bottom_state)
        for top_state in basis
        for bottom_state in basis
    }
    enumerated_orbit_count = len(pair_patterns)
    expected_orbit_count = partition_centralizer_orbit_count(n, rank)
    diagrams = set_partitions_of_size(2 * rank)
    diagram_matrices = tuple(_diagram_action_matrix(n, rank, blocks) for blocks in diagrams)
    generator_matrices = tuple(
        _adjacent_value_transposition_matrix(n, rank, adjacent)
        for adjacent in range(max(0, n - 1))
    )

    commutes_with_generators = all(
        exact_matrix_equal(
            exact_matrix_matmul(generator, diagram),
            exact_matrix_matmul(diagram, generator),
        )
        for generator in generator_matrices
        for diagram in diagram_matrices
    )
    if diagram_matrices:
        rows = [_flatten_matrix(matrix) for matrix in diagram_matrices]
        span_rank = exact_rational_rank(rows)
    else:
        span_rank = 0
    rank_matches_orbit_count = (
        span_rank == expected_orbit_count == enumerated_orbit_count
    )
    stable_range = n >= 2 * rank
    passed = bool(commutes_with_generators and rank_matches_orbit_count)
    detail = (
        f"n={n}, rank={rank}, basis_dim={basis_dim}, diagrams={len(diagrams)}, "
        f"expected_orbit_count={expected_orbit_count}, "
        f"enumerated_orbit_count={enumerated_orbit_count}, span_rank={span_rank}, "
        f"commutes_with_generators={commutes_with_generators}"
    )
    return PartitionCentralizerValidationReport(
        n=n,
        rank=rank,
        basis_dim=basis_dim,
        diagram_count=len(diagrams),
        expected_orbit_count=expected_orbit_count,
        enumerated_orbit_count=enumerated_orbit_count,
        span_rank=span_rank,
        stable_range=bool(stable_range),
        commutes_with_generators=bool(commutes_with_generators),
        rank_matches_orbit_count=bool(rank_matches_orbit_count),
        passed=passed,
        detail=detail,
    )


__all__ = [
    "PartitionCentralizerValidationReport",
    "partition_centralizer_orbit_count",
    "set_partitions_of_size",
    "validate_partition_centralizer",
]
