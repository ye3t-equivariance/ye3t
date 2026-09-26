"""Exact subgroup-chain sums against independent full permutation tables."""

from itertools import permutations
from math import factorial

import pytest
import sympy as sp

from ye3t.representations.coset_units import (
    adjacent_transposition_generators, block_stabilizer_sum,
)
from ye3t.representations.generalized_irreps import Partition
from ye3t.representations.projectors import canonical_irrep_matrices_native


@pytest.mark.parametrize("partition,block", [
    ((1,), (0,)), ((2, 1), (0, 1)), ((2, 1), (0, 1, 2)),
    ((3, 1), (1, 2, 3)), ((2, 2), (0, 1, 2, 3)),
    ((3, 2), (1, 2, 3, 4)), ((3, 2), (0, 1, 2, 3, 4)),
    ((4, 1), (0, 1, 2, 3)),
])
def test_chain_matches_permutation_sum_and_group_invariants(partition, block):
    size = sum(partition)
    dimension = Partition(partition).dimension
    generators = adjacent_transposition_generators(partition, size)
    actual = block_stabilizer_sum(block, generators, dimension)
    table = canonical_irrep_matrices_native(partition)
    expected = sp.zeros(dimension)
    for ordering in permutations(block):
        permutation = list(range(size))
        for source, target in zip(block, ordering):
            permutation[source] = target
        # Sum over inverses equals the full subgroup sum. This independent
        # table uses the established factorial reference construction.
        matrix = table[tuple(permutation)]
        expected += sp.Matrix(dimension, dimension, lambda i, j: sp.sympify(matrix[int(i)][int(j)]))
    assert (actual-expected).applyfunc(sp.simplify) == sp.zeros(dimension)
    assert actual == actual.T
    assert (actual*actual-factorial(len(block))*actual).applyfunc(sp.simplify) == sp.zeros(dimension)
    for index in range(block[0], block[-1]):
        assert (generators[index]*actual-actual).applyfunc(sp.simplify) == sp.zeros(dimension)
        assert (actual*generators[index]-actual).applyfunc(sp.simplify) == sp.zeros(dimension)


def test_large_trivial_stabilizer_and_invalid_block():
    # A factorial table at this size is impossible; the exact chain is small.
    generators = (sp.ones(1),)*19
    assert block_stabilizer_sum(tuple(range(20)), generators, 1) == sp.Matrix([[factorial(20)]])
    with pytest.raises(ValueError, match="contiguous"):
        block_stabilizer_sum((0, 2), generators, 1)
