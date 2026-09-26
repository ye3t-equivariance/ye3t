"""Exact equivalence and scaling guards for conjugacy-class character sums."""

import math
import time

from ye3t.message_passing import SameRankKroneckerMultiplicity
from ye3t.representations.generalized_irreps import Partition
from ye3t.representations.projectors import (
    _partition_character_classes,
    _symmetric_group_conjugacy_classes,
    all_permutations,
    permutation_cycle_type,
    symmetric_group_character,
)
from ye3t.representations.tensor_products import (
    _induced_partition_product_multiplicity,
)


def _brute_same_rank(left, right, target):
    n = int(left.size)
    total = 0
    for permutation in all_permutations(n):
        cycle_type = permutation_cycle_type(permutation)
        total += (
            int(symmetric_group_character(left, cycle_type))
            * int(symmetric_group_character(right, cycle_type))
            * int(symmetric_group_character(target, cycle_type))
        )
    return int(total // math.factorial(n))


def _brute_induced(left, right, target):
    total = 0
    for left_permutation in all_permutations(int(left.size)):
        left_cycle = permutation_cycle_type(left_permutation)
        left_character = int(symmetric_group_character(left, left_cycle))
        for right_permutation in all_permutations(int(right.size)):
            right_cycle = permutation_cycle_type(right_permutation)
            right_character = int(symmetric_group_character(right, right_cycle))
            merged_cycle = tuple(
                sorted(tuple(left_cycle) + tuple(right_cycle), reverse=True)
            )
            total += (
                left_character
                * right_character
                * int(symmetric_group_character(target, merged_cycle))
            )
    denominator = math.factorial(int(left.size)) * math.factorial(int(right.size))
    return int(total // denominator)


def test_conjugacy_class_records_cover_every_group_element():
    expected_class_counts = (1, 1, 2, 3, 5, 7, 11, 15, 22)
    for n, expected_count in enumerate(expected_class_counts):
        records = _symmetric_group_conjugacy_classes(n)
        assert len(records) == expected_count
        assert sum(int(class_size) for _cycle, class_size in records) == (
            math.factorial(n)
        )


def test_character_class_table_matches_direct_character_evaluation():
    partition = Partition((3, 2, 1))
    for cycle_type, class_size, character in _partition_character_classes(
        tuple(partition.parts)
    ):
        assert class_size > 0
        assert character == symmetric_group_character(partition, cycle_type)


def test_kronecker_class_sum_matches_brute_permutation_sum():
    cases = (
        ((2, 1), (2, 1), (3,)),
        ((2, 1), (2, 1), (1, 1, 1)),
        ((3, 1), (2, 2), (3, 1)),
        ((2, 2), (2, 2), (2, 2)),
    )
    for left_parts, right_parts, target_parts in cases:
        left = Partition(left_parts)
        right = Partition(right_parts)
        target = Partition(target_parts)
        assert SameRankKroneckerMultiplicity(
            left_parts, right_parts, target_parts
        ) == _brute_same_rank(left, right, target)


def test_induced_class_sum_matches_brute_young_subgroup_sum():
    cases = (
        ((2,), (2,), (4,)),
        ((2,), (1, 1), (3, 1)),
        ((2, 1), (2,), (3, 2)),
        ((2, 1), (1, 1), (2, 2, 1)),
    )
    for left_parts, right_parts, target_parts in cases:
        left = Partition(left_parts)
        right = Partition(right_parts)
        target = Partition(target_parts)
        assert _induced_partition_product_multiplicity(
            left, right, target
        ) == _brute_induced(left, right, target)


def test_rank8_kronecker_decomposition_is_exact_and_bounded():
    left = Partition((4, 4))
    right = Partition((4, 4))
    targets = tuple(
        Partition(parts)
        for parts in (
            (8,),
            (7, 1),
            (6, 2),
            (6, 1, 1),
            (5, 3),
            (5, 2, 1),
            (5, 1, 1, 1),
            (4, 4),
            (4, 3, 1),
            (4, 2, 2),
            (4, 2, 1, 1),
            (4, 1, 1, 1, 1),
            (3, 3, 2),
            (3, 3, 1, 1),
            (3, 2, 2, 1),
            (3, 2, 1, 1, 1),
            (3, 1, 1, 1, 1, 1),
            (2, 2, 2, 2),
            (2, 2, 2, 1, 1),
            (2, 2, 1, 1, 1, 1),
            (2, 1, 1, 1, 1, 1, 1),
            (1, 1, 1, 1, 1, 1, 1, 1),
        )
    )
    started = time.perf_counter()
    multiplicities = tuple(
        SameRankKroneckerMultiplicity(left, right, target)
        for target in targets
    )
    elapsed = time.perf_counter() - started
    assert sum(
        int(multiplicity) * int(target.dimension)
        for multiplicity, target in zip(multiplicities, targets)
    ) == int(left.dimension) * int(right.dimension)
    assert elapsed < 5.0
