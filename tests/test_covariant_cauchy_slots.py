"""Slot-resolved general-parent Cauchy blocks: counts, symmetry, and completeness."""

import sympy as sp

from ye3t.couplings import lifted_cauchy_scalar as scalar
from ye3t.couplings.covariant_cauchy import compile_slot_resolved_cauchy_block
from ye3t.representations.kronecker_intertwiners import kronecker_multiplicity
from ye3t.representations.projectors import standard_tableaux


PARTITIONS_OF_3 = ((3,), (2, 1), (1, 1, 1))
ROLE_DIMENSION = 2
ANGULAR_L = 1
SIZE = 3


def angular_counts(partition):
    return {
        int(Lambda): int(count)
        for Lambda, count in scalar._angular_counts(SIZE, ANGULAR_L, partition).items()
        if int(count) > 0
    }


def test_sector_dimensions_match_the_schur_functor_of_the_product_carrier():
    carrier = ROLE_DIMENSION * (2 * ANGULAR_L + 1)
    total = 0
    for parent in PARTITIONS_OF_3:
        dimension = 0
        for role in PARTITIONS_OF_3:
            for angular in PARTITIONS_OF_3:
                g = kronecker_multiplicity(parent, role, angular)
                role_count = scalar._hook_content_dimension(role, ROLE_DIMENSION)
                for Lambda, count in angular_counts(angular).items():
                    dimension += g * role_count * count * (2 * Lambda + 1)
        assert dimension == scalar._hook_content_dimension(parent, carrier)
        total += len(standard_tableaux(parent)) * dimension
    assert total == carrier**SIZE


def test_emitted_vectors_are_complete_in_the_ordered_tensor_power():
    columns = []
    for parent in PARTITIONS_OF_3:
        for role in PARTITIONS_OF_3:
            for angular in PARTITIONS_OF_3:
                if not kronecker_multiplicity(parent, role, angular):
                    continue
                if not scalar._hook_content_dimension(role, ROLE_DIMENSION):
                    continue
                for Lambda in angular_counts(angular):
                    block = compile_slot_resolved_cauchy_block(
                        ROLE_DIMENSION, SIZE, parent, role, angular, ANGULAR_L, Lambda
                    )
                    assert block["validation_report"]["passed"]
                    columns.extend(block["vectors"].values())
    carrier = ROLE_DIMENSION * (2 * ANGULAR_L + 1)
    assert len(columns) == carrier**SIZE
    assert sp.Matrix.hstack(*columns).rank() == carrier**SIZE


def test_trivial_and_sign_parents_are_the_two_cauchy_pairings():
    symmetric = compile_slot_resolved_cauchy_block(2, 3, (3,), (2, 1), (2, 1), 1, 1)
    skew = compile_slot_resolved_cauchy_block(2, 3, (1, 1, 1), (2, 1), (2, 1), 1, 1)
    assert symmetric["parent_dimension"] == skew["parent_dimension"] == 1
    states = symmetric["joint_states"]
    index = {state: position for position, state in enumerate(states)}
    for block, sign in ((symmetric, 1), (skew, -1)):
        for vector in block["vectors"].values():
            for position, state in enumerate(states):
                swapped = (state[1], state[0], state[2])
                assert sp.simplify(vector[index[swapped]] - sign * vector[position]) == 0
    # The skew pairing matches a partition with its conjugate; (3) and (2,1) do not pair.
    unmatched = compile_slot_resolved_cauchy_block(2, 3, (1, 1, 1), (3,), (2, 1), 1, 1)
    assert unmatched["multiplicity"] == 0


def test_nontrivial_parents_vanish_on_commuting_densities():
    block = compile_slot_resolved_cauchy_block(2, 3, (2, 1), (2, 1), (3,), 1, 1)
    assert block["multiplicity"] == 1
    symbols = {
        (role, m): sp.Symbol("a_%d_%d" % (role, m + 1))
        for role in range(2)
        for m in (-1, 0, 1)
    }
    for vector in block["vectors"].values():
        polynomial = sum(
            vector[position] * sp.prod(symbols[slot] for slot in state)
            for position, state in enumerate(block["joint_states"])
        )
        assert sp.expand(polynomial) == 0
