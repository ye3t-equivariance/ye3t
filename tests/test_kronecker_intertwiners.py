"""Exact Kronecker intertwiners: counts, intertwining, gauge, and Cauchy limits."""

import pytest
import sympy as sp

from ye3t.representations.kronecker_intertwiners import (
    exact_kronecker_intertwiners,
    kronecker_multiplicity,
)
from ye3t.representations.projectors import standard_tableaux


PARTITIONS_OF_4 = ((4,), (3, 1), (2, 2), (2, 1, 1), (1, 1, 1, 1))
CONJUGATE = {
    (4,): (1, 1, 1, 1),
    (3, 1): (2, 1, 1),
    (2, 2): (2, 2),
    (2, 1, 1): (3, 1),
    (1, 1, 1, 1): (4,),
}


def test_rank_four_multiplicities_fill_every_tensor_product():
    for left in PARTITIONS_OF_4:
        for right in PARTITIONS_OF_4:
            total = sum(
                kronecker_multiplicity(parent, left, right) * len(standard_tableaux(parent))
                for parent in PARTITIONS_OF_4
            )
            assert total == len(standard_tableaux(left)) * len(standard_tableaux(right))


def test_cauchy_identities_are_the_trivial_and_sign_parents():
    for left in PARTITIONS_OF_4:
        for right in PARTITIONS_OF_4:
            assert kronecker_multiplicity((4,), left, right) == int(left == right)
            assert kronecker_multiplicity((1, 1, 1, 1), left, right) == int(
                CONJUGATE[left] == right
            )


def test_algebraic_curvature_parent_has_nine_unit_pairings():
    pairs = [
        (left, right)
        for left in PARTITIONS_OF_4
        for right in PARTITIONS_OF_4
        if kronecker_multiplicity((2, 2), left, right)
    ]
    assert len(pairs) == 9
    for left, right in pairs:
        result = exact_kronecker_intertwiners((2, 2), left, right)
        assert result["multiplicity"] == 1
        (intertwiner,) = result["intertwiners"]
        assert intertwiner.shape == (
            len(standard_tableaux(left)) * len(standard_tableaux(right)),
            2,
        )
        # Construction already certifies exact intertwining and T^T T = I.
        assert (intertwiner.T * intertwiner).applyfunc(sp.simplify) == sp.eye(2)


def test_symmetric_cauchy_pairing_is_the_normalized_identity():
    (pairing,) = exact_kronecker_intertwiners((4,), (3, 1), (3, 1))["intertwiners"]
    expected = sp.Matrix([1 if row == column else 0 for row in range(3) for column in range(3)]) / sp.sqrt(3)
    assert (pairing - expected).applyfunc(sp.simplify) == sp.zeros(9, 1)


def test_first_multiplicity_two_case_has_an_orthonormal_gauge():
    assert kronecker_multiplicity((3, 2), (3, 1, 1), (3, 1, 1)) == 2
    result = exact_kronecker_intertwiners((3, 2), (3, 1, 1), (3, 1, 1))
    first, second = result["intertwiners"]
    assert (first.T * second).applyfunc(sp.simplify) == sp.zeros(5, 5)
    assert "gram_schmidt" in result["gauge"]


def test_mismatched_orders_are_rejected():
    with pytest.raises(ValueError, match="share one tensor order"):
        kronecker_multiplicity((2, 2), (3,), (2, 2))
