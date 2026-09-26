from itertools import permutations

import pytest

from ye3t.couplings.lifted_cauchy_scalar import _exact_matrix_from_payload
from ye3t.couplings.tagged_cauchy_image import (
    _MATCHING_ACTION_ID,
    _compile_s2_tag_placement_bimodule,
    _compile_s2_tag_placement_sector,
    _exact_real_component,
    _s2_tag_placement_sector_labels,
)
from ye3t.exact_scalars import ExactRadical
from ye3t.representations.projectors import adjacent_transposition


def _sympy():
    from ye3t._optional_sympy import sp

    return sp


def _explicit_injection_action(basis, permutation):
    sp = _sympy()
    index = {pair: position for position, pair in enumerate(basis)}
    matrix = sp.zeros(len(basis), len(basis))
    for column, (left, right) in enumerate(basis):
        target = (permutation[left], permutation[right])
        matrix[index[target], column] = 1
    return matrix


def _explicit_tag_swap(basis):
    sp = _sympy()
    index = {pair: position for position, pair in enumerate(basis)}
    matrix = sp.zeros(len(basis), len(basis))
    for column, (left, right) in enumerate(basis):
        matrix[index[(right, left)], column] = 1
    return matrix


def test_s2_n4_sector_labels_keep_both_typed_standard_copies():
    labels = _s2_tag_placement_sector_labels(4)
    pairs = tuple(
        (tuple(label["tag_kappa"]), tuple(label["placement_parent_lambda"]))
        for label in labels
    )
    assert pairs == (
        ((2,), (4,)),
        ((2,), (3, 1)),
        ((2,), (2, 2)),
        ((1, 1), (3, 1)),
        ((1, 1), (2, 1, 1)),
    )
    assert pairs.count(((2,), (3, 1))) == 1
    assert pairs.count(((1, 1), (3, 1))) == 1


def test_s2_n4_bimodule_matches_independent_left_and_right_actions():
    sp = _sympy()
    compiled = _compile_s2_tag_placement_bimodule(4)
    basis = tuple(tuple(value) for value in compiled["ordered_injection_basis"])
    assert basis == tuple(permutations(range(4), 2))
    assert len(basis) == 12
    assert compiled["certificate"]["passed"] is True
    assert compiled["right_sector_dimensions"] == (
        {"tag_kappa": (2,), "dimension": 6},
        {"tag_kappa": (1, 1), "dimension": 6},
    )

    independent_right = _explicit_tag_swap(basis)
    identity = sp.eye(len(basis))
    for sector in compiled["sectors"]:
        right = _exact_matrix_from_payload(sector["right_tag_swap"])
        assert right == independent_right
        kappa = tuple(sector["tag_kappa"])
        sign = 1 if kappa == (2,) else -1
        projector = _exact_matrix_from_payload(sector["right_sector_projector"])
        assert projector == sp.simplify((identity + sign * independent_right) / 2)
        assert sp.simplify(projector * projector - projector) == sp.zeros(12, 12)
        assert sp.simplify(projector * (identity - projector)) == sp.zeros(12, 12)

        for index, payload in enumerate(sector["left_adjacent_actions"]):
            actual = _exact_matrix_from_payload(payload)
            expected = _explicit_injection_action(
                basis, adjacent_transposition(4, index)
            )
            assert actual == expected
            assert sp.simplify(actual * right - right * actual) == sp.zeros(12, 12)


def test_s2_sign_uniform_functional_zero_but_formal_dual_survives():
    sp = _sympy()
    sector = _compile_s2_tag_placement_sector(4, (1, 1), (3, 1))
    embedding = _exact_matrix_from_payload(sector["sector_embedding"])
    pairing = _exact_matrix_from_payload(sector["normalized_dual_pairing"])
    right = _exact_matrix_from_payload(sector["right_tag_swap"])
    uniform = sp.ones(embedding.rows, 1)

    assert sp.simplify(uniform.T * embedding) == sp.zeros(1, embedding.cols)
    assert sp.simplify((pairing.T * pairing).trace()) == 1
    assert sp.simplify(right * pairing * right.T - pairing) == sp.zeros(
        pairing.rows, pairing.cols
    )
    witness = sector["formal_dual_witness"]
    index = int(witness["basis_index"])
    assert sp.simplify(pairing[index, index]) != 0

    for payload in sector["left_adjacent_actions"]:
        left = _exact_matrix_from_payload(payload)
        assert sp.simplify(left * pairing * left.T - pairing) == sp.zeros(
            pairing.rows, pairing.cols
        )


def test_s2_closed_form_edge_case_and_scope_rejections():
    compiled = _compile_s2_tag_placement_bimodule(2)
    pairs = tuple(
        (tuple(sector["tag_kappa"]), tuple(sector["placement_parent_lambda"]))
        for sector in compiled["sectors"]
    )
    assert pairs == (((2,), (2,)), ((1, 1), (1, 1)))

    with pytest.raises(ValueError, match="tag_count=2"):
        _compile_s2_tag_placement_bimodule(4, tag_count=3)
    with pytest.raises(ValueError, match="tag_slot_multiplicities"):
        _compile_s2_tag_placement_bimodule(
            4, tag_slot_multiplicities=(2, 1)
        )
    with pytest.raises(ValueError, match="repeated-block kappa"):
        _compile_s2_tag_placement_sector(
            4,
            (1, 1),
            (3, 1),
            matching_action_id="repeated_block_S_2",
        )
    with pytest.raises(ValueError, match="zero LR multiplicity"):
        _compile_s2_tag_placement_sector(
            4,
            (1, 1),
            (4,),
            matching_action_id=_MATCHING_ACTION_ID,
        )


def test_exact_real_component_cancels_imaginary_terms_and_rejects_nonreal():
    sp = _sympy()
    value = (sp.sqrt(2) + sp.I * sp.sqrt(3)) + (
        sp.sqrt(2) - sp.I * sp.sqrt(3)
    )
    assert _exact_real_component(value, "test value") == (
        2 * ExactRadical.sqrt(2)
    )
    with pytest.raises(ValueError, match="nonzero exact imaginary"):
        _exact_real_component(sp.sqrt(2) + sp.I, "test value")
    with pytest.raises(TypeError, match="Unsupported exact scalar"):
        _exact_real_component(sp.real_root(2, 3), "test value")
