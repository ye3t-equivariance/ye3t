from fractions import Fraction

import pytest


def test_su2_irrep_accepts_half_integer_spin_and_tensor_products():
    from ye3t.core.basis.characters import cg_allowed
    from ye3t.representations import SU2Irrep, su2_tensor_product_js

    spin_half = SU2Irrep(Fraction(1, 2))
    spin_one = SU2Irrep(1)

    assert spin_half.group == "SU(2)"
    assert spin_half.dim == 2
    assert spin_half.m_values == (Fraction(-1, 2), Fraction(1, 2))
    assert spin_half.to_string() == "j=1/2"

    assert su2_tensor_product_js(Fraction(1, 2), Fraction(1, 2)) == (Fraction(0), Fraction(1))
    assert su2_tensor_product_js(Fraction(1, 2), 1) == (Fraction(1, 2), Fraction(3, 2))
    assert tuple(irrep.j for irrep in spin_half.tensor_product(spin_one)) == (
        Fraction(1, 2),
        Fraction(3, 2),
    )

    assert cg_allowed(1, 2) == (1, 2, 3)
    assert cg_allowed(Fraction(1, 2), 1) == (Fraction(1, 2), Fraction(3, 2))

    with pytest.raises(ValueError, match="integer or half-integer"):
        SU2Irrep(Fraction(1, 3))


def test_su2_cg_backend_normalizes_half_integer_maps():
    from ye3t.representations import SU2Irrep, build_su2_cg_map, su2_clebsch_gordan

    singlet = build_su2_cg_map(SU2Irrep(Fraction(1, 2)), SU2Irrep(Fraction(1, 2)), SU2Irrep(0))
    triplet = build_su2_cg_map(Fraction(1, 2), Fraction(1, 2), 1)

    assert singlet.validation["target_m_normalized"] is True
    assert singlet.validation["max_target_m_norm_error"] < 1e-12
    assert len(singlet.entries) == 2
    assert triplet.validation["target_m_normalized"] is True
    assert triplet.validation["target_m_count"] == 3

    coeff = su2_clebsch_gordan(
        Fraction(1, 2),
        Fraction(1, 2),
        Fraction(1, 2),
        Fraction(-1, 2),
        0,
        0,
    )
    assert abs(abs(coeff) - 2**-0.5) < 1e-12

    with pytest.raises(ValueError, match="not reachable"):
        build_su2_cg_map(Fraction(1, 2), Fraction(1, 2), 2)


def test_spin_spatial_sign_selection_uses_conjugate_young_sector():
    from ye3t.representations import spin_spatial_sign_selection

    symmetric_space_spin_singlet = spin_spatial_sign_selection((2,), (1, 1))
    antisymmetric_space_spin_triplet = spin_spatial_sign_selection((1, 1), (2,))
    invalid_two_fermion = spin_spatial_sign_selection((2,), (2,))
    self_conjugate_three_body = spin_spatial_sign_selection((2, 1), (2, 1))

    assert symmetric_space_spin_singlet.allowed is True
    assert symmetric_space_spin_singlet.multiplicity == 1
    assert antisymmetric_space_spin_triplet.allowed is True
    assert antisymmetric_space_spin_triplet.multiplicity == 1
    assert invalid_two_fermion.allowed is False
    assert invalid_two_fermion.multiplicity == 0
    assert self_conjugate_three_body.allowed is True
    assert tuple(self_conjugate_three_body.required_spin_partition.parts) == (2, 1)

    with pytest.raises(ValueError, match="same symmetric group degree"):
        spin_spatial_sign_selection((3,), (1, 1))
