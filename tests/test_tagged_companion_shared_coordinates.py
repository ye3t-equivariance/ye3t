"""Independent qualification of the bounded companion's legacy input conversion."""
import pytest
import sympy as sp

from ye3t.couplings import tagged_cauchy_image as companion
from ye3t.representations.projectors import adjacent_transposition_representation_matrix
from ye3t.representations.kronecker_intertwiners import exact_kronecker_intertwiners


def test_legacy_polynomial_coordinates_convert_to_complete_orthonormal_yy_carriers():
    role_states, roles, _ = companion._role_schur_vectors(3,4,(2,1,1))
    role_copy, = companion._role_copy_indices_for_content(3,4,(2,1,1),(1,1,2))
    role_raw = sp.Matrix.hstack(*(roles[(role_copy,t)] for t in range(3)))
    angular_states, angular, count = companion._angular_schur_vectors(4,1,(2,2),0)
    angular_raw = sp.Matrix.hstack(*(angular[(0,t,0)] for t in range(2)))
    assert count == 1
    assert (role_raw.T*role_raw).applyfunc(sp.simplify) == 6*sp.eye(3)
    assert (angular_raw.T*angular_raw).applyfunc(sp.simplify) == sp.diag(12,9)
    converted = (role_raw/sp.sqrt(6), angular_raw*sp.diag(1/sp.sqrt(12),-sp.Rational(1,3)))
    for states, basis, partition in zip((role_states,angular_states),converted,((2,1,1),(2,2))):
        assert (basis.T*basis).applyfunc(sp.simplify) == sp.eye(basis.cols)
        # Direct permutation of raw coordinates checks the actual tableau axes.
        indices = {tuple(state):i for i,state in enumerate(states)}
        for g in range(3):
            permutation = []
            for state in states:
                changed = list(state)
                changed[g],changed[g+1] = changed[g+1],changed[g]
                permutation.append(indices[tuple(changed)])
            moved = basis[permutation,:]
            assert (moved-basis*adjacent_transposition_representation_matrix(partition,g)).applyfunc(sp.simplify).is_zero_matrix
    left_actions = companion._restricted_carrier_actions(role_states,tuple(role_raw[:,i] for i in range(3)),4)
    right_actions = companion._restricted_carrier_actions(angular_states,tuple(angular_raw[:,i] for i in range(2)),4)
    old,_ = companion._reference_exact_kronecker_intertwiner((3,1),left_actions,right_actions)
    reference = sp.kronecker_product(role_raw,angular_raw)*old/12
    shared, = exact_kronecker_intertwiners((3,1),(2,1,1),(2,2))["intertwiners"]
    actual = sp.kronecker_product(*converted)*shared
    assert (actual+reference).applyfunc(sp.simplify).is_zero_matrix
    assert (actual.T*actual).applyfunc(sp.simplify) == sp.eye(3)


def test_cold_companion_uses_shared_map_and_preserves_cartesian_convention(tmp_path,monkeypatch):
    from ye3t.couplings.orthogonal_shifted_jacobi import (
        build_radial_species_product_record, ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY)
    monkeypatch.setenv("YE3T_CACHE_DIR",str(tmp_path))
    def forbidden(*args,**kwargs):
        pytest.fail("Physical companion called its historical generic solve")
    monkeypatch.setattr(companion,"_reference_exact_kronecker_intertwiner",forbidden)
    source=dict(neighbor_species="X",q=0,l=1,source_family_id=ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY)
    angular=companion.racah_harmonic_product_plan((1,),maximum_collision_arity=2)
    product=build_radial_species_product_record((source,),angular,maximum_collision_arity=2)
    with monkeypatch.context() as context:
        context.setattr(companion,"_evaluate_two_neighbor_companion",forbidden)
        context.setattr(companion,"_evaluate_moment_terms_on_axes",forbidden)
        context.setattr(companion,"_diagnose_moment_schedule",forbidden)
        context.setattr(companion,"_product_polynomial_multiply",forbidden)
        payload=companion._compile_s2_n4_physical_companion(product,angular,source_key=source)
        catalogue = companion._build_n4_s012_physical_image(
            (payload,), product, (payload["source_key"],))
    assert "two_neighbor_witness" not in payload
    assert "two_axis_witness" not in catalogue
    diagnostic=companion._diagnose_s2_n4_physical_companion(payload,product,angular)
    assert companion._diagnose_n4_s012_physical_image(catalogue, product)["passed"]
    witness=diagnostic["two_neighbor_witness"]
    assert companion._exact_scalar_from_payload(witness["value"]) == -1
    assert companion._exact_scalar_from_payload(witness["second_value"]) == -4
    assert payload["certificate"]["passed"]
    monkeypatch.setattr(companion,"_build_s2_n4_physical_companion",forbidden)
    assert companion._compile_s2_n4_physical_companion(product,angular,source_key=source) == payload


def test_explicit_adjoint_diagnostic_detects_an_incorrect_emitted_derivative():
    import copy
    from fractions import Fraction
    generator = ("X", "family", "support", 0, 0, 0)
    schedule = companion._compile_moment_schedule({(generator, generator): Fraction(3, 2)})
    assert companion._diagnose_moment_schedule(schedule)
    broken = copy.deepcopy(schedule)
    broken["adjoint_terms"][0]["coefficient"] = companion._exact_scalar_payload(sp.Rational(5, 2))
    with pytest.raises(ValueError, match="adjoint is inconsistent"):
        companion._diagnose_moment_schedule(broken)
