"""Ordinary ACE compact labels bound to complete Young/rotation coordinates."""

import math

import numpy as np
import pytest

from ye3t.couplings import compile as compile_coupling
from ye3t.couplings import compile_ace_coordinate, count, plan


@pytest.mark.parametrize(
    "content,input_Ls,target_L,expected_count",
    (
        ((1, 1), (0, 1), 1, 1),
        ((1, 1, 1), (0, 1, 1), 0, 1),
        ((1, 1, 1, 2), (0, 1, 1, 1), 1, 2),
        ((1, 2, 3), (1, 1, 1), 1, 3),
        ((1, 1, 1, 1), (2, 2, 2, 2), 4, 2),
    ),
)
def test_ordinary_compact_labels_span_the_typed_physical_projector(
    content, input_Ls, target_L, expected_count,
):
    report = count(
        content=content, input_Ls=input_Ls, target_L=target_L,
        target_permutation="trivial", carrier="ACE_density",
    )
    assert len(report.labels_for_target(target_L)) == expected_count
    compiled = compile_coupling(
        plan(report), subduction_materialization_backend="exact"
    )
    coupler = compiled.coupler
    assert compiled.certificate.passed
    assert len(coupler.alpha_labels()) == expected_count
    assert len(coupler.sparse_coefficient_tables) == 2
    assert all(item["passed"] for item in coupler.validate_sparse_coefficient_tables())
    typed_table, compact_table = coupler.sparse_coefficient_tables
    assert typed_table["kind"] == "typed_joint_orbit_isometry"
    assert compact_table["kind"] == "ace_density_compact_physical_coordinates"
    assert tuple(compact_table["compact_labels"]) == tuple(
        label.to_dict() for label in report.labels_for_target(target_L)
    )
    magnetic_width = 2 * target_L + 1
    assert compact_table["shape"][1] == expected_count * magnetic_width
    typed = np.asarray(coupler.sparse_coefficient_matrix(0), dtype=np.complex128)
    compact = np.asarray(coupler.sparse_coefficient_matrix(1), dtype=np.complex128)
    orbit_count = len(typed_table["coset_representatives"])
    raw_dimension = typed.shape[0] // orbit_count
    projected = typed.reshape(orbit_count, raw_dimension, -1).sum(axis=0)
    projected /= math.sqrt(orbit_count)
    assert compact.shape == projected.shape
    np.testing.assert_allclose(
        projected.conj().T @ projected,
        np.eye(expected_count * magnetic_width),
        atol=1e-11,
    )
    assert np.linalg.matrix_rank(compact, tol=1e-10) == expected_count * magnetic_width
    compact_frame, _ = np.linalg.qr(compact, mode="reduced")
    np.testing.assert_allclose(
        compact_frame @ compact_frame.conj().T,
        projected @ projected.conj().T,
        atol=1e-10,
    )
    binding = coupler.factorized_coefficient_tables[1]
    assert binding["kind"] == "ace_density_compact_to_typed_binding_v1"
    transform = np.asarray(binding["copy_transport_real"]) + 1j * np.asarray(
        binding["copy_transport_imag"]
    )
    np.testing.assert_allclose(
        projected @ np.kron(transform, np.eye(magnetic_width)),
        compact,
        atol=1e-10,
    )
    assert binding["copy_transport_condition_number"] < 1.0e10
    assert compiled.certificate.residuals["compact_and_typed_physical_projector"] < 1e-10


def test_rank_four_compact_copies_require_a_nonunitary_transport():
    report = count(
        content=(1, 1, 1, 1), input_Ls=(2, 2, 2, 2), target_L=4,
        target_permutation="trivial", carrier="ACE_density",
    )
    coupler = compile_coupling(
        report, subduction_materialization_backend="exact"
    ).coupler
    binding = coupler.factorized_coefficient_tables[1]
    transform = np.asarray(binding["copy_transport_real"]) + 1j * np.asarray(
        binding["copy_transport_imag"]
    )
    assert transform.shape == (2, 2)
    assert abs(transform[0, 1]) > 0.1
    assert abs(transform[1, 0]) > 0.1
    assert np.linalg.norm(transform.conj().T @ transform - np.eye(2)) > 0.1
    assert binding["copy_transport_condition_number"] < 3


def test_mixed_rank_two_projector_has_the_normalized_coset_factor():
    report = count(
        content=(1, 1), input_Ls=(0, 1), target_L=1,
        target_permutation="trivial", carrier="ACE_density",
    )
    compiled = compile_coupling(
        report, subduction_materialization_backend="exact"
    )
    typed = np.asarray(
        compiled.coupler.sparse_coefficient_matrix(0), dtype=float
    )
    independent = np.kron(np.ones((2, 2)) / 2, np.eye(3))
    np.testing.assert_allclose(typed @ typed.T, independent, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(compiled.coupler.sparse_coefficient_matrix(1), dtype=float),
        np.eye(3),
        atol=1e-12,
    )


def test_mixed_rank_three_projector_is_uniform_cosets_times_l1_singlet():
    report = count(
        content=(1, 1, 1), input_Ls=(0, 1, 1), target_L=0,
        target_permutation="trivial", carrier="ACE_density",
    )
    compiled = compile_coupling(
        report, subduction_materialization_backend="exact"
    )
    typed = np.asarray(
        compiled.coupler.sparse_coefficient_matrix(0), dtype=float
    )
    singlet = np.asarray([
        ((-1) ** (1 - left)) / math.sqrt(3) if right == -left else 0
        for left in range(-1, 2) for right in range(-1, 2)
    ])
    independent = np.kron(
        np.ones((3, 3)) / 3, np.outer(singlet, singlet)
    )
    np.testing.assert_allclose(typed @ typed.T, independent, atol=1e-12)
    label = report.labels_for_target(0)[0]
    compact_hash = compile_ace_coordinate(label)["certificate"][
        "collected_coefficient_sha256"
    ]
    assert compact_hash == (
        "9eec4609abbcc114096b961b26896c16fb78847cfb5ec14087d1139755340da7"
    )
    assert compiled.coupler.sparse_coefficient_tables[1][
        "compact_coefficient_hashes"
    ] == (compact_hash,)


def test_three_angular_paths_match_independent_total_spin_one_projector():
    report = count(
        content=(1, 2, 3), input_Ls=(1, 1, 1), target_L=1,
        target_permutation="trivial", carrier="ACE_density",
    )
    compiled = compile_coupling(
        report, subduction_materialization_backend="exact"
    )
    typed = np.asarray(
        compiled.coupler.sparse_coefficient_matrix(0), dtype=float
    )
    identity = np.eye(3)
    raising = np.zeros((3, 3))
    raising[1, 0] = math.sqrt(2)
    raising[2, 1] = math.sqrt(2)
    lowering = raising.T
    z = np.diag([-1.0, 0.0, 1.0])

    def total(operator):
        return (
            np.kron(np.kron(operator, identity), identity)
            + np.kron(np.kron(identity, operator), identity)
            + np.kron(np.kron(identity, identity), operator)
        )

    total_z = total(z)
    total_plus = total(raising)
    total_minus = total(lowering)
    casimir = (
        total_z @ total_z
        + (total_plus @ total_minus + total_minus @ total_plus) / 2
    )
    angular_projector = np.eye(27)
    for other_L in (0, 2, 3):
        other_eigenvalue = other_L * (other_L + 1)
        angular_projector = angular_projector @ (
            casimir - other_eigenvalue * np.eye(27)
        ) / (2 - other_eigenvalue)
    independent = np.kron(np.ones((6, 6)) / 6, angular_projector)
    np.testing.assert_allclose(typed @ typed.T, independent, atol=1e-11)
    np.testing.assert_allclose(
        np.kron(np.eye(6), total_plus) @ typed,
        typed @ np.kron(np.eye(3), raising),
        atol=1e-11,
    )


def test_cached_numeric_and_exact_bind_the_same_compact_coefficients(tmp_path):
    from ye3t.global_coupler import CompileGlobalYE3TCouplers

    report = count(
        content=(1, 1, 1, 2), input_Ls=(0, 1, 1, 1), target_L=1,
        target_permutation="trivial", carrier="ACE_density",
    )
    exact = compile_coupling(
        report, subduction_materialization_backend="exact"
    )
    numeric = compile_coupling(
        report,
        subduction_materialization_backend="numeric_cached",
        subduction_cache_dir=tmp_path,
    )
    direct = CompileGlobalYE3TCouplers(
        plan(report).spec, input_Ls=(0, 1, 1, 1),
        subduction_materialization_backend="numeric_cached",
        subduction_cache_dir=tmp_path,
    )
    assert numeric.certificate.passed and direct.certificate.passed
    assert numeric.coupler.sparse_coefficient_tables[1][
        "compact_coefficient_hashes"
    ] == exact.coupler.sparse_coefficient_tables[1][
        "compact_coefficient_hashes"
    ]
    np.testing.assert_allclose(
        np.asarray(numeric.coupler.sparse_coefficient_matrix(1), dtype=float),
        np.asarray(exact.coupler.sparse_coefficient_matrix(1), dtype=float),
        atol=0, rtol=0,
    )
    assert direct.sparse_coefficient_tables[1]["hash"] == (
        numeric.coupler.sparse_coefficient_tables[1]["hash"]
    )


def test_direct_untyped_ace_assembly_rejects_missing_angular_paths():
    from ye3t.global_coupler import (
        AngularCGMap, AssembleJointYoungE3Coupler,
    )
    from ye3t.representations.young_subgroup_specht_coupling import (
        build_young_subgroup_specht_coupling,
    )

    report = count(
        content=(1, 2, 3), input_Ls=(1, 1, 1), target_L=1,
        target_permutation="trivial", carrier="ACE_density",
    )
    spec = plan(report).spec
    young = build_young_subgroup_specht_coupling(
        ((1,), (1,), (1,)), (3,),
    )
    angular = AngularCGMap.build((1, 1, 1), 1)
    with pytest.raises(ValueError, match="Direct assembly of non-scalar ACE"):
        AssembleJointYoungE3Coupler(
            spec, young, angular, input_Ls=(1, 1, 1)
        )
