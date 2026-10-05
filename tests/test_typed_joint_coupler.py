import numpy as np
import pytest


def test_typed_route_inventory_matches_exact_public_multiplicity_labels():
    from ye3t.couplings import count, plan

    cases = (
        ((1, 1), (0, 1), "trivial", 1, 1),
        ((1, 1), (0, 1), "antisymmetric", 1, 1),
        ((1, 1, 2), (0, 1, 1), "young:(2,1)", 0, 2),
        ((1, 2, 3), (1, 1, 1), "trivial", 1, 3),
        ((1, 1, 1), (2, 2, 2), "young:(2,1)", 2, 2),
    )
    for content, input_Ls, target, output_L, expected in cases:
        report = count(
            content=content, input_Ls=input_Ls, target_L=output_L,
            target_permutation=target, carrier="Phi",
        )
        coupling_plan = plan(report)
        bindings = coupling_plan.validation_report["alpha_bindings"]
        assert len(bindings) == report.counts_by_target[output_L] == expected
        assert tuple(row["alpha_index"] for row in bindings) == tuple(range(expected))
        assert tuple(
            label.multiplicity_index for label in report.labels_for_target(output_L)
        ) == tuple(range(expected))


def test_count_plan_binds_crossed_local_young_and_angular_copies():
    from ye3t.couplings import count, plan

    report = count(
        content=(1, 1, 1, 2, 3),
        input_Ls=(2, 2, 2, 1, 1),
        target_L=1,
        target_permutation="young:(3,2)",
        carrier="Phi",
    )
    bindings = plan(report).validation_report["alpha_bindings"]
    assert len(bindings) == report.counts_by_target[1] == 20
    assert any(
        max(route["local_copy_indices"]) > 0
        and route["young_copy"] > 0
        and route["angular_copy"] > 0
        for route in bindings
    )


def test_local_repeated_type_two_copy_isometry_and_permutation_projector():
    from ye3t.global_coupler import _typed_local_isometry

    matrix, states, copies, tableau_dim = _typed_local_isometry(3, 2, (2, 1), 2)
    values = np.asarray(matrix, dtype=float)
    assert copies == 2
    assert tableau_dim == 2
    assert values.shape == (125, 20)
    np.testing.assert_allclose(values.T @ values, np.eye(20), atol=1e-12)
    projector = values @ values.T
    np.testing.assert_allclose(projector @ projector, projector, atol=1e-12)
    np.testing.assert_allclose(np.trace(projector), 20, atol=1e-12)
    index = {state: row for row, state in enumerate(states)}
    for permutation in ((1, 0, 2), (0, 2, 1)):
        action = np.zeros((125, 125))
        for row, state in enumerate(states):
            moved = tuple(state[permutation[position]] for position in range(3))
            action[index[moved], row] = 1.0
        np.testing.assert_allclose(action @ projector, projector @ action, atol=1e-12)


@pytest.mark.parametrize(
    "content,input_Ls,target,output_L,expected",
    (
        ((1, 1), (0, 1), "trivial", 1, 1),
        ((1, 1), (0, 1), "antisymmetric", 1, 1),
        ((1, 1, 2), (0, 1, 1), "young:(2,1)", 0, 2),
        ((1, 2, 3), (1, 1, 1), "trivial", 1, 3),
        ((1, 1, 1), (2, 2, 2), "young:(2,1)", 2, 2),
    ),
)
def test_typed_joint_exact_compile_has_every_public_copy(
    content, input_Ls, target, output_L, expected
):
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import count, plan

    report = count(
        content=content, input_Ls=input_Ls, target_L=output_L,
        target_permutation=target, carrier="Phi",
    )
    compiled = compile_coupling(plan(report), subduction_materialization_backend="exact")
    assert compiled.certificate.passed
    assert len(compiled.coupler.alpha_labels()) == expected
    table = compiled.coupler.sparse_coefficient_tables[0]
    assert table["kind"] == "typed_joint_orbit_isometry"
    matrix = np.asarray(compiled.coupler.sparse_coefficient_matrix(), dtype=float)
    assert matrix.shape[1] == expected * (2 * output_L + 1) * (
        2 if target == "young:(2,1)" else 1
    )
    np.testing.assert_allclose(
        matrix.T @ matrix, np.eye(matrix.shape[1]), atol=1e-12
    )


def test_mixed_rank_two_global_young_projectors_are_complementary():
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import count

    projectors = {}
    swap = np.kron(np.array([[0.0, 1.0], [1.0, 0.0]]), np.eye(3))
    for target, sign in (("trivial", 1), ("antisymmetric", -1)):
        compiled = compile_coupling(
            count(
                content=(1, 1), input_Ls=(0, 1), target_L=1,
                target_permutation=target, carrier="Phi",
            ),
            subduction_materialization_backend="exact",
        )
        matrix = np.asarray(compiled.coupler.sparse_coefficient_matrix(), dtype=float)
        projector = matrix @ matrix.T
        np.testing.assert_allclose(
            projector, (np.eye(6) + sign * swap) / 2, atol=1e-12
        )
        projectors[target] = projector
    np.testing.assert_allclose(
        projectors["trivial"] + projectors["antisymmetric"], np.eye(6), atol=1e-12
    )


def test_mixed_rank_three_projector_matches_independent_standard_singlet():
    from itertools import product
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import count

    compiled = compile_coupling(
        count(
            content=(1, 1, 2), input_Ls=(0, 1, 1), target_L=0,
            target_permutation="young:(2,1)", carrier="Phi",
        ),
        subduction_materialization_backend="exact",
    )
    table = compiled.coupler.sparse_coefficient_tables[0]
    matrix = np.asarray(compiled.coupler.sparse_coefficient_matrix(), dtype=float)
    cosets = tuple(tuple(rep) for rep in table["coset_representatives"])
    signs = np.array([
        (-1) ** sum(
            rep[left] > rep[right]
            for left in range(3) for right in range(left + 1, 3)
        )
        for rep in cosets
    ], dtype=float)
    young_projector = (
        np.eye(6) - np.ones((6, 6)) / 6 - np.outer(signs, signs) / 6
    )
    singlet = np.array([
        ((-1) ** (1 - m1)) / np.sqrt(3) if m2 == -m1 else 0.0
        for m1, m2 in product(range(-1, 2), repeat=2)
    ])
    independent = np.kron(young_projector, np.outer(singlet, singlet))
    np.testing.assert_allclose(matrix @ matrix.T, independent, atol=1e-12)
    np.testing.assert_allclose(np.trace(independent), 4, atol=1e-12)

    cycle = (1, 2, 0)
    action = np.zeros((6, 6))
    for source, rep in enumerate(cosets):
        target = tuple(cycle[position] for position in rep)
        action[cosets.index(target), source] = 1.0
    assert not np.array_equal(action, action.T)
    physical_action = np.kron(action, np.eye(9))
    np.testing.assert_allclose(
        physical_action @ independent, independent @ physical_action, atol=1e-12
    )


def test_mixed_rank_three_compiled_analysis_intertwines_noninvolutive_cycle_and_rotation():
    import torch
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import count
    from ye3t.representations.projectors import canonical_irrep_matrices_numeric

    compiled = compile_coupling(
        count(
            content=(1, 1, 2), input_Ls=(0, 1, 1), target_L=0,
            target_permutation="young:(2,1)", carrier="Phi",
        ),
        subduction_materialization_backend="exact",
    )
    coupler = compiled.coupler
    table = coupler.sparse_coefficient_tables[0]
    matrix = np.asarray(coupler.sparse_coefficient_matrix(), dtype=float)
    cosets = tuple(tuple(rep) for rep in table["coset_representatives"])
    cycle = (1, 2, 0)
    inverse_cycle = tuple(cycle.index(position) for position in range(3))
    permutation = np.zeros((6, 6))
    for source, rep in enumerate(cosets):
        moved = tuple(inverse_cycle[position] for position in rep)
        permutation[cosets.index(moved), source] = 1
    assert not np.array_equal(permutation, permutation.T)
    raw_action = np.kron(permutation, np.eye(9))
    tableau_action = np.asarray(
        canonical_irrep_matrices_numeric((2, 1))[cycle], dtype=float
    )
    output_action = np.kron(np.eye(2), tableau_action)
    np.testing.assert_allclose(
        raw_action @ matrix, matrix @ output_action, atol=1e-12
    )

    raw = torch.arange(54, dtype=torch.float64).reshape(1, 54) ** 2 + 0.37
    evaluated = coupler.evaluate_reference_torch(raw)
    assert evaluated.coefficient_axes == ("alpha_x_target_tableau_x_target_M",)
    assert evaluated.metadata["logical_output_shape"] == (2, 2, 1)
    base = evaluated.values
    moved = coupler.evaluate_reference_torch(
        raw @ torch.as_tensor(raw_action.T, dtype=torch.float64)
    ).values
    torch.testing.assert_close(
        moved, base @ torch.as_tensor(output_action.T, dtype=torch.float64),
        atol=1e-12, rtol=1e-12,
    )
    complex_raw = raw.to(torch.complex128) + 1j * raw.flip(-1)
    complex_result = coupler.evaluate_reference_torch(complex_raw).values
    torch.testing.assert_close(
        complex_result,
        complex_raw @ torch.as_tensor(matrix, dtype=torch.complex128),
        atol=1e-12, rtol=1e-12,
    )

    axis = np.array([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    cross = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    angle = 0.43
    rotation = (
        np.cos(angle) * np.eye(3)
        + np.sin(angle) * cross
        + (1 - np.cos(angle)) * np.outer(axis, axis)
    )
    spherical = np.array([
        [1 / np.sqrt(2), -1j / np.sqrt(2), 0],
        [0, 0, 1],
        [-1 / np.sqrt(2), -1j / np.sqrt(2), 0],
    ])
    spin_one = spherical @ rotation @ np.linalg.inv(spherical)
    raw_rotation = np.kron(np.eye(6), np.kron(spin_one, spin_one))
    np.testing.assert_allclose(raw_rotation @ matrix, matrix, atol=1e-12)


def test_three_angular_paths_exhaust_independent_spin_one_projector():
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import count

    compiled = compile_coupling(
        count(
            content=(1, 2, 3), input_Ls=(1, 1, 1), target_L=1,
            target_permutation="trivial", carrier="Phi",
        ),
        subduction_materialization_backend="exact",
    )
    matrix = np.asarray(compiled.coupler.sparse_coefficient_matrix(), dtype=float)
    assert matrix.shape == (162, 9)
    assert {
        route["angular_copy"]
        for route in compiled.coupler.sparse_coefficient_tables[0]["alpha_bindings"]
    } == {0, 1, 2}

    raising = np.array([
        [0.0, 0.0, 0.0],
        [np.sqrt(2), 0.0, 0.0],
        [0.0, np.sqrt(2), 0.0],
    ])
    lowering = raising.T
    jx = (raising + lowering) / 2
    jy = (raising - lowering) / (2j)
    jz = np.diag([-1.0, 0.0, 1.0])
    identity = np.eye(3)
    total = [
        np.kron(np.kron(generator, identity), identity)
        + np.kron(np.kron(identity, generator), identity)
        + np.kron(np.kron(identity, identity), generator)
        for generator in (jx, jy, jz)
    ]
    casimir = sum(generator @ generator for generator in total)
    spin_one_projector = np.eye(27, dtype=complex)
    for other_spin in (0, 2, 3):
        other_eigenvalue = other_spin * (other_spin + 1)
        spin_one_projector = (
            spin_one_projector
            @ (casimir - other_eigenvalue * np.eye(27))
            / (2 - other_eigenvalue)
        )
    independent = np.kron(
        np.ones((6, 6)) / 6, spin_one_projector
    )
    np.testing.assert_allclose(
        matrix @ matrix.T, independent, atol=1e-11
    )
    np.testing.assert_allclose(
        np.trace(independent), 9, atol=1e-11
    )
    axis = np.array([2.0, -1.0, 3.0])
    axis /= np.linalg.norm(axis)
    cross = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    angle = 0.61
    rotation = (
        np.cos(angle) * np.eye(3)
        + np.sin(angle) * cross
        + (1 - np.cos(angle)) * np.outer(axis, axis)
    )
    spherical = np.array([
        [1 / np.sqrt(2), -1j / np.sqrt(2), 0],
        [0, 0, 1],
        [-1 / np.sqrt(2), -1j / np.sqrt(2), 0],
    ])
    spin_one = spherical @ rotation @ np.linalg.inv(spherical)
    raw_rotation = np.kron(
        np.eye(6), np.kron(np.kron(spin_one, spin_one), spin_one)
    )
    output_rotation = np.kron(np.eye(3), spin_one)
    np.testing.assert_allclose(
        raw_rotation @ matrix, matrix @ output_rotation, atol=1e-11
    )


def test_mixed_joint_sector_projector_is_independent_of_tree_bracketing():
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import count

    projectors = []
    for tree in ("balanced", "left"):
        compiled = compile_coupling(
            count(
                content=(1, 1, 2), input_Ls=(0, 1, 1), target_L=0,
                target_permutation="young:(2,1)", carrier="Phi",
                tree_schedule=tree,
            ),
            subduction_materialization_backend="exact",
        )
        matrix = np.asarray(compiled.coupler.sparse_coefficient_matrix(), dtype=float)
        projectors.append(matrix @ matrix.T)
    for projector in projectors[1:]:
        np.testing.assert_allclose(projector, projectors[0], atol=1e-12)


def test_cached_numeric_mixed_joint_projector_matches_exact_reference(tmp_path):
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import count

    report = count(
        content=(1, 1, 2), input_Ls=(0, 1, 1), target_L=0,
        target_permutation="young:(2,1)", carrier="Phi",
    )
    exact = compile_coupling(
        report, subduction_materialization_backend="exact"
    )
    numeric = compile_coupling(
        report,
        subduction_materialization_backend="numeric_cached",
        subduction_cache_dir=tmp_path,
        compare_exact_projector=True,
        subduction_exact_reference_max_rank=3,
    )
    assert numeric.certificate.passed
    assert numeric.coupler.subduction_maps[0].source.validation.passed
    retained = numeric.certificate.provenance["numeric_subduction_reports"]
    assert retained
    assert all(report["ok"] for report in retained)
    assert all(report["rank_gap"] > 10 for report in retained)
    assert all(report["exact_reference"]["projector_compared_to_exact"] for report in retained)
    assert numeric.certificate.checks["numeric_rank_gap_and_projector_validated"]
    assert numeric.coupler.subduction_maps[0].source.as_dict()["numeric_validation_report"]
    assert numeric.coupler.sparse_coefficient_tables[0]["entry_format"] == "numeric_real"
    assert numeric.coupler.validate_sparse_coefficient_tables()[0]["passed"]
    assert not numeric.coupler.validate_sparse_coefficient_tables(exact=True)[0]["passed"]
    exact_matrix = np.asarray(exact.coupler.sparse_coefficient_matrix(), dtype=float)
    numeric_matrix = np.asarray(numeric.coupler.sparse_coefficient_matrix(), dtype=float)
    np.testing.assert_allclose(
        numeric_matrix.T @ numeric_matrix, np.eye(4), atol=1e-8
    )
    np.testing.assert_allclose(
        numeric_matrix @ numeric_matrix.T,
        exact_matrix @ exact_matrix.T,
        atol=1e-8,
    )


def test_repeated_typed_block_does_not_advertise_unimplemented_slot_evaluator():
    import torch
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import count
    from ye3t.global_coupler import (
        evaluate_joint_ye3t_factorized_slots_torch,
        joint_ye3t_factorized_slot_evaluator_report,
    )

    compiled = compile_coupling(
        count(
            content=(1, 1), input_Ls=(1, 1), target_L=0,
            target_permutation="trivial", carrier="Phi",
        ),
        subduction_materialization_backend="exact",
    )
    assert compiled.certificate.passed
    report = joint_ye3t_factorized_slot_evaluator_report(compiled.coupler)
    assert not report["passed"]
    assert not report["all_typed_blocks_singleton"]
    with pytest.raises(ValueError, match="singleton child Specht"):
        evaluate_joint_ye3t_factorized_slots_torch(
            compiled.coupler, (torch.ones(3), torch.ones(3))
        )


def test_explicit_incomplete_block_selection_rejects_typed_compilation():
    from ye3t import CompileGlobalYE3TCouplers, YE3TRotationTarget, YE3TSpec

    spec = YE3TSpec(
        content=(1, 1),
        target_permutation="trivial",
        block_permutation=((2,),),
        target_rotation=YE3TRotationTarget(L_R=1),
        carrier="Phi",
        coefficient_backend="global_coupler",
        metadata={"input_Ls": (0, 1)},
    )
    with pytest.raises(ValueError, match="Explicit block-permutation selection"):
        CompileGlobalYE3TCouplers(spec)


def test_repeated_typed_block_with_cosets_has_independent_standard_projector():
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import count
    from ye3t.representations.projectors import canonical_irrep_matrices_numeric

    compiled = compile_coupling(
        count(
            content=(1, 1, 2), input_Ls=(1, 1, 0), target_L=0,
            target_permutation="young:(2,1)", carrier="Phi",
        ),
        subduction_materialization_backend="exact",
    )
    matrix = np.asarray(compiled.coupler.sparse_coefficient_matrix(), dtype=float)
    assert matrix.shape == (27, 2)
    singlet = np.array([
        ((-1) ** (1 - m1)) / np.sqrt(3) if m2 == -m1 else 0.0
        for m1 in range(-1, 2) for m2 in range(-1, 2)
    ])
    independent = np.kron(
        np.eye(3) - np.ones((3, 3)) / 3,
        np.outer(singlet, singlet),
    )
    np.testing.assert_allclose(matrix @ matrix.T, independent, atol=1e-12)
    np.testing.assert_allclose(np.trace(independent), 2, atol=1e-12)

    cosets = tuple(
        tuple(rep)
        for rep in compiled.coupler.sparse_coefficient_tables[0]["coset_representatives"]
    )
    cycle = (1, 2, 0)
    inverse_cycle = tuple(cycle.index(position) for position in range(3))
    subgroup = ((0, 1, 2), (1, 0, 2))
    action = np.zeros((3, 3))
    for source, rep in enumerate(cosets):
        moved = tuple(inverse_cycle[position] for position in rep)
        target = next(
            index for index, candidate in enumerate(cosets)
            if moved in {
                tuple(candidate[h[position]] for position in range(3))
                for h in subgroup
            }
        )
        action[target, source] = 1
    tableau_action = np.asarray(
        canonical_irrep_matrices_numeric((2, 1))[cycle], dtype=float
    )
    np.testing.assert_allclose(
        np.kron(action, np.eye(9)) @ matrix,
        matrix @ tableau_action,
        atol=1e-12,
    )


def test_antisymmetric_local_block_transports_through_nontrivial_cosets():
    from itertools import product
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import count
    from ye3t.representations.projectors import canonical_irrep_matrices_numeric

    compiled = compile_coupling(
        count(
            content=(1, 1, 2), input_Ls=(1, 1, 0), target_L=1,
            target_permutation="young:(2,1)", carrier="Phi",
        ),
        subduction_materialization_backend="exact",
    )
    matrix = np.asarray(compiled.coupler.sparse_coefficient_matrix(), dtype=float)
    assert matrix.shape == (27, 6)
    table = compiled.coupler.sparse_coefficient_tables[0]
    assert {
        route["block_partitions"][0]
        for route in table["alpha_bindings"]
    } == {(1, 1)}
    cosets = tuple(tuple(rep) for rep in table["coset_representatives"])
    signs = np.array([
        (-1) ** sum(
            rep[left] > rep[right]
            for left in range(3) for right in range(left + 1, 3)
        )
        for rep in cosets
    ], dtype=float)
    magnetic_swap = np.zeros((9, 9))
    magnetic_states = tuple(product(range(-1, 2), repeat=2))
    for source, state in enumerate(magnetic_states):
        magnetic_swap[magnetic_states.index(state[::-1]), source] = 1
    expected = np.kron(
        np.eye(3) - np.outer(signs, signs) / 3,
        (np.eye(9) - magnetic_swap) / 2,
    )
    np.testing.assert_allclose(matrix @ matrix.T, expected, atol=1e-12)
    np.testing.assert_allclose(np.trace(expected), 6, atol=1e-12)

    cycle = (1, 2, 0)
    inverse_cycle = tuple(cycle.index(position) for position in range(3))
    subgroup = ((0, 1, 2), (1, 0, 2))
    action = np.zeros((27, 27))
    for source_coset, rep in enumerate(cosets):
        moved = tuple(inverse_cycle[position] for position in rep)
        target_coset, local_permutation = next(
            (target_index, h)
            for target_index, candidate in enumerate(cosets)
            for h in subgroup
            if moved == tuple(candidate[h[position]] for position in range(3))
        )
        for source_magnetic, state in enumerate(magnetic_states):
            transformed = tuple(state[local_permutation[position]] for position in range(2))
            target_magnetic = magnetic_states.index(transformed)
            action[target_coset * 9 + target_magnetic, source_coset * 9 + source_magnetic] = 1
    tableau_action = np.asarray(
        canonical_irrep_matrices_numeric((2, 1))[cycle], dtype=float
    )
    np.testing.assert_allclose(
        action @ matrix,
        matrix @ np.kron(tableau_action, np.eye(3)),
        atol=1e-12,
    )
