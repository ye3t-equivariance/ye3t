import math

import numpy as np
import pytest


def test_couplings_count_plan_compile_provenance_and_reject_invalid_label():
    from ye3t.core.labels import CompactLabel
    from ye3t.couplings import CompiledCoupler, CouplerPlan, MultiplicityReport
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import count, plan

    report = count(content=(1, 1), input_Ls=(0, 0), target_L=0)

    assert isinstance(report, MultiplicityReport)
    assert report.content == (1, 1)
    assert report.carrier == "ACE_density"
    assert report.target["permutation"] == "trivial"
    assert report.backend in {"symmetric_power_fast_path", "global_coupler"}
    assert report.convention_hash
    assert report.validation_report["valid_labels_from"] == "ye3t.fixed_content.FixedContentDecomposition.compact_labels_by_target"
    assert report.validation_report["fixed_content_validation"]["passed"] is True
    assert report.validation_report["passed"] is True
    assert report.labels_for_target(0)
    report.require_label(report.labels_for_target(0)[0], target_L=0)

    invalid = CompactLabel((1, 1), (0, 0), (1,), "balanced", ())
    with pytest.raises(ValueError, match="Invalid descriptor label"):
        report.require_label(invalid, target_L=0)

    coupler_plan = plan(report)

    assert isinstance(coupler_plan, CouplerPlan)
    assert coupler_plan.content == report.content
    assert coupler_plan.carrier == report.carrier
    assert coupler_plan.target == report.target
    assert coupler_plan.backend == report.backend
    assert coupler_plan.convention_hash
    assert coupler_plan.validation_report["backend_plan_selected"] == coupler_plan.backend

    compiled = compile_coupling(coupler_plan, subduction_materialization_backend="exact")

    assert isinstance(compiled, CompiledCoupler)
    assert compiled.content == report.content
    assert compiled.carrier == report.carrier
    assert compiled.target == report.target
    assert compiled.backend == coupler_plan.backend
    assert compiled.convention_hash
    assert "compiled_certificate_passed" in compiled.validation_report
    assert compiled.certificate.provenance


@pytest.mark.parametrize(
    "content,target,partition,multiplicity",
    (
        ((1, 1), "antisymmetric", (1, 1), 0),
        ((1, 2), "antisymmetric", (1, 1), 1),
        ((1, 1, 2), "young:(2,1)", (2, 1), 1),
        ((1, 1, 2), "antisymmetric", (1, 1, 1), 0),
        ((1, 2, 3), "young:(2,1)", (2, 1), 2),
        ((1, 2, 3), "antisymmetric", (1, 1, 1), 1),
    ),
)
def test_fixed_content_target_sector_count_matches_compiled_multiplicity(
    content, target, partition, multiplicity
):
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import count, plan
    from ye3t.fixed_content import FixedContentMultiplicityLabel

    report = count(
        content=content,
        input_Ls=(0,) * len(content),
        target_L=0,
        target_permutation=target,
        carrier="Phi",
    )
    labels = report.labels_for_target(0)
    assert report.counts_by_target[0] == multiplicity
    assert labels == tuple(
        FixedContentMultiplicityLabel(partition, 0, alpha)
        for alpha in range(multiplicity)
    )
    assert report.validation_report["label_kind"] == "multiplicity"
    assert report.validation_report["target_partition"] == partition
    assert plan(report).report.labels_for_target(0) == labels
    assert report.to_dict()["labels_by_target"][0] == [label.to_dict() for label in labels]
    for label in labels:
        assert report.contains_label(label.to_dict(), target_L=0)
        assert report.require_label(label, target_L=0) == label
    assert not report.contains_label((partition, 0, multiplicity), target_L=0)

    if multiplicity == 0:
        with pytest.raises(ValueError, match="zero fixed-content multiplicity"):
            compile_coupling(plan(report), subduction_materialization_backend="exact")
        return

    compiled = compile_coupling(plan(report), subduction_materialization_backend="exact")
    assert compiled.certificate.passed
    assert {label.gamma for label in compiled.coupler.labels} == {tuple(range(multiplicity))}
    subduction = compiled.coupler.subduction_maps[0]
    assert subduction.target_partition == partition
    columns = np.asarray(subduction.coefficient_matrix(), dtype=float)
    gram = columns.T @ columns
    projector = columns @ columns.T
    specht_dim = 2 if partition == (2, 1) else 1
    assert columns.shape[1] == multiplicity * specht_dim
    np.testing.assert_allclose(gram, np.eye(columns.shape[1]), atol=1e-12)
    np.testing.assert_allclose(projector @ projector, projector, atol=1e-12)
    assert np.trace(projector) == pytest.approx(multiplicity * specht_dim, abs=1e-12)
    assert subduction.multiplicity == multiplicity


@pytest.mark.parametrize("target", ("young:(N)", "young:N", "symmetric"))
def test_ace_density_rejects_nontrivial_parent_and_preserves_compact_labels(target):
    from ye3t.core.labels import CompactLabel
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import count

    report = count(
        content=(2, 5), input_Ls=(0, 0), target_L=0,
        target_permutation=target,
        fast_path_policy="force:symmetric_power_fast_path",
    )
    assert report.validation_report["target_partition"] == (2,)
    assert report.validation_report["label_kind"] == "compact"
    assert report.backend == "symmetric_power_fast_path"
    assert all(isinstance(label, CompactLabel) for label in report.labels_for_target(0))
    assert compile_coupling(report, subduction_materialization_backend="exact").certificate.passed
    with pytest.raises(ValueError, match="globally trivial Young sector"):
        count(content=(1, 2), input_Ls=(0, 0), target_L=0, target_permutation="antisymmetric")


@pytest.mark.parametrize("target", ("young:(1,1)", "sign"))
def test_sign_alias_selects_exterior_backend_and_incompatible_parity_errors_early(target):
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import count

    report = count(
        content=(1, 2), input_Ls=(0, 0), target_L=0,
        target_permutation=target, carrier="Phi",
        fast_path_policy="force:exterior_power_fast_path",
    )
    assert report.backend == "exterior_power_fast_path"
    assert compile_coupling(report, subduction_materialization_backend="exact").certificate.passed
    with pytest.raises(ValueError, match="incompatible with the natural product parity"):
        count(
            content=(1, 2), input_Ls=(0, 0), target_L=0,
            carrier="Phi", request={"target_rotation": {"L_R": 0, "group": "O3", "parity": "odd"}},
        )


@pytest.mark.parametrize(
    "content,input_Ls,target_L,target,multiplicity",
    (
        ((1, 1), (0, 1), 1, "trivial", 1),
        ((1, 1), (0, 1), 1, "antisymmetric", 1),
        ((1, 1, 2), (0, 1, 1), 0, "young:(2,1)", 2),
    ),
)
def test_generic_compile_materializes_mixed_angular_typed_orbit(
    content, input_Ls, target_L, target, multiplicity
):
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import count, plan

    report = count(
        content=content,
        input_Ls=input_Ls,
        target_L=target_L,
        target_permutation=target,
        carrier="Phi",
    )
    assert report.counts_by_target[target_L] == multiplicity
    compiled = compile_coupling(plan(report), subduction_materialization_backend="exact")
    assert compiled.certificate.passed
    assert len(compiled.coupler.alpha_labels()) == multiplicity
    assert compiled.coupler.sparse_coefficient_tables[0]["kind"] == "typed_joint_orbit_isometry"


def test_direct_global_compiler_uses_full_mixed_angular_stabilizer():
    from ye3t.couplings import count, plan
    from ye3t.global_coupler import CompileGlobalYE3TCouplers

    report = count(
        content=(1, 1), input_Ls=(0, 1), target_L=1,
        target_permutation="trivial", carrier="Phi",
    )
    assert report.counts_by_target[1] == 1
    compiled = CompileGlobalYE3TCouplers(
        plan(report).spec,
        input_Ls=(0, 1),
        subduction_materialization_backend="exact",
    )
    assert compiled.certificate.passed
    assert tuple(block["type"] for block in compiled.block_maps[0]["blocks"]) == (
        (1, 0), (1, 1)
    )
    assert compiled.sparse_coefficient_matrix().shape == (6, 3)


def test_public_compile_materializes_all_angular_beta_labels():
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import count, plan

    report = count(
        content=(1, 2, 3), input_Ls=(1, 1, 1), target_L=1,
        target_permutation="trivial", carrier="Phi",
    )
    assert report.counts_by_target[1] == 3
    coupler_plan = plan(report)
    compiled = compile_coupling(coupler_plan, subduction_materialization_backend="exact")
    assert compiled.certificate.passed
    assert tuple(
        row["angular_copy"] for row in
        compiled.coupler.sparse_coefficient_tables[0]["alpha_bindings"]
    ) == (0, 1, 2)
    assert compiled.coupler.sparse_coefficient_matrix().shape == (162, 9)


def test_compile_ace_factorized_schedules_by_l_public_facade_matches_counts():
    from ye3t.couplings import ACEFactorizedScheduleReport
    from ye3t.couplings import compile_ace_factorized_schedules_by_L, count

    report = compile_ace_factorized_schedules_by_L(
        content=(1, 1, 1),
        input_Ls=(1, 1, 1),
        constructor_backend="python",
        cache_policy="clear_before",
    )

    assert isinstance(report, ACEFactorizedScheduleReport)
    assert report.backend == "ace_factorized_coefficient_schedule"
    assert report.provenance["api"] == "ye3t.couplings.compile_ace_factorized_schedules_by_L"
    assert report.validation_report["valid_labels_from"] == "ye3t.couplings.count"
    assert report.validation_report["coefficient_source"].startswith("ye3t.core.couplings")
    assert report.validation_report["cache_policy"] == "clear_before"
    counts_by_L = {}
    for target_L in sorted(report.schedules_by_target):
        count_report = count(
            content=(1, 1, 1),
            input_Ls=(1, 1, 1),
            target_L=target_L,
            carrier="ACE_density",
        )
        counts_by_L[int(target_L)] = int(count_report.counts_by_target[int(target_L)])
    assert {
        int(target_L): int(schedule.basis_count)
        for target_L, schedule in report.schedules_by_target.items()
    } == counts_by_L
    payload = report.to_dict()
    assert payload["schedule_summary"][1]["basis_count"] == 1
    assert payload["schedule_summary"][3]["basis_count"] == 1
    assert report.convention_hash


def test_cy_factor_product_plan_records_descriptor_adjoint_provenance():
    from ye3t.couplings import (
        CYFactorProductPlan,
        YE3TDescriptorAdjointPlan,
        cy_factor_product_plan,
        ye3t_descriptor_adjoint_plan,
    )

    plan = cy_factor_product_plan(
        descriptor_count=4,
        channel_count=9,
        explicit_term_count=17,
        carrier="ACE_density",
        target={"permutation": "trivial", "rotation": {"L_R": 0, "M_R_values": (0,)}},
        coefficient_source="ye3t.couplings.compile compact descriptor payload",
    )

    assert isinstance(plan, CYFactorProductPlan)
    assert plan.carrier == "ACE_density"
    assert plan.young_coupling["stage"] == "Young/permutation"
    assert plan.young_coupling["sector"] == "globally_trivial"
    assert plan.rotation_coupling["stage"] == "O(3)/CG"
    assert plan.validation_report["valid_labels_from"] == "ye3t.couplings.count"
    assert plan.validation_report["coefficient_source"].startswith("ye3t.couplings.compile")
    assert plan.provenance["compiler_owner"] == "ye3t"
    assert plan.convention_hash

    adjoint = ye3t_descriptor_adjoint_plan(plan)
    assert isinstance(adjoint, YE3TDescriptorAdjointPlan)
    assert adjoint.factor_product_plan is plan
    assert adjoint.validation_report["descriptor_adjoint_plan"] is True
    assert adjoint.adjoint_target == "d_descriptor_d_normalized_factor"
    assert adjoint.convention_hash


def test_angular_edge_coupling_paths_report_uses_ye3t_cg_provenance():
    from ye3t.couplings import angular_edge_coupling_paths

    report = angular_edge_coupling_paths(
        input_Ls=(0, 1),
        edge_Ls=(1,),
        output_Ls=(0, 1),
    )

    assert report["backend"] == "SO3_CG_edge_message_path_report"
    assert report["provenance"]["api"] == "ye3t.couplings.angular_edge_coupling_paths"
    assert report["validation_report"]["valid_paths_from"] == "ye3t.core.basis.characters.cg_allowed"
    assert report["validation_report"]["coefficient_source"] == "ye3t.paired_cg.couple_packed_real_tesseral"
    assert report["validation_report"]["full_young_changing_path"] is False
    paths = {
        (row["input_L"], row["edge_L"], row["output_L"])
        for row in report["paths"]
    }
    assert (0, 1, 1) in paths
    assert (1, 1, 0) in paths
    assert (0, 1, 0) not in paths
    assert report["convention_hash"]


def test_symmetric_power_product_plan_compiles_exponents_and_lower_cache():
    from ye3t.couplings import SymmetricPowerProductPlan, symmetric_power_product_plan

    plan = symmetric_power_product_plan(
        (
            {
                "descriptor_index": 0,
                "channel_indices": (0, 1, 2),
                "power": 2,
                "input_L": 1,
                "output_L": 0,
                "multiplicity_index": 0,
                "component_index": 0,
            },
        ),
        descriptor_count=1,
        channel_count=3,
        normalization_convention="A_normalized",
        label_source="ye3t.couplings.count compact symmetric labels",
    )

    assert isinstance(plan, SymmetricPowerProductPlan)
    assert plan.backend == "symmetric_power_exponent_vector"
    assert plan.provenance["compiler_owner"] == "ye3t"
    assert plan.validation_report["derivative_rule"] == "alpha_q * M_{alpha-e_q}"
    assert plan.validation_report["young_sector"] == "globally_trivial"
    assert plan.active_descriptor_indices == (0,)
    entry = plan.entries[0]
    assert entry.channel_indices == (0, 1, 2)
    assert entry.validation_report["valid_labels_from"].startswith("ye3t.couplings.count")
    assert entry.component_terms
    assert entry.lower_degree_exponents
    assert all(sum(term["exponents"]) == 2 for term in entry.component_terms)
    assert all(sum(exponents) == 1 for exponents in entry.lower_degree_exponents)
    assert plan.convention_hash


@pytest.mark.parametrize(
    "power,input_L,output_L,multiplicity_index",
    (
        (2, 1, 0, 0),
        (3, 1, 1, 0),
        (4, 2, 0, 0),
    ),
)
def test_complex_symmetric_power_direct_monomials_match_ordered_reference(
    power,
    input_L,
    output_L,
    multiplicity_index,
):
    from ye3t.couplings import (
        symmetric_power_coefficient_entries,
        symmetric_power_product_plan,
    )

    ordered = symmetric_power_coefficient_entries(
        power,
        input_L,
        output_L,
        multiplicity_index,
        basis_convention="complex_magnetic",
    )
    expected_by_component = {}
    for component, local_row, coefficient in ordered:
        exponents = [0] * (2 * input_L + 1)
        for local_component in local_row:
            exponents[int(local_component)] += 1
        key = (int(component), tuple(exponents))
        expected_by_component[key] = (
            expected_by_component.get(key, 0.0 + 0.0j)
            + complex(coefficient)
        )

    for component in range(2 * output_L + 1):
        plan = symmetric_power_product_plan(
            (
                {
                    "descriptor_index": 0,
                    "channel_indices": tuple(range(2 * input_L + 1)),
                    "power": power,
                    "input_L": input_L,
                    "output_L": output_L,
                    "multiplicity_index": multiplicity_index,
                    "component_index": component,
                },
            ),
            descriptor_count=1,
            channel_count=2 * input_L + 1,
            basis_convention="complex_magnetic",
        )
        entry = plan.entries[0]
        actual = {
            tuple(term["exponents"]): complex(term["coefficient"])
            for term in entry.component_terms
        }
        expected = {
            exponents: coefficient
            for (output_component, exponents), coefficient in expected_by_component.items()
            if output_component == component and abs(coefficient) > 1.0e-12
        }
        assert entry.validation_report["coefficient_backend"] == (
            "direct_complex_occupancy_monomials"
        )
        assert actual.keys() == expected.keys()
        for exponents, coefficient in expected.items():
            assert actual[exponents] == pytest.approx(coefficient, abs=1.0e-12)


def test_symmetric_power_resource_policy_selects_bounded_rank9_backend():
    from ye3t.couplings import (
        symmetric_power_materialization_resource_report,
        symmetric_power_product_plan,
    )

    low = symmetric_power_materialization_resource_report(4, 2)
    high = symmetric_power_materialization_resource_report(9, 2)
    assert low["selected_backend"] == "exact_symbolic_independent_occupancy"
    assert low["exceeds_exact_symbolic_limit"] is False
    assert high["selected_backend"] == "certified_numeric_orthogonal_occupancy"
    assert high["exceeds_exact_symbolic_limit"] is True
    assert high["occupancy_dimension"] == 715

    with pytest.raises(MemoryError, match="configured resource limit"):
        symmetric_power_product_plan(
            (
                {
                    "descriptor_index": 0,
                    "channel_indices": tuple(range(5)),
                    "power": 9,
                    "input_L": 2,
                    "output_L": 0,
                    "multiplicity_index": 0,
                    "component_index": 0,
                },
            ),
            descriptor_count=1,
            channel_count=5,
            basis_convention="complex_magnetic",
            coefficient_materialization="exact",
        )


@pytest.mark.parametrize(
    "repeat_count,branch_output_L,target_L,expected_components",
    (
        (3, 1, 1, 3),
        (4, 2, 4, 18),
        (6, 1, 0, 1),
    ),
)
def test_repeated_subtree_symmetric_power_plan_compiles_rank6_to_rank12(
    repeat_count,
    branch_output_L,
    target_L,
    expected_components,
):
    from ye3t.couplings import repeated_subtree_symmetric_power_plan

    plan = repeated_subtree_symmetric_power_plan(
        branch_input_Ls=(1, 1),
        branch_output_L=branch_output_L,
        repeat_count=repeat_count,
        target_L=target_L,
    )

    assert plan.carrier == "Phi_graph_subtree"
    assert plan.target["permutation"] == f"young:{2 * repeat_count}"
    assert plan.target["rotation"]["group"] == "O3"
    assert plan.target["rotation"]["parity"] == 1
    assert plan.validation_report["factorial_branch_table_materialized"] is False
    assert plan.validation_report["nontrivial_outer_identical_branch_projection"] == "provably_zero"
    assert plan.validation_report["source_assembly_index"] == math.comb(
        2 * repeat_count, repeat_count
    )
    assert plan.validation_report["source_assembly_scale"] == pytest.approx(
        math.comb(2 * repeat_count, repeat_count) ** -0.5
    )
    assert plan.provenance["compiler_owner"] == "ye3t"
    assert len(plan.entries) == expected_components
    assert plan.active_descriptor_indices == tuple(range(expected_components))
    assert {entry.power for entry in plan.entries} == {repeat_count}
    assert {entry.input_L for entry in plan.entries} == {branch_output_L}
    assert {entry.output_L for entry in plan.entries} == {target_L}
    assert {entry.component_index for entry in plan.entries} == set(
        range(2 * target_L + 1)
    )


def test_repeated_subtree_symmetric_power_plan_rejects_zero_or_unrealized_sectors():
    from ye3t.couplings import repeated_subtree_symmetric_power_plan

    with pytest.raises(ValueError, match="only outer_partition"):
        repeated_subtree_symmetric_power_plan(
            branch_input_Ls=(1, 1),
            branch_output_L=0,
            repeat_count=4,
            target_L=0,
            outer_partition=(3, 1),
        )
    with pytest.raises(ValueError, match="global trivial parent"):
        repeated_subtree_symmetric_power_plan(
            branch_input_Ls=(1, 1),
            branch_output_L=0,
            repeat_count=4,
            target_L=0,
            parent_partition=(6, 2),
        )
    with pytest.raises(ValueError, match="zero multiplicity"):
        repeated_subtree_symmetric_power_plan(
            branch_input_Ls=(0, 0),
            branch_output_L=0,
            repeat_count=4,
            target_L=1,
        )


def test_typed_repeated_subtree_product_plan_compiles_nontrivial_rank8_parent():
    from ye3t.couplings import typed_repeated_subtree_product_plan

    branch_classes = (
        {
            "branch_input_Ls": (1, 1),
            "branch_output_L": 1,
            "repeat_count": 2,
            "block_output_L": 2,
        },
        {
            "branch_input_Ls": (1, 1),
            "branch_output_L": 1,
            "repeat_count": 2,
            "block_output_L": 2,
        },
    )
    plan = typed_repeated_subtree_product_plan(
        branch_classes=branch_classes,
        parent_partition=(4, 4),
        target_L=0,
    )

    assert plan.rank == 8
    assert plan.block_partitions == ((4,), (4,))
    assert plan.block_output_Ls == (2, 2)
    assert plan.parent_partition == (4, 4)
    assert plan.target_parity == 1
    assert plan.lr_multiplicity == 1
    assert plan.parent_tableau_count == 14
    assert plan.recoupler_report["induced_basis_size"] == 70
    assert plan.recoupler_report["validation"]["passed"] is True
    assert plan.validation_report["factorial_branch_table_materialized"] is False
    assert plan.provenance["rank_coupling_mode"] == (
        "rank_additive_LR_induction"
    )
    assert plan.convention_hash


def test_block_separated_lr_closed_form_matches_exact_young_orthogonal_gauge():
    from ye3t.couplings import typed_repeated_subtree_product_plan
    from ye3t.representations.young_subgroup_specht_coupling import (
        build_young_subgroup_specht_coupling,
    )

    branch_class = {
        "branch_input_Ls": (1, 1),
        "branch_output_L": 0,
        "repeat_count": 1,
        "block_output_L": 0,
    }
    plan = typed_repeated_subtree_product_plan(
        branch_classes=(branch_class, branch_class),
        parent_partition=(2, 2),
        target_L=0,
    )
    exact = build_young_subgroup_specht_coupling(
        ((2,), (2,)),
        (2, 2),
        coefficient_backend="subduction_graph",
    )
    row = next(
        index
        for index, entry in enumerate(exact.tensor.induced_basis)
        if tuple(entry.child_tableau_indices) == (0, 0)
    )
    exact_row = tuple(
        float(vector.coefficients[row])
        for vector in exact.tensor.vectors
    )
    assert plan.canonical_lr_coefficients[0] == pytest.approx(exact_row)


def test_fully_symmetric_lr_closed_form_matches_exact_and_scales_to_rank12():
    from ye3t.couplings import typed_repeated_subtree_product_plan
    from ye3t.representations.young_subgroup_specht_coupling import (
        build_young_subgroup_specht_coupling,
    )

    rank4_branch = {
        "branch_input_Ls": (0, 0),
        "branch_output_L": 0,
        "repeat_count": 1,
        "block_output_L": 0,
    }
    rank4 = typed_repeated_subtree_product_plan(
        branch_classes=(rank4_branch, rank4_branch),
        parent_partition=(4,),
        target_L=0,
    )
    exact = build_young_subgroup_specht_coupling(
        ((2,), (2,)),
        (4,),
        coefficient_backend="subduction_graph",
    )
    row = next(
        index
        for index, entry in enumerate(exact.tensor.induced_basis)
        if tuple(entry.child_tableau_indices) == (0, 0)
    )
    exact_row = tuple(
        float(vector.coefficients[row])
        for vector in exact.tensor.vectors
    )
    assert rank4.canonical_lr_coefficients[0] == pytest.approx(exact_row)

    rank12_branch = {
        "branch_input_Ls": (1, 1),
        "branch_output_L": 1,
        "repeat_count": 3,
        "block_output_L": 1,
    }
    rank12 = typed_repeated_subtree_product_plan(
        branch_classes=(rank12_branch, rank12_branch),
        parent_partition=(12,),
        target_L=0,
    )
    assert rank12.parent_tableau_count == 1
    assert rank12.recoupler_report["induced_basis_size"] == 924
    assert rank12.recoupler_report["coefficient_backend"] == (
        "fully_symmetric_one_row_induction_exact"
    )
    assert rank12.canonical_lr_coefficients[0][0] == pytest.approx(
        1.0 / math.sqrt(924.0)
    )


def test_typed_repeated_subtree_product_plan_compiles_nontrivial_rank12_parent():
    from ye3t.couplings import typed_repeated_subtree_product_plan

    branch_class = {
        "branch_input_Ls": (1, 1),
        "branch_output_L": 1,
        "repeat_count": 3,
        "block_output_L": 1,
    }
    plan = typed_repeated_subtree_product_plan(
        branch_classes=(branch_class, branch_class),
        parent_partition=(6, 6),
        target_L=0,
    )

    assert plan.rank == 12
    assert plan.block_partitions == ((6,), (6,))
    assert plan.parent_partition == (6, 6)
    assert plan.lr_multiplicity == 1
    assert plan.parent_tableau_count == 132
    assert plan.recoupler_report["induced_basis_size"] == 924
    assert plan.validation_report["factorial_branch_table_materialized"] is False


def test_blockwise_symmetric_power_labels_returns_rank12_mixed_representative():
    from ye3t.couplings import blockwise_symmetric_power_labels

    with pytest.warns(RuntimeWarning, match="largest repeated block below half the rank"):
        report = blockwise_symmetric_power_labels(
            content=(1,) * 12,
            input_Ls=(1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3),
            target_L=0,
            label_strategy="representative",
            max_labels=1,
        )

    assert report["provenance"]["compiler_owner"] == "ye3t"
    assert report["validation_report"]["backend"] == "representative_blockwise_symmetric_power_search"
    assert report["validation_report"]["label_count"] == 1
    assert report["validation_report"]["largest_repeated_block_fraction"] == pytest.approx(1.0 / 3.0)
    assert report["validation_report"]["policy_warnings"]
    label = report["labels"][0]
    assert label.rank == 12
    assert label.L_R == 0
    assert label.basis_key[0] == "node"
    assert tuple(label.l_tuple) == (1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3)


def test_blockwise_symmetric_power_product_plan_records_mixed_blocks():
    from ye3t.couplings import BlockwiseSymmetricPowerProductPlan, blockwise_symmetric_power_product_plan

    plan = blockwise_symmetric_power_product_plan(
        (
            {
                "descriptor_index": 0,
                "rank": 12,
                "blocks": (
                    {"kind": "sym", "n": 1, "l": 1, "k_b": 4, "Lambda": 0, "channel_indices": (0, 1, 2)},
                    {"kind": "sym", "n": 1, "l": 2, "k_b": 4, "Lambda": 0, "channel_indices": (3, 4, 5, 6, 7)},
                    {"kind": "sym", "n": 1, "l": 3, "k_b": 4, "Lambda": 0, "channel_indices": (8, 9, 10, 11, 12, 13, 14)},
                ),
                "schedule_term_count": 1,
                "stabilizer": "S_4 x S_4 x S_4",
            },
        ),
        descriptor_count=1,
        channel_count=15,
        normalization_convention="soft_neighbor",
    )

    assert isinstance(plan, BlockwiseSymmetricPowerProductPlan)
    assert plan.active_descriptor_indices == (0,)
    assert plan.validation_report["supports_descriptor_adjoint"] is True
    assert plan.provenance["compiler_owner"] == "ye3t"
    payload = plan.to_dict()
    assert payload["entries"][0]["stabilizer"] == "S_4 x S_4 x S_4"
    assert payload["entries"][0]["blocks"][2]["l"] == 3
    assert payload["backend"] == "blockwise_symmetric_power_factorized_schedule"


def test_exterior_power_product_plan_records_sign_basis_and_duplicate_policy():
    from ye3t.couplings import (
        ExteriorPowerProductPlan,
        compile_exterior_power_product,
        exterior_basis_tuples,
        exterior_component_index_and_sign,
        exterior_power_count,
        exterior_power_product_plan,
    )

    plan = exterior_power_product_plan(one_particle_dim=4, rank=2, normalization_convention="A_normalized")

    assert isinstance(plan, ExteriorPowerProductPlan)
    assert plan.basis_tuples == ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    assert exterior_basis_tuples(4, 2) == plan.basis_tuples
    assert plan.output_dim == 6
    assert plan.sign_convention["partition"] == (1, 1)
    assert plan.sign_convention["duplicate_policy"] == "zero"
    assert plan.validation_report["duplicate_one_particle_indices_vanish"] is True
    assert plan.validation_report["not_spinful_su2"] is True
    assert plan.provenance["compiler_owner"] == "ye3t"
    assert plan.convention_hash

    assert exterior_component_index_and_sign((2, 0), one_particle_dim=4, rank=2) == (1, -1)
    assert exterior_component_index_and_sign((0, 2), one_particle_dim=4, rank=2) == (1, 1)
    assert exterior_component_index_and_sign((2, 2), one_particle_dim=4, rank=2) is None

    count_report = exterior_power_count(one_particle_dim=5, rank=3)
    compiled = compile_exterior_power_product(one_particle_dim=5, rank=3)
    assert count_report["count"] == 10
    assert count_report["basis_tuples"] == compiled.basis_tuples
    assert count_report["sign_partition"] == (1, 1, 1)
    assert count_report["spin_scope"] == "spinless_exterior_only_not_SU2_fermions"
    assert compiled.validation_report["count_matches_basis_size"] is True
    assert compiled.provenance["api"] == "ye3t.couplings.compile_exterior_power_product"
    assert compiled.index_and_sign((3, 1, 4)) == (8, -1)
    assert compiled.index_and_sign((4, 4, 1)) is None


def test_couplings_reports_cover_a_s_phi_and_message_state_carriers():
    from ye3t.couplings import count, plan, sector_records, slot_specht_partitions

    a_s = count(
        content=(1, 1, 1),
        input_Ls=(0, 1, 1),
        target_L=0,
        carrier="A_s",
        target_permutation="young:2,1",
        carrier_options={
            "role_coordinate_policy": "role_resolved",
            "slot_count": 3,
            "permuted_slot_count": 2,
            "slot_specht_partitions": ("trivial", "sign"),
        },
    )

    assert a_s.carrier == "A_s"
    assert a_s.validation_report["carrier_validation"]["slot_permutation_scope"]["permuted_slot_count"] == 2
    assert a_s.validation_report["carrier_validation"]["slot_specht_partitions"] == ((2,), (1, 1))
    assert a_s.validation_report["carrier_validation"]["role_contract"]["role_module"]["permuted_role_count"] == 2
    assert a_s.validation_report["carrier_validation"]["role_contract"]["nontrivial_sector_requested"] is True
    assert slot_specht_partitions("standard", slot_count=3) == ((2, 1),)

    phi = count(
        content=(1, 2, 3),
        input_Ls=(0, 1, 1),
        target_L=0,
        carrier="Phi",
        target_permutation="young:2,1",
        carrier_options={
            "slot_count": 3,
            "permuted_slot_count": 3,
            "slot_orbit_partition": (3,),
            "factor_action": "permute_explicit_phi_tensor_product_factors",
            "distinguish_slots": True,
        },
    )
    assert phi.validation_report["carrier_validation"]["scope"] == "Phi_cluster_basis_label_request"
    assert phi.validation_report["carrier_validation"]["slot_count"] == 3
    assert phi.validation_report["carrier_validation"]["slot_permutation_scope"]["permuted_slot_count"] == 3
    assert phi.validation_report["carrier_validation"]["slot_orbit_partition"] == (3,)
    assert phi.validation_report["carrier_validation"]["same_factor_permutation_contract_as_role_resolved_A_s"] is True
    assert phi.validation_report["carrier_validation"]["materialization_owner"] == "ye3t-ace"

    a_s_same_factors = count(
        content=(1, 2, 3),
        input_Ls=(0, 1, 1),
        target_L=0,
        carrier="A_s",
        target_permutation="young:2,1",
        carrier_options={
            "role_coordinate_policy": "role_resolved",
            "slot_count": 3,
            "permuted_slot_count": 3,
        },
    )
    assert phi.counts_by_target == a_s_same_factors.counts_by_target
    assert phi.labels_by_target == a_s_same_factors.labels_by_target

    message = plan(
        content=(1, 1),
        input_Ls=(0, 0),
        target_L=0,
        carrier="message_state",
        target_permutation="trivial",
    )
    assert message.validation_report["carrier_validation"]["valid_labels_from"] == "ye3t.couplings.sector_records"
    assert sector_records((1, 1), (0, 0))


def test_couplings_reject_invalid_a_s_slot_scope():
    import pytest

    from ye3t.couplings import count

    with pytest.raises(ValueError, match="permuted_slot_count"):
        count(
            content=(1, 1),
            input_Ls=(0, 0),
            target_L=0,
            carrier="A_s",
            target_permutation="young:1,1",
            carrier_options={"role_coordinate_policy": "role_resolved", "slot_count": 2, "permuted_slot_count": 3},
        )


def test_couplings_reject_invalid_phi_slot_metadata():
    import pytest

    from ye3t.couplings import count

    with pytest.raises(ValueError, match="slot_count must match content rank"):
        count(
            content=(1, 2),
            input_Ls=(1, 1),
            target_L=0,
            carrier="Phi",
            target_permutation="trivial",
            carrier_options={
                "slot_count": 3,
                "factor_action": "permute_explicit_phi_tensor_product_factors",
            },
        )

    with pytest.raises(ValueError, match="slot_orbit_partition"):
        count(
            content=(1, 2, 3),
            input_Ls=(1, 1, 1),
            target_L=0,
            carrier="Phi",
            target_permutation="trivial",
            carrier_options={
                "slot_count": 3,
                "slot_orbit_partition": (2,),
                "factor_action": "permute_explicit_phi_tensor_product_factors",
            },
        )


def test_couplings_reject_nontrivial_a_s_sector_when_roles_collapse():
    import pytest

    from ye3t.couplings import count

    with pytest.raises(ValueError, match="identical role filters"):
        count(
            content=(1, 1, 1),
            input_Ls=(0, 0, 0),
            target_L=0,
            carrier="A_s",
            target_permutation="young:2,1",
            carrier_options={
                "role_coordinate_policy": "role_resolved",
                "slot_count": 3,
                "slot_specht_partitions": ("standard",),
                "identical_role_filters_declared": True,
            },
        )


def test_couplings_force_gates_a_s_ordinary_ace_symmetric_density_fallback():
    import pytest

    from ye3t.couplings import count

    with pytest.raises(ValueError, match="must be forced"):
        count(
            content=(1, 1),
            input_Ls=(0, 0),
            target_L=0,
            carrier="A_s",
            target_permutation="trivial",
            carrier_options={
                "ordinary_ace_symmetric_density": True,
                "slot_count": 2,
            },
        )

    report = count(
        content=(1, 1),
        input_Ls=(0, 0),
        target_L=0,
        carrier="A_s",
        target_permutation="trivial",
        carrier_options={
            "ordinary_ace_symmetric_density": True,
            "force_ordinary_ace_symmetric_density": True,
            "slot_count": 2,
            "slot_specht_partitions": ("trivial",),
        },
    )
    carrier_validation = report.validation_report["carrier_validation"]

    assert carrier_validation["role_contract"]["ordinary_ace_symmetric_density"] is True
    assert carrier_validation["role_contract"]["warnings"]
    assert carrier_validation["passed"] is True

    with pytest.raises(ValueError, match="ordinary ACE symmetric-density"):
        count(
            content=(1, 1),
            input_Ls=(0, 0),
            target_L=0,
            carrier="A_s",
            target_permutation="young:1,1",
            carrier_options={
                "ordinary_ace_symmetric_density": True,
                "force_ordinary_ace_symmetric_density": True,
                "slot_count": 2,
                "slot_specht_partitions": ("sign",),
            },
        )


def test_couplings_count_owns_primitive_basis_label_filtering():
    from ye3t.couplings import count, primitive_compact_label_report

    exact = count(
        content=(1,),
        input_Ls=(0,),
        target_L=0,
        carrier="ACE_density",
        target_permutation="trivial",
    )
    primitive = count(
        content=(1,),
        input_Ls=(0,),
        target_L=0,
        carrier="ACE_density",
        target_permutation="trivial",
        carrier_options={"basis_mode": "primitive_full"},
    )

    assert primitive.validation_report["basis_mode"] == "primitive_full"
    assert primitive.validation_report["valid_labels_from"] == "ye3t.couplings.primitive_compact_label_report"
    assert primitive.validation_report["primitive_validation"]["backend"] == "exact_product_engine"
    assert set(primitive.labels_for_target(0)).issubset(set(exact.labels_for_target(0)))
    assert primitive.labels_for_target(0)

    direct = primitive_compact_label_report(
        nin=(1,),
        lin=(0,),
        target_L=0,
        tree_type="balanced",
        basis_mode="primitive_full",
    )
    assert tuple(direct["labels"]) == primitive.labels_for_target(0)


def test_symmetric_power_product_plan_uses_native_rank16_scalar_terms():
    from ye3t.couplings import symmetric_power_product_plan

    plan = symmetric_power_product_plan(
        (
            {
                "descriptor_index": 0,
                "channel_indices": (0, 1, 2),
                "power": 16,
                "input_L": 1,
                "output_L": 0,
                "multiplicity_index": 0,
                "component_index": 0,
            },
        ),
        descriptor_count=1,
        channel_count=3,
        basis_convention="real_tesseral",
    )
    entry = plan.entries[0]
    assert entry.validation_report["coefficient_backend"] == "native_real_l1_even_scalar_norm_power"
    assert len(entry.component_terms) == 45

    x = (0.2, -0.3, 0.4)
    value = 0.0
    for term in entry.component_terms:
        monomial = 1.0
        for component, exponent in zip(x, term["exponents"]):
            monomial *= float(component) ** int(exponent)
        value += float(complex(term["coefficient"]).real) * monomial
    expected = sum(component * component for component in x) ** 8
    assert abs(value - expected) <= 1.0e-14


def test_complex_scalar_symmetric_power_keeps_auxiliary_routes_out_of_plan_hash():
    from ye3t.couplings import symmetric_power_product_plan

    plan = symmetric_power_product_plan(
        (
            {
                "descriptor_index": 0,
                "channel_indices": (0, 1, 2),
                "power": 8,
                "input_L": 1,
                "output_L": 0,
                "multiplicity_index": 0,
                "component_index": 0,
            },
        ),
        descriptor_count=1,
        channel_count=3,
        basis_convention="complex_magnetic",
    )
    entry = plan.entries[0]
    assert "scalar_invariant_power_factorization" not in entry.validation_report

    values = (0.2 - 0.1j, -0.3 + 0.4j, 0.15 + 0.05j)

    def polynomial(terms):
        result = 0.0 + 0.0j
        for term in terms:
            monomial = complex(term["coefficient"])
            for value, exponent in zip(values, term["exponents"]):
                monomial *= value ** int(exponent)
            result += monomial
        return result

    direct = polynomial(entry.component_terms)
    quadratic = values[1] ** 2 - 2.0 * values[0] * values[2]
    factored = quadratic**4
    assert abs(direct - factored) <= 2.0e-12 * max(1.0, abs(direct))


def test_couplings_compile_a_s_slot_specht_projectors():
    from ye3t.couplings import compile_A_s_slot_specht_projectors

    report = compile_A_s_slot_specht_projectors(
        slot_count=4,
        blocks=((0, 1), (2, 3)),
    )
    payload = report.to_dict()

    assert report.validation_report["passed"] is True
    assert report.validation_report["coordinate_resolved"] is True
    assert report.validation_report["matrix_unit_resolved"] is False
    assert report.validation_report["central_global_coupler_consumed"] is False
    assert report.validation_report["max_coordinate_orthonormality_error"] <= 1.0e-12
    assert report.validation_report["max_projector_reconstruction_error"] <= 1.0e-12
    assert report.validation_report["max_generator_orthogonality_error"] <= 1.0e-12
    assert report.validation_report["max_generator_equivariance_error"] <= 1.0e-12
    assert report.provenance["api"] == "ye3t.couplings.compile_A_s_slot_specht_projectors"
    assert report.provenance["coordinate_basis"] == "orthonormal_projector_image_basis"
    assert report.provenance["valid_labels_from"] == "ye3t.couplings.slot_specht_partitions"
    assert len(report.records) == 4
    assert {tuple(record["partition"]) for record in report.records} == {(2,), (1, 1)}
    assert all(record["coordinate_dim"] == record["rank"] for record in report.records)
    assert all(record["generator_actions"] for record in report.records)
    assert payload["backend"] == "finite_group_central_idempotent"
    assert payload["validation_report"]["max_idempotency_error"] <= 1.0e-12


def test_couplings_compile_a_s_young_subgroup_slot_intertwiners():
    from ye3t.couplings import compile_A_s_young_subgroup_slot_intertwiners

    report = compile_A_s_young_subgroup_slot_intertwiners(
        slot_count=4,
        blocks=((0, 1), (2, 3)),
    )
    payload = report.to_dict()

    assert report.validation_report["passed"] is True
    assert report.validation_report["orbital_count"] == 6
    assert report.validation_report["generator_count"] == 2
    assert report.validation_report["max_equivariance_error"] <= 1.0e-12
    assert report.validation_report["max_orthonormality_error"] <= 1.0e-12
    assert report.validation_report["matrix_unit_resolved"] is False
    assert report.validation_report["central_global_coupler_consumed"] is False
    assert report.provenance["api"] == "ye3t.couplings.compile_A_s_young_subgroup_slot_intertwiners"
    assert report.provenance["basis_formula"] == "diagonal_orbit_indicator_basis_for_permutation_module_Hom_space"
    assert payload["backend"] == "exact_diagonal_orbital_permutation_module"
    assert len(payload["records"]) == 6
    assert all(record["coefficient_source"] == "ye3t.couplings.compile_A_s_young_subgroup_slot_intertwiners" for record in report.records)


def test_couplings_compile_a_s_slot_specht_matrix_units():
    from ye3t.couplings import compile_A_s_slot_specht_matrix_units

    report = compile_A_s_slot_specht_matrix_units(
        slot_count=3,
        power=3,
        partition=(2, 1),
    )
    payload = report.to_dict()

    assert report.validation_report["passed"] is True
    assert report.validation_report["matrix_unit_resolved"] is True
    assert report.validation_report["central_global_coupler_consumed"] is False
    assert report.validation_report["matrix_unit_count"] == 4
    assert report.validation_report["max_matrix_unit_algebra_residual"] <= 1.0e-10
    assert report.validation_report["max_projector_difference_from_diagonal_sum"] <= 1.0e-10
    assert report.provenance["api"] == "ye3t.couplings.compile_A_s_slot_specht_matrix_units"
    assert report.provenance["runtime_scope"] == "matrix_unit_carrier_not_global_Young_E3_coupler"
    assert payload["backend"] == "young_orthogonal_matrix_units_on_slot_tuple_power"
    assert len(payload["records"]) == 4
    assert all(record["coefficient_source"] == "ye3t.couplings.compile_A_s_slot_specht_matrix_units" for record in report.records)
