import copy
import os
from pathlib import Path

import pytest
import torch

from ye3t.api import (
    JointYoungCGProduct,
    JointYoungCGTensorSchedule,
    SchurWeylGuidedTreeProduct,
    YoungE3TensorProduct,
    compile_schur_weyl_guided_tree_product,
    compile_schur_weyl_guided_tree_product_from_coupler,
)
from ye3t.core.subtree_dag import cg_exact
from ye3t.representations import ExactSymbolicProjectorGeneralizedBasisBuilder, PermutationIrrep, PermutationSubgroup
from ye3t.representations.young_subgroup_specht_coupling import (
    build_cached_young_subgroup_specht_coupling,
)
from ye3t.runtime.generalized import (
    GeneralizedExactRuntimeBlock,
    GeneralizedExactRuntimeIrreps,
    GeneralizedFullTensorProduct,
    _require_numerically_real,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _rank_one_trivial_sector(L=1):
    subgroup = PermutationSubgroup.from_nl((1,), (int(L),))
    irrep = PermutationIrrep.trivial_for_subgroup(subgroup)
    return ExactSymbolicProjectorGeneralizedBasisBuilder((1,), (int(L),), irrep).build()


def _rank_two_trivial_sector():
    subgroup = PermutationSubgroup.from_nl((1, 1), (1, 1))
    irrep = PermutationIrrep.trivial_for_subgroup(subgroup)
    return ExactSymbolicProjectorGeneralizedBasisBuilder((1, 1), (1, 1), irrep).build()


def test_public_joint_young_cg_product_uses_exact_generalized_product():
    assert issubclass(JointYoungCGProduct, GeneralizedFullTensorProduct)
    assert YoungE3TensorProduct is JointYoungCGProduct


def test_joint_young_cg_product_builds_exact_static_instructions():
    sector = _rank_one_trivial_sector()
    irreps_left = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(1, sector, 1, 0),))
    irreps_right = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(1, sector, 1, 0),))

    product = JointYoungCGProduct(irreps_left, irreps_right, tree_type="balanced")
    report = product.instruction_report()

    assert report["num_instructions"] > 0
    assert product.irreps_out.dim > 0
    for instruction in report["instructions"]:
        assert 0 <= instruction["output_L"] <= 2
        assert instruction["source_tensor_shape"][1:] == (3, 3)
        assert instruction["source_tensor_nnz"] > 0
        assert instruction["coordinate_nnz"] > 0


def test_joint_young_cg_product_rejects_unfused_path_output_mode():
    import pytest

    sector = _rank_one_trivial_sector()
    irreps_left = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(1, sector, 1, 0),))
    irreps_right = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(1, sector, 1, 0),))

    with pytest.raises(ValueError, match="fused_irrep"):
        JointYoungCGProduct(
            irreps_left,
            irreps_right,
            tree_type="balanced",
            output_mode="unfused_path_diagnostic",
        )


def test_schur_weyl_guided_tree_materializes_rank3_nonrepeated_slots():
    product = compile_schur_weyl_guided_tree_product((1, 1, 1), (0, 1, 2))

    assert isinstance(product, SchurWeylGuidedTreeProduct)
    assert product.dim == 15
    reports = product.node_reports()
    internal_reports = [report for report in reports if not report["is_leaf"]]
    assert internal_reports
    assert all(report["dimension_matches_plan"] for report in internal_reports)
    assert all(not report["duplicate_output_labels"] for report in internal_reports)
    assert reports[-1]["span"] == (0, 3)
    assert reports[-1]["plan_selected_dim"] == 15

    leaves = (
        torch.arange(2, dtype=torch.float64).reshape(2, 1) + 1.0,
        torch.arange(6, dtype=torch.float64).reshape(2, 3) / 7.0,
        torch.arange(10, dtype=torch.float64).reshape(2, 5) / 11.0,
    )
    out = product(leaves)
    assert out.shape == (2, product.dim)
    assert torch.isfinite(out.real).all()
    assert torch.isfinite(out.imag).all()


def test_schur_weyl_guided_tree_materializes_rank4_balanced_scalar_tree():
    product = compile_schur_weyl_guided_tree_product((1, 1, 1, 1), (0, 0, 0, 0))

    assert product.dim == 1
    reports = product.node_reports()
    assert reports[-1]["span"] == (0, 4)
    assert reports[-1]["plan_selected_dim"] == 1
    assert len(product.products) == 3
    assert all(report["dimension_matches_plan"] for report in reports)

    leaves = tuple(torch.full((2, 1), float(index + 1), dtype=torch.float64) for index in range(4))
    out = product(leaves)
    assert out.shape == (2, 1)
    assert torch.isfinite(out.real).all()
    assert torch.isfinite(out.imag).all()


def test_schur_weyl_guided_tree_can_compile_from_global_coupler_certificate():
    from ye3t import CompileBalancedTree, CompileGlobalYE3TCouplers, YE3TRotationTarget, YE3TSpec

    spec = YE3TSpec(
        content=(1, 1, 1, 1),
        target_permutation="trivial",
        target_rotation=YE3TRotationTarget(L_R=0),
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        validation_scope="projectors",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (0, 0, 0, 0)},
    )
    coupler = CompileGlobalYE3TCouplers(spec)
    product = compile_schur_weyl_guided_tree_product_from_coupler(coupler)
    report = product.static_schedule_report()
    compilation = CompileBalancedTree(spec, build_runtime_tree=True)
    compilation_report = compilation.runtime_tree.static_schedule_report()

    assert product.dim == 1
    assert report["backend_role"] == "global_coupler_guided_schur_weyl_tree_backend"
    assert report["provenance"]["consumes_global_coupler_record"] is True
    assert report["provenance"]["global_coupler_coefficient_hash"] == coupler.certificate.coefficient_hash
    assert report["provenance"]["global_coupler_alpha_label_count"] == len(coupler.alpha_labels())
    assert report["provenance"]["global_coupler_certificate"]["passed"] is True
    assert report["backend_certificate"]["passed"] is True
    assert (
        report["backend_certificate"]["backend_role"]
        == "global_coupler_guided_schur_weyl_tree_backend"
    )
    assert report["backend_certificate"]["runtime_status"] == "implemented_under_validation"
    assert report["backend_certificate"]["checks"]["consumes_global_coupler_record"] is True
    assert report["backend_certificate"]["checks"]["global_coupler_certificate_passed"] is True
    assert report["backend_certificate"]["checks"]["global_coupler_label_count_positive"] is True
    assert report["backend_certificate"]["checks"]["root_angular_target_from_global_coupler"] is True
    assert report["backend_certificate"]["checks"]["root_permutation_target_from_global_coupler"] is True
    assert report["backend_certificate"]["checks"]["root_permutation_target_enforced_when_representable"] is True
    assert report["backend_certificate"]["checks"]["single_root_factor_target_enforced"] is True
    assert report["backend_certificate"]["checks"]["full_global_induction_coset_lift_not_required"] is True
    assert report["backend_certificate"]["checks"]["root_output_partitions_match_enforced_target"] is True
    assert report["backend_certificate"]["checks"]["all_node_dimensions_match_plans"] is True
    assert report["backend_certificate"]["global_coupler_coefficient_hash"] == coupler.certificate.coefficient_hash
    assert report["backend_certificate"]["global_coupler_target_permutation"] == "trivial"
    assert report["backend_certificate"]["global_coupler_target_partition"] == (4,)
    assert report["backend_certificate"]["root_permutation_target_partition_enforced"] == (4,)
    assert report["backend_certificate"]["root_subgroup_factor_count"] == 1
    assert report["backend_certificate"]["root_subgroup_factor_multiplicities"] == (4,)
    assert report["backend_certificate"]["root_permutation_target_directly_enforceable"] is True
    assert report["backend_certificate"]["single_root_factor_target_enforced"] is True
    assert report["backend_certificate"]["full_global_induction_coset_lift_required"] is False
    assert (
        report["backend_certificate"]["root_permutation_target_direct_enforcement_kind"]
        == "single_root_symmetric_group_factor"
    )
    assert set(report["backend_certificate"]["root_output_partition_signatures"]) == {((4,),)}
    assert report["backend_certificate"]["root_angular_target"]["L_R"] == 0
    assert (
        report["backend_certificate"]["root_permutation_target_status"]
        == "enforced_single_root_symmetric_group_factor"
    )
    assert product.backend_certificate_report() == report["backend_certificate"]
    assert compilation.certificate.checks["runtime_tree_built"] is True
    assert compilation.certificate.checks["runtime_tree_backend_certificate_passed"] is True
    assert compilation.certificate.checks["runtime_tree_single_root_factor_target_enforced"] is True
    assert compilation.certificate.checks["runtime_tree_full_global_induction_coset_lift_not_required"] is True
    assert compilation.certificate.provenance["runtime_tree"] == "compile_schur_weyl_guided_tree_product_from_coupler"
    assert compilation.certificate.provenance["runtime_tree_backend_certificate"]["passed"] is True
    assert compilation_report["provenance"]["consumes_global_coupler_record"] is True
    assert compilation_report["backend_certificate"]["passed"] is True


def test_schur_weyl_guided_tree_reports_multifactor_root_target_limitation():
    from ye3t import CompileGlobalYE3TCouplers, YE3TRotationTarget, YE3TSpec

    spec = YE3TSpec(
        content=(1, 2),
        target_permutation="trivial",
        target_rotation=YE3TRotationTarget(L_R=0),
        carrier="external_tensor",
        coefficient_backend="global_coupler",
        validation_scope="projectors",
        runtime_status="planned_not_public",
        metadata={"input_Ls": (0, 0)},
    )
    coupler = CompileGlobalYE3TCouplers(spec)
    product = compile_schur_weyl_guided_tree_product_from_coupler(coupler)
    report = product.backend_certificate_report()

    assert report["global_coupler_target_partition"] == (2,)
    assert report["root_permutation_target_partition_enforced"] is None
    assert report["root_subgroup_factor_count"] == 2
    assert report["root_subgroup_factor_multiplicities"] == (1, 1)
    assert report["root_permutation_target_directly_enforceable"] is False
    assert report["single_root_factor_target_enforced"] is False
    assert report["full_global_induction_coset_lift_required"] is True
    assert report["root_permutation_target_direct_enforcement_kind"] == "requires_global_induction_coset_lift"
    assert report["root_permutation_target_status"] == (
        "not_directly_enforced_multiple_root_subgroup_factors_requires_global_induction_coset_lift"
    )
    assert report["checks"]["root_permutation_target_from_global_coupler"] is True
    assert report["checks"]["root_permutation_target_enforced_when_representable"] is False
    assert report["checks"]["single_root_factor_target_enforced"] is False
    assert report["checks"]["full_global_induction_coset_lift_not_required"] is False
    assert report["checks"]["root_output_partitions_match_enforced_target"] is True
    assert report["passed"] is False
    assert report["runtime_status"] == "planned_not_public"


def test_schur_weyl_guided_tree_accepts_symbolic_factor_channels():
    from ye3t import CompileGlobalYE3TCouplers, YE3TRotationTarget, YE3TSpec

    coupler = CompileGlobalYE3TCouplers(
        YE3TSpec(
            content=("radial_a", "radial_a"),
            target_permutation="trivial",
            target_rotation=YE3TRotationTarget(L_R=0),
            carrier="external_tensor",
            coefficient_backend="global_coupler",
            validation_scope="projectors",
            runtime_status="planned_not_public",
            metadata={"input_Ls": (1, 1)},
        )
    )
    product = compile_schur_weyl_guided_tree_product_from_coupler(coupler)

    assert product.dim == 1
    assert product.provenance["global_coupler_factor_channels"] == (
        "radial_a", "radial_a"
    )
    assert product.provenance["global_coupler_alpha_label_count"] == len(coupler.alpha_labels())
    assert product.provenance["global_coupler_target_partition"] == (2,)
    assert product.provenance["root_permutation_target_partition_enforced"] == (2,)


def test_schur_weyl_guided_tree_materializes_rank3_repeated_slots_as_fused_irreps():
    product = compile_schur_weyl_guided_tree_product((1, 1, 1), (1, 1, 1))

    assert product.dim == 27
    reports = product.node_reports()
    assert reports[-1]["span"] == (0, 3)
    assert reports[-1]["plan_selected_dim"] == 27
    assert all(report["dimension_matches_plan"] for report in reports)
    assert all(not report["duplicate_output_labels"] for report in reports)
    assert {
        label
        for label, count in reports[-1]["output_label_counts"]
        if int(count) == 1
    } == {
        "L=0 x S_3:[1,1,1]",
        "L=1 x S_3:[2,1]",
        "L=1 x S_3:[3]",
        "L=2 x S_3:[2,1]",
        "L=3 x S_3:[3]",
    }


def test_schur_weyl_guided_tree_filters_root_parity():
    import pytest

    odd_scalar = compile_schur_weyl_guided_tree_product(
        (1, 1, 1),
        (1, 1, 1),
        root_target_Ls=(0,),
        root_target_parity="odd",
    )
    assert odd_scalar.dim == 1
    assert tuple(block.label.to_string() for block in odd_scalar.irreps_out.blocks) == (
        "L=0 x S_3:[1,1,1]",
    )

    with pytest.raises(ValueError, match="No requested labels"):
        compile_schur_weyl_guided_tree_product(
            (1, 1, 1),
            (1, 1, 1),
            root_target_Ls=(0,),
            root_target_parity="even",
        )

    even_scalar = compile_schur_weyl_guided_tree_product(
        (1, 1),
        (1, 1),
        root_target_Ls=(0,),
        root_target_parity="even",
    )
    assert even_scalar.dim == 1
    assert tuple(block.label.to_string() for block in even_scalar.irreps_out.blocks) == ("L=0 x S_2:[2]",)


def test_joint_young_cg_product_target_parity_filters_output_sectors():
    import pytest

    sector = _rank_one_trivial_sector()
    irreps_left = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(1, sector, 1, 0),))
    irreps_right = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(1, sector, 1, 0),))

    even = JointYoungCGProduct(irreps_left, irreps_right, tree_type="balanced", target_parity="even")
    assert even.irreps_out.dim == 9

    with pytest.raises(ValueError, match="No valid generalized tensor-product output blocks"):
        JointYoungCGProduct(irreps_left, irreps_right, tree_type="balanced", target_parity="odd")


def test_coupled_label_and_young_product_paths_multiply_o3_parity():
    from ye3t.representations import AngularIrrep, CoupledIrrepLabel
    from ye3t.representations.young_sectors import young_product_paths

    sector = _rank_one_trivial_sector()
    odd = CoupledIrrepLabel(
        angular=AngularIrrep(1),
        permutation=sector.permutation_irrep,
        parity=-1,
    )
    even = CoupledIrrepLabel(
        angular=AngularIrrep(1),
        permutation=sector.permutation_irrep,
        parity=1,
    )

    odd_squared = young_product_paths(odd, odd)
    assert odd_squared
    assert {path.output_label.parity for path in odd_squared} == {1}
    assert young_product_paths(odd, odd, target_parity=-1) == ()
    assert "p=-1" in odd.to_string()
    assert CoupledIrrepLabel.from_dict(odd.as_dict()) == odd

    with pytest.raises(ValueError, match="cannot mix"):
        young_product_paths(odd, CoupledIrrepLabel(
            angular=AngularIrrep(1),
            permutation=sector.permutation_irrep,
        ))
    assert {path.output_label.parity for path in young_product_paths(odd, even)} == {-1}


def test_joint_young_cg_product_forward_is_finite_and_reported_by_joint_name():
    sector = _rank_one_trivial_sector()
    irreps_left = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(2, sector, 1, 0),))
    irreps_right = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(1, sector, 1, 0),))
    product = JointYoungCGProduct(irreps_left, irreps_right, tree_type="balanced")

    left = torch.arange(12, dtype=torch.float64).reshape(2, 6) / 7.0
    right = torch.arange(6, dtype=torch.float64).reshape(2, 3) / 5.0
    out = product(left, right)

    assert out.shape == (2, product.irreps_out.dim)
    assert torch.isfinite(out.real).all()
    assert torch.isfinite(out.imag).all()
    assert "JointYoungCGProduct" in product.format_instruction_report()


def test_joint_young_cg_product_filters_by_L_max_and_requested_target():
    sector = _rank_one_trivial_sector()
    irreps_left = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(1, sector, 1, 0),))
    irreps_right = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(1, sector, 1, 0),))

    capped = JointYoungCGProduct(irreps_left, irreps_right, tree_type="balanced", L_max=1)
    assert {int(block.L) for block in capped.irreps_out.blocks} == {0, 1}
    assert all(int(block.L) <= 1 for block in capped.irreps_out.blocks)

    target = capped.irreps_out.blocks[0].label
    requested = JointYoungCGProduct(
        irreps_left,
        irreps_right,
        tree_type="balanced",
        requested_targets=(target,),
    )
    assert tuple(block.label.to_string() for block in requested.irreps_out.blocks) == (target.to_string(),)


def test_o3_joint_young_cg_product_accepts_requested_target_with_parity_label():
    subgroup = PermutationSubgroup.from_nl((1,), (1,))
    irrep = PermutationIrrep.trivial_for_subgroup(subgroup)
    sector = ExactSymbolicProjectorGeneralizedBasisBuilder(
        (1,),
        (1,),
        irrep,
        spatial_symmetry="O3",
    ).build()
    runtime_irreps = GeneralizedExactRuntimeIrreps(
        (GeneralizedExactRuntimeBlock(1, sector, 1, 0),)
    )

    requested = JointYoungCGProduct(
        runtime_irreps,
        runtime_irreps,
        tree_type="balanced",
        requested_targets=("L=0,p=+1 x S_2:[2]",),
    )

    assert tuple(block.label.to_string() for block in requested.irreps_out.blocks) == (
        "L=0,p=+1 x S_2:[2]",
    )


def _rank_three_mixed_parent_irreps():
    rank_one_sector = _rank_one_trivial_sector()
    rank_one = GeneralizedExactRuntimeIrreps(
        (GeneralizedExactRuntimeBlock(1, rank_one_sector, 1, 0),)
    )
    rank_two_product = JointYoungCGProduct(
        rank_one,
        rank_one,
        requested_targets=("L=1 x S_2:[1,1]",),
        L_max=1,
    )
    rank_three_product = JointYoungCGProduct(
        rank_two_product.irreps_out,
        rank_one,
        requested_targets=("L=1 x S_3:[2,1]",),
        L_max=1,
    )
    return rank_three_product.irreps_out


def test_exact_parent_runtime_irreps_construct_from_compiled_carrier_layout():
    from ye3t.execution_plan import (
        YE3T_PRIMARY_CONVENTION,
        YE3TCarrierKey,
        YE3TCarrierLayout,
    )
    from ye3t.runtime.generalized import exact_runtime_carrier_records

    layout = YE3TCarrierLayout(
        key=YE3TCarrierKey(
            rank=3,
            partition=(2, 1),
            rotation_L=2,
            convention_id=YE3T_PRIMARY_CONVENTION,
        ),
        channel_count=3,
        tableau_count=2,
        magnetic_count=5,
    )
    irreps = GeneralizedExactRuntimeIrreps.from_carrier_layout(layout)
    records = exact_runtime_carrier_records(
        irreps,
        convention_id=YE3T_PRIMARY_CONVENTION,
    )

    assert irreps.dim == layout.width
    assert len(irreps.blocks) == 1
    assert irreps.blocks[0].mul == 3
    assert irreps.blocks[0].label.to_string() == "L=2 x S_3:[2,1]"
    assert records[0]["carrier_key"] == layout.key.to_dict()
    assert records[0]["carrier_layout"] == layout.to_dict()


def test_o3_exact_parent_runtime_irreps_preserve_signed_parity():
    from ye3t.execution_plan import (
        YE3T_O3_PRIMARY_CONVENTION,
        YE3TCarrierKey,
        YE3TCarrierLayout,
    )
    from ye3t.runtime.generalized import exact_runtime_carrier_records

    layout = YE3TCarrierLayout(
        key=YE3TCarrierKey(
            rank=3,
            partition=(2, 1),
            rotation_L=2,
            convention_id=YE3T_O3_PRIMARY_CONVENTION,
            parity=-1,
        ),
        channel_count=3,
        tableau_count=2,
        magnetic_count=5,
    )
    irreps = GeneralizedExactRuntimeIrreps.from_carrier_layout(layout)
    records = exact_runtime_carrier_records(
        irreps,
        convention_id=YE3T_O3_PRIMARY_CONVENTION,
    )

    assert irreps.blocks[0].label.parity == -1
    assert records[0]["parity"] == -1
    assert records[0]["carrier_status"] == "exact_parent_S_N_x_O3"
    assert records[0]["carrier_key"] == layout.key.to_dict()


def test_o3_generalized_sector_labels_use_polar_source_parity():
    from ye3t.representations import ExactSymbolicProjectorGeneralizedBasisBuilder

    subgroup = PermutationSubgroup.from_nl((1, 1, 1), (1, 1, 1))
    irrep = PermutationIrrep.trivial_for_subgroup(subgroup)
    odd_sector = ExactSymbolicProjectorGeneralizedBasisBuilder(
        (1, 1, 1),
        (1, 1, 1),
        irrep,
        spatial_symmetry="O3",
    ).build()

    assert {
        label.parity
        for labels in odd_sector.labels_by_L.values()
        for label in labels
    } == {-1}


def test_o3_runtime_irrep_action_handles_inversion_and_reflection():
    from ye3t.representations import ExactSymbolicProjectorGeneralizedBasisBuilder

    subgroup = PermutationSubgroup.from_nl((1,), (1,))
    irrep = PermutationIrrep.trivial_for_subgroup(subgroup)
    sector = ExactSymbolicProjectorGeneralizedBasisBuilder(
        (1,),
        (1,),
        irrep,
        spatial_symmetry="O3",
    ).build()
    runtime_irreps = GeneralizedExactRuntimeIrreps(
        (GeneralizedExactRuntimeBlock(1, sector, 1, 0),)
    )

    inversion = runtime_irreps.D_from_matrix_real(
        -torch.eye(3, dtype=torch.float64),
        dtype=torch.float64,
    )
    reflection = runtime_irreps.D_from_matrix_real(
        torch.diag(torch.tensor((-1.0, 1.0, 1.0), dtype=torch.float64)),
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        inversion,
        -torch.eye(3, dtype=torch.float64),
        rtol=0.0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        reflection @ reflection,
        torch.eye(3, dtype=torch.float64),
        rtol=0.0,
        atol=1e-12,
    )


def test_real_tesseral_actions_project_fp32_roundoff_through_L16():
    rotation = torch.tensor(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=torch.float32,
    )
    for L in (1, 2, 3, 4, 6, 8, 12, 16):
        sector = _rank_one_trivial_sector(L)
        irreps = GeneralizedExactRuntimeIrreps(
            (GeneralizedExactRuntimeBlock(1, sector, L, 0),)
        )
        action = irreps.D_from_matrix_real(
            rotation,
            dtype=torch.float32,
        )

        assert action.dtype == torch.float32
        identity = torch.eye(int(action.shape[0]), dtype=torch.float32)
        torch.testing.assert_close(
            action @ action.mT,
            identity,
            rtol=2.0e-5,
            atol=2.0e-5,
        )


def test_real_tesseral_projection_rejects_material_imaginary_content():
    matrix = torch.eye(3, dtype=torch.complex64)
    matrix[0, 0] = matrix[0, 0] + 1.0e-3j

    with pytest.raises(RuntimeError, match="imaginary residual"):
        _require_numerically_real(matrix, "test_action")


def test_exact_parent_runtime_irreps_reject_inconsistent_layout_axes():
    from ye3t.execution_plan import (
        YE3T_PRIMARY_CONVENTION,
        YE3TCarrierKey,
        YE3TCarrierLayout,
    )

    key = YE3TCarrierKey(
        rank=3,
        partition=(2, 1),
        rotation_L=2,
        convention_id=YE3T_PRIMARY_CONVENTION,
    )
    with pytest.raises(ValueError, match="tableau count"):
        GeneralizedExactRuntimeIrreps.from_carrier_layout(
            YE3TCarrierLayout(
                key=key,
                channel_count=1,
                tableau_count=1,
                magnetic_count=5,
            )
        )
    with pytest.raises(ValueError, match="magnetic_count"):
        GeneralizedExactRuntimeIrreps.from_carrier_layout(
            YE3TCarrierLayout(
                key=key,
                channel_count=1,
                tableau_count=2,
                magnetic_count=3,
            )
        )


def test_direct_parent_subduction_compiles_rank6_lr_copies_with_explicit_source_assembly():
    rank_three = _rank_three_mixed_parent_irreps()
    product = JointYoungCGProduct(
        rank_three,
        rank_three,
        requested_targets=("L=0 x S_6:[3,2,1]",),
        rank_cap=6,
        L_max=0,
    )

    assert product.product_policy_report()["parent_subduction_backend"] == (
        "validated_cached_parent_subduction_C_dagger_L_v"
    )
    assert tuple(block.label.to_string() for block in product.irreps_out.blocks) == (
        "L=0 x S_6:[3,2,1]",
        "L=0 x S_6:[3,2,1]#1",
    )
    assert product.irreps_out.dim == 32
    assert len(product._instructions) == 2
    assert product.coefficient_report()["coefficient_sources"]["young_coupling"] == (
        "YoungSubgroupSpechtCoupling.C_dagger_L_v"
    )

    coupling = build_cached_young_subgroup_specht_coupling(
        ((2, 1), (2, 1)),
        (3, 2, 1),
    )
    coefficient_matrix = coupling.coefficient_matrix()
    for rho, instruction in enumerate(product._instructions):
        expected = torch.tensor(
            [
                [
                    complex(coefficient_matrix[child_index, rho * 16 + target_index])
                    for child_index in range(4)
                ]
                for target_index in range(16)
            ],
            dtype=torch.complex128,
        )
        assert torch.allclose(instruction.coordinate_matrix, expected)
        assert instruction.source_assembly["analysis_orientation"] == "C_dagger_L_v"
        assert instruction.source_assembly["active_induced_rows"] == (0, 1, 2, 3)
        assert instruction.source_assembly["induced_dimension"] == 80
        assert instruction.source_assembly["validation"]["passed"] is True

    schedule = product.static_schedule()
    restored = JointYoungCGProduct.from_static_schedule(schedule)
    assert restored.product_policy_report()["parent_subduction_backend"] == (
        "validated_cached_parent_subduction_C_dagger_L_v"
    )
    left = torch.randn((3, rank_three.dim), dtype=torch.float64)
    right = torch.randn((3, rank_three.dim), dtype=torch.float64)
    assert torch.allclose(restored(left, right), product(left, right))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_direct_parent_rank6_packed_cuda_matches_cpu_values_vjp_and_hvp():
    rank_three = _rank_three_mixed_parent_irreps()
    product = JointYoungCGProduct(
        rank_three,
        rank_three,
        requested_targets=("L=0 x S_6:[3,2,1]",),
        rank_cap=6,
        L_max=0,
    )
    cpu_product = JointYoungCGProduct.from_static_schedule(product.static_schedule())
    cpu_product.enable_real_basis_product(True).enable_streaming_vjp(True)
    gpu_product = JointYoungCGProduct.from_static_schedule(product.static_schedule())
    gpu_product.enable_real_basis_product(True).enable_streaming_vjp(True)

    left_values = torch.randn((4, rank_three.dim), dtype=torch.float64)
    right_values = torch.randn((4, rank_three.dim), dtype=torch.float64)
    cpu_left = left_values.clone().requires_grad_(True)
    cpu_right = right_values.clone().requires_grad_(True)
    gpu_left = left_values.cuda().requires_grad_(True)
    gpu_right = right_values.cuda().requires_grad_(True)

    cpu_output = cpu_product(cpu_left, cpu_right)
    gpu_output = gpu_product(gpu_left, gpu_right)
    assert torch.allclose(gpu_output.cpu(), cpu_output, atol=1.0e-10, rtol=1.0e-10)

    cpu_grads = torch.autograd.grad(cpu_output.square().sum(), (cpu_left, cpu_right), create_graph=True)
    gpu_grads = torch.autograd.grad(gpu_output.square().sum(), (gpu_left, gpu_right), create_graph=True)
    assert torch.allclose(gpu_grads[0].cpu(), cpu_grads[0], atol=1.0e-10, rtol=1.0e-10)
    assert torch.allclose(gpu_grads[1].cpu(), cpu_grads[1], atol=1.0e-10, rtol=1.0e-10)

    cpu_hvp = torch.autograd.grad(
        cpu_grads[0].square().sum() + cpu_grads[1].square().sum(),
        (cpu_left, cpu_right),
    )
    gpu_hvp = torch.autograd.grad(
        gpu_grads[0].square().sum() + gpu_grads[1].square().sum(),
        (gpu_left, gpu_right),
    )
    assert torch.allclose(gpu_hvp[0].cpu(), cpu_hvp[0], atol=1.0e-9, rtol=1.0e-9)
    assert torch.allclose(gpu_hvp[1].cpu(), cpu_hvp[1], atol=1.0e-9, rtol=1.0e-9)
    report = gpu_product.fallback_report()
    assert report["runtime"] == "triton_real_basis_exact_schedule"
    assert report["young_coupling_source"] == "YoungSubgroupSpechtCoupling.C_dagger_L_v"
    assert report["source_assembly"] == "canonical_binary_tree_identity_coset"


def test_joint_young_cg_product_trivial_only_excludes_nontrivial_permutation_targets():
    sector = _rank_two_trivial_sector()
    irreps = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(1, sector, 0, 0),))

    mixed = JointYoungCGProduct(irreps, irreps, tree_type="balanced", L_max=0)
    trivial = JointYoungCGProduct(irreps, irreps, tree_type="balanced", L_max=0, permutation_policy="trivial_only")

    assert any(not block.label.permutation.is_totally_symmetric() for block in mixed.irreps_out.blocks)
    assert all(block.label.permutation.is_totally_symmetric() for block in trivial.irreps_out.blocks)
    assert len(trivial.irreps_out.blocks) < len(mixed.irreps_out.blocks)


def test_joint_young_cg_product_static_schedule_roundtrip():
    from ye3t.api import JointYoungCGProduct as CurrentJointYoungCGProduct
    from ye3t.api import JointYoungCGTensorSchedule as CurrentJointYoungCGTensorSchedule
    from ye3t.representations import ExactSymbolicProjectorGeneralizedBasisBuilder as CurrentBuilder
    from ye3t.representations import PermutationIrrep as CurrentPermutationIrrep
    from ye3t.representations import PermutationSubgroup as CurrentPermutationSubgroup
    from ye3t.runtime.generalized import GeneralizedExactRuntimeBlock as CurrentRuntimeBlock
    from ye3t.runtime.generalized import GeneralizedExactRuntimeIrreps as CurrentRuntimeIrreps

    subgroup = CurrentPermutationSubgroup.from_nl((1,), (1,))
    irrep = CurrentPermutationIrrep.trivial_for_subgroup(subgroup)
    sector = CurrentBuilder((1,), (1,), irrep).build()
    irreps_left = CurrentRuntimeIrreps((CurrentRuntimeBlock(1, sector, 1, 0),))
    irreps_right = CurrentRuntimeIrreps((CurrentRuntimeBlock(1, sector, 1, 0),))
    product = CurrentJointYoungCGProduct(irreps_left, irreps_right, tree_type="balanced", L_max=1)

    left = torch.arange(6, dtype=torch.float64).reshape(2, 3) / 7.0
    right = torch.arange(6, dtype=torch.float64).reshape(2, 3) / 5.0
    reference = product(left, right)

    schedule = product.static_schedule()
    assert isinstance(schedule, CurrentJointYoungCGTensorSchedule)
    assert schedule.provenance["format"] == "joint_young_cg_tensor_schedule_v1"
    assert schedule.provenance["schedule_hash"]

    loaded = CurrentJointYoungCGProduct.from_static_schedule(schedule)
    assert loaded.composer is None
    assert torch.allclose(loaded(left, right), reference)

    scratch = PACKAGE_ROOT / ".tmp" / "test-joint-young-cg"
    scratch.mkdir(parents=True, exist_ok=True)
    path = scratch / f"joint_young_cg_schedule_{os.getpid()}.pt"
    try:
        product.save_static_schedule(path)
        loaded_from_file = CurrentJointYoungCGProduct.load_static_schedule(path)
        assert torch.allclose(loaded_from_file(left, right), reference)
    finally:
        path.unlink(missing_ok=True)


def test_joint_young_cg_packed_product_plan_roundtrip_preserves_vjp_hvp_and_o3():
    subgroup = PermutationSubgroup.from_nl((1,), (1,))
    irrep = PermutationIrrep.trivial_for_subgroup(subgroup)
    sector = ExactSymbolicProjectorGeneralizedBasisBuilder(
        (1,),
        (1,),
        irrep,
        spatial_symmetry="O3",
    ).build()
    irreps = GeneralizedExactRuntimeIrreps(
        (GeneralizedExactRuntimeBlock(2, sector, 1, 0),)
    )
    base = JointYoungCGProduct(
        irreps,
        irreps,
        requested_targets=("L=0,p=+1 x S_2:[2]",),
        rank_cap=2,
        L_max=0,
    )
    reference = JointYoungCGProduct.from_static_schedule(
        base.static_schedule()
    ).enable_real_basis_product(True).enable_streaming_vjp(True)
    plan = reference.packed_real_product_plan()
    restored = JointYoungCGProduct.from_packed_real_product_plan(
        plan,
        irreps,
        irreps,
    )

    assert restored.packed_real_product_plan() == plan
    assert tuple(
        block.label.to_string() for block in restored.irreps_out.blocks
    ) == tuple(
        block.label.to_string() for block in reference.irreps_out.blocks
    )
    assert {block.label.parity for block in restored.irreps_out.blocks} == {1}

    left_values = torch.randn((4, irreps.dim), dtype=torch.float64)
    right_values = torch.randn((4, irreps.dim), dtype=torch.float64)
    reference_left = left_values.clone().requires_grad_(True)
    reference_right = right_values.clone().requires_grad_(True)
    restored_left = left_values.clone().requires_grad_(True)
    restored_right = right_values.clone().requires_grad_(True)
    reference_output = reference(reference_left, reference_right)
    restored_output = restored(restored_left, restored_right)
    torch.testing.assert_close(
        restored_output, reference_output, atol=1.0e-12, rtol=0.0
    )
    reference_gradients = torch.autograd.grad(
        reference_output.square().sum(),
        (reference_left, reference_right),
        create_graph=True,
    )
    restored_gradients = torch.autograd.grad(
        restored_output.square().sum(),
        (restored_left, restored_right),
        create_graph=True,
    )
    for actual, expected in zip(restored_gradients, reference_gradients):
        torch.testing.assert_close(actual, expected, atol=1.0e-12, rtol=0.0)
    directions = (
        torch.randn_like(restored_left),
        torch.randn_like(restored_right),
    )
    reference_hvp = torch.autograd.grad(
        sum(
            torch.sum(gradient * direction)
            for gradient, direction in zip(reference_gradients, directions)
        ),
        (reference_left, reference_right),
    )
    restored_hvp = torch.autograd.grad(
        sum(
            torch.sum(gradient * direction)
            for gradient, direction in zip(restored_gradients, directions)
        ),
        (restored_left, restored_right),
    )
    for actual, expected in zip(restored_hvp, reference_hvp):
        torch.testing.assert_close(actual, expected, atol=1.0e-12, rtol=0.0)

    corrupted = copy.deepcopy(plan)
    corrupted["table"]["coefficient"][0] += 0.25
    with pytest.raises(ValueError, match="plan hash mismatch"):
        JointYoungCGProduct.from_packed_real_product_plan(
            corrupted,
            irreps,
            irreps,
        )


def test_packed_product_plan_reconstructs_multiple_parent_copies():
    rank_three = _rank_three_mixed_parent_irreps()
    base = JointYoungCGProduct(
        rank_three,
        rank_three,
        requested_targets=("L=0 x S_6:[3,2,1]",),
        rank_cap=6,
        L_max=0,
    )
    reference = JointYoungCGProduct.from_static_schedule(
        base.static_schedule()
    ).enable_real_basis_product(True)
    restored = JointYoungCGProduct.from_packed_real_product_plan(
        reference.packed_real_product_plan(),
        rank_three,
        rank_three,
    )
    assert tuple(
        block.label.to_string() for block in restored.irreps_out.blocks
    ) == (
        "L=0 x S_6:[3,2,1]",
        "L=0 x S_6:[3,2,1]#1",
    )
    left = torch.randn((3, rank_three.dim), dtype=torch.float64)
    right = torch.randn((3, rank_three.dim), dtype=torch.float64)
    torch.testing.assert_close(
        restored(left, right),
        reference(left, right),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_joint_young_cg_product_streaming_vjp_matches_eager_gradients():
    sector = _rank_one_trivial_sector()
    irreps_left = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(2, sector, 1, 0),))
    irreps_right = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(1, sector, 1, 0),))
    eager = JointYoungCGProduct(irreps_left, irreps_right, tree_type="balanced", L_max=1)
    streaming = JointYoungCGProduct.from_static_schedule(eager.static_schedule()).enable_streaming_vjp(True)

    left_values = torch.arange(12, dtype=torch.float64).reshape(2, 6) / 7.0
    right_values = torch.arange(6, dtype=torch.float64).reshape(2, 3) / 5.0
    eager_left = left_values.clone().requires_grad_(True)
    eager_right = right_values.clone().requires_grad_(True)
    streaming_left = left_values.clone().requires_grad_(True)
    streaming_right = right_values.clone().requires_grad_(True)

    eager_out = eager(eager_left, eager_right)
    streaming_out = streaming(streaming_left, streaming_right)
    assert torch.allclose(streaming_out, eager_out)

    eager_loss = (eager_out.real.square() + eager_out.imag.square()).sum()
    streaming_loss = (streaming_out.real.square() + streaming_out.imag.square()).sum()
    eager_grads = torch.autograd.grad(eager_loss, (eager_left, eager_right), create_graph=True)
    streaming_grads = torch.autograd.grad(streaming_loss, (streaming_left, streaming_right), create_graph=True)

    assert torch.allclose(streaming_grads[0], eager_grads[0])
    assert torch.allclose(streaming_grads[1], eager_grads[1])

    eager_second_loss = eager_grads[0].square().sum() + eager_grads[1].square().sum()
    streaming_second_loss = streaming_grads[0].square().sum() + streaming_grads[1].square().sum()
    eager_second_grads = torch.autograd.grad(eager_second_loss, (eager_left, eager_right))
    streaming_second_grads = torch.autograd.grad(streaming_second_loss, (streaming_left, streaming_right))

    assert torch.allclose(streaming_second_grads[0], eager_second_grads[0])
    assert torch.allclose(streaming_second_grads[1], eager_second_grads[1])


def _phase_adjusted_real_product_output(product, output):
    output_real_basis = product.irreps_out.generalized_irreps().to_real_basis(output)
    adjusted = torch.zeros(output_real_basis.shape, dtype=output_real_basis.real.dtype, device=output_real_basis.device)
    out_slices = product.irreps_out.slices()
    for instruction in product._instructions:
        left_block = product.irreps_in1.blocks[instruction.left_block_index]
        right_block = product.irreps_in2.blocks[instruction.right_block_index]
        out_slice = out_slices[instruction.output_block_index]
        if (int(left_block.L) + int(right_block.L) - int(instruction.output_L)) % 2:
            adjusted[..., out_slice] = output_real_basis[..., out_slice].imag
        else:
            adjusted[..., out_slice] = output_real_basis[..., out_slice].real
    return adjusted


def test_generalized_real_basis_roundtrip_preserves_fp32_precision():
    sector = _rank_one_trivial_sector()
    irreps = GeneralizedExactRuntimeIrreps(
        (GeneralizedExactRuntimeBlock(2, sector, 1, 0),)
    ).generalized_irreps()
    values = torch.arange(12, dtype=torch.float32).reshape(2, 6) / 7.0

    canonical = irreps.from_real_basis(values)
    roundtrip = irreps.to_real_basis(canonical)

    assert canonical.dtype == torch.complex64
    assert roundtrip.dtype == torch.complex64
    torch.testing.assert_close(
        roundtrip.real,
        values,
        atol=2.0e-6,
        rtol=2.0e-6,
    )
    torch.testing.assert_close(
        roundtrip.imag,
        torch.zeros_like(values),
        atol=2.0e-6,
        rtol=0.0,
    )


def test_generalized_real_basis_matches_declared_tesseral_conversion():
    from ye3t.core.tesseral import real_tesseral_to_complex_multiplet

    for L in (1, 2, 3):
        sector = _rank_one_trivial_sector(L)
        irreps = GeneralizedExactRuntimeIrreps(
            (GeneralizedExactRuntimeBlock(1, sector, L, 0),)
        ).generalized_irreps()
        values = torch.arange(2 * L + 1, dtype=torch.float64).reshape(1, -1) / 7.0
        actual = irreps.from_real_basis(values)
        expected = real_tesseral_to_complex_multiplet(values, L)
        torch.testing.assert_close(actual, expected, atol=1.0e-12, rtol=1.0e-12)


def test_joint_young_cg_product_real_basis_forward_vjp_and_hjp_match_complex_reference():
    sector = _rank_one_trivial_sector()
    irreps_left = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(2, sector, 1, 0),))
    irreps_right = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(1, sector, 1, 0),))
    eager = JointYoungCGProduct(irreps_left, irreps_right, tree_type="balanced", L_max=1)
    real_product = JointYoungCGProduct.from_static_schedule(eager.static_schedule())
    real_product.enable_real_basis_product(True).enable_streaming_vjp(True)

    left_values = torch.arange(12, dtype=torch.float64).reshape(2, 6) / 7.0
    right_values = torch.arange(6, dtype=torch.float64).reshape(2, 3) / 5.0
    eager_left_real = left_values.clone().requires_grad_(True)
    eager_right_real = right_values.clone().requires_grad_(True)
    eager_left = irreps_left.generalized_irreps().from_real_basis(eager_left_real)
    eager_right = irreps_right.generalized_irreps().from_real_basis(eager_right_real)
    real_left = left_values.clone().requires_grad_(True)
    real_right = right_values.clone().requires_grad_(True)

    eager_out_real = _phase_adjusted_real_product_output(eager, eager(eager_left, eager_right))
    real_out = real_product(real_left, real_right)
    assert real_out.dtype == torch.float64
    assert torch.allclose(real_out, eager_out_real)
    assert real_product.fallback_report()["runtime"] == "torch_real_basis_exact_schedule"

    eager_loss = eager_out_real.square().sum()
    real_loss = real_out.square().sum()
    eager_grads = torch.autograd.grad(eager_loss, (eager_left_real, eager_right_real), create_graph=True)
    real_grads = torch.autograd.grad(real_loss, (real_left, real_right), create_graph=True)

    assert torch.allclose(real_grads[0], eager_grads[0])
    assert torch.allclose(real_grads[1], eager_grads[1])

    eager_second_loss = eager_grads[0].square().sum() + eager_grads[1].square().sum()
    real_second_loss = real_grads[0].square().sum() + real_grads[1].square().sum()
    eager_second_grads = torch.autograd.grad(eager_second_loss, (eager_left_real, eager_right_real))
    real_second_grads = torch.autograd.grad(real_second_loss, (real_left, real_right))

    assert torch.allclose(real_second_grads[0], eager_second_grads[0])
    assert torch.allclose(real_second_grads[1], eager_second_grads[1])


def test_packed_real_bilinear_table_matches_instruction_einsum_in_fp32():
    from ye3t.backends.triton_joint import sparse_bilinear_reference
    from ye3t.runtime.generalized import _packed_real_bilinear_table

    sector = _rank_one_trivial_sector()
    irreps_left = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(2, sector, 1, 0),))
    irreps_right = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(1, sector, 1, 0),))
    base = JointYoungCGProduct(irreps_left, irreps_right, tree_type="balanced", L_max=1)
    reference = JointYoungCGProduct.from_static_schedule(base.static_schedule())
    reference.enable_real_basis_product(True).enable_packed_cuda(False)

    left = torch.arange(24, dtype=torch.float32).reshape(4, 6) / 11.0
    right = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 7.0
    expected = reference(left, right)
    table = _packed_real_bilinear_table(reference)
    actual = sparse_bilinear_reference(left, right, table)

    torch.testing.assert_close(actual, expected, atol=5.0e-6, rtol=5.0e-6)
    report = reference._packed_real_bilinear_report
    assert report["term_count"] > 0
    assert report["instruction_count"] == len(reference._instructions)
    assert report["dropped_max_abs"] <= 64.0 * torch.finfo(torch.float64).eps
    assert report["logical_output_axes"] == (
        "channel_or_multiplicity",
        "tableau_t",
        "magnetic_M",
    )
    assert report["basis"] == "declared_real_unitary_transform_of_complex_primary_convention"


def test_packed_sparse_bilinear_group_matches_multiple_exact_product_paths():
    from ye3t.backends.triton_joint import PackedSparseBilinearGroup

    sector = _rank_one_trivial_sector()
    wide = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(2, sector, 1, 0),))
    narrow = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(1, sector, 1, 0),))
    wide_narrow_base = JointYoungCGProduct(wide, narrow, tree_type="balanced", L_max=1)
    narrow_wide_base = JointYoungCGProduct(narrow, wide, tree_type="balanced", L_max=1)
    wide_narrow = JointYoungCGProduct.from_static_schedule(wide_narrow_base.static_schedule())
    narrow_wide = JointYoungCGProduct.from_static_schedule(narrow_wide_base.static_schedule())
    wide_narrow.enable_real_basis_product(True).enable_packed_cuda(False)
    narrow_wide.enable_real_basis_product(True).enable_packed_cuda(False)
    group = PackedSparseBilinearGroup(
        (
            wide_narrow.packed_real_bilinear_table(),
            narrow_wide.packed_real_bilinear_table(),
        ),
        ((0, 1), (1, 0)),
        (int(wide.dim), int(narrow.dim)),
    )

    wide_values = (torch.arange(24, dtype=torch.float64).reshape(4, 6) - 4.0) / 9.0
    narrow_values = (torch.arange(12, dtype=torch.float64).reshape(4, 3) + 2.0) / 7.0
    expected = (
        wide_narrow(wide_values, narrow_values),
        narrow_wide(narrow_values, wide_values),
    )
    actual = group((wide_values, narrow_values))

    torch.testing.assert_close(actual[0], expected[0], atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(actual[1], expected[1], atol=1.0e-12, rtol=1.0e-12)
    report = group.report()
    assert report["path_count"] == 2
    assert report["source_count"] == 2
    assert report["single_numeric_launch"] is True
    assert report["source_storage"] == "declared_real_basis"
    assert report["index_validation"] == "once_at_packed_group_construction"
    assert report["logical_sector_axes"] == (
        "channel_or_multiplicity",
        "tableau_t",
        "magnetic_M",
    )
    assert group.state_dict() == {}


def test_packed_sparse_bilinear_group_rejects_invalid_table_before_runtime():
    from ye3t.backends.triton_joint import (
        PackedSparseBilinearGroup,
        SparseBilinearTable,
    )

    invalid = SparseBilinearTable(
        left_index=torch.tensor((-1,), dtype=torch.long),
        right_index=torch.tensor((0,), dtype=torch.long),
        output_index=torch.tensor((0,), dtype=torch.long),
        coefficient=torch.tensor((1.0,), dtype=torch.float64),
        output_width=1,
    )
    with pytest.raises(ValueError, match="left_index is out of range"):
        PackedSparseBilinearGroup(
            (invalid,),
            ((0, 0),),
            (1,),
        )


def test_weighted_packed_group_fuses_multiplicity_map_and_preserves_tableau_magnetic_axes():
    from ye3t.backends.triton_joint import (
        PackedWeightedSparseBilinearGroup,
        weighted_sparse_bilinear_reference,
    )

    sector = _rank_one_trivial_sector()
    wide = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(2, sector, 1, 0),))
    narrow = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(1, sector, 1, 0),))
    base = JointYoungCGProduct(wide, narrow, tree_type="balanced", L_max=1)
    product = JointYoungCGProduct.from_static_schedule(base.static_schedule())
    product.enable_real_basis_product(True).enable_packed_cuda(False)
    group = PackedWeightedSparseBilinearGroup(
        (product.packed_real_bilinear_table(),),
        ((0, 1),),
        (int(wide.dim), int(narrow.dim)),
        (product.packed_real_output_blocks(),),
    ).double()

    wide_values = (torch.arange(24, dtype=torch.float64).reshape(4, 6) - 2.0) / 13.0
    narrow_values = (torch.arange(12, dtype=torch.float64).reshape(4, 3) + 1.0) / 9.0
    raw = product(wide_values, narrow_values)
    identity_output = group((wide_values, narrow_values))[0]
    torch.testing.assert_close(identity_output, raw, atol=1.0e-12, rtol=1.0e-12)

    with torch.no_grad():
        for block in group.weight_blocks:
            rows, columns = block["shape"]
            matrix = group.mixing_weight[int(block["start"]):int(block["stop"])].reshape(rows, columns)
            matrix.add_(torch.arange(rows * columns, dtype=matrix.dtype).reshape(rows, columns) / 17.0)
    packed = group.pack_sources((wide_values, narrow_values))
    expected = weighted_sparse_bilinear_reference(
        packed,
        packed,
        group.mixing_weight,
        group._table(),
    )
    actual = torch.cat(group((wide_values, narrow_values)), dim=1)
    torch.testing.assert_close(actual, expected, atol=1.0e-12, rtol=1.0e-12)
    report = group.report()
    assert tuple(group.state_dict()) == ("mixing_weight",)
    assert report["learned_map_axis"] == "channel_or_multiplicity"
    assert report["preserved_axes"] == ("tableau_t", "magnetic_M")
    assert report["fused_operations"] == (
        "joint_young_angular_analysis",
        "multiplicity_mixing",
        "output_packing",
    )
    compressed = PackedWeightedSparseBilinearGroup(
        (product.packed_real_bilinear_table(),),
        ((0, 1),),
        (int(wide.dim), int(narrow.dim)),
        (product.packed_real_output_blocks(),),
        output_channel_counts_by_path=((1, 1),),
    ).double()
    raw_blocks = product.packed_real_output_blocks()
    expected_compressed = torch.cat(
        tuple(
            raw[:, int(block["start"]):int(block["stop"])].reshape(
                int(raw.shape[0]),
                int(block["channel_count"]),
                int(block["component_width"]),
            )[:, :1].reshape(int(raw.shape[0]), -1)
            for block in raw_blocks
        ),
        dim=1,
    )
    actual_compressed = compressed((wide_values, narrow_values))[0]
    torch.testing.assert_close(
        actual_compressed,
        expected_compressed,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    assert all(
        block["shape"] == (1, 2)
        for block in compressed.weight_blocks
    )
    assert compressed.report()["output_width"] == sum(
        int(block["component_width"])
        for block in raw_blocks
    )
    shared_base = torch.randn(4, 9, dtype=torch.float64, requires_grad=True)
    shared_sources = (shared_base[:, :6], shared_base[:, 6:])
    shared_packed = group.pack_sources(shared_sources)
    shared_output = group.evaluate_packed(shared_packed)
    shared_split = group.split_output(shared_output)
    shared_grad = torch.autograd.grad(shared_output.square().sum(), shared_base)[0]

    assert shared_packed.untyped_storage().data_ptr() == shared_base.untyped_storage().data_ptr()
    assert shared_packed.storage_offset() == shared_base.storage_offset()
    assert shared_packed.is_contiguous()
    torch.testing.assert_close(shared_packed, shared_base)
    torch.testing.assert_close(torch.cat(shared_split, dim=1), shared_output)
    assert torch.isfinite(shared_grad).all()
    assert group.report()["source_pack_operation"] == "zero_copy_shared_packed_carrier_view"
    assert group.report()["flat_packed_output_available"] is True
    assert torch.autograd.gradgradcheck(
        lambda left, right, weight: weighted_sparse_bilinear_reference(
            left,
            right,
            weight,
            group._table(),
        ),
        (
            packed.detach().requires_grad_(True),
            (packed.detach() * 0.7).requires_grad_(True),
            group.mixing_weight.detach().requires_grad_(True),
        ),
        eps=1.0e-6,
        atol=1.0e-8,
        rtol=1.0e-6,
    )


def test_weighted_packed_group_expands_only_complete_multiplicity_axis():
    from ye3t.backends.triton_joint import (
        PackedWeightedSparseBilinearGroup,
        weighted_sparse_bilinear_reference,
    )

    sector = _rank_one_trivial_sector()
    left_irreps = GeneralizedExactRuntimeIrreps(
        (GeneralizedExactRuntimeBlock(2, sector, 1, 0),)
    )
    right_irreps = GeneralizedExactRuntimeIrreps(
        (GeneralizedExactRuntimeBlock(1, sector, 1, 0),)
    )
    base = JointYoungCGProduct(
        left_irreps,
        right_irreps,
        tree_type="balanced",
        L_max=1,
    )
    product = JointYoungCGProduct.from_static_schedule(
        base.static_schedule()
    )
    product.enable_real_basis_product(True).enable_packed_cuda(False)
    blocks = product.packed_real_output_blocks()
    expanded_counts = tuple(
        int(block["channel_count"]) + 1 for block in blocks
    )
    arguments = (
        (product.packed_real_bilinear_table(),),
        ((0, 1),),
        (int(left_irreps.dim), int(right_irreps.dim)),
        (blocks,),
    )
    with pytest.raises(ValueError, match="explicit channel expansion"):
        PackedWeightedSparseBilinearGroup(
            *arguments,
            output_channel_counts_by_path=(expanded_counts,),
        )

    group = PackedWeightedSparseBilinearGroup(
        *arguments,
        output_channel_counts_by_path=(expanded_counts,),
        allow_output_channel_expansion=True,
    ).double()
    with torch.no_grad():
        for block in group.weight_blocks:
            start = int(block["start"])
            stop = int(block["stop"])
            rows, columns = tuple(int(value) for value in block["shape"])
            values = torch.arange(
                1,
                rows * columns + 1,
                dtype=torch.float64,
            ).reshape(rows, columns)
            group.mixing_weight[start:stop].copy_(
                (values / float(rows * columns + 1)).reshape(-1)
            )
    left = torch.randn(3, int(left_irreps.dim), dtype=torch.float64)
    right = torch.randn(3, int(right_irreps.dim), dtype=torch.float64)
    packed = group.pack_sources((left, right)).requires_grad_(True)
    expected = weighted_sparse_bilinear_reference(
        packed,
        packed,
        group.mixing_weight,
        group._table(),
    )
    actual = group.evaluate_packed(packed)
    torch.testing.assert_close(actual, expected, atol=1.0e-12, rtol=1.0e-12)
    gradient = torch.autograd.grad(actual.square().sum(), packed, create_graph=True)[0]
    hvp = torch.autograd.grad((gradient * torch.randn_like(gradient)).sum(), packed)[0]
    assert torch.isfinite(gradient).all()
    assert torch.isfinite(hvp).all()
    report = group.report()
    assert report["allow_output_channel_expansion"] is True
    assert report["expanded_weight_block_count"] == len(blocks)
    assert all(
        block["output_channel_expansion"]
        for block in report["weight_blocks"]
    )


def test_weighted_packed_group_indexes_persistent_arena_without_gather():
    from ye3t.backends.triton_joint import PackedWeightedSparseBilinearGroup

    sector = _rank_one_trivial_sector()
    wide = GeneralizedExactRuntimeIrreps(
        (GeneralizedExactRuntimeBlock(2, sector, 1, 0),)
    )
    narrow = GeneralizedExactRuntimeIrreps(
        (GeneralizedExactRuntimeBlock(1, sector, 1, 0),)
    )
    base = JointYoungCGProduct(
        wide,
        narrow,
        tree_type="balanced",
        L_max=1,
    )
    product = JointYoungCGProduct.from_static_schedule(
        base.static_schedule()
    )
    product.enable_real_basis_product(True).enable_packed_cuda(False)
    wide_offset = 2
    narrow_offset = 10
    arena_width = 15
    group = PackedWeightedSparseBilinearGroup(
        (product.packed_real_bilinear_table(),),
        ((0, 1),),
        (int(wide.dim), int(narrow.dim)),
        (product.packed_real_output_blocks(),),
        arena_source_offsets=(wide_offset, narrow_offset),
        arena_source_width=arena_width,
        reduction_mode="stable",
    ).double()
    generator = torch.Generator().manual_seed(6421)
    arena_reference = torch.randn(
        (5, arena_width),
        dtype=torch.float64,
        generator=generator,
        requires_grad=True,
    )
    arena_direct = arena_reference.detach().clone().requires_grad_(True)
    direction = torch.randn(
        (5, arena_width),
        dtype=torch.float64,
        generator=generator,
    )

    compact = torch.cat(
        (
            arena_reference[
                :, wide_offset:wide_offset + int(wide.dim)
            ],
            arena_reference[
                :, narrow_offset:narrow_offset + int(narrow.dim)
            ],
        ),
        dim=1,
    )
    compact_output = group.evaluate_packed(
        compact,
        source_layout="compact",
    )
    arena_output = group.evaluate_packed(
        arena_direct,
        source_layout="arena",
    )
    torch.testing.assert_close(
        arena_output,
        compact_output,
        atol=1.0e-12,
        rtol=1.0e-12,
    )

    compact_gradient = torch.autograd.grad(
        compact_output.square().sum(),
        arena_reference,
        create_graph=True,
    )[0]
    arena_gradient = torch.autograd.grad(
        arena_output.square().sum(),
        arena_direct,
        create_graph=True,
    )[0]
    compact_hvp = torch.autograd.grad(
        (compact_gradient * direction).sum(),
        arena_reference,
    )[0]
    arena_hvp = torch.autograd.grad(
        (arena_gradient * direction).sum(),
        arena_direct,
    )[0]
    torch.testing.assert_close(
        arena_gradient,
        compact_gradient,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    torch.testing.assert_close(
        arena_hvp,
        compact_hvp,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    report = group.report()
    assert report["direct_arena_indexing_available"] is True
    assert report["last_source_layout"] == "arena"
    assert report["arena_source_offsets"] == (
        wide_offset,
        narrow_offset,
    )


def test_weighted_packed_group_reports_supplied_compact_aligned_arena():
    from ye3t.backends.triton_joint import PackedWeightedSparseBilinearGroup

    sector = _rank_one_trivial_sector()
    left = GeneralizedExactRuntimeIrreps(
        (GeneralizedExactRuntimeBlock(1, sector, 1, 0),)
    )
    right = GeneralizedExactRuntimeIrreps(
        (GeneralizedExactRuntimeBlock(1, sector, 1, 0),)
    )
    product = JointYoungCGProduct(left, right, L_max=1)
    product.enable_real_basis_product(True).enable_packed_cuda(False)
    left_width = int(left.dim)
    right_width = int(right.dim)
    group = PackedWeightedSparseBilinearGroup(
        (product.packed_real_bilinear_table(),),
        ((0, 1),),
        (left_width, right_width),
        (product.packed_real_output_blocks(),),
        arena_source_offsets=(0, left_width),
        arena_source_width=left_width + right_width,
        reduction_mode="stable",
    ).double()

    values = torch.randn(
        (3, left_width + right_width),
        dtype=torch.float64,
    )
    group.evaluate_packed(values, source_layout="arena")
    report = group.report()

    assert report["direct_arena_indexing_available"] is True
    assert report["arena_layout_distinct"] is False
    assert report["last_source_layout"] == "arena"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_weighted_packed_direct_arena_cuda_matches_compact_vjp_hvp():
    from ye3t.backends.triton_joint import (
        PackedWeightedSparseBilinearGroup,
        SparseBilinearTable,
    )

    table = SparseBilinearTable(
        left_index=torch.tensor((0, 1), dtype=torch.long),
        right_index=torch.tensor((0, 0), dtype=torch.long),
        output_index=torch.tensor((0, 0), dtype=torch.long),
        coefficient=torch.tensor((0.75, -0.25), dtype=torch.float64),
        output_width=1,
    )
    group = PackedWeightedSparseBilinearGroup(
        (table,),
        ((0, 1),),
        (2, 1),
        (
            (
                {
                    "start": 0,
                    "stop": 1,
                    "channel_count": 1,
                    "component_width": 1,
                },
            ),
        ),
        arena_source_offsets=(1, 4),
        arena_source_width=6,
        reduction_mode="segmented",
    ).float().cuda()
    generator = torch.Generator(device="cuda").manual_seed(6422)
    direct_input = torch.randn(
        (31, 6),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
        requires_grad=True,
    )
    compact_input_base = direct_input.detach().clone().requires_grad_(True)
    direction = torch.randn(
        direct_input.shape,
        dtype=direct_input.dtype,
        device=direct_input.device,
        generator=generator,
    )
    compact_input = torch.cat(
        (
            compact_input_base[:, 1:3],
            compact_input_base[:, 4:5],
        ),
        dim=1,
    )
    direct_output = group.evaluate_packed(
        direct_input,
        source_layout="arena",
    )
    direct_backend = str(group.last_backend)
    compact_output = group.evaluate_packed(
        compact_input,
        source_layout="compact",
    )
    direct_gradient = torch.autograd.grad(
        direct_output.square().sum(),
        direct_input,
        create_graph=True,
    )[0]
    compact_gradient = torch.autograd.grad(
        compact_output.square().sum(),
        compact_input_base,
        create_graph=True,
    )[0]
    direct_hvp = torch.autograd.grad(
        (direct_gradient * direction).sum(),
        direct_input,
    )[0]
    compact_hvp = torch.autograd.grad(
        (compact_gradient * direction).sum(),
        compact_input_base,
    )[0]
    torch.testing.assert_close(
        direct_output,
        compact_output,
        atol=2.0e-6,
        rtol=2.0e-6,
    )
    torch.testing.assert_close(
        direct_gradient,
        compact_gradient,
        atol=3.0e-6,
        rtol=3.0e-6,
    )
    torch.testing.assert_close(
        direct_hvp,
        compact_hvp,
        atol=3.0e-6,
        rtol=3.0e-6,
    )
    assert direct_backend.endswith(
        "arena_compact_adjoint_autograd"
    )
    assert group.report()["arena_derivative_strategy"] == (
        "compact_selected_adjoint_then_single_arena_scatter"
    )


def test_weighted_packed_stable_reduction_matches_reference_and_is_derived_state():
    from ye3t.backends.triton_joint import (
        PackedWeightedSparseBilinearGroup,
        weighted_sparse_bilinear_reference,
    )

    sector = _rank_one_trivial_sector()
    wide = GeneralizedExactRuntimeIrreps(
        (GeneralizedExactRuntimeBlock(2, sector, 1, 0),)
    )
    narrow = GeneralizedExactRuntimeIrreps(
        (GeneralizedExactRuntimeBlock(1, sector, 1, 0),)
    )
    base = JointYoungCGProduct(
        wide,
        narrow,
        tree_type="balanced",
        L_max=1,
    )
    product = JointYoungCGProduct.from_static_schedule(base.static_schedule())
    product.enable_real_basis_product(True).enable_packed_cuda(False)
    group = PackedWeightedSparseBilinearGroup(
        (product.packed_real_bilinear_table(),),
        ((0, 1),),
        (int(wide.dim), int(narrow.dim)),
        (product.packed_real_output_blocks(),),
        reduction_mode="stable",
    ).double()
    generator = torch.Generator().manual_seed(987)
    wide_values = torch.randn((7, int(wide.dim)), dtype=torch.float64, generator=generator)
    narrow_values = torch.randn((7, int(narrow.dim)), dtype=torch.float64, generator=generator)
    packed = group.pack_sources((wide_values, narrow_values))
    actual = group.evaluate_packed(packed)
    expected = weighted_sparse_bilinear_reference(
        packed,
        packed,
        group.mixing_weight,
        group._table(),
    )
    torch.testing.assert_close(actual, expected, atol=1.0e-12, rtol=1.0e-12)
    assert group.report()["last_backend"] == "torch_sorted_segment_weighted_sparse_bilinear"
    assert group.report()["single_numeric_launch"] is False
    assert group.report()["effective_reduction_mode"] == "stable"
    assert group.report()["stable_reduction"]["output"][
        "max_segment_length"
    ] > 0
    assert not any(
        "segment_" in name
        for name in group.state_dict()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_weighted_packed_stable_cuda_repeats_fp32_value_vjp_hvp_exactly():
    from ye3t.backends.triton_joint import PackedWeightedSparseBilinearGroup

    sector = _rank_one_trivial_sector()
    wide = GeneralizedExactRuntimeIrreps(
        (GeneralizedExactRuntimeBlock(3, sector, 1, 0),)
    )
    narrow = GeneralizedExactRuntimeIrreps(
        (GeneralizedExactRuntimeBlock(2, sector, 1, 0),)
    )
    base = JointYoungCGProduct(
        wide,
        narrow,
        tree_type="balanced",
        L_max=1,
    )
    product = JointYoungCGProduct.from_static_schedule(base.static_schedule())
    product.enable_real_basis_product(True).enable_packed_cuda(False)
    group = PackedWeightedSparseBilinearGroup(
        (product.packed_real_bilinear_table(),),
        ((0, 1),),
        (int(wide.dim), int(narrow.dim)),
        (product.packed_real_output_blocks(),),
        reduction_mode="stable",
    ).float().cuda()
    generator = torch.Generator(device="cuda").manual_seed(988)
    seed = torch.randn(
        (31, int(wide.dim + narrow.dim)),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    direction = torch.randn(
        seed.shape,
        dtype=seed.dtype,
        device=seed.device,
        generator=generator,
    )

    def evaluate():
        packed = seed.detach().clone().requires_grad_(True)
        output = group.evaluate_packed(packed)
        gradient = torch.autograd.grad(
            output.square().sum(),
            packed,
            create_graph=True,
        )[0]
        hvp = torch.autograd.grad(
            (gradient * direction).sum(),
            packed,
        )[0]
        return output.detach(), gradient.detach(), hvp.detach()

    rows = tuple(evaluate() for _ in range(5))
    for row in rows[1:]:
        for actual, expected in zip(row, rows[0]):
            torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    report = group.report()
    assert report["last_backend"] == (
        "triton_segmented_weighted_sparse_bilinear_autograd"
    )
    assert report["single_numeric_launch"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_weighted_packed_auto_cuda_fallback_is_never_silent(monkeypatch):
    import ye3t.backends.triton_joint as triton_joint
    from ye3t.backends.triton_joint import PackedWeightedSparseBilinearGroup

    sector = _rank_one_trivial_sector()
    left = GeneralizedExactRuntimeIrreps(
        (GeneralizedExactRuntimeBlock(1, sector, 1, 0),)
    )
    right = GeneralizedExactRuntimeIrreps(
        (GeneralizedExactRuntimeBlock(1, sector, 1, 0),)
    )
    base = JointYoungCGProduct(left, right, tree_type="balanced", L_max=1)
    product = JointYoungCGProduct.from_static_schedule(base.static_schedule())
    product.enable_real_basis_product(True).enable_packed_cuda(False)
    group = PackedWeightedSparseBilinearGroup(
        (product.packed_real_bilinear_table(),),
        ((0, 1),),
        (int(left.dim), int(right.dim)),
        (product.packed_real_output_blocks(),),
        reduction_mode="auto",
    ).float().cuda()
    packed = torch.randn(
        (4, int(sum(group.source_widths))),
        dtype=torch.float32,
        device="cuda",
    )
    monkeypatch.setattr(triton_joint, "_can_use_triton", lambda *_args: False)

    with pytest.warns(RuntimeWarning, match="fell back to the PyTorch reference"):
        output = group.evaluate_packed(packed)

    assert torch.all(torch.isfinite(output))
    assert group.report()["last_backend"] == (
        "torch_weighted_sparse_bilinear_reference"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
@pytest.mark.parametrize("streaming_vjp", (False, True))
def test_packed_cuda_joint_young_product_matches_values_vjp_and_double_backward(streaming_vjp):
    pytest.importorskip("triton")

    sector = _rank_one_trivial_sector()
    irreps_left = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(2, sector, 1, 0),))
    irreps_right = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(1, sector, 1, 0),))
    base = JointYoungCGProduct(irreps_left, irreps_right, tree_type="balanced", L_max=1)
    reference = JointYoungCGProduct.from_static_schedule(base.static_schedule())
    reference.enable_real_basis_product(True).enable_streaming_vjp(bool(streaming_vjp)).enable_packed_cuda(False)
    packed = JointYoungCGProduct.from_static_schedule(base.static_schedule())
    packed.enable_real_basis_product(True).enable_streaming_vjp(bool(streaming_vjp)).enable_packed_cuda(True, strict=True)

    generator = torch.Generator(device="cuda").manual_seed(541)
    left_values = torch.randn((64, 6), dtype=torch.float64, device="cuda", generator=generator)
    right_values = torch.randn((64, 3), dtype=torch.float64, device="cuda", generator=generator)
    reference_left = left_values.clone().requires_grad_(True)
    reference_right = right_values.clone().requires_grad_(True)
    packed_left = left_values.clone().requires_grad_(True)
    packed_right = right_values.clone().requires_grad_(True)

    reference_out = reference(reference_left, reference_right)
    packed_out = packed(packed_left, packed_right)
    torch.testing.assert_close(packed_out, reference_out, atol=1.0e-11, rtol=1.0e-11)

    reference_grads = torch.autograd.grad(
        reference_out.square().sum(),
        (reference_left, reference_right),
        create_graph=True,
    )
    packed_grads = torch.autograd.grad(
        packed_out.square().sum(),
        (packed_left, packed_right),
        create_graph=True,
    )
    torch.testing.assert_close(packed_grads[0], reference_grads[0], atol=1.0e-10, rtol=1.0e-10)
    torch.testing.assert_close(packed_grads[1], reference_grads[1], atol=1.0e-10, rtol=1.0e-10)

    reference_hessian = torch.autograd.grad(
        reference_grads[0].square().sum() + reference_grads[1].square().sum(),
        (reference_left, reference_right),
    )
    packed_hessian = torch.autograd.grad(
        packed_grads[0].square().sum() + packed_grads[1].square().sum(),
        (packed_left, packed_right),
    )
    torch.testing.assert_close(packed_hessian[0], reference_hessian[0], atol=1.0e-9, rtol=1.0e-9)
    torch.testing.assert_close(packed_hessian[1], reference_hessian[1], atol=1.0e-9, rtol=1.0e-9)

    report = packed.fallback_report()
    assert report["runtime"] == "triton_real_basis_exact_schedule"
    assert report["accelerator_backend"] == "triton_sparse_bilinear"
    assert report["packed_sparse_bilinear_table"]["term_count"] > 0
    assert report["last_real_product_backend"] == "packed_sparse_bilinear"
    if bool(streaming_vjp):
        assert report["last_real_product_backward_backend"] == "packed_sparse_bilinear"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
@pytest.mark.parametrize(
    "dtype,value_tolerance,gradient_tolerance,hessian_tolerance",
    (
        (torch.float64, 1.0e-11, 1.0e-10, 1.0e-9),
        (torch.float32, 2.0e-5, 5.0e-5, 5.0e-5),
    ),
)
def test_packed_cuda_group_matches_cross_path_values_and_double_backward(
    dtype,
    value_tolerance,
    gradient_tolerance,
    hessian_tolerance,
):
    from ye3t.backends.triton_joint import PackedSparseBilinearGroup
    from torch.utils._python_dispatch import TorchDispatchMode

    class ScalarReadCounter(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.count = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if str(func) == "aten._local_scalar_dense.default":
                self.count += 1
            return func(*args, **({} if kwargs is None else kwargs))

    pytest.importorskip("triton")
    sector = _rank_one_trivial_sector()
    wide = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(2, sector, 1, 0),))
    narrow = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(1, sector, 1, 0),))
    base = JointYoungCGProduct(wide, narrow, tree_type="balanced", L_max=1)
    reference = JointYoungCGProduct.from_static_schedule(base.static_schedule())
    reference.enable_real_basis_product(True).enable_packed_cuda(False)
    group = PackedSparseBilinearGroup(
        (
            reference.packed_real_bilinear_table(),
            reference.packed_real_bilinear_table(),
        ),
        ((0, 1), (0, 1)),
        (int(wide.dim), int(narrow.dim)),
        strict=True,
    ).cuda()

    generator = torch.Generator(device="cuda").manual_seed(542)
    wide_values = torch.randn((64, 6), dtype=dtype, device="cuda", generator=generator)
    narrow_values = torch.randn((64, 3), dtype=dtype, device="cuda", generator=generator)
    reference_wide = wide_values.clone().requires_grad_(True)
    reference_narrow = narrow_values.clone().requires_grad_(True)
    packed_wide = wide_values.clone().requires_grad_(True)
    packed_narrow = narrow_values.clone().requires_grad_(True)

    expected_one = reference(reference_wide, reference_narrow)
    expected = torch.cat((expected_one, expected_one), dim=1)
    actual_blocks = group((packed_wide, packed_narrow))
    actual = torch.cat(actual_blocks, dim=1)
    torch.testing.assert_close(
        actual,
        expected,
        atol=value_tolerance,
        rtol=value_tolerance,
    )

    expected_grads = torch.autograd.grad(
        expected.square().sum(),
        (reference_wide, reference_narrow),
        create_graph=True,
    )
    actual_grads = torch.autograd.grad(
        actual.square().sum(),
        (packed_wide, packed_narrow),
        create_graph=True,
    )
    torch.testing.assert_close(
        actual_grads[0],
        expected_grads[0],
        atol=gradient_tolerance,
        rtol=gradient_tolerance,
    )
    torch.testing.assert_close(
        actual_grads[1],
        expected_grads[1],
        atol=gradient_tolerance,
        rtol=gradient_tolerance,
    )

    expected_hessian = torch.autograd.grad(
        expected_grads[0].square().sum() + expected_grads[1].square().sum(),
        (reference_wide, reference_narrow),
    )
    actual_hessian = torch.autograd.grad(
        actual_grads[0].square().sum() + actual_grads[1].square().sum(),
        (packed_wide, packed_narrow),
    )
    torch.testing.assert_close(
        actual_hessian[0],
        expected_hessian[0],
        atol=hessian_tolerance,
        rtol=hessian_tolerance,
    )
    torch.testing.assert_close(
        actual_hessian[1],
        expected_hessian[1],
        atol=hessian_tolerance,
        rtol=hessian_tolerance,
    )
    report = group.report()
    assert report["last_backend"] == (
        "triton_segmented_sparse_quadratic_autograd"
    )
    assert report["destination_segmented_quadratic_active"] is True
    assert report["segmented_reduction"]["quadratic_output"][
        "max_segment_length"
    ] > 0
    scalar_reads = ScalarReadCounter()
    with scalar_reads:
        group((packed_wide.detach(), packed_narrow.detach()))
    torch.cuda.synchronize()
    assert scalar_reads.count == 0
    assert group.report()["last_backend"] == (
        "triton_segmented_sparse_quadratic"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
@pytest.mark.parametrize(
    "dtype,tolerance",
    (
        (torch.float64, 2.0e-11),
        (torch.float32, 2.0e-4),
    ),
)
def test_packed_cuda_quadratic_segment_length_buckets_match_stable_control(
    dtype,
    tolerance,
):
    from ye3t.backends.triton_joint import (
        PackedSparseBilinearGroup,
        SparseBilinearTable,
    )

    pytest.importorskip("triton")
    output_width = 1024
    source_width = 256
    lengths = torch.tensor(
        (1, 3, 9, 65) * (output_width // 4),
        dtype=torch.long,
    )
    term_count = int(lengths.sum())
    term = torch.arange(term_count, dtype=torch.long)
    output_index = torch.repeat_interleave(
        torch.arange(output_width, dtype=torch.long),
        lengths,
    )
    table = SparseBilinearTable(
        left_index=(term * 17 + 3).remainder(source_width),
        right_index=(term * 29 + 11).remainder(source_width),
        output_index=output_index,
        coefficient=torch.linspace(
            -0.75,
            0.9,
            term_count,
            dtype=torch.float64,
        ),
        output_width=output_width,
    )
    bucketed = PackedSparseBilinearGroup(
        (table,),
        ((0, 0),),
        (source_width,),
        strict=True,
        reduction_mode="segmented",
    ).cuda()
    stable = PackedSparseBilinearGroup(
        (table,),
        ((0, 0),),
        (source_width,),
        strict=True,
        reduction_mode="stable",
    ).cuda()
    generator = torch.Generator(device="cuda").manual_seed(546)
    values = torch.randn(
        (8, source_width),
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    actual_values = values.clone().requires_grad_(True)
    expected_values = values.clone().requires_grad_(True)
    actual = bucketed((actual_values,))[0]
    expected = stable((expected_values,))[0]
    torch.testing.assert_close(
        actual,
        expected,
        atol=tolerance,
        rtol=tolerance,
    )
    actual_gradient = torch.autograd.grad(
        actual.square().mean(),
        actual_values,
        create_graph=True,
    )[0]
    expected_gradient = torch.autograd.grad(
        expected.square().mean(),
        expected_values,
        create_graph=True,
    )[0]
    torch.testing.assert_close(
        actual_gradient,
        expected_gradient,
        atol=tolerance,
        rtol=tolerance,
    )
    direction = torch.randn(
        actual_gradient.shape,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    actual_hvp = torch.autograd.grad(
        (actual_gradient * direction).sum(),
        actual_values,
    )[0]
    expected_hvp = torch.autograd.grad(
        (expected_gradient * direction).sum(),
        expected_values,
    )[0]
    torch.testing.assert_close(
        actual_hvp,
        expected_hvp,
        atol=tolerance,
        rtol=tolerance,
    )
    report = bucketed.report()
    output_report = report["segmented_reduction"]["quadratic_output"]
    assert output_report["bucket_count"] == 4
    assert output_report["auto_eligible"] is True
    assert output_report["predicted_padding_reduction"] > 2.0
    assert report["single_numeric_launch"] is False
    assert stable.report()["single_numeric_launch"] is True


def test_packed_group_common_base_handoff_preserves_vjp_and_hvp():
    from ye3t.backends.triton_joint import PackedSparseBilinearGroup

    sector = _rank_one_trivial_sector()
    wide = GeneralizedExactRuntimeIrreps(
        (GeneralizedExactRuntimeBlock(2, sector, 1, 0),)
    )
    narrow = GeneralizedExactRuntimeIrreps(
        (GeneralizedExactRuntimeBlock(1, sector, 1, 0),)
    )
    base = JointYoungCGProduct(wide, narrow, tree_type="balanced", L_max=1)
    reference = JointYoungCGProduct.from_static_schedule(
        base.static_schedule()
    )
    reference.enable_real_basis_product(True).enable_streaming_vjp(False)
    group = PackedSparseBilinearGroup(
        (reference.packed_real_bilinear_table(),),
        ((0, 1),),
        (int(wide.dim), int(narrow.dim)),
    ).double()
    generator = torch.Generator().manual_seed(549)
    values = torch.randn(
        (8, int(wide.dim + narrow.dim)),
        dtype=torch.float64,
        generator=generator,
    )
    expected_base = values.clone().requires_grad_(True)
    actual_base = values.clone().requires_grad_(True)
    expected = reference(
        expected_base[:, : int(wide.dim)],
        expected_base[:, int(wide.dim) :],
    )
    actual = group(
        (
            actual_base[:, : int(wide.dim)],
            actual_base[:, int(wide.dim) :],
        )
    )[0]
    torch.testing.assert_close(actual, expected, atol=1.0e-12, rtol=1.0e-12)

    expected_gradient = torch.autograd.grad(
        expected.square().sum(),
        expected_base,
        create_graph=True,
    )[0]
    actual_gradient = torch.autograd.grad(
        actual.square().sum(),
        actual_base,
        create_graph=True,
    )[0]
    torch.testing.assert_close(
        actual_gradient,
        expected_gradient,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    expected_hessian = torch.autograd.grad(
        expected_gradient.square().sum(),
        expected_base,
    )[0]
    actual_hessian = torch.autograd.grad(
        actual_gradient.square().sum(),
        actual_base,
    )[0]
    torch.testing.assert_close(
        actual_hessian,
        expected_hessian,
        atol=1.0e-11,
        rtol=1.0e-11,
    )
    assert group.report()["source_pack_operation"] == (
        "zero_copy_shared_packed_carrier_view"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
def test_weighted_packed_cuda_group_matches_parameter_vjp_and_double_backward():
    from ye3t.backends.triton_joint import (
        PackedWeightedSparseBilinearGroup,
        weighted_sparse_bilinear_reference,
    )

    pytest.importorskip("triton")
    sector = _rank_one_trivial_sector()
    wide = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(2, sector, 1, 0),))
    narrow = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(1, sector, 1, 0),))
    base = JointYoungCGProduct(wide, narrow, tree_type="balanced", L_max=1)
    product = JointYoungCGProduct.from_static_schedule(base.static_schedule())
    product.enable_real_basis_product(True)
    group = PackedWeightedSparseBilinearGroup(
        (product.packed_real_bilinear_table(),),
        ((0, 1),),
        (int(wide.dim), int(narrow.dim)),
        (product.packed_real_output_blocks(),),
        strict=True,
    ).double().cuda()
    with torch.no_grad():
        group.mixing_weight.add_(
            torch.arange(
                int(group.mixing_weight.numel()),
                dtype=torch.float64,
                device="cuda",
            )
            / 19.0
        )
    generator = torch.Generator(device="cuda").manual_seed(543)
    wide_values = torch.randn((48, 6), dtype=torch.float64, device="cuda", generator=generator)
    narrow_values = torch.randn((48, 3), dtype=torch.float64, device="cuda", generator=generator)
    packed_actual = group.pack_sources(
        (
            wide_values.clone().requires_grad_(True),
            narrow_values.clone().requires_grad_(True),
        )
    )
    packed_reference = packed_actual.detach().clone().requires_grad_(True)
    weight_actual = group.mixing_weight
    weight_reference = weight_actual.detach().clone().requires_grad_(True)

    actual = torch.cat(group.forward_packed(packed_actual), dim=1)
    expected = weighted_sparse_bilinear_reference(
        packed_reference,
        packed_reference,
        weight_reference,
        group._table(),
    )
    torch.testing.assert_close(actual, expected, atol=1.0e-11, rtol=1.0e-11)
    actual_grads = torch.autograd.grad(
        actual.square().sum(),
        (packed_actual, weight_actual),
        create_graph=True,
    )
    expected_grads = torch.autograd.grad(
        expected.square().sum(),
        (packed_reference, weight_reference),
        create_graph=True,
    )
    torch.testing.assert_close(actual_grads[0], expected_grads[0], atol=1.0e-10, rtol=1.0e-10)
    torch.testing.assert_close(actual_grads[1], expected_grads[1], atol=1.0e-10, rtol=1.0e-10)
    actual_second = torch.autograd.grad(
        actual_grads[0].square().sum() + actual_grads[1].square().sum(),
        (packed_actual, weight_actual),
    )
    expected_second = torch.autograd.grad(
        expected_grads[0].square().sum() + expected_grads[1].square().sum(),
        (packed_reference, weight_reference),
    )
    torch.testing.assert_close(actual_second[0], expected_second[0], atol=1.0e-9, rtol=1.0e-9)
    torch.testing.assert_close(actual_second[1], expected_second[1], atol=1.0e-9, rtol=1.0e-9)
    report = group.report()
    assert report["last_backend"] == (
        "triton_segmented_fast_weighted_sparse_bilinear_autograd"
    )
    assert report["reduction_mode"] == "auto"
    assert report["effective_reduction_mode"] == "segmented"
    assert report["promotion_basis"] == (
        "fp64_value_vjp_hvp_and_interleaved_timing_gate_v1"
    )


def test_joint_young_cg_product_exposes_real_basis_coefficients():
    sector = _rank_one_trivial_sector()
    irreps_left = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(1, sector, 1, 0),))
    irreps_right = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(1, sector, 1, 0),))
    product = JointYoungCGProduct(irreps_left, irreps_right, tree_type="balanced", L_max=1)

    tensors = product.coefficient_tensors(instruction_index=0, basis="real")[0]
    assert tensors["basis"] == "real"
    assert tensors["joint_coupling_tensor"].dtype == torch.float64
    assert torch.isfinite(tensors["joint_coupling_tensor"]).all()


def test_joint_young_cg_product_exposes_exact_cg_young_and_joint_coefficients():
    sector = _rank_one_trivial_sector()
    irreps_left = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(1, sector, 1, 0),))
    irreps_right = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(1, sector, 1, 0),))
    product = JointYoungCGProduct(irreps_left, irreps_right, tree_type="balanced", L_max=1)

    tensors = product.coefficient_tensors(instruction_index=0)[0]
    cg_tensor = tensors["clebsch_gordan_tensor"]
    young_matrix = tensors["young_coupling_matrix"]
    joint_tensor = tensors["joint_coupling_tensor"]

    assert cg_tensor.shape[0] == len(tensors["source_basis_labels"])
    assert cg_tensor.shape[1] == len(tensors["left_basis_labels"])
    assert cg_tensor.shape[2] == len(tensors["right_basis_labels"])
    assert young_matrix.shape == (len(tensors["target_basis_labels"]), len(tensors["source_basis_labels"]))
    assert joint_tensor.shape == (
        len(tensors["target_basis_labels"]),
        len(tensors["left_basis_labels"]),
        len(tensors["right_basis_labels"]),
    )
    assert torch.allclose(joint_tensor, torch.einsum("ts,sij->tij", young_matrix, cg_tensor))

    report = product.coefficient_report(include_values=True, instruction_index=0)
    instruction = report["instructions"][0]
    assert instruction["clebsch_gordan"]["entries"]
    assert instruction["young_coupling"]["entries"]
    assert instruction["joint_coupling"]["entries"]
    assert report["coefficient_sources"]["clebsch_gordan"] == "ye3t.core.subtree_dag.cg_exact"
    assert report["coefficient_sources"]["young_coupling"] == "ExactLoweredChangeOfGroupBasisBranch.coordinate_matrix"
    assert report["coefficient_sources"]["joint_coupling"] == "young_coupling_matrix @ clebsch_gordan_tensor"

    cg_entry = instruction["clebsch_gordan"]["entries"][0]
    source_label = cg_entry["source_label"]
    left_label = cg_entry["left_basis_label"]
    right_label = cg_entry["right_basis_label"]
    M_out = int(source_label[2])
    left_M = int(left_label[1])
    right_M = int(right_label[1])
    value = complex(cg_entry["value"]["real"], cg_entry["value"]["imag"])
    expected = complex(cg_exact(1, left_M, 1, right_M, instruction["output_L"], M_out).evalf())
    assert abs(value - expected) < 1.0e-12

    fallback = product.fallback_report()
    assert fallback["uses_scalar_proxy"] is False
    assert fallback["uses_norm_shortcut"] is False
    assert fallback["uses_approximate_intertwiner"] is False


def test_joint_young_cg_static_schedule_manifest_carries_coefficient_hashes():
    sector = _rank_one_trivial_sector()
    irreps_left = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(1, sector, 1, 0),))
    irreps_right = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(1, sector, 1, 0),))
    product = JointYoungCGProduct(irreps_left, irreps_right, tree_type="balanced", L_max=1)

    manifest = product.static_schedule_manifest()
    schedule = product.static_schedule()

    assert manifest["format"] == "joint_young_cg_tensor_schedule_manifest_v1"
    assert manifest["coefficient_hash"]
    assert manifest["manifest_hash"]
    assert manifest["fallback_report"]["uses_scalar_proxy"] is False
    assert schedule.provenance["coefficient_hash"] == manifest["coefficient_hash"]
    assert schedule.provenance["manifest_hash"] == manifest["manifest_hash"]
    assert schedule.manifest()["coefficient_hash"] == manifest["coefficient_hash"]
    assert schedule.coefficient_report()["coefficient_hash"] == product.coefficient_report()["coefficient_hash"]

    loaded = JointYoungCGProduct.from_static_schedule(schedule)
    assert loaded.coefficient_report()["coefficient_hash"] == product.coefficient_report()["coefficient_hash"]
