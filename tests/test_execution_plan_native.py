import math
import os
import importlib
from types import SimpleNamespace

import pytest
import torch

from ye3t.couplings import (
    YE3TSourceAssemblyPlan,
    YE3TSourceRealization,
    YE3TSynthesisTable,
    apply_factorized_angular_analysis_reference,
    apply_source_analysis_reference,
    compile as compile_coupler,
    execution_plan_from_compiled_coupler,
    plan as coupling_plan,
    source_assembly_from_induction,
)
from ye3t.runtime.execution_plan import (
    YE3TFactorizedAngularModule,
    _factorized_angular_packed_reference,
    _PackedRepeatedBlockPowerDerivativeDestinationGroup,
    _PackedRepeatedBlockPowerPlanGroup,
    _YE3TFactorizedAngularModuleGroup,
    _YE3TSegmentedFactorizedAngularModuleGroup,
    _repeated_block_power_layout,
    _load_extension,
    apply_source_analysis,
    apply_source_analysis_native,
    cheb_exp_cos_radial_table_with_derivative,
    cheb_exp_cos_radial_with_derivative,
    compact_pair_product,
    compact_pair_product_reference,
    compact_exterior_power_product,
    compact_exterior_power_product_reference,
    carrier_channel_update,
    carrier_channel_update_adjoint,
    carrier_channel_transform,
    carrier_channel_transform_adjoint,
    carrier_role_channel_map_adjoint,
    carrier_gated_scatter,
    carrier_gated_scatter_adjoint,
    carrier_residual_gated_scatter,
    carrier_residual_gated_scatter_adjoint,
    carrier_segmented_residual_gated_scatter,
    carrier_segmented_residual_gated_scatter_adjoint,
    density_accumulate,
    edge_outer_accumulate,
    native_execution_plan_capabilities,
    plain_site_basis_product_adjoint,
    plain_site_basis_product_with_derivative,
    prepare_carrier_scatter_segments,
    prepare_source_arena_schedule,
    scheduled_radial_angular_channels_with_derivative,
    scheduled_softmax_gaussian_role_density,
    softmax_gaussian_role_density,
    spherical_harmonics_table_with_derivative,
    spherical_harmonics_with_derivative,
    source_arena_gather,
    source_arena_gather_adjoint,
    source_arena_channel_transform,
    sparse_monomial_support_from_counts,
    symmetric_power_monomial_contraction,
    symmetric_power_monomial_adjoint_reference,
    symmetric_power_monomial_reference,
    symmetric_power_shared_monomial_contraction,
    symmetric_power_shared_monomial_adjoint_reference,
    symmetric_power_shared_monomial_batched_adjoint,
    symmetric_power_shared_monomial_batched_adjoint_reference,
    symmetric_power_shared_monomial_reference,
    symmetric_power_shared_sparse_monomial_contraction,
)


pytestmark = [
    pytest.mark.fast,
    pytest.mark.skipif(
        not (
            os.environ.get("YE3T_TEST_CPP_EXTENSION")
            or native_execution_plan_capabilities()["prebuilt_extension"]
        ),
        reason="requires the prebuilt extension or YE3T_TEST_CPP_EXTENSION=1",
    ),
]


def _squared_norm(value):
    return (value.conj() * value).real.sum()


def test_torch_contract_registration_is_idempotent_across_module_reload():
    import ye3t.runtime.execution_plan as runtime_module

    runtime_module._load_extension()
    reloaded = importlib.reload(runtime_module)
    assert reloaded._load_extension() is not None


def _case(complex_coefficients):
    source_realization = YE3TSourceRealization(
        kind="lifted_density_roles",
        rank=2,
        content=(1, 1),
        role_labels=("left", "right"),
        retain_role_order=True,
    )
    assembly_values = (1.25, -0.75)
    synthesis_values = (
        1.0 / math.sqrt(2.0),
        (
            1j / math.sqrt(2.0)
            if complex_coefficients
            else -1.0 / math.sqrt(2.0)
        ),
    )
    assembly = YE3TSourceAssemblyPlan(
        assembly_id="native_test_assembly",
        source_realization=source_realization,
        source_dimension=2,
        induced_dimension=2,
        row_indices=(0, 1),
        column_indices=(0, 1),
        values=assembly_values,
    )
    synthesis = YE3TSynthesisTable(
        table_id="native_test_synthesis",
        input_dimension=2,
        output_dimension=1,
        row_indices=(0, 1),
        column_indices=(0, 0),
        values=synthesis_values,
    )
    return assembly, synthesis


def _rank3_factorized_plan():
    compiled = compile_coupler(
        coupling_plan(
            content=(1, 2, 3),
            input_Ls=(1, 2, 1),
            target_L=2,
        ),
        subduction_materialization_backend="exact",
    )
    return execution_plan_from_compiled_coupler(compiled)


def _rank3_factorized_role_plan(target_L=2):
    target_L = int(target_L)
    compiled = compile_coupler(
        coupling_plan(
            content=(1, 2, 3),
            input_Ls=(1, 2, 1),
            target_L=target_L,
            carrier="A_s",
            target_permutation="young:2,1",
            carrier_options={
                "role_coordinate_policy": "role_resolved",
                "slot_count": 3,
                "permuted_slot_count": 3,
            },
        ),
        subduction_materialization_backend="exact",
    )
    source = YE3TSourceRealization(
        kind="lifted_density_roles",
        rank=3,
        content=(1, 2, 3),
        role_labels=("role_0", "role_1", "role_2"),
        retain_role_order=True,
    )
    assembly = source_assembly_from_induction(
        compiled.coupler.induction_couplers[0],
        source,
        assembly_id=f"rank3_roles_L{target_L}",
    )
    return execution_plan_from_compiled_coupler(
        compiled,
        source_realization=source,
        source_assembly=assembly,
    )


def _rank5_factorized_plan(target_L=2):
    compiled = compile_coupler(
        coupling_plan(
            content=(1, 2, 3, 4, 5),
            input_Ls=(1, 1, 1, 1, 1),
            target_L=int(target_L),
        ),
        subduction_materialization_backend="exact",
    )
    return execution_plan_from_compiled_coupler(compiled)


def _rank4_homogeneous_factorized_role_plan(target_L):
    target_L = int(target_L)
    compiled = compile_coupler(
        coupling_plan(
            content=(1, 1, 1, 1),
            input_Ls=(1, 1, 1, 1),
            target_L=target_L,
            carrier="A_s",
            target_permutation="young:4",
            carrier_options={
                "role_coordinate_policy": "role_resolved",
                "slot_count": 4,
                "permuted_slot_count": 4,
            },
        ),
        subduction_materialization_backend="exact",
    )
    source = YE3TSourceRealization(
        kind="lifted_density_roles",
        rank=4,
        content=(1, 1, 1, 1),
        role_labels=("role_0",) * 4,
        retain_role_order=True,
    )
    assembly = source_assembly_from_induction(
        compiled.coupler.induction_couplers[0],
        source,
        assembly_id=(
            "test_homogeneous_rank4_L" + str(target_L)
        ),
    )
    return execution_plan_from_compiled_coupler(
        compiled,
        source_realization=source,
        source_assembly=assembly,
    )


def _rank4_two_block_factorized_role_plan(target_L):
    target_L = int(target_L)
    compiled = compile_coupler(
        coupling_plan(
            content=(1, 1, 2, 2),
            input_Ls=(1, 1, 1, 1),
            target_L=target_L,
            carrier="A_s",
            target_permutation="young:3,1",
            carrier_options={
                "role_coordinate_policy": "role_resolved",
                "slot_count": 4,
                "permuted_slot_count": 4,
            },
            metadata={"subgroup_partitions": ((2,), (2,))},
        ),
        subduction_materialization_backend="exact",
    )
    source = YE3TSourceRealization(
        kind="lifted_density_roles",
        rank=4,
        content=(1, 1, 2, 2),
        role_labels=("role_0", "role_0", "role_1", "role_1"),
        retain_role_order=True,
    )
    assembly = source_assembly_from_induction(
        compiled.coupler.induction_couplers[0],
        source,
        assembly_id="test_two_block_rank4_L" + str(target_L),
    )
    return execution_plan_from_compiled_coupler(
        compiled,
        source_realization=source,
        source_assembly=assembly,
    )


def _rank3_nontrivial_child_tableau_plan():
    compiled = compile_coupler(
        coupling_plan(
            content=(1, 1, 1),
            input_Ls=(1, 1, 1),
            target_L=2,
            carrier="A_s",
            target_permutation="young:2,1",
            carrier_options={
                "role_coordinate_policy": "role_resolved",
                "slot_count": 3,
                "permuted_slot_count": 3,
            },
            metadata={"subgroup_partitions": ((2, 1),)},
        ),
        subduction_materialization_backend="exact",
    )
    source = YE3TSourceRealization(
        kind="lifted_density_roles",
        rank=3,
        content=(1, 1, 1),
        role_labels=("role_0", "role_1", "role_2"),
        retain_role_order=True,
    )
    assembly = source_assembly_from_induction(
        compiled.coupler.induction_couplers[0],
        source,
        assembly_id="rank3_nontrivial_child_tableau_native",
    )
    return execution_plan_from_compiled_coupler(
        compiled,
        source_realization=source,
        source_assembly=assembly,
    )


def test_native_extension_reports_factorized_abi():
    _load_extension()
    capabilities = native_execution_plan_capabilities()
    assert _load_extension().core_abi_version() >= 21
    assert capabilities["heterogeneous_factorized_roots"]
    assert "cheb_exp_cos_radial_with_derivative" in capabilities["operations"]
    assert (
        "cheb_exp_cos_radial_table_with_derivative"
        in capabilities["operations"]
    )
    assert (
        "cheb_exp_cos_radial_table_double_backward"
        in capabilities["operations"]
    )
    assert (
        "complex_spherical_harmonics_with_derivative"
        in capabilities["operations"]
    )
    assert (
        "real_spherical_harmonics_with_derivative"
        in capabilities["operations"]
    )
    assert (
        "complex_spherical_harmonics_table_with_derivative"
        in capabilities["operations"]
    )
    assert (
        "real_spherical_harmonics_table_with_derivative"
        in capabilities["operations"]
    )
    if capabilities["cuda"]:
        assert (
            "cheb_exp_cos_radial_table_double_backward"
            in capabilities["cuda_operations"]
        )
        assert (
            "real_spherical_harmonics_table_double_backward"
            in capabilities["operations"]
        )
        assert (
            "spherical_harmonics_table_double_backward"
            in capabilities["cuda_operations"]
        )
    assert (
        "plain_site_basis_product_with_derivative"
        in capabilities["operations"]
    )
    assert "plain_site_basis_product_adjoint" in capabilities["operations"]
    assert (
        "scheduled_radial_angular_channels_with_derivative"
        in capabilities["operations"]
    )
    assert "density_accumulate" in capabilities["operations"]
    assert "density_accumulate_adjoint" in capabilities["operations"]
    assert "edge_outer_accumulate" in capabilities["operations"]
    assert "edge_outer_accumulate_adjoint" in capabilities["operations"]
    assert (
        "edge_outer_accumulate_double_backward"
        in capabilities["operations"]
    )
    assert "softmax_gaussian_role_density" in capabilities["operations"]
    assert (
        "softmax_gaussian_role_density_adjoint"
        in capabilities["operations"]
    )
    assert (
        "softmax_gaussian_role_density_double_backward"
        in capabilities["operations"]
    )
    assert capabilities["scheduled_softmax_gaussian_role_density"]
    assert (
        "scheduled_softmax_gaussian_role_density"
        in capabilities["operations"]
    )
    assert capabilities["core_abi_version"] >= 33
    assert capabilities["segmented_factorized_angular"]
    assert "factorized_angular_segmented" in capabilities["operations"]
    assert (
        "factorized_angular_segmented_adjoint"
        in capabilities["operations"]
    )
    assert (
        "factorized_angular_segmented_double_backward"
        in capabilities["operations"]
    )
    assert capabilities["real_control_carrier_gates"]
    assert capabilities["real_control_carrier_channel_update"]
    assert "carrier_gated_scatter" in capabilities["operations"]
    assert "carrier_gated_scatter_adjoint" in capabilities["operations"]
    assert (
        "carrier_gated_scatter_double_backward"
        in capabilities["operations"]
    )
    assert capabilities["residual_carrier_gated_scatter"]
    assert "carrier_residual_gated_scatter" in capabilities["operations"]
    assert (
        "carrier_residual_gated_scatter_adjoint"
        in capabilities["operations"]
    )
    assert (
        "carrier_residual_gated_scatter_double_backward"
        in capabilities["operations"]
    )
    assert capabilities["segmented_carrier_gated_scatter"]
    if capabilities["cuda"]:
        assert capabilities[
            "cuda_segmented_carrier_scatter_min_node_feature_work"
        ] == (1 << 20)
    else:
        assert capabilities[
            "cuda_segmented_carrier_scatter_min_node_feature_work"
        ] is None
    assert (
        "carrier_segmented_residual_gated_scatter"
        in capabilities["operations"]
    )
    assert (
        "carrier_segmented_residual_gated_scatter_adjoint"
        in capabilities["operations"]
    )
    assert (
        "carrier_segmented_residual_gated_scatter_double_backward"
        in capabilities["operations"]
    )
    assert capabilities[
        "cpu_carrier_gated_scatter_auto_min_work_items"
    ] == (1 << 15)
    if capabilities["cuda"]:
        assert capabilities[
            "cuda_carrier_gated_scatter_auto_min_work_items"
        ] == (1 << 18)
    else:
        assert capabilities[
            "cuda_carrier_gated_scatter_auto_min_work_items"
        ] is None
    assert "carrier_channel_update" in capabilities["operations"]
    assert "carrier_channel_update_adjoint" in capabilities["operations"]
    assert (
        "carrier_channel_update_double_backward"
        in capabilities["operations"]
    )
    assert (
        capabilities[
            "cpu_carrier_channel_update_auto_max_work_items"
        ]
        == (1 << 15)
    )
    if capabilities["cuda"]:
        assert capabilities[
            "cuda_carrier_channel_update_auto_max_work_items"
        ] is None
        assert capabilities[
            "cuda_carrier_channel_update_auto_min_work_items"
        ] == 0
    assert capabilities[
        "cpu_edge_outer_auto_min_work_items"
    ] == (1 << 16)
    if capabilities["cuda"]:
        assert capabilities[
            "cuda_edge_outer_auto_min_work_items"
        ] == (1 << 18)
    assert "source_analysis_linear" in capabilities["operations"]
    assert "source_analysis_linear_adjoint" in capabilities["operations"]
    assert "factorized_angular" in capabilities["operations"]
    assert "factorized_angular_adjoint" in capabilities["operations"]
    assert (
        "factorized_angular_double_backward"
        in capabilities["operations"]
    )
    assert (
        "factorized_angular_heterogeneous"
        in capabilities["operations"]
    )
    assert (
        "factorized_angular_heterogeneous_adjoint"
        in capabilities["operations"]
    )
    assert (
        "factorized_angular_heterogeneous_double_backward"
        in capabilities["operations"]
    )
    if capabilities["cuda"]:
        assert capabilities["cached_heterogeneous_factorized_reverse"]
        assert (
            "factorized_angular_heterogeneous_adjoint_with_workspace"
            in capabilities["operations"]
        )
        assert (
            "factorized_angular_heterogeneous_double_backward_from_workspace"
            in capabilities["operations"]
        )
    else:
        assert not capabilities["cached_heterogeneous_factorized_reverse"]
    assert "factorized_angular_linear" in capabilities["operations"]
    assert "factorized_angular_linear_adjoint" in capabilities["operations"]
    assert "symmetric_power_monomial" in capabilities["operations"]
    assert "symmetric_power_monomial_adjoint" in capabilities["operations"]
    if capabilities["cuda"]:
        assert capabilities["symmetric_power_monomial_double_backward"]
        assert (
            "symmetric_power_monomial_double_backward"
            in capabilities["cuda_operations"]
        )
        assert capabilities[
            "symmetric_power_shared_monomial_double_backward"
        ]
        assert (
            "symmetric_power_shared_monomial_double_backward"
            in capabilities["cuda_operations"]
        )
        assert capabilities[
            "symmetric_power_shared_monomial_factored_double_backward"
        ]
        assert (
            "symmetric_power_shared_monomial_factored_double_backward"
            in capabilities["cuda_operations"]
        )
        assert (
            capabilities[
                "symmetric_power_shared_double_backward_default_algorithm"
            ]
            == "auto"
        )
        assert capabilities["symmetric_power_shared_sparse_monomial"]
        for operation in (
            "symmetric_power_shared_sparse_monomial",
            "symmetric_power_shared_sparse_monomial_adjoint",
            "symmetric_power_shared_sparse_monomial_double_backward",
        ):
            assert operation in capabilities["cuda_operations"]
    assert "symmetric_power_shared_monomial" in capabilities["operations"]
    assert (
        "symmetric_power_shared_monomial_adjoint"
        in capabilities["operations"]
    )
    assert (
        "symmetric_power_shared_monomial_batched_adjoint"
        in capabilities["operations"]
    )
    if capabilities["cuda"]:
        assert "factorized_angular" in capabilities["cuda_operations"]
        assert "factorized_angular_adjoint" in capabilities["cuda_operations"]
        assert (
            "factorized_angular_double_backward"
            in capabilities["cuda_operations"]
        )
        assert (
            "factorized_angular_heterogeneous"
            in capabilities["cuda_operations"]
        )
        assert (
            "factorized_angular_heterogeneous_adjoint"
            in capabilities["cuda_operations"]
        )
        assert (
            "factorized_angular_heterogeneous_double_backward"
            in capabilities["cuda_operations"]
        )
        assert (
            "factorized_angular_heterogeneous_adjoint_with_workspace"
            in capabilities["cuda_operations"]
        )
        assert (
            "factorized_angular_heterogeneous_double_backward_from_workspace"
            in capabilities["cuda_operations"]
        )
        assert "factorized_angular_linear" in capabilities["cuda_operations"]
        assert (
            "factorized_angular_linear_adjoint"
            in capabilities["cuda_operations"]
        )
    else:
        assert capabilities["cuda_operations"] == ()
    assert capabilities["symmetric_power_cpu_auto_max_batch"] == 32
    assert capabilities["symmetric_power_cpu_auto_min_power"] == 6
    assert capabilities["symmetric_power_shared_cpu_min_reuse"] == 3.0
    assert capabilities["symmetric_power_shared_cpu_min_batch"] == 8
    if capabilities["cuda"]:
        assert capabilities["symmetric_power_shared_cuda_min_reuse"] == 8.0
        assert capabilities["symmetric_power_shared_cuda_grad_min_batch"] == 256
        assert (
            capabilities["symmetric_power_shared_cuda_forward_min_batch"]
            == 1024
        )
    assert "triton_available" in capabilities
    assert "triton_cuda_available" in capabilities
    assert (
        capabilities["triton_real_convention_id"]
        == "real_tesseral_from_complex_condon_shortley_young_orthogonal_v1"
    )
    assert (
        "dense_source_analysis_C_dagger_L_v"
        in capabilities["triton_operations"]
    )
    assert (
        "dense_source_analysis_channel_mixing"
        in capabilities["triton_operations"]
    )


@pytest.mark.parametrize("radial_index", range(7))
def test_native_cheb_exp_cos_values_derivatives_and_autograd(
    monkeypatch,
    radial_index,
):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    radii = torch.tensor(
        [0.0, 0.17, 0.83, 1.72, 2.49, 3.1],
        dtype=torch.float64,
        requires_grad=True,
    )
    cutoffs = torch.tensor(
        [2.0, 2.2, 2.4, 2.6, 3.0, 3.0],
        dtype=torch.float64,
    )
    lambdas = torch.tensor(
        [0.15, 0.25, 0.4, 0.7, 1.1, 0.35],
        dtype=torch.float64,
    )
    actual_values, actual_derivatives = (
        cheb_exp_cos_radial_with_derivative(
            radii,
            cutoffs,
            lambdas,
            radial_index,
            backend="native",
        )
    )
    expected_values, expected_derivatives = (
        cheb_exp_cos_radial_with_derivative(
            radii,
            cutoffs,
            lambdas,
            radial_index,
            backend="reference",
        )
    )
    torch.testing.assert_close(
        actual_values,
        expected_values,
        rtol=2e-13,
        atol=2e-13,
    )
    torch.testing.assert_close(
        actual_derivatives,
        expected_derivatives,
        rtol=2e-12,
        atol=2e-12,
    )

    def evaluate(value):
        return cheb_exp_cos_radial_with_derivative(
            value,
            cutoffs,
            lambdas,
            radial_index,
            backend="native",
        )[0]

    assert torch.autograd.gradcheck(
        evaluate,
        (radii,),
        eps=1e-6,
        atol=1e-9,
        rtol=1e-7,
    )
    assert torch.autograd.gradgradcheck(
        evaluate,
        (radii,),
        eps=1e-6,
        atol=2e-8,
        rtol=2e-6,
    )


def test_cheb_exp_cos_cutoff_regularity_is_explicit():
    cutoff = torch.tensor([4.0], dtype=torch.float64)
    radial_lambda = torch.tensor([0.1], dtype=torch.float64)
    epsilon = 1.0e-6

    def derivatives(radius_value, radial_index):
        radius = torch.tensor(
            [radius_value],
            dtype=torch.float64,
            requires_grad=True,
        )
        value = cheb_exp_cos_radial_with_derivative(
            radius,
            cutoff,
            radial_lambda,
            radial_index,
            backend="reference",
        )[0]
        first = torch.autograd.grad(
            value.sum(),
            radius,
            create_graph=True,
        )[0]
        second = torch.autograd.grad(first.sum(), radius)[0]
        return tuple(
            float(item.detach()) for item in (value, first, second)
        )

    n1_left = derivatives(4.0 - epsilon, 1)
    n1_right = derivatives(4.0 + epsilon, 1)
    assert abs(n1_left[0]) < 1.0e-10
    assert abs(n1_left[1]) < 1.0e-5
    assert n1_right == (0.0, 0.0, 0.0)
    assert abs(n1_left[2]) > 0.1

    n2_left = derivatives(4.0 - epsilon, 2)
    n2_right = derivatives(4.0 + epsilon, 2)
    assert abs(n2_left[0]) < 1.0e-15
    assert abs(n2_left[1]) < 1.0e-9
    assert abs(n2_left[2]) < 1.0e-3
    assert n2_right == (0.0, 0.0, 0.0)


def test_native_radial_table_matches_fixed_indices_and_passes_opcheck(
    monkeypatch,
):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    radii = torch.tensor(
        [0.0, 0.17, 0.83, 1.72, 2.49, 3.1],
        dtype=torch.float64,
        requires_grad=True,
    )
    cutoffs = torch.tensor(
        [2.0, 2.2, 2.4, 2.6, 3.0, 3.0],
        dtype=torch.float64,
    )
    lambdas = torch.tensor(
        [0.15, 0.25, 0.4, 0.7, 1.1, 0.35],
        dtype=torch.float64,
    )
    values, derivatives = cheb_exp_cos_radial_table_with_derivative(
        radii,
        cutoffs,
        lambdas,
        6,
        backend="native",
    )
    expected = [
        cheb_exp_cos_radial_with_derivative(
            radii,
            cutoffs,
            lambdas,
            radial_index,
            backend="reference",
        )
        for radial_index in range(7)
    ]
    torch.testing.assert_close(
        values,
        torch.stack([item[0] for item in expected], dim=1),
        rtol=2e-13,
        atol=2e-13,
    )
    torch.testing.assert_close(
        derivatives,
        torch.stack([item[1] for item in expected], dim=1),
        rtol=2e-12,
        atol=2e-12,
    )
    assert torch.autograd.gradcheck(
        lambda value: cheb_exp_cos_radial_table_with_derivative(
            value,
            cutoffs,
            lambdas,
            6,
            backend="native",
        )[0],
        (radii,),
        eps=1e-6,
        atol=1e-9,
        rtol=1e-7,
    )
    assert torch.autograd.gradgradcheck(
        lambda value: cheb_exp_cos_radial_table_with_derivative(
            value,
            cutoffs,
            lambdas,
            6,
            backend="native",
        )[0],
        (radii,),
        eps=1e-6,
        atol=2e-8,
        rtol=2e-6,
    )
    result = torch.library.opcheck(
        torch.ops.ye3t_runtime.cheb_exp_cos_radial_table_with_derivative,
        (
            radii.detach(),
            cutoffs,
            lambdas,
            6,
        ),
    )
    assert all(item == "SUCCESS" for item in result.values())


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA native radial runtime",
)
def test_native_cuda_radial_table_vjp_hvp_matches_reference(monkeypatch):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    radii = torch.linspace(
        0.05,
        2.85,
        257,
        dtype=torch.float64,
        device="cuda",
        requires_grad=True,
    )
    reference_radii = radii.detach().clone().requires_grad_(True)
    cutoffs = torch.full_like(radii, 3.0)
    lambdas = torch.linspace(
        0.15,
        0.85,
        int(radii.numel()),
        dtype=radii.dtype,
        device=radii.device,
    )
    actual = cheb_exp_cos_radial_table_with_derivative(
        radii,
        cutoffs,
        lambdas,
        6,
        backend="native",
    )[0]
    expected = cheb_exp_cos_radial_table_with_derivative(
        reference_radii,
        cutoffs,
        lambdas,
        6,
        backend="reference",
    )[0]
    weights = torch.linspace(
        -0.7,
        0.9,
        int(actual.numel()),
        dtype=actual.dtype,
        device=actual.device,
    ).reshape_as(actual)
    actual_grad = torch.autograd.grad(
        (actual * weights).sum(),
        radii,
        create_graph=True,
    )[0]
    expected_grad = torch.autograd.grad(
        (expected * weights).sum(),
        reference_radii,
        create_graph=True,
    )[0]
    direction = torch.linspace(
        0.8,
        -0.4,
        int(radii.numel()),
        dtype=radii.dtype,
        device=radii.device,
    )
    actual_hvp = torch.autograd.grad(
        (actual_grad * direction).sum(),
        radii,
    )[0]
    expected_hvp = torch.autograd.grad(
        (expected_grad * direction).sum(),
        reference_radii,
    )[0]

    torch.testing.assert_close(actual, expected, atol=2.0e-12, rtol=2.0e-12)
    torch.testing.assert_close(
        actual_grad,
        expected_grad,
        atol=2.0e-11,
        rtol=2.0e-11,
    )
    torch.testing.assert_close(
        actual_hvp,
        expected_hvp,
        atol=2.0e-10,
        rtol=2.0e-10,
    )


@pytest.mark.parametrize("real_output", [False, True])
@pytest.mark.parametrize("angular_momentum", range(6))
def test_native_spherical_values_derivatives_and_autograd(
    monkeypatch,
    real_output,
    angular_momentum,
):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    edge_vectors = torch.tensor(
        [
            [0.31, 0.47, 0.83],
            [-0.42, 0.58, 0.27],
            [0.73, -0.24, 0.51],
            [-0.67, -0.19, 0.37],
            [0.28, 0.63, -0.44],
            [-0.35, 0.22, -0.91],
            [0.59, -0.71, -0.18],
            [-0.76, -0.41, -0.29],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    actual_values, actual_derivatives = spherical_harmonics_with_derivative(
        edge_vectors,
        angular_momentum,
        real_output=real_output,
        backend="native",
    )
    expected_values, expected_derivatives = (
        spherical_harmonics_with_derivative(
            edge_vectors,
            angular_momentum,
            real_output=real_output,
            backend="reference",
        )
    )
    torch.testing.assert_close(
        actual_values,
        expected_values,
        rtol=2e-12,
        atol=2e-12,
    )
    torch.testing.assert_close(
        actual_derivatives,
        expected_derivatives,
        rtol=2e-11,
        atol=2e-11,
    )

    def evaluate(value):
        return spherical_harmonics_with_derivative(
            value,
            angular_momentum,
            real_output=real_output,
            backend="native",
        )[0]

    assert torch.autograd.gradcheck(
        evaluate,
        (edge_vectors,),
        eps=1e-6,
        atol=2e-8,
        rtol=2e-6,
    )
    assert torch.autograd.gradgradcheck(
        evaluate,
        (edge_vectors,),
        eps=1e-6,
        atol=2e-7,
        rtol=2e-5,
    )


@pytest.mark.parametrize("real_output", [False, True])
def test_native_spherical_table_matches_fixed_degrees_and_passes_opcheck(
    monkeypatch,
    real_output,
):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    edge_vectors = torch.tensor(
        [
            [0.31, 0.47, 0.83],
            [-0.42, 0.58, 0.27],
            [0.73, -0.24, 0.51],
            [-0.67, -0.19, 0.37],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    values, derivatives = spherical_harmonics_table_with_derivative(
        edge_vectors,
        5,
        real_output=real_output,
        backend="native",
    )
    expected = [
        spherical_harmonics_with_derivative(
            edge_vectors,
            angular_momentum,
            real_output=real_output,
            backend="reference",
        )
        for angular_momentum in range(6)
    ]
    torch.testing.assert_close(
        values,
        torch.cat([item[0] for item in expected], dim=1),
        rtol=2e-12,
        atol=2e-12,
    )
    torch.testing.assert_close(
        derivatives,
        torch.cat([item[1] for item in expected], dim=1),
        rtol=2e-11,
        atol=2e-11,
    )
    assert torch.autograd.gradcheck(
        lambda value: spherical_harmonics_table_with_derivative(
            value,
            3,
            real_output=real_output,
            backend="native",
        )[0],
        (edge_vectors,),
        eps=1e-6,
        atol=2e-8,
        rtol=2e-6,
    )
    result = torch.library.opcheck(
        torch.ops.ye3t_runtime.spherical_harmonics_table_with_derivative,
        (
            edge_vectors.detach(),
            3,
            real_output,
            1.0e-12,
        ),
    )
    assert all(item == "SUCCESS" for item in result.values())


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA native spherical runtime",
)
@pytest.mark.parametrize("maximum_angular_momentum", [1, 3, 5])
def test_native_cuda_real_spherical_table_double_backward_matches_reference(
    monkeypatch,
    maximum_angular_momentum,
):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    edge_vectors = torch.tensor(
        [
            [0.31, 0.47, 0.83],
            [-0.42, 0.58, 0.27],
            [0.73, -0.24, 0.51],
            [-0.67, -0.19, 0.37],
            [0.28, 0.63, -0.44],
            [-0.35, 0.22, -0.91],
            [0.59, -0.71, -0.18],
            [-0.76, -0.41, -0.29],
        ],
        dtype=torch.float64,
        device="cuda",
    )
    width = (maximum_angular_momentum + 1) ** 2
    value_adjoint = torch.linspace(
        -0.8,
        0.9,
        int(edge_vectors.size(0)) * width,
        dtype=edge_vectors.dtype,
        device=edge_vectors.device,
    ).reshape(int(edge_vectors.size(0)), width)
    edge_direction = torch.linspace(
        0.7,
        -0.5,
        int(edge_vectors.numel()),
        dtype=edge_vectors.dtype,
        device=edge_vectors.device,
    ).reshape_as(edge_vectors)

    reference_edges = edge_vectors.detach().clone().requires_grad_(True)
    reference_adjoint = value_adjoint.detach().clone().requires_grad_(True)
    reference_values = spherical_harmonics_table_with_derivative(
        reference_edges,
        maximum_angular_momentum,
        real_output=True,
        backend="reference",
    )[0]
    first = torch.autograd.grad(
        (reference_values * reference_adjoint).sum(),
        reference_edges,
        create_graph=True,
    )[0]
    expected_adjoint_gradient, expected_edge_gradient = (
        torch.autograd.grad(
            (first * edge_direction).sum(),
            (reference_adjoint, reference_edges),
        )
    )
    actual_adjoint_gradient, actual_edge_gradient = (
        torch.ops.ye3t_runtime
        .spherical_harmonics_table_double_backward(
            edge_vectors.contiguous(),
            value_adjoint.contiguous(),
            edge_direction.contiguous(),
            maximum_angular_momentum,
            1.0e-12,
        )
    )

    torch.testing.assert_close(
        actual_adjoint_gradient,
        expected_adjoint_gradient,
        atol=2.0e-11,
        rtol=2.0e-11,
    )
    torch.testing.assert_close(
        actual_edge_gradient,
        expected_edge_gradient,
        atol=2.0e-10,
        rtol=2.0e-10,
    )

    native_edges = edge_vectors.detach().clone().requires_grad_(True)
    native_adjoint = value_adjoint.detach().clone().requires_grad_(True)
    native_values = spherical_harmonics_table_with_derivative(
        native_edges,
        maximum_angular_momentum,
        real_output=True,
        backend="native",
    )[0]
    native_first = torch.autograd.grad(
        (native_values * native_adjoint).sum(),
        native_edges,
        create_graph=True,
    )[0]
    native_adjoint_gradient, native_edge_gradient = torch.autograd.grad(
        (native_first * edge_direction).sum(),
        (native_adjoint, native_edges),
    )
    torch.testing.assert_close(
        native_adjoint_gradient,
        expected_adjoint_gradient,
        atol=2.0e-11,
        rtol=2.0e-11,
    )
    torch.testing.assert_close(
        native_edge_gradient,
        expected_edge_gradient,
        atol=2.0e-10,
        rtol=2.0e-10,
    )


@pytest.mark.parametrize("complex_values", [False, True])
def test_native_plain_site_basis_product_matches_reference_and_derivatives(
    monkeypatch,
    complex_values,
):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    dtype = torch.complex128 if complex_values else torch.float64
    group_count = 3
    edge_count = 5
    term_count = 6
    channel_count = 8
    radial_values = torch.randn(
        group_count,
        edge_count,
        dtype=dtype,
        requires_grad=True,
    )
    radial_derivatives = torch.randn(
        group_count,
        edge_count,
        dtype=dtype,
    )
    angular_values = torch.randn(
        term_count,
        edge_count,
        dtype=dtype,
        requires_grad=True,
    )
    angular_derivatives = torch.randn(
        term_count,
        edge_count,
        3,
        dtype=dtype,
    )
    prefactors = torch.randn(
        group_count,
        edge_count,
        dtype=dtype,
        requires_grad=True,
    )
    prefactor_derivatives_center = torch.randn(
        group_count,
        edge_count,
        dtype=dtype,
    )
    prefactor_derivatives_neighbor = torch.randn(
        group_count,
        edge_count,
        dtype=dtype,
    )
    radial_directions = torch.randn(
        edge_count,
        3,
        dtype=dtype,
    )
    term_groups = torch.tensor(
        [0, 0, 1, 1, 2, 2],
        dtype=torch.int64,
    )
    term_channels = torch.tensor(
        [7, 1, 5, 0, 4, 2],
        dtype=torch.int64,
    )
    inputs = (
        radial_values,
        radial_derivatives,
        angular_values,
        angular_derivatives,
        prefactors,
        prefactor_derivatives_center,
        prefactor_derivatives_neighbor,
        radial_directions,
        term_groups,
        term_channels,
        channel_count,
    )
    actual = plain_site_basis_product_with_derivative(
        *inputs,
        backend="native",
    )
    expected = plain_site_basis_product_with_derivative(
        *inputs,
        backend="reference",
    )
    for actual_value, expected_value in zip(actual, expected):
        torch.testing.assert_close(
            actual_value,
            expected_value,
            rtol=2e-13,
            atol=2e-13,
        )

    def evaluate(radial, angular, prefactor):
        return plain_site_basis_product_with_derivative(
            radial,
            radial_derivatives,
            angular,
            angular_derivatives,
            prefactor,
            prefactor_derivatives_center,
            prefactor_derivatives_neighbor,
            radial_directions,
            term_groups,
            term_channels,
            channel_count,
            backend="native",
        )[0]

    assert torch.autograd.gradcheck(
        evaluate,
        (radial_values, angular_values, prefactors),
        eps=1e-6,
        atol=2e-8,
        rtol=2e-6,
    )
    assert torch.autograd.gradgradcheck(
        evaluate,
        (radial_values, angular_values, prefactors),
        eps=1e-6,
        atol=2e-7,
        rtol=2e-5,
    )
    result = torch.library.opcheck(
        torch.ops.ye3t_runtime.plain_site_basis_product_with_derivative,
        tuple(value.detach() if torch.is_tensor(value) else value for value in inputs),
    )
    assert all(item == "SUCCESS" for item in result.values())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_scheduled_radial_angular_channels_values_vjp_hvp_and_opcheck(
    monkeypatch,
    device,
):
    if device == "cuda" and (
        not torch.cuda.is_available() or not _load_extension().has_cuda()
    ):
        pytest.skip("requires native CUDA")
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    edge_count = 5
    radial_width = 5
    angular_width = 9
    channel_count = 6
    radial_values = torch.randn(
        edge_count,
        radial_width,
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    radial_derivatives = torch.randn_like(radial_values)
    angular_values = torch.randn(
        edge_count,
        angular_width,
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    angular_derivatives = torch.randn(
        edge_count,
        angular_width,
        3,
        dtype=torch.float64,
        device=device,
    )
    radial_directions = torch.randn(
        edge_count,
        3,
        dtype=torch.float64,
        device=device,
    )
    edge_types = torch.tensor(
        [0, 1, 2, 1, 0],
        dtype=torch.int64,
        device=device,
    )
    channel_radial_indices = torch.tensor(
        [0, 1, 2, 3, 4, 2],
        dtype=torch.int64,
        device=device,
    )
    channel_angular_indices = torch.tensor(
        [0, 1, 4, 8, 6, 3],
        dtype=torch.int64,
        device=device,
    )
    channel_types = torch.tensor(
        [-1, 0, 1, 2, 1, -1],
        dtype=torch.int64,
        device=device,
    )
    channel_scales = torch.randn(
        channel_count,
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    inputs = (
        radial_values,
        radial_derivatives,
        angular_values,
        angular_derivatives,
        radial_directions,
        edge_types,
        channel_radial_indices,
        channel_angular_indices,
        channel_types,
        channel_scales,
    )
    actual = scheduled_radial_angular_channels_with_derivative(
        *inputs,
        backend="native",
    )
    expected = scheduled_radial_angular_channels_with_derivative(
        *inputs,
        backend="reference",
    )
    for actual_value, expected_value in zip(actual, expected):
        torch.testing.assert_close(
            actual_value,
            expected_value,
            rtol=2e-13,
            atol=2e-13,
        )

    def evaluate(radial, angular, scales):
        return scheduled_radial_angular_channels_with_derivative(
            radial,
            radial_derivatives,
            angular,
            angular_derivatives,
            radial_directions,
            edge_types,
            channel_radial_indices,
            channel_angular_indices,
            channel_types,
            scales,
            backend="native",
        )[0]

    assert torch.autograd.gradcheck(
        evaluate,
        (radial_values, angular_values, channel_scales),
        eps=1e-6,
        atol=2e-8,
        rtol=2e-6,
    )
    assert torch.autograd.gradgradcheck(
        evaluate,
        (radial_values, angular_values, channel_scales),
        eps=1e-6,
        atol=2e-7,
        rtol=2e-5,
    )
    result = torch.library.opcheck(
        torch.ops.ye3t_runtime.scheduled_radial_angular_channels_with_derivative,
        tuple(value.detach() for value in inputs),
    )
    assert all(item == "SUCCESS" for item in result.values())


@pytest.mark.parametrize("complex_values", [False, True])
def test_native_plain_site_basis_product_adjoint_matches_reference_and_hvp(
    monkeypatch,
    complex_values,
):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    dtype = torch.complex128 if complex_values else torch.float64
    group_count = 3
    edge_count = 4
    term_count = 6
    channel_count = 8
    data = (
        torch.randn(group_count, edge_count, dtype=dtype),
        torch.randn(group_count, edge_count, dtype=dtype),
        torch.randn(term_count, edge_count, dtype=dtype),
        torch.randn(term_count, edge_count, 3, dtype=dtype),
        torch.randn(group_count, edge_count, dtype=dtype),
        torch.randn(group_count, edge_count, dtype=dtype),
        torch.randn(group_count, edge_count, dtype=dtype),
        torch.randn(edge_count, 3, dtype=dtype),
        torch.randn(edge_count, dtype=dtype),
        torch.randn(edge_count, 3, dtype=dtype),
        torch.randn(edge_count, channel_count, dtype=dtype),
    )
    native_data = tuple(
        value.detach().clone().requires_grad_(True)
        for value in data
    )
    reference_data = tuple(
        value.detach().clone().requires_grad_(True)
        for value in data
    )
    term_groups = torch.tensor(
        [0, 0, 1, 1, 2, 2],
        dtype=torch.int64,
    )
    term_channels = torch.tensor(
        [7, 1, 5, 0, 4, 2],
        dtype=torch.int64,
    )

    def evaluate(values, backend):
        return plain_site_basis_product_adjoint(
            *values[:10],
            term_groups,
            term_channels,
            values[10],
            backend=backend,
        )

    actual = evaluate(native_data, "native")
    expected = evaluate(reference_data, "reference")
    for actual_value, expected_value in zip(actual, expected):
        torch.testing.assert_close(
            actual_value,
            expected_value,
            rtol=2e-13,
            atol=2e-13,
        )

    output_adjoints = tuple(
        torch.randn_like(value)
        for value in actual
    )
    native_gradients = torch.autograd.grad(
        actual,
        native_data,
        output_adjoints,
        create_graph=True,
    )
    reference_gradients = torch.autograd.grad(
        expected,
        reference_data,
        output_adjoints,
        create_graph=True,
    )
    for actual_value, expected_value in zip(
        native_gradients,
        reference_gradients,
    ):
        torch.testing.assert_close(
            actual_value,
            expected_value,
            rtol=2e-12,
            atol=2e-12,
        )
    gradient_adjoints = tuple(
        torch.randn_like(value)
        for value in native_gradients
    )
    native_hvp = torch.autograd.grad(
        native_gradients,
        native_data,
        gradient_adjoints,
    )
    reference_hvp = torch.autograd.grad(
        reference_gradients,
        reference_data,
        gradient_adjoints,
    )
    for actual_value, expected_value in zip(native_hvp, reference_hvp):
        torch.testing.assert_close(
            actual_value,
            expected_value,
            rtol=2e-11,
            atol=2e-11,
        )
    arguments = (
        *tuple(value.detach() for value in native_data[:10]),
        term_groups,
        term_channels,
        native_data[10].detach(),
    )
    result = torch.library.opcheck(
        torch.ops.ye3t_runtime.plain_site_basis_product_adjoint,
        arguments,
    )
    assert all(item == "SUCCESS" for item in result.values())


@pytest.mark.parametrize("complex_values", [False, True])
def test_native_density_accumulation_matches_reference_and_derivatives(
    monkeypatch,
    complex_values,
):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    dtype = torch.complex128 if complex_values else torch.float64
    edge_values = torch.randn(
        9,
        7,
        dtype=dtype,
        requires_grad=True,
    )
    centers = torch.tensor(
        [0, 0, 2, 1, 2, 2, 3, 0, 3],
        dtype=torch.int64,
    )
    actual = density_accumulate(
        edge_values,
        centers,
        4,
        backend="native",
    )
    expected = density_accumulate(
        edge_values,
        centers,
        4,
        backend="reference",
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)

    def evaluate(values):
        return density_accumulate(
            values,
            centers,
            4,
            backend="native",
        )

    assert torch.autograd.gradcheck(
        evaluate,
        (edge_values,),
        eps=1e-6,
        atol=1e-10,
        rtol=1e-8,
    )
    assert torch.autograd.gradgradcheck(
        evaluate,
        (edge_values,),
        eps=1e-6,
        atol=1e-10,
        rtol=1e-8,
    )


def test_native_density_accumulation_passes_opcheck(monkeypatch):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    edge_values = torch.randn(
        9,
        7,
        dtype=torch.float64,
        requires_grad=True,
    )
    centers = torch.tensor(
        [0, 0, 2, 1, 2, 2, 3, 0, 3],
        dtype=torch.int64,
    )
    density_accumulate(edge_values, centers, 4, backend="native")
    result = torch.library.opcheck(
        torch.ops.ye3t_runtime.density_accumulate.default,
        (edge_values, centers, 4),
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in result.values()), result


def test_native_edge_outer_accumulation_matches_reference_and_derivatives(
):
    generator = torch.Generator().manual_seed(211)
    left = torch.randn(
        9,
        3,
        dtype=torch.float64,
        generator=generator,
        requires_grad=True,
    )
    right = torch.randn(
        9,
        5,
        dtype=torch.float64,
        generator=generator,
        requires_grad=True,
    )
    centers = torch.tensor(
        [0, 0, 2, 1, 2, 2, 3, 0, 3],
        dtype=torch.int64,
    )
    actual = edge_outer_accumulate(
        left,
        right,
        centers,
        4,
        backend="native",
    )
    expected = edge_outer_accumulate(
        left,
        right,
        centers,
        4,
        backend="reference",
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)

    def evaluate(left_value, right_value):
        return edge_outer_accumulate(
            left_value,
            right_value,
            centers,
            4,
            backend="native",
        )

    assert torch.autograd.gradcheck(
        evaluate,
        (left, right),
        eps=1e-6,
        atol=1e-10,
        rtol=1e-8,
    )
    grad_output = torch.randn(
        actual.shape,
        dtype=actual.dtype,
        generator=generator,
    )
    assert torch.autograd.gradgradcheck(
        evaluate,
        (left, right),
        grad_outputs=grad_output,
        eps=1e-6,
        atol=1e-10,
        rtol=1e-8,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_native_softmax_gaussian_role_density_values_vjp_hvp_and_opcheck(
    device,
):
    capabilities = native_execution_plan_capabilities()
    if device == "cuda" and (
        not torch.cuda.is_available() or not capabilities["cuda"]
    ):
        pytest.skip("requires native CUDA")
    generator = torch.Generator().manual_seed(229)
    distance_values = 0.4 + torch.rand(
        11,
        dtype=torch.float64,
        generator=generator,
    )
    edge_data = torch.randn(
        11,
        5,
        dtype=torch.float64,
        generator=generator,
    )
    cutoffs = torch.linspace(2.5, 4.0, 11, dtype=torch.float64).to(device)
    filter_centers = torch.tensor(
        (0.15, 0.5, 0.85),
        dtype=torch.float64,
        device=device,
    )
    atom_centers = torch.tensor(
        (0, 1, 2, 1, 3, 0, 2, 3, 1, 0, 2),
        dtype=torch.int64,
        device=device,
    )
    native_distances = distance_values.to(device).requires_grad_(True)
    native_edges = edge_data.to(device).requires_grad_(True)
    reference_distances = distance_values.to(device).requires_grad_(True)
    reference_edges = edge_data.to(device).requires_grad_(True)
    native = softmax_gaussian_role_density(
        native_distances,
        cutoffs,
        filter_centers,
        0.23,
        native_edges,
        atom_centers,
        4,
        backend="native",
    )
    reference = softmax_gaussian_role_density(
        reference_distances,
        cutoffs,
        filter_centers,
        0.23,
        reference_edges,
        atom_centers,
        4,
        backend="reference",
    )
    torch.testing.assert_close(native, reference, rtol=1e-13, atol=1e-13)
    weights = torch.linspace(
        -0.7,
        0.9,
        native.numel(),
        dtype=native.dtype,
        device=device,
    ).reshape(native.shape)
    native_gradient = torch.autograd.grad(
        (native.square() * weights).sum(),
        (native_distances, native_edges),
        create_graph=True,
    )
    reference_gradient = torch.autograd.grad(
        (reference.square() * weights).sum(),
        (reference_distances, reference_edges),
        create_graph=True,
    )
    for actual, expected in zip(native_gradient, reference_gradient):
        torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-11)
    distance_direction = torch.linspace(
        -0.3,
        0.4,
        native_distances.numel(),
        dtype=native.dtype,
        device=device,
    )
    edge_direction = torch.linspace(
        -0.2,
        0.5,
        native_edges.numel(),
        dtype=native.dtype,
        device=device,
    ).reshape(native_edges.shape)
    native_hvp = torch.autograd.grad(
        (native_gradient[0] * distance_direction).sum()
        + (native_gradient[1] * edge_direction).sum(),
        (native_distances, native_edges),
    )
    reference_hvp = torch.autograd.grad(
        (reference_gradient[0] * distance_direction).sum()
        + (reference_gradient[1] * edge_direction).sum(),
        (reference_distances, reference_edges),
    )
    for actual, expected in zip(native_hvp, reference_hvp):
        torch.testing.assert_close(actual, expected, rtol=4e-10, atol=4e-10)
    result = torch.library.opcheck(
        torch.ops.ye3t_runtime.softmax_gaussian_role_density.default,
        (
            native_distances.detach().requires_grad_(True),
            cutoffs,
            filter_centers,
            0.23,
            native_edges.detach().requires_grad_(True),
            atom_centers,
            4,
        ),
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in result.values()), result


def test_native_scheduled_role_density_cpu_values_gradients_and_hvp():
    generator = torch.Generator().manual_seed(230)
    radial = torch.randn(
        9,
        4,
        dtype=torch.float64,
        generator=generator,
        requires_grad=True,
    )
    angular = torch.randn(
        9,
        7,
        dtype=torch.float64,
        generator=generator,
        requires_grad=True,
    )
    distances = (
        0.3
        + torch.rand(9, dtype=torch.float64, generator=generator)
    ).requires_grad_(True)
    soft_weights = torch.randn(
        9,
        dtype=torch.float64,
        generator=generator,
        requires_grad=True,
    )
    cutoffs = torch.linspace(2.4, 3.6, 9, dtype=torch.float64)
    filter_centers = torch.tensor(
        (0.12, 0.48, 0.84),
        dtype=torch.float64,
    )
    edge_types = torch.tensor(
        (0, 1, 2, 1, 0, 2, 1, 0, 2),
        dtype=torch.long,
    )
    radial_indices = torch.tensor(
        (1, 2, 1, 3, 2, 1),
        dtype=torch.long,
    )
    angular_indices = torch.tensor(
        (0, 3, 5, 3, 6, 5),
        dtype=torch.long,
    )
    channel_types = torch.tensor(
        (-1, 1, 2, -1, 0, 1),
        dtype=torch.long,
    )
    channel_scales = torch.tensor(
        (0.8, -0.4, 1.1, 0.6, -0.7, 0.3),
        dtype=torch.float64,
    )
    atom_centers = torch.tensor(
        (0, 1, 2, 1, 3, 0, 2, 3, 1),
        dtype=torch.long,
    )

    def evaluate(local_radial, local_angular, local_distances, local_soft):
        return scheduled_softmax_gaussian_role_density(
            local_radial,
            local_angular,
            local_distances,
            cutoffs,
            filter_centers,
            0.21,
            local_soft,
            edge_types,
            radial_indices,
            angular_indices,
            channel_types,
            channel_scales,
            atom_centers,
            4,
            backend="native",
        )

    native = evaluate(radial, angular, distances, soft_weights)
    reference = scheduled_softmax_gaussian_role_density(
        radial,
        angular,
        distances,
        cutoffs,
        filter_centers,
        0.21,
        soft_weights,
        edge_types,
        radial_indices,
        angular_indices,
        channel_types,
        channel_scales,
        atom_centers,
        4,
        backend="reference",
    )
    torch.testing.assert_close(native, reference, rtol=0.0, atol=2.0e-13)
    assert torch.autograd.gradcheck(
        evaluate,
        (radial, angular, distances, soft_weights),
        eps=1.0e-6,
        atol=3.0e-6,
        rtol=3.0e-5,
    )
    assert torch.autograd.gradgradcheck(
        evaluate,
        (radial, angular, distances, soft_weights),
        eps=1.0e-6,
        atol=5.0e-6,
        rtol=5.0e-5,
    )
    result = torch.library.opcheck(
        torch.ops.ye3t_runtime.scheduled_softmax_gaussian_role_density.default,
        (
            radial.detach(),
            angular.detach(),
            distances.detach(),
            cutoffs,
            filter_centers,
            0.21,
            soft_weights.detach(),
            edge_types,
            radial_indices,
            angular_indices,
            channel_types,
            channel_scales,
            atom_centers,
            4,
        ),
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in result.values()), result


def test_native_scheduled_role_density_cuda_values_vjp_hvp_opcheck_and_compile():
    capabilities = native_execution_plan_capabilities()
    if not torch.cuda.is_available() or not capabilities["cuda"]:
        pytest.skip("requires native CUDA")
    generator = torch.Generator().manual_seed(231)
    radial_values = torch.randn(
        13,
        5,
        dtype=torch.float64,
        generator=generator,
    )
    angular_values = torch.randn(
        13,
        9,
        dtype=torch.float64,
        generator=generator,
    )
    distances = 0.25 + torch.rand(
        13,
        dtype=torch.float64,
        generator=generator,
    )
    soft_weights = torch.randn(
        13,
        dtype=torch.float64,
        generator=generator,
    )
    cutoffs = torch.linspace(2.2, 4.1, 13, dtype=torch.float64).cuda()
    filter_centers = torch.tensor(
        (0.1, 0.4, 0.7, 0.9),
        dtype=torch.float64,
        device="cuda",
    )
    edge_types = torch.tensor(
        (0, 1, 2, 1, 0, 2, 1, 0, 2, 2, 1, 0, 1),
        dtype=torch.long,
        device="cuda",
    )
    radial_indices = torch.tensor(
        (1, 3, 2, 4, 1, 0, 3),
        dtype=torch.long,
        device="cuda",
    )
    angular_indices = torch.tensor(
        (0, 4, 7, 3, 8, 5, 2),
        dtype=torch.long,
        device="cuda",
    )
    channel_types = torch.tensor(
        (-1, 1, 2, -1, 0, 2, 1),
        dtype=torch.long,
        device="cuda",
    )
    channel_scales = torch.tensor(
        (0.8, -0.4, 1.1, 0.6, -0.7, 0.3, 0.2),
        dtype=torch.float64,
        device="cuda",
    )
    atom_centers = torch.tensor(
        (0, 1, 2, 1, 3, 0, 2, 3, 1, 0, 2, 3, 1),
        dtype=torch.long,
        device="cuda",
    )

    def evaluate(local_radial, local_angular, local_distances, local_soft, backend):
        return scheduled_softmax_gaussian_role_density(
            local_radial,
            local_angular,
            local_distances,
            cutoffs,
            filter_centers,
            0.19,
            local_soft,
            edge_types,
            radial_indices,
            angular_indices,
            channel_types,
            channel_scales,
            atom_centers,
            4,
            backend=backend,
        )

    native_inputs = tuple(
        value.cuda().requires_grad_(True)
        for value in (radial_values, angular_values, distances, soft_weights)
    )
    reference_inputs = tuple(
        value.cuda().requires_grad_(True)
        for value in (radial_values, angular_values, distances, soft_weights)
    )
    native = evaluate(*native_inputs, "native")
    reference = evaluate(*reference_inputs, "reference")
    torch.testing.assert_close(native, reference, rtol=2.0e-13, atol=2.0e-13)
    output_weight = torch.linspace(
        -0.8,
        0.9,
        native.numel(),
        dtype=native.dtype,
        device=native.device,
    ).reshape(native.shape)
    native_gradient = torch.autograd.grad(
        (native.square() * output_weight).sum(),
        native_inputs,
        create_graph=True,
    )
    reference_gradient = torch.autograd.grad(
        (reference.square() * output_weight).sum(),
        reference_inputs,
        create_graph=True,
    )
    for actual, expected in zip(native_gradient, reference_gradient):
        torch.testing.assert_close(actual, expected, rtol=3.0e-11, atol=3.0e-11)
    directions = tuple(
        torch.linspace(
            -0.3,
            0.4,
            value.numel(),
            dtype=value.dtype,
            device=value.device,
        ).reshape(value.shape)
        for value in native_inputs
    )
    native_hvp = torch.autograd.grad(
        sum((gradient * direction).sum() for gradient, direction in zip(
            native_gradient,
            directions,
        )),
        native_inputs,
    )
    reference_hvp = torch.autograd.grad(
        sum((gradient * direction).sum() for gradient, direction in zip(
            reference_gradient,
            directions,
        )),
        reference_inputs,
    )
    for actual, expected in zip(native_hvp, reference_hvp):
        torch.testing.assert_close(actual, expected, rtol=8.0e-10, atol=8.0e-10)

    raw_arguments = (
        *(value.detach() for value in native_inputs),
        cutoffs,
        filter_centers,
        0.19,
        edge_types,
        radial_indices,
        angular_indices,
        channel_types,
        channel_scales,
        atom_centers,
        4,
    )
    raw_arguments = (
        raw_arguments[0],
        raw_arguments[1],
        raw_arguments[2],
        raw_arguments[4],
        raw_arguments[5],
        raw_arguments[6],
        raw_arguments[3],
        *raw_arguments[7:],
    )
    result = torch.library.opcheck(
        torch.ops.ye3t_runtime.scheduled_softmax_gaussian_role_density.default,
        raw_arguments,
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in result.values()), result

    def raw_evaluate(local_radial, local_angular, local_distances, local_soft):
        return torch.ops.ye3t_runtime.scheduled_softmax_gaussian_role_density(
            local_radial,
            local_angular,
            local_distances,
            cutoffs,
            filter_centers,
            0.19,
            local_soft,
            edge_types,
            radial_indices,
            angular_indices,
            channel_types,
            channel_scales,
            atom_centers,
            4,
        )

    compiled = torch.compile(raw_evaluate, backend="eager", fullgraph=True)
    torch.testing.assert_close(
        compiled(*tuple(value.detach() for value in native_inputs)),
        native.detach(),
        rtol=0.0,
        atol=2.0e-13,
    )


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_native_scheduled_role_density_shared_coordinate_hvp(device):
    capabilities = native_execution_plan_capabilities()
    if device == "cuda" and (
        not torch.cuda.is_available() or not capabilities["cuda"]
    ):
        pytest.skip("requires native CUDA")
    cutoffs = torch.full(
        (7,),
        3.2,
        dtype=torch.float64,
        device=device,
    )
    filter_centers = torch.tensor(
        (0.2, 0.5, 0.8),
        dtype=torch.float64,
        device=device,
    )
    edge_types = torch.tensor(
        (0, 1, 0, 1, 1, 0, 1),
        dtype=torch.long,
        device=device,
    )
    radial_indices = torch.tensor(
        (0, 1, 2),
        dtype=torch.long,
        device=device,
    )
    angular_indices = torch.tensor(
        (1, 2, 0),
        dtype=torch.long,
        device=device,
    )
    channel_types = torch.tensor(
        (-1, 1, 0),
        dtype=torch.long,
        device=device,
    )
    channel_scales = torch.tensor(
        (0.7, -0.4, 1.2),
        dtype=torch.float64,
        device=device,
    )
    atom_centers = torch.tensor(
        (0, 1, 2, 0, 1, 2, 0),
        dtype=torch.long,
        device=device,
    )

    def evaluate(coordinate, backend):
        radial = torch.stack(
            (coordinate, coordinate.square(), torch.sin(coordinate)),
            dim=1,
        )
        angular = torch.stack(
            (torch.cos(coordinate), coordinate.pow(3), torch.exp(-coordinate)),
            dim=1,
        )
        return scheduled_softmax_gaussian_role_density(
            radial,
            angular,
            coordinate,
            cutoffs,
            filter_centers,
            0.22,
            torch.cos(0.4 * coordinate),
            edge_types,
            radial_indices,
            angular_indices,
            channel_types,
            channel_scales,
            atom_centers,
            3,
            backend=backend,
        )

    native_coordinate = torch.linspace(
        0.3,
        1.1,
        7,
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    reference_coordinate = native_coordinate.detach().requires_grad_(True)
    native_raw = evaluate(native_coordinate, "native")
    reference_raw = evaluate(reference_coordinate, "reference")
    native = native_raw[..., :-1] / native_raw[..., -1:].clamp_min(1.0e-8)
    reference = (
        reference_raw[..., :-1]
        / reference_raw[..., -1:].clamp_min(1.0e-8)
    )
    weight = torch.linspace(
        -0.6,
        0.8,
        native.numel(),
        dtype=native.dtype,
        device=device,
    ).reshape(native.shape)
    native_gradient = torch.autograd.grad(
        (native.square() * weight).sum(),
        native_coordinate,
        create_graph=True,
    )[0]
    reference_gradient = torch.autograd.grad(
        (reference.square() * weight).sum(),
        reference_coordinate,
        create_graph=True,
    )[0]
    direction = torch.linspace(
        -0.3,
        0.4,
        7,
        dtype=torch.float64,
        device=device,
    )
    native_hvp = torch.autograd.grad(
        (native_gradient * direction).sum(),
        native_coordinate,
    )[0]
    reference_hvp = torch.autograd.grad(
        (reference_gradient * direction).sum(),
        reference_coordinate,
    )[0]
    torch.testing.assert_close(
        native_gradient,
        reference_gradient,
        rtol=3.0e-11,
        atol=3.0e-11,
    )
    torch.testing.assert_close(
        native_hvp,
        reference_hvp,
        rtol=8.0e-10,
        atol=8.0e-10,
    )


def test_native_edge_outer_accumulation_empty_edges_and_opcheck(
):
    empty_left = torch.empty(
        0,
        3,
        dtype=torch.float64,
        requires_grad=True,
    )
    empty_right = torch.empty(
        0,
        5,
        dtype=torch.float64,
        requires_grad=True,
    )
    centers = torch.empty(0, dtype=torch.int64)
    for backend in ("reference", "auto", "native"):
        actual = edge_outer_accumulate(
            empty_left,
            empty_right,
            centers,
            4,
            backend=backend,
        )
        assert actual.shape == (4, 3, 5)
        torch.testing.assert_close(
            actual,
            torch.zeros_like(actual),
            rtol=0.0,
            atol=0.0,
        )

    left = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    right = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
    centers = torch.tensor([0, 1, 2, 1, 3, 0, 2], dtype=torch.int64)
    edge_outer_accumulate(left, right, centers, 4, backend="native")
    result = torch.library.opcheck(
        torch.ops.ye3t_runtime.edge_outer_accumulate.default,
        (left, right, centers, 4),
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in result.values()), result
def test_native_cuda_edge_outer_values_vjp_hvp_and_opcheck():
    capabilities = native_execution_plan_capabilities()
    if not torch.cuda.is_available() or not capabilities["cuda"]:
        pytest.skip("requires native CUDA")
    generator = torch.Generator().manual_seed(223)
    left_values = torch.randn(
        17,
        3,
        dtype=torch.float64,
        generator=generator,
    )
    right_values = torch.randn(
        17,
        6,
        dtype=torch.float64,
        generator=generator,
    )
    centers = torch.tensor(
        [0, 1, 2, 3, 4, 0, 2, 4, 1, 3, 3, 2, 0, 4, 1, 2, 3],
        dtype=torch.int64,
        device="cuda",
    )
    native_left = left_values.to("cuda").requires_grad_(True)
    native_right = right_values.to("cuda").requires_grad_(True)
    reference_left = left_values.to("cuda").requires_grad_(True)
    reference_right = right_values.to("cuda").requires_grad_(True)
    native = edge_outer_accumulate(
        native_left,
        native_right,
        centers,
        5,
        backend="native",
    )
    reference = edge_outer_accumulate(
        reference_left,
        reference_right,
        centers,
        5,
        backend="reference",
    )
    torch.testing.assert_close(native, reference, rtol=0.0, atol=1e-12)
    native_gradient = torch.autograd.grad(
        native.square().sum(),
        (native_left, native_right),
        create_graph=True,
    )
    reference_gradient = torch.autograd.grad(
        reference.square().sum(),
        (reference_left, reference_right),
        create_graph=True,
    )
    for actual, expected in zip(native_gradient, reference_gradient):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=2e-11,
            atol=2e-11,
        )
    left_direction = torch.linspace(
        -0.4,
        0.3,
        int(native_left.numel()),
        dtype=native_left.dtype,
        device=native_left.device,
    ).reshape_as(native_left)
    right_direction = torch.linspace(
        0.2,
        -0.35,
        int(native_right.numel()),
        dtype=native_right.dtype,
        device=native_right.device,
    ).reshape_as(native_right)
    native_hvp = torch.autograd.grad(
        (
            native_gradient[0] * left_direction
        ).sum()
        + (
            native_gradient[1] * right_direction
        ).sum(),
        (native_left, native_right),
    )
    reference_hvp = torch.autograd.grad(
        (
            reference_gradient[0] * left_direction
        ).sum()
        + (
            reference_gradient[1] * right_direction
        ).sum(),
        (reference_left, reference_right),
    )
    for actual, expected in zip(native_hvp, reference_hvp):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=2e-10,
            atol=2e-10,
        )
    result = torch.library.opcheck(
        torch.ops.ye3t_runtime.edge_outer_accumulate.default,
        (
            native_left.detach(),
            native_right.detach(),
            centers,
            5,
        ),
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in result.values()), result


def test_deterministic_cuda_source_and_graph_reductions_repeat_hvp_exactly():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(5071)
    radial_seed = torch.randn(
        12,
        4,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    angular_seed = torch.randn(
        12,
        5,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    distance_seed = torch.linspace(
        0.35,
        2.4,
        12,
        dtype=torch.float32,
        device=device,
    )
    soft_seed = torch.linspace(
        0.2,
        0.9,
        12,
        dtype=torch.float32,
        device=device,
    )
    cutoffs = torch.full((12,), 3.0, dtype=torch.float32, device=device)
    filter_centers = torch.tensor(
        (0.2, 0.5, 0.8),
        dtype=torch.float32,
        device=device,
    )
    edge_types = torch.tensor(
        (0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1),
        dtype=torch.long,
        device=device,
    )
    radial_indices = torch.tensor(
        (0, 1, 2, 3, 1, 2),
        dtype=torch.long,
        device=device,
    )
    angular_indices = torch.tensor(
        (0, 1, 2, 3, 4, 1),
        dtype=torch.long,
        device=device,
    )
    channel_types = torch.tensor(
        (-1, 1, 0, -1, 0, 1),
        dtype=torch.long,
        device=device,
    )
    channel_scales = torch.linspace(
        0.4,
        1.1,
        6,
        dtype=torch.float32,
        device=device,
    )
    atom_centers = torch.tensor(
        (0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3),
        dtype=torch.long,
        device=device,
    )
    graph_sources = torch.tensor(
        (0, 1, 2, 3, 0, 2, 1, 3),
        dtype=torch.long,
        device=device,
    )
    graph_targets = torch.tensor(
        (1, 2, 3, 0, 2, 0, 3, 1),
        dtype=torch.long,
        device=device,
    )
    graph_gates = torch.randn(
        8,
        3,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    direction = torch.randn(
        distance_seed.shape,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )

    def evaluate(backend):
        radial = radial_seed.detach().clone().requires_grad_(True)
        angular = angular_seed.detach().clone().requires_grad_(True)
        distances = distance_seed.detach().clone().requires_grad_(True)
        soft = soft_seed.detach().clone().requires_grad_(True)
        density = scheduled_softmax_gaussian_role_density(
            radial,
            angular,
            distances,
            cutoffs,
            filter_centers,
            0.24,
            soft,
            edge_types,
            radial_indices,
            angular_indices,
            channel_types,
            channel_scales,
            atom_centers,
            4,
            backend=backend,
        )
        nodes = density.reshape(4, -1)
        feature_channels = torch.arange(
            nodes.shape[1],
            dtype=torch.long,
            device=device,
        ) % graph_gates.shape[1]
        scattered = carrier_gated_scatter(
            nodes,
            graph_gates,
            graph_sources,
            graph_targets,
            feature_channels,
            4,
            backend=backend,
        )
        gradient = torch.autograd.grad(
            scattered.square().sum(),
            distances,
            create_graph=True,
        )[0]
        hvp = torch.autograd.grad(
            (gradient * direction).sum(),
            distances,
        )[0]
        return scattered.detach(), gradient.detach(), hvp.detach()

    deterministic = tuple(evaluate("deterministic") for _ in range(4))
    for repeated in deterministic[1:]:
        for actual, expected in zip(repeated, deterministic[0]):
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    reference = evaluate("reference")
    for actual, expected in zip(deterministic[0], reference):
        torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-5)


@pytest.mark.parametrize(
    "value_dtype,gate_dtype",
    (
        (torch.float64, torch.float64),
        (torch.complex128, torch.complex128),
        (torch.complex128, torch.float64),
    ),
)
def test_native_carrier_gated_scatter_cpu_values_gradients_and_hvp(
    value_dtype,
    gate_dtype,
):
    generator = torch.Generator().manual_seed(4201)
    node_values = torch.randn(
        5,
        8,
        dtype=value_dtype,
        generator=generator,
        requires_grad=True,
    )
    edge_gates = torch.randn(
        9,
        3,
        dtype=gate_dtype,
        generator=generator,
        requires_grad=True,
    )
    edge_sources = torch.tensor(
        (0, 1, 2, 3, 4, 0, 2, 4, 1),
        dtype=torch.long,
    )
    edge_targets = torch.tensor(
        (1, 2, 3, 4, 0, 3, 0, 2, 4),
        dtype=torch.long,
    )
    feature_channels = torch.tensor(
        (0, 0, 1, 1, 1, 2, 2, 2),
        dtype=torch.long,
    )
    actual = carrier_gated_scatter(
        node_values,
        edge_gates,
        edge_sources,
        edge_targets,
        feature_channels,
        5,
        backend="native",
    )
    expected = carrier_gated_scatter(
        node_values,
        edge_gates,
        edge_sources,
        edge_targets,
        feature_channels,
        5,
        backend="reference",
    )
    torch.testing.assert_close(
        actual,
        expected,
        rtol=0.0,
        atol=1.0e-12,
    )

    def evaluate(nodes, gates):
        return carrier_gated_scatter(
            nodes,
            gates,
            edge_sources,
            edge_targets,
            feature_channels,
            5,
            backend="native",
        )

    assert torch.autograd.gradcheck(
        evaluate,
        (node_values, edge_gates),
        eps=1.0e-6,
        atol=2.0e-6,
        rtol=2.0e-5,
    )
    assert torch.autograd.gradgradcheck(
        evaluate,
        (node_values, edge_gates),
        eps=1.0e-6,
        atol=3.0e-6,
        rtol=3.0e-5,
    )


@pytest.mark.parametrize(
    "value_dtype,gate_dtype",
    (
        (torch.float64, torch.float64),
        (torch.complex128, torch.float64),
    ),
)
def test_native_carrier_residual_gated_scatter_cpu_value_vjp_hvp(
    value_dtype,
    gate_dtype,
):
    generator = torch.Generator().manual_seed(4211)
    node_values = torch.randn(
        5,
        8,
        dtype=value_dtype,
        generator=generator,
        requires_grad=True,
    )
    edge_gates = torch.randn(
        9,
        3,
        dtype=gate_dtype,
        generator=generator,
        requires_grad=True,
    )
    edge_sources = torch.tensor(
        (0, 1, 2, 3, 4, 0, 2, 4, 1),
        dtype=torch.long,
    )
    edge_targets = torch.tensor(
        (1, 2, 3, 4, 0, 3, 0, 2, 4),
        dtype=torch.long,
    )
    feature_channels = torch.tensor(
        (0, 0, 1, 1, 1, 2, 2, 2),
        dtype=torch.long,
    )

    def evaluate(nodes, gates):
        return carrier_residual_gated_scatter(
            nodes,
            gates,
            edge_sources,
            edge_targets,
            feature_channels,
            backend="native",
        )

    actual = evaluate(node_values, edge_gates)
    expected = node_values + carrier_gated_scatter(
        node_values,
        edge_gates,
        edge_sources,
        edge_targets,
        feature_channels,
        5,
        backend="reference",
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-12)
    assert torch.autograd.gradcheck(
        evaluate,
        (node_values, edge_gates),
        eps=1.0e-6,
        atol=2.0e-6,
        rtol=2.0e-5,
    )
    assert torch.autograd.gradgradcheck(
        evaluate,
        (node_values, edge_gates),
        eps=1.0e-6,
        atol=3.0e-6,
        rtol=3.0e-5,
    )
    adjoints = carrier_residual_gated_scatter_adjoint(
        torch.ones_like(actual),
        node_values.detach(),
        edge_gates.detach(),
        edge_sources,
        edge_targets,
        feature_channels,
        backend="native",
    )
    assert adjoints[0].shape == node_values.shape
    assert adjoints[1].shape == edge_gates.shape


@pytest.mark.parametrize(
    "value_dtype,gate_dtype",
    (
        (torch.float64, torch.float64),
        (torch.complex128, torch.float64),
    ),
)
def test_native_segmented_carrier_scatter_cpu_value_vjp_hvp(
    value_dtype,
    gate_dtype,
):
    generator = torch.Generator().manual_seed(4213)
    nodes = torch.randn(
        6,
        9,
        dtype=value_dtype,
        generator=generator,
        requires_grad=True,
    )
    gates = torch.randn(
        11,
        4,
        dtype=gate_dtype,
        generator=generator,
        requires_grad=True,
    )
    sources = torch.tensor(
        (5, 1, 4, 0, 3, 2, 1, 5, 0, 4, 2),
        dtype=torch.long,
    )
    targets = torch.tensor(
        (2, 5, 1, 3, 0, 5, 2, 1, 4, 3, 0),
        dtype=torch.long,
    )
    feature_channels = torch.tensor(
        (2, 0, 3, 1, 1, 0, 3, 2, 1),
        dtype=torch.long,
    )
    segments = prepare_carrier_scatter_segments(
        sources,
        targets,
        feature_channels,
        6,
        4,
    )
    ordered_gates = gates.index_select(0, segments["edge_order"])

    def evaluate(node_values, edge_gates):
        return carrier_segmented_residual_gated_scatter(
            node_values,
            edge_gates,
            segments,
            backend="native",
        )

    actual = evaluate(nodes, ordered_gates)
    expected = carrier_residual_gated_scatter(
        nodes,
        ordered_gates,
        segments["edge_sources"],
        segments["edge_targets"],
        feature_channels,
        backend="reference",
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-12)
    assert torch.autograd.gradcheck(
        evaluate,
        (nodes, ordered_gates),
        eps=1.0e-6,
        atol=2.0e-6,
        rtol=2.0e-5,
    )
    assert torch.autograd.gradgradcheck(
        evaluate,
        (nodes, ordered_gates),
        eps=1.0e-6,
        atol=3.0e-6,
        rtol=3.0e-5,
    )
    adjoints = carrier_segmented_residual_gated_scatter_adjoint(
        torch.ones_like(actual),
        nodes.detach(),
        ordered_gates.detach(),
        segments,
        backend="native",
    )
    assert adjoints[0].shape == nodes.shape
    assert adjoints[1].shape == ordered_gates.shape


def test_native_segmented_carrier_scatter_rejects_corrupt_segments():
    nodes = torch.randn(4, 6, dtype=torch.float64)
    sources = torch.tensor((0, 1, 2), dtype=torch.long)
    targets = torch.tensor((1, 2, 3), dtype=torch.long)
    channels = torch.tensor((0, 0, 0, 1, 1, 1), dtype=torch.long)
    segments = prepare_carrier_scatter_segments(
        sources,
        targets,
        channels,
        4,
        2,
    )
    segments = dict(segments)
    segments["source_edges"] = torch.tensor((0, 0, 2), dtype=torch.long)
    gates = torch.randn(3, 2, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="source_edges must be a permutation"):
        carrier_segmented_residual_gated_scatter(
            nodes,
            gates,
            segments,
            backend="native",
        )


def test_native_carrier_gated_scatter_empty_and_opcheck():
    nodes = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
    gates = torch.empty(
        0,
        2,
        dtype=torch.float64,
        requires_grad=True,
    )
    empty = torch.empty(0, dtype=torch.long)
    feature_channels = torch.tensor(
        (0, 0, 0, 1, 1, 1),
        dtype=torch.long,
    )
    for backend in ("reference", "auto", "native"):
        output = carrier_gated_scatter(
            nodes,
            gates,
            empty,
            empty,
            feature_channels,
            4,
            backend=backend,
        )
        torch.testing.assert_close(
            output,
            torch.zeros_like(output),
            rtol=0.0,
            atol=0.0,
        )
    empty_segments = prepare_carrier_scatter_segments(
        empty,
        empty,
        feature_channels,
        4,
        2,
    )
    segmented = carrier_segmented_residual_gated_scatter(
        nodes,
        gates,
        empty_segments,
        backend="native",
    )
    torch.testing.assert_close(segmented, nodes, rtol=0.0, atol=0.0)

    sources = torch.tensor((0, 1, 2, 3, 0), dtype=torch.long)
    targets = torch.tensor((1, 2, 3, 0, 2), dtype=torch.long)
    populated_gates = torch.randn(
        5,
        2,
        dtype=torch.float64,
        requires_grad=True,
    )
    result = torch.library.opcheck(
        torch.ops.ye3t_runtime.carrier_gated_scatter.default,
        (
            nodes,
            populated_gates,
            sources,
            targets,
            feature_channels,
            4,
        ),
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in result.values()), result
    compiled = torch.compile(
        lambda node_values, edge_values: (
            torch.ops.ye3t_runtime.carrier_gated_scatter(
                node_values,
                edge_values,
                sources,
                targets,
                feature_channels,
                4,
            )
        ),
        backend="eager",
        fullgraph=True,
    )
    torch.testing.assert_close(
        compiled(nodes, populated_gates),
        carrier_gated_scatter(
            nodes,
            populated_gates,
            sources,
            targets,
            feature_channels,
            4,
            backend="native",
        ),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "value_dtype,gate_dtype",
    (
        (torch.float64, torch.float64),
        (torch.complex128, torch.complex128),
        (torch.complex128, torch.float64),
    ),
)
def test_native_cuda_carrier_gated_scatter_values_vjp_hvp(
    value_dtype,
    gate_dtype,
):
    capabilities = native_execution_plan_capabilities()
    if not capabilities["cuda"]:
        pytest.skip("requires the native CUDA execution-plan extension")
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(4202)
    nodes_native = torch.randn(
        6,
        10,
        dtype=value_dtype,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    gates_native = torch.randn(
        13,
        3,
        dtype=gate_dtype,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    nodes_reference = nodes_native.detach().clone().requires_grad_(True)
    gates_reference = gates_native.detach().clone().requires_grad_(True)
    sources = torch.tensor(
        (0, 1, 2, 3, 4, 5, 0, 2, 4, 1, 3, 5, 2),
        dtype=torch.long,
        device=device,
    )
    targets = torch.tensor(
        (1, 2, 3, 4, 5, 0, 3, 4, 0, 5, 1, 2, 0),
        dtype=torch.long,
        device=device,
    )
    feature_channels = torch.tensor(
        (0, 0, 0, 1, 1, 1, 2, 2, 2, 2),
        dtype=torch.long,
        device=device,
    )
    native = carrier_gated_scatter(
        nodes_native,
        gates_native,
        sources,
        targets,
        feature_channels,
        6,
        backend="native",
    )
    reference = carrier_gated_scatter(
        nodes_reference,
        gates_reference,
        sources,
        targets,
        feature_channels,
        6,
        backend="reference",
    )
    torch.testing.assert_close(
        native,
        reference,
        rtol=0.0,
        atol=2.0e-12,
    )
    native_loss = native.abs().square().sum()
    reference_loss = reference.abs().square().sum()
    native_gradients = torch.autograd.grad(
        native_loss,
        (nodes_native, gates_native),
        create_graph=True,
    )
    reference_gradients = torch.autograd.grad(
        reference_loss,
        (nodes_reference, gates_reference),
        create_graph=True,
    )
    for actual, expected in zip(
        native_gradients,
        reference_gradients,
    ):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=2.0e-11,
            atol=2.0e-11,
        )
    node_direction = torch.randn(
        nodes_native.shape,
        dtype=value_dtype,
        device=device,
        generator=generator,
    )
    gate_direction = torch.randn(
        gates_native.shape,
        dtype=gate_dtype,
        device=device,
        generator=generator,
    )
    native_hvp = torch.autograd.grad(
        (
            native_gradients[0].conj() * node_direction
        ).real.sum()
        + (
            native_gradients[1].conj() * gate_direction
        ).real.sum(),
        (nodes_native, gates_native),
    )
    reference_hvp = torch.autograd.grad(
        (
            reference_gradients[0].conj() * node_direction
        ).real.sum()
        + (
            reference_gradients[1].conj() * gate_direction
        ).real.sum(),
        (nodes_reference, gates_reference),
    )
    for actual, expected in zip(native_hvp, reference_hvp):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=4.0e-10,
            atol=4.0e-10,
        )

    node_adjoint, gate_adjoint = carrier_gated_scatter_adjoint(
        torch.ones_like(native),
        nodes_native.detach(),
        gates_native.detach(),
        sources,
        targets,
        feature_channels,
        backend="native",
    )
    assert node_adjoint.shape == nodes_native.shape
    assert gate_adjoint.shape == gates_native.shape
    if (
        value_dtype == torch.float64
        or gate_dtype != value_dtype
    ):
        result = torch.library.opcheck(
            torch.ops.ye3t_runtime.carrier_gated_scatter.default,
            (
                nodes_native.detach(),
                gates_native.detach(),
                sources,
                targets,
                feature_channels,
                6,
            ),
            raise_exception=False,
        )
        assert all(
            value == "SUCCESS" for value in result.values()
        ), result


@pytest.mark.parametrize(
    "value_dtype,gate_dtype",
    (
        (torch.float32, torch.float32),
        (torch.complex64, torch.float32),
    ),
)
def test_native_cuda_carrier_residual_gated_scatter_value_vjp_hvp(
    value_dtype,
    gate_dtype,
):
    capabilities = native_execution_plan_capabilities()
    if not capabilities["cuda"]:
        pytest.skip("requires the native CUDA execution-plan extension")
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(4212)
    nodes_native = torch.randn(
        32,
        96,
        dtype=value_dtype,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    gates_native = torch.randn(
        192,
        12,
        dtype=gate_dtype,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    nodes_reference = nodes_native.detach().clone().requires_grad_(True)
    gates_reference = gates_native.detach().clone().requires_grad_(True)
    sources = torch.randint(
        0,
        32,
        (192,),
        dtype=torch.long,
        device=device,
        generator=generator,
    )
    targets = torch.arange(
        32,
        dtype=torch.long,
        device=device,
    ).repeat_interleave(6)
    feature_channels = torch.arange(
        96,
        dtype=torch.long,
        device=device,
    ) % 12

    def evaluate(nodes, gates, backend):
        return carrier_residual_gated_scatter(
            nodes,
            gates,
            sources,
            targets,
            feature_channels,
            backend=backend,
        )

    native = evaluate(nodes_native, gates_native, "native")
    reference = evaluate(nodes_reference, gates_reference, "reference")
    torch.testing.assert_close(native, reference, rtol=2.0e-5, atol=2.0e-5)
    native_gradients = torch.autograd.grad(
        _squared_norm(native),
        (nodes_native, gates_native),
        create_graph=True,
    )
    reference_gradients = torch.autograd.grad(
        _squared_norm(reference),
        (nodes_reference, gates_reference),
        create_graph=True,
    )
    for actual, expected in zip(native_gradients, reference_gradients):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=6.0e-5,
            atol=6.0e-5,
        )
    node_direction = torch.randn(
        nodes_native.shape,
        dtype=value_dtype,
        device=device,
        generator=generator,
    )
    gate_direction = torch.randn(
        gates_native.shape,
        dtype=gate_dtype,
        device=device,
        generator=generator,
    )
    native_hvp = torch.autograd.grad(
        (
            native_gradients[0].conj() * node_direction
        ).real.sum()
        + (
            native_gradients[1].conj() * gate_direction
        ).real.sum(),
        (nodes_native, gates_native),
    )
    reference_hvp = torch.autograd.grad(
        (
            reference_gradients[0].conj() * node_direction
        ).real.sum()
        + (
            reference_gradients[1].conj() * gate_direction
        ).real.sum(),
        (nodes_reference, gates_reference),
    )
    for actual, expected in zip(native_hvp, reference_hvp):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=1.0e-4,
            atol=1.0e-4,
        )

    result = torch.library.opcheck(
        torch.ops.ye3t_runtime.carrier_residual_gated_scatter.default,
        (
            nodes_native.detach(),
            gates_native.detach(),
            sources,
            targets,
            feature_channels,
        ),
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in result.values()), result


@pytest.mark.parametrize(
    "value_dtype,gate_dtype",
    (
        (torch.float32, torch.float32),
        (torch.complex64, torch.float32),
    ),
)
def test_native_cuda_segmented_carrier_scatter_value_vjp_hvp(
    value_dtype,
    gate_dtype,
):
    capabilities = native_execution_plan_capabilities()
    if not capabilities["cuda"]:
        pytest.skip("requires the native CUDA execution-plan extension")
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(4214)
    sources = torch.randint(
        0,
        32,
        (384,),
        dtype=torch.long,
        device=device,
        generator=generator,
    )
    targets = torch.randint(
        0,
        32,
        (384,),
        dtype=torch.long,
        device=device,
        generator=generator,
    )
    feature_channels = torch.arange(
        128,
        dtype=torch.long,
        device=device,
    ) % 16
    segments = prepare_carrier_scatter_segments(
        sources,
        targets,
        feature_channels,
        32,
        16,
    )
    nodes_native = torch.randn(
        32,
        128,
        dtype=value_dtype,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    raw_gates = torch.randn(
        384,
        16,
        dtype=gate_dtype,
        device=device,
        generator=generator,
    )
    gates_native = raw_gates.index_select(
        0,
        segments["edge_order"],
    ).detach().requires_grad_(True)
    nodes_reference = nodes_native.detach().clone().requires_grad_(True)
    gates_reference = gates_native.detach().clone().requires_grad_(True)

    native = carrier_segmented_residual_gated_scatter(
        nodes_native,
        gates_native,
        segments,
        backend="native",
    )
    reference = carrier_residual_gated_scatter(
        nodes_reference,
        gates_reference,
        segments["edge_sources"],
        segments["edge_targets"],
        feature_channels,
        backend="reference",
    )
    torch.testing.assert_close(native, reference, rtol=2.0e-5, atol=2.0e-5)
    native_gradients = torch.autograd.grad(
        _squared_norm(native),
        (nodes_native, gates_native),
        create_graph=True,
    )
    reference_gradients = torch.autograd.grad(
        _squared_norm(reference),
        (nodes_reference, gates_reference),
        create_graph=True,
    )
    for actual, expected in zip(native_gradients, reference_gradients):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=6.0e-5,
            atol=6.0e-5,
        )
    node_direction = torch.randn(
        nodes_native.shape,
        dtype=value_dtype,
        device=device,
        generator=generator,
    )
    gate_direction = torch.randn(
        gates_native.shape,
        dtype=gate_dtype,
        device=device,
        generator=generator,
    )
    native_hvp = torch.autograd.grad(
        (native_gradients[0].conj() * node_direction).real.sum()
        + (native_gradients[1].conj() * gate_direction).real.sum(),
        (nodes_native, gates_native),
    )
    reference_hvp = torch.autograd.grad(
        (reference_gradients[0].conj() * node_direction).real.sum()
        + (reference_gradients[1].conj() * gate_direction).real.sum(),
        (nodes_reference, gates_reference),
    )
    for actual, expected in zip(native_hvp, reference_hvp):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=1.0e-4,
            atol=1.0e-4,
        )
    operator = (
        torch.ops.ye3t_runtime
        .carrier_segmented_residual_gated_scatter.default
    )
    result = torch.library.opcheck(
        operator,
        (
            nodes_native.detach(),
            gates_native.detach(),
            segments["edge_sources"],
            segments["edge_targets"],
            segments["target_offsets"],
            segments["source_offsets"],
            segments["source_edges"],
            feature_channels,
            segments["channel_offsets"],
            segments["channel_features"],
        ),
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in result.values()), result


def _carrier_channel_update_case(
    device,
    dtype,
    batch_size=3,
    control_dtype=None,
):
    if control_dtype is None:
        control_dtype = dtype
    generator = torch.Generator(device=device).manual_seed(4301)
    values = torch.randn(
        int(batch_size),
        24,
        dtype=dtype,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    gates = torch.randn(
        int(batch_size),
        5,
        dtype=control_dtype,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    channel_maps = torch.randn(
        13,
        dtype=control_dtype,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    feature_offsets = torch.tensor(
        (0, 12, 24),
        dtype=torch.long,
        device=device,
    )
    channel_offsets = torch.tensor(
        (0, 2, 5),
        dtype=torch.long,
        device=device,
    )
    map_offsets = torch.tensor(
        (0, 4, 13),
        dtype=torch.long,
        device=device,
    )
    return (
        values,
        gates,
        channel_maps,
        feature_offsets,
        channel_offsets,
        map_offsets,
    )


def _carrier_channel_transform_case(device, dtype, control_dtype=None):
    if control_dtype is None:
        control_dtype = dtype
    generator = torch.Generator(device=device).manual_seed(4351)
    values = torch.randn(
        3,
        30,
        dtype=dtype,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    channel_maps = torch.randn(
        9,
        dtype=control_dtype,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    offsets = tuple(
        torch.tensor(items, dtype=torch.long, device=device)
        for items in (
            (0, 12, 30),
            (0, 8, 14),
            (0, 3, 6),
            (0, 2, 3),
            (0, 6, 9),
        )
    )
    return (values, channel_maps, *offsets)


def _carrier_role_channel_map_adjoint_case(
    device,
    dtype,
    control_dtype=None,
):
    if control_dtype is None:
        control_dtype = dtype
    base = _carrier_channel_transform_case(device, dtype, control_dtype)
    generator = torch.Generator(device=device).manual_seed(7731)
    edge_values = torch.randn(
        7,
        30,
        dtype=dtype,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    real_dtype = edge_values.real.dtype if edge_values.is_complex() else dtype
    role_weights = torch.randn(
        7,
        3,
        dtype=real_dtype,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    atomic_output_adjoint = torch.randn(
        4,
        3,
        14,
        dtype=dtype,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    atom_centers = torch.tensor(
        (0, 1, 0, 3, 2, 1, 3),
        dtype=torch.long,
        device=device,
    )
    return (
        edge_values,
        role_weights,
        atomic_output_adjoint,
        atom_centers,
        base[1],
        *base[2:],
    )


@pytest.mark.parametrize(
    ("dtype", "control_dtype"),
    (
        (torch.float64, torch.float64),
        (torch.complex128, torch.complex128),
        (torch.complex128, torch.float64),
    ),
)
def test_native_carrier_role_channel_map_adjoint_cpu_value_vjp_and_hvp(
    dtype,
    control_dtype,
):
    case = _carrier_role_channel_map_adjoint_case(
        torch.device("cpu"), dtype, control_dtype
    )
    actual = carrier_role_channel_map_adjoint(*case, backend="native")
    expected = carrier_role_channel_map_adjoint(*case, backend="reference")
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=3.0e-12)

    def evaluate(edge_values, role_weights, atomic_output_adjoint):
        return carrier_role_channel_map_adjoint(
            edge_values,
            role_weights,
            atomic_output_adjoint,
            *case[3:],
            backend="native",
        )

    assert torch.autograd.gradcheck(
        evaluate,
        case[:3],
        eps=1.0e-6,
        atol=4.0e-6,
        rtol=4.0e-5,
    )
    assert torch.autograd.gradgradcheck(
        evaluate,
        case[:3],
        eps=1.0e-6,
        atol=5.0e-6,
        rtol=5.0e-5,
    )


def test_native_carrier_role_channel_map_adjoint_opcheck_and_compile():
    case = _carrier_role_channel_map_adjoint_case(
        torch.device("cpu"), torch.float64
    )
    result = torch.library.opcheck(
        torch.ops.ye3t_runtime.carrier_role_channel_map_adjoint.default,
        tuple(value.detach() for value in case),
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in result.values()), result
    fixed = case[3:]
    compiled = torch.compile(
        lambda edge_values, role_weights, atomic_output_adjoint: (
            torch.ops.ye3t_runtime.carrier_role_channel_map_adjoint(
                edge_values,
                role_weights,
                atomic_output_adjoint,
                *fixed,
            )
        ),
        backend="eager",
        fullgraph=True,
    )
    torch.testing.assert_close(
        compiled(*case[:3]),
        carrier_role_channel_map_adjoint(*case, backend="native"),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("dtype", "control_dtype"),
    (
        (torch.float64, torch.float64),
        (torch.complex128, torch.complex128),
        (torch.complex128, torch.float64),
    ),
)
def test_reference_carrier_channel_transform_values_vjp_and_hvp(
    dtype,
    control_dtype,
):
    case = _carrier_channel_transform_case(
        torch.device("cpu"), dtype, control_dtype
    )
    values, maps = case[:2]
    transformed = carrier_channel_transform(*case, backend="reference")
    assert transformed.shape == (3, 14)
    probe = torch.randn_like(transformed)
    direct = carrier_channel_transform_adjoint(
        probe, *case, backend="reference"
    )
    expected = torch.autograd.grad(
        transformed,
        (values, maps),
        probe,
        create_graph=True,
    )
    for actual, reference in zip(direct, expected):
        torch.testing.assert_close(
            actual,
            reference,
            rtol=2.0e-12,
            atol=2.0e-12,
        )

    def evaluate(local_values, local_maps):
        return carrier_channel_transform(
            local_values,
            local_maps,
            *case[2:],
            backend="reference",
        )

    assert torch.autograd.gradcheck(
        evaluate,
        (values, maps),
        eps=1.0e-6,
        atol=3.0e-6,
        rtol=3.0e-5,
    )
    assert torch.autograd.gradgradcheck(
        evaluate,
        (values, maps),
        eps=1.0e-6,
        atol=4.0e-6,
        rtol=4.0e-5,
    )


def test_reference_carrier_channel_transform_full_rank_identity():
    values = torch.randn(4, 18, dtype=torch.float64)
    maps = torch.eye(3, dtype=torch.float64).reshape(-1)
    offsets = (
        torch.tensor((0, 18), dtype=torch.long),
        torch.tensor((0, 18), dtype=torch.long),
        torch.tensor((0, 3), dtype=torch.long),
        torch.tensor((0, 3), dtype=torch.long),
        torch.tensor((0, 9), dtype=torch.long),
    )
    actual = carrier_channel_transform(
        values, maps, *offsets, backend="reference"
    )
    torch.testing.assert_close(actual, values, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("dtype", "control_dtype"),
    (
        (torch.float64, torch.float64),
        (torch.complex128, torch.complex128),
        (torch.complex128, torch.float64),
    ),
)
def test_native_carrier_channel_transform_cpu_values_gradients_and_hvp(
    dtype,
    control_dtype,
):
    case = _carrier_channel_transform_case(
        torch.device("cpu"), dtype, control_dtype
    )
    values, maps = case[:2]
    actual = carrier_channel_transform(*case, backend="native")
    expected = carrier_channel_transform(*case, backend="reference")
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=2.0e-12)

    def evaluate(local_values, local_maps):
        return carrier_channel_transform(
            local_values,
            local_maps,
            *case[2:],
            backend="native",
        )

    assert torch.autograd.gradcheck(
        evaluate,
        (values, maps),
        eps=1.0e-6,
        atol=3.0e-6,
        rtol=3.0e-5,
    )
    assert torch.autograd.gradgradcheck(
        evaluate,
        (values, maps),
        eps=1.0e-6,
        atol=4.0e-6,
        rtol=4.0e-5,
    )
    direct = carrier_channel_transform_adjoint(
        torch.ones_like(actual),
        *case,
        backend="native",
    )
    assert direct[0].shape == values.shape
    assert direct[1].shape == maps.shape
    assert direct[0].dtype == dtype
    assert direct[1].dtype == control_dtype


def test_native_carrier_channel_transform_opcheck_and_compile():
    case = _carrier_channel_transform_case(
        torch.device("cpu"), torch.float64
    )
    output_width = int(case[3][-1].item())
    result = torch.library.opcheck(
        torch.ops.ye3t_runtime.carrier_channel_transform.default,
        tuple(value.detach() for value in case) + (output_width,),
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in result.values()), result
    offsets = case[2:]
    compiled = torch.compile(
        lambda values, maps: torch.ops.ye3t_runtime.carrier_channel_transform(
            values,
            maps,
            *offsets,
            output_width,
        ),
        backend="eager",
        fullgraph=True,
    )
    torch.testing.assert_close(
        compiled(case[0], case[1]),
        carrier_channel_transform(*case, backend="native"),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("dtype", "control_dtype"),
    (
        (torch.float32, torch.float32),
        (torch.complex64, torch.complex64),
        (torch.complex64, torch.float32),
    ),
)
@pytest.mark.parametrize("batch_size", (3, 64))
def test_native_cuda_carrier_channel_transform_values_vjp_hvp_repeatable(
    dtype,
    control_dtype,
    batch_size,
):
    capabilities = native_execution_plan_capabilities()
    if not capabilities["cuda"]:
        pytest.skip("requires the native CUDA execution-plan extension")
    native_case = list(
        _carrier_channel_transform_case(
            torch.device("cuda"), dtype, control_dtype
        )
    )
    if batch_size != int(native_case[0].shape[0]):
        generator = torch.Generator(device="cuda").manual_seed(4352)
        native_case[0] = torch.randn(
            batch_size,
            30,
            dtype=dtype,
            device="cuda",
            generator=generator,
            requires_grad=True,
        )
    native_case = tuple(native_case)
    reference_case = tuple(
        value.detach().clone().requires_grad_(True)
        if index < 2
        else value
        for index, value in enumerate(native_case)
    )
    native = carrier_channel_transform(*native_case, backend="native")
    repeat = carrier_channel_transform(*native_case, backend="native",
        output_width=int(native_case[3][-1].cpu()))
    reference = carrier_channel_transform(
        *reference_case, backend="reference"
    )
    assert torch.equal(native, repeat)
    torch.testing.assert_close(native, reference, rtol=2.0e-6, atol=2.0e-6)
    native_loss = native.abs().square().sum()
    reference_loss = reference.abs().square().sum()
    native_gradients = torch.autograd.grad(
        native_loss,
        native_case[:2],
        create_graph=True,
    )
    reference_gradients = torch.autograd.grad(
        reference_loss,
        reference_case[:2],
        create_graph=True,
    )
    for actual, expected in zip(native_gradients, reference_gradients):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=4.0e-5,
            atol=4.0e-5,
        )
    generator = torch.Generator(device="cuda").manual_seed(4353)
    directions = tuple(
        torch.randn(
            value.shape,
            dtype=value.dtype,
            device=value.device,
            generator=generator,
        )
        for value in native_case[:2]
    )
    native_hvp = torch.autograd.grad(
        sum(
            (gradient.conj() * direction).real.sum()
            for gradient, direction in zip(native_gradients, directions)
        ),
        native_case[:2],
    )
    reference_hvp = torch.autograd.grad(
        sum(
            (gradient.conj() * direction).real.sum()
            for gradient, direction in zip(reference_gradients, directions)
        ),
        reference_case[:2],
    )
    for actual, expected in zip(native_hvp, reference_hvp):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=6.0e-5,
            atol=6.0e-5,
        )


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("dtype", "control_dtype"),
    (
        (torch.float32, torch.float32),
        (torch.complex64, torch.complex64),
        (torch.complex64, torch.float32),
    ),
)
def test_native_cuda_carrier_role_channel_map_adjoint_value_vjp_and_hvp(
    dtype,
    control_dtype,
):
    capabilities = native_execution_plan_capabilities()
    if not capabilities["cuda"]:
        pytest.skip("requires the native CUDA execution-plan extension")
    native_case = _carrier_role_channel_map_adjoint_case(
        torch.device("cuda"), dtype, control_dtype
    )
    reference_case = tuple(
        value.detach().clone().requires_grad_(True)
        if index < 3
        else value
        for index, value in enumerate(native_case)
    )
    native = carrier_role_channel_map_adjoint(
        *native_case, backend="native"
    )
    repeat = carrier_role_channel_map_adjoint(
        *native_case, backend="native"
    )
    reference = carrier_role_channel_map_adjoint(
        *reference_case, backend="reference"
    )
    assert torch.equal(native, repeat)
    torch.testing.assert_close(native, reference, rtol=3.0e-6, atol=3.0e-6)
    native_loss = native.abs().square().sum()
    reference_loss = reference.abs().square().sum()
    native_gradients = torch.autograd.grad(
        native_loss,
        native_case[:3],
        create_graph=True,
    )
    reference_gradients = torch.autograd.grad(
        reference_loss,
        reference_case[:3],
        create_graph=True,
    )
    for actual, expected in zip(native_gradients, reference_gradients):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=7.0e-5,
            atol=7.0e-5,
        )
    generator = torch.Generator(device="cuda").manual_seed(7732)
    directions = tuple(
        torch.randn(
            value.shape,
            dtype=value.dtype,
            device=value.device,
            generator=generator,
        )
        for value in native_case[:3]
    )
    native_hvp = torch.autograd.grad(
        sum(
            (gradient.conj() * direction).real.sum()
            for gradient, direction in zip(native_gradients, directions)
        ),
        native_case[:3],
    )
    reference_hvp = torch.autograd.grad(
        sum(
            (gradient.conj() * direction).real.sum()
            for gradient, direction in zip(
                reference_gradients,
                directions,
            )
        ),
        reference_case[:3],
    )
    for actual, expected in zip(native_hvp, reference_hvp):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=1.2e-4,
            atol=1.2e-4,
        )


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("dtype", "control_dtype"),
    (
        (torch.float32, torch.float32),
        (torch.complex64, torch.complex64),
        (torch.complex64, torch.float32),
    ),
)
def test_native_cuda_carrier_role_channel_map_adjoint_large_parallel_dispatch(
    dtype,
    control_dtype,
):
    capabilities = native_execution_plan_capabilities()
    if not capabilities["cuda"]:
        pytest.skip("requires the native CUDA execution-plan extension")
    case = list(
        _carrier_role_channel_map_adjoint_case(
            torch.device("cuda"), dtype, control_dtype
        )
    )
    edge_count = 5600
    generator = torch.Generator(device="cuda").manual_seed(7733)
    case[0] = torch.randn(
        edge_count,
        case[0].shape[1],
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    case[1] = torch.randn(
        edge_count,
        case[1].shape[1],
        dtype=case[1].dtype,
        device="cuda",
        generator=generator,
    )
    case[3] = torch.arange(
        edge_count, dtype=torch.long, device="cuda"
    ).remainder(case[2].shape[0])
    case = tuple(case)
    native = carrier_role_channel_map_adjoint(*case, backend="native")
    repeat = carrier_role_channel_map_adjoint(*case, backend="native")
    reference = carrier_role_channel_map_adjoint(*case, backend="reference")
    assert torch.equal(native, repeat)
    torch.testing.assert_close(
        native,
        reference,
        rtol=8.0e-5,
        atol=8.0e-4,
    )


def _source_arena_case(device, dtype):
    schedule = prepare_source_arena_schedule(
        (2, 0, 2, 3, 0, 5),
        (-1, 0, 1, 1, 0, -1),
        producer_width=7,
    )
    producer = torch.randn(
        4,
        7,
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    atom_types = torch.tensor(
        (0, 1, 0, 1),
        dtype=torch.int64,
        device=device,
    )
    return schedule, producer, atom_types


def _source_arena_channel_transform_case(device, dtype):
    schedule = prepare_source_arena_schedule(
        (2, 0, 2, 3, 0, 5),
        (-1, -1, 0, 0, 1, 1),
        producer_width=7,
    )
    schedule.update(
        {
            "input_feature_offsets": (0, 2, 4, 6),
            "output_feature_offsets": (0, 1, 3, 4),
            "input_channel_offsets": (0, 2, 3, 5),
            "output_channel_offsets": (0, 1, 2, 3),
            "map_offsets": (0, 2, 2, 4),
            "transformed_output_width": 4,
        }
    )
    producer = torch.randn(
        3,
        7,
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    map_dtype = (
        torch.float32
        if dtype == torch.complex64
        else torch.float64
        if dtype == torch.complex128
        else dtype
    )
    channel_maps = torch.tensor(
        (0.75, -0.25, 0.5, 1.25),
        dtype=map_dtype,
        device=device,
        requires_grad=True,
    )
    atom_types = torch.tensor((0, 1, 0), device=device)
    return schedule, producer, channel_maps, atom_types


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_native_source_arena_channel_transform_cpu_value_vjp_and_hvp(dtype):
    schedule, producer, channel_maps, atom_types = (
        _source_arena_channel_transform_case("cpu", dtype)
    )
    reference_producer = producer.detach().clone().requires_grad_(True)
    reference_maps = channel_maps.detach().clone().requires_grad_(True)
    actual = source_arena_channel_transform(
        producer,
        channel_maps,
        schedule,
        atom_types,
        backend="native",
    )
    expected = source_arena_channel_transform(
        reference_producer,
        reference_maps,
        schedule,
        atom_types,
        backend="reference",
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    actual_gradients = torch.autograd.grad(
        _squared_norm(actual),
        (producer, channel_maps),
        create_graph=True,
    )
    expected_gradients = torch.autograd.grad(
        _squared_norm(expected),
        (reference_producer, reference_maps),
        create_graph=True,
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=1.0e-13,
            atol=1.0e-13,
        )
    tangents = (
        torch.randn_like(producer),
        torch.randn_like(channel_maps),
    )
    actual_hvp = torch.autograd.grad(
        actual_gradients,
        (producer, channel_maps),
        tangents,
    )
    expected_hvp = torch.autograd.grad(
        expected_gradients,
        (reference_producer, reference_maps),
        tangents,
    )
    for actual_value, expected_value in zip(actual_hvp, expected_hvp):
        torch.testing.assert_close(
            actual_value,
            expected_value,
            rtol=1.0e-13,
            atol=1.0e-13,
        )


def test_native_source_arena_channel_transform_opcheck_and_compile():
    capabilities = native_execution_plan_capabilities()
    assert capabilities["core_abi_version"] >= 37
    assert capabilities["source_arena_channel_transform"] is True
    schedule, producer, channel_maps, atom_types = (
        _source_arena_channel_transform_case("cpu", torch.float64)
    )
    tensors = tuple(
        torch.tensor(schedule[name], dtype=torch.int64)
        for name in (
            "gather_indices",
            "reverse_offsets",
            "reverse_output_indices",
            "center_types",
            "input_feature_offsets",
            "output_feature_offsets",
            "input_channel_offsets",
            "output_channel_offsets",
            "map_offsets",
        )
    )
    args = (
        producer,
        channel_maps,
        tensors[0],
        tensors[1],
        tensors[2],
        tensors[3],
        atom_types,
        *tensors[4:],
        int(schedule["transformed_output_width"]),
    )
    result = torch.library.opcheck(
        torch.ops.ye3t_runtime.source_arena_channel_transform.default,
        args,
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in result.values()), result
    compiled = torch.compile(
        lambda values, maps: (
            torch.ops.ye3t_runtime.source_arena_channel_transform(
                values,
                maps,
                tensors[0],
                tensors[1],
                tensors[2],
                tensors[3],
                atom_types,
                *tensors[4:],
                int(schedule["transformed_output_width"]),
            )
        ),
        backend="eager",
        fullgraph=True,
    )
    torch.testing.assert_close(
        compiled(producer, channel_maps),
        source_arena_channel_transform(
            producer,
            channel_maps,
            schedule,
            atom_types,
            backend="native",
        ),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_native_source_arena_channel_transform_cuda_value_vjp_and_hvp(dtype):
    if not torch.cuda.is_available() or not _load_extension().has_cuda():
        pytest.skip("requires native CUDA")
    schedule, producer, channel_maps, atom_types = (
        _source_arena_channel_transform_case("cuda", dtype)
    )
    reference_producer = producer.detach().clone().requires_grad_(True)
    reference_maps = channel_maps.detach().clone().requires_grad_(True)
    actual = source_arena_channel_transform(
        producer,
        channel_maps,
        schedule,
        atom_types,
        backend="native",
    )
    expected = source_arena_channel_transform(
        reference_producer,
        reference_maps,
        schedule,
        atom_types,
        backend="reference",
    )
    torch.testing.assert_close(actual, expected, rtol=2.0e-6, atol=2.0e-6)
    actual_gradients = torch.autograd.grad(
        _squared_norm(actual),
        (producer, channel_maps),
        create_graph=True,
    )
    expected_gradients = torch.autograd.grad(
        _squared_norm(expected),
        (reference_producer, reference_maps),
        create_graph=True,
    )
    tangents = (
        torch.randn_like(producer),
        torch.randn_like(channel_maps),
    )
    actual_hvp = torch.autograd.grad(
        actual_gradients,
        (producer, channel_maps),
        tangents,
    )
    expected_hvp = torch.autograd.grad(
        expected_gradients,
        (reference_producer, reference_maps),
        tangents,
    )
    for actual_value, expected_value in zip(
        actual_gradients + actual_hvp,
        expected_gradients + expected_hvp,
    ):
        torch.testing.assert_close(
            actual_value,
            expected_value,
            rtol=3.0e-5,
            atol=3.0e-5,
        )


@pytest.mark.gpu
def test_native_source_arena_channel_transform_cuda_all_identity_vjp_and_hvp():
    if not torch.cuda.is_available() or not _load_extension().has_cuda():
        pytest.skip("requires native CUDA")
    schedule = prepare_source_arena_schedule(
        (0, 1, 2, 3),
        (-1, -1, -1, -1),
        producer_width=4,
    )
    schedule.update(
        {
            "input_feature_offsets": (0, 2, 4),
            "output_feature_offsets": (0, 2, 4),
            "input_channel_offsets": (0, 1, 3),
            "output_channel_offsets": (0, 1, 3),
            "map_offsets": (0, 0, 0),
            "transformed_output_width": 4,
        }
    )
    producer = torch.randn(
        3,
        4,
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    )
    channel_maps = torch.empty(
        0,
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    )
    atom_types = torch.tensor((0, 1, 0), device="cuda")
    actual = source_arena_channel_transform(
        producer,
        channel_maps,
        schedule,
        atom_types,
        backend="native",
    )
    torch.testing.assert_close(actual, producer, rtol=0.0, atol=0.0)
    producer_gradient = torch.autograd.grad(
        actual.square().sum(),
        producer,
        create_graph=True,
    )[0]
    torch.testing.assert_close(
        producer_gradient,
        2.0 * producer,
        rtol=0.0,
        atol=0.0,
    )
    producer_hvp = torch.autograd.grad(
        producer_gradient,
        producer,
        torch.ones_like(producer_gradient),
    )[0]
    torch.testing.assert_close(
        producer_hvp,
        torch.full_like(producer, 2.0),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_native_source_arena_cpu_values_adjoint_and_hvp(dtype):
    schedule, producer, atom_types = _source_arena_case("cpu", dtype)
    reference_producer = producer.detach().clone().requires_grad_(True)

    actual = source_arena_gather(
        producer,
        schedule,
        atom_types,
        backend="native",
    )
    expected = source_arena_gather(
        reference_producer,
        schedule,
        atom_types,
        backend="reference",
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    output_probe = torch.randn_like(actual)
    direct_adjoint = source_arena_gather_adjoint(
        output_probe,
        schedule,
        atom_types,
        backend="native",
    )
    expected_adjoint = torch.autograd.grad(
        expected,
        reference_producer,
        output_probe,
        retain_graph=True,
    )[0]
    torch.testing.assert_close(
        direct_adjoint,
        expected_adjoint,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.count_nonzero(direct_adjoint[:, (1, 4, 6)]) == 0

    actual_gradient = torch.autograd.grad(
        _squared_norm(actual),
        producer,
        create_graph=True,
    )[0]
    expected_gradient = torch.autograd.grad(
        _squared_norm(expected),
        reference_producer,
        create_graph=True,
    )[0]
    torch.testing.assert_close(
        actual_gradient,
        expected_gradient,
        rtol=1e-13,
        atol=1e-13,
    )
    tangent = torch.randn_like(producer)
    actual_hvp = torch.autograd.grad(
        actual_gradient,
        producer,
        tangent,
    )[0]
    expected_hvp = torch.autograd.grad(
        expected_gradient,
        reference_producer,
        tangent,
    )[0]
    torch.testing.assert_close(
        actual_hvp,
        expected_hvp,
        rtol=1e-13,
        atol=1e-13,
    )


def test_native_source_arena_cpu_opcheck_compile_and_schedule_validation():
    capabilities = native_execution_plan_capabilities()
    assert capabilities["core_abi_version"] >= 34
    assert capabilities["source_arena_gather"] is True
    if capabilities["cuda"]:
        assert "source_arena_gather" in capabilities["cuda_operations"]
    schedule, producer, atom_types = _source_arena_case(
        "cpu", torch.float64
    )
    tensors = tuple(
        torch.tensor(schedule[name], dtype=torch.int64)
        for name in (
            "gather_indices",
            "reverse_offsets",
            "reverse_output_indices",
            "center_types",
        )
    )
    result = torch.library.opcheck(
        torch.ops.ye3t_runtime.source_arena_gather.default,
        (producer, *tensors, atom_types),
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in result.values()), result
    compiled = torch.compile(
        lambda values: torch.ops.ye3t_runtime.source_arena_gather(
            values,
            *tensors,
            atom_types,
        ),
        backend="eager",
        fullgraph=True,
    )
    torch.testing.assert_close(
        compiled(producer),
        source_arena_gather(
            producer,
            schedule,
            atom_types,
            backend="native",
        ),
        rtol=0.0,
        atol=0.0,
    )
    invalid = dict(schedule)
    invalid["reverse_output_indices"] = (
        0,
        1,
        2,
        3,
        4,
        4,
    )
    with pytest.raises(ValueError, match="exact CSR"):
        source_arena_gather(
            producer,
            invalid,
            atom_types,
            backend="native",
        )


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_native_source_arena_cuda_values_vjp_hvp_and_repeatability(dtype):
    if not torch.cuda.is_available() or not _load_extension().has_cuda():
        pytest.skip("requires native CUDA")
    schedule, producer, atom_types = _source_arena_case("cuda", dtype)
    reference_producer = producer.detach().clone().requires_grad_(True)
    actual = source_arena_gather(
        producer,
        schedule,
        atom_types,
        backend="native",
    )
    expected = source_arena_gather(
        reference_producer,
        schedule,
        atom_types,
        backend="reference",
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    output_probe = torch.randn_like(actual)
    first = source_arena_gather_adjoint(
        output_probe,
        schedule,
        atom_types,
        backend="native",
    )
    second = source_arena_gather_adjoint(
        output_probe,
        schedule,
        atom_types,
        backend="native",
    )
    assert torch.equal(first, second)
    expected_adjoint = torch.autograd.grad(
        expected,
        reference_producer,
        output_probe,
        retain_graph=True,
    )[0]
    torch.testing.assert_close(
        first,
        expected_adjoint,
        rtol=2e-6,
        atol=2e-6,
    )
    actual_gradient = torch.autograd.grad(
        _squared_norm(actual),
        producer,
        create_graph=True,
    )[0]
    expected_gradient = torch.autograd.grad(
        _squared_norm(expected),
        reference_producer,
        create_graph=True,
    )[0]
    tangent = torch.randn_like(producer)
    actual_hvp = torch.autograd.grad(
        actual_gradient,
        producer,
        tangent,
    )[0]
    expected_hvp = torch.autograd.grad(
        expected_gradient,
        reference_producer,
        tangent,
    )[0]
    torch.testing.assert_close(
        actual_gradient,
        expected_gradient,
        rtol=3e-6,
        atol=3e-6,
    )
    torch.testing.assert_close(
        actual_hvp,
        expected_hvp,
        rtol=3e-6,
        atol=3e-6,
    )


@pytest.mark.parametrize(
    ("dtype", "control_dtype"),
    (
        (torch.float64, torch.float64),
        (torch.complex128, torch.complex128),
        (torch.complex128, torch.float64),
    ),
)
def test_native_carrier_channel_update_cpu_values_gradients_and_hvp(
    dtype,
    control_dtype,
):
    case = _carrier_channel_update_case(
        torch.device("cpu"),
        dtype,
        control_dtype=control_dtype,
    )
    values, gates, channel_maps, feature_offsets, channel_offsets, map_offsets = (
        case
    )
    actual = carrier_channel_update(
        *case,
        backend="native",
    )
    expected = carrier_channel_update(
        *case,
        backend="reference",
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=2.0e-12)

    def evaluate(local_values, local_gates, local_maps):
        return carrier_channel_update(
            local_values,
            local_gates,
            local_maps,
            feature_offsets,
            channel_offsets,
            map_offsets,
            backend="native",
        )

    assert torch.autograd.gradcheck(
        evaluate,
        (values, gates, channel_maps),
        eps=1.0e-6,
        atol=3.0e-6,
        rtol=3.0e-5,
    )
    assert torch.autograd.gradgradcheck(
        evaluate,
        (values, gates, channel_maps),
        eps=1.0e-6,
        atol=4.0e-6,
        rtol=4.0e-5,
    )
    adjoints = carrier_channel_update_adjoint(
        torch.ones_like(actual),
        *case,
        backend="native",
    )
    assert adjoints[0].shape == values.shape
    assert adjoints[1].shape == gates.shape
    assert adjoints[2].shape == channel_maps.shape
    assert adjoints[0].dtype == dtype
    assert adjoints[1].dtype == control_dtype
    assert adjoints[2].dtype == control_dtype


@pytest.mark.parametrize(
    ("dtype", "control_dtype"),
    (
        (torch.float64, torch.float64),
        (torch.complex128, torch.float64),
    ),
)
def test_native_carrier_channel_update_opcheck_and_compile(
    dtype,
    control_dtype,
):
    case = _carrier_channel_update_case(
        torch.device("cpu"),
        dtype,
        control_dtype=control_dtype,
    )
    result = torch.library.opcheck(
        torch.ops.ye3t_runtime.carrier_channel_update.default,
        tuple(value.detach() for value in case),
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in result.values()), result
    feature_offsets, channel_offsets, map_offsets = case[3:]
    compiled = torch.compile(
        lambda values, gates, maps: (
            torch.ops.ye3t_runtime.carrier_channel_update(
                values,
                gates,
                maps,
                feature_offsets,
                channel_offsets,
                map_offsets,
            )
        ),
        backend="eager",
        fullgraph=True,
    )
    torch.testing.assert_close(
        compiled(case[0], case[1], case[2]),
        carrier_channel_update(*case, backend="native"),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("dtype", "control_dtype"),
    (
        (torch.float64, torch.float64),
        (torch.complex128, torch.complex128),
        (torch.complex128, torch.float64),
    ),
)
@pytest.mark.parametrize("batch_size", (3, 64))
def test_native_cuda_carrier_channel_update_values_vjp_hvp(
    dtype,
    control_dtype,
    batch_size,
):
    capabilities = native_execution_plan_capabilities()
    if not capabilities["cuda"]:
        pytest.skip("requires the native CUDA execution-plan extension")
    device = torch.device("cuda")
    native_case = _carrier_channel_update_case(
        device,
        dtype,
        batch_size=batch_size,
        control_dtype=control_dtype,
    )
    reference_case = tuple(
        value.detach().clone().requires_grad_(True)
        if index < 3
        else value
        for index, value in enumerate(native_case)
    )
    native = carrier_channel_update(*native_case, backend="native")
    reference = carrier_channel_update(
        *reference_case,
        backend="reference",
    )
    torch.testing.assert_close(native, reference, rtol=0.0, atol=2.0e-12)
    native_loss = native.abs().square().sum()
    reference_loss = reference.abs().square().sum()
    native_gradients = torch.autograd.grad(
        native_loss,
        native_case[:3],
        create_graph=True,
    )
    reference_gradients = torch.autograd.grad(
        reference_loss,
        reference_case[:3],
        create_graph=True,
    )
    for actual, expected in zip(native_gradients, reference_gradients):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=3.0e-11,
            atol=3.0e-11,
        )
    generator = torch.Generator(device=device).manual_seed(4302)
    directions = tuple(
        torch.randn(
            value.shape,
            dtype=value.dtype,
            device=device,
            generator=generator,
        )
        for value in native_case[:3]
    )
    native_hvp = torch.autograd.grad(
        sum(
            (gradient.conj() * direction).real.sum()
            for gradient, direction in zip(
                native_gradients,
                directions,
            )
        ),
        native_case[:3],
    )
    reference_hvp = torch.autograd.grad(
        sum(
            (gradient.conj() * direction).real.sum()
            for gradient, direction in zip(
                reference_gradients,
                directions,
            )
        ),
        reference_case[:3],
    )
    for actual, expected in zip(native_hvp, reference_hvp):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=6.0e-10,
            atol=6.0e-10,
        )
    if dtype == torch.float64 or control_dtype != dtype:
        result = torch.library.opcheck(
            torch.ops.ye3t_runtime.carrier_channel_update.default,
            tuple(value.detach() for value in native_case),
            raise_exception=False,
        )
        assert all(
            value == "SUCCESS" for value in result.values()
        ), result


def test_native_cuda_carrier_channel_update_auto_has_no_large_size_cliff(
    monkeypatch,
):
    capabilities = native_execution_plan_capabilities()
    if not capabilities["cuda"]:
        pytest.skip("requires the native CUDA execution-plan extension")
    batch_size = (1 << 20) // 24 + 1
    case = _carrier_channel_update_case(
        torch.device("cuda"),
        torch.float32,
        batch_size=batch_size,
    )

    def forbid_reference(*args, **kwargs):
        raise AssertionError("large CUDA carrier update fell back to reference")

    monkeypatch.setattr(
        "ye3t.runtime.execution_plan._carrier_channel_update_reference",
        forbid_reference,
    )
    actual = carrier_channel_update(*case, backend="auto")
    assert actual.shape == case[0].shape
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("complex_coefficients", [False, True])
def test_native_matches_reference_values_adjoint_and_double_backward(
    monkeypatch,
    complex_coefficients,
):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    assembly, synthesis = _case(complex_coefficients)
    source = torch.tensor(
        [[0.4, -0.8], [1.1, 0.2]],
        dtype=torch.float64,
        requires_grad=True,
    )

    reference = apply_source_analysis_reference(source, assembly, synthesis)
    native = apply_source_analysis_native(source, assembly, synthesis)
    torch.testing.assert_close(native, reference, rtol=0.0, atol=1e-12)

    def evaluate(value):
        return apply_source_analysis_native(value, assembly, synthesis)

    assert torch.autograd.gradcheck(
        evaluate,
        (source,),
        eps=1e-6,
        atol=1e-10,
        rtol=1e-8,
    )
    assert torch.autograd.gradgradcheck(
        evaluate,
        (source,),
        eps=1e-6,
        atol=1e-10,
        rtol=1e-8,
    )


@pytest.mark.parametrize("complex_coefficients", [False, True])
def test_native_dense_source_linear_readout_matches_unfused_and_derivatives(
    monkeypatch,
    complex_coefficients,
):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    assembly, synthesis = _case(complex_coefficients)
    dtype = torch.complex128 if complex_coefficients else torch.float64
    source = torch.randn(3, 2, dtype=dtype, requires_grad=True)
    assembly_rows = torch.tensor(assembly.row_indices, dtype=torch.int64)
    assembly_columns = torch.tensor(
        assembly.column_indices,
        dtype=torch.int64,
    )
    assembly_values = torch.tensor(
        (
            assembly.values
            if complex_coefficients
            else [value.real for value in assembly.values]
        ),
        dtype=dtype,
    )
    synthesis_rows = torch.tensor(synthesis.row_indices, dtype=torch.int64)
    synthesis_columns = torch.tensor(
        synthesis.column_indices,
        dtype=torch.int64,
    )
    synthesis_values = torch.tensor(
        (
            synthesis.values
            if complex_coefficients
            else [value.real for value in synthesis.values]
        ),
        dtype=dtype,
    )
    weight = torch.randn(1, dtype=dtype, requires_grad=True)
    bias = torch.randn((), dtype=dtype, requires_grad=True)

    def evaluate(source_value, weight_value, bias_value):
        return torch.ops.ye3t_runtime.source_analysis_linear(
            source_value,
            assembly_rows,
            assembly_columns,
            assembly_values,
            synthesis_rows,
            synthesis_columns,
            synthesis_values,
            2,
            weight_value,
            bias_value,
        )

    _load_extension()
    actual = evaluate(source, weight, bias)
    expected = (
        apply_source_analysis_native(source, assembly, synthesis).reshape(3)
        * weight[0]
        + bias
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)
    assert torch.autograd.gradcheck(
        evaluate,
        (source, weight, bias),
        eps=1e-6,
        atol=1e-9,
        rtol=1e-7,
    )
    assert torch.autograd.gradgradcheck(
        evaluate,
        (source, weight, bias),
        eps=1e-6,
        atol=2e-8,
        rtol=2e-6,
    )


def test_native_operator_passes_opcheck(monkeypatch):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    assembly, synthesis = _case(False)
    source = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    apply_source_analysis_native(source, assembly, synthesis)
    rows = torch.tensor(assembly.row_indices, dtype=torch.int64)
    columns = torch.tensor(assembly.column_indices, dtype=torch.int64)
    assembly_values = torch.tensor(
        [value.real for value in assembly.values],
        dtype=torch.float64,
    )
    synthesis_rows = torch.tensor(synthesis.row_indices, dtype=torch.int64)
    synthesis_columns = torch.tensor(
        synthesis.column_indices,
        dtype=torch.int64,
    )
    synthesis_values = torch.tensor(
        [value.real for value in synthesis.values],
        dtype=torch.float64,
    )

    result = torch.library.opcheck(
        torch.ops.ye3t_runtime.source_analysis.default,
        (
            source,
            rows,
            columns,
            assembly_values,
            synthesis_rows,
            synthesis_columns,
            synthesis_values,
            2,
            1,
        ),
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in result.values()), result


def test_native_operator_supports_torch_compile_fullgraph(monkeypatch):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    assembly, synthesis = _case(False)
    source = torch.randn(3, 2, dtype=torch.float64)
    apply_source_analysis_native(source, assembly, synthesis)
    rows = torch.tensor(assembly.row_indices, dtype=torch.int64)
    columns = torch.tensor(assembly.column_indices, dtype=torch.int64)
    assembly_values = torch.tensor(
        [value.real for value in assembly.values],
        dtype=torch.float64,
    )
    synthesis_rows = torch.tensor(synthesis.row_indices, dtype=torch.int64)
    synthesis_columns = torch.tensor(
        synthesis.column_indices,
        dtype=torch.int64,
    )
    synthesis_values = torch.tensor(
        [value.real for value in synthesis.values],
        dtype=torch.float64,
    )

    def evaluate(value):
        return torch.ops.ye3t_runtime.source_analysis(
            value,
            rows,
            columns,
            assembly_values,
            synthesis_rows,
            synthesis_columns,
            synthesis_values,
            2,
            1,
        )

    eager = evaluate(source)
    compiled = torch.compile(evaluate, backend="eager", fullgraph=True)
    actual = compiled(source)
    torch.testing.assert_close(actual, eager, rtol=0.0, atol=1e-12)


def test_require_native_forbids_reference_backend(monkeypatch):
    monkeypatch.setenv("YE3T_REQUIRE_NATIVE", "1")
    assembly, synthesis = _case(False)
    source = torch.randn(2, 2, dtype=torch.float64)

    with pytest.raises(RuntimeError, match="forbids explicit reference"):
        apply_source_analysis(
            source,
            assembly,
            synthesis,
            backend="reference",
        )


@pytest.mark.parametrize("antisymmetric", [False, True])
@pytest.mark.parametrize("complex_values", [False, True])
def test_native_compact_pair_product_matches_reference_and_derivatives(
    monkeypatch,
    antisymmetric,
    complex_values,
):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    dtype = torch.complex128 if complex_values else torch.float64
    left = torch.randn(2, 4, dtype=dtype, requires_grad=True)
    right = torch.randn(2, 4, dtype=dtype, requires_grad=True)
    expected = compact_pair_product_reference(
        left,
        right,
        antisymmetric=antisymmetric,
    )
    actual = compact_pair_product(
        left,
        right,
        antisymmetric=antisymmetric,
        backend="native",
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)

    def evaluate(left_value, right_value):
        return compact_pair_product(
            left_value,
            right_value,
            antisymmetric=antisymmetric,
            backend="native",
        )

    assert torch.autograd.gradcheck(
        evaluate,
        (left, right),
        eps=1e-6,
        atol=1e-9,
        rtol=1e-7,
    )
    assert torch.autograd.gradgradcheck(
        evaluate,
        (left, right),
        eps=1e-6,
        atol=1e-9,
        rtol=1e-7,
    )
    native_left = left.detach().clone().requires_grad_(True)
    native_right = right.detach().clone().requires_grad_(True)
    reference_left = left.detach().clone().requires_grad_(True)
    reference_right = right.detach().clone().requires_grad_(True)
    native_gradient = torch.autograd.grad(
        _squared_norm(evaluate(native_left, native_right)),
        (native_left, native_right),
        create_graph=True,
    )
    reference_gradient = torch.autograd.grad(
        _squared_norm(
            compact_pair_product_reference(
                reference_left,
                reference_right,
                antisymmetric=antisymmetric,
            )
        ),
        (reference_left, reference_right),
        create_graph=True,
    )
    probes = (
        torch.randn_like(native_left),
        torch.randn_like(native_right),
    )
    native_hvp = torch.autograd.grad(
        native_gradient,
        (native_left, native_right),
        probes,
    )
    reference_hvp = torch.autograd.grad(
        reference_gradient,
        (reference_left, reference_right),
        probes,
    )
    torch.testing.assert_close(
        native_hvp[0],
        reference_hvp[0],
        rtol=2e-7,
        atol=2e-9,
    )
    torch.testing.assert_close(
        native_hvp[1],
        reference_hvp[1],
        rtol=2e-7,
        atol=2e-9,
    )


@pytest.mark.parametrize("antisymmetric", [False, True])
def test_native_compact_pair_operator_passes_opcheck(
    monkeypatch,
    antisymmetric,
):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    left = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    right = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    compact_pair_product(
        left,
        right,
        antisymmetric=antisymmetric,
        backend="native",
    )

    result = torch.library.opcheck(
        torch.ops.ye3t_runtime.compact_pair_product.default,
        (left, right, antisymmetric),
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in result.values()), result


@pytest.mark.parametrize("order,dimension", [(3, 5), (4, 5)])
@pytest.mark.parametrize("complex_values", [False, True])
def test_native_compact_exterior_power_matches_reference_and_derivatives(
    monkeypatch,
    order,
    dimension,
    complex_values,
):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    dtype = torch.complex128 if complex_values else torch.float64
    factors = torch.randn(
        2,
        order,
        dimension,
        dtype=dtype,
        requires_grad=True,
    )
    expected = compact_exterior_power_product_reference(factors)
    actual = compact_exterior_power_product(
        factors,
        backend="native",
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=2e-12)

    def evaluate(value):
        return compact_exterior_power_product(
            value,
            backend="native",
        )

    assert torch.autograd.gradcheck(
        evaluate,
        (factors,),
        eps=1e-6,
        atol=2e-9,
        rtol=2e-7,
    )
    assert torch.autograd.gradgradcheck(
        evaluate,
        (factors,),
        eps=1e-6,
        atol=3e-8,
        rtol=3e-6,
    )

    native_factors = factors.detach().clone().requires_grad_(True)
    reference_factors = factors.detach().clone().requires_grad_(True)
    native_output = evaluate(native_factors)
    reference_output = compact_exterior_power_product_reference(
        reference_factors
    )
    output_adjoint = torch.randn_like(native_output)
    native_gradient = torch.autograd.grad(
        native_output,
        native_factors,
        output_adjoint,
        create_graph=True,
    )[0]
    reference_gradient = torch.autograd.grad(
        reference_output,
        reference_factors,
        output_adjoint,
        create_graph=True,
    )[0]
    tangent = torch.randn_like(native_factors)
    native_hvp = torch.autograd.grad(
        native_gradient,
        native_factors,
        tangent,
    )[0]
    reference_hvp = torch.autograd.grad(
        reference_gradient,
        reference_factors,
        tangent,
    )[0]
    torch.testing.assert_close(
        native_hvp,
        reference_hvp,
        rtol=3e-6,
        atol=3e-8,
    )
    native_factors = factors.detach().clone().requires_grad_(True)
    reference_factors = factors.detach().clone().requires_grad_(True)
    native_gradient = torch.autograd.grad(
        _squared_norm(evaluate(native_factors)),
        native_factors,
        create_graph=True,
    )[0]
    reference_gradient = torch.autograd.grad(
        _squared_norm(
            compact_exterior_power_product_reference(reference_factors)
        ),
        reference_factors,
        create_graph=True,
    )[0]
    probe = torch.randn_like(native_factors)
    native_hvp = torch.autograd.grad(
        native_gradient,
        native_factors,
        probe,
    )[0]
    reference_hvp = torch.autograd.grad(
        reference_gradient,
        reference_factors,
        probe,
    )[0]
    torch.testing.assert_close(
        native_hvp,
        reference_hvp,
        rtol=3e-6,
        atol=3e-8,
    )


def test_native_compact_exterior_power_operator_passes_opcheck(monkeypatch):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    factors = torch.randn(
        2,
        3,
        5,
        dtype=torch.float64,
        requires_grad=True,
    )
    compact_exterior_power_product(factors, backend="native")

    result = torch.library.opcheck(
        torch.ops.ye3t_runtime.compact_exterior_power.default,
        (factors,),
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in result.values()), result


@pytest.mark.parametrize("complex_values", [False, True])
def test_native_symmetric_power_monomial_matches_reference_and_derivatives(
    monkeypatch,
    complex_values,
):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    dtype = torch.complex128 if complex_values else torch.float64
    input = torch.tensor(
        [
            [0.0, -0.5, 1.25],
            [1.5, 0.0, -0.75],
            [-0.625, 1.25, 0.0],
        ],
        dtype=dtype,
        requires_grad=True,
    )
    if complex_values:
        input = (
            input + 0.5j * torch.flip(input, dims=(1,))
        ).detach().requires_grad_(True)
    counts = torch.tensor(
        [
            [2, 0, 0],
            [0, 1, 1],
            [1, 1, 0],
            [0, 0, 2],
        ],
        dtype=torch.int64,
    )
    offsets = torch.tensor([0, 2, 3, 3, 4], dtype=torch.int64)
    indices = torch.tensor([0, 0, 1, 3], dtype=torch.int64)
    values = torch.tensor(
        [0.75, -1.25, 0.5, 1.5],
        dtype=dtype,
    )
    if complex_values:
        values = values + 1j * torch.tensor(
            [0.25, -0.5, 0.75, 0.125],
            dtype=dtype,
        )

    expected = symmetric_power_monomial_reference(
        input,
        counts,
        offsets,
        indices,
        values,
    )
    actual = symmetric_power_monomial_contraction(
        input,
        counts,
        offsets,
        indices,
        values,
        backend="native",
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)

    def evaluate(input_value):
        return symmetric_power_monomial_contraction(
            input_value,
            counts,
            offsets,
            indices,
            values,
            backend="native",
        )

    assert torch.autograd.gradcheck(
        evaluate,
        (input,),
        eps=1e-6,
        atol=1e-9,
        rtol=1e-7,
    )
    assert torch.autograd.gradgradcheck(
        evaluate,
        (input,),
        eps=1e-6,
        atol=2e-8,
        rtol=2e-6,
    )
    native_input = input.detach().clone().requires_grad_(True)
    reference_input = input.detach().clone().requires_grad_(True)
    native_gradient = torch.autograd.grad(
        _squared_norm(evaluate(native_input)),
        native_input,
        create_graph=True,
    )[0]
    reference_gradient = torch.autograd.grad(
        _squared_norm(
            symmetric_power_monomial_reference(
                reference_input,
                counts,
                offsets,
                indices,
                values,
            )
        ),
        reference_input,
        create_graph=True,
    )[0]
    probe = torch.randn_like(native_input)
    native_hvp = torch.autograd.grad(
        native_gradient,
        native_input,
        probe,
    )[0]
    reference_hvp = torch.autograd.grad(
        reference_gradient,
        reference_input,
        probe,
    )[0]
    torch.testing.assert_close(
        native_hvp,
        reference_hvp,
        rtol=2e-6,
        atol=2e-8,
    )


@pytest.mark.parametrize("complex_values", [False, True])
def test_native_cuda_symmetric_power_monomial_double_backward_matches_reference(
    monkeypatch,
    complex_values,
):
    if not torch.cuda.is_available() or not _load_extension().has_cuda():
        pytest.skip("requires CUDA native execution-plan extension")
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    dtype = torch.complex128 if complex_values else torch.float64
    input = torch.tensor(
        [
            [0.0, -0.5, 1.25],
            [1.5, 0.0, -0.75],
            [-0.625, 1.25, 0.0],
        ],
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    if complex_values:
        input = (
            input + 0.5j * torch.flip(input, dims=(1,))
        ).detach().requires_grad_(True)
    counts = torch.tensor(
        [
            [3, 0, 0],
            [0, 2, 1],
            [1, 1, 1],
            [0, 0, 3],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    offsets = torch.tensor(
        [0, 2, 3, 3, 4],
        dtype=torch.int64,
        device="cuda",
    )
    indices = torch.tensor(
        [0, 0, 1, 3],
        dtype=torch.int64,
        device="cuda",
    )
    values = torch.tensor(
        [0.75, -1.25, 0.5, 1.5],
        dtype=dtype,
        device="cuda",
    )
    if complex_values:
        values = values + 1j * torch.tensor(
            [0.25, -0.5, 0.75, 0.125],
            dtype=dtype,
            device="cuda",
        )
    output_adjoint = torch.randn(
        (input.shape[0], offsets.numel() - 1),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    input_adjoint_tangent = torch.randn_like(input)
    reference = symmetric_power_monomial_adjoint_reference(
        output_adjoint,
        input,
        counts,
        offsets,
        indices,
        values,
    )
    expected = torch.autograd.grad(
        reference,
        (output_adjoint, input),
        input_adjoint_tangent,
    )
    actual = (
        torch.ops.ye3t_runtime.symmetric_power_monomial_double_backward(
            input_adjoint_tangent,
            output_adjoint,
            input,
            counts,
            offsets,
            indices,
            values,
        )
    )
    for value, expected_value in zip(actual, expected):
        torch.testing.assert_close(
            value,
            expected_value,
            rtol=2e-12,
            atol=2e-12,
        )


def test_native_symmetric_power_monomial_operator_passes_opcheck(monkeypatch):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    input = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    counts = torch.tensor(
        [[2, 0, 0], [0, 1, 1], [1, 1, 0], [0, 0, 2]],
        dtype=torch.int64,
    )
    offsets = torch.tensor([0, 2, 3, 3, 4], dtype=torch.int64)
    indices = torch.tensor([0, 0, 1, 3], dtype=torch.int64)
    values = torch.tensor(
        [0.75, -1.25, 0.5, 1.5],
        dtype=torch.float64,
    )
    symmetric_power_monomial_contraction(
        input,
        counts,
        offsets,
        indices,
        values,
        backend="native",
    )

    result = torch.library.opcheck(
        torch.ops.ye3t_runtime.symmetric_power_monomial.default,
        (input, counts, offsets, indices, values),
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in result.values()), result


@pytest.mark.parametrize("complex_values", [False, True])
def test_native_shared_symmetric_power_matches_reference_and_derivatives(
    monkeypatch,
    complex_values,
):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    dtype = torch.complex128 if complex_values else torch.float64
    input = torch.tensor(
        [
            [0.0, -0.5, 1.25],
            [1.5, 0.0, -0.75],
            [-0.625, 1.25, 0.0],
        ],
        dtype=dtype,
        requires_grad=True,
    )
    if complex_values:
        input = (
            input + 0.5j * torch.flip(input, dims=(1,))
        ).detach().requires_grad_(True)
    counts = torch.tensor(
        [
            [2, 0, 0],
            [0, 1, 1],
            [1, 1, 0],
            [0, 0, 2],
        ],
        dtype=torch.int64,
    )
    offsets = torch.tensor([0, 2, 4, 5], dtype=torch.int64)
    terms = torch.tensor([0, 1, 1, 2, 0], dtype=torch.int64)
    outputs = torch.tensor([0, 0, 1, 1, 2], dtype=torch.int64)
    values = torch.tensor(
        [0.75, -1.25, 0.5, -0.625, 1.5],
        dtype=dtype,
    )
    if complex_values:
        values = values + 1j * torch.tensor(
            [0.25, -0.5, 0.75, 0.125, -0.375],
            dtype=dtype,
        )

    expected = symmetric_power_shared_monomial_reference(
        input,
        counts,
        offsets,
        terms,
        outputs,
        values,
    )
    actual = symmetric_power_shared_monomial_contraction(
        input,
        counts,
        offsets,
        terms,
        outputs,
        values,
        backend="native",
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)

    def evaluate(input_value):
        return symmetric_power_shared_monomial_contraction(
            input_value,
            counts,
            offsets,
            terms,
            outputs,
            values,
            backend="native",
        )

    assert torch.autograd.gradcheck(
        evaluate,
        (input,),
        eps=1e-6,
        atol=1e-9,
        rtol=1e-7,
    )
    assert torch.autograd.gradgradcheck(
        evaluate,
        (input,),
        eps=1e-6,
        atol=2e-8,
        rtol=2e-6,
    )
    native_input = input.detach().clone().requires_grad_(True)
    reference_input = input.detach().clone().requires_grad_(True)
    native_gradient = torch.autograd.grad(
        _squared_norm(evaluate(native_input)),
        native_input,
        create_graph=True,
    )[0]
    reference_gradient = torch.autograd.grad(
        _squared_norm(
            symmetric_power_shared_monomial_reference(
                reference_input,
                counts,
                offsets,
                terms,
                outputs,
                values,
            )
        ),
        reference_input,
        create_graph=True,
    )[0]
    probe = torch.randn_like(native_input)
    native_hvp = torch.autograd.grad(
        native_gradient,
        native_input,
        probe,
    )[0]
    reference_hvp = torch.autograd.grad(
        reference_gradient,
        reference_input,
        probe,
    )[0]
    torch.testing.assert_close(
        native_hvp,
        reference_hvp,
        rtol=2e-6,
        atol=2e-8,
    )


def test_native_shared_symmetric_power_operator_passes_opcheck(monkeypatch):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    input = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    counts = torch.tensor(
        [[2, 0, 0], [0, 1, 1], [1, 1, 0], [0, 0, 2]],
        dtype=torch.int64,
    )
    offsets = torch.tensor([0, 2, 4, 5], dtype=torch.int64)
    terms = torch.tensor([0, 1, 1, 2, 0], dtype=torch.int64)
    outputs = torch.tensor([0, 0, 1, 1, 2], dtype=torch.int64)
    values = torch.tensor(
        [0.75, -1.25, 0.5, -0.625, 1.5],
        dtype=torch.float64,
    )
    symmetric_power_shared_monomial_contraction(
        input,
        counts,
        offsets,
        terms,
        outputs,
        values,
        backend="native",
    )

    result = torch.library.opcheck(
        torch.ops.ye3t_runtime.symmetric_power_shared_monomial.default,
        (input, counts, offsets, terms, outputs, values),
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in result.values()), result


@pytest.mark.parametrize("complex_values", [False, True])
@pytest.mark.parametrize("algorithm", ["coefficient", "factored"])
def test_native_cuda_shared_symmetric_power_double_backward_matches_reference(
    monkeypatch,
    complex_values,
    algorithm,
):
    if not torch.cuda.is_available() or not _load_extension().has_cuda():
        pytest.skip("requires CUDA native execution-plan extension")
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    dtype = torch.complex128 if complex_values else torch.float64
    input = torch.tensor(
        [
            [0.0, -0.5, 1.25],
            [1.5, 0.0, -0.75],
            [-0.625, 1.25, 0.0],
        ],
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    if complex_values:
        input = (
            input + 0.5j * torch.flip(input, dims=(1,))
        ).detach().requires_grad_(True)
    counts = torch.tensor(
        [[3, 0, 0], [0, 2, 1], [1, 1, 1], [0, 0, 3]],
        dtype=torch.int64,
        device="cuda",
    )
    offsets = torch.tensor(
        [0, 2, 4, 5],
        dtype=torch.int64,
        device="cuda",
    )
    terms = torch.tensor(
        [0, 1, 1, 2, 3],
        dtype=torch.int64,
        device="cuda",
    )
    outputs = torch.tensor(
        [0, 0, 1, 1, 2],
        dtype=torch.int64,
        device="cuda",
    )
    values = torch.tensor(
        [0.75, -1.25, 0.5, -0.625, 1.5],
        dtype=dtype,
        device="cuda",
    )
    if complex_values:
        values = values + 1j * torch.tensor(
            [0.25, -0.5, 0.75, 0.125, -0.375],
            dtype=dtype,
            device="cuda",
        )
    output_adjoint = torch.randn(
        (input.shape[0], offsets.numel() - 1),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    input_adjoint_tangent = torch.randn_like(input)
    reference = symmetric_power_shared_monomial_adjoint_reference(
        output_adjoint,
        input,
        counts,
        offsets,
        terms,
        outputs,
        values,
    )
    expected = torch.autograd.grad(
        reference,
        (output_adjoint, input),
        input_adjoint_tangent,
    )
    operation_name = (
        "symmetric_power_shared_monomial_factored_double_backward"
        if algorithm == "factored"
        else "symmetric_power_shared_monomial_double_backward"
    )
    operation = getattr(torch.ops.ye3t_runtime, operation_name)
    actual = operation(
        input_adjoint_tangent,
        output_adjoint,
        input,
        counts,
        offsets,
        terms,
        outputs,
        values,
    )
    for value, expected_value in zip(actual, expected):
        torch.testing.assert_close(
            value,
            expected_value,
            rtol=2e-12,
            atol=2e-12,
        )

    result = torch.library.opcheck(
        operation.default,
        (
            input_adjoint_tangent,
            output_adjoint.detach(),
            input.detach(),
            counts,
            offsets,
            terms,
            outputs,
            values,
        ),
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in result.values()), result


@pytest.mark.parametrize("power", [2, 3, 4, 6, 8, 12, 16, 32])
@pytest.mark.parametrize("complex_values", [False, True])
def test_native_cuda_factored_shared_power_double_backward_is_rank_general(
    monkeypatch,
    power,
    complex_values,
):
    if not torch.cuda.is_available() or not _load_extension().has_cuda():
        pytest.skip("requires CUDA native execution-plan extension")
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    dtype = torch.complex128 if complex_values else torch.float64
    input = torch.tensor(
        [[0.75, -1.25, 1.5], [1.125, 0.625, -0.875]],
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    if complex_values:
        input = (
            input + 0.125j * torch.flip(input, dims=(1,))
        ).detach().requires_grad_(True)
    counts = torch.tensor(
        [
            [power, 0, 0],
            [power - 1, 1, 0],
            [power - 2, 1, 1],
            [0, 0, power],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    offsets = torch.tensor([0, 2, 4, 5], device="cuda")
    terms = torch.tensor([0, 1, 1, 2, 3], device="cuda")
    outputs = torch.tensor([0, 0, 1, 1, 2], device="cuda")
    values = torch.tensor(
        [0.75, -1.25, 0.5, -0.625, 1.5],
        dtype=dtype,
        device="cuda",
    )
    if complex_values:
        values = values + 0.125j * torch.flip(values, dims=(0,))
    output_adjoint = torch.randn(
        (input.shape[0], offsets.numel() - 1),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    input_adjoint_tangent = torch.randn_like(input)
    reference = symmetric_power_shared_monomial_adjoint_reference(
        output_adjoint,
        input,
        counts,
        offsets,
        terms,
        outputs,
        values,
    )
    expected = torch.autograd.grad(
        reference,
        (output_adjoint, input),
        input_adjoint_tangent,
    )
    actual = (
        torch.ops.ye3t_runtime
        .symmetric_power_shared_monomial_factored_double_backward(
            input_adjoint_tangent,
            output_adjoint,
            input,
            counts,
            offsets,
            terms,
            outputs,
            values,
        )
    )
    for value, expected_value in zip(actual, expected):
        torch.testing.assert_close(
            value,
            expected_value,
            rtol=5e-11,
            atol=5e-11,
        )


@pytest.mark.parametrize("power", [2, 3, 4, 6, 8, 12, 16, 32])
@pytest.mark.parametrize("complex_values", [False, True])
def test_native_cuda_sparse_shared_power_is_rank_general(
    monkeypatch,
    power,
    complex_values,
):
    if not torch.cuda.is_available() or not _load_extension().has_cuda():
        pytest.skip("requires CUDA native execution-plan extension")
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    dtype = torch.complex128 if complex_values else torch.float64
    input = torch.tensor(
        [[0.75, -1.25, 1.5], [1.125, 0.625, -0.875]],
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    if complex_values:
        input = (
            input + 0.125j * torch.flip(input, dims=(1,))
        ).detach().requires_grad_(True)
    counts = torch.tensor(
        [
            [power, 0, 0],
            [power - 1, 1, 0],
            [power - 2, 1, 1],
            [0, 0, power],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    support = sparse_monomial_support_from_counts(counts)
    offsets = torch.tensor([0, 2, 4, 5], device="cuda")
    terms = torch.tensor([0, 1, 1, 2, 3], device="cuda")
    outputs = torch.tensor([0, 0, 1, 1, 2], device="cuda")
    values = torch.tensor(
        [0.75, -1.25, 0.5, -0.625, 1.5],
        dtype=dtype,
        device="cuda",
    )
    if complex_values:
        values = values + 0.125j * torch.flip(values, dims=(0,))
    expected = symmetric_power_shared_monomial_reference(
        input,
        counts,
        offsets,
        terms,
        outputs,
        values,
    )
    actual = symmetric_power_shared_sparse_monomial_contraction(
        input,
        *support,
        offsets,
        terms,
        outputs,
        values,
        backend="native",
    )
    torch.testing.assert_close(actual, expected, rtol=5e-12, atol=5e-12)
    output_adjoint = torch.randn_like(actual)
    expected_vjp = torch.autograd.grad(
        expected,
        input,
        output_adjoint,
        create_graph=True,
    )[0]
    actual_vjp = torch.autograd.grad(
        actual,
        input,
        output_adjoint,
        create_graph=True,
    )[0]
    torch.testing.assert_close(
        actual_vjp,
        expected_vjp,
        rtol=5e-11,
        atol=5e-11,
    )
    tangent = torch.randn_like(input)
    expected_hvp = torch.autograd.grad(
        expected_vjp,
        input,
        tangent,
    )[0]
    actual_hvp = torch.autograd.grad(
        actual_vjp,
        input,
        tangent,
    )[0]
    torch.testing.assert_close(
        actual_hvp,
        expected_hvp,
        rtol=5e-10,
        atol=5e-10,
    )


def test_native_cuda_sparse_shared_power_opcheck():
    if not torch.cuda.is_available() or not _load_extension().has_cuda():
        pytest.skip("requires CUDA native execution-plan extension")
    input = torch.randn(
        (3, 5), dtype=torch.float64, device="cuda"
    )
    counts = torch.tensor(
        [[2, 0, 0, 0, 0], [1, 0, 1, 0, 0], [0, 0, 0, 1, 1]],
        dtype=torch.int64,
        device="cuda",
    )
    support = sparse_monomial_support_from_counts(counts)
    offsets = torch.tensor([0, 2, 4], device="cuda")
    terms = torch.tensor([0, 1, 1, 2], device="cuda")
    outputs = torch.tensor([0, 0, 1, 1], device="cuda")
    values = torch.tensor(
        [0.75, -1.25, 0.5, 1.5],
        dtype=torch.float64,
        device="cuda",
    )
    result = torch.library.opcheck(
        torch.ops.ye3t_runtime.symmetric_power_shared_sparse_monomial.default,
        (input, *support, offsets, terms, outputs, values),
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in result.values()), result


@pytest.mark.parametrize("complex_values", [False, True])
def test_native_cuda_sparse_shared_power_zero_support_derivatives(
    complex_values,
):
    if not torch.cuda.is_available() or not _load_extension().has_cuda():
        pytest.skip("requires CUDA native execution-plan extension")
    dtype = torch.complex128 if complex_values else torch.float64
    input = torch.tensor(
        [[0.0, 1.25, -0.75], [1.5, 0.0, -0.625], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device="cuda",
    )
    if complex_values:
        imaginary = torch.tensor(
            [[0.0, 0.25, -0.5], [0.125, 0.0, 0.75], [0.0, 0.0, -0.25]],
            dtype=dtype,
            device="cuda",
        )
        input = input + 1j * imaginary
    input.requires_grad_(True)
    counts = torch.tensor(
        [[2, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1], [0, 2, 0]],
        dtype=torch.int64,
        device="cuda",
    )
    support = sparse_monomial_support_from_counts(counts)
    offsets = torch.tensor([0, 2, 4, 5], device="cuda")
    terms = torch.tensor([0, 1, 2, 3, 4], device="cuda")
    outputs = torch.tensor([0, 0, 1, 1, 2], device="cuda")
    values = torch.tensor(
        [0.75, -1.25, 0.5, -0.625, 1.5],
        dtype=dtype,
        device="cuda",
    )
    expected = symmetric_power_shared_monomial_reference(
        input, counts, offsets, terms, outputs, values
    )
    actual = symmetric_power_shared_sparse_monomial_contraction(
        input,
        *support,
        offsets,
        terms,
        outputs,
        values,
        backend="native",
    )
    output_adjoint = torch.randn_like(actual)
    expected_vjp = torch.autograd.grad(
        expected, input, output_adjoint, create_graph=True
    )[0]
    actual_vjp = torch.autograd.grad(
        actual, input, output_adjoint, create_graph=True
    )[0]
    tangent = torch.randn_like(input)
    expected_hvp = torch.autograd.grad(
        expected_vjp, input, tangent
    )[0]
    actual_hvp = torch.autograd.grad(actual_vjp, input, tangent)[0]
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(
        actual_vjp, expected_vjp, rtol=2e-11, atol=2e-11
    )
    torch.testing.assert_close(
        actual_hvp, expected_hvp, rtol=2e-10, atol=2e-10
    )


@pytest.mark.parametrize("complex_values", [False, True])
def test_native_shared_symmetric_power_batched_adjoint(
    monkeypatch,
    complex_values,
):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    dtype = torch.complex128 if complex_values else torch.float64
    input = torch.tensor(
        [
            [0.0, -0.5, 1.25],
            [1.5, 0.0, -0.75],
            [-0.625, 1.25, 0.0],
        ],
        dtype=dtype,
        requires_grad=True,
    )
    if complex_values:
        input = (
            input + 0.5j * torch.flip(input, dims=(1,))
        ).detach().requires_grad_(True)
    counts = torch.tensor(
        [
            [2, 0, 0],
            [0, 1, 1],
            [1, 1, 0],
            [0, 0, 2],
        ],
        dtype=torch.int64,
    )
    offsets = torch.tensor([0, 2, 4, 5], dtype=torch.int64)
    terms = torch.tensor([0, 1, 1, 2, 0], dtype=torch.int64)
    outputs = torch.tensor([0, 0, 1, 1, 2], dtype=torch.int64)
    values = torch.tensor(
        [0.75, -1.25, 0.5, -0.625, 1.5],
        dtype=dtype,
    )
    if complex_values:
        values = values + 1j * torch.tensor(
            [0.25, -0.5, 0.75, 0.125, -0.375],
            dtype=dtype,
        )
    output_adjoint = torch.randn(
        2,
        input.shape[0],
        offsets.numel() - 1,
        dtype=dtype,
        requires_grad=True,
    )

    expected = symmetric_power_shared_monomial_batched_adjoint_reference(
        output_adjoint,
        input,
        counts,
        offsets,
        terms,
        outputs,
        values,
    )
    actual = symmetric_power_shared_monomial_batched_adjoint(
        output_adjoint,
        input,
        counts,
        offsets,
        terms,
        outputs,
        values,
        backend="native",
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)

    def evaluate(output_gradient, input_value):
        return symmetric_power_shared_monomial_batched_adjoint(
            output_gradient,
            input_value,
            counts,
            offsets,
            terms,
            outputs,
            values,
            backend="native",
        )

    assert torch.autograd.gradcheck(
        evaluate,
        (output_adjoint, input),
        eps=1e-6,
        atol=2e-9,
        rtol=2e-7,
    )
    assert torch.autograd.gradgradcheck(
        evaluate,
        (output_adjoint, input),
        eps=1e-6,
        atol=3e-8,
        rtol=3e-6,
    )


def test_native_shared_symmetric_power_batched_adjoint_coupled_seed_hessian(
    monkeypatch,
):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    counts = torch.tensor(
        [[2, 0, 0], [0, 1, 1], [1, 1, 0], [0, 0, 2]],
        dtype=torch.int64,
    )
    offsets = torch.tensor([0, 2, 4, 5], dtype=torch.int64)
    terms = torch.tensor([0, 1, 1, 2, 0], dtype=torch.int64)
    outputs = torch.tensor([0, 0, 1, 1, 2], dtype=torch.int64)
    values = torch.tensor(
        [0.75, -1.25, 0.5, -0.625, 1.5],
        dtype=torch.float64,
    )
    initial = torch.tensor(
        [[0.25, -0.5, 1.25], [1.5, 0.75, -0.625]],
        dtype=torch.float64,
    )
    probe = torch.tensor(
        [[-0.75, 0.25, 0.5], [0.125, -1.0, 0.375]],
        dtype=torch.float64,
    )

    def objective(input_value, backend):
        output = symmetric_power_shared_monomial_contraction(
            input_value,
            counts,
            offsets,
            terms,
            outputs,
            values,
            backend="reference",
        )
        output_adjoint = torch.stack((output, -0.375 * output), dim=0)
        roots = symmetric_power_shared_monomial_batched_adjoint(
            output_adjoint,
            input_value,
            counts,
            offsets,
            terms,
            outputs,
            values,
            backend=backend,
        )
        return torch.sum(roots.square())

    native_input = initial.clone().requires_grad_(True)
    native_gradient = torch.autograd.grad(
        objective(native_input, "native"),
        native_input,
        create_graph=True,
    )[0]
    native_hvp = torch.autograd.grad(
        torch.sum(native_gradient * probe),
        native_input,
    )[0]
    reference_input = initial.clone().requires_grad_(True)
    reference_gradient = torch.autograd.grad(
        objective(reference_input, "reference"),
        reference_input,
        create_graph=True,
    )[0]
    reference_hvp = torch.autograd.grad(
        torch.sum(reference_gradient * probe),
        reference_input,
    )[0]

    torch.testing.assert_close(
        native_gradient,
        reference_gradient,
        rtol=2e-10,
        atol=2e-10,
    )
    torch.testing.assert_close(
        native_hvp,
        reference_hvp,
        rtol=2e-9,
        atol=2e-9,
    )


def test_native_shared_symmetric_power_batched_adjoint_passes_opcheck(
    monkeypatch,
):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    input = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    counts = torch.tensor(
        [[2, 0, 0], [0, 1, 1], [1, 1, 0], [0, 0, 2]],
        dtype=torch.int64,
    )
    offsets = torch.tensor([0, 2, 4, 5], dtype=torch.int64)
    terms = torch.tensor([0, 1, 1, 2, 0], dtype=torch.int64)
    outputs = torch.tensor([0, 0, 1, 1, 2], dtype=torch.int64)
    values = torch.tensor(
        [0.75, -1.25, 0.5, -0.625, 1.5],
        dtype=torch.float64,
    )
    output_adjoint = torch.randn(
        2,
        input.shape[0],
        offsets.numel() - 1,
        dtype=input.dtype,
        requires_grad=True,
    )
    symmetric_power_shared_monomial_batched_adjoint(
        output_adjoint,
        input,
        counts,
        offsets,
        terms,
        outputs,
        values,
        backend="native",
    )

    result = torch.library.opcheck(
        torch.ops.ye3t_runtime
        .symmetric_power_shared_monomial_batched_adjoint.default,
        (
            output_adjoint,
            input,
            counts,
            offsets,
            terms,
            outputs,
            values,
        ),
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in result.values()), result


@pytest.mark.parametrize("complex_values", [False, True])
def test_native_factorized_angular_matches_reference_and_derivatives(
    monkeypatch,
    complex_values,
):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    execution_plan = _rank3_factorized_plan()
    dtype = torch.complex128 if complex_values else torch.float64
    native = YE3TFactorizedAngularModule(
        execution_plan,
        backend="native",
        dtype=dtype,
    )
    reference = YE3TFactorizedAngularModule(
        execution_plan,
        backend="reference",
        dtype=dtype,
    )
    assert reference.state_dict() == {}
    native_report = native.runtime_report()
    assert native_report["reference_dense_synthesis_materialized"] is False
    assert native_report["reference_dense_synthesis_bytes"] == 0
    reference_report = reference.runtime_report()
    assert reference_report["reference_dense_synthesis_materialized"] is True
    assert reference_report["reference_dense_synthesis_bytes"] > 0
    assert reference_report[
        "per_forward_synthesis_table_reconstruction"
    ] is False
    slots = tuple(
        torch.randn(
            2,
            2 * angular_L + 1,
            dtype=dtype,
            requires_grad=True,
        )
        for angular_L in (1, 2, 1)
    )

    actual = native(slots)
    expected = reference(slots)
    packed, leading_shape = reference._packed_slots(slots)
    sparse_reconstruction = _factorized_angular_packed_reference(
        packed,
        *reference._native_arguments(),
        reference.source_dimension,
    ).reshape(
        leading_shape
        + (
            reference.logical_channel_tableau_dimension,
            reference.magnetic_dimension,
        )
    )
    mathematical_reference = apply_factorized_angular_analysis_reference(
        slots,
        execution_plan,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(
        expected,
        sparse_reconstruction,
        rtol=0.0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        actual,
        mathematical_reference,
        rtol=0.0,
        atol=1e-12,
    )

    dense_slots = tuple(
        value.detach().clone().requires_grad_(True)
        for value in slots
    )
    sparse_slots = tuple(
        value.detach().clone().requires_grad_(True)
        for value in slots
    )
    reference.reference_dense_synthesis_materialized = 1
    dense_output = reference(dense_slots)
    dense_gradient = torch.autograd.grad(
        _squared_norm(dense_output),
        dense_slots,
        create_graph=True,
    )
    reference.reference_dense_synthesis_materialized = 0
    sparse_output = reference(sparse_slots)
    sparse_gradient = torch.autograd.grad(
        _squared_norm(sparse_output),
        sparse_slots,
        create_graph=True,
    )
    dense_probes = tuple(torch.randn_like(value) for value in dense_slots)
    dense_hvp = torch.autograd.grad(
        dense_gradient,
        dense_slots,
        dense_probes,
    )
    sparse_hvp = torch.autograd.grad(
        sparse_gradient,
        sparse_slots,
        dense_probes,
    )
    torch.testing.assert_close(
        dense_output,
        sparse_output,
        rtol=0.0,
        atol=1e-12,
    )
    for dense_value, sparse_value in zip(dense_gradient, sparse_gradient):
        torch.testing.assert_close(
            dense_value,
            sparse_value,
            rtol=0.0,
            atol=1e-12,
        )
    for dense_value, sparse_value in zip(dense_hvp, sparse_hvp):
        torch.testing.assert_close(
            dense_value,
            sparse_value,
            rtol=0.0,
            atol=1e-12,
        )
    reference.reference_dense_synthesis_materialized = 1

    def evaluate(*values):
        return native(values)

    assert torch.autograd.gradcheck(
        evaluate,
        slots,
        eps=1e-6,
        atol=1e-9,
        rtol=1e-7,
    )
    assert torch.autograd.gradgradcheck(
        evaluate,
        slots,
        eps=1e-6,
        atol=1e-9,
        rtol=1e-7,
    )
    native_slots = tuple(
        value.detach().clone().requires_grad_(True)
        for value in slots
    )
    reference_slots = tuple(
        value.detach().clone().requires_grad_(True)
        for value in slots
    )
    native_gradient = torch.autograd.grad(
        _squared_norm(native(native_slots)),
        native_slots,
        create_graph=True,
    )
    reference_gradient = torch.autograd.grad(
        _squared_norm(reference(reference_slots)),
        reference_slots,
        create_graph=True,
    )
    probes = tuple(torch.randn_like(value) for value in native_slots)
    native_hvp = torch.autograd.grad(
        native_gradient,
        native_slots,
        probes,
    )
    reference_hvp = torch.autograd.grad(
        reference_gradient,
        reference_slots,
        probes,
    )
    for actual_hvp, expected_hvp in zip(native_hvp, reference_hvp):
        torch.testing.assert_close(
            actual_hvp,
            expected_hvp,
            rtol=2e-6,
            atol=2e-8,
        )


@pytest.mark.parametrize("complex_values", [False, True])
def test_native_factorized_angular_handles_nontrivial_child_tableaux(
    monkeypatch,
    complex_values,
):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    execution_plan = _rank3_nontrivial_child_tableau_plan()
    dtype = torch.complex128 if complex_values else torch.float64
    native = YE3TFactorizedAngularModule(
        execution_plan,
        backend="native",
        dtype=dtype,
    )
    reference = YE3TFactorizedAngularModule(
        execution_plan,
        backend="reference",
        dtype=dtype,
    )
    assert native.source_dimension == 2
    slots = tuple(
        torch.randn(
            2,
            native.source_dimension,
            3,
            dtype=dtype,
            requires_grad=True,
        )
        for _ in range(3)
    )

    torch.testing.assert_close(
        native(slots),
        reference(slots),
        rtol=0.0,
        atol=1.0e-12,
    )
    assert torch.autograd.gradcheck(
        lambda *values: native(values),
        slots,
        eps=1e-6,
        atol=1e-9,
        rtol=1e-7,
    )
    assert torch.autograd.gradgradcheck(
        lambda *values: native(values),
        slots,
        eps=1e-6,
        atol=1e-9,
        rtol=1e-7,
    )


def test_native_factorized_angular_operator_passes_opcheck(monkeypatch):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    module = YE3TFactorizedAngularModule(
        _rank3_factorized_plan(),
        backend="native",
        dtype=torch.float64,
    )
    packed = torch.randn(
        2,
        module.input_dimension,
        dtype=torch.float64,
        requires_grad=True,
    )
    arguments = (
        packed,
        module.node_offsets,
        module.node_dimensions,
        module.node_leaf_offsets,
        module.node_left,
        module.node_right,
        module.node_coefficient_offsets,
        module.coefficient_rows,
        module.coefficient_columns,
        module.coefficient_values,
        module.root_nodes,
        module.projection_values,
        module.source_dimension,
        module.workspace_dimension,
        module.output_dimension,
    )

    result = torch.library.opcheck(
        torch.ops.ye3t_runtime.factorized_angular.default,
        arguments,
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in result.values()), result


@pytest.mark.parametrize("complex_values", [False, True])
def test_native_factorized_linear_readout_matches_unfused_and_derivatives(
    monkeypatch,
    complex_values,
):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    dtype = torch.complex128 if complex_values else torch.float64
    module = YE3TFactorizedAngularModule(
        _rank3_factorized_plan(),
        backend="native",
        dtype=dtype,
    )
    slots = tuple(
        torch.randn(
            2,
            2 * angular_L + 1,
            dtype=dtype,
            requires_grad=True,
        )
        for angular_L in (1, 2, 1)
    )
    weight = torch.randn(
        module.output_dimension,
        dtype=dtype,
        requires_grad=True,
    )
    bias = torch.randn((), dtype=dtype, requires_grad=True)
    actual = module.linear_readout(slots, weight, bias)
    expected = module(slots).reshape(2, -1) @ weight + bias
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)

    def evaluate(*values):
        return module.linear_readout(values[:-2], values[-2], values[-1])

    arguments = slots + (weight, bias)
    assert torch.autograd.gradcheck(
        evaluate,
        arguments,
        eps=1e-6,
        atol=1e-9,
        rtol=1e-7,
    )
    assert torch.autograd.gradgradcheck(
        evaluate,
        arguments,
        eps=1e-6,
        atol=2e-8,
        rtol=2e-6,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_factorized_angular_empty_batch_preserves_shapes_and_autograd(
    monkeypatch,
    device,
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    module = YE3TFactorizedAngularModule(
        _rank3_factorized_plan(),
        backend="native",
        dtype=torch.complex64,
        device=device,
    )
    slots = tuple(
        torch.empty(
            0,
            2 * angular_L + 1,
            dtype=torch.complex64,
            device=device,
            requires_grad=True,
        )
        for angular_L in (1, 2, 1)
    )

    output = module(slots)
    assert output.shape == (
        0,
        module.logical_channel_tableau_dimension,
        module.magnetic_dimension,
    )
    output.real.sum().backward()
    for slot in slots:
        assert slot.grad is not None
        assert slot.grad.shape == slot.shape

    linear_slots = tuple(slot.detach().clone().requires_grad_(True) for slot in slots)
    weight = torch.randn(
        module.output_dimension,
        dtype=torch.complex64,
        device=device,
        requires_grad=True,
    )
    bias = torch.randn(
        (), dtype=torch.complex64, device=device, requires_grad=True
    )
    linear = module.linear_readout(linear_slots, weight, bias)
    assert linear.shape == (0,)
    assert module.last_linear_readout_backend == "empty_batch"
    linear.real.sum().backward()
    for slot in linear_slots:
        assert slot.grad is not None
        assert slot.grad.shape == slot.shape


def test_native_factorized_linear_readout_opcheck_and_compile(monkeypatch):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    module = YE3TFactorizedAngularModule(
        _rank3_factorized_plan(),
        backend="native",
        dtype=torch.float64,
    )
    slots = tuple(
        torch.randn(
            2,
            2 * angular_L + 1,
            dtype=torch.float64,
            requires_grad=True,
        )
        for angular_L in (1, 2, 1)
    )
    packed, _ = module._packed_slots(slots)
    weight = torch.randn(
        module.output_dimension,
        dtype=torch.float64,
        requires_grad=True,
    )
    bias = torch.randn((), dtype=torch.float64, requires_grad=True)
    arguments = (
        packed,
        *module._native_arguments(),
        module.source_dimension,
        module.workspace_dimension,
        weight,
        bias,
    )
    result = torch.library.opcheck(
        torch.ops.ye3t_runtime.factorized_angular_linear.default,
        arguments,
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in result.values()), result

    def evaluate(packed_value, weight_value, bias_value):
        return torch.ops.ye3t_runtime.factorized_angular_linear(
            packed_value,
            *module._native_arguments(),
            module.source_dimension,
            module.workspace_dimension,
            weight_value,
            bias_value,
        )

    eager = evaluate(packed, weight, bias)
    compiled = torch.compile(evaluate, backend="eager", fullgraph=True)
    actual = compiled(packed, weight, bias)
    torch.testing.assert_close(actual, eager, rtol=0.0, atol=1e-12)


def test_native_factorized_role_coordinates_match_reference_and_derivatives(
    monkeypatch,
):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    execution_plan = _rank3_factorized_role_plan()
    native = YE3TFactorizedAngularModule(
        execution_plan,
        backend="native",
        dtype=torch.float64,
    )
    reference = YE3TFactorizedAngularModule(
        execution_plan,
        backend="reference",
        dtype=torch.float64,
    )
    assert native.source_dimension > 1
    slots = tuple(
        torch.randn(
            1,
            native.source_dimension,
            2 * angular_L + 1,
            dtype=torch.float64,
            requires_grad=True,
        )
        for angular_L in (1, 2, 1)
    )

    actual = native(slots)
    expected = reference(slots)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)
    mathematical_reference = apply_factorized_angular_analysis_reference(
        slots,
        execution_plan,
    )
    torch.testing.assert_close(
        actual,
        mathematical_reference,
        rtol=0.0,
        atol=1e-12,
    )
    assert execution_plan.carrier_layouts[0].key.partition == (2, 1)

    def evaluate(*values):
        return native(values)

    assert torch.autograd.gradcheck(
        evaluate,
        slots,
        eps=1e-6,
        atol=1e-9,
        rtol=1e-7,
    )
    assert torch.autograd.gradgradcheck(
        evaluate,
        slots,
        eps=1e-6,
        atol=1e-9,
        rtol=1e-7,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_native_factorized_plan_group_preserves_distinct_role_sources_and_derivatives(
    monkeypatch,
    device,
):
    if device == "cuda" and (
        not torch.cuda.is_available() or not _load_extension().has_cuda()
    ):
        pytest.skip("requires native CUDA")
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    execution_plan = _rank3_factorized_role_plan()
    native_modules = tuple(
        YE3TFactorizedAngularModule(
            execution_plan,
            backend="native",
            dtype=torch.complex128,
            device=device,
        )
        for _ in range(2)
    )
    reference_modules = tuple(
        YE3TFactorizedAngularModule(
            execution_plan,
            backend="reference",
            dtype=torch.complex128,
            device=device,
        )
        for _ in range(2)
    )
    grouped = _YE3TFactorizedAngularModuleGroup(
        native_modules,
        enable_native_grouping=True,
    )
    source_dimension = int(native_modules[0].source_dimension)
    values_by_plan = tuple(
        tuple(
            torch.randn(
                2,
                source_dimension,
                2 * angular_L + 1,
                dtype=torch.complex128,
                device=device,
            )
            for angular_L in (1, 2, 1)
        )
        for _ in range(2)
    )
    assert not torch.equal(values_by_plan[0][0], values_by_plan[1][0])
    native_slots = tuple(
        tuple(value.clone().requires_grad_(True) for value in values)
        for values in values_by_plan
    )
    reference_slots = tuple(
        tuple(value.clone().requires_grad_(True) for value in values)
        for values in values_by_plan
    )

    actual = grouped(native_slots)
    expected = tuple(
        module(values)
        for module, values in zip(reference_modules, reference_slots)
    )
    for value, expected_value in zip(actual, expected):
        torch.testing.assert_close(
            value,
            expected_value,
            rtol=2e-10,
            atol=2e-10,
        )
    report = grouped.runtime_report()
    assert report["native_call_count"] == 1
    assert report["groups"][0]["member_plan_indices"] == (0, 1)
    assert report["groups"][0]["member_count"] == 2
    assert report["groups"][0]["single_native_call"]
    assert not report["subtree_evaluation_shared"]

    native_arguments = tuple(
        value for values in native_slots for value in values
    )
    reference_arguments = tuple(
        value for values in reference_slots for value in values
    )
    native_gradients = torch.autograd.grad(
        sum(_squared_norm(value) for value in actual),
        native_arguments,
        create_graph=True,
    )
    reference_gradients = torch.autograd.grad(
        sum(_squared_norm(value) for value in expected),
        reference_arguments,
        create_graph=True,
    )
    for value, expected_value in zip(
        native_gradients,
        reference_gradients,
    ):
        torch.testing.assert_close(
            value,
            expected_value,
            rtol=2e-9,
            atol=2e-9,
        )
    probes = tuple(torch.randn_like(value) for value in native_arguments)
    native_hvp = torch.autograd.grad(
        native_gradients,
        native_arguments,
        probes,
    )
    reference_hvp = torch.autograd.grad(
        reference_gradients,
        reference_arguments,
        probes,
    )
    for value, expected_value in zip(native_hvp, reference_hvp):
        torch.testing.assert_close(
            value,
            expected_value,
            rtol=2e-8,
            atol=2e-8,
        )


@pytest.mark.parametrize(
    ("device", "cuda_policy"),
    (
        ("cpu", None),
        ("cuda", "auto"),
        ("cuda", "serial"),
        ("cuda", "warp"),
    ),
)
def test_native_segmented_factorized_group_preserves_ragged_source_images(
    monkeypatch,
    device,
    cuda_policy,
):
    if device == "cuda" and (
        not torch.cuda.is_available() or not _load_extension().has_cuda()
    ):
        pytest.skip("requires native CUDA")
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    if cuda_policy is not None:
        monkeypatch.setenv(
            "YE3T_FACTORIZED_ANGULAR_SEGMENTED_CUDA_POLICY",
            cuda_policy,
        )
    plans = (
        _rank3_factorized_plan(),
        _rank3_factorized_role_plan(),
        _rank5_factorized_plan(),
    )
    native_modules = tuple(
        YE3TFactorizedAngularModule(
            plan,
            backend="native",
            dtype=torch.complex128,
            device=device,
        )
        for plan in plans
    )
    reference_modules = tuple(
        YE3TFactorizedAngularModule(
            plan,
            backend="reference",
            dtype=torch.complex128,
            device=device,
        )
        for plan in plans
    )
    grouped = _YE3TSegmentedFactorizedAngularModuleGroup(
        native_modules,
        plan_indices=(7, 11, 13),
    )
    assert len(set(grouped.runtime_report()["source_dimensions"])) == 2

    values_by_plan = []
    for module in native_modules:
        values = []
        for angular_L in module.input_Ls:
            shape = (2, 2 * angular_L + 1)
            if int(module.source_dimension) != 1:
                shape = (2, int(module.source_dimension), shape[-1])
            values.append(
                torch.randn(
                    shape,
                    dtype=torch.complex128,
                    device=device,
                )
            )
        values_by_plan.append(tuple(values))
    native_slots = tuple(
        tuple(value.clone().requires_grad_(True) for value in values)
        for values in values_by_plan
    )
    reference_slots = tuple(
        tuple(value.clone().requires_grad_(True) for value in values)
        for values in values_by_plan
    )

    actual = grouped.forward_packed(native_slots)
    reference_values = tuple(
        module(values)
        for module, values in zip(reference_modules, reference_slots)
    )
    expected = torch.cat(
        tuple(value.reshape(2, -1) for value in reference_values),
        dim=1,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-10)
    report = grouped.runtime_report()
    assert report["native_call_count"] == 1
    assert report["member_plan_indices"] == (7, 11, 13)
    assert not report["runtime_path_discovery"]
    assert report["cuda_auto_policy"] == "hybrid"
    assert report["cuda_auto_warp_min_coefficient_count"] == 128
    assert tuple(
        record["plan_index"]
        for record in report["cuda_auto_segments"]
    ) == (7, 11, 13)
    assert all(
        record["policy"]
        == (
            "warp"
            if record["coefficient_count"] >= 128
            else "serial"
        )
        for record in report["cuda_auto_segments"]
    )

    native_arguments = tuple(
        value for values in native_slots for value in values
    )
    reference_arguments = tuple(
        value for values in reference_slots for value in values
    )
    native_gradients = torch.autograd.grad(
        _squared_norm(actual),
        native_arguments,
        create_graph=True,
    )
    reference_gradients = torch.autograd.grad(
        _squared_norm(expected),
        reference_arguments,
        create_graph=True,
    )
    for value, expected_value in zip(
        native_gradients,
        reference_gradients,
    ):
        torch.testing.assert_close(
            value,
            expected_value,
            rtol=2e-9,
            atol=2e-9,
        )
    probes = tuple(torch.randn_like(value) for value in native_arguments)
    native_hvp = torch.autograd.grad(
        native_gradients,
        native_arguments,
        probes,
    )
    reference_hvp = torch.autograd.grad(
        reference_gradients,
        reference_arguments,
        probes,
    )
    for value, expected_value in zip(native_hvp, reference_hvp):
        torch.testing.assert_close(
            value,
            expected_value,
            rtol=2e-8,
            atol=2e-8,
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_native_factorized_plan_group_consumes_prepacked_physical_role_bank(
    monkeypatch,
    device,
):
    if device == "cuda" and (
        not torch.cuda.is_available() or not _load_extension().has_cuda()
    ):
        pytest.skip("requires native CUDA")
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    execution_plan = _rank3_factorized_role_plan()
    native_modules = tuple(
        YE3TFactorizedAngularModule(
            execution_plan,
            backend="native",
            dtype=torch.complex128,
            device=device,
        )
        for _ in range(2)
    )
    reference_modules = tuple(
        YE3TFactorizedAngularModule(
            execution_plan,
            backend="reference",
            dtype=torch.complex128,
            device=device,
        )
        for _ in range(2)
    )
    leaf_offsets = (
        (0, 3, 8),
        (11, 14, 19),
    )
    physical_input_dimension = 22
    grouped = _YE3TFactorizedAngularModuleGroup(
        native_modules,
        enable_native_grouping=True,
        physical_leaf_offsets_by_plan=leaf_offsets,
        physical_input_dimension=physical_input_dimension,
    )
    source_dimension = int(native_modules[0].source_dimension)
    values = torch.randn(
        2,
        source_dimension,
        physical_input_dimension,
        dtype=torch.complex128,
        device=device,
    )
    native_bank = values.clone().reshape(2, -1).requires_grad_(True)
    reference_bank = (
        values.clone().reshape(2, -1).requires_grad_(True)
    )

    def reference_outputs(bank):
        source_rows = bank.reshape(
            2,
            source_dimension,
            physical_input_dimension,
        )
        outputs = []
        for module, offsets in zip(reference_modules, leaf_offsets):
            slots = tuple(
                source_rows[
                    ...,
                    int(offset):int(offset) + 2 * angular_L + 1,
                ]
                for offset, angular_L in zip(
                    offsets,
                    module.input_Ls,
                )
            )
            outputs.append(module(slots))
        return tuple(outputs)

    actual = grouped(
        None,
        packed_source_bank=native_bank,
        leading_shape=(2,),
    )
    expected = reference_outputs(reference_bank)
    for value, expected_value in zip(actual, expected):
        torch.testing.assert_close(
            value,
            expected_value,
            rtol=2e-10,
            atol=2e-10,
        )
    packed_actual = grouped.forward_packed(
        None,
        packed_source_bank=native_bank,
        leading_shape=(2,),
    )
    expected_packed = torch.cat(
        tuple(value.reshape(2, -1) for value in expected),
        dim=1,
    )
    torch.testing.assert_close(
        packed_actual,
        expected_packed,
        rtol=2e-10,
        atol=2e-10,
    )
    report = grouped.runtime_report()
    assert report["prepacked_physical_source_bank"]
    assert report["physical_input_dimension"] == 22
    assert report["groups"][0]["prepacked_physical_source_bank"]

    native_gradient = torch.autograd.grad(
        _squared_norm(packed_actual),
        native_bank,
        create_graph=True,
    )[0]
    reference_gradient = torch.autograd.grad(
        sum(_squared_norm(value) for value in expected),
        reference_bank,
        create_graph=True,
    )[0]
    torch.testing.assert_close(
        native_gradient,
        reference_gradient,
        rtol=2e-9,
        atol=2e-9,
    )
    probe = torch.randn_like(native_bank)
    native_hvp = torch.autograd.grad(
        native_gradient,
        native_bank,
        probe,
    )[0]
    reference_hvp = torch.autograd.grad(
        reference_gradient,
        reference_bank,
        probe,
    )[0]
    torch.testing.assert_close(
        native_hvp,
        reference_hvp,
        rtol=2e-8,
        atol=2e-8,
    )


@pytest.mark.parametrize(
    "device,double_backward_policy,expected_double_backward_backend",
    (
        (
            "cpu",
            "auto",
            "exact_factorized_double_backward_fallback",
        ),
        (
            "cuda",
            "auto",
            "native_destination_shared_power_factored_double_backward",
        ),
        (
            "cuda",
            "native",
            "native_destination_shared_power_factored_double_backward",
        ),
        (
            "cuda",
            "reference",
            "exact_factorized_double_backward_fallback",
        ),
    ),
)
def test_grouped_homogeneous_power_lowering_preserves_values_vjp_and_hvp(
    monkeypatch,
    device,
    double_backward_policy,
    expected_double_backward_backend,
):
    if device == "cuda" and (
        not torch.cuda.is_available() or not _load_extension().has_cuda()
    ):
        pytest.skip("requires native CUDA")
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    monkeypatch.setenv(
        "YE3T_SYMMETRIC_POWER_MONOMIAL_DOUBLE_BACKWARD_POLICY",
        double_backward_policy,
    )
    plans = tuple(
        _rank4_homogeneous_factorized_role_plan(target_L)
        for target_L in (0, 2)
    )
    native_modules = tuple(
        YE3TFactorizedAngularModule(
            plan,
            backend="native",
            dtype=torch.complex128,
            device=device,
        )
        for plan in plans
    )
    reference_modules = tuple(
        YE3TFactorizedAngularModule(
            plan,
            backend="reference",
            dtype=torch.complex128,
            device=device,
        )
        for plan in plans
    )
    leaf_offsets = ((0, 0, 0, 0), (0, 0, 0, 0))
    grouped = _YE3TFactorizedAngularModuleGroup(
        native_modules,
        enable_native_grouping=True,
        physical_leaf_offsets_by_plan=leaf_offsets,
        physical_input_dimension=3,
        enable_heterogeneous_root_grouping=True,
        enable_homogeneous_power_grouping=True,
    )
    power_group = grouped.power_groups[0]
    double_backward_backends = []
    original_power_double_backward = power_group._power_double_backward

    def record_power_double_backward(*args, **kwargs):
        result = original_power_double_backward(*args, **kwargs)
        double_backward_backends.append(
            power_group.last_derivative_backend
        )
        return result

    monkeypatch.setattr(
        power_group,
        "_power_double_backward",
        record_power_double_backward,
    )
    native_values = torch.randn(
        3,
        3,
        dtype=torch.complex128,
        device=device,
    )
    native_values[0, 0] = 0
    native_values[1] = 0
    native_bank = native_values.requires_grad_(True)
    reference_bank = native_bank.detach().clone().requires_grad_(True)

    actual = grouped(
        None,
        packed_source_bank=native_bank,
        leading_shape=(3,),
    )
    expected = tuple(
        module(tuple(reference_bank for _slot in range(4)))
        for module in reference_modules
    )
    for value, expected_value in zip(actual, expected):
        torch.testing.assert_close(
            value,
            expected_value,
            rtol=2e-10,
            atol=2e-10,
        )
    packed_actual = grouped.forward_packed(
        None,
        packed_source_bank=native_bank,
        leading_shape=(3,),
    )
    expected_packed = torch.cat(
        tuple(value.reshape(3, -1) for value in expected),
        dim=1,
    )
    torch.testing.assert_close(
        packed_actual,
        expected_packed,
        rtol=2e-10,
        atol=2e-10,
    )
    report = grouped.runtime_report()
    assert report["homogeneous_power_plan_indices"] == (0, 1)
    assert report["homogeneous_power_group_count"] == 1
    assert report["native_call_count"] == 1
    assert report["power_groups"][0]["powers"] == (4, 4)
    assert report["power_groups"][0]["monomial_count"] == 11
    assert report["power_groups"][0]["double_backward_term_count"] == (
        report["power_groups"][0]["monomial_count"]
    )
    assert report["power_groups"][0][
        "double_backward_coefficient_count"
    ] == report["power_groups"][0]["coefficient_count"]
    assert report["power_groups"][0]["double_backward_table_bytes"] == 0

    native_gradient = torch.autograd.grad(
        _squared_norm(packed_actual),
        native_bank,
        create_graph=True,
    )[0]
    reference_gradient = torch.autograd.grad(
        sum(_squared_norm(value) for value in expected),
        reference_bank,
        create_graph=True,
    )[0]
    torch.testing.assert_close(
        native_gradient,
        reference_gradient,
        rtol=2e-9,
        atol=2e-9,
    )
    probe = torch.randn_like(native_bank)
    native_hvp = torch.autograd.grad(
        native_gradient,
        native_bank,
        probe,
    )[0]
    reference_hvp = torch.autograd.grad(
        reference_gradient,
        reference_bank,
        probe,
    )[0]
    torch.testing.assert_close(
        native_hvp,
        reference_hvp,
        rtol=2e-8,
        atol=2e-8,
    )
    assert double_backward_backends[-1] == (
        expected_double_backward_backend
    )


@pytest.mark.parametrize(
    "device,double_backward_policy,expected_double_backward_backend",
    (
        (
            "cpu",
            "auto",
            "exact_destination_grouped_factorized_double_backward_fallback",
        ),
        (
            "cuda",
            "auto",
            "native_destination_grouped_shared_power_double_backward",
        ),
        (
            "cuda",
            "reference",
            "exact_destination_grouped_factorized_double_backward_fallback",
        ),
    ),
)
def test_repeated_power_destination_group_consolidates_exact_hvp_calls(
    monkeypatch,
    device,
    double_backward_policy,
    expected_double_backward_backend,
):
    if device == "cuda" and (
        not torch.cuda.is_available() or not _load_extension().has_cuda()
    ):
        pytest.skip("requires native CUDA")
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    monkeypatch.setenv(
        "YE3T_SYMMETRIC_POWER_MONOMIAL_DOUBLE_BACKWARD_POLICY",
        double_backward_policy,
    )
    plans = tuple(
        _rank4_homogeneous_factorized_role_plan(target_L)
        for target_L in (0, 2, 0, 2)
    )
    individual_modules = tuple(
        YE3TFactorizedAngularModule(
            plan,
            backend="native",
            dtype=torch.complex128,
            device=device,
        )
        for plan in plans
    )
    destination_modules = tuple(
        YE3TFactorizedAngularModule(
            plan,
            backend="native",
            dtype=torch.complex128,
            device=device,
        )
        for plan in plans
    )
    first_indices = (0, 1)
    second_indices = (2, 3)
    first_offsets = ((0, 0, 0, 0), (0, 0, 0, 0))
    second_offsets = ((0, 0, 0, 0), (0, 0, 0, 0))
    individual_groups = (
        _PackedRepeatedBlockPowerPlanGroup(
            tuple(individual_modules[index] for index in first_indices),
            first_indices,
            power_physical_leaf_offsets_by_plan=first_offsets,
            exact_physical_leaf_offsets_by_plan=first_offsets,
            physical_input_dimension=3,
        ),
        _PackedRepeatedBlockPowerPlanGroup(
            tuple(individual_modules[index] for index in second_indices),
            second_indices,
            power_physical_leaf_offsets_by_plan=second_offsets,
            exact_physical_leaf_offsets_by_plan=second_offsets,
            physical_input_dimension=3,
        ),
    )
    destination_power_groups = (
        _PackedRepeatedBlockPowerPlanGroup(
            tuple(destination_modules[index] for index in first_indices),
            first_indices,
            power_physical_leaf_offsets_by_plan=first_offsets,
            exact_physical_leaf_offsets_by_plan=first_offsets,
            physical_input_dimension=3,
            defer_exact_adjoint_group=True,
        ),
        _PackedRepeatedBlockPowerPlanGroup(
            tuple(destination_modules[index] for index in second_indices),
            second_indices,
            power_physical_leaf_offsets_by_plan=second_offsets,
            exact_physical_leaf_offsets_by_plan=second_offsets,
            physical_input_dimension=3,
            defer_exact_adjoint_group=True,
        ),
    )
    destination = _PackedRepeatedBlockPowerDerivativeDestinationGroup(
        destination_power_groups
    )
    destination_backends = []
    original_destination_double_backward = (
        destination._power_double_backward
    )

    def record_destination_double_backward(*args, **kwargs):
        result = original_destination_double_backward(*args, **kwargs)
        destination_backends.append(
            destination.last_derivative_backend
        )
        return result

    monkeypatch.setattr(
        destination,
        "_power_double_backward",
        record_destination_double_backward,
    )
    report = destination.runtime_report()
    assert report["member_power_group_count"] == 2
    assert report["prior_exact_derivative_call_count"] == 2
    assert report["exact_derivative_call_count"] == 1

    destination_bank = torch.randn(
        3,
        3,
        dtype=torch.complex128,
        device=device,
        requires_grad=True,
    )
    individual_bank = (
        destination_bank.detach().clone().requires_grad_(True)
    )
    destination_output = destination.forward_prepacked_output(
        destination_bank,
        (3,),
    )
    individual_output = torch.cat(
        tuple(
            group.forward_prepacked_output(individual_bank, (3,))
            for group in individual_groups
        ),
        dim=1,
    )
    torch.testing.assert_close(
        destination_output,
        individual_output,
        rtol=2e-10,
        atol=2e-10,
    )
    destination_gradient = torch.autograd.grad(
        _squared_norm(destination_output),
        destination_bank,
        create_graph=True,
    )[0]
    individual_gradient = torch.autograd.grad(
        _squared_norm(individual_output),
        individual_bank,
        create_graph=True,
    )[0]
    torch.testing.assert_close(
        destination_gradient,
        individual_gradient,
        rtol=2e-9,
        atol=2e-9,
    )
    probe = torch.randn_like(destination_bank)
    destination_hvp = torch.autograd.grad(
        destination_gradient,
        destination_bank,
        probe,
    )[0]
    individual_hvp = torch.autograd.grad(
        individual_gradient,
        individual_bank,
        probe,
    )[0]
    torch.testing.assert_close(
        destination_hvp,
        individual_hvp,
        rtol=2e-8,
        atol=2e-8,
    )
    assert destination_backends[-1] == (
        expected_double_backward_backend
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_factorized_group_destination_derivative_dispatch_is_optional(
    monkeypatch,
    device,
):
    if device == "cuda" and (
        not torch.cuda.is_available() or not _load_extension().has_cuda()
    ):
        pytest.skip("requires native CUDA")
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    plans = tuple(
        _rank4_homogeneous_factorized_role_plan(target_L)
        for target_L in (0, 2)
    )
    modules = tuple(
        YE3TFactorizedAngularModule(
            plan,
            backend="native",
            dtype=torch.complex128,
            device=device,
        )
        for plan in plans
    )
    grouped = _YE3TFactorizedAngularModuleGroup(
        modules,
        enable_native_grouping=True,
        physical_leaf_offsets_by_plan=(
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        ),
        physical_input_dimension=3,
        enable_heterogeneous_root_grouping=True,
        enable_homogeneous_power_grouping=True,
        enable_destination_derivative_grouping=True,
    )
    report = grouped.runtime_report()
    assert report["destination_derivative_grouping_requested"]
    assert report["repeated_block_power_group_count"] == 1
    assert report["destination_derivative_group_count"] == 0
    assert report["exact_power_derivative_call_count"] == 1


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_grouped_two_block_power_lowering_preserves_values_vjp_and_hvp(
    monkeypatch,
    device,
):
    if device == "cuda" and (
        not torch.cuda.is_available() or not _load_extension().has_cuda()
    ):
        pytest.skip("requires native CUDA")
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    plans = tuple(
        _rank4_two_block_factorized_role_plan(target_L)
        for target_L in (0, 1)
    )
    native_modules = tuple(
        YE3TFactorizedAngularModule(
            plan,
            backend="native",
            dtype=torch.complex128,
            device=device,
        )
        for plan in plans
    )
    reference_modules = tuple(
        YE3TFactorizedAngularModule(
            plan,
            backend="reference",
            dtype=torch.complex128,
            device=device,
        )
        for plan in plans
    )
    source_dimension = int(native_modules[0].source_dimension)
    assert source_dimension > 1
    leaf_offsets = ((0, 0, 3, 3), (0, 0, 3, 3))
    grouped = _YE3TFactorizedAngularModuleGroup(
        native_modules,
        enable_native_grouping=True,
        physical_leaf_offsets_by_plan=leaf_offsets,
        physical_input_dimension=6,
        enable_heterogeneous_root_grouping=True,
        enable_homogeneous_power_grouping=True,
        repeated_block_power_policy="force",
    )
    native_bank = torch.randn(
        3,
        source_dimension * 6,
        dtype=torch.complex128,
        device=device,
        requires_grad=True,
    )
    reference_bank = native_bank.detach().clone().requires_grad_(True)
    shaped_reference = reference_bank.reshape(3, source_dimension, 6)
    block_a = shaped_reference[:, :, :3]
    block_b = shaped_reference[:, :, 3:]

    packed_actual = grouped.forward_packed(
        None,
        packed_source_bank=native_bank,
        leading_shape=(3,),
    )
    expected = tuple(
        module((block_a, block_a, block_b, block_b))
        for module in reference_modules
    )
    expected_packed = torch.cat(
        tuple(value.reshape(3, -1) for value in expected),
        dim=1,
    )
    torch.testing.assert_close(
        packed_actual,
        expected_packed,
        rtol=2e-10,
        atol=2e-10,
    )
    report = grouped.runtime_report()
    assert report["repeated_block_power_plan_indices"] == (0, 1)
    assert report["repeated_block_power_group_count"] == 1
    assert report["native_call_count"] == 1
    assert report["power_groups"][0]["source_dimension"] == source_dimension
    assert report["power_groups"][0]["block_multiplicities"] == (2, 2)

    native_gradient = torch.autograd.grad(
        _squared_norm(packed_actual),
        native_bank,
        create_graph=True,
    )[0]
    reference_gradient = torch.autograd.grad(
        sum(_squared_norm(value) for value in expected),
        reference_bank,
        create_graph=True,
    )[0]
    torch.testing.assert_close(
        native_gradient,
        reference_gradient,
        rtol=2e-9,
        atol=2e-9,
    )
    probe = torch.randn_like(native_bank)
    native_hvp = torch.autograd.grad(
        native_gradient,
        native_bank,
        probe,
    )[0]
    reference_hvp = torch.autograd.grad(
        reference_gradient,
        reference_bank,
        probe,
    )[0]
    torch.testing.assert_close(
        native_hvp,
        reference_hvp,
        rtol=2e-8,
        atol=2e-8,
    )


def test_grouped_two_block_power_auto_policy_uses_exact_shared_subtrees(
    monkeypatch,
):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    plans = tuple(
        _rank4_two_block_factorized_role_plan(target_L)
        for target_L in (0, 1)
    )
    modules = tuple(
        YE3TFactorizedAngularModule(
            plan,
            backend="native",
            dtype=torch.complex128,
            device="cpu",
        )
        for plan in plans
    )
    leaf_offsets = ((0, 0, 3, 3), (0, 0, 3, 3))
    automatic = _YE3TFactorizedAngularModuleGroup(
        modules,
        enable_native_grouping=True,
        physical_leaf_offsets_by_plan=leaf_offsets,
        physical_input_dimension=6,
        enable_heterogeneous_root_grouping=True,
        enable_homogeneous_power_grouping=True,
        repeated_block_power_policy="auto",
    )
    exact = _YE3TFactorizedAngularModuleGroup(
        modules,
        enable_native_grouping=True,
        physical_leaf_offsets_by_plan=leaf_offsets,
        physical_input_dimension=6,
        enable_heterogeneous_root_grouping=True,
        enable_homogeneous_power_grouping=False,
    )
    source_dimension = int(modules[0].source_dimension)
    bank = torch.randn(3, source_dimension * 6, dtype=torch.complex128)
    torch.testing.assert_close(
        automatic.forward_packed(
            None,
            packed_source_bank=bank,
            leading_shape=(3,),
        ),
        exact.forward_packed(
            None,
            packed_source_bank=bank,
            leading_shape=(3,),
        ),
        rtol=0.0,
        atol=2e-12,
    )
    report = automatic.runtime_report()
    assert report["repeated_block_power_policy"] == "auto"
    assert report["repeated_block_power_group_count"] == 0
    assert report["native_call_count"] == 1
    assert len(report["groups"]) == 1
    assert report["power_skip_records"] == (
        {
            "plan_index": 0,
            "reason": "multi_block_exact_shared_subtree_preferred",
            "block_count": 2,
        },
        {
            "plan_index": 1,
            "reason": "multi_block_exact_shared_subtree_preferred",
            "block_count": 2,
        },
    )


@pytest.mark.parametrize("rank", (2, 3, 4, 6, 8, 12, 16, 32))
def test_repeated_block_layout_is_rank_general_and_byte_reported(rank):
    module = SimpleNamespace(input_Ls=(1,) * rank, source_dimension=1)
    homogeneous = _repeated_block_power_layout(
        module,
        (0,) * rank,
        3,
    )
    assert homogeneous["block_multiplicities"] == (rank,)
    assert homogeneous["repeated_slot_count"] == rank
    assert homogeneous["estimated_count_table_bytes"] > 0

    first_block = int((rank + 1) // 2)
    second_block = int(rank - first_block)
    balanced = _repeated_block_power_layout(
        module,
        (0,) * first_block + (3,) * second_block,
        6,
    )
    assert balanced["block_multiplicities"] == (
        first_block,
        second_block,
    )
    assert balanced["estimated_count_table_bytes"] > 0
    assert balanced["repeated_slot_count"] == sum(
        value
        for value in (first_block, second_block)
        if value > 1
    )


def test_repeated_block_power_byte_guard_retains_exact_generic_plan(
    monkeypatch,
):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    plans = tuple(
        _rank4_homogeneous_factorized_role_plan(target_L)
        for target_L in (0, 2)
    )
    modules = tuple(
        YE3TFactorizedAngularModule(
            plan,
            backend="native",
            dtype=torch.complex128,
            device="cpu",
        )
        for plan in plans
    )
    leaf_offsets = ((0, 0, 0, 0), (0, 0, 0, 0))
    guarded = _YE3TFactorizedAngularModuleGroup(
        modules,
        enable_native_grouping=True,
        physical_leaf_offsets_by_plan=leaf_offsets,
        physical_input_dimension=3,
        enable_heterogeneous_root_grouping=True,
        enable_homogeneous_power_grouping=True,
        power_table_byte_limit=1,
    )
    exact = _YE3TFactorizedAngularModuleGroup(
        modules,
        enable_native_grouping=True,
        physical_leaf_offsets_by_plan=leaf_offsets,
        physical_input_dimension=3,
        enable_heterogeneous_root_grouping=True,
        enable_homogeneous_power_grouping=False,
    )
    bank = torch.randn(3, 3, dtype=torch.complex128)
    torch.testing.assert_close(
        guarded.forward_packed(
            None,
            packed_source_bank=bank,
            leading_shape=(3,),
        ),
        exact.forward_packed(
            None,
            packed_source_bank=bank,
            leading_shape=(3,),
        ),
        rtol=0.0,
        atol=2e-12,
    )
    report = guarded.runtime_report()
    assert report["repeated_block_power_group_count"] == 0
    assert report["power_skip_records"]
    assert all(
        record["reason"] == "estimated_count_table_bytes_exceeds_limit"
        for record in report["power_skip_records"]
    )


def test_repeated_block_power_guard_includes_sparse_coefficient_storage(
    monkeypatch,
):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    plan = _rank4_homogeneous_factorized_role_plan(2)
    module = YE3TFactorizedAngularModule(
        plan,
        backend="native",
        dtype=torch.complex128,
        device="cpu",
    )
    leaf_offsets = ((0, 0, 0, 0),)
    unbounded = _YE3TFactorizedAngularModuleGroup(
        (module,),
        enable_native_grouping=True,
        physical_leaf_offsets_by_plan=leaf_offsets,
        physical_input_dimension=3,
        enable_heterogeneous_root_grouping=True,
        enable_homogeneous_power_grouping=True,
        power_table_byte_limit=64 * 1024 * 1024,
    )
    power_report = unbounded.runtime_report()["power_groups"][0]
    layout = _repeated_block_power_layout(module, leaf_offsets[0], 3)
    lower_bound = max(
        int(layout["estimated_count_table_bytes"]),
        int(power_report["estimated_count_table_bytes"]),
    )
    assert int(power_report["power_table_bytes"]) > lower_bound

    guarded = _YE3TFactorizedAngularModuleGroup(
        (module,),
        enable_native_grouping=True,
        physical_leaf_offsets_by_plan=leaf_offsets,
        physical_input_dimension=3,
        enable_heterogeneous_root_grouping=True,
        enable_homogeneous_power_grouping=True,
        power_table_byte_limit=lower_bound,
    )
    report = guarded.runtime_report()
    assert report["repeated_block_power_group_count"] == 0
    assert report["power_skip_records"] == (
        {
            "plan_index": 0,
            "reason": "materialized_power_table_bytes_exceeds_limit",
            "power_table_bytes": int(power_report["power_table_bytes"]),
        },
    )
    bank = torch.randn(2, 3, dtype=torch.complex128)
    torch.testing.assert_close(
        guarded.forward_packed(
            None,
            packed_source_bank=bank,
            leading_shape=(2,),
        ),
        unbounded.forward_packed(
            None,
            packed_source_bank=bank,
            leading_shape=(2,),
        ),
        rtol=0.0,
        atol=2e-12,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_native_heterogeneous_factorized_role_group_values_vjp_hvp_and_opcheck(
    monkeypatch,
    device,
):
    if device == "cuda" and (
        not torch.cuda.is_available() or not _load_extension().has_cuda()
    ):
        pytest.skip("requires native CUDA")
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    plans = tuple(
        _rank3_factorized_role_plan(target_L)
        for target_L in (1, 2, 3)
    )
    native_modules = tuple(
        YE3TFactorizedAngularModule(
            plan,
            backend="native",
            dtype=torch.complex128,
            device=device,
        )
        for plan in plans
    )
    reference_modules = tuple(
        YE3TFactorizedAngularModule(
            plan,
            backend="reference",
            dtype=torch.complex128,
            device=device,
        )
        for plan in plans
    )
    leaf_offsets = tuple((0, 3, 8) for _plan in plans)
    physical_input_dimension = 11
    grouped = _YE3TFactorizedAngularModuleGroup(
        native_modules,
        enable_native_grouping=True,
        physical_leaf_offsets_by_plan=leaf_offsets,
        physical_input_dimension=physical_input_dimension,
        enable_heterogeneous_root_grouping=True,
    )
    source_dimension = int(native_modules[0].source_dimension)
    values = torch.randn(
        2,
        source_dimension,
        physical_input_dimension,
        dtype=torch.complex128,
        device=device,
    )
    native_bank = values.clone().reshape(2, -1).requires_grad_(True)
    reference_bank = values.clone().reshape(2, -1).requires_grad_(True)

    def reference_outputs(bank):
        source_rows = bank.reshape(
            2,
            source_dimension,
            physical_input_dimension,
        )
        outputs = []
        for module, offsets in zip(reference_modules, leaf_offsets):
            slots = tuple(
                source_rows[
                    ...,
                    int(offset):int(offset) + 2 * angular_L + 1,
                ]
                for offset, angular_L in zip(offsets, module.input_Ls)
            )
            outputs.append(module(slots))
        return tuple(outputs)

    actual = grouped(
        None,
        packed_source_bank=native_bank,
        leading_shape=(2,),
    )
    expected = reference_outputs(reference_bank)
    for value, expected_value in zip(actual, expected):
        torch.testing.assert_close(
            value,
            expected_value,
            rtol=2e-10,
            atol=2e-10,
        )
    report = grouped.runtime_report()
    assert report["heterogeneous_root_grouped"]
    assert report["native_call_count"] == 1
    assert len(report["groups"]) == 1
    assert report["groups"][0]["heterogeneous_root_dimensions"]
    assert report["subtree_evaluation_shared"]
    assert report["groups"][0]["subtree_evaluation_shared"]
    assert report["groups"][0]["naive_node_count"] == 23
    assert report["groups"][0]["unique_node_count"] == 13
    assert report["groups"][0]["elided_node_evaluations"] == 10
    assert (
        report["groups"][0]["workspace_dimension"]
        < report["groups"][0]["naive_workspace_dimension"]
    )
    assert report["groups"][0]["composition"] == (
        "exact_shared_subtree_heterogeneous_root_dag"
    )
    assert report["groups"][0]["member_plan_hashes"] == tuple(
        module.plan_hash for module in native_modules
    )

    native_gradient = torch.autograd.grad(
        sum(_squared_norm(value) for value in actual),
        native_bank,
        create_graph=True,
    )[0]
    reference_gradient = torch.autograd.grad(
        sum(_squared_norm(value) for value in expected),
        reference_bank,
        create_graph=True,
    )[0]
    torch.testing.assert_close(
        native_gradient,
        reference_gradient,
        rtol=2e-9,
        atol=2e-9,
    )
    probe = torch.randn_like(native_bank)
    native_hvp = torch.autograd.grad(
        native_gradient,
        native_bank,
        probe,
    )[0]
    reference_hvp = torch.autograd.grad(
        reference_gradient,
        reference_bank,
        probe,
    )[0]
    torch.testing.assert_close(
        native_hvp,
        reference_hvp,
        rtol=2e-8,
        atol=2e-8,
    )

    packed_group = grouped.groups[0]
    if device == "cuda":
        output_adjoint = torch.randn(
            2,
            packed_group.output_dimension,
            dtype=torch.complex128,
            device=device,
        )
        cached_adjoint, workspace, workspace_adjoint = (
            torch.ops.ye3t_runtime
            .factorized_angular_heterogeneous_adjoint_with_workspace(
                output_adjoint,
                native_bank.detach(),
                *packed_group._native_arguments(),
                packed_group.source_dimension,
                packed_group.workspace_dimension,
            )
        )
        direct_adjoint = (
            torch.ops.ye3t_runtime.factorized_angular_heterogeneous_adjoint(
                output_adjoint,
                native_bank.detach(),
                *packed_group._native_arguments(),
                packed_group.source_dimension,
                packed_group.workspace_dimension,
            )
        )
        torch.testing.assert_close(
            cached_adjoint,
            direct_adjoint,
            rtol=0.0,
            atol=0.0,
        )
        packed_adjoint_tangent = torch.randn_like(native_bank)
        cached_second = (
            torch.ops.ye3t_runtime
            .factorized_angular_heterogeneous_double_backward_from_workspace(
                packed_adjoint_tangent,
                output_adjoint,
                workspace,
                workspace_adjoint,
                *packed_group._native_arguments(),
                packed_group.source_dimension,
                packed_group.workspace_dimension,
            )
        )
        direct_second = (
            torch.ops.ye3t_runtime
            .factorized_angular_heterogeneous_double_backward(
                packed_adjoint_tangent,
                output_adjoint,
                native_bank.detach(),
                *packed_group._native_arguments(),
                packed_group.source_dimension,
                packed_group.workspace_dimension,
            )
        )
        for cached_value, direct_value in zip(
            cached_second,
            direct_second,
        ):
            torch.testing.assert_close(
                cached_value,
                direct_value,
                rtol=0.0,
                atol=0.0,
            )

    arguments = (
        native_bank,
        *packed_group._native_arguments(),
        packed_group.source_dimension,
        packed_group.workspace_dimension,
        packed_group.output_dimension,
    )
    result = torch.library.opcheck(
        torch.ops.ye3t_runtime.factorized_angular_heterogeneous.default,
        arguments,
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in result.values()), result

    if device == "cpu":
        def raw_operator(packed):
            return (
                torch.ops.ye3t_runtime.factorized_angular_heterogeneous(
                    packed,
                    *packed_group._native_arguments(),
                    packed_group.source_dimension,
                    packed_group.workspace_dimension,
                    packed_group.output_dimension,
                )
            )

        eager = raw_operator(native_bank.detach())
        compiled = torch.compile(
            raw_operator,
            backend="eager",
            fullgraph=True,
        )(native_bank.detach())
        torch.testing.assert_close(compiled, eager, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "plan_factory",
    [_rank3_factorized_plan, _rank3_factorized_role_plan],
)
@pytest.mark.parametrize("complex_values", [False, True])
def test_native_factorized_angular_cuda_matches_reference_vjp_and_hvp(
    monkeypatch,
    plan_factory,
    complex_values,
):
    if not torch.cuda.is_available() or not _load_extension().has_cuda():
        pytest.skip("requires native CUDA")
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    dtype = torch.complex128 if complex_values else torch.float64
    execution_plan = plan_factory()
    native = YE3TFactorizedAngularModule(
        execution_plan,
        backend="native",
        dtype=dtype,
        device="cuda",
    )
    reference = YE3TFactorizedAngularModule(
        execution_plan,
        backend="reference",
        dtype=dtype,
        device="cuda",
    )
    source_shape = (
        (native.source_dimension,)
        if native.source_dimension > 1
        else ()
    )
    slots = tuple(
        torch.randn(
            (3,) + source_shape + (2 * angular_L + 1,),
            dtype=dtype,
            device="cuda",
            requires_grad=True,
        )
        for angular_L in (1, 2, 1)
    )

    actual = native(slots)
    expected = reference(slots)
    tolerance = 2.0e-10
    torch.testing.assert_close(
        actual,
        expected,
        rtol=tolerance,
        atol=tolerance,
    )
    output_adjoint = torch.randn_like(actual)
    native_gradient = torch.autograd.grad(
        actual,
        slots,
        output_adjoint,
        create_graph=True,
    )
    reference_gradient = torch.autograd.grad(
        expected,
        slots,
        output_adjoint,
        create_graph=True,
    )
    for value, expected_value in zip(
        native_gradient,
        reference_gradient,
    ):
        torch.testing.assert_close(
            value,
            expected_value,
            rtol=tolerance,
            atol=tolerance,
        )
    probes = tuple(torch.randn_like(value) for value in slots)
    native_hvp = torch.autograd.grad(
        native_gradient,
        slots,
        probes,
    )
    reference_hvp = torch.autograd.grad(
        reference_gradient,
        slots,
        probes,
    )
    for value, expected_value in zip(native_hvp, reference_hvp):
        torch.testing.assert_close(
            value,
            expected_value,
            rtol=2.0e-9,
            atol=2.0e-9,
        )

    if not complex_values and plan_factory is _rank3_factorized_role_plan:
        packed, _ = native._packed_slots(slots)
        arguments = (
            packed,
            *native._native_arguments(),
            native.source_dimension,
            native.workspace_dimension,
            native.output_dimension,
        )
        result = torch.library.opcheck(
            torch.ops.ye3t_runtime.factorized_angular.default,
            arguments,
            raise_exception=False,
        )
        assert all(value == "SUCCESS" for value in result.values()), result


def test_native_factorized_angular_cuda_serial_and_warp_policies_agree(
    monkeypatch,
):
    if not torch.cuda.is_available() or not _load_extension().has_cuda():
        pytest.skip("requires native CUDA")
    execution_plan = _rank3_factorized_role_plan()
    native = YE3TFactorizedAngularModule(
        execution_plan,
        backend="native",
        dtype=torch.complex128,
        device="cuda",
    )
    source_shape = (native.source_dimension,)
    base_slots = tuple(
        torch.randn(
            (4,) + source_shape + (2 * angular_L + 1,),
            dtype=torch.complex128,
            device="cuda",
        )
        for angular_L in (1, 2, 1)
    )
    probes = tuple(torch.randn_like(value) for value in base_slots)

    def evaluate(policy):
        monkeypatch.setenv(
            "YE3T_FACTORIZED_ANGULAR_CUDA_POLICY",
            policy,
        )
        slots = tuple(value.clone().requires_grad_(True) for value in base_slots)
        output = native(slots)
        gradients = torch.autograd.grad(
            _squared_norm(output),
            slots,
            create_graph=True,
        )
        hvp = torch.autograd.grad(gradients, slots, probes)
        return output.detach(), tuple(
            value.detach() for value in gradients
        ), tuple(value.detach() for value in hvp)

    serial = evaluate("serial")
    warp = evaluate("warp")
    for serial_group, warp_group in zip(serial, warp):
        serial_values = (
            serial_group if isinstance(serial_group, tuple) else (serial_group,)
        )
        warp_values = (
            warp_group if isinstance(warp_group, tuple) else (warp_group,)
        )
        for serial_value, warp_value in zip(serial_values, warp_values):
            torch.testing.assert_close(
                warp_value,
                serial_value,
                rtol=2.0e-10,
                atol=2.0e-10,
            )


@pytest.mark.parametrize(
    "plan_factory",
    [_rank3_factorized_plan, _rank3_factorized_role_plan],
)
@pytest.mark.parametrize("complex_values", [False, True])
def test_native_factorized_linear_cuda_matches_reference_gradients_and_hvp(
    monkeypatch,
    plan_factory,
    complex_values,
):
    if not torch.cuda.is_available() or not _load_extension().has_cuda():
        pytest.skip("requires native CUDA")
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    dtype = torch.complex128 if complex_values else torch.float64
    execution_plan = plan_factory()
    native = YE3TFactorizedAngularModule(
        execution_plan,
        backend="native",
        dtype=dtype,
        device="cuda",
    )
    reference = YE3TFactorizedAngularModule(
        execution_plan,
        backend="reference",
        dtype=dtype,
        device="cuda",
    )
    source_shape = (
        (native.source_dimension,)
        if native.source_dimension > 1
        else ()
    )
    native_slots = tuple(
        torch.randn(
            (3,) + source_shape + (2 * angular_L + 1,),
            dtype=dtype,
            device="cuda",
            requires_grad=True,
        )
        for angular_L in (1, 2, 1)
    )
    reference_slots = tuple(
        value.detach().clone().requires_grad_(True)
        for value in native_slots
    )
    native_weight = torch.randn(
        native.output_dimension,
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    reference_weight = (
        native_weight.detach().clone().requires_grad_(True)
    )
    native_bias = torch.randn(
        (),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    reference_bias = native_bias.detach().clone().requires_grad_(True)

    actual = native.linear_readout(
        native_slots,
        native_weight,
        native_bias,
    )
    expected = reference.linear_readout(
        reference_slots,
        reference_weight,
        reference_bias,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-10)
    native_arguments = native_slots + (native_weight, native_bias)
    reference_arguments = reference_slots + (
        reference_weight,
        reference_bias,
    )
    native_gradients = torch.autograd.grad(
        _squared_norm(actual),
        native_arguments,
        create_graph=True,
    )
    reference_gradients = torch.autograd.grad(
        _squared_norm(expected),
        reference_arguments,
        create_graph=True,
    )
    for value, expected_value in zip(
        native_gradients,
        reference_gradients,
    ):
        torch.testing.assert_close(
            value,
            expected_value,
            rtol=2e-9,
            atol=2e-9,
        )
    probes = tuple(torch.randn_like(value) for value in native_arguments)
    native_hvp = torch.autograd.grad(
        native_gradients,
        native_arguments,
        probes,
    )
    reference_hvp = torch.autograd.grad(
        reference_gradients,
        reference_arguments,
        probes,
    )
    for value, expected_value in zip(native_hvp, reference_hvp):
        torch.testing.assert_close(
            value,
            expected_value,
            rtol=2e-8,
            atol=2e-8,
        )

    if not complex_values and plan_factory is _rank3_factorized_role_plan:
        packed, _ = native._packed_slots(native_slots)
        result = torch.library.opcheck(
            torch.ops.ye3t_runtime.factorized_angular_linear.default,
            (
                packed,
                *native._native_arguments(),
                native.source_dimension,
                native.workspace_dimension,
                native_weight,
                native_bias,
            ),
            raise_exception=False,
        )
        assert all(value == "SUCCESS" for value in result.values()), result


def test_factorized_linear_cuda_auto_policy_materializes_complex_roles():
    if not torch.cuda.is_available() or not _load_extension().has_cuda():
        pytest.skip("requires native CUDA")
    execution_plan = _rank3_factorized_role_plan()
    automatic = YE3TFactorizedAngularModule(
        execution_plan,
        backend="auto",
        dtype=torch.complex128,
        device="cuda",
    )
    explicit = YE3TFactorizedAngularModule(
        execution_plan,
        backend="native",
        dtype=torch.complex128,
        device="cuda",
    )
    slots = tuple(
        torch.randn(
            4,
            automatic.source_dimension,
            2 * angular_L + 1,
            dtype=torch.complex128,
            device="cuda",
        )
        for angular_L in (1, 2, 1)
    )
    weight = torch.randn(
        automatic.output_dimension,
        dtype=torch.complex128,
        device="cuda",
    )
    bias = torch.randn((), dtype=torch.complex128, device="cuda")

    automatic_output = automatic.linear_readout(slots, weight, bias)
    explicit_output = explicit.linear_readout(slots, weight, bias)

    torch.testing.assert_close(
        automatic_output,
        explicit_output,
        rtol=2e-10,
        atol=2e-10,
    )
    assert (
        automatic.runtime_report()["last_linear_readout_backend"]
        == "native_materialized_measured_complex_role"
    )
    assert (
        explicit.runtime_report()["last_linear_readout_backend"]
        == "native_fused_factorized_linear"
    )


@pytest.mark.gpu
def test_factorized_auto_uses_cuda_reference_for_cpu_only_extension(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    class CPUOnlyExtension:
        @staticmethod
        def has_cuda():
            return False

    runtime = importlib.import_module("ye3t.runtime.execution_plan")
    extension = CPUOnlyExtension()
    monkeypatch.setattr(runtime, "_prebuilt_extension", lambda: extension)
    monkeypatch.setattr(runtime, "_load_extension", lambda: extension)
    automatic = runtime.YE3TFactorizedAngularModule(
        _rank3_factorized_role_plan(),
        backend="auto",
        dtype=torch.complex64,
        device="cuda",
    )
    assert automatic.resolved_backend == "torch_reference"
    assert all(buffer.device.type == "cuda" for buffer in automatic.buffers())

    with pytest.raises(RuntimeError, match="no CUDA factorized angular"):
        runtime.YE3TFactorizedAngularModule(
            _rank3_factorized_role_plan(),
            backend="native",
            dtype=torch.complex64,
            device="cuda",
        )
