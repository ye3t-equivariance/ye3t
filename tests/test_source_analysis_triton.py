import math

import pytest
import torch

from ye3t.couplings import (
    YE3TCarrierKey,
    YE3TCarrierLayout,
    YE3TRuntimeInstruction,
    YE3TSourceAssemblyPlan,
    YE3TSourceRealization,
    YE3TSynthesisTable,
    apply_source_analysis_reference,
    compile_execution_plan,
)
from ye3t.execution_plan import (
    YE3T_PRIMARY_CONVENTION,
    YE3T_REAL_TESSERAL_CONVENTION,
)
from ye3t.runtime.execution_plan import (
    YE3TSourceAnalysisModule,
    native_execution_plan_capabilities,
)


pytestmark = pytest.mark.fast


def test_capability_report_declares_triton_source_analysis_contract():
    capabilities = native_execution_plan_capabilities()

    assert isinstance(capabilities["triton_available"], bool)
    assert isinstance(capabilities["triton_cuda_available"], bool)
    assert (
        capabilities["triton_real_convention_id"]
        == YE3T_REAL_TESSERAL_CONVENTION
    )
    assert (
        "dense_source_analysis_C_dagger_L_v"
        in capabilities["triton_operations"]
    )
    assert (
        "dense_source_analysis_channel_mixing"
        in capabilities["triton_operations"]
    )


def _two_channel_vector_plan(
    convention_id=YE3T_REAL_TESSERAL_CONVENTION,
):
    key = YE3TCarrierKey(
        rank=2,
        partition=(2,),
        rotation_L=1,
        convention_id=convention_id,
    )
    source_realization = YE3TSourceRealization(
        kind="lifted_density_roles",
        rank=2,
        content=(1, 2),
        role_labels=("scalar_role", "vector_role"),
        retain_role_order=True,
        automorphisms=((1, 0),),
    )
    source_dimension = 12
    assembly = YE3TSourceAssemblyPlan(
        assembly_id="two_channel_full_role_placement",
        source_realization=source_realization,
        source_dimension=source_dimension,
        induced_dimension=source_dimension,
        row_indices=tuple(range(source_dimension)),
        column_indices=tuple(range(source_dimension)),
        values=(1.0,) * source_dimension,
        validation_report={
            "passed": True,
            "placement_orbit": "complete",
            "layout": "[raw_channel, role_placement, magnetic_M]",
        },
    )
    scale = 1.0 / math.sqrt(2.0)
    synthesis_rows = []
    synthesis_columns = []
    synthesis_values = []
    for channel in range(2):
        for placement in range(2):
            for magnetic in range(3):
                synthesis_rows.append(
                    channel * 6 + placement * 3 + magnetic
                )
                synthesis_columns.append(channel * 3 + magnetic)
                synthesis_values.append(scale)
    synthesis = YE3TSynthesisTable(
        table_id="two_channel_symmetric_vector",
        input_dimension=source_dimension,
        output_dimension=6,
        row_indices=tuple(synthesis_rows),
        column_indices=tuple(synthesis_columns),
        values=tuple(synthesis_values),
        validation_report={
            "passed": True,
            "scope": "two copies of the exact S2 symmetric projector",
        },
    )
    instruction = YE3TRuntimeInstruction(
        instruction_id="full_role_source_analysis",
        opcode="rank_additive_lr_induction",
        input_carriers=(),
        output_carrier=key,
        source_assembly_id=assembly.assembly_id,
        synthesis_table_id=synthesis.table_id,
    )
    plan = compile_execution_plan(
        carrier_layouts=(
            YE3TCarrierLayout(
                key=key,
                channel_count=2,
                tableau_count=1,
                magnetic_count=3,
            ),
        ),
        source_assemblies=(assembly,),
        synthesis_tables=(synthesis,),
        instructions=(instruction,),
        forward_schedule=(instruction.instruction_id,),
        reverse_schedule=(instruction.instruction_id,),
        second_order_schedule=(instruction.instruction_id,),
        certificate={
            "passed": True,
            "scope": "full typed S2 role placement and channel mixing",
        },
    )
    return plan, assembly, synthesis


def _mixed_reference(source, weight, assembly, synthesis):
    base = apply_source_analysis_reference(source, assembly, synthesis)
    logical = base.reshape(int(base.shape[0]), 2, 3)
    return torch.matmul(weight, logical).reshape(int(base.shape[0]), 6)


def test_source_analysis_channel_mixing_preserves_magnetic_axis():
    plan, assembly, synthesis = _two_channel_vector_plan()
    module = YE3TSourceAnalysisModule(
        plan,
        backend="reference",
        channel_mixing=True,
    )
    weight = torch.tensor(
        [[1.5, -0.25], [0.75, 2.0]],
        dtype=torch.float64,
    )
    with torch.no_grad():
        module.mixing_weight.copy_(weight)
    source = torch.arange(1.0, 25.0, dtype=torch.float64).reshape(2, 12)

    actual = module(source)
    expected = _mixed_reference(source, weight, assembly, synthesis)
    swapped = source.reshape(2, 2, 2, 3).flip(2).reshape(2, 12)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(
        module(swapped),
        actual,
        rtol=0.0,
        atol=1e-12,
    )
    logical = actual.reshape(2, 2, 1, 3)
    assert logical.shape == (2, 2, 1, 3)
    report = module.runtime_report()
    assert report["full_source_assembly_applied"]
    assert report["learned_map_axis"] == "channel_or_multiplicity"
    assert report["preserved_axes"] == ("tableau_t", "magnetic_M")
    assert not report["learned_map_fused"]


def test_source_analysis_channel_mixing_matches_reference_vjp_and_hvp():
    plan, assembly, synthesis = _two_channel_vector_plan()
    module = YE3TSourceAnalysisModule(
        plan,
        backend="reference",
        channel_mixing=True,
    )
    source = torch.randn(3, 12, dtype=torch.float64, requires_grad=True)
    reference_source = source.detach().clone().requires_grad_()
    weight = torch.randn(2, 2, dtype=torch.float64)
    with torch.no_grad():
        module.mixing_weight.copy_(weight)
    reference_weight = weight.detach().clone().requires_grad_()

    actual_loss = module(source).square().sum()
    expected_loss = _mixed_reference(
        reference_source,
        reference_weight,
        assembly,
        synthesis,
    ).square().sum()
    actual_gradients = torch.autograd.grad(
        actual_loss,
        (source, module.mixing_weight),
        create_graph=True,
    )
    expected_gradients = torch.autograd.grad(
        expected_loss,
        (reference_source, reference_weight),
        create_graph=True,
    )
    actual_hvp = torch.autograd.grad(
        actual_gradients[0].sum() + actual_gradients[1].sum(),
        (source, module.mixing_weight),
    )
    expected_hvp = torch.autograd.grad(
        expected_gradients[0].sum() + expected_gradients[1].sum(),
        (reference_source, reference_weight),
    )
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0.0, atol=1e-12)
    for actual, expected in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)
    for actual, expected in zip(actual_hvp, expected_hvp):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA",
)
@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    (
        (torch.float64, 1e-10, 1e-10),
        (torch.float32, 5e-5, 5e-5),
    ),
)
def test_triton_full_source_analysis_values_vjp_hvp_and_strict_dispatch(
    dtype,
    atol,
    rtol,
):
    plan, assembly, synthesis = _two_channel_vector_plan()
    module = YE3TSourceAnalysisModule(
        plan,
        backend="triton",
        dtype=dtype,
        device="cuda",
        channel_mixing=True,
    )
    source = torch.randn(
        5,
        12,
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    reference_source = source.detach().clone().requires_grad_()
    weight = torch.randn(2, 2, dtype=dtype, device="cuda")
    with torch.no_grad():
        module.mixing_weight.copy_(weight)
    reference_weight = weight.detach().clone().requires_grad_()

    actual = module(source)
    swapped = source.reshape(5, 2, 2, 3).flip(2).reshape(5, 12)
    expected = _mixed_reference(
        reference_source,
        reference_weight,
        assembly,
        synthesis,
    )
    actual_loss = actual.square().sum()
    expected_loss = expected.square().sum()
    actual_gradients = torch.autograd.grad(
        actual_loss,
        (source, module.mixing_weight),
        create_graph=True,
    )
    expected_gradients = torch.autograd.grad(
        expected_loss,
        (reference_source, reference_weight),
        create_graph=True,
    )
    actual_hvp = torch.autograd.grad(
        actual_gradients[0].sum() + actual_gradients[1].sum(),
        (source, module.mixing_weight),
    )
    expected_hvp = torch.autograd.grad(
        expected_gradients[0].sum() + expected_gradients[1].sum(),
        (reference_source, reference_weight),
    )
    first_source = source.detach().clone().requires_grad_()
    first_reference_source = (
        source.detach().clone().requires_grad_()
    )
    first_reference_weight = (
        module.mixing_weight.detach().clone().requires_grad_()
    )
    first_gradients = torch.autograd.grad(
        module(first_source).sum(),
        (first_source, module.mixing_weight),
    )
    first_reference_gradients = torch.autograd.grad(
        _mixed_reference(
            first_reference_source,
            first_reference_weight,
            assembly,
            synthesis,
        ).sum(),
        (first_reference_source, first_reference_weight),
    )

    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(
        module(swapped),
        actual,
        rtol=rtol,
        atol=atol,
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=rtol,
            atol=atol,
        )
    for actual_value, expected_value in zip(actual_hvp, expected_hvp):
        torch.testing.assert_close(
            actual_value,
            expected_value,
            rtol=rtol,
            atol=atol,
        )
    for actual_value, expected_value in zip(
        first_gradients,
        first_reference_gradients,
    ):
        torch.testing.assert_close(
            actual_value,
            expected_value,
            rtol=rtol,
            atol=atol,
        )
    report = module.runtime_report()
    assert report["last_backend"] == "triton_weighted_sparse_linear_autograd"
    assert report["fused_source_analysis_and_mixing"]
    assert report["learned_map_fused"]
    assert report["strict"]


def test_explicit_triton_rejects_complex_primary_plan_before_dispatch():
    plan, _, _ = _two_channel_vector_plan(YE3T_PRIMARY_CONVENTION)
    with pytest.raises(ValueError, match="serialized real-tesseral"):
        YE3TSourceAnalysisModule(
            plan,
            backend="triton",
            dtype=torch.float64,
            device="cuda",
        )
