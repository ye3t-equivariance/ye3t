"""Smoke the public YE3T execution-plan API from a CUDA-enabled wheel."""

from pathlib import Path
import sys

import torch

import ye3t
from ye3t.couplings import (
    YE3TCarrierKey,
    YE3TCarrierLayout,
    YE3TRuntimeInstruction,
    YE3TSourceAssemblyPlan,
    YE3TSourceRealization,
    YE3TSynthesisTable,
    compile_execution_plan,
    symmetric_power_product_plan,
)
from ye3t.api import (
    symmetric_power_product_plan_batched_adjoint,
    symmetric_power_product_plan_contraction,
)
from ye3t.runtime.execution_plan import (
    YE3TSourceAnalysisModule,
    _load_extension,
    cheb_exp_cos_radial_table_with_derivative,
    cheb_exp_cos_radial_with_derivative,
    compact_exterior_power_product,
    compact_exterior_power_product_reference,
    compact_pair_product,
    compact_pair_product_reference,
    density_accumulate,
    density_accumulate_adjoint,
    native_execution_plan_capabilities,
    plain_site_basis_product_adjoint,
    plain_site_basis_product_with_derivative,
    spherical_harmonics_table_with_derivative,
    spherical_harmonics_with_derivative,
)
from ye3t.runtime.symmetric_power import (
    symmetric_power_outputs_real_tesseral,
    symmetric_power_product_evaluator_real_tesseral,
    symmetric_power_real_tesseral,
)


def _execution_plan():
    carrier = YE3TCarrierKey(
        rank=2,
        partition=(2,),
        rotation_L=0,
    )
    source = YE3TSourceRealization(
        kind="ordinary_density",
        rank=2,
        content=(1, 2),
    )
    assembly = YE3TSourceAssemblyPlan(
        assembly_id="installed_cuda_source",
        source_realization=source,
        source_dimension=3,
        induced_dimension=3,
        row_indices=(0, 1, 1, 2),
        column_indices=(0, 1, 2, 0),
        values=(1.0, 0.5j, -0.25, 0.75 - 0.5j),
        validation_report={"passed": True, "scope": "installed CUDA smoke"},
        provenance={"coefficient_source": "explicit smoke fixture"},
    )
    table = YE3TSynthesisTable(
        table_id="installed_cuda_synthesis",
        input_dimension=3,
        output_dimension=2,
        row_indices=(0, 1, 1, 2),
        column_indices=(0, 0, 1, 1),
        values=(0.75, -0.5j, 1.25 + 0.25j, -0.5),
        validation_report={"passed": True, "scope": "installed CUDA smoke"},
        provenance={"coefficient_source": "explicit smoke fixture"},
    )
    instruction = YE3TRuntimeInstruction(
        instruction_id="installed_cuda_analysis",
        opcode="rank_additive_lr_induction",
        input_carriers=(),
        output_carrier=carrier,
        source_assembly_id=assembly.assembly_id,
        synthesis_table_id=table.table_id,
    )
    return compile_execution_plan(
        carrier_layouts=(
            YE3TCarrierLayout(
                key=carrier,
                channel_count=2,
                tableau_count=1,
                magnetic_count=1,
            ),
        ),
        source_assemblies=(assembly,),
        synthesis_tables=(table,),
        instructions=(instruction,),
        forward_schedule=(instruction.instruction_id,),
        reverse_schedule=(instruction.instruction_id,),
        second_order_schedule=(instruction.instruction_id,),
        certificate={"passed": True, "scope": "installed CUDA smoke"},
    )


def _factorized_cuda_smoke():
    dtype = torch.complex128
    device = torch.device("cuda")
    packed = torch.tensor(
        [
            [
                0.25,
                -0.5,
                0.75,
                1.0,
                -0.625,
                0.375,
                -0.25,
                0.5,
                1.25,
                0.75,
                -1.0,
                0.625,
            ],
            [
                -0.5,
                0.125,
                1.0,
                0.375,
                0.75,
                -0.25,
                1.25,
                -0.75,
                0.5,
                -0.625,
                0.25,
                1.0,
            ],
        ],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    node_offsets = torch.tensor([0, 3, 6], dtype=torch.int64, device=device)
    node_dimensions = torch.tensor([3, 3, 1], dtype=torch.int64, device=device)
    node_leaf_offsets = torch.tensor([0, 3, -1], dtype=torch.int64, device=device)
    node_left = torch.tensor([-1, -1, 0], dtype=torch.int64, device=device)
    node_right = torch.tensor([-1, -1, 1], dtype=torch.int64, device=device)
    node_coefficient_offsets = torch.tensor(
        [0, 0, 0, 3],
        dtype=torch.int64,
        device=device,
    )
    coefficient_rows = torch.tensor([0, 4, 8], dtype=torch.int64, device=device)
    coefficient_columns = torch.zeros(3, dtype=torch.int64, device=device)
    coefficient_values = torch.tensor(
        [0.5 + 0.125j, -0.25 - 0.375j, 0.75 + 0.25j],
        dtype=dtype,
        device=device,
    )
    root_nodes = torch.tensor([2], dtype=torch.int64, device=device)
    projection_values = torch.tensor(
        [1.0 + 0.25j, 0.5 - 0.125j, -0.25 + 0.5j, 0.75 + 0.375j],
        dtype=dtype,
        device=device,
    )
    arguments = (
        node_offsets,
        node_dimensions,
        node_leaf_offsets,
        node_left,
        node_right,
        node_coefficient_offsets,
        coefficient_rows,
        coefficient_columns,
        coefficient_values,
        root_nodes,
        projection_values,
    )
    actual = torch.ops.ye3t_runtime.factorized_angular(
        packed,
        *arguments,
        2,
        7,
        2,
    )
    sources = packed.reshape(2, 2, 6)
    roots = (
        sources[:, :, :3]
        * sources[:, :, 3:]
        * coefficient_values.conj().reshape(1, 1, 3)
    ).sum(dim=2)
    expected = roots @ projection_values.reshape(2, 2)
    torch.testing.assert_close(actual, expected, rtol=2.0e-10, atol=2.0e-10)
    output_adjoint = torch.tensor(
        [[0.5 - 0.25j, -1.0 + 0.75j], [1.25 + 0.5j, 0.25 - 0.625j]],
        dtype=dtype,
        device=device,
    )
    actual_gradient = torch.autograd.grad(
        actual,
        packed,
        output_adjoint,
    )[0]
    reference_input = packed.detach().clone().requires_grad_(True)
    reference_sources = reference_input.reshape(2, 2, 6)
    reference_roots = (
        reference_sources[:, :, :3]
        * reference_sources[:, :, 3:]
        * coefficient_values.conj().reshape(1, 1, 3)
    ).sum(dim=2)
    reference = reference_roots @ projection_values.reshape(2, 2)
    expected_gradient = torch.autograd.grad(
        reference,
        reference_input,
        output_adjoint,
    )[0]
    torch.testing.assert_close(
        actual_gradient,
        expected_gradient,
        rtol=2.0e-10,
        atol=2.0e-10,
    )
    linear_input = packed.detach().clone().requires_grad_(True)
    linear_weight = torch.tensor(
        [0.75 - 0.25j, -1.125 + 0.5j],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    linear_bias = torch.tensor(
        0.125 + 0.375j,
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    linear = torch.ops.ye3t_runtime.factorized_angular_linear(
        linear_input,
        *arguments,
        2,
        7,
        linear_weight,
        linear_bias,
    )
    linear_sources = linear_input.reshape(2, 2, 6)
    linear_roots = (
        linear_sources[:, :, :3]
        * linear_sources[:, :, 3:]
        * coefficient_values.conj().reshape(1, 1, 3)
    ).sum(dim=2)
    linear_features = linear_roots @ projection_values.reshape(2, 2)
    expected_linear = linear_features @ linear_weight + linear_bias
    torch.testing.assert_close(
        linear,
        expected_linear,
        rtol=2.0e-10,
        atol=2.0e-10,
    )
    linear_gradients = torch.autograd.grad(
        linear,
        (linear_input, linear_weight, linear_bias),
        output_adjoint[:, 0],
    )
    reference_linear_input = (
        packed.detach().clone().requires_grad_(True)
    )
    reference_linear_weight = (
        linear_weight.detach().clone().requires_grad_(True)
    )
    reference_linear_bias = (
        linear_bias.detach().clone().requires_grad_(True)
    )
    reference_linear_sources = reference_linear_input.reshape(2, 2, 6)
    reference_linear_roots = (
        reference_linear_sources[:, :, :3]
        * reference_linear_sources[:, :, 3:]
        * coefficient_values.conj().reshape(1, 1, 3)
    ).sum(dim=2)
    reference_linear_features = (
        reference_linear_roots @ projection_values.reshape(2, 2)
    )
    reference_linear = (
        reference_linear_features @ reference_linear_weight
        + reference_linear_bias
    )
    expected_linear_gradients = torch.autograd.grad(
        reference_linear,
        (
            reference_linear_input,
            reference_linear_weight,
            reference_linear_bias,
        ),
        output_adjoint[:, 0],
    )
    for value, expected_value in zip(
        linear_gradients,
        expected_linear_gradients,
    ):
        torch.testing.assert_close(
            value,
            expected_value,
            rtol=2.0e-10,
            atol=2.0e-10,
        )


def _plain_source_product_cuda_smoke():
    edge_count = 5
    radial_values = torch.randn(
        3,
        edge_count,
        dtype=torch.complex128,
        device="cuda",
    )
    radial_derivatives = torch.randn_like(radial_values)
    angular_values = torch.randn(
        6,
        edge_count,
        dtype=torch.complex128,
        device="cuda",
    )
    angular_derivatives = torch.randn(
        6,
        edge_count,
        3,
        dtype=torch.complex128,
        device="cuda",
    )
    prefactors = torch.randn_like(radial_values)
    prefactor_center = torch.randn_like(radial_values)
    prefactor_neighbor = torch.randn_like(radial_values)
    radial_directions = torch.randn(
        edge_count,
        3,
        dtype=torch.complex128,
        device="cuda",
    )
    term_groups = torch.tensor(
        [0, 0, 1, 1, 2, 2],
        dtype=torch.int64,
        device="cuda",
    )
    term_channels = torch.tensor(
        [7, 1, 5, 0, 4, 2],
        dtype=torch.int64,
        device="cuda",
    )
    arguments = (
        radial_values,
        radial_derivatives,
        angular_values,
        angular_derivatives,
        prefactors,
        prefactor_center,
        prefactor_neighbor,
        radial_directions,
        term_groups,
        term_channels,
        8,
    )
    actual = plain_site_basis_product_with_derivative(
        *arguments,
        backend="native",
    )
    expected = plain_site_basis_product_with_derivative(
        *arguments,
        backend="reference",
    )
    for actual_value, expected_value in zip(actual, expected):
        torch.testing.assert_close(
            actual_value,
            expected_value,
            rtol=2.0e-10,
            atol=2.0e-10,
        )

    edge_weights = torch.randn(
        edge_count,
        dtype=torch.complex128,
        device="cuda",
    )
    edge_weight_derivatives = torch.randn(
        edge_count,
        3,
        dtype=torch.complex128,
        device="cuda",
    )
    edge_adjoint = torch.randn(
        edge_count,
        8,
        dtype=torch.complex128,
        device="cuda",
    )
    adjoint_arguments = (
        radial_values,
        radial_derivatives,
        angular_values,
        angular_derivatives,
        prefactors,
        prefactor_center,
        prefactor_neighbor,
        radial_directions,
        edge_weights,
        edge_weight_derivatives,
        term_groups,
        term_channels,
        edge_adjoint,
    )
    actual_adjoint = plain_site_basis_product_adjoint(
        *adjoint_arguments,
        backend="native",
    )
    expected_adjoint = plain_site_basis_product_adjoint(
        *adjoint_arguments,
        backend="reference",
    )
    for actual_value, expected_value in zip(
        actual_adjoint,
        expected_adjoint,
    ):
        torch.testing.assert_close(
            actual_value,
            expected_value,
            rtol=2.0e-10,
            atol=2.0e-10,
        )


def main():
    package_path = Path(ye3t.__file__).resolve()
    expected_prefix = Path(sys.prefix).resolve()
    if not package_path.is_relative_to(expected_prefix):
        raise RuntimeError(
            "YE3T was imported outside the active installation prefix: "
            f"{package_path} is not under {expected_prefix}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("installed CUDA smoke requires an available CUDA device")
    _load_extension()

    capabilities = native_execution_plan_capabilities()
    expected_operations = (
        "cheb_exp_cos_radial_with_derivative",
        "cheb_exp_cos_radial_table_with_derivative",
        "cheb_exp_cos_radial_table_double_backward",
        "spherical_harmonics_with_derivative",
        "spherical_harmonics_table_with_derivative",
        "spherical_harmonics_table_double_backward",
        "plain_site_basis_product_with_derivative",
        "plain_site_basis_product_adjoint",
        "density_accumulate",
        "density_accumulate_adjoint",
        "compact_pair_product",
        "compact_pair_product_adjoint",
        "compact_exterior_power",
        "compact_exterior_power_adjoint",
        "symmetric_power_monomial",
        "symmetric_power_monomial_adjoint",
        "symmetric_power_shared_monomial",
        "symmetric_power_shared_monomial_adjoint",
        "symmetric_power_shared_monomial_batched_adjoint",
        "factorized_angular",
        "factorized_angular_adjoint",
        "factorized_angular_double_backward",
        "factorized_angular_heterogeneous",
        "factorized_angular_heterogeneous_adjoint",
        "factorized_angular_heterogeneous_double_backward",
        "factorized_angular_linear",
        "factorized_angular_linear_adjoint",
        "source_analysis",
        "source_analysis_adjoint",
        "source_analysis_linear",
        "source_analysis_linear_adjoint",
    )
    if not capabilities["cuda"]:
        raise RuntimeError("installed YE3T wheel does not report native CUDA")
    if tuple(capabilities["cuda_operations"]) != expected_operations:
        raise RuntimeError(
            "installed CUDA operation report is incomplete: "
            f"{capabilities['cuda_operations']}"
        )
    _factorized_cuda_smoke()
    _plain_source_product_cuda_smoke()

    radii = torch.tensor(
        [0.17, 0.83, 1.72, 2.49],
        dtype=torch.float64,
        device="cuda",
        requires_grad=True,
    )
    cutoffs = torch.tensor(
        [2.2, 2.4, 2.6, 3.0],
        dtype=torch.float64,
        device="cuda",
    )
    lambdas = torch.tensor(
        [1.3, 1.7, 2.0, 2.4],
        dtype=torch.float64,
        device="cuda",
    )
    native_radial = cheb_exp_cos_radial_with_derivative(
        radii,
        cutoffs,
        lambdas,
        5,
        backend="native",
    )
    reference_radial = cheb_exp_cos_radial_with_derivative(
        radii,
        cutoffs,
        lambdas,
        5,
        backend="reference",
    )
    torch.testing.assert_close(
        native_radial[0],
        reference_radial[0],
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    torch.testing.assert_close(
        native_radial[1],
        reference_radial[1],
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    native_radial_table = (
        cheb_exp_cos_radial_table_with_derivative(
            radii,
            cutoffs,
            lambdas,
            5,
            backend="native",
        )
    )
    reference_radial_table = torch.stack(
        [
            cheb_exp_cos_radial_with_derivative(
                radii,
                cutoffs,
                lambdas,
                radial_index,
                backend="reference",
            )[0]
            for radial_index in range(6)
        ],
        dim=1,
    )
    torch.testing.assert_close(
        native_radial_table[0],
        reference_radial_table,
        rtol=1.0e-11,
        atol=1.0e-11,
    )

    edge_vectors = torch.tensor(
        [
            [0.31, 0.47, 0.83],
            [-0.42, 0.58, 0.27],
            [0.73, -0.24, 0.51],
            [-0.67, -0.19, 0.37],
        ],
        dtype=torch.float64,
        device="cuda",
        requires_grad=True,
    )
    native_spherical = spherical_harmonics_with_derivative(
        edge_vectors,
        5,
        real_output=False,
        backend="native",
    )
    reference_spherical = spherical_harmonics_with_derivative(
        edge_vectors,
        5,
        real_output=False,
        backend="reference",
    )
    torch.testing.assert_close(
        native_spherical[0],
        reference_spherical[0],
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    torch.testing.assert_close(
        native_spherical[1],
        reference_spherical[1],
        rtol=2.0e-10,
        atol=2.0e-10,
    )
    native_spherical_table = (
        spherical_harmonics_table_with_derivative(
            edge_vectors,
            5,
            real_output=False,
            backend="native",
        )
    )
    reference_spherical_table = torch.cat(
        [
            spherical_harmonics_with_derivative(
                edge_vectors,
                angular_momentum,
                real_output=False,
                backend="reference",
            )[0]
            for angular_momentum in range(6)
        ],
        dim=1,
    )
    torch.testing.assert_close(
        native_spherical_table[0],
        reference_spherical_table,
        rtol=2.0e-11,
        atol=2.0e-11,
    )

    edge_values = torch.tensor(
        [
            [1.0 + 0.25j, -0.5 + 0.75j],
            [2.0 - 0.5j, 0.75 + 0.25j],
            [-0.25 + 1.0j, 1.5 - 0.75j],
            [0.75 + 0.5j, -1.25 - 0.25j],
        ],
        dtype=torch.complex128,
        device="cuda",
        requires_grad=True,
    )
    centers = torch.tensor([0, 2, 1, 2], dtype=torch.int64, device="cuda")
    expected_density = density_accumulate(
        edge_values,
        centers,
        3,
        backend="reference",
    )
    native_density = density_accumulate(
        edge_values,
        centers,
        3,
        backend="native",
    )
    torch.testing.assert_close(
        native_density,
        expected_density,
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    atomic_adjoint = torch.flip(expected_density.detach(), dims=(1,))
    torch.testing.assert_close(
        density_accumulate_adjoint(
            atomic_adjoint,
            centers,
            backend="native",
        ),
        density_accumulate_adjoint(
            atomic_adjoint,
            centers,
            backend="reference",
        ),
        rtol=1.0e-11,
        atol=1.0e-11,
    )

    pair_left = torch.tensor(
        [
            [0.25 + 0.5j, -0.5 + 0.75j, 1.0 - 0.25j, 0.75 + 1.0j],
            [1.5 - 0.5j, 0.125 + 0.25j, -0.75 + 1.25j, 0.5 - 0.75j],
        ],
        dtype=torch.complex128,
        device="cuda",
        requires_grad=True,
    )
    pair_right = torch.flip(pair_left.detach(), dims=(1,)).requires_grad_(True)
    for antisymmetric in (False, True):
        native_pair = compact_pair_product(
            pair_left,
            pair_right,
            antisymmetric=antisymmetric,
            backend="native",
        )
        reference_pair = compact_pair_product_reference(
            pair_left,
            pair_right,
            antisymmetric=antisymmetric,
        )
        torch.testing.assert_close(
            native_pair,
            reference_pair,
            rtol=1.0e-11,
            atol=1.0e-11,
        )
        pair_adjoint = torch.flip(native_pair.detach(), dims=(1,))
        native_gradients = torch.autograd.grad(
            native_pair,
            (pair_left, pair_right),
            pair_adjoint,
            create_graph=True,
        )
        reference_gradients = torch.autograd.grad(
            reference_pair,
            (pair_left, pair_right),
            pair_adjoint,
            create_graph=True,
        )
        torch.testing.assert_close(
            native_gradients[0],
            reference_gradients[0],
            rtol=1.0e-11,
            atol=1.0e-11,
        )
        torch.testing.assert_close(
            native_gradients[1],
            reference_gradients[1],
            rtol=1.0e-11,
            atol=1.0e-11,
        )

    exterior_coordinates = torch.arange(
        30,
        dtype=torch.float64,
        device="cuda",
    ).reshape(2, 3, 5)
    exterior_factors = (
        0.5 * torch.sin(0.37 * exterior_coordinates + 0.11)
        + 0.35 * torch.cos(0.23 * exterior_coordinates - 0.19)
        + 0.4j * torch.cos(0.29 * exterior_coordinates + 0.17)
    ).requires_grad_(True)
    native_exterior = compact_exterior_power_product(
        exterior_factors,
        backend="native",
    )
    reference_exterior = compact_exterior_power_product_reference(
        exterior_factors
    )
    torch.testing.assert_close(
        native_exterior,
        reference_exterior,
        rtol=2.0e-10,
        atol=2.0e-10,
    )
    exterior_adjoint = torch.flip(native_exterior.detach(), dims=(1,))
    native_exterior_gradient = torch.autograd.grad(
        native_exterior,
        exterior_factors,
        exterior_adjoint,
        create_graph=True,
    )[0]
    reference_exterior_gradient = torch.autograd.grad(
        reference_exterior,
        exterior_factors,
        exterior_adjoint,
        create_graph=True,
    )[0]
    torch.testing.assert_close(
        native_exterior_gradient,
        reference_exterior_gradient,
        rtol=2.0e-10,
        atol=2.0e-10,
    )

    symmetric_input = torch.tensor(
        [
            [0.25, -0.5, 0.75],
            [-0.625, 1.0, 0.375],
            [0.0, -0.75, 0.5],
        ],
        dtype=torch.float64,
        device="cuda",
        requires_grad=True,
    )
    native_symmetric = symmetric_power_real_tesseral(
        symmetric_input,
        4,
        1,
        4,
        optimization_policy="aggressive",
    )
    reference_symmetric = symmetric_power_product_evaluator_real_tesseral(
        symmetric_input,
        4,
        1,
        4,
    )
    torch.testing.assert_close(
        native_symmetric,
        reference_symmetric,
        rtol=2.0e-10,
        atol=2.0e-10,
    )
    symmetric_adjoint = torch.flip(
        native_symmetric.detach(),
        dims=(1,),
    )
    native_symmetric_gradient = torch.autograd.grad(
        native_symmetric,
        symmetric_input,
        symmetric_adjoint,
        create_graph=True,
    )[0]
    reference_symmetric_gradient = torch.autograd.grad(
        reference_symmetric,
        symmetric_input,
        symmetric_adjoint,
        create_graph=True,
    )[0]
    torch.testing.assert_close(
        native_symmetric_gradient,
        reference_symmetric_gradient,
        rtol=2.0e-10,
        atol=2.0e-10,
    )

    grouped_symmetric = symmetric_power_outputs_real_tesseral(
        symmetric_input,
        4,
        1,
        (0, 2, 4),
        optimization_policy="aggressive",
    )
    for output_L in (0, 2, 4):
        separate = symmetric_power_product_evaluator_real_tesseral(
            symmetric_input,
            4,
            1,
            output_L,
        )
        torch.testing.assert_close(
            grouped_symmetric[output_L],
            separate,
            rtol=2.0e-10,
            atol=2.0e-10,
        )

    product_plan = symmetric_power_product_plan(
        (
            {
                "descriptor_index": 0,
                "channel_indices": (0, 1, 2),
                "power": 4,
                "input_L": 1,
                "output_L": 0,
                "multiplicity_index": 0,
                "component_index": 0,
            },
        ),
        descriptor_count=2,
        channel_count=3,
    )
    plan_input = symmetric_input.detach().clone().requires_grad_(True)
    plan_output = symmetric_power_product_plan_contraction(
        plan_input,
        product_plan,
        backend="native",
    )
    expected_plan_output = torch.zeros_like(plan_output)
    for entry in product_plan.entries:
        selected = plan_input[:, tuple(entry.channel_indices)]
        for term in entry.component_terms:
            term_value = torch.ones_like(selected[:, 0])
            for component, exponent in enumerate(term["exponents"]):
                for _ in range(int(exponent)):
                    term_value = term_value * selected[:, component]
            coefficient = torch.as_tensor(
                complex(term["coefficient"]).real,
                dtype=plan_input.dtype,
                device=plan_input.device,
            )
            expected_plan_output[:, entry.descriptor_index] += (
                coefficient * term_value
            )
    torch.testing.assert_close(
        plan_output,
        expected_plan_output,
        rtol=2.0e-10,
        atol=2.0e-10,
    )
    if torch.count_nonzero(plan_output[:, 1]):
        raise RuntimeError("inactive grouped plan output is not zero")
    plan_seed_rows = torch.eye(
        2,
        dtype=plan_input.dtype,
        device=plan_input.device,
    )
    plan_output_adjoint = plan_seed_rows.unsqueeze(1).expand(
        2,
        plan_input.shape[0],
        2,
    ).contiguous()
    expected_plan_roots = torch.stack(
        tuple(
            torch.autograd.grad(
                plan_output,
                plan_input,
                plan_output_adjoint[seed],
                retain_graph=True,
            )[0]
            for seed in range(2)
        ),
        dim=0,
    )
    native_plan_roots = symmetric_power_product_plan_batched_adjoint(
        plan_output_adjoint,
        plan_input,
        product_plan,
        backend="native",
    )
    torch.testing.assert_close(
        native_plan_roots,
        expected_plan_roots,
        rtol=2.0e-10,
        atol=2.0e-10,
    )
    plan_gradient = torch.autograd.grad(
        plan_output.square().sum(),
        plan_input,
    )[0]
    if not torch.isfinite(plan_gradient).all():
        raise RuntimeError("grouped plan gradient is not finite")

    plan = _execution_plan()
    native = YE3TSourceAnalysisModule(
        plan,
        backend="native",
        dtype=torch.complex128,
        device="cuda",
        strict=True,
    )
    reference = YE3TSourceAnalysisModule(
        plan,
        backend="reference",
        dtype=torch.complex128,
        device="cuda",
    )
    source = torch.tensor(
        [
            [1.0 + 0.5j, -2.0 + 0.25j, 0.75 - 1.0j],
            [0.5 - 0.25j, 1.25 + 2.0j, -0.5 + 0.75j],
        ],
        dtype=torch.complex128,
        device="cuda",
    )
    expected = reference(source)
    actual = native(source)
    torch.testing.assert_close(actual, expected, rtol=1.0e-11, atol=1.0e-11)
    report = native.runtime_report()
    if report["backend"] != "native_cuda" or report["last_backend"] != "native_cuda":
        raise RuntimeError(f"public runtime did not use native CUDA: {report}")

    weight = torch.tensor(
        [0.75 + 0.5j, -1.25 + 0.25j],
        dtype=torch.complex128,
        device="cuda",
    )
    bias = torch.tensor(0.125 - 0.375j, dtype=torch.complex128, device="cuda")
    torch.testing.assert_close(
        native.linear_readout(source, weight, bias),
        reference.linear_readout(source, weight, bias),
        rtol=1.0e-11,
        atol=1.0e-11,
    )

    tracked_source = source.detach().clone().requires_grad_(True)
    output_adjoint = torch.tensor(
        [[0.5 + 0.25j, -1.0 + 0.75j], [1.5 - 0.5j, 0.25 + 1.0j]],
        dtype=torch.complex128,
        device="cuda",
        requires_grad=True,
    )
    native_source_adjoint = torch.autograd.grad(
        native(tracked_source),
        tracked_source,
        output_adjoint,
        create_graph=True,
    )[0]
    reference_source = source.detach().clone().requires_grad_(True)
    reference_output_adjoint = output_adjoint.detach().clone().requires_grad_(True)
    reference_source_adjoint = torch.autograd.grad(
        reference(reference_source),
        reference_source,
        reference_output_adjoint,
        create_graph=True,
    )[0]
    torch.testing.assert_close(
        native_source_adjoint,
        reference_source_adjoint,
        rtol=1.0e-11,
        atol=1.0e-11,
    )

    source_tangent = torch.flip(source, dims=(1,))
    native_output_tangent = torch.autograd.grad(
        native_source_adjoint,
        output_adjoint,
        source_tangent,
    )[0]
    reference_output_tangent = torch.autograd.grad(
        reference_source_adjoint,
        reference_output_adjoint,
        source_tangent,
    )[0]
    torch.testing.assert_close(
        native_output_tangent,
        reference_output_tangent,
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    print(f"YE3T installed CUDA smoke passed: {package_path}")


if __name__ == "__main__":
    main()
