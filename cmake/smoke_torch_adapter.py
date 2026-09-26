"""Load a CMake-built YE3T Torch adapter and execute one source analysis."""

import importlib.util
import pathlib
import sys

import torch


def run_radial_smoke(device):
    from ye3t.runtime.execution_plan import (
        cheb_exp_cos_radial_table_with_derivative,
        cheb_exp_cos_radial_with_derivative,
    )

    radii = torch.tensor(
        [0.0, 0.17, 0.83, 1.72, 2.49, 3.1],
        dtype=torch.float64,
        device=device,
    )
    cutoffs = torch.tensor(
        [2.0, 2.2, 2.4, 2.6, 3.0, 3.0],
        dtype=torch.float64,
        device=device,
    )
    lambdas = torch.tensor(
        [1.1, 1.3, 1.7, 2.0, 2.4, 2.8],
        dtype=torch.float64,
        device=device,
    )
    for radial_index in (0, 1, 2, 5):
        expected_values, expected_derivatives = (
            cheb_exp_cos_radial_with_derivative(
                radii,
                cutoffs,
                lambdas,
                radial_index,
                backend="reference",
            )
        )
        actual_values, actual_derivatives = (
            torch.ops.ye3t_runtime.cheb_exp_cos_radial_with_derivative(
                radii,
                cutoffs,
                lambdas,
                radial_index,
            )
        )
        torch.testing.assert_close(
            actual_values,
            expected_values,
            rtol=1.0e-11,
            atol=1.0e-11,
        )
        torch.testing.assert_close(
            actual_derivatives,
            expected_derivatives,
            rtol=1.0e-10,
            atol=1.0e-10,
        )
    table_values, table_derivatives = (
        torch.ops.ye3t_runtime.cheb_exp_cos_radial_table_with_derivative(
            radii,
            cutoffs,
            lambdas,
            5,
        )
    )
    expected_table = cheb_exp_cos_radial_table_with_derivative(
        radii,
        cutoffs,
        lambdas,
        5,
        backend="reference",
    )
    torch.testing.assert_close(
        table_values,
        expected_table[0],
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    torch.testing.assert_close(
        table_derivatives,
        expected_table[1],
        rtol=1.0e-10,
        atol=1.0e-10,
    )

    tracked_radii = radii.detach().clone().requires_grad_(True)
    native_values = (
        torch.ops.ye3t_runtime.cheb_exp_cos_radial_with_derivative(
            tracked_radii,
            cutoffs,
            lambdas,
            5,
        )[0]
    )
    reference_radii = radii.detach().clone().requires_grad_(True)
    reference_values = cheb_exp_cos_radial_with_derivative(
        reference_radii,
        cutoffs,
        lambdas,
        5,
        backend="reference",
    )[0]
    value_adjoint = torch.linspace(
        -0.75,
        1.25,
        radii.numel(),
        dtype=radii.dtype,
        device=device,
    )
    native_gradient = torch.autograd.grad(
        native_values,
        tracked_radii,
        value_adjoint,
        create_graph=True,
    )[0]
    reference_gradient = torch.autograd.grad(
        reference_values,
        reference_radii,
        value_adjoint,
        create_graph=True,
    )[0]
    torch.testing.assert_close(
        native_gradient,
        reference_gradient,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    tangent = torch.flip(value_adjoint, dims=(0,))
    torch.testing.assert_close(
        torch.autograd.grad(
            native_gradient,
            tracked_radii,
            tangent,
        )[0],
        torch.autograd.grad(
            reference_gradient,
            reference_radii,
            tangent,
        )[0],
        rtol=1.0e-9,
        atol=1.0e-9,
    )

    if device.type == "cuda":
        torch.library.opcheck(
            torch.ops.ye3t_runtime.cheb_exp_cos_radial_with_derivative.default,
            (radii, cutoffs, lambdas, 5),
        )
        torch.library.opcheck(
            torch.ops.ye3t_runtime.cheb_exp_cos_radial_table_with_derivative.default,
            (radii, cutoffs, lambdas, 5),
        )


def run_spherical_smoke(device):
    from ye3t.runtime.execution_plan import (
        spherical_harmonics_table_with_derivative,
        spherical_harmonics_with_derivative,
    )

    edge_vectors = torch.tensor(
        [
            [0.31, 0.47, 0.83],
            [-0.42, 0.58, 0.27],
            [0.73, -0.24, 0.51],
            [-0.67, -0.19, 0.37],
            [0.28, 0.63, -0.44],
        ],
        dtype=torch.float64,
        device=device,
    )
    for angular_momentum in (0, 1, 2, 5, 8, 12):
        for real_output in (False, True):
            expected_values, expected_derivatives = (
                spherical_harmonics_with_derivative(
                    edge_vectors,
                    angular_momentum,
                    real_output=real_output,
                    backend="reference",
                )
            )
            actual_values, actual_derivatives = (
                torch.ops.ye3t_runtime.spherical_harmonics_with_derivative(
                    edge_vectors,
                    angular_momentum,
                    real_output,
                    1.0e-12,
                )
            )
            tolerance = 5.0e-10 if angular_momentum >= 8 else 2.0e-11
            torch.testing.assert_close(
                actual_values,
                expected_values,
                rtol=tolerance,
                atol=tolerance,
            )
            torch.testing.assert_close(
                actual_derivatives,
                expected_derivatives,
                rtol=tolerance * 10.0,
                atol=tolerance * 10.0,
            )
    table_values, table_derivatives = (
        torch.ops.ye3t_runtime.spherical_harmonics_table_with_derivative(
            edge_vectors,
            5,
            False,
            1.0e-12,
        )
    )
    expected_table = spherical_harmonics_table_with_derivative(
        edge_vectors,
        5,
        real_output=False,
        backend="reference",
    )
    torch.testing.assert_close(
        table_values,
        expected_table[0],
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    torch.testing.assert_close(
        table_derivatives,
        expected_table[1],
        rtol=2.0e-10,
        atol=2.0e-10,
    )

    float_vectors = edge_vectors.to(torch.float32)
    for angular_momentum in (1, 5, 8):
        for real_output in (False, True):
            expected_values, expected_derivatives = (
                spherical_harmonics_with_derivative(
                    float_vectors,
                    angular_momentum,
                    real_output=real_output,
                    backend="reference",
                )
            )
            actual_values, actual_derivatives = (
                torch.ops.ye3t_runtime.spherical_harmonics_with_derivative(
                    float_vectors,
                    angular_momentum,
                    real_output,
                    1.0e-12,
                )
            )
            torch.testing.assert_close(
                actual_values,
                expected_values,
                rtol=5.0e-5,
                atol=5.0e-5,
            )
            torch.testing.assert_close(
                actual_derivatives,
                expected_derivatives,
                rtol=5.0e-5,
                atol=5.0e-5,
            )

    tracked_vectors = edge_vectors.detach().clone().requires_grad_(True)
    native_values = torch.ops.ye3t_runtime.spherical_harmonics_with_derivative(
        tracked_vectors,
        5,
        False,
        1.0e-12,
    )[0]
    reference_vectors = (
        edge_vectors.detach().clone().requires_grad_(True)
    )
    reference_values = spherical_harmonics_with_derivative(
        reference_vectors,
        5,
        real_output=False,
        backend="reference",
    )[0]
    value_adjoint = torch.linspace(
        -0.75,
        1.25,
        native_values.numel(),
        dtype=torch.float64,
        device=device,
    ).reshape(native_values.shape)
    value_adjoint = value_adjoint + 1j * torch.flip(
        value_adjoint,
        dims=(1,),
    )
    native_gradient = torch.autograd.grad(
        native_values,
        tracked_vectors,
        value_adjoint,
        create_graph=True,
    )[0]
    reference_gradient = torch.autograd.grad(
        reference_values,
        reference_vectors,
        value_adjoint,
        create_graph=True,
    )[0]
    torch.testing.assert_close(
        native_gradient,
        reference_gradient,
        rtol=2.0e-10,
        atol=2.0e-10,
    )
    tangent = torch.flip(tracked_vectors.detach(), dims=(0,))
    torch.testing.assert_close(
        torch.autograd.grad(
            native_gradient,
            tracked_vectors,
            tangent,
        )[0],
        torch.autograd.grad(
            reference_gradient,
            reference_vectors,
            tangent,
        )[0],
        rtol=2.0e-9,
        atol=2.0e-9,
    )

    zero_vector = edge_vectors.new_zeros((1, 3))
    zero_values, zero_derivatives = (
        torch.ops.ye3t_runtime.spherical_harmonics_with_derivative(
            zero_vector,
            4,
            False,
            1.0e-12,
        )
    )
    if not torch.isfinite(zero_values).all():
        raise RuntimeError("zero-edge spherical values must be finite")
    if not torch.isfinite(zero_derivatives).all():
        raise RuntimeError("zero-edge spherical derivatives must be finite")
    if device.type == "cuda":
        cpu_zero = torch.zeros((1, 3), dtype=torch.float64)
        cpu_values, cpu_derivatives = (
            torch.ops.ye3t_runtime.spherical_harmonics_with_derivative(
                cpu_zero,
                4,
                False,
                1.0e-12,
            )
        )
        torch.testing.assert_close(
            zero_values.cpu(),
            cpu_values,
            rtol=1.0e-11,
            atol=1.0e-11,
        )
        torch.testing.assert_close(
            zero_derivatives.cpu(),
            cpu_derivatives,
            rtol=1.0e-10,
            atol=1.0e-10,
        )

    empty_vectors = edge_vectors.new_empty((0, 3))
    empty_values, empty_derivatives = (
        torch.ops.ye3t_runtime.spherical_harmonics_with_derivative(
            empty_vectors,
            3,
            False,
            1.0e-12,
        )
    )
    if tuple(empty_values.shape) != (0, 7):
        raise RuntimeError("empty spherical values have the wrong shape")
    if tuple(empty_derivatives.shape) != (0, 7, 3):
        raise RuntimeError("empty spherical derivatives have the wrong shape")

    if device.type == "cuda":
        torch.library.opcheck(
            torch.ops.ye3t_runtime.spherical_harmonics_with_derivative.default,
            (edge_vectors, 5, False, 1.0e-12),
        )
        torch.library.opcheck(
            torch.ops.ye3t_runtime.spherical_harmonics_table_with_derivative.default,
            (edge_vectors, 5, False, 1.0e-12),
        )


def run_plain_source_product_smoke(device, dtype):
    from ye3t.runtime.execution_plan import (
        plain_site_basis_product_adjoint,
        plain_site_basis_product_with_derivative,
    )

    edge_count = 5
    radial_values = torch.randn(3, edge_count, dtype=dtype, device=device)
    radial_derivatives = torch.randn_like(radial_values)
    angular_values = torch.randn(6, edge_count, dtype=dtype, device=device)
    angular_derivatives = torch.randn(
        6,
        edge_count,
        3,
        dtype=dtype,
        device=device,
    )
    prefactors = torch.randn_like(radial_values)
    prefactor_center = torch.randn_like(radial_values)
    prefactor_neighbor = torch.randn_like(radial_values)
    radial_directions = torch.randn(
        edge_count,
        3,
        dtype=dtype,
        device=device,
    )
    term_groups = torch.tensor(
        [0, 0, 1, 1, 2, 2],
        dtype=torch.int64,
        device=device,
    )
    term_channels = torch.tensor(
        [7, 1, 5, 0, 4, 2],
        dtype=torch.int64,
        device=device,
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
    actual = (
        torch.ops.ye3t_runtime.plain_site_basis_product_with_derivative(
            *arguments
        )
    )
    expected = plain_site_basis_product_with_derivative(
        *arguments,
        backend="reference",
    )
    for actual_value, expected_value in zip(actual, expected):
        torch.testing.assert_close(
            actual_value,
            expected_value,
            rtol=2.0e-11,
            atol=2.0e-11,
        )
    if device.type == "cuda":
        result = torch.library.opcheck(
            torch.ops.ye3t_runtime.plain_site_basis_product_with_derivative.default,
            arguments,
            raise_exception=False,
        )
        if not all(value == "SUCCESS" for value in result.values()):
            raise RuntimeError(
                f"plain source product CUDA opcheck failed: {result}"
            )

    edge_weights = torch.randn(edge_count, dtype=dtype, device=device)
    edge_weight_derivatives = torch.randn(
        edge_count,
        3,
        dtype=dtype,
        device=device,
    )
    edge_adjoint = torch.randn(
        edge_count,
        8,
        dtype=dtype,
        device=device,
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
    actual_adjoint = torch.ops.ye3t_runtime.plain_site_basis_product_adjoint(
        *adjoint_arguments
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
            rtol=2.0e-11,
            atol=2.0e-11,
        )
    if device.type == "cuda":
        result = torch.library.opcheck(
            torch.ops.ye3t_runtime.plain_site_basis_product_adjoint.default,
            adjoint_arguments,
            raise_exception=False,
        )
        if not all(value == "SUCCESS" for value in result.values()):
            raise RuntimeError(
                f"plain source adjoint CUDA opcheck failed: {result}"
            )


def run_density_smoke(device, dtype):
    edge_values = torch.tensor(
        [
            [1.0, -0.5, 0.25],
            [2.0, 0.75, -1.0],
            [-0.25, 1.5, 0.5],
            [0.75, -1.25, 2.0],
            [1.25, 0.5, -0.75],
        ],
        dtype=dtype,
        device=device,
    )
    if dtype.is_complex:
        edge_values = edge_values + 1j * torch.tensor(
            [
                [0.25, 0.5, -0.75],
                [-1.0, 0.25, 0.5],
                [0.75, -0.5, 1.0],
                [0.5, 1.25, -0.25],
                [-0.5, 0.75, 0.25],
            ],
            dtype=dtype,
            device=device,
        )
    centers = torch.tensor(
        [0, 2, 1, 2, 0],
        dtype=torch.int64,
        device=device,
    )
    expected = edge_values.new_zeros((3, edge_values.shape[1]))
    expected.index_add_(0, centers, edge_values)
    actual = torch.ops.ye3t_runtime.density_accumulate(
        edge_values,
        centers,
        3,
    )
    tolerance = 1.0e-11 if dtype in (torch.float64, torch.complex128) else 5.0e-5
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

    empty_edges = edge_values.new_empty((0, edge_values.shape[1]))
    empty_centers = centers.new_empty((0,))
    torch.testing.assert_close(
        torch.ops.ye3t_runtime.density_accumulate(
            empty_edges,
            empty_centers,
            3,
        ),
        edge_values.new_zeros((3, edge_values.shape[1])),
        rtol=0.0,
        atol=0.0,
    )
    empty_adjoint = torch.ops.ye3t_runtime.density_accumulate_adjoint(
        expected,
        empty_centers,
    )
    if tuple(empty_adjoint.shape) != (0, edge_values.shape[1]):
        raise RuntimeError("empty density adjoint has the wrong shape")

    atomic_adjoint = torch.flip(expected, dims=(1,))
    expected_edge_adjoint = atomic_adjoint.index_select(0, centers)
    edge_adjoint = torch.ops.ye3t_runtime.density_accumulate_adjoint(
        atomic_adjoint,
        centers,
    )
    torch.testing.assert_close(
        edge_adjoint,
        expected_edge_adjoint,
        rtol=tolerance,
        atol=tolerance,
    )

    tracked_edges = edge_values.detach().clone().requires_grad_(True)
    tracked_atomic_adjoint = (
        atomic_adjoint.detach().clone().requires_grad_(True)
    )
    tracked_atomic = torch.ops.ye3t_runtime.density_accumulate(
        tracked_edges,
        centers,
        3,
    )
    differentiable_edge_adjoint = torch.autograd.grad(
        tracked_atomic,
        tracked_edges,
        tracked_atomic_adjoint,
        create_graph=True,
    )[0]
    torch.testing.assert_close(
        differentiable_edge_adjoint,
        expected_edge_adjoint,
        rtol=tolerance,
        atol=tolerance,
    )
    edge_tangent = torch.flip(edge_values, dims=(0,))
    atomic_tangent = torch.autograd.grad(
        differentiable_edge_adjoint,
        tracked_atomic_adjoint,
        edge_tangent,
    )[0]
    expected_atomic_tangent = edge_tangent.new_zeros(
        (3, edge_tangent.shape[1])
    )
    expected_atomic_tangent.index_add_(0, centers, edge_tangent)
    torch.testing.assert_close(
        atomic_tangent,
        expected_atomic_tangent,
        rtol=tolerance,
        atol=tolerance,
    )

    if device.type == "cuda" and dtype == torch.float64:
        torch.library.opcheck(
            torch.ops.ye3t_runtime.density_accumulate.default,
            (edge_values, centers, 3),
        )


def run_compact_pair_smoke(device, dtype):
    from ye3t.runtime.execution_plan import (
        compact_pair_product_adjoint_reference,
        compact_pair_product_reference,
    )

    left = torch.tensor(
        [
            [0.25, -0.5, 1.0, 0.75, -1.25],
            [1.5, 0.125, -0.75, 0.5, 0.25],
            [-0.625, 1.25, 0.375, -1.0, 0.875],
        ],
        dtype=dtype,
        device=device,
    )
    right = torch.tensor(
        [
            [-0.75, 1.25, 0.5, -0.25, 0.625],
            [0.875, -1.0, 0.25, 1.5, -0.5],
            [1.125, 0.5, -0.875, 0.75, 0.125],
        ],
        dtype=dtype,
        device=device,
    )
    if dtype.is_complex:
        left = left + 1j * torch.flip(left, dims=(1,))
        right = right - 0.5j * torch.flip(right, dims=(0,))

    tolerance = (
        1.0e-11
        if dtype in (torch.float64, torch.complex128)
        else 5.0e-5
    )
    for antisymmetric in (False, True):
        expected = compact_pair_product_reference(
            left,
            right,
            antisymmetric=antisymmetric,
        )
        actual = torch.ops.ye3t_runtime.compact_pair_product(
            left,
            right,
            antisymmetric,
        )
        torch.testing.assert_close(
            actual,
            expected,
            rtol=tolerance,
            atol=tolerance,
        )

        output_adjoint = torch.linspace(
            -1.0,
            1.0,
            actual.numel(),
            dtype=left.real.dtype,
            device=device,
        ).reshape(actual.shape)
        if dtype.is_complex:
            output_adjoint = output_adjoint + 0.75j * torch.flip(
                output_adjoint,
                dims=(1,),
            )
        expected_left_adjoint, expected_right_adjoint = (
            compact_pair_product_adjoint_reference(
                output_adjoint,
                left,
                right,
                antisymmetric=antisymmetric,
            )
        )
        left_adjoint, right_adjoint = (
            torch.ops.ye3t_runtime.compact_pair_product_adjoint(
                output_adjoint,
                left,
                right,
                antisymmetric,
            )
        )
        torch.testing.assert_close(
            left_adjoint,
            expected_left_adjoint,
            rtol=tolerance,
            atol=tolerance,
        )
        torch.testing.assert_close(
            right_adjoint,
            expected_right_adjoint,
            rtol=tolerance,
            atol=tolerance,
        )

        tracked_left = left.detach().clone().requires_grad_(True)
        tracked_right = right.detach().clone().requires_grad_(True)
        tracked_output_adjoint = (
            output_adjoint.detach().clone().requires_grad_(True)
        )
        differentiable_output = (
            torch.ops.ye3t_runtime.compact_pair_product(
                tracked_left,
                tracked_right,
                antisymmetric,
            )
        )
        differentiable_left, differentiable_right = torch.autograd.grad(
            differentiable_output,
            (tracked_left, tracked_right),
            tracked_output_adjoint,
            create_graph=True,
        )
        torch.testing.assert_close(
            differentiable_left,
            expected_left_adjoint,
            rtol=tolerance,
            atol=tolerance,
        )
        torch.testing.assert_close(
            differentiable_right,
            expected_right_adjoint,
            rtol=tolerance,
            atol=tolerance,
        )

        left_tangent = torch.flip(left, dims=(1,))
        right_tangent = torch.flip(right, dims=(0,))
        output_tangent = torch.autograd.grad(
            (differentiable_left, differentiable_right),
            tracked_output_adjoint,
            (left_tangent, right_tangent),
        )[0]
        expected_tangent = compact_pair_product_reference(
            left_tangent,
            right,
            antisymmetric=antisymmetric,
        ) + compact_pair_product_reference(
            left,
            right_tangent,
            antisymmetric=antisymmetric,
        )
        torch.testing.assert_close(
            output_tangent,
            expected_tangent,
            rtol=tolerance * 5.0,
            atol=tolerance * 5.0,
        )

        if device.type == "cuda" and dtype == torch.float64:
            torch.library.opcheck(
                torch.ops.ye3t_runtime.compact_pair_product.default,
                (left, right, antisymmetric),
            )


def run_compact_exterior_power_smoke(device, dtype):
    from ye3t.runtime.execution_plan import (
        compact_exterior_power_adjoint_reference,
        compact_exterior_power_product_reference,
    )

    tolerance = (
        2.0e-10
        if dtype in (torch.float64, torch.complex128)
        else 5.0e-5
    )
    for order, dimension in (
        (1, 4),
        (2, 5),
        (3, 5),
        (4, 6),
        (6, 7),
        (8, 8),
    ):
        coordinates = torch.arange(
            2 * order * dimension,
            dtype=torch.float64,
            device=device,
        ).reshape(2, order, dimension)
        factors = (
            0.5 * torch.sin(0.37 * coordinates + 0.11)
            + 0.35 * torch.cos(0.23 * coordinates - 0.19)
        ).to(dtype)
        if dtype.is_complex:
            factors = factors + 1j * (
                0.4 * torch.cos(0.29 * coordinates + 0.17)
            ).to(dtype)
        expected = compact_exterior_power_product_reference(factors)
        actual = torch.ops.ye3t_runtime.compact_exterior_power(factors)
        torch.testing.assert_close(
            actual,
            expected,
            rtol=tolerance,
            atol=tolerance,
        )

        output_adjoint = torch.linspace(
            -0.75,
            1.25,
            actual.numel(),
            dtype=factors.real.dtype,
            device=device,
        ).reshape(actual.shape)
        if dtype.is_complex:
            output_adjoint = output_adjoint + 0.5j * torch.flip(
                output_adjoint,
                dims=(0,),
            )
        expected_adjoint = compact_exterior_power_adjoint_reference(
            output_adjoint,
            factors,
        )
        actual_adjoint = (
            torch.ops.ye3t_runtime.compact_exterior_power_adjoint(
                output_adjoint,
                factors,
            )
        )
        torch.testing.assert_close(
            actual_adjoint,
            expected_adjoint,
            rtol=tolerance * 5.0,
            atol=tolerance * 5.0,
        )

    order = 3
    dimension = 5
    factors = torch.randn(
        2,
        order,
        dimension,
        dtype=dtype,
        device=device,
    )
    tracked_native = factors.detach().clone().requires_grad_(True)
    tracked_reference = factors.detach().clone().requires_grad_(True)
    native_output = torch.ops.ye3t_runtime.compact_exterior_power(
        tracked_native
    )
    reference_output = compact_exterior_power_product_reference(
        tracked_reference
    )
    output_adjoint = torch.randn_like(native_output)
    native_gradient = torch.autograd.grad(
        native_output,
        tracked_native,
        output_adjoint,
        create_graph=True,
    )[0]
    reference_gradient = torch.autograd.grad(
        reference_output,
        tracked_reference,
        output_adjoint,
        create_graph=True,
    )[0]
    torch.testing.assert_close(
        native_gradient,
        reference_gradient,
        rtol=tolerance * 5.0,
        atol=tolerance * 5.0,
    )
    tangent = torch.flip(factors, dims=(2,))
    native_hvp = torch.autograd.grad(
        native_gradient,
        tracked_native,
        tangent,
    )[0]
    reference_hvp = torch.autograd.grad(
        reference_gradient,
        tracked_reference,
        tangent,
    )[0]
    torch.testing.assert_close(
        native_hvp,
        reference_hvp,
        rtol=tolerance * 10.0,
        atol=tolerance * 10.0,
    )

    duplicated = factors.detach().clone()
    duplicated[:, 1, :] = duplicated[:, 0, :]
    duplicate_output = torch.ops.ye3t_runtime.compact_exterior_power(
        duplicated
    )
    torch.testing.assert_close(
        duplicate_output,
        torch.zeros_like(duplicate_output),
        rtol=0.0,
        atol=tolerance,
    )
    duplicate_output_adjoint = torch.randn_like(duplicate_output)
    torch.testing.assert_close(
        torch.ops.ye3t_runtime.compact_exterior_power_adjoint(
            duplicate_output_adjoint,
            duplicated,
        ),
        compact_exterior_power_adjoint_reference(
            duplicate_output_adjoint,
            duplicated,
        ),
        rtol=tolerance * 10.0,
        atol=tolerance * 10.0,
    )

    if device.type == "cuda" and dtype == torch.float64:
        torch.library.opcheck(
            torch.ops.ye3t_runtime.compact_exterior_power.default,
            (factors,),
        )


def run_symmetric_power_monomial_smoke(device, dtype):
    from ye3t.runtime.execution_plan import (
        symmetric_power_monomial_adjoint_reference,
        symmetric_power_monomial_reference,
    )

    input = torch.tensor(
        [
            [0.0, -0.5, 1.25],
            [1.5, 0.0, -0.75],
            [-0.625, 1.25, 0.0],
        ],
        dtype=dtype,
        device=device,
    )
    if dtype.is_complex:
        input = input + 0.5j * torch.flip(input, dims=(1,))
    counts = torch.tensor(
        [
            [2, 0, 0],
            [0, 1, 1],
            [1, 1, 0],
            [0, 0, 2],
        ],
        dtype=torch.int64,
        device=device,
    )
    offsets = torch.tensor(
        [0, 2, 3, 3, 4],
        dtype=torch.int64,
        device=device,
    )
    indices = torch.tensor(
        [0, 0, 1, 3],
        dtype=torch.int64,
        device=device,
    )
    values = torch.tensor(
        [0.75, -1.25, 0.5, 1.5],
        dtype=dtype,
        device=device,
    )
    if dtype.is_complex:
        values = values + 1j * torch.tensor(
            [0.25, -0.5, 0.75, 0.125],
            dtype=dtype,
            device=device,
        )
    expected = symmetric_power_monomial_reference(
        input,
        counts,
        offsets,
        indices,
        values,
    )
    actual = torch.ops.ye3t_runtime.symmetric_power_monomial(
        input,
        counts,
        offsets,
        indices,
        values,
    )
    tolerance = (
        1.0e-11
        if dtype in (torch.float64, torch.complex128)
        else 5.0e-5
    )
    torch.testing.assert_close(
        actual,
        expected,
        rtol=tolerance,
        atol=tolerance,
    )

    output_adjoint = torch.flip(actual.detach(), dims=(1,))
    expected_adjoint = symmetric_power_monomial_adjoint_reference(
        output_adjoint,
        input,
        counts,
        offsets,
        indices,
        values,
    )
    actual_adjoint = (
        torch.ops.ye3t_runtime.symmetric_power_monomial_adjoint(
            output_adjoint,
            input,
            counts,
            offsets,
            indices,
            values,
        )
    )
    torch.testing.assert_close(
        actual_adjoint,
        expected_adjoint,
        rtol=tolerance,
        atol=tolerance,
    )

    tracked_native = input.detach().clone().requires_grad_(True)
    tracked_reference = input.detach().clone().requires_grad_(True)
    native_output = torch.ops.ye3t_runtime.symmetric_power_monomial(
        tracked_native,
        counts,
        offsets,
        indices,
        values,
    )
    reference_output = symmetric_power_monomial_reference(
        tracked_reference,
        counts,
        offsets,
        indices,
        values,
    )
    fixed_output_adjoint = torch.randn_like(native_output)
    native_gradient = torch.autograd.grad(
        native_output,
        tracked_native,
        fixed_output_adjoint,
        create_graph=True,
    )[0]
    reference_gradient = torch.autograd.grad(
        reference_output,
        tracked_reference,
        fixed_output_adjoint,
        create_graph=True,
    )[0]
    torch.testing.assert_close(
        native_gradient,
        reference_gradient,
        rtol=tolerance,
        atol=tolerance,
    )
    tangent = torch.flip(input, dims=(0,))
    torch.testing.assert_close(
        torch.autograd.grad(
            native_gradient,
            tracked_native,
            tangent,
        )[0],
        torch.autograd.grad(
            reference_gradient,
            tracked_reference,
            tangent,
        )[0],
        rtol=tolerance * 5.0,
        atol=tolerance * 5.0,
    )

    if device.type == "cuda" and dtype == torch.float64:
        torch.library.opcheck(
            torch.ops.ye3t_runtime.symmetric_power_monomial.default,
            (input, counts, offsets, indices, values),
        )


def run_shared_symmetric_power_monomial_smoke(device, dtype):
    from ye3t.runtime.execution_plan import (
        symmetric_power_shared_monomial_adjoint_reference,
        symmetric_power_shared_monomial_batched_adjoint_reference,
        symmetric_power_shared_monomial_reference,
    )

    input = torch.tensor(
        [
            [0.0, -0.5, 1.25],
            [1.5, 0.0, -0.75],
            [-0.625, 1.25, 0.0],
        ],
        dtype=dtype,
        device=device,
    )
    if dtype.is_complex:
        input = input + 0.5j * torch.flip(input, dims=(1,))
    counts = torch.tensor(
        [
            [2, 0, 0],
            [0, 1, 1],
            [1, 1, 0],
            [0, 0, 2],
        ],
        dtype=torch.int64,
        device=device,
    )
    offsets = torch.tensor([0, 2, 4, 5], dtype=torch.int64, device=device)
    terms = torch.tensor([0, 1, 1, 2, 0], dtype=torch.int64, device=device)
    outputs = torch.tensor([0, 0, 1, 1, 2], dtype=torch.int64, device=device)
    values = torch.tensor(
        [0.75, -1.25, 0.5, -0.625, 1.5],
        dtype=dtype,
        device=device,
    )
    if dtype.is_complex:
        values = values + 1j * torch.tensor(
            [0.25, -0.5, 0.75, 0.125, -0.375],
            dtype=dtype,
            device=device,
        )
    expected = symmetric_power_shared_monomial_reference(
        input,
        counts,
        offsets,
        terms,
        outputs,
        values,
    )
    actual = torch.ops.ye3t_runtime.symmetric_power_shared_monomial(
        input,
        counts,
        offsets,
        terms,
        outputs,
        values,
    )
    tolerance = (
        1.0e-11
        if dtype in (torch.float64, torch.complex128)
        else 5.0e-5
    )
    torch.testing.assert_close(
        actual,
        expected,
        rtol=tolerance,
        atol=tolerance,
    )

    output_adjoint = torch.flip(actual.detach(), dims=(1,))
    expected_adjoint = symmetric_power_shared_monomial_adjoint_reference(
        output_adjoint,
        input,
        counts,
        offsets,
        terms,
        outputs,
        values,
    )
    actual_adjoint = (
        torch.ops.ye3t_runtime.symmetric_power_shared_monomial_adjoint(
            output_adjoint,
            input,
            counts,
            offsets,
            terms,
            outputs,
            values,
        )
    )
    torch.testing.assert_close(
        actual_adjoint,
        expected_adjoint,
        rtol=tolerance,
        atol=tolerance,
    )

    batched_output_adjoint = torch.stack(
        (
            output_adjoint,
            torch.roll(output_adjoint, shifts=1, dims=1),
        ),
        dim=0,
    )
    expected_batched_adjoint = (
        symmetric_power_shared_monomial_batched_adjoint_reference(
            batched_output_adjoint,
            input,
            counts,
            offsets,
            terms,
            outputs,
            values,
        )
    )
    actual_batched_adjoint = (
        torch.ops.ye3t_runtime
        .symmetric_power_shared_monomial_batched_adjoint(
            batched_output_adjoint,
            input,
            counts,
            offsets,
            terms,
            outputs,
            values,
        )
    )
    torch.testing.assert_close(
        actual_batched_adjoint,
        expected_batched_adjoint,
        rtol=tolerance,
        atol=tolerance,
    )

    tracked_native = input.detach().clone().requires_grad_(True)
    tracked_reference = input.detach().clone().requires_grad_(True)
    native_output = (
        torch.ops.ye3t_runtime.symmetric_power_shared_monomial(
            tracked_native,
            counts,
            offsets,
            terms,
            outputs,
            values,
        )
    )
    reference_output = symmetric_power_shared_monomial_reference(
        tracked_reference,
        counts,
        offsets,
        terms,
        outputs,
        values,
    )
    fixed_output_adjoint = torch.randn_like(native_output)
    native_gradient = torch.autograd.grad(
        native_output,
        tracked_native,
        fixed_output_adjoint,
        create_graph=True,
    )[0]
    reference_gradient = torch.autograd.grad(
        reference_output,
        tracked_reference,
        fixed_output_adjoint,
        create_graph=True,
    )[0]
    torch.testing.assert_close(
        native_gradient,
        reference_gradient,
        rtol=tolerance,
        atol=tolerance,
    )
    tangent = torch.flip(input, dims=(0,))
    torch.testing.assert_close(
        torch.autograd.grad(
            native_gradient,
            tracked_native,
            tangent,
        )[0],
        torch.autograd.grad(
            reference_gradient,
            tracked_reference,
            tangent,
        )[0],
        rtol=tolerance * 5.0,
        atol=tolerance * 5.0,
    )

    native_seed = (
        batched_output_adjoint.detach().clone().requires_grad_(True)
    )
    native_batched_input = input.detach().clone().requires_grad_(True)
    reference_seed = (
        batched_output_adjoint.detach().clone().requires_grad_(True)
    )
    reference_batched_input = (
        input.detach().clone().requires_grad_(True)
    )
    native_roots = (
        torch.ops.ye3t_runtime
        .symmetric_power_shared_monomial_batched_adjoint(
            native_seed,
            native_batched_input,
            counts,
            offsets,
            terms,
            outputs,
            values,
        )
    )
    reference_roots = (
        symmetric_power_shared_monomial_batched_adjoint_reference(
            reference_seed,
            reference_batched_input,
            counts,
            offsets,
            terms,
            outputs,
            values,
        )
    )
    root_tangent = torch.randn_like(native_roots)
    native_first = torch.autograd.grad(
        native_roots,
        (native_seed, native_batched_input),
        root_tangent,
        create_graph=True,
    )
    reference_first = torch.autograd.grad(
        reference_roots,
        (reference_seed, reference_batched_input),
        root_tangent,
        create_graph=True,
    )
    for actual_gradient, expected_gradient in zip(
        native_first,
        reference_first,
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=tolerance * 5.0,
            atol=tolerance * 5.0,
        )
    second_tangents = tuple(
        torch.randn_like(value)
        for value in native_first
    )
    native_second = torch.autograd.grad(
        native_first,
        (native_seed, native_batched_input),
        second_tangents,
    )
    reference_second = torch.autograd.grad(
        reference_first,
        (reference_seed, reference_batched_input),
        second_tangents,
    )
    for actual_gradient, expected_gradient in zip(
        native_second,
        reference_second,
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=tolerance * 10.0,
            atol=tolerance * 10.0,
        )

    if device.type == "cuda" and dtype == torch.float64:
        torch.library.opcheck(
            torch.ops.ye3t_runtime.symmetric_power_shared_monomial.default,
            (input, counts, offsets, terms, outputs, values),
        )
        torch.library.opcheck(
            torch.ops.ye3t_runtime
            .symmetric_power_shared_monomial_batched_adjoint.default,
            (
                batched_output_adjoint,
                input,
                counts,
                offsets,
                terms,
                outputs,
                values,
            ),
        )


def run_compiler_symmetric_power_smoke(device, dtype):
    from ye3t.runtime.symmetric_power import (
        _symmetric_power_count_table,
        symmetric_power_product_evaluator_real_tesseral,
    )

    tolerance = 2.0e-10 if dtype == torch.float64 else 5.0e-5
    for power, input_L, output_L in (
        (2, 2, 0),
        (3, 1, 3),
        (4, 2, 4),
        (8, 1, 4),
    ):
        input_dimension = 2 * input_L + 1
        coordinates = torch.arange(
            3 * input_dimension,
            dtype=torch.float64,
            device=device,
        ).reshape(3, input_dimension)
        input = (
            0.4 * torch.sin(0.31 * coordinates + 0.17)
            + 0.3 * torch.cos(0.23 * coordinates - 0.11)
        ).to(dtype)
        table = _symmetric_power_count_table(
            power,
            input_L,
            output_L,
            device=device,
            dtype=dtype,
        )
        actual = torch.ops.ye3t_runtime.symmetric_power_monomial(
            input,
            table.monomial_counts,
            table.output_offsets,
            table.output_index,
            table.monomial_value,
        )
        expected = symmetric_power_product_evaluator_real_tesseral(
            input,
            power,
            input_L,
            output_L,
        ).reshape(input.shape[0], -1)
        torch.testing.assert_close(
            actual,
            expected,
            rtol=tolerance,
            atol=tolerance,
        )
        tracked_native = input.detach().clone().requires_grad_(True)
        tracked_reference = input.detach().clone().requires_grad_(True)
        native_output = torch.ops.ye3t_runtime.symmetric_power_monomial(
            tracked_native,
            table.monomial_counts,
            table.output_offsets,
            table.output_index,
            table.monomial_value,
        )
        reference_output = (
            symmetric_power_product_evaluator_real_tesseral(
                tracked_reference,
                power,
                input_L,
                output_L,
            ).reshape(input.shape[0], -1)
        )
        output_adjoint = torch.randn_like(native_output)
        torch.testing.assert_close(
            torch.autograd.grad(
                native_output,
                tracked_native,
                output_adjoint,
            )[0],
            torch.autograd.grad(
                reference_output,
                tracked_reference,
                output_adjoint,
            )[0],
            rtol=tolerance * 5.0,
            atol=tolerance * 5.0,
        )
        if (
            power == 4
            and input_L == 2
            and output_L == 4
            and int(table.multiplicity) <= 1
        ):
            raise RuntimeError(
                "compiler symmetric-power smoke requires multiplicity > 1"
            )


def run_source_analysis_smoke(device, dtype):
    complex_dtype = dtype.is_complex
    source = torch.tensor(
        [[2.0, 4.0], [-1.0, 3.0]],
        dtype=dtype,
        device=device,
    )
    if complex_dtype:
        source = source + 1j * torch.tensor(
            [[0.5, -1.0], [2.0, 0.25]],
            dtype=dtype,
            device=device,
        )
    rows = torch.tensor([0, 1, 1], dtype=torch.int64, device=device)
    columns = torch.tensor([0, 1, 0], dtype=torch.int64, device=device)
    assembly_values = torch.tensor(
        [1.0, 0.5, -0.25],
        dtype=dtype,
        device=device,
    )
    synthesis_rows = torch.tensor(
        [0, 1, 1],
        dtype=torch.int64,
        device=device,
    )
    synthesis_columns = torch.tensor(
        [0, 0, 1],
        dtype=torch.int64,
        device=device,
    )
    synthesis_values = torch.tensor(
        [0.75, -0.5, 1.25],
        dtype=dtype,
        device=device,
    )
    if complex_dtype:
        assembly_values = assembly_values + 1j * torch.tensor(
            [0.25, -0.5, 0.75],
            dtype=dtype,
            device=device,
        )
        synthesis_values = synthesis_values + 1j * torch.tensor(
            [-0.5, 0.25, 0.5],
            dtype=dtype,
            device=device,
        )

    ambient = source.new_zeros((source.shape[0], 2))
    ambient.index_add_(
        1,
        rows,
        source.index_select(1, columns)
        * assembly_values.reshape(1, -1),
    )
    synthesis = source.new_zeros((2, 2))
    synthesis.index_put_(
        (synthesis_rows, synthesis_columns),
        synthesis_values,
        accumulate=True,
    )
    expected = ambient @ synthesis.conj()
    output = torch.ops.ye3t_runtime.source_analysis(
        source,
        rows,
        columns,
        assembly_values,
        synthesis_rows,
        synthesis_columns,
        synthesis_values,
        2,
        2,
    )
    tolerance = 1.0e-11 if dtype in (torch.float64, torch.complex128) else 5.0e-5
    torch.testing.assert_close(
        output,
        expected,
        rtol=tolerance,
        atol=tolerance,
    )

    output_adjoint = torch.tensor(
        [[0.5, -1.0], [1.5, 0.25]],
        dtype=dtype,
        device=device,
    )
    if complex_dtype:
        output_adjoint = output_adjoint + 1j * torch.tensor(
            [[0.25, 0.75], [-0.5, 1.0]],
            dtype=dtype,
            device=device,
        )
    tracked_source = source.detach().clone().requires_grad_(True)
    tracked_ambient = tracked_source.new_zeros((source.shape[0], 2))
    tracked_ambient.index_add_(
        1,
        rows,
        tracked_source.index_select(1, columns)
        * assembly_values.reshape(1, -1),
    )
    tracked_output = tracked_ambient @ synthesis.conj()
    expected_source_adjoint = torch.autograd.grad(
        tracked_output,
        tracked_source,
        output_adjoint,
    )[0]
    source_adjoint = torch.ops.ye3t_runtime.source_analysis_adjoint(
        output_adjoint.contiguous(),
        rows,
        columns,
        assembly_values,
        synthesis_rows,
        synthesis_columns,
        synthesis_values,
        2,
        2,
    )
    torch.testing.assert_close(
        source_adjoint,
        expected_source_adjoint,
        rtol=tolerance,
        atol=tolerance,
    )

    weight = torch.tensor([0.75, -1.25], dtype=dtype, device=device)
    bias = torch.tensor(0.125, dtype=dtype, device=device)
    if complex_dtype:
        weight = weight + 1j * torch.tensor(
            [0.5, -0.25],
            dtype=dtype,
            device=device,
        )
        bias = bias + 0.375j
    expected_linear = expected @ weight + bias
    linear = torch.ops.ye3t_runtime.source_analysis_linear(
        source,
        rows,
        columns,
        assembly_values,
        synthesis_rows,
        synthesis_columns,
        synthesis_values,
        2,
        weight,
        bias,
    )
    torch.testing.assert_close(
        linear,
        expected_linear,
        rtol=tolerance,
        atol=tolerance,
    )

    linear_adjoint = torch.tensor(
        [0.5, -1.5],
        dtype=dtype,
        device=device,
    )
    if complex_dtype:
        linear_adjoint = linear_adjoint + 1j * torch.tensor(
            [0.25, 0.75],
            dtype=dtype,
            device=device,
        )
    tracked_source = source.detach().clone().requires_grad_(True)
    tracked_weight = weight.detach().clone().requires_grad_(True)
    tracked_ambient = tracked_source.new_zeros((source.shape[0], 2))
    tracked_ambient.index_add_(
        1,
        rows,
        tracked_source.index_select(1, columns)
        * assembly_values.reshape(1, -1),
    )
    tracked_linear = tracked_ambient @ synthesis.conj() @ tracked_weight + bias
    expected_source, expected_weight = torch.autograd.grad(
        tracked_linear,
        (tracked_source, tracked_weight),
        linear_adjoint,
    )
    source_gradient, weight_gradient, bias_gradient = (
        torch.ops.ye3t_runtime.source_analysis_linear_adjoint(
            linear_adjoint.contiguous(),
            source,
            rows,
            columns,
            assembly_values,
            synthesis_rows,
            synthesis_columns,
            synthesis_values,
            2,
            weight,
        )
    )
    torch.testing.assert_close(
        source_gradient,
        expected_source,
        rtol=tolerance,
        atol=tolerance,
    )
    torch.testing.assert_close(
        weight_gradient,
        expected_weight,
        rtol=tolerance,
        atol=tolerance,
    )
    torch.testing.assert_close(
        bias_gradient,
        linear_adjoint.sum(),
        rtol=tolerance,
        atol=tolerance,
    )

    tracked_source = source.detach().clone().requires_grad_(True)
    tracked_output_adjoint = (
        output_adjoint.detach().clone().requires_grad_(True)
    )
    differentiable_output = torch.ops.ye3t_runtime.source_analysis(
        tracked_source,
        rows,
        columns,
        assembly_values,
        synthesis_rows,
        synthesis_columns,
        synthesis_values,
        2,
        2,
    )
    differentiable_source_adjoint = torch.autograd.grad(
        differentiable_output,
        tracked_source,
        tracked_output_adjoint,
        create_graph=True,
    )[0]
    torch.testing.assert_close(
        differentiable_source_adjoint,
        expected_source_adjoint,
        rtol=tolerance,
        atol=tolerance,
    )
    source_tangent = torch.flip(source, dims=(1,))
    output_tangent = torch.autograd.grad(
        differentiable_source_adjoint,
        tracked_output_adjoint,
        source_tangent,
    )[0]
    expected_tangent_ambient = source_tangent.new_zeros(
        (source.shape[0], 2)
    )
    expected_tangent_ambient.index_add_(
        1,
        rows,
        source_tangent.index_select(1, columns)
        * assembly_values.reshape(1, -1),
    )
    expected_output_tangent = expected_tangent_ambient @ synthesis.conj()
    torch.testing.assert_close(
        output_tangent,
        expected_output_tangent,
        rtol=tolerance,
        atol=tolerance,
    )

    if device.type == "cuda" and dtype == torch.float64:
        torch.library.opcheck(
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
                2,
            ),
        )


def run_factorized_angular_smoke(device, dtype):
    from ye3t.runtime.execution_plan import (
        _factorized_angular_packed_reference,
    )

    node_offsets = torch.tensor(
        [0, 3, 6],
        dtype=torch.int64,
        device=device,
    )
    node_dimensions = torch.tensor(
        [3, 3, 1],
        dtype=torch.int64,
        device=device,
    )
    node_leaf_offsets = torch.tensor(
        [0, 3, -1],
        dtype=torch.int64,
        device=device,
    )
    node_left = torch.tensor(
        [-1, -1, 0],
        dtype=torch.int64,
        device=device,
    )
    node_right = torch.tensor(
        [-1, -1, 1],
        dtype=torch.int64,
        device=device,
    )
    node_coefficient_offsets = torch.tensor(
        [0, 0, 0, 3],
        dtype=torch.int64,
        device=device,
    )
    coefficient_rows = torch.tensor(
        [0, 4, 8],
        dtype=torch.int64,
        device=device,
    )
    coefficient_columns = torch.tensor(
        [0, 0, 0],
        dtype=torch.int64,
        device=device,
    )
    coefficient_values = torch.tensor(
        [0.5, -0.25, 0.75],
        dtype=dtype,
        device=device,
    )
    root_nodes = torch.tensor([2], dtype=torch.int64, device=device)
    projection_values = torch.tensor(
        [1.0, 0.5, -0.25, 0.75],
        dtype=dtype,
        device=device,
    )
    if dtype.is_complex:
        coefficient_values = coefficient_values + 1j * torch.tensor(
            [0.125, -0.375, 0.25],
            dtype=dtype,
            device=device,
        )
        projection_values = projection_values + 1j * torch.tensor(
            [0.25, -0.125, 0.5, 0.375],
            dtype=dtype,
            device=device,
        )
    source_dimension = 2
    workspace_dimension = 7
    output_dimension = 2
    packed = torch.randn(
        3,
        source_dimension * 6,
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
    native_input = packed.detach().clone().requires_grad_(True)
    reference_input = packed.detach().clone().requires_grad_(True)
    native = torch.ops.ye3t_runtime.factorized_angular(
        native_input,
        *arguments,
        source_dimension,
        workspace_dimension,
        output_dimension,
    )
    heterogeneous_input = packed.detach().clone().requires_grad_(True)
    heterogeneous_root_nodes = torch.tensor(
        [2, 2],
        dtype=torch.int64,
        device=device,
    )
    root_projection_starts = torch.tensor(
        [0, 1],
        dtype=torch.int64,
        device=device,
    )
    root_projection_dimensions = torch.tensor(
        [1, 1],
        dtype=torch.int64,
        device=device,
    )
    root_output_offsets = torch.tensor(
        [0, 1],
        dtype=torch.int64,
        device=device,
    )
    heterogeneous = (
        torch.ops.ye3t_runtime.factorized_angular_heterogeneous(
            heterogeneous_input,
            node_offsets,
            node_dimensions,
            node_leaf_offsets,
            node_left,
            node_right,
            node_coefficient_offsets,
            coefficient_rows,
            coefficient_columns,
            coefficient_values,
            heterogeneous_root_nodes,
            root_projection_starts,
            root_projection_dimensions,
            root_output_offsets,
            projection_values,
            source_dimension,
            workspace_dimension,
            output_dimension,
        )
    )
    reference = _factorized_angular_packed_reference(
        reference_input,
        *arguments,
        source_dimension,
    )
    tolerance = (
        2.0e-10
        if dtype in (torch.float64, torch.complex128)
        else 5.0e-5
    )
    torch.testing.assert_close(
        native,
        reference,
        rtol=tolerance,
        atol=tolerance,
    )
    torch.testing.assert_close(
        heterogeneous,
        reference,
        rtol=tolerance,
        atol=tolerance,
    )
    native_gradient = torch.autograd.grad(
        (native.conj() * native).real.sum(),
        native_input,
        create_graph=True,
    )[0]
    reference_gradient = torch.autograd.grad(
        (reference.conj() * reference).real.sum(),
        reference_input,
        create_graph=True,
    )[0]
    heterogeneous_gradient = torch.autograd.grad(
        (heterogeneous.conj() * heterogeneous).real.sum(),
        heterogeneous_input,
        create_graph=True,
    )[0]
    torch.testing.assert_close(
        native_gradient,
        reference_gradient,
        rtol=tolerance * 5.0,
        atol=tolerance * 5.0,
    )
    torch.testing.assert_close(
        heterogeneous_gradient,
        reference_gradient,
        rtol=tolerance * 5.0,
        atol=tolerance * 5.0,
    )
    probe = torch.randn_like(native_input)
    native_hvp = torch.autograd.grad(
        (native_gradient.conj() * probe).real.sum(),
        native_input,
    )[0]
    reference_hvp = torch.autograd.grad(
        (reference_gradient.conj() * probe).real.sum(),
        reference_input,
    )[0]
    heterogeneous_hvp = torch.autograd.grad(
        (heterogeneous_gradient.conj() * probe).real.sum(),
        heterogeneous_input,
    )[0]
    torch.testing.assert_close(
        native_hvp,
        reference_hvp,
        rtol=tolerance * 10.0,
        atol=tolerance * 10.0,
    )
    torch.testing.assert_close(
        heterogeneous_hvp,
        reference_hvp,
        rtol=tolerance * 10.0,
        atol=tolerance * 10.0,
    )
    native_linear_input = packed.detach().clone().requires_grad_(True)
    reference_linear_input = packed.detach().clone().requires_grad_(True)
    native_weight = torch.randn(
        output_dimension,
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    reference_weight = (
        native_weight.detach().clone().requires_grad_(True)
    )
    native_bias = torch.randn(
        (),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    reference_bias = native_bias.detach().clone().requires_grad_(True)
    native_linear = torch.ops.ye3t_runtime.factorized_angular_linear(
        native_linear_input,
        *arguments,
        source_dimension,
        workspace_dimension,
        native_weight,
        native_bias,
    )
    reference_features = _factorized_angular_packed_reference(
        reference_linear_input,
        *arguments,
        source_dimension,
    )
    reference_linear = (
        reference_features @ reference_weight + reference_bias
    )
    torch.testing.assert_close(
        native_linear,
        reference_linear,
        rtol=tolerance,
        atol=tolerance,
    )
    native_linear_arguments = (
        native_linear_input,
        native_weight,
        native_bias,
    )
    reference_linear_arguments = (
        reference_linear_input,
        reference_weight,
        reference_bias,
    )
    native_linear_gradients = torch.autograd.grad(
        (native_linear.conj() * native_linear).real.sum(),
        native_linear_arguments,
        create_graph=True,
    )
    reference_linear_gradients = torch.autograd.grad(
        (reference_linear.conj() * reference_linear).real.sum(),
        reference_linear_arguments,
        create_graph=True,
    )
    for actual, expected in zip(
        native_linear_gradients,
        reference_linear_gradients,
    ):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=tolerance * 10.0,
            atol=tolerance * 10.0,
        )
    linear_probes = tuple(
        torch.randn_like(value)
        for value in native_linear_arguments
    )
    native_linear_hvp = torch.autograd.grad(
        native_linear_gradients,
        native_linear_arguments,
        linear_probes,
    )
    reference_linear_hvp = torch.autograd.grad(
        reference_linear_gradients,
        reference_linear_arguments,
        linear_probes,
    )
    for actual, expected in zip(
        native_linear_hvp,
        reference_linear_hvp,
    ):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=tolerance * 20.0,
            atol=tolerance * 20.0,
        )
    if device.type == "cuda" and dtype == torch.float64:
        result = torch.library.opcheck(
            torch.ops.ye3t_runtime.factorized_angular.default,
            (
                native_input,
                *arguments,
                source_dimension,
                workspace_dimension,
                output_dimension,
            ),
            raise_exception=False,
        )
        if not all(value == "SUCCESS" for value in result.values()):
            raise RuntimeError(f"factorized CUDA opcheck failed: {result}")
        linear_result = torch.library.opcheck(
            torch.ops.ye3t_runtime.factorized_angular_linear.default,
            (
                native_linear_input,
                *arguments,
                source_dimension,
                workspace_dimension,
                native_weight,
                native_bias,
            ),
            raise_exception=False,
        )
        if not all(value == "SUCCESS" for value in linear_result.values()):
            raise RuntimeError(
                f"factorized linear CUDA opcheck failed: {linear_result}"
            )


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit(
            "usage: smoke_torch_adapter.py EXTENSION_PATH [cuda]"
        )
    require_cuda = len(sys.argv) == 3 and sys.argv[2] == "cuda"
    extension_path = pathlib.Path(sys.argv[1]).resolve()
    spec = importlib.util.spec_from_file_location(
        "_execution_plan_native",
        extension_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not create an extension module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if int(module.core_abi_version()) < 20:
        raise RuntimeError("unexpected YE3T runtime ABI")
    if not bool(module.has_cpu()) or bool(module.has_cuda()) != require_cuda:
        raise RuntimeError("CMake adapter capability report is inconsistent")
    from ye3t.runtime import execution_plan

    execution_plan._register_torch_contract()

    devices = [torch.device("cpu")]
    if require_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA adapter smoke requires an available GPU")
        devices.append(torch.device("cuda"))
    for device in devices:
        run_radial_smoke(device)
        run_spherical_smoke(device)
        for dtype in (torch.float32, torch.float64):
            run_compiler_symmetric_power_smoke(device, dtype)
        for dtype in (torch.float32, torch.complex64):
            run_compact_pair_smoke(device, dtype)
            run_compact_exterior_power_smoke(device, dtype)
            run_symmetric_power_monomial_smoke(device, dtype)
            run_shared_symmetric_power_monomial_smoke(device, dtype)
        for dtype in (torch.float64, torch.complex128):
            run_plain_source_product_smoke(device, dtype)
            run_density_smoke(device, dtype)
            run_compact_pair_smoke(device, dtype)
            run_compact_exterior_power_smoke(device, dtype)
            run_symmetric_power_monomial_smoke(device, dtype)
            run_shared_symmetric_power_monomial_smoke(device, dtype)
            run_source_analysis_smoke(device, dtype)
            run_factorized_angular_smoke(device, dtype)


if __name__ == "__main__":
    main()
