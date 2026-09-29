"""PyTorch adapter for versioned YE3T execution-plan instructions."""

import os
import shutil
import math
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import torch

from ye3t.backends import triton_joint as _triton_joint
from ye3t.backends.triton_joint import (
    WeightedSparseLinearTable,
    weighted_sparse_linear_forward,
)
from ye3t.execution_plan import (
    YE3TExecutionPlan,
    YE3T_REAL_TESSERAL_CONVENTION,
    YE3TSourceAssemblyPlan,
    YE3TSynthesisTable,
    _factorized_angular_subtree_identity,
    apply_factorized_angular_analysis_reference,
    apply_source_analysis_reference,
)

_NATIVE_CUDA_DENSITY_MIN_WORK_ITEMS = 1 << 20
_NATIVE_CPU_EDGE_OUTER_MIN_WORK_ITEMS = 1 << 16
_NATIVE_CUDA_EDGE_OUTER_MIN_WORK_ITEMS = 1 << 18
_NATIVE_CPU_CARRIER_GATED_SCATTER_MIN_WORK_ITEMS = 1 << 15
_NATIVE_CUDA_CARRIER_GATED_SCATTER_MIN_WORK_ITEMS = 1 << 18
_NATIVE_CUDA_SEGMENTED_CARRIER_SCATTER_MIN_NODE_FEATURE_WORK = 1 << 20
_NATIVE_CPU_CARRIER_CHANNEL_UPDATE_MAX_WORK_ITEMS = 1 << 15
_NATIVE_CUDA_CARRIER_CHANNEL_UPDATE_MIN_WORK_ITEMS = 0
_NATIVE_EXECUTION_PLAN_ABI = 40


def _enabled(name):
    return str(os.environ.get(name, "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class _PackedConcatenateAdjoint(torch.autograd.Function):
    """Split one packed adjoint and concatenate its tangents exactly once."""

    @staticmethod
    def forward(ctx, output_adjoint, widths):
        widths = tuple(int(width) for width in widths)
        ctx.widths = widths
        ctx.set_materialize_grads(False)
        outputs = []
        offset = 0
        for width in widths:
            outputs.append(output_adjoint.narrow(-1, offset, width))
            offset += width
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *part_tangents):
        template = next(
            (value for value in part_tangents if value is not None),
            None,
        )
        if template is None:
            return None, None
        pieces = []
        for width, value in zip(ctx.widths, part_tangents):
            if value is None:
                value = template.new_zeros(
                    tuple(template.shape[:-1]) + (int(width),)
                )
            pieces.append(value)
        return torch.cat(tuple(pieces), dim=-1), None


class _PackedConcatenate(torch.autograd.Function):
    """Concatenate producer blocks with an explicit linear double adjoint."""

    @staticmethod
    def forward(ctx, widths, *parts):
        ctx.widths = tuple(int(width) for width in widths)
        ctx.set_materialize_grads(False)
        return torch.cat(tuple(parts), dim=-1)

    @staticmethod
    def backward(ctx, output_adjoint):
        if output_adjoint is None:
            return (None,) + (None,) * len(ctx.widths)
        parts = _PackedConcatenateAdjoint.apply(
            output_adjoint,
            ctx.widths,
        )
        if len(ctx.widths) == 1:
            parts = (parts,)
        return (None,) + tuple(parts)


def packed_concatenate(parts):
    """Pack tensors along their final axis with an analytic double adjoint.

    The operation is the direct-sum assembly map used by compiler-planned
    producer arenas.  Its first adjoint returns exact disjoint views, while
    its double adjoint performs one concatenation instead of one full-width
    scatter/add per producer.  It changes only storage layout and is
    independent of carrier rank, Young sector, O(3) convention, and dtype.
    """

    parts = tuple(parts)
    if not parts:
        raise ValueError("packed_concatenate requires at least one tensor")
    if len(parts) == 1:
        if not isinstance(parts[0], torch.Tensor):
            raise TypeError("packed_concatenate inputs must be tensors")
        return parts[0]
    first = parts[0]
    if not isinstance(first, torch.Tensor):
        raise TypeError("packed_concatenate inputs must be tensors")
    if first.ndim < 1:
        raise ValueError(
            "packed_concatenate inputs require a packed final axis"
        )
    leading_shape = tuple(first.shape[:-1])
    for part in parts[1:]:
        if not isinstance(part, torch.Tensor):
            raise TypeError("packed_concatenate inputs must be tensors")
        if tuple(part.shape[:-1]) != leading_shape:
            raise ValueError(
                "packed_concatenate inputs must share all leading axes"
            )
        if part.dtype != first.dtype or part.device != first.device:
            raise TypeError(
                "packed_concatenate inputs must share dtype and device"
            )
    widths = tuple(int(part.shape[-1]) for part in parts)
    return _PackedConcatenate.apply(widths, *parts)


@lru_cache(maxsize=1)
def _prebuilt_extension():
    try:
        from ye3t.runtime import _execution_plan_native
    except Exception:
        return None
    return _execution_plan_native


def _source_paths():
    directory = Path(__file__).resolve().parent / "csrc"
    return (
        directory / "execution_plan_torch.cpp",
        directory / "ye3t_runtime_core.cpp",
    )


def _ensure_compiler_env():
    if os.environ.get("CXX"):
        return
    for candidate in (
        "c++",
        "g++",
        "clang++",
        "x86_64-conda-linux-gnu-c++",
    ):
        path = shutil.which(candidate)
        if path:
            os.environ["CXX"] = path
            return


@lru_cache(maxsize=1)
def _load_extension():
    explicit_jit = _enabled("YE3T_ENABLE_EXECUTION_PLAN_JIT")
    extension = _prebuilt_extension()
    if extension is not None:
        installed_abi = int(extension.core_abi_version())
        if installed_abi < _NATIVE_EXECUTION_PLAN_ABI:
            raise RuntimeError(
                "Installed YE3T execution-plan extension ABI "
                f"{installed_abi} is older than required ABI "
                f"{_NATIVE_EXECUTION_PLAN_ABI}. Reinstall ye3t so its "
                "native extension matches the Python package."
            )
        _register_torch_contract()
        return extension
    if not explicit_jit:
        raise RuntimeError(
            "YE3T execution-plan native extension is not installed. Reinstall "
            "ye3t with native extensions enabled, or set "
            "YE3T_ENABLE_EXECUTION_PLAN_JIT=1 to permit a development JIT build."
        )
    _ensure_compiler_env()
    sources = _source_paths()
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing YE3T execution-plan C++ sources: " + ", ".join(missing)
        )
    from torch.utils.cpp_extension import load

    extension = load(
        name="ye3t_execution_plan_native_jit_v40",
        sources=[str(path) for path in sources],
        extra_include_paths=[str(sources[0].parent)],
        extra_cflags=["-O3", "-std=c++20"],
        with_cuda=False,
        verbose=_enabled("YE3T_CPP_JIT_VERBOSE"),
    )
    _register_torch_contract()
    return extension


def _extension_for_native_dispatch():
    if torch.compiler.is_compiling():
        return None
    return _load_extension()


def _native_cuda_dispatch_available(extension):
    return extension is None or bool(extension.has_cuda())


def _resolve_symmetric_power_double_backward_policy(
    input,
    operation_name="symmetric_power_monomial_double_backward",
):
    policy = os.environ.get(
        "YE3T_SYMMETRIC_POWER_MONOMIAL_DOUBLE_BACKWARD_POLICY",
        "auto",
    )
    if policy not in {"auto", "native", "reference"}:
        raise RuntimeError(
            "YE3T_SYMMETRIC_POWER_MONOMIAL_DOUBLE_BACKWARD_POLICY must "
            "be auto, native, or reference"
        )
    native_available = bool(
        input.is_cuda
        and hasattr(torch.ops.ye3t_runtime, operation_name)
    )
    if policy == "native" and not native_available:
        raise RuntimeError(
            "native symmetric-power double backward operation "
            f"{operation_name!r} was requested but is unavailable"
        )
    return policy, bool(native_available and policy != "reference")


def _resolve_shared_power_double_backward_algorithm(input):
    requested = os.environ.get(
        "YE3T_SYMMETRIC_POWER_SHARED_DOUBLE_BACKWARD_ALGORITHM",
        "auto",
    )
    if requested not in {"auto", "coefficient", "factored"}:
        raise RuntimeError(
            "YE3T_SYMMETRIC_POWER_SHARED_DOUBLE_BACKWARD_ALGORITHM must "
            "be auto, coefficient, or factored"
        )
    factored_available = bool(
        input.is_cuda
        and hasattr(
            torch.ops.ye3t_runtime,
            "symmetric_power_shared_monomial_factored_double_backward",
        )
    )
    if requested == "factored" and not factored_available:
        raise RuntimeError(
            "factored shared symmetric-power double backward was "
            "requested but is unavailable"
        )
    if requested == "auto":
        return "factored" if factored_available else "coefficient"
    return requested


def _symmetric_power_shared_double_backward(
    input_adjoint_tangent,
    output_adjoint,
    input,
    monomial_counts,
    output_offsets,
    coefficient_terms,
    coefficient_outputs,
    coefficient_values,
):
    algorithm = _resolve_shared_power_double_backward_algorithm(input)
    operation_name = (
        "symmetric_power_shared_monomial_factored_double_backward"
        if algorithm == "factored"
        else "symmetric_power_shared_monomial_double_backward"
    )
    operation = getattr(
        torch.ops.ye3t_runtime,
        operation_name,
    )
    return (
        operation(
            input_adjoint_tangent.contiguous(),
            output_adjoint.contiguous(),
            input.contiguous(),
            monomial_counts,
            output_offsets,
            coefficient_terms,
            coefficient_outputs,
            coefficient_values,
        ),
        algorithm,
    )


@lru_cache(maxsize=1)
def _register_torch_contract():
    import torch

    registration_key = "_ye3t_runtime_contract_registered_abi39"
    if bool(getattr(torch.library, registration_key, False)):
        return True

    def independent_autograd_input(value):
        return value.detach().requires_grad_(True)

    def radial_fake(radii, cutoffs, lambdas, radial_index):
        del cutoffs, lambdas, radial_index
        return radii.new_empty(radii.shape), radii.new_empty(radii.shape)

    def radial_table_fake(
        radii,
        cutoffs,
        lambdas,
        maximum_radial_index,
    ):
        del cutoffs, lambdas
        shape = (radii.shape[0], int(maximum_radial_index) + 1)
        return radii.new_empty(shape), radii.new_empty(shape)

    def radial_table_double_backward_fake(
        values_adjoint,
        radial_derivatives,
        radii,
        cutoffs,
        lambdas,
        grad_grad_radii,
        maximum_radial_index,
    ):
        del (
            radial_derivatives,
            cutoffs,
            lambdas,
            grad_grad_radii,
            maximum_radial_index,
        )
        return (
            values_adjoint.new_empty(values_adjoint.shape),
            radii.new_empty(radii.shape),
        )

    def spherical_fake(
        edge_vectors,
        angular_momentum,
        real_output,
        epsilon,
    ):
        del epsilon
        width = 2 * int(angular_momentum) + 1
        if real_output:
            values = edge_vectors.new_empty((edge_vectors.shape[0], width))
        else:
            dtype = (
                torch.complex64
                if edge_vectors.dtype == torch.float32
                else torch.complex128
            )
            values = edge_vectors.new_empty(
                (edge_vectors.shape[0], width),
                dtype=dtype,
            )
        return values, values.new_empty(
            (edge_vectors.shape[0], width, 3)
        )

    def spherical_table_fake(
        edge_vectors,
        maximum_angular_momentum,
        real_output,
        epsilon,
    ):
        del epsilon
        width = (int(maximum_angular_momentum) + 1) ** 2
        if real_output:
            values = edge_vectors.new_empty((edge_vectors.shape[0], width))
        else:
            dtype = (
                torch.complex64
                if edge_vectors.dtype == torch.float32
                else torch.complex128
            )
            values = edge_vectors.new_empty(
                (edge_vectors.shape[0], width),
                dtype=dtype,
            )
        return values, values.new_empty(
            (edge_vectors.shape[0], width, 3)
        )

    def spherical_table_double_backward_fake(
        edge_vectors,
        value_adjoint,
        edge_direction,
        maximum_angular_momentum,
        epsilon,
    ):
        del edge_vectors, maximum_angular_momentum, epsilon
        return (
            value_adjoint.new_empty(value_adjoint.shape),
            edge_direction.new_empty(edge_direction.shape),
        )

    torch.library.register_fake(
        "ye3t_runtime::cheb_exp_cos_radial_with_derivative",
        radial_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::cheb_exp_cos_radial_table_with_derivative",
        radial_table_fake,
    )
    if hasattr(
        torch.ops.ye3t_runtime,
        "cheb_exp_cos_radial_table_double_backward",
    ):
        torch.library.register_fake(
            "ye3t_runtime::cheb_exp_cos_radial_table_double_backward",
            radial_table_double_backward_fake,
        )
    torch.library.register_fake(
        "ye3t_runtime::spherical_harmonics_with_derivative",
        spherical_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::spherical_harmonics_table_with_derivative",
        spherical_table_fake,
    )
    if hasattr(
        torch.ops.ye3t_runtime,
        "spherical_harmonics_table_double_backward",
    ):
        torch.library.register_fake(
            "ye3t_runtime::spherical_harmonics_table_double_backward",
            spherical_table_double_backward_fake,
        )

    def plain_site_basis_product_fake(
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
    ):
        del (
            radial_derivatives,
            angular_values,
            angular_derivatives,
            prefactors,
            prefactor_derivatives_center,
            prefactor_derivatives_neighbor,
            radial_directions,
            term_groups,
            term_channels,
        )
        values = radial_values.new_empty(
            (radial_values.shape[1], int(channel_count))
        )
        return (
            values,
            values.new_empty(
                (
                    radial_values.shape[1],
                    int(channel_count),
                    3,
                )
            ),
            values.new_empty(values.shape),
            values.new_empty(values.shape),
        )

    torch.library.register_fake(
        "ye3t_runtime::plain_site_basis_product_with_derivative",
        plain_site_basis_product_fake,
    )

    def scheduled_radial_angular_channels_fake(
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
    ):
        del (
            radial_derivatives,
            angular_values,
            angular_derivatives,
            radial_directions,
            edge_types,
            channel_radial_indices,
            channel_angular_indices,
            channel_types,
        )
        values = radial_values.new_empty(
            (radial_values.shape[0], channel_scales.shape[0])
        )
        return values, values.new_empty(
            (
                radial_values.shape[0],
                channel_scales.shape[0],
                3,
            )
        )

    torch.library.register_fake(
        "ye3t_runtime::scheduled_radial_angular_channels_with_derivative",
        scheduled_radial_angular_channels_fake,
    )

    def plain_site_basis_product_adjoint_fake(
        radial_values,
        radial_derivatives,
        angular_values,
        angular_derivatives,
        prefactors,
        prefactor_derivatives_center,
        prefactor_derivatives_neighbor,
        radial_directions,
        edge_weights,
        edge_weight_derivatives,
        term_groups,
        term_channels,
        edge_adjoint,
    ):
        del (
            radial_derivatives,
            angular_values,
            angular_derivatives,
            prefactors,
            prefactor_derivatives_center,
            prefactor_derivatives_neighbor,
            radial_directions,
            edge_weights,
            edge_weight_derivatives,
            term_groups,
            term_channels,
            edge_adjoint,
        )
        edge_count = radial_values.shape[1]
        charge = radial_values.new_empty((edge_count,))
        return (
            radial_values.new_empty((edge_count, 3)),
            charge,
            charge.new_empty(charge.shape),
        )

    torch.library.register_fake(
        "ye3t_runtime::plain_site_basis_product_adjoint",
        plain_site_basis_product_adjoint_fake,
    )

    def density_fake(edge_values, centers, atom_count):
        del centers
        return edge_values.new_empty(
            (int(atom_count), edge_values.shape[1])
        )

    def density_adjoint_fake(atomic_adjoint, centers):
        return atomic_adjoint.new_empty(
            (centers.shape[0], atomic_adjoint.shape[1])
        )

    torch.library.register_fake(
        "ye3t_runtime::density_accumulate",
        density_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::density_accumulate_adjoint",
        density_adjoint_fake,
    )

    def edge_outer_fake(left, right, centers, atom_count):
        del centers
        return left.new_empty(
            (int(atom_count), left.shape[1], right.shape[1])
        )

    def edge_outer_adjoint_fake(
        atomic_adjoint,
        left,
        right,
        centers,
    ):
        del atomic_adjoint, centers
        return left.new_empty(left.shape), right.new_empty(right.shape)

    def edge_outer_double_backward_fake(
        atomic_adjoint,
        left,
        right,
        left_adjoint_tangent,
        right_adjoint_tangent,
        centers,
    ):
        del left_adjoint_tangent, right_adjoint_tangent, centers
        return (
            atomic_adjoint.new_empty(atomic_adjoint.shape),
            left.new_empty(left.shape),
            right.new_empty(right.shape),
        )

    torch.library.register_fake(
        "ye3t_runtime::edge_outer_accumulate",
        edge_outer_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::edge_outer_accumulate_adjoint",
        edge_outer_adjoint_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::edge_outer_accumulate_double_backward",
        edge_outer_double_backward_fake,
    )

    def softmax_gaussian_role_density_fake(
        distances,
        cutoffs,
        filter_centers,
        filter_width,
        edge_values,
        atom_centers,
        atom_count,
    ):
        del distances, cutoffs, filter_width, atom_centers
        return edge_values.new_empty(
            (int(atom_count), filter_centers.shape[0], edge_values.shape[1])
        )

    def softmax_gaussian_role_density_adjoint_fake(
        atomic_adjoint,
        distances,
        cutoffs,
        filter_centers,
        filter_width,
        edge_values,
        atom_centers,
    ):
        del atomic_adjoint, cutoffs, filter_centers, filter_width, atom_centers
        return distances.new_empty(distances.shape), edge_values.new_empty(
            edge_values.shape
        )

    def softmax_gaussian_role_density_double_backward_fake(
        atomic_adjoint,
        distances,
        cutoffs,
        filter_centers,
        filter_width,
        edge_values,
        distance_adjoint_tangent,
        edge_adjoint_tangent,
        atom_centers,
    ):
        del (
            cutoffs,
            filter_centers,
            filter_width,
            distance_adjoint_tangent,
            edge_adjoint_tangent,
            atom_centers,
        )
        return (
            atomic_adjoint.new_empty(atomic_adjoint.shape),
            distances.new_empty(distances.shape),
            edge_values.new_empty(edge_values.shape),
        )

    torch.library.register_fake(
        "ye3t_runtime::softmax_gaussian_role_density",
        softmax_gaussian_role_density_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::softmax_gaussian_role_density_adjoint",
        softmax_gaussian_role_density_adjoint_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::softmax_gaussian_role_density_double_backward",
        softmax_gaussian_role_density_double_backward_fake,
    )

    def scheduled_role_density_fake(
        radial_values,
        angular_values,
        distances,
        cutoffs,
        filter_centers,
        filter_width,
        soft_weights,
        edge_types,
        channel_radial_indices,
        channel_angular_indices,
        channel_types,
        channel_scales,
        atom_centers,
        atom_count,
    ):
        del (
            angular_values,
            distances,
            cutoffs,
            filter_width,
            soft_weights,
            edge_types,
            channel_radial_indices,
            channel_angular_indices,
            channel_types,
            atom_centers,
        )
        return radial_values.new_empty(
            (
                int(atom_count),
                filter_centers.shape[0],
                channel_scales.shape[0] + 1,
            )
        )

    def scheduled_role_density_adjoint_fake(
        atomic_adjoint,
        radial_values,
        angular_values,
        distances,
        cutoffs,
        filter_centers,
        filter_width,
        soft_weights,
        edge_types,
        channel_radial_indices,
        channel_angular_indices,
        channel_types,
        channel_scales,
        atom_centers,
    ):
        del (
            atomic_adjoint,
            cutoffs,
            filter_centers,
            filter_width,
            edge_types,
            channel_radial_indices,
            channel_angular_indices,
            channel_types,
            channel_scales,
            atom_centers,
        )
        return (
            radial_values.new_empty(radial_values.shape),
            angular_values.new_empty(angular_values.shape),
            distances.new_empty(distances.shape),
            soft_weights.new_empty(soft_weights.shape),
        )

    def scheduled_role_density_double_backward_fake(
        atomic_adjoint,
        radial_values,
        angular_values,
        distances,
        cutoffs,
        filter_centers,
        filter_width,
        soft_weights,
        radial_adjoint_tangent,
        angular_adjoint_tangent,
        distance_adjoint_tangent,
        soft_adjoint_tangent,
        edge_types,
        channel_radial_indices,
        channel_angular_indices,
        channel_types,
        channel_scales,
        atom_centers,
    ):
        del (
            cutoffs,
            filter_centers,
            filter_width,
            radial_adjoint_tangent,
            angular_adjoint_tangent,
            distance_adjoint_tangent,
            soft_adjoint_tangent,
            edge_types,
            channel_radial_indices,
            channel_angular_indices,
            channel_types,
            channel_scales,
            atom_centers,
        )
        return (
            atomic_adjoint.new_empty(atomic_adjoint.shape),
            radial_values.new_empty(radial_values.shape),
            angular_values.new_empty(angular_values.shape),
            distances.new_empty(distances.shape),
            soft_weights.new_empty(soft_weights.shape),
        )

    torch.library.register_fake(
        "ye3t_runtime::scheduled_softmax_gaussian_role_density",
        scheduled_role_density_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::scheduled_softmax_gaussian_role_density_adjoint",
        scheduled_role_density_adjoint_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::scheduled_softmax_gaussian_role_density_double_backward",
        scheduled_role_density_double_backward_fake,
    )

    def carrier_gated_scatter_fake(
        node_values,
        edge_gates,
        edge_sources,
        edge_targets,
        feature_channels,
        target_count,
    ):
        del edge_gates, edge_sources, edge_targets, feature_channels
        return node_values.new_empty(
            (int(target_count), node_values.shape[1])
        )

    def carrier_gated_scatter_adjoint_fake(
        target_adjoint,
        node_values,
        edge_gates,
        edge_sources,
        edge_targets,
        feature_channels,
    ):
        del target_adjoint, edge_sources, edge_targets, feature_channels
        return (
            node_values.new_empty(node_values.shape),
            edge_gates.new_empty(edge_gates.shape),
        )

    def carrier_gated_scatter_double_backward_fake(
        target_adjoint,
        node_values,
        edge_gates,
        node_adjoint_tangent,
        gate_adjoint_tangent,
        edge_sources,
        edge_targets,
        feature_channels,
    ):
        del (
            node_adjoint_tangent,
            gate_adjoint_tangent,
            edge_sources,
            edge_targets,
            feature_channels,
        )
        return (
            target_adjoint.new_empty(target_adjoint.shape),
            node_values.new_empty(node_values.shape),
            edge_gates.new_empty(edge_gates.shape),
        )

    torch.library.register_fake(
        "ye3t_runtime::carrier_gated_scatter",
        carrier_gated_scatter_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::carrier_gated_scatter_adjoint",
        carrier_gated_scatter_adjoint_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::carrier_gated_scatter_double_backward",
        carrier_gated_scatter_double_backward_fake,
    )

    def carrier_residual_gated_scatter_fake(
        node_values,
        edge_gates,
        edge_sources,
        edge_targets,
        feature_channels,
    ):
        del edge_gates, edge_sources, edge_targets, feature_channels
        return node_values.new_empty(node_values.shape)

    torch.library.register_fake(
        "ye3t_runtime::carrier_residual_gated_scatter",
        carrier_residual_gated_scatter_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::carrier_residual_gated_scatter_adjoint",
        carrier_gated_scatter_adjoint_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::carrier_residual_gated_scatter_double_backward",
        carrier_gated_scatter_double_backward_fake,
    )

    def carrier_segmented_residual_gated_scatter_fake(
        node_values,
        edge_gates,
        edge_sources,
        edge_targets,
        target_offsets,
        source_offsets,
        source_edges,
        feature_channels,
        channel_offsets,
        channel_features,
    ):
        del (
            edge_gates,
            edge_sources,
            edge_targets,
            target_offsets,
            source_offsets,
            source_edges,
            feature_channels,
            channel_offsets,
            channel_features,
        )
        return node_values.new_empty(node_values.shape)

    def carrier_segmented_residual_gated_scatter_adjoint_fake(
        target_adjoint,
        node_values,
        edge_gates,
        edge_sources,
        edge_targets,
        target_offsets,
        source_offsets,
        source_edges,
        feature_channels,
        channel_offsets,
        channel_features,
    ):
        del (
            target_adjoint,
            edge_sources,
            edge_targets,
            target_offsets,
            source_offsets,
            source_edges,
            feature_channels,
            channel_offsets,
            channel_features,
        )
        return (
            node_values.new_empty(node_values.shape),
            edge_gates.new_empty(edge_gates.shape),
        )

    def carrier_segmented_residual_gated_scatter_double_backward_fake(
        target_adjoint,
        node_values,
        edge_gates,
        node_adjoint_tangent,
        gate_adjoint_tangent,
        edge_sources,
        edge_targets,
        target_offsets,
        source_offsets,
        source_edges,
        feature_channels,
        channel_offsets,
        channel_features,
    ):
        del (
            node_adjoint_tangent,
            gate_adjoint_tangent,
            edge_sources,
            edge_targets,
            target_offsets,
            source_offsets,
            source_edges,
            feature_channels,
            channel_offsets,
            channel_features,
        )
        return (
            target_adjoint.new_empty(target_adjoint.shape),
            node_values.new_empty(node_values.shape),
            edge_gates.new_empty(edge_gates.shape),
        )

    torch.library.register_fake(
        "ye3t_runtime::carrier_segmented_residual_gated_scatter",
        carrier_segmented_residual_gated_scatter_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::carrier_segmented_residual_gated_scatter_adjoint",
        carrier_segmented_residual_gated_scatter_adjoint_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::carrier_segmented_residual_gated_scatter_double_backward",
        carrier_segmented_residual_gated_scatter_double_backward_fake,
    )

    def source_arena_gather_fake(
        producer,
        gather_indices,
        reverse_offsets,
        reverse_output_indices,
        center_types,
        atom_types,
    ):
        del reverse_offsets, reverse_output_indices, center_types, atom_types
        return producer.new_empty(
            (producer.shape[0], gather_indices.numel())
        )

    def source_arena_gather_adjoint_fake(
        output_adjoint,
        gather_indices,
        reverse_offsets,
        reverse_output_indices,
        center_types,
        atom_types,
    ):
        del gather_indices, reverse_output_indices, center_types, atom_types
        return output_adjoint.new_empty(
            (output_adjoint.shape[0], reverse_offsets.numel() - 1)
        )

    def source_arena_gather_double_backward_fake(
        producer_adjoint_tangent,
        gather_indices,
        reverse_offsets,
        reverse_output_indices,
        center_types,
        atom_types,
    ):
        del reverse_offsets, reverse_output_indices, center_types, atom_types
        return producer_adjoint_tangent.new_empty(
            (producer_adjoint_tangent.shape[0], gather_indices.numel())
        )

    torch.library.register_fake(
        "ye3t_runtime::source_arena_gather",
        source_arena_gather_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::source_arena_gather_adjoint",
        source_arena_gather_adjoint_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::source_arena_gather_double_backward",
        source_arena_gather_double_backward_fake,
    )

    def source_arena_channel_transform_fake(
        producer,
        channel_maps,
        gather_indices,
        reverse_offsets,
        reverse_output_indices,
        center_types,
        atom_types,
        input_feature_offsets,
        output_feature_offsets,
        input_channel_offsets,
        output_channel_offsets,
        map_offsets,
        output_width,
    ):
        del (
            channel_maps,
            gather_indices,
            reverse_offsets,
            reverse_output_indices,
            center_types,
            atom_types,
            input_feature_offsets,
            output_feature_offsets,
            input_channel_offsets,
            output_channel_offsets,
            map_offsets,
        )
        return producer.new_empty((producer.shape[0], int(output_width)))

    def source_arena_channel_transform_adjoint_fake(
        output_adjoint,
        producer,
        channel_maps,
        gather_indices,
        reverse_offsets,
        reverse_output_indices,
        center_types,
        atom_types,
        input_feature_offsets,
        output_feature_offsets,
        input_channel_offsets,
        output_channel_offsets,
        map_offsets,
    ):
        del (
            output_adjoint,
            gather_indices,
            reverse_offsets,
            reverse_output_indices,
            center_types,
            atom_types,
            input_feature_offsets,
            output_feature_offsets,
            input_channel_offsets,
            output_channel_offsets,
            map_offsets,
        )
        return (
            producer.new_empty(producer.shape),
            channel_maps.new_empty(channel_maps.shape),
        )

    def source_arena_channel_transform_double_backward_fake(
        output_adjoint,
        producer,
        channel_maps,
        producer_adjoint_tangent,
        channel_maps_adjoint_tangent,
        gather_indices,
        reverse_offsets,
        reverse_output_indices,
        center_types,
        atom_types,
        input_feature_offsets,
        output_feature_offsets,
        input_channel_offsets,
        output_channel_offsets,
        map_offsets,
    ):
        del (
            producer_adjoint_tangent,
            channel_maps_adjoint_tangent,
            gather_indices,
            reverse_offsets,
            reverse_output_indices,
            center_types,
            atom_types,
            input_feature_offsets,
            output_feature_offsets,
            input_channel_offsets,
            output_channel_offsets,
            map_offsets,
        )
        return (
            output_adjoint.new_empty(output_adjoint.shape),
            producer.new_empty(producer.shape),
            channel_maps.new_empty(channel_maps.shape),
        )

    torch.library.register_fake(
        "ye3t_runtime::source_arena_channel_transform",
        source_arena_channel_transform_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::source_arena_channel_transform_adjoint",
        source_arena_channel_transform_adjoint_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::source_arena_channel_transform_double_backward",
        source_arena_channel_transform_double_backward_fake,
    )

    def carrier_channel_update_fake(
        values,
        gates,
        channel_maps,
        feature_offsets,
        channel_offsets,
        map_offsets,
    ):
        del (
            gates,
            channel_maps,
            feature_offsets,
            channel_offsets,
            map_offsets,
        )
        return values.new_empty(values.shape)

    def carrier_channel_update_adjoint_fake(
        output_adjoint,
        values,
        gates,
        channel_maps,
        feature_offsets,
        channel_offsets,
        map_offsets,
    ):
        del (
            output_adjoint,
            feature_offsets,
            channel_offsets,
            map_offsets,
        )
        return (
            values.new_empty(values.shape),
            gates.new_empty(gates.shape),
            channel_maps.new_empty(channel_maps.shape),
        )

    def carrier_channel_update_double_backward_fake(
        output_adjoint,
        values,
        gates,
        channel_maps,
        values_adjoint_tangent,
        gates_adjoint_tangent,
        channel_maps_adjoint_tangent,
        feature_offsets,
        channel_offsets,
        map_offsets,
    ):
        del (
            values_adjoint_tangent,
            gates_adjoint_tangent,
            channel_maps_adjoint_tangent,
            feature_offsets,
            channel_offsets,
            map_offsets,
        )
        return (
            output_adjoint.new_empty(output_adjoint.shape),
            values.new_empty(values.shape),
            gates.new_empty(gates.shape),
            channel_maps.new_empty(channel_maps.shape),
        )

    torch.library.register_fake(
        "ye3t_runtime::carrier_channel_update",
        carrier_channel_update_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::carrier_channel_update_adjoint",
        carrier_channel_update_adjoint_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::carrier_channel_update_double_backward",
        carrier_channel_update_double_backward_fake,
    )

    def carrier_channel_transform_fake(
        values,
        channel_maps,
        input_feature_offsets,
        output_feature_offsets,
        input_channel_offsets,
        output_channel_offsets,
        map_offsets,
        output_width,
    ):
        del (
            channel_maps,
            input_feature_offsets,
            input_channel_offsets,
            output_channel_offsets,
            map_offsets,
        )
        del output_feature_offsets
        return values.new_empty((values.shape[0], int(output_width)))

    def carrier_channel_transform_adjoint_fake(
        output_adjoint,
        values,
        channel_maps,
        input_feature_offsets,
        output_feature_offsets,
        input_channel_offsets,
        output_channel_offsets,
        map_offsets,
    ):
        del (
            output_adjoint,
            input_feature_offsets,
            output_feature_offsets,
            input_channel_offsets,
            output_channel_offsets,
            map_offsets,
        )
        return (
            values.new_empty(values.shape),
            channel_maps.new_empty(channel_maps.shape),
        )

    def carrier_channel_transform_double_backward_fake(
        output_adjoint,
        values,
        channel_maps,
        values_adjoint_tangent,
        channel_maps_adjoint_tangent,
        input_feature_offsets,
        output_feature_offsets,
        input_channel_offsets,
        output_channel_offsets,
        map_offsets,
    ):
        del (
            values_adjoint_tangent,
            channel_maps_adjoint_tangent,
            input_feature_offsets,
            output_feature_offsets,
            input_channel_offsets,
            output_channel_offsets,
            map_offsets,
        )
        return (
            output_adjoint.new_empty(output_adjoint.shape),
            values.new_empty(values.shape),
            channel_maps.new_empty(channel_maps.shape),
        )

    torch.library.register_fake(
        "ye3t_runtime::carrier_channel_transform",
        carrier_channel_transform_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::carrier_channel_transform_adjoint",
        carrier_channel_transform_adjoint_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::carrier_channel_transform_double_backward",
        carrier_channel_transform_double_backward_fake,
    )

    def carrier_role_channel_map_adjoint_fake(
        edge_values,
        role_weights,
        atomic_output_adjoint,
        atom_centers,
        channel_maps,
        input_feature_offsets,
        output_feature_offsets,
        input_channel_offsets,
        output_channel_offsets,
        map_offsets,
    ):
        del (
            edge_values,
            role_weights,
            atomic_output_adjoint,
            atom_centers,
            input_feature_offsets,
            output_feature_offsets,
            input_channel_offsets,
            output_channel_offsets,
            map_offsets,
        )
        return channel_maps.new_empty(channel_maps.shape)

    def carrier_role_channel_map_adjoint_double_backward_fake(
        edge_values,
        role_weights,
        atomic_output_adjoint,
        atom_centers,
        channel_maps,
        channel_maps_adjoint_tangent,
        input_feature_offsets,
        output_feature_offsets,
        input_channel_offsets,
        output_channel_offsets,
        map_offsets,
    ):
        del (
            atom_centers,
            channel_maps,
            channel_maps_adjoint_tangent,
            input_feature_offsets,
            output_feature_offsets,
            input_channel_offsets,
            output_channel_offsets,
            map_offsets,
        )
        return (
            edge_values.new_empty(edge_values.shape),
            role_weights.new_empty(role_weights.shape),
            atomic_output_adjoint.new_empty(atomic_output_adjoint.shape),
        )

    torch.library.register_fake(
        "ye3t_runtime::carrier_role_channel_map_adjoint",
        carrier_role_channel_map_adjoint_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::carrier_role_channel_map_adjoint_double_backward",
        carrier_role_channel_map_adjoint_double_backward_fake,
    )

    def source_analysis_fake(
        source,
        assembly_rows,
        assembly_columns,
        assembly_values,
        synthesis_rows,
        synthesis_columns,
        synthesis_values,
        induced_dimension,
        output_dimension,
    ):
        return source.new_empty((source.shape[0], int(output_dimension)))

    def source_analysis_adjoint_fake(
        output_adjoint,
        assembly_rows,
        assembly_columns,
        assembly_values,
        synthesis_rows,
        synthesis_columns,
        synthesis_values,
        source_dimension,
        induced_dimension,
    ):
        return output_adjoint.new_empty(
            (output_adjoint.shape[0], int(source_dimension))
        )

    torch.library.register_fake(
        "ye3t_runtime::source_analysis",
        source_analysis_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::source_analysis_adjoint",
        source_analysis_adjoint_fake,
    )

    def source_linear_fake(
        source,
        assembly_rows,
        assembly_columns,
        assembly_values,
        synthesis_rows,
        synthesis_columns,
        synthesis_values,
        induced_dimension,
        weight,
        bias,
    ):
        del (
            assembly_rows,
            assembly_columns,
            assembly_values,
            synthesis_rows,
            synthesis_columns,
            synthesis_values,
            induced_dimension,
            weight,
            bias,
        )
        return source.new_empty((source.shape[0],))

    def source_linear_adjoint_fake(
        output_adjoint,
        source,
        assembly_rows,
        assembly_columns,
        assembly_values,
        synthesis_rows,
        synthesis_columns,
        synthesis_values,
        induced_dimension,
        weight,
    ):
        del (
            output_adjoint,
            assembly_rows,
            assembly_columns,
            assembly_values,
            synthesis_rows,
            synthesis_columns,
            synthesis_values,
            induced_dimension,
        )
        return (
            source.new_empty(source.shape),
            weight.new_empty(weight.shape),
            weight.new_empty(()),
        )

    torch.library.register_fake(
        "ye3t_runtime::source_analysis_linear",
        source_linear_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::source_analysis_linear_adjoint",
        source_linear_adjoint_fake,
    )

    def compact_pair_product_fake(left, right, antisymmetric):
        dimension = left.shape[1]
        if antisymmetric:
            output_dimension = dimension * (dimension - 1) // 2
        else:
            output_dimension = dimension * (dimension + 1) // 2
        return left.new_empty((left.shape[0], output_dimension))

    def compact_pair_product_adjoint_fake(
        output_adjoint,
        left,
        right,
        antisymmetric,
    ):
        return left.new_empty(left.shape), right.new_empty(right.shape)

    torch.library.register_fake(
        "ye3t_runtime::compact_pair_product",
        compact_pair_product_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::compact_pair_product_adjoint",
        compact_pair_product_adjoint_fake,
    )

    def compact_exterior_power_fake(factors):
        order = int(factors.shape[1])
        dimension = factors.shape[2]
        output_dimension = 1
        for index in range(order):
            output_dimension = (
                output_dimension * (dimension - index) // (index + 1)
            )
        return factors.new_empty((factors.shape[0], output_dimension))

    def compact_exterior_power_adjoint_fake(
        output_adjoint,
        factors,
    ):
        del output_adjoint
        return factors.new_empty(factors.shape)

    torch.library.register_fake(
        "ye3t_runtime::compact_exterior_power",
        compact_exterior_power_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::compact_exterior_power_adjoint",
        compact_exterior_power_adjoint_fake,
    )

    def symmetric_power_monomial_fake(
        input,
        monomial_counts,
        output_offsets,
        output_indices,
        monomial_values,
    ):
        del monomial_counts, output_indices, monomial_values
        return input.new_empty(
            (input.shape[0], output_offsets.shape[0] - 1)
        )

    def symmetric_power_monomial_adjoint_fake(
        output_adjoint,
        input,
        monomial_counts,
        output_offsets,
        output_indices,
        monomial_values,
    ):
        del (
            output_adjoint,
            monomial_counts,
            output_offsets,
            output_indices,
            monomial_values,
        )
        return input.new_empty(input.shape)

    def symmetric_power_monomial_double_backward_fake(
        input_adjoint_tangent,
        output_adjoint,
        input,
        monomial_counts,
        output_offsets,
        output_indices,
        monomial_values,
    ):
        del (
            input_adjoint_tangent,
            monomial_counts,
            output_offsets,
            output_indices,
            monomial_values,
        )
        return (
            output_adjoint.new_empty(output_adjoint.shape),
            input.new_empty(input.shape),
        )

    torch.library.register_fake(
        "ye3t_runtime::symmetric_power_monomial",
        symmetric_power_monomial_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::symmetric_power_monomial_adjoint",
        symmetric_power_monomial_adjoint_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::symmetric_power_monomial_double_backward",
        symmetric_power_monomial_double_backward_fake,
    )

    def symmetric_power_shared_monomial_fake(
        input,
        monomial_counts,
        output_offsets,
        coefficient_terms,
        coefficient_outputs,
        coefficient_values,
    ):
        del (
            monomial_counts,
            coefficient_terms,
            coefficient_outputs,
            coefficient_values,
        )
        return input.new_empty(
            (input.shape[0], output_offsets.shape[0] - 1)
        )

    def symmetric_power_shared_monomial_adjoint_fake(
        output_adjoint,
        input,
        monomial_counts,
        output_offsets,
        coefficient_terms,
        coefficient_outputs,
        coefficient_values,
    ):
        del (
            output_adjoint,
            monomial_counts,
            output_offsets,
            coefficient_terms,
            coefficient_outputs,
            coefficient_values,
        )
        return input.new_empty(input.shape)

    def symmetric_power_shared_monomial_double_backward_fake(
        input_adjoint_tangent,
        output_adjoint,
        input,
        monomial_counts,
        output_offsets,
        coefficient_terms,
        coefficient_outputs,
        coefficient_values,
    ):
        del (
            input_adjoint_tangent,
            monomial_counts,
            output_offsets,
            coefficient_terms,
            coefficient_outputs,
            coefficient_values,
        )
        return (
            output_adjoint.new_empty(output_adjoint.shape),
            input.new_empty(input.shape),
        )

    torch.library.register_fake(
        "ye3t_runtime::symmetric_power_shared_monomial",
        symmetric_power_shared_monomial_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::symmetric_power_shared_monomial_adjoint",
        symmetric_power_shared_monomial_adjoint_fake,
    )
    if hasattr(
        torch.ops.ye3t_runtime,
        "symmetric_power_shared_monomial_double_backward",
    ):
        torch.library.register_fake(
            "ye3t_runtime::symmetric_power_shared_monomial_double_backward",
            symmetric_power_shared_monomial_double_backward_fake,
        )
    if hasattr(
        torch.ops.ye3t_runtime,
        "symmetric_power_shared_monomial_factored_double_backward",
    ):
        torch.library.register_fake(
            "ye3t_runtime::"
            "symmetric_power_shared_monomial_factored_double_backward",
            symmetric_power_shared_monomial_double_backward_fake,
        )

    def symmetric_power_shared_sparse_monomial_fake(
        input,
        term_offsets,
        term_components,
        term_exponents,
        output_offsets,
        coefficient_terms,
        coefficient_outputs,
        coefficient_values,
    ):
        del (
            term_offsets,
            term_components,
            term_exponents,
            coefficient_terms,
            coefficient_outputs,
            coefficient_values,
        )
        return input.new_empty(
            (input.shape[0], output_offsets.shape[0] - 1)
        )

    def symmetric_power_shared_sparse_monomial_adjoint_fake(
        output_adjoint,
        input,
        term_offsets,
        term_components,
        term_exponents,
        output_offsets,
        coefficient_terms,
        coefficient_outputs,
        coefficient_values,
    ):
        del (
            output_adjoint,
            term_offsets,
            term_components,
            term_exponents,
            output_offsets,
            coefficient_terms,
            coefficient_outputs,
            coefficient_values,
        )
        return input.new_empty(input.shape)

    def symmetric_power_shared_sparse_monomial_double_backward_fake(
        input_adjoint_tangent,
        output_adjoint,
        input,
        term_offsets,
        term_components,
        term_exponents,
        output_offsets,
        coefficient_terms,
        coefficient_outputs,
        coefficient_values,
    ):
        del (
            input_adjoint_tangent,
            term_offsets,
            term_components,
            term_exponents,
            output_offsets,
            coefficient_terms,
            coefficient_outputs,
            coefficient_values,
        )
        return (
            output_adjoint.new_empty(output_adjoint.shape),
            input.new_empty(input.shape),
        )

    if hasattr(
        torch.ops.ye3t_runtime,
        "symmetric_power_shared_sparse_monomial",
    ):
        torch.library.register_fake(
            "ye3t_runtime::symmetric_power_shared_sparse_monomial",
            symmetric_power_shared_sparse_monomial_fake,
        )
        torch.library.register_fake(
            "ye3t_runtime::symmetric_power_shared_sparse_monomial_adjoint",
            symmetric_power_shared_sparse_monomial_adjoint_fake,
        )
        torch.library.register_fake(
            "ye3t_runtime::"
            "symmetric_power_shared_sparse_monomial_double_backward",
            symmetric_power_shared_sparse_monomial_double_backward_fake,
        )

    def symmetric_power_shared_monomial_batched_adjoint_fake(
        output_adjoint,
        input,
        monomial_counts,
        output_offsets,
        coefficient_terms,
        coefficient_outputs,
        coefficient_values,
    ):
        del (
            monomial_counts,
            output_offsets,
            coefficient_terms,
            coefficient_outputs,
            coefficient_values,
        )
        return input.new_empty(
            (
                output_adjoint.shape[0],
                input.shape[0],
                input.shape[1],
            )
        )

    torch.library.register_fake(
        "ye3t_runtime::symmetric_power_shared_monomial_batched_adjoint",
        symmetric_power_shared_monomial_batched_adjoint_fake,
    )

    def factorized_angular_fake(
        packed_slots,
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
        source_dimension,
        workspace_dimension,
        output_dimension,
    ):
        del workspace_dimension
        return packed_slots.new_empty(
            (packed_slots.shape[0], int(output_dimension))
        )

    def factorized_angular_adjoint_fake(
        output_adjoint,
        packed_slots,
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
        source_dimension,
        workspace_dimension,
    ):
        del workspace_dimension
        return packed_slots.new_empty(packed_slots.shape)

    def factorized_angular_double_backward_fake(
        packed_adjoint_tangent,
        output_adjoint,
        packed_slots,
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
        source_dimension,
        workspace_dimension,
    ):
        del (
            packed_adjoint_tangent,
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
            source_dimension,
            workspace_dimension,
        )
        return (
            output_adjoint.new_empty(output_adjoint.shape),
            packed_slots.new_empty(packed_slots.shape),
        )

    torch.library.register_fake(
        "ye3t_runtime::factorized_angular",
        factorized_angular_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::factorized_angular_adjoint",
        factorized_angular_adjoint_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::factorized_angular_double_backward",
        factorized_angular_double_backward_fake,
    )

    def factorized_angular_heterogeneous_fake(
        packed_slots,
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
        root_projection_starts,
        root_projection_dimensions,
        root_output_offsets,
        projection_values,
        source_dimension,
        workspace_dimension,
        output_dimension,
    ):
        del workspace_dimension
        return packed_slots.new_empty(
            (packed_slots.shape[0], int(output_dimension))
        )

    def factorized_angular_heterogeneous_adjoint_fake(
        output_adjoint,
        packed_slots,
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
        root_projection_starts,
        root_projection_dimensions,
        root_output_offsets,
        projection_values,
        source_dimension,
        workspace_dimension,
    ):
        del output_adjoint, workspace_dimension
        return packed_slots.new_empty(packed_slots.shape)

    def factorized_angular_heterogeneous_adjoint_with_workspace_fake(
        output_adjoint,
        packed_slots,
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
        root_projection_starts,
        root_projection_dimensions,
        root_output_offsets,
        projection_values,
        source_dimension,
        workspace_dimension,
    ):
        del output_adjoint
        workspace_shape = (
            packed_slots.shape[0] * int(source_dimension),
            int(workspace_dimension),
        )
        return (
            packed_slots.new_empty(packed_slots.shape),
            packed_slots.new_empty(workspace_shape),
            packed_slots.new_empty(workspace_shape),
        )

    def factorized_angular_heterogeneous_double_backward_fake(
        packed_adjoint_tangent,
        output_adjoint,
        packed_slots,
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
        root_projection_starts,
        root_projection_dimensions,
        root_output_offsets,
        projection_values,
        source_dimension,
        workspace_dimension,
    ):
        del packed_adjoint_tangent, workspace_dimension
        return (
            output_adjoint.new_empty(output_adjoint.shape),
            packed_slots.new_empty(packed_slots.shape),
        )

    def factorized_angular_heterogeneous_double_backward_from_workspace_fake(
        packed_adjoint_tangent,
        output_adjoint,
        workspace,
        workspace_adjoint,
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
        root_projection_starts,
        root_projection_dimensions,
        root_output_offsets,
        projection_values,
        source_dimension,
        workspace_dimension,
    ):
        del workspace, workspace_adjoint, source_dimension, workspace_dimension
        return (
            output_adjoint.new_empty(output_adjoint.shape),
            packed_adjoint_tangent.new_empty(packed_adjoint_tangent.shape),
        )

    torch.library.register_fake(
        "ye3t_runtime::factorized_angular_heterogeneous",
        factorized_angular_heterogeneous_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::factorized_angular_heterogeneous_adjoint",
        factorized_angular_heterogeneous_adjoint_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::factorized_angular_heterogeneous_adjoint_with_workspace",
        factorized_angular_heterogeneous_adjoint_with_workspace_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::factorized_angular_heterogeneous_double_backward",
        factorized_angular_heterogeneous_double_backward_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::"
        "factorized_angular_heterogeneous_double_backward_from_workspace",
        factorized_angular_heterogeneous_double_backward_from_workspace_fake,
    )

    def factorized_angular_segmented_fake(
        packed_slots,
        segment_source_offsets,
        segment_input_offsets,
        segment_input_dimensions,
        segment_workspace_offsets,
        segment_workspace_dimensions,
        segment_node_offsets,
        segment_root_offsets,
        segment_projection_offsets,
        segment_projection_dimensions,
        segment_output_offsets,
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
        root_projection_starts,
        root_projection_dimensions,
        root_output_offsets,
        projection_values,
        total_source_count,
        workspace_dimension,
        output_dimension,
    ):
        del total_source_count, workspace_dimension
        return packed_slots.new_empty(
            (packed_slots.shape[0], int(output_dimension))
        )

    def factorized_angular_segmented_adjoint_fake(
        output_adjoint,
        packed_slots,
        segment_source_offsets,
        segment_input_offsets,
        segment_input_dimensions,
        segment_workspace_offsets,
        segment_workspace_dimensions,
        segment_node_offsets,
        segment_root_offsets,
        segment_projection_offsets,
        segment_projection_dimensions,
        segment_output_offsets,
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
        root_projection_starts,
        root_projection_dimensions,
        root_output_offsets,
        projection_values,
        total_source_count,
        workspace_dimension,
    ):
        del output_adjoint, total_source_count, workspace_dimension
        return packed_slots.new_empty(packed_slots.shape)

    def factorized_angular_segmented_double_backward_fake(
        packed_adjoint_tangent,
        output_adjoint,
        packed_slots,
        segment_source_offsets,
        segment_input_offsets,
        segment_input_dimensions,
        segment_workspace_offsets,
        segment_workspace_dimensions,
        segment_node_offsets,
        segment_root_offsets,
        segment_projection_offsets,
        segment_projection_dimensions,
        segment_output_offsets,
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
        root_projection_starts,
        root_projection_dimensions,
        root_output_offsets,
        projection_values,
        total_source_count,
        workspace_dimension,
    ):
        del packed_adjoint_tangent, total_source_count, workspace_dimension
        return (
            output_adjoint.new_empty(output_adjoint.shape),
            packed_slots.new_empty(packed_slots.shape),
        )

    torch.library.register_fake(
        "ye3t_runtime::factorized_angular_segmented",
        factorized_angular_segmented_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::factorized_angular_segmented_adjoint",
        factorized_angular_segmented_adjoint_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::factorized_angular_segmented_double_backward",
        factorized_angular_segmented_double_backward_fake,
    )

    def factorized_linear_fake(
        packed_slots,
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
        source_dimension,
        workspace_dimension,
        weight,
        bias,
    ):
        del (
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
            source_dimension,
            workspace_dimension,
            weight,
            bias,
        )
        return packed_slots.new_empty((packed_slots.shape[0],))

    def factorized_linear_adjoint_fake(
        output_adjoint,
        packed_slots,
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
        source_dimension,
        workspace_dimension,
        weight,
    ):
        del (
            output_adjoint,
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
            source_dimension,
            workspace_dimension,
        )
        return (
            packed_slots.new_empty(packed_slots.shape),
            weight.new_empty(weight.shape),
            weight.new_empty(()),
        )

    torch.library.register_fake(
        "ye3t_runtime::factorized_angular_linear",
        factorized_linear_fake,
    )
    torch.library.register_fake(
        "ye3t_runtime::factorized_angular_linear_adjoint",
        factorized_linear_adjoint_fake,
    )

    class _RadialValueAdjoint(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            values_adjoint,
            radial_derivative,
            radii,
            cutoffs,
            lambdas,
            radial_index,
        ):
            ctx.save_for_backward(
                values_adjoint,
                radial_derivative,
                radii,
                cutoffs,
                lambdas,
            )
            ctx.radial_index = int(radial_index)
            ctx.set_materialize_grads(False)
            return values_adjoint * radial_derivative

        @staticmethod
        def backward(ctx, grad_grad_radii):
            (
                values_adjoint,
                radial_derivative,
                radii,
                cutoffs,
                lambdas,
            ) = ctx.saved_tensors
            if grad_grad_radii is None:
                grad_grad_radii = torch.zeros_like(radii)
            grad_values_adjoint = (
                grad_grad_radii * radial_derivative
            )
            with torch.enable_grad():
                reference_radii = (
                    radii.detach().requires_grad_(True)
                )
                reference_values = _cheb_exp_cos_radial_reference(
                    reference_radii,
                    cutoffs.detach(),
                    lambdas.detach(),
                    ctx.radial_index,
                )
                if reference_values.requires_grad:
                    first = torch.autograd.grad(
                        reference_values,
                        reference_radii,
                        values_adjoint.detach(),
                        create_graph=True,
                    )[0]
                    if first.requires_grad:
                        grad_radii = torch.autograd.grad(
                            first,
                            reference_radii,
                            grad_grad_radii.detach(),
                        )[0]
                    else:
                        grad_radii = torch.zeros_like(radii)
                else:
                    grad_radii = torch.zeros_like(radii)
            return (
                grad_values_adjoint,
                None,
                grad_radii,
                None,
                None,
                None,
            )

    class _RadialTableValueAdjoint(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            values_adjoint,
            radial_derivative,
            radii,
            cutoffs,
            lambdas,
            maximum_radial_index,
        ):
            ctx.save_for_backward(
                values_adjoint,
                radial_derivative,
                radii,
                cutoffs,
                lambdas,
            )
            ctx.maximum_radial_index = int(
                maximum_radial_index
            )
            ctx.set_materialize_grads(False)
            return (
                values_adjoint * radial_derivative
            ).sum(dim=1)

        @staticmethod
        def backward(ctx, grad_grad_radii):
            (
                values_adjoint,
                radial_derivative,
                radii,
                cutoffs,
                lambdas,
            ) = ctx.saved_tensors
            if grad_grad_radii is None:
                grad_grad_radii = torch.zeros_like(radii)
            if hasattr(
                torch.ops.ye3t_runtime,
                "cheb_exp_cos_radial_table_double_backward",
            ):
                (
                    grad_values_adjoint,
                    grad_radii,
                ) = (
                    torch.ops.ye3t_runtime
                    .cheb_exp_cos_radial_table_double_backward(
                        values_adjoint.contiguous(),
                        radial_derivative.contiguous(),
                        radii.contiguous(),
                        cutoffs.contiguous(),
                        lambdas.contiguous(),
                        grad_grad_radii.contiguous(),
                        ctx.maximum_radial_index,
                    )
                )
                return (
                    grad_values_adjoint,
                    None,
                    grad_radii,
                    None,
                    None,
                    None,
                )
            grad_values_adjoint = (
                grad_grad_radii.unsqueeze(-1)
                * radial_derivative
            )
            with torch.enable_grad():
                reference_radii = (
                    radii.detach().requires_grad_(True)
                )
                reference_values = (
                    _cheb_exp_cos_radial_table_reference(
                        reference_radii,
                        cutoffs.detach(),
                        lambdas.detach(),
                        ctx.maximum_radial_index,
                    )
                )
                first = torch.autograd.grad(
                    reference_values,
                    reference_radii,
                    values_adjoint.detach(),
                    create_graph=True,
                )[0]
                if first.requires_grad:
                    grad_radii = torch.autograd.grad(
                        first,
                        reference_radii,
                        grad_grad_radii.detach(),
                    )[0]
                else:
                    grad_radii = torch.zeros_like(radii)
            return (
                grad_values_adjoint,
                None,
                grad_radii,
                None,
                None,
                None,
            )

    def radial_setup(ctx, inputs, output):
        radii, cutoffs, lambdas, radial_index = inputs
        ctx.save_for_backward(
            radii,
            cutoffs,
            lambdas,
            output[1],
        )
        ctx.radial_index = int(radial_index)
        ctx.mark_non_differentiable(output[1])

    def radial_backward(ctx, values_adjoint, derivatives_adjoint):
        del derivatives_adjoint
        (
            radii,
            cutoffs,
            lambdas,
            radial_derivative,
        ) = ctx.saved_tensors
        needs = ctx.needs_input_grad[:3]
        if (
            needs == (True, False, False)
            and values_adjoint is not None
        ):
            grad_radii = _RadialValueAdjoint.apply(
                values_adjoint,
                radial_derivative,
                radii,
                cutoffs,
                lambdas,
                ctx.radial_index,
            )
            return grad_radii, None, None, None
        selected = tuple(
            value
            for value, needed in zip((radii, cutoffs, lambdas), needs)
            if needed
        )
        if not selected:
            return None, None, None, None
        with torch.enable_grad():
            values = _cheb_exp_cos_radial_reference(
                radii,
                cutoffs,
                lambdas,
                ctx.radial_index,
            )
            if values.requires_grad:
                gradients = torch.autograd.grad(
                    values,
                    selected,
                    values_adjoint,
                    create_graph=torch.is_grad_enabled(),
                    allow_unused=True,
                )
            else:
                gradients = tuple(value * 0.0 for value in selected)
        output = []
        gradient_index = 0
        for needed in needs:
            if needed:
                output.append(gradients[gradient_index])
                gradient_index += 1
            else:
                output.append(None)
        return tuple(output) + (None,)

    def radial_table_setup(ctx, inputs, output):
        radii, cutoffs, lambdas, maximum_radial_index = inputs
        ctx.save_for_backward(
            radii,
            cutoffs,
            lambdas,
            output[1],
        )
        ctx.maximum_radial_index = int(maximum_radial_index)
        ctx.mark_non_differentiable(output[1])

    def radial_table_backward(
        ctx,
        values_adjoint,
        derivatives_adjoint,
    ):
        del derivatives_adjoint
        (
            radii,
            cutoffs,
            lambdas,
            radial_derivative,
        ) = ctx.saved_tensors
        needs = ctx.needs_input_grad[:3]
        if (
            needs == (True, False, False)
            and values_adjoint is not None
        ):
            grad_radii = _RadialTableValueAdjoint.apply(
                values_adjoint,
                radial_derivative,
                radii,
                cutoffs,
                lambdas,
                ctx.maximum_radial_index,
            )
            return grad_radii, None, None, None
        selected = tuple(
            value
            for value, needed in zip((radii, cutoffs, lambdas), needs)
            if needed
        )
        if not selected:
            return None, None, None, None
        with torch.enable_grad():
            values = _cheb_exp_cos_radial_table_reference(
                radii,
                cutoffs,
                lambdas,
                ctx.maximum_radial_index,
            )
            if values.requires_grad:
                gradients = torch.autograd.grad(
                    values,
                    selected,
                    values_adjoint,
                    create_graph=torch.is_grad_enabled(),
                    allow_unused=True,
                )
            else:
                gradients = tuple(value * 0.0 for value in selected)
        output = []
        gradient_index = 0
        for needed in needs:
            if needed:
                output.append(gradients[gradient_index])
                gradient_index += 1
            else:
                output.append(None)
        return tuple(output) + (None,)

    class _SphericalValueAdjoint(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            values_adjoint,
            spherical_derivative,
            edge_vectors,
            angular_momentum,
            epsilon,
        ):
            ctx.save_for_backward(
                values_adjoint,
                spherical_derivative,
                edge_vectors,
            )
            ctx.angular_momentum = int(angular_momentum)
            ctx.epsilon = float(epsilon)
            ctx.set_materialize_grads(False)
            return (
                values_adjoint.unsqueeze(-1)
                * spherical_derivative
            ).sum(dim=1)

        @staticmethod
        def backward(ctx, grad_grad_edges):
            (
                values_adjoint,
                spherical_derivative,
                edge_vectors,
            ) = ctx.saved_tensors
            if grad_grad_edges is None:
                grad_grad_edges = torch.zeros_like(edge_vectors)
            grad_values_adjoint = (
                grad_grad_edges.unsqueeze(1)
                * spherical_derivative
            ).sum(dim=-1)
            with torch.enable_grad():
                reference_edges = (
                    edge_vectors.detach().requires_grad_(True)
                )
                reference_values = (
                    _spherical_harmonics_reference_values(
                        reference_edges,
                        ctx.angular_momentum,
                        real_output=True,
                        epsilon=ctx.epsilon,
                    )
                )
                if reference_values.requires_grad:
                    first = torch.autograd.grad(
                        reference_values,
                        reference_edges,
                        values_adjoint.detach(),
                        create_graph=True,
                    )[0]
                    if first.requires_grad:
                        grad_edges = torch.autograd.grad(
                            first,
                            reference_edges,
                            grad_grad_edges.detach(),
                        )[0]
                    else:
                        grad_edges = torch.zeros_like(edge_vectors)
                else:
                    grad_edges = torch.zeros_like(edge_vectors)
            return (
                grad_values_adjoint,
                None,
                grad_edges,
                None,
                None,
            )

    class _SphericalTableValueAdjoint(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            values_adjoint,
            spherical_derivative,
            edge_vectors,
            maximum_angular_momentum,
            epsilon,
        ):
            ctx.save_for_backward(
                values_adjoint,
                spherical_derivative,
                edge_vectors,
            )
            ctx.maximum_angular_momentum = int(
                maximum_angular_momentum
            )
            ctx.epsilon = float(epsilon)
            ctx.set_materialize_grads(False)
            return (
                values_adjoint.unsqueeze(-1)
                * spherical_derivative
            ).sum(dim=1)

        @staticmethod
        def backward(ctx, grad_grad_edges):
            (
                values_adjoint,
                spherical_derivative,
                edge_vectors,
            ) = ctx.saved_tensors
            if grad_grad_edges is None:
                grad_grad_edges = torch.zeros_like(edge_vectors)
            if (
                edge_vectors.is_cuda
                and hasattr(
                    torch.ops.ye3t_runtime,
                    "spherical_harmonics_table_double_backward",
                )
            ):
                (
                    grad_values_adjoint,
                    grad_edges,
                ) = (
                    torch.ops.ye3t_runtime
                    .spherical_harmonics_table_double_backward(
                        edge_vectors.contiguous(),
                        values_adjoint.contiguous(),
                        grad_grad_edges.contiguous(),
                        ctx.maximum_angular_momentum,
                        ctx.epsilon,
                    )
                )
                return (
                    grad_values_adjoint,
                    None,
                    grad_edges,
                    None,
                    None,
                )
            grad_values_adjoint = (
                grad_grad_edges.unsqueeze(1)
                * spherical_derivative
            ).sum(dim=-1)
            with torch.enable_grad():
                reference_edges = (
                    edge_vectors.detach().requires_grad_(True)
                )
                reference_values = (
                    _spherical_harmonics_table_reference_values(
                        reference_edges,
                        ctx.maximum_angular_momentum,
                        real_output=True,
                        epsilon=ctx.epsilon,
                    )
                )
                if reference_values.requires_grad:
                    first = torch.autograd.grad(
                        reference_values,
                        reference_edges,
                        values_adjoint.detach(),
                        create_graph=True,
                    )[0]
                    if first.requires_grad:
                        grad_edges = torch.autograd.grad(
                            first,
                            reference_edges,
                            grad_grad_edges.detach(),
                        )[0]
                    else:
                        grad_edges = torch.zeros_like(edge_vectors)
                else:
                    grad_edges = torch.zeros_like(edge_vectors)
            return (
                grad_values_adjoint,
                None,
                grad_edges,
                None,
                None,
            )

    def spherical_setup(ctx, inputs, output):
        edge_vectors, angular_momentum, real_output, epsilon = inputs
        ctx.save_for_backward(edge_vectors, output[1])
        ctx.angular_momentum = int(angular_momentum)
        ctx.real_output = bool(real_output)
        ctx.epsilon = float(epsilon)
        ctx.mark_non_differentiable(output[1])

    def spherical_backward(
        ctx,
        values_adjoint,
        derivatives_adjoint,
    ):
        del derivatives_adjoint
        if not ctx.needs_input_grad[0]:
            return None, None, None, None
        edge_vectors, spherical_derivative = ctx.saved_tensors
        if ctx.real_output and values_adjoint is not None:
            edge_adjoint = _SphericalValueAdjoint.apply(
                values_adjoint,
                spherical_derivative,
                edge_vectors,
                ctx.angular_momentum,
                ctx.epsilon,
            )
            return edge_adjoint, None, None, None
        with torch.enable_grad():
            values = _spherical_harmonics_reference_values(
                edge_vectors,
                ctx.angular_momentum,
                real_output=ctx.real_output,
                epsilon=ctx.epsilon,
            )
            edge_adjoint = torch.autograd.grad(
                values,
                edge_vectors,
                values_adjoint,
                create_graph=torch.is_grad_enabled(),
            )[0]
        return edge_adjoint, None, None, None

    def spherical_table_setup(ctx, inputs, output):
        (
            edge_vectors,
            maximum_angular_momentum,
            real_output,
            epsilon,
        ) = inputs
        ctx.save_for_backward(edge_vectors, output[1])
        ctx.maximum_angular_momentum = int(
            maximum_angular_momentum
        )
        ctx.real_output = bool(real_output)
        ctx.epsilon = float(epsilon)
        ctx.mark_non_differentiable(output[1])

    def spherical_table_backward(
        ctx,
        values_adjoint,
        derivatives_adjoint,
    ):
        del derivatives_adjoint
        if not ctx.needs_input_grad[0]:
            return None, None, None, None
        edge_vectors, spherical_derivative = ctx.saved_tensors
        if ctx.real_output and values_adjoint is not None:
            edge_adjoint = _SphericalTableValueAdjoint.apply(
                values_adjoint,
                spherical_derivative,
                edge_vectors,
                ctx.maximum_angular_momentum,
                ctx.epsilon,
            )
            return edge_adjoint, None, None, None
        with torch.enable_grad():
            values = _spherical_harmonics_table_reference_values(
                edge_vectors,
                ctx.maximum_angular_momentum,
                real_output=ctx.real_output,
                epsilon=ctx.epsilon,
            )
            edge_adjoint = torch.autograd.grad(
                values,
                edge_vectors,
                values_adjoint,
                create_graph=torch.is_grad_enabled(),
            )[0]
        return edge_adjoint, None, None, None

    torch.library.register_autograd(
        "ye3t_runtime::cheb_exp_cos_radial_with_derivative",
        radial_backward,
        setup_context=radial_setup,
    )
    torch.library.register_autograd(
        "ye3t_runtime::cheb_exp_cos_radial_table_with_derivative",
        radial_table_backward,
        setup_context=radial_table_setup,
    )
    torch.library.register_autograd(
        "ye3t_runtime::spherical_harmonics_with_derivative",
        spherical_backward,
        setup_context=spherical_setup,
    )
    torch.library.register_autograd(
        "ye3t_runtime::spherical_harmonics_table_with_derivative",
        spherical_table_backward,
        setup_context=spherical_table_setup,
    )

    def plain_site_basis_product_setup(ctx, inputs, output):
        (
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
        ) = inputs
        del (
            radial_derivatives,
            angular_derivatives,
            prefactor_derivatives_center,
            prefactor_derivatives_neighbor,
            radial_directions,
        )
        ctx.save_for_backward(
            radial_values,
            angular_values,
            prefactors,
            term_groups,
            term_channels,
        )
        ctx.channel_count = int(channel_count)
        ctx.mark_non_differentiable(output[1], output[2], output[3])

    def plain_site_basis_product_backward(
        ctx,
        values_adjoint,
        edge_derivatives_adjoint,
        charge_center_adjoint,
        charge_neighbor_adjoint,
    ):
        del (
            edge_derivatives_adjoint,
            charge_center_adjoint,
            charge_neighbor_adjoint,
        )
        (
            radial_values,
            angular_values,
            prefactors,
            term_groups,
            term_channels,
        ) = ctx.saved_tensors
        input_positions = (0, 2, 4)
        values_by_position = {
            0: radial_values,
            2: angular_values,
            4: prefactors,
        }
        selected_positions = [
            position
            for position in input_positions
            if ctx.needs_input_grad[position]
        ]
        gradients_by_position = {}
        if selected_positions:
            selected = tuple(
                values_by_position[position]
                for position in selected_positions
            )
            with torch.enable_grad():
                values = _plain_site_basis_product_reference_values(
                    radial_values,
                    angular_values,
                    prefactors,
                    term_groups,
                    term_channels,
                    ctx.channel_count,
                )
                gradients = torch.autograd.grad(
                    values,
                    selected,
                    values_adjoint,
                    create_graph=torch.is_grad_enabled(),
                    allow_unused=True,
                )
            gradients_by_position = dict(
                zip(selected_positions, gradients)
            )
        return tuple(
            gradients_by_position.get(position)
            for position in range(11)
        )

    torch.library.register_autograd(
        "ye3t_runtime::plain_site_basis_product_with_derivative",
        plain_site_basis_product_backward,
        setup_context=plain_site_basis_product_setup,
    )

    def scheduled_radial_angular_channels_setup(
        ctx,
        inputs,
        output,
    ):
        (
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
        ) = inputs
        del (
            radial_derivatives,
            angular_derivatives,
            radial_directions,
        )
        ctx.save_for_backward(
            radial_values,
            angular_values,
            edge_types,
            channel_radial_indices,
            channel_angular_indices,
            channel_types,
            channel_scales,
        )
        ctx.mark_non_differentiable(output[1])

    def scheduled_radial_angular_channels_backward(
        ctx,
        values_adjoint,
        derivatives_adjoint,
    ):
        del derivatives_adjoint
        if values_adjoint is None:
            return (None,) * 10
        (
            radial_values,
            angular_values,
            edge_types,
            channel_radial_indices,
            channel_angular_indices,
            channel_types,
            channel_scales,
        ) = ctx.saved_tensors
        values_by_position = {
            0: radial_values,
            2: angular_values,
            9: channel_scales,
        }
        selected_positions = [
            position
            for position in (0, 2, 9)
            if ctx.needs_input_grad[position]
        ]
        gradients_by_position = {}
        if selected_positions:
            selected = tuple(
                values_by_position[position]
                for position in selected_positions
            )
            with torch.enable_grad():
                values = (
                    _scheduled_radial_angular_channels_reference_values(
                        radial_values,
                        angular_values,
                        edge_types,
                        channel_radial_indices,
                        channel_angular_indices,
                        channel_types,
                        channel_scales,
                    )
                )
                gradients = torch.autograd.grad(
                    values,
                    selected,
                    values_adjoint,
                    create_graph=torch.is_grad_enabled(),
                    allow_unused=True,
                )
            gradients_by_position = dict(
                zip(selected_positions, gradients)
            )
        return tuple(
            gradients_by_position.get(position)
            for position in range(10)
        )

    torch.library.register_autograd(
        "ye3t_runtime::scheduled_radial_angular_channels_with_derivative",
        scheduled_radial_angular_channels_backward,
        setup_context=scheduled_radial_angular_channels_setup,
    )

    def plain_site_basis_product_adjoint_setup(ctx, inputs, output):
        del output
        ctx.save_for_backward(*inputs)

    def plain_site_basis_product_adjoint_backward(
        ctx,
        position_adjoint,
        charge_center_adjoint,
        charge_neighbor_adjoint,
    ):
        saved = ctx.saved_tensors
        differentiable_positions = tuple(range(10)) + (12,)
        selected_positions = [
            position
            for position in differentiable_positions
            if ctx.needs_input_grad[position]
        ]
        gradients_by_position = {}
        if selected_positions:
            selected = tuple(saved[position] for position in selected_positions)
            with torch.enable_grad():
                outputs = _plain_site_basis_product_adjoint_reference(
                    *saved
                )
                output_adjoints = (
                    position_adjoint
                    if position_adjoint is not None
                    else torch.zeros_like(outputs[0]),
                    charge_center_adjoint
                    if charge_center_adjoint is not None
                    else torch.zeros_like(outputs[1]),
                    charge_neighbor_adjoint
                    if charge_neighbor_adjoint is not None
                    else torch.zeros_like(outputs[2]),
                )
                gradients = torch.autograd.grad(
                    outputs,
                    selected,
                    output_adjoints,
                    create_graph=torch.is_grad_enabled(),
                    allow_unused=True,
                )
            gradients_by_position = dict(
                zip(selected_positions, gradients)
            )
        return tuple(
            gradients_by_position.get(position)
            for position in range(13)
        )

    torch.library.register_autograd(
        "ye3t_runtime::plain_site_basis_product_adjoint",
        plain_site_basis_product_adjoint_backward,
        setup_context=plain_site_basis_product_adjoint_setup,
    )

    def density_setup(ctx, inputs, output):
        edge_values, centers, atom_count = inputs
        del edge_values, output
        ctx.save_for_backward(centers)
        ctx.atom_count = int(atom_count)

    def density_backward(ctx, atomic_adjoint):
        centers, = ctx.saved_tensors
        return (
            torch.ops.ye3t_runtime.density_accumulate_adjoint(
                atomic_adjoint.contiguous(),
                centers,
            ),
            None,
            None,
        )

    def density_adjoint_setup(ctx, inputs, output):
        atomic_adjoint, centers = inputs
        del output
        ctx.save_for_backward(centers)
        ctx.atom_count = int(atomic_adjoint.shape[0])

    def density_adjoint_backward(ctx, edge_adjoint_tangent):
        centers, = ctx.saved_tensors
        return (
            torch.ops.ye3t_runtime.density_accumulate(
                edge_adjoint_tangent.contiguous(),
                centers,
                ctx.atom_count,
            ),
            None,
        )

    torch.library.register_autograd(
        "ye3t_runtime::density_accumulate",
        density_backward,
        setup_context=density_setup,
    )
    torch.library.register_autograd(
        "ye3t_runtime::density_accumulate_adjoint",
        density_adjoint_backward,
        setup_context=density_adjoint_setup,
    )

    def edge_outer_setup(ctx, inputs, output):
        left, right, centers, atom_count = inputs
        del output
        ctx.save_for_backward(left, right, centers)
        ctx.atom_count = int(atom_count)

    def edge_outer_backward(ctx, atomic_adjoint):
        left, right, centers = ctx.saved_tensors
        left_adjoint, right_adjoint = (
            torch.ops.ye3t_runtime.edge_outer_accumulate_adjoint(
                atomic_adjoint.contiguous(),
                left,
                right,
                centers,
            )
        )
        return left_adjoint, right_adjoint, None, None

    def edge_outer_adjoint_setup(ctx, inputs, output):
        atomic_adjoint, left, right, centers = inputs
        del output
        ctx.save_for_backward(
            atomic_adjoint,
            left,
            right,
            centers,
        )

    def edge_outer_adjoint_backward(
        ctx,
        left_adjoint_tangent,
        right_adjoint_tangent,
    ):
        atomic_adjoint, left, right, centers = ctx.saved_tensors
        if left_adjoint_tangent is None:
            left_adjoint_tangent = torch.zeros_like(left)
        if right_adjoint_tangent is None:
            right_adjoint_tangent = torch.zeros_like(right)
        return (
            *torch.ops.ye3t_runtime.edge_outer_accumulate_double_backward(
                atomic_adjoint,
                left,
                right,
                left_adjoint_tangent.contiguous(),
                right_adjoint_tangent.contiguous(),
                centers,
            ),
            None,
        )

    torch.library.register_autograd(
        "ye3t_runtime::edge_outer_accumulate",
        edge_outer_backward,
        setup_context=edge_outer_setup,
    )
    torch.library.register_autograd(
        "ye3t_runtime::edge_outer_accumulate_adjoint",
        edge_outer_adjoint_backward,
        setup_context=edge_outer_adjoint_setup,
    )

    def softmax_gaussian_role_density_setup(ctx, inputs, output):
        (
            distances,
            cutoffs,
            filter_centers,
            filter_width,
            edge_values,
            atom_centers,
            atom_count,
        ) = inputs
        del output
        ctx.save_for_backward(
            distances,
            cutoffs,
            filter_centers,
            edge_values,
            atom_centers,
        )
        ctx.filter_width = float(filter_width)
        ctx.atom_count = int(atom_count)

    def softmax_gaussian_role_density_backward(ctx, atomic_adjoint):
        (
            distances,
            cutoffs,
            filter_centers,
            edge_values,
            atom_centers,
        ) = ctx.saved_tensors
        distance_adjoint, edge_adjoint = (
            torch.ops.ye3t_runtime.softmax_gaussian_role_density_adjoint(
                atomic_adjoint.contiguous(),
                distances,
                cutoffs,
                filter_centers,
                ctx.filter_width,
                edge_values,
                atom_centers,
            )
        )
        return (
            distance_adjoint,
            None,
            None,
            None,
            edge_adjoint,
            None,
            None,
        )

    def softmax_gaussian_role_density_adjoint_setup(ctx, inputs, output):
        (
            atomic_adjoint,
            distances,
            cutoffs,
            filter_centers,
            filter_width,
            edge_values,
            atom_centers,
        ) = inputs
        del output
        ctx.save_for_backward(
            atomic_adjoint,
            distances,
            cutoffs,
            filter_centers,
            edge_values,
            atom_centers,
        )
        ctx.filter_width = float(filter_width)

    def softmax_gaussian_role_density_adjoint_backward(
        ctx,
        distance_adjoint_tangent,
        edge_adjoint_tangent,
    ):
        (
            atomic_adjoint,
            distances,
            cutoffs,
            filter_centers,
            edge_values,
            atom_centers,
        ) = ctx.saved_tensors
        if distance_adjoint_tangent is None:
            distance_adjoint_tangent = torch.zeros_like(distances)
        if edge_adjoint_tangent is None:
            edge_adjoint_tangent = torch.zeros_like(edge_values)
        (
            atomic_adjoint_tangent,
            distance_second_adjoint,
            edge_second_adjoint,
        ) = torch.ops.ye3t_runtime.softmax_gaussian_role_density_double_backward(
            atomic_adjoint,
            distances,
            cutoffs,
            filter_centers,
            ctx.filter_width,
            edge_values,
            distance_adjoint_tangent.contiguous(),
            edge_adjoint_tangent.contiguous(),
            atom_centers,
        )
        return (
            atomic_adjoint_tangent,
            distance_second_adjoint,
            None,
            None,
            None,
            edge_second_adjoint,
            None,
        )

    torch.library.register_autograd(
        "ye3t_runtime::softmax_gaussian_role_density",
        softmax_gaussian_role_density_backward,
        setup_context=softmax_gaussian_role_density_setup,
    )
    torch.library.register_autograd(
        "ye3t_runtime::softmax_gaussian_role_density_adjoint",
        softmax_gaussian_role_density_adjoint_backward,
        setup_context=softmax_gaussian_role_density_adjoint_setup,
    )

    def scheduled_role_density_setup(ctx, inputs, output):
        del output
        (
            radial_values,
            angular_values,
            distances,
            cutoffs,
            filter_centers,
            filter_width,
            soft_weights,
            edge_types,
            channel_radial_indices,
            channel_angular_indices,
            channel_types,
            channel_scales,
            atom_centers,
            atom_count,
        ) = inputs
        ctx.save_for_backward(
            radial_values,
            angular_values,
            distances,
            cutoffs,
            filter_centers,
            soft_weights,
            edge_types,
            channel_radial_indices,
            channel_angular_indices,
            channel_types,
            channel_scales,
            atom_centers,
        )
        ctx.filter_width = float(filter_width)
        ctx.atom_count = int(atom_count)

    def scheduled_role_density_backward(ctx, atomic_adjoint):
        saved = ctx.saved_tensors
        adjoints = torch.ops.ye3t_runtime.scheduled_softmax_gaussian_role_density_adjoint(
            atomic_adjoint.contiguous(),
            saved[0],
            saved[1],
            saved[2],
            saved[3],
            saved[4],
            ctx.filter_width,
            saved[5],
            saved[6],
            saved[7],
            saved[8],
            saved[9],
            saved[10],
            saved[11],
        )
        return (
            adjoints[0],
            adjoints[1],
            adjoints[2],
            None,
            None,
            None,
            adjoints[3],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    def scheduled_role_density_adjoint_setup(ctx, inputs, output):
        del output
        ctx.save_for_backward(*(
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[5],
            inputs[7],
            inputs[8],
            inputs[9],
            inputs[10],
            inputs[11],
            inputs[12],
            inputs[13],
        ))
        ctx.filter_width = float(inputs[6])

    def scheduled_role_density_adjoint_backward(
        ctx,
        radial_adjoint_tangent,
        angular_adjoint_tangent,
        distance_adjoint_tangent,
        soft_adjoint_tangent,
    ):
        saved = ctx.saved_tensors
        tangents = (
            torch.zeros_like(saved[1])
            if radial_adjoint_tangent is None
            else radial_adjoint_tangent.contiguous(),
            torch.zeros_like(saved[2])
            if angular_adjoint_tangent is None
            else angular_adjoint_tangent.contiguous(),
            torch.zeros_like(saved[3])
            if distance_adjoint_tangent is None
            else distance_adjoint_tangent.contiguous(),
            torch.zeros_like(saved[6])
            if soft_adjoint_tangent is None
            else soft_adjoint_tangent.contiguous(),
        )
        results = torch.ops.ye3t_runtime.scheduled_softmax_gaussian_role_density_double_backward(
            saved[0],
            saved[1],
            saved[2],
            saved[3],
            saved[4],
            saved[5],
            ctx.filter_width,
            saved[6],
            tangents[0],
            tangents[1],
            tangents[2],
            tangents[3],
            saved[7],
            saved[8],
            saved[9],
            saved[10],
            saved[11],
            saved[12],
        )
        return (
            results[0],
            results[1],
            results[2],
            results[3],
            None,
            None,
            None,
            results[4],
            None,
            None,
            None,
            None,
            None,
            None,
        )

    torch.library.register_autograd(
        "ye3t_runtime::scheduled_softmax_gaussian_role_density",
        scheduled_role_density_backward,
        setup_context=scheduled_role_density_setup,
    )
    torch.library.register_autograd(
        "ye3t_runtime::scheduled_softmax_gaussian_role_density_adjoint",
        scheduled_role_density_adjoint_backward,
        setup_context=scheduled_role_density_adjoint_setup,
    )

    def carrier_gated_scatter_setup(ctx, inputs, output):
        (
            node_values,
            edge_gates,
            edge_sources,
            edge_targets,
            feature_channels,
            target_count,
        ) = inputs
        del output
        ctx.save_for_backward(
            node_values,
            edge_gates,
            edge_sources,
            edge_targets,
            feature_channels,
        )
        ctx.target_count = int(target_count)

    def carrier_gated_scatter_backward(ctx, target_adjoint):
        (
            node_values,
            edge_gates,
            edge_sources,
            edge_targets,
            feature_channels,
        ) = ctx.saved_tensors
        node_adjoint, gate_adjoint = (
            torch.ops.ye3t_runtime.carrier_gated_scatter_adjoint(
                target_adjoint.contiguous(),
                node_values,
                edge_gates,
                edge_sources,
                edge_targets,
                feature_channels,
            )
        )
        return (
            node_adjoint,
            gate_adjoint,
            None,
            None,
            None,
            None,
        )

    def carrier_gated_scatter_adjoint_setup(ctx, inputs, output):
        (
            target_adjoint,
            node_values,
            edge_gates,
            edge_sources,
            edge_targets,
            feature_channels,
        ) = inputs
        del output
        ctx.save_for_backward(
            target_adjoint,
            node_values,
            edge_gates,
            edge_sources,
            edge_targets,
            feature_channels,
        )

    def carrier_gated_scatter_adjoint_backward(
        ctx,
        node_adjoint_tangent,
        gate_adjoint_tangent,
    ):
        (
            target_adjoint,
            node_values,
            edge_gates,
            edge_sources,
            edge_targets,
            feature_channels,
        ) = ctx.saved_tensors
        if node_adjoint_tangent is None:
            node_adjoint_tangent = torch.zeros_like(node_values)
        if gate_adjoint_tangent is None:
            gate_adjoint_tangent = torch.zeros_like(edge_gates)
        return (
            *torch.ops.ye3t_runtime.carrier_gated_scatter_double_backward(
                target_adjoint,
                node_values,
                edge_gates,
                node_adjoint_tangent.contiguous(),
                gate_adjoint_tangent.contiguous(),
                edge_sources,
                edge_targets,
                feature_channels,
            ),
            None,
            None,
            None,
        )

    torch.library.register_autograd(
        "ye3t_runtime::carrier_gated_scatter",
        carrier_gated_scatter_backward,
        setup_context=carrier_gated_scatter_setup,
    )
    torch.library.register_autograd(
        "ye3t_runtime::carrier_gated_scatter_adjoint",
        carrier_gated_scatter_adjoint_backward,
        setup_context=carrier_gated_scatter_adjoint_setup,
    )

    def carrier_residual_gated_scatter_setup(ctx, inputs, output):
        (
            node_values,
            edge_gates,
            edge_sources,
            edge_targets,
            feature_channels,
        ) = inputs
        del output
        ctx.save_for_backward(
            node_values,
            edge_gates,
            edge_sources,
            edge_targets,
            feature_channels,
        )

    def carrier_residual_gated_scatter_backward(
        ctx,
        target_adjoint,
    ):
        (
            node_values,
            edge_gates,
            edge_sources,
            edge_targets,
            feature_channels,
        ) = ctx.saved_tensors
        node_adjoint, gate_adjoint = (
            torch.ops.ye3t_runtime.carrier_residual_gated_scatter_adjoint(
                target_adjoint.contiguous(),
                node_values,
                edge_gates,
                edge_sources,
                edge_targets,
                feature_channels,
            )
        )
        return (
            node_adjoint,
            gate_adjoint,
            None,
            None,
            None,
        )

    torch.library.register_autograd(
        "ye3t_runtime::carrier_residual_gated_scatter",
        carrier_residual_gated_scatter_backward,
        setup_context=carrier_residual_gated_scatter_setup,
    )

    def carrier_residual_gated_scatter_adjoint_backward(
        ctx,
        node_adjoint_tangent,
        gate_adjoint_tangent,
    ):
        (
            target_adjoint,
            node_values,
            edge_gates,
            edge_sources,
            edge_targets,
            feature_channels,
        ) = ctx.saved_tensors
        if node_adjoint_tangent is None:
            node_adjoint_tangent = torch.zeros_like(node_values)
        if gate_adjoint_tangent is None:
            gate_adjoint_tangent = torch.zeros_like(edge_gates)
        return (
            *torch.ops.ye3t_runtime.carrier_residual_gated_scatter_double_backward(
                target_adjoint,
                node_values,
                edge_gates,
                node_adjoint_tangent.contiguous(),
                gate_adjoint_tangent.contiguous(),
                edge_sources,
                edge_targets,
                feature_channels,
            ),
            None,
            None,
            None,
        )

    torch.library.register_autograd(
        "ye3t_runtime::carrier_residual_gated_scatter_adjoint",
        carrier_residual_gated_scatter_adjoint_backward,
        setup_context=carrier_gated_scatter_adjoint_setup,
    )

    def carrier_segmented_residual_gated_scatter_setup(
        ctx,
        inputs,
        output,
    ):
        del output
        ctx.save_for_backward(*inputs)

    def carrier_segmented_residual_gated_scatter_backward(
        ctx,
        target_adjoint,
    ):
        (
            node_values,
            edge_gates,
            edge_sources,
            edge_targets,
            target_offsets,
            source_offsets,
            source_edges,
            feature_channels,
            channel_offsets,
            channel_features,
        ) = ctx.saved_tensors
        operator = (
            torch.ops.ye3t_runtime
            .carrier_segmented_residual_gated_scatter_adjoint
        )
        node_adjoint, gate_adjoint = (
            operator(
                target_adjoint.contiguous(),
                node_values,
                edge_gates,
                edge_sources,
                edge_targets,
                target_offsets,
                source_offsets,
                source_edges,
                feature_channels,
                channel_offsets,
                channel_features,
            )
        )
        return (
            node_adjoint,
            gate_adjoint,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    def carrier_segmented_residual_gated_scatter_adjoint_setup(
        ctx,
        inputs,
        output,
    ):
        del output
        ctx.save_for_backward(*inputs)

    def carrier_segmented_residual_gated_scatter_adjoint_backward(
        ctx,
        node_adjoint_tangent,
        gate_adjoint_tangent,
    ):
        (
            target_adjoint,
            node_values,
            edge_gates,
            edge_sources,
            edge_targets,
            target_offsets,
            source_offsets,
            source_edges,
            feature_channels,
            channel_offsets,
            channel_features,
        ) = ctx.saved_tensors
        if node_adjoint_tangent is None:
            node_adjoint_tangent = torch.zeros_like(node_values)
        if gate_adjoint_tangent is None:
            gate_adjoint_tangent = torch.zeros_like(edge_gates)
        operator = (
            torch.ops.ye3t_runtime
            .carrier_segmented_residual_gated_scatter_double_backward
        )
        return (
            *operator(
                target_adjoint,
                node_values,
                edge_gates,
                node_adjoint_tangent.contiguous(),
                gate_adjoint_tangent.contiguous(),
                edge_sources,
                edge_targets,
                target_offsets,
                source_offsets,
                source_edges,
                feature_channels,
                channel_offsets,
                channel_features,
            ),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    torch.library.register_autograd(
        "ye3t_runtime::carrier_segmented_residual_gated_scatter",
        carrier_segmented_residual_gated_scatter_backward,
        setup_context=carrier_segmented_residual_gated_scatter_setup,
    )
    torch.library.register_autograd(
        "ye3t_runtime::carrier_segmented_residual_gated_scatter_adjoint",
        carrier_segmented_residual_gated_scatter_adjoint_backward,
        setup_context=
            carrier_segmented_residual_gated_scatter_adjoint_setup,
    )

    def source_arena_gather_setup(ctx, inputs, output):
        (
            producer,
            gather_indices,
            reverse_offsets,
            reverse_output_indices,
            center_types,
            atom_types,
        ) = inputs
        del producer, output
        ctx.save_for_backward(
            gather_indices,
            reverse_offsets,
            reverse_output_indices,
            center_types,
            atom_types,
        )

    def source_arena_gather_backward(ctx, output_adjoint):
        (
            gather_indices,
            reverse_offsets,
            reverse_output_indices,
            center_types,
            atom_types,
        ) = ctx.saved_tensors
        return (
            torch.ops.ye3t_runtime.source_arena_gather_adjoint(
                output_adjoint.contiguous(),
                gather_indices,
                reverse_offsets,
                reverse_output_indices,
                center_types,
                atom_types,
            ),
            None,
            None,
            None,
            None,
            None,
        )

    def source_arena_gather_adjoint_setup(ctx, inputs, output):
        (
            output_adjoint,
            gather_indices,
            reverse_offsets,
            reverse_output_indices,
            center_types,
            atom_types,
        ) = inputs
        del output_adjoint, output
        ctx.save_for_backward(
            gather_indices,
            reverse_offsets,
            reverse_output_indices,
            center_types,
            atom_types,
        )

    def source_arena_gather_adjoint_backward(
        ctx,
        producer_adjoint_tangent,
    ):
        (
            gather_indices,
            reverse_offsets,
            reverse_output_indices,
            center_types,
            atom_types,
        ) = ctx.saved_tensors
        return (
            torch.ops.ye3t_runtime.source_arena_gather_double_backward(
                producer_adjoint_tangent.contiguous(),
                gather_indices,
                reverse_offsets,
                reverse_output_indices,
                center_types,
                atom_types,
            ),
            None,
            None,
            None,
            None,
            None,
        )

    torch.library.register_autograd(
        "ye3t_runtime::source_arena_gather",
        source_arena_gather_backward,
        setup_context=source_arena_gather_setup,
    )
    torch.library.register_autograd(
        "ye3t_runtime::source_arena_gather_adjoint",
        source_arena_gather_adjoint_backward,
        setup_context=source_arena_gather_adjoint_setup,
    )

    def source_arena_channel_transform_setup(ctx, inputs, output):
        (
            producer,
            channel_maps,
            gather_indices,
            reverse_offsets,
            reverse_output_indices,
            center_types,
            atom_types,
            input_feature_offsets,
            output_feature_offsets,
            input_channel_offsets,
            output_channel_offsets,
            map_offsets,
            output_width,
        ) = inputs
        del output, output_width
        ctx.save_for_backward(
            producer,
            channel_maps,
            gather_indices,
            reverse_offsets,
            reverse_output_indices,
            center_types,
            atom_types,
            input_feature_offsets,
            output_feature_offsets,
            input_channel_offsets,
            output_channel_offsets,
            map_offsets,
        )

    def source_arena_channel_transform_backward(ctx, output_adjoint):
        return (
            *torch.ops.ye3t_runtime.source_arena_channel_transform_adjoint(
                output_adjoint.contiguous(),
                *ctx.saved_tensors,
            ),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    def source_arena_channel_transform_adjoint_setup(
        ctx,
        inputs,
        output,
    ):
        del output
        ctx.save_for_backward(*inputs)

    def source_arena_channel_transform_adjoint_backward(
        ctx,
        producer_adjoint_tangent,
        channel_maps_adjoint_tangent,
    ):
        (
            output_adjoint,
            producer,
            channel_maps,
            gather_indices,
            reverse_offsets,
            reverse_output_indices,
            center_types,
            atom_types,
            input_feature_offsets,
            output_feature_offsets,
            input_channel_offsets,
            output_channel_offsets,
            map_offsets,
        ) = ctx.saved_tensors
        if producer_adjoint_tangent is None:
            producer_adjoint_tangent = torch.zeros_like(producer)
        if channel_maps_adjoint_tangent is None:
            channel_maps_adjoint_tangent = torch.zeros_like(channel_maps)
        return (
            *torch.ops.ye3t_runtime
            .source_arena_channel_transform_double_backward(
                output_adjoint,
                producer,
                channel_maps,
                producer_adjoint_tangent.contiguous(),
                channel_maps_adjoint_tangent.contiguous(),
                gather_indices,
                reverse_offsets,
                reverse_output_indices,
                center_types,
                atom_types,
                input_feature_offsets,
                output_feature_offsets,
                input_channel_offsets,
                output_channel_offsets,
                map_offsets,
            ),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    torch.library.register_autograd(
        "ye3t_runtime::source_arena_channel_transform",
        source_arena_channel_transform_backward,
        setup_context=source_arena_channel_transform_setup,
    )
    torch.library.register_autograd(
        "ye3t_runtime::source_arena_channel_transform_adjoint",
        source_arena_channel_transform_adjoint_backward,
        setup_context=source_arena_channel_transform_adjoint_setup,
    )

    def carrier_channel_update_setup(ctx, inputs, output):
        (
            values,
            gates,
            channel_maps,
            feature_offsets,
            channel_offsets,
            map_offsets,
        ) = inputs
        del output
        ctx.save_for_backward(
            values,
            gates,
            channel_maps,
            feature_offsets,
            channel_offsets,
            map_offsets,
        )

    def carrier_channel_update_backward(ctx, output_adjoint):
        (
            values,
            gates,
            channel_maps,
            feature_offsets,
            channel_offsets,
            map_offsets,
        ) = ctx.saved_tensors
        return (
            *torch.ops.ye3t_runtime.carrier_channel_update_adjoint(
                output_adjoint.contiguous(),
                values,
                gates,
                channel_maps,
                feature_offsets,
                channel_offsets,
                map_offsets,
            ),
            None,
            None,
            None,
        )

    def carrier_channel_update_adjoint_setup(ctx, inputs, output):
        (
            output_adjoint,
            values,
            gates,
            channel_maps,
            feature_offsets,
            channel_offsets,
            map_offsets,
        ) = inputs
        del output
        ctx.save_for_backward(
            output_adjoint,
            values,
            gates,
            channel_maps,
            feature_offsets,
            channel_offsets,
            map_offsets,
        )

    def carrier_channel_update_adjoint_backward(
        ctx,
        values_adjoint_tangent,
        gates_adjoint_tangent,
        channel_maps_adjoint_tangent,
    ):
        (
            output_adjoint,
            values,
            gates,
            channel_maps,
            feature_offsets,
            channel_offsets,
            map_offsets,
        ) = ctx.saved_tensors
        if values_adjoint_tangent is None:
            values_adjoint_tangent = torch.zeros_like(values)
        if gates_adjoint_tangent is None:
            gates_adjoint_tangent = torch.zeros_like(gates)
        if channel_maps_adjoint_tangent is None:
            channel_maps_adjoint_tangent = torch.zeros_like(channel_maps)
        return (
            *torch.ops.ye3t_runtime.carrier_channel_update_double_backward(
                output_adjoint,
                values,
                gates,
                channel_maps,
                values_adjoint_tangent.contiguous(),
                gates_adjoint_tangent.contiguous(),
                channel_maps_adjoint_tangent.contiguous(),
                feature_offsets,
                channel_offsets,
                map_offsets,
            ),
            None,
            None,
            None,
        )

    torch.library.register_autograd(
        "ye3t_runtime::carrier_channel_update",
        carrier_channel_update_backward,
        setup_context=carrier_channel_update_setup,
    )
    torch.library.register_autograd(
        "ye3t_runtime::carrier_channel_update_adjoint",
        carrier_channel_update_adjoint_backward,
        setup_context=carrier_channel_update_adjoint_setup,
    )

    def carrier_channel_transform_setup(ctx, inputs, output):
        (
            values,
            channel_maps,
            input_feature_offsets,
            output_feature_offsets,
            input_channel_offsets,
            output_channel_offsets,
            map_offsets,
            output_width,
        ) = inputs
        del output, output_width
        ctx.save_for_backward(
            values,
            channel_maps,
            input_feature_offsets,
            output_feature_offsets,
            input_channel_offsets,
            output_channel_offsets,
            map_offsets,
        )

    def carrier_channel_transform_backward(ctx, output_adjoint):
        saved = ctx.saved_tensors
        return (
            *torch.ops.ye3t_runtime.carrier_channel_transform_adjoint(
                output_adjoint.contiguous(),
                *saved,
            ),
            None,
            None,
            None,
            None,
            None,
            None,
        )

    def carrier_channel_transform_adjoint_setup(ctx, inputs, output):
        (
            output_adjoint,
            values,
            channel_maps,
            input_feature_offsets,
            output_feature_offsets,
            input_channel_offsets,
            output_channel_offsets,
            map_offsets,
        ) = inputs
        del output
        ctx.save_for_backward(
            output_adjoint,
            values,
            channel_maps,
            input_feature_offsets,
            output_feature_offsets,
            input_channel_offsets,
            output_channel_offsets,
            map_offsets,
        )

    def carrier_channel_transform_adjoint_backward(
        ctx,
        values_adjoint_tangent,
        channel_maps_adjoint_tangent,
    ):
        (
            output_adjoint,
            values,
            channel_maps,
            input_feature_offsets,
            output_feature_offsets,
            input_channel_offsets,
            output_channel_offsets,
            map_offsets,
        ) = ctx.saved_tensors
        if values_adjoint_tangent is None:
            values_adjoint_tangent = torch.zeros_like(values)
        if channel_maps_adjoint_tangent is None:
            channel_maps_adjoint_tangent = torch.zeros_like(channel_maps)
        return (
            *torch.ops.ye3t_runtime.carrier_channel_transform_double_backward(
                output_adjoint,
                values,
                channel_maps,
                values_adjoint_tangent.contiguous(),
                channel_maps_adjoint_tangent.contiguous(),
                input_feature_offsets,
                output_feature_offsets,
                input_channel_offsets,
                output_channel_offsets,
                map_offsets,
            ),
            None,
            None,
            None,
            None,
            None,
        )

    torch.library.register_autograd(
        "ye3t_runtime::carrier_channel_transform",
        carrier_channel_transform_backward,
        setup_context=carrier_channel_transform_setup,
    )
    torch.library.register_autograd(
        "ye3t_runtime::carrier_channel_transform_adjoint",
        carrier_channel_transform_adjoint_backward,
        setup_context=carrier_channel_transform_adjoint_setup,
    )

    def carrier_role_channel_map_adjoint_setup(ctx, inputs, output):
        del output
        ctx.save_for_backward(*inputs)

    def carrier_role_channel_map_adjoint_backward(
        ctx,
        channel_maps_adjoint_tangent,
    ):
        (
            edge_values,
            role_weights,
            atomic_output_adjoint,
            atom_centers,
            channel_maps,
            input_feature_offsets,
            output_feature_offsets,
            input_channel_offsets,
            output_channel_offsets,
            map_offsets,
        ) = ctx.saved_tensors
        if channel_maps_adjoint_tangent is None:
            channel_maps_adjoint_tangent = torch.zeros_like(channel_maps)
        (
            edge_values_second,
            role_weights_second,
            atomic_output_tangent,
        ) = torch.ops.ye3t_runtime.carrier_role_channel_map_adjoint_double_backward(
            edge_values,
            role_weights,
            atomic_output_adjoint,
            atom_centers,
            channel_maps,
            channel_maps_adjoint_tangent.contiguous(),
            input_feature_offsets,
            output_feature_offsets,
            input_channel_offsets,
            output_channel_offsets,
            map_offsets,
        )
        return (
            edge_values_second,
            role_weights_second,
            atomic_output_tangent,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    torch.library.register_autograd(
        "ye3t_runtime::carrier_role_channel_map_adjoint",
        carrier_role_channel_map_adjoint_backward,
        setup_context=carrier_role_channel_map_adjoint_setup,
    )

    def carrier_role_channel_map_adjoint_double_setup(
        ctx,
        inputs,
        output,
    ):
        del output
        ctx.save_for_backward(*inputs)

    def carrier_role_channel_map_adjoint_double_backward(
        ctx,
        edge_values_second_tangent,
        role_weights_second_tangent,
        atomic_output_tangent_tangent,
    ):
        (
            edge_values,
            role_weights,
            atomic_output_adjoint,
            atom_centers,
            channel_maps,
            channel_maps_adjoint_tangent,
            input_feature_offsets,
            output_feature_offsets,
            input_channel_offsets,
            output_channel_offsets,
            map_offsets,
        ) = ctx.saved_tensors
        with torch.enable_grad():
            tracked_edge_values = independent_autograd_input(edge_values)
            tracked_role_weights = independent_autograd_input(role_weights)
            tracked_atomic_output = independent_autograd_input(
                atomic_output_adjoint
            )
            tracked_map_tangent = independent_autograd_input(
                channel_maps_adjoint_tangent
            )
            map_adjoint = _carrier_role_channel_map_adjoint_reference(
                tracked_edge_values,
                tracked_role_weights,
                tracked_atomic_output,
                atom_centers,
                channel_maps,
                (
                    input_feature_offsets,
                    output_feature_offsets,
                    input_channel_offsets,
                    output_channel_offsets,
                    map_offsets,
                ),
            )
            first_gradients = torch.autograd.grad(
                map_adjoint,
                (
                    tracked_edge_values,
                    tracked_role_weights,
                    tracked_atomic_output,
                ),
                tracked_map_tangent,
                create_graph=True,
            )
            incoming = (
                edge_values_second_tangent,
                role_weights_second_tangent,
                atomic_output_tangent_tangent,
            )
            incoming = tuple(
                torch.zeros_like(value) if tangent is None else tangent
                for tangent, value in zip(incoming, first_gradients)
            )
            higher = torch.autograd.grad(
                first_gradients,
                (
                    tracked_edge_values,
                    tracked_role_weights,
                    tracked_atomic_output,
                    tracked_map_tangent,
                ),
                incoming,
                create_graph=torch.is_grad_enabled(),
                allow_unused=True,
            )
        return (
            higher[0],
            higher[1],
            higher[2],
            None,
            None,
            higher[3],
            None,
            None,
            None,
            None,
            None,
        )

    torch.library.register_autograd(
        "ye3t_runtime::carrier_role_channel_map_adjoint_double_backward",
        carrier_role_channel_map_adjoint_double_backward,
        setup_context=carrier_role_channel_map_adjoint_double_setup,
    )

    def source_analysis_setup(ctx, inputs, output):
        (
            source,
            assembly_rows,
            assembly_columns,
            assembly_values,
            synthesis_rows,
            synthesis_columns,
            synthesis_values,
            induced_dimension,
            output_dimension,
        ) = inputs
        ctx.save_for_backward(
            assembly_rows,
            assembly_columns,
            assembly_values,
            synthesis_rows,
            synthesis_columns,
            synthesis_values,
        )
        ctx.source_dimension = int(source.shape[1])
        ctx.induced_dimension = int(induced_dimension)

    def source_analysis_backward(context, output_adjoint):
        (
            assembly_rows,
            assembly_columns,
            assembly_values,
            synthesis_rows,
            synthesis_columns,
            synthesis_values,
        ) = context.saved_tensors
        source_adjoint = torch.ops.ye3t_runtime.source_analysis_adjoint(
            output_adjoint.contiguous(),
            assembly_rows,
            assembly_columns,
            assembly_values,
            synthesis_rows,
            synthesis_columns,
            synthesis_values,
            context.source_dimension,
            context.induced_dimension,
        )
        return (
            source_adjoint,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    def adjoint_setup(ctx, inputs, output):
        (
            output_adjoint,
            assembly_rows,
            assembly_columns,
            assembly_values,
            synthesis_rows,
            synthesis_columns,
            synthesis_values,
            source_dimension,
            induced_dimension,
        ) = inputs
        ctx.save_for_backward(
            assembly_rows,
            assembly_columns,
            assembly_values,
            synthesis_rows,
            synthesis_columns,
            synthesis_values,
        )
        ctx.output_dimension = int(output_adjoint.shape[1])
        ctx.induced_dimension = int(induced_dimension)

    def adjoint_backward(context, source_tangent):
        (
            assembly_rows,
            assembly_columns,
            assembly_values,
            synthesis_rows,
            synthesis_columns,
            synthesis_values,
        ) = context.saved_tensors
        output_tangent = torch.ops.ye3t_runtime.source_analysis(
            source_tangent.contiguous(),
            assembly_rows,
            assembly_columns,
            assembly_values,
            synthesis_rows,
            synthesis_columns,
            synthesis_values,
            context.induced_dimension,
            context.output_dimension,
        )
        return (
            output_tangent,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    torch.library.register_autograd(
        "ye3t_runtime::source_analysis",
        source_analysis_backward,
        setup_context=source_analysis_setup,
    )
    torch.library.register_autograd(
        "ye3t_runtime::source_analysis_adjoint",
        adjoint_backward,
        setup_context=adjoint_setup,
    )

    def source_linear_setup(ctx, inputs, output):
        del output
        (
            source,
            assembly_rows,
            assembly_columns,
            assembly_values,
            synthesis_rows,
            synthesis_columns,
            synthesis_values,
            induced_dimension,
            weight,
            bias,
        ) = inputs
        del bias
        ctx.save_for_backward(
            source,
            assembly_rows,
            assembly_columns,
            assembly_values,
            synthesis_rows,
            synthesis_columns,
            synthesis_values,
            weight,
        )
        ctx.induced_dimension = int(induced_dimension)

    def source_linear_backward(ctx, output_adjoint):
        (
            source,
            assembly_rows,
            assembly_columns,
            assembly_values,
            synthesis_rows,
            synthesis_columns,
            synthesis_values,
            weight,
        ) = ctx.saved_tensors
        source_adjoint, weight_adjoint, bias_adjoint = (
            torch.ops.ye3t_runtime.source_analysis_linear_adjoint(
                output_adjoint.contiguous(),
                source,
                assembly_rows,
                assembly_columns,
                assembly_values,
                synthesis_rows,
                synthesis_columns,
                synthesis_values,
                ctx.induced_dimension,
                weight,
            )
        )
        return (
            source_adjoint,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            weight_adjoint,
            bias_adjoint,
        )

    def source_linear_adjoint_setup(ctx, inputs, output):
        del output
        (
            output_adjoint,
            source,
            assembly_rows,
            assembly_columns,
            assembly_values,
            synthesis_rows,
            synthesis_columns,
            synthesis_values,
            induced_dimension,
            weight,
        ) = inputs
        ctx.save_for_backward(
            output_adjoint,
            source,
            assembly_rows,
            assembly_columns,
            assembly_values,
            synthesis_rows,
            synthesis_columns,
            synthesis_values,
            weight,
        )
        ctx.induced_dimension = int(induced_dimension)

    def source_linear_adjoint_backward(
        ctx,
        source_adjoint_tangent,
        weight_adjoint_tangent,
        bias_adjoint_tangent,
    ):
        (
            output_adjoint,
            source,
            assembly_rows,
            assembly_columns,
            assembly_values,
            synthesis_rows,
            synthesis_columns,
            synthesis_values,
            weight,
        ) = ctx.saved_tensors
        with torch.enable_grad():
            tracked_output_adjoint = independent_autograd_input(
                output_adjoint
            )
            tracked_source = independent_autograd_input(source)
            tracked_weight = independent_autograd_input(weight)
            ambient = tracked_source.new_zeros(
                (tracked_source.shape[0], ctx.induced_dimension)
            )
            ambient.index_add_(
                1,
                assembly_rows,
                tracked_source.index_select(1, assembly_columns)
                * assembly_values.reshape(1, -1),
            )
            synthesis = tracked_source.new_zeros(
                (ctx.induced_dimension, tracked_weight.numel())
            )
            synthesis.index_put_(
                (synthesis_rows, synthesis_columns),
                synthesis_values,
                accumulate=True,
            )
            features = ambient @ synthesis.conj()
            output = features @ tracked_weight
            source_adjoint, weight_adjoint = torch.autograd.grad(
                output,
                (tracked_source, tracked_weight),
                tracked_output_adjoint,
                create_graph=True,
            )
            bias_adjoint = tracked_output_adjoint.sum()
            (
                output_tangent,
                source_tangent,
                weight_tangent,
            ) = torch.autograd.grad(
                (source_adjoint, weight_adjoint, bias_adjoint),
                (
                    tracked_output_adjoint,
                    tracked_source,
                    tracked_weight,
                ),
                (
                    source_adjoint_tangent,
                    weight_adjoint_tangent,
                    bias_adjoint_tangent,
                ),
                create_graph=torch.is_grad_enabled(),
                allow_unused=True,
            )
        return (
            output_tangent,
            source_tangent,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            weight_tangent,
        )

    torch.library.register_autograd(
        "ye3t_runtime::source_analysis_linear",
        source_linear_backward,
        setup_context=source_linear_setup,
    )
    torch.library.register_autograd(
        "ye3t_runtime::source_analysis_linear_adjoint",
        source_linear_adjoint_backward,
        setup_context=source_linear_adjoint_setup,
    )

    def compact_setup(ctx, inputs, output):
        left, right, antisymmetric = inputs
        ctx.save_for_backward(left, right)
        ctx.antisymmetric = bool(antisymmetric)

    def compact_backward(ctx, output_adjoint):
        left, right = ctx.saved_tensors
        left_adjoint, right_adjoint = (
            torch.ops.ye3t_runtime.compact_pair_product_adjoint(
                output_adjoint.contiguous(),
                left,
                right,
                ctx.antisymmetric,
            )
        )
        return left_adjoint, right_adjoint, None

    def compact_adjoint_setup(ctx, inputs, output):
        output_adjoint, left, right, antisymmetric = inputs
        ctx.save_for_backward(output_adjoint, left, right)
        ctx.antisymmetric = bool(antisymmetric)

    def compact_adjoint_backward(
        ctx,
        left_adjoint_tangent,
        right_adjoint_tangent,
    ):
        output_adjoint, left, right = ctx.saved_tensors
        with torch.enable_grad():
            tracked_output_adjoint = independent_autograd_input(
                output_adjoint
            )
            tracked_left = independent_autograd_input(left)
            tracked_right = independent_autograd_input(right)
            reference_left, reference_right = (
                compact_pair_product_adjoint_reference(
                    tracked_output_adjoint,
                    tracked_left,
                    tracked_right,
                    antisymmetric=ctx.antisymmetric,
                )
            )
            gradients = torch.autograd.grad(
                (reference_left, reference_right),
                (
                    tracked_output_adjoint,
                    tracked_left,
                    tracked_right,
                ),
                (left_adjoint_tangent, right_adjoint_tangent),
                create_graph=torch.is_grad_enabled(),
                allow_unused=True,
            )
        return gradients[0], gradients[1], gradients[2], None

    torch.library.register_autograd(
        "ye3t_runtime::compact_pair_product",
        compact_backward,
        setup_context=compact_setup,
    )
    torch.library.register_autograd(
        "ye3t_runtime::compact_pair_product_adjoint",
        compact_adjoint_backward,
        setup_context=compact_adjoint_setup,
    )

    def exterior_power_setup(ctx, inputs, output):
        factors, = inputs
        ctx.save_for_backward(factors)

    def exterior_power_backward(ctx, output_adjoint):
        factors, = ctx.saved_tensors
        return torch.ops.ye3t_runtime.compact_exterior_power_adjoint(
            output_adjoint.contiguous(),
            factors,
        )

    def exterior_power_adjoint_setup(ctx, inputs, output):
        output_adjoint, factors = inputs
        del output
        ctx.save_for_backward(output_adjoint, factors)

    def exterior_power_adjoint_backward(
        ctx,
        factors_adjoint_tangent,
    ):
        output_adjoint, factors = ctx.saved_tensors
        with torch.enable_grad():
            tracked_output_adjoint = independent_autograd_input(
                output_adjoint
            )
            tracked_factors = independent_autograd_input(factors)
            reference = compact_exterior_power_adjoint_reference(
                tracked_output_adjoint,
                tracked_factors,
            )
            gradients = torch.autograd.grad(
                reference,
                (tracked_output_adjoint, tracked_factors),
                factors_adjoint_tangent,
                create_graph=torch.is_grad_enabled(),
                allow_unused=True,
            )
        return gradients

    torch.library.register_autograd(
        "ye3t_runtime::compact_exterior_power",
        exterior_power_backward,
        setup_context=exterior_power_setup,
    )
    torch.library.register_autograd(
        "ye3t_runtime::compact_exterior_power_adjoint",
        exterior_power_adjoint_backward,
        setup_context=exterior_power_adjoint_setup,
    )

    def symmetric_power_setup(ctx, inputs, output):
        (
            input,
            monomial_counts,
            output_offsets,
            output_indices,
            monomial_values,
        ) = inputs
        del output
        ctx.save_for_backward(
            input,
            monomial_counts,
            output_offsets,
            output_indices,
            monomial_values,
        )

    def symmetric_power_backward(ctx, output_adjoint):
        (
            input,
            monomial_counts,
            output_offsets,
            output_indices,
            monomial_values,
        ) = ctx.saved_tensors
        input_adjoint = (
            torch.ops.ye3t_runtime.symmetric_power_monomial_adjoint(
                output_adjoint.contiguous(),
                input,
                monomial_counts,
                output_offsets,
                output_indices,
                monomial_values,
            )
        )
        return input_adjoint, None, None, None, None

    def symmetric_power_adjoint_setup(ctx, inputs, output):
        (
            output_adjoint,
            input,
            monomial_counts,
            output_offsets,
            output_indices,
            monomial_values,
        ) = inputs
        del output
        ctx.save_for_backward(
            output_adjoint,
            input,
            monomial_counts,
            output_offsets,
            output_indices,
            monomial_values,
        )

    def symmetric_power_adjoint_backward(
        ctx,
        input_adjoint_tangent,
    ):
        (
            output_adjoint,
            input,
            monomial_counts,
            output_offsets,
            output_indices,
            monomial_values,
        ) = ctx.saved_tensors
        _policy, native_double_backward = (
            _resolve_symmetric_power_double_backward_policy(input)
        )
        if native_double_backward:
            output_tangent, input_tangent = (
                torch.ops.ye3t_runtime
                .symmetric_power_monomial_double_backward(
                    input_adjoint_tangent.contiguous(),
                    output_adjoint.contiguous(),
                    input.contiguous(),
                    monomial_counts,
                    output_offsets,
                    output_indices,
                    monomial_values,
                )
            )
            return (
                output_tangent,
                input_tangent,
                None,
                None,
                None,
                None,
            )
        with torch.enable_grad():
            tracked_output_adjoint = independent_autograd_input(
                output_adjoint
            )
            tracked_input = independent_autograd_input(input)
            reference = symmetric_power_monomial_adjoint_reference(
                tracked_output_adjoint,
                tracked_input,
                monomial_counts,
                output_offsets,
                output_indices,
                monomial_values,
            )
            gradients = torch.autograd.grad(
                reference,
                (tracked_output_adjoint, tracked_input),
                input_adjoint_tangent,
                create_graph=torch.is_grad_enabled(),
                allow_unused=True,
            )
        return gradients[0], gradients[1], None, None, None, None

    torch.library.register_autograd(
        "ye3t_runtime::symmetric_power_monomial",
        symmetric_power_backward,
        setup_context=symmetric_power_setup,
    )
    torch.library.register_autograd(
        "ye3t_runtime::symmetric_power_monomial_adjoint",
        symmetric_power_adjoint_backward,
        setup_context=symmetric_power_adjoint_setup,
    )

    def symmetric_power_shared_setup(ctx, inputs, output):
        (
            input,
            monomial_counts,
            output_offsets,
            coefficient_terms,
            coefficient_outputs,
            coefficient_values,
        ) = inputs
        del output
        ctx.save_for_backward(
            input,
            monomial_counts,
            output_offsets,
            coefficient_terms,
            coefficient_outputs,
            coefficient_values,
        )

    def symmetric_power_shared_backward(ctx, output_adjoint):
        (
            input,
            monomial_counts,
            output_offsets,
            coefficient_terms,
            coefficient_outputs,
            coefficient_values,
        ) = ctx.saved_tensors
        input_adjoint = (
            torch.ops.ye3t_runtime.symmetric_power_shared_monomial_adjoint(
                output_adjoint.contiguous(),
                input,
                monomial_counts,
                output_offsets,
                coefficient_terms,
                coefficient_outputs,
                coefficient_values,
            )
        )
        return input_adjoint, None, None, None, None, None

    def symmetric_power_shared_adjoint_setup(ctx, inputs, output):
        (
            output_adjoint,
            input,
            monomial_counts,
            output_offsets,
            coefficient_terms,
            coefficient_outputs,
            coefficient_values,
        ) = inputs
        del output
        ctx.save_for_backward(
            output_adjoint,
            input,
            monomial_counts,
            output_offsets,
            coefficient_terms,
            coefficient_outputs,
            coefficient_values,
        )

    def symmetric_power_shared_adjoint_backward(
        ctx,
        input_adjoint_tangent,
    ):
        (
            output_adjoint,
            input,
            monomial_counts,
            output_offsets,
            coefficient_terms,
            coefficient_outputs,
            coefficient_values,
        ) = ctx.saved_tensors
        _policy, native_double_backward = (
            _resolve_symmetric_power_double_backward_policy(
                input,
                "symmetric_power_shared_monomial_double_backward",
            )
        )
        if native_double_backward:
            (output_tangent, input_tangent), _algorithm = (
                _symmetric_power_shared_double_backward(
                    input_adjoint_tangent,
                    output_adjoint,
                    input,
                    monomial_counts,
                    output_offsets,
                    coefficient_terms,
                    coefficient_outputs,
                    coefficient_values,
                )
            )
            return (
                output_tangent,
                input_tangent,
                None,
                None,
                None,
                None,
                None,
            )
        with torch.enable_grad():
            tracked_output_adjoint = independent_autograd_input(
                output_adjoint
            )
            tracked_input = independent_autograd_input(input)
            reference = symmetric_power_shared_monomial_adjoint_reference(
                tracked_output_adjoint,
                tracked_input,
                monomial_counts,
                output_offsets,
                coefficient_terms,
                coefficient_outputs,
                coefficient_values,
            )
            gradients = torch.autograd.grad(
                reference,
                (tracked_output_adjoint, tracked_input),
                input_adjoint_tangent,
                create_graph=torch.is_grad_enabled(),
                allow_unused=True,
            )
        return (
            gradients[0],
            gradients[1],
            None,
            None,
            None,
            None,
            None,
        )

    torch.library.register_autograd(
        "ye3t_runtime::symmetric_power_shared_monomial",
        symmetric_power_shared_backward,
        setup_context=symmetric_power_shared_setup,
    )
    torch.library.register_autograd(
        "ye3t_runtime::symmetric_power_shared_monomial_adjoint",
        symmetric_power_shared_adjoint_backward,
        setup_context=symmetric_power_shared_adjoint_setup,
    )

    def symmetric_power_shared_sparse_setup(ctx, inputs, output):
        (
            input,
            term_offsets,
            term_components,
            term_exponents,
            output_offsets,
            coefficient_terms,
            coefficient_outputs,
            coefficient_values,
        ) = inputs
        del output
        ctx.save_for_backward(
            input,
            term_offsets,
            term_components,
            term_exponents,
            output_offsets,
            coefficient_terms,
            coefficient_outputs,
            coefficient_values,
        )

    def symmetric_power_shared_sparse_backward(ctx, output_adjoint):
        (
            input,
            term_offsets,
            term_components,
            term_exponents,
            output_offsets,
            coefficient_terms,
            coefficient_outputs,
            coefficient_values,
        ) = ctx.saved_tensors
        input_adjoint = (
            torch.ops.ye3t_runtime
            .symmetric_power_shared_sparse_monomial_adjoint(
                output_adjoint.contiguous(),
                input,
                term_offsets,
                term_components,
                term_exponents,
                output_offsets,
                coefficient_terms,
                coefficient_outputs,
                coefficient_values,
            )
        )
        return input_adjoint, None, None, None, None, None, None, None

    def symmetric_power_shared_sparse_adjoint_setup(
        ctx,
        inputs,
        output,
    ):
        (
            output_adjoint,
            input,
            term_offsets,
            term_components,
            term_exponents,
            output_offsets,
            coefficient_terms,
            coefficient_outputs,
            coefficient_values,
        ) = inputs
        del output
        ctx.save_for_backward(
            output_adjoint,
            input,
            term_offsets,
            term_components,
            term_exponents,
            output_offsets,
            coefficient_terms,
            coefficient_outputs,
            coefficient_values,
        )

    def symmetric_power_shared_sparse_adjoint_backward(
        ctx,
        input_adjoint_tangent,
    ):
        (
            output_adjoint,
            input,
            term_offsets,
            term_components,
            term_exponents,
            output_offsets,
            coefficient_terms,
            coefficient_outputs,
            coefficient_values,
        ) = ctx.saved_tensors
        output_tangent, input_tangent = (
            torch.ops.ye3t_runtime
            .symmetric_power_shared_sparse_monomial_double_backward(
                input_adjoint_tangent.contiguous(),
                output_adjoint.contiguous(),
                input.contiguous(),
                term_offsets,
                term_components,
                term_exponents,
                output_offsets,
                coefficient_terms,
                coefficient_outputs,
                coefficient_values,
            )
        )
        return (
            output_tangent,
            input_tangent,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    if hasattr(
        torch.ops.ye3t_runtime,
        "symmetric_power_shared_sparse_monomial",
    ):
        torch.library.register_autograd(
            "ye3t_runtime::symmetric_power_shared_sparse_monomial",
            symmetric_power_shared_sparse_backward,
            setup_context=symmetric_power_shared_sparse_setup,
        )
        torch.library.register_autograd(
            "ye3t_runtime::symmetric_power_shared_sparse_monomial_adjoint",
            symmetric_power_shared_sparse_adjoint_backward,
            setup_context=symmetric_power_shared_sparse_adjoint_setup,
        )

    def symmetric_power_shared_batched_adjoint_setup(
        ctx,
        inputs,
        output,
    ):
        (
            output_adjoint,
            input,
            monomial_counts,
            output_offsets,
            coefficient_terms,
            coefficient_outputs,
            coefficient_values,
        ) = inputs
        del output
        ctx.save_for_backward(
            output_adjoint,
            input,
            monomial_counts,
            output_offsets,
            coefficient_terms,
            coefficient_outputs,
            coefficient_values,
        )

    def symmetric_power_shared_batched_adjoint_backward(
        ctx,
        input_adjoint_tangent,
    ):
        (
            output_adjoint,
            input,
            monomial_counts,
            output_offsets,
            coefficient_terms,
            coefficient_outputs,
            coefficient_values,
        ) = ctx.saved_tensors
        with torch.enable_grad():
            needs_output = ctx.needs_input_grad[0]
            needs_input = ctx.needs_input_grad[1]
            if not needs_output and not needs_input:
                return (None, None, None, None, None, None, None)

            detached_output = output_adjoint.detach().requires_grad_(True)
            input_branch_reference = (
                symmetric_power_shared_monomial_batched_adjoint_reference(
                    detached_output,
                    input,
                    monomial_counts,
                    output_offsets,
                    coefficient_terms,
                    coefficient_outputs,
                    coefficient_values,
                )
            )
            input_branch_targets = [detached_output]
            if needs_input:
                input_branch_targets.append(input)
            input_branch_gradients = torch.autograd.grad(
                input_branch_reference,
                tuple(input_branch_targets),
                input_adjoint_tangent,
                create_graph=torch.is_grad_enabled(),
                allow_unused=True,
            )
            output_gradient = (
                input_branch_gradients[0] if needs_output else None
            )
            input_gradient = (
                input_branch_gradients[1] if needs_input else None
            )

            if needs_output and needs_input:
                detached_input = input.detach().requires_grad_(True)
                output_branch_reference = (
                    symmetric_power_shared_monomial_batched_adjoint_reference(
                        output_adjoint,
                        detached_input,
                        monomial_counts,
                        output_offsets,
                        coefficient_terms,
                        coefficient_outputs,
                        coefficient_values,
                    )
                )
                output_branch_gradients = torch.autograd.grad(
                    output_branch_reference,
                    (output_adjoint, detached_input),
                    input_adjoint_tangent.detach(),
                    create_graph=torch.is_grad_enabled(),
                    allow_unused=True,
                )
                output_gradient = output_gradient + (
                    output_branch_gradients[0]
                    - output_branch_gradients[0].detach()
                )
                input_gradient = input_gradient + (
                    output_branch_gradients[1]
                    - output_branch_gradients[1].detach()
                )
        return (
            output_gradient,
            input_gradient,
            None,
            None,
            None,
            None,
            None,
        )

    torch.library.register_autograd(
        "ye3t_runtime::symmetric_power_shared_monomial_batched_adjoint",
        symmetric_power_shared_batched_adjoint_backward,
        setup_context=symmetric_power_shared_batched_adjoint_setup,
    )

    def factorized_setup(ctx, inputs, output):
        (
            packed_slots,
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
            source_dimension,
            workspace_dimension,
            output_dimension,
        ) = inputs
        ctx.save_for_backward(
            packed_slots,
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
        ctx.source_dimension = int(source_dimension)
        ctx.workspace_dimension = int(workspace_dimension)

    def factorized_backward(ctx, output_adjoint):
        saved = ctx.saved_tensors
        packed_adjoint = torch.ops.ye3t_runtime.factorized_angular_adjoint(
            output_adjoint.contiguous(),
            *saved,
            ctx.source_dimension,
            ctx.workspace_dimension,
        )
        return (packed_adjoint,) + (None,) * 14

    def factorized_adjoint_setup(ctx, inputs, output):
        ctx.save_for_backward(*inputs[:-2])
        ctx.source_dimension = int(inputs[-2])
        ctx.workspace_dimension = int(inputs[-1])

    def factorized_adjoint_backward(ctx, packed_adjoint_tangent):
        (
            output_adjoint,
            packed_slots,
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
        ) = ctx.saved_tensors
        if hasattr(
            torch.ops.ye3t_runtime,
            "factorized_angular_double_backward",
        ):
            output_tangent, packed_tangent = (
                torch.ops.ye3t_runtime.factorized_angular_double_backward(
                    packed_adjoint_tangent.contiguous(),
                    output_adjoint.contiguous(),
                    packed_slots.contiguous(),
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
                    ctx.source_dimension,
                    ctx.workspace_dimension,
                )
            )
            return (
                output_tangent,
                packed_tangent,
            ) + (None,) * 13
        with torch.enable_grad():
            tracked_output_adjoint = independent_autograd_input(
                output_adjoint
            )
            tracked_packed_slots = independent_autograd_input(
                packed_slots
            )
            reference_output = _factorized_angular_packed_reference(
                tracked_packed_slots,
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
                ctx.source_dimension,
            )
            reference_adjoint = torch.autograd.grad(
                reference_output,
                tracked_packed_slots,
                tracked_output_adjoint,
                create_graph=True,
            )[0]
            output_tangent, packed_tangent = torch.autograd.grad(
                reference_adjoint,
                (
                    tracked_output_adjoint,
                    tracked_packed_slots,
                ),
                packed_adjoint_tangent,
                create_graph=torch.is_grad_enabled(),
                allow_unused=True,
            )
        return (
            output_tangent,
            packed_tangent,
        ) + (None,) * 13

    torch.library.register_autograd(
        "ye3t_runtime::factorized_angular",
        factorized_backward,
        setup_context=factorized_setup,
    )
    torch.library.register_autograd(
        "ye3t_runtime::factorized_angular_adjoint",
        factorized_adjoint_backward,
        setup_context=factorized_adjoint_setup,
    )

    def heterogeneous_factorized_setup(ctx, inputs, output):
        (
            packed_slots,
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
            root_projection_starts,
            root_projection_dimensions,
            root_output_offsets,
            projection_values,
            source_dimension,
            workspace_dimension,
            output_dimension,
        ) = inputs
        ctx.save_for_backward(
            packed_slots,
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
            root_projection_starts,
            root_projection_dimensions,
            root_output_offsets,
            projection_values,
        )
        ctx.source_dimension = int(source_dimension)
        ctx.workspace_dimension = int(workspace_dimension)

    def heterogeneous_factorized_backward(ctx, output_adjoint):
        if output_adjoint.is_cuda:
            packed_adjoint, _, _ = (
                torch.ops.ye3t_runtime
                .factorized_angular_heterogeneous_adjoint_with_workspace(
                    output_adjoint.contiguous(),
                    *ctx.saved_tensors,
                    ctx.source_dimension,
                    ctx.workspace_dimension,
                )
            )
        else:
            packed_adjoint = (
                torch.ops.ye3t_runtime
                .factorized_angular_heterogeneous_adjoint(
                    output_adjoint.contiguous(),
                    *ctx.saved_tensors,
                    ctx.source_dimension,
                    ctx.workspace_dimension,
                )
            )
        return (packed_adjoint,) + (None,) * 17

    def heterogeneous_factorized_adjoint_setup(ctx, inputs, output):
        ctx.save_for_backward(*inputs[:-2])
        ctx.source_dimension = int(inputs[-2])
        ctx.workspace_dimension = int(inputs[-1])

    def heterogeneous_factorized_adjoint_backward(
        ctx,
        packed_adjoint_tangent,
    ):
        (
            output_adjoint,
            packed_slots,
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
            root_projection_starts,
            root_projection_dimensions,
            root_output_offsets,
            projection_values,
        ) = ctx.saved_tensors
        output_tangent, packed_tangent = (
            torch.ops.ye3t_runtime
            .factorized_angular_heterogeneous_double_backward(
                packed_adjoint_tangent.contiguous(),
                output_adjoint.contiguous(),
                packed_slots.contiguous(),
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
                root_projection_starts,
                root_projection_dimensions,
                root_output_offsets,
                projection_values,
                ctx.source_dimension,
                ctx.workspace_dimension,
            )
        )
        return (
            output_tangent,
            packed_tangent,
        ) + (None,) * 16

    def heterogeneous_factorized_cached_adjoint_setup(ctx, inputs, output):
        output_adjoint = inputs[0]
        workspace = output[1]
        workspace_adjoint = output[2]
        ctx.mark_non_differentiable(workspace, workspace_adjoint)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(
            output_adjoint,
            *inputs[2:-2],
            workspace,
            workspace_adjoint,
        )
        ctx.source_dimension = int(inputs[-2])
        ctx.workspace_dimension = int(inputs[-1])

    def heterogeneous_factorized_cached_adjoint_backward(
        ctx,
        packed_adjoint_tangent,
        workspace_tangent,
        workspace_adjoint_tangent,
    ):
        del workspace_tangent, workspace_adjoint_tangent
        (
            output_adjoint,
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
            root_projection_starts,
            root_projection_dimensions,
            root_output_offsets,
            projection_values,
            workspace,
            workspace_adjoint,
        ) = ctx.saved_tensors
        output_tangent, packed_tangent = (
            torch.ops.ye3t_runtime
            .factorized_angular_heterogeneous_double_backward_from_workspace(
                packed_adjoint_tangent.contiguous(),
                output_adjoint,
                workspace,
                workspace_adjoint,
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
                root_projection_starts,
                root_projection_dimensions,
                root_output_offsets,
                projection_values,
                ctx.source_dimension,
                ctx.workspace_dimension,
            )
        )
        return (
            output_tangent,
            packed_tangent,
        ) + (None,) * 16

    torch.library.register_autograd(
        "ye3t_runtime::factorized_angular_heterogeneous",
        heterogeneous_factorized_backward,
        setup_context=heterogeneous_factorized_setup,
    )
    torch.library.register_autograd(
        "ye3t_runtime::factorized_angular_heterogeneous_adjoint",
        heterogeneous_factorized_adjoint_backward,
        setup_context=heterogeneous_factorized_adjoint_setup,
    )
    torch.library.register_autograd(
        "ye3t_runtime::factorized_angular_heterogeneous_adjoint_with_workspace",
        heterogeneous_factorized_cached_adjoint_backward,
        setup_context=heterogeneous_factorized_cached_adjoint_setup,
    )

    def segmented_factorized_setup(ctx, inputs, output):
        del output
        ctx.save_for_backward(*inputs[:-3])
        ctx.total_source_count = int(inputs[-3])
        ctx.workspace_dimension = int(inputs[-2])

    def segmented_factorized_backward(ctx, output_adjoint):
        packed_adjoint = (
            torch.ops.ye3t_runtime.factorized_angular_segmented_adjoint(
                output_adjoint.contiguous(),
                *ctx.saved_tensors,
                ctx.total_source_count,
                ctx.workspace_dimension,
            )
        )
        return (packed_adjoint,) + (None,) * 27

    def segmented_factorized_adjoint_setup(ctx, inputs, output):
        del output
        ctx.save_for_backward(*inputs[:-2])
        ctx.total_source_count = int(inputs[-2])
        ctx.workspace_dimension = int(inputs[-1])

    def segmented_factorized_adjoint_backward(
        ctx,
        packed_adjoint_tangent,
    ):
        output_tangent, packed_tangent = (
            torch.ops.ye3t_runtime
            .factorized_angular_segmented_double_backward(
                packed_adjoint_tangent.contiguous(),
                *ctx.saved_tensors,
                ctx.total_source_count,
                ctx.workspace_dimension,
            )
        )
        return (output_tangent, packed_tangent) + (None,) * 26

    torch.library.register_autograd(
        "ye3t_runtime::factorized_angular_segmented",
        segmented_factorized_backward,
        setup_context=segmented_factorized_setup,
    )
    torch.library.register_autograd(
        "ye3t_runtime::factorized_angular_segmented_adjoint",
        segmented_factorized_adjoint_backward,
        setup_context=segmented_factorized_adjoint_setup,
    )

    def factorized_linear_setup(ctx, inputs, output):
        del output
        (
            packed_slots,
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
            source_dimension,
            workspace_dimension,
            weight,
            bias,
        ) = inputs
        del bias
        ctx.save_for_backward(
            packed_slots,
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
            weight,
        )
        ctx.source_dimension = int(source_dimension)
        ctx.workspace_dimension = int(workspace_dimension)

    def factorized_linear_backward(ctx, output_adjoint):
        (
            packed_slots,
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
            weight,
        ) = ctx.saved_tensors
        packed_adjoint, weight_adjoint, bias_adjoint = (
            torch.ops.ye3t_runtime.factorized_angular_linear_adjoint(
                output_adjoint.contiguous(),
                packed_slots,
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
                ctx.source_dimension,
                ctx.workspace_dimension,
                weight,
            )
        )
        return (
            packed_adjoint,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            weight_adjoint,
            bias_adjoint,
        )

    def factorized_linear_adjoint_setup(ctx, inputs, output):
        del output
        (
            output_adjoint,
            packed_slots,
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
            source_dimension,
            workspace_dimension,
            weight,
        ) = inputs
        ctx.save_for_backward(
            output_adjoint,
            packed_slots,
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
            weight,
        )
        ctx.source_dimension = int(source_dimension)
        ctx.workspace_dimension = int(workspace_dimension)

    def factorized_linear_adjoint_backward(
        ctx,
        packed_adjoint_tangent,
        weight_adjoint_tangent,
        bias_adjoint_tangent,
    ):
        (
            output_adjoint,
            packed_slots,
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
            weight,
        ) = ctx.saved_tensors
        with torch.enable_grad():
            tracked_output_adjoint = independent_autograd_input(
                output_adjoint
            )
            tracked_packed_slots = independent_autograd_input(
                packed_slots
            )
            tracked_weight = independent_autograd_input(weight)
            features = _factorized_angular_packed_reference(
                tracked_packed_slots,
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
                ctx.source_dimension,
            )
            output = features @ tracked_weight
            packed_adjoint, weight_adjoint = torch.autograd.grad(
                output,
                (tracked_packed_slots, tracked_weight),
                tracked_output_adjoint,
                create_graph=True,
            )
            bias_adjoint = tracked_output_adjoint.sum()
            (
                output_tangent,
                packed_tangent,
                weight_tangent,
            ) = torch.autograd.grad(
                (packed_adjoint, weight_adjoint, bias_adjoint),
                (
                    tracked_output_adjoint,
                    tracked_packed_slots,
                    tracked_weight,
                ),
                (
                    packed_adjoint_tangent,
                    weight_adjoint_tangent,
                    bias_adjoint_tangent,
                ),
                create_graph=torch.is_grad_enabled(),
                allow_unused=True,
            )
        return (
            output_tangent,
            packed_tangent,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            weight_tangent,
        )

    torch.library.register_autograd(
        "ye3t_runtime::factorized_angular_linear",
        factorized_linear_backward,
        setup_context=factorized_linear_setup,
    )
    torch.library.register_autograd(
        "ye3t_runtime::factorized_angular_linear_adjoint",
        factorized_linear_adjoint_backward,
        setup_context=factorized_linear_adjoint_setup,
    )
    setattr(torch.library, registration_key, True)
    return True


def native_execution_plan_capabilities():
    """Report installed or explicitly requested JIT runtime capabilities."""

    extension = (
        _load_extension()
        if _enabled("YE3T_ENABLE_EXECUTION_PLAN_JIT")
        else _prebuilt_extension()
    )
    triton_available = bool(_triton_joint.triton is not None)
    triton_cuda_available = bool(
        triton_available and torch.cuda.is_available()
    )
    has_cuda = bool(extension is not None and extension.has_cuda())
    core_abi_version = (
        int(extension.core_abi_version())
        if extension is not None
        else None
    )
    has_spherical_table_double_backward = bool(
        has_cuda
        and core_abi_version is not None
        and core_abi_version >= 16
    )
    has_factorized_angular_double_backward = bool(
        extension is not None
        and core_abi_version is not None
        and core_abi_version >= 17
    )
    has_radial_table_double_backward = bool(
        extension is not None
        and core_abi_version is not None
        and core_abi_version >= 18
    )
    has_host_certified_factorized_workspace = bool(
        extension is not None
        and core_abi_version is not None
        and core_abi_version >= 19
    )
    has_heterogeneous_factorized_roots = bool(
        extension is not None
        and core_abi_version is not None
        and core_abi_version >= 20
    )
    has_edge_outer_accumulate = bool(
        extension is not None
        and core_abi_version is not None
        and core_abi_version >= 21
    )
    has_carrier_gated_scatter = bool(
        extension is not None
        and core_abi_version is not None
        and core_abi_version >= 22
    )
    has_carrier_channel_update = bool(
        extension is not None
        and core_abi_version is not None
        and core_abi_version >= 23
    )
    has_scheduled_radial_angular_channels = bool(
        extension is not None
        and core_abi_version is not None
        and core_abi_version >= 24
    )
    has_softmax_gaussian_role_density = bool(
        extension is not None
        and core_abi_version is not None
        and core_abi_version >= 25
    )
    has_cached_heterogeneous_factorized_reverse = bool(
        has_cuda
        and core_abi_version is not None
        and core_abi_version >= 26
    )
    has_real_control_carrier_gates = bool(
        extension is not None
        and core_abi_version is not None
        and core_abi_version >= 27
    )
    has_real_control_carrier_channel_update = bool(
        extension is not None
        and core_abi_version is not None
        and core_abi_version >= 27
    )
    has_scheduled_role_density = bool(
        extension is not None
        and core_abi_version is not None
        and core_abi_version >= 28
    )
    has_symmetric_power_monomial_double_backward = bool(
        has_cuda
        and core_abi_version is not None
        and core_abi_version >= 29
    )
    has_symmetric_power_shared_monomial_double_backward = bool(
        has_cuda
        and core_abi_version is not None
        and core_abi_version >= 33
        and hasattr(
            torch.ops.ye3t_runtime,
            "symmetric_power_shared_monomial_double_backward",
        )
    )
    has_symmetric_power_shared_monomial_factored_double_backward = bool(
        has_cuda
        and core_abi_version is not None
        and core_abi_version >= 38
        and hasattr(
            torch.ops.ye3t_runtime,
            "symmetric_power_shared_monomial_factored_double_backward",
        )
    )
    has_symmetric_power_shared_sparse_monomial = bool(
        has_cuda
        and core_abi_version is not None
        and core_abi_version >= 39
        and all(
            hasattr(torch.ops.ye3t_runtime, name)
            for name in (
                "symmetric_power_shared_sparse_monomial",
                "symmetric_power_shared_sparse_monomial_adjoint",
                "symmetric_power_shared_sparse_monomial_double_backward",
            )
        )
    )
    has_residual_carrier_gated_scatter = bool(
        extension is not None
        and core_abi_version is not None
        and core_abi_version >= 30
    )
    has_segmented_carrier_gated_scatter = bool(
        extension is not None
        and core_abi_version is not None
        and core_abi_version >= 31
    )
    has_segmented_factorized_angular = bool(
        extension is not None
        and core_abi_version is not None
        and core_abi_version >= 32
    )
    has_source_arena_gather = bool(
        extension is not None
        and core_abi_version is not None
        and core_abi_version >= 34
        and hasattr(torch.ops.ye3t_runtime, "source_arena_gather")
        and hasattr(
            torch.ops.ye3t_runtime,
            "source_arena_gather_adjoint",
        )
        and hasattr(
            torch.ops.ye3t_runtime,
            "source_arena_gather_double_backward",
        )
    )
    has_source_arena_channel_transform = bool(
        extension is not None
        and core_abi_version is not None
        and core_abi_version >= 37
        and hasattr(
            torch.ops.ye3t_runtime,
            "source_arena_channel_transform",
        )
        and hasattr(
            torch.ops.ye3t_runtime,
            "source_arena_channel_transform_adjoint",
        )
        and hasattr(
            torch.ops.ye3t_runtime,
            "source_arena_channel_transform_double_backward",
        )
    )
    has_carrier_channel_transform = bool(
        extension is not None
        and core_abi_version is not None
        and core_abi_version >= 35
        and hasattr(torch.ops.ye3t_runtime, "carrier_channel_transform")
        and hasattr(
            torch.ops.ye3t_runtime,
            "carrier_channel_transform_adjoint",
        )
        and hasattr(
            torch.ops.ye3t_runtime,
            "carrier_channel_transform_double_backward",
        )
    )
    has_carrier_role_channel_map_adjoint = bool(
        extension is not None
        and core_abi_version is not None
        and core_abi_version >= 36
        and hasattr(
            torch.ops.ye3t_runtime,
            "carrier_role_channel_map_adjoint",
        )
        and hasattr(
            torch.ops.ye3t_runtime,
            "carrier_role_channel_map_adjoint_double_backward",
        )
    )
    return {
        "schema": "ye3t_execution_plan_v1",
        "torch_independent_core_source": True,
        "cmake_core_target": "ye3t_runtime_core",
        "prebuilt_extension": extension is not None,
        "cpu": bool(extension is not None and extension.has_cpu()),
        "cuda": has_cuda,
        "cuda_operations": (
            (
                "cheb_exp_cos_radial_with_derivative",
                "cheb_exp_cos_radial_table_with_derivative",
                *(
                    (
                        "cheb_exp_cos_radial_table_double_backward",
                    )
                    if has_radial_table_double_backward
                    else ()
                ),
                "spherical_harmonics_with_derivative",
                "spherical_harmonics_table_with_derivative",
                *(
                    (
                        "spherical_harmonics_table_double_backward",
                    )
                    if has_spherical_table_double_backward
                    else ()
                ),
                "plain_site_basis_product_with_derivative",
                "plain_site_basis_product_adjoint",
                *(
                    (
                        "scheduled_radial_angular_channels_with_derivative",
                    )
                    if has_scheduled_radial_angular_channels
                    else ()
                ),
                "density_accumulate",
                "density_accumulate_adjoint",
                *(
                    (
                        "edge_outer_accumulate",
                        "edge_outer_accumulate_adjoint",
                        "edge_outer_accumulate_double_backward",
                    )
                    if has_edge_outer_accumulate
                    else ()
                ),
                *(
                    (
                        "softmax_gaussian_role_density",
                        "softmax_gaussian_role_density_adjoint",
                        "softmax_gaussian_role_density_double_backward",
                    )
                    if has_softmax_gaussian_role_density
                    else ()
                ),
                *(
                    (
                        "scheduled_softmax_gaussian_role_density",
                        "scheduled_softmax_gaussian_role_density_adjoint",
                        "scheduled_softmax_gaussian_role_density_double_backward",
                    )
                    if has_scheduled_role_density
                    else ()
                ),
                *(
                    (
                        "carrier_gated_scatter",
                        "carrier_gated_scatter_adjoint",
                        "carrier_gated_scatter_double_backward",
                    )
                    if has_carrier_gated_scatter
                    else ()
                ),
                *(
                    (
                        "carrier_residual_gated_scatter",
                        "carrier_residual_gated_scatter_adjoint",
                        "carrier_residual_gated_scatter_double_backward",
                    )
                    if has_residual_carrier_gated_scatter
                    else ()
                ),
                *(
                    (
                        "carrier_segmented_residual_gated_scatter",
                        "carrier_segmented_residual_gated_scatter_adjoint",
                        "carrier_segmented_residual_gated_scatter_"
                        "double_backward",
                    )
                    if has_segmented_carrier_gated_scatter
                    else ()
                ),
                *(
                    (
                        "source_arena_gather",
                        "source_arena_gather_adjoint",
                        "source_arena_gather_double_backward",
                    )
                    if has_source_arena_gather
                    else ()
                ),
                *(
                    (
                        "source_arena_channel_transform",
                        "source_arena_channel_transform_adjoint",
                        "source_arena_channel_transform_double_backward",
                    )
                    if has_source_arena_channel_transform
                    else ()
                ),
                *(
                    (
                        "carrier_channel_update",
                        "carrier_channel_update_adjoint",
                        "carrier_channel_update_double_backward",
                    )
                    if has_carrier_channel_update
                    else ()
                ),
                *(
                    (
                        "carrier_channel_transform",
                        "carrier_channel_transform_adjoint",
                        "carrier_channel_transform_double_backward",
                    )
                    if has_carrier_channel_transform
                    else ()
                ),
                *(
                    (
                        "carrier_role_channel_map_adjoint",
                        "carrier_role_channel_map_adjoint_double_backward",
                    )
                    if has_carrier_role_channel_map_adjoint
                    else ()
                ),
                "compact_pair_product",
                "compact_pair_product_adjoint",
                "compact_exterior_power",
                "compact_exterior_power_adjoint",
                "symmetric_power_monomial",
                "symmetric_power_monomial_adjoint",
                *(
                    ("symmetric_power_monomial_double_backward",)
                    if has_symmetric_power_monomial_double_backward
                    else ()
                ),
                "symmetric_power_shared_monomial",
                "symmetric_power_shared_monomial_adjoint",
                *(
                    (
                        "symmetric_power_shared_monomial_double_backward",
                    )
                    if has_symmetric_power_shared_monomial_double_backward
                    else ()
                ),
                *(
                    (
                        "symmetric_power_shared_monomial_"
                        "factored_double_backward",
                    )
                    if has_symmetric_power_shared_monomial_factored_double_backward
                    else ()
                ),
                *(
                    (
                        "symmetric_power_shared_sparse_monomial",
                        "symmetric_power_shared_sparse_monomial_adjoint",
                        "symmetric_power_shared_sparse_monomial_"
                        "double_backward",
                    )
                    if has_symmetric_power_shared_sparse_monomial
                    else ()
                ),
                "symmetric_power_shared_monomial_batched_adjoint",
                "factorized_angular",
                "factorized_angular_adjoint",
                *(
                    ("factorized_angular_double_backward",)
                    if has_factorized_angular_double_backward
                    else ()
                ),
                *(
                    (
                        "factorized_angular_heterogeneous",
                        "factorized_angular_heterogeneous_adjoint",
                        "factorized_angular_heterogeneous_double_backward",
                    )
                    if has_heterogeneous_factorized_roots
                    else ()
                ),
                *(
                    (
                        "factorized_angular_heterogeneous_"
                        "adjoint_with_workspace",
                        "factorized_angular_heterogeneous_"
                        "double_backward_from_workspace",
                    )
                    if has_cached_heterogeneous_factorized_reverse
                    else ()
                ),
                *(
                    (
                        "factorized_angular_segmented",
                        "factorized_angular_segmented_adjoint",
                        "factorized_angular_segmented_double_backward",
                    )
                    if has_segmented_factorized_angular
                    else ()
                ),
                "factorized_angular_linear",
                "factorized_angular_linear_adjoint",
                "source_analysis",
                "source_analysis_adjoint",
                "source_analysis_linear",
                "source_analysis_linear_adjoint",
            )
            if has_cuda
            else ()
        ),
        "cuda_density_auto_min_work_items": (
            _NATIVE_CUDA_DENSITY_MIN_WORK_ITEMS
            if has_cuda
            else None
        ),
        "cuda_edge_outer_auto_min_work_items": (
            _NATIVE_CUDA_EDGE_OUTER_MIN_WORK_ITEMS
            if has_cuda and has_edge_outer_accumulate
            else None
        ),
        "cpu_edge_outer_auto_min_work_items": (
            _NATIVE_CPU_EDGE_OUTER_MIN_WORK_ITEMS
            if extension is not None and has_edge_outer_accumulate
            else None
        ),
        "cpu_carrier_gated_scatter_auto_min_work_items": (
            _NATIVE_CPU_CARRIER_GATED_SCATTER_MIN_WORK_ITEMS
            if extension is not None and has_carrier_gated_scatter
            else None
        ),
        "cuda_carrier_gated_scatter_auto_min_work_items": (
            _NATIVE_CUDA_CARRIER_GATED_SCATTER_MIN_WORK_ITEMS
            if has_cuda and has_carrier_gated_scatter
            else None
        ),
        "real_control_carrier_gates": has_real_control_carrier_gates,
        "residual_carrier_gated_scatter": (
            has_residual_carrier_gated_scatter
        ),
        "segmented_carrier_gated_scatter": (
            has_segmented_carrier_gated_scatter
        ),
        "segmented_factorized_angular": (
            has_segmented_factorized_angular
        ),
        "source_arena_gather": has_source_arena_gather,
        "source_arena_channel_transform": (
            has_source_arena_channel_transform
        ),
        "carrier_channel_transform": has_carrier_channel_transform,
        "carrier_role_channel_map_adjoint": (
            has_carrier_role_channel_map_adjoint
        ),
        "cuda_segmented_carrier_scatter_min_node_feature_work": (
            _NATIVE_CUDA_SEGMENTED_CARRIER_SCATTER_MIN_NODE_FEATURE_WORK
            if has_cuda and has_segmented_carrier_gated_scatter
            else None
        ),
        "real_control_carrier_channel_update": (
            has_real_control_carrier_channel_update
        ),
        "scheduled_softmax_gaussian_role_density": (
            has_scheduled_role_density
        ),
        "symmetric_power_monomial_double_backward": (
            has_symmetric_power_monomial_double_backward
        ),
        "symmetric_power_shared_monomial_double_backward": (
            has_symmetric_power_shared_monomial_double_backward
        ),
        "symmetric_power_shared_monomial_factored_double_backward": (
            has_symmetric_power_shared_monomial_factored_double_backward
        ),
        "symmetric_power_shared_double_backward_default_algorithm": "auto",
        "symmetric_power_shared_sparse_monomial": (
            has_symmetric_power_shared_sparse_monomial
        ),
        "cpu_carrier_channel_update_auto_max_work_items": (
            _NATIVE_CPU_CARRIER_CHANNEL_UPDATE_MAX_WORK_ITEMS
            if extension is not None and has_carrier_channel_update
            else None
        ),
        "cuda_carrier_channel_update_auto_max_work_items": (
            None
        ),
        "cuda_carrier_channel_update_auto_min_work_items": (
            _NATIVE_CUDA_CARRIER_CHANNEL_UPDATE_MIN_WORK_ITEMS
            if has_cuda and has_carrier_channel_update
            else None
        ),
        "symmetric_power_cpu_auto_max_batch": (
            32 if extension is not None else None
        ),
        "symmetric_power_cpu_auto_min_power": (
            6 if extension is not None else None
        ),
        "symmetric_power_shared_cpu_min_reuse": (
            3.0 if extension is not None else None
        ),
        "symmetric_power_shared_cpu_min_batch": (
            8 if extension is not None else None
        ),
        "symmetric_power_shared_cuda_min_reuse": (
            8.0 if has_cuda else None
        ),
        "symmetric_power_shared_cuda_grad_min_batch": (
            256 if has_cuda else None
        ),
        "symmetric_power_shared_cuda_forward_min_batch": (
            1024 if has_cuda else None
        ),
        "triton_available": triton_available,
        "triton_cuda_available": triton_cuda_available,
        "triton_real_convention_id": YE3T_REAL_TESSERAL_CONVENTION,
        "triton_operations": (
            "dense_source_analysis_C_dagger_L_v",
            "dense_source_analysis_channel_mixing",
            "packed_joint_young_angular_bilinear",
            "packed_joint_young_angular_channel_mixing",
        ),
        "core_abi_version": (
            core_abi_version
        ),
        "factorized_workspace_metadata": (
            "compiler_host_scalar_no_device_read"
            if has_host_certified_factorized_workspace
            else "legacy_device_tensor_read"
        ),
        "heterogeneous_factorized_roots": (
            has_heterogeneous_factorized_roots
        ),
        "cached_heterogeneous_factorized_reverse": (
            has_cached_heterogeneous_factorized_reverse
        ),
        "dtypes": ("float32", "float64", "complex64", "complex128"),
        "analysis_orientation": "conjugate_transpose",
        "operations": (
            "cheb_exp_cos_radial_with_derivative",
            "cheb_exp_cos_radial_table_with_derivative",
            *(
                (
                    "cheb_exp_cos_radial_table_double_backward",
                )
                if has_radial_table_double_backward
                else ()
            ),
            "complex_spherical_harmonics_with_derivative",
            "real_spherical_harmonics_with_derivative",
            "complex_spherical_harmonics_table_with_derivative",
            "real_spherical_harmonics_table_with_derivative",
            *(
                (
                    "real_spherical_harmonics_table_double_backward",
                )
                if has_spherical_table_double_backward
                else ()
            ),
            "plain_site_basis_product_with_derivative",
            "plain_site_basis_product_adjoint",
            *(
                (
                    "scheduled_radial_angular_channels_with_derivative",
                )
                if has_scheduled_radial_angular_channels
                else ()
            ),
            "density_accumulate",
            "density_accumulate_adjoint",
            *(
                (
                    "edge_outer_accumulate",
                    "edge_outer_accumulate_adjoint",
                    "edge_outer_accumulate_double_backward",
                )
                if has_edge_outer_accumulate
                else ()
            ),
            *(
                (
                    "softmax_gaussian_role_density",
                    "softmax_gaussian_role_density_adjoint",
                    "softmax_gaussian_role_density_double_backward",
                )
                if has_softmax_gaussian_role_density
                else ()
            ),
            *(
                (
                    "scheduled_softmax_gaussian_role_density",
                    "scheduled_softmax_gaussian_role_density_adjoint",
                    "scheduled_softmax_gaussian_role_density_double_backward",
                )
                if has_scheduled_role_density
                else ()
            ),
            *(
                (
                    "carrier_gated_scatter",
                    "carrier_gated_scatter_adjoint",
                    "carrier_gated_scatter_double_backward",
                )
                if has_carrier_gated_scatter
                else ()
            ),
            *(
                (
                    "carrier_residual_gated_scatter",
                    "carrier_residual_gated_scatter_adjoint",
                    "carrier_residual_gated_scatter_double_backward",
                )
                if has_residual_carrier_gated_scatter
                else ()
            ),
            *(
                (
                    "carrier_segmented_residual_gated_scatter",
                    "carrier_segmented_residual_gated_scatter_adjoint",
                    "carrier_segmented_residual_gated_scatter_"
                    "double_backward",
                )
                if has_segmented_carrier_gated_scatter
                else ()
            ),
            *(
                (
                    "source_arena_gather",
                    "source_arena_gather_adjoint",
                    "source_arena_gather_double_backward",
                )
                if has_source_arena_gather
                else ()
            ),
            *(
                (
                    "carrier_channel_update",
                    "carrier_channel_update_adjoint",
                    "carrier_channel_update_double_backward",
                )
                if has_carrier_channel_update
                else ()
            ),
            *(
                (
                    "carrier_channel_transform",
                    "carrier_channel_transform_adjoint",
                    "carrier_channel_transform_double_backward",
                )
                if has_carrier_channel_transform
                else ()
            ),
            *(
                (
                    "carrier_role_channel_map_adjoint",
                    "carrier_role_channel_map_adjoint_double_backward",
                )
                if has_carrier_role_channel_map_adjoint
                else ()
            ),
            "source_analysis",
            "source_analysis_adjoint",
            "source_analysis_linear",
            "source_analysis_linear_adjoint",
            "compact_symmetric_pair_product",
            "compact_exterior_pair_product",
            "compact_pair_product_adjoint",
            "compact_exterior_power",
            "compact_exterior_power_adjoint",
            "symmetric_power_monomial",
            "symmetric_power_monomial_adjoint",
            *(
                ("symmetric_power_monomial_double_backward",)
                if has_symmetric_power_monomial_double_backward
                else ()
            ),
            "symmetric_power_shared_monomial",
            "symmetric_power_shared_monomial_adjoint",
            *(
                (
                    "symmetric_power_shared_monomial_double_backward",
                )
                if has_symmetric_power_shared_monomial_double_backward
                else ()
            ),
            "symmetric_power_shared_monomial_batched_adjoint",
            "factorized_angular",
            "factorized_angular_adjoint",
            *(
                ("factorized_angular_double_backward",)
                if has_factorized_angular_double_backward
                else ()
            ),
            *(
                (
                    "factorized_angular_heterogeneous",
                    "factorized_angular_heterogeneous_adjoint",
                    "factorized_angular_heterogeneous_double_backward",
                )
                if has_heterogeneous_factorized_roots
                else ()
            ),
            *(
                (
                    "factorized_angular_heterogeneous_"
                    "adjoint_with_workspace",
                    "factorized_angular_heterogeneous_"
                    "double_backward_from_workspace",
                )
                if has_cached_heterogeneous_factorized_reverse
                else ()
            ),
            *(
                (
                    "factorized_angular_segmented",
                    "factorized_angular_segmented_adjoint",
                    "factorized_angular_segmented_double_backward",
                )
                if has_segmented_factorized_angular
                else ()
            ),
            "factorized_angular_linear",
            "factorized_angular_linear_adjoint",
        ),
        "compact_exterior_power_max_native_order": 8,
    }


def _chebyshev_first_reference(value, order):
    if int(order) == 0:
        return torch.ones_like(value)
    if int(order) == 1:
        return value
    previous = torch.ones_like(value)
    current = value
    for _ in range(2, int(order) + 1):
        previous, current = current, 2.0 * value * current - previous
    return current


def _cheb_exp_cos_radial_reference(
    radii,
    cutoffs,
    lambdas,
    radial_index,
):
    # TODO(YE3T_C2_RADIAL_CUTOFF): add a serialized C2 cutoff-envelope option
    # for the n=0/1 channels before claiming smooth force Hessians at cutoffs.
    safe_cutoffs = torch.clamp(
        cutoffs,
        min=torch.finfo(radii.dtype).eps,
    )
    scaled = radii / safe_cutoffs
    inside = scaled <= 1.0
    if int(radial_index) == 0:
        values = torch.ones_like(radii)
    elif int(radial_index) == 1:
        values = 0.5 * (1.0 + torch.cos(torch.pi * scaled))
    else:
        numerator = torch.exp(-lambdas * (scaled - 1.0)) - 1.0
        denominator = torch.exp(lambdas) - 1.0
        warped = 1.0 - 2.0 * numerator / denominator
        chebyshev = _chebyshev_first_reference(
            warped,
            int(radial_index),
        )
        values = (
            0.25
            * (1.0 - chebyshev)
            * (1.0 + torch.cos(torch.pi * scaled))
        )
    return torch.where(inside, values, torch.zeros_like(values))

def _cheb_exp_cos_radial_table_reference(
    radii,
    cutoffs,
    lambdas,
    maximum_radial_index,
):
    return torch.stack(
        [
            _cheb_exp_cos_radial_reference(
                radii,
                cutoffs,
                lambdas,
                radial_index,
            )
            for radial_index in range(
                int(maximum_radial_index) + 1
            )
        ],
        dim=1,
    )


def _spherical_harmonics_reference_values(
    edge_vectors,
    angular_momentum,
    *,
    real_output,
    epsilon,
):
    from ye3t.core.spherical import spherical_harmonics_l

    radius = torch.linalg.norm(edge_vectors, dim=-1)
    safe_radius = torch.clamp(radius, min=float(epsilon))
    unit = edge_vectors / safe_radius.unsqueeze(-1)
    theta = torch.atan2(
        torch.linalg.norm(unit[..., :2], dim=-1),
        unit[..., 2],
    )
    phi = torch.atan2(unit[..., 1], unit[..., 0])
    complex_values = spherical_harmonics_l(
        int(angular_momentum),
        theta,
        phi,
    ).transpose(0, 1)
    if not real_output:
        return complex_values
    rows = []
    root_two = math.sqrt(2.0)
    angular_momentum = int(angular_momentum)
    for magnetic in range(-angular_momentum, angular_momentum + 1):
        if magnetic < 0:
            sign = 1.0 if ((-magnetic) % 2) else -1.0
            rows.append(
                root_two
                * sign
                * complex_values[:, angular_momentum + abs(magnetic)].imag
            )
        elif magnetic == 0:
            rows.append(complex_values[:, angular_momentum].real)
        else:
            sign = -1.0 if (magnetic % 2) else 1.0
            rows.append(
                root_two
                * sign
                * complex_values[:, angular_momentum + magnetic].real
            )
    return torch.stack(rows, dim=1)

def _spherical_harmonics_table_reference_values(
    edge_vectors,
    maximum_angular_momentum,
    *,
    real_output,
    epsilon,
):
    return torch.cat(
        [
            _spherical_harmonics_reference_values(
                edge_vectors,
                angular_momentum,
                real_output=real_output,
                epsilon=epsilon,
            )
            for angular_momentum in range(
                int(maximum_angular_momentum) + 1
            )
        ],
        dim=1,
    )

def _plain_site_basis_product_reference_values(
    radial_values,
    angular_values,
    prefactors,
    term_groups,
    term_channels,
    channel_count,
):
    radial_terms = radial_values.index_select(0, term_groups)
    prefactor_terms = prefactors.index_select(0, term_groups)
    term_values = prefactor_terms * radial_terms * angular_values
    return radial_values.new_zeros(
        (radial_values.shape[1], int(channel_count))
    ).index_copy(
        1,
        term_channels,
        term_values.transpose(0, 1),
    )


def _scheduled_radial_angular_channels_reference_values(
    radial_values,
    angular_values,
    edge_types,
    channel_radial_indices,
    channel_angular_indices,
    channel_types,
    channel_scales,
):
    selected_radial = radial_values.index_select(
        1,
        channel_radial_indices,
    )
    selected_angular = angular_values.index_select(
        1,
        channel_angular_indices,
    )
    active = (
        (channel_types < 0).unsqueeze(0)
        | (edge_types.unsqueeze(1) == channel_types.unsqueeze(0))
    ).to(radial_values.dtype)
    return (
        selected_radial
        * selected_angular
        * active
        * channel_scales.unsqueeze(0)
    )


def _scheduled_radial_angular_channels_reference(
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
):
    selected_radial = radial_values.index_select(
        1,
        channel_radial_indices,
    )
    selected_radial_derivative = radial_derivatives.index_select(
        1,
        channel_radial_indices,
    )
    selected_angular = angular_values.index_select(
        1,
        channel_angular_indices,
    )
    selected_angular_derivative = angular_derivatives.index_select(
        1,
        channel_angular_indices,
    )
    scale = (
        (
            (channel_types < 0).unsqueeze(0)
            | (edge_types.unsqueeze(1) == channel_types.unsqueeze(0))
        ).to(radial_values.dtype)
        * channel_scales.unsqueeze(0)
    )
    values = selected_radial * selected_angular * scale
    derivatives = scale.unsqueeze(-1) * (
        selected_radial_derivative.unsqueeze(-1)
        * radial_directions.unsqueeze(1)
        * selected_angular.unsqueeze(-1)
        + selected_radial.unsqueeze(-1)
        * selected_angular_derivative
    )
    return values, derivatives


def _plain_site_basis_product_reference(
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
):
    radial_terms = radial_values.index_select(0, term_groups)
    radial_derivative_terms = radial_derivatives.index_select(
        0,
        term_groups,
    )
    prefactor_terms = prefactors.index_select(0, term_groups)
    angular_radial = angular_values.unsqueeze(-1)
    term_values = prefactor_terms * radial_terms * angular_values
    term_derivatives = prefactor_terms.unsqueeze(-1) * (
        radial_derivative_terms.unsqueeze(-1)
        * radial_directions.unsqueeze(0)
        * angular_radial
        + radial_terms.unsqueeze(-1) * angular_derivatives
    )
    charge_common = radial_terms * angular_values
    charge_center_terms = (
        prefactor_derivatives_center.index_select(0, term_groups)
        * charge_common
    )
    charge_neighbor_terms = (
        prefactor_derivatives_neighbor.index_select(0, term_groups)
        * charge_common
    )
    edge_values = radial_values.new_zeros(
        (radial_values.shape[1], int(channel_count))
    ).index_copy(
        1,
        term_channels,
        term_values.transpose(0, 1),
    )
    edge_derivatives = radial_values.new_zeros(
        (
            radial_values.shape[1],
            int(channel_count),
            3,
        )
    ).index_copy(
        1,
        term_channels,
        term_derivatives.permute(1, 0, 2),
    )
    charge_center = radial_values.new_zeros(
        edge_values.shape
    ).index_copy(
        1,
        term_channels,
        charge_center_terms.transpose(0, 1),
    )
    charge_neighbor = radial_values.new_zeros(
        edge_values.shape
    ).index_copy(
        1,
        term_channels,
        charge_neighbor_terms.transpose(0, 1),
    )
    return (
        edge_values,
        edge_derivatives,
        charge_center,
        charge_neighbor,
    )


def _plain_site_basis_product_adjoint_reference(
    radial_values,
    radial_derivatives,
    angular_values,
    angular_derivatives,
    prefactors,
    prefactor_derivatives_center,
    prefactor_derivatives_neighbor,
    radial_directions,
    edge_weights,
    edge_weight_derivatives,
    term_groups,
    term_channels,
    edge_adjoint,
):
    radial_terms = radial_values.index_select(0, term_groups)
    radial_derivative_terms = radial_derivatives.index_select(
        0,
        term_groups,
    )
    prefactor_terms = prefactors.index_select(0, term_groups)
    adjoint_terms = edge_adjoint.index_select(
        1,
        term_channels,
    ).transpose(0, 1)
    term_values = prefactor_terms * radial_terms * angular_values
    term_derivatives = prefactor_terms.unsqueeze(-1) * (
        radial_derivative_terms.unsqueeze(-1)
        * radial_directions.unsqueeze(0)
        * angular_values.unsqueeze(-1)
        + radial_terms.unsqueeze(-1) * angular_derivatives
    )
    weighted_derivatives = (
        term_derivatives * edge_weights.reshape(1, -1, 1)
        + term_values.unsqueeze(-1)
        * edge_weight_derivatives.unsqueeze(0)
    )
    edge_position_adjoint = (
        adjoint_terms.unsqueeze(-1) * weighted_derivatives
    ).sum(dim=0)
    charge_common = (
        radial_terms
        * angular_values
        * edge_weights.unsqueeze(0)
    )
    edge_charge_center = (
        adjoint_terms
        * prefactor_derivatives_center.index_select(0, term_groups)
        * charge_common
    ).sum(dim=0)
    edge_charge_neighbor = (
        adjoint_terms
        * prefactor_derivatives_neighbor.index_select(0, term_groups)
        * charge_common
    ).sum(dim=0)
    return (
        edge_position_adjoint,
        edge_charge_center,
        edge_charge_neighbor,
    )


def cheb_exp_cos_radial_with_derivative(
    radii,
    cutoffs,
    lambdas,
    radial_index,
    backend="auto",
):
    """Evaluate built-in compact radial values and analytic ``dR/dr``."""

    backend = str(backend)
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")
    require_native = _enabled("YE3T_REQUIRE_NATIVE")
    if backend == "reference" and require_native:
        raise RuntimeError(
            "YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback"
        )
    use_native = backend == "native" or require_native
    use_native = use_native or (
        backend == "auto"
        and (
            _prebuilt_extension() is not None
            or _enabled("YE3T_ENABLE_EXECUTION_PLAN_JIT")
        )
    )
    radii = torch.as_tensor(radii)
    cutoffs = torch.as_tensor(
        cutoffs,
        dtype=radii.dtype,
        device=radii.device,
    )
    lambdas = torch.as_tensor(
        lambdas,
        dtype=radii.dtype,
        device=radii.device,
    )
    if use_native:
        extension = _extension_for_native_dispatch()
        if (
            radii.device.type == "cuda"
            and not _native_cuda_dispatch_available(extension)
        ):
            if backend == "auto" and not require_native:
                use_native = False
            else:
                raise RuntimeError(
                    "the installed YE3T native extension has no CUDA "
                    "radial operator"
                )
        elif radii.device.type not in {"cpu", "cuda"}:
            raise ValueError("native radial source runtime requires CPU or CUDA")
        if use_native:
            return torch.ops.ye3t_runtime.cheb_exp_cos_radial_with_derivative(
                radii.contiguous(),
                cutoffs.expand_as(radii).contiguous(),
                lambdas.expand_as(radii).contiguous(),
                int(radial_index),
            )
    values = _cheb_exp_cos_radial_reference(
        radii,
        cutoffs,
        lambdas,
        int(radial_index),
    )
    derivative = torch.autograd.functional.jvp(
        lambda value: _cheb_exp_cos_radial_reference(
            value,
            cutoffs,
            lambdas,
            int(radial_index),
        ),
        radii,
        torch.ones_like(radii),
        create_graph=torch.is_grad_enabled(),
    )[1]
    return values, derivative


def cheb_exp_cos_radial_table_with_derivative(
    radii,
    cutoffs,
    lambdas,
    maximum_radial_index,
    backend="auto",
):
    """Evaluate all built-in radial indices through the requested maximum."""

    backend = str(backend)
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")
    maximum_radial_index = int(maximum_radial_index)
    if maximum_radial_index < 0:
        raise ValueError("maximum_radial_index must be non-negative")
    require_native = _enabled("YE3T_REQUIRE_NATIVE")
    if backend == "reference" and require_native:
        raise RuntimeError(
            "YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback"
        )
    use_native = backend == "native" or require_native
    use_native = use_native or (
        backend == "auto"
        and (
            _prebuilt_extension() is not None
            or _enabled("YE3T_ENABLE_EXECUTION_PLAN_JIT")
        )
    )
    radii = torch.as_tensor(radii)
    cutoffs = torch.as_tensor(
        cutoffs,
        dtype=radii.dtype,
        device=radii.device,
    )
    lambdas = torch.as_tensor(
        lambdas,
        dtype=radii.dtype,
        device=radii.device,
    )
    if use_native:
        extension = _extension_for_native_dispatch()
        if (
            radii.device.type == "cuda"
            and not _native_cuda_dispatch_available(extension)
        ):
            if backend == "auto" and not require_native:
                use_native = False
            else:
                raise RuntimeError(
                    "the installed YE3T native extension has no CUDA "
                    "radial-table operator"
                )
        elif radii.device.type not in {"cpu", "cuda"}:
            raise ValueError(
                "native radial-table source runtime requires CPU or CUDA"
            )
        if use_native:
            return torch.ops.ye3t_runtime.cheb_exp_cos_radial_table_with_derivative(
                radii.contiguous(),
                cutoffs.expand_as(radii).contiguous(),
                lambdas.expand_as(radii).contiguous(),
                maximum_radial_index,
            )
    values = _cheb_exp_cos_radial_table_reference(
        radii,
        cutoffs,
        lambdas,
        maximum_radial_index,
    )
    derivative = torch.autograd.functional.jvp(
        lambda value: _cheb_exp_cos_radial_table_reference(
            value,
            cutoffs,
            lambdas,
            maximum_radial_index,
        ),
        radii,
        torch.ones_like(radii),
        create_graph=torch.is_grad_enabled(),
    )[1]
    return values, derivative


def spherical_harmonics_with_derivative(
    edge_vectors,
    angular_momentum,
    *,
    real_output=False,
    epsilon=1.0e-12,
    backend="auto",
):
    """Evaluate fixed-``L`` harmonics and analytic Cartesian derivatives."""

    backend = str(backend)
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")
    require_native = _enabled("YE3T_REQUIRE_NATIVE")
    if backend == "reference" and require_native:
        raise RuntimeError(
            "YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback"
        )
    use_native = backend == "native" or require_native
    use_native = use_native or (
        backend == "auto"
        and (
            _prebuilt_extension() is not None
            or _enabled("YE3T_ENABLE_EXECUTION_PLAN_JIT")
        )
    )
    edge_vectors = torch.as_tensor(edge_vectors)
    if use_native:
        extension = _extension_for_native_dispatch()
        if edge_vectors.device.type == "cuda" and not bool(
            _native_cuda_dispatch_available(extension)
        ):
            if backend == "auto" and not require_native:
                use_native = False
            else:
                raise RuntimeError(
                    "the installed YE3T native extension has no CUDA "
                    "spherical-harmonic operator"
                )
        elif edge_vectors.device.type not in {"cpu", "cuda"}:
            raise ValueError(
                "native spherical source runtime requires CPU or CUDA"
            )
        if use_native:
            return torch.ops.ye3t_runtime.spherical_harmonics_with_derivative(
                edge_vectors.contiguous(),
                int(angular_momentum),
                bool(real_output),
                float(epsilon),
            )
    values = _spherical_harmonics_reference_values(
        edge_vectors,
        int(angular_momentum),
        real_output=bool(real_output),
        epsilon=float(epsilon),
    )
    if values.requires_grad:
        working_vectors = edge_vectors
        working_values = values
    else:
        working_vectors = edge_vectors.detach().requires_grad_(True)
        working_values = _spherical_harmonics_reference_values(
            working_vectors,
            int(angular_momentum),
            real_output=bool(real_output),
            epsilon=float(epsilon),
        )
    derivative_rows = []
    for component in range(working_values.shape[1]):
        component_values = working_values[:, component]
        if component_values.is_complex():
            real_gradient = torch.autograd.grad(
                component_values.real.sum(),
                working_vectors,
                retain_graph=True,
                create_graph=torch.is_grad_enabled(),
            )[0]
            imaginary_gradient = torch.autograd.grad(
                component_values.imag.sum(),
                working_vectors,
                retain_graph=True,
                create_graph=torch.is_grad_enabled(),
            )[0]
            gradient = torch.complex(real_gradient, imaginary_gradient)
        else:
            gradient = torch.autograd.grad(
                component_values.sum(),
                working_vectors,
                retain_graph=True,
                create_graph=torch.is_grad_enabled(),
            )[0]
        derivative_rows.append(gradient)
    derivative = torch.stack(derivative_rows, dim=1)
    if working_vectors is not edge_vectors:
        derivative = derivative.detach()
    return values, derivative


def spherical_harmonics_table_with_derivative(
    edge_vectors,
    maximum_angular_momentum,
    *,
    real_output=False,
    epsilon=1.0e-12,
    backend="auto",
):
    """Evaluate packed harmonics for every ``L`` through the requested maximum."""

    backend = str(backend)
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")
    maximum_angular_momentum = int(maximum_angular_momentum)
    if maximum_angular_momentum < 0:
        raise ValueError(
            "maximum_angular_momentum must be non-negative"
        )
    require_native = _enabled("YE3T_REQUIRE_NATIVE")
    if backend == "reference" and require_native:
        raise RuntimeError(
            "YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback"
        )
    use_native = backend == "native" or require_native
    use_native = use_native or (
        backend == "auto"
        and (
            _prebuilt_extension() is not None
            or _enabled("YE3T_ENABLE_EXECUTION_PLAN_JIT")
        )
    )
    edge_vectors = torch.as_tensor(edge_vectors)
    if use_native:
        extension = _extension_for_native_dispatch()
        if edge_vectors.device.type == "cuda" and not bool(
            _native_cuda_dispatch_available(extension)
        ):
            if backend == "auto" and not require_native:
                use_native = False
            else:
                raise RuntimeError(
                    "the installed YE3T native extension has no CUDA "
                    "spherical-harmonic-table operator"
                )
        elif edge_vectors.device.type not in {"cpu", "cuda"}:
            raise ValueError(
                "native spherical-table source runtime requires CPU or CUDA"
            )
        if use_native:
            return torch.ops.ye3t_runtime.spherical_harmonics_table_with_derivative(
                edge_vectors.contiguous(),
                maximum_angular_momentum,
                bool(real_output),
                float(epsilon),
            )
    values = _spherical_harmonics_table_reference_values(
        edge_vectors,
        maximum_angular_momentum,
        real_output=bool(real_output),
        epsilon=float(epsilon),
    )
    if values.requires_grad:
        working_vectors = edge_vectors
        working_values = values
    else:
        working_vectors = edge_vectors.detach().requires_grad_(True)
        working_values = _spherical_harmonics_table_reference_values(
            working_vectors,
            maximum_angular_momentum,
            real_output=bool(real_output),
            epsilon=float(epsilon),
        )
    derivative_rows = []
    for component in range(working_values.shape[1]):
        component_values = working_values[:, component]
        if component_values.is_complex():
            real_gradient = torch.autograd.grad(
                component_values.real.sum(),
                working_vectors,
                retain_graph=True,
                create_graph=torch.is_grad_enabled(),
            )[0]
            imaginary_gradient = torch.autograd.grad(
                component_values.imag.sum(),
                working_vectors,
                retain_graph=True,
                create_graph=torch.is_grad_enabled(),
            )[0]
            gradient = torch.complex(real_gradient, imaginary_gradient)
        else:
            gradient = torch.autograd.grad(
                component_values.sum(),
                working_vectors,
                retain_graph=True,
                create_graph=torch.is_grad_enabled(),
            )[0]
        derivative_rows.append(gradient)
    derivative = torch.stack(derivative_rows, dim=1)
    if working_vectors is not edge_vectors:
        derivative = derivative.detach()
    return values, derivative


def plain_site_basis_product_with_derivative(
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
    backend="auto",
):
    """Assemble grouped plain source channels and analytic edge derivatives."""

    backend = str(backend)
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")
    require_native = _enabled("YE3T_REQUIRE_NATIVE")
    if backend == "reference" and require_native:
        raise RuntimeError(
            "YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback"
        )
    use_native = backend == "native" or require_native
    use_native = use_native or (
        backend == "auto"
        and (
            _prebuilt_extension() is not None
            or _enabled("YE3T_ENABLE_EXECUTION_PLAN_JIT")
        )
    )
    radial_values = torch.as_tensor(radial_values)
    data = [
        torch.as_tensor(
            value,
            dtype=radial_values.dtype,
            device=radial_values.device,
        )
        for value in (
            radial_derivatives,
            angular_values,
            angular_derivatives,
            prefactors,
            prefactor_derivatives_center,
            prefactor_derivatives_neighbor,
            radial_directions,
        )
    ]
    term_groups = torch.as_tensor(
        term_groups,
        dtype=torch.int64,
        device=radial_values.device,
    )
    term_channels = torch.as_tensor(
        term_channels,
        dtype=torch.int64,
        device=radial_values.device,
    )
    channel_count = int(channel_count)
    if use_native:
        extension = _extension_for_native_dispatch()
        if radial_values.device.type == "cuda" and not bool(
            _native_cuda_dispatch_available(extension)
        ):
            if backend == "auto" and not require_native:
                use_native = False
            else:
                raise RuntimeError(
                    "the installed YE3T native extension has no CUDA "
                    "plain-site-basis product operator"
                )
        elif radial_values.device.type not in {"cpu", "cuda"}:
            raise ValueError(
                "native plain-site-basis runtime requires CPU or CUDA"
            )
        if use_native:
            return torch.ops.ye3t_runtime.plain_site_basis_product_with_derivative(
                radial_values.contiguous(),
                *(value.contiguous() for value in data),
                term_groups.contiguous(),
                term_channels.contiguous(),
                channel_count,
            )
    return _plain_site_basis_product_reference(
        radial_values,
        *data,
        term_groups,
        term_channels,
        channel_count,
    )


def scheduled_radial_angular_channels_with_derivative(
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
    backend="auto",
):
    """Apply a fixed radial/angular channel schedule and analytic derivatives."""

    backend = str(backend)
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")
    require_native = _enabled("YE3T_REQUIRE_NATIVE")
    if backend == "reference" and require_native:
        raise RuntimeError(
            "YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback"
        )
    radial_values = torch.as_tensor(radial_values)
    data = tuple(
        torch.as_tensor(
            value,
            dtype=radial_values.dtype,
            device=radial_values.device,
        )
        for value in (
            radial_derivatives,
            angular_values,
            angular_derivatives,
            radial_directions,
            channel_scales,
        )
    )
    indices = tuple(
        torch.as_tensor(
            value,
            dtype=torch.int64,
            device=radial_values.device,
        )
        for value in (
            edge_types,
            channel_radial_indices,
            channel_angular_indices,
            channel_types,
        )
    )
    (
        radial_derivatives,
        angular_values,
        angular_derivatives,
        radial_directions,
        channel_scales,
    ) = data
    (
        edge_types,
        channel_radial_indices,
        channel_angular_indices,
        channel_types,
    ) = indices
    use_native = backend == "native" or require_native
    use_native = use_native or (
        backend == "auto"
        and (
            _prebuilt_extension() is not None
            or _enabled("YE3T_ENABLE_EXECUTION_PLAN_JIT")
        )
    )
    if use_native:
        extension = _extension_for_native_dispatch()
        if radial_values.device.type == "cuda" and not bool(
            _native_cuda_dispatch_available(extension)
        ):
            if backend == "auto" and not require_native:
                use_native = False
            else:
                raise RuntimeError(
                    "the installed YE3T native extension has no CUDA "
                    "scheduled radial-angular channel operator"
                )
        elif radial_values.device.type not in {"cpu", "cuda"}:
            raise ValueError(
                "native scheduled radial-angular channels require CPU or CUDA"
            )
        if use_native:
            return torch.ops.ye3t_runtime.scheduled_radial_angular_channels_with_derivative(
                radial_values.contiguous(),
                radial_derivatives.contiguous(),
                angular_values.contiguous(),
                angular_derivatives.contiguous(),
                radial_directions.contiguous(),
                edge_types.contiguous(),
                channel_radial_indices.contiguous(),
                channel_angular_indices.contiguous(),
                channel_types.contiguous(),
                channel_scales.contiguous(),
            )
    return _scheduled_radial_angular_channels_reference(
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


def plain_site_basis_product_adjoint(
    radial_values,
    radial_derivatives,
    angular_values,
    angular_derivatives,
    prefactors,
    prefactor_derivatives_center,
    prefactor_derivatives_neighbor,
    radial_directions,
    edge_weights,
    edge_weight_derivatives,
    term_groups,
    term_channels,
    edge_adjoint,
    backend="auto",
):
    """Reduce grouped plain source derivatives against edge-channel adjoints."""

    backend = str(backend)
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")
    require_native = _enabled("YE3T_REQUIRE_NATIVE")
    if backend == "reference" and require_native:
        raise RuntimeError(
            "YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback"
        )
    use_native = backend == "native" or require_native
    use_native = use_native or (
        backend == "auto"
        and (
            _prebuilt_extension() is not None
            or _enabled("YE3T_ENABLE_EXECUTION_PLAN_JIT")
        )
    )
    radial_values = torch.as_tensor(radial_values)
    data = [
        torch.as_tensor(
            value,
            dtype=radial_values.dtype,
            device=radial_values.device,
        )
        for value in (
            radial_derivatives,
            angular_values,
            angular_derivatives,
            prefactors,
            prefactor_derivatives_center,
            prefactor_derivatives_neighbor,
            radial_directions,
            edge_weights,
            edge_weight_derivatives,
        )
    ]
    term_groups = torch.as_tensor(
        term_groups,
        dtype=torch.int64,
        device=radial_values.device,
    )
    term_channels = torch.as_tensor(
        term_channels,
        dtype=torch.int64,
        device=radial_values.device,
    )
    edge_adjoint = torch.as_tensor(
        edge_adjoint,
        dtype=radial_values.dtype,
        device=radial_values.device,
    )
    if use_native:
        extension = _load_extension()
        if radial_values.device.type == "cuda" and not bool(
            extension.has_cuda()
        ):
            if backend == "auto" and not require_native:
                use_native = False
            else:
                raise RuntimeError(
                    "the installed YE3T native extension has no CUDA "
                    "plain-site-basis adjoint operator"
                )
        elif radial_values.device.type not in {"cpu", "cuda"}:
            raise ValueError(
                "native plain-site-basis adjoint requires CPU or CUDA"
            )
        if use_native:
            return torch.ops.ye3t_runtime.plain_site_basis_product_adjoint(
                radial_values.contiguous(),
                *(value.contiguous() for value in data),
                term_groups.contiguous(),
                term_channels.contiguous(),
                edge_adjoint.contiguous(),
            )
    return _plain_site_basis_product_adjoint_reference(
        radial_values,
        *data,
        term_groups,
        term_channels,
        edge_adjoint,
    )


def _deterministic_segment_sum(values, indices, output_count):
    """Sum rows by integer segment through a fixed dense incidence map."""

    values = torch.as_tensor(values)
    indices = torch.as_tensor(
        indices,
        dtype=torch.int64,
        device=values.device,
    )
    if values.ndim != 2 or indices.ndim != 1:
        raise ValueError("deterministic segment sum expects rows and indices")
    if int(values.shape[0]) != int(indices.numel()):
        raise ValueError("deterministic segment row and index counts must match")
    if int(output_count) <= 0:
        raise ValueError("deterministic segment output_count must be positive")
    incidence = torch.nn.functional.one_hot(
        indices,
        num_classes=int(output_count),
    ).transpose(0, 1).to(dtype=values.dtype)
    return incidence @ values


def density_accumulate(
    edge_values,
    centers,
    atom_count,
    *,
    backend="auto",
):
    """Accumulate edge-source rows into ordinary per-center density rows."""

    backend = str(backend)
    if backend not in {"auto", "native", "reference", "deterministic"}:
        raise ValueError(
            "backend must be 'auto', 'native', 'reference', or 'deterministic'"
        )
    require_native = _enabled("YE3T_REQUIRE_NATIVE")
    if backend in {"reference", "deterministic"} and require_native:
        raise RuntimeError(
            "YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback"
        )
    use_native = backend == "native" or require_native
    use_native = use_native or (
        backend == "auto"
        and (
            _prebuilt_extension() is not None
            or _enabled("YE3T_ENABLE_EXECUTION_PLAN_JIT")
        )
    )
    edge_values = torch.as_tensor(edge_values)
    centers = torch.as_tensor(
        centers,
        dtype=torch.int64,
        device=edge_values.device,
    )
    if (
        use_native
        and backend == "auto"
        and edge_values.device.type == "cuda"
        and int(edge_values.numel())
        < _NATIVE_CUDA_DENSITY_MIN_WORK_ITEMS
    ):
        use_native = False
    if use_native:
        extension = _load_extension()
        if edge_values.device.type == "cuda" and not bool(
            extension.has_cuda()
        ):
            if backend == "auto" and not require_native:
                use_native = False
            else:
                raise RuntimeError(
                    "the installed YE3T native extension has no CUDA "
                    "density-accumulation operators"
                )
        elif edge_values.device.type not in {"cpu", "cuda"}:
            raise ValueError(
                "native density accumulation requires CPU or CUDA"
            )
        if use_native:
            return torch.ops.ye3t_runtime.density_accumulate(
                edge_values.contiguous(),
                centers.contiguous(),
                int(atom_count),
            )
    if backend == "deterministic":
        return _deterministic_segment_sum(
            edge_values,
            centers,
            int(atom_count),
        )
    output = edge_values.new_zeros((int(atom_count), edge_values.shape[1]))
    output.index_add_(0, centers, edge_values)
    return output


def density_accumulate_adjoint(
    atomic_adjoint,
    centers,
    *,
    backend="auto",
):
    """Gather ordinary-density adjoints from center rows to edge rows."""

    backend = str(backend)
    if backend not in {"auto", "native", "reference", "deterministic"}:
        raise ValueError(
            "backend must be 'auto', 'native', 'reference', or 'deterministic'"
        )
    require_native = _enabled("YE3T_REQUIRE_NATIVE")
    if backend in {"reference", "deterministic"} and require_native:
        raise RuntimeError(
            "YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback"
        )
    use_native = backend == "native" or require_native
    use_native = use_native or (
        backend == "auto"
        and (
            _prebuilt_extension() is not None
            or _enabled("YE3T_ENABLE_EXECUTION_PLAN_JIT")
        )
    )
    atomic_adjoint = torch.as_tensor(atomic_adjoint)
    centers = torch.as_tensor(
        centers,
        dtype=torch.int64,
        device=atomic_adjoint.device,
    )
    if (
        use_native
        and backend == "auto"
        and atomic_adjoint.device.type == "cuda"
        and int(centers.numel() * atomic_adjoint.shape[1])
        < _NATIVE_CUDA_DENSITY_MIN_WORK_ITEMS
    ):
        use_native = False
    if use_native:
        extension = _load_extension()
        if atomic_adjoint.device.type == "cuda" and not bool(
            extension.has_cuda()
        ):
            if backend == "auto" and not require_native:
                use_native = False
            else:
                raise RuntimeError(
                    "the installed YE3T native extension has no CUDA "
                    "density-adjoint operator"
                )
        elif atomic_adjoint.device.type not in {"cpu", "cuda"}:
            raise ValueError(
                "native density accumulation requires CPU or CUDA"
            )
        if use_native:
            return torch.ops.ye3t_runtime.density_accumulate_adjoint(
                atomic_adjoint.contiguous(),
                centers.contiguous(),
            )
    return atomic_adjoint.index_select(0, centers)


def edge_outer_accumulate(
    left,
    right,
    centers,
    atom_count,
    *,
    backend="auto",
):
    """Accumulate per-edge outer products into center-indexed rows."""

    backend = str(backend)
    if backend not in {"auto", "native", "reference", "deterministic"}:
        raise ValueError(
            "backend must be 'auto', 'native', 'reference', or 'deterministic'"
        )
    require_native = _enabled("YE3T_REQUIRE_NATIVE")
    if backend in {"reference", "deterministic"} and require_native:
        raise RuntimeError(
            "YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback"
        )
    left = torch.as_tensor(left)
    right = torch.as_tensor(
        right,
        dtype=left.dtype,
        device=left.device,
    )
    centers = torch.as_tensor(
        centers,
        dtype=torch.int64,
        device=left.device,
    )
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("left and right must be two-dimensional")
    if int(left.shape[0]) != int(right.shape[0]):
        raise ValueError("left and right edge counts must match")
    if centers.ndim != 1 or int(centers.numel()) != int(left.shape[0]):
        raise ValueError("centers must contain one index per edge")
    if left.dtype not in {torch.float32, torch.float64}:
        raise TypeError("edge outer accumulation requires float32 or float64")
    if int(atom_count) <= 0:
        raise ValueError("atom_count must be positive")
    use_native = backend == "native" or require_native
    use_native = use_native or (
        backend == "auto"
        and (
            _prebuilt_extension() is not None
            or _enabled("YE3T_ENABLE_EXECUTION_PLAN_JIT")
        )
    )
    work_items = int(
        left.shape[0] * left.shape[1] * right.shape[1]
    )
    if (
        use_native
        and backend == "auto"
        and (
            (
                left.device.type == "cpu"
                and work_items
                < _NATIVE_CPU_EDGE_OUTER_MIN_WORK_ITEMS
            )
            or (
                left.device.type == "cuda"
                and work_items
                < _NATIVE_CUDA_EDGE_OUTER_MIN_WORK_ITEMS
            )
        )
    ):
        use_native = False
    if use_native:
        extension = _load_extension()
        if left.device.type == "cuda" and not bool(extension.has_cuda()):
            if backend == "auto" and not require_native:
                use_native = False
            else:
                raise RuntimeError(
                    "the installed YE3T native extension has no CUDA "
                    "edge-outer accumulation operators"
                )
        elif left.device.type not in {"cpu", "cuda"}:
            raise ValueError(
                "native edge outer accumulation requires CPU or CUDA"
            )
        if use_native:
            return torch.ops.ye3t_runtime.edge_outer_accumulate(
                left.contiguous(),
                right.contiguous(),
                centers.contiguous(),
                int(atom_count),
            )
    edge_outer = (
        left.unsqueeze(2) * right.unsqueeze(1)
    ).reshape(
        int(left.shape[0]),
        int(left.shape[1] * right.shape[1]),
    )
    if backend == "deterministic":
        return _deterministic_segment_sum(
            edge_outer,
            centers,
            int(atom_count),
        ).reshape(
            int(atom_count),
            int(left.shape[1]),
            int(right.shape[1]),
        )
    output = left.new_zeros(
        (int(atom_count), int(left.shape[1]), int(right.shape[1]))
    )
    output.reshape(int(atom_count), -1).index_add_(
        0,
        centers,
        edge_outer,
    )
    return output


def edge_outer_accumulate_adjoint(
    atomic_adjoint,
    left,
    right,
    centers,
    *,
    backend="auto",
):
    """Apply the adjoint of center-indexed edge outer accumulation."""

    backend = str(backend)
    if backend not in {"auto", "native", "reference", "deterministic"}:
        raise ValueError(
            "backend must be 'auto', 'native', 'reference', or 'deterministic'"
        )
    require_native = _enabled("YE3T_REQUIRE_NATIVE")
    if backend in {"reference", "deterministic"} and require_native:
        raise RuntimeError(
            "YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback"
        )
    atomic_adjoint = torch.as_tensor(atomic_adjoint)
    left = torch.as_tensor(
        left,
        dtype=atomic_adjoint.dtype,
        device=atomic_adjoint.device,
    )
    right = torch.as_tensor(
        right,
        dtype=atomic_adjoint.dtype,
        device=atomic_adjoint.device,
    )
    centers = torch.as_tensor(
        centers,
        dtype=torch.int64,
        device=atomic_adjoint.device,
    )
    if atomic_adjoint.ndim != 3:
        raise ValueError("atomic_adjoint must be three-dimensional")
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("left and right must be two-dimensional")
    if (
        int(left.shape[0]) != int(right.shape[0])
        or int(centers.numel()) != int(left.shape[0])
    ):
        raise ValueError("edge counts must agree")
    if (
        int(atomic_adjoint.shape[1]) != int(left.shape[1])
        or int(atomic_adjoint.shape[2]) != int(right.shape[1])
    ):
        raise ValueError(
            "atomic_adjoint dimensions must match left and right"
        )
    if atomic_adjoint.dtype not in {torch.float32, torch.float64}:
        raise TypeError("edge outer accumulation requires float32 or float64")
    use_native = backend == "native" or require_native
    use_native = use_native or (
        backend == "auto"
        and (
            _prebuilt_extension() is not None
            or _enabled("YE3T_ENABLE_EXECUTION_PLAN_JIT")
        )
    )
    work_items = int(
        left.shape[0] * left.shape[1] * right.shape[1]
    )
    if (
        use_native
        and backend == "auto"
        and (
            (
                atomic_adjoint.device.type == "cpu"
                and work_items
                < _NATIVE_CPU_EDGE_OUTER_MIN_WORK_ITEMS
            )
            or (
                atomic_adjoint.device.type == "cuda"
                and work_items
                < _NATIVE_CUDA_EDGE_OUTER_MIN_WORK_ITEMS
            )
        )
    ):
        use_native = False
    if use_native:
        extension = _load_extension()
        if (
            atomic_adjoint.device.type == "cuda"
            and not bool(extension.has_cuda())
        ):
            if backend == "auto" and not require_native:
                use_native = False
            else:
                raise RuntimeError(
                    "the installed YE3T native extension has no CUDA "
                    "edge-outer adjoint operators"
                )
        elif atomic_adjoint.device.type not in {"cpu", "cuda"}:
            raise ValueError(
                "native edge outer adjoint requires CPU or CUDA"
            )
        if use_native:
            return (
                torch.ops.ye3t_runtime.edge_outer_accumulate_adjoint(
                    atomic_adjoint.contiguous(),
                    left.contiguous(),
                    right.contiguous(),
                    centers.contiguous(),
                )
            )
    gathered = atomic_adjoint.index_select(0, centers)
    return (
        torch.einsum("esc,ec->es", gathered, right),
        torch.einsum("esc,es->ec", gathered, left),
    )


def _prepare_softmax_gaussian_role_density_inputs(
    distances,
    cutoffs,
    filter_centers,
    filter_width,
    edge_values,
    atom_centers,
    atom_count,
):
    distances = torch.as_tensor(distances)
    cutoffs = torch.as_tensor(
        cutoffs,
        dtype=distances.dtype,
        device=distances.device,
    )
    filter_centers = torch.as_tensor(
        filter_centers,
        dtype=distances.dtype,
        device=distances.device,
    )
    edge_values = torch.as_tensor(
        edge_values,
        dtype=distances.dtype,
        device=distances.device,
    )
    atom_centers = torch.as_tensor(
        atom_centers,
        dtype=torch.int64,
        device=distances.device,
    )
    if (
        distances.ndim != 1
        or cutoffs.ndim != 1
        or filter_centers.ndim != 1
        or edge_values.ndim != 2
        or atom_centers.ndim != 1
    ):
        raise ValueError(
            "distances, cutoffs, filter_centers, and atom_centers must be "
            "vectors and edge_values must be a matrix"
        )
    edge_count = int(distances.numel())
    if (
        int(cutoffs.numel()) != edge_count
        or int(edge_values.shape[0]) != edge_count
        or int(atom_centers.numel()) != edge_count
    ):
        raise ValueError("softmax-Gaussian role-density edge counts must match")
    if distances.dtype not in {torch.float32, torch.float64}:
        raise TypeError("softmax-Gaussian role density requires float32 or float64")
    if int(filter_centers.numel()) == 0 or int(edge_values.shape[1]) == 0:
        raise ValueError("softmax-Gaussian role density requires roles and channels")
    if float(filter_width) <= 0.0:
        raise ValueError("filter_width must be positive")
    if int(atom_count) <= 0:
        raise ValueError("atom_count must be positive")
    if bool(cutoffs.requires_grad) or bool(filter_centers.requires_grad):
        raise ValueError(
            "cutoffs and filter_centers are fixed schedule data; gradients are "
            "defined only for distances and edge_values"
        )
    return (
        distances,
        cutoffs,
        filter_centers,
        float(filter_width),
        edge_values,
        atom_centers,
        int(atom_count),
    )


def _softmax_gaussian_role_density_reference(
    distances,
    cutoffs,
    filter_centers,
    filter_width,
    edge_values,
    atom_centers,
    atom_count,
    deterministic=False,
):
    scaled = distances.unsqueeze(1) / cutoffs.unsqueeze(1)
    logits = -0.5 * (
        (scaled - filter_centers.reshape(1, -1)) / filter_width
    ).pow(2)
    filters = torch.softmax(logits, dim=1)
    contributions = (filters.unsqueeze(2) * edge_values.unsqueeze(1)).reshape(
        int(distances.numel()),
        -1,
    )
    if deterministic:
        output = _deterministic_segment_sum(
            contributions,
            atom_centers,
            int(atom_count),
        ).reshape(
            int(atom_count),
            int(filter_centers.numel()),
            int(edge_values.shape[1]),
        )
    else:
        output = edge_values.new_zeros(
            (
                int(atom_count),
                int(filter_centers.numel()),
                int(edge_values.shape[1]),
            )
        )
        output.reshape(int(atom_count), -1).index_add_(
            0,
            atom_centers,
            contributions,
        )
    return output


def softmax_gaussian_role_density(
    distances,
    cutoffs,
    filter_centers,
    filter_width,
    edge_values,
    atom_centers,
    atom_count,
    *,
    backend="auto",
):
    """Accumulate fixed softmax-Gaussian role-filtered edge channels."""

    backend = str(backend)
    if backend not in {"auto", "native", "reference", "deterministic"}:
        raise ValueError(
            "backend must be 'auto', 'native', 'reference', or 'deterministic'"
        )
    require_native = _enabled("YE3T_REQUIRE_NATIVE")
    if backend in {"reference", "deterministic"} and require_native:
        raise RuntimeError(
            "YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback"
        )
    prepared = _prepare_softmax_gaussian_role_density_inputs(
        distances,
        cutoffs,
        filter_centers,
        filter_width,
        edge_values,
        atom_centers,
        atom_count,
    )
    distances = prepared[0]
    use_native = backend == "native" or require_native
    use_native = use_native or (
        backend == "auto"
        and (
            _prebuilt_extension() is not None
            or _enabled("YE3T_ENABLE_EXECUTION_PLAN_JIT")
        )
    )
    if use_native:
        extension = _extension_for_native_dispatch()
        if (
            distances.device.type == "cuda"
            and not _native_cuda_dispatch_available(extension)
        ):
            if backend == "auto" and not require_native:
                use_native = False
            else:
                raise RuntimeError(
                    "the installed YE3T native extension has no CUDA "
                    "softmax-Gaussian role-density operators"
                )
        elif distances.device.type not in {"cpu", "cuda"}:
            raise ValueError(
                "native softmax-Gaussian role density requires CPU or CUDA"
            )
        if use_native:
            return torch.ops.ye3t_runtime.softmax_gaussian_role_density(
                prepared[0].contiguous(),
                prepared[1].contiguous(),
                prepared[2].contiguous(),
                prepared[3],
                prepared[4].contiguous(),
                prepared[5].contiguous(),
                prepared[6],
            )
    return _softmax_gaussian_role_density_reference(
        *prepared,
        deterministic=backend == "deterministic",
    )


def softmax_gaussian_role_density_adjoint(
    atomic_adjoint,
    distances,
    cutoffs,
    filter_centers,
    filter_width,
    edge_values,
    atom_centers,
    *,
    backend="auto",
):
    """Apply the adjoint of fixed softmax-Gaussian role accumulation."""

    atomic_adjoint = torch.as_tensor(atomic_adjoint)
    prepared = _prepare_softmax_gaussian_role_density_inputs(
        distances,
        cutoffs,
        filter_centers,
        filter_width,
        edge_values,
        atom_centers,
        int(atomic_adjoint.shape[0]),
    )
    if atomic_adjoint.ndim != 3 or tuple(atomic_adjoint.shape[1:]) != (
        int(prepared[2].numel()),
        int(prepared[4].shape[1]),
    ):
        raise ValueError("atomic_adjoint must match the role-density output")
    backend = str(backend)
    if backend not in {"auto", "native", "reference", "deterministic"}:
        raise ValueError(
            "backend must be 'auto', 'native', 'reference', or 'deterministic'"
        )
    require_native = _enabled("YE3T_REQUIRE_NATIVE")
    if backend in {"reference", "deterministic"} and require_native:
        raise RuntimeError(
            "YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback"
        )
    use_native = backend == "native" or require_native
    use_native = use_native or (
        backend == "auto"
        and (
            _prebuilt_extension() is not None
            or _enabled("YE3T_ENABLE_EXECUTION_PLAN_JIT")
        )
    )
    if use_native:
        extension = _load_extension()
        if prepared[0].device.type == "cuda" and not bool(extension.has_cuda()):
            if backend == "auto" and not require_native:
                use_native = False
            else:
                raise RuntimeError(
                    "the installed YE3T native extension has no CUDA "
                    "softmax-Gaussian role-density adjoint"
                )
        if use_native:
            return torch.ops.ye3t_runtime.softmax_gaussian_role_density_adjoint(
                atomic_adjoint.contiguous(),
                prepared[0].contiguous(),
                prepared[1].contiguous(),
                prepared[2].contiguous(),
                prepared[3],
                prepared[4].contiguous(),
                prepared[5].contiguous(),
            )
    distances_ref = prepared[0].detach().requires_grad_(True)
    edge_values_ref = prepared[4].detach().requires_grad_(True)
    with torch.enable_grad():
        output = _softmax_gaussian_role_density_reference(
            distances_ref,
            prepared[1],
            prepared[2],
            prepared[3],
            edge_values_ref,
            prepared[5],
            prepared[6],
            deterministic=backend == "deterministic",
        )
        return torch.autograd.grad(
            output,
            (distances_ref, edge_values_ref),
            atomic_adjoint,
            create_graph=torch.is_grad_enabled(),
        )


def _prepare_scheduled_role_density_inputs(
    radial_values,
    angular_values,
    distances,
    cutoffs,
    filter_centers,
    filter_width,
    soft_weights,
    edge_types,
    channel_radial_indices,
    channel_angular_indices,
    channel_types,
    channel_scales,
    atom_centers,
    atom_count,
):
    radial_values = torch.as_tensor(radial_values)
    device = radial_values.device
    dtype = radial_values.dtype
    floating = (
        torch.as_tensor(angular_values, dtype=dtype, device=device),
        torch.as_tensor(distances, dtype=dtype, device=device),
        torch.as_tensor(cutoffs, dtype=dtype, device=device),
        torch.as_tensor(filter_centers, dtype=dtype, device=device),
        torch.as_tensor(soft_weights, dtype=dtype, device=device),
        torch.as_tensor(channel_scales, dtype=dtype, device=device),
    )
    integers = tuple(
        torch.as_tensor(value, dtype=torch.int64, device=device)
        for value in (
            edge_types,
            channel_radial_indices,
            channel_angular_indices,
            channel_types,
            atom_centers,
        )
    )
    (
        angular_values,
        distances,
        cutoffs,
        filter_centers,
        soft_weights,
        channel_scales,
    ) = floating
    (
        edge_types,
        channel_radial_indices,
        channel_angular_indices,
        channel_types,
        atom_centers,
    ) = integers
    if dtype not in {torch.float32, torch.float64}:
        raise TypeError("scheduled role density requires float32 or float64")
    if radial_values.ndim != 2 or angular_values.ndim != 2:
        raise ValueError("radial_values and angular_values must be matrices")
    edge_count = int(distances.numel())
    if any(value.ndim != 1 for value in floating[1:] + integers):
        raise ValueError("scheduled role-density metadata must be vectors")
    if (
        int(radial_values.shape[0]) != edge_count
        or int(angular_values.shape[0]) != edge_count
        or int(cutoffs.numel()) != edge_count
        or int(soft_weights.numel()) != edge_count
        or int(edge_types.numel()) != edge_count
        or int(atom_centers.numel()) != edge_count
    ):
        raise ValueError("scheduled role-density edge counts must match")
    channel_count = int(channel_scales.numel())
    if (
        channel_count <= 0
        or int(channel_radial_indices.numel()) != channel_count
        or int(channel_angular_indices.numel()) != channel_count
        or int(channel_types.numel()) != channel_count
    ):
        raise ValueError("scheduled role-density channel schedules must match")
    if int(filter_centers.numel()) <= 0 or float(filter_width) <= 0.0:
        raise ValueError("scheduled role density requires roles and positive width")
    if int(atom_count) <= 0:
        raise ValueError("atom_count must be positive")
    if any(
        bool(value.requires_grad)
        for value in (cutoffs, filter_centers, channel_scales)
    ):
        raise ValueError(
            "cutoffs, filter centers, and channel scales are fixed plan data"
        )
    return (
        radial_values,
        angular_values,
        distances,
        cutoffs,
        filter_centers,
        float(filter_width),
        soft_weights,
        edge_types,
        channel_radial_indices,
        channel_angular_indices,
        channel_types,
        channel_scales,
        atom_centers,
        int(atom_count),
    )


def _scheduled_role_density_reference(*prepared, deterministic=False):
    (
        radial_values,
        angular_values,
        distances,
        cutoffs,
        filter_centers,
        filter_width,
        soft_weights,
        edge_types,
        channel_radial_indices,
        channel_angular_indices,
        channel_types,
        channel_scales,
        atom_centers,
        atom_count,
    ) = prepared
    selected_radial = radial_values.index_select(
        1,
        channel_radial_indices,
    )
    selected_angular = angular_values.index_select(
        1,
        channel_angular_indices,
    )
    edge_values = (
        selected_radial
        * selected_angular
        * channel_scales.reshape(1, -1)
    )
    type_mask = (
        channel_types.reshape(1, -1) < 0
    ) | (
        channel_types.reshape(1, -1)
        == edge_types.reshape(-1, 1)
    )
    edge_values = edge_values * type_mask.to(dtype=edge_values.dtype)
    packed = torch.cat((edge_values, soft_weights.reshape(-1, 1)), dim=1)
    scaled = distances.reshape(-1, 1) / cutoffs.reshape(-1, 1)
    logits = -0.5 * (
        (scaled - filter_centers.reshape(1, -1)) / filter_width
    ).pow(2)
    filters = torch.softmax(logits, dim=1)
    contributions = (filters.unsqueeze(2) * packed.unsqueeze(1)).reshape(
        int(distances.numel()),
        -1,
    )
    if deterministic:
        return _deterministic_segment_sum(
            contributions,
            atom_centers,
            int(atom_count),
        ).reshape(
            int(atom_count),
            int(filter_centers.numel()),
            int(packed.shape[1]),
        )
    output = radial_values.new_zeros(
        (int(atom_count), int(filter_centers.numel()), int(packed.shape[1]))
    )
    output.reshape(int(atom_count), -1).index_add_(
        0,
        atom_centers,
        contributions,
    )
    return output


def scheduled_softmax_gaussian_role_density(
    radial_values,
    angular_values,
    distances,
    cutoffs,
    filter_centers,
    filter_width,
    soft_weights,
    edge_types,
    channel_radial_indices,
    channel_angular_indices,
    channel_types,
    channel_scales,
    atom_centers,
    atom_count,
    *,
    backend="auto",
):
    """Fuse scheduled radial/angular channels into lifted role density."""

    backend = str(backend)
    if backend not in {"auto", "native", "reference", "deterministic"}:
        raise ValueError(
            "backend must be 'auto', 'native', 'reference', or 'deterministic'"
        )
    prepared = _prepare_scheduled_role_density_inputs(
        radial_values,
        angular_values,
        distances,
        cutoffs,
        filter_centers,
        filter_width,
        soft_weights,
        edge_types,
        channel_radial_indices,
        channel_angular_indices,
        channel_types,
        channel_scales,
        atom_centers,
        atom_count,
    )
    require_native = _enabled("YE3T_REQUIRE_NATIVE")
    if backend in {"reference", "deterministic"} and require_native:
        raise RuntimeError("YE3T_REQUIRE_NATIVE=1 forbids reference fallback")
    use_native = backend == "native" or require_native
    use_native = use_native or (
        backend == "auto"
        and (
            _prebuilt_extension() is not None
            or _enabled("YE3T_ENABLE_EXECUTION_PLAN_JIT")
        )
    )
    if use_native:
        extension = _extension_for_native_dispatch()
        if (
            prepared[0].device.type == "cuda"
            and not _native_cuda_dispatch_available(extension)
        ):
            if backend == "auto" and not require_native:
                use_native = False
            else:
                raise RuntimeError(
                    "the installed extension has no scheduled role-density CUDA dispatch"
                )
        elif prepared[0].device.type not in {"cpu", "cuda"}:
            raise ValueError("native scheduled role density requires CPU or CUDA")
        if use_native:
            return torch.ops.ye3t_runtime.scheduled_softmax_gaussian_role_density(
                *(value.contiguous() if torch.is_tensor(value) else value for value in prepared)
            )
    if backend == "deterministic":
        return _scheduled_role_density_reference(
            *prepared,
            deterministic=True,
        )
    return _scheduled_role_density_reference(*prepared)


def _prepare_carrier_gated_scatter_inputs(
    node_values,
    edge_gates,
    edge_sources,
    edge_targets,
    feature_channels,
    target_count,
):
    node_values = torch.as_tensor(node_values)
    edge_gates = torch.as_tensor(
        edge_gates,
        device=node_values.device,
    )
    if node_values.is_complex():
        real_dtype = (
            torch.float32
            if node_values.dtype == torch.complex64
            else torch.float64
        )
        edge_gates = edge_gates.to(
            dtype=(
                node_values.dtype
                if edge_gates.is_complex()
                else real_dtype
            )
        )
    else:
        edge_gates = edge_gates.to(dtype=node_values.dtype)
    edge_sources = torch.as_tensor(
        edge_sources,
        dtype=torch.int64,
        device=node_values.device,
    )
    edge_targets = torch.as_tensor(
        edge_targets,
        dtype=torch.int64,
        device=node_values.device,
    )
    feature_channels = torch.as_tensor(
        feature_channels,
        dtype=torch.int64,
        device=node_values.device,
    )
    if node_values.ndim != 2 or edge_gates.ndim != 2:
        raise ValueError(
            "node_values and edge_gates must be two-dimensional"
        )
    if (
        edge_sources.ndim != 1
        or edge_targets.ndim != 1
        or int(edge_sources.numel()) != int(edge_gates.shape[0])
        or int(edge_targets.numel()) != int(edge_gates.shape[0])
    ):
        raise ValueError(
            "edge_sources and edge_targets must contain one index per edge"
        )
    if (
        feature_channels.ndim != 1
        or int(feature_channels.numel()) != int(node_values.shape[1])
    ):
        raise ValueError(
            "feature_channels must contain one channel index per feature"
        )
    if node_values.dtype not in {
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    }:
        raise TypeError(
            "carrier gated scatter requires floating or complex inputs"
        )
    if int(node_values.shape[0]) <= 0:
        raise ValueError("node_values must contain at least one node")
    if int(edge_gates.shape[1]) <= 0:
        raise ValueError("edge_gates must contain at least one channel")
    if int(target_count) <= 0:
        raise ValueError("target_count must be positive")
    return (
        node_values,
        edge_gates,
        edge_sources,
        edge_targets,
        feature_channels,
        int(target_count),
    )


def _carrier_gated_scatter_use_native(
    backend,
    node_values,
    edge_count,
    require_native,
):
    if backend == "native" or require_native:
        return True
    if backend != "auto":
        return False
    threshold = (
        _NATIVE_CUDA_CARRIER_GATED_SCATTER_MIN_WORK_ITEMS
        if node_values.device.type == "cuda"
        else _NATIVE_CPU_CARRIER_GATED_SCATTER_MIN_WORK_ITEMS
    )
    return bool(
        threshold is not None
        and int(edge_count * node_values.shape[1]) >= int(threshold)
        and (
            _prebuilt_extension() is not None
            or _enabled("YE3T_ENABLE_EXECUTION_PLAN_JIT")
        )
    )


def carrier_gated_scatter(
    node_values,
    edge_gates,
    edge_sources,
    edge_targets,
    feature_channels,
    target_count,
    *,
    backend="auto",
):
    """Scatter channel-gated carriers without mixing tableau or M axes."""

    backend = str(backend)
    if backend not in {"auto", "native", "reference", "deterministic"}:
        raise ValueError(
            "backend must be 'auto', 'native', 'reference', or 'deterministic'"
        )
    require_native = _enabled("YE3T_REQUIRE_NATIVE")
    if backend in {"reference", "deterministic"} and require_native:
        raise RuntimeError(
            "YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback"
        )
    (
        node_values,
        edge_gates,
        edge_sources,
        edge_targets,
        feature_channels,
        target_count,
    ) = _prepare_carrier_gated_scatter_inputs(
        node_values,
        edge_gates,
        edge_sources,
        edge_targets,
        feature_channels,
        target_count,
    )
    use_native = _carrier_gated_scatter_use_native(
        backend,
        node_values,
        int(edge_gates.shape[0]),
        require_native,
    )
    if use_native:
        extension = _extension_for_native_dispatch()
        if (
            node_values.device.type == "cuda"
            and not _native_cuda_dispatch_available(extension)
        ):
            if backend == "auto" and not require_native:
                use_native = False
            else:
                raise RuntimeError(
                    "the installed YE3T native extension has no CUDA "
                    "carrier-gated scatter operators"
                )
        elif node_values.device.type not in {"cpu", "cuda"}:
            raise ValueError(
                "native carrier-gated scatter requires CPU or CUDA"
            )
        if use_native:
            return torch.ops.ye3t_runtime.carrier_gated_scatter(
                node_values.contiguous(),
                edge_gates.contiguous(),
                edge_sources.contiguous(),
                edge_targets.contiguous(),
                feature_channels.contiguous(),
                target_count,
            )
    source_values = node_values.index_select(0, edge_sources)
    feature_gates = edge_gates.index_select(1, feature_channels)
    contributions = source_values * feature_gates
    if backend == "deterministic":
        return _deterministic_segment_sum(
            contributions,
            edge_targets,
            target_count,
        )
    output = node_values.new_zeros(
        (target_count, int(node_values.shape[1]))
    )
    output.index_add_(
        0,
        edge_targets,
        contributions,
    )
    return output


def carrier_gated_scatter_adjoint(
    target_adjoint,
    node_values,
    edge_gates,
    edge_sources,
    edge_targets,
    feature_channels,
    *,
    backend="auto",
):
    """Apply the adjoint of channel-gated carrier scatter."""

    target_adjoint = torch.as_tensor(target_adjoint)
    if target_adjoint.ndim != 2:
        raise ValueError("target_adjoint must be two-dimensional")
    (
        node_values,
        edge_gates,
        edge_sources,
        edge_targets,
        feature_channels,
        target_count,
    ) = _prepare_carrier_gated_scatter_inputs(
        node_values,
        edge_gates,
        edge_sources,
        edge_targets,
        feature_channels,
        int(target_adjoint.shape[0]),
    )
    target_adjoint = target_adjoint.to(
        dtype=node_values.dtype,
        device=node_values.device,
    )
    if int(target_adjoint.shape[1]) != int(node_values.shape[1]):
        raise ValueError(
            "target_adjoint width must match node_values"
        )
    backend = str(backend)
    if backend not in {"auto", "native", "reference", "deterministic"}:
        raise ValueError(
            "backend must be 'auto', 'native', 'reference', or 'deterministic'"
        )
    require_native = _enabled("YE3T_REQUIRE_NATIVE")
    if backend in {"reference", "deterministic"} and require_native:
        raise RuntimeError(
            "YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback"
        )
    use_native = _carrier_gated_scatter_use_native(
        backend,
        node_values,
        int(edge_gates.shape[0]),
        require_native,
    )
    if use_native:
        extension = _load_extension()
        if (
            node_values.device.type == "cuda"
            and not bool(extension.has_cuda())
        ):
            if backend == "auto" and not require_native:
                use_native = False
            else:
                raise RuntimeError(
                    "the installed YE3T native extension has no CUDA "
                    "carrier-gated scatter adjoint"
                )
        elif node_values.device.type not in {"cpu", "cuda"}:
            raise ValueError(
                "native carrier-gated scatter requires CPU or CUDA"
            )
        if use_native:
            return torch.ops.ye3t_runtime.carrier_gated_scatter_adjoint(
                target_adjoint.contiguous(),
                node_values.contiguous(),
                edge_gates.contiguous(),
                edge_sources.contiguous(),
                edge_targets.contiguous(),
                feature_channels.contiguous(),
            )
    gathered = target_adjoint.index_select(0, edge_targets)
    feature_gates = edge_gates.index_select(1, feature_channels)
    source_values = node_values.index_select(0, edge_sources)
    node_contributions = gathered * feature_gates.conj()
    gate_contributions = gathered * source_values.conj()
    if not edge_gates.is_complex():
        gate_contributions = gate_contributions.real
    if backend == "deterministic":
        node_adjoint = _deterministic_segment_sum(
            node_contributions,
            edge_sources,
            int(node_values.shape[0]),
        )
        gate_adjoint = _deterministic_segment_sum(
            gate_contributions.transpose(0, 1),
            feature_channels,
            int(edge_gates.shape[1]),
        ).transpose(0, 1)
        return node_adjoint, gate_adjoint
    node_adjoint = torch.zeros_like(node_values)
    node_adjoint.index_add_(0, edge_sources, node_contributions)
    gate_adjoint = torch.zeros_like(edge_gates)
    gate_adjoint.index_add_(
        1,
        feature_channels,
        gate_contributions,
    )
    return node_adjoint, gate_adjoint


def _native_residual_carrier_scatter_available(extension):
    return bool(
        extension is not None
        and int(extension.core_abi_version()) >= 30
        and hasattr(
            torch.ops.ye3t_runtime,
            "carrier_residual_gated_scatter",
        )
    )


def carrier_residual_gated_scatter(
    node_values,
    edge_gates,
    edge_sources,
    edge_targets,
    feature_channels,
    *,
    backend="auto",
):
    """Scatter invariant-gated carriers directly into their residual state."""

    backend = str(backend)
    if backend not in {"auto", "native", "reference", "deterministic"}:
        raise ValueError(
            "backend must be 'auto', 'native', 'reference', or 'deterministic'"
        )
    require_native = _enabled("YE3T_REQUIRE_NATIVE")
    if backend in {"reference", "deterministic"} and require_native:
        raise RuntimeError(
            "YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback"
        )
    (
        node_values,
        edge_gates,
        edge_sources,
        edge_targets,
        feature_channels,
        target_count,
    ) = _prepare_carrier_gated_scatter_inputs(
        node_values,
        edge_gates,
        edge_sources,
        edge_targets,
        feature_channels,
        int(torch.as_tensor(node_values).shape[0]),
    )
    use_native = _carrier_gated_scatter_use_native(
        backend,
        node_values,
        int(edge_gates.shape[0]),
        require_native,
    )
    if use_native:
        extension = _extension_for_native_dispatch()
        if not _native_residual_carrier_scatter_available(extension):
            if backend == "auto" and not require_native:
                use_native = False
            else:
                raise RuntimeError(
                    "the installed YE3T native extension has no residual "
                    "carrier-gated scatter operators"
                )
        elif (
            node_values.device.type == "cuda"
            and not _native_cuda_dispatch_available(extension)
        ):
            if backend == "auto" and not require_native:
                use_native = False
            else:
                raise RuntimeError(
                    "the installed YE3T native extension has no CUDA "
                    "residual carrier-gated scatter operators"
                )
        elif node_values.device.type not in {"cpu", "cuda"}:
            raise ValueError(
                "native residual carrier scatter requires CPU or CUDA"
            )
        if use_native:
            return torch.ops.ye3t_runtime.carrier_residual_gated_scatter(
                node_values.contiguous(),
                edge_gates.contiguous(),
                edge_sources.contiguous(),
                edge_targets.contiguous(),
                feature_channels.contiguous(),
            )
    messages = carrier_gated_scatter(
        node_values,
        edge_gates,
        edge_sources,
        edge_targets,
        feature_channels,
        target_count,
        backend=backend,
    )
    return node_values + messages


def carrier_residual_gated_scatter_adjoint(
    target_adjoint,
    node_values,
    edge_gates,
    edge_sources,
    edge_targets,
    feature_channels,
    *,
    backend="auto",
):
    """Apply the adjoint of residual invariant-gated carrier scatter."""

    target_adjoint = torch.as_tensor(target_adjoint)
    (
        node_values,
        edge_gates,
        edge_sources,
        edge_targets,
        feature_channels,
        _target_count,
    ) = _prepare_carrier_gated_scatter_inputs(
        node_values,
        edge_gates,
        edge_sources,
        edge_targets,
        feature_channels,
        int(target_adjoint.shape[0]),
    )
    target_adjoint = target_adjoint.to(
        dtype=node_values.dtype,
        device=node_values.device,
    )
    if target_adjoint.shape != node_values.shape:
        raise ValueError(
            "residual scatter target_adjoint must match node_values"
        )
    backend = str(backend)
    if backend not in {"auto", "native", "reference", "deterministic"}:
        raise ValueError(
            "backend must be 'auto', 'native', 'reference', or 'deterministic'"
        )
    require_native = _enabled("YE3T_REQUIRE_NATIVE")
    if backend in {"reference", "deterministic"} and require_native:
        raise RuntimeError(
            "YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback"
        )
    use_native = _carrier_gated_scatter_use_native(
        backend,
        node_values,
        int(edge_gates.shape[0]),
        require_native,
    )
    if use_native:
        extension = _extension_for_native_dispatch()
        if not _native_residual_carrier_scatter_available(extension):
            if backend == "auto" and not require_native:
                use_native = False
            else:
                raise RuntimeError(
                    "the installed YE3T native extension has no residual "
                    "carrier-gated scatter adjoint"
                )
        elif (
            node_values.device.type == "cuda"
            and not _native_cuda_dispatch_available(extension)
        ):
            if backend == "auto" and not require_native:
                use_native = False
            else:
                raise RuntimeError(
                    "the installed YE3T native extension has no CUDA "
                    "residual carrier-gated scatter adjoint"
                )
        if use_native:
            return torch.ops.ye3t_runtime.carrier_residual_gated_scatter_adjoint(
                target_adjoint.contiguous(),
                node_values.contiguous(),
                edge_gates.contiguous(),
                edge_sources.contiguous(),
                edge_targets.contiguous(),
                feature_channels.contiguous(),
            )
    node_adjoint, gate_adjoint = carrier_gated_scatter_adjoint(
        target_adjoint,
        node_values,
        edge_gates,
        edge_sources,
        edge_targets,
        feature_channels,
        backend=backend,
    )
    return target_adjoint + node_adjoint, gate_adjoint


def prepare_carrier_scatter_segments(
    edge_sources,
    edge_targets,
    feature_channels,
    node_count,
    channel_count,
):
    """Build exact target/source/channel segments for carrier scatter."""

    edge_sources = torch.as_tensor(edge_sources, dtype=torch.int64)
    edge_targets = torch.as_tensor(
        edge_targets,
        dtype=torch.int64,
        device=edge_sources.device,
    )
    feature_channels = torch.as_tensor(
        feature_channels,
        dtype=torch.int64,
        device=edge_sources.device,
    )
    node_count = int(node_count)
    channel_count = int(channel_count)
    if node_count <= 0 or channel_count <= 0:
        raise ValueError("node_count and channel_count must be positive")
    if (
        edge_sources.ndim != 1
        or edge_targets.ndim != 1
        or edge_sources.shape != edge_targets.shape
    ):
        raise ValueError(
            "edge_sources and edge_targets must be equal-length vectors"
        )
    if feature_channels.ndim != 1 or int(feature_channels.numel()) <= 0:
        raise ValueError(
            "feature_channels must contain at least one feature"
        )
    if bool(
        torch.any(
            (edge_sources < 0) | (edge_sources >= node_count)
        ).item()
    ):
        raise ValueError("edge_sources contains an out-of-range node")
    if bool(
        torch.any(
            (edge_targets < 0) | (edge_targets >= node_count)
        ).item()
    ):
        raise ValueError("edge_targets contains an out-of-range node")
    if bool(
        torch.any(
            (feature_channels < 0)
            | (feature_channels >= channel_count)
        ).item()
    ):
        raise ValueError(
            "feature_channels contains an out-of-range channel"
        )
    edge_key = edge_targets * node_count + edge_sources
    edge_order = torch.argsort(edge_key, stable=True)
    sorted_sources = edge_sources.index_select(0, edge_order)
    sorted_targets = edge_targets.index_select(0, edge_order)
    target_counts = torch.bincount(
        sorted_targets,
        minlength=node_count,
    )
    target_offsets = torch.cat(
        (
            torch.zeros(
                1,
                dtype=torch.int64,
                device=edge_sources.device,
            ),
            torch.cumsum(target_counts, dim=0),
        )
    )
    source_edges = torch.argsort(sorted_sources, stable=True)
    source_counts = torch.bincount(
        sorted_sources,
        minlength=node_count,
    )
    source_offsets = torch.cat(
        (
            torch.zeros(
                1,
                dtype=torch.int64,
                device=edge_sources.device,
            ),
            torch.cumsum(source_counts, dim=0),
        )
    )
    channel_features = torch.argsort(feature_channels, stable=True)
    channel_counts = torch.bincount(
        feature_channels,
        minlength=channel_count,
    )
    channel_offsets = torch.cat(
        (
            torch.zeros(
                1,
                dtype=torch.int64,
                device=edge_sources.device,
            ),
            torch.cumsum(channel_counts, dim=0),
        )
    )
    return {
        "schema": "ye3t_carrier_scatter_segments_v1",
        "edge_order": edge_order.contiguous(),
        "edge_sources": sorted_sources.contiguous(),
        "edge_targets": sorted_targets.contiguous(),
        "target_offsets": target_offsets.contiguous(),
        "source_offsets": source_offsets.contiguous(),
        "source_edges": source_edges.contiguous(),
        "feature_channels": feature_channels.contiguous(),
        "channel_offsets": channel_offsets.contiguous(),
        "channel_features": channel_features.contiguous(),
        "node_count": node_count,
        "channel_count": channel_count,
    }


def _prepare_segmented_carrier_scatter_inputs(
    node_values,
    edge_gates,
    segments,
):
    if not isinstance(segments, dict):
        raise TypeError("segments must be a compiled segment dictionary")
    if segments.get("schema") != "ye3t_carrier_scatter_segments_v1":
        raise ValueError("unsupported carrier scatter segment schema")
    required = (
        "edge_sources",
        "edge_targets",
        "target_offsets",
        "source_offsets",
        "source_edges",
        "feature_channels",
        "channel_offsets",
        "channel_features",
        "node_count",
        "channel_count",
    )
    missing = tuple(key for key in required if key not in segments)
    if missing:
        raise ValueError(
            "carrier scatter segments are missing " + ", ".join(missing)
        )
    prepared = _prepare_carrier_gated_scatter_inputs(
        node_values,
        edge_gates,
        segments["edge_sources"],
        segments["edge_targets"],
        segments["feature_channels"],
        segments["node_count"],
    )
    (
        node_values,
        edge_gates,
        edge_sources,
        edge_targets,
        feature_channels,
        target_count,
    ) = prepared
    if target_count != int(node_values.shape[0]):
        raise ValueError(
            "segmented residual scatter requires one target per node"
        )
    if int(segments["channel_count"]) != int(edge_gates.shape[1]):
        raise ValueError(
            "segment channel_count must match edge_gates width"
        )
    index_names = (
        "target_offsets",
        "source_offsets",
        "source_edges",
        "channel_offsets",
        "channel_features",
    )
    index_tensors = tuple(
        torch.as_tensor(
            segments[name],
            dtype=torch.int64,
            device=node_values.device,
        ).contiguous()
        for name in index_names
    )
    (
        target_offsets,
        source_offsets,
        source_edges,
        channel_offsets,
        channel_features,
    ) = index_tensors
    if (
        int(target_offsets.numel()) != int(node_values.shape[0]) + 1
        or int(source_offsets.numel()) != int(node_values.shape[0]) + 1
        or int(source_edges.numel()) != int(edge_gates.shape[0])
        or int(channel_offsets.numel()) != int(edge_gates.shape[1]) + 1
        or int(channel_features.numel()) != int(node_values.shape[1])
    ):
        raise ValueError(
            "carrier scatter segment dimensions do not match primals"
        )
    return (
        node_values,
        edge_gates,
        edge_sources,
        edge_targets,
        target_offsets,
        source_offsets,
        source_edges,
        feature_channels,
        channel_offsets,
        channel_features,
    )


def _native_segmented_carrier_scatter_available(extension):
    return bool(
        extension is not None
        and int(extension.core_abi_version()) >= 31
        and hasattr(
            torch.ops.ye3t_runtime,
            "carrier_segmented_residual_gated_scatter",
        )
    )


def carrier_segmented_residual_gated_scatter(
    node_values,
    edge_gates,
    segments,
    *,
    backend="native",
):
    """Apply a presegmented residual carrier graph update."""

    backend = str(backend)
    if backend not in {"native", "reference", "deterministic"}:
        raise ValueError(
            "segmented scatter backend must be native, reference, or deterministic"
        )
    prepared = _prepare_segmented_carrier_scatter_inputs(
        node_values,
        edge_gates,
        segments,
    )
    if backend == "native":
        extension = _extension_for_native_dispatch()
        if not _native_segmented_carrier_scatter_available(extension):
            raise RuntimeError(
                "the installed YE3T native extension has no segmented "
                "carrier-gated scatter operators"
            )
        if (
            prepared[0].device.type == "cuda"
            and not _native_cuda_dispatch_available(extension)
        ):
            raise RuntimeError(
                "the installed YE3T native extension has no CUDA "
                "segmented carrier-gated scatter operators"
            )
        if prepared[0].device.type not in {"cpu", "cuda"}:
            raise ValueError(
                "native segmented carrier scatter requires CPU or CUDA"
            )
        operator = (
            torch.ops.ye3t_runtime
            .carrier_segmented_residual_gated_scatter
        )
        return operator(*(value.contiguous() for value in prepared))
    return carrier_residual_gated_scatter(
        prepared[0],
        prepared[1],
        prepared[2],
        prepared[3],
        prepared[7],
        backend=backend,
    )


def carrier_segmented_residual_gated_scatter_adjoint(
    target_adjoint,
    node_values,
    edge_gates,
    segments,
    *,
    backend="native",
):
    """Apply the adjoint of a presegmented residual graph update."""

    prepared = _prepare_segmented_carrier_scatter_inputs(
        node_values,
        edge_gates,
        segments,
    )
    target_adjoint = torch.as_tensor(
        target_adjoint,
        dtype=prepared[0].dtype,
        device=prepared[0].device,
    )
    if target_adjoint.shape != prepared[0].shape:
        raise ValueError("target_adjoint must match node_values")
    backend = str(backend)
    if backend == "native":
        extension = _extension_for_native_dispatch()
        if not _native_segmented_carrier_scatter_available(extension):
            raise RuntimeError(
                "the installed YE3T native extension has no segmented "
                "carrier-gated scatter adjoint"
            )
        operator = (
            torch.ops.ye3t_runtime
            .carrier_segmented_residual_gated_scatter_adjoint
        )
        return operator(
            target_adjoint.contiguous(),
            *(value.contiguous() for value in prepared),
        )
    if backend not in {"reference", "deterministic"}:
        raise ValueError(
            "segmented scatter backend must be native, reference, or deterministic"
        )
    return carrier_residual_gated_scatter_adjoint(
        target_adjoint,
        prepared[0],
        prepared[1],
        prepared[2],
        prepared[3],
        prepared[7],
        backend=backend,
    )


def prepare_source_arena_schedule(
    gather_indices,
    center_types=None,
    producer_width=None,
):
    """Compile an exact gather and deterministic reverse CSR schedule."""

    gather_values = tuple(
        int(value)
        for value in torch.as_tensor(
            gather_indices,
            dtype=torch.int64,
            device="cpu",
        ).reshape(-1).tolist()
    )
    if not gather_values:
        raise ValueError("source arena gather_indices may not be empty")
    if producer_width is None:
        producer_width = max(gather_values) + 1
    producer_width = int(producer_width)
    if producer_width <= 0:
        raise ValueError("source arena producer_width must be positive")
    if any(
        value < 0 or value >= producer_width
        for value in gather_values
    ):
        raise ValueError(
            "source arena gather_indices contains an out-of-range coordinate"
        )
    if center_types is None:
        center_values = (-1,) * len(gather_values)
    else:
        center_values = tuple(
            int(value)
            for value in torch.as_tensor(
                center_types,
                dtype=torch.int64,
                device="cpu",
            ).reshape(-1).tolist()
        )
    if len(center_values) != len(gather_values):
        raise ValueError(
            "source arena center_types must match gather_indices"
        )
    if any(value < -1 for value in center_values):
        raise ValueError(
            "source arena center_types must be -1 or nonnegative"
        )
    reverse_rows = [[] for _ in range(producer_width)]
    for output_index, producer_index in enumerate(gather_values):
        reverse_rows[producer_index].append(int(output_index))
    reverse_offsets = [0]
    reverse_output_indices = []
    for row in reverse_rows:
        reverse_output_indices.extend(row)
        reverse_offsets.append(len(reverse_output_indices))
    return {
        "schema": "ye3t_source_arena_schedule_v1",
        "producer_width": int(producer_width),
        "output_width": int(len(gather_values)),
        "gather_indices": gather_values,
        "reverse_offsets": tuple(reverse_offsets),
        "reverse_output_indices": tuple(reverse_output_indices),
        "center_types": center_values,
    }


def _prepare_source_arena_inputs(producer, schedule, atom_types):
    producer = torch.as_tensor(producer)
    if producer.ndim != 2:
        raise ValueError("source arena producer must be two-dimensional")
    if producer.dtype not in {
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    }:
        raise TypeError(
            "source arena producer requires floating or complex values"
        )
    if not isinstance(schedule, dict):
        raise TypeError("source arena schedule must be a dictionary")
    if str(schedule.get("schema")) != "ye3t_source_arena_schedule_v1":
        raise ValueError("unsupported source arena schedule schema")
    producer_width = int(schedule.get("producer_width", -1))
    output_width = int(schedule.get("output_width", -1))
    if producer_width != int(producer.shape[1]):
        raise ValueError(
            "source arena schedule producer width does not match producer"
        )
    names = (
        "gather_indices",
        "reverse_offsets",
        "reverse_output_indices",
        "center_types",
    )
    tensors = []
    cpu_values = {}
    for name in names:
        if name not in schedule:
            raise ValueError("source arena schedule is missing " + name)
        original = torch.as_tensor(schedule[name], dtype=torch.int64)
        if original.ndim != 1:
            raise ValueError(name + " must be one-dimensional")
        if original.device.type == "cpu":
            cpu_values[name] = tuple(
                int(value) for value in original.tolist()
            )
        tensors.append(original.to(device=producer.device))
    (
        gather_indices,
        reverse_offsets,
        reverse_output_indices,
        center_types,
    ) = tensors
    if (
        int(gather_indices.numel()) != output_width
        or int(center_types.numel()) != output_width
        or int(reverse_offsets.numel()) != producer_width + 1
        or int(reverse_output_indices.numel()) != output_width
    ):
        raise ValueError("source arena schedule dimensions are inconsistent")
    if len(cpu_values) == len(names):
        canonical = prepare_source_arena_schedule(
            cpu_values["gather_indices"],
            cpu_values["center_types"],
            producer_width=producer_width,
        )
        for name in names:
            if tuple(canonical[name]) != tuple(cpu_values[name]):
                raise ValueError(
                    "source arena reverse schedule is not the exact CSR of "
                    "gather_indices"
                )
    if atom_types is None:
        atom_types = torch.zeros(
            int(producer.shape[0]),
            dtype=torch.int64,
            device=producer.device,
        )
    else:
        atom_types = torch.as_tensor(
            atom_types,
            dtype=torch.int64,
            device=producer.device,
        ).reshape(-1)
    if int(atom_types.numel()) != int(producer.shape[0]):
        raise ValueError(
            "atom_types must contain one entry per source arena row"
        )
    return (
        producer,
        gather_indices,
        reverse_offsets,
        reverse_output_indices,
        center_types,
        atom_types,
    )


def _native_source_arena_available(extension):
    return bool(
        (extension is None or int(extension.core_abi_version()) >= 34)
        and hasattr(torch.ops.ye3t_runtime, "source_arena_gather")
        and hasattr(torch.ops.ye3t_runtime, "source_arena_gather_adjoint")
        and hasattr(
            torch.ops.ye3t_runtime,
            "source_arena_gather_double_backward",
        )
    )


def _source_arena_reference(prepared):
    (
        producer,
        gather_indices,
        reverse_offsets,
        reverse_output_indices,
        center_types,
        atom_types,
    ) = prepared
    del reverse_offsets, reverse_output_indices
    active = (center_types < 0).reshape(1, -1) | (
        atom_types.reshape(-1, 1) == center_types.reshape(1, -1)
    )
    return producer.index_select(1, gather_indices) * active


def _source_arena_adjoint_reference(prepared):
    (
        output_adjoint,
        gather_indices,
        reverse_offsets,
        reverse_output_indices,
        center_types,
        atom_types,
    ) = prepared
    producer_width = int(reverse_offsets.numel()) - 1
    reverse_offsets_values = tuple(
        int(value) for value in reverse_offsets.detach().cpu().tolist()
    )
    reverse_output_values = tuple(
        int(value)
        for value in reverse_output_indices.detach().cpu().tolist()
    )
    active = (center_types < 0).reshape(1, -1) | (
        atom_types.reshape(-1, 1) == center_types.reshape(1, -1)
    )
    masked = output_adjoint * active
    outputs = []
    for producer_index in range(producer_width):
        start = reverse_offsets_values[producer_index]
        stop = reverse_offsets_values[producer_index + 1]
        if start == stop:
            outputs.append(output_adjoint.new_zeros(output_adjoint.shape[0]))
        else:
            indices = torch.as_tensor(
                reverse_output_values[start:stop],
                dtype=torch.int64,
                device=output_adjoint.device,
            )
            outputs.append(masked.index_select(1, indices).sum(dim=1))
    del gather_indices
    return torch.stack(tuple(outputs), dim=1)


def source_arena_gather(
    producer,
    schedule,
    atom_types=None,
    *,
    backend="auto",
):
    """Gather compiler-selected sources and apply the center-type mask."""

    prepared = _prepare_source_arena_inputs(
        producer,
        schedule,
        atom_types,
    )
    backend = str(backend)
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")
    require_native = _enabled("YE3T_REQUIRE_NATIVE")
    if backend == "reference" and require_native:
        raise RuntimeError(
            "YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback"
        )
    use_native = bool(
        backend == "native"
        or require_native
        or (backend == "auto" and prepared[0].device.type == "cuda")
    )
    if use_native:
        extension = _extension_for_native_dispatch()
        if not _native_source_arena_available(extension):
            if backend == "auto" and not require_native:
                use_native = False
            else:
                raise RuntimeError(
                    "the installed YE3T native extension has no source-arena "
                    "gather/adjoint runtime"
                )
        elif (
            prepared[0].device.type == "cuda"
            and not _native_cuda_dispatch_available(extension)
        ):
            if backend == "auto" and not require_native:
                use_native = False
            else:
                raise RuntimeError(
                    "the installed YE3T native extension has no CUDA "
                    "source-arena runtime"
                )
        if use_native:
            return torch.ops.ye3t_runtime.source_arena_gather(
                *(value.contiguous() for value in prepared)
            )
    return _source_arena_reference(prepared)


def source_arena_gather_adjoint(
    output_adjoint,
    schedule,
    atom_types=None,
    *,
    backend="auto",
):
    """Apply the deterministic reverse-CSR source-arena adjoint."""

    output_adjoint = torch.as_tensor(output_adjoint)
    producer_width = int(schedule.get("producer_width", -1))
    placeholder = output_adjoint.new_empty(
        (int(output_adjoint.shape[0]), producer_width)
    )
    prepared_forward = _prepare_source_arena_inputs(
        placeholder,
        schedule,
        atom_types,
    )
    if (
        output_adjoint.ndim != 2
        or int(output_adjoint.shape[1])
        != int(schedule.get("output_width", -1))
    ):
        raise ValueError(
            "output_adjoint must match the source arena output shape"
        )
    prepared = (output_adjoint,) + prepared_forward[1:]
    backend = str(backend)
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")
    require_native = _enabled("YE3T_REQUIRE_NATIVE")
    if backend == "reference" and require_native:
        raise RuntimeError(
            "YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback"
        )
    use_native = bool(
        backend == "native"
        or require_native
        or (backend == "auto" and output_adjoint.device.type == "cuda")
    )
    if use_native:
        extension = _extension_for_native_dispatch()
        if not _native_source_arena_available(extension):
            if backend == "auto" and not require_native:
                use_native = False
            else:
                raise RuntimeError(
                    "the installed YE3T native extension has no source-arena "
                    "gather/adjoint runtime"
                )
        elif (
            output_adjoint.device.type == "cuda"
            and not _native_cuda_dispatch_available(extension)
        ):
            if backend == "auto" and not require_native:
                use_native = False
            else:
                raise RuntimeError(
                    "the installed YE3T native extension has no CUDA "
                    "source-arena adjoint"
                )
        if use_native:
            return torch.ops.ye3t_runtime.source_arena_gather_adjoint(
                *(value.contiguous() for value in prepared)
            )
    return _source_arena_adjoint_reference(prepared)


def _prepare_source_arena_channel_transform_inputs(
    producer,
    channel_maps,
    schedule,
    atom_types,
):
    prepared_source = _prepare_source_arena_inputs(
        producer,
        schedule,
        atom_types,
    )
    producer = prepared_source[0]
    channel_maps = torch.as_tensor(
        channel_maps,
        device=producer.device,
    )
    if channel_maps.ndim != 1:
        raise ValueError("channel_maps must be a packed vector")
    if producer.is_complex():
        real_dtype = (
            torch.float32
            if producer.dtype == torch.complex64
            else torch.float64
        )
        if channel_maps.dtype not in {producer.dtype, real_dtype}:
            raise TypeError(
                "channel_maps must match producer dtype or use its real "
                "component dtype"
            )
    elif channel_maps.dtype != producer.dtype:
        raise TypeError("channel_maps must match producer dtype")
    names = (
        "input_feature_offsets",
        "output_feature_offsets",
        "input_channel_offsets",
        "output_channel_offsets",
        "map_offsets",
    )
    values = {}
    tensors = []
    for name in names:
        if name not in schedule:
            raise ValueError(
                "source-arena channel-transform schedule is missing " + name
            )
        original = torch.as_tensor(
            schedule[name], dtype=torch.int64
        ).reshape(-1)
        if original.device.type == "cpu":
            values[name] = tuple(int(value) for value in original.tolist())
        tensors.append(original.to(device=producer.device))
    offset_count = int(tensors[0].numel())
    if offset_count < 2 or any(
        int(tensor.numel()) != offset_count for tensor in tensors
    ):
        raise ValueError(
            "source-arena channel-transform offsets must share one "
            "block_count + 1 length"
        )
    if len(values) == len(names):
        if any(values[name][0] != 0 for name in names):
            raise ValueError(
                "source-arena channel-transform offsets must begin at zero"
            )
        if values["input_feature_offsets"][-1] != int(
            prepared_source[1].numel()
        ):
            raise ValueError(
                "source-arena channel-transform input offsets must cover "
                "the logical source arena"
            )
        if values["map_offsets"][-1] != int(channel_maps.numel()):
            raise ValueError(
                "source-arena channel-transform map offsets must cover "
                "channel_maps"
            )
        center_types = tuple(
            int(value) for value in prepared_source[4].tolist()
        )
        block_count = offset_count - 1
        for block in range(block_count):
            input_start = values["input_feature_offsets"][block]
            input_stop = values["input_feature_offsets"][block + 1]
            output_start = values["output_feature_offsets"][block]
            output_stop = values["output_feature_offsets"][block + 1]
            input_channels = (
                values["input_channel_offsets"][block + 1]
                - values["input_channel_offsets"][block]
            )
            output_channels = (
                values["output_channel_offsets"][block + 1]
                - values["output_channel_offsets"][block]
            )
            map_count = (
                values["map_offsets"][block + 1]
                - values["map_offsets"][block]
            )
            if (
                input_stop <= input_start
                or output_stop <= output_start
                or input_channels <= 0
                or output_channels <= 0
            ):
                raise ValueError(
                    "source-arena channel-transform blocks require "
                    "positive dimensions"
                )
            if (
                (input_stop - input_start) % input_channels
                or (output_stop - output_start) % output_channels
                or (input_stop - input_start) // input_channels
                != (output_stop - output_start) // output_channels
            ):
                raise ValueError(
                    "source-arena channel-transform blocks must preserve "
                    "complete carrier axes"
                )
            if output_channels > input_channels:
                raise ValueError(
                    "source-arena channel-transform blocks must be "
                    "non-expanding"
                )
            if map_count not in {0, input_channels * output_channels}:
                raise ValueError(
                    "source-arena channel-transform blocks require "
                    "rectangular maps or exact identity passthrough"
                )
            if map_count == 0 and input_channels != output_channels:
                raise ValueError(
                    "identity source-arena blocks must preserve channel "
                    "count"
                )
            if len(set(center_types[input_start:input_stop])) != 1:
                raise ValueError(
                    "one source-arena transform block cannot mix center "
                    "types"
                )
    declared_output_width = int(
        schedule.get("transformed_output_width", -1)
    )
    if declared_output_width <= 0 or (
        "output_feature_offsets" in values
        and declared_output_width
        != values["output_feature_offsets"][-1]
    ):
        raise ValueError(
            "source-arena channel-transform output width is inconsistent"
        )
    return (
        producer,
        channel_maps,
        *prepared_source[1:],
        *tensors,
    )


def _source_arena_channel_transform_reference(prepared):
    (
        producer,
        channel_maps,
        gather_indices,
        reverse_offsets,
        reverse_output_indices,
        center_types,
        atom_types,
        input_feature_offsets,
        output_feature_offsets,
        input_channel_offsets,
        output_channel_offsets,
        map_offsets,
    ) = prepared
    del reverse_offsets, reverse_output_indices
    gathered = producer.index_select(1, gather_indices)
    outputs = []
    block_count = int(input_feature_offsets.numel()) - 1
    for block in range(block_count):
        input_start = int(input_feature_offsets[block].item())
        input_stop = int(input_feature_offsets[block + 1].item())
        output_start = int(output_feature_offsets[block].item())
        output_stop = int(output_feature_offsets[block + 1].item())
        input_channels = int(
            input_channel_offsets[block + 1].item()
            - input_channel_offsets[block].item()
        )
        output_channels = int(
            output_channel_offsets[block + 1].item()
            - output_channel_offsets[block].item()
        )
        inner_width = (input_stop - input_start) // input_channels
        values = gathered[:, input_start:input_stop].reshape(
            int(producer.shape[0]), input_channels, inner_width
        )
        map_start = int(map_offsets[block].item())
        map_stop = int(map_offsets[block + 1].item())
        if map_start == map_stop:
            transformed = values
        else:
            matrix = channel_maps[map_start:map_stop].reshape(
                output_channels, input_channels
            )
            transformed = torch.einsum(
                "oi,bik->bok",
                matrix.to(dtype=values.dtype),
                values,
            )
        center_type = center_types[input_start]
        active = (center_type < 0) | (atom_types == center_type)
        outputs.append(
            transformed.reshape(
                int(producer.shape[0]), output_stop - output_start
            ) * active.reshape(-1, 1)
        )
    return torch.cat(tuple(outputs), dim=1)


def _native_source_arena_channel_transform_available(extension):
    return bool(
        (extension is None or int(extension.core_abi_version()) >= 37)
        and hasattr(
            torch.ops.ye3t_runtime,
            "source_arena_channel_transform",
        )
        and hasattr(
            torch.ops.ye3t_runtime,
            "source_arena_channel_transform_adjoint",
        )
        and hasattr(
            torch.ops.ye3t_runtime,
            "source_arena_channel_transform_double_backward",
        )
    )


def source_arena_channel_transform(
    producer,
    channel_maps,
    schedule,
    atom_types=None,
    *,
    backend="auto",
):
    """Fuse exact source reuse, center masking, and channel projection.

    Every block is a complete ``(N, lambda, L, parity)`` carrier.  The
    channel map acts only on multiplicity, while the tableau and magnetic
    axes are preserved exactly.  A zero-width map block is an explicit
    identity passthrough.
    """

    prepared = _prepare_source_arena_channel_transform_inputs(
        producer,
        channel_maps,
        schedule,
        atom_types,
    )
    backend = str(backend)
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")
    require_native = _enabled("YE3T_REQUIRE_NATIVE")
    if backend == "reference" and require_native:
        raise RuntimeError(
            "YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback"
        )
    use_native = bool(
        backend == "native"
        or require_native
        or (backend == "auto" and prepared[0].device.type == "cuda")
    )
    if use_native:
        extension = _extension_for_native_dispatch()
        if not _native_source_arena_channel_transform_available(extension):
            if backend == "auto" and not require_native:
                use_native = False
            else:
                raise RuntimeError(
                    "the installed YE3T native extension has no "
                    "source-arena channel-transform runtime"
                )
        elif (
            prepared[0].device.type == "cuda"
            and not _native_cuda_dispatch_available(extension)
        ):
            if backend == "auto" and not require_native:
                use_native = False
            else:
                raise RuntimeError(
                    "the installed YE3T native extension has no CUDA "
                    "source-arena channel-transform runtime"
                )
        if use_native:
            return torch.ops.ye3t_runtime.source_arena_channel_transform(
                *(value.contiguous() for value in prepared),
                int(schedule["transformed_output_width"]),
            )
    return _source_arena_channel_transform_reference(prepared)


def _prepare_carrier_channel_update_inputs(
    values,
    gates,
    channel_maps,
    feature_offsets,
    channel_offsets,
    map_offsets,
):
    values = torch.as_tensor(values)
    gates = torch.as_tensor(
        gates,
        device=values.device,
    )
    channel_maps = torch.as_tensor(
        channel_maps,
        device=values.device,
    )
    if values.is_complex():
        real_dtype = (
            torch.float32
            if values.dtype == torch.complex64
            else torch.float64
        )
        if gates.is_complex() != channel_maps.is_complex():
            raise TypeError(
                "gates and channel_maps must both be real or both be complex"
            )
        control_dtype = (
            values.dtype if gates.is_complex() else real_dtype
        )
        gates = gates.to(dtype=control_dtype)
        channel_maps = channel_maps.to(dtype=control_dtype)
    else:
        gates = gates.to(dtype=values.dtype)
        channel_maps = channel_maps.to(dtype=values.dtype)
    offset_tensors = []
    offset_values = []
    for name, offsets in (
        ("feature_offsets", feature_offsets),
        ("channel_offsets", channel_offsets),
        ("map_offsets", map_offsets),
    ):
        tensor = torch.as_tensor(offsets, dtype=torch.int64)
        if tensor.ndim != 1 or int(tensor.numel()) < 2:
            raise ValueError(
                name + " must be a one-dimensional block_count + 1 vector"
            )
        values_tuple = (
            tuple(int(value) for value in tensor.tolist())
            if tensor.device.type == "cpu"
            else None
        )
        offset_values.append(values_tuple)
        offset_tensors.append(tensor.to(device=values.device))
    (
        feature_offsets,
        channel_offsets,
        map_offsets,
    ) = offset_tensors
    if values.ndim != 2 or gates.ndim != 2:
        raise ValueError("values and gates must be two-dimensional")
    if channel_maps.ndim != 1:
        raise ValueError("channel_maps must be one-dimensional")
    if values.dtype not in {
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    }:
        raise TypeError(
            "carrier channel update requires floating or complex inputs"
        )
    if int(values.shape[0]) != int(gates.shape[0]):
        raise ValueError("values and gates must share the batch axis")
    if any(
        int(offset.numel()) != int(feature_offsets.numel())
        for offset in (channel_offsets, map_offsets)
    ):
        raise ValueError(
            "carrier update offsets must have one common length"
        )
    if all(items is not None for items in offset_values):
        feature_values, channel_values, map_values = offset_values
        if (
            feature_values[0] != 0
            or channel_values[0] != 0
            or map_values[0] != 0
            or feature_values[-1] != int(values.shape[1])
            or channel_values[-1] != int(gates.shape[1])
            or map_values[-1] != int(channel_maps.numel())
        ):
            raise ValueError(
                "carrier update offsets must start at zero and end at "
                "the packed tensor widths"
            )
        for block in range(len(feature_values) - 1):
            local_features = (
                feature_values[block + 1] - feature_values[block]
            )
            local_channels = (
                channel_values[block + 1] - channel_values[block]
            )
            local_maps = map_values[block + 1] - map_values[block]
            if (
                local_features <= 0
                or local_channels <= 0
                or local_features % local_channels
                or local_maps != local_channels * local_channels
            ):
                raise ValueError(
                    "each carrier update block requires a positive "
                    "[channel,t,M] width and one square channel map"
                )
    return (
        values,
        gates,
        channel_maps,
        feature_offsets,
        channel_offsets,
        map_offsets,
        tuple(offset_values),
    )


def _prepare_carrier_channel_transform_inputs(
    values,
    channel_maps,
    input_feature_offsets,
    output_feature_offsets,
    input_channel_offsets,
    output_channel_offsets,
    map_offsets,
):
    values = torch.as_tensor(values)
    channel_maps = torch.as_tensor(channel_maps, device=values.device)
    if values.is_complex():
        real_dtype = (
            torch.float32
            if values.dtype == torch.complex64
            else torch.float64
        )
        if channel_maps.dtype not in {values.dtype, real_dtype}:
            raise TypeError(
                "channel maps must match complex carriers or their real dtype"
            )
    else:
        channel_maps = channel_maps.to(dtype=values.dtype)
    offset_tensors = []
    offset_values = []
    for name, offsets in (
        ("input_feature_offsets", input_feature_offsets),
        ("output_feature_offsets", output_feature_offsets),
        ("input_channel_offsets", input_channel_offsets),
        ("output_channel_offsets", output_channel_offsets),
        ("map_offsets", map_offsets),
    ):
        tensor = torch.as_tensor(offsets, dtype=torch.int64)
        if tensor.ndim != 1 or int(tensor.numel()) < 2:
            raise ValueError(
                name + " must be a one-dimensional block_count + 1 vector"
            )
        offset_values.append(
            tuple(int(value) for value in tensor.tolist())
            if tensor.device.type == "cpu"
            else None
        )
        offset_tensors.append(tensor.to(device=values.device))
    if values.ndim != 2 or channel_maps.ndim != 1:
        raise ValueError("values must be 2D and channel_maps must be 1D")
    if values.dtype not in {
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    }:
        raise TypeError(
            "carrier channel transforms require floating or complex values"
        )
    if any(
        int(offset.numel()) != int(offset_tensors[0].numel())
        for offset in offset_tensors[1:]
    ):
        raise ValueError(
            "carrier channel-transform offsets must share one length"
        )
    if all(items is not None for items in offset_values):
        (
            input_features,
            output_features,
            input_channels,
            output_channels,
            map_values,
        ) = offset_values
        if (
            any(items[0] != 0 for items in offset_values)
            or input_features[-1] != int(values.shape[1])
            or map_values[-1] != int(channel_maps.numel())
        ):
            raise ValueError(
                "channel-transform offsets must start at zero and end at "
                "their packed input widths"
            )
        for block in range(len(input_features) - 1):
            input_width = input_features[block + 1] - input_features[block]
            output_width = output_features[block + 1] - output_features[block]
            input_count = input_channels[block + 1] - input_channels[block]
            output_count = output_channels[block + 1] - output_channels[block]
            map_count = map_values[block + 1] - map_values[block]
            if (
                min(input_width, output_width, input_count, output_count) <= 0
                or input_width % input_count
                or output_width % output_count
                or input_width // input_count
                != output_width // output_count
                or output_count > input_count
                or map_count != input_count * output_count
            ):
                raise ValueError(
                    "each channel-transform block requires matching complete "
                    "carrier component axes and a non-expanding rectangular map"
                )
    return (
        values,
        channel_maps,
        *offset_tensors,
        tuple(offset_values),
    )


def _carrier_channel_transform_reference(
    values,
    channel_maps,
    offset_values,
):
    (
        input_features,
        output_features,
        input_channels,
        output_channels,
        map_values,
    ) = offset_values
    if not all(isinstance(items, tuple) for items in offset_values):
        offset_values = tuple(
            tuple(int(value) for value in items.detach().cpu().tolist())
            for items in offset_values
        )
        (
            input_features,
            output_features,
            input_channels,
            output_channels,
            map_values,
        ) = offset_values
    outputs = []
    for block in range(len(input_features) - 1):
        input_count = input_channels[block + 1] - input_channels[block]
        output_count = output_channels[block + 1] - output_channels[block]
        local_values = values[
            :, input_features[block] : input_features[block + 1]
        ].reshape(int(values.shape[0]), input_count, -1)
        local_map = channel_maps[
            map_values[block] : map_values[block + 1]
        ].reshape(output_count, input_count)
        outputs.append(
            torch.einsum(
                "oi,biq->boq",
                local_map.to(dtype=local_values.dtype),
                local_values,
            ).reshape(int(values.shape[0]), -1)
        )
    return torch.cat(tuple(outputs), dim=1)


def _carrier_channel_transform_adjoint_reference(
    output_adjoint,
    values,
    channel_maps,
    offset_values,
):
    if not all(isinstance(items, tuple) for items in offset_values):
        offset_values = tuple(
            tuple(int(value) for value in items.detach().cpu().tolist())
            for items in offset_values
        )
    (
        input_features,
        output_features,
        input_channels,
        output_channels,
        map_values,
    ) = offset_values
    value_blocks = []
    map_blocks = []
    for block in range(len(input_features) - 1):
        input_count = input_channels[block + 1] - input_channels[block]
        output_count = output_channels[block + 1] - output_channels[block]
        local_values = values[
            :, input_features[block] : input_features[block + 1]
        ].reshape(int(values.shape[0]), input_count, -1)
        local_adjoint = output_adjoint[
            :, output_features[block] : output_features[block + 1]
        ].reshape(int(values.shape[0]), output_count, -1)
        local_map = channel_maps[
            map_values[block] : map_values[block + 1]
        ].reshape(output_count, input_count)
        value_blocks.append(
            torch.einsum(
                "oi,boq->biq",
                local_map.conj().to(dtype=local_adjoint.dtype),
                local_adjoint,
            ).reshape(int(values.shape[0]), -1)
        )
        local_map_adjoint = torch.einsum(
            "boq,biq->oi",
            local_adjoint,
            local_values.conj(),
        )
        if not channel_maps.is_complex() and local_map_adjoint.is_complex():
            local_map_adjoint = local_map_adjoint.real
        map_blocks.append(
            local_map_adjoint.to(dtype=channel_maps.dtype).reshape(-1)
        )
    return torch.cat(tuple(value_blocks), dim=1), torch.cat(
        tuple(map_blocks), dim=0
    )


def carrier_channel_transform(
    values,
    channel_maps,
    input_feature_offsets,
    output_feature_offsets,
    input_channel_offsets,
    output_channel_offsets,
    map_offsets,
    *,
    backend="auto",
    output_width=None,
):
    """Apply rectangular channel maps while preserving complete carriers.

    A compiled caller may supply the immutable host output width to avoid a
    CUDA scalar synchronization. It must equal the final output offset.
    """

    backend = str(backend)
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")
    prepared = _prepare_carrier_channel_transform_inputs(
        values,
        channel_maps,
        input_feature_offsets,
        output_feature_offsets,
        input_channel_offsets,
        output_channel_offsets,
        map_offsets,
    )
    values, channel_maps = prepared[:2]
    offsets = prepared[2:7]
    offset_values = prepared[7]
    if output_width is not None:
        output_width = int(output_width)
        if output_width < 0 or (offset_values[1] is not None and output_width != int(offset_values[1][-1])):
            raise ValueError("compiled output_width disagrees with carrier output offsets")
    require_native = _enabled("YE3T_REQUIRE_NATIVE")
    if backend == "reference" and require_native:
        raise RuntimeError(
            "YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback"
        )
    native_available = bool(
        (
            _prebuilt_extension() is not None
            or _enabled("YE3T_ENABLE_EXECUTION_PLAN_JIT")
        )
        and hasattr(torch.ops.ye3t_runtime, "carrier_channel_transform")
    )
    use_native = bool(
        backend == "native"
        or require_native
        or (backend == "auto" and values.device.type == "cuda")
    )
    if use_native and not native_available:
        if backend == "auto" and not require_native:
            use_native = False
        else:
            _load_extension()
            native_available = hasattr(
                torch.ops.ye3t_runtime, "carrier_channel_transform"
            )
            if not native_available:
                raise RuntimeError(
                    "the installed YE3T native extension has no rectangular "
                    "carrier channel-transform runtime"
                )
    if use_native:
        extension = _extension_for_native_dispatch()
        if (
            values.device.type == "cuda"
            and extension is not None
            and not bool(extension.has_cuda())
        ):
            if backend == "auto" and not require_native:
                use_native = False
            else:
                raise RuntimeError(
                    "the installed YE3T native extension has no CUDA "
                    "carrier channel-transform runtime"
                )
    if use_native:
        if output_width is None:
            output_width = (
                int(offset_values[1][-1])
                if offset_values[1] is not None
                else int(offsets[1][-1].item())
            )
        return torch.ops.ye3t_runtime.carrier_channel_transform(
            values.contiguous(),
            channel_maps.contiguous(),
            *(value.contiguous() for value in offsets),
            output_width,
        )
    return _carrier_channel_transform_reference(
        values,
        channel_maps,
        tuple(
            offset_values[index]
            if offset_values[index] is not None
            else offsets[index]
            for index in range(5)
        ),
    )


def carrier_channel_transform_adjoint(
    output_adjoint,
    values,
    channel_maps,
    input_feature_offsets,
    output_feature_offsets,
    input_channel_offsets,
    output_channel_offsets,
    map_offsets,
    *,
    backend="auto",
):
    """Apply the adjoint of packed rectangular channel transforms."""

    prepared = _prepare_carrier_channel_transform_inputs(
        values,
        channel_maps,
        input_feature_offsets,
        output_feature_offsets,
        input_channel_offsets,
        output_channel_offsets,
        map_offsets,
    )
    values, channel_maps = prepared[:2]
    offsets = prepared[2:7]
    offset_values = prepared[7]
    output_adjoint = torch.as_tensor(
        output_adjoint,
        dtype=values.dtype,
        device=values.device,
    )
    expected_output_width = int(offsets[1][-1].item())
    if output_adjoint.shape != (int(values.shape[0]), expected_output_width):
        raise ValueError(
            "output_adjoint must match the packed transform output"
        )
    backend = str(backend)
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")
    require_native = _enabled("YE3T_REQUIRE_NATIVE")
    if backend == "reference" and require_native:
        raise RuntimeError(
            "YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback"
        )
    native_available = bool(
        (
            _prebuilt_extension() is not None
            or _enabled("YE3T_ENABLE_EXECUTION_PLAN_JIT")
        )
        and hasattr(
            torch.ops.ye3t_runtime, "carrier_channel_transform_adjoint"
        )
    )
    use_native = bool(
        backend == "native"
        or require_native
        or (backend == "auto" and values.device.type == "cuda")
    )
    if use_native and not native_available:
        if backend == "auto" and not require_native:
            use_native = False
        else:
            _load_extension()
            native_available = hasattr(
                torch.ops.ye3t_runtime,
                "carrier_channel_transform_adjoint",
            )
            if not native_available:
                raise RuntimeError(
                    "the installed YE3T native extension has no rectangular "
                    "carrier channel-transform adjoint"
                )
    if use_native:
        extension = _extension_for_native_dispatch()
        if (
            values.device.type == "cuda"
            and extension is not None
            and not bool(extension.has_cuda())
        ):
            if backend == "auto" and not require_native:
                use_native = False
            else:
                raise RuntimeError(
                    "the installed YE3T native extension has no CUDA "
                    "carrier channel-transform adjoint"
                )
    if use_native:
        return torch.ops.ye3t_runtime.carrier_channel_transform_adjoint(
            output_adjoint.contiguous(),
            values.contiguous(),
            channel_maps.contiguous(),
            *(value.contiguous() for value in offsets),
        )
    return _carrier_channel_transform_adjoint_reference(
        output_adjoint,
        values,
        channel_maps,
        tuple(
            offset_values[index]
            if offset_values[index] is not None
            else offsets[index]
            for index in range(5)
        ),
    )


def _carrier_role_channel_map_adjoint_reference(
    edge_values,
    role_weights,
    atomic_output_adjoint,
    atom_centers,
    channel_maps,
    offset_values,
):
    if not all(isinstance(items, tuple) for items in offset_values):
        offset_values = tuple(
            tuple(int(value) for value in items.detach().cpu().tolist())
            for items in offset_values
        )
    (
        input_features,
        output_features,
        input_channels,
        output_channels,
        map_values,
    ) = offset_values
    gathered_adjoint = atomic_output_adjoint.index_select(0, atom_centers)
    weighted_roles = role_weights.to(dtype=edge_values.dtype)
    map_blocks = []
    for block in range(len(input_features) - 1):
        input_count = input_channels[block + 1] - input_channels[block]
        output_count = output_channels[block + 1] - output_channels[block]
        local_values = edge_values[
            :, input_features[block] : input_features[block + 1]
        ].reshape(int(edge_values.shape[0]), input_count, -1)
        local_adjoint = gathered_adjoint[
            :, :, output_features[block] : output_features[block + 1]
        ].reshape(
            int(edge_values.shape[0]),
            int(role_weights.shape[1]),
            output_count,
            -1,
        )
        local_map_adjoint = torch.einsum(
            "er,eroq,eiq->oi",
            weighted_roles,
            local_adjoint,
            local_values.conj(),
        )
        if not channel_maps.is_complex() and local_map_adjoint.is_complex():
            local_map_adjoint = local_map_adjoint.real
        map_blocks.append(
            local_map_adjoint.to(dtype=channel_maps.dtype).reshape(-1)
        )
    output = torch.cat(tuple(map_blocks), dim=0)
    if int(output.numel()) != int(map_values[-1]):
        raise RuntimeError(
            "role-factored channel-map adjoint does not match map offsets"
        )
    return output


def carrier_role_channel_map_adjoint(
    edge_values,
    role_weights,
    atomic_output_adjoint,
    atom_centers,
    channel_maps,
    input_feature_offsets,
    output_feature_offsets,
    input_channel_offsets,
    output_channel_offsets,
    map_offsets,
    *,
    backend="auto",
):
    """Contract role-factored edge values directly into map adjoints.

    This is the exact adjoint of a channel transform applied after the
    role-resolved radial map. Complete carrier axes remain compiler blocked;
    the operation contracts only scalar role and channel coordinates.
    """

    prepared = _prepare_carrier_channel_transform_inputs(
        edge_values,
        channel_maps,
        input_feature_offsets,
        output_feature_offsets,
        input_channel_offsets,
        output_channel_offsets,
        map_offsets,
    )
    edge_values, channel_maps = prepared[:2]
    offsets = prepared[2:7]
    offset_values = prepared[7]
    real_dtype = (
        edge_values.real.dtype
        if edge_values.is_complex()
        else edge_values.dtype
    )
    role_weights = torch.as_tensor(
        role_weights,
        dtype=real_dtype,
        device=edge_values.device,
    )
    atomic_output_adjoint = torch.as_tensor(
        atomic_output_adjoint,
        dtype=edge_values.dtype,
        device=edge_values.device,
    )
    atom_centers = torch.as_tensor(
        atom_centers,
        dtype=torch.long,
        device=edge_values.device,
    ).reshape(-1)
    if role_weights.ndim != 2 or int(role_weights.shape[0]) != int(
        edge_values.shape[0]
    ):
        raise ValueError(
            "role_weights must provide one role row per physical edge"
        )
    if atomic_output_adjoint.ndim != 3 or int(
        atomic_output_adjoint.shape[1]
    ) != int(role_weights.shape[1]):
        raise ValueError(
            "atomic_output_adjoint must share the declared role axis"
        )
    expected_output_width = int(offsets[1][-1].item())
    if int(atomic_output_adjoint.shape[2]) != expected_output_width:
        raise ValueError(
            "atomic_output_adjoint does not match the transform output"
        )
    if int(atom_centers.numel()) != int(edge_values.shape[0]):
        raise ValueError("atom_centers must match the physical edge count")
    if atom_centers.numel() and (
        int(atom_centers.min().item()) < 0
        or int(atom_centers.max().item()) >= int(
            atomic_output_adjoint.shape[0]
        )
    ):
        raise ValueError("atom_centers contains an out-of-range atom index")
    backend = str(backend)
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")
    require_native = _enabled("YE3T_REQUIRE_NATIVE")
    if backend == "reference" and require_native:
        raise RuntimeError(
            "YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback"
        )
    extension = _prebuilt_extension()
    native_available = bool(
        extension is not None
        and int(extension.core_abi_version()) >= 36
        and hasattr(
            torch.ops.ye3t_runtime,
            "carrier_role_channel_map_adjoint",
        )
        and hasattr(
            torch.ops.ye3t_runtime,
            "carrier_role_channel_map_adjoint_double_backward",
        )
    )
    use_native = bool(
        backend == "native"
        or require_native
        or (backend == "auto" and edge_values.device.type == "cuda")
    )
    if use_native and not native_available:
        if backend == "auto" and not require_native:
            use_native = False
        else:
            extension = _load_extension()
            native_available = bool(
                int(extension.core_abi_version()) >= 36
                and hasattr(
                    torch.ops.ye3t_runtime,
                    "carrier_role_channel_map_adjoint",
                )
            )
            if not native_available:
                raise RuntimeError(
                    "the installed YE3T native extension has no "
                    "role-factored channel-map adjoint"
                )
    if use_native:
        extension = _extension_for_native_dispatch()
        native_available = bool(
            extension is None
            or (
                int(extension.core_abi_version()) >= 36
                and hasattr(
                    torch.ops.ye3t_runtime,
                    "carrier_role_channel_map_adjoint",
                )
                and hasattr(
                    torch.ops.ye3t_runtime,
                    "carrier_role_channel_map_adjoint_double_backward",
                )
            )
        )
        if not native_available:
            raise RuntimeError(
                "the installed YE3T native extension has no derivative-"
                "complete role-factored channel-map adjoint"
            )
    if use_native and edge_values.device.type == "cuda":
        if extension is None or not bool(extension.has_cuda()):
            if backend == "auto" and not require_native:
                use_native = False
            else:
                raise RuntimeError(
                    "the installed YE3T native extension has no CUDA "
                    "role-factored channel-map adjoint"
                )
    if use_native:
        return torch.ops.ye3t_runtime.carrier_role_channel_map_adjoint(
            edge_values.contiguous(),
            role_weights.contiguous(),
            atomic_output_adjoint.contiguous(),
            atom_centers.contiguous(),
            channel_maps.contiguous(),
            *(value.contiguous() for value in offsets),
        )
    return _carrier_role_channel_map_adjoint_reference(
        edge_values,
        role_weights,
        atomic_output_adjoint,
        atom_centers,
        channel_maps,
        tuple(
            offset_values[index]
            if offset_values[index] is not None
            else offsets[index]
            for index in range(5)
        ),
    )


def _carrier_channel_update_use_native(
    backend,
    values,
    require_native,
):
    if backend == "native" or require_native:
        return True
    if backend != "auto":
        return False
    if values.device.type == "cuda":
        threshold = _NATIVE_CUDA_CARRIER_CHANNEL_UPDATE_MIN_WORK_ITEMS
        size_eligible = int(values.numel()) >= int(threshold)
    else:
        threshold = _NATIVE_CPU_CARRIER_CHANNEL_UPDATE_MAX_WORK_ITEMS
        size_eligible = int(values.numel()) <= int(threshold)
    return bool(
        threshold is not None
        and size_eligible
        and (
            _prebuilt_extension() is not None
            or _enabled("YE3T_ENABLE_EXECUTION_PLAN_JIT")
        )
    )


def _carrier_channel_update_reference(
    values,
    gates,
    channel_maps,
    offset_values,
):
    feature_values, channel_values, map_values = offset_values
    if not all(isinstance(items, tuple) for items in offset_values):
        feature_values = tuple(
            int(value) for value in offset_values[0].detach().cpu().tolist()
        )
        channel_values = tuple(
            int(value) for value in offset_values[1].detach().cpu().tolist()
        )
        map_values = tuple(
            int(value) for value in offset_values[2].detach().cpu().tolist()
        )
    outputs = []
    for block in range(len(feature_values) - 1):
        feature_start = feature_values[block]
        feature_stop = feature_values[block + 1]
        channel_start = channel_values[block]
        channel_stop = channel_values[block + 1]
        local_channels = channel_stop - channel_start
        local_values = values[:, feature_start:feature_stop].reshape(
            int(values.shape[0]),
            local_channels,
            -1,
        )
        local_map = channel_maps[
            map_values[block] : map_values[block + 1]
        ].reshape(local_channels, local_channels)
        mixed = torch.einsum(
            "oi,biq->boq",
            local_map.to(dtype=local_values.dtype),
            local_values,
        )
        outputs.append(
            (
                local_values
                + gates[
                    :,
                    channel_start:channel_stop,
                ].unsqueeze(2)
                * mixed
            ).reshape(int(values.shape[0]), -1)
        )
    return torch.cat(tuple(outputs), dim=1)


def carrier_channel_update(
    values,
    gates,
    channel_maps,
    feature_offsets,
    channel_offsets,
    map_offsets,
    *,
    backend="auto",
):
    """Apply block-local channel maps and invariant gates to packed carriers."""

    backend = str(backend)
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")
    require_native = _enabled("YE3T_REQUIRE_NATIVE")
    if backend == "reference" and require_native:
        raise RuntimeError(
            "YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback"
        )
    (
        values,
        gates,
        channel_maps,
        feature_offsets,
        channel_offsets,
        map_offsets,
        offset_values,
    ) = _prepare_carrier_channel_update_inputs(
        values,
        gates,
        channel_maps,
        feature_offsets,
        channel_offsets,
        map_offsets,
    )
    use_native = _carrier_channel_update_use_native(
        backend,
        values,
        require_native,
    )
    if use_native:
        extension = _extension_for_native_dispatch()
        if (
            values.device.type == "cuda"
            and not _native_cuda_dispatch_available(extension)
        ):
            if backend == "auto" and not require_native:
                use_native = False
            else:
                raise RuntimeError(
                    "the installed YE3T native extension has no CUDA "
                    "carrier channel-update operators"
                )
        elif values.device.type not in {"cpu", "cuda"}:
            raise ValueError(
                "native carrier channel update requires CPU or CUDA"
            )
        if use_native:
            return torch.ops.ye3t_runtime.carrier_channel_update(
                values.contiguous(),
                gates.contiguous(),
                channel_maps.contiguous(),
                feature_offsets.contiguous(),
                channel_offsets.contiguous(),
                map_offsets.contiguous(),
            )
    return _carrier_channel_update_reference(
        values,
        gates,
        channel_maps,
        (
            offset_values[0]
            if offset_values[0] is not None
            else feature_offsets,
            offset_values[1]
            if offset_values[1] is not None
            else channel_offsets,
            offset_values[2]
            if offset_values[2] is not None
            else map_offsets,
        ),
    )


def carrier_channel_update_adjoint(
    output_adjoint,
    values,
    gates,
    channel_maps,
    feature_offsets,
    channel_offsets,
    map_offsets,
    *,
    backend="auto",
):
    """Apply the adjoint of packed block-local carrier channel updates."""

    (
        values,
        gates,
        channel_maps,
        feature_offsets,
        channel_offsets,
        map_offsets,
        offset_values,
    ) = _prepare_carrier_channel_update_inputs(
        values,
        gates,
        channel_maps,
        feature_offsets,
        channel_offsets,
        map_offsets,
    )
    output_adjoint = torch.as_tensor(
        output_adjoint,
        dtype=values.dtype,
        device=values.device,
    )
    if output_adjoint.shape != values.shape:
        raise ValueError("output_adjoint must match values")
    backend = str(backend)
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")
    require_native = _enabled("YE3T_REQUIRE_NATIVE")
    if backend == "reference" and require_native:
        raise RuntimeError(
            "YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback"
        )
    use_native = _carrier_channel_update_use_native(
        backend,
        values,
        require_native,
    )
    if use_native:
        extension = _load_extension()
        if values.device.type == "cuda" and not bool(extension.has_cuda()):
            if backend == "auto" and not require_native:
                use_native = False
            else:
                raise RuntimeError(
                    "the installed YE3T native extension has no CUDA "
                    "carrier channel-update adjoint"
                )
        elif values.device.type not in {"cpu", "cuda"}:
            raise ValueError(
                "native carrier channel update requires CPU or CUDA"
            )
        if use_native:
            return torch.ops.ye3t_runtime.carrier_channel_update_adjoint(
                output_adjoint.contiguous(),
                values.contiguous(),
                gates.contiguous(),
                channel_maps.contiguous(),
                feature_offsets.contiguous(),
                channel_offsets.contiguous(),
                map_offsets.contiguous(),
            )
    feature_values, channel_values, map_values = offset_values
    if any(items is None for items in offset_values):
        feature_values = tuple(
            int(value) for value in feature_offsets.detach().cpu().tolist()
        )
        channel_values = tuple(
            int(value) for value in channel_offsets.detach().cpu().tolist()
        )
        map_values = tuple(
            int(value) for value in map_offsets.detach().cpu().tolist()
        )
    value_adjoint_blocks = []
    gate_adjoint_blocks = []
    map_adjoint_blocks = []
    for block in range(len(feature_values) - 1):
        feature_start = feature_values[block]
        feature_stop = feature_values[block + 1]
        channel_start = channel_values[block]
        channel_stop = channel_values[block + 1]
        local_channels = channel_stop - channel_start
        local_values = values[:, feature_start:feature_stop].reshape(
            int(values.shape[0]),
            local_channels,
            -1,
        )
        local_adjoint = output_adjoint[
            :, feature_start:feature_stop
        ].reshape(int(values.shape[0]), local_channels, -1)
        local_gates = gates[:, channel_start:channel_stop]
        local_map = channel_maps[
            map_values[block] : map_values[block + 1]
        ].reshape(local_channels, local_channels)
        map_for_values = local_map.to(dtype=local_values.dtype)
        gates_for_values = local_gates.to(dtype=local_values.dtype)
        mixed = torch.einsum(
            "oi,biq->boq",
            map_for_values,
            local_values,
        )
        value_adjoint_blocks.append(
            (
                local_adjoint
                + torch.einsum(
                    "boq,bo,oi->biq",
                    local_adjoint,
                    gates_for_values.conj(),
                    map_for_values.conj(),
                )
            ).reshape(int(values.shape[0]), -1)
        )
        gate_adjoint = torch.sum(
            local_adjoint * mixed.conj(),
            dim=2,
        )
        map_adjoint = torch.einsum(
            "boq,bo,biq->oi",
            local_adjoint,
            gates_for_values.conj(),
            local_values.conj(),
        )
        if not gates.is_complex():
            gate_adjoint = gate_adjoint.real
            map_adjoint = map_adjoint.real
        gate_adjoint_blocks.append(gate_adjoint)
        map_adjoint_blocks.append(map_adjoint.reshape(-1))
    return (
        torch.cat(tuple(value_adjoint_blocks), dim=1),
        torch.cat(tuple(gate_adjoint_blocks), dim=1),
        torch.cat(tuple(map_adjoint_blocks), dim=0),
    )


def _coefficient_dtype(source, source_assembly, synthesis_table):
    import torch

    has_complex = source.is_complex()
    has_complex = has_complex or any(
        value.imag != 0.0 for value in source_assembly.values
    )
    has_complex = has_complex or any(
        value.imag != 0.0 for value in synthesis_table.values
    )
    if has_complex:
        if source.dtype in (torch.float64, torch.complex128):
            return torch.complex128
        return torch.complex64
    if source.dtype not in (torch.float32, torch.float64):
        raise TypeError("native execution-plan CPU supports float32/float64 inputs")
    return source.dtype


def _coefficient_values(values, dtype):
    import torch

    if dtype in (torch.complex64, torch.complex128):
        return tuple(values)
    return tuple(float(value.real) for value in values)


def _compose_source_analysis_table(assembly, synthesis):
    assembly_by_row = {}
    for row, column, value in zip(
        assembly.row_indices,
        assembly.column_indices,
        assembly.values,
    ):
        assembly_by_row.setdefault(int(row), []).append(
            (int(column), complex(value))
        )
    synthesis_by_row = {}
    for row, column, value in zip(
        synthesis.row_indices,
        synthesis.column_indices,
        synthesis.values,
    ):
        synthesis_by_row.setdefault(int(row), []).append(
            (int(column), complex(value).conjugate())
        )
    combined = {}
    for row, assembly_terms in assembly_by_row.items():
        for source_column, assembly_value in assembly_terms:
            for output_column, synthesis_value in synthesis_by_row.get(row, ()):
                key = (int(source_column), int(output_column))
                combined[key] = (
                    combined.get(key, 0.0j)
                    + assembly_value * synthesis_value
                )
    tolerance = 1e-15
    return tuple(
        (source_column, output_column, value)
        for (source_column, output_column), value in sorted(combined.items())
        if abs(value) > tolerance
    )


def _expand_source_analysis_mixing_table(
    composed_terms,
    channel_count,
    component_width,
):
    source_indices = []
    output_indices = []
    weight_indices = []
    coefficients = []
    for source_index, base_output, coefficient in composed_terms:
        raw_channel = int(base_output) // int(component_width)
        component = int(base_output) % int(component_width)
        if raw_channel < 0 or raw_channel >= int(channel_count):
            raise ValueError(
                "source-analysis output does not match the carrier channel layout"
            )
        for output_channel in range(int(channel_count)):
            source_indices.append(int(source_index))
            output_indices.append(
                int(output_channel) * int(component_width) + int(component)
            )
            weight_indices.append(
                int(output_channel) * int(channel_count) + int(raw_channel)
            )
            coefficients.append(complex(coefficient))
    return (
        tuple(source_indices),
        tuple(output_indices),
        tuple(weight_indices),
        tuple(coefficients),
    )


def apply_source_analysis_native(source, source_assembly, synthesis_table):
    """Apply one sparse source-analysis instruction with the native runtime."""

    import torch

    extension = _load_extension()
    if not isinstance(source_assembly, YE3TSourceAssemblyPlan):
        source_assembly = YE3TSourceAssemblyPlan.from_dict(source_assembly)
    if not isinstance(synthesis_table, YE3TSynthesisTable):
        synthesis_table = YE3TSynthesisTable.from_dict(synthesis_table)
    if source.device.type not in {"cpu", "cuda"}:
        raise ValueError("native execution-plan runtime requires CPU or CUDA")
    if source.device.type == "cuda" and not bool(extension.has_cuda()):
        raise RuntimeError(
            "the installed YE3T native extension has no CUDA source-analysis "
            "operators"
        )
    if int(source.shape[-1]) != int(source_assembly.source_dimension):
        raise ValueError("source last dimension does not match source assembly")
    if int(source_assembly.induced_dimension) != int(synthesis_table.input_dimension):
        raise ValueError("source assembly and synthesis table dimensions do not match")
    dtype = _coefficient_dtype(source, source_assembly, synthesis_table)
    working = source.to(dtype=dtype)
    flat = working.reshape(-1, source_assembly.source_dimension).contiguous()
    assembly_rows = torch.tensor(
        source_assembly.row_indices,
        dtype=torch.int64,
        device=source.device,
    )
    assembly_columns = torch.tensor(
        source_assembly.column_indices,
        dtype=torch.int64,
        device=source.device,
    )
    assembly_values = torch.tensor(
        _coefficient_values(source_assembly.values, dtype),
        dtype=dtype,
        device=source.device,
    )
    synthesis_rows = torch.tensor(
        synthesis_table.row_indices,
        dtype=torch.int64,
        device=source.device,
    )
    synthesis_columns = torch.tensor(
        synthesis_table.column_indices,
        dtype=torch.int64,
        device=source.device,
    )
    synthesis_values = torch.tensor(
        _coefficient_values(synthesis_table.values, dtype),
        dtype=dtype,
        device=source.device,
    )
    result = torch.ops.ye3t_runtime.source_analysis(
        flat,
        assembly_rows,
        assembly_columns,
        assembly_values,
        synthesis_rows,
        synthesis_columns,
        synthesis_values,
        source_assembly.induced_dimension,
        synthesis_table.output_dimension,
    )
    return result.reshape(
        tuple(source.shape[:-1]) + (synthesis_table.output_dimension,)
    )


def apply_source_analysis(source, source_assembly, synthesis_table, backend="auto"):
    """Apply source assembly and analysis with explicit dispatch semantics."""

    backend = str(backend)
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")
    require_native = _enabled("YE3T_REQUIRE_NATIVE")
    if backend == "reference":
        if require_native:
            raise RuntimeError(
                "YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback"
            )
        return apply_source_analysis_reference(
            source,
            source_assembly,
            synthesis_table,
        )
    if backend == "native" or require_native or _prebuilt_extension() is not None:
        return apply_source_analysis_native(
            source,
            source_assembly,
            synthesis_table,
        )
    return apply_source_analysis_reference(
        source,
        source_assembly,
        synthesis_table,
    )


def compact_pair_product_reference(left, right, antisymmetric=False):
    """Normalized compact symmetric or exterior rank-two product."""

    if left.shape != right.shape:
        raise ValueError("left and right must have identical shapes")
    if left.ndim < 1:
        raise ValueError("compact pair products require a feature axis")
    dimension = int(left.shape[-1])
    values = []
    scale = 1.0 / (2.0 ** 0.5)
    for i in range(dimension):
        start = i + 1 if antisymmetric else i
        for j in range(start, dimension):
            if not antisymmetric and i == j:
                values.append(left[..., i] * right[..., i])
            elif antisymmetric:
                values.append(
                    scale
                    * (
                        left[..., i] * right[..., j]
                        - left[..., j] * right[..., i]
                    )
                )
            else:
                values.append(
                    scale
                    * (
                        left[..., i] * right[..., j]
                        + left[..., j] * right[..., i]
                    )
                )
    if values:
        return torch.stack(values, dim=-1)
    return left.new_empty(tuple(left.shape[:-1]) + (0,))


def compact_pair_product_adjoint_reference(
    output_adjoint,
    left,
    right,
    antisymmetric=False,
):
    """Analytic adjoint of the normalized compact rank-two product."""

    if left.shape != right.shape:
        raise ValueError("left and right must have identical shapes")
    dimension = int(left.shape[-1])
    flat_left = left.reshape(-1, dimension)
    flat_right = right.reshape(-1, dimension)
    flat_output = output_adjoint.reshape(-1, int(output_adjoint.shape[-1]))
    if int(flat_output.shape[0]) != int(flat_left.shape[0]):
        raise ValueError("output_adjoint leading dimensions must match inputs")
    left_targets = []
    right_targets = []
    left_values = []
    right_values = []
    scale = 1.0 / (2.0 ** 0.5)
    output_index = 0
    for i in range(dimension):
        start = i + 1 if antisymmetric else i
        for j in range(start, dimension):
            gradient = flat_output[:, output_index]
            if not antisymmetric and i == j:
                left_targets.append(i)
                left_values.append(gradient * flat_right[:, i].conj())
                right_targets.append(i)
                right_values.append(gradient * flat_left[:, i].conj())
            elif antisymmetric:
                left_targets.extend((i, j))
                left_values.extend(
                    (
                        scale * gradient * flat_right[:, j].conj(),
                        -scale * gradient * flat_right[:, i].conj(),
                    )
                )
                right_targets.extend((j, i))
                right_values.extend(
                    (
                        scale * gradient * flat_left[:, i].conj(),
                        -scale * gradient * flat_left[:, j].conj(),
                    )
                )
            else:
                left_targets.extend((i, j))
                left_values.extend(
                    (
                        scale * gradient * flat_right[:, j].conj(),
                        scale * gradient * flat_right[:, i].conj(),
                    )
                )
                right_targets.extend((j, i))
                right_values.extend(
                    (
                        scale * gradient * flat_left[:, i].conj(),
                        scale * gradient * flat_left[:, j].conj(),
                    )
                )
            output_index += 1
    left_adjoint = flat_left.new_zeros(flat_left.shape)
    right_adjoint = flat_right.new_zeros(flat_right.shape)
    if left_values:
        left_indices = torch.tensor(
            left_targets,
            dtype=torch.int64,
            device=left.device,
        )
        right_indices = torch.tensor(
            right_targets,
            dtype=torch.int64,
            device=right.device,
        )
        left_adjoint = left_adjoint.index_add(
            1,
            left_indices,
            torch.stack(left_values, dim=1),
        )
        right_adjoint = right_adjoint.index_add(
            1,
            right_indices,
            torch.stack(right_values, dim=1),
        )
    return (
        left_adjoint.reshape(left.shape),
        right_adjoint.reshape(right.shape),
    )


def compact_pair_product(left, right, antisymmetric=False, backend="auto"):
    """Dispatch compact symmetric/exterior products without ordered storage."""

    backend = str(backend)
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")
    require_native = _enabled("YE3T_REQUIRE_NATIVE")
    if backend == "reference":
        if require_native:
            raise RuntimeError(
                "YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback"
            )
        return compact_pair_product_reference(
            left,
            right,
            antisymmetric=bool(antisymmetric),
        )
    use_native = backend == "native" or require_native
    use_native = use_native or (
        backend == "auto"
        and (
            _prebuilt_extension() is not None
            or _enabled("YE3T_ENABLE_EXECUTION_PLAN_JIT")
        )
    )
    if not use_native:
        return compact_pair_product_reference(
            left,
            right,
            antisymmetric=bool(antisymmetric),
        )
    if left.shape != right.shape:
        raise ValueError("left and right must have identical shapes")
    if left.ndim < 1:
        raise ValueError("compact pair products require a feature axis")
    if left.device != right.device:
        raise ValueError("left and right must use the same device")
    extension = _load_extension()
    if left.device.type == "cuda" and not bool(extension.has_cuda()):
        if backend == "auto" and not require_native:
            return compact_pair_product_reference(
                left,
                right,
                antisymmetric=bool(antisymmetric),
            )
        raise RuntimeError(
            "the installed YE3T native extension has no CUDA compact-pair "
            "operators"
        )
    if left.device.type not in {"cpu", "cuda"}:
        raise ValueError("native compact pair products require CPU or CUDA")
    dimension = int(left.shape[-1])
    flat_left = left.reshape(-1, dimension).contiguous()
    flat_right = right.reshape(-1, dimension).contiguous()
    output = torch.ops.ye3t_runtime.compact_pair_product(
        flat_left,
        flat_right,
        bool(antisymmetric),
    )
    return output.reshape(tuple(left.shape[:-1]) + (output.shape[-1],))


def compact_symmetric_pair_product(left, right, backend="auto"):
    return compact_pair_product(
        left,
        right,
        antisymmetric=False,
        backend=backend,
    )


def compact_exterior_pair_product(left, right, backend="auto"):
    return compact_pair_product(
        left,
        right,
        antisymmetric=True,
        backend=backend,
    )


def compact_exterior_power_product_reference(factors):
    """Normalized compact wedge of one-body factors in lexicographic order."""

    if factors.ndim < 2:
        raise ValueError(
            "compact exterior power factors require [order, feature] axes"
        )
    order = int(factors.shape[-2])
    dimension = int(factors.shape[-1])
    if order < 1:
        raise ValueError("compact exterior power order must be positive")
    if order > dimension:
        scalar_zero = factors.sum(dim=(-2, -1)) * 0
        return scalar_zero.unsqueeze(-1).expand(
            tuple(factors.shape[:-2]) + (0,)
        )
    scale = 1.0 / math.sqrt(float(math.factorial(order)))
    values = []
    for coordinate in combinations(range(dimension), order):
        indices = torch.tensor(
            coordinate,
            dtype=torch.int64,
            device=factors.device,
        )
        square = factors.index_select(-1, indices)
        values.append(scale * torch.linalg.det(square))
    return torch.stack(values, dim=-1)


def compact_exterior_power_adjoint_reference(
    output_adjoint,
    factors,
):
    """Differentiable signed adjoint of the compact exterior power."""

    expected = math.comb(
        int(factors.shape[-1]),
        int(factors.shape[-2]),
    )
    if tuple(output_adjoint.shape[:-1]) != tuple(factors.shape[:-2]):
        raise ValueError(
            "output_adjoint leading dimensions must match factors"
        )
    if int(output_adjoint.shape[-1]) != int(expected):
        raise ValueError(
            "output_adjoint width does not match compact exterior power"
        )
    create_graph = torch.is_grad_enabled()
    with torch.enable_grad():
        tracked = factors
        if not tracked.requires_grad:
            tracked = factors.detach().requires_grad_(True)
        output = compact_exterior_power_product_reference(tracked)
        factors_adjoint = torch.autograd.grad(
            output,
            tracked,
            output_adjoint,
            create_graph=create_graph,
        )[0]
    return factors_adjoint


def compact_exterior_power_product(factors, backend="auto"):
    """Dispatch a compact normalized exterior power without ordered storage."""

    backend = str(backend)
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")
    if factors.ndim < 2:
        raise ValueError(
            "compact exterior power factors require [order, feature] axes"
        )
    order = int(factors.shape[-2])
    dimension = int(factors.shape[-1])
    if order < 1:
        raise ValueError("compact exterior power order must be positive")
    require_native = _enabled("YE3T_REQUIRE_NATIVE")
    if backend == "reference":
        if require_native:
            raise RuntimeError(
                "YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback"
            )
        return compact_exterior_power_product_reference(factors)
    if order > 8:
        if backend == "native" or require_native:
            raise ValueError(
                "native compact exterior power currently supports order <= 8"
            )
        return compact_exterior_power_product_reference(factors)
    use_native = backend == "native" or require_native
    use_native = use_native or (
        backend == "auto"
        and (
            _prebuilt_extension() is not None
            or _enabled("YE3T_ENABLE_EXECUTION_PLAN_JIT")
        )
    )
    if not use_native:
        return compact_exterior_power_product_reference(factors)
    extension = _load_extension()
    if factors.device.type == "cuda" and not bool(extension.has_cuda()):
        if backend == "auto" and not require_native:
            return compact_exterior_power_product_reference(factors)
        raise RuntimeError(
            "the installed YE3T native extension has no CUDA compact "
            "exterior-power operators"
        )
    if factors.device.type not in {"cpu", "cuda"}:
        raise ValueError(
            "native compact exterior power requires CPU or CUDA"
        )
    flat = factors.reshape(-1, order, dimension).contiguous()
    output = torch.ops.ye3t_runtime.compact_exterior_power(flat)
    return output.reshape(
        tuple(factors.shape[:-2]) + (int(output.shape[-1]),)
    )


def symmetric_power_monomial_reference(
    input,
    monomial_counts,
    output_offsets,
    output_indices,
    monomial_values,
):
    """Evaluate one compiler-grouped symmetric-power monomial table."""

    input = torch.as_tensor(input)
    if input.ndim < 1:
        raise ValueError("symmetric-power input requires a feature axis")
    input_dimension = int(input.shape[-1])
    counts = torch.as_tensor(
        monomial_counts,
        dtype=torch.int64,
        device=input.device,
    )
    offsets = torch.as_tensor(
        output_offsets,
        dtype=torch.int64,
        device=input.device,
    ).reshape(-1)
    indices = torch.as_tensor(
        output_indices,
        dtype=torch.int64,
        device=input.device,
    ).reshape(-1)
    values = torch.as_tensor(
        monomial_values,
        dtype=input.dtype,
        device=input.device,
    ).reshape(-1)
    if counts.ndim != 2 or int(counts.shape[1]) != input_dimension:
        raise ValueError(
            "monomial_counts must have shape [term, input_dimension]"
        )
    if int(indices.numel()) != int(counts.shape[0]):
        raise ValueError("output_indices length must match the term count")
    if int(values.numel()) != int(counts.shape[0]):
        raise ValueError("monomial_values length must match the term count")
    if int(offsets.numel()) < 1:
        raise ValueError("output_offsets must contain at least one entry")
    output_dimension = int(offsets.numel()) - 1
    flat = input.reshape(-1, input_dimension)
    monomials = torch.ones(
        (flat.shape[0], counts.shape[0]),
        dtype=input.dtype,
        device=input.device,
    )
    for component in range(input_dimension):
        component_counts = counts[:, component].reshape(1, -1)
        component_power = torch.ones_like(monomials)
        maximum = (
            int(component_counts.max().detach().cpu())
            if int(component_counts.numel())
            else 0
        )
        for exponent in range(maximum):
            component_power = torch.where(
                component_counts > exponent,
                component_power * flat[:, component : component + 1],
                component_power,
            )
        monomials = monomials * component_power
    weighted = monomials * values
    output = input.new_zeros((flat.shape[0], output_dimension))
    output.index_add_(1, indices, weighted)
    return output.reshape(
        tuple(input.shape[:-1]) + (output_dimension,)
    )


def symmetric_power_monomial_adjoint_reference(
    output_adjoint,
    input,
    monomial_counts,
    output_offsets,
    output_indices,
    monomial_values,
):
    """Differentiable adjoint of one symmetric-power monomial table."""

    create_graph = torch.is_grad_enabled()
    with torch.enable_grad():
        tracked = input
        if not tracked.requires_grad:
            tracked = input.detach().requires_grad_(True)
        output = symmetric_power_monomial_reference(
            tracked,
            monomial_counts,
            output_offsets,
            output_indices,
            monomial_values,
        )
        input_adjoint = torch.autograd.grad(
            output,
            tracked,
            output_adjoint,
            create_graph=create_graph,
        )[0]
    return input_adjoint


def symmetric_power_monomial_contraction(
    input,
    monomial_counts,
    output_offsets,
    output_indices,
    monomial_values,
    *,
    backend="auto",
):
    """Dispatch a compiler-grouped symmetric-power table contraction."""

    backend = str(backend)
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")
    input = torch.as_tensor(input)
    if input.ndim < 1:
        raise ValueError("symmetric-power input requires a feature axis")
    input_dimension = int(input.shape[-1])
    counts = torch.as_tensor(
        monomial_counts,
        dtype=torch.int64,
        device=input.device,
    )
    offsets = torch.as_tensor(
        output_offsets,
        dtype=torch.int64,
        device=input.device,
    ).reshape(-1)
    indices = torch.as_tensor(
        output_indices,
        dtype=torch.int64,
        device=input.device,
    ).reshape(-1)
    values = torch.as_tensor(
        monomial_values,
        dtype=input.dtype,
        device=input.device,
    ).reshape(-1)
    require_native = _enabled("YE3T_REQUIRE_NATIVE")
    if backend == "reference":
        if require_native:
            raise RuntimeError(
                "YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback"
            )
        return symmetric_power_monomial_reference(
            input,
            counts,
            offsets,
            indices,
            values,
        )
    use_native = backend == "native" or require_native
    use_native = use_native or (
        backend == "auto"
        and (
            _prebuilt_extension() is not None
            or _enabled("YE3T_ENABLE_EXECUTION_PLAN_JIT")
        )
    )
    if not use_native:
        return symmetric_power_monomial_reference(
            input,
            counts,
            offsets,
            indices,
            values,
        )
    extension = _load_extension()
    if input.device.type == "cuda" and not bool(extension.has_cuda()):
        if backend == "auto" and not require_native:
            return symmetric_power_monomial_reference(
                input,
                counts,
                offsets,
                indices,
                values,
            )
        raise RuntimeError(
            "the installed YE3T native extension has no CUDA "
            "symmetric-power monomial operators"
        )
    if input.device.type not in {"cpu", "cuda"}:
        raise ValueError(
            "native symmetric-power monomials require CPU or CUDA"
        )
    if counts.ndim != 2 or int(counts.shape[1]) != input_dimension:
        raise ValueError(
            "monomial_counts must have shape [term, input_dimension]"
        )
    flat = input.reshape(-1, input_dimension).contiguous()
    output = torch.ops.ye3t_runtime.symmetric_power_monomial(
        flat,
        counts.contiguous(),
        offsets.contiguous(),
        indices.contiguous(),
        values.contiguous(),
    )
    return output.reshape(
        tuple(input.shape[:-1]) + (int(output.shape[-1]),)
    )


def symmetric_power_shared_monomial_reference(
    input,
    monomial_counts,
    output_offsets,
    coefficient_terms,
    coefficient_outputs,
    coefficient_values,
):
    """Evaluate unique monomials once and apply a sparse output map."""

    input = torch.as_tensor(input)
    if input.ndim < 1:
        raise ValueError("symmetric-power input requires a feature axis")
    input_dimension = int(input.shape[-1])
    counts = torch.as_tensor(
        monomial_counts,
        dtype=torch.int64,
        device=input.device,
    )
    offsets = torch.as_tensor(
        output_offsets,
        dtype=torch.int64,
        device=input.device,
    ).reshape(-1)
    terms = torch.as_tensor(
        coefficient_terms,
        dtype=torch.int64,
        device=input.device,
    ).reshape(-1)
    outputs = torch.as_tensor(
        coefficient_outputs,
        dtype=torch.int64,
        device=input.device,
    ).reshape(-1)
    values = torch.as_tensor(
        coefficient_values,
        dtype=input.dtype,
        device=input.device,
    ).reshape(-1)
    if counts.ndim != 2 or int(counts.shape[1]) != input_dimension:
        raise ValueError(
            "monomial_counts must have shape [monomial, input_dimension]"
        )
    coefficient_count = int(terms.numel())
    if int(outputs.numel()) != coefficient_count:
        raise ValueError(
            "coefficient_outputs length must match coefficient_terms"
        )
    if int(values.numel()) != coefficient_count:
        raise ValueError(
            "coefficient_values length must match coefficient_terms"
        )
    if int(offsets.numel()) < 1:
        raise ValueError("output_offsets must contain at least one entry")
    output_dimension = int(offsets.numel()) - 1
    if coefficient_count:
        if int(terms.min().detach().cpu()) < 0:
            raise ValueError("coefficient_terms must be nonnegative")
        if int(terms.max().detach().cpu()) >= int(counts.shape[0]):
            raise ValueError("coefficient_terms contains an invalid monomial")
        if int(outputs.min().detach().cpu()) < 0:
            raise ValueError("coefficient_outputs must be nonnegative")
        if int(outputs.max().detach().cpu()) >= output_dimension:
            raise ValueError("coefficient_outputs contains an invalid output")
    flat = input.reshape(-1, input_dimension)
    monomials = torch.ones(
        (flat.shape[0], counts.shape[0]),
        dtype=input.dtype,
        device=input.device,
    )
    for component in range(input_dimension):
        component_counts = counts[:, component].reshape(1, -1)
        component_power = torch.ones_like(monomials)
        maximum = (
            int(component_counts.max().detach().cpu())
            if int(component_counts.numel())
            else 0
        )
        for exponent in range(maximum):
            component_power = torch.where(
                component_counts > exponent,
                component_power * flat[:, component : component + 1],
                component_power,
            )
        monomials = monomials * component_power
    weighted = monomials[:, terms] * values
    output = input.new_zeros((flat.shape[0], output_dimension))
    output.index_add_(1, outputs, weighted)
    return output.reshape(
        tuple(input.shape[:-1]) + (output_dimension,)
    )


def symmetric_power_shared_monomial_adjoint_reference(
    output_adjoint,
    input,
    monomial_counts,
    output_offsets,
    coefficient_terms,
    coefficient_outputs,
    coefficient_values,
):
    """Differentiable adjoint of a shared symmetric-power table."""

    create_graph = torch.is_grad_enabled()
    with torch.enable_grad():
        tracked = input
        if not tracked.requires_grad:
            tracked = input.detach().requires_grad_(True)
        output = symmetric_power_shared_monomial_reference(
            tracked,
            monomial_counts,
            output_offsets,
            coefficient_terms,
            coefficient_outputs,
            coefficient_values,
        )
        input_adjoint = torch.autograd.grad(
            output,
            tracked,
            output_adjoint,
            create_graph=create_graph,
        )[0]
    return input_adjoint


def symmetric_power_shared_monomial_batched_adjoint_reference(
    output_adjoint,
    input,
    monomial_counts,
    output_offsets,
    coefficient_terms,
    coefficient_outputs,
    coefficient_values,
):
    """Reference VJPs for many output seeds over one physical input batch."""

    input = torch.as_tensor(input)
    output_adjoint = torch.as_tensor(
        output_adjoint,
        dtype=input.dtype,
        device=input.device,
    )
    if input.ndim != 2:
        raise ValueError("batched symmetric-power input must be two-dimensional")
    if output_adjoint.ndim != 3:
        raise ValueError(
            "output_adjoint must have shape [seed, batch, output]"
        )
    if int(output_adjoint.shape[1]) != int(input.shape[0]):
        raise ValueError("output_adjoint batch must match input")
    counts = torch.as_tensor(
        monomial_counts,
        dtype=torch.int64,
        device=input.device,
    )
    terms = torch.as_tensor(
        coefficient_terms,
        dtype=torch.int64,
        device=input.device,
    ).reshape(-1)
    outputs = torch.as_tensor(
        coefficient_outputs,
        dtype=torch.int64,
        device=input.device,
    ).reshape(-1)
    values = torch.as_tensor(
        coefficient_values,
        dtype=input.dtype,
        device=input.device,
    ).reshape(-1)
    input_dimension = input.shape[1]
    identity = torch.eye(
        input_dimension,
        dtype=torch.int64,
        device=input.device,
    )
    lowered = torch.clamp(
        counts[:, None, :] - identity[None, :, :],
        min=0,
    )
    powers = input.new_ones(
        (
            input.shape[0],
            counts.shape[0],
            input_dimension,
            input_dimension,
        )
    )
    factor = input[:, None, None, :].expand_as(powers)
    remaining = lowered[None, :, :, :]
    for _ in range(7):
        powers = torch.where(
            torch.bitwise_and(remaining, 1) == 1,
            powers * factor,
            powers,
        )
        factor = factor * factor
        remaining = torch.div(
            remaining,
            2,
            rounding_mode="floor",
        )
    derivatives = torch.prod(powers, dim=-1)
    derivatives = derivatives * counts[None, :, :]
    selected_adjoint = output_adjoint.index_select(2, outputs)
    weighted_adjoint = selected_adjoint * values.conj()
    monomial_adjoint = input.new_zeros(
        (
            output_adjoint.shape[0],
            input.shape[0],
            counts.shape[0],
        )
    )
    monomial_adjoint.index_add_(2, terms, weighted_adjoint)
    return torch.einsum(
        "sbu,bui->sbi",
        monomial_adjoint,
        derivatives.conj(),
    )


def symmetric_power_shared_monomial_batched_adjoint(
    output_adjoint,
    input,
    monomial_counts,
    output_offsets,
    coefficient_terms,
    coefficient_outputs,
    coefficient_values,
    *,
    backend="auto",
):
    """Dispatch many shared-table VJPs without expanding the input batch."""

    backend = str(backend)
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")
    input = torch.as_tensor(input)
    if input.ndim != 2:
        raise ValueError("batched symmetric-power input must be two-dimensional")
    output_adjoint = torch.as_tensor(
        output_adjoint,
        dtype=input.dtype,
        device=input.device,
    )
    if output_adjoint.ndim != 3:
        raise ValueError(
            "output_adjoint must have shape [seed, batch, output]"
        )
    if int(output_adjoint.shape[1]) != int(input.shape[0]):
        raise ValueError("output_adjoint batch must match input")
    counts = torch.as_tensor(
        monomial_counts,
        dtype=torch.int64,
        device=input.device,
    )
    offsets = torch.as_tensor(
        output_offsets,
        dtype=torch.int64,
        device=input.device,
    ).reshape(-1)
    terms = torch.as_tensor(
        coefficient_terms,
        dtype=torch.int64,
        device=input.device,
    ).reshape(-1)
    outputs = torch.as_tensor(
        coefficient_outputs,
        dtype=torch.int64,
        device=input.device,
    ).reshape(-1)
    values = torch.as_tensor(
        coefficient_values,
        dtype=input.dtype,
        device=input.device,
    ).reshape(-1)
    reference_arguments = (
        output_adjoint,
        input,
        counts,
        offsets,
        terms,
        outputs,
        values,
    )
    require_native = _enabled("YE3T_REQUIRE_NATIVE")
    if backend == "reference":
        if require_native:
            raise RuntimeError(
                "YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback"
            )
        return symmetric_power_shared_monomial_batched_adjoint_reference(
            *reference_arguments
        )
    use_native = backend == "native" or require_native
    use_native = use_native or (
        backend == "auto"
        and (
            _prebuilt_extension() is not None
            or _enabled("YE3T_ENABLE_EXECUTION_PLAN_JIT")
        )
    )
    if not use_native:
        return symmetric_power_shared_monomial_batched_adjoint_reference(
            *reference_arguments
        )
    extension = _load_extension()
    if input.device.type == "cuda" and not bool(extension.has_cuda()):
        if backend == "auto" and not require_native:
            return symmetric_power_shared_monomial_batched_adjoint_reference(
                *reference_arguments
            )
        raise RuntimeError(
            "the installed YE3T native extension has no CUDA shared "
            "symmetric-power batched adjoint"
        )
    if input.device.type not in {"cpu", "cuda"}:
        raise ValueError(
            "native batched symmetric-power adjoints require CPU or CUDA"
        )
    if counts.ndim != 2 or int(counts.shape[1]) != int(input.shape[1]):
        raise ValueError(
            "monomial_counts must have shape [monomial, input_dimension]"
        )
    return (
        torch.ops.ye3t_runtime
        .symmetric_power_shared_monomial_batched_adjoint(
            output_adjoint.contiguous(),
            input.contiguous(),
            counts.contiguous(),
            offsets.contiguous(),
            terms.contiguous(),
            outputs.contiguous(),
            values.contiguous(),
        )
    )


def symmetric_power_shared_monomial_contraction(
    input,
    monomial_counts,
    output_offsets,
    coefficient_terms,
    coefficient_outputs,
    coefficient_values,
    *,
    backend="auto",
):
    """Dispatch a shared-monomial symmetric-power contraction."""

    backend = str(backend)
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")
    input = torch.as_tensor(input)
    if input.ndim < 1:
        raise ValueError("symmetric-power input requires a feature axis")
    input_dimension = int(input.shape[-1])
    counts = torch.as_tensor(
        monomial_counts,
        dtype=torch.int64,
        device=input.device,
    )
    offsets = torch.as_tensor(
        output_offsets,
        dtype=torch.int64,
        device=input.device,
    ).reshape(-1)
    terms = torch.as_tensor(
        coefficient_terms,
        dtype=torch.int64,
        device=input.device,
    ).reshape(-1)
    outputs = torch.as_tensor(
        coefficient_outputs,
        dtype=torch.int64,
        device=input.device,
    ).reshape(-1)
    values = torch.as_tensor(
        coefficient_values,
        dtype=input.dtype,
        device=input.device,
    ).reshape(-1)
    require_native = _enabled("YE3T_REQUIRE_NATIVE")
    reference_arguments = (
        input,
        counts,
        offsets,
        terms,
        outputs,
        values,
    )
    if backend == "reference":
        if require_native:
            raise RuntimeError(
                "YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback"
            )
        return symmetric_power_shared_monomial_reference(
            *reference_arguments
        )
    use_native = backend == "native" or require_native
    use_native = use_native or (
        backend == "auto"
        and (
            _prebuilt_extension() is not None
            or _enabled("YE3T_ENABLE_EXECUTION_PLAN_JIT")
        )
    )
    if not use_native:
        return symmetric_power_shared_monomial_reference(
            *reference_arguments
        )
    extension = _load_extension()
    if input.device.type == "cuda" and not bool(extension.has_cuda()):
        if backend == "auto" and not require_native:
            return symmetric_power_shared_monomial_reference(
                *reference_arguments
            )
        raise RuntimeError(
            "the installed YE3T native extension has no CUDA shared "
            "symmetric-power monomial operators"
        )
    if input.device.type not in {"cpu", "cuda"}:
        raise ValueError(
            "native shared symmetric-power monomials require CPU or CUDA"
        )
    if counts.ndim != 2 or int(counts.shape[1]) != input_dimension:
        raise ValueError(
            "monomial_counts must have shape [monomial, input_dimension]"
        )
    flat = input.reshape(-1, input_dimension).contiguous()
    output = torch.ops.ye3t_runtime.symmetric_power_shared_monomial(
        flat,
        counts.contiguous(),
        offsets.contiguous(),
        terms.contiguous(),
        outputs.contiguous(),
        values.contiguous(),
    )
    return output.reshape(
        tuple(input.shape[:-1]) + (int(output.shape[-1]),)
    )


def sparse_monomial_support_from_counts(monomial_counts):
    """Compile dense nonnegative occupations to canonical sparse support."""

    counts = torch.as_tensor(monomial_counts, dtype=torch.int64)
    if counts.ndim != 2:
        raise ValueError("monomial_counts must be a matrix")
    if bool(torch.any(counts < 0)):
        raise ValueError("monomial_counts must be nonnegative")
    offsets = [0]
    components = []
    exponents = []
    for row in counts.detach().cpu().tolist():
        for component, exponent in enumerate(row):
            if int(exponent) <= 0:
                continue
            components.append(int(component))
            exponents.append(int(exponent))
        offsets.append(len(components))
    device = counts.device
    return (
        torch.tensor(offsets, dtype=torch.int64, device=device),
        torch.tensor(components, dtype=torch.int64, device=device),
        torch.tensor(exponents, dtype=torch.int64, device=device),
    )


def _dense_counts_from_sparse_monomial_support(
    term_offsets,
    term_components,
    term_exponents,
    input_dimension,
):
    offsets = torch.as_tensor(term_offsets, dtype=torch.int64).reshape(-1)
    components = torch.as_tensor(
        term_components,
        dtype=torch.int64,
        device=offsets.device,
    ).reshape(-1)
    exponents = torch.as_tensor(
        term_exponents,
        dtype=torch.int64,
        device=offsets.device,
    ).reshape(-1)
    if offsets.numel() < 1:
        raise ValueError("term_offsets must contain at least one entry")
    if components.numel() != exponents.numel():
        raise ValueError(
            "term_components and term_exponents must have equal length"
        )
    offset_values = tuple(int(value) for value in offsets.cpu().tolist())
    if (
        offset_values[0] != 0
        or offset_values[-1] != int(components.numel())
        or any(
            left > right
            for left, right in zip(offset_values[:-1], offset_values[1:])
        )
    ):
        raise ValueError(
            "term_offsets must be monotone from zero through support size"
        )
    if bool(torch.any(exponents <= 0)):
        raise ValueError("sparse monomial exponents must be positive")
    if components.numel() and bool(
        torch.any((components < 0) | (components >= int(input_dimension)))
    ):
        raise ValueError("sparse monomial component is outside input width")
    counts = torch.zeros(
        (int(offsets.numel()) - 1, int(input_dimension)),
        dtype=torch.int64,
        device=offsets.device,
    )
    if components.numel():
        lengths = offsets[1:] - offsets[:-1]
        terms = torch.repeat_interleave(
            torch.arange(
                int(offsets.numel()) - 1,
                dtype=torch.int64,
                device=offsets.device,
            ),
            lengths,
        )
        counts.index_put_((terms, components), exponents, accumulate=True)
    return counts


def symmetric_power_shared_sparse_monomial_contraction(
    input,
    term_offsets,
    term_components,
    term_exponents,
    output_offsets,
    coefficient_terms,
    coefficient_outputs,
    coefficient_values,
    *,
    backend="auto",
):
    """Evaluate compiler-derived sparse monomial support exactly on CUDA."""

    backend = str(backend)
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")
    input = torch.as_tensor(input)
    if input.ndim < 1:
        raise ValueError("symmetric-power input requires a feature axis")
    input_dimension = int(input.shape[-1])
    support = tuple(
        torch.as_tensor(value, dtype=torch.int64, device=input.device)
        .reshape(-1)
        .contiguous()
        for value in (term_offsets, term_components, term_exponents)
    )
    offsets = torch.as_tensor(
        output_offsets,
        dtype=torch.int64,
        device=input.device,
    ).reshape(-1).contiguous()
    terms = torch.as_tensor(
        coefficient_terms,
        dtype=torch.int64,
        device=input.device,
    ).reshape(-1).contiguous()
    outputs = torch.as_tensor(
        coefficient_outputs,
        dtype=torch.int64,
        device=input.device,
    ).reshape(-1).contiguous()
    values = torch.as_tensor(
        coefficient_values,
        dtype=input.dtype,
        device=input.device,
    ).reshape(-1).contiguous()
    require_native = _enabled("YE3T_REQUIRE_NATIVE")
    if backend == "reference" and require_native:
        raise RuntimeError("YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback")
    use_native = bool(
        backend != "reference"
        and input.device.type == "cuda"
        and (
            _prebuilt_extension() is not None
            or _enabled("YE3T_ENABLE_EXECUTION_PLAN_JIT")
        )
    )
    if backend == "native" or require_native:
        use_native = True
    if use_native:
        extension = _load_extension()
        native_available = bool(
            input.device.type == "cuda"
            and extension.has_cuda()
            and int(extension.core_abi_version()) >= 39
            and hasattr(
                torch.ops.ye3t_runtime,
                "symmetric_power_shared_sparse_monomial",
            )
        )
        if native_available:
            flat = input.reshape(-1, input_dimension).contiguous()
            output = (
                torch.ops.ye3t_runtime
                .symmetric_power_shared_sparse_monomial(
                    flat,
                    support[0],
                    support[1],
                    support[2],
                    offsets,
                    terms,
                    outputs,
                    values,
                )
            )
            return output.reshape(
                tuple(input.shape[:-1]) + (int(output.shape[-1]),)
            )
        if backend == "native" or require_native:
            raise RuntimeError(
                "native sparse-support shared symmetric-power execution "
                "requires CUDA ABI 39"
            )
    dense_counts = _dense_counts_from_sparse_monomial_support(
        support[0],
        support[1],
        support[2],
        input_dimension,
    )
    return symmetric_power_shared_monomial_reference(
        input,
        dense_counts,
        offsets,
        terms,
        outputs,
        values,
    )


def _factorized_angular_packed_reference(
    packed_slots,
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
    source_dimension,
    reference_metadata=None,
    dense_analysis_values=None,
):
    source_dimension = int(source_dimension)
    if int(packed_slots.shape[1]) % source_dimension:
        raise ValueError(
            "packed slot width must be divisible by source_dimension"
        )
    input_dimension = int(packed_slots.shape[1]) // source_dimension
    source_slots = packed_slots.reshape(
        packed_slots.shape[0] * source_dimension,
        input_dimension,
    )
    nodes = []
    if reference_metadata is None:
        dimensions = tuple(int(value) for value in node_dimensions.tolist())
        leaf_offsets = tuple(
            int(value) for value in node_leaf_offsets.tolist()
        )
        left_nodes = tuple(int(value) for value in node_left.tolist())
        right_nodes = tuple(int(value) for value in node_right.tolist())
        coefficient_offsets = tuple(
            int(value) for value in node_coefficient_offsets.tolist()
        )
        roots = tuple(int(value) for value in root_nodes.tolist())
        dense_analysis_offsets = None
    else:
        dimensions = reference_metadata["node_dimensions"]
        leaf_offsets = reference_metadata["node_leaf_offsets"]
        left_nodes = reference_metadata["node_left"]
        right_nodes = reference_metadata["node_right"]
        coefficient_offsets = reference_metadata[
            "node_coefficient_offsets"
        ]
        roots = reference_metadata["root_nodes"]
        dense_analysis_offsets = reference_metadata[
            "dense_analysis_offsets"
        ]
    for node_index, dimension in enumerate(dimensions):
        leaf_offset = leaf_offsets[node_index]
        if leaf_offset >= 0:
            nodes.append(
                source_slots[
                    :,
                    leaf_offset : leaf_offset + dimension,
                ]
            )
            continue
        left = nodes[left_nodes[node_index]]
        right = nodes[right_nodes[node_index]]
        outer = torch.einsum("ba,bc->bac", left, right).reshape(
            source_slots.shape[0],
            -1,
        )
        if dense_analysis_values is None:
            synthesis = source_slots.new_zeros(
                (outer.shape[1], dimension)
            )
            start = coefficient_offsets[node_index]
            stop = coefficient_offsets[node_index + 1]
            synthesis = synthesis.index_put(
                (
                    coefficient_rows[start:stop],
                    coefficient_columns[start:stop],
                ),
                coefficient_values[start:stop],
                accumulate=True,
            )
            nodes.append(outer @ synthesis.conj())
        else:
            start = dense_analysis_offsets[node_index]
            stop = dense_analysis_offsets[node_index + 1]
            analysis = dense_analysis_values[start:stop].reshape(
                outer.shape[1],
                dimension,
            )
            nodes.append(outer @ analysis)
    outputs = []
    projection_dimension = (
        int(projection_values.numel()) // source_dimension
    )
    projection = projection_values.reshape(
        1,
        source_dimension,
        projection_dimension,
        1,
    )
    for root_index in roots:
        root = nodes[root_index].reshape(
            packed_slots.shape[0],
            source_dimension,
            1,
            -1,
        )
        outputs.append((root * projection).sum(dim=1))
    return torch.cat(tuple(outputs), dim=1).reshape(
        packed_slots.shape[0],
        -1,
    )


def _factorized_plan_buffers(
    execution_plan,
    angular_plan,
    dtype,
    device,
    materialize_reference_dense_tables=True,
):
    tables = {
        table.table_id: table for table in execution_plan.synthesis_tables
    }
    assemblies = {
        assembly.assembly_id: assembly
        for assembly in execution_plan.source_assemblies
    }
    assembly = assemblies[angular_plan.source_assembly_id]
    young = tables[angular_plan.young_synthesis_table_id]
    source_dimension = int(assembly.source_dimension)
    projection_values = [
        [0j] * int(young.output_dimension)
        for _ in range(source_dimension)
    ]
    for row, column, value in zip(
        assembly.row_indices,
        assembly.column_indices,
        assembly.values,
    ):
        source_column = int(column)
        for young_row, young_column, young_value in zip(
            young.row_indices,
            young.column_indices,
            young.values,
        ):
            if int(young_row) == int(row):
                projection_values[source_column][int(young_column)] += (
                    complex(value) * complex(young_value).conjugate()
                )
    input_offsets = []
    running_input_offset = 0
    for angular_L in angular_plan.input_Ls:
        input_offsets.append(running_input_offset)
        running_input_offset += 2 * int(angular_L) + 1
    node_indices = {
        node.node_id: index
        for index, node in enumerate(angular_plan.nodes)
    }
    node_offsets = []
    node_dimensions = []
    node_leaf_offsets = []
    node_left = []
    node_right = []
    node_coefficient_offsets = [0]
    coefficient_rows = []
    coefficient_columns = []
    coefficient_values = []
    node_synthesis_table_ids = []
    running_node_offset = 0
    for node in angular_plan.nodes:
        dimension = 2 * int(node.output_L) + 1
        node_offsets.append(running_node_offset)
        node_dimensions.append(dimension)
        running_node_offset += dimension
        if node.kind == "leaf":
            node_synthesis_table_ids.append(None)
            node_leaf_offsets.append(input_offsets[node.leaf_index])
            node_left.append(-1)
            node_right.append(-1)
        else:
            node_synthesis_table_ids.append(str(node.synthesis_table_id))
            node_leaf_offsets.append(-1)
            node_left.append(node_indices[node.left_node_id])
            node_right.append(node_indices[node.right_node_id])
            table = tables[node.synthesis_table_id]
            coefficient_rows.extend(table.row_indices)
            coefficient_columns.extend(table.column_indices)
            coefficient_values.extend(table.values)
        node_coefficient_offsets.append(len(coefficient_values))
    coefficient_values = _coefficient_values(
        tuple(coefficient_values),
        dtype,
    )
    reference_dense_table_byte_limit = 64 * 1024 * 1024
    reference_dense_element_count = int(
        sum(
            int(tables[table_id].input_dimension)
            * int(tables[table_id].output_dimension)
            for table_id in node_synthesis_table_ids
            if table_id is not None
        )
    )
    element_size = int(torch.empty((), dtype=dtype).element_size())
    reference_dense_synthesis_materialized = bool(
        materialize_reference_dense_tables
        and reference_dense_element_count * element_size
        <= reference_dense_table_byte_limit
    )
    dense_analysis_offsets = [0]
    dense_analysis_values = []
    if reference_dense_synthesis_materialized:
        for table_id in node_synthesis_table_ids:
            if table_id is not None:
                table = tables[table_id]
                dense = [
                    0j
                    for _index in range(
                        int(table.input_dimension)
                        * int(table.output_dimension)
                    )
                ]
                for row, column, value in zip(
                    table.row_indices,
                    table.column_indices,
                    table.values,
                ):
                    flat_index = (
                        int(row) * int(table.output_dimension)
                        + int(column)
                    )
                    dense[flat_index] += complex(value).conjugate()
                dense_analysis_values.extend(dense)
            dense_analysis_offsets.append(len(dense_analysis_values))
    else:
        dense_analysis_offsets.extend(
            0 for _table_id in node_synthesis_table_ids
        )
    projection_values = _coefficient_values(
        tuple(
            value
            for source_values in projection_values
            for value in source_values
        ),
        dtype,
    )
    subtree_identity = _factorized_angular_subtree_identity(
        execution_plan,
        angular_plan,
    )
    return {
        "node_offsets": torch.tensor(
            node_offsets,
            dtype=torch.int64,
            device=device,
        ),
        "node_dimensions": torch.tensor(
            node_dimensions,
            dtype=torch.int64,
            device=device,
        ),
        "node_leaf_offsets": torch.tensor(
            node_leaf_offsets,
            dtype=torch.int64,
            device=device,
        ),
        "node_left": torch.tensor(
            node_left,
            dtype=torch.int64,
            device=device,
        ),
        "node_right": torch.tensor(
            node_right,
            dtype=torch.int64,
            device=device,
        ),
        "node_coefficient_offsets": torch.tensor(
            node_coefficient_offsets,
            dtype=torch.int64,
            device=device,
        ),
        "coefficient_rows": torch.tensor(
            coefficient_rows,
            dtype=torch.int64,
            device=device,
        ),
        "coefficient_columns": torch.tensor(
            coefficient_columns,
            dtype=torch.int64,
            device=device,
        ),
        "coefficient_values": torch.tensor(
            coefficient_values,
            dtype=dtype,
            device=device,
        ),
        "dense_analysis_offsets": torch.tensor(
            dense_analysis_offsets,
            dtype=torch.int64,
            device=device,
        ),
        "dense_analysis_values": torch.tensor(
            _coefficient_values(tuple(dense_analysis_values), dtype),
            dtype=dtype,
            device=device,
        ),
        "reference_dense_synthesis_materialized": int(
            reference_dense_synthesis_materialized
        ),
        "reference_dense_synthesis_element_count": int(
            reference_dense_element_count
        ),
        "reference_dense_synthesis_byte_limit": int(
            reference_dense_table_byte_limit
        ),
        "root_nodes": torch.tensor(
            tuple(
                node_indices[node_id]
                for node_id in angular_plan.root_node_ids
            ),
            dtype=torch.int64,
            device=device,
        ),
        "projection_values": torch.tensor(
            projection_values,
            dtype=dtype,
            device=device,
        ),
        "input_dimension": running_input_offset,
        "source_dimension": source_dimension,
        "workspace_dimension": running_node_offset,
        "output_dimension": (
            len(angular_plan.root_node_ids)
            * int(young.output_dimension)
            * (2 * int(angular_plan.output_L) + 1)
        ),
        "logical_channel_tableau_dimension": (
            len(angular_plan.root_node_ids)
            * int(young.output_dimension)
        ),
        "magnetic_dimension": 2 * int(angular_plan.output_L) + 1,
        "node_operator_hashes": subtree_identity[
            "node_operator_hashes"
        ],
        "node_subtree_hashes": subtree_identity[
            "node_subtree_hashes"
        ],
    }


def _repeated_block_power_layout(
    module,
    physical_leaf_offsets,
    physical_input_dimension,
):
    """Describe repeated compiler-bound physical leaves without changing them."""

    input_Ls = tuple(int(value) for value in module.input_Ls)
    physical_leaf_offsets = tuple(physical_leaf_offsets)
    physical_input_dimension = int(physical_input_dimension)
    source_dimension = int(module.source_dimension)
    nested_offsets = bool(
        len(physical_leaf_offsets) == source_dimension
        and physical_leaf_offsets
        and isinstance(physical_leaf_offsets[0], (tuple, list))
    )
    if nested_offsets:
        offsets_by_source = tuple(
            tuple(int(value) for value in offsets)
            for offsets in physical_leaf_offsets
        )
    else:
        offsets = tuple(int(value) for value in physical_leaf_offsets)
        offsets_by_source = tuple(
            offsets for _source_index in range(source_dimension)
        )
    if any(len(offsets) != len(input_Ls) for offsets in offsets_by_source):
        raise ValueError(
            "repeated-block physical offsets must cover every source "
            "coordinate and plan leaf"
        )
    if physical_input_dimension <= 0:
        raise ValueError(
            "repeated-block physical input dimension must be positive"
        )
    local_offsets = []
    local_offset = 0
    for angular_L in input_Ls:
        magnetic_dimension = 2 * int(angular_L) + 1
        local_offsets.append(int(local_offset))
        local_offset += magnetic_dimension

    local_to_variable_offset_by_source = []
    source_block_physical_offsets = []
    source_block_angular_Ls = []
    source_block_multiplicities = []
    source_repeated_slot_counts = []
    physical_indices = []
    variable_offset = 0
    monomial_upper_bound = 0
    for source_index, offsets in enumerate(offsets_by_source):
        blocks = {}
        local_to_block = {}
        occupied = {}
        for local, physical_offset, angular_L in zip(
            local_offsets,
            offsets,
            input_Ls,
        ):
            magnetic_dimension = 2 * int(angular_L) + 1
            if (
                physical_offset < 0
                or physical_offset + magnetic_dimension
                > physical_input_dimension
            ):
                raise ValueError(
                    "repeated-block physical leaf is outside the source bank"
                )
            block = (int(physical_offset), int(angular_L))
            blocks[block] = int(blocks.get(block, 0)) + 1
            local_to_block[int(local)] = block

        ordered_blocks = tuple(sorted(blocks))
        block_variable_offsets = {}
        source_upper_bound = 1
        for physical_offset, angular_L in ordered_blocks:
            magnetic_dimension = 2 * int(angular_L) + 1
            for coordinate in range(
                int(physical_offset),
                int(physical_offset + magnetic_dimension),
            ):
                previous = occupied.get(int(coordinate))
                if previous is not None and previous != (
                    int(physical_offset),
                    int(angular_L),
                ):
                    raise ValueError(
                        "repeated-block physical leaf slices overlap"
                    )
                occupied[int(coordinate)] = (
                    int(physical_offset),
                    int(angular_L),
                )
                physical_indices.append(
                    int(
                        source_index * physical_input_dimension
                        + coordinate
                    )
                )
            block_variable_offsets[
                (int(physical_offset), int(angular_L))
            ] = int(variable_offset)
            variable_offset += magnetic_dimension
            source_upper_bound *= int(
                math.comb(
                    magnetic_dimension + int(blocks[(physical_offset, angular_L)]) - 1,
                    int(blocks[(physical_offset, angular_L)]),
                )
            )
        monomial_upper_bound += int(source_upper_bound)
        local_to_variable_offset_by_source.append(
            {
                int(local): int(
                    block_variable_offsets[local_to_block[int(local)]]
                )
                for local in local_offsets
            }
        )
        source_block_physical_offsets.append(
            tuple(int(block[0]) for block in ordered_blocks)
        )
        source_block_angular_Ls.append(
            tuple(int(block[1]) for block in ordered_blocks)
        )
        source_block_multiplicities.append(
            tuple(int(blocks[block]) for block in ordered_blocks)
        )
        source_repeated_slot_counts.append(
            int(
                sum(
                    multiplicity
                    for multiplicity in blocks.values()
                    if int(multiplicity) > 1
                )
            )
        )

    uniform_block_shape = bool(
        source_block_angular_Ls
        and all(
            angular_Ls == source_block_angular_Ls[0]
            and multiplicities == source_block_multiplicities[0]
            for angular_Ls, multiplicities in zip(
                source_block_angular_Ls,
                source_block_multiplicities,
            )
        )
    )
    total_input_dimension = int(variable_offset)
    return {
        "physical_input_indices": tuple(physical_indices),
        "physical_leaf_offsets_by_source": tuple(offsets_by_source),
        "physical_input_dimension": int(physical_input_dimension),
        "source_dimension": int(source_dimension),
        "input_dimension": int(total_input_dimension),
        "local_to_variable_offset_by_source": tuple(
            local_to_variable_offset_by_source
        ),
        "block_physical_offsets": (
            tuple(source_block_physical_offsets[0])
            if uniform_block_shape
            else ()
        ),
        "block_angular_Ls": (
            tuple(source_block_angular_Ls[0])
            if uniform_block_shape
            else ()
        ),
        "block_multiplicities": (
            tuple(source_block_multiplicities[0])
            if uniform_block_shape
            else ()
        ),
        "source_block_physical_offsets": tuple(source_block_physical_offsets),
        "source_block_angular_Ls": tuple(source_block_angular_Ls),
        "source_block_multiplicities": tuple(source_block_multiplicities),
        "repeated_slot_count": int(min(source_repeated_slot_counts)),
        "total_repeated_slot_count": int(sum(source_repeated_slot_counts)),
        "monomial_upper_bound": int(monomial_upper_bound),
        "estimated_count_table_bytes": int(
            monomial_upper_bound * total_input_dimension * 8
        ),
    }


def _repeated_block_power_polynomial_buffers(
    module,
    physical_leaf_offsets,
    physical_input_dimension,
):
    """Lower one exact repeated-block angular DAG to sparse monomials."""

    layout = _repeated_block_power_layout(
        module,
        physical_leaf_offsets,
        physical_input_dimension,
    )
    input_Ls = tuple(int(value) for value in module.input_Ls)
    if len(input_Ls) < 2 or int(layout["repeated_slot_count"]) < 2:
        raise ValueError(
            "repeated-block power lowering requires repeated physical leaves"
    )
    input_dimension = int(layout["input_dimension"])
    dimensions = tuple(
        int(value) for value in module.node_dimensions.detach().cpu().tolist()
    )
    leaf_offsets = tuple(
        int(value) for value in module.node_leaf_offsets.detach().cpu().tolist()
    )
    left_nodes = tuple(
        int(value) for value in module.node_left.detach().cpu().tolist()
    )
    right_nodes = tuple(
        int(value) for value in module.node_right.detach().cpu().tolist()
    )
    coefficient_offsets = tuple(
        int(value)
        for value in module.node_coefficient_offsets.detach().cpu().tolist()
    )
    coefficient_rows = tuple(
        int(value) for value in module.coefficient_rows.detach().cpu().tolist()
    )
    coefficient_columns = tuple(
        int(value)
        for value in module.coefficient_columns.detach().cpu().tolist()
    )
    coefficient_values = tuple(
        complex(value)
        for value in module.coefficient_values.detach().cpu().tolist()
    )
    nodes_by_source = []
    for source_index in range(int(module.source_dimension)):
        nodes = []
        for node_index, dimension in enumerate(dimensions):
            if leaf_offsets[node_index] >= 0:
                variable_offset = int(
                    layout["local_to_variable_offset_by_source"][
                        source_index
                    ][int(leaf_offsets[node_index])]
                )
                components = []
                for component in range(int(dimension)):
                    exponents = [0] * input_dimension
                    exponents[variable_offset + component] = 1
                    components.append(
                        {tuple(exponents): 1.0 + 0.0j}
                    )
                nodes.append(tuple(components))
                continue
            left = nodes[left_nodes[node_index]]
            right = nodes[right_nodes[node_index]]
            right_dimension = len(right)
            components = [dict() for _component in range(dimension)]
            start = coefficient_offsets[node_index]
            stop = coefficient_offsets[node_index + 1]
            for row, column, value in zip(
                coefficient_rows[start:stop],
                coefficient_columns[start:stop],
                coefficient_values[start:stop],
            ):
                left_component = int(row) // int(right_dimension)
                right_component = int(row) % int(right_dimension)
                for left_exponents, left_value in left[
                    left_component
                ].items():
                    for right_exponents, right_value in right[
                        right_component
                    ].items():
                        exponents = tuple(
                            int(a) + int(b)
                            for a, b in zip(
                                left_exponents,
                                right_exponents,
                            )
                        )
                        contribution = (
                            complex(left_value)
                            * complex(right_value)
                            * complex(value).conjugate()
                        )
                        components[int(column)][exponents] = (
                            components[int(column)].get(
                                exponents,
                                0.0 + 0.0j,
                            )
                            + contribution
                        )
            nodes.append(tuple(components))
        nodes_by_source.append(tuple(nodes))

    root_nodes = tuple(
        int(value) for value in module.root_nodes.detach().cpu().tolist()
    )
    magnetic_dimension = int(module.magnetic_dimension)
    projection = tuple(
        complex(value)
        for value in module.projection_values.detach().cpu().tolist()
    )
    source_dimension = int(module.source_dimension)
    if len(projection) % source_dimension:
        raise ValueError(
            "repeated-block source projection is not divisible by its source dimension"
        )
    projection_dimension = int(len(projection) // source_dimension)
    if projection_dimension * len(root_nodes) * magnetic_dimension != int(
        module.output_dimension
    ):
        raise ValueError(
            "repeated-block power output layout does not match root projection"
        )
    output_polynomials = []
    for root_index in root_nodes:
        for projection_index in range(projection_dimension):
            for component_index in range(magnetic_dimension):
                polynomial = {}
                for source_index in range(source_dimension):
                    root = nodes_by_source[source_index][root_index]
                    if len(root) != magnetic_dimension:
                        raise ValueError(
                            "repeated-block power root dimension does not match output L"
                        )
                    projection_value = projection[
                        source_index * projection_dimension
                        + projection_index
                    ]
                    for exponents, value in root[
                        component_index
                    ].items():
                        contribution = (
                            complex(value) * projection_value
                        )
                        polynomial[exponents] = (
                            polynomial.get(exponents, 0.0 + 0.0j)
                            + contribution
                        )
                output_polynomials.append(
                    {
                        exponents: value
                        for exponents, value in polynomial.items()
                        if value != 0.0
                    }
                )

    monomial_set = {
        exponents
        for polynomial in output_polynomials
        for exponents in polynomial
    }
    monomials = tuple(sorted(monomial_set))
    monomial_indices = {
        exponents: index for index, exponents in enumerate(monomials)
    }
    output_offsets = [0]
    coefficient_terms = []
    coefficient_outputs = []
    sparse_values = []
    for output_index, polynomial in enumerate(output_polynomials):
        for exponents, value in sorted(polynomial.items()):
            coefficient_terms.append(monomial_indices[exponents])
            coefficient_outputs.append(int(output_index))
            sparse_values.append(value)
        output_offsets.append(len(coefficient_terms))
    device = module.coefficient_values.device
    return {
        "monomial_counts": torch.tensor(
            monomials,
            dtype=torch.int64,
            device=device,
        ),
        "output_offsets": torch.tensor(
            output_offsets,
            dtype=torch.int64,
            device=device,
        ),
        "coefficient_terms": torch.tensor(
            coefficient_terms,
            dtype=torch.int64,
            device=device,
        ),
        "coefficient_outputs": torch.tensor(
            coefficient_outputs,
            dtype=torch.int64,
            device=device,
        ),
        "coefficient_values": torch.tensor(
            sparse_values,
            dtype=module.coefficient_values.dtype,
            device=device,
        ),
        "input_dimension": int(input_dimension),
        "power": int(len(input_Ls)),
        "output_dimension": int(module.output_dimension),
        **layout,
    }


class YE3TFactorizedAngularModule(torch.nn.Module):
    """Prepared ordinary or role-resolved factorized angular DAG runtime."""

    def __init__(
        self,
        execution_plan,
        factorized_angular_plan_id=None,
        backend="auto",
        dtype=None,
        device=None,
    ):
        super().__init__()
        if not isinstance(execution_plan, YE3TExecutionPlan):
            execution_plan = YE3TExecutionPlan.from_dict(execution_plan)
        angular_plans = {
            plan.plan_id: plan
            for plan in execution_plan.factorized_angular_plans
        }
        if factorized_angular_plan_id is None:
            if len(angular_plans) != 1:
                raise ValueError(
                    "factorized_angular_plan_id is required for zero or "
                    "multiple angular plans"
                )
            angular_plan = next(iter(angular_plans.values()))
        else:
            angular_plan = angular_plans[
                str(factorized_angular_plan_id)
            ]
        backend = str(backend)
        if backend not in {"auto", "native", "reference"}:
            raise ValueError("backend must be 'auto', 'native', or 'reference'")
        require_native = _enabled("YE3T_REQUIRE_NATIVE")
        if backend == "reference" and require_native:
            raise RuntimeError(
                "YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback"
            )
        use_native = backend == "native" or require_native
        use_native = use_native or (
            backend == "auto"
            and (
                _prebuilt_extension() is not None
                or _enabled("YE3T_ENABLE_EXECUTION_PLAN_JIT")
            )
        )
        extension = _load_extension() if use_native else None
        tables = {
            table.table_id: table for table in execution_plan.synthesis_tables
        }
        coefficient_complex = any(
            value.imag != 0.0
            for table in tables.values()
            for value in table.values
        )
        if dtype is None:
            dtype = torch.complex128 if coefficient_complex else torch.float64
        device = torch.device("cpu" if device is None else device)
        if (
            use_native
            and device.type == "cuda"
            and not bool(extension.has_cuda())
            and backend == "auto"
            and not require_native
        ):
            use_native = False
            extension = None
        if use_native and device.type == "cuda":
            if not bool(extension.has_cuda()):
                raise RuntimeError(
                    "the installed YE3T native extension has no CUDA "
                    "factorized angular runtime"
                )
            resolved_backend = "native_cuda"
        elif use_native and device.type == "cpu":
            resolved_backend = "native_cpu"
        elif use_native:
            raise ValueError(
                "native factorized angular runtime requires CPU or CUDA"
            )
        else:
            resolved_backend = "torch_reference"
        buffers = _factorized_plan_buffers(
            execution_plan,
            angular_plan,
            dtype,
            device,
            materialize_reference_dense_tables=(
                resolved_backend == "torch_reference"
            ),
        )
        reference_metadata = {
            "node_dimensions": tuple(
                int(value)
                for value in buffers["node_dimensions"].detach().cpu().tolist()
            ),
            "node_leaf_offsets": tuple(
                int(value)
                for value in buffers["node_leaf_offsets"].detach().cpu().tolist()
            ),
            "node_left": tuple(
                int(value)
                for value in buffers["node_left"].detach().cpu().tolist()
            ),
            "node_right": tuple(
                int(value)
                for value in buffers["node_right"].detach().cpu().tolist()
            ),
            "node_coefficient_offsets": tuple(
                int(value)
                for value in buffers["node_coefficient_offsets"]
                .detach()
                .cpu()
                .tolist()
            ),
            "root_nodes": tuple(
                int(value)
                for value in buffers["root_nodes"].detach().cpu().tolist()
            ),
            "dense_analysis_offsets": tuple(
                int(value)
                for value in buffers["dense_analysis_offsets"]
                .detach()
                .cpu()
                .tolist()
            ),
        }
        for name, value in buffers.items():
            if isinstance(value, torch.Tensor):
                self.register_buffer(name, value, persistent=False)
            elif name in {
                "node_operator_hashes",
                "node_subtree_hashes",
            }:
                setattr(self, name, tuple(str(item) for item in value))
            else:
                setattr(self, name, int(value))
        self._reference_metadata = reference_metadata
        self.input_Ls = tuple(int(value) for value in angular_plan.input_Ls)
        self.plan_hash = str(execution_plan.plan_hash)
        self.factorized_angular_plan_id = str(angular_plan.plan_id)
        self.requested_backend = backend
        self.resolved_backend = resolved_backend
        self.last_linear_readout_backend = None
        self.analysis_orientation = "conjugate_transpose"

    def _packed_slots(self, slot_values):
        slots = tuple(slot_values)
        if len(slots) != len(self.input_Ls):
            raise ValueError("slot tensor count does not match input_Ls")
        if self.source_dimension == 1:
            leading_shape = tuple(slots[0].shape[:-1])
        else:
            if slots[0].ndim < 2:
                raise ValueError(
                    "role-resolved slots require a source-coordinate axis"
                )
            leading_shape = tuple(slots[0].shape[:-2])
        working = []
        for slot, angular_L in zip(slots, self.input_Ls):
            if int(slot.shape[-1]) != 2 * angular_L + 1:
                raise ValueError("slot magnetic dimension does not match input_Ls")
            if self.source_dimension == 1:
                if tuple(slot.shape[:-1]) != leading_shape:
                    raise ValueError(
                        "slot tensors must share leading dimensions"
                    )
                slot = slot.unsqueeze(-2)
            else:
                if tuple(slot.shape[:-2]) != leading_shape:
                    raise ValueError(
                        "slot tensors must share leading dimensions"
                    )
                if int(slot.shape[-2]) != self.source_dimension:
                    raise ValueError(
                        "slot source axis does not match source assembly"
                    )
            working.append(
                slot.to(
                    dtype=self.coefficient_values.dtype,
                    device=self.coefficient_values.device,
                )
            )
        packed = torch.cat(tuple(working), dim=-1).reshape(
            -1,
            self.source_dimension * self.input_dimension,
        ).contiguous()
        return packed, leading_shape

    def _native_arguments(self):
        return (
            self.node_offsets,
            self.node_dimensions,
            self.node_leaf_offsets,
            self.node_left,
            self.node_right,
            self.node_coefficient_offsets,
            self.coefficient_rows,
            self.coefficient_columns,
            self.coefficient_values,
            self.root_nodes,
            self.projection_values,
        )

    def forward(self, slot_values):
        packed, leading_shape = self._packed_slots(slot_values)
        if int(packed.shape[0]) == 0:
            output = packed[:, :1].expand(-1, self.output_dimension)
            return output.reshape(
                leading_shape
                + (
                    self.logical_channel_tableau_dimension,
                    self.magnetic_dimension,
                )
            )
        arguments = self._native_arguments()
        if self.resolved_backend in {"native_cpu", "native_cuda"}:
            output = torch.ops.ye3t_runtime.factorized_angular(
                packed,
                *arguments,
                self.source_dimension,
                self.workspace_dimension,
                self.output_dimension,
            )
        else:
            output = _factorized_angular_packed_reference(
                packed,
                *arguments,
                self.source_dimension,
                reference_metadata=self._reference_metadata,
                dense_analysis_values=(
                    self.dense_analysis_values
                    if self.reference_dense_synthesis_materialized
                    else None
                ),
            )
        return output.reshape(
            leading_shape
            + (
                self.logical_channel_tableau_dimension,
                self.magnetic_dimension,
            )
        )

    def linear_readout(self, slot_values, weight, bias):
        packed, leading_shape = self._packed_slots(slot_values)
        weight = torch.as_tensor(
            weight,
            dtype=self.coefficient_values.dtype,
            device=self.coefficient_values.device,
        )
        bias = torch.as_tensor(
            bias,
            dtype=self.coefficient_values.dtype,
            device=self.coefficient_values.device,
        ).reshape(())
        if weight.ndim != 1 or int(weight.numel()) != self.output_dimension:
            raise ValueError(
                "weight must be a vector matching factorized output dimension"
            )
        if int(packed.shape[0]) == 0:
            self.last_linear_readout_backend = "empty_batch"
            return packed[:, 0].reshape(leading_shape)
        native_backend = self.resolved_backend in {
            "native_cpu",
            "native_cuda",
        }
        measured_complex_role_materialization = bool(
            self.requested_backend == "auto"
            and self.resolved_backend == "native_cuda"
            and torch.is_complex(self.coefficient_values)
            and self.source_dimension > 1
        )
        if native_backend and not measured_complex_role_materialization:
            output = torch.ops.ye3t_runtime.factorized_angular_linear(
                packed,
                *self._native_arguments(),
                self.source_dimension,
                self.workspace_dimension,
                weight.contiguous(),
                bias.contiguous(),
            )
            self.last_linear_readout_backend = (
                "native_fused_factorized_linear"
            )
        elif native_backend:
            features = torch.ops.ye3t_runtime.factorized_angular(
                packed,
                *self._native_arguments(),
                self.source_dimension,
                self.workspace_dimension,
                self.output_dimension,
            )
            output = features @ weight + bias
            self.last_linear_readout_backend = (
                "native_materialized_measured_complex_role"
            )
        else:
            features = _factorized_angular_packed_reference(
                packed,
                *self._native_arguments(),
                self.source_dimension,
                reference_metadata=self._reference_metadata,
                dense_analysis_values=(
                    self.dense_analysis_values
                    if self.reference_dense_synthesis_materialized
                    else None
                ),
            )
            output = features @ weight + bias
            self.last_linear_readout_backend = "torch_reference_materialized"
        return output.reshape(leading_shape)

    def runtime_report(self):
        return {
            "backend": str(self.resolved_backend),
            "native": bool(
                self.resolved_backend in {"native_cpu", "native_cuda"}
            ),
            "operation": "factorized_angular_source_analysis",
            "factorized_angular_plan_id": str(
                self.factorized_angular_plan_id
            ),
            "plan_hash": str(self.plan_hash),
            "input_Ls": tuple(self.input_Ls),
            "source_dimension": int(self.source_dimension),
            "workspace_dimension": int(self.workspace_dimension),
            "runtime_device_scalar_metadata_reads": 0,
            "root_path_count": int(self.root_nodes.numel()),
            "unique_node_count": int(self.node_offsets.numel()),
            "reference_dense_synthesis_materialized": bool(
                self.reference_dense_synthesis_materialized
            ),
            "reference_dense_synthesis_element_count": int(
                self.reference_dense_synthesis_element_count
            ),
            "reference_dense_synthesis_bytes": int(
                self.dense_analysis_values.numel()
                * self.dense_analysis_values.element_size()
            ),
            "reference_dense_synthesis_byte_limit": int(
                self.reference_dense_synthesis_byte_limit
            ),
            "per_forward_synthesis_table_reconstruction": bool(
                self.resolved_backend == "torch_reference"
                and not self.reference_dense_synthesis_materialized
            ),
            "last_linear_readout_backend": (
                self.last_linear_readout_backend
            ),
            "linear_readout_auto_policy": (
                "materialize_native_complex_role_otherwise_fuse"
            ),
            "analysis_orientation": str(self.analysis_orientation),
        }


def _pack_shared_physical_factorized_nodes(
    modules,
    physical_leaf_offsets,
    physical_input_dimension,
):
    """Pack exact cross-plan common subtrees over one physical source bank."""

    first = modules[0]
    device = first.node_offsets.device
    identity_to_global = {}
    local_to_global_by_module = []
    node_offsets = []
    node_dimensions = []
    node_leaf_offsets = []
    node_left = []
    node_right = []
    node_coefficient_offsets = [0]
    coefficient_rows = []
    coefficient_columns = []
    coefficient_values = []
    workspace_dimension = 0
    naive_node_count = 0
    naive_workspace_dimension = 0

    for module_index, module in enumerate(modules):
        dimensions = tuple(
            int(value)
            for value in module.node_dimensions.detach().cpu().tolist()
        )
        leaf_offsets = tuple(
            int(value)
            for value in module.node_leaf_offsets.detach().cpu().tolist()
        )
        left_nodes = tuple(
            int(value)
            for value in module.node_left.detach().cpu().tolist()
        )
        right_nodes = tuple(
            int(value)
            for value in module.node_right.detach().cpu().tolist()
        )
        coefficient_offsets = tuple(
            int(value)
            for value in (
                module.node_coefficient_offsets.detach().cpu().tolist()
            )
        )
        offsets = physical_leaf_offsets[module_index]
        original_offset = 0
        physical_by_local_offset = {}
        for physical_offset, angular_L in zip(offsets, module.input_Ls):
            magnetic_dimension = 2 * int(angular_L) + 1
            if (
                physical_offset < 0
                or physical_offset + magnetic_dimension
                > physical_input_dimension
            ):
                raise ValueError(
                    "physical leaf offset is outside the source bank"
                )
            physical_by_local_offset[int(original_offset)] = int(
                physical_offset
            )
            original_offset += magnetic_dimension

        local_to_global = []
        for local_node, dimension in enumerate(dimensions):
            leaf_offset = leaf_offsets[local_node]
            operator_hash = module.node_operator_hashes[local_node]
            if leaf_offset >= 0:
                if leaf_offset not in physical_by_local_offset:
                    raise ValueError(
                        "factorized leaf offset is not bound to a physical "
                        "source-bank slice"
                    )
                physical_offset = physical_by_local_offset[leaf_offset]
                identity = (
                    "physical_leaf",
                    str(operator_hash),
                    int(physical_offset),
                )
                global_left = -1
                global_right = -1
            else:
                global_left = local_to_global[left_nodes[local_node]]
                global_right = local_to_global[right_nodes[local_node]]
                identity = (
                    "merge",
                    str(operator_hash),
                    int(global_left),
                    int(global_right),
                )
                physical_offset = -1

            global_node = identity_to_global.get(identity)
            if global_node is None:
                global_node = len(node_offsets)
                identity_to_global[identity] = global_node
                node_offsets.append(int(workspace_dimension))
                node_dimensions.append(int(dimension))
                node_leaf_offsets.append(int(physical_offset))
                node_left.append(int(global_left))
                node_right.append(int(global_right))
                workspace_dimension += int(dimension)
                if leaf_offset < 0:
                    start = coefficient_offsets[local_node]
                    stop = coefficient_offsets[local_node + 1]
                    coefficient_rows.append(
                        module.coefficient_rows[start:stop]
                    )
                    coefficient_columns.append(
                        module.coefficient_columns[start:stop]
                    )
                    coefficient_values.append(
                        module.coefficient_values[start:stop]
                    )
                    node_coefficient_offsets.append(
                        node_coefficient_offsets[-1] + stop - start
                    )
                else:
                    node_coefficient_offsets.append(
                        node_coefficient_offsets[-1]
                    )
            local_to_global.append(int(global_node))

        naive_node_count += len(dimensions)
        naive_workspace_dimension += int(module.workspace_dimension)
        local_to_global_by_module.append(tuple(local_to_global))

    def integer_tensor(values):
        return torch.tensor(
            values,
            dtype=torch.int64,
            device=device,
        )

    return {
        "node_offsets": integer_tensor(node_offsets),
        "node_dimensions": integer_tensor(node_dimensions),
        "node_leaf_offsets": integer_tensor(node_leaf_offsets),
        "node_left": integer_tensor(node_left),
        "node_right": integer_tensor(node_right),
        "node_coefficient_offsets": integer_tensor(
            node_coefficient_offsets
        ),
        "coefficient_rows": (
            torch.cat(tuple(coefficient_rows))
            if coefficient_rows
            else first.coefficient_rows.new_empty((0,))
        ),
        "coefficient_columns": (
            torch.cat(tuple(coefficient_columns))
            if coefficient_columns
            else first.coefficient_columns.new_empty((0,))
        ),
        "coefficient_values": (
            torch.cat(tuple(coefficient_values))
            if coefficient_values
            else first.coefficient_values.new_empty((0,))
        ),
        "local_to_global_by_module": tuple(
            local_to_global_by_module
        ),
        "naive_node_count": int(naive_node_count),
        "unique_node_count": int(len(node_offsets)),
        "naive_workspace_dimension": int(
            naive_workspace_dimension
        ),
        "workspace_dimension": int(workspace_dimension),
    }


class _PackedFactorizedAngularPlanGroup(torch.nn.Module):
    """Block-compose compatible exact factorized plans into one native call."""

    def __init__(
        self,
        modules,
        plan_indices,
        physical_leaf_offsets=None,
        physical_input_dimension=None,
        heterogeneous_roots=False,
        share_physical_subtrees=True,
    ):
        super().__init__()
        modules = tuple(modules)
        plan_indices = tuple(int(index) for index in plan_indices)
        heterogeneous_roots = bool(heterogeneous_roots)
        if not modules or len(modules) != len(plan_indices):
            raise ValueError(
                "factorized plan groups require matching modules and indices"
            )
        first = modules[0]
        if any(
            module.resolved_backend != first.resolved_backend
            or int(module.source_dimension) != int(first.source_dimension)
            or module.coefficient_values.dtype
            != first.coefficient_values.dtype
            or module.coefficient_values.device
            != first.coefficient_values.device
            or (
                not heterogeneous_roots
                and (
                    int(module.magnetic_dimension)
                    != int(first.magnetic_dimension)
                    or not torch.equal(
                        module.projection_values,
                        first.projection_values,
                    )
                )
            )
            for module in modules[1:]
        ):
            raise ValueError(
                "factorized plan group members are not exactly compatible"
            )
        if (physical_leaf_offsets is None) != (
            physical_input_dimension is None
        ):
            raise ValueError(
                "physical leaf offsets and input dimension must be supplied "
                "together"
            )
        if physical_leaf_offsets is not None:
            physical_leaf_offsets = tuple(
                tuple(int(offset) for offset in offsets)
                for offsets in physical_leaf_offsets
            )
            if len(physical_leaf_offsets) != len(modules):
                raise ValueError(
                    "physical leaf offsets must cover every grouped plan"
                )
            physical_input_dimension = int(physical_input_dimension)
            if physical_input_dimension <= 0:
                raise ValueError(
                    "physical input dimension must be positive"
                )
        shared_nodes = None
        share_physical_subtrees = bool(share_physical_subtrees)
        if (
            physical_leaf_offsets is not None
            and heterogeneous_roots
            and share_physical_subtrees
        ):
            shared_nodes = _pack_shared_physical_factorized_nodes(
                modules,
                physical_leaf_offsets,
                physical_input_dimension,
            )

        node_offsets = []
        node_dimensions = []
        node_leaf_offsets = []
        node_left = []
        node_right = []
        node_coefficient_offsets = []
        coefficient_rows = []
        coefficient_columns = []
        coefficient_values = []
        root_nodes = []
        root_projection_starts = []
        root_projection_dimensions = []
        root_output_offsets = []
        projection_matrices = []
        node_base = 0
        input_base = 0
        workspace_base = 0
        coefficient_base = 0
        projection_base = 0
        output_base = 0
        output_slices = []
        for module_index, (module, plan_index) in enumerate(
            zip(modules, plan_indices)
        ):
            if shared_nodes is None:
                node_offsets.append(
                    module.node_offsets + int(workspace_base)
                )
                node_dimensions.append(module.node_dimensions)
                if physical_leaf_offsets is None:
                    remapped_leaf_offsets = torch.where(
                        module.node_leaf_offsets >= 0,
                        module.node_leaf_offsets + int(input_base),
                        module.node_leaf_offsets,
                    )
                else:
                    offsets = physical_leaf_offsets[module_index]
                    if len(offsets) != len(module.input_Ls):
                        raise ValueError(
                            "physical leaf offsets must cover every plan leaf"
                        )
                    original_offset = 0
                    remapped_leaf_offsets = module.node_leaf_offsets
                    for offset, angular_L in zip(
                        offsets,
                        module.input_Ls,
                    ):
                        magnetic_dimension = 2 * int(angular_L) + 1
                        if (
                            offset < 0
                            or offset + magnetic_dimension
                            > physical_input_dimension
                        ):
                            raise ValueError(
                                "physical leaf offset is outside the "
                                "source bank"
                            )
                        remapped_leaf_offsets = torch.where(
                            module.node_leaf_offsets == int(original_offset),
                            int(offset),
                            remapped_leaf_offsets,
                        )
                        original_offset += magnetic_dimension
                node_leaf_offsets.append(remapped_leaf_offsets)
                node_left.append(
                    torch.where(
                        module.node_left >= 0,
                        module.node_left + int(node_base),
                        module.node_left,
                    )
                )
                node_right.append(
                    torch.where(
                        module.node_right >= 0,
                        module.node_right + int(node_base),
                        module.node_right,
                    )
                )
                offsets = (
                    module.node_coefficient_offsets
                    + int(coefficient_base)
                )
                if node_coefficient_offsets:
                    offsets = offsets[1:]
                node_coefficient_offsets.append(offsets)
                coefficient_rows.append(module.coefficient_rows)
                coefficient_columns.append(module.coefficient_columns)
                coefficient_values.append(module.coefficient_values)
                root_nodes.append(module.root_nodes + int(node_base))
            else:
                local_to_global = shared_nodes[
                    "local_to_global_by_module"
                ][module_index]
                root_nodes.append(
                    torch.tensor(
                        tuple(
                            local_to_global[int(root)]
                            for root in (
                                module.root_nodes.detach().cpu().tolist()
                            )
                        ),
                        dtype=torch.int64,
                        device=module.root_nodes.device,
                    )
                )
            if heterogeneous_roots:
                if (
                    int(module.projection_values.numel())
                    % int(module.source_dimension)
                    != 0
                ):
                    raise ValueError(
                        "factorized projection table does not match source "
                        "dimension"
                    )
                projection_dimension = int(
                    module.projection_values.numel()
                    // int(module.source_dimension)
                )
                expected_output_dimension = (
                    int(module.root_nodes.numel())
                    * projection_dimension
                    * int(module.magnetic_dimension)
                )
                if expected_output_dimension != int(
                    module.output_dimension
                ):
                    raise ValueError(
                        "factorized output dimension does not match its roots"
                    )
                projection_matrices.append(
                    module.projection_values.reshape(
                        int(module.source_dimension),
                        projection_dimension,
                    )
                )
                for local_root in range(int(module.root_nodes.numel())):
                    root_projection_starts.append(int(projection_base))
                    root_projection_dimensions.append(
                        int(projection_dimension)
                    )
                    root_output_offsets.append(
                        int(
                            output_base
                            + local_root
                            * projection_dimension
                            * int(module.magnetic_dimension)
                        )
                    )
            output_slices.append(
                {
                    "plan_index": int(plan_index),
                    "start": int(output_base),
                    "stop": int(
                        output_base + int(module.output_dimension)
                    ),
                    "logical_channel_tableau_dimension": int(
                        module.logical_channel_tableau_dimension
                    ),
                    "magnetic_dimension": int(
                        module.magnetic_dimension
                    ),
                    "plan_hash": str(module.plan_hash),
                }
            )
            if shared_nodes is None:
                node_base += int(module.node_offsets.numel())
            input_base += int(module.input_dimension)
            if shared_nodes is None:
                workspace_base += int(module.workspace_dimension)
                coefficient_base += int(
                    module.coefficient_values.numel()
                )
            if heterogeneous_roots:
                projection_base += int(projection_dimension)
            output_base += int(module.output_dimension)

        if shared_nodes is None:
            packed_nodes = {
                "node_offsets": torch.cat(tuple(node_offsets)),
                "node_dimensions": torch.cat(tuple(node_dimensions)),
                "node_leaf_offsets": torch.cat(
                    tuple(node_leaf_offsets)
                ),
                "node_left": torch.cat(tuple(node_left)),
                "node_right": torch.cat(tuple(node_right)),
                "node_coefficient_offsets": torch.cat(
                    tuple(node_coefficient_offsets)
                ),
                "coefficient_rows": torch.cat(tuple(coefficient_rows)),
                "coefficient_columns": torch.cat(
                    tuple(coefficient_columns)
                ),
                "coefficient_values": torch.cat(
                    tuple(coefficient_values)
                ),
            }
        else:
            packed_nodes = shared_nodes
            workspace_base = int(shared_nodes["workspace_dimension"])
        for name in (
            "node_offsets",
            "node_dimensions",
            "node_leaf_offsets",
            "node_left",
            "node_right",
            "node_coefficient_offsets",
            "coefficient_rows",
            "coefficient_columns",
            "coefficient_values",
        ):
            self.register_buffer(
                name,
                packed_nodes[name],
                persistent=False,
            )
        self.register_buffer(
            "root_nodes",
            torch.cat(tuple(root_nodes)),
            persistent=False,
        )
        self.register_buffer(
            "projection_values",
            (
                torch.cat(tuple(projection_matrices), dim=1)
                .reshape(-1)
                .contiguous()
                if heterogeneous_roots
                else first.projection_values.detach().clone()
            ),
            persistent=False,
        )
        if heterogeneous_roots:
            metadata_options = {
                "dtype": torch.int64,
                "device": first.root_nodes.device,
            }
            self.register_buffer(
                "root_projection_starts",
                torch.tensor(
                    root_projection_starts,
                    **metadata_options,
                ),
                persistent=False,
            )
            self.register_buffer(
                "root_projection_dimensions",
                torch.tensor(
                    root_projection_dimensions,
                    **metadata_options,
                ),
                persistent=False,
            )
            self.register_buffer(
                "root_output_offsets",
                torch.tensor(
                    root_output_offsets,
                    **metadata_options,
                ),
                persistent=False,
            )
        self.plan_indices = plan_indices
        self.input_dimensions = tuple(
            int(module.input_dimension) for module in modules
        )
        self.physical_input_dimension = (
            None
            if physical_input_dimension is None
            else int(physical_input_dimension)
        )
        self.source_dimension = int(first.source_dimension)
        self.workspace_dimension = int(workspace_base)
        self.output_dimension = int(output_base)
        self.output_slices = tuple(output_slices)
        self.resolved_backend = str(first.resolved_backend)
        self.heterogeneous_roots = bool(heterogeneous_roots)
        self.shared_subtree_grouping_requested = bool(
            share_physical_subtrees
        )
        self.subtree_evaluation_shared = bool(
            shared_nodes is not None
            and int(shared_nodes["unique_node_count"])
            < int(shared_nodes["naive_node_count"])
        )
        self.naive_node_count = (
            int(sum(module.node_offsets.numel() for module in modules))
            if shared_nodes is None
            else int(shared_nodes["naive_node_count"])
        )
        self.unique_node_count = int(self.node_offsets.numel())
        self.naive_workspace_dimension = (
            int(sum(module.workspace_dimension for module in modules))
            if shared_nodes is None
            else int(shared_nodes["naive_workspace_dimension"])
        )
        self.last_backend = None

    def _native_arguments(self):
        arguments = (
            self.node_offsets,
            self.node_dimensions,
            self.node_leaf_offsets,
            self.node_left,
            self.node_right,
            self.node_coefficient_offsets,
            self.coefficient_rows,
            self.coefficient_columns,
            self.coefficient_values,
            self.root_nodes,
        )
        if self.heterogeneous_roots:
            return arguments + (
                self.root_projection_starts,
                self.root_projection_dimensions,
                self.root_output_offsets,
                self.projection_values,
            )
        return arguments + (self.projection_values,)

    def _evaluate_packed_output(self, packed_group):
        if self.heterogeneous_roots:
            output = (
                torch.ops.ye3t_runtime.factorized_angular_heterogeneous(
                    packed_group,
                    *self._native_arguments(),
                    self.source_dimension,
                    self.workspace_dimension,
                    self.output_dimension,
                )
            )
            self.last_backend = (
                "native_heterogeneous_root_factorized_angular"
            )
        else:
            output = torch.ops.ye3t_runtime.factorized_angular(
                packed_group,
                *self._native_arguments(),
                self.source_dimension,
                self.workspace_dimension,
                self.output_dimension,
            )
            self.last_backend = "native_block_composed_factorized_angular"
        return output

    def _split_output(self, output, leading_shape):
        return tuple(
            output[
                :,
                int(record["start"]):int(record["stop"]),
            ].reshape(
                leading_shape
                + (
                    int(
                        record[
                            "logical_channel_tableau_dimension"
                        ]
                    ),
                    int(record["magnetic_dimension"]),
                )
            )
            for record in self.output_slices
        )

    def forward_packed(self, packed_values, leading_shapes):
        if self.physical_input_dimension is not None:
            raise ValueError(
                "a physical-source plan group requires forward_prepacked"
            )
        packed_values = tuple(packed_values)
        leading_shapes = tuple(tuple(shape) for shape in leading_shapes)
        if len(packed_values) != len(self.input_dimensions):
            raise ValueError(
                "packed factorized group input count does not match its plans"
            )
        if any(shape != leading_shapes[0] for shape in leading_shapes[1:]):
            raise ValueError(
                "grouped factorized plans require matching leading shapes"
            )
        batch_size = int(packed_values[0].shape[0])
        source_rows = []
        for packed, input_dimension in zip(
            packed_values,
            self.input_dimensions,
        ):
            if tuple(packed.shape) != (
                batch_size,
                self.source_dimension * int(input_dimension),
            ):
                raise ValueError(
                    "packed factorized group input width is inconsistent"
                )
            source_rows.append(
                packed.reshape(
                    batch_size,
                    self.source_dimension,
                    int(input_dimension),
                )
            )
        packed_group = torch.cat(
            tuple(source_rows),
            dim=2,
        ).reshape(
            batch_size,
            self.source_dimension * sum(self.input_dimensions),
        ).contiguous()
        return self._evaluate_packed_output(packed_group)

    def forward(self, packed_values, leading_shapes):
        leading_shapes = tuple(tuple(shape) for shape in leading_shapes)
        output = self.forward_packed(packed_values, leading_shapes)
        return self._split_output(output, leading_shapes[0])

    def forward_prepacked_output(self, packed_source_bank, leading_shape):
        if self.physical_input_dimension is None:
            raise ValueError(
                "this factorized plan group has no physical source bank"
            )
        leading_shape = tuple(leading_shape)
        batch_size = math.prod(leading_shape)
        expected_shape = (
            int(batch_size),
            int(
                self.source_dimension
                * self.physical_input_dimension
            ),
        )
        if tuple(packed_source_bank.shape) != expected_shape:
            raise ValueError(
                "prepacked physical source bank shape is inconsistent"
            )
        return self._evaluate_packed_output(packed_source_bank.contiguous())

    def forward_prepacked(self, packed_source_bank, leading_shape):
        leading_shape = tuple(leading_shape)
        output = self.forward_prepacked_output(
            packed_source_bank,
            leading_shape,
        )
        return self._split_output(output, leading_shape)

    def runtime_report(self):
        return {
            "runtime": "_PackedFactorizedAngularPlanGroup",
            "member_plan_indices": tuple(self.plan_indices),
            "member_plan_hashes": tuple(
                str(record["plan_hash"])
                for record in self.output_slices
            ),
            "member_count": int(len(self.plan_indices)),
            "source_dimension": int(self.source_dimension),
            "workspace_dimension": int(self.workspace_dimension),
            "output_dimension": int(self.output_dimension),
            "physical_input_dimension": (
                None
                if self.physical_input_dimension is None
                else int(self.physical_input_dimension)
            ),
            "prepacked_physical_source_bank": bool(
                self.physical_input_dimension is not None
            ),
            "single_native_call": True,
            "flat_output_available": True,
            "heterogeneous_root_dimensions": bool(
                self.heterogeneous_roots
            ),
            "subtree_evaluation_shared": bool(
                self.subtree_evaluation_shared
            ),
            "shared_subtree_grouping_requested": bool(
                self.shared_subtree_grouping_requested
            ),
            "naive_node_count": int(self.naive_node_count),
            "unique_node_count": int(self.unique_node_count),
            "elided_node_evaluations": int(
                self.naive_node_count - self.unique_node_count
            ),
            "naive_workspace_dimension": int(
                self.naive_workspace_dimension
            ),
            "workspace_dimension": int(self.workspace_dimension),
            "composition": (
                "exact_shared_subtree_heterogeneous_root_dag"
                if self.subtree_evaluation_shared
                else "exact_heterogeneous_root_node_forest"
                if self.heterogeneous_roots
                else "exact_block_diagonal_node_forest"
            ),
            "last_backend": self.last_backend,
        }


class _RepeatedBlockPowerExactAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packed_source_bank, group):
        ctx.group = group
        ctx.save_for_backward(packed_source_bank)
        return group._power_output_from_bank(packed_source_bank)

    @staticmethod
    def backward(ctx, output_adjoint):
        (packed_source_bank,) = ctx.saved_tensors
        source_adjoint = _RepeatedBlockPowerAdjoint.apply(
            output_adjoint,
            packed_source_bank,
            ctx.group,
        )
        return source_adjoint, None


class _RepeatedBlockPowerAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output_adjoint, packed_source_bank, group):
        ctx.group = group
        ctx.save_for_backward(output_adjoint, packed_source_bank)
        return group._power_input_adjoint(
            output_adjoint,
            packed_source_bank,
        )

    @staticmethod
    def backward(ctx, source_adjoint_tangent):
        output_adjoint, packed_source_bank = ctx.saved_tensors
        needs_output = bool(ctx.needs_input_grad[0])
        needs_source = bool(ctx.needs_input_grad[1])
        if not needs_output and not needs_source:
            return None, None, None
        output_gradient, source_gradient = ctx.group._power_double_backward(
            source_adjoint_tangent,
            output_adjoint,
            packed_source_bank,
            needs_output=needs_output,
            needs_source=needs_source,
        )
        return output_gradient, source_gradient, None


class _PackedRepeatedBlockPowerPlanGroup(torch.nn.Module):
    """Evaluate exact repeated-block parent plans through shared monomials."""

    def __init__(
        self,
        modules,
        plan_indices,
        power_physical_leaf_offsets_by_plan,
        exact_physical_leaf_offsets_by_plan,
        physical_input_dimension,
        defer_exact_adjoint_group=False,
    ):
        super().__init__()
        modules = tuple(modules)
        plan_indices = tuple(int(index) for index in plan_indices)
        if not modules or len(modules) != len(plan_indices):
            raise ValueError(
                "repeated-block power groups require matching modules and indices"
            )
        first = modules[0]
        power_physical_leaf_offsets_by_plan = tuple(
            tuple(
                tuple(int(offset) for offset in source_offsets)
                if isinstance(source_offsets, (tuple, list))
                else int(source_offsets)
                for source_offsets in offsets
            )
            for offsets in power_physical_leaf_offsets_by_plan
        )
        exact_physical_leaf_offsets_by_plan = tuple(
            tuple(int(offset) for offset in offsets)
            for offsets in exact_physical_leaf_offsets_by_plan
        )
        if (
            len(power_physical_leaf_offsets_by_plan) != len(modules)
            or len(exact_physical_leaf_offsets_by_plan) != len(modules)
        ):
            raise ValueError(
                "repeated-block physical offsets must cover every member"
            )
        buffers_by_module = tuple(
            _repeated_block_power_polynomial_buffers(
                module,
                offsets,
                physical_input_dimension,
            )
            for module, offsets in zip(
                modules,
                power_physical_leaf_offsets_by_plan,
            )
        )
        if any(
            int(buffers["source_dimension"])
            != int(buffers_by_module[0]["source_dimension"])
            or module.resolved_backend != first.resolved_backend
            or module.coefficient_values.dtype
            != first.coefficient_values.dtype
            or module.coefficient_values.device
            != first.coefficient_values.device
            for module, buffers in zip(modules[1:], buffers_by_module[1:])
        ):
            raise ValueError(
                "repeated-block power group members are not exactly compatible"
            )
        physical_input_dimension = int(physical_input_dimension)
        physical_indices = tuple(
            sorted(
                {
                    int(value)
                    for buffers in buffers_by_module
                    for value in buffers["physical_input_indices"]
                }
            )
        )
        physical_index_to_union = {
            int(value): int(index)
            for index, value in enumerate(physical_indices)
        }
        input_dimension = int(len(physical_indices))
        if input_dimension <= 0:
            raise ValueError(
                "repeated-block power group has no physical input coordinates"
            )

        monomial_indices = {}
        monomial_counts = []
        coefficient_terms = []
        coefficient_outputs = []
        coefficient_values = []
        output_offsets = [0]
        output_slices = []
        output_base = 0
        for module, plan_index, buffers in zip(
            modules,
            plan_indices,
            buffers_by_module,
        ):
            local_physical_indices = tuple(
                int(value) for value in buffers["physical_input_indices"]
            )
            local_to_union = tuple(
                int(physical_index_to_union[value])
                for value in local_physical_indices
            )
            local_to_global = []
            for counts in buffers["monomial_counts"].detach().cpu().tolist():
                union_counts = [0] * input_dimension
                for local_index, value in enumerate(counts):
                    union_counts[local_to_union[int(local_index)]] += int(
                        value
                    )
                counts = tuple(union_counts)
                global_index = monomial_indices.get(counts)
                if global_index is None:
                    global_index = len(monomial_counts)
                    monomial_indices[counts] = global_index
                    monomial_counts.append(counts)
                local_to_global.append(int(global_index))
            local_terms = buffers["coefficient_terms"].detach().cpu().tolist()
            local_outputs = (
                buffers["coefficient_outputs"].detach().cpu().tolist()
            )
            local_values = (
                buffers["coefficient_values"].detach().cpu().tolist()
            )
            by_output = [[] for _output in range(int(module.output_dimension))]
            for term, output, value in zip(
                local_terms,
                local_outputs,
                local_values,
            ):
                by_output[int(output)].append(
                    (local_to_global[int(term)], complex(value))
                )
            for local_output, records in enumerate(by_output):
                for term, value in records:
                    coefficient_terms.append(int(term))
                    coefficient_outputs.append(
                        int(output_base + local_output)
                    )
                    coefficient_values.append(value)
                output_offsets.append(len(coefficient_terms))
            output_slices.append(
                {
                    "plan_index": int(plan_index),
                    "start": int(output_base),
                    "stop": int(output_base + module.output_dimension),
                    "logical_channel_tableau_dimension": int(
                        module.logical_channel_tableau_dimension
                    ),
                    "magnetic_dimension": int(module.magnetic_dimension),
                    "plan_hash": str(module.plan_hash),
                    "power": int(buffers["power"]),
                    "block_multiplicities": tuple(
                        int(value)
                        for value in buffers["block_multiplicities"]
                    ),
                }
            )
            output_base += int(module.output_dimension)

        device = first.coefficient_values.device
        self.register_buffer(
            "monomial_counts",
            torch.tensor(
                monomial_counts,
                dtype=torch.int64,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "output_offsets",
            torch.tensor(
                output_offsets,
                dtype=torch.int64,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "coefficient_terms",
            torch.tensor(
                coefficient_terms,
                dtype=torch.int64,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "coefficient_outputs",
            torch.tensor(
                coefficient_outputs,
                dtype=torch.int64,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "coefficient_values",
            torch.tensor(
                coefficient_values,
                dtype=first.coefficient_values.dtype,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "physical_input_indices",
            torch.tensor(
                physical_indices,
                dtype=torch.int64,
                device=device,
            ),
            persistent=False,
        )
        object.__setattr__(self, "_source_modules", modules)
        self.exact_physical_leaf_offsets_by_plan = (
            exact_physical_leaf_offsets_by_plan
        )
        self.exact_adjoint_group = None
        object.__setattr__(self, "_shared_exact_derivative_owner", None)
        self._shared_exact_output_slice = None
        self.exact_physical_input_dimension = int(physical_input_dimension)
        self.plan_indices = plan_indices
        self.output_slices = tuple(output_slices)
        self.input_dimension = int(input_dimension)
        self.physical_input_dimension = int(physical_input_dimension)
        self.source_dimension = int(
            buffers_by_module[0]["source_dimension"]
        )
        self.physical_input_contiguous = bool(
            physical_indices
            and physical_indices
            == tuple(
                range(
                    int(physical_indices[0]),
                    int(physical_indices[0] + len(physical_indices)),
                )
            )
        )
        self.physical_input_start = (
            int(physical_indices[0]) if physical_indices else 0
        )
        self.member_block_layouts = tuple(
            {
                "plan_index": int(plan_index),
                "source_block_physical_offsets": tuple(
                    tuple(int(value) for value in offsets)
                    for offsets in buffers["source_block_physical_offsets"]
                ),
                "source_block_angular_Ls": tuple(
                    tuple(int(value) for value in angular_Ls)
                    for angular_Ls in buffers["source_block_angular_Ls"]
                ),
                "source_block_multiplicities": tuple(
                    tuple(int(value) for value in multiplicities)
                    for multiplicities in buffers[
                        "source_block_multiplicities"
                    ]
                ),
                "physical_input_coordinate_count": int(
                    len(buffers["physical_input_indices"])
                ),
            }
            for plan_index, buffers in zip(plan_indices, buffers_by_module)
        )
        self.block_physical_offsets = (
            tuple(
                int(value)
                for value in buffers_by_module[0]["block_physical_offsets"]
            )
            if all(
                tuple(buffers["block_physical_offsets"])
                == tuple(buffers_by_module[0]["block_physical_offsets"])
                for buffers in buffers_by_module[1:]
            )
            else ()
        )
        self.block_angular_Ls = (
            tuple(
                int(value)
                for value in buffers_by_module[0]["block_angular_Ls"]
            )
            if all(
                tuple(buffers["block_angular_Ls"])
                == tuple(buffers_by_module[0]["block_angular_Ls"])
                for buffers in buffers_by_module[1:]
            )
            else ()
        )
        self.block_multiplicities = (
            tuple(
                int(value)
                for value in buffers_by_module[0]["block_multiplicities"]
            )
            if all(
                tuple(buffers["block_multiplicities"])
                == tuple(buffers_by_module[0]["block_multiplicities"])
                for buffers in buffers_by_module[1:]
            )
            else ()
        )
        self.source_block_physical_offsets = (
            tuple(
                tuple(int(value) for value in offsets)
                for offsets in buffers_by_module[0][
                    "source_block_physical_offsets"
                ]
            )
            if all(
                tuple(buffers["source_block_physical_offsets"])
                == tuple(
                    buffers_by_module[0]["source_block_physical_offsets"]
                )
                for buffers in buffers_by_module[1:]
            )
            else ()
        )
        self.source_block_angular_Ls = (
            tuple(
                tuple(int(value) for value in angular_Ls)
                for angular_Ls in buffers_by_module[0][
                    "source_block_angular_Ls"
                ]
            )
            if all(
                tuple(buffers["source_block_angular_Ls"])
                == tuple(buffers_by_module[0]["source_block_angular_Ls"])
                for buffers in buffers_by_module[1:]
            )
            else ()
        )
        self.source_block_multiplicities = (
            tuple(
                tuple(int(value) for value in multiplicities)
                for multiplicities in buffers_by_module[0][
                    "source_block_multiplicities"
                ]
            )
            if all(
                tuple(buffers["source_block_multiplicities"])
                == tuple(
                    buffers_by_module[0]["source_block_multiplicities"]
                )
                for buffers in buffers_by_module[1:]
            )
            else ()
        )
        self.monomial_upper_bound = int(
            sum(
                buffers["monomial_upper_bound"]
                for buffers in buffers_by_module
            )
        )
        self.estimated_count_table_bytes = int(
            self.monomial_counts.shape[0]
            * self.monomial_counts.shape[1]
            * self.monomial_counts.element_size()
        )
        self.power_table_bytes = int(
            self.monomial_counts.numel()
            * self.monomial_counts.element_size()
            + self.output_offsets.numel()
            * self.output_offsets.element_size()
            + self.coefficient_terms.numel()
            * self.coefficient_terms.element_size()
            + self.coefficient_outputs.numel()
            * self.coefficient_outputs.element_size()
            + self.coefficient_values.numel()
            * self.coefficient_values.element_size()
            + self.physical_input_indices.numel()
            * self.physical_input_indices.element_size()
        )
        self.double_backward_table_bytes = 0
        self.output_dimension = int(output_base)
        self.resolved_backend = str(first.resolved_backend)
        self.last_backend = None
        self.last_derivative_backend = None
        if not bool(defer_exact_adjoint_group):
            self._install_exact_adjoint_group()

    def _install_exact_adjoint_group(self):
        if self.exact_adjoint_group is not None:
            return
        exact_adjoint_group = _PackedFactorizedAngularPlanGroup(
            self._source_modules,
            self.plan_indices,
            physical_leaf_offsets=(
                self.exact_physical_leaf_offsets_by_plan
            ),
            physical_input_dimension=int(self.physical_input_dimension),
            heterogeneous_roots=True,
            share_physical_subtrees=True,
        )
        minimum_cuda_input_dimension = int(
            math.ceil(
                int(exact_adjoint_group.workspace_dimension)
                / max(1, int(exact_adjoint_group.node_offsets.numel()))
            )
        )
        self.exact_physical_input_dimension = int(
            max(
                self.physical_input_dimension,
                minimum_cuda_input_dimension,
            )
        )
        if (
            self.exact_physical_input_dimension
            != self.physical_input_dimension
        ):
            exact_adjoint_group = _PackedFactorizedAngularPlanGroup(
                self._source_modules,
                self.plan_indices,
                physical_leaf_offsets=(
                    self.exact_physical_leaf_offsets_by_plan
                ),
                physical_input_dimension=(
                    self.exact_physical_input_dimension
                ),
                heterogeneous_roots=True,
                share_physical_subtrees=True,
            )
        self.exact_adjoint_group = exact_adjoint_group

    def _share_exact_derivative_group(self, owner, start, stop):
        if self.exact_adjoint_group is not None:
            raise RuntimeError(
                "an individual exact derivative group cannot also use a "
                "shared destination group"
            )
        object.__setattr__(self, "_shared_exact_derivative_owner", owner)
        self._shared_exact_output_slice = (int(start), int(stop))

    def _power_output(self, input):
        output = symmetric_power_shared_monomial_contraction(
            input,
            self.monomial_counts,
            self.output_offsets,
            self.coefficient_terms,
            self.coefficient_outputs,
            self.coefficient_values,
            backend=(
                "native"
                if self.resolved_backend in {"native_cpu", "native_cuda"}
                else "reference"
            ),
        )
        self.last_backend = "native_shared_repeated_block_power_monomial"
        return output

    def _select_power_input(self, packed_source_bank):
        if self.physical_input_contiguous:
            return packed_source_bank[
                :,
                self.physical_input_start : (
                    self.physical_input_start + self.input_dimension
                ),
            ]
        return packed_source_bank.index_select(
            1,
            self.physical_input_indices,
        )

    def _power_output_from_bank(self, packed_source_bank):
        return self._power_output(
            self._select_power_input(packed_source_bank)
        )

    def _power_input_adjoint(
        self,
        output_adjoint,
        packed_source_bank,
    ):
        input = self._select_power_input(packed_source_bank)
        input_adjoint = (
            torch.ops.ye3t_runtime
            .symmetric_power_shared_monomial_adjoint(
                output_adjoint.contiguous(),
                input.contiguous(),
                self.monomial_counts,
                self.output_offsets,
                self.coefficient_terms,
                self.coefficient_outputs,
                self.coefficient_values,
            )
        )
        source_adjoint = torch.zeros_like(packed_source_bank)
        source_adjoint.index_copy_(
            1,
            self.physical_input_indices,
            input_adjoint,
        )
        self.last_derivative_backend = (
            "native_repeated_block_power_adjoint"
        )
        return source_adjoint

    def _power_double_backward(
        self,
        source_adjoint_tangent,
        output_adjoint,
        packed_source_bank,
        *,
        needs_output,
        needs_source,
    ):
        _policy, native_double_backward = (
            _resolve_symmetric_power_double_backward_policy(
                packed_source_bank,
                "symmetric_power_shared_monomial_double_backward",
            )
        )
        if native_double_backward:
            selected_input = self._select_power_input(
                packed_source_bank
            )
            selected_tangent = self._select_power_input(
                source_adjoint_tangent
            )
            (output_tangent, input_tangent), algorithm = (
                _symmetric_power_shared_double_backward(
                    selected_tangent,
                    output_adjoint,
                    selected_input,
                    self.monomial_counts,
                    self.output_offsets,
                    self.coefficient_terms,
                    self.coefficient_outputs,
                    self.coefficient_values,
                )
            )
            source_tangent = None
            if needs_source:
                source_tangent = torch.zeros_like(packed_source_bank)
                source_tangent.index_copy_(
                    1,
                    self.physical_input_indices,
                    input_tangent,
                )
            self.last_derivative_backend = (
                "native_destination_shared_power_"
                f"{algorithm}_double_backward"
            )
            return (
                output_tangent if needs_output else None,
                source_tangent,
            )

        with torch.enable_grad():
            tracked_output = output_adjoint.detach().requires_grad_(True)
            tracked_source = (
                packed_source_bank.detach().requires_grad_(True)
            )
            exact_output = self._exact_output(tracked_source)
            exact_source_adjoint = torch.autograd.grad(
                exact_output,
                tracked_source,
                tracked_output,
                create_graph=True,
            )[0]
            targets = []
            if needs_output:
                targets.append(tracked_output)
            if needs_source:
                targets.append(tracked_source)
            gradients = torch.autograd.grad(
                exact_source_adjoint,
                tuple(targets),
                source_adjoint_tangent,
                create_graph=torch.is_grad_enabled(),
                allow_unused=True,
            )
        gradient_index = 0
        output_gradient = None
        source_gradient = None
        if needs_output:
            output_gradient = gradients[gradient_index]
            gradient_index += 1
        if needs_source:
            source_gradient = gradients[gradient_index]
        self.last_derivative_backend = (
            "exact_factorized_double_backward_fallback"
        )
        return output_gradient, source_gradient

    def _exact_output(self, packed_source_bank):
        if self.exact_adjoint_group is None:
            owner = self._shared_exact_derivative_owner
            if owner is None or self._shared_exact_output_slice is None:
                raise RuntimeError(
                    "repeated-block power group has no exact derivative "
                    "executor"
                )
            start, stop = self._shared_exact_output_slice
            return owner._exact_output(packed_source_bank)[:, start:stop]
        exact_source_bank = packed_source_bank
        if (
            self.exact_physical_input_dimension
            != self.physical_input_dimension
        ):
            exact_source_bank = torch.nn.functional.pad(
                packed_source_bank.reshape(
                    int(packed_source_bank.shape[0]),
                    int(self.source_dimension),
                    int(self.physical_input_dimension),
                ),
                (
                    0,
                    int(
                        self.exact_physical_input_dimension
                        - self.physical_input_dimension
                    ),
                ),
            ).reshape(
                int(packed_source_bank.shape[0]),
                int(
                    self.source_dimension
                    * self.exact_physical_input_dimension
                ),
            )
        output = self.exact_adjoint_group.forward_prepacked_output(
            exact_source_bank,
            (int(packed_source_bank.shape[0]),),
        )
        self.last_derivative_backend = (
            "native_exact_factorized_adjoint_double_adjoint"
        )
        return output

    def forward_prepacked_output(self, packed_source_bank, leading_shape):
        leading_shape = tuple(leading_shape)
        batch_size = math.prod(leading_shape)
        if tuple(packed_source_bank.shape) != (
            int(batch_size),
            int(self.source_dimension * self.physical_input_dimension),
        ):
            raise ValueError(
                "repeated-block power physical source bank shape is inconsistent"
            )
        if torch.is_grad_enabled() and packed_source_bank.requires_grad:
            return _RepeatedBlockPowerExactAdjoint.apply(
                packed_source_bank,
                self,
            )
        return self._power_output_from_bank(packed_source_bank)

    def forward_prepacked(self, packed_source_bank, leading_shape):
        leading_shape = tuple(leading_shape)
        output = self.forward_prepacked_output(
            packed_source_bank,
            leading_shape,
        )
        return tuple(
            output[:, record["start"] : record["stop"]].reshape(
                leading_shape
                + (
                    int(record["logical_channel_tableau_dimension"]),
                    int(record["magnetic_dimension"]),
                )
            )
            for record in self.output_slices
        )

    def runtime_report(self):
        return {
            "runtime": "_PackedRepeatedBlockPowerPlanGroup",
            "member_plan_indices": tuple(self.plan_indices),
            "member_plan_hashes": tuple(
                record["plan_hash"] for record in self.output_slices
            ),
            "member_count": int(len(self.plan_indices)),
            "powers": tuple(
                int(record["power"]) for record in self.output_slices
            ),
            "source_dimension": int(self.source_dimension),
            "physical_input_dimension": int(self.physical_input_dimension),
            "exact_derivative_input_dimension": int(
                self.exact_physical_input_dimension
            ),
            "exact_derivative_padding_required": bool(
                self.exact_physical_input_dimension
                != self.physical_input_dimension
            ),
            "physical_input_coordinate_count": int(
                self.physical_input_indices.numel()
            ),
            "physical_input_gather_required": bool(
                not self.physical_input_contiguous
            ),
            "input_dimension": int(self.input_dimension),
            "block_physical_offsets": tuple(self.block_physical_offsets),
            "block_angular_Ls": tuple(self.block_angular_Ls),
            "block_multiplicities": tuple(self.block_multiplicities),
            "source_block_physical_offsets": tuple(
                self.source_block_physical_offsets
            ),
            "source_block_angular_Ls": tuple(
                self.source_block_angular_Ls
            ),
            "source_block_multiplicities": tuple(
                self.source_block_multiplicities
            ),
            "member_block_layouts": tuple(
                dict(record) for record in self.member_block_layouts
            ),
            "monomial_count": int(self.monomial_counts.shape[0]),
            "monomial_upper_bound": int(self.monomial_upper_bound),
            "estimated_count_table_bytes": int(
                self.estimated_count_table_bytes
            ),
            "power_table_bytes": int(self.power_table_bytes),
            "double_backward_table_bytes": int(
                self.double_backward_table_bytes
            ),
            "double_backward_term_count": int(
                self.monomial_counts.shape[0]
            ),
            "double_backward_coefficient_count": int(
                self.coefficient_values.numel()
            ),
            "double_backward_policy": os.environ.get(
                "YE3T_SYMMETRIC_POWER_MONOMIAL_DOUBLE_BACKWARD_POLICY",
                "auto",
            ),
            "coefficient_count": int(self.coefficient_values.numel()),
            "output_dimension": int(self.output_dimension),
            "flat_output_available": True,
            "exact_derivative_group_count": 1,
            "exact_derivative_group": (
                self.exact_adjoint_group.runtime_report()
                if self.exact_adjoint_group is not None
                else {
                    "runtime": (
                        "shared_destination_factorized_derivative_group"
                    ),
                    "output_slice": tuple(
                        self._shared_exact_output_slice or ()
                    ),
                }
            ),
            "composition": (
                "exact_sparse_dag_to_shared_repeated_block_power_monomials"
            ),
            "derivative_backend": (
                "native_power_adjoint_destination_shared_double_adjoint"
                "_with_exact_factorized_fallback"
            ),
            "last_backend": self.last_backend,
            "last_derivative_backend": self.last_derivative_backend,
        }


class _PackedRepeatedBlockPowerDerivativeDestinationGroup(torch.nn.Module):
    """Share one exact Hessian traversal across compatible power groups."""

    def __init__(self, power_groups):
        super().__init__()
        power_groups = tuple(power_groups)
        if len(power_groups) < 2:
            raise ValueError(
                "a derivative destination group requires at least two "
                "power groups"
            )
        first = power_groups[0]
        if any(
            group.resolved_backend != first.resolved_backend
            or int(group.source_dimension) != int(first.source_dimension)
            or int(group.physical_input_dimension)
            != int(first.physical_input_dimension)
            or group.coefficient_values.dtype
            != first.coefficient_values.dtype
            or group.coefficient_values.device
            != first.coefficient_values.device
            for group in power_groups[1:]
        ):
            raise ValueError(
                "power derivative destination groups require compatible "
                "backend, dtype, device, source, and physical layouts"
            )
        object.__setattr__(self, "_power_groups", power_groups)
        modules = tuple(
            module
            for group in power_groups
            for module in group._source_modules
        )
        plan_indices = tuple(
            int(index)
            for group in power_groups
            for index in group.plan_indices
        )
        exact_offsets = tuple(
            offsets
            for group in power_groups
            for offsets in group.exact_physical_leaf_offsets_by_plan
        )
        physical_input_dimension = int(first.physical_input_dimension)
        exact_group = _PackedFactorizedAngularPlanGroup(
            modules,
            plan_indices,
            physical_leaf_offsets=exact_offsets,
            physical_input_dimension=physical_input_dimension,
            heterogeneous_roots=True,
            share_physical_subtrees=True,
        )
        minimum_cuda_input_dimension = int(
            math.ceil(
                int(exact_group.workspace_dimension)
                / max(1, int(exact_group.node_offsets.numel()))
            )
        )
        self.exact_physical_input_dimension = int(
            max(physical_input_dimension, minimum_cuda_input_dimension)
        )
        if self.exact_physical_input_dimension != physical_input_dimension:
            exact_group = _PackedFactorizedAngularPlanGroup(
                modules,
                plan_indices,
                physical_leaf_offsets=exact_offsets,
                physical_input_dimension=(
                    self.exact_physical_input_dimension
                ),
                heterogeneous_roots=True,
                share_physical_subtrees=True,
            )
        self.exact_adjoint_group = exact_group
        self.plan_indices = plan_indices
        self.source_dimension = int(first.source_dimension)
        self.physical_input_dimension = physical_input_dimension
        self.resolved_backend = str(first.resolved_backend)
        output_slices = []
        group_slices = []
        output_base = 0
        for group in power_groups:
            group_start = int(output_base)
            for record in group.output_slices:
                width = int(record["stop"] - record["start"])
                output_slices.append(
                    {
                        **record,
                        "start": int(output_base),
                        "stop": int(output_base + width),
                    }
                )
                output_base += width
            group_slices.append((group_start, int(output_base)))
            group._share_exact_derivative_group(
                self,
                group_start,
                output_base,
            )
        self.output_slices = tuple(output_slices)
        self.group_slices = tuple(group_slices)
        self.output_dimension = int(output_base)
        self.last_backend = None
        self.last_derivative_backend = None

    def _power_output_from_bank(self, packed_source_bank):
        outputs = tuple(
            group._power_output_from_bank(packed_source_bank)
            for group in self._power_groups
        )
        self.last_backend = (
            "native_destination_grouped_repeated_block_power_monomial"
        )
        return torch.cat(outputs, dim=1)

    def _power_input_adjoint(self, output_adjoint, packed_source_bank):
        source_adjoint = None
        for group, (start, stop) in zip(
            self._power_groups,
            self.group_slices,
        ):
            contribution = group._power_input_adjoint(
                output_adjoint[:, int(start):int(stop)],
                packed_source_bank,
            )
            source_adjoint = (
                contribution
                if source_adjoint is None
                else source_adjoint + contribution
            )
        self.last_derivative_backend = (
            "native_power_adjoint_destination_grouped_exact_double_adjoint"
        )
        return source_adjoint

    def _power_double_backward(
        self,
        source_adjoint_tangent,
        output_adjoint,
        packed_source_bank,
        *,
        needs_output,
        needs_source,
    ):
        _policy, native_double_backward = (
            _resolve_symmetric_power_double_backward_policy(
                packed_source_bank,
                "symmetric_power_shared_monomial_double_backward",
            )
        )
        if native_double_backward:
            output_gradients = []
            source_gradient = None
            for group, (start, stop) in zip(
                self._power_groups,
                self.group_slices,
            ):
                output_contribution, source_contribution = (
                    group._power_double_backward(
                        source_adjoint_tangent,
                        output_adjoint[:, int(start):int(stop)],
                        packed_source_bank,
                        needs_output=needs_output,
                        needs_source=needs_source,
                    )
                )
                if needs_output:
                    output_gradients.append(output_contribution)
                if needs_source:
                    source_gradient = (
                        source_contribution
                        if source_gradient is None
                        else source_gradient + source_contribution
                    )
            self.last_derivative_backend = (
                "native_destination_grouped_shared_power_"
                "double_backward"
            )
            return (
                torch.cat(tuple(output_gradients), dim=1)
                if needs_output
                else None,
                source_gradient,
            )

        with torch.enable_grad():
            tracked_output = output_adjoint.detach().requires_grad_(True)
            tracked_source = (
                packed_source_bank.detach().requires_grad_(True)
            )
            exact_output = self._exact_output(tracked_source)
            exact_source_adjoint = torch.autograd.grad(
                exact_output,
                tracked_source,
                tracked_output,
                create_graph=True,
            )[0]
            targets = []
            if needs_output:
                targets.append(tracked_output)
            if needs_source:
                targets.append(tracked_source)
            gradients = torch.autograd.grad(
                exact_source_adjoint,
                tuple(targets),
                source_adjoint_tangent,
                create_graph=torch.is_grad_enabled(),
                allow_unused=True,
            )
        gradient_index = 0
        output_gradient = None
        source_gradient = None
        if needs_output:
            output_gradient = gradients[gradient_index]
            gradient_index += 1
        if needs_source:
            source_gradient = gradients[gradient_index]
        self.last_derivative_backend = (
            "exact_destination_grouped_factorized_double_backward_"
            "fallback"
        )
        return output_gradient, source_gradient

    def _exact_output(self, packed_source_bank):
        exact_source_bank = packed_source_bank
        if (
            self.exact_physical_input_dimension
            != self.physical_input_dimension
        ):
            exact_source_bank = torch.nn.functional.pad(
                packed_source_bank.reshape(
                    int(packed_source_bank.shape[0]),
                    int(self.source_dimension),
                    int(self.physical_input_dimension),
                ),
                (
                    0,
                    int(
                        self.exact_physical_input_dimension
                        - self.physical_input_dimension
                    ),
                ),
            ).reshape(
                int(packed_source_bank.shape[0]),
                int(
                    self.source_dimension
                    * self.exact_physical_input_dimension
                ),
            )
        output = self.exact_adjoint_group.forward_prepacked_output(
            exact_source_bank,
            (int(packed_source_bank.shape[0]),),
        )
        self.last_derivative_backend = (
            "native_destination_grouped_exact_factorized_double_adjoint"
        )
        return output

    def forward_prepacked_output(self, packed_source_bank, leading_shape):
        leading_shape = tuple(leading_shape)
        batch_size = math.prod(leading_shape)
        if tuple(packed_source_bank.shape) != (
            int(batch_size),
            int(self.source_dimension * self.physical_input_dimension),
        ):
            raise ValueError(
                "destination-grouped repeated-block source bank shape is "
                "inconsistent"
            )
        if torch.is_grad_enabled() and packed_source_bank.requires_grad:
            return _RepeatedBlockPowerExactAdjoint.apply(
                packed_source_bank,
                self,
            )
        return self._power_output_from_bank(packed_source_bank)

    def runtime_report(self):
        return {
            "runtime": (
                "_PackedRepeatedBlockPowerDerivativeDestinationGroup"
            ),
            "member_power_group_count": int(len(self._power_groups)),
            "member_plan_count": int(len(self.plan_indices)),
            "member_plan_indices": tuple(self.plan_indices),
            "source_dimension": int(self.source_dimension),
            "physical_input_dimension": int(
                self.physical_input_dimension
            ),
            "exact_derivative_input_dimension": int(
                self.exact_physical_input_dimension
            ),
            "output_dimension": int(self.output_dimension),
            "prior_exact_derivative_call_count": int(
                len(self._power_groups)
            ),
            "exact_derivative_call_count": 1,
            "native_double_backward_call_count": int(
                len(self._power_groups)
            ),
            "double_backward_policy": os.environ.get(
                "YE3T_SYMMETRIC_POWER_MONOMIAL_DOUBLE_BACKWARD_POLICY",
                "auto",
            ),
            "destination_order": "compiler_plan_order",
            "runtime_path_discovery": False,
            "exact_derivative_group": (
                self.exact_adjoint_group.runtime_report()
            ),
            "last_backend": self.last_backend,
            "last_derivative_backend": self.last_derivative_backend,
        }


class _YE3TFactorizedAngularModuleGroup(torch.nn.Module):
    """Internal compiler-runtime grouping for several exact parent plans."""

    def __init__(
        self,
        modules,
        enable_native_grouping=False,
        physical_leaf_offsets_by_plan=None,
        power_physical_leaf_offsets_by_plan=None,
        physical_input_dimension=None,
        enable_heterogeneous_root_grouping=False,
        enable_homogeneous_power_grouping=False,
        power_table_byte_limit=64 * 1024 * 1024,
        repeated_block_power_policy="auto",
        enable_destination_derivative_grouping=False,
    ):
        super().__init__()
        modules = tuple(modules)
        if not modules:
            raise ValueError(
                "factorized angular module group requires at least one plan"
            )
        object.__setattr__(self, "_source_modules", modules)
        self.native_capable = all(
            module.resolved_backend in {"native_cpu", "native_cuda"}
            for module in modules
        )
        self.native_grouping_requested = bool(enable_native_grouping)
        self.native_grouped = bool(
            self.native_capable and self.native_grouping_requested
        )
        self.heterogeneous_root_grouping_requested = bool(
            enable_heterogeneous_root_grouping
        )
        self.homogeneous_power_grouping_requested = bool(
            enable_homogeneous_power_grouping
        )
        self.power_table_byte_limit = int(power_table_byte_limit)
        if self.power_table_byte_limit <= 0:
            raise ValueError("power_table_byte_limit must be positive")
        self.repeated_block_power_policy = str(
            repeated_block_power_policy
        )
        if self.repeated_block_power_policy not in {"auto", "force"}:
            raise ValueError(
                "repeated_block_power_policy must be auto or force"
            )
        self.destination_derivative_grouping_requested = bool(
            enable_destination_derivative_grouping
        )
        if (physical_leaf_offsets_by_plan is None) != (
            physical_input_dimension is None
        ):
            raise ValueError(
                "physical leaf offsets and input dimension must be supplied "
                "together"
            )
        if physical_leaf_offsets_by_plan is not None:
            physical_leaf_offsets_by_plan = tuple(
                tuple(int(offset) for offset in offsets)
                for offsets in physical_leaf_offsets_by_plan
            )
            if len(physical_leaf_offsets_by_plan) != len(modules):
                raise ValueError(
                    "physical leaf offsets must cover every plan"
                )
        if power_physical_leaf_offsets_by_plan is None:
            power_physical_leaf_offsets_by_plan = (
                physical_leaf_offsets_by_plan
            )
        else:
            power_physical_leaf_offsets_by_plan = tuple(
                tuple(
                    tuple(int(offset) for offset in source_offsets)
                    if isinstance(source_offsets, (tuple, list))
                    else int(source_offsets)
                    for source_offsets in offsets
                )
                for offsets in power_physical_leaf_offsets_by_plan
            )
            if len(power_physical_leaf_offsets_by_plan) != len(modules):
                raise ValueError(
                    "power physical leaf offsets must cover every plan"
                )
        if (
            power_physical_leaf_offsets_by_plan is not None
            and physical_input_dimension is None
        ):
            raise ValueError(
                "power physical leaf offsets require an input dimension"
            )
        self.physical_input_dimension = (
            None
            if physical_input_dimension is None
            else int(physical_input_dimension)
        )
        self.heterogeneous_root_grouped = bool(
            self.native_grouped
            and self.heterogeneous_root_grouping_requested
            and self.physical_input_dimension is not None
            and hasattr(
                torch.ops.ye3t_runtime,
                "factorized_angular_heterogeneous",
            )
        )
        groups = []
        power_groups = []
        power_plan_indices = set()
        power_skip_records = []
        if self.native_grouped:
            if (
                self.homogeneous_power_grouping_requested
                and power_physical_leaf_offsets_by_plan is not None
            ):
                records = {}
                for plan_index, (module, offsets) in enumerate(
                    zip(modules, power_physical_leaf_offsets_by_plan)
                ):
                    if len(module.input_Ls) < 4:
                        continue
                    layout = _repeated_block_power_layout(
                        module,
                        offsets,
                        self.physical_input_dimension,
                    )
                    if int(layout["repeated_slot_count"]) < 2:
                        continue
                    if (
                        self.repeated_block_power_policy == "auto"
                        and any(
                            len(tuple(angular_Ls)) > 1
                            for angular_Ls in layout[
                                "source_block_angular_Ls"
                            ]
                        )
                    ):
                        power_skip_records.append(
                            {
                                "plan_index": int(plan_index),
                                "reason": (
                                    "multi_block_exact_shared_subtree_preferred"
                                ),
                                "block_count": int(
                                    max(
                                        len(tuple(angular_Ls))
                                        for angular_Ls in layout[
                                            "source_block_angular_Ls"
                                        ]
                                    )
                                ),
                            }
                        )
                        continue
                    if int(layout["estimated_count_table_bytes"]) > int(
                        self.power_table_byte_limit
                    ):
                        power_skip_records.append(
                            {
                                "plan_index": int(plan_index),
                                "reason": "estimated_count_table_bytes_exceeds_limit",
                                "estimated_count_table_bytes": int(
                                    layout["estimated_count_table_bytes"]
                                ),
                            }
                        )
                        continue
                    key = (
                        str(module.resolved_backend),
                        str(module.coefficient_values.dtype),
                        str(module.coefficient_values.device),
                        int(module.source_dimension),
                    )
                    records.setdefault(key, []).append(
                        {
                            "plan_index": int(plan_index),
                            "layout": layout,
                        }
                    )

                grouped_records = []
                for candidates in records.values():
                    current = []
                    current_physical_indices = set()
                    current_monomial_upper_bound = 0
                    for candidate in candidates:
                        layout = candidate["layout"]
                        candidate_physical_indices = (
                            current_physical_indices.union(
                                int(value)
                                for value in layout[
                                    "physical_input_indices"
                                ]
                            )
                        )
                        candidate_monomial_upper_bound = int(
                            current_monomial_upper_bound
                            + int(layout["monomial_upper_bound"])
                        )
                        candidate_bytes = int(
                            candidate_monomial_upper_bound
                            * len(candidate_physical_indices)
                            * 8
                        )
                        if (
                            current
                            and candidate_bytes
                            > int(self.power_table_byte_limit)
                        ):
                            grouped_records.append(tuple(current))
                            current = []
                            current_physical_indices = set()
                            current_monomial_upper_bound = 0
                            candidate_physical_indices = set(
                                int(value)
                                for value in layout[
                                    "physical_input_indices"
                                ]
                            )
                            candidate_monomial_upper_bound = int(
                                layout["monomial_upper_bound"]
                            )
                        current.append(candidate)
                        current_physical_indices = (
                            candidate_physical_indices
                        )
                        current_monomial_upper_bound = (
                            candidate_monomial_upper_bound
                        )
                    if current:
                        grouped_records.append(tuple(current))

                for records_in_group in grouped_records:
                    indices = tuple(
                        int(record["plan_index"])
                        for record in records_in_group
                    )
                    candidate_group = _PackedRepeatedBlockPowerPlanGroup(
                        tuple(modules[index] for index in indices),
                        tuple(indices),
                        power_physical_leaf_offsets_by_plan=tuple(
                            power_physical_leaf_offsets_by_plan[index]
                            for index in indices
                        ),
                        exact_physical_leaf_offsets_by_plan=tuple(
                            physical_leaf_offsets_by_plan[index]
                            for index in indices
                        ),
                        physical_input_dimension=int(
                            self.physical_input_dimension
                        ),
                        defer_exact_adjoint_group=True,
                    )
                    if int(candidate_group.power_table_bytes) > int(
                        self.power_table_byte_limit
                    ):
                        power_skip_records.extend(
                            {
                                "plan_index": int(index),
                                "reason": (
                                    "materialized_power_table_bytes_exceeds_limit"
                                ),
                                "power_table_bytes": int(
                                    candidate_group.power_table_bytes
                                ),
                            }
                            for index in indices
                        )
                        continue
                    power_groups.append(candidate_group)
                    power_plan_indices.update(indices)
            generic_indices = tuple(
                index
                for index in range(len(modules))
                if index not in power_plan_indices
            )
            if self.heterogeneous_root_grouped:
                if generic_indices:
                    groups.append(
                        _PackedFactorizedAngularPlanGroup(
                            tuple(modules[index] for index in generic_indices),
                            generic_indices,
                            physical_leaf_offsets=tuple(
                                physical_leaf_offsets_by_plan[index]
                                for index in generic_indices
                            ),
                            physical_input_dimension=(
                                self.physical_input_dimension
                            ),
                            heterogeneous_roots=True,
                        )
                    )
            else:
                records = []
                for plan_index in generic_indices:
                    module = modules[plan_index]
                    projection_signature = tuple(
                        complex(value)
                        for value in (
                            module.projection_values.detach().cpu().tolist()
                        )
                    )
                    key = (
                        str(module.resolved_backend),
                        str(module.coefficient_values.dtype),
                        str(module.coefficient_values.device),
                        int(module.source_dimension),
                        int(module.magnetic_dimension),
                        projection_signature,
                    )
                    matching = next(
                        (
                            record
                            for record in records
                            if record["key"] == key
                        ),
                        None,
                    )
                    if matching is None:
                        matching = {
                            "key": key,
                            "indices": [],
                        }
                        records.append(matching)
                    matching["indices"].append(int(plan_index))
                for record in records:
                    indices = tuple(record["indices"])
                    groups.append(
                        _PackedFactorizedAngularPlanGroup(
                            tuple(modules[index] for index in indices),
                            indices,
                            physical_leaf_offsets=(
                                None
                                if physical_leaf_offsets_by_plan is None
                                else tuple(
                                    physical_leaf_offsets_by_plan[index]
                                    for index in indices
                                )
                            ),
                            physical_input_dimension=(
                                self.physical_input_dimension
                            ),
                        )
                    )
        power_destination_groups = []
        power_execution_records = []
        power_runs = []
        for power_group_index, power_group in enumerate(power_groups):
            compatibility_key = (
                str(power_group.resolved_backend),
                str(power_group.coefficient_values.dtype),
                str(power_group.coefficient_values.device),
                int(power_group.source_dimension),
                int(power_group.physical_input_dimension),
            )
            if (
                power_runs
                and power_runs[-1]["key"] == compatibility_key
            ):
                power_runs[-1]["indices"].append(
                    int(power_group_index)
                )
            else:
                power_runs.append(
                    {
                        "key": compatibility_key,
                        "indices": [int(power_group_index)],
                    }
                )
        for run in power_runs:
            indices = tuple(int(value) for value in run["indices"])
            if (
                self.destination_derivative_grouping_requested
                and len(indices) > 1
            ):
                destination_index = len(power_destination_groups)
                power_destination_groups.append(
                    _PackedRepeatedBlockPowerDerivativeDestinationGroup(
                        tuple(power_groups[index] for index in indices)
                    )
                )
                power_execution_records.append(
                    {
                        "kind": "destination_group",
                        "index": int(destination_index),
                        "power_group_indices": indices,
                    }
                )
            else:
                for index in indices:
                    power_groups[int(index)]._install_exact_adjoint_group()
                    power_execution_records.append(
                        {
                            "kind": "individual",
                            "index": int(index),
                            "power_group_indices": (int(index),),
                        }
                    )
        self.groups = torch.nn.ModuleList(groups)
        self.power_groups = torch.nn.ModuleList(power_groups)
        self.power_destination_groups = torch.nn.ModuleList(
            power_destination_groups
        )
        self._power_execution_records = tuple(
            dict(record) for record in power_execution_records
        )
        self.homogeneous_power_plan_indices = tuple(
            sorted(power_plan_indices)
        )
        self.repeated_block_power_plan_indices = tuple(
            self.homogeneous_power_plan_indices
        )
        self.power_skip_records = tuple(
            dict(record) for record in power_skip_records
        )
        self.output_dimensions = tuple(
            int(module.output_dimension) for module in modules
        )
        output_slices = []
        output_base = 0
        for plan_index, module in enumerate(modules):
            output_slices.append(
                {
                    "plan_index": int(plan_index),
                    "start": int(output_base),
                    "stop": int(output_base + module.output_dimension),
                    "logical_channel_tableau_dimension": int(
                        module.logical_channel_tableau_dimension
                    ),
                    "magnetic_dimension": int(module.magnetic_dimension),
                    "plan_hash": str(module.plan_hash),
                }
            )
            output_base += int(module.output_dimension)
        self.output_slices = tuple(output_slices)
        self.output_dimension = int(output_base)
        execution_offsets = {}
        execution_base = 0
        if self.native_grouped:
            for group in tuple(self.groups) + tuple(
                self._power_execution_modules()
            ):
                for record in group.output_slices:
                    execution_offsets[int(record["plan_index"])] = (
                        int(execution_base + int(record["start"])),
                        int(execution_base + int(record["stop"])),
                    )
                execution_base += int(group.output_dimension)
            if set(execution_offsets) != set(range(len(modules))):
                raise RuntimeError(
                    "native factorized grouping left an unbound plan output"
                )
            packed_output_reorder_indices = tuple(
                coordinate
                for plan_index in range(len(modules))
                for coordinate in range(
                    int(execution_offsets[int(plan_index)][0]),
                    int(execution_offsets[int(plan_index)][1]),
                )
            )
        else:
            packed_output_reorder_indices = tuple(
                range(int(self.output_dimension))
            )
        self.register_buffer(
            "packed_output_reorder_indices",
            torch.tensor(
                packed_output_reorder_indices,
                dtype=torch.long,
                device=modules[0].coefficient_values.device,
            ),
            persistent=False,
        )
        self.packed_output_reorder_required = bool(
            packed_output_reorder_indices
            != tuple(range(int(self.output_dimension)))
        )
        self.last_backend = None

    def _power_execution_modules(self):
        modules = []
        for record in self._power_execution_records:
            if record["kind"] == "destination_group":
                modules.append(
                    self.power_destination_groups[int(record["index"])]
                )
            else:
                modules.append(self.power_groups[int(record["index"])])
        return tuple(modules)

    def _forward_packed_with_shape(
        self,
        slot_values_by_plan,
        packed_source_bank=None,
        leading_shape=None,
        use_native_grouping=None,
    ):
        if slot_values_by_plan is None:
            if (
                self.physical_input_dimension is None
                or packed_source_bank is None
            ):
                raise ValueError(
                    "factorized plan slots may be omitted only for a packed "
                    "physical source bank"
                )
            slot_values_by_plan = tuple(
                () for _module in self._source_modules
            )
        else:
            slot_values_by_plan = tuple(
                tuple(values) for values in slot_values_by_plan
            )
        if len(slot_values_by_plan) != len(self._source_modules):
            raise ValueError(
                "factorized angular group input count does not match plans"
            )
        grouped_dispatch = bool(self.native_grouped)
        if use_native_grouping is not None:
            grouped_dispatch = bool(
                grouped_dispatch and bool(use_native_grouping)
            )
        if not grouped_dispatch:
            if packed_source_bank is not None:
                raise ValueError(
                    "independent plan execution does not consume a packed "
                    "physical source bank"
                )
            self.last_backend = "individual_factorized_modules"
            outputs = tuple(
                module(values)
                for module, values in zip(
                    self._source_modules,
                    slot_values_by_plan,
                )
            )
            leading_shapes = tuple(
                tuple(value.shape[:-2]) for value in outputs
            )
            if any(
                shape != leading_shapes[0]
                for shape in leading_shapes[1:]
            ):
                raise ValueError(
                    "factorized plans require matching leading shapes"
                )
            batch_size = math.prod(leading_shapes[0])
            packed_output = torch.cat(
                tuple(
                    value.reshape(
                        int(batch_size),
                        int(module.output_dimension),
                    )
                    for module, value in zip(
                        self._source_modules,
                        outputs,
                    )
                ),
                dim=1,
            )
            return packed_output, leading_shapes[0]
        use_physical_bank = self.physical_input_dimension is not None
        if use_physical_bank:
            if packed_source_bank is None or leading_shape is None:
                raise ValueError(
                    "grouped physical execution requires its packed source "
                    "bank and leading shape"
                )
            leading_shape = tuple(leading_shape)
        else:
            if packed_source_bank is not None or leading_shape is not None:
                raise ValueError(
                    "packed source arguments require physical leaf offsets"
                )
            packed = []
            leading_shapes = []
            for module, values in zip(
                self._source_modules,
                slot_values_by_plan,
            ):
                packed_value, module_leading_shape = (
                    module._packed_slots(values)
                )
                packed.append(packed_value)
                leading_shapes.append(module_leading_shape)
            if any(
                shape != leading_shapes[0]
                for shape in leading_shapes[1:]
            ):
                raise ValueError(
                    "grouped factorized plans require matching leading shapes"
                )
            leading_shape = tuple(leading_shapes[0])
        packed_group_outputs = []
        for group in self.groups:
            if use_physical_bank:
                group_output = group.forward_prepacked_output(
                    packed_source_bank,
                    leading_shape,
                )
            else:
                group_output = group.forward_packed(
                    tuple(
                        packed[index]
                        for index in group.plan_indices
                    ),
                    tuple(
                        leading_shapes[index]
                        for index in group.plan_indices
                    ),
                )
            packed_group_outputs.append(group_output)
        if use_physical_bank:
            for group in self._power_execution_modules():
                group_output = group.forward_prepacked_output(
                    packed_source_bank,
                    leading_shape,
                )
                packed_group_outputs.append(group_output)
        if not packed_group_outputs:
            raise RuntimeError(
                "factorized angular group left an unbound plan output"
            )
        packed_output = (
            packed_group_outputs[0]
            if len(packed_group_outputs) == 1
            else torch.cat(tuple(packed_group_outputs), dim=1)
        )
        if int(packed_output.shape[1]) != int(self.output_dimension):
            raise RuntimeError(
                "factorized angular packed output width is inconsistent"
            )
        if self.packed_output_reorder_required:
            packed_output = packed_output.index_select(
                1,
                self.packed_output_reorder_indices.to(
                    device=packed_output.device
                ),
            )
        self.last_backend = "native_block_composed_factorized_groups"
        return packed_output, tuple(leading_shape)

    def forward_packed(
        self,
        slot_values_by_plan,
        packed_source_bank=None,
        leading_shape=None,
        use_native_grouping=None,
    ):
        output, _leading_shape = self._forward_packed_with_shape(
            slot_values_by_plan,
            packed_source_bank=packed_source_bank,
            leading_shape=leading_shape,
            use_native_grouping=use_native_grouping,
        )
        return output

    def forward(
        self,
        slot_values_by_plan,
        packed_source_bank=None,
        leading_shape=None,
        use_native_grouping=None,
    ):
        output, resolved_leading_shape = self._forward_packed_with_shape(
            slot_values_by_plan,
            packed_source_bank=packed_source_bank,
            leading_shape=leading_shape,
            use_native_grouping=use_native_grouping,
        )
        return tuple(
            output[:, int(record["start"]):int(record["stop"])].reshape(
                tuple(resolved_leading_shape)
                + (
                    int(record["logical_channel_tableau_dimension"]),
                    int(record["magnetic_dimension"]),
                )
            )
            for record in self.output_slices
        )

    def runtime_report(self):
        return {
            "runtime": "_YE3TFactorizedAngularModuleGroup",
            "member_plan_count": int(len(self._source_modules)),
            "native_capable": bool(self.native_capable),
            "native_grouping_requested": bool(
                self.native_grouping_requested
            ),
            "native_grouped": bool(self.native_grouped),
            "heterogeneous_root_grouping_requested": bool(
                self.heterogeneous_root_grouping_requested
            ),
            "heterogeneous_root_grouped": bool(
                self.heterogeneous_root_grouped
            ),
            "homogeneous_power_grouping_requested": bool(
                self.homogeneous_power_grouping_requested
            ),
            "homogeneous_power_plan_indices": tuple(
                self.homogeneous_power_plan_indices
            ),
            "homogeneous_power_group_count": int(len(self.power_groups)),
            "repeated_block_power_plan_indices": tuple(
                self.repeated_block_power_plan_indices
            ),
            "repeated_block_power_group_count": int(
                len(self.power_groups)
            ),
            "power_table_byte_limit": int(self.power_table_byte_limit),
            "repeated_block_power_policy": str(
                self.repeated_block_power_policy
            ),
            "destination_derivative_grouping_requested": bool(
                self.destination_derivative_grouping_requested
            ),
            "destination_derivative_group_count": int(
                len(self.power_destination_groups)
            ),
            "exact_power_derivative_call_count": int(
                len(self._power_execution_records)
            ),
            "individual_exact_power_derivative_call_count": int(
                len(self.power_groups)
            ),
            "power_skip_records": tuple(
                dict(record) for record in self.power_skip_records
            ),
            "prepacked_physical_source_bank": bool(
                self.physical_input_dimension is not None
            ),
            "physical_input_dimension": (
                None
                if self.physical_input_dimension is None
                else int(self.physical_input_dimension)
            ),
            "native_call_count": (
                int(len(self.groups) + len(self.power_groups))
                if self.native_grouped
                else int(len(self._source_modules))
            ),
            "flat_output_available": True,
            "output_dimension": int(self.output_dimension),
            "packed_output_reorder_required": bool(
                self.packed_output_reorder_required
            ),
            "member_plan_hashes": tuple(
                str(module.plan_hash)
                for module in self._source_modules
            ),
            "groups": tuple(
                group.runtime_report() for group in self.groups
            ),
            "power_groups": tuple(
                group.runtime_report() for group in self.power_groups
            ),
            "power_destination_groups": tuple(
                group.runtime_report()
                for group in self.power_destination_groups
            ),
            "subtree_evaluation_shared": any(
                group.subtree_evaluation_shared
                for group in self.groups
            ),
            "last_backend": self.last_backend,
        }


class _YE3TSegmentedFactorizedAngularModuleGroup(torch.nn.Module):
    """Pack exact factorized plans with different source dimensions."""

    def __init__(self, modules, plan_indices=None, enable_native=True):
        super().__init__()
        modules = tuple(modules)
        if not modules:
            raise ValueError(
                "segmented factorized group requires at least one plan"
            )
        if plan_indices is None:
            plan_indices = tuple(range(len(modules)))
        else:
            plan_indices = tuple(int(value) for value in plan_indices)
        if len(plan_indices) != len(modules):
            raise ValueError(
                "segmented factorized plan indices must match modules"
            )
        first = modules[0]
        if any(
            module.resolved_backend != first.resolved_backend
            or module.coefficient_values.dtype
            != first.coefficient_values.dtype
            or module.coefficient_values.device
            != first.coefficient_values.device
            for module in modules[1:]
        ):
            raise ValueError(
                "segmented factorized plans must share backend, dtype, and device"
            )
        object.__setattr__(self, "_source_modules", modules)
        self.plan_indices = plan_indices
        self.native_requested = bool(enable_native)
        self.native_capable = bool(
            first.resolved_backend in {"native_cpu", "native_cuda"}
            and hasattr(
                torch.ops.ye3t_runtime,
                "factorized_angular_segmented",
            )
        )
        self.native_enabled = bool(
            self.native_requested and self.native_capable
        )

        source_offsets = [0]
        input_offsets = []
        input_dimensions = []
        workspace_offsets = []
        workspace_dimensions = []
        node_group_offsets = [0]
        root_group_offsets = [0]
        projection_offsets = []
        projection_dimensions = []
        output_group_offsets = [0]
        node_offsets = []
        node_dimensions = []
        node_leaf_offsets = []
        node_left = []
        node_right = []
        node_coefficient_offsets = []
        coefficient_rows = []
        coefficient_columns = []
        coefficient_values = []
        root_nodes = []
        root_projection_starts = []
        root_projection_dimensions = []
        root_output_offsets = []
        projection_values = []
        output_slices = []
        input_base = 0
        workspace_base = 0
        node_base = 0
        coefficient_base = 0
        projection_base = 0
        output_base = 0
        for module, plan_index in zip(modules, plan_indices):
            source_dimension = int(module.source_dimension)
            input_dimension = int(module.input_dimension)
            workspace_dimension = int(module.workspace_dimension)
            node_count = int(module.node_offsets.numel())
            root_count = int(module.root_nodes.numel())
            if source_dimension <= 0 or input_dimension <= 0:
                raise ValueError(
                    "segmented factorized plans require positive source and input dimensions"
                )
            if (
                int(module.projection_values.numel())
                % source_dimension
                != 0
            ):
                raise ValueError(
                    "factorized projection table does not match source dimension"
                )
            projection_dimension = int(
                module.projection_values.numel() // source_dimension
            )
            expected_output_dimension = int(
                root_count
                * projection_dimension
                * int(module.magnetic_dimension)
            )
            if expected_output_dimension != int(module.output_dimension):
                raise ValueError(
                    "segmented factorized output does not match roots"
                )
            source_offsets.append(
                int(source_offsets[-1]) + source_dimension
            )
            input_offsets.append(int(input_base))
            input_dimensions.append(input_dimension)
            workspace_offsets.append(int(workspace_base))
            workspace_dimensions.append(workspace_dimension)
            projection_offsets.append(int(projection_base))
            projection_dimensions.append(projection_dimension)

            node_offsets.append(module.node_offsets)
            node_dimensions.append(module.node_dimensions)
            node_leaf_offsets.append(module.node_leaf_offsets)
            node_left.append(
                module.node_left
            )
            node_right.append(
                module.node_right
            )
            local_coefficient_offsets = (
                module.node_coefficient_offsets + int(coefficient_base)
            )
            if node_coefficient_offsets:
                local_coefficient_offsets = local_coefficient_offsets[1:]
            node_coefficient_offsets.append(local_coefficient_offsets)
            coefficient_rows.append(module.coefficient_rows)
            coefficient_columns.append(module.coefficient_columns)
            coefficient_values.append(module.coefficient_values)
            root_nodes.append(module.root_nodes)
            root_projection_starts.extend((0,) * root_count)
            root_projection_dimensions.extend(
                (projection_dimension,) * root_count
            )
            root_output_offsets.extend(
                int(
                    local_root
                    * projection_dimension
                    * int(module.magnetic_dimension)
                )
                for local_root in range(root_count)
            )
            projection_values.append(module.projection_values)
            node_base += node_count
            coefficient_base += int(module.coefficient_values.numel())
            projection_base += int(module.projection_values.numel())
            input_base += source_dimension * input_dimension
            workspace_base += source_dimension * workspace_dimension
            output_slices.append(
                {
                    "plan_index": int(plan_index),
                    "start": int(output_base),
                    "stop": int(output_base + module.output_dimension),
                    "logical_channel_tableau_dimension": int(
                        module.logical_channel_tableau_dimension
                    ),
                    "magnetic_dimension": int(
                        module.magnetic_dimension
                    ),
                    "plan_hash": str(module.plan_hash),
                }
            )
            output_base += int(module.output_dimension)
            output_group_offsets.append(int(output_base))
            node_group_offsets.append(int(node_base))
            root_group_offsets.append(
                int(root_group_offsets[-1]) + root_count
            )

        device = first.node_offsets.device

        def register_integer(name, values):
            self.register_buffer(
                name,
                torch.as_tensor(
                    values,
                    dtype=torch.int64,
                    device=device,
                ).contiguous(),
                persistent=False,
            )

        register_integer("segment_source_offsets", source_offsets)
        register_integer("segment_input_offsets", input_offsets)
        register_integer("segment_input_dimensions", input_dimensions)
        register_integer("segment_workspace_offsets", workspace_offsets)
        register_integer(
            "segment_workspace_dimensions", workspace_dimensions
        )
        register_integer("segment_node_offsets", node_group_offsets)
        register_integer("segment_root_offsets", root_group_offsets)
        register_integer("segment_projection_offsets", projection_offsets)
        register_integer(
            "segment_projection_dimensions", projection_dimensions
        )
        register_integer("segment_output_offsets", output_group_offsets)
        self.register_buffer(
            "node_offsets",
            torch.cat(tuple(node_offsets)).contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "node_dimensions",
            torch.cat(tuple(node_dimensions)).contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "node_leaf_offsets",
            torch.cat(tuple(node_leaf_offsets)).contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "node_left",
            torch.cat(tuple(node_left)).contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "node_right",
            torch.cat(tuple(node_right)).contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "node_coefficient_offsets",
            torch.cat(tuple(node_coefficient_offsets)).contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "coefficient_rows",
            torch.cat(tuple(coefficient_rows)).contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "coefficient_columns",
            torch.cat(tuple(coefficient_columns)).contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "coefficient_values",
            torch.cat(tuple(coefficient_values)).contiguous(),
            persistent=False,
        )
        register_integer(
            "root_nodes",
            torch.cat(tuple(root_nodes)),
        )
        register_integer(
            "root_projection_starts", root_projection_starts
        )
        register_integer(
            "root_projection_dimensions", root_projection_dimensions
        )
        register_integer("root_output_offsets", root_output_offsets)
        self.register_buffer(
            "projection_values",
            torch.cat(tuple(projection_values)).contiguous(),
            persistent=False,
        )
        self.total_source_count = int(source_offsets[-1])
        self.input_dimension = int(input_base)
        self.workspace_dimension = int(workspace_base)
        self.output_dimension = int(output_base)
        self.cuda_auto_warp_min_coefficient_count = 128
        self.cuda_auto_segments = tuple(
            {
                "plan_index": int(plan_index),
                "source_dimension": int(module.source_dimension),
                "node_count": int(module.node_offsets.numel()),
                "coefficient_count": int(
                    module.coefficient_values.numel()
                ),
                "policy": (
                    "warp"
                    if int(module.coefficient_values.numel())
                    >= int(self.cuda_auto_warp_min_coefficient_count)
                    else "serial"
                ),
            }
            for module, plan_index in zip(modules, plan_indices)
        )
        cuda_auto_policies = {
            str(record["policy"]) for record in self.cuda_auto_segments
        }
        self.cuda_auto_policy = (
            next(iter(cuda_auto_policies))
            if len(cuda_auto_policies) == 1
            else "hybrid"
        )
        self.output_slices = tuple(output_slices)
        self.last_backend = None

    def _native_arguments(self):
        return (
            self.segment_source_offsets,
            self.segment_input_offsets,
            self.segment_input_dimensions,
            self.segment_workspace_offsets,
            self.segment_workspace_dimensions,
            self.segment_node_offsets,
            self.segment_root_offsets,
            self.segment_projection_offsets,
            self.segment_projection_dimensions,
            self.segment_output_offsets,
            self.node_offsets,
            self.node_dimensions,
            self.node_leaf_offsets,
            self.node_left,
            self.node_right,
            self.node_coefficient_offsets,
            self.coefficient_rows,
            self.coefficient_columns,
            self.coefficient_values,
            self.root_nodes,
            self.root_projection_starts,
            self.root_projection_dimensions,
            self.root_output_offsets,
            self.projection_values,
            self.total_source_count,
            self.workspace_dimension,
            self.output_dimension,
        )

    def _reference_output(self, slot_values_by_plan):
        outputs = tuple(
            module(values)
            for module, values in zip(
                self._source_modules,
                slot_values_by_plan,
            )
        )
        leading_shapes = tuple(
            tuple(value.shape[:-2]) for value in outputs
        )
        if any(shape != leading_shapes[0] for shape in leading_shapes[1:]):
            raise ValueError(
                "segmented factorized plans require matching leading shapes"
            )
        batch_size = math.prod(leading_shapes[0])
        return (
            torch.cat(
                tuple(
                    value.reshape(batch_size, int(module.output_dimension))
                    for module, value in zip(self._source_modules, outputs)
                ),
                dim=1,
            ),
            leading_shapes[0],
        )

    def forward_packed(self, slot_values_by_plan, use_native=None):
        slot_values_by_plan = tuple(
            tuple(values) for values in slot_values_by_plan
        )
        if len(slot_values_by_plan) != len(self._source_modules):
            raise ValueError(
                "segmented factorized input count does not match plans"
            )
        native = self.native_enabled
        if use_native is not None:
            native = bool(native and bool(use_native))
        if not native:
            output, _leading_shape = self._reference_output(
                slot_values_by_plan
            )
            self.last_backend = "independent_exact_factorized_reference"
            return output
        packed = []
        leading_shapes = []
        for module, values in zip(
            self._source_modules,
            slot_values_by_plan,
        ):
            packed_value, leading_shape = module._packed_slots(values)
            packed.append(packed_value)
            leading_shapes.append(tuple(leading_shape))
        if any(shape != leading_shapes[0] for shape in leading_shapes[1:]):
            raise ValueError(
                "segmented factorized plans require matching leading shapes"
            )
        packed_slots = torch.cat(tuple(packed), dim=1).contiguous()
        if int(packed_slots.shape[1]) != int(self.input_dimension):
            raise RuntimeError(
                "segmented factorized packed input width is inconsistent"
            )
        output = torch.ops.ye3t_runtime.factorized_angular_segmented(
            packed_slots,
            *self._native_arguments(),
        )
        self.last_backend = "native_destination_segmented_factorized_angular"
        return output

    def forward(self, slot_values_by_plan, use_native=None):
        slot_values_by_plan = tuple(
            tuple(values) for values in slot_values_by_plan
        )
        output = self.forward_packed(
            slot_values_by_plan,
            use_native=use_native,
        )
        first_values = tuple(slot_values_by_plan)[0][0]
        leading_shape = tuple(
            first_values.shape[
                :-1
                if int(self._source_modules[0].source_dimension) == 1
                else -2
            ]
        )
        return tuple(
            output[:, int(record["start"]):int(record["stop"])].reshape(
                leading_shape
                + (
                    int(record["logical_channel_tableau_dimension"]),
                    int(record["magnetic_dimension"]),
                )
            )
            for record in self.output_slices
        )

    def runtime_report(self):
        return {
            "runtime": "_YE3TSegmentedFactorizedAngularModuleGroup",
            "member_plan_count": int(len(self._source_modules)),
            "member_plan_indices": tuple(self.plan_indices),
            "member_plan_hashes": tuple(
                str(module.plan_hash) for module in self._source_modules
            ),
            "source_dimensions": tuple(
                int(module.source_dimension)
                for module in self._source_modules
            ),
            "total_source_count": int(self.total_source_count),
            "input_dimension": int(self.input_dimension),
            "workspace_dimension": int(self.workspace_dimension),
            "output_dimension": int(self.output_dimension),
            "cuda_auto_policy": str(self.cuda_auto_policy),
            "cuda_auto_segments": tuple(
                dict(record) for record in self.cuda_auto_segments
            ),
            "cuda_auto_warp_min_coefficient_count": int(
                self.cuda_auto_warp_min_coefficient_count
            ),
            "native_requested": bool(self.native_requested),
            "native_capable": bool(self.native_capable),
            "native_enabled": bool(self.native_enabled),
            "native_call_count": 1 if self.native_enabled else int(
                len(self._source_modules)
            ),
            "destination_order": "compiler_plan_order",
            "runtime_path_discovery": False,
            "composition": (
                "ragged_source_C_dagger_to_compiler_destination_segments"
            ),
            "last_backend": self.last_backend,
        }


class YE3TSourceAnalysisModule(torch.nn.Module):
    """Prepared Torch module for one execution-plan source-analysis instruction."""

    def __init__(
        self,
        execution_plan,
        instruction_id=None,
        backend="auto",
        dtype=None,
        device=None,
        channel_mixing=False,
        strict=False,
    ):
        super().__init__()
        if not isinstance(execution_plan, YE3TExecutionPlan):
            execution_plan = YE3TExecutionPlan.from_dict(execution_plan)
        instructions = {
            instruction.instruction_id: instruction
            for instruction in execution_plan.instructions
        }
        if instruction_id is None:
            if len(execution_plan.forward_schedule) != 1:
                raise ValueError(
                    "instruction_id is required for a multi-instruction plan"
                )
            instruction_id = execution_plan.forward_schedule[0]
        instruction_id = str(instruction_id)
        if instruction_id not in instructions:
            raise ValueError("instruction_id is not present in the execution plan")
        instruction = instructions[instruction_id]
        if instruction.source_assembly_id is None:
            raise ValueError("instruction does not reference a source assembly")
        if instruction.synthesis_table_id is None:
            raise ValueError("instruction does not reference a synthesis table")
        assemblies = {
            assembly.assembly_id: assembly
            for assembly in execution_plan.source_assemblies
        }
        tables = {
            table.table_id: table
            for table in execution_plan.synthesis_tables
        }
        assembly = assemblies[instruction.source_assembly_id]
        table = tables[instruction.synthesis_table_id]
        if int(assembly.induced_dimension) != int(table.input_dimension):
            raise ValueError("instruction source and synthesis dimensions disagree")
        output_layouts = tuple(
            layout
            for layout in execution_plan.carrier_layouts
            if layout.key == instruction.output_carrier
        )
        if len(output_layouts) != 1:
            raise ValueError(
                "source-analysis instruction must resolve one output carrier layout"
            )
        output_layout = output_layouts[0]
        if int(output_layout.width) != int(table.output_dimension):
            raise ValueError(
                "source-analysis table width does not match its output carrier layout"
            )
        declared_real_basis = (
            str(output_layout.key.convention_id)
            == YE3T_REAL_TESSERAL_CONVENTION
        )
        backend = str(backend)
        if backend not in {"auto", "native", "reference", "triton"}:
            raise ValueError(
                "backend must be 'auto', 'native', 'reference', or 'triton'"
            )
        require_native = _enabled("YE3T_REQUIRE_NATIVE")
        if backend == "reference" and require_native:
            raise RuntimeError(
                "YE3T_REQUIRE_NATIVE=1 forbids explicit reference fallback"
            )
        if dtype is None:
            has_complex = any(value.imag != 0.0 for value in assembly.values)
            has_complex = has_complex or any(
                value.imag != 0.0 for value in table.values
            )
            dtype = torch.complex128 if has_complex else torch.float64
        device = torch.device("cpu" if device is None else device)
        has_complex_dtype = dtype in (torch.complex64, torch.complex128)
        if backend == "triton":
            if device.type != "cuda":
                raise ValueError("triton source analysis requires a CUDA device")
            if has_complex_dtype or not declared_real_basis:
                raise ValueError(
                    "triton source analysis requires a serialized real-tesseral "
                    "basis transform and real dtype"
                )
            resolved_backend = "triton_source_analysis"
        elif backend == "native":
            extension = _load_extension()
            if device.type == "cuda":
                if not bool(extension.has_cuda()):
                    raise RuntimeError(
                        "the installed YE3T native extension has no CUDA "
                        "source-analysis operators"
                    )
                resolved_backend = "native_cuda"
            elif device.type == "cpu":
                resolved_backend = "native_cpu"
            else:
                raise ValueError(
                    "native execution-plan runtime requires CPU or CUDA"
                )
        elif backend == "reference":
            resolved_backend = "torch_reference"
        elif device.type == "cuda":
            extension = _prebuilt_extension()
            native_cuda = bool(
                extension is not None and extension.has_cuda()
            )
            if require_native and native_cuda:
                resolved_backend = "native_cuda"
            elif not has_complex_dtype and declared_real_basis:
                resolved_backend = "triton_source_analysis"
            elif native_cuda:
                resolved_backend = "native_cuda"
            else:
                if require_native:
                    raise RuntimeError(
                        "YE3T_REQUIRE_NATIVE=1 cannot execute a source-analysis "
                        "plan because the installed extension has no CUDA "
                        "operators and the real-only Triton kernel does not "
                        "support this convention/dtype"
                    )
                resolved_backend = "torch_reference"
        else:
            use_native = require_native or (
                _prebuilt_extension() is not None
                or _enabled("YE3T_ENABLE_EXECUTION_PLAN_JIT")
            )
            if use_native:
                _load_extension()
                resolved_backend = "native_cpu"
            else:
                resolved_backend = "torch_reference"
        coefficient_values = _coefficient_values(assembly.values, dtype)
        synthesis_values = _coefficient_values(table.values, dtype)
        self.register_buffer(
            "assembly_rows",
            torch.tensor(assembly.row_indices, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.register_buffer(
            "assembly_columns",
            torch.tensor(
                assembly.column_indices,
                dtype=torch.int64,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "assembly_values",
            torch.tensor(coefficient_values, dtype=dtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "synthesis_rows",
            torch.tensor(table.row_indices, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.register_buffer(
            "synthesis_columns",
            torch.tensor(table.column_indices, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.register_buffer(
            "synthesis_values",
            torch.tensor(synthesis_values, dtype=dtype, device=device),
            persistent=False,
        )
        self.source_dimension = int(assembly.source_dimension)
        self.induced_dimension = int(assembly.induced_dimension)
        self.output_dimension = int(table.output_dimension)
        self.channel_count = int(output_layout.channel_count)
        self.tableau_count = int(output_layout.tableau_count)
        self.magnetic_count = int(output_layout.magnetic_count)
        self.component_width = int(self.tableau_count * self.magnetic_count)
        mixing_weight = torch.eye(
            self.channel_count,
            dtype=dtype,
            device=device,
        )
        if bool(channel_mixing):
            self.mixing_weight = torch.nn.Parameter(mixing_weight)
        else:
            self.register_buffer(
                "mixing_weight",
                mixing_weight,
                persistent=False,
            )
        composed_terms = _compose_source_analysis_table(assembly, table)
        (
            source_indices,
            output_indices,
            weight_indices,
            mixed_coefficients,
        ) = _expand_source_analysis_mixing_table(
            composed_terms,
            self.channel_count,
            self.component_width,
        )
        if has_complex_dtype:
            mixed_values = tuple(mixed_coefficients)
        else:
            if any(abs(value.imag) > 1e-15 for value in mixed_coefficients):
                raise ValueError(
                    "real source-analysis dtype cannot represent complex coefficients"
                )
            mixed_values = tuple(float(value.real) for value in mixed_coefficients)
        self.register_buffer(
            "mixed_source_index",
            torch.tensor(source_indices, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.register_buffer(
            "mixed_output_index",
            torch.tensor(output_indices, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.register_buffer(
            "mixed_weight_index",
            torch.tensor(weight_indices, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.register_buffer(
            "mixed_coefficient",
            torch.tensor(mixed_values, dtype=dtype, device=device),
            persistent=False,
        )
        self.instruction_id = instruction_id
        self.opcode = str(instruction.opcode)
        self.plan_hash = str(execution_plan.plan_hash)
        self.coefficient_hash = str(execution_plan.coefficient_hash)
        self.source_kind = str(assembly.source_realization.kind)
        self.convention_id = str(output_layout.key.convention_id)
        self.declared_real_basis = bool(declared_real_basis)
        self.resolved_backend = resolved_backend
        self.analysis_orientation = str(instruction.analysis_orientation)
        self.channel_mixing = bool(channel_mixing)
        self.strict = bool(strict or backend == "triton" or require_native)
        self.last_backend = None
        self.last_linear_readout_backend = None

    def _mixed_table(self):
        return WeightedSparseLinearTable(
            input_index=self.mixed_source_index,
            output_index=self.mixed_output_index,
            weight_index=self.mixed_weight_index,
            coefficient=self.mixed_coefficient,
            output_width=int(self.output_dimension),
            weight_count=int(self.mixing_weight.numel()),
        )

    def _apply_channel_mixing(self, output):
        logical = output.reshape(
            int(output.shape[0]),
            self.channel_count,
            self.component_width,
        )
        return torch.matmul(self.mixing_weight, logical).reshape(
            int(output.shape[0]),
            self.output_dimension,
        )

    def forward(self, source):
        if int(source.shape[-1]) != self.source_dimension:
            raise ValueError("source last dimension does not match prepared plan")
        working = source.to(
            device=self.assembly_values.device,
            dtype=self.assembly_values.dtype,
        )
        flat = working.reshape(-1, self.source_dimension).contiguous()
        if self.resolved_backend == "triton_source_analysis":
            if flat.is_complex():
                raise ValueError(
                    "triton source analysis requires declared real-basis values"
                )
            output = weighted_sparse_linear_forward(
                flat,
                self.mixing_weight.reshape(-1),
                self._mixed_table(),
                prefer_triton=True,
                strict=bool(self.strict),
            )
            self.last_backend = (
                _triton_joint._LAST_WEIGHTED_SPARSE_LINEAR_BACKEND
            )
        elif self.resolved_backend in {"native_cpu", "native_cuda"}:
            output = torch.ops.ye3t_runtime.source_analysis(
                flat,
                self.assembly_rows,
                self.assembly_columns,
                self.assembly_values,
                self.synthesis_rows,
                self.synthesis_columns,
                self.synthesis_values,
                self.induced_dimension,
                self.output_dimension,
            )
            self.last_backend = str(self.resolved_backend)
            if self.channel_mixing:
                output = self._apply_channel_mixing(output)
        else:
            ambient = flat.new_zeros(
                (flat.shape[0], self.induced_dimension)
            )
            ambient.index_add_(
                1,
                self.assembly_rows,
                flat.index_select(1, self.assembly_columns)
                * self.assembly_values.reshape(1, -1),
            )
            synthesis = flat.new_zeros(
                (self.induced_dimension, self.output_dimension)
            )
            synthesis.index_put_(
                (self.synthesis_rows, self.synthesis_columns),
                self.synthesis_values,
                accumulate=True,
            )
            output = ambient @ synthesis.conj()
            self.last_backend = "torch_reference"
            if self.channel_mixing:
                output = self._apply_channel_mixing(output)
        return output.reshape(
            tuple(source.shape[:-1]) + (self.output_dimension,)
        )

    def linear_readout(self, source, weight, bias):
        if int(source.shape[-1]) != self.source_dimension:
            raise ValueError("source last dimension does not match prepared plan")
        working = source.to(
            device=self.assembly_values.device,
            dtype=self.assembly_values.dtype,
        )
        flat = working.reshape(-1, self.source_dimension).contiguous()
        weight = torch.as_tensor(
            weight,
            dtype=self.assembly_values.dtype,
            device=self.assembly_values.device,
        )
        bias = torch.as_tensor(
            bias,
            dtype=self.assembly_values.dtype,
            device=self.assembly_values.device,
        ).reshape(())
        if weight.ndim != 1 or int(weight.numel()) != self.output_dimension:
            raise ValueError(
                "weight must be a vector matching source-analysis output"
            )
        if (
            self.resolved_backend in {"native_cpu", "native_cuda"}
            and not self.channel_mixing
        ):
            output = torch.ops.ye3t_runtime.source_analysis_linear(
                flat,
                self.assembly_rows,
                self.assembly_columns,
                self.assembly_values,
                self.synthesis_rows,
                self.synthesis_columns,
                self.synthesis_values,
                self.induced_dimension,
                weight.contiguous(),
                bias.contiguous(),
            )
            self.last_linear_readout_backend = (
                "native_fused_source_analysis_linear"
            )
        else:
            features = self.forward(source).reshape(
                -1,
                self.output_dimension,
            )
            output = features @ weight + bias
            self.last_linear_readout_backend = (
                f"materialized_{self.last_backend}"
            )
        return output.reshape(tuple(source.shape[:-1]))

    def runtime_report(self):
        return {
            "runtime": "YE3TSourceAnalysisModule",
            "instruction_id": str(self.instruction_id),
            "opcode": str(self.opcode),
            "plan_hash": str(self.plan_hash),
            "coefficient_hash": str(self.coefficient_hash),
            "source_kind": str(self.source_kind),
            "convention_id": str(self.convention_id),
            "declared_real_basis": bool(self.declared_real_basis),
            "source_dimension": int(self.source_dimension),
            "induced_dimension": int(self.induced_dimension),
            "output_dimension": int(self.output_dimension),
            "backend": str(self.resolved_backend),
            "last_backend": self.last_backend,
            "last_linear_readout_backend": (
                self.last_linear_readout_backend
            ),
            "analysis_orientation": str(self.analysis_orientation),
            "channel_mixing": bool(self.channel_mixing),
            "learned_map_axis": "channel_or_multiplicity",
            "preserved_axes": ("tableau_t", "magnetic_M"),
            "channel_count": int(self.channel_count),
            "tableau_count": int(self.tableau_count),
            "magnetic_count": int(self.magnetic_count),
            "fused_source_analysis_and_mixing": bool(
                self.resolved_backend == "triton_source_analysis"
            ),
            "source_analysis_terms": int(self.mixed_coefficient.numel()),
            "strict": bool(self.strict),
            "full_source_assembly_applied": True,
            "fused_operations": (
                "source_placement_assembly_L_v",
                "joint_young_angular_analysis_C_dagger",
                "channel_or_multiplicity_mixing",
                "output_packing",
            ),
            "learned_map_fused": bool(
                self.channel_mixing
                and self.resolved_backend == "triton_source_analysis"
            ),
        }


__all__ = [
    "YE3TSourceAnalysisModule",
    "apply_source_analysis",
    "apply_source_analysis_native",
    "cheb_exp_cos_radial_with_derivative",
    "cheb_exp_cos_radial_table_with_derivative",
    "compact_exterior_pair_product",
    "compact_exterior_power_adjoint_reference",
    "compact_exterior_power_product",
    "compact_exterior_power_product_reference",
    "compact_pair_product",
    "compact_pair_product_adjoint_reference",
    "compact_pair_product_reference",
    "compact_symmetric_pair_product",
    "carrier_channel_update",
    "carrier_channel_update_adjoint",
    "carrier_channel_transform",
    "carrier_channel_transform_adjoint",
    "carrier_role_channel_map_adjoint",
    "carrier_gated_scatter",
    "carrier_gated_scatter_adjoint",
    "carrier_residual_gated_scatter",
    "carrier_residual_gated_scatter_adjoint",
    "carrier_segmented_residual_gated_scatter",
    "carrier_segmented_residual_gated_scatter_adjoint",
    "density_accumulate",
    "density_accumulate_adjoint",
    "edge_outer_accumulate",
    "edge_outer_accumulate_adjoint",
    "softmax_gaussian_role_density",
    "softmax_gaussian_role_density_adjoint",
    "scheduled_softmax_gaussian_role_density",
    "native_execution_plan_capabilities",
    "plain_site_basis_product_with_derivative",
    "packed_concatenate",
    "prepare_source_arena_schedule",
    "prepare_carrier_scatter_segments",
    "scheduled_radial_angular_channels_with_derivative",
    "plain_site_basis_product_adjoint",
    "spherical_harmonics_with_derivative",
    "spherical_harmonics_table_with_derivative",
    "source_arena_gather",
    "source_arena_gather_adjoint",
    "source_arena_channel_transform",
    "symmetric_power_monomial_adjoint_reference",
    "symmetric_power_monomial_contraction",
    "symmetric_power_monomial_reference",
    "symmetric_power_shared_monomial_adjoint_reference",
    "symmetric_power_shared_monomial_batched_adjoint",
    "symmetric_power_shared_monomial_batched_adjoint_reference",
    "symmetric_power_shared_monomial_contraction",
    "symmetric_power_shared_monomial_reference",
]
