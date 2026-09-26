"""Runtime package facade with compiler-heavy paths loaded lazily."""

from importlib import import_module

from .features import *
from .symmetric_power import *


_NATIVE_EXPORTS = {
    "NativeRuntimeSegment",
    "NativeRuntimePlan",
    "NativeYE3TOperatorModule",
    "MultiNativeYE3TOperatorModule",
    "build_native_runtime_plan",
}

_GENERALIZED_EXPORTS = {
    "GeneralizedExactRuntimeBlock",
    "GeneralizedExactRuntimeIrreps",
    "GeneralizedFullTensorProduct",
    "GeneralizedIrreps",
    "GeneralizedLinear",
    "GeneralizedMulIrrep",
    "JointYoungCGTensorSchedule",
    "JointYoungCGProduct",
    "YoungE3TensorProduct",
}

_PACKED_REDUCTION_EXPORTS = {
    "PackedWeightedSparseBilinearGroup",
}

_SCHUR_WEYL_TREE_EXPORTS = {
    "SchurWeylGuidedTreeProduct",
    "SchurWeylTreeNode",
    "compile_schur_weyl_guided_tree_product",
    "compile_schur_weyl_guided_tree_product_from_coupler",
}

_EXECUTION_PLAN_EXPORTS = {
    "YE3TFactorizedAngularModule",
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
    "prepare_carrier_scatter_segments",
    "density_accumulate",
    "density_accumulate_adjoint",
    "edge_outer_accumulate",
    "edge_outer_accumulate_adjoint",
    "softmax_gaussian_role_density",
    "softmax_gaussian_role_density_adjoint",
    "native_execution_plan_capabilities",
    "plain_site_basis_product_adjoint",
    "plain_site_basis_product_with_derivative",
    "packed_concatenate",
    "prepare_source_arena_schedule",
    "scheduled_radial_angular_channels_with_derivative",
    "scheduled_softmax_gaussian_role_density",
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
}

_HIERARCHICAL_EXPORTS = {
    "YE3THierarchicalRepeatedBlockModule",
    "YE3THierarchicalRepeatedBlockModuleGroup",
}

_ALGEBRAIC_CURVATURE_EXPORTS = {
    "YE3TAlgebraicCurvatureOutput",
    "YE3TAlgebraicCurvatureReadout",
    "YE3TBindingAttachedAlgebraicCurvatureReadout",
}

_LAZY_EXPORT_MODULES = {}
for _name in _NATIVE_EXPORTS:
    _LAZY_EXPORT_MODULES[_name] = "ye3t.runtime.native"
for _name in _GENERALIZED_EXPORTS:
    _LAZY_EXPORT_MODULES[_name] = "ye3t.runtime.generalized"
for _name in _PACKED_REDUCTION_EXPORTS:
    _LAZY_EXPORT_MODULES[_name] = "ye3t.backends.triton_joint"
for _name in _SCHUR_WEYL_TREE_EXPORTS:
    _LAZY_EXPORT_MODULES[_name] = "ye3t.runtime.schur_weyl_tree"
for _name in _EXECUTION_PLAN_EXPORTS:
    _LAZY_EXPORT_MODULES[_name] = "ye3t.runtime.execution_plan"
for _name in _HIERARCHICAL_EXPORTS:
    _LAZY_EXPORT_MODULES[_name] = "ye3t.runtime.hierarchical"
for _name in _ALGEBRAIC_CURVATURE_EXPORTS:
    _LAZY_EXPORT_MODULES[_name] = "ye3t.runtime.algebraic_curvature"
del _name


def __getattr__(name):
    """Load runtime/compiler-heavy modules only when their symbols are requested."""
    module_name = _LAZY_EXPORT_MODULES.get(name)
    if module_name is not None:
        module = import_module(module_name)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(name)


__all__ = [
    name
    for name in tuple(globals())
    if not name.startswith("_") and name not in {"import_module"}
] + sorted(_LAZY_EXPORT_MODULES)
