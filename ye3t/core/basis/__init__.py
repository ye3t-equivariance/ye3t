"""Public import surface for ye3t basis construction."""

from .characters import cg_allowed, sym_square_allowed
from .formatters import LabelPrinter, TreePrinter
from .sector import ExactBasisEntry, ExactBasisHandle, ExactBasisSector, ExactSectorSignature
from .validation import (
    count_canonical_leaf_labelings,
    iter_canonical_leaf_labelings,
    reachable_total_angular_momenta,
    validate_leaf_quantum_numbers,
    validate_target_angular_momentum,
    validate_tree_type,
)


_BUILDER_EXPORTS = {
    "BalancedPairwiseACELabeler",
    "ExactACELabeler",
    "YE3TBasisLabeler",
}

_BENCHMARK_EXPORTS = {
    "MethodComparisonResult",
    "compare_schur_weyl_vs_gramian",
}

_METADATA_EXPORTS = {
    "BlockMultiplicityMetadata",
    "ExactLabelMetadata",
    "InternalAngularNodeMetadata",
    "ReconstructionRecipe",
    "TensorBasisExpansionCache",
    "TensorBasisTerm",
}

_THEORY_EXPORTS = {
    "FinalSO3CouplingPath",
    "InvariantSubspaceDecomposition",
    "SymmetricPowerDecomposition",
    "TensorProductChannel",
    "YoungBlockDecomposition",
    "YoungResolvedSO3CouplingPath",
    "YoungResolvedSubspaceDecomposition",
    "YoungSubgroupBlock",
    "ace_invariant_subspace_decomposition",
    "channels_from_nl",
    "final_so3_coupling_multiplicity",
    "invariant_subspace_decomposition",
    "symmetric_power_decomposition",
    "young_block_decomposition",
    "young_resolved_subspace_decomposition",
    "young_subgroup_blocks",
    "young_subgroup_order",
    "young_subgroup_projector_prefactor",
}

_MULTIPLICITY_EXPORTS = {
    "couple_block_irrep_multiplicities",
    "couple_so3_multiplicity_distributions",
    "symmetric_power_irrep_multiplicities",
    "symmetric_power_weight_counts",
}

_LAZY_EXPORT_MODULES = {}
for _name in _BUILDER_EXPORTS:
    _LAZY_EXPORT_MODULES[_name] = "ye3t.core.basis.builder"
for _name in _BENCHMARK_EXPORTS:
    _LAZY_EXPORT_MODULES[_name] = "ye3t.core.basis.benchmark"
for _name in _METADATA_EXPORTS:
    _LAZY_EXPORT_MODULES[_name] = "ye3t.core.basis.metadata"
for _name in _THEORY_EXPORTS:
    _LAZY_EXPORT_MODULES[_name] = "ye3t.core.basis.theory"
for _name in _MULTIPLICITY_EXPORTS:
    _LAZY_EXPORT_MODULES[_name] = "ye3t.core.basis.multiplicity"
del _name


def __getattr__(name):
    module_name = _LAZY_EXPORT_MODULES.get(name)
    if module_name is not None:
        module = __import__(module_name, fromlist=[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(name)


__all__ = [
    "BalancedPairwiseACELabeler",
    "ExactACELabeler",
    "YE3TBasisLabeler",
    "cg_allowed",
    "sym_square_allowed",
    "LabelPrinter",
    "TreePrinter",
    "compare_schur_weyl_vs_gramian",
    "MethodComparisonResult",
    "ExactSectorSignature",
    "ExactBasisHandle",
    "ExactBasisEntry",
    "ExactBasisSector",
    "TensorProductChannel",
    "YoungSubgroupBlock",
    "SymmetricPowerDecomposition",
    "FinalSO3CouplingPath",
    "InvariantSubspaceDecomposition",
    "YoungBlockDecomposition",
    "YoungResolvedSO3CouplingPath",
    "YoungResolvedSubspaceDecomposition",
    "channels_from_nl",
    "young_subgroup_blocks",
    "young_subgroup_order",
    "young_subgroup_projector_prefactor",
    "symmetric_power_decomposition",
    "young_block_decomposition",
    "final_so3_coupling_multiplicity",
    "invariant_subspace_decomposition",
    "ace_invariant_subspace_decomposition",
    "young_resolved_subspace_decomposition",
    "symmetric_power_weight_counts",
    "symmetric_power_irrep_multiplicities",
    "couple_so3_multiplicity_distributions",
    "couple_block_irrep_multiplicities",
    "count_canonical_leaf_labelings",
    "iter_canonical_leaf_labelings",
    "reachable_total_angular_momenta",
    "validate_leaf_quantum_numbers",
    "validate_target_angular_momentum",
    "validate_tree_type",
    "BlockMultiplicityMetadata",
    "InternalAngularNodeMetadata",
    "TensorBasisTerm",
    "TensorBasisExpansionCache",
    "ReconstructionRecipe",
    "ExactLabelMetadata",
]
