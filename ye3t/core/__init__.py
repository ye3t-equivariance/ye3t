from .labels import *
from .spherical import *
from .tesseral import *
from .rotation import *
from .couplings import *
from .irreps import *


_E3_EXPORTS = {
    "ACEChannelBlockDiagonalization",
    "ACEChannelReduction",
    "NaiveCoupledTreeBlockDiagonalization",
    "NaiveCoupledTreeReduction",
    "PrimitiveQuotientDimensionSummary",
    "RepeatedChannelBlockDiagonalization",
    "RepeatedChannelReduction",
    "TensorProductReduction",
    "YE3TBlockDiagonalization",
    "YE3TChannelReduction",
    "YE3TSymbolicSummary",
    "ace_channel_reduction",
    "block_diag_dense",
    "block_diagonalize_ace_channel_D_wig",
    "block_diagonalize_ace_channel_from_leaf_D_wigs",
    "block_diagonalize_naive_coupled_tree_D_wig",
    "block_diagonalize_naive_coupled_tree_from_leaf_D_wigs",
    "block_diagonalize_repeated_channel_D_wig",
    "block_diagonalize_repeated_channel_from_leaf_D_wig",
    "block_diagonalize_ye3t_D_wig",
    "block_diagonalize_ye3t_from_leaf_D_wigs",
    "cg_tensor_product_reduction",
    "format_irreps",
    "irrep_direct_sum_D_wig",
    "naive_coupled_tree_reduction",
    "parse_irreps",
    "primitive_quotient_dimension_summary",
    "reduce_tensor_product_D_wig",
    "repeated_channel_reduction",
    "tensor_product_output_irreps",
    "torch_cg_tensor_product",
    "total_dim",
    "wigner_D_wig_from_matrix",
    "wigner_D_wig_z",
    "ye3t_channel_reduction",
    "ye3t_symbolic_summary",
}

_PRODUCT_ENGINE_EXPORTS = {
    "ExactFeatureSpace",
    "ExactProductColumnDescriptor",
    "ExactProductExpansionEngine",
    "ExactSparseColumn",
    "ExactSparseOperator",
    "IndependentDecomposableProductSubspace",
    "PrimitiveGeneratorReconstructionSummary",
    "PrimitiveQuotientSummary",
    "PrimitiveReconstructionBenchmark",
    "ProductExpansionResult",
}

_API_EXPORTS = {
    "YE3TAPI",
    "infer_repeated_channel_blocks",
    "make_pattern_spec",
}

_SPECS_EXPORTS = {
    "E3PatternSpec",
    "ExactPathCatalog",
    "RepeatedChannelBlockSpec",
    "RepresentationReductionResult",
}

_GRADED_ALGEBRA_EXPORTS = {
    "ExactSymbolicPrimitiveFilter",
    "GradedBasisRegistry",
    "HomogeneousSector",
    "PrimitiveDecompositionSummary",
    "SampledGeneratedSubspaceAnalyzer",
    "SampledSpanResult",
}

_LAZY_EXPORT_MODULES = {}
for _name in _E3_EXPORTS:
    _LAZY_EXPORT_MODULES[_name] = "ye3t.core.e3"
for _name in _PRODUCT_ENGINE_EXPORTS:
    _LAZY_EXPORT_MODULES[_name] = "ye3t.core.product_engine"
for _name in _API_EXPORTS:
    _LAZY_EXPORT_MODULES[_name] = "ye3t.core.api"
for _name in _SPECS_EXPORTS:
    _LAZY_EXPORT_MODULES[_name] = "ye3t.core.specs"
for _name in _GRADED_ALGEBRA_EXPORTS:
    _LAZY_EXPORT_MODULES[_name] = "ye3t.core.graded_algebra"
del _name


def __getattr__(name):
    module_name = _LAZY_EXPORT_MODULES.get(name)
    if module_name is not None:
        module = __import__(module_name, fromlist=[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
