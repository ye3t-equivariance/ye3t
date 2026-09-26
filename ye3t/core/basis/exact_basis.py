"""Exact-basis import surface for the constructive Schur-Weyl + Young path.

Importing from this module gives access to the runtime exact-basis machinery
without pulling in the optional Gram/SVD benchmark helpers.

Recommended entry points:
- ``YoungSymmetrizerBackend`` for the constructive label builder
- ``YE3TBasisLabeler`` for the public-facing exact labeler interface
- theory helpers such as ``ace_invariant_subspace_decomposition`` when you want
  the paper-facing decomposition objects
"""

from .builder import BalancedPairwiseACELabeler, ExactACELabeler, YE3TBasisLabeler
from .homogeneous import AlgebraicSymmetricPowerDecomposer, HomogeneousRepresentativeGenerator
from .metadata import (
    BlockMultiplicityMetadata,
    ExactLabelMetadata,
    InternalAngularNodeMetadata,
    ReconstructionRecipe,
    TensorBasisExpansionCache,
    TensorBasisTerm,
)
from .sector import ExactBasisEntry, ExactBasisHandle, ExactBasisSector, ExactSectorSignature
from .theory import (
    FinalSO3CouplingPath,
    InvariantSubspaceDecomposition,
    SymmetricPowerDecomposition,
    TensorProductChannel,
    YoungSubgroupBlock,
    ace_invariant_subspace_decomposition,
    channels_from_nl,
    final_so3_coupling_multiplicity,
    invariant_subspace_decomposition,
    symmetric_power_decomposition,
    young_subgroup_blocks,
    young_subgroup_order,
    young_subgroup_projector_prefactor,
)
from .validation import (
    reachable_total_angular_momenta,
    validate_leaf_quantum_numbers,
    validate_target_angular_momentum,
    validate_tree_type,
)
from .young_exact import YoungSymmetrizerBackend

__all__ = [
    "BalancedPairwiseACELabeler",
    "ExactACELabeler",
    "YE3TBasisLabeler",
    "YoungSymmetrizerBackend",
    "AlgebraicSymmetricPowerDecomposer",
    "HomogeneousRepresentativeGenerator",
    "ExactSectorSignature",
    "ExactBasisHandle",
    "ExactBasisEntry",
    "ExactBasisSector",
    "TensorProductChannel",
    "BlockMultiplicityMetadata",
    "InternalAngularNodeMetadata",
    "TensorBasisTerm",
    "TensorBasisExpansionCache",
    "ReconstructionRecipe",
    "ExactLabelMetadata",
    "YoungSubgroupBlock",
    "SymmetricPowerDecomposition",
    "FinalSO3CouplingPath",
    "InvariantSubspaceDecomposition",
    "channels_from_nl",
    "young_subgroup_blocks",
    "young_subgroup_order",
    "young_subgroup_projector_prefactor",
    "symmetric_power_decomposition",
    "final_so3_coupling_multiplicity",
    "invariant_subspace_decomposition",
    "ace_invariant_subspace_decomposition",
    "reachable_total_angular_momenta",
    "validate_leaf_quantum_numbers",
    "validate_target_angular_momentum",
    "validate_tree_type",
]
