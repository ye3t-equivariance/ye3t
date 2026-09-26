
from dataclasses import field

from .e3 import (
    RepeatedChannelReduction,
    TensorProductReduction,
    YE3TChannelReduction,
    YE3TSymbolicSummary,
)
from .irreps import IrrepTerm
from .product_engine import (
    ExactFeatureSpace,
    ExactProductColumnDescriptor,
    IndependentDecomposableProductSubspace,
    PrimitiveQuotientSummary,
)
from ye3t._record import recordclass


@recordclass(('eta', 'l', 'multiplicity'), frozen = True)
class RepeatedChannelBlockSpec:
    """One repeated-channel block in a ye3t input specification."""


@recordclass(('nin', 'lin', 'blocks'), frozen = True)
class E3PatternSpec:
    r"""Canonical ye3t input pattern.

    This is the E(3)/SO(3) analogue of the repeated-channel block pattern
    :math:`\boldsymbol{\nu}=\{(\eta_b,l_b)^{k_b}\}_{b=1}^B`.
    """
    blocks = tuple()

    @property
    def rank(self):
        return len(self.lin)


@recordclass(('target', 'subspace', 'primitive', 'descriptors', 'path_count_by_signature'), frozen = True)
class ExactPathCatalog:
    """Exact symbolic coupling catalog for one target E(3) sector."""


@recordclass(('repeated_channel', 'tensor_product', 'ye3t', 'ye3t_symbolic', 'irreps_out', 'metadata'), frozen = True)
class RepresentationReductionResult:
    """Unified reduction result for the generic E(3) representation API."""

    repeated_channel = None
    tensor_product = None
    ye3t = None
    ye3t_symbolic = None
    irreps_out = tuple()
    metadata = field(default_factory=dict)
