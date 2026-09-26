
from collections import Counter

from .e3 import (
    cg_tensor_product_reduction,
    repeated_channel_reduction,
    tensor_product_output_irreps,
    ye3t_channel_reduction,
    ye3t_symbolic_summary,
)
from .irreps import IrrepTerm
from .labels import normalize_compact_label
from .product_engine import ExactProductExpansionEngine
from .basis import YE3TBasisLabeler

from .specs import E3PatternSpec, ExactPathCatalog, RepeatedChannelBlockSpec, RepresentationReductionResult


def infer_repeated_channel_blocks(nin, lin):
    """Infer repeated-channel blocks from exact ACE/E3 leaf labels."""
    counts = Counter((int(n), int(l)) for n, l in zip(nin, lin))
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    return tuple(
        RepeatedChannelBlockSpec(eta=int(eta), l=int(l), multiplicity=int(mult))
        for (eta, l), mult in ordered
    )


def make_pattern_spec(nin, lin):
    return E3PatternSpec(tuple(int(x) for x in nin), tuple(int(x) for x in lin), infer_repeated_channel_blocks(nin, lin))


class YE3TAPI:
    """Public ye3t representation-reduction and exact-basis API."""

    def __init__(self, tree_type = 'balanced', block_basis_mode = 'independent'):
        self.tree_type = str(tree_type)
        self.block_basis_mode = str(block_basis_mode).strip().lower()
        if self.block_basis_mode not in {"independent", "orthogonal"}:
            raise ValueError(
                f"Unknown block_basis_mode '{block_basis_mode}'. "
                "Use 'independent' or 'orthogonal'."
            )
        self.engine = ExactProductExpansionEngine(tree_type=self.tree_type)

    def reduce_repeated_channel(self, l, multiplicity):
        red = repeated_channel_reduction(int(l), int(multiplicity), tree_type=self.tree_type)
        return RepresentationReductionResult(repeated_channel=red, irreps_out=tuple(red.irreps_out))

    def reduce_tensor_product(self, L_left, L_right):
        red = cg_tensor_product_reduction(int(L_left), int(L_right))
        irreps_out = tuple(IrrepTerm(1, int(L), 'e') for L in red.output_Ls)
        return RepresentationReductionResult(tensor_product=red, irreps_out=irreps_out)

    def tensor_product_output_irreps(self, left_terms, right_terms):
        return tuple(tensor_product_output_irreps(left_terms, right_terms))

    def reduce_ye3t(self, nin, lin):
        red = ye3t_channel_reduction(tuple(int(x) for x in nin), tuple(int(x) for x in lin), tree_type=self.tree_type)
        return RepresentationReductionResult(ye3t=red, irreps_out=tuple(red.irreps_out))

    def summarize_ye3t(self, nin, lin):
        red = ye3t_symbolic_summary(tuple(int(x) for x in nin), tuple(int(x) for x in lin), tree_type=self.tree_type)
        return RepresentationReductionResult(ye3t_symbolic=red, irreps_out=tuple(red.irreps_out))

    def allowed_target_Ls(
        self,
        nin,
        lin,
        *,
        max_target_L = None,
    ):
        """Return all nonzero exact target SO(3) irreps for one E(3) pattern.

        This uses the symbolic Schur-Weyl/Young backend once for the whole
        pattern, so requesting ``target_L='all'`` does not require probing every
        angular momentum separately.
        """
        values = tuple(int(L) for L in self.engine.available_L_for_pattern(tuple(nin), tuple(lin)))
        if max_target_L is not None:
            cap = int(max_target_L)
            values = tuple(L for L in values if L <= cap)
        return values

    def labels_by_target_L(
        self,
        nin,
        lin,
        *,
        max_target_L = None,
    ):
        """Return compact exact basis labels keyed by allowed target irrep ``L_R``.

        This uses the fast compact tuple enumeration path and builds all allowed
        target sectors together before slicing, so ``target_L='all'``-style
        callers share the Schur-Weyl / block-coupling work across sectors.
        """
        labeler = YE3TBasisLabeler(
            list(nin),
            list(lin),
            strict_target_validation=False,
            tree_type=self.tree_type,
            block_basis_mode=self.block_basis_mode,
        )
        return {
            int(L): tuple(normalize_compact_label(label) for label in labels)
            for L, labels in labeler.compact_labels_for_targets("all", max_target_L=max_target_L).items()
        }

    def counts_by_target_L(
        self,
        nin,
        lin,
        *,
        max_target_L = None,
    ):
        """Return exact multiplicities ``alpha_{L_R}`` without materializing labels."""
        labeler = YE3TBasisLabeler(
            list(nin),
            list(lin),
            strict_target_validation=False,
            tree_type=self.tree_type,
            block_basis_mode=self.block_basis_mode,
        )
        counts = {int(L): int(count) for L, count in labeler.counts_by_L().items() if int(count) > 0}
        if max_target_L is not None:
            cap = int(max_target_L)
            counts = {int(L): int(count) for L, count in counts.items() if int(L) <= cap}
        return counts

    def compact_labels_by_target_L(
        self,
        nin,
        lin,
        *,
        max_target_L = None,
    ):
        """Explicit alias for the compact all-sector label path."""
        return self.labels_by_target_L(nin, lin, max_target_L=max_target_L)

    def structured_labels_by_target_L(
        self,
        nin,
        lin,
        *,
        max_target_L = None,
    ):
        """Return structured representation-theory labels keyed by target ``L_R``."""
        labeler = YE3TBasisLabeler(
            list(nin),
            list(lin),
            strict_target_validation=False,
            tree_type=self.tree_type,
            block_basis_mode=self.block_basis_mode,
        )
        return {
            int(L): tuple(labels)
            for L, labels in labeler.structured_labels_for_targets("all", max_target_L=max_target_L).items()
        }

    def build_exact_path_catalog(
        self,
        nin,
        lin,
        L_R,
        *,
        factorization_policy = 'full',
        include_target_primitive = False,
        primitive_first = False,
        generator_ranks = None,
        generator_Ls = None,
        max_generator_rank = None,
        max_generator_L = None,
        max_recoupling_L = None,
    ):
        target = self.engine.feature_space(tuple(nin), tuple(lin), int(L_R))
        subspace = self.engine.independent_decomposable_product_subspace(
            tuple(nin),
            tuple(lin),
            int(L_R),
            factorization_policy=factorization_policy,
            generator_ranks=generator_ranks,
            generator_Ls=generator_Ls,
            max_generator_rank=max_generator_rank,
            max_generator_L=max_generator_L,
            max_recoupling_L=max_recoupling_L,
            include_target_primitive=include_target_primitive or primitive_first,
        )
        primitive = self.engine.primitive_quotient(tuple(nin), tuple(lin), int(L_R), mode=factorization_policy)
        descriptors = tuple(subspace.independent_product_descriptors if subspace is not None else tuple())
        signature_counts = Counter()
        for desc in descriptors:
            sig = _descriptor_signature(desc)
            if sig is not None:
                signature_counts[sig] += 1
        return ExactPathCatalog(
            target=target,
            subspace=subspace,
            primitive=primitive,
            descriptors=descriptors,
            path_count_by_signature=dict(signature_counts),
        )


def _descriptor_signature(desc):
    if getattr(desc, 'kind', None) != 'product' or desc.left is None or desc.right is None:
        return None
    return (int(desc.left.L_R), int(desc.right.L_R), int(desc.L_R))


__all__ = [
    'RepeatedChannelBlockSpec',
    'E3PatternSpec',
    'ExactPathCatalog',
    'RepresentationReductionResult',
    'YE3TAPI',
    'infer_repeated_channel_blocks',
    'make_pattern_spec',
]
