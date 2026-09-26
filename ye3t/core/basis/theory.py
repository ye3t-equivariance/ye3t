
r"""Theory-facing helpers for the exact ACE representation decomposition.

The functions in this module are named to mirror the representation-theory
equations used in the paper:

1. partition the leaf channels into Young-subgroup blocks ``(eta, l)_b^{k_b}``
2. form the invariant subspace ``H^G`` by blockwise symmetric powers
3. decompose each block
   ``Sym^{k_b}(V_{l_b}) = direct_sum_{Lambda_b} d_{Lambda_b} V_{Lambda_b}``
4. couple the block irreps to final angular momentum ``L_R``
5. collect the multiplicity-space dimensions ``alpha_{L_R}``

In the current exact ACE labeler the non-angular channel index ``eta`` is the
leaf-channel label carried by ``nin``. Chemistry and auxiliary scalar/vector
channels are layered later in `equivariant_calc`.

This module is split into two conceptual parts:

1. exact theory objects/functions used by the constructive block-first basis
   builder in [young_exact.py]
2. optional validation helpers, such as ``TheoryConsistencyReport``, that are
   useful when checking the implementation against independent counting routes

Nothing in this file performs numerical projection, SVD, or character
integration. The validation helpers here are still exact and symbolic/algebraic;
they are separated only by *purpose*, not by numerical method.
"""

from collections import OrderedDict
from functools import lru_cache
from itertools import product
from math import factorial

from .characters import cg_allowed
from .homogeneous import AlgebraicSymmetricPowerDecomposer
from .exhaustive_enumeration import integer_partitions
from .multiplicity import couple_block_irrep_multiplicities, so3_irrep_multiplicities_from_weight_counts
from ye3t._record import recordclass


_EXPLICIT_COUPLING_PATH_PRODUCT_LIMIT = 100000
_EXPLICIT_COUPLING_PATH_WORK_LIMIT = 10000000


@recordclass(('eta', 'l'), frozen = True)
class TensorProductChannel:
    """One tensor-product factor ``V_l^(eta)``.

    In the paper notation this is a single leaf space carrying:
    - angular momentum ``l``
    - a non-angular channel label ``eta``

    In the current ACE implementation ``eta`` is built from the leaf radial
    label ``n``. This object is part of the exact constructive path.
    """

    @property
    def channel_key(self):
        return (self.eta, self.l)


@recordclass(('eta', 'l', 'multiplicity'), frozen = True)
class YoungSubgroupBlock:
    r"""Repeated-channel block ``(\eta, l)_b^{k_b}``.

    A block groups together identical tensor-product factors. For example, if
    the same ``(n,l)`` pair appears three times among the leaves, those three
    copies become one block with ``multiplicity = 3``.

    This block structure is the starting point for the exact Schur-Weyl +
    Young-subgroup construction, because permutation symmetry is enforced within
    each block before the final SO(3) coupling step.
    """

    @property
    def block_key(self):
        return (self.eta, self.l)

    @property
    def symmetric_power_symbol(self):
        return f"Sym^{self.multiplicity}(V_{self.l})"


@recordclass(('block', 'd_by_Lambda'), frozen = True)
class SymmetricPowerDecomposition:
    r"""Exact decomposition of one repeated block into irreducible SO(3) pieces.

    For a repeated block ``(\eta, l)_b^{k_b}``, the Young-subgroup-invariant
    subspace is the symmetric power ``Sym^{k_b}(V_{l_b})``. This object stores
    the multiplicities ``d_{\Lambda_b}`` in

    ``Sym^{k_b}(V_{l_b}) = direct_sum_{\Lambda_b} d_{\Lambda_b} V_{\Lambda_b}``

    In plain terms:
    - ``Lambda_b`` are the allowed intermediate angular momenta of that block
    - ``d_by_Lambda[Lambda_b]`` tells you how many independent symmetry-adapted
      ways that block can transform like angular momentum ``Lambda_b``

    This decomposition is exact and is used directly by the constructive basis
    builder in [young_exact.py].
    """


@recordclass(('Lambda_tuple', 'block_multiplicity_product', 'coupling_multiplicity_by_L_R'), frozen = True)
class FinalSO3CouplingPath:
    r"""One block-level coupling channel contributing to the final ``L_R`` count.

    After each repeated block is reduced separately, the remaining task is to
    couple the block angular momenta ``{\Lambda_b}`` to final resultant angular
    momentum ``L_R``. This object records one such block-angular tuple together
    with:
    - the product of block multiplicities ``prod_b d_{\Lambda_b}``
    - the final SO(3) coupling multiplicities for each reachable ``L_R``

    This is part of the exact theory decomposition. It is also useful for
    understanding how the paper's intermediate labels contribute to the final
    multiplicity-space dimension ``alpha_{L_R}``.
    """


@recordclass(('block', 'kappa', 'd_by_Lambda'), frozen = True)
class YoungBlockDecomposition:
    r"""Exact decomposition of one repeated block for a selected ``\kappa_b``.

    This generalizes the ACE symmetric-power block decomposition from
    ``\kappa_b=(k_b)`` to an arbitrary Young-subgroup block irrep
    ``\kappa_b``.  The values ``d_by_Lambda`` are the exact multiplicities in
    the corresponding Schur functor of ``V_{l_b}``.
    """


@recordclass(('Lambda_tuple', 'd_blocks', 'block_multiplicity_product', 'coupling_multiplicity_by_L_R'), frozen = True)
class YoungResolvedSO3CouplingPath:
    r"""One fully resolved ``(\boldsymbol\kappa,\boldsymbol\Lambda)`` path."""


@recordclass(
    (
        'channels',
        'blocks',
        'kappa_tuple',
        'parent_lambda',
        'c_kappa_lambda',
        'block_decompositions',
        'coupling_paths',
        'alpha_by_L_R',
    ),
    frozen = True,
)
class YoungResolvedSubspaceDecomposition:
    r"""Count-only decomposition resolved by ``\boldsymbol\kappa`` and ``\boldsymbol\Lambda``."""


@recordclass(('channels', 'blocks', 'symmetric_power_decompositions', 'coupling_paths', 'alpha_by_L_R'), frozen = True)
class InvariantSubspaceDecomposition:
    """Exact block-first decomposition of the Young-subgroup invariant space.

    This is the theory-level object corresponding to

    ``H^G = direct_sum_{L_R} C^{alpha_{L_R}} tensor V_{L_R}``

    It separates:
    - the multiplicity-space dimension ``alpha_{L_R}``
    - the carrier irrep ``V_{L_R}``

    The constructive backend in [young_exact.py] uses the block information in
    this object to build a concrete canonical basis, rather than only counting
    dimensions.

    Very large count-only decompositions may leave ``coupling_paths`` empty.
    In that case ``alpha_by_L_R`` is still exact and is computed from the same
    blockwise symmetric-power multiplicity formula, but explicit block-angular
    path records are intentionally not materialized.
    """

    def multiplicity_space_dimension(self, L_R):
        return int(self.alpha_by_L_R.get(int(L_R), 0))


@recordclass(('backend_counts_by_L', 'theory_alpha_by_L_R', 'difference_by_L_R', 'consistent'), frozen = True)
class TheoryConsistencyReport:
    """Validation-only comparison between implementation counts and theory counts.

    This object is not part of the runtime exact-basis construction. It is used
    when auditing whether a given backend reproduces the tree-independent theory
    multiplicities ``alpha_{L_R}``.
    """


def channels_from_nl(nin, lin):
    r"""Convert ACE leaf labels into theory channels.

    This is the bridge from implementation data to paper notation:
    each ACE leaf pair ``(n_i, l_i)`` is reinterpreted as a channel
    ``V_{l_i}^{(\eta_i)}`` with ``eta_i = n_i``.
    """
    if len(nin) != len(lin):
        raise ValueError("nin and lin must have the same length")
    return tuple(TensorProductChannel(eta=int(n), l=int(l)) for n, l in zip(nin, lin))


def young_subgroup_blocks(channels):
    """Group identical channels into Young-subgroup blocks.

    The Young subgroup acts by permuting identical channels within each block.
    This function therefore finds the repeated-channel structure needed to write

    ``G = prod_b S_{k_b}``

    and to replace the raw tensor product with blockwise symmetric powers.
    """
    counts = OrderedDict()
    for channel in channels:
        counts[channel.channel_key] = counts.get(channel.channel_key, 0) + 1
    return tuple(
        YoungSubgroupBlock(eta=eta, l=int(l), multiplicity=int(k_b))
        for (eta, l), k_b in counts.items()
    )


def young_subgroup_order(blocks):
    """Return the order ``|G|`` of the Young subgroup.

    This is mainly a theory-facing helper: it gives the normalization factor
    needed in the projector ``P_G = (1/|G|) sum_{sigma in G} sigma``.
    """
    order = 1
    for block in blocks:
        order *= factorial(int(block.multiplicity))
    return order


def young_subgroup_projector_prefactor(blocks):
    """Return the scalar prefactor ``1 / |G|`` for the Young-subgroup projector."""
    return 1.0 / float(max(young_subgroup_order(blocks), 1))


def symmetric_power_decomposition(block):
    r"""Decompose one repeated block ``Sym^{k_b}(V_{l_b})`` exactly.

    Mathematically, this computes the multiplicities ``d_{\Lambda_b}`` in

    ``Sym^{k_b}(V_{l_b}) = direct_sum_{\Lambda_b} d_{\Lambda_b} V_{\Lambda_b}``

    Conceptually, this answers the question:
    "If I take ``k_b`` identical neighbors/channels with angular momentum
    ``l_b`` and enforce permutation symmetry among them, what angular momenta
    can that whole block carry, and how many linearly independent symmetry-
    adapted block states exist for each one-"

    In the code, those multiplicities are supplied by the algebraic symmetric-
    power decomposer in [homogeneous.py]. The result is part of the exact basis
    construction, not a numerical validation path.
    """
    d_by_Lambda = AlgebraicSymmetricPowerDecomposer.decompose(
        l=int(block.l),
        k_b=int(block.multiplicity),
    )
    return SymmetricPowerDecomposition(block=block, d_by_Lambda=dict(d_by_Lambda))


def _normalize_partition_parts(parts, size):
    parts = tuple(int(part) for part in parts)
    if not parts:
        raise ValueError("partition must not be empty")
    if any(part <= 0 for part in parts):
        raise ValueError(f"partition parts must be positive, got {parts!r}")
    if any(left < right for left, right in zip(parts, parts[1:])):
        raise ValueError(f"partition parts must be weakly decreasing, got {parts!r}")
    if sum(parts) != int(size):
        raise ValueError(f"partition {parts!r} has size {sum(parts)}, expected {int(size)}")
    return parts


def _normalize_parent_lambda(parent_lambda, rank):
    if parent_lambda is None:
        return (int(rank),)
    if isinstance(parent_lambda, str):
        text = parent_lambda.strip().lower()
        if text in {"trivial", "symmetric", "lambda=(n)", "(n)"}:
            return (int(rank),)
        if text in {"sign", "antisymmetric", "lambda=(1^n)", "(1^n)"}:
            return tuple(1 for _ in range(int(rank)))
    return _normalize_partition_parts(parent_lambda, int(rank))


def _normalize_kappa_tuple(kappa_tuple, blocks):
    if kappa_tuple is None:
        return None
    kappa_tuple = tuple(kappa_tuple)
    if len(kappa_tuple) != len(blocks):
        raise ValueError("kappa_tuple must contain one partition per repeated-channel block")
    return tuple(
        _normalize_partition_parts(parts, int(block.multiplicity))
        for parts, block in zip(kappa_tuple, blocks, strict=False)
    )


def _kappa_choices_for_blocks(blocks):
    choices = []
    for block in blocks:
        choices.append(tuple(tuple(int(part) for part in parts) for parts in integer_partitions(int(block.multiplicity))))
    out = [tuple()]
    for block_choices in choices:
        out = [prefix + (choice,) for prefix in out for choice in block_choices]
    return tuple(out)


def _block_irrep_multiplicities_for_kappa(block, kappa):
    if int(block.l) == 0:
        # V_0 is one-dimensional. Its tensor powers carry only the trivial
        # S_k action and one scalar SO(3) copy.
        if tuple(int(part) for part in kappa) == (
            int(block.multiplicity),
        ):
            return {0: 1}
        return {}

    from ye3t.representations.young_sectors import _projected_block_weight_counts_cached

    weight_counts = dict(
        _projected_block_weight_counts_cached(
            int(block.multiplicity),
            int(block.l),
            tuple(int(part) for part in kappa),
        )
    )
    return so3_irrep_multiplicities_from_weight_counts(
        weight_counts,
        Lmax=int(block.l) * int(block.multiplicity),
    )


def young_block_decomposition(block, kappa):
    r"""Decompose one repeated block for a selected Young-subgroup irrep."""
    kappa = _normalize_partition_parts(kappa, int(block.multiplicity))
    d_by_Lambda = _block_irrep_multiplicities_for_kappa(block, kappa)
    return YoungBlockDecomposition(
        block=block,
        kappa=kappa,
        d_by_Lambda=dict(d_by_Lambda),
    )


def young_resolved_subspace_decomposition(
    nin,
    lin,
    parent_lambda = None,
    kappa_tuple = None,
):
    r"""Return exact counts resolved by ``\boldsymbol\kappa`` and ``\boldsymbol\Lambda``.

    This is the count-only paper decomposition

    ``c_{\boldsymbol\kappa}^{\lambda}
      d_{\boldsymbol\kappa\boldsymbol\Lambda}
      M_{\boldsymbol\Lambda}^{L}``

    for one fixed parent Young sector ``lambda`` and one selected
    ``boldsymbol{kappa}``.  If ``kappa_tuple`` is omitted, every Young-subgroup
    irrep with nonzero branching multiplicity into ``parent_lambda`` is
    returned as one decomposition record.
    """
    channels = channels_from_nl(nin, lin)
    blocks = young_subgroup_blocks(channels)
    parent = _normalize_parent_lambda(parent_lambda, len(tuple(nin)))
    selected_kappas = _normalize_kappa_tuple(kappa_tuple, blocks)
    kappa_choices = (selected_kappas,) if selected_kappas is not None else _kappa_choices_for_blocks(blocks)
    records = []
    from ye3t.representations.young_subgroup_specht_coupling import young_subgroup_specht_coupling_multiplicity

    for kappa_choice in kappa_choices:
        c_value = int(young_subgroup_specht_coupling_multiplicity(kappa_choice, parent))
        if c_value <= 0:
            continue
        block_decompositions = tuple(
            young_block_decomposition(block, kappa)
            for block, kappa in zip(blocks, kappa_choice, strict=False)
        )
        lambda_choices = tuple(tuple(sorted(decomp.d_by_Lambda)) for decomp in block_decompositions)
        alpha_by_L_R = {}
        coupling_paths = []
        if _should_use_count_only_final_coupling(lambda_choices):
            block_maps = tuple(decomp.d_by_Lambda for decomp in block_decompositions)
            alpha_by_L_R = {
                int(L_R): int(c_value) * int(count)
                for L_R, count in couple_block_irrep_multiplicities(block_maps).items()
            }
        else:
            if not lambda_choices:
                lambda_products = [tuple()]
            else:
                lambda_products = product(*lambda_choices)
            for lambda_tuple in lambda_products:
                d_blocks = []
                block_product = 1
                for decomp, Lambda_b in zip(block_decompositions, lambda_tuple, strict=False):
                    value = int(decomp.d_by_Lambda[int(Lambda_b)])
                    d_blocks.append(value)
                    block_product *= value
                coupling_mult = final_so3_coupling_multiplicity(lambda_tuple)
                for L_R, count in coupling_mult.items():
                    alpha_by_L_R[int(L_R)] = (
                        int(alpha_by_L_R.get(int(L_R), 0))
                        + int(c_value) * int(block_product) * int(count)
                    )
                coupling_paths.append(
                    YoungResolvedSO3CouplingPath(
                        Lambda_tuple=tuple(int(value) for value in lambda_tuple),
                        d_blocks=tuple(int(value) for value in d_blocks),
                        block_multiplicity_product=int(block_product),
                        coupling_multiplicity_by_L_R=dict(coupling_mult),
                    )
                )
        records.append(
            YoungResolvedSubspaceDecomposition(
                channels=tuple(channels),
                blocks=tuple(blocks),
                kappa_tuple=tuple(kappa_choice),
                parent_lambda=tuple(parent),
                c_kappa_lambda=int(c_value),
                block_decompositions=block_decompositions,
                coupling_paths=tuple(coupling_paths),
                alpha_by_L_R=dict(sorted(alpha_by_L_R.items())),
            )
        )
    return tuple(records)


def final_so3_coupling_multiplicity(Lambda_tuple):
    r"""Compute final SO(3) multiplicities for a fixed tuple of block angular momenta.

    Given one choice of intermediate block angular momenta ``{\Lambda_b}``, this
    function recursively applies Clebsch-Gordan coupling to determine how many
    times each final ``L_R`` appears.

    In the paper's notation, this computes
    ``M_{ {\Lambda_b} }^{L_R}``.
    """
    current = {0: 1}
    for Lambda in Lambda_tuple:
        next_counts = {}
        for left_L, left_mult in current.items():
            for out_L in cg_allowed(int(left_L), int(Lambda)):
                next_counts[out_L] = next_counts.get(out_L, 0) + int(left_mult)
        current = next_counts
    return current


def _lambda_choices_for_decompositions(block_decompositions):
    return tuple(tuple(sorted(decomp.d_by_Lambda)) for decomp in block_decompositions)


def _lambda_path_product_size(lambda_choices):
    if not lambda_choices:
        return 1
    total = 1
    for choices in lambda_choices:
        total *= max(1, len(choices))
    return int(total)


def _lambda_path_work_estimate(lambda_choices):
    if not lambda_choices:
        return 1
    product_size = _lambda_path_product_size(lambda_choices)
    largest_total = 0
    for choices in lambda_choices:
        if choices:
            largest_total += max(int(value) for value in choices)
    return int(product_size * max(1, largest_total + 1))


def _should_use_count_only_final_coupling(lambda_choices):
    return (
        _lambda_path_product_size(lambda_choices) > _EXPLICIT_COUPLING_PATH_PRODUCT_LIMIT
        or _lambda_path_work_estimate(lambda_choices) > _EXPLICIT_COUPLING_PATH_WORK_LIMIT
    )


def _count_alpha_by_L_R_from_block_decompositions(block_decompositions):
    return couple_block_irrep_multiplicities(tuple(decomp.d_by_Lambda for decomp in block_decompositions))


def invariant_subspace_decomposition(
    channels,
):
    r"""Assemble the exact theory decomposition of the invariant subspace ``H^G``.

    This function performs the block-first logic described in the paper:
    1. identify repeated-channel blocks
    2. decompose each block as a symmetric power
    3. enumerate the allowed block-angular tuples ``{\Lambda_b}``
    4. couple those block irreps to final ``L_R``
    5. sum contributions into ``alpha_{L_R}``

    The output is a theory object. The constructive basis builder uses this
    structure to decide what needs to be represented explicitly.
    """
    blocks = young_subgroup_blocks(channels)
    block_decompositions = tuple(symmetric_power_decomposition(block) for block in blocks)
    coupling_paths = []
    alpha_by_L_R = _count_alpha_by_L_R_from_block_decompositions(block_decompositions)

    # Each lambda_tuple chooses one allowed angular momentum from every block.
    lambda_choices = _lambda_choices_for_decompositions(block_decompositions)
    if _should_use_count_only_final_coupling(lambda_choices):
        return InvariantSubspaceDecomposition(
            channels=tuple(channels),
            blocks=tuple(blocks),
            symmetric_power_decompositions=block_decompositions,
            coupling_paths=tuple(),
            alpha_by_L_R=alpha_by_L_R,
        )

    if not lambda_choices:
        lambda_products = [tuple()]
    else:
        lambda_products = product(*lambda_choices)

    for lambda_tuple in lambda_products:
        # The block multiplicity spaces contribute multiplicatively before the
        # final SO(3) reduction across blocks.
        block_multiplicity_product = 1
        for decomp, lambda_b in zip(block_decompositions, lambda_tuple, strict=False):
            block_multiplicity_product *= int(decomp.d_by_Lambda[int(lambda_b)])
        coupling_mult = final_so3_coupling_multiplicity(lambda_tuple)
        coupling_paths.append(
            FinalSO3CouplingPath(
                Lambda_tuple=tuple(int(v) for v in lambda_tuple),
                block_multiplicity_product=int(block_multiplicity_product),
                coupling_multiplicity_by_L_R=dict(coupling_mult),
            )
        )

    return InvariantSubspaceDecomposition(
        channels=tuple(channels),
        blocks=tuple(blocks),
        symmetric_power_decompositions=block_decompositions,
        coupling_paths=tuple(coupling_paths),
        alpha_by_L_R=dict(sorted(alpha_by_L_R.items())),
    )


@lru_cache(maxsize=None)
def _ace_invariant_subspace_decomposition_cached(
    nin,
    lin,
):
    return invariant_subspace_decomposition(channels_from_nl(nin, lin))


def ace_invariant_subspace_decomposition(
    nin,
    lin,
):
    """Convenience wrapper for ACE leaf labels.

    This is the easiest entry point if you want the theory decomposition for a
    specific ACE leaf pattern without manually constructing channel objects.
    """
    return _ace_invariant_subspace_decomposition_cached(
        tuple(int(x) for x in nin),
        tuple(int(x) for x in lin),
    )


def compare_backend_counts_to_theory(
    backend_counts_by_L,
    nin,
    lin,
):
    """Validation-only check against the exact block-theory counts.

    This helper compares externally supplied backend counts to the exact,
    tree-independent theory multiplicities ``alpha_{L_R}``.

    It is meant for auditing and regression testing. It is not needed to use the
    constructive exact basis itself.
    """
    decomposition = ace_invariant_subspace_decomposition(nin, lin)
    support = sorted(set(backend_counts_by_L) | set(decomposition.alpha_by_L_R))
    difference = {
        int(L_R): int(backend_counts_by_L.get(int(L_R), 0)) - int(decomposition.alpha_by_L_R.get(int(L_R), 0))
        for L_R in support
    }
    consistent = all(delta == 0 for delta in difference.values())
    return TheoryConsistencyReport(
        backend_counts_by_L=dict(sorted(backend_counts_by_L.items())),
        theory_alpha_by_L_R=dict(decomposition.alpha_by_L_R),
        difference_by_L_R=difference,
        consistent=consistent,
    )
