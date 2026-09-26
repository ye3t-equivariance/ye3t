
"""Exact ACE basis from block decomposition followed by tree coupling.

This backend follows the paper's representation-theory construction directly:

1. partition the leaf channels into repeated-channel blocks ``(eta, l)_b^{k_b}``
2. decompose each block exactly as ``Sym^{k_b}(V_{l_b})``
3. choose canonical multiplicity-space representatives inside each block
4. couple the block irreps through a selected recursive full binary tree

The key theoretical idea is:
- first apply Schur-Weyl/Young-subgroup reasoning to remove permutation
  redundancy within repeated-channel blocks
- only then perform the final SO(3) coupling between block irreps

This is different from a naive "couple everything first, then clean up"
workflow. Here, the symmetry reduction is built into the representation spaces
before the final tree coupling step.

No floating-point orthogonalization or SVD is used in the basis construction.
The only numerical route retained elsewhere is the optional Gram comparison.
"""

from collections import OrderedDict
import os

from .homogeneous import HomogeneousRepresentativeGenerator
from .labels import LeafLabel, NodeLabel, SymBlockLabel
from .metadata import (
    BlockMultiplicityMetadata,
    ExactLabelMetadata,
    InternalAngularNodeMetadata,
    ReconstructionRecipe,
)
from .sector import ExactBasisEntry, ExactBasisHandle, ExactBasisSector, ExactSectorSignature
from .theory import ace_invariant_subspace_decomposition
from .tree import HomShape, get_tree_factory
from .characters import cg_allowed
from .validation import canonicalize_leaf_quantum_numbers
from ye3t._record import recordclass


def _env_flag(name, legacy_name = None):
    raw = os.getenv(name)
    if raw is None and legacy_name is not None:
        raw = os.getenv(legacy_name)
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name, default, legacy_name = None):
    if _env_flag("YE3T_DISABLE_CACHES", "gne3_DISABLE_CACHES"):
        return 0
    raw = os.getenv(name)
    if raw is None and legacy_name is not None:
        raw = os.getenv(legacy_name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


_LABEL_TEMPLATE_CACHE_MAX_ENTRIES = _env_int("YE3T_LABEL_TEMPLATE_CACHE_MAX_ENTRIES", 16, "gne3_LABEL_TEMPLATE_CACHE_MAX_ENTRIES")
_LABEL_TEMPLATE_CACHE_MAX_TOTAL_LABELS = _env_int("YE3T_LABEL_TEMPLATE_CACHE_MAX_TOTAL_LABELS", 300000, "gne3_LABEL_TEMPLATE_CACHE_MAX_TOTAL_LABELS")
_LABEL_TEMPLATE_CACHE_MAX_ENTRY_LABELS = _env_int("YE3T_LABEL_TEMPLATE_CACHE_MAX_ENTRY_LABELS", 250000, "gne3_LABEL_TEMPLATE_CACHE_MAX_ENTRY_LABELS")
_LABEL_SECTOR_CACHE_MAX_ENTRIES = _env_int("YE3T_LABEL_SECTOR_CACHE_MAX_ENTRIES", 8, "gne3_LABEL_SECTOR_CACHE_MAX_ENTRIES")
_LABEL_SECTOR_CACHE_MAX_TOTAL_LABELS = _env_int("YE3T_LABEL_SECTOR_CACHE_MAX_TOTAL_LABELS", 300000, "gne3_LABEL_SECTOR_CACHE_MAX_TOTAL_LABELS")
_LABEL_SECTOR_CACHE_MAX_ENTRY_LABELS = _env_int("YE3T_LABEL_SECTOR_CACHE_MAX_ENTRY_LABELS", 250000, "gne3_LABEL_SECTOR_CACHE_MAX_ENTRY_LABELS")


def _basis_template_signature(blocks, tree_type, block_basis_mode):
    return (
        str(tree_type),
        str(block_basis_mode),
        tuple((int(block.multiplicity), int(block.l)) for block in blocks),
    )


class _BoundedLabelTemplateCache:
    def __init__(self, max_entries, max_total_labels, max_entry_labels):
        self.max_entries = int(max_entries)
        self.max_total_labels = int(max_total_labels)
        self.max_entry_labels = int(max_entry_labels)
        self._items = OrderedDict()
        self._total_labels = 0

    def get(self, key):
        item = self._items.get(key)
        if item is None:
            return None
        self._items.move_to_end(key)
        return item[0]

    def put(self, key, value):
        if self.max_entries <= 0 or self.max_total_labels == 0:
            return
        label_count = sum(len(items) for items in value.values())
        if self.max_entry_labels >= 0 and label_count > self.max_entry_labels:
            return
        existing = self._items.pop(key, None)
        if existing is not None:
            self._total_labels -= existing[1]
        self._items[key] = (value, int(label_count))
        self._total_labels += int(label_count)
        self._evict()

    def _evict(self):
        while self._items:
            too_many_entries = self.max_entries >= 0 and len(self._items) > self.max_entries
            too_many_labels = self.max_total_labels >= 0 and self._total_labels > self.max_total_labels
            if not too_many_entries and not too_many_labels:
                break
            _, (_, label_count) = self._items.popitem(last=False)
            self._total_labels -= int(label_count)


_LABEL_TEMPLATE_CACHE = _BoundedLabelTemplateCache(
    _LABEL_TEMPLATE_CACHE_MAX_ENTRIES,
    _LABEL_TEMPLATE_CACHE_MAX_TOTAL_LABELS,
    _LABEL_TEMPLATE_CACHE_MAX_ENTRY_LABELS,
)


class _BoundedLabelSectorCache:
    def __init__(self, max_entries, max_total_labels, max_entry_labels):
        self.max_entries = int(max_entries)
        self.max_total_labels = int(max_total_labels)
        self.max_entry_labels = int(max_entry_labels)
        self._items = OrderedDict()
        self._total_labels = 0

    def get(self, key):
        item = self._items.get(key)
        if item is None:
            return None
        self._items.move_to_end(key)
        return item[0]

    def put(
        self,
        key,
        value,
    ):
        if self.max_entries <= 0 or self.max_total_labels == 0:
            return
        labels_cache, structured_cache = value
        label_count = max(
            sum(len(items) for items in labels_cache.values()),
            sum(len(items) for items in structured_cache.values()),
        )
        if self.max_entry_labels >= 0 and label_count > self.max_entry_labels:
            return
        existing = self._items.pop(key, None)
        if existing is not None:
            self._total_labels -= existing[1]
        self._items[key] = (value, int(label_count))
        self._total_labels += int(label_count)
        self._evict()

    def _evict(self):
        while self._items:
            too_many_entries = self.max_entries >= 0 and len(self._items) > self.max_entries
            too_many_labels = self.max_total_labels >= 0 and self._total_labels > self.max_total_labels
            if not too_many_entries and not too_many_labels:
                break
            _, (_, label_count) = self._items.popitem(last=False)
            self._total_labels -= int(label_count)


_LABEL_SECTOR_CACHE = _BoundedLabelSectorCache(
    _LABEL_SECTOR_CACHE_MAX_ENTRIES,
    _LABEL_SECTOR_CACHE_MAX_TOTAL_LABELS,
    _LABEL_SECTOR_CACHE_MAX_ENTRY_LABELS,
)


@recordclass(('compact',), frozen = True)
class CompactTupleLabel:
    """Small wrapper for a canonical compact ACE label tuple.

    The constructive backend builds labels internally using structured label
    objects (`LeafLabel`, `SymBlockLabel`, `NodeLabel`). This wrapper is the
    final lightweight public-facing object returned by `labels_by_L()` when the
    backend has finished constructing the basis.
    """

    def compact_label(self):
        return self.compact

    def pretty(self):
        return str(self.compact)

    def sort_key(self):
        return self.compact


class YoungSymmetrizerBackend:
    """Exact constructive backend matching the block-first theory derivation.

    This class is the runtime implementation of the paper's exact basis
    construction. Its job is not just to count dimensions, but to build a
    canonical set of compact ACE labels spanning the exact multiplicity space.

    The implementation mirrors the theory in four stages:

    1. sort and canonicalize the leaf pattern `(n_i, l_i)`
    2. apply Schur-Weyl/Young-subgroup decomposition blockwise
    3. choose exact representatives for each block multiplicity space
    4. couple those block irreps along the chosen recursive tree

    Unlike the Gram comparison helpers, this class is part of the
    actual exact ACE construction used by the runtime descriptor pipeline.
    """

    def __init__(
        self,
        nin,
        lin,
        qr_tol = 1e-10,
        tree_type = "balanced",
        block_basis_mode = "independent",
    ):
        del qr_tol  # accepted by the public constructor but unused by this backend
        # Canonical ordering makes the constructive basis independent of the raw
        # input ordering of identical leaf channels. The actual block structure
        # depends only on repeated `(n,l)` content, not on the original list order.
        self.nin, self.lin = canonicalize_leaf_quantum_numbers(nin, lin)
        self.tree_type = tree_type
        self.block_basis_mode = str(block_basis_mode).strip().lower()
        if self.block_basis_mode not in {"independent", "orthogonal"}:
            raise ValueError(
                f"Unknown block_basis_mode '{block_basis_mode}'. "
                "Use 'independent' or 'orthogonal'."
            )
        self.tree_factory = get_tree_factory(tree_type)
        self.hom_reps = HomogeneousRepresentativeGenerator()
        # This is the theory-level object
        #   H^G = direct_sum_{L_R} C^{alpha_{L_R}} tensor V_{L_R}
        # from which we extract repeated-channel blocks and exact block counts.
        self.decomposition = ace_invariant_subspace_decomposition(self.nin, self.lin)
        self.blocks = self.decomposition.blocks
        self._template_signature = _basis_template_signature(
            self.blocks,
            self.tree_type,
            self.block_basis_mode,
        )
        self._sector_signature = (
            self.tree_type,
            self.block_basis_mode,
            tuple(int(x) for x in self.nin),
            tuple(int(x) for x in self.lin),
        )
        # The final SO(3) coupling is done between blocks, not between raw leaves.
        self.block_tree_shape = self.tree_factory.build_hom_shape(len(self.blocks)) if self.blocks else None
        # Compact block labels are the fast-path representation used for tuple
        # enumeration. Rich structured block labels with occupancy expansions
        # are built separately and only when the analytical path needs them.
        self._block_label_maps = None
        self._structured_block_label_maps = None
        self._lightweight_structured_labels_cache = {}
        self._labels_cache = {}
        self._metadata_cache = {}
        self._structured_labels_cache = {}
        self._sector_data_cache = {}

    def _ensure_block_label_maps(self):
        if self._block_label_maps is None:
            self._block_label_maps = [self._block_labels_by_L(block, include_occupancy_expansion=False) for block in self.blocks]
        return self._block_label_maps

    def _ensure_structured_block_label_maps(self):
        if self._structured_block_label_maps is None:
            self._structured_block_label_maps = [self._block_labels_by_L(block, include_occupancy_expansion=True) for block in self.blocks]
        return self._structured_block_label_maps

    def _relabel_block_etas(self, label, block_etas, pos = 0):
        if isinstance(label, LeafLabel):
            return LeafLabel(n=int(block_etas[pos]), l=int(label.l), tree_type=label.tree_type), pos + 1
        if isinstance(label, SymBlockLabel):
            return (
                SymBlockLabel(
                    n=int(block_etas[pos]),
                    l=int(label.l),
                    k_b=int(label.k_b),
                    Lambda=int(label.Lambda),
                    multiplicity_index=int(label.multiplicity_index),
                    representative_internal_Ls=tuple(int(v) for v in label.representative_internal_Ls),
                    tree_type=label.tree_type,
                    basis_key=tuple(label.basis_key),
                    occupancy_expansion_by_M=tuple(label.occupancy_expansion_by_M),
                ),
                pos + 1,
            )
        left, pos = self._relabel_block_etas(label.left, block_etas, pos)
        right, pos = self._relabel_block_etas(label.right, block_etas, pos)
        return NodeLabel(left=left, right=right, L=int(label.L), tree_type=label.tree_type), pos

    def _template_structured_label(self, label):
        template_etas = tuple(range(1, len(self.blocks) + 1))
        templated, consumed = self._relabel_block_etas(label, template_etas)
        if consumed != len(template_etas):
            raise RuntimeError("Did not consume every block while building the label template cache.")
        return templated

    def _populate_cache_from_template(self):
        template = _LABEL_TEMPLATE_CACHE.get(self._template_signature)
        if template is None:
            return False
        block_etas = tuple(int(block.eta) for block in self.blocks)
        structured_cache = {}
        labels_cache = {}
        for L, template_labels in sorted(template.items()):
            actual_labels = []
            compact_labels = []
            for template_label in template_labels:
                actual_label, consumed = self._relabel_block_etas(template_label, block_etas)
                if consumed != len(block_etas):
                    raise RuntimeError("Did not consume every block while restoring the label template cache.")
                actual_labels.append(actual_label)
                compact_labels.append(tuple(actual_label.compact_label()))
            structured_cache[int(L)] = actual_labels
            labels_cache[int(L)] = compact_labels
        self._structured_labels_cache = structured_cache
        self._labels_cache = labels_cache
        return True

    def _populate_cache_from_sector(self):
        cached = _LABEL_SECTOR_CACHE.get(self._sector_signature)
        if cached is None:
            return False
        labels_cache, structured_cache = cached
        self._labels_cache = {int(L): list(items) for L, items in labels_cache.items()}
        return True

    @staticmethod
    def _dedupe_sorted(labels):
        """Deduplicate labels by compact form while keeping canonical ordering."""
        uniq = {}
        for label in labels:
            uniq.setdefault(label.compact_label(), label)
        return [uniq[key] for key in sorted(uniq)]

    def _block_labels_by_L(self, block, *, include_occupancy_expansion):
        r"""Build the exact label set for one repeated-channel block.

        For a block ``(\eta, l)_b^{k_b}``, Schur-Weyl/Young-subgroup theory says
        the invariant block space is ``Sym^{k_b}(V_{l_b})``. This method turns
        that abstract decomposition into explicit canonical labels:

        - if ``k_b = 1``, the block is just a leaf irrep ``V_l``
        - if ``k_b > 1``, we use the homogeneous algebraic decomposer to obtain
          the exact multiplicities ``d_{\Lambda_b}``
        - for each allowed ``\Lambda_b``, we attach a block-local
          ``multiplicity_index`` and a canonical representative internal-coupling
          tuple

        The name ``multiplicity_index`` is deliberate: in ACE notation ``mu`` is
        commonly reserved for chemical/channel indices, so we avoid reusing that
        symbol here for multiplicity-space coordinates.

        The returned dictionary is keyed by the block's final angular momentum.
        """
        n = int(block.eta)
        l = int(block.l)
        k_b = int(block.multiplicity)
        if k_b == 1:
            return {l: [LeafLabel(n=n, l=l, tree_type=self.tree_type)]}
        out = {}
        if not include_occupancy_expansion:
            multiplicities = self.hom_reps.decompose(int(k_b), int(l))
            for Lambda_b, multiplicity in sorted(multiplicities.items()):
                out[int(Lambda_b)] = [
                    SymBlockLabel(
                        n=n,
                        l=l,
                        k_b=k_b,
                        Lambda=int(Lambda_b),
                        multiplicity_index=int(multiplicity_index),
                        representative_internal_Ls=tuple(),
                        tree_type=self.tree_type,
                        basis_key=("sym", int(Lambda_b), int(multiplicity_index)),
                        occupancy_expansion_by_M=tuple(),
                    )
                    for multiplicity_index in range(int(multiplicity))
                ]
            return out
        # This is where we apply the exact symmetric-power decomposition of the
        # repeated block, rather than generating a redundant naive tree basis.
        basis_states = self.hom_reps.basis_states_by_L(
            k_b=k_b,
            l=l,
            tree_type=self.tree_type,
            basis_mode=self.block_basis_mode,
        )
        for Lambda_b, states in basis_states.items():
            out[Lambda_b] = [
                SymBlockLabel(
                    n=n,
                    l=l,
                    k_b=k_b,
                    Lambda=Lambda_b,
                    multiplicity_index=int(state.multiplicity_index),
                    representative_internal_Ls=tuple(state.representative_internal_Ls),
                    tree_type=self.tree_type,
                    basis_key=tuple(state.basis_key),
                    occupancy_expansion_by_M=tuple(state.occupancy_expansion_by_M),
                )
                for state in states
            ]
        return out

    def _combine_block_subspaces(
        self,
        left_map,
        right_map,
    ):
        """Couple two already symmetry-adapted block subspaces.

        At this stage permutation redundancy has already been removed inside each
        repeated block. What remains is ordinary SO(3) coupling between the left
        and right subspaces along the selected binary tree.
        """
        out = {}
        for J_left, left_labels in left_map.items():
            for J_right, right_labels in right_map.items():
                for left_label in left_labels:
                    for right_label in right_labels:
                        for L_out in cg_allowed(int(J_left), int(J_right)):
                            out.setdefault(L_out, []).append(
                                NodeLabel(left=left_label, right=right_label, L=L_out, tree_type=self.tree_type)
                            )
        return {L: self._dedupe_sorted(labels) for L, labels in out.items()}

    def _labels_on_block_shape(
        self,
        shape,
        block_idx = 0,
        *,
        structured = False,
    ):
        """Recursively build labels on the block-coupling tree.

        The leaves of this tree are *blocks*, not raw neighbor channels. This is
        the concrete realization of the paper's statement that one should first
        reduce each repeated block and only then couple the resulting irreps.
        """
        block_label_maps = (
            self._ensure_structured_block_label_maps()
            if structured
            else self._ensure_block_label_maps()
        )
        if shape.is_leaf:
            return block_label_maps[block_idx], block_idx + 1
        left_map, next_idx = self._labels_on_block_shape(shape.left, block_idx, structured=structured)
        right_map, next_idx = self._labels_on_block_shape(shape.right, next_idx, structured=structured)
        return self._combine_block_subspaces(left_map, right_map), next_idx

    def _leaf_spans(self, label, start = 0):
        """Return canonical leaf spans for a structured exact label tree."""
        if isinstance(label, (LeafLabel, SymBlockLabel)):
            stop = start + len(label.n_sequence())
            return tuple(range(start, stop)), stop
        left_span, next_idx = self._leaf_spans(label.left, start)
        right_span, next_idx = self._leaf_spans(label.right, next_idx)
        return left_span + right_span, next_idx

    def _internal_node_metadata(self, label, start = 0, postorder_index = 0):
        """Collect internal-node metadata in compact-label postorder."""
        if isinstance(label, (LeafLabel, SymBlockLabel)):
            span, next_leaf = self._leaf_spans(label, start)
            return [], next_leaf, postorder_index, span
        left_nodes, next_leaf, post_idx, left_span = self._internal_node_metadata(label.left, start, postorder_index)
        right_nodes, next_leaf, post_idx, right_span = self._internal_node_metadata(label.right, next_leaf, post_idx)
        node = InternalAngularNodeMetadata(
            postorder_index=post_idx,
            coupled_L=int(label.L),
            left_leaf_span=tuple(left_span),
            right_leaf_span=tuple(right_span),
        )
        return left_nodes + right_nodes + [node], next_leaf, post_idx + 1, tuple(left_span) + tuple(right_span)

    def _block_multiplicity_metadata(self, label):
        """Collect blockwise Schur--Weyl / Young multiplicity labels."""
        if isinstance(label, LeafLabel):
            return [
                BlockMultiplicityMetadata(
                    eta=int(label.n),
                    l=int(label.l),
                    block_size=1,
                    Lambda=int(label.l),
                    multiplicity_index=0,
                    representative_internal_Ls=tuple(),
                )
            ]
        if isinstance(label, SymBlockLabel):
            return [
                BlockMultiplicityMetadata(
                    eta=int(label.n),
                    l=int(label.l),
                    block_size=int(label.k_b),
                    Lambda=int(label.Lambda),
                    multiplicity_index=int(label.multiplicity_index),
                    representative_internal_Ls=tuple(label.representative_internal_Ls),
                    basis_key=tuple(label.basis_key),
                )
            ]
        return self._block_multiplicity_metadata(label.left) + self._block_multiplicity_metadata(label.right)

    def _exact_label_metadata(self, label):
        """Build rich metadata for one canonical exact basis vector."""
        internal_nodes, _, _, _ = self._internal_node_metadata(label, 0, 0)
        reconstruction = ReconstructionRecipe(
            tree_type=self.tree_type,
            tree_signature=self.block_tree_shape.signature() if self.block_tree_shape is not None else ("leaf", 1),
            internal_Ls_postorder=tuple(label.internal_Ls_postorder()),
        )
        return ExactLabelMetadata(
            compact_label=tuple(label.compact_label()),
            eta_tuple=tuple(int(x) for x in label.n_sequence()),
            l_tuple=tuple(int(x) for x in label.l_sequence()),
            root_L=int(label.total_L()),
            tree_type=self.tree_type,
            canonical_tree_signature=reconstruction.tree_signature,
            young_block_multiplicities=tuple(self._block_multiplicity_metadata(label)),
            internal_nodes=tuple(internal_nodes),
            reconstruction=reconstruction,
            tensor_basis_expansion=None,
        )

    def _ensure_structured_labels(self):
        if self._structured_labels_cache:
            return
        cached = _LABEL_SECTOR_CACHE.get(self._sector_signature)
        if cached is not None:
            labels_cache, structured_cache = cached
            if not self._labels_cache:
                self._labels_cache = {int(L): list(items) for L, items in labels_cache.items()}
            if structured_cache:
                self._structured_labels_cache = {int(L): list(items) for L, items in structured_cache.items()}
                return
        if self._populate_cache_from_template():
            return
        if not self.blocks:
            self._labels_cache = {}
            self._structured_labels_cache = {}
            return
        full_map, consumed = self._labels_on_block_shape(self.block_tree_shape, structured=True)
        if consumed != len(self.blocks):
            raise RuntimeError("Did not consume every block when building the exact block-coupled basis.")
        labels_cache = {}
        structured_cache = {}
        for L, labels in full_map.items():
            unique_labels = self._dedupe_sorted(labels)
            compact = [tuple(label.compact_label()) for label in unique_labels]
            if compact:
                labels_cache[int(L)] = compact
                structured_cache[int(L)] = list(unique_labels)
        self._labels_cache = labels_cache
        self._structured_labels_cache = structured_cache
        self._metadata_cache = {}
        self._sector_data_cache = {}
        _LABEL_TEMPLATE_CACHE.put(
            self._template_signature,
            {
                int(L): tuple(self._template_structured_label(label) for label in labels)
                for L, labels in structured_cache.items()
            },
        )
        _LABEL_SECTOR_CACHE.put(
            self._sector_signature,
            (
                {int(L): tuple(labels) for L, labels in self._labels_cache.items()},
                {int(L): tuple(labels) for L, labels in self._structured_labels_cache.items()},
            ),
        )

    def _ensure_lightweight_structured_labels(self):
        if self._lightweight_structured_labels_cache:
            return
        if not self.blocks:
            self._lightweight_structured_labels_cache = {}
            return
        full_map, consumed = self._labels_on_block_shape(self.block_tree_shape, structured=False)
        if consumed != len(self.blocks):
            raise RuntimeError("Did not consume every block when building the lightweight exact basis.")
        self._lightweight_structured_labels_cache = {
            int(L): self._dedupe_sorted(labels)
            for L, labels in full_map.items()
            if labels
        }

    def _ensure_compact_labels(self):
        if self._labels_cache:
            return
        if self._populate_cache_from_sector():
            return
        if not self.blocks:
            self._labels_cache = {}
            self._metadata_cache = {}
            self._structured_labels_cache = {}
            self._sector_data_cache = {}
            return
        full_map, consumed = self._labels_on_block_shape(self.block_tree_shape, structured=False)
        if consumed != len(self.blocks):
            raise RuntimeError("Did not consume every block when building the exact compact block-coupled basis.")
        self._lightweight_structured_labels_cache = {
            int(L): self._dedupe_sorted(labels)
            for L, labels in full_map.items()
            if labels
        }
        labels_cache = {}
        for L, unique_labels in self._lightweight_structured_labels_cache.items():
            compact = [tuple(label.compact_label()) for label in unique_labels]
            if compact:
                labels_cache[int(L)] = compact
        self._labels_cache = labels_cache
        self._metadata_cache = {}
        self._structured_labels_cache = {}
        self._sector_data_cache = {}
        _LABEL_SECTOR_CACHE.put(
            self._sector_signature,
            (
                {int(L): tuple(labels) for L, labels in self._labels_cache.items()},
                {},
            ),
        )

    def _ensure_metadata(self):
        if self._metadata_cache:
            return
        self._ensure_structured_labels()
        self._metadata_cache = {
            int(L): [self._exact_label_metadata(label) for label in labels]
            for L, labels in self._structured_labels_cache.items()
        }

    def metadata_by_L(self):
        """Return rich metadata for every exact basis vector grouped by final ``L``."""
        self._ensure_metadata()
        return {L: list(items) for L, items in sorted(self._metadata_cache.items())}

    def metadata_for_target(self, L_target):
        """Return rich metadata for one target ``L_R``."""
        self._ensure_metadata()
        return list(self._metadata_cache.get(int(L_target), []))

    def _sector_signature_for_target(self, L_target):
        return ExactSectorSignature(
            nin=tuple(int(x) for x in self.nin),
            lin=tuple(int(x) for x in self.lin),
            L_R=int(L_target),
            tree_type=str(self.tree_type),
        )

    def _build_sector_data_cache(self):
        if self._sector_data_cache:
            return
        self._ensure_structured_labels()
        self._ensure_metadata()
        sectors = {}
        all_targets = sorted(set(self._labels_cache) | set(self._structured_labels_cache))
        for L in all_targets:
            compact_labels = list(self._labels_cache.get(int(L), []))
            structured_labels = list(self._structured_labels_cache.get(int(L), []))
            metadata_items = list(self._metadata_cache.get(int(L), []))
            if not compact_labels or len(compact_labels) != len(structured_labels):
                continue
            signature = self._sector_signature_for_target(int(L))
            entries = []
            for basis_index, (compact_label, structured_label) in enumerate(zip(compact_labels, structured_labels)):
                metadata = metadata_items[basis_index] if basis_index < len(metadata_items) else None
                entries.append(
                    ExactBasisEntry(
                        handle=ExactBasisHandle(sector=signature, basis_index=int(basis_index)),
                        compact_label=tuple(compact_label),
                        structured_label=structured_label,
                        metadata=metadata,
                    )
                )
            sectors[int(L)] = ExactBasisSector(signature=signature, entries=tuple(entries))
        self._sector_data_cache = sectors

    def sector_data_by_L(self):
        """Return structured exact basis sectors keyed by final ``L_R``."""
        self._build_sector_data_cache()
        return {int(L): sector for L, sector in sorted(self._sector_data_cache.items())}

    def sector_data_for_target(self, L_target):
        """Return the structured exact basis sector for one target ``L_R``."""
        self._build_sector_data_cache()
        return self._sector_data_cache.get(int(L_target))

    def structured_label_objects_by_L(self):
        """Return structured exact label objects grouped by final ``L_R`` without metadata."""
        self._ensure_structured_labels()
        return {int(L): list(labels) for L, labels in sorted(self._structured_labels_cache.items())}

    def structured_label_objects_for_target(self, L_target):
        """Return structured exact label objects for one target without building sector metadata."""
        self._ensure_structured_labels()
        return list(self._structured_labels_cache.get(int(L_target), []))

    def lightweight_structured_label_objects_by_L(self):
        """Return structured labels without homogeneous occupancy expansions.

        This is the hot path for factorized coefficient schedules: it keeps the
        ACE block decomposition and inter-block SO(3) tree, but does not build
        exact symmetric-power magnetic expansions for repeated blocks.
        """
        self._ensure_lightweight_structured_labels()
        return {int(L): list(labels) for L, labels in sorted(self._lightweight_structured_labels_cache.items())}

    def lightweight_structured_label_objects_for_target(self, L_target):
        """Return lightweight structured labels for one target ``L_R``."""
        self._ensure_lightweight_structured_labels()
        return list(self._lightweight_structured_labels_cache.get(int(L_target), []))

    def compact_label_objects_by_L(self):
        """Return compact canonical labels grouped by final ``L``.

        This is the default hot-path enumeration API. It builds and caches the
        canonical compact tuple labels without forcing metadata or sector-table
        construction.
        """
        if self._labels_cache:
            return {
                L: [CompactTupleLabel(compact) for compact in labels]
                for L, labels in sorted(self._labels_cache.items())
            }
        self._ensure_compact_labels()
        return {
            L: [CompactTupleLabel(compact) for compact in labels]
            for L, labels in sorted(self._labels_cache.items())
        }

    def labels_by_L(self):
        """Compatibility alias for :meth:`compact_label_objects_by_L`."""
        return self.compact_label_objects_by_L()

    def compact_labels_by_L(self):
        """Return compact canonical label tuples grouped by final ``L``."""
        if not self._labels_cache:
            self.compact_label_objects_by_L()
        return {int(L): list(labels) for L, labels in sorted(self._labels_cache.items())}

    def compact_labels_for_target(self, L_target):
        """Return the exact canonical compact labels for one target ``L_R``."""
        if not self._labels_cache:
            self.compact_label_objects_by_L()
        return list(self._labels_cache.get(int(L_target), []))

    def compact_label_objects_for_target(self, L_target):
        """Return compact canonical label wrapper objects for one target ``L_R``."""
        return [CompactTupleLabel(label) for label in self.compact_labels_for_target(L_target)]

    def labels_for_target(self, L_target):
        """Compatibility alias for :meth:`compact_label_objects_for_target`."""
        return self.compact_label_objects_for_target(L_target)

    def structured_labels_by_L(self):
        """Return structured representation-theory labels grouped by final ``L``."""
        self._ensure_structured_labels()
        return {int(L): list(labels) for L, labels in sorted(self._structured_labels_cache.items())}

    def structured_labels_for_target(self, L_target):
        """Return structured representation-theory labels for one target ``L_R``."""
        self._ensure_structured_labels()
        return list(self._structured_labels_cache.get(int(L_target), []))

    def counts_by_L(self):
        """Return the number of exact basis labels for each reachable final ``L``."""
        if self._labels_cache:
            return {int(L): len(labels) for L, labels in sorted(self._labels_cache.items())}
        return {
            int(L): int(count)
            for L, count in sorted(self.decomposition.alpha_by_L_R.items())
            if int(count) > 0
        }

    def count_for_target(self, L_target):
        """Return the exact basis size for one target ``L_R``."""
        if self._labels_cache:
            return len(self._labels_cache.get(int(L_target), []))
        return int(self.decomposition.alpha_by_L_R.get(int(L_target), 0))

    def invariant_subspace_decomposition(self):
        """Return the theory-side decomposition used by the constructor.

        This is useful if you want to inspect the abstract block structure and
        multiplicity counts that underlie the concrete labels built by this
        backend.
        """
        return self.decomposition
