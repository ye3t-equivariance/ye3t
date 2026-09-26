
"""Exact symbolic caches for primitive labels, CG expansions, and derived features."""

from collections import OrderedDict
import hashlib
import time
import warnings

import numpy as np

from ye3t.core.basis.sector import ExactBasisHandle
from ye3t.core.basis.validation import canonicalize_leaf_quantum_numbers
from ye3t.core.labels import CompactLabel, normalize_compact_label
from ye3t.core.subtree_dag import expand_tree_key_exact, expand_tree_key_numeric, raw_tree_key
from ye3t.exact_scalars import ExactRadical, exact_scalar
from ye3t._record import recordclass


def _sympy():
    from ye3t._optional_sympy import sp

    return sp


def _hash_parts(*parts):
    payload = repr(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_feature_lookup_policy(policy):
    normalized = str(policy).strip().lower().replace("-", "_")
    if normalized not in {"eager", "cached_only", "off"}:
        raise ValueError(
            "feature_lookup_policy must be one of {'eager', 'cached_only', 'off'}, "
            f"got {policy!r}."
        )
    return normalized


def _build_graded_basis_registry(tree_type):
    from ye3t.core.graded_algebra import GradedBasisRegistry

    return GradedBasisRegistry(tree_type=tree_type)


def _build_exact_product_engine(tree_type):
    from ye3t.core.product_engine import ExactProductExpansionEngine

    return ExactProductExpansionEngine(tree_type=tree_type)


@recordclass(('rank', 'L', 'multiplicity_id', 'n_tuple', 'l_tuple', 'tree_type'), frozen = True)
class PrimitiveCacheKey:
    """Identity for a primitive basis element."""
    n_tuple = ()
    l_tuple = ()
    tree_type = "balanced"

    def __post_init__(self):
        if self.rank < 1:
            raise ValueError(f"Rank must be positive, got {self.rank}")
        if self.L < 0:
            raise ValueError(f"Angular momentum must be non-negative, got {self.L}")
        if self.multiplicity_id < 0:
            raise ValueError(f"Multiplicity ID must be non-negative, got {self.multiplicity_id}")


@recordclass(('left_primitive_key', 'right_primitive_key', 'coupling_coefficients', 'target_L'), frozen = True)
class ProductRecipe:
    """Metadata for a generated product direction."""

    def __post_init__(self):
        if not self.coupling_coefficients:
            raise ValueError("Coupling coefficients cannot be empty")
        if self.target_L < 0:
            raise ValueError("Target L must be non-negative")
        if (
            self.left_primitive_key.rank == 2
            and self.right_primitive_key.rank == 2
            and len(self.coupling_coefficients) != 2
        ):
            raise ValueError("Rank-2 by rank-2 recipes require two coefficients")


@recordclass(('label', 'expansion', 'creation_time', 'access_count'))
class PrimitiveFeature:
    """Cached primitive expansion in the uncoupled magnetic basis."""
    creation_time = 0.0
    access_count = 0

    def __post_init__(self):
        if self.creation_time == 0.0:
            self.creation_time = time.time()


@recordclass(('digest', 'component_count', 'nonzero_count'), frozen = True)
class SparseExpansionFingerprint:
    """Compact identity for a sparse exact magnetic-basis expansion."""

    def as_key(self):
        return (str(self.digest), int(self.component_count), int(self.nonzero_count))


class PrimitiveCacheSystem:
    """LRU cache for exact primitive feature expansions."""

    def __init__(
        self,
        max_cache_size = 1000,
        tree_type = "balanced",
        max_cache_rank = None,
        primitive_mode = "auto",
        feature_lookup_policy = "cached_only",
    ):
        self.max_cache_size = int(max_cache_size)
        self.tree_type = str(tree_type)
        self.max_cache_rank = None if max_cache_rank is None else int(max_cache_rank)
        self.primitive_mode = str(primitive_mode)
        self.feature_lookup_policy = _validate_feature_lookup_policy(feature_lookup_policy)
        self._primitive_cache = {}
        self._recipe_cache = {}
        self._access_order = []
        self._label_to_key = OrderedDict()
        self._registry = _build_graded_basis_registry(self.tree_type)
        self._product_engine = _build_exact_product_engine(self.tree_type)
        self._product_engines = {self.tree_type: self._product_engine}
        self._primitive_membership_cache = OrderedDict()
        self._primitive_membership_cache_size = 4096
        self._primitive_basis_indices_cache = OrderedDict()
        self._primitive_basis_indices_cache_size = 512
        self._label_index_cache = OrderedDict()
        self._label_index_cache_size = 256
        self._canonical_label_alias_cache = OrderedDict()
        self._canonical_label_alias_cache_size = 512
        self._sector_fingerprint_lookup_cache = OrderedDict()
        self._sector_fingerprint_lookup_cache_size = 256
        self.cache_hits = 0
        self.cache_misses = 0
        self.cached_only_hits = 0
        self.cached_only_misses = 0
        self.reconstruction_count = 0
        self.full_computation_count = 0

    def _engine_for_tree_type(self, tree_type):
        engine = self._product_engines.get(str(tree_type))
        if engine is None:
            engine = _build_exact_product_engine(str(tree_type))
            self._product_engines[str(tree_type)] = engine
        return engine

    def _mode_for_label(self, label):
        if self.primitive_mode != "auto":
            return self.primitive_mode
        return "invariant" if int(label.L_R) == 0 else "module"

    @staticmethod
    def _legacy_key(key):
        return PrimitiveCacheKey(rank=key.rank, L=key.L, multiplicity_id=key.multiplicity_id)

    def _cache_key_for_label(self, label):
        label = self.canonicalize_label_for_cache(label)
        basis_index = self._label_index_for_label(label)
        if basis_index is None:
            return None
        key = PrimitiveCacheKey(
            rank=label.rank,
            L=label.L_R,
            multiplicity_id=int(basis_index),
            n_tuple=tuple(label.n_tuple),
            l_tuple=tuple(label.l_tuple),
            tree_type=label.tree_type,
        )
        legacy = self._legacy_key(key)
        return legacy if legacy in self._primitive_cache else key

    def _handle_for_label(self, label):
        label = self.canonicalize_label_for_cache(label)
        basis_index = self._label_index_for_label(label)
        if basis_index is None:
            return None
        try:
            space = self._engine_for_tree_type(label.tree_type).feature_space(label.n_tuple, label.l_tuple, label.L_R)
        except ValueError:
            return None
        if space.signature is None:
            return None
        return ExactBasisHandle(sector=space.signature, basis_index=int(basis_index))

    def label_for_handle(self, handle):
        sector = handle.sector
        try:
            space = self._engine_for_tree_type(sector.tree_type).feature_space(
                sector.nin,
                sector.lin,
                sector.L_R,
            )
        except ValueError:
            return None
        basis_index = int(handle.basis_index)
        if basis_index < 0 or basis_index >= len(space.labels):
            return None
        return space.labels[basis_index]

    def is_primitive_handle(self, handle):
        label = self.label_for_handle(handle)
        return False if label is None else self.is_primitive(label)

    def _should_cache_by_rank(self, label):
        return self.max_cache_rank is None or int(label.rank) <= int(self.max_cache_rank)

    def clear(self):
        """Clear primitive features, recipes, and counters."""
        self._primitive_cache.clear()
        self._recipe_cache.clear()
        self._access_order.clear()
        self._label_to_key.clear()
        self._primitive_membership_cache.clear()
        self._primitive_basis_indices_cache.clear()
        self._label_index_cache.clear()
        self._canonical_label_alias_cache.clear()
        self._sector_fingerprint_lookup_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.cached_only_hits = 0
        self.cached_only_misses = 0
        self.reconstruction_count = 0
        self.full_computation_count = 0

    def _evict_if_needed(self):
        while len(self._primitive_cache) > self.max_cache_size and self._access_order:
            key = self._access_order.pop(0)
            feature = self._primitive_cache.pop(key, None)
            if feature is not None:
                self._label_to_key.pop(normalize_compact_label(feature.label), None)

    def _update_access_order(self, key):
        try:
            self._access_order.remove(key)
        except ValueError:
            pass
        self._access_order.append(key)

    def _membership_cache_get(self, key):
        value = self._primitive_membership_cache.get(key)
        if value is not None:
            self._primitive_membership_cache.move_to_end(key)
        return value

    def _membership_cache_put(self, key, value):
        self._primitive_membership_cache[key] = bool(value)
        self._primitive_membership_cache.move_to_end(key)
        while len(self._primitive_membership_cache) > self._primitive_membership_cache_size:
            self._primitive_membership_cache.popitem(last=False)

    def _canonical_alias_cache_get(self, label):
        if label in self._canonical_label_alias_cache:
            value = self._canonical_label_alias_cache[label]
            self._canonical_label_alias_cache.move_to_end(label)
            return True, value
        return False, None

    def _canonical_alias_cache_put(self, label, resolved):
        self._canonical_label_alias_cache[label] = resolved
        self._canonical_label_alias_cache.move_to_end(label)
        while len(self._canonical_label_alias_cache) > self._canonical_label_alias_cache_size:
            self._canonical_label_alias_cache.popitem(last=False)

    @staticmethod
    def _leaf_sort_permutation(label):
        canonical_nin, canonical_lin = canonicalize_leaf_quantum_numbers(label.n_tuple, label.l_tuple)
        items = [
            (int(n), int(l), int(idx))
            for idx, (n, l) in enumerate(zip(label.n_tuple, label.l_tuple))
        ]
        sorted_items = sorted(items)
        permutation = tuple(int(old_idx) for _, _, old_idx in sorted_items)
        return canonical_nin, canonical_lin, permutation

    @staticmethod
    def _permute_expansion(
        expansion,
        permutation,
    ):
        if not permutation:
            return {int(M): dict(block) for M, block in expansion.items()}
        out = {}
        for M, block in expansion.items():
            permuted_block = {}
            for ms, coeff in block.items():
                permuted_ms = tuple(int(ms[idx]) for idx in permutation)
                permuted_block[permuted_ms] = coeff
            out[int(M)] = permuted_block
        return out

    @staticmethod
    def _native_exact_scalar(value):
        if isinstance(value, ExactRadical):
            return value
        try:
            return exact_scalar(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _stable_scalar_payload(value):
        native = PrimitiveCacheSystem._native_exact_scalar(value)
        if native is not None:
            return repr(native.stable_key())
        return _sympy().srepr(value)

    @staticmethod
    def _expansion_fingerprint(
        expansion,
    ):
        pivot = None
        hasher = hashlib.sha256()
        component_count = 0
        nonzero_count = 0
        for M in sorted(int(x) for x in expansion):
            block = expansion[int(M)]
            block_has_support = False
            for ms, coeff in sorted(block.items()):
                native = PrimitiveCacheSystem._native_exact_scalar(coeff)
                if native is not None:
                    simplified = native
                else:
                    simplified = _sympy().simplify(coeff)
                if simplified == 0:
                    continue
                if pivot is None:
                    pivot = simplified
                if isinstance(simplified, ExactRadical) and isinstance(pivot, ExactRadical):
                    try:
                        normalized = simplified / pivot
                    except (ValueError, ZeroDivisionError):
                        normalized = _sympy().together(_sympy().simplify(simplified._sympy_() / pivot._sympy_()))
                else:
                    normalized = _sympy().together(_sympy().simplify(simplified / pivot))
                hasher.update(str(int(M)).encode("utf-8"))
                hasher.update(repr(tuple(int(x) for x in ms)).encode("utf-8"))
                hasher.update(PrimitiveCacheSystem._stable_scalar_payload(normalized).encode("utf-8"))
                nonzero_count += 1
                block_has_support = True
            if block_has_support:
                component_count += 1
        if pivot is None:
            return None
        return SparseExpansionFingerprint(
            digest=hasher.hexdigest(),
            component_count=component_count,
            nonzero_count=nonzero_count,
        )

    def _direct_label_index_for_label(self, label):
        label = normalize_compact_label(label)
        sector_key = (
            str(label.tree_type),
            tuple(label.n_tuple),
            tuple(label.l_tuple),
            int(label.L_R),
        )
        cached = self._label_index_cache.get(sector_key)
        if cached is None:
            try:
                space = self._engine_for_tree_type(label.tree_type).feature_space(label.n_tuple, label.l_tuple, label.L_R)
            except ValueError:
                return None
            cached = {space_label: int(idx) for idx, space_label in enumerate(space.labels)}
            self._label_index_cache[sector_key] = cached
            self._label_index_cache.move_to_end(sector_key)
            while len(self._label_index_cache) > self._label_index_cache_size:
                self._label_index_cache.popitem(last=False)
        else:
            self._label_index_cache.move_to_end(sector_key)
        return cached.get(label)

    def _sector_fingerprint_lookup(
        self,
        *,
        n_tuple,
        l_tuple,
        L_R,
        tree_type,
    ):
        cache_key = (str(tree_type), tuple(n_tuple), tuple(l_tuple), int(L_R))
        cached = self._sector_fingerprint_lookup_cache.get(cache_key)
        if cached is not None:
            self._sector_fingerprint_lookup_cache.move_to_end(cache_key)
            return cached
        lookup = {}
        try:
            space = self._engine_for_tree_type(tree_type).feature_space(tuple(n_tuple), tuple(l_tuple), int(L_R))
        except ValueError:
            space = None
        if space is not None:
            engine = self._engine_for_tree_type(tree_type)
            for candidate in space.labels:
                fingerprint = self._expansion_fingerprint(engine._m_vectors(candidate))
                if fingerprint is not None:
                    lookup[fingerprint.as_key()] = candidate
        self._sector_fingerprint_lookup_cache[cache_key] = lookup
        self._sector_fingerprint_lookup_cache.move_to_end(cache_key)
        while len(self._sector_fingerprint_lookup_cache) > self._sector_fingerprint_lookup_cache_size:
            self._sector_fingerprint_lookup_cache.popitem(last=False)
        return lookup

    def _raw_cache_representative(self, label):
        label = normalize_compact_label(label)
        if len(label.n_tuple) != len(label.l_tuple):
            return label
        canonical_nin, canonical_lin, _ = self._leaf_sort_permutation(label)
        return CompactLabel(
            n_tuple=canonical_nin,
            l_tuple=canonical_lin,
            internal_Ls=tuple(label.internal_Ls),
            tree_type=str(label.tree_type),
            basis_key=tuple(label.basis_key),
        )

    def canonicalize_label_for_cache(self, label):
        """Best-effort canonical leaf-order representative for cache lookup.

        This is intentionally conservative: if an out-of-order leaf labeling can
        be matched exactly to one canonical basis label after transporting its
        magnetic expansion into canonical leaf order, we return that canonical
        label. Otherwise we keep the original normalized label.
        """
        label = normalize_compact_label(label)
        if len(label.n_tuple) != len(label.l_tuple):
            return label
        found, cached = self._canonical_alias_cache_get(label)
        if found:
            return label if cached is None else cached
        direct_index = self._direct_label_index_for_label(label)
        if direct_index is not None:
            self._canonical_alias_cache_put(label, label)
            return label
        canonical_nin, canonical_lin, permutation = self._leaf_sort_permutation(label)
        simple_candidate = self._raw_cache_representative(label)
        if self._direct_label_index_for_label(simple_candidate) is not None:
            self._canonical_alias_cache_put(label, simple_candidate)
            return simple_candidate
        fallback_candidate = (
            label
            if (
                tuple(label.n_tuple) == canonical_nin
                and tuple(label.l_tuple) == canonical_lin
            )
            else simple_candidate
        )
        try:
            lookup = self._sector_fingerprint_lookup(
                n_tuple=canonical_nin,
                l_tuple=canonical_lin,
                L_R=int(label.L_R),
                tree_type=str(label.tree_type),
            )
        except ValueError:
            self._canonical_alias_cache_put(label, fallback_candidate if fallback_candidate != label else None)
            return fallback_candidate
        transported = self._permute_expansion(self._engine_for_tree_type(label.tree_type)._m_vectors(label), permutation)
        transported_fingerprint = self._expansion_fingerprint(transported)
        if transported_fingerprint is not None:
            candidate = lookup.get(transported_fingerprint.as_key())
            if candidate is not None:
                self._canonical_alias_cache_put(label, candidate)
                return candidate
        self._canonical_alias_cache_put(label, fallback_candidate if fallback_candidate != label else None)
        return fallback_candidate

    def _label_index_for_label(self, label):
        label = normalize_compact_label(label)
        direct = self._direct_label_index_for_label(label)
        if direct is not None:
            return direct
        resolved = self.canonicalize_label_for_cache(label)
        if resolved == label:
            return None
        return self._direct_label_index_for_label(resolved)

    def _primitive_basis_indices_for_sector(
        self,
        *,
        n_tuple,
        l_tuple,
        L_R,
        tree_type,
        mode,
    ):
        cache_key = (str(tree_type), tuple(n_tuple), tuple(l_tuple), int(L_R), str(mode))
        cached = self._primitive_basis_indices_cache.get(cache_key)
        if cached is not None:
            self._primitive_basis_indices_cache.move_to_end(cache_key)
            return cached
        quotient = self._engine_for_tree_type(tree_type).primitive_quotient(
            tuple(n_tuple),
            tuple(l_tuple),
            int(L_R),
            mode=str(mode),
        )
        cached = tuple(int(i) for i in quotient.primitive_basis_indices)
        self._primitive_basis_indices_cache[cache_key] = cached
        self._primitive_basis_indices_cache.move_to_end(cache_key)
        while len(self._primitive_basis_indices_cache) > self._primitive_basis_indices_cache_size:
            self._primitive_basis_indices_cache.popitem(last=False)
        return cached

    def is_primitive(self, label):
        """Return whether ``label`` survives the exact primitive quotient."""
        label = self.canonicalize_label_for_cache(label)
        key = (
            tuple(label.n_tuple),
            tuple(label.l_tuple),
            tuple(label.internal_Ls),
            label.tree_type,
            tuple(label.basis_key),
            int(label.L_R),
            self._mode_for_label(label),
        )
        cached = self._membership_cache_get(key)
        if cached is not None:
            return cached
        basis_index = self._label_index_for_label(label)
        if basis_index is None:
            self._membership_cache_put(key, False)
            return False
        primitive_basis_indices = self._primitive_basis_indices_for_sector(
            n_tuple=tuple(label.n_tuple),
            l_tuple=tuple(label.l_tuple),
            L_R=int(label.L_R),
            tree_type=str(label.tree_type),
            mode=self._mode_for_label(label),
        )
        result = int(basis_index) in primitive_basis_indices
        self._membership_cache_put(key, result)
        return result

    def get_primitive_feature(self, label):
        """Return a cached primitive expansion, computing it on first use."""
        label = self.canonicalize_label_for_cache(label)
        if not self.is_primitive(label):
            return None
        cache_key = self._cache_key_for_label(label)
        if cache_key is None:
            return None
        if cache_key in self._primitive_cache:
            feature = self._primitive_cache[cache_key]
            feature.access_count += 1
            self.cache_hits += 1
            self._update_access_order(cache_key)
            return feature
        self.cache_misses += 1
        feature = PrimitiveFeature(label=label, expansion=self._engine_for_tree_type(label.tree_type)._m_vectors(label))
        if self._should_cache_by_rank(label):
            self.store_primitive_feature(feature, cache_key=cache_key)
        return feature

    def get_cached_primitive_for_label(self, label):
        """Return a cached primitive only if it is already stored."""
        label = self.canonicalize_label_for_cache(label)
        cache_key = self._label_to_key.get(label)
        if cache_key is None:
            return None
        feature = self._primitive_cache.get(cache_key)
        if feature is None:
            self._label_to_key.pop(label, None)
            return None
        feature.access_count += 1
        self._update_access_order(cache_key)
        return feature

    def get_primitive_feature_by_handle(self, handle):
        label = self.label_for_handle(handle)
        if label is None:
            return None
        return self.get_primitive_feature(label)

    def get_cached_primitive(self, key):
        """Return a cached primitive feature by key."""
        return self._primitive_cache.get(key)

    def store_primitive_feature(
        self,
        feature,
        *,
        cache_key = None,
    ):
        """Store a primitive feature under its canonical cache key."""
        resolved_key = cache_key if cache_key is not None else self._cache_key_for_label(feature.label)
        if resolved_key is None:
            return None
        normalized_label = normalize_compact_label(feature.label)
        self._primitive_cache[resolved_key] = feature
        self._label_to_key[normalized_label] = resolved_key
        self._label_to_key.move_to_end(normalized_label)
        self._update_access_order(resolved_key)
        self._evict_if_needed()
        return resolved_key

    def iter_cached_primitives(self):
        """Return cached primitive entries in storage order."""
        return tuple(self._primitive_cache.items())

    def iter_product_recipes(self):
        """Return stored product recipes."""
        return tuple(self._recipe_cache.items())

    def retain_most_recent_primitives(self, keep_entries):
        """Trim cached primitives to the ``keep_entries`` most recently used items."""
        keep = max(0, int(keep_entries))
        if len(self._access_order) <= keep:
            return
        keys_to_keep = set(self._access_order[-keep:]) if keep else set()
        for key in list(self._primitive_cache.keys()):
            if key not in keys_to_keep:
                self._primitive_cache.pop(key, None)
        self._access_order = [key for key in self._access_order if key in keys_to_keep]

    def store_product_recipe(self, target_label, recipe):
        """Store product metadata for ``target_label``."""
        original = normalize_compact_label(target_label)
        direct_handle = None
        if self._direct_label_index_for_label(original) is not None:
            direct_handle = self._handle_for_label(original)
        if direct_handle is not None:
            self._recipe_cache[direct_handle] = recipe
            return
        canonical_raw = self._raw_cache_representative(original)
        self._recipe_cache[normalize_compact_label(canonical_raw)] = recipe

    def get_product_recipe(self, target_label):
        """Return stored product metadata for ``target_label``."""
        original = normalize_compact_label(target_label)
        direct_handle = None
        if self._direct_label_index_for_label(original) is not None:
            direct_handle = self._handle_for_label(original)
        if direct_handle is not None and direct_handle in self._recipe_cache:
            return self._recipe_cache.get(direct_handle)
        resolved = self.canonicalize_label_for_cache(original)
        resolved_handle = self._handle_for_label(resolved)
        if direct_handle is not None:
            return None
        if resolved_handle is not None and resolved_handle in self._recipe_cache:
            return self._recipe_cache.get(resolved_handle)
        canonical_raw = self._raw_cache_representative(original)
        return self._recipe_cache.get(normalize_compact_label(canonical_raw))

    def get_product_recipe_by_handle(self, handle):
        recipe = self._recipe_cache.get(handle)
        if recipe is not None:
            return recipe
        label = self.label_for_handle(handle)
        if label is None:
            return None
        resolved = self.canonicalize_label_for_cache(label)
        if resolved != label:
            resolved_handle = self._handle_for_label(resolved)
            if resolved_handle is not None and resolved_handle in self._recipe_cache:
                return self._recipe_cache.get(resolved_handle)
        canonical_raw = self._raw_cache_representative(label)
        return self._recipe_cache.get(normalize_compact_label(canonical_raw))

    def reconstruct_from_primitives(self, label):
        """Return ``None`` unless an exact runtime reconstruction is available."""
        if self.get_product_recipe(label) is None:
            return None
        self.reconstruction_count += 1
        return None

    def reconstruct_from_primitives_by_handle(self, handle):
        if self.get_product_recipe_by_handle(handle) is None:
            return None
        self.reconstruction_count += 1
        return None

    def get_feature_by_any_means(
        self,
        label,
        *,
        feature_lookup_policy = None,
    ):
        """Return a primitive expansion or the exact full expansion."""
        label = normalize_compact_label(label)
        canonical = self.canonicalize_label_for_cache(label)
        policy = self.feature_lookup_policy if feature_lookup_policy is None else _validate_feature_lookup_policy(feature_lookup_policy)
        if policy == "eager":
            primitive = self.get_primitive_feature(canonical)
            if primitive is not None:
                return primitive.expansion
        elif policy == "cached_only":
            primitive = self.get_cached_primitive_for_label(canonical)
            if primitive is not None:
                self.cached_only_hits += 1
                return primitive.expansion
            self.cached_only_misses += 1
        self.full_computation_count += 1
        return self._engine_for_tree_type(label.tree_type)._m_vectors(label)

    def get_feature_by_handle(self, handle):
        label = self.label_for_handle(handle)
        if label is None:
            raise KeyError(f"Unknown basis handle: {handle!r}")
        return self.get_feature_by_any_means(label)

    def generate_recipes_for_sector(self, nin, lin, L_R):
        """Record generated directions for a sector."""
        target = self._engine_for_tree_type(self.tree_type).feature_space(tuple(nin), tuple(lin), int(L_R))
        mode = self._mode_for_sector(int(L_R))
        quotient = self._engine_for_tree_type(self.tree_type).primitive_quotient(tuple(nin), tuple(lin), int(L_R), mode=mode)
        generated = set(int(i) for i in quotient.generated_basis_indices)
        primitive_indices = set(
            self._primitive_basis_indices_for_sector(
                n_tuple=tuple(nin),
                l_tuple=tuple(lin),
                L_R=int(L_R),
                tree_type=str(self.tree_type),
                mode=mode,
            )
        )
        for idx, label in enumerate(target.labels):
            if idx in primitive_indices or idx not in generated:
                continue
            recipe = self._recipe_metadata_for_generated_label(label)
            self.store_product_recipe(label, recipe)

    def _mode_for_sector(self, L_R):
        if self.primitive_mode != "auto":
            return self.primitive_mode
        return "invariant" if int(L_R) == 0 else "module"

    def _recipe_metadata_for_generated_label(self, label):
        left = PrimitiveCacheKey(rank=max(1, label.rank // 2), L=0, multiplicity_id=0)
        right = PrimitiveCacheKey(rank=max(1, label.rank - left.rank), L=label.L_R, multiplicity_id=0)
        one = ExactRadical.rational(1)
        coeffs = (one, one) if left.rank == 2 and right.rank == 2 else (one,)
        return ProductRecipe(left, right, coeffs, target_L=label.L_R)

    def validate_reconstruction(
        self,
        nin,
        lin,
        L_R,
        tolerance = 1e-10,
        timeout = 120,
    ):
        """Summarize the exact primitive quotient for a sector."""
        del tolerance
        start = time.time()
        target = self._engine_for_tree_type(self.tree_type).feature_space(tuple(nin), tuple(lin), int(L_R))
        result = {
            "sector": (tuple(nin), tuple(lin), int(L_R)),
            "total_features": int(target.dim),
            "primitives": 0,
            "factorizable": 0,
            "validated": 0,
            "failed": 0,
            "validation_errors": [],
            "time_elapsed": 0.0,
        }
        for label in target.labels:
            if time.time() - start > float(timeout):
                warnings.warn(f"Validation timeout reached after {timeout} seconds")
                break
            if self.is_primitive(label):
                result["primitives"] += 1
                self.get_primitive_feature(label)
            else:
                result["factorizable"] += 1
        self.generate_recipes_for_sector(tuple(nin), tuple(lin), int(L_R))
        result["validated"] = int(result["factorizable"])
        result["time_elapsed"] = time.time() - start
        return result

    def get_cache_stats(self):
        """Return cache counters and sizes."""
        return {
            "primitive_cache_size": len(self._primitive_cache),
            "recipe_cache_size": len(self._recipe_cache),
            "feature_lookup_policy": self.feature_lookup_policy,
            "cache_hits": int(self.cache_hits),
            "cache_misses": int(self.cache_misses),
            "cached_only_hits": int(self.cached_only_hits),
            "cached_only_misses": int(self.cached_only_misses),
            "reconstruction_count": int(self.reconstruction_count),
            "full_computation_count": int(self.full_computation_count),
            "eviction_count": max(0, int(self.cache_misses) - len(self._primitive_cache)),
        }

    def get_primitive_cache_info(self):
        """Return cached primitive metadata ordered by use count."""
        now = time.time()
        info = [
            {
                "rank": int(key.rank),
                "L": int(key.L),
                "multiplicity_id": int(key.multiplicity_id),
                "n_tuple": tuple(key.n_tuple),
                "l_tuple": tuple(key.l_tuple),
                "tree_type": key.tree_type,
                "label": feature.label,
                "access_count": int(feature.access_count),
                "age": float(now - feature.creation_time),
            }
            for key, feature in self._primitive_cache.items()
        ]
        return sorted(info, key=lambda item: item["access_count"], reverse=True)


@recordclass(('rank', 'L', 'channel_multiset', 'internal_Ls', 'coupling_scheme', 'channel_tuple', 'basis_handle'), frozen = True)
class CGCacheKey:
    """Key for one exact coupled-tree expansion."""
    coupling_scheme = "balanced"
    channel_tuple = ()
    basis_handle = None

    @property
    def hash_key(self):
        if self.basis_handle is not None:
            return _hash_parts(self.basis_handle.sector, int(self.basis_handle.basis_index))
        channels = self.channel_tuple or tuple(sorted(self.channel_multiset))
        return _hash_parts(self.rank, self.L, channels, self.internal_Ls, self.coupling_scheme)


@recordclass(('key', 'expansion', 'coupling_coefficients', 'creation_time', 'access_count'))
class CGCachedEntry:
    """Cached exact Clebsch-Gordan expansion."""
    creation_time = 0.0
    access_count = 0

    def __post_init__(self):
        if self.creation_time == 0.0:
            self.creation_time = time.time()


class ClebschGordanCache:
    """LRU cache for symbolic coupled-tree expansions."""

    def __init__(self, max_size = 1000):
        self.max_size = int(max_size)
        self._cache = {}
        self._access_order = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.evictions = 0
        self._product_engines = {}

    def _engine_for_tree_type(self, tree_type):
        engine = self._product_engines.get(str(tree_type))
        if engine is None:
            engine = _build_exact_product_engine(str(tree_type))
            self._product_engines[str(tree_type)] = engine
        return engine

    def clear(self):
        self._cache.clear()
        self._access_order.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.evictions = 0

    def _evict_if_needed(self):
        while len(self._cache) > self.max_size and self._access_order:
            self._cache.pop(self._access_order.pop(0), None)
            self.evictions += 1

    def _touch(self, key):
        try:
            self._access_order.remove(key)
        except ValueError:
            pass
        self._access_order.append(key)

    @staticmethod
    def _key_for_label(label):
        label = normalize_compact_label(label)
        channel_tuple = tuple(zip(label.n_tuple, label.l_tuple))
        return CGCacheKey(
            rank=label.rank,
            L=label.L_R,
            channel_multiset=frozenset(channel_tuple),
            channel_tuple=channel_tuple,
            internal_Ls=label.internal_Ls + tuple(label.basis_key),
            coupling_scheme=label.tree_type,
        )

    def get_cg_expansion(self, label):
        """Return the exact expansion for ``label``."""
        label = normalize_compact_label(label)
        key = self._key_for_label(label)
        if label.basis_key:
            handle = self._engine_for_tree_type(label.tree_type).feature_space(
                label.n_tuple,
                label.l_tuple,
                label.L_R,
            ).handle_for_label(label)
            if handle is not None:
                key = CGCacheKey(
                    rank=key.rank,
                    L=key.L,
                    channel_multiset=key.channel_multiset,
                    internal_Ls=key.internal_Ls,
                    coupling_scheme=key.coupling_scheme,
                    channel_tuple=key.channel_tuple,
                    basis_handle=handle,
                )
        hash_key = key.hash_key
        if hash_key in self._cache:
            entry = self._cache[hash_key]
            entry.access_count += 1
            self.cache_hits += 1
            self._touch(hash_key)
            return entry.expansion
        self.cache_misses += 1
        expansion = self._ordinary_tree_expansion_for_label(label)
        if expansion is None:
            expansion = self._engine_for_tree_type(label.tree_type)._m_vectors(label)
        entry = CGCachedEntry(
            key=key,
            expansion=expansion,
            coupling_coefficients=self._nonzero_coefficients(expansion),
        )
        self._cache[hash_key] = entry
        self._touch(hash_key)
        self._evict_if_needed()
        return expansion

    @staticmethod
    def _ordinary_tree_expansion_for_label(label):
        label = normalize_compact_label(label)
        if label.basis_key:
            return None
        try:
            key = raw_tree_key(label.l_tuple, label.internal_Ls, label.tree_type)
        except ValueError:
            return None
        return expand_tree_key_exact(key)

    @staticmethod
    def _nonzero_coefficients(expansion):
        coeffs = []
        seen = set()
        for block in expansion.values():
            for coeff in block.values():
                if isinstance(coeff, ExactRadical):
                    value = coeff
                else:
                    try:
                        value = exact_scalar(coeff)
                    except (TypeError, ValueError):
                        value = _sympy().simplify(coeff)
                if value != 0 and value not in seen:
                    seen.add(value)
                    coeffs.append(value)
        return tuple(coeffs) or (ExactRadical.rational(0),)

    def get_cache_stats(self):
        total = self.cache_hits + self.cache_misses
        return {
            "cg_cache_size": len(self._cache),
            "cache_hits": int(self.cache_hits),
            "cache_misses": int(self.cache_misses),
            "evictions": int(self.evictions),
            "hit_rate": float(self.cache_hits / total) if total else 0.0,
        }


@recordclass(('rank', 'L', 'parity', 'channel_multiset', 'coupling_scheme', 'channel_tuple', 'basis_handle'), frozen = True)
class FeatureCacheKey:
    """Key for a numerical feature cache entry."""
    coupling_scheme = "balanced"
    channel_tuple = ()
    basis_handle = None

    @property
    def hash_key(self):
        if self.basis_handle is not None:
            return _hash_parts(self.basis_handle.sector, int(self.basis_handle.basis_index), int(self.parity))
        channels = self.channel_tuple or tuple(sorted(self.channel_multiset))
        return _hash_parts(self.rank, self.L, self.parity, channels, self.coupling_scheme)


@recordclass(('key', 'feature_values', 'metadata', 'creation_time', 'access_count'))
class FeatureCachedEntry:
    """Cached numerical feature vector with provenance metadata."""
    creation_time = 0.0
    access_count = 0

    def __post_init__(self):
        if self.creation_time == 0.0:
            self.creation_time = time.time()


class FeatureCache:
    """LRU cache for numerical feature values derived from magnetic expansions."""

    def __init__(self, max_size = 500, cg_cache = None):
        self.max_size = int(max_size)
        self._cache = {}
        self._access_order = []
        self.cg_cache = cg_cache
        self.cache_hits = 0
        self.cache_misses = 0
        self.evictions = 0
        self._product_engines = {}

    def _engine_for_tree_type(self, tree_type):
        engine = self._product_engines.get(str(tree_type))
        if engine is None:
            engine = _build_exact_product_engine(str(tree_type))
            self._product_engines[str(tree_type)] = engine
        return engine

    def clear(self):
        self._cache.clear()
        self._access_order.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.evictions = 0

    def _evict_if_needed(self):
        while len(self._cache) > self.max_size and self._access_order:
            self._cache.pop(self._access_order.pop(0), None)
            self.evictions += 1

    def _touch(self, key):
        try:
            self._access_order.remove(key)
        except ValueError:
            pass
        self._access_order.append(key)

    @staticmethod
    def _parity(l_tuple):
        return -1 if sum(int(l) for l in l_tuple) % 2 else 1

    def _key_for_label(self, label):
        label = normalize_compact_label(label)
        channel_tuple = tuple(zip(label.n_tuple, label.l_tuple))
        handle = None
        if label.basis_key:
            handle = self._engine_for_tree_type(label.tree_type).feature_space(
                label.n_tuple,
                label.l_tuple,
                label.L_R,
            ).handle_for_label(label)
        return FeatureCacheKey(
            rank=label.rank,
            L=label.L_R,
            parity=self._parity(label.l_tuple),
            channel_multiset=frozenset(channel_tuple),
            channel_tuple=channel_tuple,
            coupling_scheme=label.tree_type,
            basis_handle=handle,
        )

    def get_feature(self, label):
        """Return a deterministic numerical summary of the exact expansion."""
        label = normalize_compact_label(label)
        key = self._key_for_label(label)
        hash_key = key.hash_key
        if hash_key in self._cache:
            entry = self._cache[hash_key]
            entry.access_count += 1
            self.cache_hits += 1
            self._touch(hash_key)
            return entry.feature_values
        self.cache_misses += 1
        values = self._compute_feature_values(label)
        self._cache[hash_key] = FeatureCachedEntry(
            key=key,
            feature_values=values,
            metadata={"label": label, "uses_cg_cache": self.cg_cache is not None},
        )
        self._touch(hash_key)
        self._evict_if_needed()
        return values

    def _compute_feature_values(self, label):
        expansion = self._numeric_expansion_for_label(label)
        numeric = expansion is not None
        if expansion is None:
            cache = self.cg_cache or ClebschGordanCache(max_size=0)
            expansion = cache.get_cg_expansion(label)
        rows = []
        for M, block in sorted(expansion.items()):
            if numeric:
                norm_sq = sum(abs(complex(c)) ** 2 for c in block.values())
            else:
                norm_sq = sum(self._coefficient_norm_squared_float(c) for c in block.values())
            rows.append((float(M), float(norm_sq), float(len(block))))
        return np.asarray(rows, dtype=float)

    @staticmethod
    def _coefficient_norm_squared_float(coeff):
        if isinstance(coeff, ExactRadical):
            value = complex(coeff.evalf())
            return float((value.conjugate() * value).real)
        try:
            native = exact_scalar(coeff)
        except (TypeError, ValueError):
            try:
                value = complex(coeff)
                return float((value.conjugate() * value).real)
            except (TypeError, ValueError):
                return float(_sympy().N(_sympy().conjugate(coeff) * coeff))
        value = complex(native.evalf())
        return float((value.conjugate() * value).real)

    @staticmethod
    def _numeric_expansion_for_label(label):
        label = normalize_compact_label(label)
        if label.basis_key:
            return None
        try:
            key = raw_tree_key(label.l_tuple, label.internal_Ls, label.tree_type)
        except ValueError:
            return None
        return expand_tree_key_numeric(key)

    def get_cache_stats(self):
        total = self.cache_hits + self.cache_misses
        return {
            "feature_cache_size": len(self._cache),
            "cache_hits": int(self.cache_hits),
            "cache_misses": int(self.cache_misses),
            "evictions": int(self.evictions),
            "hit_rate": float(self.cache_hits / total) if total else 0.0,
            "has_cg_cache": self.cg_cache is not None,
        }


class DualCacheSystem:
    """Pair of symbolic CG and numerical feature caches."""

    def __init__(self, cg_max_size = 1000, feature_max_size = 500):
        self.cg_cache = ClebschGordanCache(max_size=cg_max_size)
        self.feature_cache = FeatureCache(max_size=feature_max_size, cg_cache=self.cg_cache)
        self.creation_time = time.time()

    def clear(self):
        self.cg_cache.clear()
        self.feature_cache.clear()

    def get_cg_expansion(self, label):
        return self.cg_cache.get_cg_expansion(label)

    def get_feature(self, label):
        return self.feature_cache.get_feature(label)

    def get_stats(self):
        return {
            "cg_cache": self.cg_cache.get_cache_stats(),
            "feature_cache": self.feature_cache.get_cache_stats(),
            "uptime": float(time.time() - self.creation_time),
        }
