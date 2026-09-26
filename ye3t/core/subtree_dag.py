
"""Bounded subtree caches for exact and numeric CG recursion."""

from collections import OrderedDict
from functools import lru_cache
import os

from ye3t.core.basis.labels import LeafLabel, NodeLabel, SymBlockLabel
from ye3t.core.basis.tree import RawLeaf, get_tree_factory
from ye3t.core.basis.validation import validate_tree_type
from ye3t.core.cg import cg_exact as cg_exact_native, cg_numeric as cg_numeric_native, cg_numeric_integer
from ye3t.exact_scalars import ExactRadical

SymExpr = object
MagneticTuple = tuple
SubtreeKey = tuple
ExactExpansion = dict
NumericExpansion = dict


def _occupancy_expansion_to_m_vectors(*args, **kwargs):
    from ye3t.core.basis.homogeneous import occupancy_expansion_to_m_vectors

    return occupancy_expansion_to_m_vectors(*args, **kwargs)


def _occupancy_expansion_to_m_vectors_numeric(*args, **kwargs):
    from ye3t.core.basis.homogeneous import occupancy_expansion_to_m_vectors_numeric

    return occupancy_expansion_to_m_vectors_numeric(*args, **kwargs)


def _thaw_magnetic_expansion(*args, **kwargs):
    from ye3t.core.basis.homogeneous import thaw_magnetic_expansion

    return thaw_magnetic_expansion(*args, **kwargs)


def _thaw_numeric_magnetic_expansion(*args, **kwargs):
    from ye3t.core.basis.homogeneous import thaw_numeric_magnetic_expansion

    return thaw_numeric_magnetic_expansion(*args, **kwargs)


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


_MAX_SUBTREE_RANK = _env_int("YE3T_SUBTREE_CACHE_MAX_RANK", 12, "gne3_SUBTREE_CACHE_MAX_RANK")
_MAX_CACHE_ENTRIES = _env_int("YE3T_SUBTREE_CACHE_MAX_ENTRIES", 4096, "gne3_SUBTREE_CACHE_MAX_ENTRIES")
_MAX_TOTAL_TERMS = _env_int("YE3T_SUBTREE_CACHE_MAX_TOTAL_TERMS", 200000, "gne3_SUBTREE_CACHE_MAX_TOTAL_TERMS")
_MAX_ENTRY_TERMS = _env_int("YE3T_SUBTREE_CACHE_MAX_ENTRY_TERMS", 20000, "gne3_SUBTREE_CACHE_MAX_ENTRY_TERMS")


@lru_cache(maxsize=None)
def cg_exact(j1, m1, j2, m2, j3, m3):
    values = (j1, m1, j2, m2, j3, m3)
    if all(int(x) == x for x in values):
        return cg_exact_native(*(int(x) for x in values))
    return cg_exact_native(j1, m1, j2, m2, j3, m3)


@lru_cache(maxsize=None)
def cg_numeric(j1, m1, j2, m2, j3, m3):
    values = (j1, m1, j2, m2, j3, m3)
    if all(int(x) == x for x in values):
        return cg_numeric_integer(*(int(x) for x in values))
    return cg_numeric_native(j1, m1, j2, m2, j3, m3)


@lru_cache(maxsize=None)
def _raw_tree_skeleton(l_values, tree_type):
    tree_type = validate_tree_type(tree_type)
    tree_factory = get_tree_factory(tree_type)
    return tree_factory.build_raw_tree([0] * len(l_values), list(l_values))


@lru_cache(maxsize=None)
def _raw_tree_key_cached(l_values, internal_Ls, tree_type):
    raw_tree = _raw_tree_skeleton(l_values, tree_type)
    if isinstance(raw_tree, RawLeaf):
        if len(internal_Ls) not in {0, 1}:
            raise ValueError("Rank-1 compact labels must use zero or one internal_L entry.")
        if len(internal_Ls) == 1 and int(internal_Ls[0]) != int(raw_tree.l):
            raise ValueError("Rank-1 compact-label internal_L must equal the leaf angular momentum.")
        return ("leaf", int(raw_tree.l))

    def build_key(node, pos = 0):
        if isinstance(node, RawLeaf):
            return ("leaf", int(node.l)), pos
        left_key, pos = build_key(node.left, pos)
        right_key, pos = build_key(node.right, pos)
        root_L = int(internal_Ls[pos])
        pos += 1
        return ("node", left_key, right_key, root_L), pos

    coupled_key, consumed = build_key(raw_tree, 0)
    if consumed != len(internal_Ls):
        raise ValueError("internal_Ls tuple length does not match the canonical coupling tree.")
    return coupled_key


def raw_tree_key(l_values, internal_Ls, tree_type):
    """Return the canonical coupled-tree key for one compact label."""
    return _raw_tree_key_cached(tuple(int(l) for l in l_values), tuple(int(L) for L in internal_Ls), str(tree_type))


@lru_cache(maxsize=None)
def structured_label_key(label_obj):
    """Return the subtree key for one structured exact label object."""
    if isinstance(label_obj, LeafLabel):
        return ("leaf", int(label_obj.l))
    if isinstance(label_obj, SymBlockLabel):
        if label_obj.occupancy_expansion_by_M and not label_obj.representative_internal_Ls:
            raise ValueError("Exact homogeneous labels without representative tuples do not have a raw coupled-tree key.")
        return raw_tree_key(label_obj.l_sequence(), tuple(label_obj.representative_internal_Ls), label_obj.tree_type)
    if isinstance(label_obj, NodeLabel):
        return (
            "node",
            structured_label_key(label_obj.left),
            structured_label_key(label_obj.right),
            int(label_obj.L),
        )
    raise TypeError(f"Unsupported structured label type: {type(label_obj)!r}")


@lru_cache(maxsize=None)
def expand_structured_label_exact(label_obj):
    """Return the exact magnetic-basis expansion of one structured label."""
    if isinstance(label_obj, LeafLabel):
        return {m: {(m,): ExactRadical.rational(1)} for m in range(-int(label_obj.l), int(label_obj.l) + 1)}
    if isinstance(label_obj, SymBlockLabel):
        if label_obj.occupancy_expansion_by_M:
            frozen = _occupancy_expansion_to_m_vectors(
                int(label_obj.l),
                int(label_obj.k_b),
                tuple(label_obj.occupancy_expansion_by_M),
            )
            return _thaw_magnetic_expansion(frozen)
        return expand_tree_key_exact(structured_label_key(label_obj))
    if isinstance(label_obj, NodeLabel):
        left_map = expand_structured_label_exact(label_obj.left)
        right_map = expand_structured_label_exact(label_obj.right)
        left_total = int(label_obj.left.total_L())
        right_total = int(label_obj.right.total_L())
        out = {}
        for M1, vec1 in left_map.items():
            for M2, vec2 in right_map.items():
                M = int(M1 + M2)
                if abs(M) > int(label_obj.L):
                    continue
                coeff = cg_exact(left_total, int(M1), right_total, int(M2), int(label_obj.L), M)
                if coeff == 0:
                    continue
                slot = out.setdefault(M, {})
                for ms1, c1 in vec1.items():
                    for ms2, c2 in vec2.items():
                        key_ms = ms1 + ms2
                        slot[key_ms] = slot.get(key_ms, ExactRadical.rational(0)) + c1 * c2 * coeff
        return out
    raise TypeError(f"Unsupported structured label type: {type(label_obj)!r}")


@lru_cache(maxsize=None)
def expand_structured_label_numeric(label_obj):
    """Return the numeric magnetic-basis expansion of one structured label."""
    if isinstance(label_obj, LeafLabel):
        return {m: {(m,): 1.0 + 0.0j} for m in range(-int(label_obj.l), int(label_obj.l) + 1)}
    if isinstance(label_obj, SymBlockLabel):
        if label_obj.occupancy_expansion_by_M:
            frozen = _occupancy_expansion_to_m_vectors_numeric(
                int(label_obj.l),
                int(label_obj.k_b),
                tuple(label_obj.occupancy_expansion_by_M),
            )
            return _thaw_numeric_magnetic_expansion(frozen)
        return expand_tree_key_numeric(structured_label_key(label_obj))
    if isinstance(label_obj, NodeLabel):
        left_map = expand_structured_label_numeric(label_obj.left)
        right_map = expand_structured_label_numeric(label_obj.right)
        left_total = int(label_obj.left.total_L())
        right_total = int(label_obj.right.total_L())
        out = {}
        for M1, vec1 in left_map.items():
            for M2, vec2 in right_map.items():
                M = int(M1 + M2)
                if abs(M) > int(label_obj.L):
                    continue
                coeff = cg_numeric(left_total, int(M1), right_total, int(M2), int(label_obj.L), M)
                if abs(coeff) < 1e-14:
                    continue
                slot = out.setdefault(M, {})
                for ms1, c1 in vec1.items():
                    for ms2, c2 in vec2.items():
                        key_ms = ms1 + ms2
                        slot[key_ms] = slot.get(key_ms, 0.0 + 0.0j) + c1 * c2 * coeff
        return out
    raise TypeError(f"Unsupported structured label type: {type(label_obj)!r}")


@lru_cache(maxsize=None)
def subtree_leaf_count(key):
    if key[0] == "leaf":
        return 1
    return int(subtree_leaf_count(key[1]) + subtree_leaf_count(key[2]))


@lru_cache(maxsize=None)
def subtree_root_L(key):
    if key[0] == "leaf":
        return int(key[1])
    return int(key[3])


def _term_count(expansion):
    return sum(len(block) for block in expansion.values())


class _BoundedSubtreeExpansionCache:
    def __init__(
        self,
        *,
        max_rank,
        max_entries,
        max_total_terms,
        max_entry_terms,
    ):
        self.max_rank = int(max_rank)
        self.max_entries = int(max_entries)
        self.max_total_terms = int(max_total_terms)
        self.max_entry_terms = int(max_entry_terms)
        self._items = OrderedDict()
        self._total_terms = 0
        self._hit_count = 0
        self._miss_count = 0

    def get(self, key):
        item = self._items.get(key)
        if item is None:
            self._miss_count += 1
            return None
        self._items.move_to_end(key)
        self._hit_count += 1
        return item[0]

    def put(self, key, value):
        if self.max_entries == 0 or self.max_total_terms == 0:
            return
        if self.max_rank >= 0 and subtree_leaf_count(key) > self.max_rank:
            return
        terms = _term_count(value)
        if self.max_entry_terms >= 0 and terms > self.max_entry_terms:
            return
        existing = self._items.pop(key, None)
        if existing is not None:
            self._total_terms -= existing[1]
        self._items[key] = (value, terms)
        self._total_terms += terms
        self._evict()

    def _evict(self):
        while self._items:
            too_many_entries = self.max_entries >= 0 and len(self._items) > self.max_entries
            too_many_terms = self.max_total_terms >= 0 and self._total_terms > self.max_total_terms
            if not too_many_entries and not too_many_terms:
                break
            _, (_, terms) = self._items.popitem(last=False)
            self._total_terms -= int(terms)

    def clear(self):
        self._items.clear()
        self._total_terms = 0
        self._hit_count = 0
        self._miss_count = 0

    def stats(self):
        return {
            "max_subtree_rank": int(self.max_rank),
            "max_entries": int(self.max_entries),
            "max_total_terms": int(self.max_total_terms),
            "max_entry_terms": int(self.max_entry_terms),
            "entry_count": int(len(self._items)),
            "cached_term_count": int(self._total_terms),
            "hit_count": int(self._hit_count),
            "miss_count": int(self._miss_count),
        }


_EXACT_SUBTREE_CACHE = _BoundedSubtreeExpansionCache(
    max_rank=_MAX_SUBTREE_RANK,
    max_entries=_MAX_CACHE_ENTRIES,
    max_total_terms=_MAX_TOTAL_TERMS,
    max_entry_terms=_MAX_ENTRY_TERMS,
)
_NUMERIC_SUBTREE_CACHE = _BoundedSubtreeExpansionCache(
    max_rank=_MAX_SUBTREE_RANK,
    max_entries=_MAX_CACHE_ENTRIES,
    max_total_terms=_MAX_TOTAL_TERMS,
    max_entry_terms=_MAX_ENTRY_TERMS,
)


def expand_tree_key_exact(key):
    cached = _EXACT_SUBTREE_CACHE.get(key)
    if cached is not None:
        return cached
    if key[0] == "leaf":
        l = int(key[1])
        out = {m: {(m,): ExactRadical.rational(1)} for m in range(-l, l + 1)}
        _EXACT_SUBTREE_CACHE.put(key, out)
        return out

    _, left_key, right_key, root_L = key
    left_map = expand_tree_key_exact(left_key)
    right_map = expand_tree_key_exact(right_key)
    left_total = subtree_root_L(left_key)
    right_total = subtree_root_L(right_key)
    out = {}
    for M1, vec1 in left_map.items():
        for M2, vec2 in right_map.items():
            M = int(M1 + M2)
            if abs(M) > int(root_L):
                continue
            coeff = cg_exact(left_total, int(M1), right_total, int(M2), int(root_L), M)
            if coeff == 0:
                continue
            slot = out.setdefault(M, {})
            for ms1, c1 in vec1.items():
                for ms2, c2 in vec2.items():
                    key_ms = ms1 + ms2
                    slot[key_ms] = slot.get(key_ms, ExactRadical.rational(0)) + c1 * c2 * coeff
    _EXACT_SUBTREE_CACHE.put(key, out)
    return out


def expand_tree_key_numeric(key):
    cached = _NUMERIC_SUBTREE_CACHE.get(key)
    if cached is not None:
        return cached
    if key[0] == "leaf":
        l = int(key[1])
        out = {m: {(m,): 1.0 + 0.0j} for m in range(-l, l + 1)}
        _NUMERIC_SUBTREE_CACHE.put(key, out)
        return out

    _, left_key, right_key, root_L = key
    left_map = expand_tree_key_numeric(left_key)
    right_map = expand_tree_key_numeric(right_key)
    left_total = subtree_root_L(left_key)
    right_total = subtree_root_L(right_key)
    out = {}
    for M1, vec1 in left_map.items():
        for M2, vec2 in right_map.items():
            M = int(M1 + M2)
            if abs(M) > int(root_L):
                continue
            coeff = cg_numeric(left_total, int(M1), right_total, int(M2), int(root_L), M)
            if abs(coeff) < 1e-14:
                continue
            slot = out.setdefault(M, {})
            for ms1, c1 in vec1.items():
                for ms2, c2 in vec2.items():
                    key_ms = ms1 + ms2
                    slot[key_ms] = slot.get(key_ms, 0.0 + 0.0j) + c1 * c2 * coeff
    _NUMERIC_SUBTREE_CACHE.put(key, out)
    return out


def clear_subtree_dag_caches():
    _EXACT_SUBTREE_CACHE.clear()
    _NUMERIC_SUBTREE_CACHE.clear()


def exact_subtree_cache_stats():
    return _EXACT_SUBTREE_CACHE.stats()


def numeric_subtree_cache_stats():
    return _NUMERIC_SUBTREE_CACHE.stats()
