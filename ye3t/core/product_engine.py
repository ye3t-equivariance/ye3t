
r"""Exact product expansions and primitive quotients in the canonical basis.

The engine expands exact labels into the uncoupled magnetic basis, forms exact
products, permutes leaves into canonical order, and projects back into the
package's exact basis. Native exact kernels handle bounded integer/rational
subproblems where available; algebraic exact matrix reference paths use the
optional symbolic backend.
"""

from collections import OrderedDict
from functools import lru_cache
from itertools import combinations
import os

from ye3t.core.basis import YE3TBasisLabeler
from ye3t.core.basis.sector import ExactBasisHandle, ExactSectorSignature
from ye3t.core.basis.validation import canonicalize_leaf_quantum_numbers, validate_tree_type
from ye3t.core.labels import CompactLabel, normalize_compact_label
from ye3t.core.product_descriptors import ExactProductColumnDescriptor
from ye3t.core.subtree_dag import (
    cg_exact,
    expand_structured_label_exact,
    expand_tree_key_exact,
    raw_tree_key,
    structured_label_key,
)
from ye3t.exact_linalg import exact_algebraic_rank_or_none
from ye3t.exact_scalars import ExactRadical, exact_scalar
from ye3t._record import recordclass


def _sympy():
    from ye3t._optional_sympy import sp

    return sp

SymExpr = object
MagneticTuple = tuple


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


class _BoundedExactResultCache:
    def __init__(self, max_entries):
        self.max_entries = int(max_entries)
        self._items = OrderedDict()

    def get(self, key):
        item = self._items.get(key)
        if item is None:
            return None
        self._items.move_to_end(key)
        return item

    def put(self, key, value):
        if self.max_entries <= 0:
            return
        self._items.pop(key, None)
        self._items[key] = value
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)


@recordclass(('rows', 'entries'), frozen = True)
class ExactSparseColumn:
    """Sparse exact column in canonical basis coordinates."""

    def to_matrix(self):
        return _sympy().SparseMatrix(int(self.rows), 1, {(int(row), 0): value for row, value in self.entries})


@recordclass(('rows', 'cols', 'columns'), frozen = True)
class ExactSparseOperator:
    """Sparse exact linear operator stored column-by-column."""

    def to_matrix(self):
        data = {}
        for col_idx, column in enumerate(self.columns):
            for row_idx, value in column.entries:
                data[(int(row_idx), int(col_idx))] = value
        return _sympy().SparseMatrix(int(self.rows), int(self.cols), data)


class _ExactIndependentColumnAccumulator:
    """Keep an exact independent column set and stop early at a target rank."""

    def __init__(self, rows, *, stop_rank = 0):
        self.rows = int(rows)
        self.stop_rank = min(int(stop_rank), self.rows) if int(stop_rank) > 0 else 0
        self._basis = None
        self._cols = []
        self._native_exact_cols = []
        self._seen_signatures = set()

    @property
    def rank(self):
        return len(self._cols)

    def reached_target(self):
        return self.stop_rank > 0 and self.rank >= self.stop_rank

    def matrix(self):
        if not self._cols:
            return _sympy().zeros(self.rows, 0)
        if self._basis is None:
            self._basis = _sympy().Matrix.hstack(*(self._symbolic_column_matrix(column) for column in self._cols))
        return self._basis

    @staticmethod
    def _symbolic_column_matrix(column):
        if isinstance(column, ExactSparseColumn):
            return column.to_matrix()
        return _sympy().SparseMatrix(column)

    @staticmethod
    def _native_exact_value_or_none(value):
        value_module = str(getattr(type(value), "__module__", ""))
        if value_module.startswith("sympy"):
            return None
        try:
            return value if isinstance(value, ExactRadical) else exact_scalar(value)
        except (TypeError, ValueError):
            return None

    def _native_exact_column_from_entries(self, entries):
        column = [ExactRadical.rational(0) for _ in range(self.rows)]
        for row, value in entries:
            row = int(row)
            if row < 0 or row >= self.rows:
                raise ValueError(f"Column row index {row} is outside {self.rows} rows.")
            native = self._native_exact_value_or_none(value)
            if native is None:
                return None
            column[row] = native
        return tuple(column)

    def _native_exact_column_from_dense(self, column):
        values = []
        for row in range(self.rows):
            try:
                value = column[row, 0]
            except (TypeError, IndexError):
                value = column[row]
            native = self._native_exact_value_or_none(value)
            if native is None:
                return None
            values.append(native)
        return tuple(values)

    def _native_exact_rank_with_column(self, native_column):
        rows = tuple(
            tuple(column[row] for column in self._native_exact_cols + [native_column])
            for row in range(self.rows)
        )
        return exact_algebraic_rank_or_none(rows)

    def _append_native_exact_column(self, native_column, symbolic_column):
        self._native_exact_cols.append(native_column)
        self._cols.append(symbolic_column)
        self._basis = None

    def try_add_sparse(self, column):
        signature = ExactProductExpansionEngine._column_signature_from_entries(column.entries)
        if signature is None or signature in self._seen_signatures:
            return False
        native_column = self._native_exact_column_from_entries(column.entries)
        if native_column is not None and len(self._native_exact_cols) == self.rank:
            native_rank = self._native_exact_rank_with_column(native_column)
            if native_rank is not None and (self.rank == 0 or native_rank > self.rank):
                self._append_native_exact_column(native_column, column)
                self._seen_signatures.add(signature)
                return True
            if native_rank is not None:
                self._seen_signatures.add(signature)
                return False
        column_matrix = column.to_matrix()
        if self.rank == 0:
            self._basis = column_matrix
            self._cols.append(column_matrix)
            self._seen_signatures.add(signature)
            return True
        if self._basis is None:
            self._basis = _sympy().Matrix.hstack(*(self._symbolic_column_matrix(existing) for existing in self._cols))
        augmented = self._basis.row_join(column_matrix)
        if int(augmented.rank(simplify=False)) <= self.rank:
            self._seen_signatures.add(signature)
            return False
        self._basis = augmented
        self._cols.append(column_matrix)
        self._native_exact_cols.clear()
        self._seen_signatures.add(signature)
        return True

    def try_add(self, column):
        signature = ExactProductExpansionEngine._column_signature(column)
        if signature is None or signature in self._seen_signatures:
            return False
        native_column = self._native_exact_column_from_dense(column)
        if native_column is not None and len(self._native_exact_cols) == self.rank:
            native_rank = self._native_exact_rank_with_column(native_column)
            if native_rank is not None and (self.rank == 0 or native_rank > self.rank):
                self._append_native_exact_column(native_column, column)
                self._seen_signatures.add(signature)
                return True
            if native_rank is not None:
                self._seen_signatures.add(signature)
                return False
        column_matrix = _sympy().SparseMatrix(column)
        if self.rank == 0:
            self._basis = column_matrix
            self._cols.append(column_matrix)
            self._seen_signatures.add(signature)
            return True
        if self._basis is None:
            self._basis = _sympy().Matrix.hstack(*(self._symbolic_column_matrix(existing) for existing in self._cols))
        augmented = self._basis.row_join(column_matrix)
        if int(augmented.rank(simplify=False)) <= self.rank:
            self._seen_signatures.add(signature)
            return False
        self._basis = augmented
        self._cols.append(column_matrix)
        self._native_exact_cols.clear()
        self._seen_signatures.add(signature)
        return True


_GLOBAL_PRODUCT_ENGINE_CACHE_MAX_ENTRIES = _env_int("YE3T_PRODUCT_ENGINE_GLOBAL_CACHE_MAX_ENTRIES", 64, "gne3_PRODUCT_ENGINE_GLOBAL_CACHE_MAX_ENTRIES")
_GLOBAL_FEATURE_SPACE_CACHE = _BoundedExactResultCache(_GLOBAL_PRODUCT_ENGINE_CACHE_MAX_ENTRIES)
_GLOBAL_AVAILABLE_L_CACHE = _BoundedExactResultCache(_GLOBAL_PRODUCT_ENGINE_CACHE_MAX_ENTRIES)
_GLOBAL_EXPAND_PRODUCT_CACHE = _BoundedExactResultCache(_GLOBAL_PRODUCT_ENGINE_CACHE_MAX_ENTRIES)
_GLOBAL_PRODUCT_MATRIX_CACHE = _BoundedExactResultCache(_GLOBAL_PRODUCT_ENGINE_CACHE_MAX_ENTRIES)
_GLOBAL_PRODUCT_OPERATOR_CACHE = _BoundedExactResultCache(_GLOBAL_PRODUCT_ENGINE_CACHE_MAX_ENTRIES)
_GLOBAL_TARGET_GRAM_INVERSE_CACHE = _BoundedExactResultCache(_GLOBAL_PRODUCT_ENGINE_CACHE_MAX_ENTRIES)
_GLOBAL_GENERATED_COLUMNS_CACHE = _BoundedExactResultCache(_GLOBAL_PRODUCT_ENGINE_CACHE_MAX_ENTRIES)
_GLOBAL_PRIMITIVE_QUOTIENT_CACHE = _BoundedExactResultCache(_GLOBAL_PRODUCT_ENGINE_CACHE_MAX_ENTRIES)
_GLOBAL_FACTORIZABLE_COLUMNS_CACHE = _BoundedExactResultCache(_GLOBAL_PRODUCT_ENGINE_CACHE_MAX_ENTRIES)
_GLOBAL_INDEPENDENT_SUBSPACE_CACHE = _BoundedExactResultCache(_GLOBAL_PRODUCT_ENGINE_CACHE_MAX_ENTRIES)
_GLOBAL_SUBTREE_KEY_MAP_CACHE = _BoundedExactResultCache(_GLOBAL_PRODUCT_ENGINE_CACHE_MAX_ENTRIES)
_GLOBAL_EXACT_VECTOR_MAP_CACHE = _BoundedExactResultCache(_GLOBAL_PRODUCT_ENGINE_CACHE_MAX_ENTRIES)


@recordclass(('nin', 'lin', 'L_R', 'labels', 'signature', 'basis_handles'), frozen = True)
class ExactFeatureSpace:
    r"""Exact basis sector for one canonical ``(n, l, L_R)`` pattern."""
    signature = None
    basis_handles = tuple()

    @property
    def rank(self):
        return len(self.nin)

    @property
    def dim(self):
        return len(self.labels)

    def handle_for_index(self, basis_index):
        if 0 <= int(basis_index) < len(self.basis_handles):
            return self.basis_handles[int(basis_index)]
        return None

    def handle_for_label(self, label):
        try:
            basis_index = self.labels.index(normalize_compact_label(label))
        except ValueError:
            return None
        return self.handle_for_index(int(basis_index))


@recordclass(('label_left', 'label_right', 'L_out', 'target_space', 'coefficients', 'coefficients_by_M', 'max_M_inconsistency'), frozen = True)
class ProductExpansionResult:
    r"""Exact expansion of one coupled product into the target basis."""

@recordclass(('target_space', 'generated_rank', 'primitive_rank', 'generated_basis_indices', 'primitive_basis_indices'), frozen = True)
class PrimitiveQuotientSummary:
    r"""Summary of the exact lower-generated quotient for one sector."""

@recordclass(('target_space', 'factorization_policy', 'primitive_rank', 'primitive_basis_indices', 'primitive_labels', 'factorizable_rank', 'factorizable_from_primitive_rank', 'factorizable_missing_rank', 'factorizable_basis_indices', 'product_column_count', 'reconstructed_rank', 'missing_rank', 'missing_basis_indices'), frozen = True)
class PrimitiveGeneratorReconstructionSummary:
    """Exact reconstruction audit from primitive generators."""

@recordclass(('nin', 'lin', 'L_R', 'factorization_policy', 'space_dim', 'primitive_rank', 'factorizable_rank', 'factorizable_from_primitive_rank', 'factorizable_missing_rank', 'reconstructed_rank', 'missing_rank', 'direct_space_seconds', 'primitive_quotient_seconds', 'primitive_reconstruction_seconds'), frozen = True)
class PrimitiveReconstructionBenchmark:
    """Runtime comparison of direct, quotient, and primitive-generator routes."""


@recordclass(('target_space', 'factorization_policy', 'generator_ranks', 'generator_Ls', 'max_generator_rank', 'max_generator_L', 'max_recoupling_L', 'include_target_primitive', 'raw_product_column_count', 'independent_product_rank', 'independent_product_column_indices', 'product_descriptors', 'independent_product_descriptors', 'target_basis_indices', 'missing_rank', 'missing_basis_indices', 'coordinate_matrix'), frozen = True)
class IndependentDecomposableProductSubspace:
    """Exact independent product image generated from primitive representatives."""

class ExactProductExpansionEngine:
    """Exact product-expansion engine for the canonical coupled-tree basis."""

    def __init__(self, tree_type = "balanced"):
        self.tree_type = validate_tree_type(tree_type)

    # ------------------------------------------------------------------
    # Exact homogeneous feature spaces
    # ------------------------------------------------------------------
    @staticmethod
    def _canonical_pairs(nin, lin):
        return canonicalize_leaf_quantum_numbers(nin, lin)

    def feature_space(self, nin, lin, L_R):
        nin_c, lin_c = self._canonical_pairs(nin, lin)
        return self._feature_space_for_canonical_pattern(nin_c, lin_c, int(L_R))

    def sector(self, nin, lin, L_R):
        """Alias for :meth:`feature_space`."""
        return self.feature_space(nin, lin, int(L_R))

    @lru_cache(maxsize=None)
    def _feature_space_for_canonical_pattern(self, nin, lin, L_R):
        cache_key = ("feature_space", str(self.tree_type), tuple(nin), tuple(lin), int(L_R))
        cached = _GLOBAL_FEATURE_SPACE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        labeler = YE3TBasisLabeler(list(nin), list(lin), strict_target_validation=False, tree_type=self.tree_type)
        sector = labeler.sector_data_for_target(int(L_R))
        if sector is not None:
            labels = tuple(normalize_compact_label(entry.compact_label) for entry in sector.entries)
            basis_handles = tuple(entry.handle for entry in sector.entries)
            structured_labels = tuple(entry.structured_label for entry in sector.entries)
            exact_vector_map = {
                label: expand_structured_label_exact(structured)
                for label, structured in zip(labels, structured_labels)
            }
            _GLOBAL_EXACT_VECTOR_MAP_CACHE.put(
                ("exact_vectors", str(self.tree_type), tuple(nin), tuple(lin), int(L_R)),
                exact_vector_map,
            )
            subtree_key_map = {}
            for label, structured in zip(labels, structured_labels):
                if getattr(label, "basis_key", tuple()):
                    continue
                subtree_key_map[label] = structured_label_key(structured)
            if subtree_key_map:
                _GLOBAL_SUBTREE_KEY_MAP_CACHE.put(
                    ("subtree_keys", str(self.tree_type), tuple(nin), tuple(lin), int(L_R)),
                    subtree_key_map,
                )
            signature = sector.signature
        else:
            labels = tuple(normalize_compact_label(x) for x in labeler.compact_labels_for_target(int(L_R)))
            signature = ExactSectorSignature(
                nin=tuple(nin),
                lin=tuple(lin),
                L_R=int(L_R),
                tree_type=str(self.tree_type),
            )
            basis_handles = tuple(
                ExactBasisHandle(sector=signature, basis_index=int(idx))
                for idx in range(len(labels))
            )
        result = ExactFeatureSpace(
            nin=nin,
            lin=lin,
            L_R=int(L_R),
            labels=labels,
            signature=signature,
            basis_handles=basis_handles,
        )
        _GLOBAL_FEATURE_SPACE_CACHE.put(cache_key, result)
        return result

    def available_L_for_pattern(self, nin, lin):
        nin_c, lin_c = self._canonical_pairs(nin, lin)
        return self._available_L_for_canonical_pattern(nin_c, lin_c)

    @lru_cache(maxsize=None)
    def _available_L_for_canonical_pattern(self, nin, lin):
        cache_key = ("available_L", str(self.tree_type), tuple(nin), tuple(lin))
        cached = _GLOBAL_AVAILABLE_L_CACHE.get(cache_key)
        if cached is not None:
            return cached
        labeler = YE3TBasisLabeler(list(nin), list(lin), strict_target_validation=False, tree_type=self.tree_type)
        result = tuple(sorted(int(k) for k, v in labeler.counts_by_L().items() if int(v) > 0))
        _GLOBAL_AVAILABLE_L_CACHE.put(cache_key, result)
        return result

    # ------------------------------------------------------------------
    # Exact coupled-tree expansion in magnetic basis
    # ------------------------------------------------------------------
    @lru_cache(maxsize=None)
    def _m_vectors(self, label):
        """Return exact uncoupled magnetic-basis vectors for each root ``M``."""
        label = normalize_compact_label(label)
        sector_key = ("exact_vectors", str(self.tree_type), tuple(label.n_tuple), tuple(label.l_tuple), int(label.L_R))
        cached = _GLOBAL_EXACT_VECTOR_MAP_CACHE.get(sector_key)
        if cached is None:
            self._feature_space_for_canonical_pattern(tuple(label.n_tuple), tuple(label.l_tuple), int(label.L_R))
            cached = _GLOBAL_EXACT_VECTOR_MAP_CACHE.get(sector_key)
        if cached is not None:
            exact_vectors = cached.get(label)
            if exact_vectors is not None:
                return exact_vectors
        subtree_key = self._subtree_key_for_label(label)
        return expand_tree_key_exact(subtree_key)

    @lru_cache(maxsize=None)
    def _subtree_key_for_label(self, label):
        label = normalize_compact_label(label)
        sector_key = ("subtree_keys", str(self.tree_type), tuple(label.n_tuple), tuple(label.l_tuple), int(label.L_R))
        cached = _GLOBAL_SUBTREE_KEY_MAP_CACHE.get(sector_key)
        if cached is None:
            self._feature_space_for_canonical_pattern(tuple(label.n_tuple), tuple(label.l_tuple), int(label.L_R))
            cached = _GLOBAL_SUBTREE_KEY_MAP_CACHE.get(sector_key)
        if cached is not None:
            subtree_key = cached.get(label)
            if subtree_key is not None:
                return subtree_key
        return raw_tree_key(label.l_tuple, label.internal_Ls, label.tree_type)

    @staticmethod
    def _combine_and_sort_leaves(label_left, label_right):
        """Merge two leaf lists and return the canonical-sort permutation."""
        items = [
            (int(n), int(l), 0, i) for i, (n, l) in enumerate(zip(label_left.n_tuple, label_left.l_tuple))
        ] + [
            (int(n), int(l), 1, i) for i, (n, l) in enumerate(zip(label_right.n_tuple, label_right.l_tuple))
        ]
        sorted_items = sorted(items)
        # permutation[new_position] = old_position in the concatenated tuple
        concat_items = items
        old_index = {item: i for i, item in enumerate(concat_items)}
        permutation = tuple(old_index[item] for item in sorted_items)
        nin = tuple(item[0] for item in sorted_items)
        lin = tuple(item[1] for item in sorted_items)
        return nin, lin, permutation

    @staticmethod
    def _permute_ms(ms, permutation):
        return tuple(ms[idx] for idx in permutation)

    @lru_cache(maxsize=None)
    def _product_vector_for_M(
        self,
        label_left,
        label_right,
        L_out,
        M_out,
    ):
        r"""Construct the exact magnetic-basis vector for one coupled product.

        This evaluates the exact tensor product

        .. math::

            [B^{(L_1)}_{\alpha} \otimes B^{(L_2)}_{\beta}]_{L_{\rm out} M}

        directly in the uncoupled magnetic basis of the combined leaf list.
        """
        label_left = normalize_compact_label(label_left)
        label_right = normalize_compact_label(label_right)
        if abs(M_out) > L_out:
            raise ValueError(f"M_out={M_out} is incompatible with L_out={L_out}.")
        vec_left = self._m_vectors(label_left)
        vec_right = self._m_vectors(label_right)
        nin, lin, permutation = self._combine_and_sort_leaves(label_left, label_right)
        out = {}
        for M1, block_left in vec_left.items():
            M2 = int(M_out - M1)
            if M2 not in vec_right:
                continue
            cg = cg_exact(label_left.L_R, int(M1), label_right.L_R, int(M2), int(L_out), int(M_out))
            if cg == 0:
                continue
            for ms_left, c_left in block_left.items():
                for ms_right, c_right in vec_right[M2].items():
                    combined = ms_left + ms_right
                    permuted = self._permute_ms(combined, permutation)
                    out[permuted] = _sympy().simplify(out.get(permuted, _sympy().Integer(0)) + c_left * c_right * cg)
        return nin, lin, out

    def _target_gram_inverse(self, target_space, target_vectors, M):
        cache_key = (
            "target_gram_inverse",
            str(self.tree_type),
            tuple(target_space.nin),
            tuple(target_space.lin),
            int(target_space.L_R),
            int(M),
        )
        cached = _GLOBAL_TARGET_GRAM_INVERSE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        gram = _sympy().zeros(int(target_space.dim), int(target_space.dim))
        for left_index, left_map in enumerate(target_vectors):
            left = left_map[int(M)]
            for right_index, right_map in enumerate(target_vectors):
                right = right_map[int(M)]
                value = _sympy().Integer(0)
                for magnetic_indices, coefficient in left.items():
                    value += _sympy().conjugate(coefficient) * right.get(
                        magnetic_indices, _sympy().Integer(0)
                    )
                gram[left_index, right_index] = _sympy().simplify(value)
        if int(gram.rank(simplify=False)) != int(target_space.dim):
            raise ValueError("exact target basis has a singular magnetic Gram matrix")
        inverse = gram.inv()
        _GLOBAL_TARGET_GRAM_INVERSE_CACHE.put(cache_key, inverse)
        return inverse

    # ------------------------------------------------------------------
    # Exact product-expansion maps
    # ------------------------------------------------------------------
    def expand_product(self, label_left, label_right, L_out):
        """Expand one exact coupled product in the target exact basis."""
        return self._expand_product_cached(
            normalize_compact_label(label_left),
            normalize_compact_label(label_right),
            int(L_out),
        )

    @lru_cache(maxsize=None)
    def _expand_product_cached(
        self,
        label_left,
        label_right,
        L_out,
    ):
        r"""Expand one exact coupled product in the target basis."""
        cache_key = ("expand_product", str(self.tree_type), label_left, label_right, int(L_out))
        cached = _GLOBAL_EXPAND_PRODUCT_CACHE.get(cache_key)
        if cached is not None:
            return cached
        nin, lin, _ = self._combine_and_sort_leaves(label_left, label_right)
        target_space = self.feature_space(nin, lin, int(L_out))
        target_vectors = [self._m_vectors(lab) for lab in target_space.labels]

        coeffs_by_M = {}
        for M_out in range(-int(L_out), int(L_out) + 1):
            _, _, product_vec = self._product_vector_for_M(label_left, label_right, int(L_out), int(M_out))
            overlaps = []
            for target_map in target_vectors:
                target_vec = target_map[M_out]
                coeff = _sympy().Integer(0)
                for ms, c in target_vec.items():
                    coeff += _sympy().conjugate(c) * product_vec.get(ms, _sympy().Integer(0))
                overlaps.append(_sympy().simplify(coeff))
            gram_inverse = self._target_gram_inverse(
                target_space, target_vectors, M_out
            )
            coordinates = gram_inverse * _sympy().Matrix(overlaps)
            coeffs_by_M[M_out] = tuple(
                _sympy().simplify(value) for value in coordinates
            )

        # Multiplicity coefficients should not depend on M.
        ref_M = 0 if 0 in coeffs_by_M else next(iter(coeffs_by_M))
        ref = coeffs_by_M[ref_M]
        max_inconsistency = _sympy().Integer(0)
        for M, coeffs_M in coeffs_by_M.items():
            for a, b in zip(ref, coeffs_M):
                diff = _sympy().simplify(a - b)
                if diff != 0:
                    max_inconsistency = max(max_inconsistency, _sympy().Abs(diff))
        result = ProductExpansionResult(
            label_left=label_left,
            label_right=label_right,
            L_out=int(L_out),
            target_space=target_space,
            coefficients=tuple(_sympy().simplify(x) for x in ref),
            coefficients_by_M=coeffs_by_M,
            max_M_inconsistency=max_inconsistency,
        )
        _GLOBAL_EXPAND_PRODUCT_CACHE.put(cache_key, result)
        return result

    def product_expansion_matrix(self, space_left, space_right, L_out):
        """Return the exact product-expansion matrix for all basis pairs."""
        return self.product_expansion_operator(space_left, space_right, int(L_out)).to_matrix()

    def product_expansion_operator(self, space_left, space_right, L_out):
        """Return the exact product-expansion operator in sparse column form."""
        return self._product_expansion_operator_cached(space_left, space_right, int(L_out))

    @lru_cache(maxsize=None)
    def _product_expansion_operator_cached(
        self,
        space_left,
        space_right,
        L_out,
    ):
        """Return the exact product-expansion operator for all basis pairs."""
        cache_key = ("product_operator", str(self.tree_type), space_left, space_right, int(L_out))
        cached = _GLOBAL_PRODUCT_OPERATOR_CACHE.get(cache_key)
        if cached is not None:
            return cached
        target_nin, target_lin = self._canonical_pairs(
            tuple(space_left.nin) + tuple(space_right.nin),
            tuple(space_left.lin) + tuple(space_right.lin),
        )
        target_space = self.feature_space(target_nin, target_lin, int(L_out))
        if space_left.dim == 0 or space_right.dim == 0:
            result = ExactSparseOperator(rows=target_space.dim, cols=0, columns=tuple())
            _GLOBAL_PRODUCT_OPERATOR_CACHE.put(cache_key, result)
            return result
        columns = []
        for lab_left in space_left.labels:
            for lab_right in space_right.labels:
                res = self.expand_product(lab_left, lab_right, int(L_out))
                entries = tuple(
                    (int(row_idx), coeff)
                    for row_idx, coeff in enumerate(res.coefficients)
                    if coeff != 0
                )
                columns.append(ExactSparseColumn(rows=target_space.dim, entries=entries))
        result = ExactSparseOperator(rows=target_space.dim, cols=len(columns), columns=tuple(columns))
        _GLOBAL_PRODUCT_OPERATOR_CACHE.put(cache_key, result)
        return result

    @lru_cache(maxsize=None)
    def _product_expansion_matrix_cached(
        self,
        space_left,
        space_right,
        L_out,
    ):
        """Return the exact product-expansion matrix for all basis pairs."""
        cache_key = ("product_matrix", str(self.tree_type), space_left, space_right, int(L_out))
        cached = _GLOBAL_PRODUCT_MATRIX_CACHE.get(cache_key)
        if cached is not None:
            return cached
        result = self.product_expansion_operator(space_left, space_right, int(L_out)).to_matrix()
        _GLOBAL_PRODUCT_MATRIX_CACHE.put(cache_key, result)
        return result

    # ------------------------------------------------------------------
    # Exact generated-subspace / primitive quotient
    # ------------------------------------------------------------------
    @staticmethod
    def _submultiset_partitions(rank):
        """Yield canonical bipartitions of a fixed sorted leaf list by indices."""
        all_idx = tuple(range(rank))
        seen = set()
        for k in range(1, rank):
            for left in combinations(all_idx, k):
                right = tuple(i for i in all_idx if i not in left)
                key = tuple(sorted((left, right)))
                if key in seen:
                    continue
                seen.add(key)
                yield left, right

    def _submultiset_pattern_partitions(
        self,
        nin,
        lin,
    ):
        """Yield unique unordered bipartitions of a sorted target leaf multiset."""
        seen = set()
        for left_idx, right_idx in self._submultiset_partitions(len(nin)):
            left_pattern = self._restrict_pattern(nin, lin, left_idx)
            right_pattern = self._restrict_pattern(nin, lin, right_idx)
            key = tuple(sorted((left_pattern, right_pattern)))
            if key in seen:
                continue
            seen.add(key)
            yield left_pattern, right_pattern

    @staticmethod
    def _restrict_pattern(nin, lin, idxs):
        return tuple(nin[i] for i in idxs), tuple(lin[i] for i in idxs)

    @staticmethod
    def _column_signature_from_entries(entries):
        """Return a scale-invariant signature from sparse nonzero entries."""
        if not entries:
            return None
        pivot = entries[0][1]
        if pivot == 0:
            return None
        normalized = []
        for row_idx, value in entries:
            if value == 0:
                continue
            normalized.append((int(row_idx), value / pivot))
        return tuple(normalized)

    @staticmethod
    def _column_signature(column):
        """Return a scale-invariant signature for a generated-subspace column."""
        if column.cols != 1:
            raise ValueError("Expected a single-column SymPy matrix.")
        sparse_entries = tuple(
            (int(i), column[i, 0])
            for i in range(column.rows)
            if column[i, 0] != 0
        )
        return ExactProductExpansionEngine._column_signature_from_entries(sparse_entries)

    @staticmethod
    def _policy_for_subspace(L_R, factorization_policy):
        if factorization_policy == "module" and int(L_R) == 0:
            return "invariant"
        return str(factorization_policy)

    def _candidate_product_irreps(
        self,
        target_space,
        Ls_left,
        Ls_right,
        factorization_policy,
    ):
        policy = self._policy_for_subspace(target_space.L_R, factorization_policy)
        if policy == "invariant":
            if target_space.L_R != 0:
                return []
            return [(0, 0)] if (0 in Ls_left and 0 in Ls_right) else []
        if policy == "module":
            pairs = []
            if 0 in Ls_left and target_space.L_R in Ls_right:
                pairs.append((0, target_space.L_R))
            if target_space.L_R in Ls_left and 0 in Ls_right:
                pairs.append((target_space.L_R, 0))
            return pairs
        if policy == "full":
            return [
                (int(L1), int(L2))
                for L1 in Ls_left
                for L2 in Ls_right
                if abs(int(L1) - int(L2)) <= target_space.L_R <= int(L1) + int(L2)
            ]
        raise ValueError("factorization_policy must be one of {'invariant', 'module', 'full'}")

    @lru_cache(maxsize=None)
    def _generated_columns_for_target(self, target_space, mode, stop_at_rank = 0):
        """Assemble the exact lower-generated subspace for one target space."""
        cache_key = (
            "generated_columns",
            str(self.tree_type),
            tuple(target_space.nin),
            tuple(target_space.lin),
            int(target_space.L_R),
            str(mode),
            int(stop_at_rank),
        )
        cached = _GLOBAL_GENERATED_COLUMNS_CACHE.get(cache_key)
        if cached is not None:
            return cached
        if target_space.rank <= 1 or target_space.dim == 0:
            result = _sympy().zeros(target_space.dim, 0)
            _GLOBAL_GENERATED_COLUMNS_CACHE.put(cache_key, result)
            return result
        stop_rank = min(int(stop_at_rank), int(target_space.dim)) if int(stop_at_rank) > 0 else 0
        accumulator = _ExactIndependentColumnAccumulator(target_space.dim, stop_rank=stop_rank)
        for (nin_left, lin_left), (nin_right, lin_right) in self._submultiset_pattern_partitions(target_space.nin, target_space.lin):
            Ls_left = self.available_L_for_pattern(nin_left, lin_left)
            Ls_right = self.available_L_for_pattern(nin_right, lin_right)
            candidate_pairs = self._candidate_product_irreps(target_space, Ls_left, Ls_right, mode)

            for L1, L2 in candidate_pairs:
                space_left = self.feature_space(nin_left, lin_left, int(L1))
                space_right = self.feature_space(nin_right, lin_right, int(L2))
                operator = self.product_expansion_operator(space_left, space_right, target_space.L_R)
                for column in operator.columns:
                    accumulator.try_add_sparse(column)
                    if accumulator.reached_target():
                        result = accumulator.matrix()
                        _GLOBAL_GENERATED_COLUMNS_CACHE.put(cache_key, result)
                        return result
        result = accumulator.matrix()
        _GLOBAL_GENERATED_COLUMNS_CACHE.put(cache_key, result)
        return result

    def primitive_quotient(
        self,
        nin,
        lin,
        L_R,
        mode = "full",
        factorization_policy = None,
    ):
        """Compute the exact primitive quotient for one target space."""
        policy = self._policy_for_subspace(int(L_R), factorization_policy if factorization_policy is not None else mode)
        cache_key = ("primitive_quotient", str(self.tree_type), tuple(int(x) for x in nin), tuple(int(x) for x in lin), int(L_R), str(policy))
        cached = _GLOBAL_PRIMITIVE_QUOTIENT_CACHE.get(cache_key)
        if cached is not None:
            return cached
        target = self.feature_space(tuple(nin), tuple(lin), int(L_R))
        if target.dim == 0:
            result = PrimitiveQuotientSummary(target, 0, 0, tuple(), tuple())
            _GLOBAL_PRIMITIVE_QUOTIENT_CACHE.put(cache_key, result)
            return result
        generated = self._generated_columns_for_target(target, mode=policy, stop_at_rank=target.dim)
        if generated.cols == 0:
            result = PrimitiveQuotientSummary(target, 0, target.dim, tuple(), tuple(range(target.dim)))
            _GLOBAL_PRIMITIVE_QUOTIENT_CACHE.put(cache_key, result)
            return result
        # Row pivots of ``generated.T`` are coordinate directions in the target basis.
        _, row_pivots = generated.T.rref(simplify=False)
        generated_rank = len(row_pivots)
        primitive = tuple(i for i in range(target.dim) if i not in row_pivots)
        result = PrimitiveQuotientSummary(
            target_space=target,
            generated_rank=generated_rank,
            primitive_rank=target.dim - generated_rank,
            generated_basis_indices=tuple(int(i) for i in row_pivots),
            primitive_basis_indices=primitive,
        )
        _GLOBAL_PRIMITIVE_QUOTIENT_CACHE.put(cache_key, result)
        return result

    @lru_cache(maxsize=None)
    def primitive_representative_labels(
        self,
        nin,
        lin,
        L_R,
        factorization_policy = "full",
    ):
        """Return the selected representatives of the primitive quotient."""
        policy = self._policy_for_subspace(int(L_R), factorization_policy)
        summary = self.primitive_quotient(tuple(nin), tuple(lin), int(L_R), mode=policy)
        return tuple(summary.target_space.labels[i] for i in summary.primitive_basis_indices)

    def primitive_basis_labels(
        self,
        nin,
        lin,
        L_R,
        mode = "full",
    ):
        """Alias for :meth:`primitive_representative_labels`."""
        return self.primitive_representative_labels(tuple(nin), tuple(lin), int(L_R), mode)

    @staticmethod
    def _identity_columns(indices, dim):
        cols = []
        for idx in indices:
            cols.append(_sympy().SparseMatrix(int(dim), 1, {(int(idx), 0): _sympy().Integer(1)}))
        return cols

    @staticmethod
    def _append_independent_column(cols, seen_signatures, col):
        signature = ExactProductExpansionEngine._column_signature(col)
        if signature is None or signature in seen_signatures:
            return False
        seen_signatures.add(signature)
        cols.append(col)
        return True

    @staticmethod
    def _columns_reach_rank(cols, target_rank):
        if int(target_rank) <= 0 or len(cols) < int(target_rank):
            return False
        matrix = _sympy().Matrix.hstack(*cols)
        return int(matrix.rank(simplify=False)) >= int(target_rank)

    @staticmethod
    def _normalize_optional_int_tuple(values):
        if values is None:
            return tuple()
        return tuple(sorted({int(value) for value in values}))

    @staticmethod
    def _space_key(space):
        return tuple(space.nin), tuple(space.lin), int(space.L_R)

    @staticmethod
    def _primitive_generator_allowed(
        space,
        allowed_ranks,
        allowed_Ls,
        max_generator_rank,
        max_generator_L,
    ):
        if allowed_ranks and int(space.rank) not in allowed_ranks:
            return False
        if allowed_Ls and int(space.L_R) not in allowed_Ls:
            return False
        if int(max_generator_rank) >= 0 and int(space.rank) > int(max_generator_rank):
            return False
        if int(max_generator_L) >= 0 and int(space.L_R) > int(max_generator_L):
            return False
        return True

    @staticmethod
    def _recoupling_allowed(space, max_recoupling_L):
        return int(max_recoupling_L) < 0 or int(space.L_R) <= int(max_recoupling_L)

    def _product_column_from_coordinate_columns(
        self,
        left_space,
        right_space,
        col_left,
        col_right,
        L_out,
    ):
        """Expand a product of two coordinate vectors into the target basis."""
        product_operator = self.product_expansion_operator(left_space, right_space, int(L_out))
        if product_operator.cols == 0:
            return _sympy().zeros(product_operator.rows, 1)
        left_support = [(int(i), col_left[i, 0]) for i in range(left_space.dim) if col_left[i, 0] != 0]
        right_support = [(int(j), col_right[j, 0]) for j in range(right_space.dim) if col_right[j, 0] != 0]
        if not left_support or not right_support:
            return _sympy().zeros(product_operator.rows, 1)
        right_dim = int(right_space.dim)
        out_entries = {}
        for i, coeff_left in left_support:
            for j, coeff_right in right_support:
                product_coeff = coeff_left * coeff_right
                if product_coeff == 0:
                    continue
                column = product_operator.columns[int(i) * right_dim + int(j)]
                for row_idx, value in column.entries:
                    out_entries[int(row_idx)] = out_entries.get(int(row_idx), _sympy().Integer(0)) + product_coeff * value
        return _sympy().SparseMatrix(product_operator.rows, 1, {(row_idx, 0): value for row_idx, value in out_entries.items() if value != 0})

    @lru_cache(maxsize=None)
    def _independent_product_columns_from_primitive_policy(
        self,
        nin,
        lin,
        L_R,
        factorization_policy,
        allowed_generator_ranks,
        allowed_generator_Ls,
        max_generator_rank,
        max_generator_L,
        max_recoupling_L,
        root_key,
        include_target_primitive,
    ):
        columns, _ = self._independent_product_columns_and_descriptors_from_primitive_policy(
            tuple(nin),
            tuple(lin),
            int(L_R),
            factorization_policy,
            allowed_generator_ranks,
            allowed_generator_Ls,
            int(max_generator_rank),
            int(max_generator_L),
            int(max_recoupling_L),
            root_key,
            bool(include_target_primitive),
        )
        return columns

    @lru_cache(maxsize=None)
    def _independent_product_columns_and_descriptors_from_primitive_policy(
        self,
        nin,
        lin,
        L_R,
        factorization_policy,
        allowed_generator_ranks,
        allowed_generator_Ls,
        max_generator_rank,
        max_generator_L,
        max_recoupling_L,
        root_key,
        include_target_primitive,
    ):
        """Return exact columns generated by a controlled primitive policy.

        Primitive representatives are admitted as generators only when their
        rank/irrep pass the user policy.  Higher-rank columns are then generated
        recursively by exact CG product maps.  The returned matrix may still be
        overcomplete; public callers select an independent subset by exact RREF.
        """
        target = self.feature_space(tuple(nin), tuple(lin), int(L_R))
        is_root = self._space_key(target) == root_key
        if target.dim == 0 or (not is_root and not self._recoupling_allowed(target, int(max_recoupling_L))):
            return _sympy().zeros(target.dim, 0), tuple()

        accumulator = _ExactIndependentColumnAccumulator(target.dim, stop_rank=target.dim)
        descriptors = []
        if (bool(include_target_primitive) or not is_root) and self._primitive_generator_allowed(
            target,
            allowed_generator_ranks,
            allowed_generator_Ls,
            int(max_generator_rank),
            int(max_generator_L),
        ):
            quotient = self.primitive_quotient(tuple(nin), tuple(lin), int(L_R), mode=factorization_policy)
            for basis_index, col in zip(
                quotient.primitive_basis_indices,
                self._identity_columns(quotient.primitive_basis_indices, target.dim),
            ):
                if accumulator.try_add(col):
                    descriptors.append(
                        ExactProductColumnDescriptor(
                            kind="primitive",
                            space_nin=tuple(target.nin),
                            space_lin=tuple(target.lin),
                            L_R=int(target.L_R),
                            basis_index=int(basis_index),
                            basis_label=target.labels[int(basis_index)],
                            basis_handle=target.handle_for_index(int(basis_index)),
                        )
                    )

        if target.rank <= 1:
            return accumulator.matrix(), tuple(descriptors)

        for (nin_left, lin_left), (nin_right, lin_right) in self._submultiset_pattern_partitions(target.nin, target.lin):
            Ls_left = self.available_L_for_pattern(nin_left, lin_left)
            Ls_right = self.available_L_for_pattern(nin_right, lin_right)
            for L1, L2 in self._candidate_product_irreps(target, Ls_left, Ls_right, factorization_policy):
                if int(max_recoupling_L) >= 0 and (int(L1) > int(max_recoupling_L) or int(L2) > int(max_recoupling_L)):
                    continue
                left_space = self.feature_space(nin_left, lin_left, int(L1))
                right_space = self.feature_space(nin_right, lin_right, int(L2))
                left_cols, left_descriptors = self._independent_product_columns_and_descriptors_from_primitive_policy(
                    tuple(nin_left),
                    tuple(lin_left),
                    int(L1),
                    factorization_policy,
                    allowed_generator_ranks,
                    allowed_generator_Ls,
                    int(max_generator_rank),
                    int(max_generator_L),
                    int(max_recoupling_L),
                    root_key,
                    bool(include_target_primitive),
                )
                right_cols, right_descriptors = self._independent_product_columns_and_descriptors_from_primitive_policy(
                    tuple(nin_right),
                    tuple(lin_right),
                    int(L2),
                    factorization_policy,
                    allowed_generator_ranks,
                    allowed_generator_Ls,
                    int(max_generator_rank),
                    int(max_generator_L),
                    int(max_recoupling_L),
                    root_key,
                    bool(include_target_primitive),
                )
                if left_cols.cols == 0 or right_cols.cols == 0:
                    continue
                for i in range(left_cols.cols):
                    for j in range(right_cols.cols):
                        col = self._product_column_from_coordinate_columns(
                            left_space,
                            right_space,
                            left_cols[:, i],
                            right_cols[:, j],
                            target.L_R,
                        )
                        if accumulator.try_add(col):
                            descriptors.append(
                                ExactProductColumnDescriptor(
                                    kind="product",
                                    space_nin=tuple(target.nin),
                                    space_lin=tuple(target.lin),
                                    L_R=int(target.L_R),
                                    left=left_descriptors[i],
                                    right=right_descriptors[j],
                                )
                            )
                        if accumulator.reached_target():
                            return accumulator.matrix(), tuple(descriptors)

        return accumulator.matrix(), tuple(descriptors)

    def independent_decomposable_product_subspace(
        self,
        nin,
        lin,
        L_R,
        factorization_policy = "full",
        *,
        generator_ranks = None,
        generator_Ls = None,
        max_generator_rank = None,
        max_generator_L = None,
        max_recoupling_L = None,
        include_target_primitive = False,
    ):
        """Tabulate an exact independent product image from primitive generators."""
        nin_key = tuple(int(x) for x in nin)
        lin_key = tuple(int(x) for x in lin)
        generator_ranks_key = self._normalize_optional_int_tuple(generator_ranks)
        generator_Ls_key = self._normalize_optional_int_tuple(generator_Ls)
        max_rank = int(max_generator_rank) if max_generator_rank is not None else -1
        max_gen_L = int(max_generator_L) if max_generator_L is not None else -1
        max_rec_L = int(max_recoupling_L) if max_recoupling_L is not None else -1
        policy = self._policy_for_subspace(int(L_R), factorization_policy)
        cache_key = (
            "independent_decomposable_subspace",
            str(self.tree_type),
            nin_key,
            lin_key,
            int(L_R),
            str(policy),
            generator_ranks_key,
            generator_Ls_key,
            int(max_rank),
            int(max_gen_L),
            int(max_rec_L),
            bool(include_target_primitive),
        )
        cached = _GLOBAL_INDEPENDENT_SUBSPACE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        target = self.feature_space(tuple(nin), tuple(lin), int(L_R))
        allowed_ranks = generator_ranks_key
        allowed_Ls = generator_Ls_key
        root_key = self._space_key(target)
        raw, descriptors = self._independent_product_columns_and_descriptors_from_primitive_policy(
            tuple(target.nin),
            tuple(target.lin),
            int(target.L_R),
            policy,
            allowed_ranks,
            allowed_Ls,
            max_rank,
            max_gen_L,
            max_rec_L,
            root_key,
            bool(include_target_primitive),
        )
        if raw.cols:
            _, product_pivots = raw.rref(simplify=False)
            independent_indices = tuple(int(i) for i in product_pivots)
            independent = raw.extract(range(raw.rows), list(independent_indices))
            independent_descriptors = tuple(descriptors[i] for i in independent_indices)
            _, target_pivots = independent.T.rref(simplify=False)
            target_basis_indices = tuple(int(i) for i in target_pivots)
        else:
            independent_indices = tuple()
            independent = _sympy().zeros(target.dim, 0)
            independent_descriptors = tuple()
            target_basis_indices = tuple()
        missing_basis_indices = tuple(i for i in range(target.dim) if i not in target_basis_indices)
        result = IndependentDecomposableProductSubspace(
            target_space=target,
            factorization_policy=policy,
            generator_ranks=allowed_ranks,
            generator_Ls=allowed_Ls,
            max_generator_rank=None if max_rank < 0 else max_rank,
            max_generator_L=None if max_gen_L < 0 else max_gen_L,
            max_recoupling_L=None if max_rec_L < 0 else max_rec_L,
            include_target_primitive=bool(include_target_primitive),
            raw_product_column_count=int(raw.cols),
            independent_product_rank=len(independent_indices),
            independent_product_column_indices=independent_indices,
            product_descriptors=tuple(descriptors),
            independent_product_descriptors=independent_descriptors,
            target_basis_indices=target_basis_indices,
            missing_rank=int(target.dim - len(target_basis_indices)),
            missing_basis_indices=missing_basis_indices,
            coordinate_matrix=independent,
        )
        _GLOBAL_INDEPENDENT_SUBSPACE_CACHE.put(cache_key, result)
        return result

    @lru_cache(maxsize=None)
    def _primitive_generated_columns_for_space(
        self,
        nin,
        lin,
        L_R,
        factorization_policy,
    ):
        """Return columns spanning a feature space from primitive generators."""
        policy = self._policy_for_subspace(int(L_R), factorization_policy)
        target = self.feature_space(tuple(nin), tuple(lin), int(L_R))
        if target.dim == 0:
            return _sympy().zeros(0, 0)

        quotient = self.primitive_quotient(tuple(nin), tuple(lin), int(L_R), mode=policy)
        cols = self._identity_columns(quotient.primitive_basis_indices, target.dim)
        seen_signatures = {self._column_signature(col) for col in cols}
        seen_signatures.discard(None)

        if target.rank > 1:
            product_cols, _ = self._factorizable_columns_from_primitive_generators(target, policy)
            for idx in range(product_cols.cols):
                self._append_independent_column(cols, seen_signatures, product_cols[:, idx])
        if not cols:
            return _sympy().zeros(target.dim, 0)
        return _sympy().Matrix.hstack(*cols)

    @lru_cache(maxsize=None)
    def _factorizable_columns_from_primitive_generators(
        self,
        target_space,
        factorization_policy,
    ):
        """Build factorizable columns using recursively generated lower spaces."""
        cache_key = (
            "factorizable_columns",
            str(self.tree_type),
            tuple(target_space.nin),
            tuple(target_space.lin),
            int(target_space.L_R),
            str(factorization_policy),
        )
        cached = _GLOBAL_FACTORIZABLE_COLUMNS_CACHE.get(cache_key)
        if cached is not None:
            return cached
        if target_space.rank <= 1 or target_space.dim == 0:
            result = (_sympy().zeros(target_space.dim, 0), 0)
            _GLOBAL_FACTORIZABLE_COLUMNS_CACHE.put(cache_key, result)
            return result
        accumulator = _ExactIndependentColumnAccumulator(target_space.dim, stop_rank=target_space.dim)
        product_column_count = 0
        for (nin_left, lin_left), (nin_right, lin_right) in self._submultiset_pattern_partitions(target_space.nin, target_space.lin):
            Ls_left = self.available_L_for_pattern(nin_left, lin_left)
            Ls_right = self.available_L_for_pattern(nin_right, lin_right)
            for L1, L2 in self._candidate_product_irreps(target_space, Ls_left, Ls_right, factorization_policy):
                left_space = self.feature_space(nin_left, lin_left, int(L1))
                right_space = self.feature_space(nin_right, lin_right, int(L2))
                left_cols = self._primitive_generated_columns_for_space(tuple(nin_left), tuple(lin_left), int(L1), factorization_policy)
                right_cols = self._primitive_generated_columns_for_space(tuple(nin_right), tuple(lin_right), int(L2), factorization_policy)
                for i in range(left_cols.cols):
                    for j in range(right_cols.cols):
                        product_column_count += 1
                        col = self._product_column_from_coordinate_columns(
                            left_space,
                            right_space,
                            left_cols[:, i],
                            right_cols[:, j],
                            target_space.L_R,
                        )
                        accumulator.try_add(col)
                        if accumulator.reached_target():
                            result = (accumulator.matrix(), product_column_count)
                            _GLOBAL_FACTORIZABLE_COLUMNS_CACHE.put(cache_key, result)
                            return result
        result = (accumulator.matrix(), product_column_count)
        _GLOBAL_FACTORIZABLE_COLUMNS_CACHE.put(cache_key, result)
        return result

    def primitive_generator_reconstruction(
        self,
        nin,
        lin,
        L_R,
        factorization_policy = "full",
    ):
        """Audit reconstruction of ``F_{N,L}`` from primitive generators."""
        target = self.feature_space(tuple(nin), tuple(lin), int(L_R))
        policy = self._policy_for_subspace(int(L_R), factorization_policy)
        quotient = self.primitive_quotient(tuple(nin), tuple(lin), int(L_R), mode=policy)
        primitive_labels = tuple(target.labels[i] for i in quotient.primitive_basis_indices)
        product_cols, product_column_count = self._factorizable_columns_from_primitive_generators(target, policy)

        if product_cols.cols:
            _, product_pivots = product_cols.T.rref(simplify=False)
            factorizable_basis_indices = tuple(int(i) for i in product_pivots)
            factorizable_rank = len(factorizable_basis_indices)
        else:
            factorizable_basis_indices = tuple()
            factorizable_rank = 0

        combined_cols = self._identity_columns(quotient.primitive_basis_indices, target.dim)
        for idx in range(product_cols.cols):
            combined_cols.append(product_cols[:, idx])
        if combined_cols:
            combined = _sympy().Matrix.hstack(*combined_cols)
            _, reconstructed_pivots = combined.T.rref(simplify=False)
            reconstructed_basis_indices = tuple(int(i) for i in reconstructed_pivots)
            reconstructed_rank = len(reconstructed_basis_indices)
        else:
            reconstructed_basis_indices = tuple()
            reconstructed_rank = 0
        missing_basis_indices = tuple(i for i in range(target.dim) if i not in reconstructed_basis_indices)
        return PrimitiveGeneratorReconstructionSummary(
            target_space=target,
            factorization_policy=policy,
            primitive_rank=int(quotient.primitive_rank),
            primitive_basis_indices=tuple(int(i) for i in quotient.primitive_basis_indices),
            primitive_labels=primitive_labels,
            factorizable_rank=int(quotient.generated_rank),
            factorizable_from_primitive_rank=int(factorizable_rank),
            factorizable_missing_rank=int(quotient.generated_rank - factorizable_rank),
            factorizable_basis_indices=factorizable_basis_indices,
            product_column_count=int(product_column_count),
            reconstructed_rank=int(reconstructed_rank),
            missing_rank=int(target.dim - reconstructed_rank),
            missing_basis_indices=missing_basis_indices,
        )

    def primitive_reconstruction(
        self,
        nin,
        lin,
        L_R,
        mode = "full",
    ):
        """Alias for :meth:`primitive_generator_reconstruction`."""
        return self.primitive_generator_reconstruction(nin, lin, L_R, factorization_policy=mode)

    def benchmark_primitive_reconstruction(
        self,
        nin,
        lin,
        L_R,
        factorization_policy = "full",
    ):
        """Compare direct exact, primitive quotient, and reconstruction timings."""
        from time import perf_counter

        nin_t = tuple(int(x) for x in nin)
        lin_t = tuple(int(x) for x in lin)
        policy = self._policy_for_subspace(int(L_R), factorization_policy)
        t0 = perf_counter()
        target = self.feature_space(nin_t, lin_t, int(L_R))
        t1 = perf_counter()
        quotient = self.primitive_quotient(nin_t, lin_t, int(L_R), mode=policy)
        t2 = perf_counter()
        reconstruction = self.primitive_generator_reconstruction(nin_t, lin_t, int(L_R), factorization_policy=policy)
        t3 = perf_counter()
        return PrimitiveReconstructionBenchmark(
            nin=nin_t,
            lin=lin_t,
            L_R=int(L_R),
            factorization_policy=policy,
            space_dim=int(target.dim),
            primitive_rank=int(quotient.primitive_rank),
            factorizable_rank=int(quotient.generated_rank),
            factorizable_from_primitive_rank=int(reconstruction.factorizable_from_primitive_rank),
            factorizable_missing_rank=int(reconstruction.factorizable_missing_rank),
            reconstructed_rank=int(reconstruction.reconstructed_rank),
            missing_rank=int(reconstruction.missing_rank),
            direct_space_seconds=float(t1 - t0),
            primitive_quotient_seconds=float(t2 - t1),
            primitive_reconstruction_seconds=float(t3 - t2),
        )

    def benchmark_reconstruction_paths(
        self,
        nin,
        lin,
        L_R,
        mode = "full",
    ):
        """Alias for :meth:`benchmark_primitive_reconstruction`."""
        return self.benchmark_primitive_reconstruction(nin, lin, L_R, factorization_policy=mode)


ExactSector = ExactFeatureSpace
PrimitiveReconstructionSummary = PrimitiveGeneratorReconstructionSummary
ReconstructionBenchmarkResult = PrimitiveReconstructionBenchmark

try:
    from .graded_algebra import (
        GradedBasisRegistry,
        HomogeneousSector,
        PrimitiveDecompositionSummary,
        SampledGeneratedSubspaceAnalyzer,
        SampledSpanResult,
    )
except Exception:  # pragma: no cover - avoids circular import during partial initialization
    GradedBasisRegistry = None  # type: ignore
    HomogeneousSector = None  # type: ignore
    PrimitiveDecompositionSummary = None  # type: ignore
    SampledGeneratedSubspaceAnalyzer = None  # type: ignore
    SampledSpanResult = None  # type: ignore

try:
    from .benchmarking import (
        ExactPrimitiveBenchmarkResult,
        benchmark_exact_primitive_case,
        benchmark_exact_primitive_suite,
    )
except Exception:  # pragma: no cover
    ExactPrimitiveBenchmarkResult = None  # type: ignore
    benchmark_exact_primitive_case = None  # type: ignore
    benchmark_exact_primitive_suite = None  # type: ignore
