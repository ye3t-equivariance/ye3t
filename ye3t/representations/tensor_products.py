
"""Tree-aware and merged symbolic tensor-product tools for the generalized path.

This module is intentionally separate from the runtime-oriented exact tensor
product path. It now provides two related layers:

1. tree-aware symbolic bookkeeping along a chosen binary coupling tree
2. exact conversion from that tree-local subgroup picture to merged global
   subgroup sectors, together with exact basis-level intertwiners

The merged conversion is exact and small-``N`` symbolic. It works by:

- decomposing subgroup products channel-by-channel through exact character
  inner products
- building merged output sectors through the separate generalized symbolic
  builder
- CG-coupling the left/right generalized multiplets in raw space
- expanding those coupled input vectors into canonical merged output bases

This remains part of the separate generalized path and does not affect the fast
ACE / ye3t runtime path.
"""

from collections import OrderedDict
from functools import lru_cache
from itertools import product
from math import factorial

from ye3t.core.basis.validation import validate_tree_type
from ye3t.core.subtree_dag import cg_exact
from ye3t.exact_linalg import (
    exact_matrix_from_entries,
    exact_matrix_matmul,
    exact_matrix_transpose,
)

from .generalized_irreps import AngularIrrep, CoupledIrrepLabel, GeneralizedTensorProduct, Partition, PermutationIrrep, PermutationSubgroup, PermutationSubgroupFactor
from .generalized_sector_data import GeneralizedSectorData
from .projectors import _partition_character_classes, symmetric_group_character
from ye3t._record import recordclass


def _sympy():
    from ye3t._optional_sympy import sp

    return sp


class _BoundedObjectCache:
    """Small LRU cache with lightweight stats for generalized exact artifacts."""

    def __init__(self, *, max_entries):
        self.max_entries = None if max_entries is None else max(1, int(max_entries))
        self._items = OrderedDict()
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0

    def get(self, key):
        item = self._items.get(key)
        if item is None:
            self.miss_count += 1
            return None
        self._items.move_to_end(key)
        self.hit_count += 1
        return item

    def set(self, key, value):
        if key in self._items:
            self._items.move_to_end(key)
            self._items[key] = value
            return
        self._items[key] = value
        if self.max_entries is None:
            return
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)
            self.eviction_count += 1

    def clear(self):
        self._items.clear()
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0

    def stats(self):
        total = int(self.hit_count + self.miss_count)
        return {
            "size": int(len(self._items)),
            "max_entries": None if self.max_entries is None else int(self.max_entries),
            "hit_count": int(self.hit_count),
            "miss_count": int(self.miss_count),
            "eviction_count": int(self.eviction_count),
            "hit_rate_ppm": 0 if total == 0 else int(round(1_000_000 * self.hit_count / total)),
        }


def _tagged_factor(factor, prefix):
    return PermutationSubgroupFactor(
        channel_label=(tuple(str(x) for x in prefix), factor.channel_label),
        l=int(factor.l),
        multiplicity=int(factor.multiplicity),
    )


def _base_channel_label(channel_label):
    if (
        isinstance(channel_label, tuple)
        and len(channel_label) == 2
        and isinstance(channel_label[0], tuple)
    ):
        return channel_label[1]
    return channel_label


def _base_channel_key(factor):
    return (_base_channel_label(factor.channel_label), int(factor.l))


def _partitions_of_n(n, *, max_part = None):
    n = int(n)
    if n == 0:
        return (tuple(),)
    cap = int(n if max_part is None else min(max_part, n))
    out = []
    for first in range(cap, 0, -1):
        for tail in _partitions_of_n(n - first, max_part=first):
            out.append((int(first),) + tuple(int(x) for x in tail))
    return tuple(out)


@lru_cache(maxsize=None)
def _induced_partition_product_multiplicity_cached(
    left_parts,
    right_parts,
    target_parts,
):
    left_partition = Partition(tuple(int(value) for value in left_parts))
    right_partition = Partition(tuple(int(value) for value in right_parts))
    target_partition = Partition(tuple(int(value) for value in target_parts))
    a = int(left_partition.size)
    b = int(right_partition.size)
    if int(target_partition.size) != a + b:
        return 0
    total = 0
    for left_cycle, left_size, left_character in _partition_character_classes(
        tuple(left_partition.parts)
    ):
        for right_cycle, right_size, right_character in _partition_character_classes(
            tuple(right_partition.parts)
        ):
            merged_cycle = tuple(
                sorted(tuple(left_cycle) + tuple(right_cycle), reverse=True)
            )
            total += (
                int(left_size)
                * int(right_size)
                * int(left_character)
                * int(right_character)
                * int(symmetric_group_character(target_partition, merged_cycle))
            )
    denominator = int(factorial(a) * factorial(b))
    if total % denominator != 0:
        raise ArithmeticError(
            "character inner product did not produce an integer LR multiplicity"
        )
    return int(total // denominator)


def _induced_partition_product_multiplicity(
    left_partition,
    right_partition,
    target_partition,
):
    """Exact multiplicity from ``S_a x S_b`` to ``S_{a+b}`` by Frobenius reciprocity."""

    return _induced_partition_product_multiplicity_cached(
        tuple(left_partition.parts),
        tuple(right_partition.parts),
        tuple(target_partition.parts),
    )


def tree_aware_permutation_irrep_product(
    left,
    right,
    *,
    node_path = ("root",),
):
    """Combine subgroup irreps without merging factors across the tree cut."""

    node_path = tuple(str(x) for x in node_path)
    subgroup = PermutationSubgroup(
        tuple(_tagged_factor(factor, (*node_path, "L")) for factor in left.subgroup.factors)
        + tuple(_tagged_factor(factor, (*node_path, "R")) for factor in right.subgroup.factors)
    )
    return PermutationIrrep(
        subgroup=subgroup,
        partitions=tuple(left.partitions) + tuple(right.partitions),
    )


def tree_aware_tensor_product_output_labels(
    left,
    right,
    *,
    node_path = ("root",),
):
    """Return the exact tree-aware symbolic output label inventory."""

    if (left.parity is None) != (right.parity is None):
        raise ValueError("cannot mix signed O(3) and unspecified SO(3)-legacy parity")
    output_parity = (
        None
        if left.parity is None
        else int(left.parity) * int(right.parity)
    )

    output_permutation = tree_aware_permutation_irrep_product(
        left.permutation,
        right.permutation,
        node_path=node_path,
    )
    L_min = abs(int(left.angular.l) - int(right.angular.l))
    L_max = int(left.angular.l) + int(right.angular.l)
    return tuple(
        CoupledIrrepLabel(
            angular=AngularIrrep(int(L)),
            permutation=output_permutation,
            multiplicity_index=0,
            parity=output_parity,
        )
        for L in range(int(L_min), int(L_max) + 1)
    )


@recordclass(('left', 'right', 'output_labels', 'tree_type', 'node_path', 'notes'), frozen = True)
class TreeAwareTensorProductRule:
    """One tree-aware symbolic generalized tensor-product rule."""


def build_tree_aware_tensor_product_rule(
    left,
    right,
    *,
    tree_type = "balanced",
    node_path = ("root",),
):
    """Build a symbolic tensor-product rule along one chosen tree edge."""

    tree_type = validate_tree_type(tree_type)
    outputs = tree_aware_tensor_product_output_labels(left, right, node_path=node_path)
    return TreeAwareTensorProductRule(
        left=left,
        right=right,
        output_labels=outputs,
        tree_type=str(tree_type),
        node_path=tuple(str(x) for x in node_path),
        notes=(
            "Tree-aware subgroup rule: child subgroup factors remain separate across this coupling edge.",
            "Merged global subgroup conversion is available through the exact merged tensor-product helpers.",
        ),
    )


def build_tree_aware_generalized_tensor_product(
    left,
    right,
    *,
    tree_type = "balanced",
    node_path = ("root",),
):
    """Build a symbolic tensor-product descriptor for one chosen output branch."""

    tree_type = validate_tree_type(tree_type)
    rule = build_tree_aware_tensor_product_rule(left, right, tree_type=tree_type, node_path=node_path)
    output = rule.output_labels[0] if rule.output_labels else None
    return GeneralizedTensorProduct(
        inputs=(left, right),
        output=output,
        tree_type=str(tree_type),
        notes=rule.notes,
    )


@recordclass(('permutation_irrep', 'multiplicity'), frozen = True)
class MergedPermutationIrrepTerm:
    """One exact merged global subgroup-irrep term."""


@recordclass(('left_term', 'right_term', 'source_weight'), frozen = True)
class ExactChangeOfGroupSourcePair:
    """One source-pair column in an exact change-of-group map."""


@recordclass(('left_terms', 'right_terms', 'source_pairs', 'target_terms', 'induction_matrix', 'restriction_matrix', 'source_weight_vector', 'target_weight_vector', 'tree_type', 'node_span', 'notes'), frozen = True)
class ExactChangeOfGroupMap:
    """Reusable exact induction/restriction map between child and merged subgroup sectors."""

    def _column_vector(self, weights):
        weights = tuple(weights)
        if weights and isinstance(weights[0], (tuple, list)):
            return tuple(tuple(int(value) for value in row) for row in weights)
        return tuple((int(value),) for value in weights)

    def induced_target_weights(
        self,
        source_weights = None,
    ):
        if source_weights is None:
            source_vector = self.source_weight_vector
        else:
            source_vector = self._column_vector(source_weights)
        return exact_matrix_matmul(self.induction_matrix, source_vector)

    def restricted_source_weights(
        self,
        target_weights,
    ):
        target_vector = self._column_vector(target_weights)
        return exact_matrix_matmul(self.restriction_matrix, target_vector)


@recordclass(('node_id', 'leaf_span', 'depth', 'left_child_id', 'right_child_id', 'is_leaf'), frozen = True)
class CouplingScheduleNode:
    """One node in a scheduled generalized tensor-product coupling tree."""


@recordclass(('tree_type', 'num_inputs', 'root_id', 'nodes', 'level_node_ids', 'internal_node_ids_postorder', 'peak_parallel_width', 'notes'), frozen = True)
class GeneralizedTensorProductSchedule:
    """A scheduled exact coupling plan for the separate generalized path."""


@recordclass(('schedule', 'root_branches', 'branches_by_node', 'change_of_group_maps_by_node', 'max_intermediate_branch_count', 'total_binary_decompositions'), frozen = True)
class ScheduledGeneralizedTensorProductResult:
    """Executed result for a scheduled exact generalized tensor-product composition."""


@recordclass(('permutation_irrep', 'output_L', 'source_basis_labels', 'target_basis_labels', 'source_basis_matrix', 'target_basis_matrix', 'raw_projected_source_matrix', 'coordinate_matrix'), frozen = True)
class ExactChangeOfGroupBasisBranch:
    """Exact canonical highest-weight basis map for one merged global subgroup sector."""


@recordclass(('tree_type', 'node_span', 'left_L', 'right_L', 'output_L', 'left_permutation', 'right_permutation', 'source_basis_labels', 'branches', 'notes'), frozen = True)
class ExactChangeOfGroupBasisMap:
    """Exact canonical highest-weight change-of-group map at fixed output ``L``."""


@recordclass(('permutation_irrep', 'output_L', 'subgroup_multiplicity', 'output_sector', 'source_basis_labels', 'target_basis_labels', 'source_basis_matrix', 'target_basis_matrix', 'raw_projected_source_matrix', 'coordinate_matrix'), frozen = True)
class ExactLoweredChangeOfGroupBasisBranch:
    """Exact canonical full-multiplet basis map for one merged global subgroup sector."""
    subgroup_multiplicity = 1


@recordclass(('tree_type', 'node_span', 'left_L', 'right_L', 'output_L', 'left_permutation', 'right_permutation', 'source_basis_labels', 'branches', 'notes'), frozen = True)
class ExactLoweredChangeOfGroupBasisMap:
    """Exact canonical full lowered-multiplet change-of-group map at fixed output ``L``."""


def _sympy_matrix_nnz(matrix):
    count = 0
    for row in range(int(matrix.rows)):
        for col in range(int(matrix.cols)):
            if _sympy().simplify(matrix[row, col]) != 0:
                count += 1
    return int(count)


def _sympy_matrix_abs_heatmap_data(
    matrix,
):
    return tuple(
        tuple(float(abs(complex(matrix[row, col].evalf()))) for col in range(int(matrix.cols)))
        for row in range(int(matrix.rows))
    )


def _permutation_irrep_signature(permutation_irrep):
    return (
        tuple(
            (
                factor.channel_label,
                int(factor.l),
                int(factor.multiplicity),
            )
            for factor in permutation_irrep.subgroup.factors
        ),
        tuple(tuple(int(x) for x in partition.parts) for partition in permutation_irrep.partitions),
    )


def _term_collection_signature(
    terms,
):
    return tuple(
        (
            _permutation_irrep_signature(term.permutation_irrep),
            int(term.multiplicity),
        )
        for term in terms
    )


def merged_permutation_irrep_product_terms(
    left,
    right,
):
    """Convert tree-local child subgroup irreps into merged global subgroup terms."""

    left_by_key = {_base_channel_key(factor): (factor, partition) for factor, partition in zip(left.subgroup.factors, left.partitions, strict=True)}
    right_by_key = {_base_channel_key(factor): (factor, partition) for factor, partition in zip(right.subgroup.factors, right.partitions, strict=True)}
    all_keys = tuple(sorted(set(left_by_key) | set(right_by_key), key=lambda item: (str(item[0]), int(item[1]))))

    merged_factor_options = []
    for key in all_keys:
        left_item = left_by_key.get(key)
        right_item = right_by_key.get(key)
        channel_label, l_value = key
        if left_item is None:
            factor_right, partition_right = right_item  # type: ignore[misc]
            merged_factor_options.append([
                (
                    PermutationSubgroupFactor(channel_label=channel_label, l=int(l_value), multiplicity=int(factor_right.multiplicity)),
                    partition_right,
                    1,
                )
            ])
            continue
        if right_item is None:
            factor_left, partition_left = left_item
            merged_factor_options.append([
                (
                    PermutationSubgroupFactor(channel_label=channel_label, l=int(l_value), multiplicity=int(factor_left.multiplicity)),
                    partition_left,
                    1,
                )
            ])
            continue
        factor_left, partition_left = left_item
        factor_right, partition_right = right_item
        total_mult = int(factor_left.multiplicity) + int(factor_right.multiplicity)
        options = []
        for parts in _partitions_of_n(total_mult):
            target_partition = Partition(parts)
            multiplicity = _induced_partition_product_multiplicity(partition_left, partition_right, target_partition)
            if multiplicity > 0:
                options.append(
                    (
                        PermutationSubgroupFactor(channel_label=channel_label, l=int(l_value), multiplicity=total_mult),
                        target_partition,
                        int(multiplicity),
                    )
                )
        merged_factor_options.append(options)

    terms = []
    for choice in product(*merged_factor_options):
        subgroup = PermutationSubgroup(tuple(item[0] for item in choice))
        permutation_irrep = PermutationIrrep(
            subgroup=subgroup,
            partitions=tuple(item[1] for item in choice),
        )
        multiplicity = 1
        for item in choice:
            multiplicity *= int(item[2])
        terms.append(MergedPermutationIrrepTerm(permutation_irrep=permutation_irrep, multiplicity=int(multiplicity)))
    terms.sort(key=lambda term: (term.permutation_irrep.to_string(), term.multiplicity))
    return tuple(terms)


def build_exact_change_of_group_map(
    left_terms,
    right_terms,
    *,
    tree_type = "balanced",
    node_span = (0, 0),
):
    """Build a reusable exact induction/restriction map for one regrouping edge."""

    tree_type = validate_tree_type(tree_type)
    left_terms = tuple(left_terms)
    right_terms = tuple(right_terms)
    source_pairs = []
    target_by_signature = {}
    pair_contributions = []

    for left_term in left_terms:
        for right_term in right_terms:
            source_weight = int(left_term.multiplicity) * int(right_term.multiplicity)
            source_pairs.append(
                ExactChangeOfGroupSourcePair(
                    left_term=left_term,
                    right_term=right_term,
                    source_weight=int(source_weight),
                )
            )
            contributions = {}
            for merged_term in merged_permutation_irrep_product_terms(
                left_term.permutation_irrep,
                right_term.permutation_irrep,
            ):
                signature = _permutation_irrep_signature(merged_term.permutation_irrep)
                contributions[signature] = int(merged_term.multiplicity)
                existing = target_by_signature.get(signature)
                added_weight = int(source_weight) * int(merged_term.multiplicity)
                if existing is None:
                    target_by_signature[signature] = (merged_term.permutation_irrep, int(added_weight))
                else:
                    target_by_signature[signature] = (existing[0], int(existing[1] + added_weight))
            pair_contributions.append(contributions)

    target_terms = tuple(
        MergedPermutationIrrepTerm(permutation_irrep=perm_irrep, multiplicity=int(weight))
        for _, (perm_irrep, weight) in sorted(
            target_by_signature.items(),
            key=lambda item: item[1][0].to_string(),
        )
    )
    row_index = {
        _permutation_irrep_signature(term.permutation_irrep): idx
        for idx, term in enumerate(target_terms)
    }
    induction_entries = {}
    for col, contributions in enumerate(pair_contributions):
        for signature, multiplicity in contributions.items():
            induction_entries[(row_index[signature], col)] = int(multiplicity)
    induction = exact_matrix_from_entries(len(target_terms), len(source_pairs), induction_entries)
    source_weight_vector = tuple((int(pair.source_weight),) for pair in source_pairs)
    target_weight_vector = exact_matrix_matmul(induction, source_weight_vector)
    return ExactChangeOfGroupMap(
        left_terms=left_terms,
        right_terms=right_terms,
        source_pairs=tuple(source_pairs),
        target_terms=target_terms,
        induction_matrix=induction,
        restriction_matrix=exact_matrix_transpose(induction),
        source_weight_vector=source_weight_vector,
        target_weight_vector=target_weight_vector,
        tree_type=str(tree_type),
        node_span=(int(node_span[0]), int(node_span[1])),
        notes=(
            "Exact change-of-group map on subgroup-irrep multiplicity data.",
            "Rows are merged global target sectors and columns are child-pair source sectors.",
            "This library remains separate from the fast ACE/ye3t runtime path.",
        ),
    )


def restrict_merged_permutation_irrep_to_children(
    merged,
    left,
    right,
):
    """Return the exact restriction multiplicity back to one child pair."""

    for term in merged_permutation_irrep_product_terms(left, right):
        if _permutation_irrep_signature(term.permutation_irrep) == _permutation_irrep_signature(merged):
            return int(term.multiplicity)
    return 0


def _combine_merged_term_collections(
    left_terms,
    right_terms,
):
    accumulated = {}
    for left_term in left_terms:
        for right_term in right_terms:
            for merged_term in merged_permutation_irrep_product_terms(left_term.permutation_irrep, right_term.permutation_irrep):
                signature = _permutation_irrep_signature(merged_term.permutation_irrep)
                total_mult = int(left_term.multiplicity) * int(right_term.multiplicity) * int(merged_term.multiplicity)
                existing = accumulated.get(signature)
                if existing is None:
                    accumulated[signature] = (merged_term.permutation_irrep, int(total_mult))
                else:
                    accumulated[signature] = (existing[0], int(existing[1] + total_mult))
    out = [
        MergedPermutationIrrepTerm(permutation_irrep=perm_irrep, multiplicity=int(mult))
        for perm_irrep, mult in accumulated.values()
        if int(mult) > 0
    ]
    out.sort(key=lambda term: (term.permutation_irrep.to_string(), term.multiplicity))
    return tuple(out)


def merge_permutation_irrep_sequence(
    irreps,
    *,
    regrouping = "balanced",
):
    """Merge a deeper tree-local sequence into global subgroup sectors exactly."""

    regrouping = validate_tree_type(regrouping)
    irreps = tuple(irreps)
    if not irreps:
        return tuple()
    singleton_terms = tuple((MergedPermutationIrrepTerm(permutation_irrep=irrep, multiplicity=1),) for irrep in irreps)
    if regrouping == "left":
        current = singleton_terms[0]
        for next_terms in singleton_terms[1:]:
            current = _combine_merged_term_collections(current, next_terms)
        return current

    def _merge_balanced(term_groups):
        if len(term_groups) == 1:
            return tuple(term_groups[0])
        midpoint = len(term_groups) // 2
        left_group = _merge_balanced(term_groups[:midpoint])
        right_group = _merge_balanced(term_groups[midpoint:])
        return _combine_merged_term_collections(left_group, right_group)

    return _merge_balanced(singleton_terms)


def build_generalized_tensor_product_schedule(
    num_inputs,
    *,
    tree_type = "balanced",
):
    """Build a generalized exact coupling schedule with repo-consistent tree types."""

    tree_type = validate_tree_type(tree_type)
    num_inputs = int(num_inputs)
    if num_inputs <= 0:
        raise ValueError(f"num_inputs must be positive, got {num_inputs!r}")

    nodes = []
    levels = {}
    postorder = []

    def _rec_balanced(start, end, depth):
        levels.setdefault(depth, [])
        if end - start == 1:
            node_id = f"leaf:{start}:{end}"
            node = CouplingScheduleNode(
                node_id=node_id,
                leaf_span=(int(start), int(end)),
                depth=int(depth),
                left_child_id=None,
                right_child_id=None,
                is_leaf=True,
            )
            nodes.append(node)
            levels[depth].append(node_id)
            return node_id
        midpoint = start + (end - start) // 2
        left_id = _rec_balanced(start, midpoint, depth + 1)
        right_id = _rec_balanced(midpoint, end, depth + 1)
        node_id = f"node:{start}:{end}"
        node = CouplingScheduleNode(
            node_id=node_id,
            leaf_span=(int(start), int(end)),
            depth=int(depth),
            left_child_id=left_id,
            right_child_id=right_id,
            is_leaf=False,
        )
        nodes.append(node)
        levels[depth].append(node_id)
        postorder.append(node_id)
        return node_id

    def _make_leaf(index, depth):
        levels.setdefault(depth, [])
        node_id = f"leaf:{index}:{index + 1}"
        node = CouplingScheduleNode(
            node_id=node_id,
            leaf_span=(int(index), int(index + 1)),
            depth=int(depth),
            left_child_id=None,
            right_child_id=None,
            is_leaf=True,
        )
        nodes.append(node)
        levels[depth].append(node_id)
        return node_id

    def _rec_left(start, end, depth):
        levels.setdefault(depth, [])
        if end - start == 1:
            return _make_leaf(start, depth)
        left_id = _rec_left(start, end - 1, depth + 1)
        right_id = _make_leaf(end - 1, depth + 1)
        node_id = f"node:{start}:{end}"
        node = CouplingScheduleNode(
            node_id=node_id,
            leaf_span=(int(start), int(end)),
            depth=int(depth),
            left_child_id=left_id,
            right_child_id=right_id,
            is_leaf=False,
        )
        nodes.append(node)
        levels[depth].append(node_id)
        postorder.append(node_id)
        return node_id

    root_id = _rec_balanced(0, num_inputs, 0) if tree_type == "balanced" else _rec_left(0, num_inputs, 0)
    ordered_levels = tuple(tuple(levels[depth]) for depth in sorted(levels))
    peak_parallel_width = max(len(level) for level in ordered_levels if level)
    return GeneralizedTensorProductSchedule(
        tree_type=str(tree_type),
        num_inputs=int(num_inputs),
        root_id=root_id,
        nodes=tuple(nodes),
        level_node_ids=ordered_levels,
        internal_node_ids_postorder=tuple(postorder),
        peak_parallel_width=int(peak_parallel_width),
        notes=(
            f"{tree_type} schedule for the separate exact generalized tensor-product path.",
            "Leaf spans identify the regrouping blocks used by exact change-of-group maps and cached binary decompositions.",
        ),
    )


def build_balanced_pairwise_schedule(
    num_inputs,
):
    """Compatibility helper for the default balanced generalized schedule."""

    return build_generalized_tensor_product_schedule(int(num_inputs), tree_type="balanced")


def build_left_schedule(
    num_inputs,
):
    """Build the explicit left-justified schedule for the generalized path."""

    return build_generalized_tensor_product_schedule(int(num_inputs), tree_type="left")


def _sector_basis_matrix_for_L(
    sector,
    L,
):
    L = int(L)
    columns = []
    labels = []
    for (copy_index, carrier_index), multiplet in sorted(
        sector.lowered_multiplets_by_L.get(int(L), {}).items(),
        key=lambda item: (item[0][0], item[0][1]),
    ):
        for M in range(-int(L), int(L) + 1):
            columns.append(_sympy().Matrix(multiplet[int(M)]))
            labels.append((int(copy_index), tuple(int(x) for x in carrier_index), int(M)))
    if not columns:
        return _sympy().zeros(int(sector.raw_dim), 0), tuple()
    return _sympy().Matrix.hstack(*columns), tuple(labels)


def _highest_weight_basis_matrix_for_L(
    sector,
    L,
):
    L = int(L)
    columns = []
    labels = []
    for key, vector in sorted(
        sector.canonical_highest_weight_vectors_by_L.get(L, {}).items(),
        key=lambda item: (item[0][0], item[0][1]),
    ):
        columns.append(_sympy().Matrix(vector))
        labels.append((int(key[0]), tuple(int(x) for x in key[1])))
    if not columns:
        return _sympy().zeros(int(sector.raw_dim), 0), tuple()
    return _sympy().Matrix.hstack(*columns), tuple(labels)


def _basis_left_inverse(basis_matrix):
    gram = _sympy().simplify(basis_matrix.T * basis_matrix)
    return _sympy().simplify(gram.inv() * basis_matrix.T)


def _coupled_input_basis_matrix(
    left_sector,
    right_sector,
    *,
    left_L,
    right_L,
    output_L,
):
    left_L = int(left_L)
    right_L = int(right_L)
    output_L = int(output_L)
    columns = []
    labels = []
    left_items = sorted(left_sector.lowered_multiplets_by_L.get(left_L, {}).items(), key=lambda item: (item[0][0], item[0][1]))
    right_items = sorted(right_sector.lowered_multiplets_by_L.get(right_L, {}).items(), key=lambda item: (item[0][0], item[0][1]))
    rows = int(left_sector.raw_dim) * int(right_sector.raw_dim)
    for left_key, left_multiplet in left_items:
        for right_key, right_multiplet in right_items:
            for M_out in range(-output_L, output_L + 1):
                vector = _sympy().zeros(rows, 1)
                for M_left in range(-left_L, left_L + 1):
                    M_right = int(M_out - M_left)
                    if M_right < -right_L or M_right > right_L:
                        continue
                    coeff = cg_exact(left_L, M_left, right_L, M_right, output_L, M_out)
                    if coeff == 0:
                        continue
                    vector += _sympy().simplify(coeff) * _sympy().kronecker_product(
                        _sympy().Matrix(left_multiplet[int(M_left)]),
                        _sympy().Matrix(right_multiplet[int(M_right)]),
                    )
                columns.append(_sympy().Matrix(vector))
                labels.append(
                    (
                        (int(left_key[0]), tuple(int(x) for x in left_key[1])),
                        (int(right_key[0]), tuple(int(x) for x in right_key[1])),
                        int(M_out),
                    )
                )
    if not columns:
        return _sympy().zeros(rows, 0), tuple()
    return _sympy().Matrix.hstack(*columns), tuple(labels)


def _coupled_highest_weight_basis_matrix(
    left_sector,
    right_sector,
    *,
    left_L,
    right_L,
    output_L,
):
    left_L = int(left_L)
    right_L = int(right_L)
    output_L = int(output_L)
    columns = []
    labels = []
    left_items = sorted(
        left_sector.lowered_multiplets_by_L.get(left_L, {}).items(),
        key=lambda item: (item[0][0], item[0][1]),
    )
    right_items = sorted(
        right_sector.lowered_multiplets_by_L.get(right_L, {}).items(),
        key=lambda item: (item[0][0], item[0][1]),
    )
    rows = int(left_sector.raw_dim) * int(right_sector.raw_dim)
    for left_key, left_multiplet in left_items:
        for right_key, right_multiplet in right_items:
            vector = _sympy().zeros(rows, 1)
            for M_left in range(-left_L, left_L + 1):
                M_right = int(output_L - M_left)
                if M_right < -right_L or M_right > right_L:
                    continue
                coeff = cg_exact(left_L, M_left, right_L, M_right, output_L, output_L)
                if coeff == 0:
                    continue
                vector += _sympy().simplify(coeff) * _sympy().kronecker_product(
                    _sympy().Matrix(left_multiplet[int(M_left)]),
                    _sympy().Matrix(right_multiplet[int(M_right)]),
                )
            columns.append(_sympy().Matrix(vector))
            labels.append(
                (
                    (int(left_key[0]), tuple(int(x) for x in left_key[1])),
                    (int(right_key[0]), tuple(int(x) for x in right_key[1])),
                )
            )
    if not columns:
        return _sympy().zeros(rows, 0), tuple()
    return _sympy().Matrix.hstack(*columns), tuple(labels)


def _raw_magnetic_basis_states(lin):
    ranges = [tuple(range(-int(l), int(l) + 1)) for l in tuple(lin)]
    return tuple(tuple(int(m) for m in state) for state in product(*ranges))


def _slot_reorder_matrix(lin, slot_order):
    lin = tuple(int(l) for l in lin)
    slot_order = tuple(int(idx) for idx in slot_order)
    if slot_order == tuple(range(len(lin))):
        return _sympy().eye(int(_sympy().prod([2 * int(l) + 1 for l in lin])) if lin else 1)
    old_states = _raw_magnetic_basis_states(lin)
    new_lin = tuple(int(lin[idx]) for idx in slot_order)
    new_states = _raw_magnetic_basis_states(new_lin)
    new_index = {state: idx for idx, state in enumerate(new_states)}
    entries = {}
    for old_index, state in enumerate(old_states):
        new_state = tuple(int(state[idx]) for idx in slot_order)
        entries[(int(new_index[new_state]), int(old_index))] = _sympy().Integer(1)
    return _sympy().SparseMatrix(len(new_states), len(old_states), entries)


def _slot_order_for_permutation_irrep(nin, lin, permutation_irrep):
    nin = tuple(int(x) for x in nin)
    lin = tuple(int(x) for x in lin)
    used = set()
    order = []
    for factor in permutation_irrep.subgroup.factors:
        key = (_base_channel_label(factor.channel_label), int(factor.l))
        matches = [
            idx
            for idx, pair in enumerate(zip(nin, lin, strict=True))
            if idx not in used and (pair[0], int(pair[1])) == key
        ]
        if len(matches) != int(factor.multiplicity):
            return tuple(range(len(nin)))
        order.extend(matches)
        used.update(matches)
    if len(order) != len(nin):
        return tuple(range(len(nin)))
    return tuple(int(idx) for idx in order)


@recordclass(('permutation_irrep', 'output_L', 'subgroup_multiplicity', 'output_sector', 'input_basis_labels', 'output_basis_labels', 'coupled_input_basis_matrix', 'output_basis_matrix', 'raw_projected_input_matrix', 'coordinate_matrix'), frozen = True)
class ExactMergedTensorProductIntertwiner:
    """Exact basis-level intertwiner from tree-local input to one merged global output sector."""


@recordclass(('nin', 'lin', 'left_L', 'right_L', 'output_L', 'left_permutation', 'right_permutation', 'merged_terms', 'intertwiners', 'notes'), frozen = True)
class ExactMergedTensorProductDecomposition:
    """Exact merged global decomposition for one generalized tensor product."""


@recordclass(('sector', 'output_L', 'permutation_irrep', 'branch_history'), frozen = True)
class ComposedGeneralizedTensorProductBranch:
    """One branch in a cached multi-step generalized tensor-product composition."""


def build_exact_merged_tensor_product_decomposition(
    left_sector,
    right_sector,
    *,
    left_L,
    right_L,
    output_L,
    target_permutation_keys = None,
):
    """Build the exact merged global tensor-product decomposition and intertwiners."""

    from .builder import ExactSymbolicProjectorGeneralizedBasisBuilder

    left_L = int(left_L)
    right_L = int(right_L)
    output_L = int(output_L)
    merged_terms = merged_permutation_irrep_product_terms(left_sector.permutation_irrep, right_sector.permutation_irrep)
    coupled_input_basis_matrix, input_basis_labels = _coupled_input_basis_matrix(
        left_sector,
        right_sector,
        left_L=left_L,
        right_L=right_L,
        output_L=output_L,
    )
    output_nin = tuple(int(x) for x in left_sector.nin + right_sector.nin)
    output_lin = tuple(int(x) for x in left_sector.lin + right_sector.lin)
    target_permutation_keys = None if target_permutation_keys is None else frozenset(str(key) for key in target_permutation_keys)
    intertwiners = []
    for term in merged_terms:
        if target_permutation_keys is not None and term.permutation_irrep.to_string() not in target_permutation_keys:
            continue
        slot_order = _slot_order_for_permutation_irrep(output_nin, output_lin, term.permutation_irrep)
        ordered_output_nin = tuple(int(output_nin[idx]) for idx in slot_order)
        ordered_output_lin = tuple(int(output_lin[idx]) for idx in slot_order)
        source_basis_for_term = coupled_input_basis_matrix
        if slot_order != tuple(range(len(output_nin))):
            source_basis_for_term = _sympy().simplify(_slot_reorder_matrix(output_lin, slot_order) * coupled_input_basis_matrix)
        output_sector = ExactSymbolicProjectorGeneralizedBasisBuilder(ordered_output_nin, ordered_output_lin, term.permutation_irrep).build()
        if int(output_L) not in output_sector.lowered_multiplets_by_L:
            continue
        output_basis_matrix, output_basis_labels = _sector_basis_matrix_for_L(output_sector, int(output_L))
        output_left_inverse = _basis_left_inverse(output_basis_matrix)
        raw_projected_input_matrix = _sympy().simplify(output_sector.projector_matrix * source_basis_for_term)
        coordinate_matrix = _sympy().simplify(output_left_inverse * raw_projected_input_matrix)
        intertwiners.append(
            ExactMergedTensorProductIntertwiner(
                permutation_irrep=term.permutation_irrep,
                output_L=int(output_L),
                subgroup_multiplicity=int(term.multiplicity),
                output_sector=output_sector,
                input_basis_labels=input_basis_labels,
                output_basis_labels=output_basis_labels,
                coupled_input_basis_matrix=source_basis_for_term,
                output_basis_matrix=output_basis_matrix,
                raw_projected_input_matrix=raw_projected_input_matrix,
                coordinate_matrix=coordinate_matrix,
            )
        )
    return ExactMergedTensorProductDecomposition(
        nin=output_nin,
        lin=output_lin,
        left_L=int(left_L),
        right_L=int(right_L),
        output_L=int(output_L),
        left_permutation=left_sector.permutation_irrep,
        right_permutation=right_sector.permutation_irrep,
        merged_terms=merged_terms,
        intertwiners=tuple(intertwiners),
        notes=(
            "Merged subgroup conversion is exact and symbolic, built from subgroup character inner products.",
            "Intertwiners are exact CG-coupled input bases re-expanded in canonical merged output bases.",
            "This remains part of the separate generalized path and does not affect the fast ACE/ye3t runtime path.",
        ),
    )


def build_exact_change_of_group_basis_map(
    left_sector,
    right_sector,
    *,
    left_L,
    right_L,
    output_L,
    tree_type = "balanced",
    node_span = (0, 0),
    target_permutation_keys = None,
):
    """Build an exact canonical highest-weight basis map across one regrouping edge."""

    tree_type = validate_tree_type(tree_type)
    left_L = int(left_L)
    right_L = int(right_L)
    output_L = int(output_L)
    decomposition = build_exact_merged_tensor_product_decomposition(
        left_sector,
        right_sector,
        left_L=left_L,
        right_L=right_L,
        output_L=output_L,
        target_permutation_keys=target_permutation_keys,
    )
    source_basis_matrix, source_basis_labels = _coupled_highest_weight_basis_matrix(
        left_sector,
        right_sector,
        left_L=left_L,
        right_L=right_L,
        output_L=output_L,
    )
    branches = []
    for intertwiner in decomposition.intertwiners:
        target_basis_matrix, target_basis_labels = _highest_weight_basis_matrix_for_L(
            intertwiner.output_sector,
            output_L,
        )
        if target_basis_matrix.cols == 0:
            continue
        target_left_inverse = _basis_left_inverse(target_basis_matrix)
        raw_projected_source_matrix = _sympy().simplify(intertwiner.output_sector.projector_matrix * source_basis_matrix)
        coordinate_matrix = _sympy().simplify(target_left_inverse * raw_projected_source_matrix)
        branches.append(
            ExactChangeOfGroupBasisBranch(
                permutation_irrep=intertwiner.permutation_irrep,
                output_L=int(output_L),
                source_basis_labels=source_basis_labels,
                target_basis_labels=target_basis_labels,
                source_basis_matrix=source_basis_matrix,
                target_basis_matrix=target_basis_matrix,
                raw_projected_source_matrix=raw_projected_source_matrix,
                coordinate_matrix=coordinate_matrix,
            )
        )
    return ExactChangeOfGroupBasisMap(
        tree_type=str(tree_type),
        node_span=(int(node_span[0]), int(node_span[1])),
        left_L=int(left_L),
        right_L=int(right_L),
        output_L=int(output_L),
        left_permutation=left_sector.permutation_irrep,
        right_permutation=right_sector.permutation_irrep,
        source_basis_labels=source_basis_labels,
        branches=tuple(branches),
        notes=(
            "Exact canonical highest-weight basis map between child-pair subgroup bases and merged subgroup bases.",
            "Basis ordering follows the canonical highest-weight carrier/copy convention from the generalized symbolic builder.",
            "The default generalized tree policy is balanced; explicit left trees must be requested by the caller.",
        ),
    )


def build_exact_lowered_change_of_group_basis_map(
    left_sector,
    right_sector,
    *,
    left_L,
    right_L,
    output_L,
    tree_type = "balanced",
    node_span = (0, 0),
    target_permutation_keys = None,
):
    """Build an exact canonical full-multiplet basis map across one regrouping edge."""

    tree_type = validate_tree_type(tree_type)
    left_L = int(left_L)
    right_L = int(right_L)
    output_L = int(output_L)
    decomposition = build_exact_merged_tensor_product_decomposition(
        left_sector,
        right_sector,
        left_L=left_L,
        right_L=right_L,
        output_L=output_L,
        target_permutation_keys=target_permutation_keys,
    )
    source_basis_matrix, source_basis_labels = _coupled_input_basis_matrix(
        left_sector,
        right_sector,
        left_L=left_L,
        right_L=right_L,
        output_L=output_L,
    )
    branches = tuple(
        ExactLoweredChangeOfGroupBasisBranch(
            permutation_irrep=intertwiner.permutation_irrep,
            output_L=int(output_L),
            subgroup_multiplicity=int(intertwiner.subgroup_multiplicity),
            output_sector=intertwiner.output_sector,
            source_basis_labels=intertwiner.input_basis_labels,
            target_basis_labels=intertwiner.output_basis_labels,
            source_basis_matrix=intertwiner.coupled_input_basis_matrix,
            target_basis_matrix=intertwiner.output_basis_matrix,
            raw_projected_source_matrix=intertwiner.raw_projected_input_matrix,
            coordinate_matrix=intertwiner.coordinate_matrix,
        )
        for intertwiner in decomposition.intertwiners
    )
    return ExactLoweredChangeOfGroupBasisMap(
        tree_type=str(tree_type),
        node_span=(int(node_span[0]), int(node_span[1])),
        left_L=int(left_L),
        right_L=int(right_L),
        output_L=int(output_L),
        left_permutation=left_sector.permutation_irrep,
        right_permutation=right_sector.permutation_irrep,
        source_basis_labels=source_basis_labels,
        branches=branches,
        notes=(
            "Exact canonical full lowered-multiplet basis map between child-pair subgroup bases and merged subgroup bases.",
            "Source labels run over canonical child-side copy/carrier indices together with magnetic quantum number M.",
            "Target labels run over canonical merged-side copy/carrier indices together with magnetic quantum number M.",
        ),
    )


def lowered_change_of_group_basis_map_report(
    basis_map,
):
    """Return a structured exact report for one lowered change-of-group map."""

    branch_reports = []
    for branch in basis_map.branches:
        branch_reports.append(
            {
                "permutation_irrep": branch.permutation_irrep.to_string(),
                "output_L": int(branch.output_L),
                "source_basis_dim": len(branch.source_basis_labels),
                "target_basis_dim": len(branch.target_basis_labels),
                "coordinate_shape": (int(branch.coordinate_matrix.rows), int(branch.coordinate_matrix.cols)),
                "coordinate_nnz": _sympy_matrix_nnz(branch.coordinate_matrix),
                "projected_source_shape": (int(branch.raw_projected_source_matrix.rows), int(branch.raw_projected_source_matrix.cols)),
                "target_basis_labels": tuple(branch.target_basis_labels),
            }
        )
    return {
        "tree_type": str(basis_map.tree_type),
        "node_span": tuple(int(x) for x in basis_map.node_span),
        "left_L": int(basis_map.left_L),
        "right_L": int(basis_map.right_L),
        "output_L": int(basis_map.output_L),
        "left_permutation": basis_map.left_permutation.to_string(),
        "right_permutation": basis_map.right_permutation.to_string(),
        "source_basis_dim": len(basis_map.source_basis_labels),
        "source_basis_labels": tuple(basis_map.source_basis_labels),
        "num_branches": len(basis_map.branches),
        "branches": tuple(branch_reports),
        "notes": tuple(str(note) for note in basis_map.notes),
    }


def format_lowered_change_of_group_basis_map_report(
    basis_map,
):
    """Return a readable text summary for one lowered change-of-group map."""

    report = lowered_change_of_group_basis_map_report(basis_map)
    lines = [
        "ExactLoweredChangeOfGroupBasisMap",
        f"tree_type={report['tree_type']} node_span={report['node_span']} output_L={report['output_L']}",
        f"left={report['left_permutation']} x right={report['right_permutation']}",
        f"source_basis_dim={report['source_basis_dim']} num_branches={report['num_branches']}",
    ]
    for index, branch in enumerate(report["branches"]):
        lines.append(
            "branch[{idx}] {perm} shape={shape} nnz={nnz} target_dim={target_dim}".format(
                idx=index,
                perm=branch["permutation_irrep"],
                shape=branch["coordinate_shape"],
                nnz=branch["coordinate_nnz"],
                target_dim=branch["target_basis_dim"],
            )
        )
    return "\n".join(lines)


def lowered_change_of_group_basis_map_heatmap_data(
    basis_map,
):
    """Return plot-ready heatmap payloads for one lowered change-of-group map."""

    branch_reports = []
    for branch in basis_map.branches:
        coordinate_shape = (int(branch.coordinate_matrix.rows), int(branch.coordinate_matrix.cols))
        projected_shape = (int(branch.raw_projected_source_matrix.rows), int(branch.raw_projected_source_matrix.cols))
        branch_reports.append(
            {
                "permutation_irrep": branch.permutation_irrep.to_string(),
                "output_L": int(branch.output_L),
                "coordinate_shape": coordinate_shape,
                "projected_source_shape": projected_shape,
                "coordinate_abs": _sympy_matrix_abs_heatmap_data(branch.coordinate_matrix),
                "projected_source_abs": _sympy_matrix_abs_heatmap_data(branch.raw_projected_source_matrix),
                "coordinate_nnz": _sympy_matrix_nnz(branch.coordinate_matrix),
                "projected_source_nnz": _sympy_matrix_nnz(branch.raw_projected_source_matrix),
                "source_basis_labels": tuple(branch.source_basis_labels),
                "target_basis_labels": tuple(branch.target_basis_labels),
            }
        )
    return {
        "tree_type": str(basis_map.tree_type),
        "node_span": tuple(int(x) for x in basis_map.node_span),
        "left_permutation": basis_map.left_permutation.to_string(),
        "right_permutation": basis_map.right_permutation.to_string(),
        "output_L": int(basis_map.output_L),
        "num_branches": len(branch_reports),
        "branches": tuple(branch_reports),
    }


def plot_lowered_change_of_group_basis_map_heatmaps(
    basis_map,
    *,
    branch_indices = None,
    cmap = "viridis",
    include_colorbar = True,
):
    """Plot absolute-value heatmaps for exact lowered change-of-group matrices."""

    import matplotlib.pyplot as plt

    heatmap = lowered_change_of_group_basis_map_heatmap_data(basis_map)
    selected = (
        tuple(int(index) for index in branch_indices)
        if branch_indices is not None
        else tuple(range(len(heatmap["branches"])))
    )
    if not selected:
        raise ValueError("At least one branch index is required to plot heatmaps.")

    fig, axes = plt.subplots(
        len(selected),
        2,
        figsize=(8.8, max(3.2, 3.0 * len(selected))),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, branch_index in enumerate(selected):
        branch = heatmap["branches"][branch_index]
        coordinate_ax = axes[row_index][0]
        projected_ax = axes[row_index][1]

        coord_im = coordinate_ax.imshow(branch["coordinate_abs"], aspect="auto", cmap=cmap)
        coordinate_ax.set_title(
            f"{branch['permutation_irrep']} coordinate |.|"
        )
        coordinate_ax.set_xlabel("source basis")
        coordinate_ax.set_ylabel("target basis")

        projected_im = projected_ax.imshow(branch["projected_source_abs"], aspect="auto", cmap=cmap)
        projected_ax.set_title(
            f"{branch['permutation_irrep']} projected source |.|"
        )
        projected_ax.set_xlabel("source basis")
        projected_ax.set_ylabel("raw ambient basis")

        if include_colorbar:
            fig.colorbar(coord_im, ax=coordinate_ax, shrink=0.8)
            fig.colorbar(projected_im, ax=projected_ax, shrink=0.8)

    fig.suptitle(
        "Exact Lowered Change-of-Group Heatmaps\n"
        f"tree_type={heatmap['tree_type']} output_L={heatmap['output_L']}",
        fontsize=11,
    )
    return fig


class CachedGeneralizedTensorProductComposer:
    """Cached exact composition helper for deeper generalized tensor-product trees.

    This wrapper keeps the generalized symbolic path usable when chaining many
    exact couplings by caching:

    - generalized output sectors
    - exact binary merged decompositions

    It remains entirely separate from the fast ACE / runtime path.
    """

    def __init__(
        self,
        *,
        max_sector_entries = 256,
        max_decomposition_entries = 256,
        max_change_of_group_entries = 256,
        max_basis_change_entries = 128,
        max_lowered_basis_change_entries = 128,
        max_schedule_entries = 64,
    ):
        self._sector_cache = _BoundedObjectCache(max_entries=max_sector_entries)
        self._decomposition_cache = _BoundedObjectCache(max_entries=max_decomposition_entries)
        self._change_of_group_cache = _BoundedObjectCache(max_entries=max_change_of_group_entries)
        self._basis_change_cache = _BoundedObjectCache(max_entries=max_basis_change_entries)
        self._lowered_basis_change_cache = _BoundedObjectCache(max_entries=max_lowered_basis_change_entries)
        self._schedule_cache = _BoundedObjectCache(max_entries=max_schedule_entries)

    def clear_caches(self):
        for cache in (
            self._sector_cache,
            self._decomposition_cache,
            self._change_of_group_cache,
            self._basis_change_cache,
            self._lowered_basis_change_cache,
            self._schedule_cache,
        ):
            cache.clear()

    def cache_stats(self):
        caches = {
            "sectors": self._sector_cache.stats(),
            "decompositions": self._decomposition_cache.stats(),
            "change_of_group": self._change_of_group_cache.stats(),
            "basis_change": self._basis_change_cache.stats(),
            "lowered_basis_change": self._lowered_basis_change_cache.stats(),
            "schedules": self._schedule_cache.stats(),
        }
        total_entries = sum(int(entry["size"]) for entry in caches.values())
        total_hits = sum(int(entry["hit_count"]) for entry in caches.values())
        total_misses = sum(int(entry["miss_count"]) for entry in caches.values())
        return {
            "total_entries": int(total_entries),
            "total_hits": int(total_hits),
            "total_misses": int(total_misses),
            "caches": caches,
        }

    def sector_for(
        self,
        nin,
        lin,
        permutation_irrep,
    ):
        from .builder import ExactSymbolicProjectorGeneralizedBasisBuilder

        key = (
            tuple(int(x) for x in nin),
            tuple(int(x) for x in lin),
            _permutation_irrep_signature(permutation_irrep),
        )
        cached = self._sector_cache.get(key)
        if cached is not None:
            return cached
        sector = ExactSymbolicProjectorGeneralizedBasisBuilder(tuple(int(x) for x in nin), tuple(int(x) for x in lin), permutation_irrep).build()
        self._sector_cache.set(key, sector)
        return sector

    def binary_decomposition(
        self,
        left_sector,
        right_sector,
        *,
        left_L,
        right_L,
        output_L,
    ):
        key = (
            tuple(int(x) for x in left_sector.nin),
            tuple(int(x) for x in left_sector.lin),
            _permutation_irrep_signature(left_sector.permutation_irrep),
            int(left_L),
            tuple(int(x) for x in right_sector.nin),
            tuple(int(x) for x in right_sector.lin),
            _permutation_irrep_signature(right_sector.permutation_irrep),
            int(right_L),
            int(output_L),
        )
        cached = self._decomposition_cache.get(key)
        if cached is not None:
            return cached
        decomposition = build_exact_merged_tensor_product_decomposition(
            left_sector,
            right_sector,
            left_L=int(left_L),
            right_L=int(right_L),
            output_L=int(output_L),
        )
        self._decomposition_cache.set(key, decomposition)
        return decomposition

    def change_of_group_map(
        self,
        left_terms,
        right_terms,
        *,
        tree_type = "balanced",
        node_span = (0, 0),
    ):
        key = (
            _term_collection_signature(left_terms),
            _term_collection_signature(right_terms),
            str(tree_type),
            tuple(int(x) for x in node_span),
        )
        cached = self._change_of_group_cache.get(key)
        if cached is not None:
            return cached
        mapping = build_exact_change_of_group_map(
            left_terms,
            right_terms,
            tree_type=str(tree_type),
            node_span=tuple(int(x) for x in node_span),
        )
        self._change_of_group_cache.set(key, mapping)
        return mapping

    def balanced_schedule(
        self,
        num_inputs,
    ):
        key = ("balanced", int(num_inputs))
        cached = self._schedule_cache.get(key)
        if cached is not None:
            return cached
        schedule = build_generalized_tensor_product_schedule(int(num_inputs), tree_type="balanced")
        self._schedule_cache.set(key, schedule)
        return schedule

    def schedule_for(
        self,
        num_inputs,
        *,
        tree_type = "balanced",
    ):
        tree_type = validate_tree_type(tree_type)
        key = (str(tree_type), int(num_inputs))
        cached = self._schedule_cache.get(key)
        if cached is not None:
            return cached
        schedule = build_generalized_tensor_product_schedule(int(num_inputs), tree_type=str(tree_type))
        self._schedule_cache.set(key, schedule)
        return schedule

    def change_of_group_basis_map(
        self,
        left_sector,
        right_sector,
        *,
        left_L,
        right_L,
        output_L,
        tree_type = "balanced",
        node_span = (0, 0),
    ):
        tree_type = validate_tree_type(tree_type)
        key = (
            tuple(int(x) for x in left_sector.nin),
            tuple(int(x) for x in left_sector.lin),
            _permutation_irrep_signature(left_sector.permutation_irrep),
            int(left_L),
            tuple(int(x) for x in right_sector.nin),
            tuple(int(x) for x in right_sector.lin),
            _permutation_irrep_signature(right_sector.permutation_irrep),
            int(right_L),
            int(output_L),
            str(tree_type),
            tuple(int(x) for x in node_span),
        )
        cached = self._basis_change_cache.get(key)
        if cached is not None:
            return cached
        mapping = build_exact_change_of_group_basis_map(
            left_sector,
            right_sector,
            left_L=int(left_L),
            right_L=int(right_L),
            output_L=int(output_L),
            tree_type=str(tree_type),
            node_span=tuple(int(x) for x in node_span),
        )
        self._basis_change_cache.set(key, mapping)
        return mapping

    def lowered_change_of_group_basis_map(
        self,
        left_sector,
        right_sector,
        *,
        left_L,
        right_L,
        output_L,
        tree_type = "balanced",
        node_span = (0, 0),
        target_permutation_keys = None,
    ):
        tree_type = validate_tree_type(tree_type)
        target_permutation_keys = None if target_permutation_keys is None else tuple(sorted(str(key) for key in target_permutation_keys))
        key = (
            tuple(int(x) for x in left_sector.nin),
            tuple(int(x) for x in left_sector.lin),
            _permutation_irrep_signature(left_sector.permutation_irrep),
            int(left_L),
            tuple(int(x) for x in right_sector.nin),
            tuple(int(x) for x in right_sector.lin),
            _permutation_irrep_signature(right_sector.permutation_irrep),
            int(right_L),
            int(output_L),
            str(tree_type),
            tuple(int(x) for x in node_span),
            target_permutation_keys,
        )
        cached = self._lowered_basis_change_cache.get(key)
        if cached is not None:
            return cached
        mapping = build_exact_lowered_change_of_group_basis_map(
            left_sector,
            right_sector,
            left_L=int(left_L),
            right_L=int(right_L),
            output_L=int(output_L),
            tree_type=str(tree_type),
            node_span=tuple(int(x) for x in node_span),
            target_permutation_keys=target_permutation_keys,
        )
        self._lowered_basis_change_cache.set(key, mapping)
        return mapping

    def compose_left_associated_chain(
        self,
        labeled_inputs,
        output_Ls,
    ):
        """Chain cached exact binary couplings along a left-associated tree."""

        labeled_inputs = tuple((sector, int(L)) for sector, L in labeled_inputs)
        output_Ls = tuple(int(L) for L in output_Ls)
        if len(labeled_inputs) == 0:
            return tuple()
        if len(labeled_inputs) == 1:
            sector, L_value = labeled_inputs[0]
            return (
                ComposedGeneralizedTensorProductBranch(
                    sector=sector,
                    output_L=int(L_value),
                    permutation_irrep=sector.permutation_irrep,
                    branch_history=(sector.permutation_irrep.to_string(),),
                ),
            )
        if len(output_Ls) != len(labeled_inputs) - 1:
            raise ValueError(
                f"Expected {len(labeled_inputs) - 1} output-L choices for {len(labeled_inputs)} inputs, got {len(output_Ls)}."
            )

        branches = (
            ComposedGeneralizedTensorProductBranch(
                sector=labeled_inputs[0][0],
                output_L=int(labeled_inputs[0][1]),
                permutation_irrep=labeled_inputs[0][0].permutation_irrep,
                branch_history=(labeled_inputs[0][0].permutation_irrep.to_string(),),
            ),
        )
        for idx, ((next_sector, next_L), output_L) in enumerate(zip(labeled_inputs[1:], output_Ls, strict=True), start=1):
            new_branches = []
            for branch in branches:
                decomposition = self.binary_decomposition(
                    branch.sector,
                    next_sector,
                    left_L=int(branch.output_L),
                    right_L=int(next_L),
                    output_L=int(output_L),
                )
                for intertwiner in decomposition.intertwiners:
                    new_branches.append(
                        ComposedGeneralizedTensorProductBranch(
                            sector=intertwiner.output_sector,
                            output_L=int(output_L),
                            permutation_irrep=intertwiner.permutation_irrep,
                            branch_history=branch.branch_history + (
                                f"step{idx}:{intertwiner.permutation_irrep.to_string()}@L={int(output_L)}",
                            ),
                        )
                    )
            branches = tuple(new_branches)
        return branches

    def compose_with_schedule(
        self,
        labeled_inputs,
        *,
        schedule,
        output_L_by_span,
    ):
        """Execute a scheduled exact composition plan with cached intermediates."""

        labeled_inputs = tuple((sector, int(L)) for sector, L in labeled_inputs)
        if len(labeled_inputs) != int(schedule.num_inputs):
            raise ValueError(
                f"Schedule expects {schedule.num_inputs} inputs, got {len(labeled_inputs)}."
            )
        nodes_by_id = {node.node_id: node for node in schedule.nodes}
        branches_by_node = {}
        terms_by_node = {}
        change_maps_by_node = {}
        max_intermediate_branch_count = 0
        total_binary_decompositions = 0

        for node in schedule.nodes:
            if not node.is_leaf:
                continue
            start, end = node.leaf_span
            if end - start != 1:
                raise RuntimeError(f"Leaf node {node.node_id} does not cover exactly one input.")
            sector, L_value = labeled_inputs[start]
            branches_by_node[node.node_id] = (
                ComposedGeneralizedTensorProductBranch(
                    sector=sector,
                    output_L=int(L_value),
                    permutation_irrep=sector.permutation_irrep,
                    branch_history=(f"{node.node_id}:{sector.permutation_irrep.to_string()}@L={int(L_value)}",),
                ),
            )
            terms_by_node[node.node_id] = (
                MergedPermutationIrrepTerm(permutation_irrep=sector.permutation_irrep, multiplicity=1),
            )
            max_intermediate_branch_count = max(max_intermediate_branch_count, 1)

        for node_id in schedule.internal_node_ids_postorder:
            node = nodes_by_id[node_id]
            if node.left_child_id is None or node.right_child_id is None:
                raise RuntimeError(f"Internal node {node_id} is missing children.")
            span = tuple(int(x) for x in node.leaf_span)
            if span not in output_L_by_span:
                raise ValueError(
                    f"Missing output_L for span {span!r}. Provide output_L_by_span for every internal schedule node."
                )
            target_L = int(output_L_by_span[span])
            left_branches = branches_by_node[node.left_child_id]
            right_branches = branches_by_node[node.right_child_id]
            left_terms = terms_by_node[node.left_child_id]
            right_terms = terms_by_node[node.right_child_id]
            change_maps_by_node[node_id] = self.change_of_group_map(
                left_terms,
                right_terms,
                tree_type=str(schedule.tree_type),
                node_span=span,
            )
            merged_terms = change_maps_by_node[node_id].target_terms
            terms_by_node[node_id] = merged_terms

            new_branches = []
            for left_branch in left_branches:
                for right_branch in right_branches:
                    decomposition = self.binary_decomposition(
                        left_branch.sector,
                        right_branch.sector,
                        left_L=int(left_branch.output_L),
                        right_L=int(right_branch.output_L),
                        output_L=int(target_L),
                    )
                    total_binary_decompositions += 1
                    for intertwiner in decomposition.intertwiners:
                        new_branches.append(
                            ComposedGeneralizedTensorProductBranch(
                                sector=intertwiner.output_sector,
                                output_L=int(target_L),
                                permutation_irrep=intertwiner.permutation_irrep,
                                branch_history=left_branch.branch_history
                                + right_branch.branch_history
                                + (f"{node_id}:{intertwiner.permutation_irrep.to_string()}@L={int(target_L)}",),
                            )
                        )
            branches_by_node[node_id] = tuple(new_branches)
            max_intermediate_branch_count = max(max_intermediate_branch_count, len(new_branches))

        root_branches = branches_by_node.get(schedule.root_id, tuple())
        return ScheduledGeneralizedTensorProductResult(
            schedule=schedule,
            root_branches=root_branches,
            branches_by_node=branches_by_node,
            change_of_group_maps_by_node=change_maps_by_node,
            max_intermediate_branch_count=int(max_intermediate_branch_count),
            total_binary_decompositions=int(total_binary_decompositions),
        )

    def compose_balanced_pairwise_tree(
        self,
        labeled_inputs,
        *,
        output_L_by_span,
    ):
        """Execute a balanced pairwise exact composition schedule."""

        schedule = self.balanced_schedule(len(tuple(labeled_inputs)))
        return self.compose_with_schedule(
            labeled_inputs,
            schedule=schedule,
            output_L_by_span=output_L_by_span,
        )

    def compose_tree(
        self,
        labeled_inputs,
        *,
        output_L_by_span,
        tree_type = "balanced",
    ):
        """Execute an exact generalized composition using a user-selected tree type."""

        schedule = self.schedule_for(len(tuple(labeled_inputs)), tree_type=tree_type)
        return self.compose_with_schedule(
            labeled_inputs,
            schedule=schedule,
            output_L_by_span=output_L_by_span,
        )


__all__ = [
    "CouplingScheduleNode",
    "ExactChangeOfGroupBasisBranch",
    "ExactChangeOfGroupBasisMap",
    "ExactLoweredChangeOfGroupBasisBranch",
    "ExactLoweredChangeOfGroupBasisMap",
    "ExactChangeOfGroupMap",
    "ExactChangeOfGroupSourcePair",
    "ExactMergedTensorProductDecomposition",
    "ExactMergedTensorProductIntertwiner",
    "GeneralizedTensorProductSchedule",
    "MergedPermutationIrrepTerm",
    "CachedGeneralizedTensorProductComposer",
    "ComposedGeneralizedTensorProductBranch",
    "ScheduledGeneralizedTensorProductResult",
    "TreeAwareTensorProductRule",
    "build_balanced_pairwise_schedule",
    "build_exact_change_of_group_basis_map",
    "build_exact_change_of_group_map",
    "build_exact_lowered_change_of_group_basis_map",
    "build_exact_merged_tensor_product_decomposition",
    "format_lowered_change_of_group_basis_map_report",
    "lowered_change_of_group_basis_map_heatmap_data",
    "lowered_change_of_group_basis_map_report",
    "build_generalized_tensor_product_schedule",
    "build_left_schedule",
    "build_tree_aware_generalized_tensor_product",
    "build_tree_aware_tensor_product_rule",
    "merge_permutation_irrep_sequence",
    "merged_permutation_irrep_product_terms",
    "plot_lowered_change_of_group_basis_map_heatmaps",
    "restrict_merged_permutation_irrep_to_children",
    "tree_aware_permutation_irrep_product",
    "tree_aware_tensor_product_output_labels",
]
