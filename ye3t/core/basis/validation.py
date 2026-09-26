
"""Input validation for exact ACE labels and target angular momenta.

The exact algebraic construction only works for integer SO(3) angular momenta
that satisfy the SO(3) coupling rules implied by the selected leaf tuple. These
checks are cheap and help fail early on impossible inputs.
"""

from itertools import combinations_with_replacement
from collections import Counter
from math import comb, factorial

from .characters import cg_allowed


def validate_tree_type(tree_type):
    normalized = "balanced" if tree_type is None else str(tree_type).strip().lower()
    if normalized in {"balanced", "balanced_pairwise", "pairwise"}:
        return "balanced"
    if normalized in {"left", "left_justified", "left-justified"}:
        return "left"
    raise ValueError(f"Unknown tree_type '{tree_type}'. Use 'balanced' or 'left'.")


def validate_leaf_quantum_numbers(nin, lin):
    if len(nin) != len(lin):
        raise ValueError("nin and lin must have the same length.")
    if len(nin) == 0:
        raise ValueError("nin and lin must be non-empty.")
    if any(int(n) < 1 for n in nin):
        raise ValueError(f"All radial quantum numbers n must be positive integers; got nin={tuple(nin)}")
    if any(int(l) < 0 for l in lin):
        raise ValueError(f"All angular quantum numbers l must be non-negative integers; got lin={tuple(lin)}")


def canonicalize_leaf_quantum_numbers(nin, lin):
    """Return the canonical leaf ordering used by the exact basis machinery.

    The current convention is ``l``-first:

    1. group leaves by repeated angular channel ``l``,
    2. order those groups by decreasing multiplicity and then by ``l`` value,
    3. within each repeated-``l`` Young subgroup, order the non-angular labels
       ``n`` increasingly.

    This keeps repeated angular channels contiguous and canonicalizes
    permutations inside the induced Young subgroups.
    """
    validate_leaf_quantum_numbers(nin, lin)
    l_counts = Counter(int(l) for l in lin)
    canonical_pairs = tuple(
        sorted(
            ((int(n), int(l)) for n, l in zip(nin, lin)),
            key=lambda pair: (-int(l_counts[int(pair[1])]), int(pair[1]), int(pair[0])),
        )
    )
    return (
        tuple(n for n, _ in canonical_pairs),
        tuple(l for _, l in canonical_pairs),
    )


def iter_canonical_leaf_labelings(
    rank,
    n_values,
    l_values,
    multiplicity_partitions=None,
):
    """Enumerate unique canonical leaf labelings under the ``l``-first convention.

    ``multiplicity_partitions`` restricts the multiplicities of identical
    ``(n, l)`` pairs.  This bounded route generates only the requested content
    patterns and does not first materialize every rank-``N`` multiset.
    """
    n_values = tuple(int(n) for n in n_values)
    l_values = tuple(int(l) for l in l_values)
    pair_values = tuple(
        sorted({(n, l) for n in n_values for l in l_values})
    )
    if multiplicity_partitions is not None:
        partitions = _normalize_leaf_multiplicity_partitions(
            rank,
            multiplicity_partitions,
        )

        def extend(multiplicities, position, used, selected):
            if position == len(multiplicities):
                pairs = tuple(
                    pair_values[pair_index]
                    for pair_index, multiplicity in zip(
                        selected,
                        multiplicities,
                        strict=True,
                    )
                    for _ in range(multiplicity)
                )
                yield canonicalize_leaf_quantum_numbers(
                    tuple(n for n, _ in pairs),
                    tuple(l for _, l in pairs),
                )
                return
            lower = 0
            if position and multiplicities[position] == multiplicities[position - 1]:
                lower = selected[position - 1] + 1
            for pair_index in range(lower, len(pair_values)):
                if pair_index in used:
                    continue
                used.add(pair_index)
                selected.append(pair_index)
                yield from extend(
                    multiplicities,
                    position + 1,
                    used,
                    selected,
                )
                selected.pop()
                used.remove(pair_index)

        for multiplicities in partitions:
            yield from extend(multiplicities, 0, set(), [])
        return
    seen = set()
    items = []
    for pair_multiset in combinations_with_replacement(pair_values, int(rank)):
        canonical = canonicalize_leaf_quantum_numbers(
            tuple(n for n, _ in pair_multiset),
            tuple(l for _, l in pair_multiset),
        )
        if canonical in seen:
            continue
        seen.add(canonical)
        items.append(canonical)
    for item in sorted(items, key=lambda value: (value[1], value[0])):
        yield item


def _normalize_leaf_multiplicity_partitions(rank, partitions):
    rank = int(rank)
    normalized = []
    seen = set()
    for raw in partitions:
        partition = tuple(sorted((int(value) for value in raw), reverse=True))
        if not partition or any(value <= 0 for value in partition):
            raise ValueError("Leaf multiplicity partitions must contain positive integers.")
        if sum(partition) != rank:
            raise ValueError(
                "Leaf multiplicity partitions must sum to the requested rank."
            )
        if partition not in seen:
            normalized.append(partition)
            seen.add(partition)
    if not normalized:
        raise ValueError("At least one leaf multiplicity partition is required.")
    return tuple(sorted(normalized, reverse=True))


def count_canonical_leaf_labelings(
    rank,
    n_values,
    l_values,
    multiplicity_partitions=None,
):
    """Count canonical leaf labelings without materializing their tuples."""

    rank = int(rank)
    if rank <= 0:
        raise ValueError("rank must be positive.")
    n_values = tuple(int(n) for n in n_values)
    l_values = tuple(int(l) for l in l_values)
    pair_count = len(
        {
            (n, l)
            for n in n_values
            for l in l_values
        }
    )
    if pair_count == 0:
        return 0
    if multiplicity_partitions is None:
        return comb(pair_count + rank - 1, rank)
    total = 0
    for partition in _normalize_leaf_multiplicity_partitions(
        rank,
        multiplicity_partitions,
    ):
        block_count = len(partition)
        if block_count > pair_count:
            continue
        assignments = factorial(pair_count) // factorial(pair_count - block_count)
        for repeated_block_count in Counter(partition).values():
            assignments //= factorial(repeated_block_count)
        total += assignments
    return int(total)


def reachable_total_angular_momenta(lin):
    """Return the exact set of final SO(3) angular momenta reachable from ``lin``.

    This uses Clebsch-Gordan recursion directly. It is stricter and more correct
    than simple parity heuristics, which can reject valid couplings such as
    ``lin=[1, 1], L_target=1``.
    """
    validate_leaf_quantum_numbers([1] * len(lin), lin)
    reachable = {0}
    for l in (int(val) for val in lin):
        next_reachable = set()
        for left_L in reachable:
            next_reachable.update(int(out_L) for out_L in cg_allowed(int(left_L), l))
        reachable = next_reachable
    return reachable


def is_target_angular_momentum_allowed(lin, L_target):
    if int(L_target) < 0:
        return False
    return int(L_target) in reachable_total_angular_momenta(lin)


def validate_target_angular_momentum(lin, L_target):
    if int(L_target) < 0:
        raise ValueError(f"L_target must be non-negative; got {L_target}")
    if not is_target_angular_momentum_allowed(lin, L_target):
        reachable = sorted(reachable_total_angular_momenta(lin))
        raise ValueError(
            "Invalid angular-momentum input: the requested L_target is not reachable "
            "under exact Clebsch-Gordan coupling of the supplied leaf angular momenta. "
            f"Got lin={tuple(lin)} and L_target={L_target}. Reachable values are {reachable}."
        )
