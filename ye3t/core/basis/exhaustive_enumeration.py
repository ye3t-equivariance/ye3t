
"""Exhaustive orbit-pattern and label enumeration helpers.

This module provides reusable enumeration logic for both ``ye3t`` and
``ye3t_methods`` without pulling benchmark or plotting code into the runtime
packages.
"""

from collections import Counter
from functools import lru_cache
from itertools import permutations
from math import ceil, floor, isclose

from .validation import canonicalize_leaf_quantum_numbers

__all__ = [
    "build_label_bank",
    "canonicalize_composite_leaf_quantum_numbers",
    "enumerate_rank_labels",
    "integer_partitions",
    "make_spec_key",
    "make_target_l_avs",
    "eta_labels",
    "pair_partition",
    "partition_from_values",
]


@lru_cache(maxsize=None)
def integer_partitions(n, max_part = None):
    """Return all integer partitions of ``n`` in non-increasing order."""
    if n < 0:
        return ()
    if max_part is None or max_part > n:
        max_part = n
    if n == 0:
        return ((),)

    out = []
    for first in range(max_part, 0, -1):
        for rest in integer_partitions(n - first, first):
            out.append((first,) + rest)
    return tuple(out)


def _validate_partition(rank, part):
    if part is None:
        return None
    part = tuple(int(x) for x in part)
    if any(x <= 0 for x in part):
        raise ValueError("All partition entries must be positive.")
    if sum(part) != rank:
        raise ValueError(f"sum(part)={sum(part)} must equal rank={rank}.")
    return part


def _normalize_partition_list(
    rank,
    parts,
):
    if parts is None:
        return None
    return tuple(_validate_partition(rank, p) for p in parts)


def _normalize_mu_values(values):
    if values is None:
        return None
    values = tuple(int(x) for x in values)
    if any(x <= 0 for x in values):
        raise ValueError("All mu_values entries must be positive.")
    if len(set(values)) != len(values):
        raise ValueError("mu_values entries must be distinct.")
    return tuple(sorted(values))


def partition_from_values(values):
    """Return the multiplicity partition induced by repeated values."""
    counts = Counter(values)
    return tuple(sorted(counts.values(), reverse=True))


def pair_partition(n_in, l_in):
    """Return the multiplicity partition of repeated ``(n, l)`` leaf pairs."""
    counts = Counter(zip(n_in, l_in))
    return tuple(sorted(counts.values(), reverse=True))


def eta_labels(mu_in, n_in):
    """Return canonical composite non-angular channels ``eta=(mu,n)``."""
    return _relabel_by_first_appearance(tuple((int(mu), int(n)) for mu, n in zip(mu_in, n_in)))


@lru_cache(maxsize=None)
def make_target_l_avs(l_av_max, l_av_inc):
    """Build the target average-``l`` grid used by the exhaustive enumerator."""
    if l_av_inc <= 0:
        raise ValueError("l_av_inc must be positive.")
    if l_av_max < 0:
        raise ValueError("l_av_max must be nonnegative.")

    vals = []
    k = 0
    while True:
        x = k * l_av_inc
        if x > l_av_max + 1e-12:
            break
        vals.append(round(x, 12))
        k += 1

    if vals and not isclose(vals[-1], l_av_max, abs_tol=1e-12):
        if l_av_max > vals[-1]:
            vals.append(round(l_av_max, 12))
    elif not vals:
        vals = [round(l_av_max, 12)]

    return tuple(vals)


def _relabel_by_first_appearance(values):
    mapping = {}
    next_label = 1
    out = []
    for value in values:
        if value not in mapping:
            mapping[value] = next_label
            next_label += 1
        out.append(mapping[value])
    return tuple(out)


@lru_cache(maxsize=None)
def _choose_l_blocks(
    num_blocks,
    target_l_av,
    strict_max_li = None,
    minimum_l = 1,
):
    """Choose distinct ``l``-values for each ``l``-orbit block."""
    if num_blocks <= 0:
        return ()
    minimum_l = int(minimum_l)
    if minimum_l < 0:
        raise ValueError("minimum_l must be nonnegative.")

    if num_blocks == 1:
        value = max(minimum_l, round(target_l_av))
        if strict_max_li is not None:
            value = min(value, strict_max_li)
        if value < minimum_l:
            raise ValueError("strict_max_li is smaller than minimum_l.")
        return (value,)

    if strict_max_li is not None:
        if strict_max_li < minimum_l + num_blocks - 1:
            raise ValueError(
                f"Cannot assign {num_blocks} distinct l-values from "
                f"minimum_l={minimum_l} through strict_max_li={strict_max_li}."
            )
        lo = minimum_l
        hi = strict_max_li - num_blocks + 1
    else:
        lo = minimum_l
        hi = None

    ideal_start = target_l_av - 0.5 * (num_blocks - 1)
    starts = {
        max(minimum_l, floor(ideal_start)),
        max(minimum_l, ceil(ideal_start)),
    }

    best = None
    best_score = None
    for start in starts:
        if hi is not None:
            start = min(max(lo, start), hi)

        vals = tuple(start + i for i in range(num_blocks))
        element_deviation = sum(abs(v - target_l_av) for v in vals)
        average_deviation = abs(sum(vals) / num_blocks - target_l_av)
        score = (element_deviation, average_deviation, vals)
        if best_score is None or score < best_score:
            best_score = score
            best = vals

    return best


def _equal_value_groups(part):
    groups = []
    i = 0
    while i < len(part):
        j = i + 1
        while j < len(part) and part[j] == part[i]:
            j += 1
        groups.append((i, j))
        i = j
    return groups


def _canonicalize_matrix(
    mat,
    row_part,
    col_part,
):
    arr = [list(row) for row in mat]
    row_groups = _equal_value_groups(row_part)
    col_groups = _equal_value_groups(col_part)

    changed = True
    while changed:
        changed = False

        for i0, i1 in row_groups:
            old = arr[i0:i1]
            new = sorted(old, reverse=True)
            if new != old:
                arr[i0:i1] = new
                changed = True

        nrows = len(arr)
        ncols = len(arr[0]) if nrows else 0
        cols = [tuple(arr[i][j] for i in range(nrows)) for j in range(ncols)]

        for j0, j1 in col_groups:
            old = cols[j0:j1]
            new = sorted(old, reverse=True)
            if new != old:
                cols[j0:j1] = new
                changed = True

        if changed:
            arr = [[cols[j][i] for j in range(len(cols))] for i in range(len(cols[0]))]

    return tuple(tuple(row) for row in arr)


@lru_cache(maxsize=None)
def _is_orbit_compatible(
    rank,
    n_orbits,
    l_orbits,
    pair_orbits,
):
    if sum(n_orbits) != rank or sum(l_orbits) != rank or sum(pair_orbits) != rank:
        return False
    if len(pair_orbits) > len(n_orbits) * len(l_orbits):
        return False
    if pair_orbits and max(pair_orbits) > min(max(n_orbits), max(l_orbits)):
        return False
    return True


@lru_cache(maxsize=None)
def _enumerate_pair_matrices(
    rank,
    n_orbits,
    l_orbits,
    pair_orbits,
):
    if not _is_orbit_compatible(rank, n_orbits, l_orbits, pair_orbits):
        return ()

    row_count = len(n_orbits)
    col_count = len(l_orbits)
    pair_blocks = tuple(sorted(pair_orbits, reverse=True))

    seen = set()
    out = []

    def rec(
        k,
        row_rem,
        col_rem,
        used_mask,
        entries,
        min_cell,
    ):
        if k == len(pair_blocks):
            if all(x == 0 for x in row_rem) and all(x == 0 for x in col_rem):
                mat = [[0] * col_count for _ in range(row_count)]
                for i, j, value in entries:
                    mat[i][j] = value
                canonical = _canonicalize_matrix(tuple(tuple(row) for row in mat), n_orbits, l_orbits)
                if canonical not in seen:
                    seen.add(canonical)
                    out.append(canonical)
            return

        remaining_total = sum(pair_blocks[k:])
        if sum(row_rem) != remaining_total or sum(col_rem) != remaining_total:
            return

        block = pair_blocks[k]
        continuing_same_run = k > 0 and pair_blocks[k - 1] == block
        next_same_run = k + 1 < len(pair_blocks) and pair_blocks[k + 1] == block
        start_cell = min_cell if continuing_same_run else 0

        for cell in range(start_cell, row_count * col_count):
            if (used_mask >> cell) & 1:
                continue

            i, j = divmod(cell, col_count)
            if row_rem[i] < block or col_rem[j] < block:
                continue

            new_row = list(row_rem)
            new_col = list(col_rem)
            new_row[i] -= block
            new_col[j] -= block

            if any(x < 0 for x in new_row) or any(x < 0 for x in new_col):
                continue

            rec(
                k + 1,
                tuple(new_row),
                tuple(new_col),
                used_mask | (1 << cell),
                entries + ((i, j, block),),
                cell + 1 if next_same_run else 0,
            )

    rec(
        k=0,
        row_rem=n_orbits,
        col_rem=l_orbits,
        used_mask=0,
        entries=(),
        min_cell=0,
    )
    return tuple(sorted(out))


@lru_cache(maxsize=None)
def _matrix_to_labels_cached(
    mat,
    target_l_av,
    strict_max_li,
    homogeneous_n,
    keep_l_ordered,
    minimum_l,
):
    row_count = len(mat)
    col_count = len(mat[0]) if row_count else 0

    if homogeneous_n and row_count > 1:
        raise ValueError("homogeneous_n=True only works when there is one n-block.")
    if not keep_l_ordered:
        raise NotImplementedError(
            "keep_l_ordered=False is not supported in the shared module yet because "
            "the original utility depended on an undefined canonicalization helper."
        )

    n_block_vals = (1,) * row_count if homogeneous_n else tuple(range(1, row_count + 1))
    l_block_vals = _choose_l_blocks(
        col_count,
        target_l_av,
        strict_max_li,
        minimum_l,
    )
    col_sizes = tuple(sum(mat[i][j] for i in range(row_count)) for j in range(col_count))

    l_in = []
    col_slots = []
    pos = 0
    for j in range(col_count):
        size = col_sizes[j]
        l_in.extend([l_block_vals[j]] * size)
        col_slots.append(list(range(pos, pos + size)))
        pos += size

    n_in = [None] * len(l_in)
    for j in range(col_count):
        slot_ptr = 0
        for i in range(row_count):
            count = mat[i][j]
            if count <= 0:
                continue
            slots = col_slots[j][slot_ptr:slot_ptr + count]
            for idx in slots:
                n_in[idx] = n_block_vals[i]
            slot_ptr += count

    raw_n = _relabel_by_first_appearance(tuple(n_in))
    canonical_n, canonical_l = canonicalize_leaf_quantum_numbers(raw_n, tuple(l_in))
    canonical_n = _relabel_by_first_appearance(canonical_n)
    return canonical_n, canonical_l


def _distinct_permutations_from_counts(counts):
    values = tuple(sorted(int(k) for k in counts))
    total = sum(int(v) for v in counts.values())
    out = []

    def rec(prefix, remaining):
        if len(prefix) == total:
            out.append(tuple(prefix))
            return
        for value in values:
            if int(remaining.get(value, 0)) <= 0:
                continue
            remaining[value] -= 1
            prefix.append(value)
            rec(prefix, remaining)
            prefix.pop()
            remaining[value] += 1

    rec([], {int(k): int(v) for k, v in counts.items()})
    return tuple(out)


@lru_cache(maxsize=None)
def _mu_labelings_for_partition(rank, mu_orbits, mu_values):
    rank = int(rank)
    mu_orbits = tuple(int(x) for x in mu_orbits)
    if sum(mu_orbits) != rank:
        return ()
    if mu_values is None:
        base_values = tuple(range(1, len(mu_orbits) + 1))
    else:
        base_values = tuple(int(x) for x in mu_values)
        if len(base_values) < len(mu_orbits):
            return ()

    count_maps = set()
    for chosen in permutations(base_values, len(mu_orbits)):
        counts = Counter()
        for value, count in zip(chosen, mu_orbits):
            counts[int(value)] += int(count)
        count_maps.add(tuple(sorted((int(k), int(v)) for k, v in counts.items() if int(v) > 0)))

    labelings = set()
    for count_map in count_maps:
        counts = {int(k): int(v) for k, v in count_map}
        for labeling in _distinct_permutations_from_counts(counts):
            labelings.add(tuple(int(x) for x in labeling))
    return tuple(sorted(labelings))


def _canonicalize_mu_n_l(mu_in, n_in, l_in, relabel_mu):
    l_counts = Counter(int(l) for l in l_in)
    triples = tuple(
        sorted(
            ((int(mu), int(n), int(l)) for mu, n, l in zip(mu_in, n_in, l_in)),
            key=lambda item: (-int(l_counts[int(item[2])]), int(item[2]), int(item[1]), int(item[0])),
        )
    )
    mu = tuple(int(item[0]) for item in triples)
    if relabel_mu:
        mu = _relabel_by_first_appearance(mu)
    n = tuple(int(item[1]) for item in triples)
    l = tuple(int(item[2]) for item in triples)
    eta = eta_labels(mu, n)
    return mu, n, l, eta


def canonicalize_composite_leaf_quantum_numbers(
    mu_in,
    n_in,
    l_in,
    relabel_mu=False,
):
    """Normalize an exact chemical/radial/angular leaf request.

    This is the request-side counterpart of exhaustive label enumeration.  It
    applies the same canonical slot ordering and returns ``(mu, n, l, eta)``
    without enumerating neighboring labels or inventing coupling paths.
    Subsequent multiplicity and coupling construction must still use YE3T's
    exact count/plan/compile APIs.
    """
    mu_in = tuple(int(value) for value in mu_in)
    n_in = tuple(int(value) for value in n_in)
    l_in = tuple(int(value) for value in l_in)
    if not (len(mu_in) == len(n_in) == len(l_in)):
        raise ValueError("mu_in, n_in, and l_in must have the same length.")
    if not mu_in:
        raise ValueError("composite leaf requests must not be empty.")
    if any(value <= 0 for value in mu_in):
        raise ValueError("all chemical channel indices must be positive.")
    canonicalize_leaf_quantum_numbers(n_in, l_in)
    return _canonicalize_mu_n_l(
        mu_in,
        n_in,
        l_in,
        bool(relabel_mu),
    )


def _enumerate_rank_labels_plain(
    rank,
    target_l_avs,
    strict_max_li,
    homogeneous_n,
    spec_key,
    max_labels,
    keep_l_ordered,
    minimum_l,
):
    orbit_triples = _enumerate_feasible_orbit_triples(
        rank,
        spec_key,
        strict_max_li,
        homogeneous_n,
        minimum_l,
    )
    target_l_avs = tuple(target_l_avs)
    labels_by_key = {}

    for n_orbits, l_orbits, p_orbits in orbit_triples:
        mats = _enumerate_pair_matrices(rank, n_orbits, l_orbits, p_orbits)
        if not mats:
            continue

        av_groups = {}
        num_l_blocks = len(l_orbits)
        for target_l_av in target_l_avs:
            l_block_vals = _choose_l_blocks(
                num_l_blocks,
                target_l_av,
                strict_max_li,
                minimum_l,
            )
            av_groups.setdefault(l_block_vals, []).append(target_l_av)

        for avs_for_same_l in av_groups.values():
            representative_target = avs_for_same_l[0]
            for mat in mats:
                try:
                    n_in, l_in = _matrix_to_labels_cached(
                        mat=mat,
                        target_l_av=representative_target,
                        strict_max_li=strict_max_li,
                        homogeneous_n=homogeneous_n,
                        keep_l_ordered=keep_l_ordered,
                        minimum_l=minimum_l,
                    )
                except ValueError:
                    continue

                key = (n_in, l_in)
                realized_average_l = sum(l_in) / len(l_in)
                min_err = min(abs(realized_average_l - av) for av in avs_for_same_l)

                if key not in labels_by_key:
                    labels_by_key[key] = {
                        "rank": rank,
                        "n_in": n_in,
                        "l_in": l_in,
                        "target_l_avs": set(),
                        "orbit_triples": set(),
                        "matrices": set(),
                        "realized_average_l": realized_average_l,
                        "best_average_error_l": min_err,
                    }

                record = labels_by_key[key]
                record["target_l_avs"].update(avs_for_same_l)
                record["orbit_triples"].add((n_orbits, l_orbits, p_orbits))
                record["matrices"].add(mat)
                if min_err < record["best_average_error_l"]:
                    record["best_average_error_l"] = min_err

                if max_labels is not None and len(labels_by_key) >= max_labels:
                    break
            if max_labels is not None and len(labels_by_key) >= max_labels:
                break
        if max_labels is not None and len(labels_by_key) >= max_labels:
            break

    rank_labels = []
    for record in labels_by_key.values():
        orbit_triples_sorted = tuple(sorted(record["orbit_triples"]))
        matrices_sorted = tuple(sorted(record["matrices"]))
        rep_n_orbits, rep_l_orbits, rep_pair_orbits = orbit_triples_sorted[0]

        rank_labels.append(
            {
                "rank": record["rank"],
                "target_l_avs": tuple(sorted(record["target_l_avs"])),
                "realized_average_l": record["realized_average_l"],
                "best_average_error_l": record["best_average_error_l"],
                "n_orbits": rep_n_orbits,
                "l_orbits": rep_l_orbits,
                "pair_orbits": rep_pair_orbits,
                "orbit_triples": orbit_triples_sorted,
                "matrices": matrices_sorted,
                "n_in": record["n_in"],
                "l_in": record["l_in"],
            }
        )

    rank_labels.sort(key=lambda rec: (rec["n_in"], rec["l_in"], rec["target_l_avs"]))
    return rank_labels


def _enumerate_rank_labels_with_mu(
    rank,
    target_l_avs,
    strict_max_li,
    homogeneous_n,
    spec_key,
    max_labels,
    keep_l_ordered,
    minimum_l,
):
    _, n_parts, l_parts, p_parts, mu_parts, mu_values, nl_pair_parts = spec_key
    base_key = ("restricted", n_parts, l_parts, nl_pair_parts)
    final_pair_parts = integer_partitions(rank) if p_parts is None else tuple(tuple(p) for p in p_parts)
    relabel_mu = mu_values is None
    base_limit = None if max_labels is None else max(1, int(max_labels))
    while True:
        base_labels = _enumerate_rank_labels_plain(
            rank,
            target_l_avs,
            strict_max_li,
            homogeneous_n,
            base_key,
            base_limit,
            keep_l_ordered,
            minimum_l,
        )
        labels_by_key = {}
        for base in base_labels:
            for mu_orbits in mu_parts:
                for raw_mu in _mu_labelings_for_partition(rank, tuple(mu_orbits), mu_values):
                    mu_in, n_in, l_in, eta_in = _canonicalize_mu_n_l(raw_mu, base["n_in"], base["l_in"], relabel_mu)
                    pair_orbits = partition_from_values(tuple(zip(eta_in, l_in)))
                    if tuple(pair_orbits) not in final_pair_parts:
                        continue
                    key = (mu_in, n_in, l_in)
                    if key in labels_by_key:
                        labels_by_key[key]["target_l_avs"].update(base["target_l_avs"])
                        labels_by_key[key]["orbit_triples"].update(base["orbit_triples"])
                        labels_by_key[key]["matrices"].update(base["matrices"])
                        continue
                    labels_by_key[key] = {
                        "rank": int(rank),
                        "target_l_avs": set(base["target_l_avs"]),
                        "realized_average_l": base["realized_average_l"],
                        "best_average_error_l": base["best_average_error_l"],
                        "n_orbits": partition_from_values(n_in),
                        "l_orbits": partition_from_values(l_in),
                        "mu_orbits": partition_from_values(mu_in),
                        "eta_orbits": partition_from_values(eta_in),
                        "pair_orbits": pair_orbits,
                        "nl_pair_orbits": pair_partition(n_in, l_in),
                        "orbit_triples": set(base["orbit_triples"]),
                        "matrices": set(base["matrices"]),
                        "mu_in": mu_in,
                        "n_in": n_in,
                        "eta_in": eta_in,
                        "l_in": l_in,
                    }
                    if max_labels is not None and len(labels_by_key) >= max_labels:
                        break
                if max_labels is not None and len(labels_by_key) >= max_labels:
                    break
            if max_labels is not None and len(labels_by_key) >= max_labels:
                break
        if max_labels is None or len(labels_by_key) >= int(max_labels):
            break
        if len(base_labels) < int(base_limit):
            break
        base_limit *= 2

    rank_labels = []
    for record in labels_by_key.values():
        row = dict(record)
        row["target_l_avs"] = tuple(sorted(row["target_l_avs"]))
        row["orbit_triples"] = tuple(sorted(row["orbit_triples"]))
        row["matrices"] = tuple(sorted(row["matrices"]))
        rank_labels.append(row)
    rank_labels.sort(key=lambda rec: (rec["mu_in"], rec["n_in"], rec["l_in"], rec["target_l_avs"]))
    return rank_labels


@lru_cache(maxsize=None)
def _expand_rank_spec_cached(
    rank,
    spec_key,
):
    if spec_key is None:
        spec_key = ("exhaustive", None, None, None)

    mode = spec_key[0]
    if mode == "exhaustive":
        return integer_partitions(rank), integer_partitions(rank), integer_partitions(rank)
    if mode == "restricted_mu":
        _, n_parts, l_parts, _p_parts, _mu_parts, _mu_values, nl_pair_parts = spec_key
        n_parts = integer_partitions(rank) if n_parts is None else tuple(tuple(p) for p in n_parts)
        l_parts = integer_partitions(rank) if l_parts is None else tuple(tuple(p) for p in l_parts)
        p_parts = integer_partitions(rank) if nl_pair_parts is None else tuple(tuple(p) for p in nl_pair_parts)
        return n_parts, l_parts, p_parts
    if mode != "restricted":
        raise ValueError(f"Unknown mode: {mode}")

    _, n_parts, l_parts, p_parts = spec_key
    n_parts = integer_partitions(rank) if n_parts is None else tuple(tuple(p) for p in n_parts)
    l_parts = integer_partitions(rank) if l_parts is None else tuple(tuple(p) for p in l_parts)
    p_parts = integer_partitions(rank) if p_parts is None else tuple(tuple(p) for p in p_parts)
    return n_parts, l_parts, p_parts


def make_spec_key(rank, spec):
    """Normalize a rank-restricted orbit spec into a cacheable key."""
    if spec is None:
        return ("exhaustive", None, None, None)

    mode = spec.get("mode", "exhaustive")
    if mode == "exhaustive":
        return ("exhaustive", None, None, None)
    if mode != "restricted":
        raise ValueError(f"Unknown mode: {mode}")

    n_parts = _normalize_partition_list(rank, spec.get("n_orbits"))
    l_parts = _normalize_partition_list(rank, spec.get("l_orbits"))
    p_parts = _normalize_partition_list(rank, spec.get("pair_orbits"))
    mu_parts = _normalize_partition_list(rank, spec.get("mu_orbits"))
    mu_values = _normalize_mu_values(spec.get("mu_values"))
    nl_pair_parts = _normalize_partition_list(rank, spec.get("nl_pair_orbits"))
    if mu_parts is not None:
        return ("restricted_mu", n_parts, l_parts, p_parts, mu_parts, mu_values, nl_pair_parts)
    return ("restricted", n_parts, l_parts, p_parts)


@lru_cache(maxsize=None)
def _enumerate_feasible_orbit_triples(
    rank,
    spec_key,
    strict_max_li,
    homogeneous_n,
    minimum_l,
):
    n_parts, l_parts, p_parts = _expand_rank_spec_cached(rank, spec_key)

    triples = []
    for n_orbits in n_parts:
        if homogeneous_n and len(n_orbits) != 1:
            continue

        for l_orbits in l_parts:
            if (
                strict_max_li is not None
                and len(l_orbits) > strict_max_li - int(minimum_l) + 1
            ):
                continue
            for p_orbits in p_parts:
                if _is_orbit_compatible(rank, n_orbits, l_orbits, p_orbits):
                    triples.append((n_orbits, l_orbits, p_orbits))

    return tuple(triples)


def enumerate_rank_labels(
    rank,
    target_l_avs,
    strict_max_li = None,
    homogeneous_n = False,
    spec = None,
    max_labels = None,
    keep_l_ordered = True,
    minimum_l = 1,
):
    """Enumerate canonical label records for one rank."""
    spec_key = make_spec_key(rank, spec)
    if spec_key[0] == "restricted_mu":
        return _enumerate_rank_labels_with_mu(
            rank,
            target_l_avs,
            strict_max_li,
            homogeneous_n,
            spec_key,
            max_labels,
            keep_l_ordered,
            minimum_l,
        )
    return _enumerate_rank_labels_plain(
        rank,
        target_l_avs,
        strict_max_li,
        homogeneous_n,
        spec_key,
        max_labels,
        keep_l_ordered,
        minimum_l,
    )


def build_label_bank(
    ranks,
    l_av_max_per_rank,
    l_av_inc,
    strict_max_li_per_rank = None,
    orbit_spec_per_rank = None,
    homogeneous_n = False,
    max_labels_per_rank = None,
    keep_l_ordered = True,
    minimum_l_per_rank = None,
):
    """Build a grouped exhaustive label bank over multiple ranks."""
    strict_max_li_per_rank = strict_max_li_per_rank or {}
    orbit_spec_per_rank = orbit_spec_per_rank or {}
    minimum_l_per_rank = minimum_l_per_rank or {}

    all_labels = {}
    for rank in ranks:
        target_l_avs = make_target_l_avs(l_av_max_per_rank[rank], l_av_inc)
        strict_max_li = strict_max_li_per_rank.get(rank)
        spec = orbit_spec_per_rank.get(rank, {"mode": "exhaustive"})
        all_labels[rank] = enumerate_rank_labels(
            rank=rank,
            target_l_avs=target_l_avs,
            strict_max_li=strict_max_li,
            homogeneous_n=homogeneous_n,
            spec=spec,
            max_labels=max_labels_per_rank,
            keep_l_ordered=keep_l_ordered,
            minimum_l=minimum_l_per_rank.get(rank, 1),
        )
    return all_labels


def _print_rank_summary(all_labels, max_rows = 10):
    for rank, rank_labels in all_labels.items():
        print(f"\nrank = {rank}, count = {len(rank_labels)}")
        for rec in rank_labels[:max_rows]:
            print(
                f"target_l_avs={rec['target_l_avs']}, "
                f"realized_average_l={rec['realized_average_l']:.6g}, "
                f"best_average_error_l={rec['best_average_error_l']:.6g}, "
                f"n_orbits={rec['n_orbits']}, "
                f"l_orbits={rec['l_orbits']}, "
                f"pair_orbits={rec['pair_orbits']}, "
                f"n_in={rec['n_in']}, "
                f"l_in={rec['l_in']}"
            )
