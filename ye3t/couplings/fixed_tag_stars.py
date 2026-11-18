"""Analytical planning primitives for depth-one fixed-tag sources.

Retained-tag counts and final pooled counts describe different linear maps.
The functions here state their selection assumptions explicitly. They do not
construct coupling coefficients or infer independence by matrix reduction.
Normalized couplings continue to come from the shared fixed-content compiler.
"""

from collections import Counter
from itertools import product

from ye3t.core.basis.exhaustive_enumeration import integer_partitions
from ye3t.fixed_content import FixedContentModule, FixedContentSpec


def _profile_a_split_inventory(content, input_Ls, tag_count, *, tag_contents=None):
    """Independent retained copies for singleton tags and a distinct background.

    For each complete content split, couple alpha_T^(tau,J) to
    alpha_B^((p),K). Every angular output in |J-K|,...,J+K occurs once.
    The physical independence assertion uses independent primitive functions
    and a domain of freely varying distinct occurrences. Final tag pooling is
    an additional operation; these labels are retained coordinates.
    """
    content, input_Ls = tuple(content), tuple(input_Ls)
    if len(content) != len(input_Ls):
        raise ValueError("Content and angular inputs must have the same length.")
    if not isinstance(tag_count, int) or not 0 <= tag_count <= len(content):
        raise ValueError("tag_count must be an integer between zero and the factor rank.")
    if any(not isinstance(l, int) or l < 0 for l in input_Ls):
        raise ValueError("Angular inputs must be nonnegative integers.")
    # Complete channels include angular type; eta equality alone is insufficient.
    channels = tuple(dict.fromkeys(zip(content, input_Ls)))
    full = Counter(zip(content, input_Ls))
    sizes = tuple(full[channel] for channel in channels)
    if tag_contents is None:
        splits = tuple(values for values in product(*(range(size+1) for size in sizes))
                       if sum(values) == tag_count)
    else:
        splits = tuple(tuple(values) for values in tag_contents)
        if len(set(splits)) != len(splits) or any(
                len(values) != len(sizes) or sum(values) != tag_count or any(
                    not isinstance(value, int) or not 0 <= value <= size
                    for value, size in zip(values, sizes)) for values in splits):
            raise ValueError("Tag content splits must be distinct bounded multiplicity tuples.")
    counts = {}

    def sectors(values):
        if not any(values):
            return (((), 0, 1),)
        if values not in counts:
            eta, angular = [], []
            for index, ((_, l), size) in enumerate(zip(channels, values), 1):
                eta.extend([index]*size)
                angular.extend([l]*size)
            decomposition = FixedContentModule(FixedContentSpec(eta, angular)).decompose()
            counts[values] = tuple((sector.partition, sector.L_R, sector.multiplicity)
                                   for sector in decomposition.valid_young_rotation_sectors)
        return counts[values]

    inventory = []
    background_partition = (len(content)-tag_count,) if tag_count < len(content) else ()
    for tags in splits:
        background = tuple(size-value for size, value in zip(sizes, tags))
        for tau, J, tag_copies in sectors(tags):
            for beta, K, background_copies in sectors(background):
                if beta != background_partition:
                    continue
                for L in range(abs(J-K), J+K+1):
                    for u in range(tag_copies):
                        for v in range(background_copies):
                            inventory.append({"tag_content": tags, "background_content": background,
                                "tag_partition": tau, "tag_L": J, "background_L": K,
                                "target_L": L, "tag_copy": u, "background_copy": v})
    return tuple(inventory)


def _local_product_angular_support(input_Ls):
    """One copy of each nonzero Gaunt-product output for fixed local content.

    Triangle and even l_left+l_right+L are the exact zero-magnetic-index
    selection rules (DLMF 34.3.5, 34.3.20). Radial/chemical products must also
    be nonzero; combining different local contents needs their actual image.
    """
    support = {0}
    for l in input_Ls:
        if not isinstance(l, int) or l < 0:
            raise ValueError("Angular inputs must be nonnegative integers.")
        support = {L for J in support for L in range(abs(J-l), J+l+1)
                   if not (J+l+L) % 2}
    return tuple(sorted(support))


def _scalar_power_pooled_inventory(rank, tag_counts):
    """Exact pooled B inventory for one nonconstant scalar primitive f.

    The domain allows arbitrary finite sets of independently varying f values.
    For one tag count s, all positive occupancies and inclusive background
    degrees are admitted. Leading power-sum monomials are distinct, proving
    independence. For all s=0,...,k, Mobius inversion supplies the power-sum
    monomials indexed by partitions with at most k parts greater than one.
    These are finite source spans; no extra products are added to the source.
    """
    tags = tuple(sorted(set(tag_counts)))
    if not isinstance(rank, int) or rank < 1 or not tags or any(
            not isinstance(s, int) or not 0 <= s <= rank for s in tags):
        raise ValueError("Positive rank and tag counts between zero and rank are required.")
    if len(tags) == 1:
        s = tags[0]
        if s == 0:
            return ({"tag_powers": (), "background_degree": rank},)
        return tuple({"tag_powers": tuple(parts), "background_degree": rank-degree}
                     for degree in range(s, rank+1) for parts in integer_partitions(degree)
                     if len(parts) == s)
    if tags == tuple(range(tags[-1]+1)):
        return tuple({"moment_powers": tuple(parts)} for parts in integer_partitions(rank)
                     if sum(part > 1 for part in parts) <= tags[-1])
    raise NotImplementedError(
        "Exact pooled scalar-power planning supports one tag count or the complete "
        "prefix 0,...,k. An arbitrary union needs its own overlap rule.")


def _repeated_pair_pooled_inventory(rank1_nmax, pair_nmax, pair_l, tag_counts):
    """Analytical Jacobi rank-one/repeated-rank-two fixture, for one species.

    Rank-one radial indices are 0,...,rank1_nmax-1 at l=0. Rank two contains
    two identical (n,l) factors for every n below pair_nmax. All occupation
    coordinates of each chosen tag count are included. C_n has highest radial
    index 2*n+2*l+2. Distinct n therefore add independent collision directions
    precisely when this index lies beyond the rank-one prefix.
    """
    if any(not isinstance(value, int) or value < 0
           for value in (rank1_nmax, pair_nmax, pair_l)):
        raise ValueError("Radial counts and angular degree must be nonnegative integers.")
    tags = tuple(sorted(set(tag_counts)))
    if not tags or any(not isinstance(s, int) or s not in (0, 1, 2) for s in tags):
        raise ValueError("A repeated pair admits tag counts zero, one and two.")
    independent = [{"kind": "radial_moment", "q": q} for q in range(rank1_nmax)]
    separate_collision = 1 in tags or (0 in tags and 2 in tags)
    for n in range(pair_nmax):
        independent.append({"kind": "distinct_pair" if tags == (2,) else "density_pair",
                            "n": n, "l": pair_l})
        degree = 2*n+2*pair_l+2
        if separate_collision and degree >= rank1_nmax:
            independent.append({"kind": "collision", "n": n, "l": pair_l,
                                "leading_radial_index": degree})
    return tuple(independent)


def _general_analytical_selection(request, labels):
    """Select existing normalized coordinates for proved pooled source scopes.

    Return None for a scope whose selected-image rule is still unavailable.
    This is not an independence claim for an unhandled finite catalogue.
    """
    from .lifted_cauchy_scalar import _role_schur_pivot_metadata

    catalogue = request["catalogue"]
    profile = catalogue.get("source_profile", "B")
    if all(row["tag_count"] == 0 for row in labels):
        selected = list(labels)
        named = {int(rank): tuple(ids) for rank, ids in
                 catalogue.get("selected_basis_coordinates_by_rank", {}).items()}
        if any(not set(ids) <= {row["coordinate_id"] for row in selected if row["tensor_order"] == rank}
               for rank, ids in named.items()):
            raise ValueError("A selected coordinate is absent from the independent symmetric inventory at its requested rank.")
        selected = [row for row in selected if row["tensor_order"] not in named or
                    row["coordinate_id"] in named[row["tensor_order"]]]
        # Preserve the historical capped span: interleave components and use
        # their established angular/coordinate ordering before applying caps.
        ordered = []
        for rank in sorted({row["tensor_order"] for row in selected}):
            groups = {}
            for row in selected:
                if row["tensor_order"] == rank:
                    groups.setdefault(row["component_index"], []).append(row)
            for group in groups.values():
                group.sort(key=lambda row: (sum(row["label"]["block_Lambdas"]), row["coordinate_id"]))
            while groups:
                for component in tuple(groups):
                    ordered.append(groups[component].pop(0))
                    if not groups[component]:
                        del groups[component]

        selected = ordered
        rule = ("distinct_occurrence_pooled_symmetric_v1" if profile == "A"
                else "independent_primitive_moment_polynomials_v1")
    else:
        if catalogue.get("selected_basis_coordinates_by_rank"):
            return None
        if "intermediate_selection" in catalogue or any(
                len(row["label"]["block_sizes"]) != 1 for row in labels):
            return None
        contents, by_rank, seen = {}, {}, {}
        for row in labels:
            label, s, rank = row["label"], row["tag_count"], row["tensor_order"]
            if tuple(label["block_kappas"][0]) != (rank,):
                return None
            key = (s+1, rank)
            if key not in contents:
                contents[key] = _role_schur_pivot_metadata(s+1, rank, (rank,))
            occupation = contents[key][label["role_copy_indices"][0]][1]
            channel = label["block_complete_channel_keys"][0]
            source = (channel["neighbor_species"], channel["radial_channel"], channel["l"])
            seen.setdefault(rank, set()).add(source)
            if all(occupation[a] > 0 for a in range(s)):
                by_rank.setdefault(rank, []).append((row, occupation, source))
        sources = set().union(*seen.values())
        selected = []
        if len(sources) == 1 and next(iter(sources))[2] == 0:
            rule = "single_scalar_power_pooled_v1"
            for rank, entries in sorted(by_rank.items()):
                tags = tuple(sorted({row["tag_count"] for row, _, _ in entries}))
                try:
                    inventory = _scalar_power_pooled_inventory(rank, tags)
                except NotImplementedError:
                    return None
                candidates = {}
                for row, occupation, _ in entries:
                    powers, p = occupation[:-1], occupation[-1]
                    if len(tags) == 1:
                        key = (tuple(sorted(powers, reverse=True)), p)
                    else:
                        if any(r == 1 for r in powers):
                            continue
                        key = tuple(sorted((*powers, *(1,)*p), reverse=True))
                    candidates.setdefault(key, row)
                for coordinate in inventory:
                    key = (coordinate["tag_powers"], coordinate["background_degree"]) if len(tags) == 1 else coordinate["moment_powers"]
                    if key not in candidates:
                        return None
                    selected.append(candidates[key])
        elif set(by_rank) <= {1, 2} and 2 in by_rank and len(request["species"]) == 1:
            rule = "jacobi_repeated_pair_radial_prefix_v1"
            species = request["species"][0]
            rank1 = sorted(seen.get(1, ()))
            k = len(rank1)
            if rank1 != [(species, q, 0) for q in range(k)]:
                return None
            angular = {source[2] for source in seen[2]}
            if len(angular) != 1:
                return None
            l = next(iter(angular))
            n2 = len(seen[2])
            if sorted(seen[2]) != [(species, n, l) for n in range(n2)]:
                return None
            cap1 = catalogue.get("max_features_per_rank", 256)
            cap1 = cap1.get("1", cap1.get(1, 256)) if isinstance(cap1, dict) else cap1
            if int(cap1) < k:
                return None
            first = {}
            for row, occupation, source in by_rank.get(1, ()):
                first.setdefault(source[1], row)
            if len(first) != k:
                return None
            selected.extend(first[q] for q in range(k))
            tags = sorted({row["tag_count"] for row, _, _ in by_rank[2]})
            candidates = {}
            for row, occupation, source in by_rank[2]:
                s = row["tag_count"]
                kind = ("distinct_pair" if s == 2 else "collision" if s == 1 and occupation[0] == 2 else "density_pair")
                candidates.setdefault((source[1], kind), row)
            for coordinate in _repeated_pair_pooled_inventory(k, n2, l, tags):
                if coordinate["kind"] == "radial_moment":
                    continue
                key = (coordinate["n"], coordinate["kind"])
                if key[1] == "collision" and key not in candidates:
                    # B and B-C span the same two-dimensional space as B,C.
                    key = (key[0], "distinct_pair")
                if key not in candidates:
                    return None
                selected.append(candidates[key])
        else:
            return None
    full_counts = Counter(row["tensor_order"] for row in selected)
    caps = catalogue.get("max_features_per_rank", 256)
    if any(row["tag_count"] for row in labels) and any(
            int(caps.get(str(rank), caps.get(rank, 256)) if isinstance(caps, dict) else caps) < size
            for rank, size in full_counts.items()):
        # A different independent basis followed by truncation can change the
        # admitted span. Its selected-image rule needs separate qualification.
        return None
    counts, capped = Counter(), []
    for row in selected:
        rank = row["tensor_order"]
        cap = catalogue.get("max_features_per_rank", 256)
        cap = cap.get(str(rank), cap.get(rank, 256)) if isinstance(cap, dict) else cap
        if int(cap) < 1:
            raise ValueError("max_features_per_rank must be positive.")
        if counts[rank] < int(cap):
            capped.append(row["coordinate_id"])
            counts[rank] += 1
    if any(not set(ids) <= set(capped) for ids in
           catalogue.get("selected_basis_coordinates_by_rank", {}).values()):
        raise ValueError("Named coordinates exceed the rank cap of the independent inventory.")
    return {"rule": rule, "independent_coordinate_ids": tuple(capped),
            "exact_image_dimension": len(capped), "count_by_rank": dict(sorted(counts.items())),
            "source_profile": profile, "readout": "pooled_scalar",
            "coefficient_materialization_performed": False, "row_reduction_performed": False}
