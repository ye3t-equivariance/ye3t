"""Physical content inputs for the shared catalogue requests.

These functions resolve supplied factor identities. Coupling labels, admissible
intermediates and independence remain the responsibility of count/plan/compile.
"""

from collections import Counter
from collections.abc import Mapping
from itertools import product
import json
from math import factorial
from numbers import Integral


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must contain nonnegative integer indices.")
    return int(value)


def _multiset_permutations(values):
    counts = Counter(values)
    keys = sorted(counts)
    row = []

    def visit():
        if len(row) == len(values):
            yield tuple(row)
            return
        for key in keys:
            if counts[key]:
                counts[key] -= 1
                row.append(key)
                yield from visit()
                row.pop()
                counts[key] += 1

    yield from visit()


def normalize_explicit_labels(labels, *, species, source_family_id,
                              maximum_variations=4096):
    """Purpose: Resolve aligned inputs into canonical complete-channel contents.

    Mathematical condition: Reordering preserves each complete factor and root.
    Inputs: Factor-aligned records, ordered species and a physical source ID.
    Outputs: Deduplicated content records for existing catalogue requests.
    Does not: Construct coupling paths or establish physical independence.

    Each entry supplies center, chemical, radial and angular. Chemical names
    or zero-based indices refer to the configured species order. ``variations``
    optionally enumerates pairings of the three multisets, keeping the root
    fixed. Optional tag_counts and intermediates are carried to the compiler;
    this function does not infer an independent feature from a content.
    """
    species = tuple(species)
    if (not species or len(set(species)) != len(species) or
            any(not isinstance(name, str) or not name for name in species)):
        raise ValueError("Explicit labels require unique nonempty species names.")
    if not isinstance(labels, (list, tuple)) or not labels:
        raise ValueError("explicit_labels must be a nonempty list of factor-aligned records.")
    if (isinstance(maximum_variations, bool) or not isinstance(maximum_variations, Integral)
            or maximum_variations < 1):
        raise ValueError("maximum_variations must be a positive integer.")

    def chemical_name(value):
        if isinstance(value, Integral) and not isinstance(value, bool) and 0 <= value < len(species):
            return species[int(value)]
        if isinstance(value, str) and value in species:
            return value
        raise ValueError("Explicit chemical indices must name a configured species or its zero-based index.")

    result, seen = [], set()
    for record in labels:
        required = {"center", "chemical", "radial", "angular"}
        allowed = required | {"tag_counts", "intermediates", "variations"}
        if not isinstance(record, Mapping) or not required <= set(record) or set(record) - allowed:
            raise ValueError("An explicit label needs center, chemical, radial and angular; optional fields are tag_counts, intermediates and variations.")
        center = chemical_name(record["center"])
        if any(not isinstance(record[key], (list, tuple)) for key in ("chemical", "radial", "angular")):
            raise ValueError("chemical, radial and angular must be aligned sequences.")
        chemical = tuple(chemical_name(value) for value in record["chemical"])
        radial = tuple(_integer(value, "radial") for value in record["radial"])
        angular = tuple(_integer(value, "angular") for value in record["angular"])
        rank = len(chemical)
        if not rank or len(radial) != rank or len(angular) != rank:
            raise ValueError("chemical, radial and angular must have the same positive factor rank; center is separate.")
        tags = record.get("tag_counts")
        if tags is not None:
            if not isinstance(tags, (list, tuple)) or not tags:
                raise ValueError("tag_counts must be a nonempty sequence.")
            tags = tuple(_integer(value, "tag_counts") for value in tags)
            if len(set(tags)) != len(tags) or max(tags) > rank:
                raise ValueError("Explicit tag counts must be distinct and at most the factor rank.")
            tags = tuple(sorted(tags))
        intermediates = record.get("intermediates")
        if intermediates is not None:
            # Use precisely the public representation parser; no second policy
            # grammar or coupling-path enumerator is introduced here.
            from ye3t.representation_config import YE3TRepresentation
            parsed = YE3TRepresentation.from_config({
                "group": "O3", "ranks": [rank],
                "parent": {"young_lambda": "(N)", "L": 0, "parity": "even"},
                "factorization": "cauchy", "subspace": "full",
                "uncoupled_factor_inputs": {"eta_max_per_rank": {rank: rank},
                                             "l_max_per_rank": {rank: max(angular)}},
                "intermediates": intermediates,
            })
            intermediates = parsed.to_dict()["intermediates"]
        variations = record.get("variations", False)
        if not isinstance(variations, bool):
            raise ValueError("variations must be boolean.")
        if variations:
            estimate = 1
            for values in (radial, angular):
                possibilities = factorial(rank)
                for count in Counter(values).values():
                    possibilities //= factorial(count)
                estimate *= possibilities
            if estimate > maximum_variations:
                raise MemoryError(f"Explicit multiset pairings need up to {estimate} assignments, exceeding maximum_variations={maximum_variations}; supply aligned labels or raise the explicit limit.")
            chemical = tuple(sorted(chemical))
            assignments = product(_multiset_permutations(radial), _multiset_permutations(angular))
        else:
            assignments = ((radial, angular),)
        for radial_word, angular_word in assignments:
            counts = Counter(zip(chemical, radial_word, angular_word, strict=True))
            blocks = sorted(counts.items(), key=lambda item: (-item[1], item[0][2], item[0][0], item[0][1]))
            channels = tuple({"neighbor_species": name, "radial_channel": n, "l": l,
                              "source_family_id": source_family_id}
                             for (name, n, l), _size in blocks)
            sizes = tuple(size for _channel, size in blocks)
            key = (center, tuple((channel, size) for channel, size in blocks), tags,
                   json.dumps(intermediates, sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            result.append({"center": center, "rank": rank, "channels": channels,
                           "block_sizes": sizes,
                           **({"tag_counts": tags} if tags is not None else {}),
                           **({"intermediates": intermediates} if intermediates is not None else {})})
    return tuple(sorted(result, key=lambda row: (
        row["center"], row["rank"],
        tuple((-size, channel["l"], channel["neighbor_species"], channel["radial_channel"])
              for size, channel in zip(row["block_sizes"], row["channels"], strict=True)),
        row.get("tag_counts", ()), json.dumps(row.get("intermediates"), sort_keys=True))))
