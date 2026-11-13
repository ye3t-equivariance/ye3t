"""Exact one-occurrence lowering in shared normalized angular coordinates.

Retained factors use the existing species/radial/Racah function algebra;
pooled factors remain independent moment generators. All angular assembly
reuses normalized CG paths. No global magnetic tensor is constructed.

References: tensor_product_math.md 10.2, equations33--34; NIST DLMF34.3.20
(harmonic product), with the existing Racah normalization. The reduced
same-direction factor is the product of the existing zero-magnetic CGs.
"""
from itertools import product
import json

import sympy as sp

from ye3t.core.cg import cg_exact_integer
from ye3t.exact_scalars import exact_scalar
from ye3t.representations.young_angular_paths import (
    _permuted_chain_column, _inner, _association_coefficient)
from .density_product_images import (
    _normalized_sector_path_columns, _add, _channel_identity, _formal_shell_coordinate)
from .tagged_cauchy_general import _radial_product
from .chebyshev_c2 import CHEBYSHEV_C2_SOURCE_FAMILY, chebyshev_c2_radial_product
from .orthogonal_shifted_jacobi import ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY


def _retained_source_multiplets(source, lowering):
    """Reuse exact source role maps with explicit physical factor bindings."""
    from .retained_cauchy_paths import _occupation_families
    from .cauchy_paths import _role_path_analysis
    from .lifted_cauchy_scalar import _freeze_json

    groups = {}
    for position, coordinate in enumerate(source["coordinates"]):
        groups.setdefault(json.dumps(coordinate["coupling_request"], sort_keys=True), []).append((position, coordinate))
    result = [None]*len(source["coordinates"])
    for serialized, selected in groups.items():
        coupling = json.loads(serialized)
        tags = lowering.tag_count
        occupation = [1]*tags + [sum(coupling["block_sizes"])-tags]
        if (coupling["role_dimension"] != tags+1 or occupation[-1] < 0
                or coupling.get("role_occupations") != [occupation]):
            raise ValueError("Retained sources require one factor per tag and the remaining factors in the pooled role")
        base, families = _occupation_families(coupling)
        metadata = []
        for _, coordinate in selected:
            label = coordinate["label"]
            family = families.get(label["role_family_id"])
            if (family is None or _freeze_json(coordinate["channels"]) != _freeze_json(coupling["channels"])
                    or any(_freeze_json(label[key]) != _freeze_json(value) for key, value in family["family"].items())):
                raise ValueError("Retained source labels or channels differ from their canonical coupling request")
            metadata.append({"coordinate_id": coordinate["coordinate_id"], "family_id": label["role_family_id"],
                             "partition": label["tag_partition"], "role_copy": label["tag_role_copy"]})
        roles = coupling["role_dimension"]
        if roles != lowering.tag_count+1:
            raise ValueError("Retained source role dimension differs from its physical occurrence binding")
        analyses = _role_path_analysis(base, families, metadata, roles, lowering.tag_count)
        channels = [{**channel, "physical_role": role} for channel in coupling["channels"] for role in range(roles)]
        needed = {index for analysis in analyses.values() for row in analysis for index in row}
        coordinates = {index: _formal_shell_coordinate({"label": base[index], "channels": channels,
            "coupling_request": coupling}, lowering.channel_ids) for index in needed}
        for position, coordinate in selected:
            multiplet = []
            for row in analyses[coordinate["coordinate_id"]]:
                values = {}
                for index, coefficient in row.items():
                    _add(values, coordinates[index], coefficient)
                multiplet.append({key: value for key, item in values.items() if (value := sp.simplify(item)) != 0})
            result[position] = multiplet
    return result


def _same_occurrence_angular_image(spins, occurrences, column):
    """Reduce CG paths to one Racah harmonic per physical occurrence.

    Keys are (remaining angular degrees, normalized chain). Occurrences are
    nonnegative integer identities; output factors follow sorted identity order.
    Angular recoupling and zero-M CGs use the established normalized maps.
    Radial multipliers and overall real-form phases are supplied by callers.

    DLMF 34.3.20 gives the harmonic product; DLMF 34.4 gives the association
    used to bring each adjacent pair together. Traverse only sparse input
    paths and their nonzero recouplings, with no full target enumeration.
    """
    spins, occurrences = tuple(spins), tuple(occurrences)
    if (not spins or len(spins) != len(occurrences)
            or any(type(value) is not int or value < 0 for value in spins+occurrences)):
        raise ValueError("Angular occurrence image needs matching nonnegative integer degrees and identities")
    column = dict(column)
    for path in column:
        if (len(path) != len(spins) or path[0] != spins[0]
                or any(not abs(a-b) <= c <= a+b for a, b, c in
                       zip(path[:-1], spins[1:], path[1:], strict=True))):
            raise ValueError("Angular occurrence image requires valid normalized CG paths")
    if len({path[-1] for path in column}) > 1:
        raise ValueError("An angular occurrence column must have one output L")
    roles = list(occurrences)
    rows = {(spins, path): value for path, value in column.items() if value}

    def reorder(values, order):
        if order == tuple(range(len(order))):
            return values
        groups, result = {}, {}
        for (degrees, path), coefficient in values.items():
            groups.setdefault(degrees, {})[path] = coefficient
        for degrees, sparse in groups.items():
            permuted_degrees = tuple(degrees[i] for i in order)
            for path, coefficient in _permuted_chain_column(sparse, degrees, order).items():
                result[permuted_degrees, path] = coefficient
        return result

    # Apply the same exact maps with each harmonic reduction immediately after
    # its pair is aligned. Subsequent permutations then act on fewer factors;
    # no unused full grouped chain needs to be constructed.
    while len(set(roles)) < len(roles):
        role = next(role for role in sorted(set(roles)) if roles.count(role) > 1)
        position = roles.index(role)
        other = roles.index(role, position+1)
        order = list(range(len(roles)))
        order.insert(position+1, order.pop(other))
        rows = reorder(rows, tuple(order))
        roles = [roles[i] for i in order]
        updated = {}
        for (degrees, path), value in rows.items():
            left, right = degrees[position:position+2]
            outputs = ((path[1],) if position == 0 else range(
                max(abs(left-right), abs(path[position-1]-path[position+1])),
                min(left+right, path[position-1]+path[position+1])+1))
            for L in outputs:
                coefficient = cg_exact_integer(left, 0, right, 0, L, 0)
                if not coefficient:
                    continue
                if position:
                    coefficient *= _association_coefficient(path[position-1], left, right,
                        path[position], L, path[position+1])
                if not coefficient:
                    continue
                key = (degrees[:position]+(L,)+degrees[position+2:],
                       path[:position]+path[position+1:])
                total = updated.get(key, 0)+value*coefficient
                if total:
                    updated[key] = total
                else:
                    updated.pop(key, None)
        rows = updated
        del roles[position+1]
    return reorder(rows, tuple(sorted(range(len(roles)), key=lambda index: roles[index])))


class _RetainedOccurrenceImage:
    """Per-planning-call physical coordinate table and reduced lowering cache.

    Channel IDs identify actual function/role bindings. Newly generated radial
    degrees use the existing exact product algebra. They are image coordinates,
    not additions to a user's source catalogue or learned width.
    """

    def __init__(self, channels, tag_count):
        self.channels = list(channels)
        self.channel_ids = {_channel_identity(channel): i for i, channel in enumerate(channels, 1)}
        self.tag_count = tag_count
        self.lowered = {}
        self.radial_lowered = {}
        for channel in channels:
            if not 0 <= channel["physical_role"] <= tag_count:
                raise ValueError("Retained product role lies outside its physical binding")
            if channel["source_family_id"] not in {
                    CHEBYSHEV_C2_SOURCE_FAMILY, ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY}:
                raise ValueError("Retained products require an established exact one-occurrence radial algebra")

    def _channel(self, channel):
        identity = _channel_identity(channel)
        if identity not in self.channel_ids:
            self.channels.append(channel)
            self.channel_ids[identity] = len(self.channels)
        return self.channel_ids[identity]

    def _physical_radial_group(self, word, L):
        if (word, L) in self.radial_lowered:
            return self.radial_lowered[word, L]
        channels = [self.channels[token-1] for token, _ in word]
        if len({channel["neighbor_species"] for channel in channels}) != 1:
            return {}
        families = {channel["source_family_id"] for channel in channels}
        if len(families) != 1:
            raise ValueError("Retained products cannot mix radial families on one occurrence")
        radial = tuple((channel["radial_channel"], ell) for channel, (_, ell) in zip(channels, word, strict=True))
        expansion = (chebyshev_c2_radial_product(radial, L)
                     if families == {CHEBYSHEV_C2_SOURCE_FAMILY} else _radial_product(radial, L))
        result = {}
        for degree, value in expansion:
            channel = {**channels[0], "radial_channel": degree, "l": L}
            token = self._channel(channel)
            result[token, L] = exact_scalar(value)
        self.radial_lowered[word, L] = result
        return result

    def coordinate(self, word, L, copy):
        key = (word, L, copy)
        if key in self.lowered:
            return self.lowered[key]
        grouped, identities = {}, []
        for index, (token, ell) in enumerate(word):
            role = self.channels[token-1]["physical_role"]
            # A pooled moment is not a value on one retained direction. Its
            # factors stay separate until the fixed-content symmetric map.
            identity = self.tag_count+index if role == self.tag_count else role
            identities.append(identity)
            grouped.setdefault(identity, []).append(index)
        roles = tuple(sorted(grouped))
        words = tuple(tuple(word[index] for index in grouped[role]) for role in roles)
        original = _normalized_sector_path_columns(word, L)[copy]
        if _inner(dict(original), dict(original)) != exact_scalar(1):
            raise ArithmeticError("Physical occurrence input must use a normalized angular coordinate")
        image = _same_occurrence_angular_image(tuple(ell for _, ell in word), identities, original)
        result = {}
        for (spins, path), angular_weight in image.items():
            values = [({group[0]: exact_scalar(1)} if role >= self.tag_count else
                       self._physical_radial_group(group, spin))
                      for role, group, spin in zip(roles, words, spins, strict=True)]
            for selected in product(*(tuple(value.items()) for value in values)):
                factors = tuple(item[0] for item in selected)
                weight = angular_weight
                for _, radial_weight in selected:
                    weight *= radial_weight
                order = tuple(sorted(range(len(factors)), key=lambda i: factors[i]))
                target_word = tuple(factors[i] for i in order)
                column = _permuted_chain_column({path: weight}, spins, order)
                for target_copy, target in enumerate(_normalized_sector_path_columns(target_word, L)):
                    coefficient = _inner(dict(target), column)
                    if coefficient:
                        target_key = (target_word, L, target_copy)
                        result[target_key] = result.get(target_key, 0)+coefficient._sympy_()
        result = {key: value for key, item in result.items() if (value := sp.simplify(item)) != 0}
        self.lowered[key] = result
        return result

    def multiplet(self, multiplet):
        result = []
        for row in multiplet:
            values = {}
            for (word, L, copy), coefficient in row.items():
                _add(values, self.coordinate(word, L, copy), coefficient)
            result.append({key: value for key, item in values.items() if (value := sp.simplify(item)) != 0})
        return result
