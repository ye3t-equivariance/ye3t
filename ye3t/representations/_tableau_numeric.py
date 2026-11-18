"""Arithmetic for the shared selected-tableau constructor.

Discrete seed preparation uses rational seminormal coordinates. Coefficient
materialization uses Young-orthogonal FP64 actions and the shared projector
and component-transport schedules. Preparation returns indices, never a full
exact coefficient table. Its cost must be measured separately and included in
first-use construction. This private implementation is under qualification.

References: Vershik--Okounkov, math/0503040, Section 6 (seminormal/orthogonal
forms); Doty--Lauve--Seelinger, 1606.08900, Theorem 3.8 and Section 4.
Independent implementation; no external implementation copied.
"""

from fractions import Fraction
from functools import lru_cache

import numpy as np

import ye3t.representations.projectors as p


@lru_cache(maxsize=128)
def _generator_layout(partition):
    """Adjacent actions in O(rank * tableau dimension) integer storage."""
    tableaux = p.standard_tableaux(partition)
    lookup = {tableau: index for index, tableau in enumerate(tableaux)}
    result = []
    for generator in range(sum(partition)-1):
        distances, destinations = [], []
        for index, tableau in enumerate(tableaux):
            distances.append(p._tableau_content(tableau, generator+2)
                             - p._tableau_content(tableau, generator+1))
            swapped = p._swap_tableau_entries(tableau, generator+1, generator+2)
            destinations.append(lookup[swapped] if swapped is not None else index)
        distance = np.array(distances, dtype=np.int64)
        destination = np.array(destinations, dtype=np.int64)
        distance.flags.writeable = destination.flags.writeable = False
        result.append((distance, destination))
    return tuple(result)


@lru_cache(maxsize=128)
def _seminormal_weights(partition):
    """Squared diagonal basis scales from the existing component spanning tree."""
    reference, _, edges, _ = p._tableau_construction_schedule(partition)
    weights = [None]*len(p.standard_tableaux(partition))
    weights[reference] = Fraction(1)
    for source, target, _, distance in edges:
        ratio = Fraction(distance*distance-1, distance*distance)
        weights[target] = weights[source]*ratio if distance > 0 else weights[source]/ratio
    return tuple(weights)


def _rational_adjacent(layout, vector):
    distance, destination = layout
    result = {}
    for source, coefficient in vector.items():
        d = int(distance[source])
        result[source] = result.get(source, 0) + coefficient/Fraction(d)
        if abs(d) > 1:
            target = int(destination[source])
            value = coefficient if d > 0 else coefficient*Fraction(d*d-1, d*d)
            result[target] = result.get(target, 0) + value
    return {index: value for index, value in result.items() if value}


def _orthogonal_actions(partition):
    actions = []
    for distance, destination in _generator_layout(partition):
        diagonal = 1.0/distance
        off_diagonal = np.sqrt(1.0-diagonal*diagonal)
        actions.append((diagonal, destination, off_diagonal))
    return tuple(actions)


def _numeric_adjacent(action, columns):
    diagonal, destination, off_diagonal = action
    result = diagonal[:, None]*columns
    # A standard-tableau adjacent swap is an involution, hence destination
    # has no duplicate indices. No dense generator or scatter reduction.
    result[destination] += off_diagonal[:, None]*columns
    return result


def _tableau_request(kind, left, right, parent):
    """Existing LR/Kronecker counts and the canonical seed priority."""
    if kind == "split":
        from .young_orthogonal import littlewood_richardson_coefficient
        count = (int(parent == (left or right)) if not left or not right else
                 littlewood_richardson_coefficient(left, right, parent))
        dimension = len(p.standard_tableaux(parent))
        priority = p._binary_split_seed_priority(left, parent)
        return (left, right), dimension, count, priority
    if kind == "same_action":
        from .kronecker_intertwiners import kronecker_multiplicity
        count = kronecker_multiplicity(parent, left, right)
        right_dimension = len(p.standard_tableaux(right))
        dimension = len(p.standard_tableaux(left))*right_dimension
        priority = tuple(i*right_dimension+j for i in p._tableau_construction_schedule(left)[3]
                         for j in p._tableau_construction_schedule(right)[3])
        return (parent,), dimension, count, priority
    raise ValueError("Tableau arithmetic requires split or same_action")


def _tableau_action(kind, left, right, parent, arithmetic):
    """Bind shared adjacent actions to subgroup restriction or diagonal product."""
    rational = arithmetic == "rational"
    if kind == "split":
        actions = _generator_layout(parent) if rational else _orthogonal_actions(parent)
        offsets = (0, sum(left))
        apply = _rational_adjacent if rational else _numeric_adjacent
        return lambda factor, generator, value: apply(actions[offsets[factor]+generator], value)
    left_dimension = len(p.standard_tableaux(left))
    right_dimension = len(p.standard_tableaux(right))
    if rational:
        a, b = _generator_layout(left), _generator_layout(right)
        def action(factor, generator, vector):
            first = {}
            for source, coefficient in vector.items():
                i, j = divmod(source, right_dimension)
                for target, value in _rational_adjacent(a[generator], {i: coefficient}).items():
                    key = target*right_dimension+j
                    first[key] = first.get(key, 0)+value
            result = {}
            for source, coefficient in first.items():
                i, j = divmod(source, right_dimension)
                for target, value in _rational_adjacent(b[generator], {j: coefficient}).items():
                    key = i*right_dimension+target
                    result[key] = result.get(key, 0)+value
            return {index: value for index, value in result.items() if value}
    else:
        a, b = _orthogonal_actions(left), _orthogonal_actions(right)
        def action(factor, generator, columns):
            copies = columns.shape[1]
            value = _numeric_adjacent(a[generator], columns.reshape(left_dimension, -1))
            value = value.reshape(left_dimension, right_dimension, copies).transpose(1, 0, 2)
            value = _numeric_adjacent(b[generator], value.reshape(right_dimension, -1))
            return value.reshape(right_dimension, left_dimension, copies).transpose(1, 0, 2).reshape(columns.shape)
    return action


@lru_cache(maxsize=128)
def _prepare_tableau_recipe(kind, left, right, parent):
    """Reuse the small discrete coordinate recipe, independently of FP64 blocks.

    Cache reads check schema, convention and planned index bounds. Exact
    reconstruction is available only with the explicit full diagnostic policy.
    """
    from ye3t.cache.artifacts import YE3TArtifactStore

    store = YE3TArtifactStore()
    request = dict(kind=kind, left=left, right=right, parent=parent,
                   convention=p.TABLEAU_PROJECTOR_COPY_CONVENTION,
                   recipe="rational_seminormal_selected_coordinates_v1")
    def build():
        indices, candidates, support, bits = _construct_tableau_recipe(kind, left, right, parent)
        return dict(indices=indices, candidates=candidates, support=support, fraction_bits=bits)
    def validate(payload):
        _, dimension, count, _ = _tableau_request(kind, left, right, parent)
        indices = payload["indices"]
        if (len(indices) != count or len(set(indices)) != count or
                any(type(i) is not int or not 0 <= i < dimension for i in indices) or
                any(type(payload[k]) is not int or payload[k] < 0
                    for k in ("candidates", "support", "fraction_bits"))):
            raise ValueError("Stored tableau recipe disagrees with its planned dimensions")
        if store.verify == "full":
            expected = _construct_tableau_recipe(kind, left, right, parent)
            actual = (tuple(indices), payload["candidates"], payload["support"], payload["fraction_bits"])
            if actual != expected:
                raise ValueError("Stored tableau recipe differs from exact coordinate preparation")
        return True
    payload = store.resolve("tableau_coordinate_recipe", "ye3t_tableau_coordinate_recipe_v1",
        request, builder=build, validator=validate,
        certificate={"passed": True, "checks": {"canonical_coordinate_recipe": True}},
        required_certificate_checks=("canonical_coordinate_recipe",),
        producer={"compiler_revision": 1})["payload"]
    return tuple(payload["indices"]), payload["candidates"], payload["support"], payload["fraction_bits"]


def _construct_tableau_recipe(kind, left, right, parent):
    """Exact seed selection with rational LDL pivots; m is already known.

    Returns (indices, candidate_count, peak_selected_column_support,
    peak_fraction_bits). No component transport or exact coefficient table is
    constructed. The counters expose the actual discrete preparation work.
    """
    partitions, _, count, priority = _tableau_request(kind, left, right, parent)
    if not count:
        return (), 0, 0, 0
    action = _tableau_action(kind, left, right, parent, "rational")
    if kind == "split":
        weights = _seminormal_weights(parent)
    else:
        weights = tuple(a*b for a in _seminormal_weights(left) for b in _seminormal_weights(right))
    indices, lower, pivots = [], [], []
    candidates = support = bits = 0
    for index in priority:
        vector = p._tableau_projector_columns(
            action, partitions, {index: Fraction(1)}, arithmetic="rational")
        candidates += 1
        support = max(support, len(vector))
        bits = max(bits, max((max(abs(v.numerator).bit_length(), v.denominator.bit_length())
                              for v in vector.values()), default=0))
        row = []
        for j, selected in enumerate(indices):
            value = weights[selected]*vector.get(selected, 0)
            value -= sum(row[k]*pivots[k]*lower[j][k] for k in range(j))
            row.append(value/pivots[j])
        pivot = weights[index]*vector.get(index, 0)-sum(v*v*d for v, d in zip(row, pivots))
        if not pivot:
            continue
        if pivot < 0:
            raise ArithmeticError("Negative exact tableau coordinate pivot")
        indices.append(index)
        lower.append(row)
        pivots.append(pivot)
        if len(indices) == count:
            return tuple(indices), candidates, support, bits
    raise ArithmeticError("Tableau recipe exhausted coordinates for the known multiplicity")


def _materialize_tableau_recipe(kind, left, right, parent, seed_indices):
    """FP64 coefficients from a prepared recipe using shared construction."""
    partitions, dimension, count, priority = _tableau_request(kind, left, right, parent)
    action = _tableau_action(kind, left, right, parent, "numeric")
    return p._tableau_projector_copies(action, partitions, dimension, count, priority,
                                     seed_indices=seed_indices)
