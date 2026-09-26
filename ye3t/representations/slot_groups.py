"""Finite slot-group utilities for motif and channel symmetries.

General low-order helpers use explicit permutation enumeration. Rooted typed
stars additionally have an exact generator representation, so repeated blocks
can remain compact at high tensor rank.
"""

import math
from itertools import permutations


def _as_perm(perm):
    return tuple(int(x) for x in perm)


def apply_slot_permutation(values, perm):
    """Return ``values`` with output slot ``a`` receiving input slot ``perm[a]``."""

    perm = _as_perm(perm)
    if len(values) != len(perm):
        raise ValueError("Permutation length must match the number of values.")
    return tuple(values[int(idx)] for idx in perm)


def graph_template_automorphisms(vertex_count, edges=()):
    """Return graph automorphisms as one-line slot permutations.

    ``edges`` is interpreted as an undirected simple graph on
    ``range(vertex_count)``.  A permutation ``p`` is returned when
    ``{p[a], p[b]}`` is an edge exactly when ``{a, b}`` is an edge.
    """

    vertex_count = int(vertex_count)
    if vertex_count < 0:
        raise ValueError("vertex_count must be nonnegative.")
    normalized_edges = {frozenset((int(a), int(b))) for a, b in edges if int(a) != int(b)}
    for edge in normalized_edges:
        if any(v < 0 or v >= vertex_count for v in edge):
            raise ValueError("Graph edge contains a vertex outside range(vertex_count).")
    out = []
    for perm in permutations(range(vertex_count)):
        mapped = {frozenset((perm[int(a)], perm[int(b)])) for a, b in (tuple(edge) for edge in normalized_edges)}
        if mapped == normalized_edges:
            out.append(_as_perm(perm))
    return tuple(out)


def repeated_channel_group(channel_pattern):
    """Return slot permutations preserving equal channel labels."""

    pattern = tuple(channel_pattern)
    out = []
    for perm in permutations(range(len(pattern))):
        if all(pattern[int(perm[a])] == pattern[a] for a in range(len(pattern))):
            out.append(_as_perm(perm))
    return tuple(out)


def decorated_automorphism_group(vertex_count, edges=(), channel_pattern=()):
    """Return ``Aut(H)`` intersected with the repeated-channel slot group."""

    pattern = tuple(channel_pattern)
    if pattern and len(pattern) != int(vertex_count):
        raise ValueError("channel_pattern length must match vertex_count.")
    aut = graph_template_automorphisms(vertex_count, edges)
    if not pattern:
        return aut
    channel_group = set(repeated_channel_group(pattern))
    return tuple(perm for perm in aut if perm in channel_group)


def rooted_typed_graph_automorphisms(
    vertex_count,
    edges=(),
    *,
    root_vertex=0,
    vertex_types=(),
    edge_types=(),
    directed=False,
):
    """Return root- and type-preserving support-graph automorphisms.

    This explicit permutation enumeration is an offline/compiler helper for
    small fixed support graphs. It must not be called from an atomistic
    timestep kernel.
    """

    vertex_count = int(vertex_count)
    root_vertex = int(root_vertex)
    if vertex_count <= 0:
        raise ValueError("vertex_count must be positive.")
    if root_vertex < 0 or root_vertex >= vertex_count:
        raise ValueError("root_vertex is outside range(vertex_count).")
    edges = tuple((int(a), int(b)) for a, b in edges)
    if any(a == b for a, b in edges):
        raise ValueError("support graph edges must not contain self loops.")
    if any(min(a, b) < 0 or max(a, b) >= vertex_count for a, b in edges):
        raise ValueError("support graph edge contains an invalid vertex.")
    vertex_types = tuple(vertex_types)
    if vertex_types and len(vertex_types) != vertex_count:
        raise ValueError("vertex_types must contain one label per vertex.")
    if not vertex_types:
        vertex_types = tuple(None for _ in range(vertex_count))
    edge_types = tuple(edge_types)
    if edge_types and len(edge_types) != len(edges):
        raise ValueError("edge_types must contain one label per edge.")
    if not edge_types:
        edge_types = tuple(None for _ in edges)

    def edge_key(a, b, label):
        if not bool(directed) and b < a:
            a, b = b, a
        return (int(a), int(b), label)

    edge_records = tuple(
        edge_key(a, b, label)
        for (a, b), label in zip(edges, edge_types)
    )
    if len(set(edge_records)) != len(edge_records):
        raise ValueError(
            "typed support graph edges must be unique; use "
            "factor_edge_indices for repeated tensor factors."
        )
    edge_record_set = set(edge_records)
    out = []
    for perm in permutations(range(vertex_count)):
        if int(perm[root_vertex]) != root_vertex:
            continue
        if any(
            vertex_types[int(perm[index])] != vertex_types[index]
            for index in range(vertex_count)
        ):
            continue
        mapped = {
            edge_key(perm[a], perm[b], label)
            for (a, b), label in zip(edges, edge_types)
        }
        if mapped == edge_record_set:
            out.append(_as_perm(perm))
    return tuple(out)


def rooted_typed_star_automorphism_generators(
    vertex_count,
    edges=(),
    *,
    root_vertex=0,
    vertex_types=(),
    edge_types=(),
    directed=False,
):
    """Return a compact exact automorphism description for a rooted star.

    Adjacent transpositions generate each interchangeable typed/decorated leaf
    block. ``None`` is returned when the support is not a rooted star.
    """

    vertex_count = int(vertex_count)
    root_vertex = int(root_vertex)
    edges = tuple((int(a), int(b)) for a, b in edges)
    vertex_types = tuple(vertex_types) or tuple(
        None for _ in range(vertex_count)
    )
    edge_types = tuple(edge_types) or tuple(None for _ in edges)
    if (
        vertex_count <= 0
        or root_vertex < 0
        or root_vertex >= vertex_count
        or len(vertex_types) != vertex_count
        or len(edge_types) != len(edges)
        or len(edges) != vertex_count - 1
    ):
        return None
    leaf_records = {}
    seen_leaves = set()
    for edge_index, ((left, right), edge_type) in enumerate(
        zip(edges, edge_types)
    ):
        if bool(directed):
            if int(left) != root_vertex or int(right) == root_vertex:
                return None
            leaf = int(right)
            orientation = "root_to_leaf"
        else:
            if int(left) == root_vertex and int(right) != root_vertex:
                leaf = int(right)
            elif int(right) == root_vertex and int(left) != root_vertex:
                leaf = int(left)
            else:
                return None
            orientation = "undirected"
        if leaf in seen_leaves:
            return None
        seen_leaves.add(leaf)
        key = (vertex_types[leaf], edge_type, orientation)
        leaf_records.setdefault(key, []).append((leaf, int(edge_index)))
    expected_leaves = set(range(vertex_count)) - {root_vertex}
    if seen_leaves != expected_leaves:
        return None
    identity = tuple(range(vertex_count))
    generators = [identity]
    blocks = []
    order = 1
    for records in sorted(
        leaf_records.values(),
        key=lambda values: tuple(item[0] for item in values),
    ):
        vertices = tuple(sorted(int(item[0]) for item in records))
        blocks.append(vertices)
        order *= math.factorial(len(vertices))
        for left, right in zip(vertices[:-1], vertices[1:]):
            permutation = list(identity)
            permutation[int(left)] = int(right)
            permutation[int(right)] = int(left)
            generators.append(tuple(permutation))
    return {
        "generators": tuple(generators),
        "order": int(order),
        "symmetric_vertex_blocks": tuple(
            block for block in blocks if len(block) > 1
        ),
    }


def rooted_typed_tree_automorphism_generators(
    vertex_count,
    edges=(),
    *,
    root_vertex=0,
    vertex_types=(),
    edge_types=(),
    directed=False,
):
    """Return compact exact generators for a typed rooted support tree.

    Isomorphic child subtrees are exchanged by adjacent transpositions.  The
    generators internal to every child subtree are retained separately.  This
    is the standard recursive wreath-product description of a rooted-tree
    automorphism group and avoids enumerating all vertex permutations.
    """

    vertex_count = int(vertex_count)
    root_vertex = int(root_vertex)
    edges = tuple((int(a), int(b)) for a, b in edges)
    vertex_types = tuple(vertex_types) or tuple(
        None for _ in range(vertex_count)
    )
    edge_types = tuple(edge_types) or tuple(None for _ in edges)
    if (
        vertex_count <= 0
        or root_vertex < 0
        or root_vertex >= vertex_count
        or len(vertex_types) != vertex_count
        or len(edge_types) != len(edges)
        or len(edges) != vertex_count - 1
    ):
        return None
    adjacency = [[] for _ in range(vertex_count)]
    for edge_index, ((left, right), edge_type) in enumerate(
        zip(edges, edge_types)
    ):
        if (
            left == right
            or min(left, right) < 0
            or max(left, right) >= vertex_count
        ):
            return None
        adjacency[left].append((right, edge_index, 1, edge_type))
        adjacency[right].append((left, edge_index, -1, edge_type))
    parents = [-2] * vertex_count
    parents[root_vertex] = -1
    traversal = []
    queue = [root_vertex]
    children = [[] for _ in range(vertex_count)]
    while queue:
        vertex = queue.pop(0)
        traversal.append(vertex)
        for child, edge_index, sign, edge_type in sorted(
            adjacency[vertex], key=lambda item: (item[0], item[1])
        ):
            if child == parents[vertex]:
                continue
            if parents[child] != -2:
                return None
            parents[child] = vertex
            children[vertex].append(
                (child, edge_index, sign, edge_type)
            )
            queue.append(child)
    if len(traversal) != vertex_count:
        return None

    identity = tuple(range(vertex_count))
    signatures = [None] * vertex_count
    canonical_orders = [None] * vertex_count
    automorphism_orders = [1] * vertex_count
    generators = [identity]
    symmetric_blocks = []
    for vertex in reversed(traversal):
        grouped = {}
        records = []
        for child, edge_index, sign, edge_type in children[vertex]:
            orientation = int(sign) if bool(directed) else 0
            descriptor = (
                edge_type,
                orientation,
                signatures[child],
            )
            grouped.setdefault(repr(descriptor), []).append(child)
            records.append((repr(descriptor), child))
        records.sort(key=lambda item: (item[0], item[1]))
        signatures[vertex] = (
            vertex_types[vertex],
            tuple(item[0] for item in records),
        )
        canonical_order = [vertex]
        for _descriptor, child in records:
            canonical_order.extend(canonical_orders[child])
        canonical_orders[vertex] = tuple(canonical_order)
        order = 1
        for descriptor in sorted(grouped):
            child_vertices = tuple(sorted(grouped[descriptor]))
            child_order = int(automorphism_orders[child_vertices[0]])
            order *= child_order ** len(child_vertices)
            order *= math.factorial(len(child_vertices))
            if len(child_vertices) > 1:
                symmetric_blocks.append(child_vertices)
            for left, right in zip(
                child_vertices[:-1], child_vertices[1:]
            ):
                left_order = canonical_orders[left]
                right_order = canonical_orders[right]
                if len(left_order) != len(right_order):
                    raise AssertionError(
                        "isomorphic rooted subtrees have unequal sizes"
                    )
                permutation = list(identity)
                for left_vertex, right_vertex in zip(
                    left_order, right_order
                ):
                    permutation[left_vertex] = right_vertex
                    permutation[right_vertex] = left_vertex
                generators.append(tuple(permutation))
        automorphism_orders[vertex] = int(order)
    return {
        "generators": tuple(generators),
        "order": int(automorphism_orders[root_vertex]),
        "symmetric_vertex_blocks": tuple(symmetric_blocks),
    }


def induced_factor_role_permutations(
    edges,
    factor_edge_indices,
    vertex_automorphisms,
    *,
    factor_labels=(),
    directed=False,
):
    """Lift support automorphisms to stored factor-role permutations.

    Repeated factors on one support edge are matched by stable ordinal within
    each equal-label block. This fixes one exact compiler convention that must
    be serialized with the source plan.
    """

    edges = tuple((int(a), int(b)) for a, b in edges)
    factor_edge_indices = tuple(int(index) for index in factor_edge_indices)
    if any(index < 0 or index >= len(edges) for index in factor_edge_indices):
        raise ValueError("factor_edge_indices contains an invalid support edge.")
    factor_labels = tuple(factor_labels)
    if factor_labels and len(factor_labels) != len(factor_edge_indices):
        raise ValueError("factor_labels must contain one label per factor role.")
    if not factor_labels:
        factor_labels = tuple(None for _ in factor_edge_indices)

    def edge_key(edge):
        a, b = edge
        if not bool(directed) and b < a:
            a, b = b, a
        return (int(a), int(b))

    edge_lookup = {edge_key(edge): index for index, edge in enumerate(edges)}
    if len(edge_lookup) != len(edges):
        raise ValueError("support edges must be unique.")
    factor_groups = {}
    factor_ordinals = []
    for factor_index, (edge_index, label) in enumerate(
        zip(factor_edge_indices, factor_labels)
    ):
        key = (int(edge_index), label)
        group = factor_groups.setdefault(key, [])
        factor_ordinals.append(len(group))
        group.append(int(factor_index))

    lifts = []
    for vertex_perm in tuple(vertex_automorphisms):
        vertex_perm = _as_perm(vertex_perm)
        mapped_edge_indices = []
        for a, b in edges:
            mapped_key = edge_key((vertex_perm[a], vertex_perm[b]))
            if mapped_key not in edge_lookup:
                raise ValueError(
                    "vertex permutation is not a support-graph automorphism."
                )
            mapped_edge_indices.append(int(edge_lookup[mapped_key]))
        factor_perm = []
        for factor_index, (edge_index, label) in enumerate(
            zip(factor_edge_indices, factor_labels)
        ):
            mapped_key = (mapped_edge_indices[int(edge_index)], label)
            candidates = factor_groups.get(mapped_key, ())
            ordinal = int(factor_ordinals[factor_index])
            if ordinal >= len(candidates):
                raise ValueError(
                    "support automorphism does not preserve the decorated "
                    "factor-support multiset."
                )
            factor_perm.append(int(candidates[ordinal]))
        lifts.append(tuple(factor_perm))
    return tuple(lifts)


def slot_orbits(vertex_count, group):
    """Return sorted slot orbits for a finite permutation group."""

    vertex_count = int(vertex_count)
    group = tuple(_as_perm(perm) for perm in group)
    unseen = set(range(vertex_count))
    orbits = []
    while unseen:
        seed = min(unseen)
        orbit = {seed}
        changed = True
        while changed:
            changed = False
            for perm in group:
                for item in tuple(orbit):
                    image = int(perm[item])
                    if image not in orbit:
                        orbit.add(image)
                        changed = True
        unseen.difference_update(orbit)
        orbits.append(tuple(sorted(orbit)))
    return tuple(orbits)


def pair_orbits(vertex_count, group):
    """Return orbits on ordered slot pairs under simultaneous group action."""

    vertex_count = int(vertex_count)
    group = tuple(_as_perm(perm) for perm in group)
    pairs = {(a, b) for a in range(vertex_count) for b in range(vertex_count)}
    orbits = []
    while pairs:
        seed = min(pairs)
        orbit = {seed}
        changed = True
        while changed:
            changed = False
            for perm in group:
                for a, b in tuple(orbit):
                    image = (int(perm[a]), int(perm[b]))
                    if image not in orbit:
                        orbit.add(image)
                        changed = True
        pairs.difference_update(orbit)
        orbits.append(tuple(sorted(orbit)))
    return tuple(orbits)


__all__ = [
    "apply_slot_permutation",
    "decorated_automorphism_group",
    "graph_template_automorphisms",
    "induced_factor_role_permutations",
    "pair_orbits",
    "repeated_channel_group",
    "rooted_typed_graph_automorphisms",
    "rooted_typed_star_automorphism_generators",
    "rooted_typed_tree_automorphism_generators",
    "slot_orbits",
]
