"""Reduced local-density products in canonical multiplicity coordinates.

Only untransported source constructions enter this calculation. Coefficients
depend symbolically on the learned channel maps; physical factor variables are
never expanded through a hidden network. Block fusion and angular recoupling
reuse the exact representation routines. Source counting remains separate.
"""

from collections import Counter
from functools import lru_cache
import copy
import json

import sympy as sp

from ye3t.exact_scalars import exact_scalar
from ye3t.representations.factorized_product_images import (
    factorized_requested_sector, _single_factor_carriers, _sparse_inner,
    _couple_multiplets, _angular_block_recoupling, _angular_tree_coefficients)
from ye3t.representations.young_sectors import (
    permutation_irrep_for_character, _permutation_irrep_signature)


@lru_cache(maxsize=256)
def _sector(word, L):
    nin, lin = zip(*word)
    parts = tuple((size,) for size in Counter(word).values())
    irrep = permutation_irrep_for_character(nin, lin, parts)
    return factorized_requested_sector(nin, lin, _permutation_irrep_signature(irrep), L,
                                      materialize_carriers=False)


@lru_cache(maxsize=256)
def _block_gram(size, ell, L):
    vectors = _single_factor_carriers(size, ell, (size,), L)
    count = len(vectors)
    gram = sp.Matrix([[_sparse_inner(vectors[a, 0][L], vectors[b, 0][L])._sympy_()
                       for b in range(count)] for a in range(count)])
    return gram.inv()


@lru_cache(maxsize=512)
def _density_block_fusion(left_size, right_size, ell, left_L, right_L, L):
    """Identity-placement multiplication in a symmetric block's copy space."""
    left = _single_factor_carriers(left_size, ell, (left_size,), left_L)
    right = _single_factor_carriers(right_size, ell, (right_size,), right_L)
    target = _single_factor_carriers(left_size + right_size, ell, (left_size + right_size,), L)
    gram_inverse = _block_gram(left_size + right_size, ell, L)
    result = {}
    for a in range(len(left)):
        for b in range(len(right)):
            vector = _couple_multiplets(left[a, 0], left_L, right[b, 0], right_L, L)[L]
            overlap = sp.Matrix([_sparse_inner(target[c, 0][L], vector)._sympy_() for c in range(len(target))])
            for c, value in enumerate(gram_inverse * overlap):
                if value != 0:
                    result[c, a, b] = sp.simplify(value)
    return result


@lru_cache(maxsize=512)
def _product_matrix(left_word, left_L, right_word, right_L, L):
    """Sparse target-copy by child-copy product map; no global magnetic array."""
    word = tuple(sorted(left_word + right_word))
    left, right, target = _sector(left_word, left_L), _sector(right_word, right_L), _sector(word, L)
    left_counts, right_counts = Counter(left_word), Counter(right_word)
    channels = tuple(Counter(word))
    left_positions = tuple(channels.index(channel) for channel in left_counts)
    right_positions = tuple(channels.index(channel) for channel in right_counts)
    left_at, right_at = {v: i for i, v in enumerate(left_positions)}, {v: i for i, v in enumerate(right_positions)}
    result = {}
    for c, target_copy in enumerate(target.copy_descriptors):
        for a, left_copy in enumerate(left.copy_descriptors):
            for b, right_copy in enumerate(right.copy_descriptors):
                value = _angular_block_recoupling(left_copy, right_copy, target_copy,
                    left_positions, right_positions, left_L, right_L, L)
                if not value:
                    continue
                for i, channel in enumerate(channels):
                    li, ri = left_at.get(i), right_at.get(i)
                    if li is None or ri is None:
                        descriptor, position = (right_copy, ri) if li is None else (left_copy, li)
                        if (descriptor[0][position], descriptor[1][position]) != (target_copy[0][i], target_copy[1][i]):
                            value = 0
                            break
                    else:
                        fusion = _density_block_fusion(left_counts[channel], right_counts[channel], channel[1],
                            left_copy[0][li], right_copy[0][ri], target_copy[0][i])
                        value *= fusion.get((target_copy[1][i], left_copy[1][li], right_copy[1][ri]), 0)
                        if not value:
                            break
                if value:
                    result[c, a, b] = sp.simplify(value)
    return word, result


@lru_cache(maxsize=256)
def _source_block_change(size, ell, L, source_copy):
    """Convert the source's normalized Schur copy into the reduced-map gauge."""
    from .tagged_cauchy_carriers import _angular_data
    states, vectors, _ = _angular_data(size, ell, (size,), L)
    supplied = {tuple(state): exact_scalar(value)
                for state, value in zip(states, vectors[source_copy, 0, L], strict=True) if value != 0}
    target = _single_factor_carriers(size, ell, (size,), L)
    overlap = sp.Matrix([_sparse_inner(target[c, 0][L], supplied)._sympy_() for c in range(len(target))])
    return tuple(sp.simplify(value) for value in _block_gram(size, ell, L) * overlap)


def _source_coordinate(coordinate, channel_ids):
    if coordinate["label"].get("basis_coordinates") == "formal_cauchy":
        return _formal_shell_coordinate(coordinate, channel_ids)
    from .tagged_cauchy_carriers import _outer_rows
    label = coordinate["label"]
    channels = [(_channel_identity(channel), channel["l"]) for channel in coordinate["channels"]]
    word = tuple(sorted((channel_ids[channel], ell) for (channel, ell), size in
                        zip(channels, label["block_sizes"], strict=True) for _ in range(size)))
    canonical = tuple(Counter(word))
    source_positions = {(channel_ids[identity], ell): i for i, (identity, ell) in enumerate(channels)}
    order = [source_positions[channel] for channel in canonical]
    L = label["target_L"]
    sector = _sector(word, L)
    changes = [_source_block_change(size, channel["l"], spin, source_copy)
        for size, channel, spin, source_copy in zip(label["block_sizes"], coordinate["channels"],
            label["block_Lambdas"], label["angular_copy_indices"], strict=True)]
    outer = _outer_rows(tuple(label["block_Lambdas"]), L, label["outer_copy_index"])[L]
    phase = (-sp.I) ** ((sum(ell for _, ell in word) - L) % 2)
    result = {}
    for a, descriptor in enumerate(sector.copy_descriptors):
        spins, copies, intermediates = descriptor
        if spins != tuple(label["block_Lambdas"][i] for i in order):
            continue
        target_tree = _angular_tree_coefficients(spins, intermediates, L)
        overlap = sum((value * target_tree.get(tuple(magnetic[i] for i in order), exact_scalar(0))._sympy_()
                       for magnetic, value in outer.items()), sp.S.Zero)
        value = sp.simplify(phase * overlap * sp.prod(changes[i][copies[j]] for j, i in enumerate(order)))
        if value:
            result[word, L, a] = value
    return result


@lru_cache(maxsize=256)
def _role_weight_block_change(size, ell, kappa, L, weight, role_copy, angular_copy, role_order):
    """A local Schur weight copy in the complete-shell-channel gauge.

    Only one repeated base-channel block is expanded. The solve acts on its
    multiplicity space; subsequent blocks combine through reduced fusion maps.
    """
    from itertools import product
    from sympy.utilities.iterables import multiset_permutations
    from . import lifted_cauchy_scalar as scalar

    metadata = scalar._role_schur_pivot_metadata(len(weight), size, kappa)
    candidates = [a for a, (_, content) in enumerate(metadata) if tuple(content) == weight]
    template = scalar._compile_block_template((len(weight), size, kappa, ell, L),
                                             angular_basis_backend="exact_weight_space_v1")["payload"]
    row = scalar._block_template_from_payload(template)["canonical_rows"][candidates[role_copy], angular_copy, L]
    word = tuple((position + 1, ell) for position, role in enumerate(role_order) for _ in range(weight[role]))
    source = {}
    for monomial, coefficient in row.items():
        permutations = [tuple(tuple(values) for values in multiset_permutations(
            [magnetic - ell for selected_role, magnetic in monomial if selected_role == role]))
            for role in role_order]
        divisor = sp.prod(len(values) for values in permutations)
        for parts in product(*permutations):
            magnetic = tuple(m for part in parts for m in part)
            source[magnetic] = source.get(magnetic, exact_scalar(0)) + exact_scalar(coefficient / divisor)
    sector = _sector(word, L)
    counts = tuple(Counter(word).values())
    vectors = []
    for spins, copies, intermediates in sector.copy_descriptors:
        multiplet = _single_factor_carriers(counts[0], ell, (counts[0],), spins[0])[copies[0], 0]
        current_L = spins[0]
        for i in range(1, len(counts)):
            right = _single_factor_carriers(counts[i], ell, (counts[i],), spins[i])[copies[i], 0]
            multiplet = _couple_multiplets(multiplet, current_L, right, spins[i], intermediates[i - 1])
            current_L = intermediates[i - 1]
        vectors.append(multiplet[L])
    gram = sp.Matrix([[_sparse_inner(a, b)._sympy_() for b in vectors] for a in vectors])
    overlap = sp.Matrix([_sparse_inner(vector, source)._sympy_() for vector in vectors])
    return word, tuple(sp.simplify(value) for value in gram.inv() * overlap)


def _formal_shell_coordinate(coordinate, channel_ids):
    from ye3t.global_coupler import _angular_factorized_paths
    from .tagged_cauchy_carriers import _outer_tree_row
    from .covariant_cauchy import _outer_multiplet_template
    from . import lifted_cauchy_scalar as scalar

    label, request = coordinate["label"], coordinate["coupling_request"]
    roles = request["role_dimension"]
    blocks = []
    for b, (size, channel, kappa, L, weight, role, angular) in enumerate(zip(
            request["block_sizes"], request["channels"], label["block_kappas"],
            label["block_Lambdas"], label["block_role_contents"], label["role_weight_copy_indices"],
            label["angular_copy_indices"], strict=True)):
        role_order = tuple(sorted((a for a, count in enumerate(weight) if count),
            key=lambda a: channel_ids[_channel_identity(coordinate["channels"][b * roles + a])]))
        local_word, values = _role_weight_block_change(
            size, channel["l"], tuple(kappa), L, tuple(weight), role, angular, role_order)
        mapped = tuple((channel_ids[_channel_identity(coordinate["channels"][b * roles + role_order[a - 1]])], ell)
                       for a, ell in local_word)
        blocks.append({(mapped, L, a): value for a, value in enumerate(values) if value})
    spins, target_L = tuple(label["block_Lambdas"]), label["target_L"]
    outer = _outer_multiplet_template(spins, target_L)
    selected = next(row for row in outer["analysis_rows"]
                    if row["outer_copy_index"] == label["outer_copy_index"] and row["M"] == target_L)
    coefficients = {tuple(item[0] for item in magnetic): value
                    for magnetic, value in scalar._terms_from_records(selected["terms"]).items()}
    result = {}

    def coupled(tree):
        if tree["kind"] == "leaf":
            return blocks[tree["index"]]
        return _coupled(coupled(tree["left"]), coupled(tree["right"]), tree["L"])

    for path in _angular_factorized_paths(spins, target_L, "balanced"):
        tree = _outer_tree_row(path["tree"], target_L)
        overlap = sum((value * tree.get(magnetic, sp.S.Zero) for magnetic, value in coefficients.items()), sp.S.Zero)
        if overlap:
            _add(result, coupled(path["tree"]), overlap)
    phase = (-sp.I) ** ((sum(size * channel["l"] for size, channel in
                           zip(request["block_sizes"], request["channels"], strict=True)) - target_L) % 2)
    factor = phase * scalar._parent_factor(label["block_sizes"])
    return {key: sp.simplify(factor * value) for key, value in result.items() if value}


def _channel_identity(channel):
    return json.dumps({key: value for key, value in channel.items() if key not in {"channel_id", "channel_index"}}, sort_keys=True)


def _add(destination, source, scale=sp.S.One):
    for key, value in source.items():
        destination[key] = destination.get(key, 0) + scale * value


def _coupled(left, right, L):
    result = {}
    for (left_word, left_L, a), x in left.items():
        for (right_word, right_L, b), y in right.items():
            word, matrix = _product_matrix(left_word, left_L, right_word, right_L, L)
            for (c, aa, bb), value in matrix.items():
                if (aa, bb) == (a, b):
                    result[word, L, c] = result.get((word, L, c), 0) + value * x * y
    return {key: value for key, item in result.items() if (value := sp.expand(item)) != 0}


def _routes(left, right, request):
    repeated = request["argument_ids"][0] == request["argument_ids"][1]
    sign = (-1) ** (request["left"]["key"]["rotation_L"] + request["right"]["key"]["rotation_L"] - request["target"]["rotation_L"])
    rows = []
    for a, x in enumerate(left):
        for b, y in enumerate(right):
            if repeated and (b < a or (a == b and sign == -1)):
                continue
            row = _coupled(x, y, request["target"]["rotation_L"])
            if repeated and a != b:
                row = {key: sp.sqrt(2) * value for key, value in row.items()}
            rows.append(row)
    if "selected_copy_indices" in request:
        return [rows[i] for i in request["selected_copy_indices"]]
    return rows[:request["copy_limit"]] if "copy_limit" in request else rows


@lru_cache(maxsize=64)
def _plan(serialized):
    request = json.loads(serialized)
    nodes, arguments = request["full_path"]["nodes"], request["full_path"]["arguments"]
    leaves = set()

    def supported(name):
        node = nodes[name]
        if node.get("value_scope") == "propagated_hidden_state":
            return False
        if node["operation"] == "source":
            # A trainable one-factor channel map needs its own canonical
            # factor bindings. The present source coordinates describe fixed
            # channels; do not silently erase an upstream learned dependency.
            if node.get("parameter_bindings"):
                return False
            source = node.get("source", {})
            binding = source.get("physical_binding", {})
            if source.get("tag_count") != 0 or binding.get("actual_action") != "ordinary_density:S0" or binding.get("density_context") != "inclusive":
                return False
            leaves.add(name)
            return True
        if node.get("value_scope") != "local_source" or node["operation"] not in {"learned_channel_projection", "same_action_tensor_product"}:
            return False
        return all(supported(child) for child in node["inputs"])

    if not all(supported(name) for name in arguments):
        return None
    bindings = {json.dumps(nodes[name]["source"]["physical_binding"], sort_keys=True) for name in leaves}
    if len(bindings) != 1:
        return None
    channels = sorted({_channel_identity(channel) for name in leaves
                       for coordinate in nodes[name]["source"]["coordinates"] for channel in coordinate["channels"]})
    channel_ids = {key: i for i, key in enumerate(channels, 1)}
    memo, parameters, layer_bindings = {}, {}, {}

    def project(rows, node, binding):
        width = node["layout"]["channel_count"]
        result = []
        for a in range(width):
            row = {}
            for b, source in enumerate(rows):
                key = (binding, a, b)
                symbol = parameters.setdefault(key, sp.Symbol("w" + str(len(parameters)), real=True))
                _add(row, source, symbol)
            result.append(row)
        return result

    def evaluate(name):
        if name in memo:
            return memo[name]
        node = nodes[name]
        if node["operation"] == "source":
            rows = [_source_coordinate(coordinate, channel_ids) for coordinate in node["source"]["coordinates"]]
        elif node["operation"] == "learned_channel_projection":
            rows = []
            for child in node["inputs"]:
                values = evaluate(child)
                scale = node.get("source_scales", {}).get(child.removeprefix("source."), 1)
                scale = scale if isinstance(scale, list) else [scale] * len(values)
                chemical_scale = sp.S.One
                if (node.get("chemical") or {}).get("kind") == "learned_scalar_embedding":
                    # Runtime applies exp(N * log_embedding) before grouping
                    # different ranks. Keep that shared dependency exactly.
                    key = ("chemical.log_embedding", 0, 0)
                    chemical = parameters.setdefault(key, sp.Symbol("z_chemical", positive=True))
                    chemical_scale = chemical ** nodes[child]["source"]["factor_rank"]
                rows.extend({key: chemical_scale * sp.Rational(str(s)) * value for key, value in row.items()}
                            for row, s in zip(values, scale, strict=True))
            preconditioner = node.get("source_preconditioner")
            if preconditioner is not None:
                matrix = preconditioner["matrix"]
                if len(matrix) != len(rows) or any(len(row) != len(rows) for row in matrix):
                    raise ValueError("Source preconditioner must match the selected source multiplicities")
                transformed = []
                for coefficients in matrix:
                    row = {}
                    for value, source in zip(coefficients, rows, strict=True):
                        _add(row, source, sp.Rational(str(value)))
                    transformed.append(row)
                rows = transformed
            rows = project(rows, node, node["parameter_binding"])
        else:
            left, right = (evaluate(child) for child in node["inputs"])
            rows = _routes(left, right, node["coupling_selection"])
            layer_bindings[node["parameter_bindings"][0]] = node["parameter_bindings"]
            rows = project(rows, node, node["parameter_bindings"][0])
        memo[name] = rows
        return rows

    left, right = (evaluate(name) for name in arguments)
    routes = _routes(left, right, request)
    routes = [{key: value for key, item in row.items() if (value := sp.expand(item)) != 0} for row in routes]
    targets = sorted({key for row in routes for key in row})
    matrix = sp.SparseMatrix(len(targets), len(routes), {
        (i, j): sp.expand(routes[j][key]) for i, key in enumerate(targets) for j in range(len(routes)) if key in routes[j]})
    # Exact parameter specialization plus the matrix dimension proves generic
    # rank when the upper bound is attained. No geometries or floating ranks
    # participate. Otherwise exact symbolic elimination supplies the bound.
    import random
    generator = random.Random(19026)
    witness = {value: sp.Integer(generator.randint(1, 19)) for value in parameters.values()}
    witness_matrix = matrix.subs(witness)
    witness_rank = witness_matrix.rank()
    rank = witness_rank if witness_rank == min(matrix.shape) else matrix.rank()
    result = {"schema": "ye3t_reduced_density_product_image_v1",
        "target_basis": [{"content": [[json.loads(channels[token-1]), ell] for token, ell in word],
                          "L": L, "multiplicity_index": a, "copy_descriptor": _sector(word, L).copy_descriptors[a]}
                         for word, L, a in targets],
        "route_count": len(routes), "independent_image_count": rank,
        "parameter_bindings": [{"symbol": str(symbol), "parameter": name, "row": a, "column": b,
                                "transform": "exp" if name == "chemical.log_embedding" else "identity",
                                **({"per_layer_parameters": layer_bindings[name]} if name in layer_bindings else {})}
                               for (name, a, b), symbol in parameters.items()],
        "route_to_target": [[i, j, str(value)] for (i, j), value in sorted(matrix.todok().items()) if value != 0],
        "rank_proof": {"exact_parameter_witness": {str(key): str(value) for key, value in witness.items()},
                       "witness_rank": witness_rank, "upper_bound": min(matrix.shape),
                       "symbolic_elimination": witness_rank != min(matrix.shape)},
        "rank_scope": "generic_learned_parameters_with_possible_rank_drop_at_special_weights",
        "runtime_parameter_dependencies": "original_factorized_contractions_preserved",
        "global_magnetic_matrix_materialized": False}
    return json.loads(json.dumps(result))


def reduced_density_product_image(request):
    if request.get("full_path") is None or request["operation"] != "same_action_tensor_product" or request["target"]["rank"] != 0:
        return None
    return copy.deepcopy(_plan(json.dumps(request, sort_keys=True)))
