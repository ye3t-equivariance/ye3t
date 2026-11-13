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
    factorized_requested_sector, _sparse_inner,
    _angular_block_recoupling, _angular_tree_coefficients)
from ye3t.representations.young_angular_paths import (
    young_angular_path_basis, paired_occupation_path_basis, _coupled_path_columns,
    _young_angular_product_overlap, _permuted_chain_column, _inner, _add_scaled)
from ye3t.representations.young_sectors import (
    permutation_irrep_for_character, _permutation_irrep_signature)


@lru_cache(maxsize=256)
def _sector(word, L):
    nin, lin = zip(*word)
    parts = tuple((size,) for size in Counter(word).values())
    irrep = permutation_irrep_for_character(nin, lin, parts)
    return factorized_requested_sector(nin, lin, _permutation_irrep_signature(irrep), L,
                                      materialize_carriers=False)


def _normalized_sector_path_columns(word, L):
    """Existing fixed-content copy labels realized by normalized CG paths."""
    # Angular coordinates depend on equality and order of complete channels,
    # not their integer names. Generated radial-image channel IDs otherwise
    # cause repeated construction of the same normalized map.
    renaming = {token: i+1 for i, token in enumerate(sorted({token for token, _ in word}))}
    canonical = tuple((renaming[token], ell) for token, ell in word)
    return _canonical_sector_path_columns(canonical, L)


@lru_cache(maxsize=256)
def _canonical_sector_path_columns(word, L):
    sector = _sector(word, L)
    blocks = tuple(Counter(word).items())
    columns = []
    for spins, copies, intermediates in sector.copy_descriptors:
        column = ()
        for b, ((_, ell), size) in enumerate(blocks):
            local = young_angular_path_basis(size, ell, (size,), spins[b])[copies[b]][0]
            column = local if b == 0 else tuple(sorted(_coupled_path_columns(
                column, local, (ell,)*size, intermediates[b-1]).items()))
        columns.append(column)
    return tuple(columns)


def _legacy_magnetic_path_column(column, spins, M):
    """Explicit conversion boundary for historical magnetic source gauges.

    Normalized source requests use path overlaps directly. Only historical
    coefficient conventions need this bounded compatibility conversion.
    """
    result = {}
    for path, value in column:
        result = _add_scaled(result, _angular_tree_coefficients(spins, path[1:], M), value)
    return result


@lru_cache(maxsize=512)
def _density_block_fusion(left_size, right_size, ell, left_L, right_L, L):
    """Shared normalized identity-placement multiplication; exact copy overlap."""
    return {(c, a, b): value._sympy_() for (c, t, a, u, b, v), value in
            _young_angular_product_overlap(left_size, right_size, ell,
                (left_size,), (right_size,), (left_size+right_size,), left_L, right_L, L)}


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
    """Convert a historical Schur source copy into normalized path coordinates."""
    from .tagged_cauchy_carriers import _angular_data
    states, vectors, _ = _angular_data(size, ell, (size,), L)
    supplied = {tuple(state): exact_scalar(value)
                for state, value in zip(states, vectors[source_copy, 0, L], strict=True) if value != 0}
    target = young_angular_path_basis(size, ell, (size,), L)
    return tuple(_sparse_inner(_legacy_magnetic_path_column(copy[0], (ell,)*size, L), supplied)._sympy_()
                 for copy in target)


def _source_coordinate(coordinate, channel_ids):
    if coordinate["label"].get("basis_coordinates") == "symmetric_occupation_v1":
        return _ordinary_occupation_coordinate(coordinate, channel_ids)
    if (coordinate["label"].get("basis_coordinates") == "formal_cauchy" or
            (coordinate["label"].get("tag_count") == 0 and
             coordinate.get("coupling_request", {}).get("angular_basis_backend") == "normalized_cg_paths_v1")):
        return _formal_shell_coordinate(coordinate, channel_ids)
    from ye3t.global_coupler import _indexed_angular_paths
    label = coordinate["label"]
    channels = [(_channel_identity(channel), channel["l"]) for channel in coordinate["channels"]]
    word = tuple(sorted((channel_ids[channel], ell) for (channel, ell), size in
                        zip(channels, label["block_sizes"], strict=True) for _ in range(size)))
    L = label["target_L"]
    changes = [_source_block_change(size, channel["l"], spin, source_copy)
        for size, channel, spin, source_copy in zip(label["block_sizes"], coordinate["channels"],
            label["block_Lambdas"], label["angular_copy_indices"], strict=True)]
    blocks = [{(((channel_ids[identity], ell),)*size, spin, a): value
               for a, value in enumerate(change) if value}
              for (identity, ell), size, spin, change in zip(channels, label["block_sizes"],
                  label["block_Lambdas"], changes, strict=True)]
    trees = _indexed_angular_paths(tuple(label["block_Lambdas"]), L, "balanced")
    result = _couple_source_tree(trees[label["outer_copy_index"]][1], blocks)
    phase = (-sp.I) ** ((sum(ell for _, ell in word) - L) % 2)
    return {key: sp.simplify(phase*value) for key, value in result.items() if value}


def _ordinary_occupation_inventory(labels):
    """Describe existing compiler labels without constructing coefficients.

    The complete-factor parent remains (N). Evaluated commuting moments have
    no retained permutation coordinate; their emitted action is trivial S0.
    The original compact labels fix both block gauges and the outer CG tree.
    """
    from ye3t.core.couplings import _lightweight_structured_label_from_compact
    from ye3t.core.factorized_dag import label_block_specs
    from ye3t.core.labels import normalize_compact_label
    coordinates = []
    for label in labels:
        compact = normalize_compact_label(label)
        structured = _lightweight_structured_label_from_compact(compact)
        if structured is None:
            raise ValueError("Ordinary source requires a canonical structured compiler label")
        blocks = label_block_specs(structured)
        channels = [dict(n=block["n"], l=block["l"]) for block in blocks]
        coordinates.append(dict(channels=channels, compact_label=compact.to_dict(), label=dict(
            basis_coordinates="symmetric_occupation_v1", rank=compact.rank,
            block_sizes=[block["k_b"] for block in blocks],
            block_kappas=[[block["k_b"]] for block in blocks],
            block_Lambdas=[block["Lambda"] for block in blocks],
            angular_copy_indices=[block["multiplicity_index"] for block in blocks],
            tag_count=0, tag_partition=[], target_L=compact.L_R,
            target_parity=(-1)**sum(compact.l_tuple))))
    return coordinates


def _ordinary_occupation_coordinate(coordinate, channel_ids):
    """Express one existing occupation/outer-tree feature in shared copy axes.

    This is the approved compact change of coordinates. It preserves original
    non-unit norms and complex Condon--Shortley phases; no real-form phase is
    introduced into the historical complex ordinary-density evaluator.
    """
    from ye3t.core.basis.labels import LeafLabel, SymBlockLabel, NodeLabel
    from ye3t.core.couplings import _lightweight_structured_label_from_compact
    from ye3t.core.labels import normalize_compact_label
    from ye3t.representations.young_angular_paths import _symmetric_occupation_path_change
    compact = normalize_compact_label(coordinate["compact_label"])
    if coordinate != _ordinary_occupation_inventory([compact])[0]:
        raise ValueError("Ordinary source inventory differs from its canonical compiler label")
    structured = _lightweight_structured_label_from_compact(compact)
    if structured is None:
        raise ValueError("Ordinary source lacks its canonical structured label")
    def evaluate(node):
        if isinstance(node, NodeLabel):
            return _coupled(evaluate(node.left), evaluate(node.right), node.L)
        if not isinstance(node, (LeafLabel, SymBlockLabel)):
            raise ValueError("Unsupported ordinary source compiler label")
        token = channel_ids[_channel_identity(dict(n=node.n, l=node.l))]
        size = node.k_b if isinstance(node, SymBlockLabel) else 1
        word = ((token, node.l),)*size
        if isinstance(node, LeafLabel):
            return {(word, node.l, 0): sp.S.One}
        change = _symmetric_occupation_path_change(size, node.l, node.Lambda)
        return {(word, node.Lambda, a): row[node.multiplicity_index]._sympy_()
                for a, row in enumerate(change) if row[node.multiplicity_index]}
    try:
        return evaluate(structured)
    finally:
        evaluate = None


def _couple_source_tree(tree, blocks):
    if tree[0] == "leaf":
        return blocks[tree[1]]
    return _coupled(_couple_source_tree(tree[2], blocks), _couple_source_tree(tree[3], blocks), tree[1])


@lru_cache(maxsize=256)
def _role_weight_block_change(size, ell, kappa, L, weight, role_copy, angular_copy, role_order,
                              angular_backend="exact_weight_space_v1"):
    """A Schur weight copy in normalized complete-channel coordinates.

    Normalized sources use exact path overlaps. Historical source coefficients
    cross the explicit magnetic conversion boundary once. Subsequent blocks
    combine through the same reduced fusion maps in either case.
    """
    from itertools import product
    from sympy.utilities.iterables import multiset_permutations
    from . import lifted_cauchy_scalar as scalar

    word = tuple((position + 1, ell) for position, role in enumerate(role_order) for _ in range(weight[role]))
    columns = _normalized_sector_path_columns(word, L)
    if angular_backend == "normalized_cg_paths_v1":
        from math import factorial
        angular_count = len(young_angular_path_basis(size, ell, kappa, L))
        paired = paired_occupation_path_basis(weight, ell, kappa, L)
        source = paired[role_copy*angular_count+angular_copy][0]
        original = tuple(role for role, count in enumerate(weight) for _ in range(count))
        order = tuple(i for role in role_order for i, value in enumerate(original) if value == role)
        source = _permuted_chain_column(source, (ell,)*size, order)
        scale = exact_scalar(sp.sqrt(sp.Rational(factorial(size), sp.prod(factorial(n) for n in weight))))
        return word, tuple((_inner(dict(column), source)*scale)._sympy_() for column in columns)
    if angular_backend not in {"legacy_exact", "exact_weight_space_v1"}:
        raise ValueError("Unknown source angular coefficient convention")
    metadata = scalar._role_schur_pivot_metadata(len(weight), size, kappa)
    candidates = [a for a, (_, content) in enumerate(metadata) if tuple(content) == weight]
    template = scalar._compile_block_template((len(weight), size, kappa, ell, L),
                                             angular_basis_backend=angular_backend)["payload"]
    row = scalar._block_template_from_payload(template)["canonical_rows"][candidates[role_copy], angular_copy, L]
    source = {}
    for monomial, coefficient in row.items():
        permutations = [tuple(tuple(values) for values in multiset_permutations(
            [magnetic - ell for selected_role, magnetic in monomial if selected_role == role]))
            for role in role_order]
        divisor = sp.prod(len(values) for values in permutations)
        for parts in product(*permutations):
            magnetic = tuple(m for part in parts for m in part)
            source[magnetic] = source.get(magnetic, exact_scalar(0)) + exact_scalar(coefficient / divisor)
    return word, tuple(_sparse_inner(
        _legacy_magnetic_path_column(column, (ell,)*size, L), source)._sympy_() for column in columns)


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
            size, channel["l"], tuple(kappa), L, tuple(weight), role, angular, role_order,
            request.get("angular_basis_backend", "legacy_exact"))
        mapped = tuple((channel_ids[_channel_identity(coordinate["channels"][b * roles + role_order[a - 1]])], ell)
                       for a, ell in local_word)
        blocks.append({(mapped, L, a): value for a, value in enumerate(values) if value})
    spins, target_L = tuple(label["block_Lambdas"]), label["target_L"]
    if request.get("angular_basis_backend") == "normalized_cg_paths_v1":
        from ye3t.global_coupler import _indexed_angular_paths
        tree = _indexed_angular_paths(spins, target_L, "left")[label["outer_copy_index"]][1]
        result = _couple_source_tree(tree, blocks)
        phase = (-sp.I) ** ((sum(size * channel["l"] for size, channel in
                               zip(request["block_sizes"], request["channels"], strict=True)) - target_L) % 2)
        factor = phase * scalar._parent_factor(label["block_sizes"])
        return {key: sp.simplify(factor*value) for key, value in result.items() if value}
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
    from ye3t.execution_plan import YE3T_O3_REAL_TESSERAL_CONVENTION
    repeated = request["argument_ids"][0] == request["argument_ids"][1]
    sign = (-1) ** (request["left"]["key"]["rotation_L"] + request["right"]["key"]["rotation_L"] - request["target"]["rotation_L"])
    phase = -sp.I if sign == -1 and request["target"]["convention_id"] == YE3T_O3_REAL_TESSERAL_CONVENTION else sp.S.One
    rows = []
    for a, x in enumerate(left):
        for b, y in enumerate(right):
            if repeated and (b < a or (a == b and sign == -1)):
                continue
            row = _coupled(x, y, request["target"]["rotation_L"])
            if phase != 1:
                row = {key: phase * value for key, value in row.items()}
            if repeated and a != b:
                row = {key: sp.sqrt(2) * value for key, value in row.items()}
            rows.append(row)
    if "selected_copy_indices" in request:
        return [rows[i] for i in request["selected_copy_indices"]]
    return rows[:request["copy_limit"]] if "copy_limit" in request else rows


def _constant_operation_image(matrix):
    """Independent maps of a fixed selected image in normalized coordinates.

    Reuse exact orthogonalization of the reduced target-copy columns. The
    resulting analysis acts on selected existing product routes. Reconstruction
    preserves arbitrary coefficients on their full image. Learned parameter
    values never enter this operation or choose its independent coordinates.
    """
    from ye3t.core.basis.homogeneous import _exact_gram_schmidt

    if matrix.free_symbols:
        raise ValueError("Fixed operation maps require parameter-independent source coordinates")
    _, pivots = matrix.rref()
    dimension = len(pivots)
    selected = matrix.extract(range(matrix.rows), pivots)
    if dimension:
        basis = sp.Matrix.hstack(*_exact_gram_schmidt([selected[:, i] for i in range(dimension)]))
        from_selected, free = selected.gauss_jordan_solve(basis)
        if free.rows:
            raise ArithmeticError("Independent selected product maps have undetermined coordinates")
    else:
        basis, from_selected = sp.zeros(matrix.rows, 0), sp.zeros(0, 0)
    reconstruction = (basis.conjugate().T*matrix).applyfunc(sp.simplify)
    if ((basis.conjugate().T*basis-sp.eye(dimension)).applyfunc(sp.simplify) != sp.zeros(dimension)
            or (basis*reconstruction-matrix).applyfunc(sp.simplify) != sp.zeros(*matrix.shape)
            or (selected*from_selected-basis).applyfunc(sp.simplify) != sp.zeros(*basis.shape)):
        raise ArithmeticError("Independent operation maps fail exact isometry or image reconstruction")

    def entries(value):
        return [[i, j, str(sp.simplify(value[i, j]))] for i in range(value.rows)
                for j in range(value.cols) if value[i, j] != 0]

    return {"schema": "ye3t_independent_fixed_product_maps_v1",
        "independent_map_count": dimension, "route_count": matrix.cols,
        "selected_route_indices": list(pivots),
        "target_embedding": entries(basis),
        "selected_route_analysis": entries(from_selected.T),
        "route_reconstruction": entries(reconstruction.T),
        "target_coordinate_convention": "normalized_symmetric_block_CG_paths_v1",
        "parameter_dependent": False, "complete_tableau_magnetic_components": True,
        "validation": {"exact_isometry": True, "exact_image_reconstruction": True,
                       "unrestricted_selected_image_coefficients": True}}


@lru_cache(maxsize=128)
def _shell_role_coordinates(serialized):
    """Reuse the source compiler's exact occupation-to-tableau analysis."""
    from .shell_cauchy import count_shell_cauchy
    from .factorized_shell import _shell_role_path_analysis

    report = count_shell_cauchy(json.loads(serialized))
    base, coupling, analysis = _shell_role_path_analysis(report)
    # Keep representation maps exact. Physical channel assignments are supplied
    # by the same source inventory as the evaluator, including any fixed roles.
    return base, coupling, analysis, report


def _shell_source_multiplets(source, channel_ids):
    # One role analysis per selected request, shared across its copy/tableau
    # outputs. Never compile all unselected angular families to find these maps.
    groups = {}
    for position, coordinate in enumerate(source["coordinates"]):
        request = {"coupling_request": coordinate["coupling_request"], **coordinate["role_binding"]}
        groups.setdefault(json.dumps(request, sort_keys=True), []).append((position, coordinate))
    result = [None]*len(source["coordinates"])
    for serialized, selected in groups.items():
        request = {**json.loads(serialized), "selected_coordinates": [row["coordinate_id"] for _, row in selected]}
        base, coupling, analyses, report = _shell_role_coordinates(json.dumps(request, sort_keys=True))
        labels = {row["coordinate_id"]: json.loads(json.dumps(row)) for row in report["labels"]}
        channels = json.loads(json.dumps(report["request"]["channels"]))
        for _, coordinate in selected:
            if coordinate["label"] != labels[coordinate["coordinate_id"]] or coordinate["channels"] != channels:
                raise ValueError("Shell product source labels or physical channels differ from the canonical source request")
        needed = {index for analysis in analyses.values() for row in analysis for index in row}
        coordinates = {index: _formal_shell_coordinate({"label": base[index],
            "channels": channels, "coupling_request": coupling}, channel_ids) for index in needed}
        for position, coordinate in selected:
            multiplet = []
            for row in analyses[coordinate["coordinate_id"]]:
                values = {}
                for index, coefficient in row.items():
                    _add(values, coordinates[index], coefficient)
                multiplet.append({key: value for key, item in values.items()
                                  if (value := sp.simplify(item)) != 0})
            result[position] = multiplet
    return result


def _fixed_operation_multiplets(routes, tableaux):
    """Resolve one fixed image on complete tableaux in normalized path coordinates.

    This is the existing shell image construction shared with fixed local
    product nodes. Its maps act identically on every tableau/magnetic component.
    Only reduced multiplicity matrices are assembled.
    """
    targets = sorted({key for route in routes for row in route for key in row})
    target_indices = {key: i for i, key in enumerate(targets)}
    matrices = [sp.SparseMatrix(len(targets), len(routes), {
        (target_indices[key], j): value for j, route in enumerate(routes)
        for key, value in route[t].items()}) for t in range(tableaux)]
    assembly = _constant_operation_image(matrices[0])
    q = assembly["independent_map_count"]
    analysis = sp.zeros(q)
    for i, j, value in assembly["selected_route_analysis"]:
        analysis[i, j] = sp.sympify(value)
    reconstruction = sp.zeros(len(routes), q)
    for i, j, value in assembly["route_reconstruction"]:
        reconstruction[i, j] = sp.sympify(value)
    embeddings = [matrix[:, assembly["selected_route_indices"]]*analysis.T for matrix in matrices]
    for t, (matrix, embedding) in enumerate(zip(matrices, embeddings, strict=True)):
        if (embedding*reconstruction.T-matrix).applyfunc(sp.simplify) != sp.zeros(*matrix.shape):
            raise ArithmeticError("Selected product maps fail complete-tableau reconstruction")
        for u, other in enumerate(embeddings):
            expected = sp.eye(q) if t == u else sp.zeros(q)
            if (embedding.conjugate().T*other-expected).applyfunc(sp.simplify) != sp.zeros(q):
                raise ArithmeticError("Selected product maps fail complete-tableau isometry")
    assembly["validation"]["complete_tableau_reconstruction"] = True
    assembly["validation"]["complete_tableau_isometry"] = True
    assembly["target_embedding_by_tableau"] = [
        [t, i, j, str(value)] for t, embedding in enumerate(embeddings)
        for (i, j), value in sorted(embedding.todok().items()) if value]
    multiplets = [[{targets[i]: embedding[i, a] for i in range(len(targets)) if embedding[i, a]}
                   for embedding in embeddings] for a in range(q)]
    return targets, matrices, assembly, multiplets


def _role_product_routes(inputs, request):
    """Apply the shared complete Young maps and reduced angular product maps."""
    from .carrier_products import _compile_young, _hash
    from ye3t.execution_plan import YE3T_O3_REAL_TESSERAL_CONVENTION
    structural = {key: value for key, value in request.items() if key not in {"full_path", "request_hash"}}
    structural["request_hash"] = _hash(structural)
    compiled = _compile_young(json.dumps(structural, sort_keys=True))
    tableaux = compiled["output_layout"]["tableau_count"]
    L = request["target"]["rotation_L"]
    phase = (-sp.I if request["target"]["convention_id"] == YE3T_O3_REAL_TESSERAL_CONVENTION
             and (request["left"]["key"]["rotation_L"] + request["right"]["key"]["rotation_L"] - L) % 2
             else sp.S.One)
    routes = []
    for channel in compiled["channels"]:
        a, b, h = (channel[key] for key in ("left_channel", "right_channel", "young_copy"))
        rows = [{} for _ in range(tableaux)]
        scale = phase*(sp.sqrt(2) if channel["exchange"] and a != b else 1)
        for placement, mode, u, v, t, coefficient in compiled["young"]:
            if mode == h:
                _add(rows[t], _coupled(inputs[0][a][u], inputs[1][b][v], L),
                     scale*sp.sympify(coefficient))
        routes.append([{key: value for key, item in row.items()
                        if (value := sp.simplify(item)) != 0} for row in rows])
    return routes, tableaux


def _fixed_channel_multiplets(multiplets, node):
    """Apply a fixed copy map identically on every tableau, using exact entries."""
    if "subspace_request" in node:
        from .carrier_subspaces import compile_carrier_subspace, _matrix
        compiled = compile_carrier_subspace(node["subspace_request"])
        matrix = _matrix(compiled["feature_analysis_shape"], compiled["feature_analysis"])
        result = [[{} for _ in multiplets[0]] for _ in range(matrix.rows)]
        for (i, j), coefficient in matrix.todok().items():
            for row, values in zip(result[i], multiplets[j], strict=True):
                _add(row, values, coefficient)
        return [[{key: value for key, item in row.items() if (value := sp.simplify(item)) != 0}
                 for row in multiplet] for multiplet in result]
    scales = node.get("scales", [1]*len(multiplets))
    multiplets = [[{key: sp.Rational(str(scale))*value for key, value in row.items()} for row in multiplet]
                  for scale, multiplet in zip(scales, multiplets, strict=True)]
    if "copy_indices" in node:
        return [multiplets[index] for index in node["copy_indices"]]
    if node.get("matrix") is None:
        return multiplets
    result = []
    for coefficients in node["matrix"]:
        rows = [{} for _ in multiplets[0]]
        for coefficient, multiplet in zip(coefficients, multiplets, strict=True):
            for row, values in zip(rows, multiplet, strict=True):
                _add(row, values, sp.Rational(str(coefficient)))
        result.append([{key: value for key, item in row.items()
                        if (value := sp.simplify(item)) != 0} for row in rows])
    return result


def _fixed_source_coordinates(path, degree):
    """Evaluate selected fixed graph nodes in one canonical coordinate frame.

    Source and product nodes reuse the same normalized maps. The returned
    values retain complete tableaux; the channel identities are common to all
    requested nodes. Only nodes reachable from path.arguments are considered.
    This does not select a primitive/decomposable subspace.
    """
    source_sets = {}
    def sources_at(name):
        if name in source_sets:
            return source_sets[name]
        node = path["nodes"][name]
        if node.get("parameter_binding") or node.get("parameter_bindings"):
            return None
        if node["operation"] == "source":
            source_sets[name] = {name}
            return source_sets[name]
        if node["operation"] not in {"direct_sum", "fixed_carrier_product", "fixed_channel_map"} or node.get("value_scope") != "local_source":
            return None
        children = [sources_at(child) for child in node["inputs"]]
        source_sets[name] = None if any(child is None for child in children) else set().union(*children)
        return source_sets[name]

    try:
        groups = [sources_at(name) for name in path["arguments"]]
    finally:
        sources_at = None
    if any(group is None for group in groups):
        return None
    groups = [[path["nodes"][name]["source"] for name in sorted(group)] for group in groups]
    if any("vertex_binding" in source.get("physical_binding", {}) for group in groups for source in group):
        from .graph_source_images import _bound_graph_subspace_coordinates
        return _bound_graph_subspace_coordinates(path)
    if any(source.get("family") == "shared_permutation_factors" for group in groups for source in group):
        from .shared_factor_product_images import _shared_factor_coordinates
        return _shared_factor_coordinates(path)
    bindings = [source.get("physical_binding", {}) for group in groups for source in group]
    ordinary = degree == 0
    retained = bool(bindings and bindings[0].get("actual_action") == "tagged_occurrence:S" + str(degree))
    action = "ordinary_density:S0" if ordinary else ("tagged_occurrence:S" if retained else "shell_roles:S") + str(degree)
    # For one fixed retained motif, excluded and inclusive free backgrounds
    # are related by an invertible affine change with the retained values fixed.
    # Use the same normalized product image within a single binding. Unequal
    # motif/context bindings remain rejected below; their union needs its own
    # physical evaluation relations.
    contexts = {"inclusive", "motif_excluded"} if retained else {"inclusive"}
    if (not bindings or any(binding.get("actual_action") != action
            or binding.get("density_context") not in contexts for binding in bindings)
            or any(binding != bindings[0] for binding in bindings[1:])):
        return None
    sources = [{"coordinates": [coordinate for source in group for coordinate in source["coordinates"]]}
               for group in groups]
    if any((ordinary and coordinate["label"].get("tag_count") != 0)
           or (not ordinary and not retained and coordinate.get("role_binding", {}).get("shell_role_action") != "permutation")
           or (not ordinary and coordinate.get("coupling_request", {}).get("angular_basis_backend") != "normalized_cg_paths_v1")
           for source in sources for coordinate in source["coordinates"]):
        return None
    channels = sorted({_channel_identity({**channel, "physical_role": role} if retained else channel)
                       for source in sources for coordinate in source["coordinates"]
                       for channel in coordinate["channels"] for role in (range(degree+1) if retained else (0,))})
    channel_ids = {key: i for i, key in enumerate(channels, 1)}
    lowering = None
    if retained:
        from .retained_product_images import _RetainedOccurrenceImage, _retained_source_multiplets
        lowering = _RetainedOccurrenceImage([json.loads(channel) for channel in channels], degree)
    memo = {}
    def evaluate(name):
        if name in memo:
            return memo[name]
        node = path["nodes"][name]
        layout = node["layout"]
        if node["operation"] == "source":
            multiplets = ([[_source_coordinate(coordinate, channel_ids)] for coordinate in node["source"]["coordinates"]]
                          if ordinary else _retained_source_multiplets(node["source"], lowering) if retained else
                          _shell_source_multiplets(node["source"], channel_ids))
            for coordinate in node["source"]["coordinates"]:
                label, key = coordinate["label"], layout["key"]
                if (label["tag_count" if retained or ordinary else "shell_role_degree"] != key["rank"]
                        or label["tag_partition" if retained or ordinary else "shell_partition"] != key["partition"]
                        or label["target_L"] != key["rotation_L"] or label["target_parity"] != key["parity"]):
                    raise ValueError(("Retained" if retained else "Shell") + " product source symmetry differs from its input carrier")
        elif node["operation"] == "direct_sum":
            multiplets = [value for child in node["inputs"] for value in evaluate(child)]
        elif node["operation"] == "fixed_channel_map":
            multiplets = _fixed_channel_multiplets(evaluate(node["inputs"][0]), node)
        else:
            routes, tableaux = _role_product_routes([evaluate(child) for child in node["inputs"]],
                                                     node["coupling_selection"])
            if retained:
                routes = [lowering.multiplet(route) for route in routes]
            _, _, _, multiplets = _fixed_operation_multiplets(routes, tableaux)
        if len(multiplets) != layout["channel_count"] or any(
                len(row) != layout["tableau_count"] for row in multiplets):
            raise ValueError("Role source coordinates differ from the complete input carrier layout")
        memo[name] = multiplets
        return multiplets
    try:
        inputs = [evaluate(name) for name in path["arguments"]]
    finally:
        evaluate = None
    if retained:
        # Products can add exact radial/angular image coordinates. Return the
        # complete physical identity table that those coordinates reference.
        channels = [_channel_identity(channel) for channel in lowering.channels]
    return dict(kind="ordinary" if ordinary else "retained" if retained else "shell", inputs=inputs,
                channels=channels, lowering=lowering, physical_binding=bindings[0])


@lru_cache(maxsize=64)
def _fixed_role_product_image(serialized):
    """Exact product image, reconstructed on every complete tableau.

    Shared source coordinates and normalized reduced angular maps precede
    the fixed multiplicity analysis. Learned weights never enter these maps.
    """
    request = json.loads(serialized)
    if request["operation"] != "same_action_tensor_product" or request["target"].get("permutation_blocks"):
        return None
    coordinates = _fixed_source_coordinates(request["full_path"], request["target"]["rank"])
    if coordinates is None:
        return None
    if coordinates["kind"] == "shared_tree":
        from .shared_factor_product_images import shared_factor_product_image
        return shared_factor_product_image(request, _coordinates=coordinates)
    inputs, channels, lowering = (coordinates[name] for name in ("inputs", "channels", "lowering"))
    retained = coordinates["kind"] == "retained"
    routes, tableaux = _role_product_routes(inputs, request)
    if retained:
        routes = [lowering.multiplet(route) for route in routes]
        channels = [_channel_identity(channel) for channel in lowering.channels]
    targets, matrices, assembly, _ = _fixed_operation_multiplets(routes, tableaux)
    q = assembly["independent_map_count"]
    return {"schema": ("ye3t_reduced_retained_product_image_v1" if retained else "ye3t_reduced_shell_product_image_v1"),
        "target_coordinate_convention": "normalized_symmetric_block_CG_paths_v1",
        "target_basis": [{"content": [[json.loads(channels[token-1]), ell] for token, ell in word],
                          "L": L, "multiplicity_index": a} for word, L, a in targets],
        "route_count": len(routes), "independent_image_count": q,
        "independent_operation_maps": assembly, "parameter_bindings": [],
        "rank_scope": "fixed_source_multiplicity_image", "runtime_parameter_dependencies": "fixed_source_maps",
        "global_magnetic_matrix_materialized": False,
        "route_to_target_by_tableau": [[t, i, j, str(value)] for t, matrix in enumerate(matrices)
                                      for (i, j), value in sorted(matrix.todok().items()) if value]}


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
        if node.get("value_scope") != "local_source" or node["operation"] not in {
                "direct_sum", "fixed_carrier_product", "fixed_channel_map", "learned_channel_projection", "same_action_tensor_product"}:
            return False
        return all(supported(child) for child in node["inputs"])

    try:
        if not all(supported(name) for name in arguments):
            return None
    finally:
        supported = None
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
        elif node["operation"] == "direct_sum":
            rows = [row for child in node["inputs"] for row in evaluate(child)]
        elif node["operation"] == "fixed_channel_map":
            multiplets = _fixed_channel_multiplets([[row] for row in evaluate(node["inputs"][0])], node)
            rows = [multiplet[0] for multiplet in multiplets]
        elif node["operation"] == "fixed_carrier_product":
            left, right = (evaluate(child) for child in node["inputs"])
            routes = _routes(left, right, node["coupling_selection"])
            _, _, _, multiplets = _fixed_operation_multiplets([[row] for row in routes], 1)
            rows = [multiplet[0] for multiplet in multiplets]
            if len(rows) != node["layout"]["channel_count"]:
                raise ValueError("Fixed product layout differs from its independent image")
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

    try:
        left, right = (evaluate(name) for name in arguments)
    finally:
        evaluate = None
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
    result = {"schema": "ye3t_reduced_density_product_image_v2",
        "target_coordinate_convention": "normalized_symmetric_block_CG_paths_v1",
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
    if not parameters:
        assembly = _constant_operation_image(matrix)
        if assembly["independent_map_count"] != rank:
            raise ArithmeticError("Fixed operation maps disagree with the exact selected image dimension")
        result["independent_operation_maps"] = assembly
        result["rank_scope"] = "fixed_source_multiplicity_image"
        result["runtime_parameter_dependencies"] = "fixed_source_maps"
    else:
        # Full column rank at an exact parameter witness establishes that no
        # canonical carrier map is annihilated identically by this realization.
        # The maps themselves remain the fixed complete-child couplings; their
        # learned values may lose rank at special weights (V5 sections 6/9).
        result["operation_map_status"] = (
            "canonical_carrier_maps_with_generic_source_injectivity" if rank == len(routes)
            else "parameter_dependent_image_requires_assembled_input_planning")
    return json.loads(json.dumps(result))


def reduced_density_product_image(request):
    if request.get("full_path") is None or request["operation"] != "same_action_tensor_product":
        return None
    path = request["full_path"]
    arguments = [path["nodes"][name] for name in path["arguments"]]
    if request["target"]["rank"]:
        # Match the fixed-shell planner's existing root checks before encoding
        # a potentially large construction graph into a retained LRU key.
        # Transported learned carriers use their actual complete-child maps.
        if request["target"].get("permutation_blocks") or any(
                node.get("parameter_binding") or node.get("parameter_bindings")
                or (node["operation"] != "source" and (
                    node["operation"] not in {"direct_sum", "fixed_carrier_product", "fixed_channel_map"}
                    or node.get("value_scope") != "local_source")) for node in arguments):
            return None
        return copy.deepcopy(_fixed_role_product_image(json.dumps(request, sort_keys=True)))
    if any(node.get("value_scope") == "propagated_hidden_state" for node in arguments):
        return None
    return copy.deepcopy(_plan(json.dumps(request, sort_keys=True)))
