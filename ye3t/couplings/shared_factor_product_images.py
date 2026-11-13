"""Exact fixed shared-factor products on retained edges and pooled root moments.

All symmetry maps come from the shared constructors. Physical coordinates use
one Racah harmonic and one ChebExpCos polynomial per edge, plus the existing
normalized fixed-content coordinates for root moments. This module binds those
existing algebras; it constructs no alternative Young/CG coefficients.
"""
from functools import lru_cache
from itertools import product
import json
import math

import sympy as sp

from ye3t.exact_scalars import exact_scalar
from ye3t.execution_plan import YE3T_O3_REAL_TESSERAL_CONVENTION
from ye3t.representations.young_angular_paths import _coupled_path_columns, _permuted_chain_column
from .chebexp_cos import chebexp_cos_radial_product, _chebexp_cos_radial_identity
from .shared_permutation_factors import _shared_factor_path_columns
from .retained_product_images import _same_occurrence_angular_image


@lru_cache(maxsize=1024)
def _coupled_product_path(left_path, right_spins, right_path, L):
    return tuple(_coupled_path_columns(((left_path, exact_scalar(1)),),
        ((right_path, exact_scalar(1)),), right_spins, L).items())


def _moment_identity(factor):
    if (set(factor) != {"content", "center_type", "neighbor_type", "n", "l", "cutoff", "radial_lambda"}
            or any(type(factor[key]) is not int or factor[key] < 0 for key in ("center_type", "neighbor_type", "l"))):
        raise ValueError("Root moments require complete physical radial/angular/species assignments")
    return (factor["center_type"], factor["neighbor_type"], factor["l"],
            *_chebexp_cos_radial_identity(factor["n"], factor["cutoff"], factor["radial_lambda"]))


def _moment_path_image(radial, spins, tokens, column, edge_count):
    """Use harmonic products on edges and fixed-content maps on pooled moments."""
    from .density_product_images import _normalized_sector_path_columns
    # Pooled factors remain separate until the normalized symmetric map. Their
    # temporary occurrence IDs must not trigger same-direction harmonic rules.
    occurrences = tuple(token if token < edge_count else edge_count+i for i, token in enumerate(tokens))
    moments = tuple(token for token in tokens if token >= edge_count)
    result = {}
    for (degrees, path), value in _same_occurrence_angular_image(spins, occurrences, column).items():
        word = tuple(zip(tuple(range(edge_count))+moments, degrees, strict=True))
        order = tuple(sorted(range(len(word)), key=lambda i: word[i]))
        word = tuple(word[i] for i in order)
        projected = _permuted_chain_column({path: value}, degrees, order)
        for copy, target in enumerate(_normalized_sector_path_columns(word, path[-1])):
            # Columns are real normalized maps; the physical real-form phase
            # can make their complex-harmonic coordinates imaginary.
            coefficient = sum((coefficient._sympy_()*projected[path]
                               for path, coefficient in target if path in projected), sp.S.Zero)
            if coefficient:
                key = radial, word, path[-1], copy
                result[key] = result.get(key, 0)+coefficient
    return {key: value for key, item in result.items() if (value := sp.simplify(item)) != 0}


def _source_multiplets(source, layout, moment_channels=None):
    report = _shared_factor_path_columns(source["coupling_request"])
    if report["layout"] != layout or report["labels"] != source["coordinates"]:
        raise ValueError("Shared product source differs from its canonical selected coordinates")
    binding, channels = source["physical_binding"], source["channels"]
    profiles = binding["radial_profiles"]
    n = len(report["input_Ls"])
    edge_count = len(profiles)
    backgrounds = binding.get("background_factors", [])
    if len(channels) != n or edge_count+len(backgrounds) != n:
        raise ValueError("Shared edge product requires every physical channel and radial profile")
    edge_positions = tuple(i for i, channel in enumerate(channels) if type(channel["occurrence_id"]) is int)
    background_positions = tuple(i for i in range(n) if i not in edge_positions)
    identities = tuple(channels[i]["occurrence_id"] for i in edge_positions)
    if sorted(identities) != list(range(edge_count)):
        raise ValueError("A retained source must use one factor per physical edge")
    if sorted(i for group in report["request"].get("fixed_factor_groups", []) for i in group) != list(background_positions):
        raise ValueError("Fixed factor groups must match the physical root-moment positions")
    moment_tokens, repeated = {}, {}
    for i, factor in zip(background_positions, backgrounds, strict=True):
        if (factor["content"] != report["request"]["content"][i]
                or (factor["n"], factor["l"]) != (channels[i]["n"], channels[i]["l"])):
            raise ValueError("Root-moment assignments disagree with the selected source")
        identity = _moment_identity(factor)
        moment_tokens[i] = moment_channels[identity]
        repeated.setdefault(identity, []).append(i)
    if list(repeated.values()) != report["request"].get("fixed_factor_groups", []):
        raise ValueError("Fixed groups must resolve repeated physical root-moment functions")
    normalization = source["factor_normalization"]
    if normalization not in {"bounded", "none"}:
        raise ValueError("Unsupported shared edge factor normalization")
    for i, channel in enumerate(channels):
        if (type(channel["l"]) is not int or type(channel["n"]) is not int
                or channel["l"] != report["input_Ls"][i] or channel["n"] < 1):
            raise ValueError("Shared physical angular/radial assignments disagree with the source")
    for profile in profiles:
        if (profile.get("family") != "chebexp_cos" or set(profile) != {"family", "cutoff", "lambda"}
                or any(type(profile[key]) not in {int, float} or not math.isfinite(profile[key])
                       for key in ("cutoff", "lambda")) or profile["cutoff"] <= 0 or profile["lambda"] == 0):
            raise ValueError("Exact ChebExpCos products need positive cutoffs and finite nonzero real warp parameters")
    for group in report["request"]["factor_groups"]:
        assignments = [(channels[i]["n"], channels[i]["l"], profiles[channels[i]["occurrence_id"]]) for i in group]
        if any(assignment != assignments[0] for assignment in assignments[1:]):
            raise ValueError("Shared factor groups must preserve complete physical function assignments")
    order = tuple(edge_positions[identities.index(i)] for i in range(edge_count))+background_positions
    spins = tuple(report["input_Ls"][i] for i in order)
    radial = [chebexp_cos_radial_product((channels[i]["n"],)) for i in order[:edge_count]]
    scale = (-sp.I)**report["real_phase_exponent"]
    if normalization == "none":
        scale *= sp.sqrt(sp.prod(2*ell+1 for ell in spins)/(4*sp.pi)**n)
    result = []
    for copy in report["columns"]:
        rows = []
        for column in copy:
            paths = _permuted_chain_column(column, report["input_Ls"], order)
            row = {}
            for combination in product(*radial):
                powers = tuple((h, p) for h, p, _ in combination)
                weight = scale*sp.prod(sp.Rational(c.numerator, c.denominator) for _, _, c in combination)
                if moment_channels is None:
                    for path, coefficient in paths.items():
                        row[powers, spins, path] = weight*coefficient._sympy_()
                else:
                    tokens = tuple(range(edge_count))+tuple(moment_tokens[i] for i in background_positions)
                    lowered = _moment_path_image(powers, spins, tokens, paths, edge_count)
                    for key, coefficient in lowered.items():
                        row[key] = row.get(key, 0)+weight*coefficient
            rows.append(row)
        result.append(rows)
    return result


def _product_rows(inputs, request, moment_channels=None, edge_count=None):
    from .carrier_products import _compile_young, _hash
    structural = {key: value for key, value in request.items() if key not in {"full_path", "request_hash"}}
    structural["request_hash"] = _hash(structural)
    compiled = _compile_young(json.dumps(structural, sort_keys=True))
    L = request["target"]["rotation_L"]
    real = request["target"]["convention_id"] == YE3T_O3_REAL_TESSERAL_CONVENTION
    phase = (-sp.I if real and (request["left"]["key"]["rotation_L"]+
        request["right"]["key"]["rotation_L"]-L) % 2 else sp.S.One)
    routes = []
    for channel in compiled["channels"]:
        rows = [{} for _ in range(compiled["output_layout"]["tableau_count"])]
        assembled = [{} for _ in rows]
        a, b = channel["left_channel"], channel["right_channel"]
        scale = phase*(sp.sqrt(2) if channel["exchange"] and a != b else 1)
        for placement, mode, u, v, t, coefficient in compiled["young"]:
            if mode != channel["young_copy"]:
                continue
            if placement:
                raise ValueError("Shared edge products require diagonal action alignment")
            for key_a, x in inputs[0][a][u].items():
                for key_b, y in inputs[1][b][v].items():
                    if moment_channels is None:
                        radial_a, spins_a, path_a = key_a
                        radial_b, spins_b, path_b = key_b
                        angular = _coupled_product_path(path_a, spins_b, path_b, L)
                        tokens = None
                    else:
                        from .density_product_images import _normalized_sector_path_columns
                        radial_a, word_a, left_L, copy_a = key_a
                        radial_b, word_b, right_L, copy_b = key_b
                        spins_a, spins_b = tuple(ell for _, ell in word_a), tuple(ell for _, ell in word_b)
                        tokens = tuple(token for token, _ in word_a+word_b)
                        angular = _coupled_path_columns(_normalized_sector_path_columns(word_a, left_L)[copy_a],
                            _normalized_sector_path_columns(word_b, right_L)[copy_b], spins_b, L).items()
                    radial = tuple((ha+hb, pa+pb) for (ha, pa), (hb, pb) in zip(radial_a, radial_b, strict=True))
                    column = assembled[t].setdefault((radial, spins_a+spins_b, tokens), {})
                    weight = scale*sp.sympify(coefficient)*x*y
                    for path, factor in angular:
                        column[path] = column.get(path, 0)+weight*factor._sympy_()
        # Permutation and same-occurrence harmonic reduction are linear. Sum
        # compatible contributions first so cancellations happen before the
        # costly map application. Radial coordinates and full tableaux stay
        # separate. Symbolic scalars preserve physical normalization and phases.
        for t, groups in enumerate(assembled):
            for (radial, spins, tokens), column in groups.items():
                column = {path: value for path, item in column.items() if (value := sp.simplify(item)) != 0}
                if moment_channels is None:
                    identities = tuple(range(len(spins)//2))*2
                    image = {(radial, degrees, path): value for (degrees, path), value in
                             _same_occurrence_angular_image(spins, identities, column).items()}
                else:
                    image = _moment_path_image(radial, spins, tokens, column, edge_count)
                for key, value in image.items():
                    rows[t][key] = rows[t].get(key, 0)+value
        routes.append([{key: value for key, item in row.items() if (value := sp.simplify(item)) != 0} for row in rows])
    return routes, compiled["output_layout"]["tableau_count"]


def _shared_factor_coordinates(path, *, _route_indices=None):
    """Evaluate requested source/product nodes with shared physical identities."""
    from .density_product_images import _fixed_channel_multiplets, _fixed_operation_multiplets
    nodes = path["nodes"]
    live, pending = set(), list(path["arguments"])
    while pending:
        name = pending.pop()
        if name not in live:
            live.add(name)
            pending.extend(nodes[name].get("inputs", []))
    sources = [nodes[name]["source"] for name in sorted(live) if nodes[name]["operation"] == "source"]
    if not sources or not any(source.get("family") == "shared_permutation_factors" for source in sources):
        return None
    if any(source.get("family") != "shared_permutation_factors" for source in sources):
        raise NotImplementedError("Mixed shared-edge and density product images require their joint physical binding")
    bindings = [source["physical_binding"] for source in sources]
    common = ("actual_action", "domain", "radial_profiles")
    if any(any(binding.get(key) != bindings[0].get(key) for key in common) for binding in bindings[1:]):
        raise NotImplementedError("Shared edge products require matching physical domains and radial profiles; cross-profile images remain unresolved")
    if not bindings[0].get("actual_action") or any(binding.get("kind") not in {
            "independent_tree_edges", "tree_edges_and_root_moments"} for binding in bindings):
        raise ValueError("Shared product images need the actual independent-edge physical binding")
    backgrounds = [binding for binding in bindings if binding["kind"] == "tree_edges_and_root_moments"]
    moment_channels, moment_records = None, []
    edge_count = len(bindings[0]["radial_profiles"])
    if backgrounds:
        contexts = {binding.get("density_context") for binding in backgrounds}
        roots = {binding.get("root_vertex") for binding in backgrounds}
        if (len(contexts) != 1 or not contexts <= {"inclusive", "motif_excluded"}
                or len(roots) != 1 or any(type(root) is not int or root < 0 for root in roots)):
            raise NotImplementedError("Joint root-moment images require one physical root and density context; mixed-context images need their explicit affine maps")
        identities = sorted({_moment_identity(factor) for binding in backgrounds for factor in binding["background_factors"]})
        moment_channels = {identity: edge_count+i for i, identity in enumerate(identities)}
        moment_records = [dict(token=moment_channels[identity], **dict(zip(
            ("center_type", "neighbor_type", "l", "n", "cutoff", "radial_lambda"), identity, strict=True)))
            for identity in identities]
    memo = {}
    def evaluate(name):
        if name in memo:
            return memo[name]
        node = nodes[name]
        if node.get("parameter_binding") or node.get("parameter_bindings"):
            raise ValueError("Fixed shared-edge product images must precede learned mixing")
        operation = node["operation"]
        if operation == "source":
            result = _source_multiplets(node["source"], node["layout"], moment_channels)
        elif operation == "direct_sum":
            result = [row for child in node["inputs"] for row in evaluate(child)]
        elif operation == "fixed_channel_map":
            result = _fixed_channel_multiplets(evaluate(node["inputs"][0]), node)
        elif operation == "fixed_carrier_product":
            routes, tableaux = _product_rows([evaluate(child) for child in node["inputs"]],
                node["coupling_selection"], moment_channels, edge_count)
            if _route_indices is not None and name in _route_indices:
                # Bound images retain original independent routes. Preserve
                # that gauge when inheriting this established target metric.
                result = [routes[index] for index in _route_indices[name]]
            else:
                _, _, _, result = _fixed_operation_multiplets(routes, tableaux)
        else:
            raise ValueError("Unsupported fixed shared-edge operation: "+operation)
        layout = node["layout"]
        if len(result) != layout["channel_count"] or any(len(row) != layout["tableau_count"] for row in result):
            raise ValueError("Shared physical product image differs from its complete carrier layout")
        memo[name] = result
        return result
    try:
        inputs = [evaluate(name) for name in path["arguments"]]
    finally:
        evaluate = None
    return dict(kind="shared_tree", inputs=inputs, moment_channels=moment_channels,
        moment_records=moment_records, edge_count=edge_count, physical_binding=bindings[0],
        **({"density_context": next(iter(contexts)), "root_vertex": next(iter(roots))} if backgrounds else {}))


def shared_factor_product_image(request, *, _coordinates=None):
    """Resolve fixed source/product coordinates before learned channel mixing."""
    from .density_product_images import _fixed_operation_multiplets
    coordinates = _shared_factor_coordinates(request["full_path"]) if _coordinates is None else _coordinates
    if coordinates is None:
        return None
    inputs, moment_channels, moment_records, edge_count = (coordinates[name] for name in
        ("inputs", "moment_channels", "moment_records", "edge_count"))
    routes, tableaux = _product_rows(inputs, request, moment_channels, edge_count)
    targets, matrices, assembly, _ = _fixed_operation_multiplets(routes, tableaux)
    convention = ("independent_edge_chebexp_polynomials_and_normalized_angular_paths_v1" if moment_channels is None else
                  "edge_chebexp_and_root_moment_fixed_content_v1")
    assembly["target_coordinate_convention"] = convention
    target_basis = ([dict(
        radial_powers=[list(powers) for powers in radial], angular_degrees=list(spins),
        angular_path=list(chain)) for radial, spins, chain in targets] if moment_channels is None else
        [dict(radial_powers=[list(powers) for powers in radial], channel_word=[list(item) for item in word],
              target_L=L, multiplicity_index=copy) for radial, word, L, copy in targets])
    return dict(schema="ye3t_shared_edge_product_image_v1", target_basis=target_basis,
        **({"moment_channels": moment_records, "edge_count": edge_count,
            "density_context": coordinates["density_context"], "root_vertex": coordinates["root_vertex"]}
           if moment_channels is not None else {}),
        target_coordinate_convention=convention,
        route_count=len(routes), independent_image_count=assembly["independent_map_count"],
        independent_operation_maps=assembly, parameter_bindings=[], rank_scope="fixed_source_multiplicity_image",
        runtime_parameter_dependencies="fixed_source_maps", global_magnetic_matrix_materialized=False,
        route_to_target_by_tableau=[[t, i, j, str(value)] for t, matrix in enumerate(matrices)
                                   for (i, j), value in sorted(matrix.todok().items()) if value])
