"""Selected physical target subspaces using existing reduced product maps.

The supplied supplement, sec:supp_primitive_quotient, defines the image of
admitted proper products and its metric-orthogonal complement. This module
connects the existing physical-coordinate and complement implementations;
it does not construct Young/CG coefficients or expand magnetic tensor states.
"""
import copy
from functools import lru_cache
import json

import sympy as sp

from ye3t.execution_plan import YE3TCarrierLayout
from ye3t.representations.young_sectors import _orthonormal_primitive_copy_coefficients
from .carrier_products import _construction_full_path, _hash
from .density_product_images import _fixed_source_coordinates


def is_carrier_subspace_request(value):
    return isinstance(value, dict) and value.get("family") == "carrier_subspace"


def _request(value):
    raw = value.get("request", value)
    if not isinstance(raw, dict) or raw.get("family") != "carrier_subspace":
        raise ValueError("Expected a canonical carrier_subspace request")
    if set(raw)-{"family", "full_path", "subspace"}:
        raise ValueError("Unknown carrier_subspace fields: " + str(sorted(set(raw)-{
            "family", "full_path", "subspace"})))
    selection = raw.get("subspace", "full")
    if selection not in {"full", "primitive", "decomposable", "split"}:
        raise ValueError("subspace must be full, primitive, decomposable or split")
    path = raw.get("full_path")
    if not isinstance(path, dict) or not path.get("arguments") or "nodes" not in path:
        raise ValueError("Subspace full_path.arguments must name the target followed by admitted product nodes")
    try:
        layouts = [YE3TCarrierLayout.from_dict(path["nodes"][name]["layout"]) for name in path["arguments"]]
    except (KeyError, TypeError) as error:
        raise ValueError("Subspace arguments need existing nodes with complete carrier layouts") from error
    target = layouts[0]
    if any((layout.key, layout.tableau_count, layout.magnetic_count) !=
           (target.key, target.tableau_count, target.magnetic_count) for layout in layouts[1:]):
        raise ValueError("Admitted products must have the selected target's actual carrier and convention")
    path = _construction_full_path(path, layouts)
    ranks = {}

    def rank(name):
        if name in ranks:
            return ranks[name]
        node = path["nodes"][name]
        if node.get("parameter_binding") or node.get("parameter_bindings"):
            raise ValueError("Primitive/decomposable selection requires fixed inputs before learned mixing")
        if node["operation"] == "source":
            source = node["source"]
            if source.get("family") == "shared_permutation_factors":
                from .shared_permutation_factors import shared_permutation_count
                values = {shared_permutation_count(source["coupling_request"])["factor_rank"]}
            else:
                if not source.get("coordinates"):
                    raise ValueError("Source lacks the canonical independent physical coordinate inventory required for subspace selection")
                for coordinate in source["coordinates"]:
                    label = coordinate["label"]
                    if (type(label["rank"]) is not int or
                            label["rank"] != sum(label["block_sizes"])):
                        raise ValueError("Source formal factor rank differs from its complete-channel blocks")
                values = {coordinate["label"]["rank"] for coordinate in source["coordinates"]}
        elif node["operation"] == "fixed_carrier_product":
            values = {sum(rank(child) for child in node["inputs"])}
        elif node["operation"] in {"direct_sum", "fixed_channel_map"}:
            values = {rank(child) for child in node["inputs"]}
        else:
            raise ValueError("Subspace selection requires fixed source/product coordinates")
        if len(values) != 1 or next(iter(values)) <= 0:
            raise ValueError("Select a homogeneous positive factor-rank target before mixing factor grades")
        ranks[name] = next(iter(values))
        return ranks[name]

    def proper(name, target_rank):
        node = path["nodes"][name]
        if rank(name) != target_rank:
            raise ValueError("Admitted product and target must have the same formal factor rank")
        if node["operation"] == "fixed_carrier_product":
            if not all(0 < rank(child) < target_rank for child in node["inputs"]):
                raise ValueError("Admitted factorization must have two proper lower-rank children")
        elif node["operation"] in {"direct_sum", "fixed_channel_map"}:
            for child in node["inputs"]:
                proper(child, target_rank)
        else:
            raise ValueError("Admitted factorization routes must originate in fixed proper products")

    try:
        target_rank = rank(path["arguments"][0])
        for name in path["arguments"][1:]:
            proper(name, target_rank)
    finally:
        rank = proper = None
    return dict(family="carrier_subspace", full_path=path, subspace=selection), target_rank


def _entries(matrix):
    return [[int(i), int(j), str(sp.simplify(value))] for (i, j), value in sorted(matrix.todok().items()) if value]


def _matrix(shape, entries):
    return sp.SparseMatrix(*shape, {(i, j): sp.sympify(value) for i, j, value in entries})


@lru_cache(maxsize=8)
def _image(serialized):
    path = json.loads(serialized)
    layout = YE3TCarrierLayout.from_dict(path["nodes"][path["arguments"][0]]["layout"])
    coordinates = _fixed_source_coordinates(path, layout.key.rank)
    if coordinates is None:
        raise ValueError("Selected target/product physical coordinates are unresolved; align supported fixed bindings first")
    target, *products = coordinates["inputs"]
    routes = [copy for copies in products for copy in copies]
    keys = sorted({key for copies in (target, routes) for multiplet in copies for row in multiplet for key in row})
    indices = {key: index for index, key in enumerate(keys)}
    # Sparse reduced coordinates only. The metric and solves have target-copy
    # dimensions; no uncoupled magnetic tensor or numerical rank test is used.
    component_count = len(target[0]) if target else layout.tableau_count
    targets = [sp.SparseMatrix(len(keys), len(target), {
        (indices[key], q): value for q, multiplet in enumerate(target) for key, value in multiplet[t].items()})
        for t in range(component_count)]
    inherited = "target_metric" in coordinates
    gram = (coordinates["target_metric"] if inherited else
            (targets[0].H*targets[0]).applyfunc(sp.simplify))
    route_matrices = [sp.SparseMatrix(len(keys), len(routes), {
        (indices[key], q): value for q, multiplet in enumerate(routes) for key, value in multiplet[t].items()})
        for t in range(component_count)]
    try:
        # Cholesky also rejects dependent or zero target coordinates, including
        # the no-route case. This is the existing target multiplicity metric.
        gram.cholesky()
        if inherited:
            # Equality rows have no implied Euclidean metric. Select exact
            # independent equations and solve only in target-copy coordinates.
            _, independent_rows = targets[0].T.rref(simplify=False)
            if len(independent_rows) != len(target):
                raise ValueError("Bound target coordinates are physically dependent")
            square = targets[0].extract(independent_rows, range(len(target)))
            image = square.LUsolve(route_matrices[0].extract(independent_rows, range(len(routes)))) if routes else sp.zeros(len(target), 0)
        else:
            image = gram.LUsolve(targets[0].H*route_matrices[0]) if routes else sp.zeros(len(target), 0)
    except (ValueError, ZeroDivisionError, sp.matrices.exceptions.NonInvertibleMatrixError,
            sp.matrices.exceptions.NonPositiveDefiniteMatrixError) as error:
        raise ValueError("Selected target coordinates must be independent and have a positive invariant metric") from error
    image = image.applyfunc(sp.simplify)
    for t, (target_matrix, product_matrix) in enumerate(zip(targets, route_matrices, strict=True)):
        if (target_matrix*image-product_matrix).applyfunc(sp.simplify).is_zero_matrix is not True:
            raise ValueError("Admitted product leaves the selected target; retain matching target copies or restrict its routes")
        if not inherited:
            for u, other in enumerate(targets):
                expected = gram if t == u else sp.zeros(len(target))
                if (target_matrix.H*other-expected).applyfunc(sp.simplify).is_zero_matrix is not True:
                    raise ArithmeticError("Target reduced coordinates do not preserve the complete carrier metric")
    if inherited and not coordinates.get("complete_target_metric_verified"):
        raise ArithmeticError("Bound target metric requires its canonical complete-carrier validation")
    pivots = tuple(int(index) for index in image.rref(simplify=False)[1])
    return dict(target_layout=layout.to_dict(), source_kind=coordinates["kind"],
        full_count=len(target), decomposable_count=len(pivots), primitive_count=len(target)-len(pivots),
        route_count=len(routes), independent_route_indices=list(pivots),
        target_metric=_entries(gram), product_coordinates=_entries(image),
        metric_convention=coordinates.get("metric_convention", "canonical_reduced_physical_coordinates"),
        validation=dict(exact_membership=True, complete_carrier_metric=True,
                        **({"physical_coordinate_metric_used": False} if inherited else {}),
                        numerical_rank_used=False, magnetic_tensor_expanded=False))


def carrier_subspace_plan(value):
    """Count independent selected target copies and their admitted product image.

    full_path.arguments names the target first, then fixed product routes.
    Parent/intermediate restrictions are those of these canonical graph nodes.
    Existing reduced maps resolve physical membership before selection. Runtime
    subspace coefficients are materialized only by compile. Full keeps the
    original target basis; split orders normalized primitive then decomposable.
    """
    request, factor_rank = _request(value)
    image = _image(json.dumps(request["full_path"], sort_keys=True, allow_nan=False))
    result = copy.deepcopy(image)
    selection = request["subspace"]
    selected = image["full_count"] if selection in {"full", "split"} else image[selection+"_count"]
    result.update(family="carrier_subspace", schema="ye3t_carrier_subspace_plan_v1", request=request,
        output_channel_count=selected, target_factor_rank=factor_rank,
        component_count=selected*image["target_layout"]["tableau_count"]*image["target_layout"]["magnetic_count"],
        coefficient_materialization="not_requested",
        provenance=dict(api="ye3t.couplings.plan", compiler_owner="ye3t",
                        product_coordinates="shared_fixed_source_coordinates",
                        complement="orthonormal_primitive_copy_coefficients"))
    result["admitted_products"] = [dict(node=name,
        copy_count=request["full_path"]["nodes"][name]["layout"]["channel_count"])
        for name in request["full_path"]["arguments"][1:]]
    return result


@lru_cache(maxsize=8)
def _compiled_maps(serialized):
    # Retain only the compact multiplicity data; source graphs stay in the
    # existing image cache. All selections reuse the same normalized split.
    result = json.loads(serialized)
    count = result["full_count"]
    gram = _matrix((count, count), result["target_metric"])
    image = _matrix((count, result["route_count"]), result["product_coordinates"])
    primitive, _ = _orthonormal_primitive_copy_coefficients(image, gram)
    projector = (primitive*primitive.H*gram).applyfunc(sp.simplify)
    # The projector has the same primitive image. Remove column-normalization
    # radicals before the second exact complement solve; feeding normalized
    # vectors back into elimination can cause enormous symbolic expressions.
    # The shared constructor's coordinate seeds and normalized output gauge
    # depend only on this image, so its established maps are preserved.
    decomposable, _ = _orthonormal_primitive_copy_coefficients(projector, gram)
    result = {}
    for name, embedding, selected_projector in (
            ("primitive", primitive, projector),
            ("decomposable", decomposable, sp.eye(count)-projector),
            ("split", primitive.row_join(decomposable), sp.eye(count))):
        result[name] = dict(coefficient_embedding=_entries(embedding),
            feature_analysis=_entries(embedding.T),
            feature_analysis_shape=[embedding.cols, embedding.rows],
            coefficient_projector=_entries(selected_projector))
    return result


def compile_carrier_subspace(value):
    """Materialize compact copy maps, preserving every tableau and M component."""
    result = carrier_subspace_plan(value)
    count = result["full_count"]
    selection = result["request"]["subspace"]
    if selection == "full":
        identity = _entries(sp.eye(count))
        maps = dict(coefficient_embedding=identity, feature_analysis=copy.deepcopy(identity),
                    feature_analysis_shape=[count, count], coefficient_projector=copy.deepcopy(identity))
    else:
        data = {key: result[key] for key in ("full_count", "route_count", "target_metric", "product_coordinates")}
        maps = copy.deepcopy(_compiled_maps(json.dumps(data, sort_keys=True, allow_nan=False))[selection])
    if maps["feature_analysis_shape"][0] != result["output_channel_count"]:
        raise ArithmeticError("Compiled subspace dimension differs from the independent count")
    result.update(schema="ye3t_carrier_subspace_compiled_v1", coefficient_materialization="selected_copy_maps",
                  **maps)
    result["self_hash"] = _hash(result)
    return result
