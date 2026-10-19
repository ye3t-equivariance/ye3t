"""Analytical selected-factor restriction of one vector and repeated scalars.

The permutation module Ind(S_(N-1) x S_1, 1) is decomposed by the
established Young-orthogonal subduction engine. Pooled-factor invariants and
the two selected-factor characters are extracted using exact projectors.
Reference: October 12 planner/compiler mathematical handoff, SR12--SR12a.
"""

import hashlib
import json
from functools import lru_cache
from itertools import combinations_with_replacement, product


KIND = "selected_scalar_vector_marginal"
TYPE_NAMES = ("even_scalar", "odd_scalar", "even_polar_vector", "odd_polar_vector")


def is_selected_scalar_vector_request(request):
    return isinstance(request, dict) and request.get("kind") == KIND


def _request(request):
    request = dict(request.get("request", request))
    unknown = set(request) - {"kind", "rank", "hidden_width"}
    if unknown or request.get("kind") != KIND:
        raise ValueError("Unsupported selected scalar/vector request: " + str(unknown))
    for name, default in (("rank", 4), ("hidden_width", 2)):
        value = request.get(name, default)
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(name + " must be an integer")
    rank, width = int(request.get("rank", 4)), int(request.get("hidden_width", 2))
    if rank != 4 or width < 1:
        raise ValueError("This source specialization implements rank 4 with positive hidden width")
    return {"kind": KIND, "rank": rank, "hidden_width": width}


def selected_scalar_vector_count(request):
    from ye3t.fixed_content import FixedContentModule

    request = _request(request)
    rank = request["rank"]
    decomposition = FixedContentModule(content=(1,) * (rank - 1) + (2,),
        input_Ls=(0,) * (rank - 1) + (1,)).decompose(max_target_L=1)
    records = []
    for parent in ((rank,), (rank - 1, 1)):
        multiplicity = decomposition.sector_multiplicity(parent, 1)
        if multiplicity != 1:
            raise ArithmeticError("One-vector fixed-content multiplicity must be one")
        from ye3t.representations.young_subgroup_specht_coupling import young_subgroup_specht_coupling_multiplicity
        for tau in ((2,), (1, 1)):
            restriction_multiplicity = young_subgroup_specht_coupling_multiplicity(((rank - 2,), tau), parent)
            if not restriction_multiplicity:
                continue
            if restriction_multiplicity != 1:
                raise ArithmeticError("Selected restriction must have multiplicity one")
            records.append({"parent_partition": list(parent), "actual_partition": list(tau),
                            "id": "parent_" + "".join(map(str, parent)) + ("_even" if tau == (2,) else "_odd"),
                            "L": 1, "parity": -1, "multiplicity": multiplicity,
                            "parent_multiplicity": multiplicity, "restriction_multiplicity": restriction_multiplicity,
                            "parent_copy": 0, "restriction_copy": 0})
    return {"kind": KIND, "request": request, "source_types": records,
            "source_copy_count": len(records), "selected_dimension": 3 * len(records),
            "full_parent_dimension": rank * 3, "actual_action": "shared_auxiliary_S2_times_O3"}


def selected_scalar_vector_plan(request):
    report = selected_scalar_vector_count(request)
    result = {**report, "schema": "ye3t_selected_scalar_vector_plan_v1",
        "backend": "exact_young_subduction_and_character_projectors",
        "analysis_orientation": "B=C_adjoint_Z", "normalization_id": "full_content_scaled_marginal_v1",
        "moment_binding": ["A^2*v_anchor*(A-q_anchor)", "A^2*q_anchor*(V-v_anchor)",
                           "sqrt(2)*A*V*q_anchor*(A-q_anchor)"],
        "occurrence_requirements": {"pooled": "inclusive", "explicit_pair": "distinct_complete_identities",
                                    "marginal_weight": 1, "marginal_scale": "sqrt(2)/2"},
        "product_families": ["scalar_scalar", "vector_dot_vector", "scalar_vector"],
        "product_support_relation": "same_action_pointwise",
        "product_basis": "normalized_S2_characters_and_orthonormal_Cartesian",
        "output_order": [row["id"] for row in report["source_types"]]}
    result["plan_hash"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return result


def selected_scalar_vector_product_table(compiled):
    """Lower canonical Cartesian product routes to the native weighted table.

    Channels are packed in compiler type order. Weights have route-major,
    output-channel-minor order and remain supplied by the application.
    """
    import torch
    from ye3t.backends.triton_joint import WeightedSparseBilinearTable, _validate_weighted

    expected = compile_selected_scalar_vector(compiled["request"])
    if compiled != expected:
        raise ValueError("Product table requires unchanged canonical compiler maps")
    width = int(compiled["request"]["hidden_width"])
    dimensions = [2 * row["L"] + 1 for row in compiled["hidden_types"]]
    offsets, cursor = [], 0
    for dimension in dimensions:
        offsets.append(cursor)
        cursor += width * dimension
    left, right, output, weight, coefficient = [], [], [], [], []
    for route_index, route in enumerate(compiled["products"]):
        lt, lc = route["left"]
        rt, rc = route["right"]
        target = route["output_type"]
        required = {"scalar_scalar": (1, 1, 1), "vector_dot_vector": (3, 3, 1), "scalar_vector": (1, 3, 3)}
        if required.get(route["operation"]) != (dimensions[lt], dimensions[rt], dimensions[target]):
            raise ValueError("Native Cartesian lowering requires a supported scalar/vector product")
        for channel in range(width):
            for m in range(max(dimensions[lt], dimensions[rt])):
                left.append(offsets[lt] + lc * dimensions[lt] + (m if dimensions[lt] > 1 else 0))
                right.append(offsets[rt] + rc * dimensions[rt] + (m if dimensions[rt] > 1 else 0))
                output.append(offsets[target] + channel * dimensions[target] + (m if dimensions[target] > 1 else 0))
                weight.append(route_index * width + channel)
                coefficient.append(route["role_coefficient"])
    table = WeightedSparseBilinearTable(
        left_index=torch.tensor(left, dtype=torch.long),
        right_index=torch.tensor(right, dtype=torch.long),
        output_index=torch.tensor(output, dtype=torch.long),
        weight_index=torch.tensor(weight, dtype=torch.long),
        coefficient=torch.tensor(coefficient, dtype=torch.float64),
        output_width=cursor, weight_count=width * len(compiled["products"]))
    dummy = torch.zeros((1, cursor), dtype=torch.float64)
    _validate_weighted(dummy, dummy, torch.zeros(table.weight_count, dtype=torch.float64), table)
    return table


@lru_cache(maxsize=32)
def _fixed_maps(rank):
    import sympy as sp
    from ye3t.representations.young_subgroup_specht_coupling import build_young_subgroup_specht_coupling

    pooled = rank - 2
    placement = sp.zeros(rank, 3)
    placement[rank - 2, 0] = placement[rank - 1, 1] = 1
    for f in range(pooled):
        placement[f, 2] = 1 / sp.sqrt(pooled)
    swap = sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    rows, provenance = [], []
    for parent in ((rank,), (rank - 1, 1)):
        coupling = build_young_subgroup_specht_coupling(((rank - 1,), (1,)), parent,
            coefficient_backend="subduction_graph")
        if not coupling.validation.passed or coupling.multiplicity != 1:
            raise ArithmeticError("Exact parent subduction failed")
        injection = sp.zeros(rank, coupling.induced_dim)
        for column, entry in enumerate(coupling.tensor.induced_basis):
            injection[int(entry.coset_rep[-1]), column] = 1
        if injection.T * injection != sp.eye(rank):
            raise ArithmeticError("Content placement must preserve the induced counting metric")
        canonical = sp.simplify(injection * coupling.coefficient_matrix())
        selected = sp.simplify(placement.T * canonical * canonical.T * placement)
        for sign in ((1,) if len(parent) == 1 else (1, -1)):
            projector = sp.simplify(selected * (sp.eye(3) + sign * swap) / 2)
            # Fixed orientation: first selected factor for even character,
            # second selected factor for odd character. No sampled sign fit.
            vector = projector[:, 0 if sign == 1 else 1]
            vector = sp.simplify(vector / sp.sqrt((vector.T * vector)[0]))
            if sp.simplify(projector - vector * vector.T) != sp.zeros(3):
                raise ArithmeticError("Selected character projector is not multiplicity one")
            rows.append(vector.T)
            provenance.append({"parent_partition": list(parent), "sign": sign,
                "canonical_parent_embedding": [[str(v) for v in row] for row in canonical.tolist()],
                "selected_to_canonical": [str(v) for v in sp.simplify(canonical.T * placement * vector)]})
    analysis = sp.Matrix.vstack(*rows)
    if sp.simplify(analysis * analysis.T) != sp.eye(3):
        raise ArithmeticError("Selected source analysis must be orthonormal")
    # Each row acts on (F2,F1,sqrt(N-2)P). R1 supplies that marginal at
    # role 1 and its exchanged marginal at role 2. Apply R1/sqrt(2), then
    # the normalized character analysis (1,sign)/sqrt(2).
    marginal = []
    for row, sign in zip(rows, (1, 1, -1), strict=True):
        two_role_marginal = sp.Matrix.vstack(row, row * swap) / sp.sqrt(2)
        marginal.append(sp.simplify(sp.Matrix([[1, sign]]) / sp.sqrt(2) * two_role_marginal))
    return analysis, sp.Matrix.vstack(*marginal), provenance


def compile_selected_scalar_vector(request):
    import sympy as sp
    from ye3t.core.cg import cg_exact_integer

    report = selected_scalar_vector_count(request)
    compiler_plan = selected_scalar_vector_plan(request)
    if request.get("schema") == "ye3t_selected_scalar_vector_plan_v1" and request != compiler_plan:
        raise ValueError("Selected-source compiler plan identity or operation binding mismatch")
    rank, width = (report["request"][key] for key in ("rank", "hidden_width"))
    analysis, marginal, provenance = _fixed_maps(rank)
    types = [{"id": name, "actual_partition": [2] if index % 2 == 0 else [1, 1],
              "L": index // 2, "parity": 1 if index < 2 else -1, "width": width}
             for index, name in enumerate(TYPE_NAMES)]
    scalar = [(t, c) for t in range(2) for c in range(width)]
    vector = [(t, c) for t in range(2, 4) for c in range(width)]
    products = []
    characters = sp.Matrix([[1, 1], [1, -1]]) / sp.sqrt(2)
    spherical = sp.Matrix([[1/sp.sqrt(2), -sp.I/sp.sqrt(2), 0],
                           [0, 0, 1], [-1/sp.sqrt(2), -sp.I/sp.sqrt(2), 0]])
    scalar_cg = sp.Matrix(3, 3, lambda i, j: cg_exact_integer(1, i-1, 1, j-1, 0, 0)._sympy_())
    metric = sp.simplify(-sp.sqrt(3) * spherical.T * scalar_cg * spherical)
    if metric != sp.eye(3):
        raise ArithmeticError("Core spherical scalar coupling disagrees with the Cartesian metric")
    for operation, pairs in (("scalar_scalar", combinations_with_replacement(scalar, 2)),
                             ("vector_dot_vector", combinations_with_replacement(vector, 2)),
                             ("scalar_vector", product(scalar, vector))):
        for left, right in pairs:
            out = (left[0] % 2) ^ (right[0] % 2)
            if operation == "scalar_vector":
                out += 2
            role_coefficient = sp.simplify(sum(characters[out % 2, role] *
                characters[left[0] % 2, role] * characters[right[0] % 2, role] for role in range(2)))
            products.append({"id": operation + "_" + str(len(products)), "operation": operation,
                "left": list(left), "right": list(right), "output_type": out,
                "role_coefficient": float(role_coefficient),
                "role_coefficient_exact": str(role_coefficient), "actual_product": "same_action_pointwise",
                "repeated_complete_child": left == right})
    result = {**report, "schema": "ye3t_selected_scalar_vector_v1", "compiler_plan": compiler_plan,
        "construction": "exact_young_subduction_and_character_projectors",
        "analysis_exact": [[str(v) for v in row] for row in analysis.tolist()],
        "analysis_orientation": "B=C_adjoint_Z; rows_are_analysis_coefficients",
        "moment_order": ["A^2*v_anchor*(A-q_anchor)", "A^2*q_anchor*(V-v_anchor)",
                         "sqrt(2)*A*V*q_anchor*(A-q_anchor)"],
        "marginal_scale_exact": "sqrt(2)/2",
        "role_analysis_exact": [["sqrt(2)/2", "sqrt(2)/2"], ["sqrt(2)/2", "-sqrt(2)/2"]],
        "marginal_analysis_exact": [[str(v) for v in row] for row in marginal.tolist()],
        "marginal_analysis": [[float(v) for v in row] for row in marginal.tolist()],
        "parent_conversion": provenance, "hidden_types": types, "products": products,
        "normalization_id": "full_content_scaled_marginal_v1",
        "primitive_basis": "orthonormal_Cartesian_scalar_and_polar_vector",
        "cartesian_to_CS_spherical": [["sqrt(2)/2", "-I*sqrt(2)/2", "0"],
                                      ["0", "0", "1"], ["-sqrt(2)/2", "-I*sqrt(2)/2", "0"]],
        "dot_to_normalized_CS_CG": "-sqrt(3)",
        "spherical_magnetic_order": [-1, 0, 1],
        "cartesian_dot_metric": [[float(v) for v in row] for row in metric.tolist()],
        "spatial_operations": {"scalar_scalar": "ordinary_product", "scalar_vector": "scalar_times_vector",
                               "vector_dot_vector": "orthonormal_Cartesian_metric"},
        "operation_relationships": {"same_action_pointwise": "character_product_then_role_multiplication",
            "disjoint": "LR_induction_requires_placement", "overlapping": "restrict_diagonal_induce_requires_bindings"}}
    result["self_hash"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return result
