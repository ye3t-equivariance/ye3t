"""Exact selected-source construction and physical product conventions."""

import numpy as np
import pytest
import sympy as sp

from ye3t.couplings import count, plan, compile
from ye3t.core.cg import cg_exact_integer


def test_native_product_lowering_preserves_routes_and_rejects_changed_maps():
    import copy
    import torch
    from ye3t.couplings import selected_scalar_vector_product_table
    from ye3t.backends.triton_joint import weighted_sparse_bilinear_forward
    compiled = compile(plan(count({"kind": "selected_scalar_vector_marginal", "hidden_width": 3})))
    table = selected_scalar_vector_product_table(compiled)
    torch.manual_seed(82)
    states = [torch.randn(2, 3, size, dtype=torch.float64) for size in (1, 1, 3, 3)]
    weights = torch.randn(len(compiled["products"]), 3, dtype=torch.float64)
    expected = [torch.zeros_like(value) for value in states]
    for route, weight in zip(compiled["products"], weights, strict=True):
        lt, lc = route["left"]
        rt, rc = route["right"]
        value = states[lt][:, lc] * states[rt][:, rc]
        if route["operation"] == "vector_dot_vector":
            value = value.sum(-1, keepdim=True)
        expected[route["output_type"]] += value[:, None]*weight[None, :, None]*route["role_coefficient"]
    packed = torch.cat([value.flatten(1) for value in states], -1)
    actual = weighted_sparse_bilinear_forward(packed, packed, weights.flatten(), table, prefer_triton=False)
    torch.testing.assert_close(actual, torch.cat([value.flatten(1) for value in expected], -1))
    changed = copy.deepcopy(compiled)
    changed["products"][0]["role_coefficient"] = 2.
    with pytest.raises(ValueError, match="unchanged canonical"):
        selected_scalar_vector_product_table(changed)
    bad = copy.deepcopy(table)
    bad.weight_index[0] = table.weight_count
    with pytest.raises(ValueError, match="weight_index"):
        weighted_sparse_bilinear_forward(packed, packed, weights.flatten(), bad, prefer_triton=False)
    malformed = copy.deepcopy(table)
    malformed.weight_index.resize_(1)
    with pytest.raises(ValueError, match="equal lengths"):
        weighted_sparse_bilinear_forward(packed, packed, weights.flatten(), malformed,
                                        prefer_triton=False, indices_validated=True)


def test_selected_parent_maps_counts_and_restriction():
    request = {"kind": "selected_scalar_vector_marginal", "rank": 4, "hidden_width": 2}
    report = count(request)
    compiler_plan = plan(report)
    assert compiler_plan["schema"] == "ye3t_selected_scalar_vector_plan_v1"
    assert compiler_plan["output_order"] == ["parent_4_even", "parent_31_even", "parent_31_odd"]
    compiled = compile(compiler_plan)
    with pytest.raises(ValueError, match="plan identity"):
        compile({**compiler_plan, "product_support_relation": "disjoint"})
    assert report["selected_dimension"] == 9
    assert report["full_parent_dimension"] == 12
    assert [r["parent_partition"] for r in report["source_types"]] == [[4], [3, 1], [3, 1]]
    assert [r["actual_partition"] for r in report["source_types"]] == [[2], [2], [1, 1]]
    analysis = sp.Matrix(compiled["analysis_exact"])
    expected = sp.Matrix([[sp.Rational(1, 2), sp.Rational(1, 2), 1/sp.sqrt(2)],
                          [sp.Rational(1, 2), sp.Rational(1, 2), -1/sp.sqrt(2)],
                          [-1/sp.sqrt(2), 1/sp.sqrt(2), 0]])
    assert sp.simplify(analysis - expected) == sp.zeros(3)
    swap = sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert sp.simplify(analysis * swap * analysis.T) == sp.diag(1, 1, -1)
    pooled = sp.Matrix([[0, 0, 1/sp.sqrt(2)], [0, 0, 1/sp.sqrt(2)], [1, 0, 0], [0, 1, 0]])
    for index, record in enumerate(compiled["parent_conversion"]):
        embedding = sp.Matrix(record["canonical_parent_embedding"])
        conversion = sp.Matrix(record["selected_to_canonical"])
        assert sp.simplify(embedding * conversion - pooled * analysis[index, :].T) == sp.zeros(4, 1)
    assert all(row["parity"] == -1 for row in report["source_types"])
    assert compile(request) == compiled
    for update in ({"rank": 5}, {"unsupported": True}):
        invalid = {**request, **update}
        with pytest.raises(ValueError):
            count(invalid)


def test_normalized_role_products_complete_children_and_spatial_metric():
    compiled = compile({"kind": "selected_scalar_vector_marginal"})
    assert len(compiled["products"]) == 36
    chars = sp.Matrix([[1, 1], [1, -1]]) / sp.sqrt(2)
    for route in compiled["products"]:
        left, right = route["left"], route["right"]
        coefficients = chars * sp.matrix_multiply_elementwise(chars[left[0] % 2, :], chars[right[0] % 2, :]).T
        output = route["output_type"] % 2
        assert sp.simplify(coefficients[output] - sp.sympify(route["role_coefficient_exact"])) == 0
        assert coefficients[1-output] == 0
        assert route["repeated_complete_child"] == (left == right)
    conversion = np.array(sp.Matrix(compiled["cartesian_to_CS_spherical"])).astype(complex)
    rng = np.random.default_rng(2718)
    for _ in range(6):
        left, right = rng.normal(size=(2, 3))
        a, b = conversion @ left, conversion @ right
        coupled = sum(float(cg_exact_integer(1, m, 1, -m, 0, 0)._sympy_()) * a[m+1] * b[-m+1] for m in (-1, 0, 1))
        np.testing.assert_allclose(-np.sqrt(3) * coupled, left @ right, atol=2e-15)
