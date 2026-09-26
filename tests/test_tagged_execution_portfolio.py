import math

import numpy as np
import pytest

from ye3t.execution_plan import compile_tagged_moment_execution_portfolio


def _forward_reverse(plan, bases, routes):
    values = list(float(value) for value in bases)
    for node in plan["nodes"]:
        values.append(values[node["left_value"]] * values[node["right_value"]])
    adjoint = [0.0] * len(values)
    output = 0.0
    for route in routes:
        root = int(route["root_value"])
        coefficient = float(route["coefficient"])
        value = 1.0 if root < 0 else values[root]
        output += coefficient * value
        if root >= 0:
            adjoint[root] += coefficient
    base_count = int(plan["base_count"])
    for node_index in range(len(plan["nodes"]) - 1, -1, -1):
        node = plan["nodes"][node_index]
        value_index = base_count + node_index
        seed = adjoint[value_index]
        left = int(node["left_value"])
        right = int(node["right_value"])
        adjoint[left] += seed * values[right]
        adjoint[right] += seed * values[left]
    return output, np.asarray(adjoint[:base_count])


def _evaluate_portfolio(portfolio, source):
    moment_plan = portfolio["moment_plan"]
    moment_routes = tuple(
        {
            "root_value": root,
            "coefficient": 1.0,
        }
        for root in moment_plan["roots"]
    )
    moment_values = []
    for route in moment_routes:
        value, _ = _forward_reverse(moment_plan, source, (route,))
        moment_values.append(value)
    outer_bases = tuple(source) + tuple(moment_values)
    output, outer_adjoint = _forward_reverse(
        portfolio["outer_plan"],
        outer_bases,
        portfolio["species_routes"]["X"],
    )
    source_adjoint = np.asarray(outer_adjoint[: len(source)], dtype=np.float64)
    moment_adjoint = outer_adjoint[len(source) :]
    for moment_index, seed in enumerate(moment_adjoint):
        if seed == 0.0:
            continue
        _, local = _forward_reverse(
            moment_plan,
            source,
            (
                {
                    "root_value": moment_plan["roots"][moment_index],
                    "coefficient": float(seed),
                },
            ),
        )
        source_adjoint += local
    return output, source_adjoint


def _fixture():
    keys = [[0, 0], [0, 1], [0, 2]]
    program = {
        "feature_count": 3,
        "real_density_keys": keys,
        "real_moment_keys": [
            [keys[0], keys[0], keys[0], keys[0]],
            [keys[0], keys[1]],
            [keys[2]],
        ],
        "terms": [
            {
                "feature_index": 0,
                "coefficient": 2.0,
                "density_factor_indices": [],
                "p": 0,
                "moment_indices": [0],
            },
            {
                "feature_index": 1,
                "coefficient": -1.0,
                "density_factor_indices": [1],
                "p": 0,
                "moment_indices": [1],
            },
            {
                "feature_index": 2,
                "coefficient": 0.5,
                "density_factor_indices": [0, 0],
                "p": 0,
                "moment_indices": [2],
            },
        ],
    }
    return program, {"X": [1.5, -2.0, 3.0]}


def test_tagged_portfolio_uses_certified_symmetric_power_blocks():
    program, beta = _fixture()
    portfolio = compile_tagged_moment_execution_portfolio(program, beta)
    assert portfolio["certificate"]["passed"]
    assert portfolio["certificate"]["runtime_label_inference_required"] is False
    assert portfolio["moment_plan"]["factorization"] == "commutative_symmetric_power"
    assert portfolio["outer_plan"]["factorization"] == "canonical_prefix"
    assert portfolio["moment_plan"]["repeated_factor_product_count"] == 1
    assert portfolio["moment_plan"]["binary_node_count"] < 5
    assert portfolio["moment_plan"]["division_operations"] == 0
    assert portfolio["outer_plan"]["certificate"][
        "preserves_canonical_left_to_right_order"
    ]
    assert not portfolio["outer_plan"]["certificate"][
        "repeated_factors_use_symmetric_power_nodes"
    ]
    assert portfolio["certificate"][
        "preserves_direct_route_accumulation_order"
    ]
    assert [
        route["coefficient"] for route in portfolio["species_routes"]["X"]
    ] == [3.0, 2.0, 1.5]
    candidates = {row["candidate_id"]: row for row in portfolio["candidates"]}
    assert candidates["symmetric_power"]["eligible"]
    assert candidates["block"]["eligible"]


@pytest.mark.parametrize(
    "source",
    (
        np.asarray([0.0, 1.25, -0.5]),
        np.asarray([0.7, -0.2, 1.1]),
    ),
)
def test_tagged_portfolio_forward_and_division_free_adjoint(source):
    program, beta = _fixture()
    portfolio = compile_tagged_moment_execution_portfolio(program, beta)
    value, gradient = _evaluate_portfolio(portfolio, source)
    x0, x1, x2 = source
    expected = 3.0 * x0**4 + 2.0 * x0 * x1**2 + 1.5 * x0**2 * x2
    expected_gradient = np.asarray(
        [12.0 * x0**3 + 2.0 * x1**2 + 3.0 * x0 * x2,
         4.0 * x0 * x1,
         1.5 * x0**2]
    )
    assert math.isclose(value, expected, rel_tol=0.0, abs_tol=1.0e-13)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=0.0, atol=1.0e-13)


def test_tagged_portfolio_rejects_unknown_moment_source():
    program, beta = _fixture()
    program["real_moment_keys"][0][0] = [9, 9]
    with pytest.raises(ValueError, match="unknown real-density key"):
        compile_tagged_moment_execution_portfolio(program, beta)
