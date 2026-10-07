"""Full magnetic ACE coordinates from certified symmetric blocks."""

import numpy as np
import pytest

from ye3t.couplings import compile_ace_coordinate


@pytest.mark.parametrize("label,expected_terms", (
    ({"n_tuple": [1, 1, 1], "l_tuple": [1, 1, 1],
      "internal_Ls": [1], "L_R": 1, "tree_type": "balanced",
      "basis_key": ["sym", 1, 0]}, 2),
    ({"n_tuple": [1, 1, 1, 1], "l_tuple": [0, 0, 0, 1],
      "internal_Ls": [0, 1], "L_R": 1, "tree_type": "balanced",
      "basis_key": ["node", ["sym", 0, 0], []]}, 1),
    ({"n_tuple": [1, 1, 1], "l_tuple": [1, 1, 1],
      "internal_Ls": [3], "L_R": 3, "tree_type": "balanced",
      "basis_key": ["sym", 3, 0]}, None),
))
def test_covariant_ace_collected_rows_match_independent_block_contraction(
    label, expected_terms,
):
    compiled = compile_ace_coordinate(label)
    table = compiled["coefficient_table"]
    schedule = compiled["factorized_schedule"]
    assert compiled["certificate"]["passed"]
    assert compiled["certificate"]["magnetic_sign_reversal_residual"] < 1e-12
    assert compiled["certificate"]["so3_ladder_relative_residual"] < 1e-12
    M_values = tuple(range(-label["L_R"], label["L_R"] + 1))
    assert tuple(table.component_M_R) == M_values
    if expected_terms is None:
        assert np.all(np.diff(table.component_offsets) > 0)
    else:
        assert np.diff(table.component_offsets).tolist() == [expected_terms] * len(M_values)
    rng = np.random.default_rng(93)
    sources = {}
    for spec in schedule.block_specs[0]:
        key = int(spec["n"]), int(spec["l"])
        if key not in sources:
            sources[key] = (
                rng.normal(size=2 * key[1] + 1)
                + 1j * rng.normal(size=2 * key[1] + 1)
            )
    block_values = []
    for spec, block_plan in zip(schedule.block_specs[0], compiled["block_plans"]):
        source = sources[int(spec["n"]), int(spec["l"])]
        if block_plan is None:
            block_values.append(source)
            continue
        values = []
        for entry in block_plan.entries:
            values.append(sum(
                complex(term["coefficient"])
                * np.prod(source ** np.asarray(term["exponents"], dtype=int))
                for term in entry.component_terms
            ))
        block_values.append(np.asarray(values))
    for component, M in enumerate(M_values):
        outer_rows, outer_coefficients = schedule.component_terms(component)
        factorized = sum(
            complex(coefficient) * np.prod([
                block_values[index][int(magnetic) + int(
                    spec["Lambda"] if spec["kind"] == "sym" else spec["l"]
                )]
                for index, (spec, magnetic) in enumerate(zip(
                    schedule.block_specs[0], magnetic_row
                ))
            ])
            for magnetic_row, coefficient in zip(outer_rows, outer_coefficients)
        )
        raw_rows, raw_coefficients = table.component_terms(component)
        collected = sum(
            complex(coefficient) * np.prod([
                sources[int(n), int(l)][int(m) + int(l)]
                for n, l, m in zip(label["n_tuple"], label["l_tuple"], magnetic_row)
            ])
            for magnetic_row, coefficient in zip(raw_rows, raw_coefficients)
        )
        np.testing.assert_allclose(collected, factorized, atol=1e-12, rtol=1e-12)
        assert all(int(sum(row)) == M for row in raw_rows)


def test_covariant_ace_distinct_symmetric_copies_have_independent_rows():
    common = {"n_tuple": [1, 1, 1, 1], "l_tuple": [2, 2, 2, 2],
              "internal_Ls": [4], "L_R": 4, "tree_type": "balanced"}
    compiled = [compile_ace_coordinate({**common, "basis_key": ["sym", 4, copy]})
                for copy in (0, 1)]
    assert compiled[0]["label"].angular_key() != compiled[1]["label"].angular_key()
    columns = []
    for item in compiled:
        rows, coefficients = item["coefficient_table"].component_terms(4)
        columns.append(dict(zip(map(tuple, rows.tolist()), coefficients.tolist())))
    all_rows = sorted(set(columns[0]) | set(columns[1]))
    matrix = np.asarray([[column.get(row, 0.0) for column in columns]
                         for row in all_rows], dtype=np.complex128)
    assert np.linalg.matrix_rank(matrix, tol=1e-10) == 2


def test_covariant_ace_resource_caps_fail_during_collection():
    label = {"n_tuple": [1, 1, 1], "l_tuple": [1, 1, 1],
             "internal_Ls": [1], "L_R": 1, "tree_type": "balanced",
             "basis_key": ["sym", 1, 0]}
    with pytest.raises(MemoryError, match="maximum_unique_monomials"):
        compile_ace_coordinate(label, maximum_unique_monomials=5)
    with pytest.raises(MemoryError, match="maximum_term_contributions"):
        compile_ace_coordinate(label, maximum_term_contributions=1)
    with pytest.raises(MemoryError, match="maximum_coordinate_bytes during collection"):
        compile_ace_coordinate(label, maximum_coordinate_bytes=1)
