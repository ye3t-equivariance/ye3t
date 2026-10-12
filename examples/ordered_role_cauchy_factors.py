"""Evaluate complete Cauchy factor couplings at ranks 2–8.

Each ordered factor supplies two role channels and a complete angular multiplet
in complex Condon-Shortley order. The output retains every multiplicity copy,
Young-tableau component, and magnetic component. The parent partitions below
select the mixed (N-1,1) family, beginning with (1,1) at rank two.
Vector triples combine with scalar factors at ranks four through eight;
the second rank-two case uses quadrupole factors.
Requires PyTorch and SymPy. See docs/covariant_cauchy_basis.md.
"""

from ye3t import YE3TRepresentation
from ye3t.workflows import count_coupling_cases, compile_coupling_cases, evaluate_coupling_cases


cfg_ye3t = {
    "metadata": {"schema": "ye3t_config_v1", "name": "ordered_role_cauchy_factors", "status": "experimental"},
    "representation": {
        "group": "SO3", "ranks": [2, 3, 4, 5, 6, 7, 8],
        "parent": {"young_lambda": {2: [1, 1], 3: [2, 1], 4: [3, 1], 5: [4, 1], 6: [5, 1], 7: [6, 1], 8: [7, 1]}, "L": 1, "parity": None},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2},
            "l_max_per_rank": {2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2},
        },
        "intermediates": {
            "young_kappa": "all_valid", "block_rotation": {"policy": "all_valid"},
        },
    },
    "basis": {
        "carrier": "ordered_role", "role_dimension": 2, "role_kappa_policy": "all",
        "channels": [
            {"factor_type": "vector", "l": 1, "source_family_id": "user_supplied"},
            {"factor_type": "scalar", "l": 0, "source_family_id": "user_supplied"},
            {"factor_type": "quadrupole", "l": 2, "source_family_id": "user_supplied"},
        ],
        "cases_by_rank": {
            2: [{"channel_indices": [0], "block_sizes": [2]},
                {"channel_indices": [2], "block_sizes": [2]}],
            3: [{"channel_indices": [0], "block_sizes": [3]}],
            4: [{"channel_indices": [0, 1], "block_sizes": [3, 1]}],
            5: [{"channel_indices": [0, 1], "block_sizes": [3, 2]}],
            6: [{"channel_indices": [0, 1], "block_sizes": [3, 3]}],
            7: [{"channel_indices": [0, 1], "block_sizes": [3, 4]}],
            8: [{"channel_indices": [0, 1], "block_sizes": [3, 5]}],
        },
        # l -> array with axes (factor, role, magnetic component).
        "multiplets": {
            0: [[[0.6], [-0.4]], [[0.3], [0.7]], [[-0.2], [0.9]],
                [[0.8], [0.1]], [[0.4], [-0.3]], [[-0.5], [0.6]],
                [[0.7], [0.2]], [[0.3], [-0.8]]],
            1: [[[0.2, 0.4, -0.1], [0.3, -0.7, 0.5]],
                [[-0.3, 0.1, 0.7], [0.8, 0.2, -0.4]],
                [[0.8, -0.2, 0.5], [-0.1, 0.6, 0.3]]],
            2: [[[0.2, 0.4, -0.1, 0.6, -0.3], [0.7, -0.2, 0.5, -0.4, 0.1]],
                [[-0.3, 0.8, 0.2, 0.1, -0.5], [0.6, 0.1, -0.4, 0.3, 0.7]]],
        },
    },
    "runtime": {"dtype": "complex128", "device": "cpu"},
    "model": {}, "targets": {},
    "validation": {"checks": ["exact_count", "finite_values"]},
}


representation = YE3TRepresentation.from_config(cfg_ye3t["representation"])
cases = count_coupling_cases(representation, cfg_ye3t["basis"])
print("multiplicity counts (rank, case):",
      {key: case["multiplicity_count"] for key, case in cases.items()})
compiled_cases = compile_coupling_cases(cases)
values = evaluate_coupling_cases(
    compiled_cases, cfg_ye3t["basis"]["multiplets"],
    dtype=cfg_ye3t["runtime"]["dtype"], device=cfg_ye3t["runtime"]["device"],
)
print("coupled Cauchy axes (a, t, M):",
      {key: tuple(value.shape) for key, value in values.items()})
print("rank-eight artifact hash:", compiled_cases[8, 0]["compiled"]["self_hash"])
