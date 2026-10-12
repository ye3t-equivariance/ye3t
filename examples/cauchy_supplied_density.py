"""Evaluate Cauchy role-density multiplets across ranks 2–8.

The inputs have two role channels and complete real tesseral multiplets.
All valid local Young paths are retained for each content. SO(3) allows both
parities in the rank catalogue; each count report records the natural O(3)
parity. Edit the input channels, role densities, and parent.L together.
Vector triples combine with scalar-density powers at ranks four through eight;
the second rank-two case uses quadrupole inputs.
Requires NumPy, SymPy, and PyTorch. This is an experimental supplied-density workflow.
"""

from ye3t import YE3TRepresentation
from ye3t.workflows import count_coupling_cases, compile_coupling_cases, evaluate_coupling_cases


cfg_ye3t = {
    "metadata": {"schema": "ye3t_config_v1", "name": "cauchy_supplied_density", "status": "experimental"},
    "representation": {
        "group": "SO3", "ranks": [2, 3, 4, 5, 6, 7, 8],
        "parent": {"young_lambda": "(N)", "L": 1, "parity": None},
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
        "carrier": "A_s", "role_dimension": 2, "role_kappa_policy": "all",
        "channels": [
            {"neighbor_species": "X", "radial_channel": 0, "l": 1,
             "source_family_id": "user_supplied"},
            {"neighbor_species": "X", "radial_channel": 1, "l": 0,
             "source_family_id": "user_supplied"},
            {"neighbor_species": "X", "radial_channel": 0, "l": 2,
             "source_family_id": "user_supplied"},
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
        # Tesseral order: cosine l,...,1, then m=0, then sine 1,...,l.
        "role_densities": {
            0: [[0.2, 0.5, -0.3], [-0.7, 0.1, 0.4]],
            1: [[0.6], [-0.4]],
            2: [[0.2, 0.4, -0.1, 0.6, -0.3], [0.7, -0.2, 0.5, -0.4, 0.1]],
        },
    },
    "runtime": {"dtype": "float64", "device": "cpu"},
    "model": {}, "targets": {},
    "validation": {"checks": ["exact_count", "finite_values"]},
}


representation = YE3TRepresentation.from_config(cfg_ye3t["representation"])
cases = count_coupling_cases(representation, cfg_ye3t["basis"])
print("multiplicity counts (rank, case):",
      {key: case["multiplicity_count"] for key, case in cases.items()})
compiled_cases = compile_coupling_cases(cases)
values = evaluate_coupling_cases(
    compiled_cases, cfg_ye3t["basis"]["role_densities"],
    dtype=cfg_ye3t["runtime"]["dtype"], device=cfg_ye3t["runtime"]["device"],
)
print("Cauchy output shapes (a, M):", {key: value.shape for key, value in values.items()})
print("rank-eight local Young paths:",
      sorted({label["block_kappas"] for label in cases[8, 0]["report"]["labels"]}))
print("rank-eight first three multiplets:", values[8, 0][:3])
