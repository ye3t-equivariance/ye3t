"""Evaluate independent scalar ACE couplings of supplied density multiplets at ranks 1–8.

Edit the complete input contents and angular degrees below. Each case retains
every independent coupling for its selected parent. Rank two includes two
different angular contents; higher-rank cases include multiple coupling paths.
The parent and every Young intermediate are permutation invariant. Angular
intermediates keep all allowed values that couple to the scalar parent.
Use user_supplied_factors.py for mixed Young parents and ordered factors.
Requires the installed ye3t package and its numerical dependencies.
"""

from ye3t import YE3TRepresentation
from ye3t.workflows import count_coupling_cases, compile_coupling_cases, evaluate_coupling_cases


cfg_ye3t = {
    "metadata": {"schema": "ye3t_config_v1", "name": "symmetric_density_factors", "status": "stable"},
    "representation": {
        "group": "O3", "ranks": [1, 2, 3, 4, 5, 6, 7, 8],
        "parent": {"young_lambda": "(N)", "L": 0, "parity": "even"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8},
            "l_max_per_rank": {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2},
        },
        "intermediates": {
            "young_kappa": "symmetric_only", "block_rotation": {"policy": "all_valid"},
        },
    },
    "basis": {
        "carrier": "ACE_density",
        "cases_by_rank": {
            1: [{"content": [1], "input_Ls": [0]},
                {"content": [2], "input_Ls": [0]}],
            2: [{"content": [1, 2], "input_Ls": [1, 1]},
                {"content": [1, 2], "input_Ls": [2, 2]}],
            3: [{"content": [1, 2, 3], "input_Ls": [1, 1, 2]}],
            4: [{"content": [1, 2, 3, 4], "input_Ls": [1, 1, 2, 2]},
                {"content": [1, 1, 2, 2], "input_Ls": [1, 1, 2, 2]}],
            5: [{"content": [1, 2, 3, 4, 5], "input_Ls": [1, 1, 1, 1, 2]}],
            6: [{"content": [1, 2, 3, 4, 5, 6], "input_Ls": [1, 1, 1, 1, 1, 1]}],
            7: [{"content": [1, 2, 3, 4, 5, 6, 7], "input_Ls": [1, 1, 1, 1, 1, 1, 2]}],
            8: [{"content": [1, 2, 3, 4, 5, 6, 7, 8], "input_Ls": [1, 1, 1, 1, 1, 1, 1, 1]}],
        },
        # Supply complete m=-l,...,l arrays from your own basis or model.
        # Rows identify density channels n=1,...,8.
        "multiplets": {
            0: [[0.4], [0.7], [-0.2], [0.5], [0.8], [-0.4], [0.6], [0.3]],
            1: [[1.0, 0.2, -0.4], [0.3, -0.5, 0.9], [-0.2, 0.7, 0.6],
                [0.8, -0.1, 0.3], [-0.4, 0.6, 0.2], [0.5, 0.3, -0.7],
                [0.2, -0.8, 0.4], [-0.6, 0.9, 0.1]],
            2: [[0.2, 0.4, -0.1, 0.6, -0.3], [0.7, -0.2, 0.5, -0.4, 0.1],
                [-0.3, 0.8, 0.2, 0.1, -0.5], [0.6, 0.1, -0.4, 0.3, 0.7],
                [0.4, -0.6, 0.3, 0.2, -0.1], [-0.2, 0.5, 0.7, -0.3, 0.4],
                [0.1, 0.3, -0.8, 0.5, 0.2], [0.5, -0.4, 0.6, 0.1, -0.7]],
        },
    },
    "runtime": {"constructor_backend": "auto", "dtype": "complex128",
                "device": "cpu", "backend": "auto"},
    "model": {}, "targets": {},
    "validation": {},
}


representation = YE3TRepresentation.from_config(cfg_ye3t["representation"])
cases = count_coupling_cases(representation, cfg_ye3t["basis"])
print("multiplicity counts (rank, case):",
      {key: case["multiplicity_count"] for key, case in cases.items()})
compiled_cases = compile_coupling_cases(
    cases, constructor_backend=cfg_ye3t["runtime"]["constructor_backend"],
)
values = evaluate_coupling_cases(
    compiled_cases, cfg_ye3t["basis"]["multiplets"],
    dtype=cfg_ye3t["runtime"]["dtype"], device=cfg_ye3t["runtime"]["device"],
    backend=cfg_ye3t["runtime"]["backend"],
)
print("output shapes (a, M):", {key: tuple(value.shape) for key, value in values.items()})
print("rank-eight first three scalars:", values[8, 0][:3, 0])
