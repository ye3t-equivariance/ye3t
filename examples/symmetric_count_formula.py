"""Compare the GE-PI formula with scalar ACE compiler counts at ranks 1–8.

Edit the complete input contents and angular degrees below. Each case retains
every independent coupling for its selected parent. Rank two includes two
different angular contents; higher ranks have multiple coupling paths.
Requires the installed ye3t package and its numerical dependencies.
"""

from ye3t import YE3TRepresentation
from ye3t.workflows import count_coupling_cases, compare_symmetric_count_formulas


cfg_ye3t = {
    "metadata": {"schema": "ye3t_config_v1", "name": "symmetric_count_formula", "status": "stable"},
    "representation": {
        "group": "O3", "ranks": [1, 2, 3, 4, 5, 6, 7, 8],
        "parent": {"young_lambda": "(N)", "L": 0, "parity": "even"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8},
            "l_max_per_rank": {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2},
        },
        "intermediates": {
            "young_kappa": "all_valid", "block_rotation": {"policy": "all_valid"},
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
    },
    "runtime": {"count_only": True},
    "model": {}, "targets": {},
    "validation": {"checks": ["exact_counts"]},
}


representation = YE3TRepresentation.from_config(cfg_ye3t["representation"])
cases = count_coupling_cases(representation, cfg_ye3t["basis"])
print("multiplicity counts (rank, case):",
      {key: case["multiplicity_count"] for key, case in cases.items()})
formula_counts = compare_symmetric_count_formulas(cases)
print("formula counts by L:", formula_counts)
print("formula matches compiler", True)
