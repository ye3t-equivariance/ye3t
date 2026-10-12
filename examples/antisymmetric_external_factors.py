"""Evaluate permutation-antisymmetric factor couplings at ranks 2–8.

The complete m=-l,...,l inputs can come from any basis or upstream model.
Edit parent.L and the rank-specific contents to choose another output.
SO(3) permits both polar and axial results in one rank catalogue; the natural
O(3) parity of each case is (-1)**sum(input_Ls).
Repeated l=1 blocks have at most two factors, so their exterior powers are nonzero.
Requires PyTorch. See docs/basic_examples.rst for the input conventions.
"""

from ye3t import YE3TRepresentation
from ye3t.workflows import count_coupling_cases, compile_coupling_cases, evaluate_coupling_cases


cfg_ye3t = {
    "metadata": {"schema": "ye3t_config_v1", "name": "antisymmetric_external_factors", "status": "experimental"},
    "representation": {
        "group": "SO3", "ranks": [2, 3, 4, 5, 6, 7, 8],
        "parent": {"young_lambda": "antisymmetric", "L": 1, "parity": None},
        "factorization": "standard_split", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {2: 2, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4},
            "l_max_per_rank": {2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2},
        },
        "intermediates": {
            "young_kappa": "all_valid", "block_rotation": {"policy": "all_valid"},
        },
    },
    "basis": {
        "carrier": "external_tensor",
        "cases_by_rank": {
            2: [{"content": [1, 2], "input_Ls": [1, 1]},
                {"content": [1, 2], "input_Ls": [2, 2]}],
            3: [{"content": [1, 1, 2], "input_Ls": [1, 1, 1]}],
            4: [{"content": [1, 1, 2, 2], "input_Ls": [1, 1, 1, 1]}],
            5: [{"content": [1, 1, 2, 2, 3], "input_Ls": [1, 1, 1, 1, 1]}],
            6: [{"content": [1, 1, 2, 2, 3, 3], "input_Ls": [1, 1, 1, 1, 1, 1]}],
            7: [{"content": [1, 1, 2, 2, 3, 3, 4], "input_Ls": [1, 1, 1, 1, 1, 1, 1]}],
            8: [{"content": [1, 1, 2, 2, 3, 3, 4, 4], "input_Ls": [1, 1, 1, 1, 1, 1, 1, 1]}],
        },
        # Supply complete m=-l,...,l arrays from your own basis or model.
        # multiplets[l][f] uses the global ordered factor index f=0,...,7.
        "multiplets": {
            1: [[1.0, 0.2, -0.4], [0.3, -0.5, 0.9], [-0.2, 0.7, 0.6],
                [0.8, -0.1, 0.3], [-0.4, 0.6, 0.2], [0.5, 0.3, -0.7],
                [0.2, -0.8, 0.4], [-0.6, 0.9, 0.1]],
            2: [[0.2, 0.4, -0.1, 0.6, -0.3], [0.7, -0.2, 0.5, -0.4, 0.1],
                [-0.3, 0.8, 0.2, 0.1, -0.5], [0.6, 0.1, -0.4, 0.3, 0.7],
                [0.4, -0.6, 0.3, 0.2, -0.1], [-0.2, 0.5, 0.7, -0.3, 0.4],
                [0.1, 0.3, -0.8, 0.5, 0.2], [0.5, -0.4, 0.6, 0.1, -0.7]],
        },
    },
    "runtime": {"dtype": "complex128", "device": "cpu",
                "subduction_backend": "exact"},
    "model": {}, "targets": {},
    "validation": {"checks": ["exact_count", "finite_values"]},
}


representation = YE3TRepresentation.from_config(cfg_ye3t["representation"])
cases = count_coupling_cases(representation, cfg_ye3t["basis"])
print("multiplicity counts (rank, case):",
      {key: case["multiplicity_count"] for key, case in cases.items()})
compiled_cases = compile_coupling_cases(
    cases, subduction_backend=cfg_ye3t["runtime"]["subduction_backend"],
)
values = evaluate_coupling_cases(
    compiled_cases, cfg_ye3t["basis"]["multiplets"],
    dtype=cfg_ye3t["runtime"]["dtype"], device=cfg_ye3t["runtime"]["device"],
)
print("coupled axes (a, t, M):", {key: tuple(value.shape) for key, value in values.items()})
print("rank-eight first multiplet:", values[8, 0][0])
