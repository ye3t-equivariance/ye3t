"""Compare a symmetric fixed-content count with the GE-PI count formula.

Only multiplicities are calculated. No coupling coefficients or atomistic
descriptors are materialized. Edit the repeated channel and angular degrees
to compare another permutation-invariant sector.
"""

from ye3t import YE3TRepresentation, couplings, dusson_barthelemy_trivial_sector_counts


cfg_ye3t = {
    "metadata": {
        "schema": "ye3t_config_v1", "name": "symmetric_count_formula",
        "status": "stable",
    },
    "representation": {
        "group": "O3", "ranks": [4],
        "parent": {"young_lambda": "(N)", "L": 0, "parity": "even"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {4: 1}, "l_max_per_rank": {4: 1},
        },
        "intermediates": {
            "young_kappa": "all_valid", "block_rotation": {"policy": "all_valid"},
        },
    },
    "basis": {
        "carrier": "ACE_density",
        "fixed_content": [1, 1, 1, 1],
        "input_Ls": [1, 1, 1, 1],
    },
    "runtime": {"count_only": True},
    "model": {}, "targets": {},
    "validation": {"checks": ["formula_matches_compiler"]},
}


representation = YE3TRepresentation.from_config(cfg_ye3t["representation"])
content = cfg_ye3t["basis"]["fixed_content"]
input_Ls = cfg_ye3t["basis"]["input_Ls"]
formula_counts = dusson_barthelemy_trivial_sector_counts(content, input_Ls)
compiler_counts = couplings.counts_for_partitions(
    content, input_Ls, (representation.parent_partition(4),),
    spatial_symmetry="O3",
)
report = representation.count_fixed_content(
    content, input_Ls, carrier=cfg_ye3t["basis"]["carrier"],
)
target_L = representation.L
assert formula_counts == compiler_counts
assert compiler_counts.get(target_L, 0) == report.counts_by_target[target_L]

print("formula counts by L", formula_counts)
print("compiler counts by L", compiler_counts)
print("compiler count at L", target_L, report.counts_by_target[target_L])
print("formula matches compiler", True)
