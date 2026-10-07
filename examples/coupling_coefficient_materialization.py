"""Count a symmetric ACE sector and compile its factorized coefficients.

Edit the representation and fixed-content inputs below for another sector.
The schedule couples already adapted repeated blocks and is the practical
ACE coefficient route. The bounded full typed-orbit matrix is a test reference.
"""

from ye3t import YE3TRepresentation, couplings


cfg_ye3t = {
    "metadata": {"schema": "ye3t_config_v1", "name": "fixed_content_compile",
                 "status": "stable"},
    "representation": {
        "group": "O3", "ranks": [4],
        "parent": {"young_lambda": "(N)", "L": 0, "parity": "even"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {4: 2}, "l_max_per_rank": {4: 2},
        },
        "intermediates": {
            "young_kappa": "all_valid", "block_rotation": {"policy": "all_valid"},
        },
    },
    "basis": {
        "fixed_content": [1, 1, 2, 2],
        "input_Ls": [1, 1, 2, 2],
        "carrier": "ACE_density",
    },
    "runtime": {"constructor_backend": "auto"},
    "model": {}, "targets": {}, "validation": {},
}

representation = YE3TRepresentation.from_config(cfg_ye3t["representation"])
report = representation.count_fixed_content(
    cfg_ye3t["basis"]["fixed_content"], cfg_ye3t["basis"]["input_Ls"],
    carrier=cfg_ye3t["basis"]["carrier"],
)
compiler_plan = couplings.plan(report)
compiled = couplings.compile_ace_factorized_schedules_by_L(
    content=report.content,
    input_Ls=cfg_ye3t["basis"]["input_Ls"],
    constructor_backend=cfg_ye3t["runtime"]["constructor_backend"],
)
schedule = compiled.schedules_by_L[representation.L]
assert schedule.basis_count == report.counts_by_target[representation.L]
print("coupling count source: ye3t.couplings.count")
print("coupling plan source: ye3t.couplings.plan")
print("coefficient source: ye3t.couplings.compile_ace_factorized_schedules_by_L")
print("representation:", representation)
print("content:", tuple(compiled.content))
print("target:", compiled.target)
print("exact count:", report.counts_by_target[representation.L])
print("planned backend:", compiler_plan.backend)
print("compiled backend:", compiled.backend)
print("factorized basis count:", schedule.basis_count)
print("factorized term count:", schedule.term_count)
print("convention hash:", compiled.convention_hash)
print("validation passed:", bool(compiled.validation_report.get("passed", False)))
