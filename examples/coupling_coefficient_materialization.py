"""Count, plan, and compile one fixed-content YE3T coupling sector.

Edit the representation and fixed-content inputs below for another sector.
Coefficient materialization is explicit because it can cost much more than a
count or backend plan at larger ranks.
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
    "runtime": {"subduction_materialization_backend": "numeric_cached"},
    "model": {}, "targets": {}, "validation": {},
}

representation = YE3TRepresentation.from_config(cfg_ye3t["representation"])
report = representation.count_fixed_content(
    cfg_ye3t["basis"]["fixed_content"], cfg_ye3t["basis"]["input_Ls"],
    carrier=cfg_ye3t["basis"]["carrier"],
)
compiler_plan = couplings.plan(report)
compiled = couplings.compile(
    compiler_plan,
    subduction_materialization_backend=cfg_ye3t["runtime"]["subduction_materialization_backend"],
)
certificate = compiled.certificate
print("coupling count source: ye3t.couplings.count")
print("coupling plan source: ye3t.couplings.plan")
print("coefficient source: ye3t.couplings.compile")
print("representation:", representation)
print("content:", tuple(compiled.content))
print("carrier:", compiled.carrier)
print("target:", compiled.target)
print("exact count:", report.counts_by_target[representation.L])
print("planned backend:", compiler_plan.backend)
print("compiled backend:", compiled.backend)
print("backend plan selected:", compiled.validation_report.get("backend_plan_selected"))
print("convention hash:", compiled.convention_hash)
print("validation passed:", bool(compiled.validation_report.get("passed", False)))
print("certificate passed:", bool(certificate.passed))
print("coefficient hash:", certificate.coefficient_hash)
