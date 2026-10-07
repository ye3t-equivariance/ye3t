"""Inspect compiler-valid multiplicities for one fixed-content ACE sector."""

from ye3t import YE3TRepresentation


cfg_ye3t = {
    "metadata": {"schema": "ye3t_config_v1",
                 "name": "fixed_content_count", "status": "stable",
                 "label_preview": 2},
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
    "runtime": {}, "model": {}, "targets": {}, "validation": {},
}

representation = YE3TRepresentation.from_config(cfg_ye3t["representation"])
report = representation.count_fixed_content(
    cfg_ye3t["basis"]["fixed_content"], cfg_ye3t["basis"]["input_Ls"],
    carrier=cfg_ye3t["basis"]["carrier"],
)
labels = report.labels_for_target(representation.L)
print("coupling count source: ye3t.couplings.count")
print("representation:", representation)
print("content:", tuple(report.content))
print("carrier:", report.carrier)
print("target:", report.target)
print("counts by target:", dict(report.counts_by_target))
print("total labels for target:", len(labels))
print("validation passed:", bool(report.validation_report.get("passed", False)))
for index, label in enumerate(labels[:cfg_ye3t["metadata"]["label_preview"]]):
    print("label", index, label)
