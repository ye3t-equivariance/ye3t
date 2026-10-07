"""Couple user-supplied spherical tensor factors without choosing their source.

The three rows may come from any radial basis or upstream model, provided
each row uses the same complex spherical-harmonic convention and carries its
full ``m=-l,...,l`` multiplet. The repeated first factor type makes this a
nontrivial fixed-content example. The compiler returns every independent
multiplicity coordinate ``a`` and its complete Young ``t`` and magnetic ``M`` axes.

Every route is evaluated from shared local, angular, and Young factors.
This is one ordered orbit fiber; physical density or motif pooling is separate.
"""

import torch

from ye3t import YE3TRepresentation, couplings


cfg_ye3t = {
    "metadata": {
        "schema": "ye3t_config_v1", "name": "user_supplied_factors",
        "status": "experimental",
    },
    "representation": {
        "group": "O3", "ranks": [3],
        "parent": {"young_lambda": "(2,1)", "L": 1, "parity": "odd"},
        "factorization": "standard_split", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {3: 2}, "l_max_per_rank": {3: 1},
        },
        "intermediates": {
            "young_kappa": "all_valid", "block_rotation": {"policy": "all_valid"},
        },
    },
    "basis": {
        "carrier": "external_tensor", "fixed_content": [1, 1, 2],
        "input_Ls": [1, 1, 1],
        "factor_multiplets": [
            [1.0, 0.2, -0.4], [0.3, -0.5, 0.9], [-0.2, 0.7, 0.6],
        ],
    },
    "runtime": {"dtype": "complex128", "device": "cpu",
                "subduction_backend": "exact", "example_message_count": 64},
    "model": {}, "targets": {},
    "validation": {"checks": ["exact_count", "complete_factorization"]},
}


representation = YE3TRepresentation.from_config(cfg_ye3t["representation"])
report = representation.count_fixed_content(
    cfg_ye3t["basis"]["fixed_content"],
    cfg_ye3t["basis"]["input_Ls"],
    carrier=cfg_ye3t["basis"]["carrier"],
)
full_plan = couplings.plan(report)
compiled = couplings.compile(
    full_plan,
    subduction_materialization_backend=cfg_ye3t["runtime"]["subduction_backend"],
)
execution_plan = couplings.execution_plan_from_compiled_coupler(compiled)
factors = torch.tensor(
    cfg_ye3t["basis"]["factor_multiplets"],
    dtype=getattr(torch, cfg_ye3t["runtime"]["dtype"]),
    device=cfg_ye3t["runtime"]["device"],
)
runtime = couplings.bind_typed_factorized_execution_plan_torch(
    execution_plan, dtype=factors.dtype, device=factors.device)
values = runtime.evaluate(factors)
message_factors = factors.unsqueeze(0).expand(
    cfg_ye3t["runtime"]["example_message_count"], -1, -1,
).clone().requires_grad_(True)
message_features = runtime.evaluate(message_factors)
message_gradient = torch.autograd.grad(
    message_features.abs().square().sum(), message_factors,
)[0]
assert report.counts_by_target[representation.L] == len(
    full_plan.validation_report["alpha_bindings"]
)
assert compiled.certificate.passed
assert not compiled.coupler.sparse_coefficient_tables
assert values.shape == (report.counts_by_target[representation.L], 2, 3)

print("full independent multiplicity count", report.counts_by_target[representation.L])
print("all multiplicity coordinates a", list(range(values.shape[0])))
print("coupled axes (a, t, M)", tuple(values.shape))
print("message batch axes (edge, a, t, M)", tuple(message_features.shape))
print("message factor gradient shape", tuple(message_gradient.shape))
print("factorized routes", compiled.coupler.factorized_coefficient_tables[0]["kind"])
print("coefficient hash", compiled.coupler.cache_key())
print("execution plan hash", execution_plan.plan_hash)
