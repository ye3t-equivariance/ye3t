"""Apply YE3T's ACE factorized schedule to supplied density multiplets.

Each input is a complete ``u_{a l m}`` multiplet supplied by another code.
Here the channel types differ, so each adapted block has size one. For
repeated channel types, supply the block-adapted values specified by the
compiled schedule. Magnetic indices use complex Condon-Shortley order
``m=-l,...,l``. YE3T constructs the independent scalar coupling.
"""

import torch

from ye3t import YE3TRepresentation, couplings


cfg_ye3t = {
    "metadata": {
        "schema": "ye3t_config_v1", "name": "symmetric_density_factors",
        "status": "experimental",
    },
    "representation": {
        "group": "O3", "ranks": [2],
        "parent": {"young_lambda": "(N)", "L": 0, "parity": "even"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {2: 2}, "l_max_per_rank": {2: 1},
        },
        "intermediates": {
            "young_kappa": "all_valid", "block_rotation": {"policy": "all_valid"},
        },
    },
    "basis": {
        "carrier": "ACE_density", "fixed_content": [1, 2], "input_Ls": [1, 1],
        "density_multiplets": [[1.0, 0.2, -1.0], [0.3, -0.7, -0.3]],
    },
    "runtime": {"dtype": "complex128", "device": "cpu", "backend": "auto"},
    "model": {}, "targets": {},
    "validation": {"checks": ["independent_count", "factor_exchange"]},
}


representation = YE3TRepresentation.from_config(cfg_ye3t["representation"])
report = representation.count_fixed_content(
    cfg_ye3t["basis"]["fixed_content"],
    cfg_ye3t["basis"]["input_Ls"],
    carrier=cfg_ye3t["basis"]["carrier"],
)
compiled = couplings.compile_ace_factorized_schedules_by_L(
    content=report.content, input_Ls=cfg_ye3t["basis"]["input_Ls"],
)
dtype = getattr(torch, cfg_ye3t["runtime"]["dtype"])
blocks = torch.tensor(
    [[cfg_ye3t["basis"]["density_multiplets"]]],
    dtype=dtype, device=cfg_ye3t["runtime"]["device"],
)
feature = compiled.evaluate(
    blocks, target_L=representation.L, backend=cfg_ye3t["runtime"]["backend"],
)
exchanged = compiled.evaluate(
    blocks.flip(2), target_L=representation.L,
    backend=cfg_ye3t["runtime"]["backend"],
)
assert compiled.validation_report["passed"]
assert compiled.schedules_by_L[representation.L].basis_count == report.counts_by_target[representation.L]
torch.testing.assert_close(feature, exchanged, atol=1e-12, rtol=1e-12)

print("independent symmetric labels", report.counts_by_target[representation.L])
print("feature shape", tuple(feature.shape))
print("factor exchange invariant", True)
print("coefficient path", compiled.backend)
