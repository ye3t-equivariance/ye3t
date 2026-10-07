"""Every permutation-antisymmetric multiplet from ordered polar factors.

Edit ``parent.L`` to any reachable value. Repeated factor types retain
ordered values, so the alternating block is nonzero when those values are
independent. This is not an ordinary commutative density.
"""

import torch

from ye3t import YE3TRepresentation, couplings


cfg_ye3t = {
    "metadata": {"schema": "ye3t_config_v1", "name": "antisymmetric_external_factors",
                 "status": "experimental"},
    "representation": {
        "group": "O3", "ranks": [3],
        "parent": {"young_lambda": "antisymmetric", "L": 1, "parity": "odd"},
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
                "subduction_backend": "exact"},
    "model": {}, "targets": {},
    "validation": {"checks": ["exact_count", "permutation_antisymmetry"]},
}


representation = YE3TRepresentation.from_config(cfg_ye3t["representation"])
report = representation.count_fixed_content(
    cfg_ye3t["basis"]["fixed_content"], cfg_ye3t["basis"]["input_Ls"],
    carrier=cfg_ye3t["basis"]["carrier"],
)
compiled = couplings.compile(
    couplings.plan(report),
    subduction_materialization_backend=cfg_ye3t["runtime"]["subduction_backend"],
)
factors = torch.tensor(
    cfg_ye3t["basis"]["factor_multiplets"],
    dtype=getattr(torch, cfg_ye3t["runtime"]["dtype"]),
    device=cfg_ye3t["runtime"]["device"],
)
runtime = compiled.coupler.bind_factorized_factors_torch(
    dtype=factors.dtype, device=factors.device,
)
values = runtime.evaluate(factors)
assert values.abs().max() > 1e-8
swapped = runtime.evaluate(factors[[1, 0, 2]])
torch.testing.assert_close(values, -swapped, atol=1e-12, rtol=1e-12)
crossed = runtime.evaluate(
    factors[[2, 1, 0]], factor_types=((2, 1), (1, 1), (1, 1)),
)
torch.testing.assert_close(values, -crossed, atol=1e-12, rtol=1e-12)
duplicated = factors.clone()
duplicated[1] = duplicated[0]
zero = runtime.evaluate(duplicated)
torch.testing.assert_close(zero, torch.zeros_like(zero), atol=1e-12, rtol=0)
assert values.shape == (report.counts_by_target[representation.L], 1,
                        2 * representation.L + 1)
assert not compiled.coupler.sparse_coefficient_tables

print("independent multiplicity count", report.counts_by_target[representation.L])
print("coupled axes (a, t, M)", tuple(values.shape))
print("repeated-block kernel", compiled.coupler.factorized_coefficient_tables[0]
      ["local_tables"][0]["strategy"])
print("coefficient hash", compiled.coupler.cache_key())
