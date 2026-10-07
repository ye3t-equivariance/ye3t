"""Couple ordered role-resolved factors into every Cauchy ``(a,t,M)``.

The rows in ``factor_multiplets`` can come from any source that supplies a
complete role-by-spherical-harmonic multiplet for each ordered factor.  Edit
the channels, repeated-channel sizes, Young parent, and output L together.
The source here is small so the exact compiler and its symmetry checks run
quickly.  This is an experimental factor workflow, not an atomistic model.

Requires PyTorch.  Expected output includes a positive multiplicity count,
a two-dimensional tableau axis, and matching compiler and evaluator shapes.
See ``docs/covariant_cauchy_basis.md`` for the carrier and copy-space formula.
"""

import torch

from ye3t import YE3TRepresentation, couplings
from ye3t.representations.projectors import (
    adjacent_transposition_representation_matrix_numeric,
)


cfg_ye3t = {
    "metadata": {
        "schema": "ye3t_config_v1", "name": "ordered_role_cauchy_factors",
        "status": "experimental",
    },
    "basis": {
        "carrier": "ordered_role",
        "channels": [
            {"neighbor_species": "X", "radial_channel": 0, "l": 1,
             "source_family_id": "example"},
            {"neighbor_species": "X", "radial_channel": 1, "l": 1,
             "source_family_id": "example"},
        ],
        "block_sizes": [2, 1], "role_dimension": 1,
        "role_kappa_policy": "all",
        "factor_multiplets": [
            [[0.2, 0.4, -0.1]],
            [[-0.3, 0.1, 0.7]],
            [[0.8, -0.2, 0.5]],
        ],
    },
    "representation": {
        "group": "O3", "ranks": [3],
        "parent": {"young_lambda": "(2,1)", "L": 1, "parity": "odd"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {3: 2}, "l_max_per_rank": {3: 1},
        },
        "intermediates": {"young_kappa": "all_valid",
                          "block_rotation": {"policy": "all_valid"}},
    },
    "runtime": {"dtype": "complex128", "device": "cpu",
                "magnetic_basis": "complex_condon_shortley"},
    "model": {}, "targets": {},
    "validation": {"checks": ["exact_count", "artifact_hash",
                              "factor_permutation"]},
}


representation = YE3TRepresentation.from_config(cfg_ye3t["representation"])
if representation.young_kappa[0] != "all_valid":
    raise ValueError("This example expects all valid local block Young parents.")
parent = representation.parent_partition(representation.ranks[0])
request = couplings.covariant_cauchy_request(
    cfg_ye3t["basis"]["channels"], cfg_ye3t["basis"]["block_sizes"],
    target_L=representation.L,
    target_parity=(-1 if representation.parity == "odd" else 1),
    target_permutation="young:" + ",".join(str(part) for part in parent),
    role_dimension=cfg_ye3t["basis"]["role_dimension"],
    kappa_policy=cfg_ye3t["basis"]["role_kappa_policy"],
    carrier=cfg_ye3t["basis"]["carrier"],
)
report = couplings.count(request)
compiled = couplings.compile(couplings.plan(report))
assert couplings.validate_covariant_cauchy(compiled)

factors = torch.tensor(
    cfg_ye3t["basis"]["factor_multiplets"],
    dtype=getattr(torch, cfg_ye3t["runtime"]["dtype"]),
    device=cfg_ye3t["runtime"]["device"],
)
runtime = couplings.bind_ordered_role_cauchy_torch(
    compiled, dtype=factors.dtype, device=factors.device,
    basis=cfg_ye3t["runtime"]["magnetic_basis"],
)
values, _ = runtime.evaluate(factors)
assert values.shape == (
    report["multiplet_count"], report["tableau_count"],
    2 * representation.L + 1,
)
swapped, _ = runtime.evaluate(factors[[1, 0, 2]])
action = torch.as_tensor(
    adjacent_transposition_representation_matrix_numeric(parent, 0),
    dtype=values.dtype, device=values.device,
)
torch.testing.assert_close(
    swapped, torch.einsum("tu,aum->atm", action, values),
    atol=1e-11, rtol=1e-11,
)

print("independent Cauchy copies", report["multiplet_count"])
print("coupled axes (a, t, M)", tuple(values.shape))
print("artifact hash", compiled["self_hash"])
