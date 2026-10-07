"""Count and apply a Cauchy coupling to user-supplied role densities.

The two rows of ``basis.role_density`` are separate role channels of one
complete l=1 multiplet. This rank-two example selects the nontrivial local
Young path kappa=(1,1), whose angular output is an L=1 axial vector. The
global parent remains permutation invariant. Replace these rows with your
own role-resolved density values in YE3T's real tesseral order:
cosine components l,...,1, then m=0, then sine components 1,...,l.
"""

import numpy as np

from ye3t import couplings


cfg_ye3t = {
    "metadata": {
        "schema": "ye3t_config_v1", "name": "cauchy_supplied_density",
        "status": "experimental",
    },
    "representation": {
        "group": "O3", "ranks": [2],
        "parent": {"young_lambda": "(N)", "L": 1, "parity": "even"},
        "factorization": "cauchy", "subspace": "full",
        "intermediates": {"young_kappa": "all"},
    },
    "basis": {
        "tensor_product": {"kind": "role_density", "role_dimension": 2},
        "channel": {
            "neighbor_species": "X", "radial_channel": 0, "l": 1,
            "source_family_id": "user_supplied",
        },
        "block_size": 2,
        "role_density": [[0.2, 0.5, -0.3], [-0.7, 0.1, 0.4]],
    },
    "runtime": {"dtype": "float64"},
    "model": {}, "targets": {},
    "validation": {"checks": ["coefficient_certificate", "finite_values"]},
}


request = couplings.covariant_cauchy_request(
    (cfg_ye3t["basis"]["channel"],),
    (cfg_ye3t["basis"]["block_size"],),
    target_L=cfg_ye3t["representation"]["parent"]["L"],
    target_parity=1 if cfg_ye3t["representation"]["parent"]["parity"] == "even" else -1,
    role_dimension=cfg_ye3t["basis"]["tensor_product"]["role_dimension"],
    kappa_policy=cfg_ye3t["representation"]["intermediates"]["young_kappa"],
)
count = couplings.count(request)
compiled = couplings.compile(request)
role_density = np.asarray(
    cfg_ye3t["basis"]["role_density"], dtype=cfg_ye3t["runtime"]["dtype"]
)
values, _ = couplings.evaluate_covariant_cauchy(compiled, {0: role_density})

assert compiled["validation_report"]["passed"]
assert np.isfinite(values).all()
assert all(label["block_kappas"] == ((1, 1),) for label in count["labels"])

print("Cauchy count", count["multiplet_count"])
print("local Young path", count["labels"][0]["block_kappas"])
print("output shape", values.shape)
print("compiled coefficients valid", compiled["validation_report"]["passed"])
