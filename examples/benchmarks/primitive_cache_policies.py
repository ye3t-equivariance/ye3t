"""Compare primitive-cache lookup policies on a small exact sector."""

# primitive_cache_policies.py
# Compare conservative and eager primitive-cache lookup on one exact sector and
# report how many primitive products are reused versus recomputed.

from ye3t.utils.printing import primitive_cache_summary
from ye3t.workflows import merge_workflow_config


# YE3T config dictionary.
cfg_ye3t = {
    "metadata": {
        "status": "stable",  # Diagnostic benchmark on a documented exact sector.
        "name": "primitive_cache_policies",
        "schema": "ye3t_config_v1",
    },
    "basis": {
        "type": "ACE",
        "coordinates": "exact_primitive_quotient",  # Exact product-expansion primitives.
    },
    "representation": {
        "carrier": "ACE_density",  # Ordinary commutative ACE density carrier.
        "target": {"permutation": "trivial", "L": 0, "parity": "even"},
        "coupling": {"source": "ye3t.couplings", "backend": "exact"},
    },
    "runtime": {
        "benchmark": {
          "case": {  # Exact sector used for the cache-policy comparison.
            "nin": (1, 1, 1, 1),  # Non-angular channel eta_i for each slot.
            "lin": (1, 1, 1, 1),  # Angular momentum l_i for each slot.
            "target_L": 0,  # Target output angular momentum L_R.
            "max_cache_rank": 4,  # Highest rank stored as primitive cache entries.
          },
          "policies": ("cached_only", "eager"),  # Lookup policies to compare.
        },
    },
    "model": {"type": "none"},  # No fitted model; this is a cache diagnostic.
    "targets": {"energy": "none"},
    "validation": {"checks": ["cache_policy_counts"]},
}


def run_primitive_cache_policies(config=None):
    """Print primitive-cache behavior for conservative and eager policies."""
    settings = merge_workflow_config(cfg_ye3t, config)
    benchmark = settings["runtime"]["benchmark"]
    case = benchmark["case"]
    nin = tuple(case["nin"])
    lin = tuple(case["lin"])
    target_L = int(case["target_L"])
    print(f"primitive cache sector: nin={nin} lin={lin} L_R={target_L}")
    for policy in benchmark["policies"]:
        row = primitive_cache_summary(case, policy)
        print(
            "policy={policy} sector_dim={sector_dim} primitives={primitive_count} "
            "cached_primitives={primitive_cache_size} cached_only_misses={cached_only_misses} "
            "full_expansions={full_computation_count} first_terms={first_expansion_terms}".format(**row)
        )


if __name__ == "__main__":
    run_primitive_cache_policies(config=cfg_ye3t)
