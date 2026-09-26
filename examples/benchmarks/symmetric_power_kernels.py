"""Compare optional symmetric-power kernels with reference evaluation."""

# symmetric_power_kernels.py
# Compare folded symmetric-power monomial kernels with the unfused reference.
# Use this to check which repeated angular inputs have accelerated paths and
# whether forward outputs match the reference evaluator.

import torch

from ye3t.utils.illustrators import (
    compare_symmetric_power,
    compare_symmetric_square,
    print_symmetric_power_output_catalog,
)
from ye3t.workflows import merge_workflow_config


# YE3T config dictionary.
cfg_ye3t = {
    "metadata": {
        "status": "stable",  # Bounded correctness and timing benchmark of optional kernels.
        "name": "symmetric_power_kernels",
        "config_schema": "ye3t_example_config_v1",
    },
    "basis": {
        "type": "angular_symmetric_power",  # Symmetric tensor powers of one angular input.
    },
    "representation": {
        "carrier": "angular_features",  # Batched O(3) feature rows, no site density.
        "target": {"permutation": "trivial", "parity": "even"},
        "coupling": {"source": "ye3t.couplings", "backend": "exact", "fast_path": "auto"},
    },
    "runtime": {
        "device": "cpu",  # Timing runs on one CPU thread for reproducibility.
        "threads": 1,
        "catalog_cases": ((2, (1, 2, 3)), (3, (1, 2)), (4, (1, 2)), (8, (1,))),  # Powers and input L values to list.
        "square_case": {  # Fast symmetric-square correctness/timing case.
            "L": 2,  # Input angular momentum.
            "output_L": 4,  # Target angular momentum after symmetric square.
            "seed": 41,  # Random seed for generated features.
            "batch": 4096,  # Number of feature rows used in timing.
            "warmups": 3,  # Warmup evaluations before timing.
            "runs": 12,  # Timed evaluations.
        },
        "power_cases": (  # Higher-power correctness/timing cases.
            {
                "power": 3,  # Symmetric tensor power.
                "L": 2,  # Input angular momentum.
                "output_L": 6,  # Target angular momentum.
                "seed": 43,  # Random seed for generated features.
                "batch": 1024,  # Number of feature rows used in timing.
                "warmups": 2,  # Warmup evaluations before timing.
                "runs": 5,  # Timed evaluations.
            },
            {
                "power": 4,  # Symmetric tensor power.
                "L": 2,  # Input angular momentum.
                "output_L": 4,  # Target angular momentum.
                "seed": 44,  # Random seed for generated features.
                "batch": 512,  # Number of feature rows used in timing.
                "warmups": 2,  # Warmup evaluations before timing.
                "runs": 3,  # Timed evaluations.
            },
            {
                "power": 8,  # Symmetric tensor power.
                "L": 1,  # Input angular momentum.
                "output_L": 8,  # Target angular momentum.
                "seed": 48,  # Random seed for generated features.
                "batch": 256,  # Number of feature rows used in timing.
                "warmups": 2,  # Warmup evaluations before timing.
                "runs": 3,  # Timed evaluations.
            },
        ),
    },
    "model": {"type": "none"},  # No fitted model; kernels are compared on random features.
    "targets": {"energy": "none"},
    "validation": {"checks": ["forward_agreement_with_reference", "timing_ratio"]},
}


def run_symmetric_power_kernels(config=None):
    """Run bounded symmetric-power correctness and timing comparisons."""
    settings = merge_workflow_config(cfg_ye3t, config)
    runtime = settings["runtime"]
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(int(runtime["threads"]))
    try:
        print("accelerated policy:", "auto")
        print("reference policy:", "off")
        print_symmetric_power_output_catalog(runtime["catalog_cases"])
        compare_symmetric_square(runtime["square_case"])
        for case in runtime["power_cases"]:
            compare_symmetric_power(case)
    finally:
        torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    run_symmetric_power_kernels(config=cfg_ye3t)
