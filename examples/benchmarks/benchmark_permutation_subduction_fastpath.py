"""Benchmark experimental numeric permutation-subduction construction.

This script compares the exact symbolic definition path with the experimental
numeric generator-nullspace path for Young-subgroup families.  The numeric path
is validated by generator residuals and, for small cases, exact restricted
projector comparison.  It is not an analytical multiplicity-basis convention.
"""

from pathlib import Path
import time

from ye3t import Partition, validate_young_orthogonal_nary_subduction, young_orthogonal_nary_subduction
from ye3t.representations.numeric_subduction import numeric_subduction_nullspace
from ye3t.workflows import merge_workflow_config


# YE3T config dictionary.
cfg_ye3t = {
    "metadata": {
        "name": "benchmark_permutation_subduction_fastpath",
        "package": "ye3t",
        "status": "experimental",
        "example_role": "benchmark",
        "schema": "ye3t_config_v1",
    },
    "basis": {
        "carrier": "abstract_young_tensor",
    },
    "representation": {
        "group": "SN",
        "sectors": "runtime.benchmark.cases",
        "coupling_source": "ye3t",
    },
    "runtime": {
        "device": "cpu",
        "backend": "ye3t.couplings",
        "timeout_seconds": 300.0,
        "benchmark": {
            "machine_metadata": False,
            "include_slow": False,
            "cache_dir": str(Path.home() / "ye3t-workflows" / "benchmarks" / "permutation_subduction_cache"),
            "constraint_backend": "auto",
            "cases": (
                {"name": "S1xS1xS1_to_21", "subgroup_partitions": ((1,), (1,), (1,)),
                 "target_partition": (2, 1), "compare_exact_projector": True,
                 "slow": False},
                {"name": "S2xS2_to_31", "subgroup_partitions": ((2,), (2,)),
                 "target_partition": (3, 1), "compare_exact_projector": True,
                 "slow": False},
                {"name": "S2xS2xS2_to_42", "subgroup_partitions": ((2,), (2,), (2,)),
                 "target_partition": (4, 2), "compare_exact_projector": False,
                 "slow": True},
                {"name": "S3xS3_to_42", "subgroup_partitions": ((3,), (3,)),
                 "target_partition": (4, 2), "compare_exact_projector": False,
                 "slow": True},
            ),
        },
    },
    "model": {},
    "targets": {},
    "validation": {
        "quick_variant_required_for_long_runtime": True,
        "full_workflow_timeout_seconds": 300.0,
    },
}


def _time_exact(subgroup_partitions, target_partition):
    start = time.perf_counter()
    tensor = young_orthogonal_nary_subduction(
        tuple(Partition(parts) for parts in subgroup_partitions),
        Partition(target_partition),
        coefficient_backend="subduction_graph",
    )
    report = validate_young_orthogonal_nary_subduction(tensor)
    return time.perf_counter() - start, tensor.multiplicity, bool(report.passed)


def _time_numeric(case, cache_dir, constraint_backend):
    start = time.perf_counter()
    result = numeric_subduction_nullspace(
        case["subgroup_partitions"],
        case["target_partition"],
        constraint_backend=constraint_backend,
        cache_dir=cache_dir,
        compare_exact_projector=bool(case.get("compare_exact_projector", False)),
    )
    return time.perf_counter() - start, result


def run_permutation_subduction_fastpath_benchmark(config=None):
    """Run bounded exact-vs-numeric permutation-subduction benchmark cases."""
    settings = merge_workflow_config(cfg_ye3t, config)
    benchmark = settings["runtime"]["benchmark"]
    cache_dir = Path(benchmark["cache_dir"])
    constraint_backend = str(benchmark["constraint_backend"])
    if constraint_backend not in {"auto", "python", "cpp"}:
        raise ValueError("constraint_backend must be one of {'auto', 'python', 'cpp'}.")
    print("case,exact_s,numeric_s,numeric_cached_s,multiplicity,backend,valid,cache_status")
    for case in benchmark["cases"]:
        if bool(case.get("slow", False)) and not bool(benchmark["include_slow"]):
            continue
        exact_s, exact_mult, exact_passed = _time_exact(case["subgroup_partitions"], case["target_partition"])
        numeric_s, numeric = _time_numeric(case, cache_dir, constraint_backend)
        cached_s, cached = _time_numeric(case, cache_dir, constraint_backend)
        valid = bool(exact_passed and numeric.validation["passed"] and int(numeric.multiplicity) == int(exact_mult))
        print(
            f"{case['name']},{exact_s:.6f},{numeric_s:.6f},{cached_s:.6f},"
            f"{numeric.multiplicity},{numeric.constraint_backend},{valid},{cached.cache_status}"
        )


if __name__ == "__main__":
    run_permutation_subduction_fastpath_benchmark(config=cfg_ye3t)
