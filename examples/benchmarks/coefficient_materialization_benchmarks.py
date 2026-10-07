"""Write bounded coefficient-materialization benchmark artifacts.

The joint Young/O(3) generic case times the dense reference constructor for
validation and backend comparison. It is not a production descriptor-runtime
benchmark; use the ACE or Cauchy factorized paths for practical workloads.
"""

# coefficient_materialization_benchmarks.py
# Time exact, numeric-cached, and native coefficient materialization for the
# rotation-only, permutation-only, and joint Young/O(3) benchmark families and
# write the rows, a CSV table, and an optional plot under output_dir.

from pathlib import Path

from ye3t.coefficient_benchmarks import coefficient_materialization_benchmark_artifacts
from ye3t.workflows import merge_workflow_config


# YE3T config dictionary.
cfg_ye3t = {
    "metadata": {
        "status": "stable",  # Bounded benchmark of the documented materialization backends.
        "name": "coefficient_materialization_benchmarks",
        "schema": "ye3t_config_v1",
    },
    "basis": {
        "type": "ACE",
        "coordinates": "coupling_coefficient_tables",  # Rotation, permutation, and joint sectors.
    },
    "representation": {
        "carrier": "ACE_density",  # Ordinary commutative ACE density carrier.
        "coupling": {"source": "ye3t.couplings", "backend": "exact_numeric_native"},
    },
    "runtime": {
        "device": "cpu",
        "timeout_seconds": 300.0,  # Upper bound for the full benchmark set.
        "benchmark": {
            "output_dir": str(Path.home() / "ye3t-workflows" / "benchmarks" / "coefficient_materialization"),
            "repeat": 1,
            "make_plots": True,
            "include_optional_external": True,
        },
    },
    "model": {"type": "none"},  # Coefficient tables only; no fitted model.
    "targets": {"energy": "none"},
    "validation": {"checks": ["coefficient_certificates", "backend_agreement"]},
}


def run_coefficient_materialization_benchmarks(config=None):
    """Run representative coefficient-materialization benchmarks and write artifacts."""
    settings = merge_workflow_config(cfg_ye3t, config)
    benchmark = settings["runtime"]["benchmark"]
    output_dir = Path(benchmark["output_dir"])
    artifact = coefficient_materialization_benchmark_artifacts(output_dir, benchmark)
    print("coefficient materialization benchmark artifact")
    print("rows:", len(artifact["rows"]))
    print("json:", artifact["files"]["json"])
    print("csv:", artifact["files"]["csv"])
    print("plot:", artifact["files"]["plot"] or artifact["files"]["plot_report"]["reason"])
    for row in artifact["rows"]:
        print(
            row["name"],
            row["benchmark_family"],
            row["construction_mode"],
            row.get("validation_passed", row.get("status", "not_run")),
        )


if __name__ == "__main__":
    run_coefficient_materialization_benchmarks(config=cfg_ye3t)
