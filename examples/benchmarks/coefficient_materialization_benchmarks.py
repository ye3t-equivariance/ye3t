"""Write coefficient-materialization benchmark artifacts."""

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
        "config_schema": "ye3t_example_config_v1",
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
    },
    "model": {"type": "none"},  # Coefficient tables only; no fitted model.
    "targets": {"energy": "none"},
    "validation": {"checks": ["coefficient_certificates", "backend_agreement"]},
    "output_dir": str(Path("examples") / "generated" / "coefficient_materialization_benchmarks"),  # Artifact folder.
    "repeat": 1,  # Timed repetitions per cold/cache row.
    "make_plots": True,  # Write transparent-background matplotlib plots when matplotlib is installed.
    "include_optional_external": True,  # Include external-library comparison placeholders.
}


def run_coefficient_materialization_benchmarks(config=None):
    """Run representative coefficient-materialization benchmarks and write artifacts."""
    settings = merge_workflow_config(cfg_ye3t, config)
    output_dir = Path(settings.pop("output_dir"))
    artifact = coefficient_materialization_benchmark_artifacts(output_dir, settings)
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
