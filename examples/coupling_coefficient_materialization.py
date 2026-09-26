"""Plan and compile YE3T coupling coefficients for a fixed-content sector."""

# coupling_coefficient_materialization.py
# Use ye3t.couplings.plan and ye3t.couplings.compile to materialize coupling
# coefficients with provenance, convention hash, and validation/certificate
# metadata.

from ye3t import couplings
from ye3t.workflows import merge_workflow_config


# YE3T config dictionary.
cfg_ye3t = {
    "content": (1, 1, 2, 2),  # Fixed non-angular channel content.
    "input_Ls": (1, 1, 2, 2),  # Slot angular momenta for the same content.
    "target_L": 0,  # Target angular momentum L_R.
    "target_permutation": "trivial",  # ACE invariant sector.
    "carrier": "ACE_density",  # Ordinary compact ACE density carrier.
    "subduction_materialization_backend": "numeric_cached",  # Practical default backend.
}


def run_coupling_coefficient_materialization(config=None):
    """Print plan and compile provenance for one coupling sector."""
    settings = merge_workflow_config(cfg_ye3t, config)
    request = {
        "content": tuple(settings["content"]),
        "input_Ls": tuple(settings["input_Ls"]),
        "target_L": int(settings["target_L"]),
        "target_permutation": str(settings["target_permutation"]),
        "carrier": str(settings["carrier"]),
    }
    plan = couplings.plan(**request)
    compiled = couplings.compile(
        **request,
        subduction_materialization_backend=str(settings["subduction_materialization_backend"]),
    )
    certificate = compiled.certificate
    print("coupling plan source:", "ye3t.couplings.plan")
    print("coefficient source:", "ye3t.couplings.compile")
    print("content:", tuple(compiled.content))
    print("carrier:", compiled.carrier)
    print("target:", compiled.target)
    print("planned backend:", plan.backend)
    print("compiled backend:", compiled.backend)
    print("backend plan selected:", compiled.validation_report.get("backend_plan_selected"))
    print("convention hash:", compiled.convention_hash)
    print("validation passed:", bool(compiled.validation_report.get("passed", False)))
    print("certificate passed:", bool(certificate.passed))
    print("coefficient hash:", certificate.coefficient_hash)


if __name__ == "__main__":
    run_coupling_coefficient_materialization(config=cfg_ye3t)
