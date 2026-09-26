"""Count, plan, and optionally compile the bounded tagged-Cauchy physical image."""

# tagged_cauchy_image_catalogue.py
# Build the exact product algebra of a user-chosen shifted-Jacobi source set
# with ye3t.couplings.build_radial_species_product_record, bind it to the Racah
# angular product plan, and let ye3t.couplings.count / plan / compile be the
# single source of the tagged labels and their exact physical image.
# The Racah plan needs the optional sympy dependency; the compile step writes
# its hash-bound artifact to the YE3T artifact cache (YE3T_CACHE_DIR).

from ye3t import couplings
from ye3t.workflows import merge_workflow_config


# YE3T config dictionary.
cfg_ye3t = {
    "metadata": {
        "status": "experimental",  # Bounded N=4, two-tag physical-image rung of the tagged compiler.
        "name": "tagged_cauchy_image_catalogue",
        "config_schema": "ye3t_example_config_v1",
    },
    "basis": {
        "type": "tagged_cauchy_image",  # Tagged-Cauchy source slots with an exact physical image.
        "tensor_order": 4,  # Tensor order N of the bounded public rung.
        "tag_count": 2,  # Two unit tags with the nontrivial sign representation.
    },
    "representation": {
        "carrier": "tagged_source_slots",  # Role-resolved source slots, not a commutative density.
        "species": "Ta",  # Neighbor species of every source function.
        "angular_degrees": (1,),  # Angular degree l of the primitive sources (l=1 is required).
        "radial_degrees": (0, 1),  # Shifted-Jacobi degrees q of the primitive sources.
        "maximum_collision_arity": 2,  # Same-neighbor collisions closed through this arity.
        "selected_raw_tag_counts": (0, 1, 2),  # Raw tag counts kept in the catalogue.
        "target": {"L": 0, "parity": "even"},  # Scalar, even-parity readout.
        "coupling": {"source": "ye3t.couplings", "backend": "exact"},
    },
    "runtime": {
        "materialize_coefficients": False,  # True also runs ye3t.couplings.compile.
        "label_preview": 3,  # Raw labels printed from the count report.
    },
    "model": {"type": "none"},  # Catalogue only; no fitted model.
    "targets": {"energy": "none"},
    "validation": {"checks": ["record_reconstruction", "request_hash", "dimension_certificate"]},
}


def run_tagged_cauchy_image_catalogue(config=None):
    """Print the tagged-image count report, its plan, and optionally its artifact."""
    settings = merge_workflow_config(cfg_ye3t, config)
    representation = settings["representation"]
    runtime = settings["runtime"]

    angular_plan = couplings.racah_harmonic_product_plan(
        tuple(representation["angular_degrees"]),
        maximum_collision_arity=int(representation["maximum_collision_arity"]),
    )
    sources = tuple(
        {
            "neighbor_species": str(representation["species"]),
            "q": int(radial_degree),
            "l": int(angular_degree),
            "source_family_id": couplings.ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY,
        }
        for angular_degree in representation["angular_degrees"]
        for radial_degree in representation["radial_degrees"]
    )
    product_record = couplings.build_radial_species_product_record(
        sources,
        angular_plan,
        maximum_collision_arity=int(representation["maximum_collision_arity"]),
    )
    couplings.validate_radial_species_product_record(product_record, angular_plan)
    request = couplings.tagged_cauchy_image_request(
        product_record,
        angular_plan,
        source_keys=sources,
        tensor_order=int(settings["basis"]["tensor_order"]),
        tag_count=int(settings["basis"]["tag_count"]),
        target_L=int(representation["target"]["L"]),
        selected_raw_tag_counts=tuple(representation["selected_raw_tag_counts"]),
    )

    report = couplings.count(request)
    print("tagged image count source: ye3t.couplings.count")
    print("product record hash:", product_record["record_hash"])
    print("angular plan hash:", angular_plan["plan_hash"])
    print("primitive sources:", len(product_record["primitive_source_keys"]))
    print("raw label count:", int(report.raw_label_count))
    print("image dimension upper bound:", int(report.image_dimension_upper_bound))
    print("exact image dimension:", int(report.exact_image_dimension))
    print(
        "coefficients materialized:",
        bool(report.resource_report["coefficient_materialization_performed"]),
    )
    for index, label in enumerate(report.labels[: max(0, int(runtime["label_preview"]))]):
        print("raw label", index, dict(label))

    compiler_plan = couplings.plan(report)
    print("tagged image plan source: ye3t.couplings.plan")
    print("materialization steps:", tuple(compiler_plan.materialization_steps))
    print("plan validation passed:", bool(compiler_plan.validation_report.get("passed", False)))

    if bool(runtime["materialize_coefficients"]):
        artifact = couplings.compile(compiler_plan)
        payload = artifact.payload
        print("tagged image coefficient source: ye3t.couplings.compile")
        print("certificate passed:", bool(payload["certificate"]["passed"]))
        print("raw rows:", len(payload["raw_rows"]))
        print("image rows:", len(payload["image_rows"]))
        print("artifact hash:", artifact.self_hash)


if __name__ == "__main__":
    run_tagged_cauchy_image_catalogue(config=cfg_ye3t)
