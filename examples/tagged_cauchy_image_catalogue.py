"""Count a bounded tagged-Cauchy physical image from two radial sources.

This source-specific catalogue is experimental. Set
``runtime.materialize_coefficients`` to compile its physical-image map;
the default counts and plans without writing a compiler cache artifact.
The optional Racah plan requires SymPy.
"""

from ye3t import couplings


cfg_ye3t = {
    "metadata": {
        "schema": "ye3t_config_v1", "name": "tagged_cauchy_image_catalogue",
        "status": "experimental",
    },
    "representation": {
        "group": "O3", "ranks": [4],
        "parent": {"young_lambda": "(N)", "L": 0, "parity": "even"},
        "factorization": "cauchy", "subspace": "full",
        "intermediates": {"young_kappa": "all_valid"},
    },
    "basis": {
        "tensor_product": {
            "kind": "tagged_cauchy_image", "tag_count": 2,
            "maximum_collision_arity": 2,
            "selected_raw_tag_counts": [0, 1, 2],
        },
        "single_factors": {
            "sources": [
                {"neighbor_species": "Ta", "q": 0, "l": 1,
                 "source_family_id": couplings.ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY},
                {"neighbor_species": "Ta", "q": 1, "l": 1,
                 "source_family_id": couplings.ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY},
            ],
        },
    },
    "runtime": {"materialize_coefficients": False, "label_preview": 3},
    "model": {}, "targets": {},
    "validation": {
        "checks": ["record_reconstruction", "request_hash", "dimension_certificate"],
    },
}


source_keys = tuple(cfg_ye3t["basis"]["single_factors"]["sources"])
maximum_collision_arity = cfg_ye3t["basis"]["tensor_product"]["maximum_collision_arity"]
angular_plan = couplings.racah_harmonic_product_plan(
    tuple(sorted({source["l"] for source in source_keys})),
    maximum_collision_arity=maximum_collision_arity,
)
product_record = couplings.build_radial_species_product_record(
    source_keys, angular_plan, maximum_collision_arity=maximum_collision_arity,
)
couplings.validate_radial_species_product_record(product_record, angular_plan)
request = couplings.tagged_cauchy_image_request(
    product_record,
    angular_plan,
    source_keys=source_keys,
    tensor_order=cfg_ye3t["representation"]["ranks"][0],
    tag_count=cfg_ye3t["basis"]["tensor_product"]["tag_count"],
    target_L=cfg_ye3t["representation"]["parent"]["L"],
    selected_raw_tag_counts=tuple(
        cfg_ye3t["basis"]["tensor_product"]["selected_raw_tag_counts"]
    ),
)
report = couplings.count(request)
compiler_plan = couplings.plan(report)

print("tagged image count source: ye3t.couplings.count")
print("product record hash:", product_record["record_hash"])
print("angular plan hash:", angular_plan["plan_hash"])
print("primitive sources:", len(product_record["primitive_source_keys"]))
print("raw label count:", report.raw_label_count)
print("image dimension upper bound:", report.image_dimension_upper_bound)
print("exact image dimension:", report.exact_image_dimension)
print(
    "coefficients materialized:",
    report.resource_report["coefficient_materialization_performed"],
)
print("raw label preview:", tuple(
    dict(label)
    for label in report.labels[:cfg_ye3t["runtime"]["label_preview"]]
))
print("tagged image plan source: ye3t.couplings.plan")
print("materialization steps:", compiler_plan.materialization_steps)
print("plan validation passed:", compiler_plan.validation_report["passed"])

if cfg_ye3t["runtime"]["materialize_coefficients"]:
    artifact = couplings.compile(compiler_plan)
    print("tagged image coefficient source: ye3t.couplings.compile")
    print("compiled coefficients valid:", artifact.payload["certificate"]["passed"])
    print("image rows:", len(artifact.payload["image_rows"]))
    print("artifact hash:", artifact.self_hash)
