"""Compile scalar tagged Cauchy coordinates for a later linear potential.

This uses the same rank-four, two-tag catalogue as the fitted Ta example in
``ye3t-methods/examples/quickstart/tagged_fit.py``. YE3T owns the independent
coordinate labels and exact physical image; ye3t-methods turns the matched
catalogue into ASE energy, force, and stress rows for fitting.
"""

from ye3t import YE3TRepresentation, couplings


cfg_ye3t = {
    "metadata": {
        "schema": "ye3t_config_v1", "name": "tagged_cauchy_linear_coordinates",
        "status": "experimental",
    },
    "representation": {
        "group": "O3", "ranks": [4],
        "parent": {"young_lambda": "(N)", "L": 0, "parity": "even"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {4: 2}, "l_max_per_rank": {4: 1},
        },
        "intermediates": {
            "young_kappa": "all_valid", "block_rotation": {"policy": "all_valid"},
        },
    },
    "basis": {
        "single_factors": {
            "species": ["Ta"],
            "radial": {"family": "shifted_jacobi", "cutoff_A": 4.8},
            "chemical": {"kind": "explicit"},
        },
        "tensor_product": {"kind": "tagged", "tag_counts_per_rank": {4: [2]}},
        "catalogue": {
            "ranks": [4], "nmax_per_rank": {4: 2}, "lmax_per_rank": {4: 1},
            "source_block_partitions_by_rank": {4: [[2, 2]]},
            "angular_patterns_by_rank": {4: [[1, 1, 1, 1]]},
            "max_records_per_rank": {4: 6},
            "max_features_per_rank": {4: 81},
            "angular_basis_backend": "exact_weight_space_v1",
        },
    },
    "runtime": {"backend": "exact_physical_image"},
    "model": {"kind": "linear"},
    "targets": {"energy": "energy", "forces": "forces", "stress": "stress"},
    "validation": {"checks": ["exact_count", "physical_image", "coordinate_identity"]},
}


representation = YE3TRepresentation.from_config(cfg_ye3t["representation"])
catalogue = {
    "nmax_per_rank": cfg_ye3t["basis"]["catalogue"]["nmax_per_rank"],
    "lmax_per_rank": cfg_ye3t["basis"]["catalogue"]["lmax_per_rank"],
    "source_block_partitions_by_rank": cfg_ye3t["basis"]["catalogue"]["source_block_partitions_by_rank"],
    "angular_patterns_by_rank": cfg_ye3t["basis"]["catalogue"]["angular_patterns_by_rank"],
    "tag_counts_by_rank": cfg_ye3t["basis"]["tensor_product"]["tag_counts_per_rank"],
    "max_records_per_rank": cfg_ye3t["basis"]["catalogue"]["max_records_per_rank"],
    "max_features_per_rank": cfg_ye3t["basis"]["catalogue"]["max_features_per_rank"],
    "angular_basis_backend": cfg_ye3t["basis"]["catalogue"]["angular_basis_backend"],
}
request = couplings.tagged_cauchy_image_request(
    catalogue=catalogue,
    species=cfg_ye3t["basis"]["single_factors"]["species"],
)
report = couplings.count(request)
compiler_plan = couplings.plan(report)
compiled = couplings.compile(compiler_plan)
coordinates = compiled.payload["image_coordinate_provenance"]
assert representation.L == 0
assert compiler_plan.validation_report["passed"]
assert compiled.validation_report["passed"]
assert len(coordinates) == len(compiled.payload["image_rows"])
assert all(item["coordinate_id"] in {label["coordinate_id"] for label in report.labels}
           for item in coordinates)
joint = next(
    item for item in coordinates
    if tuple(tuple(kappa) for kappa in item["label"]["block_kappas"])
    == ((1, 1), (1, 1))
    and tuple(item["label"]["block_Lambdas"]) == (1, 1)
)

print("parent Young and L", representation.parent_partitions[0][1], representation.L)
print("compiler raw coordinates", len(report.labels))
print("independent physical coordinates", len(coordinates))
print("first coordinate", coordinates[0]["coordinate_id"])
print("nontrivial local Young coordinate", joint["coordinate_id"])
print("coefficient path", compiled.provenance["coefficient_compiler"])
print("artifact hash", compiled.self_hash)
print("request hash", request["request_hash"])
