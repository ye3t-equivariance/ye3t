"""Count and optionally compile scalar tagged Cauchy coordinates across ranks 1–8.

The catalogue combines radial channels, vector pairs, and scalar powers.
Every coordinate in each configured content is considered during exact
physical-image reduction. The rank-four-only fit and its coordinate system
are shown separately in tagged_cauchy_linear_coordinates.py.
"""

from collections import Counter

from ye3t import YE3TRepresentation, couplings
from ye3t.workflows import merge_workflow_config


cfg_ye3t = {
    "metadata": {"schema": "ye3t_config_v1", "name": "tagged_cauchy_image_catalogue", "status": "experimental"},
    "representation": {
        "group": "O3", "ranks": [1, 2, 3, 4, 5, 6, 7, 8],
        "parent": {"young_lambda": "(N)", "L": 0, "parity": "even"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {1: 1, 2: 1, 3: 1, 4: 2, 5: 1, 6: 1, 7: 1, 8: 1},
            "l_max_per_rank": {1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1},
        },
        "intermediates": {"young_kappa": "all_valid", "block_rotation": {"policy": "all_valid"}},
    },
    "basis": {
        "single_factors": {
            "species": ["Ta"], "radial": {"family": "shifted_jacobi", "cutoff_A": 4.8},
            "chemical": {"kind": "explicit"},
        },
        "tensor_product": {
            "kind": "tagged",
            "tag_counts_per_rank": {1: [1], 2: [2], 3: [2], 4: [2], 5: [2], 6: [2], 7: [2], 8: [2]},
        },
        "catalogue": {
            "ranks": [1, 2, 3, 4, 5, 6, 7, 8],
            "nmax_per_rank": {1: 1, 2: 1, 3: 1, 4: 2, 5: 1, 6: 1, 7: 1, 8: 1},
            "lmax_per_rank": {1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1},
            "source_block_partitions_by_rank": {
                1: [[1]], 2: [[2]], 3: [[2, 1]], 4: [[2, 2]],
                5: [[3, 2]], 6: [[4, 2]], 7: [[5, 2]], 8: [[6, 2]],
            },
            "angular_patterns_by_rank": {
                1: [[0]], 2: [[1, 1]], 3: [[0, 1, 1]], 4: [[1, 1, 1, 1]],
                5: [[0, 0, 0, 1, 1]], 6: [[0, 0, 0, 0, 1, 1]],
                7: [[0, 0, 0, 0, 0, 1, 1]], 8: [[0, 0, 0, 0, 0, 0, 1, 1]],
            },
            "max_records_per_rank": 256, "max_features_per_rank": 256,
            "angular_basis_backend": "exact_weight_space_v1",
        },
    },
    "runtime": {"backend": "exact_physical_image", "materialize_coefficients": False},
    "model": {"kind": "linear"},
    "targets": {"energy": "energy", "forces": "forces", "stress": "stress"},
    "validation": {"checks": ["exact_count", "physical_image", "coordinate_identity"]},
}



def run_tagged_cauchy_image_catalogue(config=None):
    """Count and plan the catalogue; optionally compile its complete physical image."""
    settings = merge_workflow_config(cfg_ye3t, config)
    representation = YE3TRepresentation.from_config(settings["representation"])
    # This physical-source compiler supports globally symmetric even scalar parents.
    if (representation.group != "O3" or representation.L != 0 or representation.parity != "even"
            or representation.factorization != "cauchy" or representation.subspace != "full"
            or representation.young_kappa[0] != "all_valid" or representation.block_rotation[0] != "all_valid"
            or any(representation.parent_partition(rank) != (rank,) for rank in representation.ranks)):
        raise ValueError("This physical-image compiler requires full scalar symmetric O3 Cauchy sectors.")
    catalogue = dict(settings["basis"]["catalogue"])
    if (tuple(catalogue.pop("ranks")) != representation.ranks
            or catalogue["nmax_per_rank"] != dict(representation.eta_count_per_rank)
            or catalogue["lmax_per_rank"] != dict(representation.l_max_per_rank)):
        raise ValueError("The source catalogue and representation rank/channel settings must match.")
    if settings["basis"]["single_factors"]["radial"]["family"] != "shifted_jacobi":
        raise ValueError("This exact physical-image compiler requires shifted_jacobi sources.")
    if (settings["basis"]["tensor_product"]["kind"] != "tagged"
            or settings["basis"]["single_factors"]["chemical"]["kind"] != "explicit"
            or settings["runtime"]["backend"] != "exact_physical_image"):
        raise ValueError("Select tagged factors, explicit species, and the exact_physical_image backend.")
    catalogue["tag_counts_by_rank"] = settings["basis"]["tensor_product"]["tag_counts_per_rank"]
    request = couplings.tagged_cauchy_image_request(
        catalogue=catalogue, species=settings["basis"]["single_factors"]["species"])
    report = couplings.count(request)
    compiler_plan = couplings.plan(report)
    print("formal Cauchy coordinates by rank:",
          dict(sorted(Counter(row["tensor_order"] for row in report.labels).items())))

    if settings["runtime"]["materialize_coefficients"]:
        artifact = couplings.compile(compiler_plan)
        coordinates = artifact.payload["image_coordinate_provenance"]
        print("independent physical coordinates by rank:",
              dict(sorted(Counter(row["tensor_order"] for row in coordinates).items())))
        print("total physical coordinates:", len(coordinates))
        print("artifact hash:", artifact.self_hash)
    return {
        "representation": representation, "catalogue": catalogue, "report": report,
        "compiler_plan": compiler_plan,
        "artifact": artifact if settings["runtime"]["materialize_coefficients"] else None,
    }


if __name__ == "__main__":
    run_tagged_cauchy_image_catalogue(config=cfg_ye3t)
