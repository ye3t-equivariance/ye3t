"""Export all scalar ACE couplings for selected contents at ranks 1–8.

The additional homogeneous rank-16 case exercises the symmetric-power path.
ye3t-methods supplies fitted weights and physical source bindings for LAMMPS.
"""

from pathlib import Path

from ye3t import YE3TRepresentation
from ye3t.couplings import YE3T_O3_PRIMARY_CONVENTION
from ye3t.scalar_ace_export import export_scalar_ace_plans
from ye3t.workflows import count_coupling_cases, merge_workflow_config


cfg_ye3t = {
    "metadata": {"schema": "ye3t_config_v1", "name": "compile_scalar_ace_lammps_plans", "status": "experimental"},
    "representation": {
        "group": "O3", "ranks": [1, 2, 3, 4, 5, 6, 7, 8],
        "parent": {"young_lambda": "(N)", "L": 0, "parity": "even"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3},
            "l_max_per_rank": {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2},
        },
        "intermediates": {"young_kappa": "all_valid", "block_rotation": {"policy": "all_valid"}},
    },
    "basis": {
        "carrier": "ACE_density", "species": ["Ta"], "radial": {"family": "pace_chebexpcos"},
        "cases_by_rank": {
            1: [{"name": "ta_r01_n1", "content": [1], "input_Ls": [0]},
                {"name": "ta_r01_n2", "content": [2], "input_Ls": [0]}],
            2: [{"name": "ta_r02_n12", "content": [1, 2], "input_Ls": [1, 1]},
                {"name": "ta_r02_n11_l22", "content": [1, 1], "input_Ls": [2, 2]}],
            3: [{"name": "ta_r03_n223_l112", "content": [2, 2, 3], "input_Ls": [1, 1, 2]}],
            4: [{"name": "ta_r04_block_2_2", "content": [1, 1, 2, 2], "input_Ls": [1, 1, 1, 1]},
                {"name": "ta_h04", "content": [1, 1, 1, 1], "input_Ls": [1, 1, 1, 1]}],
            5: [{"name": "ta_r05_block_2_2_1", "content": [1, 1, 2, 2, 3], "input_Ls": [1, 1, 1, 1, 2]}],
            6: [{"name": "ta_r06_block_3_3", "content": [1, 1, 1, 2, 2, 2], "input_Ls": [1, 1, 1, 1, 1, 1]}],
            7: [{"name": "ta_r07_block_3_3_1", "content": [1, 1, 1, 2, 2, 2, 3], "input_Ls": [1, 1, 1, 1, 1, 1, 2]}],
            8: [{"name": "ta_r08_block_4_4", "content": [1, 1, 1, 1, 2, 2, 2, 2], "input_Ls": [1, 1, 1, 1, 1, 1, 1, 1]}],
        },
        "additional_coordinates": [{
            "name": "ta_h16", "membership_mode": "constructive_factorized",
            "label": {"n_tuple": [1] * 16, "l_tuple": [1] * 16, "internal_Ls": [0],
                      "L_R": 0, "tree_type": "balanced", "basis_key": ["sym", 0, 0]},
        }],
    },
    "runtime": {
        "constructor_backend": "python", "dtype": "float64", "device": "cpu",
        "maximum_unique_monomials": 250000, "maximum_term_contributions": 2000000,
        "maximum_coordinate_bytes": 128 * 1024 * 1024,
        "output_dir": str(Path.home() / "ye3t-workflows" / "benchmarks" / "scalar_ace_lammps_plans"),
        "coupled_product_request": {
            "nin": (2, 2, 3),
            "lin": (1, 1, 2),
            "target_basis_index": 0,
            "source_channels": (
                {
                    "channel_id": "Ta:Ta:pace:n2:l1",
                    "content_token": 2,
                    "central_species": "Ta",
                    "neighbor_species": "Ta",
                    "radial_basis_id": "pace_chebexpcos",
                    "radial_index": 1,
                    "l": 1,
                    "convention_id": YE3T_O3_PRIMARY_CONVENTION,
                },
                {
                    "channel_id": "Ta:Ta:pace:n3:l2",
                    "content_token": 3,
                    "central_species": "Ta",
                    "neighbor_species": "Ta",
                    "radial_basis_id": "pace_chebexpcos",
                    "radial_index": 2,
                    "l": 2,
                    "convention_id": YE3T_O3_PRIMARY_CONVENTION,
                },
            ),
            "source_model_id": "ta_scalar_ace_example",
            "yace_function_id": "ta_r03_n223_l112",
            "direct_fallback_binding_id": "ta_r03_n223_l112_ctilde",
            "factorization_policy": "full",
            "id_prefix": "ta_r03_n223_l112_gram",
            "resource_limits": {
                "max_rank": 8,
                "max_target_dimension": 128,
                "max_product_columns": 256,
                "max_nodes": 2048,
                "max_static_bytes": 16 * 1024 * 1024,
                # Linux counts pages private to the forked compiler worker.
                "max_compile_peak_bytes": 1024 * 1024 * 1024,
                "max_serialized_plan_bytes": 16 * 1024 * 1024,
                "max_compile_seconds": 60.0,
            },
        },
    },
    "model": {"type": "linear_ace_execution_plan", "readout": "external_linear_coefficients"},
    "targets": {"energy": "scalar_site_energy", "forces": "plan_adjoint"},
    "validation": {"checks": ["exact_count", "coordinate_validation", "execution_plan_round_trip", "plan_hash"]},
}


def run_compile_scalar_ace_lammps_plans(config=None):
    """Count, compile, and export every selected scalar coordinate."""
    settings = merge_workflow_config(cfg_ye3t, config)
    representation = YE3TRepresentation.from_config(settings["representation"])
    cases = count_coupling_cases(representation, settings["basis"])
    print("multiplicity counts (rank, case):",
          {key: case["multiplicity_count"] for key, case in cases.items()})
    manifest = export_scalar_ace_plans(cases, settings)
    print("Wrote", manifest)
    return manifest


if __name__ == "__main__":
    run_compile_scalar_ace_lammps_plans(config=cfg_ye3t)
