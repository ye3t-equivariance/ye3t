"""Compile representative scalar ACE coordinates and YE3T execution plans."""

# compile_scalar_ace_lammps_plans.py
# Compile four representative Ta scalar ACE coordinates (rank-3 n223/l112,
# homogeneous rank 4, the [4,4] rank-8 block pair, and homogeneous rank 16)
# into exact coefficient tables and repeated-angular-block execution plans,
# plus a rank-3 coupled-product execution-plan candidate, and write them with
# a manifest for downstream LAMMPS binding (ye3t-lammps). Edit cfg_ye3t to change
# the resource caps or the output directory; edit CASES to change coordinates.

from hashlib import sha256
import json
from pathlib import Path

from ye3t.couplings import (
    YE3TExecutionPlan,
    YE3TSourceRealization,
    YE3T_O3_PRIMARY_CONVENTION,
    compile_execution_plan,
    compile_scalar_ace_coordinate,
    execution_plan_from_repeated_angular_blocks,
)
from ye3t.workflows import merge_workflow_config


# YE3T config dictionary.
cfg_ye3t = {
    "metadata": {
        "status": "experimental",  # Maturity of this workflow.
        "name": "compile_scalar_ace_lammps_plans",
        "schema": "ye3t_config_v1",
    },
    "basis": {
        "type": "ACE",
        "species": ["Ta"],  # Single-species Ta coordinates.
        "radial": {"type": "PACE_ChebExpCos", "n_max": 3},
        "angular": {
            "l_max": 4,
            "kind": "complex",
            "normalization": "pace_y00_one",
        },
        "coordinates": [  # Names of the CASES compiled below.
            "rank3_n223_l112",
            "homogeneous_rank4",
            "rank8_blocks_4_4",
            "homogeneous_rank16",
        ],
    },
    "representation": {
        "carrier": "ACE_density",  # Ordinary commutative ACE density carrier.
        "target": {"permutation": "trivial", "L": 0, "parity": "even"},
        "coupling": {
            "source": "ye3t.couplings",  # Single source of truth for labels and coefficients.
            "backend": "exact",
            "fast_path": "auto",
        },
    },
    "runtime": {
        "cache": "auto",
        "derivatives": "explicit_compiler_adjoint",
        "dtype": "float64",
        "device": "cpu",
        "maximum_coordinate_bytes": 128 * 1024 * 1024,  # Exact-symbolic and coordinate byte cap per case.
        "output_dir": str(Path.home() / "ye3t-workflows" / "benchmarks" / "scalar_ace_lammps_plans"),  # Must be new or empty.
    },
    "model": {
        "type": "linear_ace_execution_plan",
        "readout": "external_linear_coefficients",  # Coefficients are bound downstream in ye3t-methods.
    },
    "targets": {"energy": "scalar_site_energy", "forces": "plan_adjoint"},
    "validation": {
        "checks": [
            "compiler_certificate",
            "execution_plan_round_trip",
            "plan_hash",
        ]
    },
}


CASES = (
    {
        "name": "ta_r03_n223_l112",
        "label": {
            "n_tuple": [2, 2, 3],
            "l_tuple": [1, 1, 2],
            "internal_Ls": [2, 0],
            "L_R": 0,
            "tree_type": "balanced",
            "basis_key": ["node", ["sym", 2, 0], []],
        },
        "content": ["Ta:n2:l1", "Ta:n2:l1", "Ta:n3:l2"],
        "blocks": [
            {
                "power": 2,
                "input_L": 1,
                "slot_indices": [0, 1],
                "source_binding": {"binding_id": "Ta:n2:l1", "input_L": 1},
            },
            {
                "power": 1,
                "input_L": 2,
                "slot_indices": [2],
                "source_binding": {"binding_id": "Ta:n3:l2", "input_L": 2},
            },
        ],
    },
    {
        "name": "ta_h04",
        "label": {
            "n_tuple": [1, 1, 1, 1],
            "l_tuple": [1, 1, 1, 1],
            "internal_Ls": [0],
            "L_R": 0,
            "tree_type": "balanced",
            "basis_key": ["sym", 0, 0],
        },
        "content": ["Ta:n1:l1"] * 4,
        "blocks": [
            {
                "power": 4,
                "input_L": 1,
                "slot_indices": list(range(4)),
                "source_binding": {"binding_id": "Ta:n1:l1", "input_L": 1},
            }
        ],
    },
    {
        "name": "ta_r08_block_4_4",
        "label": {
            "n_tuple": [1, 1, 1, 1, 2, 2, 2, 2],
            "l_tuple": [1] * 8,
            "internal_Ls": [4, 4, 0],
            "L_R": 0,
            "tree_type": "balanced",
            "basis_key": ["node", ["sym", 4, 0], ["sym", 4, 0]],
        },
        "content": ["Ta:n1:l1"] * 4 + ["Ta:n2:l1"] * 4,
        "blocks": [
            {
                "power": 4,
                "input_L": 1,
                "slot_indices": list(range(4)),
                "source_binding": {"binding_id": "Ta:n1:l1", "input_L": 1},
            },
            {
                "power": 4,
                "input_L": 1,
                "slot_indices": list(range(4, 8)),
                "source_binding": {"binding_id": "Ta:n2:l1", "input_L": 1},
            },
        ],
    },
    {
        "name": "ta_h16",
        "label": {
            "n_tuple": [1] * 16,
            "l_tuple": [1] * 16,
            "internal_Ls": [0],
            "L_R": 0,
            "tree_type": "balanced",
            "basis_key": ["sym", 0, 0],
        },
        "membership_mode": "constructive_factorized",
        "content": ["Ta:n1:l1"] * 16,
        "blocks": [
            {
                "power": 16,
                "input_L": 1,
                "slot_indices": list(range(16)),
                "source_binding": {"binding_id": "Ta:n1:l1", "input_L": 1},
            }
        ],
    },
)


def scalar_coordinate_plan(case, maximum_coordinate_bytes):
    compilation = compile_scalar_ace_coordinate(
        case["label"],
        membership_mode=case.get("membership_mode", "auto"),
        coordinate_contract="pace_compatible_exact",
        coefficient_materialization="exact",
        constructor_backend="python",
        maximum_unique_monomials=250000,
        maximum_term_contributions=2000000,
        maximum_exact_symbolic_bytes=maximum_coordinate_bytes,
        maximum_coordinate_bytes=maximum_coordinate_bytes,
    )
    source = YE3TSourceRealization(
        kind="ordinary_density",
        rank=len(case["content"]),
        content=tuple(case["content"]),
    )
    plan = execution_plan_from_repeated_angular_blocks(
        tuple(case["blocks"]),
        id_prefix=case["name"],
        parent_partition=(len(case["content"]),),
        target_L=0,
        source_realization=source,
        spatial_symmetry="O3",
        compiler_labels=(compilation["label"],),
        coefficient_materialization="exact",
        maximum_exact_symbolic_bytes=maximum_coordinate_bytes,
    )
    return compilation, plan


def rank3_coupled_product_plan():
    source_channels = (
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
    )
    return compile_execution_plan(
        ace_coupled_product_request={
            "nin": (2, 2, 3),
            "lin": (1, 1, 2),
            "target_basis_index": 0,
            "source_channels": source_channels,
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
        }
    )


def write_plan(output_dir, filename, plan):
    text = plan.to_json(indent=2) + "\n"
    restored = YE3TExecutionPlan.from_json(text)
    if restored.plan_hash != plan.plan_hash:
        raise RuntimeError("Execution-plan JSON round trip changed the plan hash.")
    path = output_dir / filename
    path.write_text(text, encoding="utf-8")
    return {
        "file": filename,
        "plan_hash": plan.plan_hash,
        "schema": plan.schema_version,
        "sha256": sha256(text.encode("utf-8")).hexdigest(),
    }


def run_compile_scalar_ace_lammps_plans(config=None):
    """Compile every case, write the execution plans, and write the manifest."""
    settings = merge_workflow_config(cfg_ye3t, config)
    output_dir = Path(settings["runtime"]["output_dir"]).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    maximum_coordinate_bytes = int(settings["runtime"]["maximum_coordinate_bytes"])

    recorded_settings = dict(settings)
    recorded_settings["runtime"] = dict(settings["runtime"], output_dir=str(output_dir))
    manifest = {
        "schema": "ye3t_scalar_ace_lammps_plan_example_v1",
        "config": recorded_settings,
        "cases": [],
    }
    for case in CASES:
        compilation, plan = scalar_coordinate_plan(case, maximum_coordinate_bytes)
        plan_record = write_plan(
            output_dir,
            case["name"] + ".block.execution_plan.json",
            plan,
        )
        manifest["cases"].append(
            {
                "name": case["name"],
                "label": compilation["label"].to_dict(),
                "coordinate_certificate": compilation["certificate"],
                "coordinate_term_count": compilation["coefficient_table"].term_count,
                "execution_plan": plan_record,
            }
        )

    gram_plan = rank3_coupled_product_plan()
    manifest["rank3_coupled_product_candidate"] = write_plan(
        output_dir,
        "ta_r03_n223_l112.coupled_product.execution_plan.json",
        gram_plan,
    )
    manifest["rank3_coupled_product_candidate"]["deployment_status"] = (
        "compiler_candidate_requires_ye3t_methods_model_binding"
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    run_compile_scalar_ace_lammps_plans(config=cfg_ye3t)
