"""Dictionary-style workflow helpers for public YE3T examples."""

from ye3t.representations import (
    ExactSymbolicProjectorGeneralizedBasisBuilder,
    Partition,
    PermutationIrrep,
    PermutationSubgroup,
)
from ye3t.runtime.generalized import (
    GeneralizedExactRuntimeBlock,
    GeneralizedExactRuntimeIrreps,
    JointYoungCGProduct,
)


def merge_workflow_config(base, *updates):
    """Recursively merge dictionary-style workflow settings."""
    merged = dict(base or {})
    for update in updates:
        if update is None:
            continue
        for key, value in dict(update).items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = merge_workflow_config(merged[key], value)
            elif value is not None:
                merged[key] = value
    return merged


def basis_label_count_config():
    """Return default count-only basis label cases."""
    return {
        "cases": (
            {
                "name": "rank-2 repeated channel",
                "rank": 2,
                "target_l_avs": (0.0, 1.0, 2.0),
                "strict_max_li": 3,
                "homogeneous_n": False,
                "print_limit": 4,
                "spec": {
                    "mode": "restricted",
                    "n_orbits": ((1, 1),),
                    "l_orbits": ((1, 1),),
                    "pair_orbits": ((1, 1),),
                },
            },
            {
                "name": "rank-3 mixed orbit partitions",
                "rank": 3,
                "target_l_avs": (0.0, 1.0, 2.0),
                "strict_max_li": 3,
                "homogeneous_n": False,
                "print_limit": 4,
                "spec": {
                    "mode": "restricted",
                    "n_orbits": ((3,),),
                    "l_orbits": ((2, 1),),
                    "pair_orbits": ((2, 1),),
                },
            },
        )
    }


def basis_sector_enumeration_config():
    """Return default exact-sector enumeration settings."""
    return {
        "mode": "counts_by_L",
        "max_target_L": 4,
        "print_limit": 3,
        "include_rank16": False,
        "tree_type": "balanced",
        "cases": (
            {
                "name": "rank-2 repeated",
                "nin": (1, 1),
                "lin": (1, 1),
            },
            {
                "name": "rank-4 mixed",
                "nin": (1, 1, 2, 2),
                "lin": (1, 1, 2, 2),
            },
        ),
        "rank16_case": {
            "name": "rank-16 mixed",
            "nin": (1,) * 16,
            "lin": (2,) * 8 + (3,) * 8,
        },
    }


def primitive_catalog_config():
    """Return default primitive quotient catalog settings."""
    return {
        "tree_type": "balanced",
        "summary_cases": (
            {
                "name": "rank-2 invariant primitive quotient:",
                "nin": (1, 1),
                "lin": (1, 1),
                "L_R": 0,
                "mode": "invariant",
                "print_limit": 3,
            },
            {
                "name": "rank-3 mixed equivariant-module primitive quotient:",
                "nin": (1, 1, 1),
                "lin": (1, 2, 1),
                "L_R": 2,
                "mode": "module",
                "print_limit": 3,
            },
        ),
    }


def symmetric_power_kernel_config():
    """Return default symmetric-power diagnostic settings."""
    return {
        "catalog_cases": (
            (2, (1, 2, 3)),
            (3, (1, 2)),
            (4, (1, 2)),
            (8, (1,)),
        ),
        "square_case": {
            "L": 2,
            "output_L": 4,
            "seed": 41,
            "batch": 4096,
            "warmups": 3,
            "runs": 12,
        },
        "power_cases": (
            {"power": 3, "L": 2, "output_L": 6, "seed": 43, "batch": 1024, "warmups": 2, "runs": 5},
            {"power": 4, "L": 2, "output_L": 4, "seed": 44, "batch": 512, "warmups": 2, "runs": 3},
            {"power": 8, "L": 1, "output_L": 8, "seed": 48, "batch": 256, "warmups": 2, "runs": 3},
        ),
    }


def _int_tuple(values, name):
    if values is None:
        raise KeyError(f"Missing required {name!r} entry.")
    return tuple(int(value) for value in values)


def _partition_from_config(payload):
    if isinstance(payload, Partition):
        return payload
    if isinstance(payload, dict):
        return Partition.from_dict(payload)
    return Partition(tuple(int(value) for value in payload))


def _permutation_irrep_from_block(block, subgroup):
    if "permutation_irrep" in block:
        payload = block["permutation_irrep"]
        if isinstance(payload, PermutationIrrep):
            return payload
        return PermutationIrrep.from_dict(payload)
    if "partition" in block:
        return PermutationIrrep(subgroup=subgroup, partitions=(_partition_from_config(block["partition"]),))
    if "partitions" in block:
        return PermutationIrrep(
            subgroup=subgroup,
            partitions=tuple(_partition_from_config(item) for item in block["partitions"]),
        )
    return PermutationIrrep.trivial_for_subgroup(subgroup)


def build_runtime_irreps_from_config(config):
    """Build exact generalized runtime irreps from a dictionary.

    Each block requires ``nin`` and ``lin``. If ``partition``, ``partitions``,
    or ``permutation_irrep`` is omitted, the block uses the trivial Young
    subgroup irrep. ``L`` defaults only when the sector contains one angular
    copy.
    """

    if isinstance(config, dict):
        blocks_config = config.get("blocks", (config,))
    else:
        blocks_config = config
    blocks = []
    for block_config in blocks_config:
        block = dict(block_config)
        nin = _int_tuple(block.get("nin"), "nin")
        lin = _int_tuple(block.get("lin"), "lin")
        subgroup = PermutationSubgroup.from_nl(nin, lin)
        permutation_irrep = _permutation_irrep_from_block(block, subgroup)
        sector = ExactSymbolicProjectorGeneralizedBasisBuilder(nin, lin, permutation_irrep).build()

        if "L" in block:
            L_value = int(block["L"])
        else:
            allowed_L = tuple(sorted(int(value) for value in sector.labels_by_L))
            if len(allowed_L) != 1:
                raise ValueError(
                    "Runtime irrep config must specify L when the symbolic sector contains "
                    f"multiple angular channels: {allowed_L!r}."
                )
            L_value = allowed_L[0]

        blocks.append(
            GeneralizedExactRuntimeBlock(
                int(block.get("mul", block.get("multiplicity", 1))),
                sector,
                L_value,
                int(block.get("copy_index", 0)),
            )
        )
    return GeneralizedExactRuntimeIrreps(tuple(blocks))


def build_joint_young_cg_product_from_config(config):
    """Build a ``JointYoungCGProduct`` from dictionary-style left/right specs."""

    if "left" not in config or "right" not in config:
        raise KeyError("Joint Young-CG product config requires left and right entries.")
    left = config["left"]
    right = config["right"]
    irreps_left = left if isinstance(left, GeneralizedExactRuntimeIrreps) else build_runtime_irreps_from_config(left)
    irreps_right = right if isinstance(right, GeneralizedExactRuntimeIrreps) else build_runtime_irreps_from_config(right)
    return JointYoungCGProduct(
        irreps_left,
        irreps_right,
        tree_type=str(config.get("tree_type", "balanced")),
        requested_targets=config.get("requested_targets", None),
        rank_cap=config.get("rank_cap", None),
        L_max=config.get("L_max", None),
        permutation_policy=str(config.get("permutation_policy", "mixed_character")),
        allowed_permutation_irreps=config.get("allowed_permutation_irreps", None),
    )


__all__ = [
    "basis_label_count_config",
    "basis_sector_enumeration_config",
    "build_joint_young_cg_product_from_config",
    "build_runtime_irreps_from_config",
    "merge_workflow_config",
    "primitive_catalog_config",
    "symmetric_power_kernel_config",
]
