"""Inspect exact Young/rotation primitive and decomposable bases.

Install ye3t with its SymPy extra. These are abstract fixed-content carriers;
physical cluster placement and graph automorphisms require an application map.
"""

from ye3t.api import YE3TFixedContentBasis


cfg_ye3t = {
    "metadata": {"name": "exact_full_primitive_catalog", "status": "experimental"},
    "basis": {
        "type": "abstract_fixed_content",
        "rank": {
            4: {"n": (1, 1, 2, 2), "l": (1, 1, 2, 2)},
            6: {"n": (1,) * 6, "l": (1, 1, 2, 2, 3, 3)},
            16: {
                "n": tuple(i for i in range(1, 9) for _ in range(2)),
                "l": (1, 1) + (0,) * 14,
            },
        },
    },
    "representation": {
        "L": 1,
        "factorization": "matched_pairs",  # or "all_lower_products"
        "rank": {
            4: {"partition": (3, 1), "max_child_L": (1, 2),  # Compare two child-L limits.
                "child_partitions": {2: ((2,),)}},  # Allow this rank-2 Young sector in factors.
            6: {"partition": (3, 2, 1), "max_child_L": (3, 4),
                "local_partitions": ((1, 1), (1, 1), (2,))},  # Also inspect this G_nu sector.
            16: {"partition": (15, 1), "max_child_L": 0},
        },
    },
    "runtime": {"backend": "exact_symbolic", "device": "cpu"},
    "model": {"type": "none"},
    "targets": {"quantity": "primitive_quotient"},
    "validation": {"rank_accounting": True},
}


def run_exact_full_primitive_catalog(config=None):
    yb = YE3TFixedContentBasis(cfg_ye3t if config is None else config)
    print("representations:\n", yb.representations)
    yb.build_basis()  # Compile the selected Young and angular sectors.
    print("basis:", yb.basis)
    yb.calculate_pd()  # Split each sector into product image and primitive complement.
    print("primitive representations:\n", yb.primitive.representations)
    print("decomposable representations:\n", yb.decomposable.representations)
    print("primitive basis:", yb.primitive.basis)
    print("decomposable basis:", yb.decomposable.basis)
    # Options: copy, young_index, magnetic_M; global adds subgroup_partitions, induction_index.
    # scope="local" uses the fixed-content stabilizer G_nu; "global" uses all S_N.
    sample = yb.primitive.basis.vector(6, 3, scope="local")  # Rank 6, child-L cap 3.
    print("one normalized primitive vector:", sample)
    return yb


if __name__ == "__main__":
    run_exact_full_primitive_catalog(config=cfg_ye3t)
