"""Expand symbolic Young partitions and count valid O(3) sectors.

Run from an installed YE3T checkout:

    python examples/symbolic_young_partition_catalogue.py

The script has no command-line interface. Edit the complete config below.
"""

from ye3t import couplings


cfg_ye3t = {
    "metadata": {  # Public example identity and maturity.
        "name": "symbolic_young_partition_catalogue",
        "status": "experimental",
    },
    "basis": {  # Ranks and fixed radial/angular content to count.
        "ranks": [4, 5],
        "n_in_by_rank": {
            4: [1, 1, 1, 1],
            5: [1, 1, 1, 1, 1],
        },
        "l_in_by_rank": {
            4: [1, 1, 1, 1],
            5: [1, 1, 1, 1, 1],
        },
    },
    "representation": {  # O(3) convention and symbolic parent partitions.
        "spatial_symmetry": "O3",
        "internal_lambda_restriction": {
            "mode": "templates",
            "templates": [
                ("N",),
                ("N-1", "1"),
                ("N-2", "2"),
                ("N/2", "N/2"),
                ("N/2", "N/4", "N/4"),
                ("2N/3", "N/6", "N/6"),
            ],
            "rounding_tolerance": 0.5,
            "closure_policy": "strict",
        },
    },
    "runtime": {  # Count-only execution mode.
        "count_only": True,
    },
    "model": {  # Intended downstream catalogue consumer.
        "consumer": "YE3T message-passing catalogue selector",
    },
    "targets": {  # Scalar-energy readout target.
        "readout": {"lambda": "(N)", "L": 0, "parity": 1},
    },
    "validation": {  # Invalid-template and zero-multiplicity policy.
        "print_inapplicable_templates": True,
        "reject_zero_multiplicity_coordinates": True,
    },
}


def run_symbolic_young_partition_catalogue(config=None):
    """Expand the configured templates and print positive O(3) counts."""
    config = cfg_ye3t if config is None else config
    restriction = config["representation"]["internal_lambda_restriction"]
    expansion = couplings.expand_partition_templates(
        config["basis"]["ranks"],
        restriction["templates"],
        rounding_tolerance=restriction["rounding_tolerance"],
    )

    print("Partition expansion source: ye3t.couplings.expand_partition_templates")
    for record in expansion["records"]:
        print(
            "N=" + str(record["rank"]),
            tuple(record["expressions"]),
            "->",
            record["partition"],
            "[" + record["status"] + "]",
            record["reason"],
        )

    print("\nPositive O(3) multiplicity coordinates from ye3t.couplings:")
    for rank in config["basis"]["ranks"]:
        n_in = tuple(config["basis"]["n_in_by_rank"][rank])
        l_in = tuple(config["basis"]["l_in_by_rank"][rank])
        for partition in expansion["partitions_by_rank"][str(rank)]:
            counts = couplings.counts_for_partitions(
                n_in,
                l_in,
                (tuple(partition),),
                spatial_symmetry="O3",
            )
            positive = {
                int(L): int(multiplicity)
                for L, multiplicity in sorted(counts.items())
                if int(multiplicity) > 0
            }
            if not positive:
                continue
            print(
                "N=" + str(rank),
                "lambda=" + str(tuple(partition)),
                "L multiplicities=" + str(positive),
                "parity=" + str(1 if sum(l_in) % 2 == 0 else -1),
            )
    return expansion


if __name__ == "__main__":
    run_symbolic_young_partition_catalogue(config=cfg_ye3t)
