"""Build an exact product expansion and primitive quotient catalog."""

# exact_full_primitive_catalog.py
# Compare full exact product spaces against primitive quotient representatives.
# This is the place to inspect which reduced permutation sectors survive after
# invariant/module primitive factorization.

from ye3t import ExactProductExpansionEngine
from ye3t.utils.printing import print_primitive_summary
from ye3t.workflows import merge_workflow_config


# YE3T config dictionary.
cfg_ye3t = {
    "tree_type": "balanced",  # Recoupling tree used for exact products.
    "summary_cases": (  # Primitive quotient sectors to summarize.
        {
            "name": "rank-2 invariant primitive quotient:",  # Name used in printed output.
            "nin": (1, 1),  # Non-angular channel eta_i for each slot.
            "lin": (1, 1),  # Angular momentum l_i for each slot.
            "L_R": 0,  # Target output angular momentum.
            "mode": "invariant",  # Primitive factorization for scalar invariant outputs.
            "print_limit": 3,  # Maximum primitive basis labels to print.
        },
        {
            "name": "rank-3 mixed equivariant-module primitive quotient:",  # Printed label.
            "nin": (1, 1, 1),  # Non-angular channel eta_i for each slot.
            "lin": (1, 2, 1),  # Angular momentum l_i for each slot.
            "L_R": 2,  # Target output angular momentum.
            "mode": "module",  # Primitive factorization for equivariant module outputs.
            "print_limit": 3,  # Maximum primitive basis labels to print.
        },
    ),
}


def run_exact_full_primitive_catalog(config=None):
    """Compare exact/full products, primitive quotients, and reconstruction."""
    settings = merge_workflow_config(cfg_ye3t, config)
    engine = ExactProductExpansionEngine(tree_type=settings["tree_type"])
    rank2_scalar = engine.feature_space((1, 1), (1, 1), 0)
    label = rank2_scalar.labels[0]
    expansion = engine.expand_product(label, label, L_out=0)

    print("rank-2 scalar basis dimension:", rank2_scalar.dim)
    print("rank-4 scalar target dimension:", expansion.target_space.dim)
    print("nonzero expansion coefficients:", sum(1 for coeff in expansion.coefficients if coeff != 0))
    print("max M inconsistency:", expansion.max_M_inconsistency)
    for case in settings["summary_cases"]:
        print()
        print_primitive_summary(engine, case)


if __name__ == '__main__':
    run_exact_full_primitive_catalog(config=cfg_ye3t)
