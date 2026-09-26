"""Stable printing helpers for YE3T workflow examples."""

import time

from ye3t import ExactProductExpansionEngine, YE3TAPI, enumerate_rank_labels, format_ye3t_basis, format_ye3t_sector
from ye3t.api import build_exact_schedule, build_operator_ir
from ye3t.cache import PrimitiveCacheSystem


def parse_ints(raw):
    """Parse a compact integer sequence."""
    if isinstance(raw, (list, tuple)):
        return tuple(int(token) for token in raw)
    return tuple(int(token.strip()) for token in str(raw).split(",") if token.strip())


def print_label_count_case(case):
    """Print count-only label records for one permutation/angular sector."""
    labels = enumerate_rank_labels(
        rank=int(case["rank"]),
        target_l_avs=tuple(case["target_l_avs"]),
        strict_max_li=int(case.get("strict_max_li", 3)),
        homogeneous_n=bool(case.get("homogeneous_n", False)),
        spec=case["spec"],
    )
    print(case["name"])
    print("  rank:", int(case["rank"]))
    print("  target L averages:", tuple(case["target_l_avs"]))
    print("  count:", len(labels))
    for row in labels[: int(case.get("print_limit", 4))]:
        print(" ", row)
    print()


def basis_sector_cases(include_rank16=False):
    """Return bounded exact sectors used by enumeration workflows."""
    cases = [
        ("rank-2 repeated", (1, 1), (1, 1)),
        ("rank-4 mixed", (1, 1, 2, 2), (1, 1, 2, 2)),
    ]
    if include_rank16:
        cases.append(("rank-16 mixed", (1,) * 16, (2,) * 8 + (3,) * 8))
    return tuple(cases)


def print_basis_sector_counts(api, name, nin, lin, max_target_L):
    """Print exact multiplicities without building CG path labels."""
    t0 = time.perf_counter()
    counts = api.counts_by_target_L(nin, lin, max_target_L=max_target_L)
    elapsed_s = time.perf_counter() - t0

    print(name)
    print(f"mode=counts_by_L elapsed_s={elapsed_s:.6f}")
    for L_R, count in counts.items():
        print(format_ye3t_sector(nin, lin, L_R, alpha=count))
    print(f"total={sum(counts.values())}")
    print()


def print_basis_sector_labels(api, name, nin, lin, max_target_L, print_limit):
    """Materialize compact CG path labels grouped by target ``L_R``."""
    t0 = time.perf_counter()
    labels_by_L = api.labels_by_target_L(nin, lin, max_target_L=max_target_L)
    elapsed_s = time.perf_counter() - t0

    print(name)
    print(f"mode=labels_by_L elapsed_s={elapsed_s:.6f}")
    for L_R, labels in labels_by_L.items():
        print(format_ye3t_sector(nin, lin, L_R, alpha=len(labels)))
        for index, label in enumerate(labels[:print_limit]):
            print(" ", format_ye3t_basis(index, label))
        if len(labels) > print_limit:
            print(f"  ... and {len(labels) - print_limit} more")
    print(f"total={sum(len(labels) for labels in labels_by_L.values())}")
    print()


def print_structured_basis_sector(case):
    """Print one exact sector with shared context and coordinate labels."""
    engine = ExactProductExpansionEngine(tree_type=case.get("tree_type", "balanced"))
    space = engine.feature_space(tuple(case["nin"]), tuple(case["lin"]), int(case["L_R"]))
    print_limit = int(case.get("print_limit", 4))

    print(f"=== {case['name']} ===")
    print(format_ye3t_sector(space.nin, space.lin, space.L_R, alpha=space.dim))
    for index, label in enumerate(space.labels[:print_limit]):
        print(" ", format_ye3t_basis(index, label))
    if space.dim > print_limit:
        print(f"  ... and {space.dim - print_limit} more")
    print()


def print_primitive_summary(engine, case):
    """Print quotient representatives and reconstruction coverage."""
    nin = tuple(case["nin"])
    lin = tuple(case["lin"])
    L_R = int(case["L_R"])
    mode = str(case["mode"])
    quotient = engine.primitive_quotient(nin, lin, L_R, mode=mode)
    reconstruction = engine.primitive_generator_reconstruction(nin, lin, L_R, factorization_policy=mode)

    print(case["name"])
    print("  target dim             =", quotient.target_space.dim)
    print("  generated rank         =", quotient.generated_rank)
    print("  primitive rank         =", quotient.primitive_rank)
    for index in quotient.primitive_basis_indices[: int(case.get("print_limit", 3))]:
        print("  primitive basis label  =", quotient.target_space.labels[index])
    if len(quotient.primitive_basis_indices) > int(case.get("print_limit", 3)):
        print("  primitive labels left  =", len(quotient.primitive_basis_indices) - int(case.get("print_limit", 3)))
    print("  product columns tried  =", reconstruction.product_column_count)
    print("  reconstructed rank     =", reconstruction.reconstructed_rank)
    print("  missing rank           =", reconstruction.missing_rank)


def primitive_cache_summary(case, policy):
    """Return cache statistics after evaluating a representative sector."""
    nin = tuple(case["nin"])
    lin = tuple(case["lin"])
    target_L = int(case["target_L"])
    engine = ExactProductExpansionEngine()
    sector = engine.feature_space(nin, lin, target_L)
    cache = PrimitiveCacheSystem(max_cache_rank=int(case.get("max_cache_rank", 4)), feature_lookup_policy=policy)

    primitive_count = 0
    expansion_lengths = []
    for label in sector.labels:
        if cache.is_primitive(label):
            primitive_count += 1
        expansion = cache.get_feature_by_any_means(label)
        expansion_lengths.append(len(expansion))

    stats = cache.get_cache_stats()
    return {
        "policy": policy,
        "sector_dim": int(sector.dim),
        "primitive_count": int(primitive_count),
        "first_expansion_terms": int(expansion_lengths[0]),
        "primitive_cache_size": int(stats["primitive_cache_size"]),
        "cached_only_misses": int(stats["cached_only_misses"]),
        "full_computation_count": int(stats["full_computation_count"]),
    }


def print_schedule_planning_policy(policy):
    """Compile a small exact operator and print schedule-planning metadata."""
    ir = build_operator_ir(
        (1, 1, 1),
        (1, 2, 1),
        2,
        primitive_first=True,
        include_target_primitive=True,
    )
    schedule = build_exact_schedule(ir, optimization_policy=policy, backend="pytorch")
    flat = schedule.flat_metadata
    materialization = schedule.materialization_plan
    print(f"policy={policy}")
    print("  segments:", len(schedule.segments))
    print("  flat groups:", flat.metadata["group_count"])
    print("  labels:", flat.metadata["label_count"])
    print("  materialized segments:", materialization.materialized_count)
    print("  recomputed segments:", materialization.recomputed_count)
    return schedule


def print_basis_enumeration(settings):
    """Print bounded count-only or label-materialization examples."""
    api = YE3TAPI(tree_type=settings.get("tree_type", "balanced"))
    max_target_L = None if settings["include_rank16"] else settings["max_target_L"]
    cases = list(settings.get("cases", ()))
    if not cases:
        cases = [
            {"name": name, "nin": tuple(nin), "lin": tuple(lin)}
            for name, nin, lin in basis_sector_cases(include_rank16=False)
        ]
    if settings.get("include_rank16", False):
        rank16_case = settings.get("rank16_case")
        if rank16_case is None:
            rank16_case = {
                "name": "rank-16 mixed",
                "nin": (1,) * 16,
                "lin": (2,) * 8 + (3,) * 8,
            }
        cases.append(rank16_case)
    for case in cases:
        name = case["name"]
        nin = tuple(case["nin"])
        lin = tuple(case["lin"])
        if settings["mode"] == "counts_by_L":
            print_basis_sector_counts(api, name, nin, lin, max_target_L)
        else:
            print_basis_sector_labels(api, name, nin, lin, max_target_L, max(0, int(settings["print_limit"])))
