"""Atomistic-free public coupler entry points."""

from ye3t.global_coupler import AngularCGMap
from ye3t.message_passing import CompileSameRankKroneckerRuntimeTables
from ye3t.representations.young_subgroup_specht_coupling import (
    build_cached_young_subgroup_specht_coupling,
    build_young_subgroup_specht_coupling,
)


def compile_rotation_coupler(
    input_Ls,
    output_L,
    *,
    parity=None,
    group="SO3",
    bracketing="balanced",
    cache_dir=None,
    maximum_factorized_materialization_bytes=None,
):
    """Compile a pure SO(3)/O(3) angular Clebsch-Gordan coupler."""

    return AngularCGMap.build(
        input_Ls,
        output_L,
        parity=parity,
        group=group,
        bracketing=bracketing,
        cache_dir=cache_dir,
        maximum_factorized_materialization_bytes=(
            maximum_factorized_materialization_bytes
        ),
    )


def compile_permutation_coupler(
    subgroup_partitions,
    target_partition,
    *,
    bracketing="balanced",
    cache_dir=None,
    backend="numeric_cached",
    constraint_backend="auto",
    rcond=1.0e-10,
    compare_exact_projector=False,
    exact_reference_max_rank=None,
):
    """Compile a pure Young-subgroup Specht subduction coupler."""

    backend = str(backend)
    if backend in {"numeric_cached", "cached_numeric", "fast"}:
        return build_cached_young_subgroup_specht_coupling(
            subgroup_partitions,
            target_partition,
            bracketing=bracketing,
            cache_dir=cache_dir,
            constraint_backend=constraint_backend,
            rcond=rcond,
            compare_exact_projector=compare_exact_projector,
            exact_reference_max_rank=exact_reference_max_rank,
        )
    if backend in {"exact", "symbolic"}:
        return build_young_subgroup_specht_coupling(
            subgroup_partitions,
            target_partition,
            bracketing=bracketing,
        )
    raise ValueError(
        "backend must be one of 'numeric_cached', 'cached_numeric', 'fast', 'exact', or 'symbolic'."
    )


def compile_same_rank_kronecker(
    left_partition,
    right_partition,
    *,
    target_partitions=None,
):
    """Compile finite same-rank S_N Kronecker sparse intertwiner tables."""

    return CompileSameRankKroneckerRuntimeTables(
        left_partition,
        right_partition,
        target_partitions=target_partitions,
    )


__all__ = [
    "compile_rotation_coupler",
    "compile_permutation_coupler",
    "compile_same_rank_kronecker",
]
