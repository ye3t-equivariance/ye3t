"""Rank-2 Young-Yamanouchi/CG runtime coefficients.

This module materializes the finite-rank convention used by the Young-E3
planning layer for the one case currently implemented rigorously: two repeated
identical slots.  Coefficients are emitted only after Young-orthogonal Specht
symmetrization and exact YE3T Clebsch-Gordan recoupling have been summed into a
single output vector.  Invalid joint ``S_2 x SO(3)`` sectors therefore vanish
by construction; no post-hoc sector-count filter is used.

References:

* de Mello Koch, Ives, and Stephanou, "On subgroup adapted bases for
  representations of the symmetric group", arXiv:1112.4316,
  DOI: 10.1088/1751-8113/45/13/135204, for the Young-Yamanouchi/split-basis
  subduction-coefficient context.  This module does not implement their
  large-row-difference algorithm.
"""

from functools import lru_cache

from ye3t._record import recordclass
from ye3t.core.subtree_dag import cg_exact
from ye3t.exact_scalars import ExactRadical, exact_scalar

from .generalized_irreps import Partition


@recordclass(
    (
        "output_index",
        "partition_signature",
        "L_R",
        "rho",
        "target_tableau_index",
        "M",
        "source_order",
        "left_m",
        "right_m",
        "coefficient",
    ),
    frozen=True,
)
class YoungYamanouchiCGRuntimeCoefficient:
    """One sparse coefficient in the canonical rank-2 slot order."""


@recordclass(
    (
        "nin",
        "lin",
        "rank",
        "target_partitions",
        "target_Ls",
        "output_basis_labels",
        "coefficients",
        "coefficient_convention",
        "materializes_coefficients",
        "uses_young_yamanouchi_carriers",
        "uses_ye3t_cg_coefficients",
        "uses_symbolic_basis_extraction",
        "status",
        "detail",
    ),
    frozen=True,
)
class YoungYamanouchiCGRuntimeSchedule:
    """Executable sparse rank-2 Young-Yamanouchi/CG coefficient schedule."""

    detail = ""


@recordclass(
    (
        "schedule",
        "rank_supported",
        "coefficients_present",
        "convention_consistent",
        "passed",
        "detail",
    ),
    frozen=True,
)
class YoungYamanouchiCGRuntimeScheduleValidation:
    """Validation report for a materialized Young-Yamanouchi/CG runtime schedule."""

    detail = ""


def _normalize_nl(nin, lin):
    nin = tuple(int(x) for x in nin)
    lin = tuple(int(x) for x in lin)
    if len(nin) != len(lin):
        raise ValueError("nin and lin must have the same length.")
    return nin, lin


def _normalize_target_partitions(target_partitions):
    if target_partitions is None:
        target_partitions = ("all",)
    if isinstance(target_partitions, (str, Partition)):
        target_partitions = (target_partitions,)
    out = []
    for item in target_partitions:
        if isinstance(item, Partition):
            out.append(item)
            continue
        if isinstance(item, (tuple, list)):
            out.append(Partition(tuple(int(x) for x in item)))
            continue
        name = str(item).strip().lower().replace("-", "_")
        if name in {"all", "*"}:
            out.extend((Partition((2,)), Partition((1, 1))))
        elif name in {"trivial", "symmetric", "sym", "bosonic"}:
            out.append(Partition((2,)))
        elif name in {"sign", "antisymmetric", "anti", "fermionic"}:
            out.append(Partition((1, 1)))
        else:
            raise ValueError(f"Unsupported rank-2 target partition selector {item!r}.")
    seen = set()
    unique = []
    for partition in out:
        key = tuple(int(x) for x in partition.parts)
        if key not in seen:
            unique.append(partition)
            seen.add(key)
    return tuple(unique)


def _normalize_target_Ls(l_value, target_Ls):
    l_value = int(l_value)
    if target_Ls is None:
        return tuple(range(0, 2 * l_value + 1))
    return tuple(sorted({int(x) for x in target_Ls}))


def _coefficient_float(value):
    return float(exact_scalar(value))


def _canonical_leaf_pair(source_order, left_m, right_m):
    source_order = tuple(int(x) for x in source_order)
    pair = (int(left_m), int(right_m))
    return tuple(pair[index] for index in source_order)


@lru_cache(maxsize=128)
def _compile_yamanouchi_pair_schedule_cached(nin, lin, target_partition_parts, target_Ls):
    nin = tuple(int(x) for x in nin)
    lin = tuple(int(x) for x in lin)
    if len(nin) != 2:
        raise NotImplementedError("The materialized Young-Yamanouchi/CG runtime schedule currently supports rank 2.")
    if nin[0] != nin[1] or lin[0] != lin[1]:
        raise NotImplementedError("The rank-2 runtime schedule currently supports repeated identical slots only.")

    l_value = int(lin[0])
    target_partitions = tuple(Partition(tuple(int(x) for x in parts)) for parts in target_partition_parts)
    target_Ls = tuple(int(x) for x in target_Ls)
    output_labels = []
    coefficients = []
    from .young_orthogonal import young_orthogonal_induced_coupling

    for target_partition in target_partitions:
        coupling = young_orthogonal_induced_coupling(Partition((1,)), Partition((1,)), target_partition)
        for L_R in target_Ls:
            if L_R < 0 or L_R > 2 * l_value:
                continue
            for vector in coupling.vectors:
                for M in range(-int(L_R), int(L_R) + 1):
                    accumulated = {}
                    for source_index, young_coeff in enumerate(vector.coefficients):
                        young_coeff = exact_scalar(young_coeff)
                        if young_coeff.is_zero():
                            continue
                        source_order = tuple(int(x) for x in coupling.induced_basis[int(source_index)].coset_rep)
                        for left_m in range(-l_value, l_value + 1):
                            right_m = int(M - left_m)
                            if right_m < -l_value or right_m > l_value:
                                continue
                            cg_coeff = cg_exact(l_value, left_m, l_value, right_m, int(L_R), int(M))
                            coeff = young_coeff * cg_coeff
                            if coeff.is_zero():
                                continue
                            leaf_pair = _canonical_leaf_pair(source_order, left_m, right_m)
                            accumulated[leaf_pair] = accumulated.get(leaf_pair, ExactRadical.rational(0)) + coeff
                    terms = tuple(
                        sorted(
                            (
                                (leaf_pair, coeff)
                                for leaf_pair, coeff in accumulated.items()
                                if not coeff.is_zero()
                            ),
                            key=lambda item: item[0],
                        )
                    )
                    if not terms:
                        continue
                    output_index = len(output_labels)
                    output_labels.append(
                        (
                            (tuple(int(x) for x in target_partition.parts),),
                            int(L_R),
                            int(vector.rho),
                            int(vector.target_tableau_index),
                            int(M),
                        )
                    )
                    for (left_m, right_m), coeff in terms:
                        coefficients.append(
                            YoungYamanouchiCGRuntimeCoefficient(
                                output_index=int(output_index),
                                partition_signature=(tuple(int(x) for x in target_partition.parts),),
                                L_R=int(L_R),
                                rho=int(vector.rho),
                                target_tableau_index=int(vector.target_tableau_index),
                                M=int(M),
                                source_order=(0, 1),
                                left_m=int(left_m),
                                right_m=int(right_m),
                                coefficient=_coefficient_float(coeff),
                            )
                        )

    return YoungYamanouchiCGRuntimeSchedule(
        nin=nin,
        lin=lin,
        rank=2,
        target_partitions=tuple(tuple(int(x) for x in partition.parts) for partition in target_partitions),
        target_Ls=target_Ls,
        output_basis_labels=tuple(output_labels),
        coefficients=tuple(coefficients),
        coefficient_convention="young_yamanouchi_seminormal_plus_ye3t_cg_runtime_v1",
        materializes_coefficients=True,
        uses_young_yamanouchi_carriers=True,
        uses_ye3t_cg_coefficients=True,
        uses_symbolic_basis_extraction=False,
        status="rank2_repeated_slots_materialized",
        detail=(
            "Executable rank-2 repeated-slot coefficient schedule. Young-orthogonal Specht "
            "coefficients and exact CG coefficients are coalesced before outputs are emitted."
        ),
    )


def compile_young_yamanouchi_pair_cg_runtime_schedule(
    nin,
    lin,
    *,
    target_partitions=("all",),
    target_Ls=None,
):
    """Compile an executable rank-2 Young-Yamanouchi/CG coefficient schedule."""

    nin, lin = _normalize_nl(nin, lin)
    partitions = _normalize_target_partitions(target_partitions)
    target_Ls = _normalize_target_Ls(lin[0], target_Ls)
    return _compile_yamanouchi_pair_schedule_cached(
        nin,
        lin,
        tuple(tuple(int(x) for x in partition.parts) for partition in partitions),
        tuple(int(x) for x in target_Ls),
    )


def validate_young_yamanouchi_cg_runtime_schedule(schedule):
    """Check basic materialization and convention flags for a rank-2 schedule."""

    rank_supported = int(schedule.rank) == 2
    coefficients_present = len(tuple(schedule.output_basis_labels)) > 0 and len(tuple(schedule.coefficients)) > 0
    convention_consistent = (
        str(schedule.coefficient_convention) == "young_yamanouchi_seminormal_plus_ye3t_cg_runtime_v1"
        and bool(schedule.materializes_coefficients)
        and bool(schedule.uses_young_yamanouchi_carriers)
        and bool(schedule.uses_ye3t_cg_coefficients)
        and not bool(schedule.uses_symbolic_basis_extraction)
    )
    passed = bool(rank_supported and coefficients_present and convention_consistent)
    detail = "ok" if passed else (
        f"rank_supported={rank_supported}, coefficients_present={coefficients_present}, "
        f"convention_consistent={convention_consistent}"
    )
    return YoungYamanouchiCGRuntimeScheduleValidation(
        schedule=schedule,
        rank_supported=bool(rank_supported),
        coefficients_present=bool(coefficients_present),
        convention_consistent=bool(convention_consistent),
        passed=passed,
        detail=detail,
    )


def evaluate_young_yamanouchi_pair_cg_runtime_schedule(schedule, left, right):
    """Evaluate a materialized rank-2 Young-Yamanouchi/CG schedule with torch."""

    import torch

    if int(schedule.rank) != 2:
        raise ValueError("Only rank-2 schedules are supported by this evaluator.")
    if left.shape[-1] != right.shape[-1]:
        raise ValueError("left and right feature tensors must have matching trailing dimensions.")
    l_value = int(schedule.lin[0])
    expected_dim = 2 * l_value + 1
    if int(left.shape[-1]) != expected_dim:
        raise ValueError(f"Expected trailing dimension {expected_dim}, got {left.shape[-1]}.")
    if left.shape[:-1] != right.shape[:-1]:
        raise ValueError("left and right feature tensors must have matching batch dimensions.")

    dtype = torch.promote_types(left.dtype, right.dtype)
    device = left.device
    left = left.to(dtype=dtype)
    right = right.to(device=device, dtype=dtype)
    out = torch.zeros(*left.shape[:-1], len(tuple(schedule.output_basis_labels)), dtype=dtype, device=device)
    for coeff in schedule.coefficients:
        scalar = torch.as_tensor(float(coeff.coefficient), dtype=dtype, device=device)
        left_value = left[..., int(coeff.left_m) + l_value]
        right_value = right[..., int(coeff.right_m) + l_value]
        out[..., int(coeff.output_index)] = out[..., int(coeff.output_index)] + scalar * left_value * right_value
    return out


__all__ = [
    "YoungYamanouchiCGRuntimeCoefficient",
    "YoungYamanouchiCGRuntimeSchedule",
    "YoungYamanouchiCGRuntimeScheduleValidation",
    "compile_young_yamanouchi_pair_cg_runtime_schedule",
    "evaluate_young_yamanouchi_pair_cg_runtime_schedule",
    "validate_young_yamanouchi_cg_runtime_schedule",
]
