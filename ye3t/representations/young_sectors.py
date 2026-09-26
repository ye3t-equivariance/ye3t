"""Public Young-sector helpers for the generalized exact path.

These functions provide a stable, provenance-carrying surface over the
separate small-``N`` generalized projector and merged tensor-product code.  The
default ACE/ye3t invariant path remains unchanged.

Reference context: Schur-Weyl/Jacobi-Trudi count compression here uses Schur
polynomial character identities as an exact integer count path.  Subgroup
adapted Young-Yamanouchi split bases are related to the setting of de Mello
Koch, Ives, and Stephanou, arXiv:1112.4316,
DOI: 10.1088/1751-8113/45/13/135204, but the runtime materialization bridge
in this package is implemented separately and tested only for the stated
finite cases.
"""

from collections import OrderedDict
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, permutations
from math import factorial

from ye3t._record import recordclass
from ye3t.exact_linalg import exact_algebraic_rank_or_none

from .generalized_irreps import AngularIrrep, CoupledIrrepLabel, Partition, PermutationIrrep, PermutationSubgroup, PermutationSubgroupFactor, _normalize_spatial_parity
from .projectors import symmetric_group_character


_SECTOR_CACHE = OrderedDict()
_SECTOR_CACHE_MAX = 64


def _sympy():
    from ye3t._optional_sympy import sp

    return sp


def _build_exact_symbolic_projector_sector(
    nin,
    lin,
    permutation_irrep,
    spatial_symmetry="SO3_legacy",
):
    from .builder import ExactSymbolicProjectorGeneralizedBasisBuilder

    return ExactSymbolicProjectorGeneralizedBasisBuilder(
        nin,
        lin,
        permutation_irrep,
        spatial_symmetry=spatial_symmetry,
    ).build()


def _build_exact_change_of_group_basis_map(*args, **kwargs):
    from .tensor_products import build_exact_change_of_group_basis_map

    return build_exact_change_of_group_basis_map(*args, **kwargs)


def _build_balanced_pairwise_schedule(rank):
    from .tensor_products import build_balanced_pairwise_schedule

    return build_balanced_pairwise_schedule(rank)


def _build_left_schedule(rank):
    from .tensor_products import build_left_schedule

    return build_left_schedule(rank)


def _merged_permutation_irrep_product_terms(*args, **kwargs):
    from .tensor_products import merged_permutation_irrep_product_terms

    return merged_permutation_irrep_product_terms(*args, **kwargs)


def _exact_product_expansion_engine():
    from ye3t.core.product_engine import ExactProductExpansionEngine

    return ExactProductExpansionEngine()


def _reachable_total_angular_momenta(lin):
    from ye3t.core.basis.validation import reachable_total_angular_momenta

    return reachable_total_angular_momenta(lin)


@recordclass(('nin', 'lin', 'permutation_irrep', 'counts_by_L', 'labels_by_L', 'projected_dim', 'provenance', 'codepath', 'detail', 'sector'), frozen=True)
class YoungSectorCounts:
    """Counts for one ``SO(3) x G_nu`` Young sector with provenance."""
    labels_by_L = None
    projected_dim = None
    detail = ""
    sector = None


@recordclass(
    (
        'permutation_irrep',
        'partition_signature',
        'L_R',
        'multiplicity',
        'carrier_dim',
        'joint_sector_dim',
        'full_sector_dim',
        'labels',
        'provenance',
        'codepath',
        'detail',
    ),
    frozen=True,
)
class YoungAnalyticalSectorRecord:
    """One exact count-only ``SO(3) x G_nu`` Young-sector record."""
    detail = ""


@recordclass(
    (
        'nin',
        'lin',
        'channel_multiset',
        'subgroup',
        'allowed_target_Ls',
        'irreps',
        'records',
        'raw_tensor_dim',
        'total_joint_sector_dim',
        'total_full_sector_dim',
        'complete',
        'provenance',
        'codepath',
        'detail',
    ),
    frozen=True,
)
class YoungAnalyticalSectorCatalog:
    """Exact analytical catalog over all Young characters and allowed ``L_R``."""
    detail = ""


@recordclass(('catalog', 'dimension_complete', 'records_match_count_only', 'passed', 'detail'), frozen=True)
class YoungAnalyticalSectorCatalogValidation:
    """Validation report for an analytical Young/E3 sector catalog."""
    detail = ""


@recordclass(
    (
        'permutation_irrep',
        'partition_signature',
        'family_signature',
        'L_R',
        'multiplicity',
        'carrier_dim',
        'joint_sector_dim',
        'full_sector_dim',
        'construction_kind',
        'compression_kind',
        'exact',
        'labels',
        'provenance',
        'codepath',
        'detail',
    ),
    frozen=True,
)
class YoungSpechtConstructionRecord:
    """One selected Young/E3 sector in the constructive Specht plan."""
    detail = ""


@recordclass(
    (
        'nin',
        'lin',
        'channel_multiset',
        'subgroup',
        'sector_families',
        'allowed_target_Ls',
        'available_irrep_count',
        'selected_irrep_count',
        'enumerated_all_sectors',
        'records',
        'raw_tensor_dim',
        'selected_full_sector_dim',
        'complete',
        'cache_key',
        'provenance',
        'codepath',
        'detail',
    ),
    frozen=True,
)
class YoungSpechtBasisConstructionPlan:
    """Cached exact sector plan for full or selected Specht/Young-E3 sectors."""
    detail = ""


@recordclass(('plan', 'dimension_complete', 'records_are_exact', 'selection_consistent', 'passed', 'detail'), frozen=True)
class YoungSpechtBasisConstructionPlanValidation:
    """Validation report for a Specht construction plan."""
    detail = ""


@recordclass(
    (
        'nin',
        'lin',
        'tree_type',
        'sector_families',
        'target_Ls',
        'construction_plan',
        'tree_schedule',
        'leaf_labels',
        'root_labels',
        'coefficient_convention',
        'uses_young_yamanouchi_carriers',
        'uses_ye3t_cg_trees',
        'uses_symbolic_basis_extraction',
        'materializes_coefficients',
        'status',
        'cache_key',
        'detail',
    ),
    frozen=True,
)
class YoungYamanouchiCGTreeSchedule:
    """Finite-rank Young-Yamanouchi/seminormal plus YE3T-CG tree schedule."""
    detail = ""


@recordclass(('schedule', 'plan_valid', 'tree_metadata_present', 'convention_consistent', 'passed', 'detail'), frozen=True)
class YoungYamanouchiCGTreeScheduleValidation:
    """Validation report for a Young-Yamanouchi/CG-tree schedule."""
    detail = ""


@recordclass(('left_label', 'right_label', 'output_label', 'permutation_multiplicity', 'provenance', 'codepath'), frozen=True)
class YoungProductPath:
    """One angular/permutation-resolved product path."""
    codepath = "merged_character_product"


@recordclass(
    (
        'nin',
        'lin',
        'channel_multiset',
        'permutation_irrep',
        'L_R',
        'mode',
        'max_factor_L',
        'sector_count',
        'sector_basis_rank',
        'primitive_rank',
        'generated_rank',
        'generated_basis_indices',
        'primitive_basis_indices',
        'target_basis_labels',
        'generated_upper_bound',
        'primitive_lower_bound',
        'product_path_count',
        'rank_status',
        'provenance',
        'codepath',
        'detail',
        'quotient',
    ),
    frozen=True,
)
class YoungResolvedPrimitiveQuotient:
    """Primitive-quotient summary resolved by channel multiset and Young irrep."""
    sector_basis_rank = None
    generated_rank = None
    generated_basis_indices = tuple()
    primitive_basis_indices = tuple()
    target_basis_labels = tuple()
    generated_upper_bound = None
    primitive_lower_bound = None
    product_path_count = 0
    rank_status = "exact"
    detail = ""
    quotient = None


@recordclass(('quotient', 'rank_accounting', 'index_partition', 'generated_rank_matches_matrix', 'trivial_matches_ace', 'passed', 'detail'), frozen=True)
class YoungResolvedPrimitiveQuotientValidation:
    """Validation report for a Young/SO(3)-resolved primitive quotient."""
    trivial_matches_ace = None
    detail = ""


def _normalize_nl(nin, lin):
    nin = tuple(int(x) for x in nin)
    lin = tuple(int(x) for x in lin)
    if len(nin) != len(lin):
        raise ValueError(f"nin and lin must have the same length, got {len(nin)} and {len(lin)}.")
    return nin, lin


def _sector_cache_key(nin, lin, permutation_irrep, spatial_symmetry="SO3_legacy"):
    return (
        tuple(int(x) for x in nin),
        tuple(int(x) for x in lin),
        str(spatial_symmetry),
        tuple(
            (
                factor.channel_label,
                int(factor.l),
                int(factor.multiplicity),
            )
            for factor in permutation_irrep.subgroup.factors
        ),
        tuple(tuple(int(x) for x in partition.parts) for partition in permutation_irrep.partitions),
    )


def _cache_get(key):
    item = _SECTOR_CACHE.get(key)
    if item is None:
        return None
    _SECTOR_CACHE.move_to_end(key)
    return item


def _cache_put(key, value):
    _SECTOR_CACHE[key] = value
    _SECTOR_CACHE.move_to_end(key)
    while len(_SECTOR_CACHE) > _SECTOR_CACHE_MAX:
        _SECTOR_CACHE.popitem(last=False)


def _partition_for_character(character, multiplicity):
    multiplicity = int(multiplicity)
    if isinstance(character, Partition):
        return character
    if isinstance(character, (tuple, list)) and character and all(isinstance(x, int) for x in character):
        return Partition(tuple(int(x) for x in character))
    name = str(character).strip().lower()
    if name in {"trivial", "symmetric", "sym", "bosonic", "[n]"}:
        return Partition((multiplicity,))
    if name in {"sign", "antisymmetric", "anti", "fermionic", "[1^n]"}:
        return Partition(tuple(1 for _ in range(multiplicity)))
    raise ValueError(
        "character must be 'trivial', 'symmetric', 'sign', 'antisymmetric', "
        "a Partition, or explicit partition parts."
    )


def permutation_irrep_for_character(nin, lin, character):
    """Return the repeated-channel subgroup irrep for a named Young character."""

    nin, lin = _normalize_nl(nin, lin)
    subgroup = PermutationSubgroup.from_nl(nin, lin)
    if isinstance(character, PermutationIrrep):
        if character.subgroup != subgroup:
            raise ValueError("Permutation irrep subgroup does not match nin/lin.")
        return character
    if isinstance(character, (tuple, list)) and len(character) == len(subgroup.factors) and all(
        isinstance(item, (Partition, tuple, list)) for item in character
    ):
        partitions = tuple(
            item if isinstance(item, Partition) else Partition(tuple(int(x) for x in item))
            for item in character
        )
    else:
        partitions = tuple(_partition_for_character(character, factor.multiplicity) for factor in subgroup.factors)
    return PermutationIrrep(subgroup=subgroup, partitions=partitions)


def _cycle_type_class_size(cycle_type):
    counts = {}
    total = 0
    for length in cycle_type:
        length = int(length)
        counts[length] = counts.get(length, 0) + 1
        total += length
    denom = 1
    for length, count in counts.items():
        denom *= (int(length) ** int(count)) * factorial(int(count))
    return factorial(int(total)) // int(denom)


def _weight_trace_for_cycle_type(l_value, cycle_type):
    poly = {0: 1}
    for length in cycle_type:
        next_poly = {}
        for exponent, coeff in poly.items():
            for weight in range(-int(l_value), int(l_value) + 1):
                key = int(exponent) + int(length) * int(weight)
                next_poly[key] = next_poly.get(key, 0) + int(coeff)
        poly = next_poly
    return poly


def _convolve_weight_counts(left, right):
    out = {}
    for left_weight, left_coeff in left.items():
        for right_weight, right_coeff in right.items():
            weight = int(left_weight) + int(right_weight)
            out[weight] = out.get(weight, 0) + int(left_coeff) * int(right_coeff)
    return out


def _poly_add(left, right, scale=None):
    if scale is None:
        scale = 1
    scale = int(scale)
    out = dict(left)
    for exponent, coeff in right.items():
        value = int(out.get(int(exponent), 0)) + scale * int(coeff)
        if value:
            out[int(exponent)] = value
        elif int(exponent) in out:
            del out[int(exponent)]
    return out


def _poly_mul(left, right):
    out = {}
    for left_exponent, left_coeff in left.items():
        for right_exponent, right_coeff in right.items():
            exponent = int(left_exponent) + int(right_exponent)
            out[exponent] = out.get(exponent, 0) + int(left_coeff) * int(right_coeff)
    return {int(exponent): int(coeff) for exponent, coeff in out.items() if int(coeff) != 0}


def _complete_homogeneous_weight_polys(weights, max_degree):
    max_degree = int(max_degree)
    h = [{0: 1}] + [dict() for _ in range(max_degree)]
    for weight in tuple(int(x) for x in weights):
        next_h = [dict() for _ in range(max_degree + 1)]
        for degree in range(max_degree + 1):
            for exponent, coeff in h[degree].items():
                for count in range(max_degree - degree + 1):
                    new_degree = degree + count
                    new_exponent = int(exponent) + int(count) * int(weight)
                    next_h[new_degree][new_exponent] = (
                        next_h[new_degree].get(new_exponent, 0) + int(coeff)
                    )
        h = [{int(exponent): int(coeff) for exponent, coeff in poly.items() if int(coeff) != 0} for poly in next_h]
    return tuple(h)


def _det_poly(matrix):
    size = len(matrix)
    if size == 0:
        return {0: 1}
    out = {}
    for perm in permutations(range(size)):
        inversions = sum(1 for i in range(size) for j in range(i + 1, size) if int(perm[i]) > int(perm[j]))
        sign = -1 if inversions % 2 else 1
        term = {0: 1}
        for row, col in enumerate(perm):
            term = _poly_mul(term, matrix[row][col])
            if not term:
                break
        if term:
            out = _poly_add(out, term, scale=sign)
    return out


@lru_cache(maxsize=512)
def _schur_functor_block_weight_counts_cached(multiplicity, l_value, partition_parts):
    """Return weights of the Schur functor ``S_partition(V_l)``.

    This is a Schur-Weyl compressed count path.  It evaluates the Schur
    polynomial by the Jacobi-Trudi determinant in the Laurent monomials
    ``x^{-l}, ..., x^l``.  It avoids symmetric-group conjugacy-class
    enumeration and is exact over integers.
    """

    multiplicity = int(multiplicity)
    l_value = int(l_value)
    partition = Partition(tuple(int(x) for x in partition_parts))
    if int(partition.size) != multiplicity:
        raise ValueError("Partition size must match tensor multiplicity.")
    local_dim = 2 * int(l_value) + 1
    if int(partition.height) > int(local_dim):
        return tuple()
    if partition.is_sign():
        if multiplicity > local_dim:
            return tuple()
        weights = tuple(range(-l_value, l_value + 1))
        polys = [{0: 1}] + [dict() for _ in range(multiplicity)]
        for weight in weights:
            for degree in range(multiplicity - 1, -1, -1):
                for exponent, coeff in tuple(polys[degree].items()):
                    target = degree + 1
                    new_exponent = int(exponent) + int(weight)
                    polys[target][new_exponent] = polys[target].get(new_exponent, 0) + int(coeff)
        counts = polys[multiplicity]
        return tuple(sorted((int(weight), int(value)) for weight, value in counts.items() if int(value) != 0))
    max_degree = max(
        int(partition.parts[row]) - (row + 1) + (col + 1)
        for row in range(int(partition.height))
        for col in range(int(partition.height))
    )
    if max_degree < 0:
        return tuple()
    h = _complete_homogeneous_weight_polys(tuple(range(-l_value, l_value + 1)), int(max_degree))
    matrix = []
    for row in range(int(partition.height)):
        matrix_row = []
        for col in range(int(partition.height)):
            degree = int(partition.parts[row]) - (row + 1) + (col + 1)
            if degree < 0:
                matrix_row.append({})
            else:
                matrix_row.append(dict(h[int(degree)]))
        matrix.append(tuple(matrix_row))
    counts = _det_poly(tuple(matrix))
    return tuple(sorted((int(weight), int(value)) for weight, value in counts.items() if int(value) != 0))


@lru_cache(maxsize=512)
def _projected_block_weight_counts_cached(multiplicity, l_value, partition_parts):
    multiplicity = int(multiplicity)
    l_value = int(l_value)
    partition = Partition(tuple(int(x) for x in partition_parts))
    schur_counts = _schur_functor_block_weight_counts_cached(multiplicity, l_value, tuple(int(x) for x in partition.parts))
    if schur_counts or int(partition.height) > 2 * int(l_value) + 1:
        return schur_counts
    weight_counts = {}
    for cycle_type in _partition_parts_of_n(multiplicity):
        class_size = _cycle_type_class_size(cycle_type)
        character = symmetric_group_character(partition, cycle_type)
        weight_trace = _weight_trace_for_cycle_type(l_value, cycle_type)
        for weight, coeff in weight_trace.items():
            weight_counts[weight] = weight_counts.get(weight, 0) + int(class_size) * int(character) * int(coeff)
    group_order = int(factorial(multiplicity))
    out = []
    for weight, value in weight_counts.items():
        exact = Fraction(int(value), group_order)
        if exact.denominator != 1:
            raise ArithmeticError("Projected block weight count was not integral.")
        out.append((int(weight), int(exact)))
    return tuple(sorted(out))


@lru_cache(maxsize=512)
def _dusson_barthelemy_trivial_block_weight_counts_cached(multiplicity, l_value):
    """Return weight counts for the trivial ``S_N`` sector ``Sym^N(V_l)``.

    This is the count-vector form used for the GE-PI/trivial permutation
    sector in Barthelemy-Dusson-Hernandez-Zhang, arXiv:2604.01975v2,
    Theorem 3.19 and Example 3.20.  For one repeated block it counts
    nonnegative vectors ``n_m`` with ``sum_m n_m = N`` and weight
    ``sum_m m n_m``.  The final SO(3)/SU(2) multiplicity is recovered from
    highest-weight differences ``w_L - w_{L+1}``.
    """

    multiplicity = int(multiplicity)
    l_value = int(l_value)
    h = _complete_homogeneous_weight_polys(tuple(range(-l_value, l_value + 1)), int(multiplicity))
    return tuple(
        sorted(
            (int(weight), int(count))
            for weight, count in h[int(multiplicity)].items()
            if int(count) != 0
        )
    )


def dusson_barthelemy_trivial_sector_counts(nin, lin):
    """Return trivial-permutation SO(3) counts by the count-vector formula.

    The input is interpreted through the same repeated-channel grouping as the
    YE3T/ACE label helpers.  For each repeated ``(n,l)`` block of size ``N``,
    this computes the weight counts of ``Sym^N(V_l)`` by complete homogeneous
    Laurent polynomials.  The block weights are convolved and converted to
    output-angular-momentum multiplicities by ``mult(L) = w_L - w_{L+1}``.

    Reference context: Barthelemy, Dusson, Hernandez, and Zhang,
    arXiv:2604.01975v2, formulate GE-PI dimensions using permutation classes
    / count vectors and derive explicit SO(3)/SU(2) dimensionality formulas
    in Section 3, including Theorem 3.19 and Example 3.20.
    """

    nin = tuple(int(x) for x in nin)
    lin = tuple(int(x) for x in lin)
    subgroup = PermutationSubgroup.from_nl(nin, lin)
    weight_counts = {0: 1}
    max_L = 0
    for factor in subgroup.factors:
        block_counts = dict(
            _dusson_barthelemy_trivial_block_weight_counts_cached(
                int(factor.multiplicity),
                int(factor.l),
            )
        )
        max_L += int(factor.multiplicity) * int(factor.l)
        weight_counts = _convolve_weight_counts(weight_counts, block_counts)
    counts_by_L = {}
    for L_value in range(0, int(max_L) + 1):
        multiplicity_L = int(weight_counts.get(int(L_value), 0)) - int(weight_counts.get(int(L_value) + 1, 0))
        if multiplicity_L:
            counts_by_L[int(L_value)] = int(multiplicity_L)
    return counts_by_L


def _character_count_payload(
    nin,
    lin,
    permutation_irrep,
    spatial_symmetry="SO3_legacy",
):
    if len(permutation_irrep.subgroup.factors) == 0:
        return None
    if sum(int(factor.multiplicity) for factor in permutation_irrep.subgroup.factors) != len(nin):
        return None
    weight_counts = {0: 1}
    for factor, partition in zip(permutation_irrep.subgroup.factors, permutation_irrep.partitions, strict=True):
        block_counts = dict(
            _projected_block_weight_counts_cached(
                int(factor.multiplicity),
                int(factor.l),
                tuple(int(x) for x in partition.parts),
            )
        )
        weight_counts = _convolve_weight_counts(weight_counts, block_counts)
    counts_by_L = {}
    max_L = sum(int(factor.multiplicity) * int(factor.l) for factor in permutation_irrep.subgroup.factors)
    for L_value in range(0, max_L + 1):
        multiplicity_L = int(weight_counts.get(L_value, 0)) - int(weight_counts.get(L_value + 1, 0))
        if multiplicity_L:
            counts_by_L[int(L_value)] = int(multiplicity_L)
    if spatial_symmetry not in {"O3", "SO3_legacy"}:
        raise ValueError("spatial_symmetry must be 'O3' or 'SO3_legacy'")
    parity = (
        1 if sum(int(value) for value in lin) % 2 == 0 else -1
    ) if spatial_symmetry == "O3" else None
    labels_by_L = {
        int(L_value): tuple(
            CoupledIrrepLabel(
                angular=AngularIrrep(int(L_value)),
                permutation=permutation_irrep,
                multiplicity_index=int(copy_index),
                parity=parity,
            )
            for copy_index in range(int(count))
        )
        for L_value, count in counts_by_L.items()
    }
    projected_dim = sum(int(count) * (2 * int(L_value) + 1) for L_value, count in counts_by_L.items())
    return counts_by_L, labels_by_L, int(projected_dim)


def generalized_sector_counts(
    nin,
    lin,
    permutation_irrep,
    count_only=False,
    spatial_symmetry="SO3_legacy",
):
    """Build exact small-``N`` Young-sector counts with provenance."""

    nin, lin = _normalize_nl(nin, lin)
    spatial_symmetry = str(spatial_symmetry)
    if spatial_symmetry not in {"O3", "SO3_legacy"}:
        raise ValueError("spatial_symmetry must be 'O3' or 'SO3_legacy'")
    if not isinstance(permutation_irrep, PermutationIrrep):
        permutation_irrep = permutation_irrep_for_character(nin, lin, permutation_irrep)
    if bool(count_only):
        payload = _character_count_payload(
            nin,
            lin,
            permutation_irrep,
            spatial_symmetry=spatial_symmetry,
        )
        if payload is not None:
            counts_by_L, labels_by_L, projected_dim = payload
            return YoungSectorCounts(
                nin=nin,
                lin=lin,
                permutation_irrep=permutation_irrep,
                counts_by_L=counts_by_L,
                labels_by_L=labels_by_L,
                projected_dim=int(projected_dim),
                provenance="character_count",
                codepath="young_subgroup_character_count",
                detail="Count-only Young/SO(3) multiplicities from symmetric-group conjugacy classes.",
                sector=None,
            )
    key = _sector_cache_key(
        nin,
        lin,
        permutation_irrep,
        spatial_symmetry=spatial_symmetry,
    )
    cached = _cache_get(key)
    if cached is not None:
        return YoungSectorCounts(
            nin=nin,
            lin=lin,
            permutation_irrep=permutation_irrep,
            counts_by_L=dict(cached.counts_by_L),
            labels_by_L=cached.labels_by_L,
            projected_dim=int(cached.projected_dim),
            provenance="cached",
            codepath=cached.codepath,
            detail="Reused cached exact projector sector.",
            sector=cached,
        )
    try:
        sector = _build_exact_symbolic_projector_sector(
            nin,
            lin,
            permutation_irrep,
            spatial_symmetry=spatial_symmetry,
        )
    except NotImplementedError as exc:
        return YoungSectorCounts(
            nin=nin,
            lin=lin,
            permutation_irrep=permutation_irrep,
            counts_by_L={},
            labels_by_L={},
            projected_dim=None,
            provenance="unsupported_rank",
            codepath="exact_projector",
            detail=str(exc),
            sector=None,
        )
    _cache_put(key, sector)
    return YoungSectorCounts(
        nin=nin,
        lin=lin,
        permutation_irrep=permutation_irrep,
        counts_by_L=dict(sector.counts_by_L),
        labels_by_L=sector.labels_by_L,
        projected_dim=int(sector.projected_dim),
        provenance="exact_projector",
        codepath=sector.codepath,
        detail="Built exact symbolic Young-sector projector.",
        sector=sector,
    )


def _target_irrep_matches(candidate, target_irrep):
    if target_irrep is None:
        return True
    return _permutation_irrep_signature_unordered(candidate) == _permutation_irrep_signature_unordered(target_irrep)


def young_product_paths(left_label, right_label, target_irrep=None, target_L=None, target_parity=None):
    """Enumerate merged Young/permutation and angular product paths."""

    if not isinstance(left_label, CoupledIrrepLabel) or not isinstance(right_label, CoupledIrrepLabel):
        raise TypeError("young_product_paths expects CoupledIrrepLabel inputs.")
    if (left_label.parity is None) != (right_label.parity is None):
        raise ValueError("cannot mix signed O(3) and unspecified SO(3)-legacy parity")
    output_parity = (
        None
        if left_label.parity is None
        else int(left_label.parity) * int(right_label.parity)
    )
    requested_parity = _normalize_spatial_parity(target_parity)
    if requested_parity is not None:
        if output_parity is None:
            raise ValueError("target_parity requires signed O(3) input labels")
        if output_parity != requested_parity:
            return ()
    allowed_L = tuple(range(abs(int(left_label.angular.l) - int(right_label.angular.l)), int(left_label.angular.l) + int(right_label.angular.l) + 1))
    if target_L is not None:
        allowed_L = tuple(L for L in allowed_L if int(L) == int(target_L))
    paths = []
    for term in _merged_permutation_irrep_product_terms(left_label.permutation, right_label.permutation):
        if not _target_irrep_matches(term.permutation_irrep, target_irrep):
            continue
        for L in allowed_L:
            for copy_index in range(int(term.multiplicity)):
                paths.append(
                    YoungProductPath(
                        left_label=left_label,
                        right_label=right_label,
                        output_label=CoupledIrrepLabel(
                            angular=AngularIrrep(int(L)),
                            permutation=term.permutation_irrep,
                            multiplicity_index=int(copy_index),
                            parity=output_parity,
                        ),
                        permutation_multiplicity=int(term.multiplicity),
                        provenance="character",
                        codepath="merged_character_product",
                    )
                )
    return tuple(paths)


def _channel_multiset(nin, lin):
    counts = {}
    for n, l in zip(nin, lin):
        key = (int(n), int(l))
        counts[key] = counts.get(key, 0) + 1
    return tuple((key[0], key[1], int(count)) for key, count in sorted(counts.items()))


def _raw_tensor_dim(lin):
    out = 1
    for l_value in tuple(int(x) for x in lin):
        out *= 2 * int(l_value) + 1
    return int(out)


def _carrier_dim(permutation_irrep):
    out = 1
    for partition in permutation_irrep.partitions:
        out *= int(partition.dimension)
    return int(out)


def _partition_signature(permutation_irrep):
    return tuple(tuple(int(x) for x in partition.parts) for partition in permutation_irrep.partitions)


def _partition_parts_of_n(n, max_part=None):
    n = int(n)
    if max_part is None or int(max_part) > n:
        max_part = n
    if n == 0:
        return (tuple(),)
    out = []
    for first in range(int(max_part), 0, -1):
        for rest in _partition_parts_of_n(n - first, first):
            out.append((int(first),) + tuple(int(x) for x in rest))
    return tuple(out)


@lru_cache(maxsize=256)
def _all_permutation_irreps_for_pattern(nin, lin):
    subgroup = PermutationSubgroup.from_nl(tuple(nin), tuple(lin))
    choices = [
        tuple(Partition(parts) for parts in _partition_parts_of_n(int(factor.multiplicity)))
        for factor in subgroup.factors
    ]
    irreps = []

    def _rec(index, parts):
        if index == len(choices):
            irreps.append(PermutationIrrep(subgroup=subgroup, partitions=tuple(parts)))
            return
        for partition in choices[index]:
            _rec(index + 1, (*parts, partition))

    _rec(0, tuple())
    return tuple(irreps)


@lru_cache(maxsize=256)
def _partition_count_of_n(n, max_part=None):
    n = int(n)
    if max_part is None or int(max_part) > n:
        max_part = n
    max_part = int(max_part)
    if n == 0:
        return 1
    if n < 0 or max_part <= 0:
        return 0
    return int(_partition_count_of_n(n, max_part - 1) + _partition_count_of_n(n - max_part, max_part))


def _all_irrep_count_for_pattern(nin, lin):
    subgroup = PermutationSubgroup.from_nl(tuple(nin), tuple(lin))
    count = 1
    for factor in subgroup.factors:
        count *= _partition_count_of_n(int(factor.multiplicity))
    return int(count)


def all_young_character_irreps_for_pattern(nin, lin):
    """Return all Young-sector irreps for the repeated-channel subgroup.

    This enumerates every partition choice for every factor in ``G_nu``.  It is
    a label-space operation only; it does not build projectors, Gram matrices,
    SVDs, or row-reduced bases.
    """

    nin, lin = _normalize_nl(nin, lin)
    return _all_permutation_irreps_for_pattern(nin, lin)


def analytical_young_e3_sector_catalog(nin, lin, *, include_zero_records=False):
    """Catalog exact Young-character/``L_R`` sectors by character counts.

    The count path uses symmetric-group conjugacy classes and SO(3)
    highest-weight weight differences.  It intentionally avoids projector
    matrix construction, Gram-Schmidt, SVD, Gramian rank checks, and row
    reduction.  The records therefore provide exact sector multiplicities and
    labels, not coefficient matrices for evaluating basis functions.
    """

    nin, lin = _normalize_nl(nin, lin)
    irreps = all_young_character_irreps_for_pattern(nin, lin)
    allowed_target_Ls = tuple(sorted(_reachable_total_angular_momenta(lin)))
    records = []
    total_joint_sector_dim = 0
    total_full_sector_dim = 0
    for permutation_irrep in irreps:
        counts = generalized_sector_counts(nin, lin, permutation_irrep, count_only=True)
        carrier_dim = _carrier_dim(permutation_irrep)
        target_Ls = allowed_target_Ls if bool(include_zero_records) else tuple(sorted(counts.counts_by_L))
        for L_R in target_Ls:
            multiplicity = int(counts.counts_by_L.get(int(L_R), 0))
            labels = tuple(counts.labels_by_L.get(int(L_R), tuple())) if counts.labels_by_L is not None else tuple()
            joint_sector_dim = int(multiplicity) * (2 * int(L_R) + 1)
            full_sector_dim = int(joint_sector_dim) * int(carrier_dim)
            total_joint_sector_dim += int(joint_sector_dim)
            total_full_sector_dim += int(full_sector_dim)
            if multiplicity == 0 and not bool(include_zero_records):
                continue
            records.append(
                YoungAnalyticalSectorRecord(
                    permutation_irrep=permutation_irrep,
                    partition_signature=_partition_signature(permutation_irrep),
                    L_R=int(L_R),
                    multiplicity=int(multiplicity),
                    carrier_dim=int(carrier_dim),
                    joint_sector_dim=int(joint_sector_dim),
                    full_sector_dim=int(full_sector_dim),
                    labels=labels,
                    provenance=counts.provenance,
                    codepath=counts.codepath,
                    detail=(
                        "Exact count-only Young/E3 sector from subgroup characters "
                        "and SO(3) weight multiplicities; no coefficient basis was materialized."
                    ),
                )
            )
    raw_dim = _raw_tensor_dim(lin)
    return YoungAnalyticalSectorCatalog(
        nin=nin,
        lin=lin,
        channel_multiset=_channel_multiset(nin, lin),
        subgroup=PermutationSubgroup.from_nl(nin, lin),
        allowed_target_Ls=allowed_target_Ls,
        irreps=tuple(irreps),
        records=tuple(records),
        raw_tensor_dim=int(raw_dim),
        total_joint_sector_dim=int(total_joint_sector_dim),
        total_full_sector_dim=int(total_full_sector_dim),
        complete=int(total_full_sector_dim) == int(raw_dim),
        provenance="character_count",
        codepath="analytical_young_e3_sector_catalog",
        detail=(
            "Catalog over all repeated-channel Young characters and all allowed target angular momenta. "
            "Completeness is checked by sum multiplicity*(2L_R+1)*dim(Specht factors) = raw tensor dimension."
        ),
    )


def validate_analytical_young_e3_sector_catalog(catalog):
    """Validate dimension accounting and count-only provenance for a catalog."""

    dimension_complete = int(catalog.total_full_sector_dim) == int(catalog.raw_tensor_dim)
    records_match_count_only = all(
        str(record.codepath) == "young_subgroup_character_count"
        and str(record.provenance) == "character_count"
        for record in catalog.records
    )
    passed = bool(dimension_complete and records_match_count_only and bool(catalog.complete))
    detail = "ok" if passed else (
        f"dimension_complete={dimension_complete}, "
        f"records_match_count_only={records_match_count_only}, complete={bool(catalog.complete)}"
    )
    return YoungAnalyticalSectorCatalogValidation(
        catalog=catalog,
        dimension_complete=bool(dimension_complete),
        records_match_count_only=bool(records_match_count_only),
        passed=passed,
        detail=detail,
    )


def specht_partition_family(partition):
    """Classify one partition into an exact constructive family.

    The family name describes a planned coefficient construction.  It does not
    by itself materialize coefficient tensors.
    """

    if not isinstance(partition, Partition):
        partition = Partition(tuple(int(x) for x in partition))
    size = int(partition.size)
    parts = tuple(int(x) for x in partition.parts)
    if partition.is_trivial():
        return "trivial_symmetric_power"
    if partition.is_sign():
        return "sign_exterior_power"
    if size >= 2 and parts == (size - 1, 1):
        return "standard_zero_sum"
    if len(parts) >= 2 and parts[1:] == tuple(1 for _ in parts[1:]) and int(parts[0]) >= 1:
        return "hook_standard_exterior_power"
    return "generic_young_yamanouchi"


def _irrep_family_signature(permutation_irrep):
    return tuple(specht_partition_family(partition) for partition in permutation_irrep.partitions)


def _construction_kind_for_family(family):
    if family == "trivial_symmetric_power":
        return "compressed_symmetric_power"
    if family == "sign_exterior_power":
        return "compressed_exterior_power"
    if family == "standard_zero_sum":
        return "compressed_standard_residual"
    if family == "hook_standard_exterior_power":
        return "compressed_hook_exterior_standard"
    return "young_yamanouchi_seminormal"


def _compression_kind_for_family_signature(family_signature):
    family_signature = tuple(str(item) for item in family_signature)
    if all(item != "generic_young_yamanouchi" for item in family_signature):
        return "all_factors_compressed"
    if any(item != "generic_young_yamanouchi" for item in family_signature):
        return "mixed_compressed_and_generic"
    return "generic_specht"


def _normalize_sector_families(sector_families):
    if sector_families is None:
        sector_families = ("all",)
    if isinstance(sector_families, str):
        sector_families = (sector_families,)
    normalized = set()
    for item in sector_families:
        name = str(item).strip().lower().replace("-", "_")
        if name in {"all", "*"}:
            normalized.add("all")
        elif name in {"compressed", "special", "special_compressed"}:
            normalized.update(
                {
                    "trivial_symmetric_power",
                    "sign_exterior_power",
                    "standard_zero_sum",
                    "hook_standard_exterior_power",
                }
            )
        elif name in {"trivial", "symmetric", "sym", "trivial_symmetric_power"}:
            normalized.add("trivial_symmetric_power")
        elif name in {"sign", "antisymmetric", "exterior", "sign_exterior_power"}:
            normalized.add("sign_exterior_power")
        elif name in {"standard", "standard_zero_sum"}:
            normalized.add("standard_zero_sum")
        elif name in {"hook", "hooks", "hook_standard_exterior_power"}:
            normalized.add("hook_standard_exterior_power")
        elif name in {"generic", "generic_young_yamanouchi", "young_yamanouchi"}:
            normalized.add("generic_young_yamanouchi")
        else:
            raise ValueError(f"Unknown Specht sector family {item!r}.")
    if "all" in normalized:
        return ("all",)
    return tuple(sorted(normalized))


def _sector_family_selected(family_signature, sector_families):
    sector_families = _normalize_sector_families(sector_families)
    if sector_families == ("all",):
        return True
    return all(str(family) in sector_families for family in family_signature)


def _unique_partitions(partitions):
    out = []
    seen = set()
    for partition in partitions:
        if not isinstance(partition, Partition):
            partition = Partition(tuple(int(x) for x in partition))
        key = tuple(int(x) for x in partition.parts)
        if key in seen:
            continue
        seen.add(key)
        out.append(partition)
    return tuple(out)


def _compressed_partitions_for_factor(multiplicity, sector_families):
    multiplicity = int(multiplicity)
    sector_families = _normalize_sector_families(sector_families)
    if sector_families == ("all",) or "generic_young_yamanouchi" in sector_families:
        return tuple(Partition(parts) for parts in _partition_parts_of_n(multiplicity))
    partitions = []
    if "trivial_symmetric_power" in sector_families:
        partitions.append(Partition((multiplicity,)))
    if "sign_exterior_power" in sector_families:
        partitions.append(Partition(tuple(1 for _ in range(multiplicity))))
    if "standard_zero_sum" in sector_families and multiplicity >= 2:
        partitions.append(Partition((multiplicity - 1, 1)))
    if "hook_standard_exterior_power" in sector_families and multiplicity >= 3:
        for r in range(2, multiplicity - 1):
            partitions.append(Partition((multiplicity - r,) + tuple(1 for _ in range(r))))
    return _unique_partitions(partitions)


def _normalize_explicit_partition_signatures(explicit_partition_signatures):
    if explicit_partition_signatures is None:
        return tuple()
    out = []
    for signature in explicit_partition_signatures:
        out.append(tuple(tuple(int(part) for part in parts) for parts in signature))
    return tuple(out)


def _irrep_from_partition_signature(subgroup, partition_signature):
    partition_signature = tuple(tuple(int(part) for part in parts) for parts in partition_signature)
    if len(partition_signature) != len(subgroup.factors):
        raise ValueError(
            f"Explicit partition signature {partition_signature!r} has {len(partition_signature)} factors, "
            f"but subgroup has {len(subgroup.factors)} factors."
        )
    return PermutationIrrep(
        subgroup=subgroup,
        partitions=tuple(Partition(parts) for parts in partition_signature),
    )


def _selected_irreps_from_factor_partitions(subgroup, choices):
    irreps = []

    def _rec(index, parts):
        if index == len(choices):
            irreps.append(PermutationIrrep(subgroup=subgroup, partitions=tuple(parts)))
            return
        for partition in choices[index]:
            _rec(index + 1, (*parts, partition))

    _rec(0, tuple())
    return tuple(irreps)


def selected_young_character_irreps_for_pattern(
    nin,
    lin,
    *,
    sector_families=("all",),
    max_specht_dim=None,
    max_factor_specht_dim=None,
    explicit_partition_signatures=None,
):
    """Return selected repeated-channel Young irreps without materializing bases.

    ``sector_families=("all",)`` enumerates every Specht partition.  Large-rank
    callers can request only compressed families such as ``"trivial"``,
    ``"standard"``, ``"hook"``, or ``"compressed"``.
    """

    nin, lin = _normalize_nl(nin, lin)
    sector_families = _normalize_sector_families(sector_families)
    explicit_partition_signatures = _normalize_explicit_partition_signatures(explicit_partition_signatures)
    subgroup = PermutationSubgroup.from_nl(nin, lin)
    choices = tuple(
        _compressed_partitions_for_factor(int(factor.multiplicity), sector_families)
        for factor in subgroup.factors
    )
    candidates = list(_selected_irreps_from_factor_partitions(subgroup, choices))
    candidates.extend(_irrep_from_partition_signature(subgroup, signature) for signature in explicit_partition_signatures)
    selected = []
    seen = set()
    for irrep in candidates:
        partition_signature = _partition_signature(irrep)
        if partition_signature in seen:
            continue
        seen.add(partition_signature)
        family_signature = _irrep_family_signature(irrep)
        explicit = partition_signature in explicit_partition_signatures
        if not explicit and sector_families != ("all",) and not _sector_family_selected(family_signature, sector_families):
            continue
        carrier_dim = _carrier_dim(irrep)
        if max_specht_dim is not None and int(carrier_dim) > int(max_specht_dim) and not explicit:
            continue
        if max_factor_specht_dim is not None and any(
            int(partition.dimension) > int(max_factor_specht_dim)
            for partition in irrep.partitions
        ) and not explicit:
            continue
        selected.append(irrep)
    return tuple(selected)


def _specht_plan_cache_key(
    nin,
    lin,
    sector_families,
    include_zero_records,
    max_specht_dim,
    max_factor_specht_dim,
    explicit_partition_signatures,
):
    return (
        tuple(int(x) for x in nin),
        tuple(int(x) for x in lin),
        tuple(str(x) for x in _normalize_sector_families(sector_families)),
        bool(include_zero_records),
        None if max_specht_dim is None else int(max_specht_dim),
        None if max_factor_specht_dim is None else int(max_factor_specht_dim),
        _normalize_explicit_partition_signatures(explicit_partition_signatures),
    )


@lru_cache(maxsize=128)
def _young_specht_basis_construction_plan_cached(cache_key):
    nin, lin, sector_families, include_zero_records, max_specht_dim, max_factor_specht_dim, explicit_signatures = cache_key
    nin = tuple(int(x) for x in nin)
    lin = tuple(int(x) for x in lin)
    available_irrep_count = _all_irrep_count_for_pattern(nin, lin)
    selected_irreps = selected_young_character_irreps_for_pattern(
        nin,
        lin,
        sector_families=sector_families,
        max_specht_dim=max_specht_dim,
        max_factor_specht_dim=max_factor_specht_dim,
        explicit_partition_signatures=explicit_signatures,
    )
    allowed_target_Ls = tuple(sorted(_reachable_total_angular_momenta(lin)))
    records = []
    selected_full_sector_dim = 0
    for permutation_irrep in selected_irreps:
        counts = generalized_sector_counts(nin, lin, permutation_irrep, count_only=True)
        carrier_dim = _carrier_dim(permutation_irrep)
        partition_signature = _partition_signature(permutation_irrep)
        family_signature = _irrep_family_signature(permutation_irrep)
        construction_kinds = tuple(_construction_kind_for_family(family) for family in family_signature)
        compression_kind = _compression_kind_for_family_signature(family_signature)
        target_Ls = allowed_target_Ls if bool(include_zero_records) else tuple(sorted(counts.counts_by_L))
        for L_R in target_Ls:
            multiplicity = int(counts.counts_by_L.get(int(L_R), 0))
            if multiplicity == 0 and not bool(include_zero_records):
                continue
            labels = tuple(counts.labels_by_L.get(int(L_R), tuple())) if counts.labels_by_L is not None else tuple()
            joint_sector_dim = int(multiplicity) * (2 * int(L_R) + 1)
            full_sector_dim = int(joint_sector_dim) * int(carrier_dim)
            selected_full_sector_dim += int(full_sector_dim)
            records.append(
                YoungSpechtConstructionRecord(
                    permutation_irrep=permutation_irrep,
                    partition_signature=partition_signature,
                    family_signature=family_signature,
                    L_R=int(L_R),
                    multiplicity=int(multiplicity),
                    carrier_dim=int(carrier_dim),
                    joint_sector_dim=int(joint_sector_dim),
                    full_sector_dim=int(full_sector_dim),
                    construction_kind=construction_kinds,
                    compression_kind=compression_kind,
                    exact=True,
                    labels=labels,
                    provenance=counts.provenance,
                    codepath="specht_sector_construction_plan",
                    detail=(
                        "Exact sector record for planned constructive Specht/Young-E3 schedule. "
                        "Coefficient tensors are not materialized by this planning record."
                    ),
                )
            )
    raw_dim = _raw_tensor_dim(lin)
    enumerated_all = int(len(selected_irreps)) == int(available_irrep_count)
    return YoungSpechtBasisConstructionPlan(
        nin=nin,
        lin=lin,
        channel_multiset=_channel_multiset(nin, lin),
        subgroup=PermutationSubgroup.from_nl(nin, lin),
        sector_families=tuple(sector_families),
        allowed_target_Ls=allowed_target_Ls,
        available_irrep_count=int(available_irrep_count),
        selected_irrep_count=len(selected_irreps),
        enumerated_all_sectors=bool(enumerated_all),
        records=tuple(records),
        raw_tensor_dim=int(raw_dim),
        selected_full_sector_dim=int(selected_full_sector_dim),
        complete=bool(enumerated_all and int(selected_full_sector_dim) == int(raw_dim)),
        cache_key=cache_key,
        provenance="character_count",
        codepath="cached_specht_sector_construction_plan",
        detail=(
            "One-time cached sector plan. Runtime schedules should be compiled from these records "
            "and reused rather than rebuilding representation-theoretic metadata per evaluation."
        ),
    )


def young_specht_basis_construction_plan(
    nin,
    lin,
    *,
    sector_families=("all",),
    include_zero_records=False,
    max_specht_dim=None,
    max_factor_specht_dim=None,
    explicit_partition_signatures=None,
):
    """Build a cached exact plan for full or selected Specht/Young-E3 sectors."""

    nin, lin = _normalize_nl(nin, lin)
    cache_key = _specht_plan_cache_key(
        nin,
        lin,
        sector_families,
        include_zero_records,
        max_specht_dim,
        max_factor_specht_dim,
        explicit_partition_signatures,
    )
    return _young_specht_basis_construction_plan_cached(cache_key)


def compile_young_specht_basis_schedule(*args, **kwargs):
    """Return the cached exact sector plan used as the current schedule stub.

    This intentionally does not introduce a numerical fast path.  It provides a
    stable cache boundary for future coefficient-schedule materialization.
    """

    return young_specht_basis_construction_plan(*args, **kwargs)


def validate_young_specht_basis_construction_plan(plan):
    """Validate exactness and dimension accounting of a Specht plan."""

    records_are_exact = all(bool(record.exact) for record in plan.records)
    selected_dim = sum(int(record.full_sector_dim) for record in plan.records)
    selection_consistent = (
        int(selected_dim) == int(plan.selected_full_sector_dim)
        and int(plan.selected_irrep_count) <= int(plan.available_irrep_count)
        and int(plan.selected_full_sector_dim) <= int(plan.raw_tensor_dim)
    )
    dimension_complete = (not bool(plan.enumerated_all_sectors)) or (
        int(plan.selected_full_sector_dim) == int(plan.raw_tensor_dim) and bool(plan.complete)
    )
    passed = bool(records_are_exact and selection_consistent and dimension_complete)
    detail = "ok" if passed else (
        f"records_are_exact={records_are_exact}, selection_consistent={selection_consistent}, "
        f"dimension_complete={dimension_complete}"
    )
    return YoungSpechtBasisConstructionPlanValidation(
        plan=plan,
        dimension_complete=bool(dimension_complete),
        records_are_exact=bool(records_are_exact),
        selection_consistent=bool(selection_consistent),
        passed=passed,
        detail=detail,
    )


def _schedule_for_tree_type(rank, tree_type):
    tree_type = str(tree_type).strip().lower()
    if tree_type == "balanced":
        return _build_balanced_pairwise_schedule(int(rank))
    if tree_type in {"left", "left_associated", "left-associated"}:
        return _build_left_schedule(int(rank))
    raise ValueError(f"Unsupported Young-Yamanouchi/CG tree type {tree_type!r}.")


def _leaf_labels_for_pattern(nin, lin):
    labels = []
    for n_value, l_value in zip(tuple(int(x) for x in nin), tuple(int(x) for x in lin), strict=True):
        leaf_irrep = permutation_irrep_for_character((int(n_value),), (int(l_value),), "trivial")
        labels.append(
            CoupledIrrepLabel(
                angular=AngularIrrep(int(l_value)),
                permutation=leaf_irrep,
                multiplicity_index=0,
            )
        )
    return tuple(labels)


def _nonzero_root_labels_from_plan(plan, target_Ls):
    target_Ls = None if target_Ls is None else tuple(int(x) for x in target_Ls)
    labels = []
    for record in plan.records:
        if int(record.multiplicity) == 0:
            continue
        if target_Ls is not None and int(record.L_R) not in target_Ls:
            continue
        labels.extend(record.labels)
    return tuple(labels)


def _yamanouchi_schedule_cache_key(
    nin,
    lin,
    tree_type,
    sector_families,
    target_Ls,
    max_specht_dim,
    max_factor_specht_dim,
    explicit_partition_signatures,
):
    return (
        tuple(int(x) for x in nin),
        tuple(int(x) for x in lin),
        str(tree_type).strip().lower(),
        tuple(str(x) for x in _normalize_sector_families(sector_families)),
        None if target_Ls is None else tuple(int(x) for x in target_Ls),
        None if max_specht_dim is None else int(max_specht_dim),
        None if max_factor_specht_dim is None else int(max_factor_specht_dim),
        _normalize_explicit_partition_signatures(explicit_partition_signatures),
    )


@lru_cache(maxsize=128)
def _young_yamanouchi_cg_tree_schedule_cached(cache_key):
    nin, lin, tree_type, sector_families, target_Ls, max_specht_dim, max_factor_specht_dim, explicit_signatures = cache_key
    plan = young_specht_basis_construction_plan(
        nin,
        lin,
        sector_families=sector_families,
        include_zero_records=False,
        max_specht_dim=max_specht_dim,
        max_factor_specht_dim=max_factor_specht_dim,
        explicit_partition_signatures=explicit_signatures,
    )
    tree_schedule = _schedule_for_tree_type(len(tuple(nin)), tree_type)
    leaf_labels = _leaf_labels_for_pattern(nin, lin)
    root_labels = _nonzero_root_labels_from_plan(plan, target_Ls)
    return YoungYamanouchiCGTreeSchedule(
        nin=tuple(int(x) for x in nin),
        lin=tuple(int(x) for x in lin),
        tree_type=str(tree_type),
        sector_families=tuple(sector_families),
        target_Ls=None if target_Ls is None else tuple(int(x) for x in target_Ls),
        construction_plan=plan,
        tree_schedule=tree_schedule,
        leaf_labels=leaf_labels,
        root_labels=root_labels,
        coefficient_convention="young_yamanouchi_seminormal_plus_ye3t_cg_tree",
        uses_young_yamanouchi_carriers=True,
        uses_ye3t_cg_trees=True,
        uses_symbolic_basis_extraction=False,
        materializes_coefficients=False,
        status="schedule_metadata_implemented_coefficients_pending",
        cache_key=cache_key,
        detail=(
            "This is the selected finite-rank coefficient convention boundary. "
            "It fixes Young-Yamanouchi/seminormal Specht carriers and a YE3T CG coupling tree, "
            "but arbitrary-sector coefficient tensors are not materialized yet."
        ),
    )


def young_yamanouchi_cg_tree_schedule(
    nin,
    lin,
    *,
    tree_type="balanced",
    sector_families=("all",),
    target_Ls=None,
    max_specht_dim=None,
    max_factor_specht_dim=None,
    explicit_partition_signatures=None,
):
    """Build the cached Young-Yamanouchi/seminormal plus YE3T-CG tree schedule."""

    nin, lin = _normalize_nl(nin, lin)
    cache_key = _yamanouchi_schedule_cache_key(
        nin,
        lin,
        tree_type,
        sector_families,
        target_Ls,
        max_specht_dim,
        max_factor_specht_dim,
        explicit_partition_signatures,
    )
    return _young_yamanouchi_cg_tree_schedule_cached(cache_key)


def validate_young_yamanouchi_cg_tree_schedule(schedule):
    """Validate metadata consistency of a Young-Yamanouchi/CG-tree schedule."""

    plan_valid = validate_young_specht_basis_construction_plan(schedule.construction_plan).passed
    tree_metadata_present = (
        schedule.tree_schedule is not None
        and int(schedule.tree_schedule.num_inputs) == len(tuple(schedule.nin))
        and len(tuple(schedule.leaf_labels)) == len(tuple(schedule.nin))
    )
    convention_consistent = (
        str(schedule.coefficient_convention) == "young_yamanouchi_seminormal_plus_ye3t_cg_tree"
        and bool(schedule.uses_young_yamanouchi_carriers)
        and bool(schedule.uses_ye3t_cg_trees)
        and not bool(schedule.uses_symbolic_basis_extraction)
    )
    passed = bool(plan_valid and tree_metadata_present and convention_consistent)
    detail = "ok" if passed else (
        f"plan_valid={plan_valid}, tree_metadata_present={tree_metadata_present}, "
        f"convention_consistent={convention_consistent}"
    )
    return YoungYamanouchiCGTreeScheduleValidation(
        schedule=schedule,
        plan_valid=bool(plan_valid),
        tree_metadata_present=bool(tree_metadata_present),
        convention_consistent=bool(convention_consistent),
        passed=passed,
        detail=detail,
    )


def _submultiset_partitions(rank):
    all_idx = tuple(range(int(rank)))
    seen = set()
    for k in range(1, int(rank)):
        for left in combinations(all_idx, k):
            right = tuple(idx for idx in all_idx if idx not in left)
            key = tuple(sorted((left, right)))
            if key in seen:
                continue
            seen.add(key)
            yield left, right


def _restrict_pattern(nin, lin, indices):
    return tuple(int(nin[idx]) for idx in indices), tuple(int(lin[idx]) for idx in indices)


def _submultiset_pattern_partitions(nin, lin):
    seen = set()
    for left_indices, right_indices in _submultiset_partitions(len(nin)):
        left = _restrict_pattern(nin, lin, left_indices)
        right = _restrict_pattern(nin, lin, right_indices)
        key = tuple(sorted((left, right)))
        if key in seen:
            continue
        seen.add(key)
        yield left, right


def _permutation_irrep_signature(permutation_irrep):
    return (
        tuple(
            (
                factor.channel_label,
                int(factor.l),
                int(factor.multiplicity),
            )
            for factor in permutation_irrep.subgroup.factors
        ),
        tuple(tuple(int(x) for x in partition.parts) for partition in permutation_irrep.partitions),
    )


def _permutation_irrep_signature_unordered(permutation_irrep):
    return tuple(
        sorted(
            (
                str(factor.channel_label),
                int(factor.l),
                int(factor.multiplicity),
                tuple(int(x) for x in partition.parts),
            )
            for factor, partition in zip(permutation_irrep.subgroup.factors, permutation_irrep.partitions, strict=True)
        )
    )


def _permutation_irrep_from_signature(signature):
    factor_payload, partition_payload = signature
    subgroup = PermutationSubgroup(
        tuple(
            PermutationSubgroupFactor(
                channel_label=channel_label,
                l=int(l_value),
                multiplicity=int(multiplicity),
            )
            for channel_label, l_value, multiplicity in factor_payload
        )
    )
    return PermutationIrrep(
        subgroup=subgroup,
        partitions=tuple(Partition(tuple(int(x) for x in parts)) for parts in partition_payload),
    )


def _cg_allowed(left_L, right_L, target_L):
    return abs(int(left_L) - int(right_L)) <= int(target_L) <= int(left_L) + int(right_L)


def _normalize_max_factor_L(max_factor_L):
    if max_factor_L is None:
        return None
    value = int(max_factor_L)
    if value < 0:
        raise ValueError("max_factor_L must be nonnegative or None.")
    return value


def _candidate_product_L_pairs(target_L, left_Ls, right_Ls, mode, max_factor_L=None):
    mode = str(mode)
    target_L = int(target_L)
    max_factor_L = _normalize_max_factor_L(max_factor_L)

    def _within_cap(left_L, right_L):
        if max_factor_L is None:
            return True
        return int(left_L) <= max_factor_L and int(right_L) <= max_factor_L

    if mode == "invariant":
        if target_L != 0:
            return tuple()
        return ((0, 0),) if 0 in left_Ls and 0 in right_Ls and _within_cap(0, 0) else tuple()
    if mode == "module":
        pairs = []
        if 0 in left_Ls and target_L in right_Ls and _within_cap(0, target_L):
            pairs.append((0, target_L))
        if target_L in left_Ls and 0 in right_Ls and _within_cap(target_L, 0):
            pairs.append((target_L, 0))
        return tuple(pairs)
    if mode == "full":
        return tuple(
            (int(left_L), int(right_L))
            for left_L in left_Ls
            for right_L in right_Ls
            if _within_cap(left_L, right_L) and _cg_allowed(left_L, right_L, target_L)
        )
    raise ValueError("mode must be one of {'invariant', 'module', 'full'}")


def _candidate_child_irreps(left_irreps, right_irreps, target_irrep, mode):
    mode = str(mode)
    target_signature = _permutation_irrep_signature_unordered(target_irrep)
    for left_irrep in left_irreps:
        for right_irrep in right_irreps:
            if mode == "invariant" and (not left_irrep.is_totally_symmetric() or not right_irrep.is_totally_symmetric()):
                continue
            if mode == "module" and (not left_irrep.is_totally_symmetric() and not right_irrep.is_totally_symmetric()):
                continue
            if any(
                _permutation_irrep_signature_unordered(term.permutation_irrep) == target_signature
                for term in _merged_permutation_irrep_product_terms(left_irrep, right_irrep)
            ):
                yield left_irrep, right_irrep


def _candidate_child_irrep_terms(left_irreps, right_irreps, target_irrep, mode):
    """Yield child irreps and exact subgroup multiplicities that can reach the target."""

    mode = str(mode)
    target_signature = _permutation_irrep_signature_unordered(target_irrep)
    for left_irrep in left_irreps:
        for right_irrep in right_irreps:
            if mode == "invariant" and (not left_irrep.is_totally_symmetric() or not right_irrep.is_totally_symmetric()):
                continue
            if mode == "module" and (not left_irrep.is_totally_symmetric() and not right_irrep.is_totally_symmetric()):
                continue
            for term in _merged_permutation_irrep_product_terms(left_irrep, right_irrep):
                if _permutation_irrep_signature_unordered(term.permutation_irrep) == target_signature:
                    yield left_irrep, right_irrep, int(term.multiplicity)


def _highest_weight_basis_labels(sector, L_R):
    return tuple(
        (int(copy_index), tuple(int(x) for x in carrier_index))
        for copy_index, carrier_index in sorted(sector.canonical_highest_weight_vectors_by_L.get(int(L_R), {}))
    )


def _identity_column_indices(indices, dim):
    if not indices:
        return _sympy().zeros(int(dim), 0)
    return _sympy().SparseMatrix(
        int(dim),
        len(indices),
        {
            (int(index), int(col)): _sympy().Integer(1)
            for col, index in enumerate(indices)
        },
    )


def _matrix_rows(matrix):
    if hasattr(matrix, "rows") and hasattr(matrix, "cols"):
        return tuple(
            tuple(matrix[int(row), int(col)] for col in range(int(matrix.cols)))
            for row in range(int(matrix.rows))
        )
    return tuple(tuple(row) for row in matrix)


def _native_algebraic_rank_or_none(matrix):
    return exact_algebraic_rank_or_none(_matrix_rows(matrix))


class _ExactColumnAccumulator:
    def __init__(self, rows, *, stop_rank=0):
        self.rows = int(rows)
        self.stop_rank = int(stop_rank)
        self.columns = []
        self.rank = 0

    def try_add_matrix(self, matrix):
        matrix = _sympy().Matrix(matrix)
        added = False
        for col in range(matrix.cols):
            column = _sympy().Matrix(matrix[:, int(col)])
            if column == _sympy().zeros(self.rows, 1):
                continue
            candidate = column if not self.columns else _sympy().Matrix.hstack(*self.columns, column)
            candidate_rank = int(candidate.rank(simplify=False))
            if candidate_rank > self.rank:
                self.columns.append(column)
                self.rank = int(candidate_rank)
                added = True
                if self.reached_target():
                    break
        return added

    def reached_target(self):
        return self.stop_rank > 0 and self.rank >= self.stop_rank

    def matrix(self):
        if not self.columns:
            return _sympy().zeros(self.rows, 0)
        return _sympy().Matrix.hstack(*self.columns)


@lru_cache(maxsize=256)
def _generalized_sector_cached(nin, lin, permutation_irrep_signature):
    return _build_exact_symbolic_projector_sector(
        tuple(int(x) for x in nin),
        tuple(int(x) for x in lin),
        _permutation_irrep_from_signature(permutation_irrep_signature),
    )


def _sector_for_irrep(nin, lin, permutation_irrep):
    key = _sector_cache_key(tuple(nin), tuple(lin), permutation_irrep)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    sector = _generalized_sector_cached(
        tuple(int(x) for x in nin),
        tuple(int(x) for x in lin),
        _permutation_irrep_signature(permutation_irrep),
    )
    _cache_put(key, sector)
    return sector


@lru_cache(maxsize=256)
def _young_generated_upper_bound_cached(nin, lin, permutation_irrep_signature, L_R, mode, target_dim, max_factor_L):
    nin = tuple(int(x) for x in nin)
    lin = tuple(int(x) for x in lin)
    permutation_irrep = _permutation_irrep_from_signature(permutation_irrep_signature)
    if len(nin) <= 1 or target_dim == 0:
        return (0, 0)
    generated_upper_bound = 0
    product_path_count = 0
    for (left_nin, left_lin), (right_nin, right_lin) in _submultiset_pattern_partitions(nin, lin):
        left_irreps = _all_permutation_irreps_for_pattern(left_nin, left_lin)
        right_irreps = _all_permutation_irreps_for_pattern(right_nin, right_lin)
        for left_irrep, right_irrep, subgroup_multiplicity in _candidate_child_irrep_terms(left_irreps, right_irreps, permutation_irrep, mode):
            left_counts = generalized_sector_counts(left_nin, left_lin, left_irrep, count_only=True)
            right_counts = generalized_sector_counts(right_nin, right_lin, right_irrep, count_only=True)
            L_pairs = _candidate_product_L_pairs(
                int(L_R),
                tuple(int(L) for L in left_counts.counts_by_L),
                tuple(int(L) for L in right_counts.counts_by_L),
                str(mode),
                None if int(max_factor_L) < 0 else int(max_factor_L),
            )
            for left_L, right_L in L_pairs:
                left_dim = int(left_counts.counts_by_L.get(int(left_L), 0))
                right_dim = int(right_counts.counts_by_L.get(int(right_L), 0))
                path_dim = left_dim * right_dim * int(subgroup_multiplicity)
                generated_upper_bound += int(path_dim)
                product_path_count += int(path_dim)
    return (int(generated_upper_bound), int(product_path_count))


def _young_generated_upper_bound(nin, lin, permutation_irrep, L_R, mode, target_dim, max_factor_L=None):
    max_factor_L = _normalize_max_factor_L(max_factor_L)
    return _young_generated_upper_bound_cached(
        tuple(int(x) for x in nin),
        tuple(int(x) for x in lin),
        _permutation_irrep_signature(permutation_irrep),
        int(L_R),
        str(mode),
        int(target_dim),
        -1 if max_factor_L is None else int(max_factor_L),
    )


@lru_cache(maxsize=128)
def _young_generated_columns_cached(nin, lin, permutation_irrep_signature, L_R, mode, max_factor_L):
    nin = tuple(int(x) for x in nin)
    lin = tuple(int(x) for x in lin)
    permutation_irrep = _permutation_irrep_from_signature(permutation_irrep_signature)
    target_sector = _sector_for_irrep(nin, lin, permutation_irrep)
    target_dim = len(_highest_weight_basis_labels(target_sector, int(L_R)))
    if len(nin) <= 1 or target_dim == 0:
        return _sympy().zeros(target_dim, 0)
    accumulator = _ExactColumnAccumulator(target_dim, stop_rank=target_dim)
    for (left_nin, left_lin), (right_nin, right_lin) in _submultiset_pattern_partitions(nin, lin):
        left_irreps = _all_permutation_irreps_for_pattern(left_nin, left_lin)
        right_irreps = _all_permutation_irreps_for_pattern(right_nin, right_lin)
        for left_irrep, right_irrep in _candidate_child_irreps(left_irreps, right_irreps, permutation_irrep, mode):
            left_sector = _sector_for_irrep(left_nin, left_lin, left_irrep)
            right_sector = _sector_for_irrep(right_nin, right_lin, right_irrep)
            L_pairs = _candidate_product_L_pairs(
                int(L_R),
                tuple(int(L) for L in left_sector.counts_by_L),
                tuple(int(L) for L in right_sector.counts_by_L),
                str(mode),
                None if int(max_factor_L) < 0 else int(max_factor_L),
            )
            for left_L, right_L in L_pairs:
                mapping = _build_exact_change_of_group_basis_map(
                    left_sector,
                    right_sector,
                    left_L=int(left_L),
                    right_L=int(right_L),
                    output_L=int(L_R),
                    node_span=(0, len(nin)),
                )
                for branch in mapping.branches:
                    if not _target_irrep_matches(branch.permutation_irrep, permutation_irrep):
                        continue
                    accumulator.try_add_matrix(branch.coordinate_matrix)
                    if accumulator.reached_target():
                        return accumulator.matrix()
    return accumulator.matrix()


def _young_generated_columns(nin, lin, permutation_irrep, L_R, mode, max_factor_L=None):
    max_factor_L = _normalize_max_factor_L(max_factor_L)
    return _young_generated_columns_cached(
        tuple(int(x) for x in nin),
        tuple(int(x) for x in lin),
        _permutation_irrep_signature(permutation_irrep),
        int(L_R),
        str(mode),
        -1 if max_factor_L is None else int(max_factor_L),
    )


def young_resolved_primitive_quotient(nin, lin, permutation_irrep, L_R, mode="full", count_only=False, max_factor_L=None):
    """Return the exact Young/SO(3)-resolved primitive quotient for one sector.

    The quotient is computed in the canonical highest-weight basis of the joint
    ``SO(3) x G_nu`` sector.  For the totally symmetric Young sector this
    reduces to the existing ACE primitive quotient.  For sign or mixed Specht
    sectors, lower-generated products are assembled with exact generalized
    Young/SO(3) tensor-product intertwiners.
    """

    nin, lin = _normalize_nl(nin, lin)
    max_factor_L = _normalize_max_factor_L(max_factor_L)
    if not isinstance(permutation_irrep, PermutationIrrep):
        permutation_irrep = permutation_irrep_for_character(nin, lin, permutation_irrep)
    counts = generalized_sector_counts(nin, lin, permutation_irrep, count_only=bool(count_only))
    sector_count = int(counts.counts_by_L.get(int(L_R), 0))
    target_sector = counts.sector
    target_basis_labels = tuple() if target_sector is None else _highest_weight_basis_labels(target_sector, int(L_R))
    sector_basis_rank = int(sector_count) if target_sector is None else len(target_basis_labels)
    if sector_basis_rank == 0:
        return YoungResolvedPrimitiveQuotient(
            nin=nin,
            lin=lin,
            channel_multiset=_channel_multiset(nin, lin),
            permutation_irrep=permutation_irrep,
            L_R=int(L_R),
            mode=str(mode),
            max_factor_L=max_factor_L,
            sector_count=int(sector_count),
            sector_basis_rank=int(sector_basis_rank),
            primitive_rank=0,
            generated_rank=0,
            generated_basis_indices=tuple(),
            primitive_basis_indices=tuple(),
            target_basis_labels=target_basis_labels,
            generated_upper_bound=0,
            primitive_lower_bound=0,
            product_path_count=0,
            rank_status="exact",
            provenance=counts.provenance,
            codepath=counts.codepath,
            detail="Target Young/SO(3) sector is empty.",
            quotient=None,
        )
    generated_upper_bound, product_path_count = _young_generated_upper_bound(
        nin,
        lin,
        permutation_irrep,
        int(L_R),
        str(mode),
        int(sector_basis_rank),
        max_factor_L=max_factor_L,
    )
    primitive_lower_bound = max(0, int(sector_basis_rank) - int(generated_upper_bound))
    if bool(count_only):
        exact_no_generators = int(generated_upper_bound) == 0
        return YoungResolvedPrimitiveQuotient(
            nin=nin,
            lin=lin,
            channel_multiset=_channel_multiset(nin, lin),
            permutation_irrep=permutation_irrep,
            L_R=int(L_R),
            mode=str(mode),
            max_factor_L=max_factor_L,
            sector_count=int(sector_count),
            sector_basis_rank=int(sector_basis_rank),
            primitive_rank=int(sector_basis_rank) if exact_no_generators else None,
            generated_rank=0 if exact_no_generators else None,
            generated_basis_indices=tuple(),
            primitive_basis_indices=tuple(range(int(sector_basis_rank))) if exact_no_generators else tuple(),
            target_basis_labels=target_basis_labels,
            generated_upper_bound=int(generated_upper_bound),
            primitive_lower_bound=int(primitive_lower_bound),
            product_path_count=int(product_path_count),
            rank_status="exact_no_generators" if exact_no_generators else "bounded",
            provenance="character_count_bound" if counts.provenance != "cached" else "cached",
            codepath="count_only_young_so3_primitive_bound",
            detail="Count-only generated-rank upper bound; exact product coordinate maps were not built.",
            quotient=None,
        )
    if int(generated_upper_bound) == 0:
        return YoungResolvedPrimitiveQuotient(
            nin=nin,
            lin=lin,
            channel_multiset=_channel_multiset(nin, lin),
            permutation_irrep=permutation_irrep,
            L_R=int(L_R),
            mode=str(mode),
            max_factor_L=max_factor_L,
            sector_count=int(sector_count),
            sector_basis_rank=int(sector_basis_rank),
            primitive_rank=int(sector_basis_rank),
            generated_rank=0,
            generated_basis_indices=tuple(),
            primitive_basis_indices=tuple(range(int(sector_basis_rank))),
            target_basis_labels=target_basis_labels,
            generated_upper_bound=0,
            primitive_lower_bound=int(sector_basis_rank),
            product_path_count=0,
            rank_status="exact_no_generators",
            provenance="character_count_bound" if counts.provenance != "cached" else "cached",
            codepath="count_only_young_so3_primitive_bound",
            detail="No compatible lower-generated product paths exist, so the whole sector is primitive.",
            quotient=None,
        )
    if max_factor_L is not None or not permutation_irrep.is_totally_symmetric():
        generated = _young_generated_columns(nin, lin, permutation_irrep, int(L_R), str(mode), max_factor_L=max_factor_L)
        if generated.cols == 0:
            generated_rank = 0
            generated_basis_indices = tuple()
        else:
            _, row_pivots = generated.T.rref(simplify=False)
            generated_rank = len(row_pivots)
            generated_basis_indices = tuple(int(index) for index in row_pivots)
        primitive_basis_indices = tuple(index for index in range(sector_basis_rank) if index not in generated_basis_indices)
        return YoungResolvedPrimitiveQuotient(
            nin=nin,
            lin=lin,
            channel_multiset=_channel_multiset(nin, lin),
            permutation_irrep=permutation_irrep,
            L_R=int(L_R),
            mode=str(mode),
            max_factor_L=max_factor_L,
            sector_count=int(sector_count),
            sector_basis_rank=int(sector_basis_rank),
            primitive_rank=int(sector_basis_rank - generated_rank),
            generated_rank=int(generated_rank),
            generated_basis_indices=generated_basis_indices,
            primitive_basis_indices=primitive_basis_indices,
            target_basis_labels=target_basis_labels,
            generated_upper_bound=int(generated_upper_bound),
            primitive_lower_bound=int(primitive_lower_bound),
            product_path_count=int(product_path_count),
            rank_status="exact",
            provenance="exact_projector" if counts.provenance != "cached" else "cached",
            codepath="symbolic_young_so3_primitive_quotient",
            detail="Exact quotient in the resolved Young/SO(3) highest-weight basis.",
            quotient=generated,
        )
    quotient = _exact_product_expansion_engine().primitive_quotient(nin, lin, int(L_R), mode=str(mode))
    return YoungResolvedPrimitiveQuotient(
        nin=nin,
        lin=lin,
        channel_multiset=_channel_multiset(nin, lin),
        permutation_irrep=permutation_irrep,
        L_R=int(L_R),
        mode=str(mode),
        max_factor_L=max_factor_L,
        sector_count=int(sector_count),
        sector_basis_rank=int(sector_basis_rank),
        primitive_rank=int(quotient.primitive_rank),
        generated_rank=int(quotient.generated_rank),
        generated_basis_indices=tuple(int(x) for x in quotient.generated_basis_indices),
        primitive_basis_indices=tuple(int(x) for x in quotient.primitive_basis_indices),
        target_basis_labels=target_basis_labels,
        generated_upper_bound=int(generated_upper_bound),
        primitive_lower_bound=int(primitive_lower_bound),
        product_path_count=int(product_path_count),
        rank_status="exact",
        provenance="fallback" if counts.provenance == "cached" else "exact_projector",
        codepath="ace_trivial_primitive_quotient",
        detail="Trivial Young sector delegates to the existing ACE/ye3t primitive quotient.",
        quotient=quotient,
    )


def validate_young_resolved_primitive_quotient(quotient):
    """Validate rank/index consistency for a Young-resolved primitive quotient."""

    if quotient.generated_rank is None or quotient.primitive_rank is None:
        return YoungResolvedPrimitiveQuotientValidation(
            quotient=quotient,
            rank_accounting=False,
            index_partition=False,
            generated_rank_matches_matrix=True,
            trivial_matches_ace=None,
            passed=False,
            detail="count-only quotient reports bounds rather than exact ranks.",
        )
    generated_indices = tuple(int(x) for x in quotient.generated_basis_indices)
    primitive_indices = tuple(int(x) for x in quotient.primitive_basis_indices)
    target_indices = set(range(int(quotient.sector_basis_rank)))
    rank_accounting = int(quotient.generated_rank) + int(quotient.primitive_rank) == int(quotient.sector_basis_rank)
    index_partition = (
        set(generated_indices).isdisjoint(primitive_indices)
        and set(generated_indices) | set(primitive_indices) == target_indices
    )
    generated_rank_matches_matrix = True
    if quotient.codepath == "symbolic_young_so3_primitive_quotient" and quotient.quotient is not None:
        native_rank = _native_algebraic_rank_or_none(quotient.quotient)
        if native_rank is None:
            native_rank = int(_sympy().Matrix(quotient.quotient).rank(simplify=False))
        generated_rank_matches_matrix = int(native_rank) == int(quotient.generated_rank)
    trivial_matches_ace = None
    if quotient.permutation_irrep.is_totally_symmetric():
        if quotient.max_factor_L is None:
            ace = _exact_product_expansion_engine().primitive_quotient(
                tuple(quotient.nin),
                tuple(quotient.lin),
                int(quotient.L_R),
                mode=str(quotient.mode),
            )
            trivial_matches_ace = (
                int(ace.generated_rank) == int(quotient.generated_rank)
                and int(ace.primitive_rank) == int(quotient.primitive_rank)
                and tuple(int(x) for x in ace.primitive_basis_indices) == primitive_indices
            )
    passed = bool(rank_accounting and index_partition and generated_rank_matches_matrix and (trivial_matches_ace is not False))
    detail = "ok" if passed else (
        f"rank_accounting={rank_accounting}, index_partition={index_partition}, "
        f"generated_rank_matches_matrix={generated_rank_matches_matrix}, trivial_matches_ace={trivial_matches_ace}"
    )
    return YoungResolvedPrimitiveQuotientValidation(
        quotient=quotient,
        rank_accounting=bool(rank_accounting),
        index_partition=bool(index_partition),
        generated_rank_matches_matrix=bool(generated_rank_matches_matrix),
        trivial_matches_ace=trivial_matches_ace,
        passed=passed,
        detail=detail,
    )


__all__ = [
    "YoungAnalyticalSectorCatalog",
    "YoungAnalyticalSectorCatalogValidation",
    "YoungAnalyticalSectorRecord",
    "YoungProductPath",
    "YoungResolvedPrimitiveQuotient",
    "YoungResolvedPrimitiveQuotientValidation",
    "YoungSectorCounts",
    "YoungSpechtBasisConstructionPlan",
    "YoungSpechtBasisConstructionPlanValidation",
    "YoungSpechtConstructionRecord",
    "YoungYamanouchiCGTreeSchedule",
    "YoungYamanouchiCGTreeScheduleValidation",
    "all_young_character_irreps_for_pattern",
    "analytical_young_e3_sector_catalog",
    "compile_young_specht_basis_schedule",
    "dusson_barthelemy_trivial_sector_counts",
    "generalized_sector_counts",
    "permutation_irrep_for_character",
    "selected_young_character_irreps_for_pattern",
    "specht_partition_family",
    "validate_analytical_young_e3_sector_catalog",
    "validate_young_specht_basis_construction_plan",
    "validate_young_yamanouchi_cg_tree_schedule",
    "validate_young_resolved_primitive_quotient",
    "young_yamanouchi_cg_tree_schedule",
    "young_product_paths",
    "young_resolved_primitive_quotient",
    "young_specht_basis_construction_plan",
]
