"""Public Young-sector helpers for the generalized exact path.

These functions provide a stable, provenance-carrying surface over exact
character counts, factorized Young/CG product images, and a retained small
ambient-projector reference path. The default ACE/ye3t invariant path remains
unchanged.

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
from itertools import combinations, permutations, product
from math import comb, factorial

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
        'factor_scope',
        'allowed_factor_partitions_by_rank',
        'factor_route_name',
        'factor_route_policy',
        'basis_convention',
        'global_partition',
        'factorized_blocks',
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
    factor_scope = "immediate"
    allowed_factor_partitions_by_rank = tuple()
    factor_route_name = None
    factor_route_policy = "all"
    basis_convention = "canonical_projector_highest_weight"
    global_partition = None
    factorized_blocks = tuple()

    def induction_map(self, subgroup_partitions):
        """Compile the exact Young lift for one recorded global block on demand."""

        if self.global_partition is None:
            raise ValueError("An induction map requires a global S_N quotient.")
        signature = tuple(tuple(int(part) for part in parts) for parts in subgroup_partitions)
        block = next(
            (item for item in self.factorized_blocks if item["subgroup_partitions"] == signature),
            None,
        )
        if block is None:
            raise ValueError(f"No global quotient block has subgroup partitions {signature!r}.")
        from .young_orthogonal import (
            validate_young_orthogonal_nary_subduction,
            young_orthogonal_nary_subduction,
        )

        tensor = young_orthogonal_nary_subduction(signature, self.global_partition)
        if int(tensor.multiplicity) != int(block["induction_multiplicity"]):
            raise ArithmeticError("The compiled Young map disagrees with the quotient's exact LR multiplicity.")
        if not validate_young_orthogonal_nary_subduction(tensor).passed:
            raise ArithmeticError("The compiled Young map failed exact isometry or generator equivariance validation.")
        return tensor

    def orthonormal_primitive_multiplicity_blocks(self, subgroup_partitions=None):
        """Return exact orthonormal primitive vectors in multiplicity coordinates.

        Purpose:
            Materialize the orthogonal complement of retained product images.
        Mathematical contract:
            Each coefficient matrix B obeys B.H * G * B = I and D.H * G * B = 0
            in its reported multiplicity basis. Its columns are tensored with
            the identity on the Young and magnetic carrier coordinates.
        Inputs:
            An optional repeated-content subgroup partition signature limits
            a global S_N quotient to one stored induction block.
        Outputs:
            A tuple of compact dictionaries containing exact coefficient and
            Gram matrices, row labels, induction multiplicity, and carrier size.
        Does not:
            Materialize every content-orbit placement or choose an intrinsic
            basis among equivalent primitive copies.
        """

        if self.rank_status not in {"exact", "exact_no_generators"}:
            raise ValueError("Orthonormal primitive coefficients require an exact quotient.")
        if self.basis_convention == "character_count_no_basis":
            raise ValueError("Count-only quotients have no basis coefficients to orthonormalize.")
        if subgroup_partitions is not None and self.global_partition is None:
            raise ValueError("subgroup_partitions selects a block of a global S_N quotient.")
        if self.global_partition is not None:
            selected = None if subgroup_partitions is None else tuple(
                tuple(int(part) for part in parts) for parts in subgroup_partitions
            )
            blocks = tuple(
                block for block in self.factorized_blocks
                if selected is None or block["subgroup_partitions"] == selected
            )
            if selected is not None and not blocks:
                raise ValueError(f"No global quotient block has subgroup partitions {selected!r}.")
            result = []
            specht_dim = int(Partition(self.global_partition).dimension)
            for block in blocks:
                angular_labels = tuple(block["angular_basis_labels"])
                generated = _sympy().Matrix(block["angular_generated_matrix"])
                if generated.cols == len(angular_labels):
                    continue
                irrep = permutation_irrep_for_character(
                    self.nin, self.lin, block["subgroup_partitions"],
                )
                gram = _primitive_multiplicity_gram(
                    self.nin, self.lin, irrep, self.L_R,
                    block["local_basis_convention"], angular_labels,
                )
                coefficients, representatives = _orthonormal_primitive_copy_coefficients(
                    generated, gram,
                )
                if coefficients.cols != int(block["angular_target_rank"]) - int(block["angular_generated_rank"]):
                    raise ArithmeticError("Global primitive multiplicity rank disagrees with its exact local image.")
                result.append({
                    "subgroup_partitions": block["subgroup_partitions"],
                    "angular_basis_labels": angular_labels,
                    "coordinate_representative_labels": tuple(angular_labels[i] for i in representatives),
                    "coefficients": coefficients,
                    "gram": gram,
                    "generated_coefficients": generated,
                    "induction_multiplicity": int(block["induction_multiplicity"]),
                    "lr_chain_labels": block["lr_chain_labels"],
                    "young_carrier_dimension": specht_dim,
                    "magnetic_dimension": 2 * int(self.L_R) + 1,
                    "basis_convention": block["local_basis_convention"],
                })
            if selected is None and sum(
                item["coefficients"].cols * item["induction_multiplicity"] * specht_dim
                for item in result
            ) != int(self.primitive_rank):
                raise ArithmeticError("Orthonormal global blocks do not account for the primitive rank.")
            return tuple(result)

        if int(self.primitive_rank) == 0:
            return tuple()
        irrep = self.permutation_irrep
        carrier_dim = int(irrep.dim)
        if len(self.target_basis_labels) != int(self.sector_basis_rank):
            raise ValueError("The exact quotient has no materialized target basis labels.")
        if int(self.generated_rank) == 0:
            generated = _sympy().zeros(int(self.sector_basis_rank), 0)
        elif hasattr(self.quotient, "rows") and hasattr(self.quotient, "cols"):
            generated = _sympy().Matrix(self.quotient)
        else:
            engine = _exact_product_expansion_engine()
            target = engine.feature_space(self.nin, self.lin, int(self.L_R))
            generated = engine._generated_columns_for_target(
                target, mode=self.mode, stop_at_rank=target.dim,
            )
        if generated.rows != int(self.sector_basis_rank):
            raise ArithmeticError("The local product matrix has the wrong target dimension.")
        if carrier_dim == 1:
            angular_labels = tuple(self.target_basis_labels)
            angular_generated = generated
        else:
            reference_rows = tuple(range(0, int(self.sector_basis_rank), carrier_dim))
            reference = generated.extract(reference_rows, tuple(range(generated.cols)))
            independent_columns = tuple(int(index) for index in reference.rref(simplify=False)[1])
            angular_generated = reference[:, independent_columns]
            angular_labels = tuple(self.target_basis_labels[index] for index in reference_rows)
            expanded = (
                _sympy().kronecker_product(angular_generated, _sympy().eye(carrier_dim))
                if angular_generated.cols else _sympy().zeros(int(self.sector_basis_rank), 0)
            )
            if (int(self.generated_rank) != angular_generated.cols * carrier_dim
                    or int(_sympy().Matrix.hstack(generated, expanded).rank(simplify=False)) != int(self.generated_rank)):
                raise ArithmeticError("The local product image is not a complete Young submodule.")
        gram = _primitive_multiplicity_gram(
            self.nin, self.lin, irrep, self.L_R, self.basis_convention, angular_labels,
        )
        coefficients, representatives = _orthonormal_primitive_copy_coefficients(
            angular_generated, gram,
        )
        if coefficients.cols * carrier_dim != int(self.primitive_rank):
            raise ArithmeticError("Orthonormal local vectors do not account for the primitive rank.")
        return ({
            "subgroup_partitions": tuple(tuple(partition.parts) for partition in irrep.partitions),
            "angular_basis_labels": angular_labels,
            "coordinate_representative_labels": tuple(angular_labels[i] for i in representatives),
            "coefficients": coefficients,
            "gram": gram,
            "generated_coefficients": angular_generated,
            "induction_multiplicity": 1,
            "lr_chain_labels": tuple(),
            "young_carrier_dimension": carrier_dim,
            "magnetic_dimension": 2 * int(self.L_R) + 1,
            "basis_convention": self.basis_convention,
        },)

    def orthonormal_primitive_vector(
        self, primitive_copy_index, subgroup_partitions=None,
        induction_index=0, young_index=0, magnetic_M=None,
    ):
        """Return one normalized primitive vector in target-sector coordinates.

        Purpose:
            Select one column of a compact primitive block and one complete
            Young/rotation carrier component without a dense tensor product.
        Mathematical contract:
            The returned sparse target coordinates have unit invariant norm
            and are orthogonal to the retained product image.
        Inputs:
            A primitive copy index within the selected subgroup block, an LR
            induction index, a Young component index, and a magnetic number.
        Outputs:
            Tuples of ``(target_basis_label, magnetic_M, exact_coefficient)``.
        Does not:
            Materialize uncoupled magnetic-slot or full content-orbit vectors.
        """

        if self.global_partition is not None and subgroup_partitions is None:
            contributing = tuple(
                block["subgroup_partitions"] for block in self.factorized_blocks
                if block["angular_target_rank"] > block["angular_generated_rank"]
            )
            if len(contributing) != 1:
                raise ValueError("Select exactly one primitive subgroup block to materialize a vector.")
            subgroup_partitions = contributing[0]
        blocks = self.orthonormal_primitive_multiplicity_blocks(subgroup_partitions)
        if len(blocks) != 1:
            raise ValueError("Select exactly one primitive subgroup block to materialize a vector.")
        block = blocks[0]
        copy_index = int(primitive_copy_index)
        lr_index = int(induction_index)
        tableau_index = int(young_index)
        M = int(self.L_R) if magnetic_M is None else int(magnetic_M)
        if copy_index < 0 or copy_index >= block["coefficients"].cols:
            raise ValueError("primitive_copy_index is outside the selected block.")
        if lr_index < 0 or lr_index >= block["induction_multiplicity"]:
            raise ValueError("induction_index is outside the selected block.")
        if tableau_index < 0 or tableau_index >= block["young_carrier_dimension"]:
            raise ValueError("young_index is outside the selected carrier.")
        if abs(M) > int(self.L_R):
            raise ValueError("magnetic_M is outside the target angular carrier.")
        if self.global_partition is None:
            carriers = tuple(product(*(
                range(int(partition.dimension)) for partition in self.permutation_irrep.partitions
            )))
            carrier = carriers[tableau_index]
        result = []
        for row, angular_label in enumerate(block["angular_basis_labels"]):
            coefficient = block["coefficients"][row, copy_index]
            if coefficient == 0:
                continue
            if self.global_partition is not None:
                label = (
                    block["subgroup_partitions"], angular_label, lr_index, tableau_index,
                )
            elif block["young_carrier_dimension"] == 1:
                label = angular_label
            else:
                label = (angular_label[0], carrier)
            if label not in self.target_basis_labels:
                raise ArithmeticError("Normalized primitive coordinate is absent from the exact target basis.")
            result.append((label, M, coefficient))
        return tuple(result)


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


@lru_cache(maxsize=512)
def _active_child_irreps_for_pattern(nin, lin, max_factor_L):
    """Cache only child Young sectors with allowed nonzero angular content."""

    active = []
    for irrep in _all_permutation_irreps_for_pattern(nin, lin):
        counts = generalized_sector_counts(nin, lin, irrep, count_only=True).counts_by_L
        allowed = tuple(
            (int(L), int(count)) for L, count in counts.items()
            if int(count) > 0 and (int(max_factor_L) < 0 or int(L) <= int(max_factor_L))
        )
        if allowed:
            active.append((irrep, allowed))
    return tuple(active)


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


def _normalize_factor_partition_policy(allowed_factor_partitions_by_rank):
    """Normalize child Young allowlists to hashable full subgroup signatures."""

    if allowed_factor_partitions_by_rank is None:
        return tuple()
    if not hasattr(allowed_factor_partitions_by_rank, "items"):
        raise ValueError("allowed_factor_partitions_by_rank must be a rank-keyed mapping.")
    rules = []
    for rank, entries in allowed_factor_partitions_by_rank.items():
        rank = int(rank)
        if rank < 1:
            raise ValueError("Child partition policy ranks must be positive.")
        signatures = set()
        for entry in entries:
            entry = tuple(entry)
            if not entry:
                raise ValueError("A child partition signature cannot be empty.")
            if all(isinstance(part, int) for part in entry):
                signature = (tuple(Partition(entry).parts),)
            else:
                signature = tuple(tuple(Partition(tuple(parts)).parts) for parts in entry)
            if sum(sum(parts) for parts in signature) != rank:
                raise ValueError("Child partition signature sizes must sum to the declared rank.")
            signatures.add(signature)
        rules.append((rank, tuple(sorted(signatures))))
    return tuple(sorted(rules))


def _factor_route_allowed(
    left_nin, left_lin, left_irrep, left_L,
    right_nin, right_lin, right_irrep, right_L,
    target_irrep, target_L, mode, partition_policy, factor_route_filter,
    factor_route_policy="all",
):
    """Apply the same joint child policy to count bounds and exact maps."""

    if factor_route_policy == "matched_pairs":
        if len(left_nin) + len(right_nin) == 2:
            if (len(left_nin) != 1 or len(right_nin) != 1
                    or tuple(zip(left_nin, left_lin)) != tuple(zip(right_nin, right_lin))):
                return False
        elif len(left_nin) % 2 or len(right_nin) % 2:
            return False
    if mode == "module" and not (
        (int(left_L) == 0 and left_irrep.is_totally_symmetric())
        or (int(right_L) == 0 and right_irrep.is_totally_symmetric())
    ):
        return False
    for rank, irrep in ((len(left_nin), left_irrep), (len(right_nin), right_irrep)):
        allowed = next((signatures for rule_rank, signatures in partition_policy if rule_rank == rank), None)
        signature = tuple(tuple(int(part) for part in item.parts) for item in irrep.partitions)
        if allowed is not None and signature not in allowed:
            return False
    if factor_route_filter is None:
        return True
    return bool(factor_route_filter(
        left_nin=tuple(left_nin), left_lin=tuple(left_lin),
        left_permutation=left_irrep, left_L=int(left_L),
        right_nin=tuple(right_nin), right_lin=tuple(right_lin),
        right_permutation=right_irrep, right_L=int(right_L),
        target_permutation=target_irrep, target_L=int(target_L),
    ))


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


def _primitive_multiplicity_gram(nin, lin, permutation_irrep, L_R, basis_convention, angular_labels):
    """Return the exact physical Gram matrix on one Young multiplicity block."""

    sp = _sympy()
    angular_labels = tuple(angular_labels)
    if basis_convention == "factorized_three_pair_recoupled_cg":
        # The pair-character and canonical pair-CG vectors are normalized.
        return sp.eye(len(angular_labels))
    if basis_convention == "factorized_young_cg_gram":
        from .factorized_product_images import factorized_requested_sector

        sector = factorized_requested_sector(
            nin, lin, _permutation_irrep_signature(permutation_irrep), int(L_R),
        )
        carrier_dim = int(permutation_irrep.dim)
        reference_rows = tuple(index * carrier_dim for index in range(len(angular_labels)))
        if tuple(sector.basis_labels[index] for index in reference_rows) != angular_labels:
            raise ArithmeticError("Factorized primitive labels disagree with the exact Young basis.")
        return sector.gram_inverse.extract(reference_rows, reference_rows).inv()
    if basis_convention == "ace_compact_tree_highest_weight":
        engine = _exact_product_expansion_engine()
        target = engine.feature_space(nin, lin, int(L_R))
        if tuple(target.labels) != angular_labels:
            raise ArithmeticError("Primitive labels disagree with the exact ACE target basis.")
        target_vectors = tuple(engine._m_vectors(label) for label in target.labels)
        return engine._target_gram_inverse(target, target_vectors, int(L_R)).inv()
    raise ValueError(f"No exact primitive Gram matrix is defined for {basis_convention!r}.")


def _orthonormal_primitive_copy_coefficients(generated, gram):
    """Project coordinate quotient seeds and normalize them in the exact Gram.

    Algorithmic reference: weighted orthogonal projection and Cholesky
    orthonormalization (Strang, MIT OCW 18.06, Lecture 16). Independent
    implementation; no external source code is used.
    """

    from sympy.matrices.exceptions import NonPositiveDefiniteMatrixError

    sp = _sympy()
    gram = sp.Matrix(gram)
    generated = sp.Matrix(generated)
    dimension = int(gram.rows)
    if gram.cols != dimension or generated.rows != dimension:
        raise ValueError("Primitive Gram and product-image dimensions disagree.")
    if any(sp.simplify(entry) != 0 for entry in gram - gram.H):
        raise ArithmeticError("Primitive target Gram matrix is not Hermitian.")
    independent = tuple(int(index) for index in generated.rref(simplify=False)[1])
    generated = generated[:, independent]
    if generated.cols:
        _, row_pivots = generated.H.rref(simplify=False)
        row_pivots = tuple(int(index) for index in row_pivots)
    else:
        row_pivots = tuple()
    representatives = tuple(index for index in range(dimension) if index not in row_pivots)
    if not representatives:
        return sp.zeros(dimension, 0), representatives
    seeds = sp.eye(dimension)[:, representatives]
    if generated.cols:
        normal = generated.H * gram * generated
        projected = seeds - generated * normal.LUsolve(generated.H * gram * seeds)
    else:
        projected = seeds
    residual_gram = (projected.H * gram * projected).applyfunc(sp.simplify)
    try:
        cholesky = residual_gram.cholesky()
    except (ValueError, NonPositiveDefiniteMatrixError) as exc:
        raise ArithmeticError("Projected primitive representatives have a singular or nonpositive Gram matrix.") from exc
    coefficients = projected * cholesky.H.inv()
    if any(sp.simplify(entry) != 0 for entry in generated.H * gram * coefficients):
        raise ArithmeticError("Primitive coefficients are not orthogonal to generated products.")
    if any(sp.simplify(entry) != 0 for entry in coefficients.H * gram * coefficients - sp.eye(len(representatives))):
        raise ArithmeticError("Primitive coefficients are not orthonormal.")
    return coefficients, representatives


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


def _young_generated_upper_bound_impl(
    nin, lin, permutation_irrep_signature, L_R, mode, target_dim,
    max_factor_L, partition_policy, factor_route_policy, factor_route_filter,
):
    nin = tuple(int(x) for x in nin)
    lin = tuple(int(x) for x in lin)
    permutation_irrep = _permutation_irrep_from_signature(permutation_irrep_signature)
    if len(nin) <= 1 or target_dim == 0:
        return (0, 0)
    generated_upper_bound = 0
    product_path_count = 0
    for (left_nin, left_lin), (right_nin, right_lin) in _submultiset_pattern_partitions(nin, lin):
        left_content = _channel_multiset(left_nin, left_lin)
        parent_content = _channel_multiset(nin, lin)
        left_counts = {(int(n), int(l)): int(count) for n, l, count in left_content}
        shuffle_count = 1
        for n, l, multiplicity in parent_content:
            left_multiplicity = left_counts.get((int(n), int(l)), 0)
            shuffle_count *= comb(int(multiplicity), int(left_multiplicity))
        left_active = _active_child_irreps_for_pattern(left_nin, left_lin, max_factor_L)
        right_active = _active_child_irreps_for_pattern(right_nin, right_lin, max_factor_L)
        left_irreps = tuple(item[0] for item in left_active)
        right_irreps = tuple(item[0] for item in right_active)
        left_angular_counts = {irrep: dict(counts) for irrep, counts in left_active}
        right_angular_counts = {irrep: dict(counts) for irrep, counts in right_active}
        for left_irrep, right_irrep, _subgroup_multiplicity in _candidate_child_irrep_terms(left_irreps, right_irreps, permutation_irrep, mode):
            left_counts = left_angular_counts[left_irrep]
            right_counts = right_angular_counts[right_irrep]
            L_pairs = _candidate_product_L_pairs(
                int(L_R),
                tuple(left_counts),
                tuple(right_counts),
                str(mode),
                None if int(max_factor_L) < 0 else int(max_factor_L),
            )
            for left_L, right_L in L_pairs:
                if not _factor_route_allowed(
                    left_nin, left_lin, left_irrep, left_L,
                    right_nin, right_lin, right_irrep, right_L,
                    permutation_irrep, L_R, mode, partition_policy, factor_route_filter,
                    factor_route_policy,
                ):
                    continue
                left_dim = int(left_counts.get(int(left_L), 0)) * int(left_irrep.dim)
                right_dim = int(right_counts.get(int(right_L), 0)) * int(right_irrep.dim)
                path_dim = left_dim * right_dim * int(shuffle_count)
                generated_upper_bound += int(path_dim)
                product_path_count += int(path_dim)
    return (int(generated_upper_bound), int(product_path_count))


@lru_cache(maxsize=256)
def _young_generated_upper_bound_cached(nin, lin, permutation_irrep_signature, L_R, mode, target_dim, max_factor_L, partition_policy, factor_route_policy):
    return _young_generated_upper_bound_impl(
        nin, lin, permutation_irrep_signature, L_R, mode, target_dim,
        max_factor_L, partition_policy, factor_route_policy, None,
    )


def _young_generated_upper_bound(
    nin, lin, permutation_irrep, L_R, mode, target_dim, max_factor_L=None,
    partition_policy=(), factor_route_policy="all", factor_route_filter=None,
):
    max_factor_L = _normalize_max_factor_L(max_factor_L)
    args = (
        tuple(int(x) for x in nin),
        tuple(int(x) for x in lin),
        _permutation_irrep_signature(permutation_irrep),
        int(L_R),
        str(mode),
        int(target_dim),
        -1 if max_factor_L is None else int(max_factor_L),
        tuple(partition_policy),
        str(factor_route_policy),
    )
    if factor_route_filter is not None:
        return _young_generated_upper_bound_impl(*args, factor_route_filter)
    return _young_generated_upper_bound_cached(*args)


def _young_generated_columns_impl(
    nin, lin, permutation_irrep_signature, L_R, mode, max_factor_L,
    partition_policy, factor_scope, factor_route_policy, factor_route_filter, memo,
):
    from .factorized_product_images import factorized_product_image, factorized_requested_sector

    nin = tuple(int(x) for x in nin)
    lin = tuple(int(x) for x in lin)
    permutation_irrep = _permutation_irrep_from_signature(permutation_irrep_signature)
    memo_key = (
        nin, lin, permutation_irrep_signature, int(L_R), str(mode),
        int(max_factor_L), tuple(partition_policy), str(factor_scope), str(factor_route_policy),
    )
    if memo_key in memo:
        return memo[memo_key]
    target_sector = factorized_requested_sector(nin, lin, permutation_irrep_signature, int(L_R))
    target_dim = len(target_sector.basis_labels)
    if target_dim == 0:
        return _sympy().zeros(0, 0)
    if len(nin) <= 1:
        return _sympy().eye(target_dim) if factor_scope == "recursive" else _sympy().zeros(target_dim, 0)
    accumulator = _ExactColumnAccumulator(target_dim, stop_rank=target_dim)
    for (left_nin, left_lin), (right_nin, right_lin) in _submultiset_pattern_partitions(nin, lin):
        left_active = _active_child_irreps_for_pattern(left_nin, left_lin, max_factor_L)
        right_active = _active_child_irreps_for_pattern(right_nin, right_lin, max_factor_L)
        left_irreps = tuple(item[0] for item in left_active)
        right_irreps = tuple(item[0] for item in right_active)
        left_angular_counts = {irrep: dict(counts) for irrep, counts in left_active}
        right_angular_counts = {irrep: dict(counts) for irrep, counts in right_active}
        for left_irrep, right_irrep in _candidate_child_irreps(left_irreps, right_irreps, permutation_irrep, mode):
            left_counts = left_angular_counts[left_irrep]
            right_counts = right_angular_counts[right_irrep]
            L_pairs = _candidate_product_L_pairs(
                int(L_R),
                tuple(left_counts),
                tuple(right_counts),
                str(mode),
                None if int(max_factor_L) < 0 else int(max_factor_L),
            )
            for left_L, right_L in L_pairs:
                if not _factor_route_allowed(
                    left_nin, left_lin, left_irrep, left_L,
                    right_nin, right_lin, right_irrep, right_L,
                    permutation_irrep, L_R, mode, partition_policy, factor_route_filter,
                    factor_route_policy,
                ):
                    continue
                left_sector = factorized_requested_sector(
                    left_nin, left_lin, _permutation_irrep_signature(left_irrep), int(left_L),
                )
                right_sector = factorized_requested_sector(
                    right_nin, right_lin, _permutation_irrep_signature(right_irrep), int(right_L),
                )
                if factor_scope == "recursive":
                    left_retained = _young_generated_columns_impl(
                        left_nin, left_lin, _permutation_irrep_signature(left_irrep),
                        int(left_L), mode, max_factor_L, partition_policy,
                        factor_scope, factor_route_policy, factor_route_filter, memo,
                    )
                    right_retained = _young_generated_columns_impl(
                        right_nin, right_lin, _permutation_irrep_signature(right_irrep),
                        int(right_L), mode, max_factor_L, partition_policy,
                        factor_scope, factor_route_policy, factor_route_filter, memo,
                    )
                    if left_retained.cols == 0 or right_retained.cols == 0:
                        continue
                columns, source_labels = factorized_product_image(left_sector, right_sector, target_sector)
                if factor_scope == "recursive":
                    # Restrict both child images before the exact merge map
                    # (supplemental Eqs. suppdecomposablesubspace and suppprimitivequotient).
                    child_labels = tuple(
                        (left, right)
                        for left in left_sector.basis_labels
                        for right in right_sector.basis_labels
                    )
                    if not child_labels or len(source_labels) % len(child_labels):
                        raise ArithmeticError("Induced source labels have an invalid subgroup-coset size.")
                    coset_count = len(source_labels) // len(child_labels)
                    expected = tuple(
                        (coset_index, left, right)
                        for coset_index in range(coset_count)
                        for left, right in child_labels
                    )
                    if source_labels != expected:
                        raise ArithmeticError("Recursive child coordinates do not match the exact source basis.")
                    retained = _sympy().kronecker_product(left_retained, right_retained)
                    columns = columns * _sympy().kronecker_product(
                        _sympy().eye(coset_count), retained,
                    )
                accumulator.try_add_matrix(columns)
                if accumulator.reached_target():
                    result = accumulator.matrix()
                    memo[memo_key] = result
                    return result
    result = accumulator.matrix()
    memo[memo_key] = result
    return result


@lru_cache(maxsize=128)
def _young_generated_columns_cached(nin, lin, permutation_irrep_signature, L_R, mode, max_factor_L, partition_policy, factor_scope, factor_route_policy):
    return _young_generated_columns_impl(
        nin, lin, permutation_irrep_signature, L_R, mode, max_factor_L,
        partition_policy, factor_scope, factor_route_policy, None, {},
    )


def _young_generated_columns(
    nin, lin, permutation_irrep, L_R, mode, max_factor_L=None,
    partition_policy=(), factor_scope="immediate", factor_route_policy="all",
    factor_route_filter=None,
):
    max_factor_L = _normalize_max_factor_L(max_factor_L)
    args = (
        tuple(int(x) for x in nin),
        tuple(int(x) for x in lin),
        _permutation_irrep_signature(permutation_irrep),
        int(L_R),
        str(mode),
        -1 if max_factor_L is None else int(max_factor_L),
        tuple(partition_policy),
        str(factor_scope),
        str(factor_route_policy),
    )
    if factor_route_filter is not None:
        return _young_generated_columns_impl(*args, factor_route_filter, {})
    return _young_generated_columns_cached(*args)


@lru_cache(maxsize=8192)
def _three_pair_recoupling_overlap(spins, output_L, canonical_J, pair, pair_J):
    """Exact overlap of two three-spin CG trees at highest weight M=L."""

    if pair == (0, 1):
        return _sympy().Integer(int(canonical_J == pair_J))
    from ye3t.core.subtree_dag import cg_exact
    from ye3t.exact_scalars import ExactRadical

    a, b = pair
    c = next(index for index in range(3) if index not in pair)
    total = ExactRadical.rational(0)
    for m0 in range(-spins[0], spins[0] + 1):
        for m1 in range(-spins[1], spins[1] + 1):
            m2 = int(output_L) - m0 - m1
            if abs(m2) > spins[2]:
                continue
            magnetic = (m0, m1, m2)
            canonical_m = m0 + m1
            if abs(canonical_m) > canonical_J:
                continue
            pair_m = magnetic[a] + magnetic[b]
            if abs(pair_m) > pair_J:
                continue
            canonical = cg_exact(spins[0], m0, spins[1], m1, canonical_J, canonical_m)
            canonical *= cg_exact(canonical_J, canonical_m, spins[2], m2, output_L, output_L)
            route = cg_exact(spins[a], magnetic[a], spins[b], magnetic[b], pair_J, pair_m)
            route *= cg_exact(pair_J, pair_m, spins[c], magnetic[c], output_L, output_L)
            total += canonical * route
    sp = _sympy()
    return sum(
        sp.Rational(coefficient.numerator, coefficient.denominator)
        * sp.sqrt(sp.Rational(radicand.numerator, radicand.denominator))
        for radicand, coefficient in total.terms.items()
    )


def _factorized_three_pair_joint_quotient(
    nin, lin, permutation_irrep, L_R, max_factor_L, partition_policy,
    factor_route_filter=None, factor_route_name=None,
):
    """Exact three-pair quotient from small CG recoupling and pair characters.

    Algorithmic reference: CG recoupling and orthogonality, DLMF 34.3-34.4.
    This implementation independently contracts the two CG trees at M=L.
    """

    if permutation_irrep.subgroup != PermutationSubgroup.from_nl(nin, lin):
        return None
    factors = tuple(permutation_irrep.subgroup.factors)
    if len(factors) != 3 or any(int(factor.multiplicity) != 2 for factor in factors):
        return None
    if int(permutation_irrep.dim) != 1:
        return None

    pairs = []
    for factor, partition in zip(factors, permutation_irrep.partitions, strict=True):
        pair_nin = (int(factor.channel_label),) * 2
        pair_lin = (int(factor.l),) * 2
        pair_irrep = permutation_irrep_for_character(pair_nin, pair_lin, (partition.parts,))
        rank_one_irrep = permutation_irrep_for_character(pair_nin[:1], pair_lin[:1], ((1,),))
        counts = generalized_sector_counts(pair_nin, pair_lin, pair_irrep, count_only=True)
        if any(int(multiplicity) != 1 for multiplicity in counts.counts_by_L.values()):
            return None
        retained_Ls = set()
        for pair_L in counts.counts_by_L:
            if not _candidate_product_L_pairs(pair_L, (factor.l,), (factor.l,), "full", max_factor_L):
                continue
            if _factor_route_allowed(
                pair_nin[:1], pair_lin[:1], rank_one_irrep, factor.l,
                pair_nin[:1], pair_lin[:1], rank_one_irrep, factor.l,
                pair_irrep, pair_L, "full", partition_policy, factor_route_filter, "matched_pairs",
            ):
                retained_Ls.add(int(pair_L))
        pairs.append((pair_nin, pair_lin, pair_irrep, counts, frozenset(retained_Ls)))

    sp = _sympy()
    basis_labels = []
    block_columns = []
    generated_indices = []
    product_path_count = 0
    for spins in product(*(tuple(sorted(pair[3].counts_by_L)) for pair in pairs)):
        spins = tuple(int(value) for value in spins)
        canonical_Js = tuple(
            J for J in range(abs(spins[0] - spins[1]), spins[0] + spins[1] + 1)
            if _cg_allowed(J, spins[2], L_R)
        )
        if not canonical_Js:
            continue
        start = len(basis_labels)
        for J in canonical_Js:
            pair_labels = tuple(
                ((factor.channel_label, int(factor.l)), int(pair_L))
                for factor, pair_L in zip(factors, spins, strict=True)
            )
            basis_labels.append(pair_labels if int(L_R) == 0 else (*pair_labels, int(J)))
        if any(int(spins[index]) not in pairs[index][4] for index in range(3)):
            continue
        routes = []
        for last in range(3):
            a, b = tuple(index for index in range(3) if index != last)
            pair_a, pair_b, pair_c = pairs[a], pairs[b], pairs[last]
            rank_four_nin = pair_a[0] + pair_b[0]
            rank_four_lin = pair_a[1] + pair_b[1]
            rank_four_irrep = permutation_irrep_for_character(
                rank_four_nin, rank_four_lin,
                (permutation_irrep.partitions[a].parts, permutation_irrep.partitions[b].parts),
            )
            for intermediate_L in range(abs(spins[a] - spins[b]), spins[a] + spins[b] + 1):
                if not _candidate_product_L_pairs(
                    intermediate_L, (spins[a],), (spins[b],), "full", max_factor_L,
                ):
                    continue
                if not _factor_route_allowed(
                    pair_a[0], pair_a[1], pair_a[2], spins[a],
                    pair_b[0], pair_b[1], pair_b[2], spins[b],
                    rank_four_irrep, intermediate_L, "full", partition_policy, factor_route_filter, "matched_pairs",
                ):
                    continue
                if not _candidate_product_L_pairs(
                    L_R, (intermediate_L,), (spins[last],), "full", max_factor_L,
                ):
                    continue
                if not _factor_route_allowed(
                    rank_four_nin, rank_four_lin, rank_four_irrep, intermediate_L,
                    pair_c[0], pair_c[1], pair_c[2], spins[last],
                    permutation_irrep, L_R, "full", partition_policy, factor_route_filter, "matched_pairs",
                ):
                    continue
                routes.append(((a, b), int(intermediate_L)))
        product_path_count += len(routes)
        if not routes:
            continue
        recoupled = sp.Matrix([
            [
                _three_pair_recoupling_overlap(spins, int(L_R), int(J), pair, int(pair_J))
                for pair, pair_J in routes
            ]
            for J in canonical_Js
        ])
        if any(
            sp.simplify(sum(recoupled[row, col] ** 2 for row in range(recoupled.rows)) - 1) != 0
            for col in range(recoupled.cols)
        ):
            raise ArithmeticError("A three-pair CG route failed exact normalization in the canonical basis.")
        independent_columns = tuple(int(index) for index in recoupled.rref(simplify=False)[1])
        span = recoupled[:, independent_columns]
        _, row_pivots = span.T.rref(simplify=False)
        generated_indices.extend(start + int(index) for index in row_pivots)
        block_columns.append((start, span))

    target_counts = generalized_sector_counts(nin, lin, permutation_irrep, count_only=True)
    sector_count = int(target_counts.counts_by_L.get(int(L_R), 0))
    if len(basis_labels) != sector_count:
        raise ArithmeticError("Three-pair CG channel count disagrees with the exact subgroup character count.")
    generated = sp.zeros(sector_count, sum(matrix.cols for _, matrix in block_columns))
    column_offset = 0
    for row_offset, matrix in block_columns:
        generated[row_offset:row_offset + matrix.rows, column_offset:column_offset + matrix.cols] = matrix
        column_offset += matrix.cols
    exact_rank = _native_algebraic_rank_or_none(generated)
    if exact_rank is None:
        exact_rank = int(generated.rank(simplify=True))
    if int(exact_rank) != len(generated_indices):
        raise ArithmeticError("The exact three-pair product rank disagrees with its coordinate pivots.")
    generated_set = set(generated_indices)
    primitive_indices = tuple(index for index in range(sector_count) if index not in generated_set)
    return YoungResolvedPrimitiveQuotient(
        nin=nin, lin=lin, channel_multiset=_channel_multiset(nin, lin),
        permutation_irrep=permutation_irrep, L_R=int(L_R), mode="full",
        max_factor_L=max_factor_L, sector_count=sector_count,
        sector_basis_rank=sector_count, primitive_rank=len(primitive_indices),
        generated_rank=len(generated_indices),
        generated_basis_indices=tuple(generated_indices),
        primitive_basis_indices=primitive_indices, target_basis_labels=tuple(basis_labels),
        generated_upper_bound=int(product_path_count),
        primitive_lower_bound=max(0, sector_count - int(product_path_count)),
        product_path_count=int(product_path_count), rank_status="exact",
        provenance="exact_subgroup_character_count_and_CG_recoupling",
        codepath="factorized_three_pair_joint_primitive_quotient",
        detail=(
            "Exact S2^3 pair-character sectors and three-spin CG recoupling. "
            "The reported columns span retained products in the canonical pair-CG basis."
        ),
        quotient=generated, factor_scope="recursive",
        allowed_factor_partitions_by_rank=partition_policy,
        factor_route_name=factor_route_name, factor_route_policy="matched_pairs",
        basis_convention="factorized_three_pair_recoupled_cg",
    )


@lru_cache(maxsize=512)
def _global_sector_count_by_character(nin, lin, target_parts, L_R):
    """Count a global S_N/SO(3) highest-weight sector by an H-class trace.

    Algorithmic reference: Frobenius reciprocity for Young-subgroup induction
    and the SU(2) highest-weight difference w_L - w_(L+1).  The class trace
    is evaluated independently of local Young-sector decompositions.
    """

    subgroup = PermutationSubgroup.from_nl(nin, lin)
    target = Partition(target_parts)
    total = 0
    subgroup_order = 1
    for factor in subgroup.factors:
        subgroup_order *= factorial(int(factor.multiplicity))
    class_choices = tuple(
        _partition_parts_of_n(int(factor.multiplicity))
        for factor in subgroup.factors
    )
    for cycle_types in product(*class_choices):
        merged_type = tuple(sorted((length for parts in cycle_types for length in parts), reverse=True))
        character = int(symmetric_group_character(target, merged_type))
        if character == 0:
            continue
        class_size = 1
        weights = {0: 1}
        for factor, cycle_type in zip(subgroup.factors, cycle_types, strict=True):
            class_size *= _cycle_type_class_size(cycle_type)
            weights = _convolve_weight_counts(
                weights, _weight_trace_for_cycle_type(factor.l, cycle_type)
            )
        total += class_size * character * (
            int(weights.get(int(L_R), 0)) - int(weights.get(int(L_R) + 1, 0))
        )
    count, remainder = divmod(total, subgroup_order)
    if remainder or count < 0:
        raise ArithmeticError("The global character trace did not give a nonnegative integral multiplicity.")
    return int(count)


@lru_cache(maxsize=32)
def _global_local_quotient_cached(
    nin, lin, irrep_signature, L_R, mode, max_factor_L, partition_policy,
    factor_scope, factor_route_policy,
):
    """Reuse one exact G_nu product image across global Young partitions."""

    return young_resolved_primitive_quotient(
        nin, lin, _permutation_irrep_from_signature(irrep_signature), int(L_R),
        mode=mode, max_factor_L=None if int(max_factor_L) < 0 else int(max_factor_L),
        allowed_factor_partitions_by_rank=dict(partition_policy),
        factor_scope=factor_scope, factor_route_policy=factor_route_policy,
    )


def _global_fixed_content_primitive_quotient(
    nin, lin, global_partition, L_R, mode, max_factor_L, partition_policy,
    factor_scope, factor_route_policy, factor_route_filter, factor_route_name,
):
    """Induce exact local product images into a global S_N Young sector."""

    from .young_orthogonal import (
        littlewood_richardson_chain_labels,
        young_nary_induced_multiplicity_by_character,
    )

    target = global_partition if isinstance(global_partition, Partition) else Partition(global_partition)
    if int(target.size) != len(nin):
        raise ValueError("global_partition must partition the parent rank.")
    specht_dim = int(target.dimension)
    labels = []
    generated_indices = []
    primitive_indices = []
    blocks = []
    sector_count = 0
    generated_rank = 0
    generated_upper_bound = 0
    for irrep in _all_permutation_irreps_for_pattern(nin, lin):
        counts = generalized_sector_counts(nin, lin, irrep, count_only=True)
        local_copies = int(counts.counts_by_L.get(int(L_R), 0))
        if local_copies == 0:
            continue
        signature = tuple(tuple(int(part) for part in partition.parts) for partition in irrep.partitions)
        induction_multiplicity = int(young_nary_induced_multiplicity_by_character(signature, target))
        if induction_multiplicity == 0:
            continue
        if factor_route_filter is None:
            local = _global_local_quotient_cached(
                nin, lin, _permutation_irrep_signature(irrep), int(L_R), mode,
                -1 if max_factor_L is None else int(max_factor_L),
                partition_policy, factor_scope, factor_route_policy,
            )
        else:
            local = young_resolved_primitive_quotient(
                nin, lin, irrep, int(L_R), mode=mode,
                max_factor_L=max_factor_L,
                allowed_factor_partitions_by_rank=dict(partition_policy),
                factor_scope=factor_scope, factor_route_policy=factor_route_policy,
                factor_route_filter=factor_route_filter, factor_route_name=factor_route_name,
            )
        carrier_dim = int(irrep.dim)
        if (local.rank_status not in {"exact", "exact_no_generators"}
                or int(local.sector_count) != local_copies
                or int(local.sector_basis_rank) != local_copies * carrier_dim):
            raise ArithmeticError("The local exact quotient disagrees with its subgroup character count.")
        lr_labels = littlewood_richardson_chain_labels(signature, target, bracketing="balanced")
        if len(lr_labels) != induction_multiplicity:
            raise ArithmeticError("LR-chain labels disagree with the exact n-ary character multiplicity.")
        sp = _sympy()
        if int(local.generated_rank) == 0:
            local_matrix = sp.zeros(int(local.sector_basis_rank), 0)
        elif hasattr(local.quotient, "rows") and hasattr(local.quotient, "cols"):
            local_matrix = sp.Matrix(local.quotient)
        else:
            local_matrix = _young_generated_columns(
                nin, lin, irrep, int(L_R), mode,
                max_factor_L=max_factor_L, partition_policy=partition_policy,
                factor_scope=factor_scope, factor_route_policy=factor_route_policy,
                factor_route_filter=factor_route_filter,
            )
        if local_matrix.rows != local_copies * carrier_dim:
            raise ArithmeticError("The local product matrix has the wrong Young-carrier dimension.")
        local_matrix_rank = _native_algebraic_rank_or_none(local_matrix)
        if local_matrix_rank is None:
            local_matrix_rank = int(local_matrix.rank(simplify=False))
        if int(local_matrix_rank) != int(local.generated_rank):
            raise ArithmeticError("The local product matrix disagrees with its exact generated rank.")
        if carrier_dim == 1:
            angular_labels = tuple(local.target_basis_labels)
            angular_generated = local_matrix
            angular_indices = tuple(int(index) for index in angular_generated.T.rref(simplify=False)[1])
        else:
            carrier_indices = tuple(product(*(
                range(int(partition.dimension)) for partition in irrep.partitions
            )))
            expected_labels = tuple(
                (copy_index, carrier_index)
                for copy_index in range(local_copies)
                for carrier_index in carrier_indices
            )
            if tuple(local.target_basis_labels) != expected_labels:
                raise ArithmeticError("Local Young coordinates are not ordered by copy and carrier.")
            reference_rows = tuple(index * carrier_dim for index in range(local_copies))
            reference = local_matrix.extract(reference_rows, tuple(range(local_matrix.cols)))
            independent_columns = tuple(int(index) for index in reference.rref(simplify=False)[1])
            angular_generated = reference[:, independent_columns]
            angular_labels = tuple(local.target_basis_labels[index] for index in reference_rows)
            expanded = (
                sp.kronecker_product(angular_generated, sp.eye(carrier_dim))
                if angular_generated.cols else sp.zeros(int(local.sector_basis_rank), 0)
            )
            local_rank = int(local.generated_rank)
            combined_rank = int(sp.Matrix.hstack(local_matrix, expanded).rank(simplify=False))
            if local_rank != carrier_dim * angular_generated.cols or combined_rank != local_rank:
                raise ArithmeticError("The local product image is not a full subgroup-irrep module.")
            angular_indices = tuple(int(index) for index in angular_generated.T.rref(simplify=False)[1])
        angular_rank = int(angular_generated.cols)
        if carrier_dim == 1 and angular_rank != int(local.generated_rank):
            raise ArithmeticError("The local product matrix rank disagrees with the exact quotient.")
        local_copy_bound = int(local.generated_upper_bound) // carrier_dim
        if local_copy_bound < angular_rank:
            raise ArithmeticError("The local product-column bound is smaller than its exact image.")
        for lr_index in range(induction_multiplicity):
            for local_index, angular_label in enumerate(angular_labels):
                for tableau_index in range(specht_dim):
                    index = len(labels)
                    labels.append((signature, angular_label, int(lr_index), int(tableau_index)))
                    if local_index in angular_indices:
                        generated_indices.append(index)
                    else:
                        primitive_indices.append(index)
        sector_count += induction_multiplicity * local_copies
        generated_rank += induction_multiplicity * angular_rank * specht_dim
        generated_upper_bound += induction_multiplicity * local_copy_bound * specht_dim
        blocks.append({
            "subgroup_partitions": signature,
            "subgroup_carrier_dim": carrier_dim,
            "induction_multiplicity": induction_multiplicity,
            "lr_chain_labels": lr_labels,
            "angular_basis_labels": angular_labels,
            "angular_generated_matrix": angular_generated,
            "angular_generated_indices": angular_indices,
            "angular_primitive_indices": tuple(index for index in range(local_copies) if index not in angular_indices),
            "angular_target_rank": local_copies,
            "angular_generated_rank": angular_rank,
            "angular_product_path_count": int(local.product_path_count),
            "local_codepath": local.codepath,
            "local_basis_convention": local.basis_convention,
            "young_map_compiler": "young_orthogonal_nary_subduction",
        })

    expected_count = _global_sector_count_by_character(nin, lin, tuple(target.parts), int(L_R))
    if sector_count != expected_count:
        raise ArithmeticError(
            f"Induced quotient has {sector_count} global multiplicity copies; "
            f"the independent global character trace reports {expected_count}."
        )
    if len(labels) != sector_count * specht_dim:
        raise ArithmeticError("The global quotient basis size disagrees with its Young multiplicity.")
    return YoungResolvedPrimitiveQuotient(
        nin=nin, lin=lin, channel_multiset=_channel_multiset(nin, lin),
        permutation_irrep=None, L_R=int(L_R), mode=str(mode),
        max_factor_L=max_factor_L, sector_count=sector_count,
        sector_basis_rank=len(labels), primitive_rank=len(primitive_indices),
        generated_rank=int(generated_rank),
        generated_basis_indices=tuple(generated_indices),
        primitive_basis_indices=tuple(primitive_indices),
        target_basis_labels=tuple(labels),
        generated_upper_bound=int(generated_upper_bound),
        primitive_lower_bound=max(0, len(labels) - int(generated_upper_bound)),
        product_path_count=int(generated_upper_bound), rank_status="exact",
        provenance="global_character_trace_x_exact_local_product_images_x_young_induction",
        codepath="global_fixed_content_primitive_quotient",
        detail=(
            "Exact full-S_N fixed-content quotient: every retained G_nu product "
            "image is induced over its content orbit and resolved by n-ary LR. "
            "Block matrices carry exact local multiplicity-space images; "
            "induction_map(subgroup_partitions) compiles the Young coefficients on demand."
        ),
        quotient=None, factor_scope=factor_scope,
        allowed_factor_partitions_by_rank=partition_policy,
        factor_route_name=factor_route_name, factor_route_policy=factor_route_policy,
        basis_convention="local_multiplicity_x_induced_Young_LR",
        global_partition=tuple(int(part) for part in target.parts),
        factorized_blocks=tuple(blocks),
    )


def young_resolved_primitive_quotient(
    nin, lin, permutation_irrep, L_R, mode="full", count_only=False,
    max_factor_L=None, allowed_factor_partitions_by_rank=None,
    factor_route_filter=None, factor_route_name=None, factor_scope="immediate",
    factor_route_policy="all", global_partition=None,
):
    """Return the exact Young/SO(3)-resolved primitive quotient for one sector.

    The default quotient uses the canonical highest-weight basis of the joint
    ``SO(3) x G_nu`` sector. With the default full policy, the totally
    symmetric Young sector reduces to the existing ACE primitive quotient.
    For sign or mixed Specht sectors, lower-generated products are assembled
    with exact generalized
    Young/SO(3) tensor-product intertwiners. Child partition allowlists use
    full repeated-content subgroup signatures; a flat partition is accepted
    for a homogeneous child. ``factor_route_filter`` receives the left/right
    child content, Young irreps, angular momenta, and target sector as named
    arguments. It selects child pairs before exact multiplicity branches and
    requires a stable ``factor_route_name`` for reporting. Filtered calls
    bypass the global generated-column caches.

    ``factor_route_policy='matched_pairs'`` restricts a recursive construction
    to equal-content rank-one pairs followed by merges of even-rank children.
    Three distinct two-slot content blocks use exact three-spin CG recoupling
    as an optional local fast path. ``global_partition`` induces every local
    subgroup product image over the complete content orbit at any input rank
    and returns the selected full S_N Young sector. In this mode pass
    ``permutation_irrep=None``. Compact blocks hold exact multiplicity-space
    product-image columns and n-ary LR lift labels; ``induction_map`` compiles
    a selected Young coefficient table on demand.

    Other exact local images use cached homogeneous Young carriers and CG
    coupling, with sparse magnetic overlaps in a common parent basis. This
    avoids constructing full ambient Young projectors. The small three-pair
    recoupling path remains a specialized exact fast path.

    ``factor_scope='immediate'`` takes complete exact child sectors.
    ``factor_scope='recursive'`` retains only child directions generated from
    rank-one sectors under the same policy; it does not add higher-rank
    primitive directions to those children. Stabilizer-sector count-only
    reports remain upper bounds and do not compute recursive child images;
    global-partition calls require an exact quotient. Both policies concern
    abstract fixed-content tensor products; no physical cluster-placement
    or graph-automorphism map is inferred. Quotient basis indices select
    coordinate representatives, not an orthogonal complement.
    """

    nin, lin = _normalize_nl(nin, lin)
    max_factor_L = _normalize_max_factor_L(max_factor_L)
    if str(mode) not in {"invariant", "module", "full"}:
        raise ValueError("mode must be one of {'invariant', 'module', 'full'}")
    partition_policy = _normalize_factor_partition_policy(allowed_factor_partitions_by_rank)
    factor_scope = str(factor_scope)
    factor_route_policy = str(factor_route_policy)
    if factor_scope not in {"immediate", "recursive"}:
        raise ValueError("factor_scope must be 'immediate' or 'recursive'.")
    if factor_route_policy not in {"all", "matched_pairs"}:
        raise ValueError("factor_route_policy must be 'all' or 'matched_pairs'.")
    if factor_route_filter is not None and not callable(factor_route_filter):
        raise ValueError("factor_route_filter must be callable or None.")
    if factor_route_filter is not None and not str(factor_route_name or "").strip():
        raise ValueError("factor_route_name is required with factor_route_filter.")
    if factor_route_filter is None and factor_route_name is not None:
        raise ValueError("factor_route_name requires factor_route_filter.")
    use_factorized = (
        not bool(count_only)
        and (
            max_factor_L is not None or partition_policy or factor_route_filter is not None
            or factor_scope == "recursive" or factor_route_policy != "all"
            or (isinstance(permutation_irrep, PermutationIrrep) and not permutation_irrep.is_totally_symmetric())
        )
    )
    if global_partition is not None:
        if permutation_irrep is not None:
            raise ValueError("Pass permutation_irrep=None when selecting a full global S_N partition.")
        if bool(count_only):
            raise ValueError("The exact global quotient requires count_only=False.")
        return _global_fixed_content_primitive_quotient(
            nin, lin, global_partition, int(L_R), str(mode), max_factor_L,
            partition_policy, factor_scope, factor_route_policy,
            factor_route_filter, factor_route_name,
        )
    policy_fields = {
        "factor_scope": factor_scope,
        "allowed_factor_partitions_by_rank": partition_policy,
        "factor_route_name": factor_route_name,
        "factor_route_policy": factor_route_policy,
        "basis_convention": (
            "character_count_no_basis" if bool(count_only)
            else "factorized_young_cg_gram" if use_factorized
            else "canonical_projector_highest_weight"
        ),
    }
    if not isinstance(permutation_irrep, PermutationIrrep):
        permutation_irrep = permutation_irrep_for_character(nin, lin, permutation_irrep)
    if (
        not bool(count_only) and str(mode) == "full"
        and factor_scope == "recursive" and factor_route_policy == "matched_pairs"
    ):
        factorized = _factorized_three_pair_joint_quotient(
            nin, lin, permutation_irrep, int(L_R), max_factor_L, partition_policy,
            factor_route_filter=factor_route_filter, factor_route_name=factor_route_name,
        )
        if factorized is not None:
            return factorized
    if not permutation_irrep.is_totally_symmetric() and not bool(count_only):
        use_factorized = True
        policy_fields["basis_convention"] = "factorized_young_cg_gram"
    counts = generalized_sector_counts(
        nin, lin, permutation_irrep,
        count_only=bool(count_only) or use_factorized,
    )
    sector_count = int(counts.counts_by_L.get(int(L_R), 0))
    target_sector = counts.sector
    if use_factorized:
        carrier_indices = tuple(product(*(
            range(int(partition.dimension)) for partition in permutation_irrep.partitions
        )))
        target_basis_labels = tuple(
            (copy_index, carrier_index)
            for copy_index in range(sector_count)
            for carrier_index in carrier_indices
        )
    elif not bool(count_only):
        target_basis_labels = tuple(
            _exact_product_expansion_engine().feature_space(nin, lin, int(L_R)).labels
        )
        policy_fields["basis_convention"] = "ace_compact_tree_highest_weight"
    else:
        target_basis_labels = tuple() if target_sector is None else _highest_weight_basis_labels(target_sector, int(L_R))
    sector_basis_rank = int(sector_count) * int(permutation_irrep.dim) if target_sector is None else len(target_basis_labels)
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
            **policy_fields,
        )
    generated_upper_bound, product_path_count = _young_generated_upper_bound(
        nin,
        lin,
        permutation_irrep,
        int(L_R),
        str(mode),
        int(sector_basis_rank),
        max_factor_L=max_factor_L,
        partition_policy=partition_policy,
        factor_route_policy=factor_route_policy,
        factor_route_filter=factor_route_filter,
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
            **policy_fields,
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
            **policy_fields,
        )
    if use_factorized:
        generated = _young_generated_columns(
            nin, lin, permutation_irrep, int(L_R), str(mode),
            max_factor_L=max_factor_L, partition_policy=partition_policy,
            factor_scope=factor_scope, factor_route_policy=factor_route_policy,
            factor_route_filter=factor_route_filter,
        )
        from .factorized_product_images import factorized_requested_sector

        actual_labels = factorized_requested_sector(
            nin, lin, _permutation_irrep_signature(permutation_irrep), int(L_R),
        ).basis_labels
        if actual_labels != target_basis_labels:
            raise ArithmeticError("Factorized quotient labels disagree with the exact parent basis.")
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
            provenance="exact_factorized_young_cg_overlap",
            codepath="factorized_young_so3_primitive_quotient",
            detail="Exact quotient in the factorized Young/SO(3) highest-weight basis.",
            quotient=generated,
            **policy_fields,
        )
    quotient = _exact_product_expansion_engine().primitive_quotient(nin, lin, int(L_R), mode=str(mode))
    if len(quotient.target_space.labels) != int(sector_basis_rank):
        raise ArithmeticError("The exact ACE target basis disagrees with the Young-sector dimension.")
    policy_fields["basis_convention"] = "ace_compact_tree_highest_weight"
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
        target_basis_labels=tuple(quotient.target_space.labels),
        generated_upper_bound=int(generated_upper_bound),
        primitive_lower_bound=int(primitive_lower_bound),
        product_path_count=int(product_path_count),
        rank_status="exact",
        provenance="fallback" if counts.provenance == "cached" else "exact_projector",
        codepath="ace_trivial_primitive_quotient",
        detail="Trivial Young sector delegates to the existing ACE/ye3t primitive quotient.",
        quotient=quotient,
        **policy_fields,
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
        and len(set(generated_indices)) == len(generated_indices) == int(quotient.generated_rank)
        and len(set(primitive_indices)) == len(primitive_indices) == int(quotient.primitive_rank)
    )
    generated_rank_matches_matrix = True
    if quotient.codepath in {
        "symbolic_young_so3_primitive_quotient",
        "factorized_young_so3_primitive_quotient",
        "factorized_three_pair_joint_primitive_quotient",
    } and quotient.quotient is not None:
        native_rank = _native_algebraic_rank_or_none(quotient.quotient)
        if native_rank is None:
            native_rank = int(_sympy().Matrix(quotient.quotient).rank(simplify=False))
        generated_rank_matches_matrix = int(native_rank) == int(quotient.generated_rank)
        if quotient.codepath == "factorized_young_so3_primitive_quotient":
            generated_rank_matches_matrix = bool(
                generated_rank_matches_matrix
                and len(quotient.target_basis_labels) == int(quotient.sector_basis_rank)
                and quotient.basis_convention == "factorized_young_cg_gram"
            )
        if quotient.codepath == "factorized_three_pair_joint_primitive_quotient":
            matrix = _sympy().Matrix(quotient.quotient)
            generated_rank_matches_matrix = bool(
                generated_rank_matches_matrix
                and len(quotient.target_basis_labels) == int(quotient.sector_basis_rank)
                and quotient.basis_convention == "factorized_three_pair_recoupled_cg"
            )
    if quotient.codepath == "global_fixed_content_primitive_quotient":
        global_rank = 0
        global_size = 0
        specht_dim = int(Partition(quotient.global_partition).dimension)
        for block in quotient.factorized_blocks:
            matrix = _sympy().Matrix(block["angular_generated_matrix"])
            local_rank = _native_algebraic_rank_or_none(matrix)
            if local_rank is None:
                local_rank = int(matrix.rank(simplify=False))
            generated_rank_matches_matrix = bool(
                generated_rank_matches_matrix
                and int(local_rank) == int(block["angular_generated_rank"])
                and matrix.rows == int(block["angular_target_rank"])
                and len(block["lr_chain_labels"]) == int(block["induction_multiplicity"])
            )
            global_rank += int(local_rank) * int(block["induction_multiplicity"]) * specht_dim
            global_size += int(block["angular_target_rank"]) * int(block["induction_multiplicity"]) * specht_dim
        generated_rank_matches_matrix = bool(
            generated_rank_matches_matrix
            and global_rank == int(quotient.generated_rank)
            and global_size == int(quotient.sector_basis_rank)
        )
    trivial_matches_ace = None
    if quotient.permutation_irrep is not None and quotient.permutation_irrep.is_totally_symmetric():
        if (
            quotient.max_factor_L is None
            and quotient.factor_scope == "immediate"
            and quotient.factor_route_policy == "all"
            and not quotient.allowed_factor_partitions_by_rank
            and quotient.factor_route_name is None
        ):
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
