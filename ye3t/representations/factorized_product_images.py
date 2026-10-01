"""Exact fixed-content product images in a block Young/CG basis.

The parent basis is assembled from homogeneous Young carriers and binary
SO(3) Clebsch--Gordan merges. Multi-content products combine cached reduced
same-content fusion maps and recouple their CG trees. Each elementary fusion
projects sparse product vectors onto the target basis with exact Gram inner
products; matching representation labels reuse the map. Cached Young-subgroup
subduction coefficients propagate the reduced map over the remaining
placements. No full magnetic-space Young projector or parent nullspace is
constructed.

Algorithmic references: subgroup-adapted Specht bases (de Mello Koch, Ives,
and Stephanou, J. Phys. A 45 (2012) 135204) and CG orthogonality/recoupling
(NIST DLMF 34.3--34.4).  This implementation is independent; no source code
from those references is used.
"""

from functools import lru_cache
from itertools import product

from ye3t._record import recordclass
from ye3t.core.subtree_dag import cg_exact
from ye3t.exact_scalars import ExactRadical, exact_scalar


def _sympy():
    from ye3t._optional_sympy import sp

    return sp


@recordclass(
    ('nin', 'lin', 'permutation_irrep', 'L', 'content', 'basis_labels',
     'highest_vectors', 'multiplets', 'gram_inverse', 'copy_descriptors'),
    frozen=True,
)
class _FactorizedRequestedSector:
    """One exact requested-L local basis with copy then Young-carrier order."""


def _sparse_vector(states, vector):
    return {
        tuple(state): exact_scalar(value)
        for state, value in zip(states, vector, strict=True)
        if value != 0
    }


def _sparse_inner(left, right):
    if len(left) > len(right):
        left, right = right, left
    total = ExactRadical.rational(0)
    for state, value in left.items():
        other = right.get(state)
        if other is not None:
            total += value * other
    return total


def _couple_multiplets(left, left_L, right, right_L, output_L):
    """Couple sparse magnetic vectors in the exact Condon--Shortley gauge."""

    coupled = {}
    for output_M in range(-int(output_L), int(output_L) + 1):
        values = {}
        for left_M in range(-int(left_L), int(left_L) + 1):
            right_M = int(output_M) - int(left_M)
            if abs(right_M) > int(right_L):
                continue
            coefficient = cg_exact(
                int(left_L), int(left_M), int(right_L), int(right_M),
                int(output_L), int(output_M),
            )
            if not coefficient:
                continue
            for left_state, left_value in left[int(left_M)].items():
                for right_state, right_value in right[int(right_M)].items():
                    state = left_state + right_state
                    values[state] = values.get(state, ExactRadical.rational(0)) + (
                        coefficient * left_value * right_value
                    )
        coupled[int(output_M)] = {
            state: value for state, value in values.items() if not value.is_zero()
        }
    return coupled


@lru_cache(maxsize=256)
def _single_factor_carriers(multiplicity, angular_l, partition_parts, output_L):
    """Cache a homogeneous factor's exact tableau and magnetic vectors."""

    from .builder import _exact_single_factor_weight_vectors
    from .young_sectors import permutation_irrep_for_character

    nin = (0,) * int(multiplicity)
    lin = (int(angular_l),) * int(multiplicity)
    irrep = permutation_irrep_for_character(nin, lin, (tuple(partition_parts),))
    states, vectors, copy_count = _exact_single_factor_weight_vectors(
        nin, lin, irrep, int(output_L), normalize_tableaux=False,
    )
    carriers = {}
    for copy_index in range(int(copy_count)):
        for tableau in range(int(irrep.dim)):
            carriers[(int(copy_index), int(tableau))] = {
                int(magnetic): _sparse_vector(states, vectors[(copy_index, tableau, magnetic)])
                for magnetic in range(-int(output_L), int(output_L) + 1)
            }
    return carriers


@lru_cache(maxsize=256)
def _factor_angular_counts(multiplicity, angular_l, partition_parts):
    from .young_sectors import generalized_sector_counts, permutation_irrep_for_character

    nin = (0,) * int(multiplicity)
    lin = (int(angular_l),) * int(multiplicity)
    irrep = permutation_irrep_for_character(nin, lin, (tuple(partition_parts),))
    return tuple(sorted(
        (int(L), int(count))
        for L, count in generalized_sector_counts(nin, lin, irrep, count_only=True).counts_by_L.items()
        if int(count) > 0
    ))


@lru_cache(maxsize=4096)
def _content_placement_and_subgroup_shuffles(left_content, right_content, target_content):
    """Cache the content placement and the parent/child Young-subgroup cosets."""

    from .young_orthogonal import _young_subgroup_shuffle_representatives

    target_positions = {}
    for index, channel in enumerate(target_content):
        target_positions.setdefault(channel, []).append(index)
    placement = []
    for channel in left_content + right_content:
        positions = target_positions.get(channel)
        if not positions:
            raise ValueError("Child contents do not sum to the parent content.")
        placement.append(positions.pop(0))
    if any(target_positions.values()):
        raise ValueError("Child contents do not cover the parent content.")

    factor_shuffles = []
    cursor = 0
    while cursor < len(target_content):
        channel = target_content[cursor]
        size = 1
        while cursor + size < len(target_content) and target_content[cursor + size] == channel:
            size += 1
        left_count = left_content.count(channel)
        right_count = right_content.count(channel)
        if left_count + right_count != size:
            raise ValueError("Child multiplicities do not sum to the parent subgroup factor.")
        slots = tuple(range(cursor, cursor + size))
        factor_shuffles.append(tuple(
            (slots, shuffle)
            for shuffle in _young_subgroup_shuffle_representatives((left_count, right_count))
        ))
        cursor += size
    orbit_permutations = []
    for selection in product(*factor_shuffles):
        permutation = list(range(len(target_content)))
        for slots, shuffle in selection:
            for new_offset, slot in enumerate(slots):
                permutation[slot] = slots[shuffle[new_offset]]
        # The raw magnetic action indexes source slots at output positions;
        # Young induction labels active left-coset representatives.
        inverse = [0] * len(permutation)
        for output_index, source_index in enumerate(permutation):
            inverse[source_index] = output_index
        orbit_permutations.append(tuple(inverse))
    return tuple(placement), tuple(orbit_permutations)


@lru_cache(maxsize=512)
def _young_factor_restriction_table(left_parts, right_parts, target_parts):
    """Young subduction coefficients for one merged content factor."""

    from .generalized_irreps import Partition
    from .young_orthogonal import young_orthogonal_nary_subduction

    target_dim = int(Partition(target_parts).dimension)
    left_dim = int(Partition(left_parts).dimension) if left_parts else 1
    right_dim = int(Partition(right_parts).dimension) if right_parts else 1
    if not left_parts or not right_parts:
        present = left_parts or right_parts
        if tuple(present) != tuple(target_parts):
            raise ArithmeticError("An unsplit Young factor changed partition.")
        table = {}
        for tableau in range(target_dim):
            left_tableau = tableau if left_parts else 0
            right_tableau = tableau if right_parts else 0
            table[(0, tableau, left_tableau, right_tableau, 0)] = _sympy().Integer(1)
        return 1, 1, left_dim, right_dim, target_dim, table

    tensor = young_orthogonal_nary_subduction(
        (tuple(left_parts), tuple(right_parts)), tuple(target_parts),
        coefficient_backend="subduction_graph",
    )
    table = {}
    for row, entry in enumerate(tensor.induced_basis):
        left_tableau, right_tableau = entry.child_tableau_indices
        for vector in tensor.vectors:
            value = vector.coefficients[row]
            if value != 0:
                table[(
                    int(entry.coset_index), int(vector.target_tableau_index),
                    int(left_tableau), int(right_tableau), int(vector.rho),
                )] = value
    return len(tensor.coset_reps), int(tensor.multiplicity), left_dim, right_dim, target_dim, table


@lru_cache(maxsize=512)
def _local_young_subduction_template(left_signature, right_signature, target_signature):
    """Cache carrier maps and the identity-coset reduced-channel solve."""

    from .young_sectors import _permutation_irrep_from_signature

    left = _permutation_irrep_from_signature(left_signature)
    right = _permutation_irrep_from_signature(right_signature)
    target = _permutation_irrep_from_signature(target_signature)
    left_indices = {
        (factor.channel_label, int(factor.l)): index
        for index, factor in enumerate(left.subgroup.factors)
    }
    right_indices = {
        (factor.channel_label, int(factor.l)): index
        for index, factor in enumerate(right.subgroup.factors)
    }
    factor_tables = []
    for factor, partition in zip(target.subgroup.factors, target.partitions, strict=True):
        key = (factor.channel_label, int(factor.l))
        left_index = left_indices.get(key)
        right_index = right_indices.get(key)
        left_parts = tuple(left.partitions[left_index].parts) if left_index is not None else tuple()
        right_parts = tuple(right.partitions[right_index].parts) if right_index is not None else tuple()
        cosets, multiplicity, _left_dim, _right_dim, _target_dim, table = _young_factor_restriction_table(
            left_parts, right_parts, tuple(partition.parts),
        )
        factor_tables.append((left_index, right_index, cosets, multiplicity, table))

    target_carriers = tuple(product(*(range(int(partition.dimension)) for partition in target.partitions)))
    left_carriers = tuple(product(*(range(int(partition.dimension)) for partition in left.partitions)))
    right_carriers = tuple(product(*(range(int(partition.dimension)) for partition in right.partitions)))
    rho_labels = tuple(product(*(range(item[3]) for item in factor_tables)))
    coset_labels = tuple(product(*(range(item[2]) for item in factor_tables)))
    rows = tuple(
        (target_carrier, left_carrier, right_carrier)
        for target_carrier in target_carriers
        for left_carrier in left_carriers
        for right_carrier in right_carriers
    )
    matrices = []
    for coset in coset_labels:
        entries = []
        for target_carrier, left_carrier, right_carrier in rows:
            row = []
            for rho in rho_labels:
                coefficient = _sympy().Integer(1)
                for factor_index, (left_index, right_index, _cosets, _multiplicity, table) in enumerate(factor_tables):
                    left_tableau = left_carrier[left_index] if left_index is not None else 0
                    right_tableau = right_carrier[right_index] if right_index is not None else 0
                    coefficient *= table.get((
                        coset[factor_index], target_carrier[factor_index],
                        left_tableau, right_tableau, rho[factor_index],
                    ), 0)
                    if coefficient == 0:
                        break
                row.append(coefficient)
            entries.append(row)
        matrices.append(_sympy().Matrix(entries))
    identity = matrices[0]
    if identity.rank() != len(rho_labels):
        raise ArithmeticError("The identity-coset Young restriction does not resolve every LR copy.")
    left_inverse = (identity.T * identity).inv() * identity.T
    return rows, tuple(matrices), left_inverse


@lru_cache(maxsize=128)
def factorized_requested_sector(nin, lin, permutation_signature, output_L):
    """Build the exact local Young x SO(3) basis at one requested L."""

    from .young_sectors import (
        _permutation_irrep_from_signature,
        generalized_sector_counts,
    )

    nin = tuple(int(value) for value in nin)
    lin = tuple(int(value) for value in lin)
    output_L = int(output_L)
    irrep = _permutation_irrep_from_signature(permutation_signature)
    if irrep.subgroup != irrep.subgroup.from_nl(nin, lin):
        raise ValueError("Requested Young irrep does not match fixed content.")
    factors = tuple(irrep.subgroup.factors)
    content = tuple(
        (factor.channel_label, int(factor.l))
        for factor in factors
        for _ in range(int(factor.multiplicity))
    )
    expected = int(generalized_sector_counts(nin, lin, irrep, count_only=True).counts_by_L.get(output_L, 0))
    if not factors or expected == 0:
        return _FactorizedRequestedSector(
            nin, lin, irrep, output_L, content, tuple(), {}, {}, _sympy().zeros(0, 0), tuple(),
        )

    factor_options = []
    for factor_index, (factor, partition) in enumerate(zip(factors, irrep.partitions, strict=True)):
        options = []
        remaining_max = sum(
            int(other.multiplicity) * int(other.l)
            for other in factors[factor_index + 1:]
        )
        for angular_L, count in _factor_angular_counts(
            factor.multiplicity, factor.l, tuple(partition.parts),
        ):
            if factor_index == 0 and abs(int(angular_L) - output_L) > remaining_max:
                continue
            carriers = _single_factor_carriers(
                factor.multiplicity, factor.l, tuple(partition.parts), angular_L,
            )
            if len(carriers) != int(count) * int(partition.dimension):
                raise ArithmeticError("Factorized Young carrier count disagrees with the character count.")
            for (copy_index, tableau), multiplet in carriers.items():
                options.append((int(angular_L), int(copy_index), int(tableau), multiplet))
        factor_options.append(tuple(options))

    partials = [
        ((spin,), (copy,), tuple(), (tableau,), multiplet)
        for spin, copy, tableau, multiplet in factor_options[0]
    ]
    for factor_index, options in enumerate(factor_options[1:], start=1):
        remaining_max = sum(
            int(item.multiplicity) * int(item.l)
            for item in factors[factor_index + 1:]
        )
        next_partials = []
        for spins, copies, intermediates, carrier, left_multiplet in partials:
            left_L = int(intermediates[-1]) if intermediates else int(spins[0])
            for right_L, copy, tableau, right_multiplet in options:
                for coupled_L in range(abs(left_L - right_L), left_L + right_L + 1):
                    if abs(int(coupled_L) - output_L) > remaining_max:
                        continue
                    next_partials.append((
                        spins + (right_L,), copies + (copy,),
                        intermediates + (int(coupled_L),), carrier + (tableau,),
                        _couple_multiplets(
                            left_multiplet, left_L, right_multiplet, right_L, coupled_L,
                        ),
                    ))
        partials = next_partials

    completed = sorted(
        (spins, copies, intermediates, carrier, multiplet)
        for spins, copies, intermediates, carrier, multiplet in partials
        if (int(intermediates[-1]) if intermediates else int(spins[0])) == output_L
    )
    copy_descriptors = tuple(sorted({
        (spins, copies, intermediates)
        for spins, copies, intermediates, _carrier, _multiplet in completed
    }))
    copy_index = {descriptor: index for index, descriptor in enumerate(copy_descriptors)}
    highest = {}
    multiplets = {}
    for spins, copies, intermediates, carrier, multiplet in completed:
        label = (int(copy_index[(spins, copies, intermediates)]), tuple(carrier))
        highest[label] = multiplet[output_L]
        multiplets[label] = multiplet
    labels = tuple(sorted(highest))
    if len(copy_descriptors) != expected or len(labels) != expected * int(irrep.dim):
        raise ArithmeticError("Factorized basis size disagrees with the exact Young/SO(3) count.")
    reference_carrier = tuple(0 for _ in irrep.partitions)
    copy_gram = tuple(tuple(
        _sparse_inner(
            highest[(left_copy, reference_carrier)],
            highest[(right_copy, reference_carrier)],
        )
        for right_copy in range(expected)
    ) for left_copy in range(expected))
    # The unnormalized tableau vectors carry the same orthogonal Specht
    # action for every copy. Schur's lemma makes their Gram matrix
    # G_copy x I_carrier; check that identity before using the smaller solve.
    for left_label in labels:
        for right_label in labels:
            expected_overlap = (
                copy_gram[left_label[0]][right_label[0]]
                if left_label[1] == right_label[1]
                else ExactRadical.rational(0)
            )
            if _sparse_inner(highest[left_label], highest[right_label]) != expected_overlap:
                raise ArithmeticError("Factorized Young basis violates carrier Gram factorization.")
    gram = _sympy().Matrix([
        [value._sympy_() for value in row]
        for row in copy_gram
    ])
    try:
        gram_inverse = gram.inv()
    except _sympy().NonInvertibleMatrixError as exc:
        raise ArithmeticError("Factorized Young/SO(3) highest-weight basis is singular.") from exc
    return _FactorizedRequestedSector(
        nin, lin, irrep, output_L, content, labels, highest, multiplets,
        _sympy().kronecker_product(gram_inverse, _sympy().eye(int(irrep.dim))),
        copy_descriptors,
    )


@lru_cache(maxsize=8192)
def _angular_tree_coefficients(spins, intermediates, magnetic_M):
    """Exact CG coefficients of one block-spin tree in its leaf-magnetic basis."""

    if not spins:
        return {}
    vectors = {(m,): ExactRadical.rational(1) for m in range(-spins[0], spins[0] + 1)}
    current_spin = int(spins[0])
    for index, right_spin in enumerate(spins[1:]):
        output_spin = int(intermediates[index])
        next_vectors = {}
        for left_magnetic, value in vectors.items():
            left_M = sum(left_magnetic)
            for right_M in range(-int(right_spin), int(right_spin) + 1):
                coefficient = cg_exact(current_spin, left_M, int(right_spin), right_M,
                                       output_spin, left_M + right_M)
                if coefficient:
                    next_vectors[left_magnetic + (right_M,)] = value * coefficient
        vectors = next_vectors
        current_spin = output_spin
    return {magnetic: value for magnetic, value in vectors.items()
            if sum(magnetic) == int(magnetic_M)}


@lru_cache(maxsize=8192)
def _angular_block_recoupling(left_descriptor, right_descriptor, target_descriptor,
                              left_positions, right_positions, left_L, right_L, target_L):
    """Contract the two child CG trees with the parent block-spin tree."""

    left_spins, _, left_intermediates = left_descriptor
    right_spins, _, right_intermediates = right_descriptor
    target_spins, _, target_intermediates = target_descriptor
    target_vector = _angular_tree_coefficients(target_spins, target_intermediates, target_L)
    left_at = {position: index for index, position in enumerate(left_positions)}
    right_at = {position: index for index, position in enumerate(right_positions)}
    total = ExactRadical.rational(0)
    for left_M in range(-int(left_L), int(left_L) + 1):
        right_M = int(target_L) - left_M
        if abs(right_M) > int(right_L):
            continue
        outer = cg_exact(int(left_L), left_M, int(right_L), right_M,
                         int(target_L), int(target_L))
        if not outer:
            continue
        for left_magnetic, left_value in _angular_tree_coefficients(
                left_spins, left_intermediates, left_M).items():
            for right_magnetic, right_value in _angular_tree_coefficients(
                    right_spins, right_intermediates, right_M).items():
                parent_magnetic = []
                coefficient = outer * left_value * right_value
                for index, parent_spin in enumerate(target_spins):
                    li, ri = left_at.get(index), right_at.get(index)
                    if li is None:
                        parent_magnetic.append(right_magnetic[ri])
                    elif ri is None:
                        parent_magnetic.append(left_magnetic[li])
                    else:
                        left_m, right_m = left_magnetic[li], right_magnetic[ri]
                        coefficient *= cg_exact(
                            int(left_spins[li]), left_m, int(right_spins[ri]), right_m,
                            int(parent_spin), left_m + right_m,
                        )
                        parent_magnetic.append(left_m + right_m)
                    if not coefficient:
                        break
                if coefficient:
                    parent_value = target_vector.get(tuple(parent_magnetic))
                    if parent_value is not None:
                        total += coefficient * parent_value
    return total._sympy_()


@lru_cache(maxsize=512)
def _homogeneous_reduced_map(left_size, right_size, angular_l,
                             left_parts, right_parts, target_parts,
                             left_L, right_L, target_L):
    """Precompile one same-content Young/angular fusion on copy coordinates."""

    from .young_sectors import _permutation_irrep_signature, permutation_irrep_for_character

    sectors = []
    for size, parts, output_L in (
        (left_size, left_parts, left_L),
        (right_size, right_parts, right_L),
        (left_size + right_size, target_parts, target_L),
    ):
        nin, lin = (0,) * int(size), (int(angular_l),) * int(size)
        irrep = permutation_irrep_for_character(nin, lin, (tuple(parts),))
        sectors.append(factorized_requested_sector(
            nin, lin, _permutation_irrep_signature(irrep), int(output_L),
        ))
    left, right, target = sectors
    matrix, _ = _factorized_product_image_reference(left, right, target)
    rows, young_matrices, left_inverse = _local_young_subduction_template(
        _permutation_irrep_signature(left.permutation_irrep),
        _permutation_irrep_signature(right.permutation_irrep),
        _permutation_irrep_signature(target.permutation_irrep),
    )
    target_index = {label: index for index, label in enumerate(target.basis_labels)}
    child_index = {
        (left_label, right_label): index
        for index, (left_label, right_label) in enumerate(product(left.basis_labels, right.basis_labels))
    }
    reduced = {}
    for target_copy in range(len(target.copy_descriptors)):
        for left_copy in range(len(left.copy_descriptors)):
            for right_copy in range(len(right.copy_descriptors)):
                channel = _sympy().Matrix([
                    matrix[
                        target_index[(target_copy, (target_tableau,))],
                        child_index[((left_copy, (left_tableau,)),
                                     (right_copy, (right_tableau,)))],
                    ]
                    for (target_tableau,), (left_tableau,), (right_tableau,) in rows
                ])
                coefficients = left_inverse * channel
                if young_matrices[0] * coefficients != channel:
                    raise ArithmeticError("Homogeneous Young/angular fusion fails exact subduction.")
                for rho, value in enumerate(coefficients):
                    if value != 0:
                        reduced[(target_copy, left_copy, right_copy, rho)] = value
    return young_matrices[0].cols, reduced


def factorized_product_image(left, right, target, *, coefficient_backend="subduction"):
    """Compile the exact product through cached block fusions and CG recoupling.

    The direct-orbit path is retained as an independent small-case reference.
    One-content-block fusions use exact sparse-vector inner products.
    """

    if coefficient_backend == "direct_orbit" or len(target.permutation_irrep.subgroup.factors) == 1:
        return _factorized_product_image_reference(
            left, right, target, coefficient_backend=coefficient_backend,
        )
    if coefficient_backend != "subduction":
        raise ValueError("coefficient_backend must be 'subduction' or 'direct_orbit'.")
    if not abs(int(left.L) - int(right.L)) <= int(target.L) <= int(left.L) + int(right.L):
        raise ValueError("Requested output L violates the angular triangle rule.")
    from .young_sectors import _permutation_irrep_signature

    _placement, shuffles = _content_placement_and_subgroup_shuffles(
        left.content, right.content, target.content,
    )
    child_labels = tuple(product(left.basis_labels, right.basis_labels))
    source_labels = tuple(
        (coset, left_label, right_label)
        for coset in range(len(shuffles)) for left_label, right_label in child_labels
    )
    if not child_labels:
        return _sympy().zeros(len(target.basis_labels), 0), source_labels
    rows, young_matrices, _ = _local_young_subduction_template(
        _permutation_irrep_signature(left.permutation_irrep),
        _permutation_irrep_signature(right.permutation_irrep),
        _permutation_irrep_signature(target.permutation_irrep),
    )
    if len(young_matrices) != len(shuffles):
        raise ArithmeticError("Young subduction and content placement disagree on cosets.")
    target_factors = tuple(target.permutation_irrep.subgroup.factors)
    left_factors = tuple(left.permutation_irrep.subgroup.factors)
    right_factors = tuple(right.permutation_irrep.subgroup.factors)
    parent_index = {(factor.channel_label, int(factor.l)): index
                    for index, factor in enumerate(target_factors)}
    left_positions = tuple(parent_index[(factor.channel_label, int(factor.l))]
                           for factor in left_factors)
    right_positions = tuple(parent_index[(factor.channel_label, int(factor.l))]
                            for factor in right_factors)
    left_at = {position: index for index, position in enumerate(left_positions)}
    right_at = {position: index for index, position in enumerate(right_positions)}
    factor_maps = []
    for index, factor in enumerate(target_factors):
        li, ri = left_at.get(index), right_at.get(index)
        if li is None or ri is None:
            factor_maps.append((1, {}))
            continue
        left_factor, right_factor = left_factors[li], right_factors[ri]
        left_parts = tuple(left.permutation_irrep.partitions[li].parts)
        right_parts = tuple(right.permutation_irrep.partitions[ri].parts)
        target_parts = tuple(target.permutation_irrep.partitions[index].parts)
        maps_by_spins = {}
        for left_spin in {descriptor[0][li] for descriptor in left.copy_descriptors}:
            for right_spin in {descriptor[0][ri] for descriptor in right.copy_descriptors}:
                for target_spin in {descriptor[0][index] for descriptor in target.copy_descriptors}:
                    if abs(left_spin - right_spin) <= target_spin <= left_spin + right_spin:
                        maps_by_spins[(left_spin, right_spin, target_spin)] = _homogeneous_reduced_map(
                            left_factor.multiplicity, right_factor.multiplicity, factor.l,
                            left_parts, right_parts, target_parts,
                            left_spin, right_spin, target_spin,
                        )
        factor_maps.append((None, maps_by_spins))
    target_index = {label: index for index, label in enumerate(target.basis_labels)}
    child_index = {label: index for index, label in enumerate(child_labels)}
    coordinates = _sympy().zeros(len(target.basis_labels), len(source_labels))
    for target_copy, target_descriptor in enumerate(target.copy_descriptors):
        for left_copy, left_descriptor in enumerate(left.copy_descriptors):
            for right_copy, right_descriptor in enumerate(right.copy_descriptors):
                recoupling = _angular_block_recoupling(
                    left_descriptor, right_descriptor, target_descriptor,
                    left_positions, right_positions, left.L, right.L, target.L,
                )
                if recoupling == 0:
                    continue
                factor_coefficients = []
                for index, (single_rho, maps_by_spins) in enumerate(factor_maps):
                    li, ri = left_at.get(index), right_at.get(index)
                    if single_rho is not None:
                        child_descriptor = left_descriptor if ri is None else right_descriptor
                        child_position = li if ri is None else ri
                        if (target_descriptor[0][index], target_descriptor[1][index]) != (
                            child_descriptor[0][child_position], child_descriptor[1][child_position]
                        ):
                            factor_coefficients = []
                            break
                        factor_coefficients.append((_sympy().Integer(1),))
                    else:
                        spin_key = (left_descriptor[0][li], right_descriptor[0][ri],
                                    target_descriptor[0][index])
                        map_entry = maps_by_spins.get(spin_key)
                        if map_entry is None:
                            factor_coefficients = []
                            break
                        rho_count, reduced_map = map_entry
                        factor_coefficients.append(tuple(
                            reduced_map.get((target_descriptor[1][index],
                                             left_descriptor[1][li], right_descriptor[1][ri], rho), 0)
                            for rho in range(rho_count)
                        ))
                if not factor_coefficients:
                    continue
                reduced = _sympy().Matrix([
                    recoupling * _sympy().prod(values)
                    for values in product(*factor_coefficients)
                ])
                if reduced.rows != young_matrices[0].cols:
                    raise ArithmeticError("Reduced block fusions disagree with Young LR multiplicity.")
                for coset, young_matrix in enumerate(young_matrices):
                    expanded = young_matrix * reduced
                    for row, (target_carrier, left_carrier, right_carrier) in enumerate(rows):
                        coordinates[
                            target_index[(target_copy, target_carrier)],
                            coset * len(child_labels) + child_index[
                                ((left_copy, left_carrier), (right_copy, right_carrier))
                            ],
                        ] = _sympy().simplify(expanded[row])
    return coordinates, source_labels


def _factorized_product_image_reference(left, right, target, *, coefficient_backend="subduction"):
    """Exact induced child-to-parent coordinates without an ambient projector.

    Young subduction propagates the identity-placement magnetic overlap to
    every subgroup coset. ``direct_orbit`` is an exact reference backend.
    """

    if int(target.L) > int(left.L) + int(right.L) or int(target.L) < abs(int(left.L) - int(right.L)):
        raise ValueError("Requested output L violates the angular triangle rule.")
    if coefficient_backend not in {"subduction", "direct_orbit"}:
        raise ValueError("coefficient_backend must be 'subduction' or 'direct_orbit'.")
    placement, orbit_permutations = _content_placement_and_subgroup_shuffles(
        left.content, right.content, target.content,
    )

    child_labels = tuple(
        (left_label, right_label)
        for left_label in left.basis_labels
        for right_label in right.basis_labels
    )
    source_labels = tuple(
        (coset_index, left_label, right_label)
        for coset_index in range(len(orbit_permutations))
        for left_label, right_label in child_labels
    )
    if not child_labels:
        return _sympy().zeros(len(target.basis_labels), 0), source_labels

    placed_vectors = []
    for left_label, right_label in child_labels:
        coupled = _couple_multiplets(
            left.multiplets[left_label], left.L,
            right.multiplets[right_label], right.L, target.L,
        )[target.L]
        placed = {}
        for source_state, value in coupled.items():
            target_state = [0] * len(placement)
            for source_index, target_index in enumerate(placement):
                target_state[target_index] = source_state[source_index]
            state = tuple(target_state)
            placed[state] = placed.get(state, ExactRadical.rational(0)) + value
        placed_vectors.append(placed)

    def coordinates_for_coset(coset_index):
        overlap_columns = []
        for placed in placed_vectors:
            permuted = {
                tuple(state[index] for index in orbit_permutations[coset_index]): value
                for state, value in placed.items()
            }
            overlap_columns.append([
                _sparse_inner(target.highest_vectors[label], permuted)._sympy_()
                for label in target.basis_labels
            ])
        return (target.gram_inverse * _sympy().Matrix(overlap_columns).T).applyfunc(_sympy().simplify)

    identity_coordinates = coordinates_for_coset(0)
    if len(orbit_permutations) == 1:
        return identity_coordinates, source_labels
    if coefficient_backend == "direct_orbit":
        return _sympy().Matrix.hstack(*(
            identity_coordinates,
            *(coordinates_for_coset(index) for index in range(1, len(orbit_permutations))),
        )), source_labels

    from .young_sectors import _permutation_irrep_signature

    carrier_rows, young_matrices, identity_left_inverse = _local_young_subduction_template(
        _permutation_irrep_signature(left.permutation_irrep),
        _permutation_irrep_signature(right.permutation_irrep),
        _permutation_irrep_signature(target.permutation_irrep),
    )
    if len(young_matrices) != len(orbit_permutations):
        raise ArithmeticError("Young subduction and content placement disagree on subgroup cosets.")
    target_index = {label: index for index, label in enumerate(target.basis_labels)}
    child_index = {label: index for index, label in enumerate(child_labels)}
    target_copies = tuple(sorted({label[0] for label in target.basis_labels}))
    left_copies = tuple(sorted({label[0] for label in left.basis_labels}))
    right_copies = tuple(sorted({label[0] for label in right.basis_labels}))
    coordinates = _sympy().zeros(len(target.basis_labels), len(source_labels))
    for target_copy in target_copies:
        for left_copy in left_copies:
            for right_copy in right_copies:
                identity_channel = _sympy().Matrix([
                    identity_coordinates[
                        target_index[(target_copy, target_carrier)],
                        child_index[((left_copy, left_carrier), (right_copy, right_carrier))],
                    ]
                    for target_carrier, left_carrier, right_carrier in carrier_rows
                ])
                reduced = identity_left_inverse * identity_channel
                residual = young_matrices[0] * reduced - identity_channel
                if any(_sympy().simplify(value) != 0 for value in residual):
                    raise ArithmeticError("Local Young/CG map fails exact identity-coset subduction.")
                for coset_index, young_matrix in enumerate(young_matrices):
                    expanded = young_matrix * reduced
                    for row, (target_carrier, left_carrier, right_carrier) in enumerate(carrier_rows):
                        target_row = target_index[(target_copy, target_carrier)]
                        source_col = (
                            coset_index * len(child_labels)
                            + child_index[((left_copy, left_carrier), (right_copy, right_carrier))]
                        )
                        coordinates[target_row, source_col] = _sympy().simplify(expanded[row])
    return coordinates, source_labels


__all__ = []
