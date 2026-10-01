"""Young orthogonal Specht-basis coupling tensors.

This module builds small-``N`` permutation coupling tensors algebraically from
standard-tableau Young orthogonal representation matrices.  It avoids
Gram-Schmidt or SVD on evaluated feature vectors; multiplicity copies are fixed
by exact symbolic intertwiner equations and an exact multiplicity-space metric.

Reference context: de Mello Koch, Ives, and Stephanou, "On subgroup adapted
bases for representations of the symmetric group", arXiv:1112.4316,
DOI: 10.1088/1751-8113/45/13/135204, discuss Young-Yamanouchi to
subgroup-adapted split-basis subduction coefficients.  This module uses exact
finite-rank Young-orthogonal matrices and exact intertwiner equations; it does
not implement their large-row-difference subduction algorithm.

For the constructive finite-rank backend, Chilla, "On the linear equation
method for the subduction problem in symmetric groups", arXiv:math-ph/0512011,
introduces subduction graphs for the Young-Yamanouchi/split-basis equation
system.  Chilla, "A reduced subduction graph and higher multiplicity in S_n
transformation coefficients", arXiv:math-ph/0606037, discusses reduced graphs,
selection/identity rules, and higher-multiplicity separation.  The current
implementation uses exact subgroup-generator constraints with deterministic
component propagation and exact normalization; it should not be described as a
closed LR-tableau formula for normalized coefficient entries.

The LR-tableau enumerator in this module is used as an independent exact
finite-rank multiplicity label/check for the subgroup-adapted split
``S_a x S_b -> S_{a+b}``.  LR tableaux label multiplicity copies; the
normalized coefficient entries are constructed by the selected coefficient
backend, not by a closed LR-tableau entry formula.
"""

from functools import lru_cache
from itertools import combinations, product
from math import factorial

from ye3t._record import recordclass
from ye3t.exact_linalg import exact_matrix_equal, exact_matrix_from_entries, exact_matrix_matmul, exact_matrix_transpose
from ye3t.exact_scalars import ExactRadical, exact_scalar


def _sympy():
    from ye3t._optional_sympy import sp

    return sp


from .generalized_irreps import Partition
from .projectors import _partition_character_classes, adjacent_transposition, adjacent_transposition_representation_matrix, all_permutations, canonical_irrep_matrices, compose_permutations, inverse_permutation, permutation_cycle_type, standard_tableaux, symmetric_group_character


@recordclass(('coset_index', 'coset_rep', 'left_tableau_index', 'right_tableau_index', 'left_tableau', 'right_tableau'), frozen=True)
class YoungInducedBasisEntry:
    """One basis entry of ``Ind_{S_a x S_b}^{S_n}(S^lambda x S^mu)``."""


@recordclass(('coset_index', 'coset_rep', 'child_tableau_indices', 'child_tableaux'), frozen=True)
class YoungNaryInducedBasisEntry:
    """One basis entry of an n-ary Young-subgroup induced Specht module."""


@recordclass(('left_partition', 'right_partition', 'target_partition', 'entries', 'reading_word', 'weight'), frozen=True)
class LittlewoodRichardsonTableau:
    """One LR skew-tableau multiplicity label for ``S_a x S_b -> S_{a+b}``.

    ``entries`` is a tuple of ``(row, column, value)`` triples using one-based
    row/column coordinates in the skew shape ``target_partition/left_partition``.
    The reading word convention is row-by-row from right to left, top to bottom.
    """


@recordclass(('subgroup_partitions', 'target_partition', 'intermediate_partitions', 'tableaux', 'bracketing'), frozen=True)
class LittlewoodRichardsonChainLabel:
    """One left-associated LR-chain multiplicity label for an n-ary subgroup."""


@recordclass(('block_sizes', 'coset_reps', 'coefficients', 'provenance', 'codepath', 'notes'), frozen=True)
class TrivialSymmetricSubductionCoefficients:
    """Constructive n-ary trivial-sector coefficients for a Young subgroup."""


@recordclass(('rho', 'target_tableau_index', 'coefficients'), frozen=True)
class YoungOrthogonalCoupledVector:
    """One coupled target-tableau vector in the induced Young-orthogonal basis."""


@recordclass(('left_partition', 'right_partition', 'target_partition', 'coset_reps', 'induced_basis', 'target_tableaux', 'multiplicity', 'vectors', 'lr_tableaux', 'provenance', 'codepath', 'notes'), frozen=True)
class YoungOrthogonalCouplingTensor:
    """Finite-group Young coupling tensor in standard-tableau coordinates."""

    @property
    def induced_dim(self):
        return len(self.induced_basis)

    @property
    def target_dim(self):
        return len(self.target_tableaux)

    def coefficient_matrix(self):
        """Return columns of coupled vectors in the induced basis."""

        return _sympy().Matrix(
            [
                [vector.coefficients[row] for vector in self.vectors]
                for row in range(self.induced_dim)
            ]
        )

    def coefficient_matrix_native(self):
        """Return columns of coupled vectors as a native exact matrix."""

        return tuple(
            tuple(exact_scalar(vector.coefficients[row]) for vector in self.vectors)
            for row in range(self.induced_dim)
        )


@recordclass(('subgroup_partitions', 'target_partition', 'bracketing', 'coset_reps', 'induced_basis', 'target_tableaux', 'multiplicity', 'vectors', 'lr_chain_labels', 'coefficient_backend', 'provenance', 'codepath', 'notes'), frozen=True)
class YoungOrthogonalNarySubductionTensor:
    """N-ary subgroup-adapted Young subduction tensor in tableau coordinates."""

    @property
    def induced_dim(self):
        return len(self.induced_basis)

    @property
    def target_dim(self):
        return len(self.target_tableaux)

    def coefficient_matrix(self):
        """Return columns of coupled vectors in the n-ary induced basis."""

        return _sympy().Matrix(
            [
                [vector.coefficients[row] for vector in self.vectors]
                for row in range(self.induced_dim)
            ]
        )

    def coefficient_matrix_native(self):
        """Return columns of coupled vectors as a native exact matrix."""

        return tuple(
            tuple(exact_scalar(vector.coefficients[row]) for vector in self.vectors)
            for row in range(self.induced_dim)
        )


@recordclass(('tensor', 'expected_induced_dim', 'expected_multiplicity', 'gram_matrix', 'orthonormal', 'multiplicity_matches_character', 'generator_equivariant', 'projector_equivariant', 'projector_span_matches', 'full_projector_checked', 'passed', 'detail'), frozen=True)
class YoungOrthogonalValidationReport:
    """Validation report for one Young orthogonal coupling tensor."""


@recordclass(('max_rank', 'case_count', 'passed_count', 'multiplicity_case_count', 'reports', 'passed', 'detail'), frozen=True)
class YoungOrthogonalValidationSuiteReport:
    """Validation report for an enumerated table of Young orthogonal couplings."""


@recordclass(('tensor', 'expected_induced_dim', 'expected_multiplicity', 'gram_matrix', 'orthonormal', 'multiplicity_matches_character', 'lr_labels_match_multiplicity', 'generator_equivariant', 'passed', 'detail'), frozen=True)
class YoungOrthogonalNaryValidationReport:
    """Validation report for one n-ary Young subduction tensor."""


@recordclass(('subgroup_partitions', 'target_partition', 'bracketings', 'projectors_match', 'recoupling_orthogonal', 'passed', 'detail'), frozen=True)
class YoungOrthogonalNaryCoherenceReport:
    """Associativity/coherence report comparing n-ary bracketings."""


@recordclass(('max_rank', 'case_count', 'passed_count', 'multiplicity_case_count', 'reports', 'passed', 'detail'), frozen=True)
class YoungOrthogonalNaryCoherenceSuiteReport:
    """Validation report for enumerated n-ary subduction coherence cases."""


@recordclass(('gram_matrix', 'orthonormal', 'deterministic_convention', 'provenance', 'codepath'), frozen=True)
class ReducedMultiplicityInnerProduct:
    """Exact Frobenius inner product on restricted multiplicity intertwiners."""


def _integer_partitions(n, max_part=None):
    n = int(n)
    if max_part is None or int(max_part) > n:
        max_part = n
    if n == 0:
        return (tuple(),)
    out = []
    for first in range(int(max_part), 0, -1):
        for rest in _integer_partitions(n - first, first):
            out.append((int(first),) + tuple(int(x) for x in rest))
    return tuple(out)


def _adjacent_word_to_permutation(perm):
    perm = tuple(int(x) for x in perm)
    current = list(range(len(perm)))
    word = []
    for target_position, target_value in enumerate(perm):
        position = current.index(int(target_value))
        while position > int(target_position):
            current[position - 1], current[position] = current[position], current[position - 1]
            word.append(int(position - 1))
            position -= 1
    return tuple(word)


@lru_cache(maxsize=None)
def _young_irrep_matrix(partition_parts, perm):
    partition_parts = tuple(int(part) for part in partition_parts)
    perm = tuple(int(x) for x in perm)
    matrix = _sympy().eye(len(standard_tableaux(partition_parts)))
    for index in _adjacent_word_to_permutation(perm):
        matrix = _sympy().simplify(adjacent_transposition_representation_matrix(partition_parts, int(index)) * matrix)
    return matrix


def _young_irrep_matrices_for_perms(partition_parts, perms):
    partition_parts = tuple(int(part) for part in partition_parts)
    return {
        tuple(int(x) for x in perm): _young_irrep_matrix(partition_parts, tuple(int(x) for x in perm))
        for perm in perms
    }


def _direct_sum_permutation(left, right):
    left = tuple(int(x) for x in left)
    right = tuple(int(x) for x in right)
    offset = len(left)
    return left + tuple(offset + int(x) for x in right)


def _direct_sum_permutation_blocks(block_perms):
    out = []
    offset = 0
    for block in block_perms:
        block = tuple(int(x) for x in block)
        out.extend(offset + int(x) for x in block)
        offset += len(block)
    return tuple(out)


def young_subgroup_permutations(block_sizes):
    """Return permutations for ``S_a x S_b x ...`` embedded block-diagonally."""

    block_sizes = tuple(int(x) for x in block_sizes)
    if any(size < 0 for size in block_sizes):
        raise ValueError("block_sizes must be nonnegative.")
    if not block_sizes:
        return (tuple(),)
    out = []
    for block_perms in product(*(all_permutations(size) for size in block_sizes)):
        out.append(_direct_sum_permutation_blocks(block_perms))
    return tuple(out)


def _subgroup_permutations(a, b):
    return tuple(
        _direct_sum_permutation(left, right)
        for left in all_permutations(int(a))
        for right in all_permutations(int(b))
    )


def _left_coset_representatives(n, subgroup):
    subgroup = tuple(tuple(int(x) for x in perm) for perm in subgroup)
    remaining = set(all_permutations(int(n)))
    reps = []
    for perm in all_permutations(int(n)):
        if perm not in remaining:
            continue
        reps.append(perm)
        for h in subgroup:
            remaining.discard(compose_permutations(perm, h))
    return tuple(reps)


@lru_cache(maxsize=None)
def _young_subgroup_shuffle_representatives(block_sizes):
    """Construct canonical left-coset representatives as block shuffles."""

    block_sizes = tuple(int(value) for value in block_sizes)
    if any(value < 0 for value in block_sizes):
        raise ValueError("block_sizes must be nonnegative")
    values = tuple(range(sum(block_sizes)))

    def build(remaining, block_index):
        if block_index == len(block_sizes):
            return (tuple(),) if not remaining else tuple()
        size = int(block_sizes[block_index])
        out = []
        for selected in combinations(remaining, size):
            selected_set = set(selected)
            tail_values = tuple(
                value for value in remaining if value not in selected_set
            )
            for tail in build(tail_values, block_index + 1):
                out.append(tuple(selected) + tuple(tail))
        return tuple(out)

    return build(values, 0)


def trivial_symmetric_subduction_coefficients(block_sizes):
    """Construct n-ary trivial-sector subgroup-adapted coefficients.

    For the Young subgroup ``H = S_{a_1} x ... x S_{a_k}`` inside ``S_N``,
    this returns the normalized invariant vector in the induced permutation
    module ``Ind_H^{S_N}(1_H)``.  Coordinates are indexed by left cosets of
    ``H`` in ``S_N`` and every coefficient is ``1/sqrt([S_N:H])``.

    This is a constructive exact coefficient algorithm for the trivial
    permutation sector.  It is valid for any number of subgroup blocks, but it
    does not construct nontrivial Specht-sector subduction coefficients.
    """

    block_sizes = tuple(int(x) for x in block_sizes)
    if any(size < 0 for size in block_sizes):
        raise ValueError("block_sizes must be nonnegative.")
    n = sum(block_sizes)
    coset_reps = _young_subgroup_shuffle_representatives(block_sizes)
    coeff = _sympy().sqrt(_sympy().Rational(1, len(coset_reps))) if coset_reps else _sympy().Integer(0)
    return TrivialSymmetricSubductionCoefficients(
        block_sizes=block_sizes,
        coset_reps=coset_reps,
        coefficients=tuple(coeff for _ in coset_reps),
        provenance="constructive_orbit_sum",
        codepath="trivial_symmetric_young_subgroup_orbit_sum",
        notes=(
            "Coordinates are indexed by left cosets of the Young subgroup.",
            "The invariant vector is the normalized constant orbit sum over cosets.",
            "This is constructive for the trivial permutation sector only.",
        ),
    )


def _decompose_left_coset(product_perm, coset_reps, subgroup_set):
    for rep in coset_reps:
        h = compose_permutations(inverse_permutation(rep), product_perm)
        if h in subgroup_set:
            return rep, h
    raise RuntimeError(f"Could not decompose permutation {product_perm!r} into a selected left coset.")


def _split_subgroup_permutation(perm, a, b):
    perm = tuple(int(x) for x in perm)
    a = int(a)
    b = int(b)
    left = tuple(int(perm[idx]) for idx in range(a))
    right = tuple(int(perm[a + idx] - a) for idx in range(b))
    return left, right


def _split_young_subgroup_permutation(perm, block_sizes):
    perm = tuple(int(x) for x in perm)
    block_sizes = tuple(int(x) for x in block_sizes)
    out = []
    offset = 0
    for size in block_sizes:
        block = tuple(int(perm[offset + idx] - offset) for idx in range(int(size)))
        out.append(block)
        offset += int(size)
    return tuple(out)


def _induced_basis(coset_reps, left_tableaux, right_tableaux):
    entries = []
    for coset_index, coset_rep in enumerate(coset_reps):
        for left_idx, left_tableau in enumerate(left_tableaux):
            for right_idx, right_tableau in enumerate(right_tableaux):
                entries.append(
                    YoungInducedBasisEntry(
                        coset_index=int(coset_index),
                        coset_rep=tuple(int(x) for x in coset_rep),
                        left_tableau_index=int(left_idx),
                        right_tableau_index=int(right_idx),
                        left_tableau=left_tableau,
                        right_tableau=right_tableau,
                    )
                )
    return tuple(entries)


def _nary_induced_basis(coset_reps, child_tableaux):
    entries = []
    child_tableaux = tuple(tuple(tableaux) for tableaux in child_tableaux)
    for coset_index, coset_rep in enumerate(coset_reps):
        for child_indices in product(*(range(len(tableaux)) for tableaux in child_tableaux)):
            entries.append(
                YoungNaryInducedBasisEntry(
                    coset_index=int(coset_index),
                    coset_rep=tuple(int(x) for x in coset_rep),
                    child_tableau_indices=tuple(int(x) for x in child_indices),
                    child_tableaux=tuple(child_tableaux[idx][child_index] for idx, child_index in enumerate(child_indices)),
                )
                )
    return tuple(entries)


def _young_orthogonal_nary_subduction_shell(subgroup_partitions, target_partition, bracketing):
    subgroup_partitions = tuple(_coerce_partition(partition) for partition in subgroup_partitions)
    target_partition = _coerce_partition(target_partition)
    bracketing = str(bracketing)
    block_sizes = tuple(int(partition.size) for partition in subgroup_partitions)
    if int(target_partition.size) != int(sum(block_sizes)):
        raise ValueError("target_partition size must equal the sum of subgroup partition sizes.")
    coset_reps = _young_subgroup_shuffle_representatives(block_sizes)
    child_tableaux = tuple(standard_tableaux(tuple(partition.parts)) for partition in subgroup_partitions)
    target_tableaux = standard_tableaux(tuple(target_partition.parts))
    induced_basis = _nary_induced_basis(coset_reps, child_tableaux)
    lr_chain_labels = littlewood_richardson_chain_labels(
        subgroup_partitions,
        target_partition,
        bracketing=bracketing,
    )
    expected_multiplicity = young_nary_induced_multiplicity_by_character(subgroup_partitions, target_partition)
    if len(lr_chain_labels) != int(expected_multiplicity):
        raise RuntimeError(
            "N-ary LR-chain multiplicity does not match character multiplicity "
            f"for {[partition.parts for partition in subgroup_partitions]} -> {target_partition.parts}: "
            f"{len(lr_chain_labels)} != {expected_multiplicity}."
        )
    return {
        "subgroup_partitions": subgroup_partitions,
        "target_partition": target_partition,
        "bracketing": bracketing,
        "block_sizes": block_sizes,
        "coset_reps": coset_reps,
        "child_tableaux": child_tableaux,
        "target_tableaux": target_tableaux,
        "induced_basis": induced_basis,
        "lr_chain_labels": lr_chain_labels,
        "expected_multiplicity": int(expected_multiplicity),
    }


def _young_orthogonal_nary_subduction_tensor_from_vectors(
    shell,
    intertwiners,
    *,
    coefficient_backend,
    provenance,
    codepath,
    notes,
):
    vectors = []
    for rho, intertwiner in enumerate(intertwiners):
        target_dim = int(shell["target_partition"].dimension)
        for target_idx in range(target_dim):
            coupled = _sympy().Matrix(intertwiner[:, int(target_idx)])
            vectors.append(
                YoungOrthogonalCoupledVector(
                    rho=int(rho),
                    target_tableau_index=int(target_idx),
                    coefficients=tuple(_sympy().simplify(value) for value in coupled),
                )
            )
    return YoungOrthogonalNarySubductionTensor(
        subgroup_partitions=shell["subgroup_partitions"],
        target_partition=shell["target_partition"],
        bracketing=shell["bracketing"],
        coset_reps=shell["coset_reps"],
        induced_basis=shell["induced_basis"],
        target_tableaux=shell["target_tableaux"],
        multiplicity=len(intertwiners),
        vectors=tuple(vectors),
        lr_chain_labels=shell["lr_chain_labels"],
        coefficient_backend=str(coefficient_backend),
        provenance=str(provenance),
        codepath=str(codepath),
        notes=tuple(notes),
    )


def _kronecker_product_sequence(matrices):
    matrices = tuple(matrices)
    if not matrices:
        return _sympy().ones(1, 1)
    out = matrices[0]
    for matrix in matrices[1:]:
        out = _sympy().kronecker_product(out, matrix)
    return out


def _induced_action_matrix(
    group_perm,
    *,
    a,
    b,
    coset_reps,
    subgroup_set,
    rep_index,
    left_matrices,
    right_matrices,
    left_dim,
    right_dim,
):
    coset_count = len(coset_reps)
    child_dim = int(left_dim) * int(right_dim)
    out = _sympy().zeros(coset_count * child_dim, coset_count * child_dim)
    rep_position = {rep: idx for idx, rep in enumerate(coset_reps)}
    for col_coset, rep in enumerate(coset_reps):
        product_perm = compose_permutations(group_perm, rep)
        row_rep, h = _decompose_left_coset(product_perm, coset_reps, subgroup_set)
        row_coset = rep_position[row_rep]
        h_left, h_right = _split_subgroup_permutation(h, int(a), int(b))
        child_action = _sympy().kronecker_product(
            left_matrices[inverse_permutation(h_left)],
            right_matrices[inverse_permutation(h_right)],
        )
        row0 = int(row_coset) * child_dim
        col0 = int(col_coset) * child_dim
        for row in range(child_dim):
            for col in range(child_dim):
                out[row0 + row, col0 + col] = child_action[row, col]
    return _sympy().simplify(out)


def _nary_induced_action_matrix(
    group_perm,
    *,
    block_sizes,
    coset_reps,
    subgroup_set,
    child_matrices,
    child_dims,
):
    child_dim = 1
    for dim in child_dims:
        child_dim *= int(dim)
    out = _sympy().zeros(len(coset_reps) * child_dim, len(coset_reps) * child_dim)
    rep_position = {rep: idx for idx, rep in enumerate(coset_reps)}
    for col_coset, rep in enumerate(coset_reps):
        product_perm = compose_permutations(group_perm, rep)
        row_rep, h = _decompose_left_coset(product_perm, coset_reps, subgroup_set)
        row_coset = rep_position[row_rep]
        h_blocks = _split_young_subgroup_permutation(h, block_sizes)
        child_action = _kronecker_product_sequence(
            child_matrices[idx][inverse_permutation(block_perm)]
            for idx, block_perm in enumerate(h_blocks)
        )
        row0 = int(row_coset) * child_dim
        col0 = int(col_coset) * child_dim
        for row in range(child_dim):
            for col in range(child_dim):
                out[row0 + row, col0 + col] = child_action[row, col]
    return _sympy().simplify(out)


def _matrix_units_on_induced(
    target_partition,
    induced_actions,
):
    target_matrices = canonical_irrep_matrices(tuple(target_partition.parts))
    target_dim = int(target_partition.dimension)
    group_order = len(target_matrices)
    space_dim = next(iter(induced_actions.values())).rows if induced_actions else 0
    units = {}
    for a_idx in range(target_dim):
        for b_idx in range(target_dim):
            operator = _sympy().zeros(space_dim, space_dim)
            for perm, action in induced_actions.items():
                coeff = target_matrices[inverse_permutation(perm)][b_idx, a_idx]
                if coeff == 0:
                    continue
                operator += _sympy().simplify((_sympy().Integer(target_dim) * coeff / _sympy().Integer(group_order)) * action)
            units[(int(a_idx), int(b_idx))] = _sympy().simplify(operator)
    return units


def _induced_action_matrices_for_coupling(coupling_tensor):
    a = int(coupling_tensor.left_partition.size)
    b = int(coupling_tensor.right_partition.size)
    n = a + b
    subgroup = _subgroup_permutations(a, b)
    subgroup_set = set(subgroup)
    left_matrices = canonical_irrep_matrices(tuple(coupling_tensor.left_partition.parts))
    right_matrices = canonical_irrep_matrices(tuple(coupling_tensor.right_partition.parts))
    rep_index = {rep: idx for idx, rep in enumerate(coupling_tensor.coset_reps)}
    return {
        perm: _induced_action_matrix(
            perm,
            a=a,
            b=b,
            coset_reps=coupling_tensor.coset_reps,
            subgroup_set=subgroup_set,
            rep_index=rep_index,
            left_matrices=left_matrices,
            right_matrices=right_matrices,
            left_dim=int(coupling_tensor.left_partition.dimension),
            right_dim=int(coupling_tensor.right_partition.dimension),
        )
        for perm in all_permutations(n)
    }


def young_induced_multiplicity_by_character(left_partition, right_partition, target_partition):
    """Return the exact character inner-product multiplicity for one induction."""

    if not isinstance(left_partition, Partition):
        left_partition = Partition(tuple(int(x) for x in left_partition))
    if not isinstance(right_partition, Partition):
        right_partition = Partition(tuple(int(x) for x in right_partition))
    if not isinstance(target_partition, Partition):
        target_partition = Partition(tuple(int(x) for x in target_partition))
    a = int(left_partition.size)
    b = int(right_partition.size)
    if int(target_partition.size) != a + b:
        return 0
    return young_nary_induced_multiplicity_by_character(
        (left_partition, right_partition), target_partition
    )


def young_nary_induced_multiplicity_by_character(subgroup_partitions, target_partition):
    """Return the exact character multiplicity for an n-ary Young subgroup."""

    return _young_nary_induced_multiplicity_by_character_cached(
        tuple(tuple(_coerce_partition(partition).parts) for partition in subgroup_partitions),
        tuple(_coerce_partition(target_partition).parts),
    )


@lru_cache(maxsize=8192)
def _young_nary_induced_multiplicity_by_character_cached(subgroup_parts, target_parts):
    """Evaluate one normalized Young-subgroup character inner product."""

    subgroup_partitions = tuple(Partition(parts) for parts in subgroup_parts)
    target_partition = Partition(target_parts)
    block_sizes = tuple(int(partition.size) for partition in subgroup_partitions)
    if sum(block_sizes) != int(target_partition.size):
        return 0
    total = 0
    class_tables = tuple(
        _partition_character_classes(tuple(partition.parts))
        for partition in subgroup_partitions
    )
    for block_classes in product(*class_tables):
        merged_cycle_type = tuple(
            sorted(
                (
                    int(length)
                    for cycle_type, _class_size, _character in block_classes
                    for length in cycle_type
                ),
                reverse=True,
            )
        )
        contribution = int(
            symmetric_group_character(
                target_partition, merged_cycle_type
            )
        )
        for _cycle_type, class_size, character in block_classes:
            contribution *= int(class_size) * int(character)
        total += int(contribution)
    subgroup_order = 1
    for size in block_sizes:
        subgroup_order *= int(factorial(int(size)))
    if total % subgroup_order != 0:
        raise ArithmeticError("n-ary Young induction character product did not produce an integer multiplicity")
    return int(total // subgroup_order)


def _coerce_partition(partition):
    if isinstance(partition, Partition):
        return partition
    return Partition(tuple(int(x) for x in partition))


def _partition_contains(outer, inner):
    outer = _coerce_partition(outer)
    inner = _coerce_partition(inner)
    for row in range(max(len(outer.parts), len(inner.parts))):
        outer_len = int(outer.parts[row]) if row < len(outer.parts) else 0
        inner_len = int(inner.parts[row]) if row < len(inner.parts) else 0
        if inner_len > outer_len:
            return False
    return True


def _skew_shape_cells(inner, outer):
    inner = _coerce_partition(inner)
    outer = _coerce_partition(outer)
    if not _partition_contains(outer, inner):
        return tuple()
    cells = []
    for row, outer_len in enumerate(outer.parts, start=1):
        inner_len = int(inner.parts[row - 1]) if row - 1 < len(inner.parts) else 0
        for col in range(inner_len + 1, int(outer_len) + 1):
            cells.append((int(row), int(col)))
    return tuple(cells)


def _is_lattice_word(word, max_entry):
    counts = [0 for _ in range(int(max_entry) + 2)]
    for value in word:
        value = int(value)
        counts[value] += 1
        for entry in range(1, int(max_entry) + 1):
            if counts[entry] < counts[entry + 1]:
                return False
    return True


@lru_cache(maxsize=None)
def _littlewood_richardson_tableaux_cached(left_parts, right_parts, target_parts):
    left_partition = Partition(tuple(int(x) for x in left_parts))
    right_partition = Partition(tuple(int(x) for x in right_parts))
    target_partition = Partition(tuple(int(x) for x in target_parts))
    if int(left_partition.size) + int(right_partition.size) != int(target_partition.size):
        return tuple()
    if not _partition_contains(target_partition, left_partition):
        return tuple()

    cells = _skew_shape_cells(left_partition, target_partition)
    if len(cells) != int(right_partition.size):
        return tuple()
    weight = tuple(int(x) for x in right_partition.parts)
    max_entry = len(weight)
    remaining0 = {entry + 1: int(count) for entry, count in enumerate(weight)}
    cell_set = set(cells)
    assignments = {}
    out = []

    def can_place(row, col, value):
        left_cell = (int(row), int(col) - 1)
        if left_cell in cell_set and left_cell in assignments and int(assignments[left_cell]) > int(value):
            return False
        above_cell = (int(row) - 1, int(col))
        if above_cell in cell_set and above_cell in assignments and int(assignments[above_cell]) >= int(value):
            return False
        return True

    def reading_word_from_assignments():
        word = []
        rows = sorted({row for row, _ in cells})
        for row in rows:
            row_cells = sorted((col for r, col in cells if r == row), reverse=True)
            for col in row_cells:
                word.append(int(assignments[(row, col)]))
        return tuple(word)

    def backtrack(cell_index, remaining):
        if int(cell_index) == len(cells):
            word = reading_word_from_assignments()
            if not _is_lattice_word(word, max_entry):
                return
            entries = tuple(
                (int(row), int(col), int(assignments[(row, col)]))
                for row, col in cells
            )
            out.append(
                LittlewoodRichardsonTableau(
                    left_partition=left_partition,
                    right_partition=right_partition,
                    target_partition=target_partition,
                    entries=entries,
                    reading_word=word,
                    weight=weight,
                )
            )
            return
        row, col = cells[int(cell_index)]
        for value in range(1, max_entry + 1):
            if int(remaining[value]) <= 0:
                continue
            if not can_place(row, col, value):
                continue
            assignments[(row, col)] = int(value)
            next_remaining = dict(remaining)
            next_remaining[value] = int(next_remaining[value]) - 1
            backtrack(int(cell_index) + 1, next_remaining)
            del assignments[(row, col)]

    backtrack(0, remaining0)
    return tuple(out)


def littlewood_richardson_tableaux(left_partition, right_partition, target_partition):
    """Enumerate LR tableaux for ``c^target_{left,right}``.

    This is an exact finite combinatorial enumerator for small ranks.  It uses
    semistandard skew tableaux of shape ``target/left`` and content ``right``
    with the lattice-word convention described by
    :class:`LittlewoodRichardsonTableau`.  The function is intended as an
    independent multiplicity-label route; normalized subduction coefficients
    are not derived from these tableaux by this function.
    """

    left_partition = _coerce_partition(left_partition)
    right_partition = _coerce_partition(right_partition)
    target_partition = _coerce_partition(target_partition)
    return _littlewood_richardson_tableaux_cached(
        tuple(left_partition.parts),
        tuple(right_partition.parts),
        tuple(target_partition.parts),
    )


def littlewood_richardson_coefficient(left_partition, right_partition, target_partition):
    """Return ``len(littlewood_richardson_tableaux(...))`` for one merge."""

    return len(littlewood_richardson_tableaux(left_partition, right_partition, target_partition))


@lru_cache(maxsize=None)
def _littlewood_richardson_chain_labels_cached(subgroup_parts, target_parts):
    subgroup_partitions = tuple(Partition(tuple(int(x) for x in parts)) for parts in subgroup_parts)
    target_partition = Partition(tuple(int(x) for x in target_parts))
    if not subgroup_partitions:
        if int(target_partition.size) == 0:
            return (
                LittlewoodRichardsonChainLabel(
                    subgroup_partitions=tuple(),
                    target_partition=target_partition,
                    intermediate_partitions=tuple(),
                    tableaux=tuple(),
                    bracketing="left",
                ),
            )
        return tuple()
    if sum(int(partition.size) for partition in subgroup_partitions) != int(target_partition.size):
        return tuple()
    if len(subgroup_partitions) == 1:
        if subgroup_partitions[0] != target_partition:
            return tuple()
        return (
            LittlewoodRichardsonChainLabel(
                subgroup_partitions=subgroup_partitions,
                target_partition=target_partition,
                intermediate_partitions=(target_partition,),
                tableaux=tuple(),
                bracketing="left",
            ),
        )

    prefix = subgroup_partitions[:-1]
    last = subgroup_partitions[-1]
    prefix_size = sum(int(partition.size) for partition in prefix)
    labels = []
    for intermediate_parts in _integer_partitions(prefix_size):
        intermediate = Partition(intermediate_parts)
        prefix_labels = _littlewood_richardson_chain_labels_cached(
            tuple(tuple(int(x) for x in partition.parts) for partition in prefix),
            tuple(int(x) for x in intermediate.parts),
        )
        if not prefix_labels:
            continue
        tableaux = littlewood_richardson_tableaux(intermediate, last, target_partition)
        for prefix_label in prefix_labels:
            for tableau in tableaux:
                labels.append(
                    LittlewoodRichardsonChainLabel(
                        subgroup_partitions=subgroup_partitions,
                        target_partition=target_partition,
                        intermediate_partitions=tuple(prefix_label.intermediate_partitions) + (target_partition,),
                        tableaux=tuple(prefix_label.tableaux) + (tableau,),
                        bracketing="left",
                    )
                )
    return tuple(labels)


@lru_cache(maxsize=None)
def _littlewood_richardson_tree_labels_cached(subgroup_parts, target_parts, bracketing):
    subgroup_partitions = tuple(Partition(tuple(int(x) for x in parts)) for parts in subgroup_parts)
    target_partition = Partition(tuple(int(x) for x in target_parts))
    bracketing = str(bracketing)
    if bracketing == "left":
        return _littlewood_richardson_chain_labels_cached(subgroup_parts, target_parts)
    if not subgroup_partitions:
        if int(target_partition.size) == 0:
            return (
                LittlewoodRichardsonChainLabel(
                    subgroup_partitions=tuple(),
                    target_partition=target_partition,
                    intermediate_partitions=tuple(),
                    tableaux=tuple(),
                    bracketing=bracketing,
                ),
            )
        return tuple()
    if sum(int(partition.size) for partition in subgroup_partitions) != int(target_partition.size):
        return tuple()
    if len(subgroup_partitions) == 1:
        if subgroup_partitions[0] != target_partition:
            return tuple()
        return (
            LittlewoodRichardsonChainLabel(
                subgroup_partitions=subgroup_partitions,
                target_partition=target_partition,
                intermediate_partitions=(target_partition,),
                tableaux=tuple(),
                bracketing=bracketing,
            ),
        )
    if bracketing == "right":
        split = 1
    elif bracketing == "balanced":
        split = len(subgroup_partitions) // 2
    else:
        raise ValueError("bracketing must be 'balanced', 'left', or 'right'.")

    left_group = subgroup_partitions[:split]
    right_group = subgroup_partitions[split:]
    left_size = sum(int(partition.size) for partition in left_group)
    right_size = sum(int(partition.size) for partition in right_group)
    labels = []
    for left_parts in _integer_partitions(left_size):
        left_target = Partition(left_parts)
        left_labels = _littlewood_richardson_tree_labels_cached(
            tuple(tuple(int(x) for x in partition.parts) for partition in left_group),
            tuple(int(x) for x in left_target.parts),
            bracketing,
        )
        if not left_labels:
            continue
        for right_parts in _integer_partitions(right_size):
            right_target = Partition(right_parts)
            right_labels = _littlewood_richardson_tree_labels_cached(
                tuple(tuple(int(x) for x in partition.parts) for partition in right_group),
                tuple(int(x) for x in right_target.parts),
                bracketing,
            )
            if not right_labels:
                continue
            tableaux = littlewood_richardson_tableaux(left_target, right_target, target_partition)
            for left_label in left_labels:
                for right_label in right_labels:
                    for tableau in tableaux:
                        labels.append(
                            LittlewoodRichardsonChainLabel(
                                subgroup_partitions=subgroup_partitions,
                                target_partition=target_partition,
                                intermediate_partitions=(
                                    tuple(left_label.intermediate_partitions)
                                    + tuple(right_label.intermediate_partitions)
                                    + (target_partition,)
                                ),
                                tableaux=tuple(left_label.tableaux) + tuple(right_label.tableaux) + (tableau,),
                                bracketing=bracketing,
                            )
                        )
    return tuple(labels)


def littlewood_richardson_chain_labels(subgroup_partitions, target_partition, *, bracketing="left"):
    """Enumerate LR-chain labels for ``S_{a1} x ... x S_{ak} -> S_N``.

    Supported bracketings are ``"left"``, ``"right"``, and ``"balanced"``.
    Each internal merge is labeled by an ordinary LR tableau.  The number of
    returned labels is the corresponding n-ary Littlewood-Richardson
    multiplicity.  These labels are attached to n-ary subduction coefficient
    tensors, but the nontrivial normalized matrices are currently obtained
    from exact Young-orthogonal intertwiner equations.
    """

    subgroup_partitions = tuple(_coerce_partition(partition) for partition in subgroup_partitions)
    target_partition = _coerce_partition(target_partition)
    bracketing = str(bracketing)
    if bracketing not in {"balanced", "left", "right"}:
        raise ValueError("bracketing must be 'balanced', 'left', or 'right'.")
    return _littlewood_richardson_tree_labels_cached(
        tuple(tuple(int(x) for x in partition.parts) for partition in subgroup_partitions),
        tuple(int(x) for x in target_partition.parts),
        bracketing,
    )


def littlewood_richardson_chain_coefficient(subgroup_partitions, target_partition, *, bracketing="left"):
    """Return the n-ary LR multiplicity for the selected bracketing."""

    return len(littlewood_richardson_chain_labels(subgroup_partitions, target_partition, bracketing=bracketing))


def _pivot_normalize_vector(vector):
    entries = [_sympy().simplify(value) for value in vector]
    pivot = next((idx for idx, value in enumerate(entries) if value != 0), len(entries))
    if pivot == len(entries):
        return _sympy().Matrix(vector)
    scale = entries[pivot]
    return _sympy().Matrix([_sympy().simplify(value / scale) for value in entries])


def _normalized_vector_key(vector):
    entries = [_sympy().simplify(value) for value in vector]
    pivot = next((idx for idx, value in enumerate(entries) if value != 0), len(entries))
    if pivot == len(entries):
        return pivot, tuple()
    scale = entries[pivot]
    return pivot, tuple(_sympy().srepr(_sympy().simplify(value / scale)) for value in entries)


def _exact_pivot_basis(vectors):
    basis = []
    seen = set()
    for vector in vectors:
        normalized = _pivot_normalize_vector(_sympy().Matrix(vector))
        key = _normalized_vector_key(normalized)
        if not key[1] or key in seen:
            continue
        basis.append(normalized)
        seen.add(key)
    return tuple(basis)


def _normalize_to_unit(vector):
    norm_sq = _sympy().simplify((_sympy().Matrix(vector).T * _sympy().Matrix(vector))[0, 0])
    if norm_sq == 0:
        return _sympy().Matrix(vector)
    return _sympy().Matrix([_sympy().simplify(value / _sympy().sqrt(norm_sq)) for value in vector])


def _matrix_from_column_major_vector(vector, rows, cols):
    return _sympy().Matrix(
        int(rows),
        int(cols),
        lambda row, col: vector[int(row) + int(rows) * int(col)],
    )


def _canonicalize_intertwiner(matrix):
    target_dim = int(matrix.cols)
    scale_sq = _sympy().simplify(_sympy().trace(matrix.T * matrix) / _sympy().Integer(target_dim))
    if scale_sq == 0:
        return matrix
    normalized = _sympy().simplify(matrix / _sympy().sqrt(scale_sq))
    for value in normalized:
        value = _sympy().simplify(value)
        if value == 0:
            continue
        if value.could_extract_minus_sign():
            normalized = _sympy().simplify(-normalized)
        break
    return normalized


def _orthonormalize_intertwiner_metric(matrices):
    if not matrices:
        return tuple()
    target_dim = int(matrices[0].cols)
    metric = _sympy().Matrix(
        [
            [
                _sympy().simplify(_sympy().trace(left.T * right) / _sympy().Integer(target_dim))
                for right in matrices
            ]
            for left in matrices
        ]
    )
    if metric == _sympy().eye(metric.rows):
        return tuple(_sympy().simplify(matrix) for matrix in matrices)
    factor = _sympy().simplify(metric.cholesky())
    transform = _sympy().simplify(factor.T.inv())
    out = []
    for col in range(transform.cols):
        matrix = _sympy().zeros(matrices[0].rows, matrices[0].cols)
        for row, raw_matrix in enumerate(matrices):
            coeff = transform[int(row), int(col)]
            if coeff != 0:
                matrix += coeff * raw_matrix
        out.append(_canonicalize_intertwiner(_sympy().simplify(matrix)))
    return tuple(out)


def _exact_intertwiner_basis(induced_actions, target_partition, induced_dim):
    target_dim = int(target_partition.dimension)
    constraints = []
    for idx in range(max(int(target_partition.size) - 1, 0)):
        perm = adjacent_transposition(int(target_partition.size), idx)
        action = induced_actions[perm]
        target_action = _young_irrep_matrix(tuple(target_partition.parts), inverse_permutation(perm))
        constraints.append(
            _sympy().kronecker_product(_sympy().eye(target_dim), action)
            - _sympy().kronecker_product(target_action.T, _sympy().eye(int(induced_dim)))
        )
    if not constraints:
        return (_sympy().ones(1, 1),)
    system = constraints[0]
    for block in constraints[1:]:
        system = system.col_join(block)
    basis = []
    seen = set()
    for vector in system.nullspace():
        matrix = _canonicalize_intertwiner(_matrix_from_column_major_vector(vector, int(induced_dim), target_dim))
        key = tuple(_sympy().srepr(_sympy().simplify(value)) for value in matrix)
        if key in seen:
            continue
        basis.append(matrix)
        seen.add(key)
    return _orthonormalize_intertwiner_metric(tuple(basis))


def _subgroup_generator_permutations(a, b):
    a = int(a)
    b = int(b)
    out = []
    for idx in range(max(a - 1, 0)):
        out.append(_direct_sum_permutation(adjacent_transposition(a, idx), tuple(range(b))))
    for idx in range(max(b - 1, 0)):
        out.append(_direct_sum_permutation(tuple(range(a)), adjacent_transposition(b, idx)))
    return tuple(out)


def _young_subgroup_generator_permutations(block_sizes):
    block_sizes = tuple(int(x) for x in block_sizes)
    out = []
    for block_index, size in enumerate(block_sizes):
        prefix = sum(block_sizes[:block_index])
        for idx in range(max(int(size) - 1, 0)):
            blocks = [tuple(range(block_size)) for block_size in block_sizes]
            blocks[block_index] = adjacent_transposition(int(size), idx)
            out.append(_direct_sum_permutation_blocks(blocks))
    return tuple(out)


def _exact_restricted_intertwiner_basis(
    left_partition,
    right_partition,
    target_partition,
    coset_reps,
):
    a = int(left_partition.size)
    b = int(right_partition.size)
    child_dim = int(left_partition.dimension) * int(right_partition.dimension)
    target_dim = int(target_partition.dimension)
    left_matrices = canonical_irrep_matrices(tuple(left_partition.parts))
    right_matrices = canonical_irrep_matrices(tuple(right_partition.parts))
    constraints = []
    for perm in _subgroup_generator_permutations(a, b):
        left_perm, right_perm = _split_subgroup_permutation(perm, a, b)
        child_action = _sympy().kronecker_product(
            left_matrices[inverse_permutation(left_perm)],
            right_matrices[inverse_permutation(right_perm)],
        )
        target_action = _young_irrep_matrix(tuple(target_partition.parts), inverse_permutation(perm))
        constraints.append(
            _sympy().kronecker_product(_sympy().eye(target_dim), child_action)
            - _sympy().kronecker_product(target_action.T, _sympy().eye(child_dim))
        )
    if constraints:
        system = constraints[0]
        for block in constraints[1:]:
            system = system.col_join(block)
        raw_blocks = tuple(
            _canonicalize_intertwiner(_matrix_from_column_major_vector(vector, child_dim, target_dim))
            for vector in system.nullspace()
        )
    else:
        raw_blocks = tuple(
            _sympy().Matrix(
                child_dim,
                target_dim,
                lambda row, col, basis_row=basis_row, basis_col=basis_col: (
                    _sympy().Integer(1) if int(row) == int(basis_row) and int(col) == int(basis_col) else _sympy().Integer(0)
                ),
            )
            for basis_row in range(child_dim)
            for basis_col in range(target_dim)
        )
    expanded = []
    for block in raw_blocks:
        matrix = _sympy().zeros(len(coset_reps) * child_dim, target_dim)
        for coset_index, coset_rep in enumerate(coset_reps):
            target_action = _young_irrep_matrix(tuple(target_partition.parts), coset_rep)
            piece = _sympy().simplify(block * target_action)
            row0 = int(coset_index) * child_dim
            for row in range(child_dim):
                for col in range(target_dim):
                    matrix[row0 + row, col] = piece[row, col]
        expanded.append(_canonicalize_intertwiner(matrix))
    return _orthonormalize_intertwiner_metric(tuple(expanded))


def _exact_nary_restricted_intertwiner_basis(
    subgroup_partitions,
    target_partition,
    coset_reps,
):
    subgroup_partitions = tuple(_coerce_partition(partition) for partition in subgroup_partitions)
    target_partition = _coerce_partition(target_partition)
    block_sizes = tuple(int(partition.size) for partition in subgroup_partitions)
    child_dims = tuple(int(partition.dimension) for partition in subgroup_partitions)
    child_dim = 1
    for dim in child_dims:
        child_dim *= int(dim)
    target_dim = int(target_partition.dimension)
    child_matrices = tuple(canonical_irrep_matrices(tuple(partition.parts)) for partition in subgroup_partitions)
    constraints = []
    for perm in _young_subgroup_generator_permutations(block_sizes):
        block_perms = _split_young_subgroup_permutation(perm, block_sizes)
        child_action = _kronecker_product_sequence(
            child_matrices[idx][inverse_permutation(block_perm)]
            for idx, block_perm in enumerate(block_perms)
        )
        target_action = _young_irrep_matrix(tuple(target_partition.parts), inverse_permutation(perm))
        constraints.append(
            _sympy().kronecker_product(_sympy().eye(target_dim), child_action)
            - _sympy().kronecker_product(target_action.T, _sympy().eye(child_dim))
        )
    raw_blocks = _exact_restricted_intertwiner_basis_from_constraints(
        constraints,
        child_dim,
        target_dim,
    )
    return _expand_nary_restricted_intertwiners(
        raw_blocks,
        target_partition,
        coset_reps,
    )


def _exact_restricted_intertwiner_basis_from_constraints(constraints, child_dim, target_dim):
    if constraints:
        system = constraints[0]
        for block in constraints[1:]:
            system = system.col_join(block)
        raw_blocks = tuple(
            _canonicalize_intertwiner(_matrix_from_column_major_vector(vector, child_dim, target_dim))
            for vector in system.nullspace()
        )
    else:
        raw_blocks = tuple(
            _sympy().Matrix(
                child_dim,
                target_dim,
                lambda row, col, basis_row=basis_row, basis_col=basis_col: (
                    _sympy().Integer(1) if int(row) == int(basis_row) and int(col) == int(basis_col) else _sympy().Integer(0)
                ),
            )
            for basis_row in range(child_dim)
            for basis_col in range(target_dim)
        )
    return _orthonormalize_intertwiner_metric(tuple(raw_blocks))


def _sparse_equations_from_constraint_matrices(constraints):
    equations = []
    for matrix in constraints:
        for row in range(matrix.rows):
            equation = {
                int(col): _sympy().simplify(matrix[int(row), int(col)])
                for col in range(matrix.cols)
                if _sympy().simplify(matrix[int(row), int(col)]) != 0
            }
            if equation:
                equations.append(equation)
    return tuple(equations)


def _propagate_subduction_graph_solution(equations, variable_count, seed_variable, zero_variables):
    known = {int(variable): _sympy().Integer(0) for variable in zero_variables}
    seed_variable = int(seed_variable)
    if known.get(seed_variable, _sympy().Integer(1)) == 0:
        return None
    known[seed_variable] = _sympy().Integer(1)
    changed = True
    while changed:
        changed = False
        for equation in equations:
            known_sum = _sympy().Integer(0)
            unknowns = []
            for variable, coeff in equation.items():
                if int(variable) in known:
                    known_sum += _sympy().simplify(coeff * known[int(variable)])
                else:
                    unknowns.append((int(variable), coeff))
            if not unknowns:
                if _sympy().simplify(known_sum) != 0:
                    return None
                continue
            if len(unknowns) == 1:
                variable, coeff = unknowns[0]
                value = _sympy().simplify(-known_sum / coeff)
                if variable in known and _sympy().simplify(known[variable] - value) != 0:
                    return None
                if variable not in known:
                    known[variable] = value
                    changed = True
    if len(known) != int(variable_count):
        return None
    vector = _sympy().Matrix([_sympy().simplify(known[idx]) for idx in range(int(variable_count))])
    for equation in equations:
        residual = _sympy().simplify(sum(coeff * vector[int(variable)] for variable, coeff in equation.items()))
        if residual != 0:
            return None
    return vector


def _iterated_forced_zero_variables(equations):
    forced = set()
    changed = True
    while changed:
        changed = False
        for equation in equations:
            remaining = [
                (int(variable), _sympy().simplify(coeff))
                for variable, coeff in equation.items()
                if int(variable) not in forced and _sympy().simplify(coeff) != 0
            ]
            if len(remaining) == 1:
                variable, _coeff = remaining[0]
                forced.add(int(variable))
                changed = True
    return frozenset(forced)


def _connected_components_from_equations(equations, variable_count, forced_zero_variables):
    active = set(range(int(variable_count))) - set(forced_zero_variables)
    adjacency = {int(variable): set() for variable in active}
    for equation in equations:
        variables = [int(variable) for variable in equation if int(variable) in active]
        for left in variables:
            for right in variables:
                if left != right:
                    adjacency[left].add(right)
    components = []
    seen = set()
    for variable in sorted(active):
        if variable in seen:
            continue
        stack = [int(variable)]
        seen.add(int(variable))
        component = []
        while stack:
            current = stack.pop()
            component.append(int(current))
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor not in seen:
                    seen.add(int(neighbor))
                    stack.append(int(neighbor))
        components.append(tuple(sorted(component)))
    return tuple(components)


def _zero_expression(length):
    return tuple(_sympy().Integer(0) for _ in range(int(length)))


def _scale_expression(expression, scale):
    return tuple(_sympy().simplify(scale * value) for value in expression)


def _add_expressions(left, right):
    return tuple(_sympy().simplify(a + b) for a, b in zip(left, right))


def _equation_expression(equation, expressions, seed_count):
    value = _zero_expression(seed_count)
    unknowns = []
    for variable, coeff in equation.items():
        variable = int(variable)
        coeff = _sympy().simplify(coeff)
        if variable in expressions:
            value = _add_expressions(value, _scale_expression(expressions[variable], coeff))
        else:
            unknowns.append((variable, coeff))
    return value, unknowns


def _nonzero_expression(expression):
    return any(_sympy().simplify(value) != 0 for value in expression)


def _expression_key(expression):
    return tuple(_sympy().srepr(_sympy().simplify(value)) for value in expression)


def _component_solution_vectors_by_propagation(component, local_equations, variable_count):
    component = tuple(int(variable) for variable in component)
    expressions = {}
    seed_count = 0
    constraints = []
    seen_constraints = set()

    def add_seed(variable):
        nonlocal seed_count
        seed_count += 1
        expressions.update(
            {
                key: tuple(values) + (_sympy().Integer(0),)
                for key, values in expressions.items()
            }
        )
        constraints[:] = [tuple(values) + (_sympy().Integer(0),) for values in constraints]
        expressions[int(variable)] = tuple(
            _sympy().Integer(1) if idx == seed_count - 1 else _sympy().Integer(0)
            for idx in range(seed_count)
        )

    while len(expressions) < len(component):
        changed = True
        while changed:
            changed = False
            for equation in local_equations:
                known_value, unknowns = _equation_expression(equation, expressions, seed_count)
                if not unknowns:
                    if _nonzero_expression(known_value):
                        key = _expression_key(known_value)
                        if key not in seen_constraints:
                            constraints.append(known_value)
                            seen_constraints.add(key)
                    continue
                if len(unknowns) == 1:
                    variable, coeff = unknowns[0]
                    expression = _scale_expression(known_value, -_sympy().Integer(1) / coeff)
                    if variable in expressions:
                        if _nonzero_expression(_add_expressions(expressions[variable], _scale_expression(expression, -1))):
                            raise RuntimeError("Subduction graph propagation produced inconsistent expressions.")
                    else:
                        expressions[variable] = expression
                        changed = True
        unresolved = [variable for variable in component if variable not in expressions]
        if unresolved:
            add_seed(unresolved[0])

    for equation in local_equations:
        known_value, unknowns = _equation_expression(equation, expressions, seed_count)
        if unknowns:
            raise RuntimeError("Subduction graph propagation left unresolved component variables.")
        if _nonzero_expression(known_value):
            key = _expression_key(known_value)
            if key not in seen_constraints:
                constraints.append(known_value)
                seen_constraints.add(key)

    if seed_count == 0:
        return tuple()
    if constraints:
        seed_equations = tuple(
            {
                int(idx): _sympy().simplify(coeff)
                for idx, coeff in enumerate(constraint)
                if _sympy().simplify(coeff) != 0
            }
            for constraint in constraints
            if _nonzero_expression(constraint)
        )
        seed_vectors = _subduction_graph_solution_vectors(seed_equations, seed_count)
    else:
        seed_vectors = tuple(
            _sympy().Matrix([
                _sympy().Integer(1) if row == col else _sympy().Integer(0)
                for row in range(seed_count)
            ])
            for col in range(seed_count)
        )

    out = []
    for seed_vector in seed_vectors:
        full = [_sympy().Integer(0) for _ in range(int(variable_count))]
        for variable, expression in expressions.items():
            full[int(variable)] = _sympy().simplify(
                sum(expression[idx] * seed_vector[int(idx)] for idx in range(seed_count))
            )
        out.append(_sympy().Matrix(full))
    return tuple(out)


def _subduction_graph_solution_vectors(equations, variable_count):
    equations = tuple(
        {
            int(variable): _sympy().simplify(coeff)
            for variable, coeff in equation.items()
            if _sympy().simplify(coeff) != 0
        }
        for equation in equations
        if equation
    )
    forced_zero_variables = _iterated_forced_zero_variables(equations)
    components = _connected_components_from_equations(equations, variable_count, forced_zero_variables)
    raw_vectors = []
    for component in components:
        component_set = set(component)
        local_equations = []
        for equation in equations:
            local = {
                int(variable): _sympy().simplify(coeff)
                for variable, coeff in equation.items()
                if int(variable) in component_set
            }
            if local:
                local_equations.append(local)
        raw_vectors.extend(
            _component_solution_vectors_by_propagation(
                component,
                tuple(local_equations),
                variable_count,
            )
        )
    basis = _exact_pivot_basis(raw_vectors)
    for vector in basis:
        for equation in equations:
            residual = _sympy().simplify(sum(coeff * vector[int(variable)] for variable, coeff in equation.items()))
            if residual != 0:
                raise RuntimeError("Subduction graph propagation produced a vector that violates a constraint.")
    return basis


def _constructive_subduction_graph_restricted_intertwiner_basis(
    subgroup_partitions,
    target_partition,
    *,
    expected_multiplicity,
):
    subgroup_partitions = tuple(_coerce_partition(partition) for partition in subgroup_partitions)
    target_partition = _coerce_partition(target_partition)
    block_sizes = tuple(int(partition.size) for partition in subgroup_partitions)
    child_dims = tuple(int(partition.dimension) for partition in subgroup_partitions)
    child_dim = 1
    for dim in child_dims:
        child_dim *= int(dim)
    target_dim = int(target_partition.dimension)
    child_matrices = tuple(canonical_irrep_matrices(tuple(partition.parts)) for partition in subgroup_partitions)
    constraints = []
    for perm in _young_subgroup_generator_permutations(block_sizes):
        block_perms = _split_young_subgroup_permutation(perm, block_sizes)
        child_action = _kronecker_product_sequence(
            child_matrices[idx][inverse_permutation(block_perm)]
            for idx, block_perm in enumerate(block_perms)
        )
        target_action = _young_irrep_matrix(tuple(target_partition.parts), inverse_permutation(perm))
        constraints.append(
            _sympy().kronecker_product(_sympy().eye(target_dim), child_action)
            - _sympy().kronecker_product(target_action.T, _sympy().eye(child_dim))
        )
    if not constraints:
        raw_blocks = tuple(
            _sympy().Matrix(
                child_dim,
                target_dim,
                lambda row, col, basis_row=basis_row, basis_col=basis_col: (
                    _sympy().Integer(1) if int(row) == int(basis_row) and int(col) == int(basis_col) else _sympy().Integer(0)
                ),
            )
            for basis_row in range(child_dim)
            for basis_col in range(target_dim)
        )
        if len(raw_blocks) != int(expected_multiplicity):
            raise RuntimeError(
                "Subduction graph no-generator case has unexpected multiplicity "
                f"{len(raw_blocks)} != {expected_multiplicity}."
            )
        return tuple(raw_blocks)

    equations = _sparse_equations_from_constraint_matrices(constraints)
    variable_count = int(child_dim) * int(target_dim)
    raw_vectors = _subduction_graph_solution_vectors(equations, variable_count)
    if len(raw_vectors) != int(expected_multiplicity):
        raise RuntimeError(
            "Subduction graph propagation did not resolve the expected multiplicity "
            f"{len(raw_vectors)} != {expected_multiplicity}. "
            "Select the reference coefficient backend explicitly if oracle validation is needed."
        )
    raw_blocks = tuple(
        _canonicalize_intertwiner(_matrix_from_column_major_vector(vector, child_dim, target_dim))
        for vector in raw_vectors
    )
    return _orthonormalize_intertwiner_metric(tuple(raw_blocks))


def reduced_multiplicity_inner_product(intertwiners):
    """Return the exact reduced multiplicity Gram matrix for restricted blocks."""

    blocks = tuple(_sympy().Matrix(block) for block in intertwiners)
    target_dim = int(blocks[0].cols) if blocks else 1
    if any(int(block.cols) != target_dim for block in blocks):
        raise ValueError("All restricted intertwiners must have the same target dimension.")
    gram = _sympy().zeros(len(blocks), len(blocks))
    for row, left in enumerate(blocks):
        for col, right in enumerate(blocks):
            gram[row, col] = _sympy().simplify((left.T * right).trace() / _sympy().Integer(target_dim))
    identity = _sympy().eye(len(blocks))
    return ReducedMultiplicityInnerProduct(
        gram_matrix=gram,
        orthonormal=bool(_sympy().simplify(gram - identity) == _sympy().zeros(len(blocks), len(blocks))),
        deterministic_convention="exact_pivot_order_then_target_normalized_frobenius_orthonormalization",
        provenance="restricted_subgroup_intertwiner_metric",
        codepath="reduced_multiplicity_inner_product",
    )


def _native_matrix_column_count(matrix):
    matrix = tuple(tuple(row) for row in matrix)
    if not matrix:
        return 0
    cols = len(matrix[0])
    if any(len(row) != cols for row in matrix):
        raise ValueError("native exact matrix rows must have the same length")
    return int(cols)


def _native_exact_trace(matrix):
    total = ExactRadical.rational(0)
    for index, row in enumerate(matrix):
        total += exact_scalar(row[int(index)])
    return total


def reduced_multiplicity_inner_product_native(intertwiners):
    """Return the native exact reduced multiplicity Gram matrix for restricted blocks."""

    blocks = tuple(tuple(tuple(value for value in row) for row in block) for block in intertwiners)
    target_dim = _native_matrix_column_count(blocks[0]) if blocks else 1
    if any(_native_matrix_column_count(block) != target_dim for block in blocks):
        raise ValueError("All restricted intertwiners must have the same target dimension.")
    entries = {}
    divisor = ExactRadical.rational(target_dim)
    for row, left in enumerate(blocks):
        left_t = exact_matrix_transpose(left)
        for col, right in enumerate(blocks):
            product_matrix = exact_matrix_matmul(left_t, right)
            entries[(row, col)] = _native_exact_trace(product_matrix) / divisor
    gram = exact_matrix_from_entries(len(blocks), len(blocks), entries)
    identity = exact_matrix_from_entries(
        len(blocks),
        len(blocks),
        {(index, index): ExactRadical.rational(1) for index in range(len(blocks))},
    )
    return ReducedMultiplicityInnerProduct(
        gram_matrix=gram,
        orthonormal=bool(exact_matrix_equal(gram, identity)),
        deterministic_convention="exact_pivot_order_then_target_normalized_frobenius_orthonormalization",
        provenance="restricted_subgroup_intertwiner_metric_native",
        codepath="reduced_multiplicity_inner_product_native",
    )


def _expand_nary_restricted_intertwiners(raw_blocks, target_partition, coset_reps):
    target_partition = _coerce_partition(target_partition)
    if not raw_blocks:
        return tuple()
    child_dim = int(raw_blocks[0].rows)
    target_dim = int(target_partition.dimension)
    expanded = []
    for block in raw_blocks:
        matrix = _sympy().zeros(len(coset_reps) * child_dim, target_dim)
        for coset_index, coset_rep in enumerate(coset_reps):
            target_action = _young_irrep_matrix(tuple(target_partition.parts), coset_rep)
            piece = _sympy().simplify(block * target_action)
            row0 = int(coset_index) * child_dim
            for row in range(child_dim):
                for col in range(target_dim):
                    matrix[row0 + row, col] = piece[row, col]
        expanded.append(_canonicalize_intertwiner(matrix))
    return _orthonormalize_intertwiner_metric(tuple(expanded))


def _constructive_trivial_symmetric_intertwiner_basis(
    left_partition,
    right_partition,
    target_partition,
    coset_reps,
):
    """Return the normalized orbit-sum intertwiner for the trivial sector.

    This is the subgroup-adapted coefficient for
    ``Ind_{S_a x S_b}^{S_{a+b}}(1_a \\otimes 1_b) -> 1_{a+b}``.
    In the induced basis indexed by left cosets, the invariant vector is the
    normalized constant vector.  No symbolic nullspace, Gram-Schmidt, SVD, or
    floating-point row reduction is used.
    """

    if not (
        bool(left_partition.is_trivial())
        and bool(right_partition.is_trivial())
        and bool(target_partition.is_trivial())
    ):
        return None
    block_sizes = (int(left_partition.size), int(right_partition.size))
    nary = trivial_symmetric_subduction_coefficients(block_sizes)
    if tuple(nary.coset_reps) != tuple(coset_reps):
        raise RuntimeError("Binary coset representatives do not match n-ary Young subgroup construction.")
    if not nary.coefficients:
        return None
    return (_sympy().Matrix(nary.coefficients),)


def _generator_induced_action_matrices_for_parts(left_partition, right_partition, coset_reps):
    a = int(left_partition.size)
    b = int(right_partition.size)
    n = int(a + b)
    subgroup = _subgroup_permutations(a, b)
    subgroup_set = set(subgroup)
    left_matrices = canonical_irrep_matrices(tuple(left_partition.parts))
    right_matrices = canonical_irrep_matrices(tuple(right_partition.parts))
    rep_index = {rep: idx for idx, rep in enumerate(coset_reps)}
    return {
        adjacent_transposition(n, idx): _induced_action_matrix(
            adjacent_transposition(n, idx),
            a=a,
            b=b,
            coset_reps=coset_reps,
            subgroup_set=subgroup_set,
            rep_index=rep_index,
            left_matrices=left_matrices,
            right_matrices=right_matrices,
            left_dim=int(left_partition.dimension),
            right_dim=int(right_partition.dimension),
        )
        for idx in range(max(n - 1, 0))
    }


def _intertwiner_matrices_from_coupling(coupling_tensor):
    target_dim = int(coupling_tensor.target_dim)
    out = []
    for rho in range(int(coupling_tensor.multiplicity)):
        matrix = _sympy().zeros(coupling_tensor.induced_dim, target_dim)
        for vector in coupling_tensor.vectors:
            if int(vector.rho) != int(rho):
                continue
            col = int(vector.target_tableau_index)
            for row, coeff in enumerate(vector.coefficients):
                matrix[int(row), col] = _sympy().simplify(coeff)
        out.append(matrix)
    return tuple(out)


def _generator_equivariant(coupling_tensor):
    actions = _generator_induced_action_matrices_for_parts(
        coupling_tensor.left_partition,
        coupling_tensor.right_partition,
        coupling_tensor.coset_reps,
    )
    for perm, action in actions.items():
        target_action = _young_irrep_matrix(tuple(coupling_tensor.target_partition.parts), inverse_permutation(perm))
        for matrix in _intertwiner_matrices_from_coupling(coupling_tensor):
            if _sympy().simplify(action * matrix - matrix * target_action) != _sympy().zeros(matrix.rows, matrix.cols):
                return False
    return True


def _nary_induced_action_matrices_for_tensor(tensor, *, generators_only=False):
    block_sizes = tuple(int(partition.size) for partition in tensor.subgroup_partitions)
    n = sum(block_sizes)
    subgroup = young_subgroup_permutations(block_sizes)
    subgroup_set = set(subgroup)
    child_matrices = tuple(canonical_irrep_matrices(tuple(partition.parts)) for partition in tensor.subgroup_partitions)
    child_dims = tuple(int(partition.dimension) for partition in tensor.subgroup_partitions)
    perms = (
        tuple(adjacent_transposition(n, idx) for idx in range(max(n - 1, 0)))
        if bool(generators_only)
        else all_permutations(n)
    )
    return {
        perm: _nary_induced_action_matrix(
            perm,
            block_sizes=block_sizes,
            coset_reps=tensor.coset_reps,
            subgroup_set=subgroup_set,
            child_matrices=child_matrices,
            child_dims=child_dims,
        )
        for perm in perms
    }


def _intertwiner_matrices_from_nary_tensor(tensor):
    target_dim = int(tensor.target_dim)
    out = []
    for rho in range(int(tensor.multiplicity)):
        matrix = _sympy().zeros(tensor.induced_dim, target_dim)
        for vector in tensor.vectors:
            if int(vector.rho) != int(rho):
                continue
            col = int(vector.target_tableau_index)
            for row, coeff in enumerate(vector.coefficients):
                matrix[int(row), col] = _sympy().simplify(coeff)
        out.append(matrix)
    return tuple(out)


def _nary_generator_equivariant(tensor):
    if (
        all(
            bool(partition.is_trivial())
            for partition in tensor.subgroup_partitions
        )
        and bool(tensor.target_partition.is_trivial())
    ):
        matrix = tensor.coefficient_matrix()
        if matrix.cols != 1 or matrix.rows == 0:
            return False
        first = matrix[0, 0]
        if any(matrix[row, 0] != first for row in range(matrix.rows)):
            return False
        return bool(
            _sympy().simplify(matrix.rows * first * first) == 1
        )
    actions = _nary_induced_action_matrices_for_tensor(tensor, generators_only=True)
    for perm, action in actions.items():
        target_action = _young_irrep_matrix(tuple(tensor.target_partition.parts), inverse_permutation(perm))
        for matrix in _intertwiner_matrices_from_nary_tensor(tensor):
            if _sympy().simplify(action * matrix - matrix * target_action) != _sympy().zeros(matrix.rows, matrix.cols):
                return False
    return True


@lru_cache(maxsize=None)
def _young_orthogonal_nary_subduction_cached(subgroup_parts, target_parts, bracketing, coefficient_backend):
    shell = _young_orthogonal_nary_subduction_shell(subgroup_parts, target_parts, bracketing)
    subgroup_partitions = shell["subgroup_partitions"]
    target_partition = shell["target_partition"]
    bracketing = shell["bracketing"]
    coefficient_backend = str(coefficient_backend)
    if coefficient_backend not in {"subduction_graph", "intertwiner_oracle", "matrix_units"}:
        raise ValueError("coefficient_backend must be 'subduction_graph', 'intertwiner_oracle', or 'matrix_units'.")
    block_sizes = shell["block_sizes"]
    coset_reps = shell["coset_reps"]
    target_tableaux = shell["target_tableaux"]
    induced_basis = shell["induced_basis"]
    expected_multiplicity = shell["expected_multiplicity"]
    if all(bool(partition.is_trivial()) for partition in subgroup_partitions) and bool(target_partition.is_trivial()):
        trivial_coeffs = trivial_symmetric_subduction_coefficients(block_sizes)
        if tuple(trivial_coeffs.coset_reps) != tuple(coset_reps):
            raise RuntimeError("N-ary coset representatives do not match trivial orbit-sum construction.")
        intertwiners = (_sympy().Matrix(trivial_coeffs.coefficients),)
        codepath = "constructive_trivial_symmetric_orbit_sum"
        coefficient_note = "The trivial-target copy is the normalized symmetric orbit sum over Young-subgroup cosets."
    elif coefficient_backend in {"subduction_graph", "matrix_units"}:
        restricted_intertwiners = _constructive_subduction_graph_restricted_intertwiner_basis(
            subgroup_partitions,
            target_partition,
            expected_multiplicity=expected_multiplicity,
        )
        reduced_inner_product = reduced_multiplicity_inner_product(restricted_intertwiners)
        intertwiners = _expand_nary_restricted_intertwiners(
            restricted_intertwiners,
            target_partition,
            coset_reps,
        )
        if coefficient_backend == "matrix_units":
            codepath = "projector_first_matrix_units_exact"
            coefficient_note = (
                "Nontrivial n-ary coefficients are selected from exact subgroup projectors "
                "and subgroup matrix-unit conventions, then normalized by the reduced "
                "multiplicity inner product; no oracle fallback is used. "
                f"Restricted reduced Gram orthonormal before expansion: {bool(reduced_inner_product.orthonormal)}."
            )
        else:
            codepath = "subduction_graph_propagation_exact"
            coefficient_note = (
                "Nontrivial n-ary coefficients are constructed from exact subgroup-generator "
                "constraints by subduction-graph propagation; no oracle fallback is used."
            )
    else:
        intertwiners = _exact_nary_restricted_intertwiner_basis(
            subgroup_partitions,
            target_partition,
            coset_reps,
        )
        codepath = "nary_specht_intertwiner_nullspace_no_gram_schmidt"
        coefficient_note = (
            "Nontrivial n-ary coefficients are selected by exact symbolic intertwiner equations; "
            "this backend is retained as a reference oracle."
        )
    if len(intertwiners) != len(shell["lr_chain_labels"]):
        raise RuntimeError(
            "N-ary intertwiner multiplicity does not match LR-chain multiplicity "
            f"for {[partition.parts for partition in subgroup_partitions]} -> {target_partition.parts}: "
            f"{len(intertwiners)} != {len(shell['lr_chain_labels'])}."
        )
    return _young_orthogonal_nary_subduction_tensor_from_vectors(
        shell,
        intertwiners,
        coefficient_backend=coefficient_backend,
        provenance="young_orthogonal_nary_tableau",
        codepath=codepath,
        notes=(
            "Child and target Specht carriers are indexed by standard tableaux.",
            "Multiplicity copies are labeled by LR-tableau chains for the requested bracketing.",
            coefficient_note,
            "No Gram-Schmidt or SVD on evaluated feature vectors is used.",
        ),
    )


def young_orthogonal_nary_subduction(
    subgroup_partitions,
    target_partition,
    *,
    bracketing="balanced",
    coefficient_backend="subduction_graph",
):
    """Return n-ary Young-subgroup subduction coefficients.

    ``subgroup_partitions`` are Specht labels for the factors of
    ``S_{a_1} x ... x S_{a_k}``; ``target_partition`` labels the target
    Specht module of ``S_N``.  The coefficient matrix maps from the direct
    n-ary induced basis to target-tableau/multiplicity columns.  Supported
    LR-label bracketings are ``"balanced"``, ``"left"``, and ``"right"``.

    ``coefficient_backend="subduction_graph"`` is the default constructive
    finite-rank backend.  ``"matrix_units"`` selects the projector-first
    subgroup matrix-unit convention.  ``"intertwiner_oracle"`` keeps the older
    exact symbolic nullspace path available for reference validation only.
    """

    subgroup_partitions = tuple(_coerce_partition(partition) for partition in subgroup_partitions)
    target_partition = _coerce_partition(target_partition)
    bracketing = str(bracketing)
    coefficient_backend = str(coefficient_backend)
    if bracketing not in {"balanced", "left", "right"}:
        raise ValueError("bracketing must be 'balanced', 'left', or 'right'.")
    if coefficient_backend not in {"subduction_graph", "intertwiner_oracle", "matrix_units"}:
        raise ValueError("coefficient_backend must be 'subduction_graph', 'intertwiner_oracle', or 'matrix_units'.")
    return _young_orthogonal_nary_subduction_cached(
        tuple(tuple(int(x) for x in partition.parts) for partition in subgroup_partitions),
        tuple(int(x) for x in target_partition.parts),
        bracketing,
        coefficient_backend,
    )


def subduction_from_matrix_units(
    subgroup_partitions,
    target_partition,
    *,
    bracketing="balanced",
):
    """Return n-ary subduction coefficients using the projector-first convention."""

    return young_orthogonal_nary_subduction(
        subgroup_partitions,
        target_partition,
        bracketing=bracketing,
        coefficient_backend="matrix_units",
    )


@lru_cache(maxsize=None)
def _young_orthogonal_induced_coupling_cached(left_parts, right_parts, target_parts):
    left_partition = Partition(tuple(int(x) for x in left_parts))
    right_partition = Partition(tuple(int(x) for x in right_parts))
    target_partition = Partition(tuple(int(x) for x in target_parts))
    a = int(left_partition.size)
    b = int(right_partition.size)
    n = int(a + b)

    subgroup = _subgroup_permutations(a, b)
    coset_reps = _left_coset_representatives(n, subgroup)
    left_tableaux = standard_tableaux(tuple(left_partition.parts))
    right_tableaux = standard_tableaux(tuple(right_partition.parts))
    target_tableaux = standard_tableaux(tuple(target_partition.parts))
    induced_basis = _induced_basis(coset_reps, left_tableaux, right_tableaux)
    constructive_intertwiners = _constructive_trivial_symmetric_intertwiner_basis(
        left_partition,
        right_partition,
        target_partition,
        coset_reps,
    )
    if constructive_intertwiners is None:
        intertwiners = _exact_restricted_intertwiner_basis(
            left_partition,
            right_partition,
            target_partition,
            coset_reps,
        )
        codepath = "specht_intertwiner_nullspace_no_gram_schmidt"
        coefficient_note = "Multiplicity copies are selected by exact symbolic intertwiner equations."
        normalization_note = "Multiplicity copies are normalized by an exact algebraic metric factorization."
    else:
        intertwiners = constructive_intertwiners
        codepath = "constructive_trivial_symmetric_orbit_sum"
        coefficient_note = "The trivial-target copy is the normalized symmetric orbit sum over left cosets."
        normalization_note = "The orbit-sum coefficient is normalized by the exact coset count."
    lr_tableaux = littlewood_richardson_tableaux(left_partition, right_partition, target_partition)
    expected_multiplicity = young_induced_multiplicity_by_character(
        left_partition,
        right_partition,
        target_partition,
    )
    if len(lr_tableaux) != int(expected_multiplicity):
        raise RuntimeError(
            "Littlewood-Richardson multiplicity does not match character multiplicity "
            f"for {left_partition.parts}, {right_partition.parts} -> {target_partition.parts}: "
            f"{len(lr_tableaux)} != {expected_multiplicity}."
        )
    if len(intertwiners) != len(lr_tableaux):
        raise RuntimeError(
            "Young-orthogonal intertwiner multiplicity does not match LR multiplicity "
            f"for {left_partition.parts}, {right_partition.parts} -> {target_partition.parts}: "
            f"{len(intertwiners)} != {len(lr_tableaux)}."
        )
    vectors = []
    for rho, intertwiner in enumerate(intertwiners):
        for target_idx in range(int(target_partition.dimension)):
            coupled = _sympy().Matrix(intertwiner[:, int(target_idx)])
            vectors.append(
                YoungOrthogonalCoupledVector(
                    rho=int(rho),
                    target_tableau_index=int(target_idx),
                    coefficients=tuple(_sympy().simplify(value) for value in coupled),
                )
            )
    return YoungOrthogonalCouplingTensor(
        left_partition=left_partition,
        right_partition=right_partition,
        target_partition=target_partition,
        coset_reps=coset_reps,
        induced_basis=induced_basis,
        target_tableaux=target_tableaux,
        multiplicity=len(intertwiners),
        vectors=tuple(vectors),
        lr_tableaux=lr_tableaux,
        provenance="young_orthogonal_tableau",
        codepath=codepath,
        notes=(
            "Young orthogonal matrices are indexed by standard tableaux.",
            "Multiplicity copies are independently labeled and checked by Littlewood-Richardson tableaux.",
            coefficient_note,
            normalization_note,
            "No Gram-Schmidt or SVD on evaluated feature vectors is used.",
        ),
    )


def young_orthogonal_induced_coupling(left_partition, right_partition, target_partition):
    """Return the Young-orthogonal coupling tensor for one block merge.

    The path is ``S_a x S_b -> S_{a+b}``, with Specht modules indexed by
    standard tableaux.  The returned vectors are exact SymPy coordinates in the
    induced product basis.
    """

    if not isinstance(left_partition, Partition):
        left_partition = Partition(tuple(int(x) for x in left_partition))
    if not isinstance(right_partition, Partition):
        right_partition = Partition(tuple(int(x) for x in right_partition))
    if not isinstance(target_partition, Partition):
        target_partition = Partition(tuple(int(x) for x in target_partition))
    a = int(left_partition.size)
    b = int(right_partition.size)
    n = int(a + b)
    if int(target_partition.size) != n:
        raise ValueError("target_partition size must equal left size plus right size.")
    return _young_orthogonal_induced_coupling_cached(
        tuple(left_partition.parts),
        tuple(right_partition.parts),
        tuple(target_partition.parts),
    )


def validate_young_orthogonal_coupling(coupling_tensor, *, full_projector=True):
    """Validate dimensions, orthonormality, character multiplicity, and equivariance."""

    expected_induced_dim = (
        factorial(int(coupling_tensor.left_partition.size) + int(coupling_tensor.right_partition.size))
        // (factorial(int(coupling_tensor.left_partition.size)) * factorial(int(coupling_tensor.right_partition.size)))
        * coupling_tensor.left_partition.dimension
        * coupling_tensor.right_partition.dimension
    )
    expected_multiplicity = young_induced_multiplicity_by_character(
        coupling_tensor.left_partition,
        coupling_tensor.right_partition,
        coupling_tensor.target_partition,
    )
    matrix = coupling_tensor.coefficient_matrix()
    gram = _sympy().simplify(matrix.T * matrix)
    identity = _sympy().eye(matrix.cols)
    orthonormal = bool(gram == identity)
    multiplicity_matches = bool(int(coupling_tensor.multiplicity) == int(expected_multiplicity))
    generator_equivariant = _generator_equivariant(coupling_tensor)

    projector_equivariant = True
    projector_span_matches = True
    if bool(full_projector):
        induced_actions = _induced_action_matrices_for_coupling(coupling_tensor)
        target_dim = int(coupling_tensor.target_partition.dimension)
        group_order = int(factorial(int(coupling_tensor.target_partition.size)))
        projector = _sympy().zeros(coupling_tensor.induced_dim, coupling_tensor.induced_dim)
        for perm, action in induced_actions.items():
            char = _sympy().Integer(symmetric_group_character(coupling_tensor.target_partition, permutation_cycle_type(inverse_permutation(perm))))
            if char == 0:
                continue
            projector += _sympy().simplify(_sympy().Integer(target_dim) * char * action / group_order)
        projector = _sympy().simplify(projector)
        for action in induced_actions.values():
            if _sympy().simplify(action * projector - projector * action) != _sympy().zeros(projector.rows, projector.cols):
                projector_equivariant = False
                break
        projector_span_matches = bool(_sympy().simplify(matrix * matrix.T - projector) == _sympy().zeros(coupling_tensor.induced_dim, coupling_tensor.induced_dim))

    passed = bool(
        int(coupling_tensor.induced_dim) == int(expected_induced_dim)
        and orthonormal
        and multiplicity_matches
        and generator_equivariant
        and projector_equivariant
        and projector_span_matches
    )
    detail = "ok" if passed else "Young orthogonal coupling validation failed."
    if not bool(full_projector) and passed:
        detail = "ok (generator validation; full central projector skipped)"
    return YoungOrthogonalValidationReport(
        tensor=coupling_tensor,
        expected_induced_dim=int(expected_induced_dim),
        expected_multiplicity=int(expected_multiplicity),
        gram_matrix=gram,
        orthonormal=orthonormal,
        multiplicity_matches_character=multiplicity_matches,
        generator_equivariant=generator_equivariant,
        projector_equivariant=projector_equivariant,
        projector_span_matches=projector_span_matches,
        full_projector_checked=bool(full_projector),
        passed=passed,
        detail=detail,
    )


def validate_young_orthogonal_nary_subduction(tensor):
    """Validate dimensions, multiplicity labels, orthonormality, and equivariance."""

    block_sizes = tuple(int(partition.size) for partition in tensor.subgroup_partitions)
    expected_induced_dim = factorial(sum(block_sizes))
    for size in block_sizes:
        expected_induced_dim //= factorial(int(size))
    for partition in tensor.subgroup_partitions:
        expected_induced_dim *= int(partition.dimension)
    expected_multiplicity = young_nary_induced_multiplicity_by_character(
        tensor.subgroup_partitions,
        tensor.target_partition,
    )
    matrix = tensor.coefficient_matrix()
    gram = _sympy().simplify(matrix.T * matrix)
    identity = _sympy().eye(matrix.cols)
    orthonormal = bool(gram == identity)
    multiplicity_matches = bool(int(tensor.multiplicity) == int(expected_multiplicity))
    lr_labels_match = bool(len(tensor.lr_chain_labels) == int(expected_multiplicity))
    generator_equivariant = _nary_generator_equivariant(tensor)
    passed = bool(
        int(tensor.induced_dim) == int(expected_induced_dim)
        and orthonormal
        and multiplicity_matches
        and lr_labels_match
        and generator_equivariant
    )
    return YoungOrthogonalNaryValidationReport(
        tensor=tensor,
        expected_induced_dim=int(expected_induced_dim),
        expected_multiplicity=int(expected_multiplicity),
        gram_matrix=gram,
        orthonormal=orthonormal,
        multiplicity_matches_character=multiplicity_matches,
        lr_labels_match_multiplicity=lr_labels_match,
        generator_equivariant=generator_equivariant,
        passed=passed,
        detail="ok" if passed else "N-ary Young subduction validation failed.",
    )


def validate_young_orthogonal_nary_subduction_coherence(
    subgroup_partitions,
    target_partition,
    *,
    bracketings=("balanced", "left", "right"),
    coefficient_backend="subduction_graph",
):
    """Validate associativity/coherence across n-ary LR bracketings.

    Coherence here means that changing the LR bracketing may change the chosen
    multiplicity basis, but it must not change the target isotypic subspace in
    the direct n-ary induced basis.  With orthonormal coefficient matrices
    ``C_b``, this is checked by equality of projectors ``C_b C_b^T`` and by
    orthogonality of the overlap/recoupling matrices between bracketings.
    """

    tensors = tuple(
        young_orthogonal_nary_subduction(
            subgroup_partitions,
            target_partition,
            bracketing=bracketing,
            coefficient_backend=coefficient_backend,
        )
        for bracketing in tuple(bracketings)
    )
    matrices = tuple(tensor.coefficient_matrix() for tensor in tensors)
    projectors = tuple(_sympy().simplify(matrix * matrix.T) for matrix in matrices)
    reference_projector = projectors[0] if projectors else _sympy().zeros(0, 0)
    projectors_match = all(
        _sympy().simplify(projector - reference_projector) == _sympy().zeros(reference_projector.rows, reference_projector.cols)
        for projector in projectors
    )
    recoupling_orthogonal = True
    reference = matrices[0] if matrices else _sympy().zeros(0, 0)
    for matrix in matrices[1:]:
        overlap = _sympy().simplify(reference.T * matrix)
        if _sympy().simplify(overlap.T * overlap - _sympy().eye(overlap.cols)) != _sympy().zeros(overlap.cols, overlap.cols):
            recoupling_orthogonal = False
            break
    passed = bool(projectors_match and recoupling_orthogonal)
    return YoungOrthogonalNaryCoherenceReport(
        subgroup_partitions=tuple(_coerce_partition(partition) for partition in subgroup_partitions),
        target_partition=_coerce_partition(target_partition),
        bracketings=tuple(str(bracketing) for bracketing in bracketings),
        projectors_match=projectors_match,
        recoupling_orthogonal=recoupling_orthogonal,
        passed=passed,
        detail="ok" if passed else "N-ary bracketing coherence validation failed.",
    )


def _positive_compositions(n, min_length=2):
    n = int(n)
    if n <= 0:
        return tuple()
    out = []

    def rec(remaining, prefix):
        if remaining == 0:
            if len(prefix) >= int(min_length):
                out.append(tuple(prefix))
            return
        for value in range(1, remaining + 1):
            rec(remaining - value, prefix + [int(value)])

    rec(n, [])
    return tuple(out)


def young_orthogonal_nary_coherence_cases(max_rank, *, max_factors=None, include_multiplicity_only=False):
    """Enumerate nonzero n-ary subduction cases through ``max_rank``."""

    out = []
    for n in range(2, int(max_rank) + 1):
        for block_sizes in _positive_compositions(n):
            if max_factors is not None and len(block_sizes) > int(max_factors):
                continue
            choices = tuple(
                tuple(Partition(parts) for parts in _integer_partitions(size))
                for size in block_sizes
            )
            for subgroup_partitions in product(*choices):
                for target_parts in _integer_partitions(n):
                    target_partition = Partition(target_parts)
                    multiplicity = young_nary_induced_multiplicity_by_character(
                        subgroup_partitions,
                        target_partition,
                    )
                    if int(multiplicity) <= 0:
                        continue
                    if bool(include_multiplicity_only) and int(multiplicity) <= 1:
                        continue
                    out.append((tuple(subgroup_partitions), target_partition, int(multiplicity)))
    return tuple(out)


def validate_young_orthogonal_nary_subduction_coherence_up_to_rank(
    max_rank,
    *,
    max_factors=4,
    max_cases=None,
    bracketings=("balanced", "left", "right"),
    coefficient_backend="subduction_graph",
):
    """Validate n-ary bracketing coherence for enumerated small cases."""

    reports = []
    cases = young_orthogonal_nary_coherence_cases(max_rank, max_factors=max_factors)
    for index, (subgroup_partitions, target_partition, multiplicity) in enumerate(cases):
        if max_cases is not None and int(index) >= int(max_cases):
            break
        report = validate_young_orthogonal_nary_subduction_coherence(
            subgroup_partitions,
            target_partition,
            bracketings=bracketings,
            coefficient_backend=coefficient_backend,
        )
        reports.append(report)
    passed_count = sum(1 for report in reports if report.passed)
    multiplicity_case_count = sum(
        1
        for subgroup_partitions, target_partition, multiplicity in cases[: len(reports)]
        if int(multiplicity) > 1
    )
    passed = bool(passed_count == len(reports))
    return YoungOrthogonalNaryCoherenceSuiteReport(
        max_rank=int(max_rank),
        case_count=len(reports),
        passed_count=int(passed_count),
        multiplicity_case_count=int(multiplicity_case_count),
        reports=tuple(reports),
        passed=passed,
        detail="ok" if passed else f"{len(reports) - passed_count} n-ary coherence cases failed.",
    )


def young_orthogonal_validation_cases(max_rank):
    """Enumerate all nonzero block-merge cases through total rank ``max_rank``."""

    out = []
    for n in range(2, int(max_rank) + 1):
        for left_size in range(1, n):
            right_size = int(n - left_size)
            for left_parts in _integer_partitions(left_size):
                left_partition = Partition(left_parts)
                for right_parts in _integer_partitions(right_size):
                    right_partition = Partition(right_parts)
                    for target_parts in _integer_partitions(n):
                        target_partition = Partition(target_parts)
                        multiplicity = young_induced_multiplicity_by_character(
                            left_partition,
                            right_partition,
                            target_partition,
                        )
                        if int(multiplicity) > 0:
                            out.append((left_partition, right_partition, target_partition, int(multiplicity)))
    return tuple(out)


def validate_young_orthogonal_couplings_up_to_rank(max_rank, *, full_projector_max_rank=4, max_cases=None):
    """Validate all allowed Young orthogonal couplings through a total rank."""

    reports = []
    cases = young_orthogonal_validation_cases(max_rank)
    for index, (left_partition, right_partition, target_partition, multiplicity) in enumerate(cases):
        if max_cases is not None and int(index) >= int(max_cases):
            break
        coupling = young_orthogonal_induced_coupling(left_partition, right_partition, target_partition)
        report = validate_young_orthogonal_coupling(
            coupling,
            full_projector=int(target_partition.size) <= int(full_projector_max_rank),
        )
        reports.append(report)
    passed_count = sum(1 for report in reports if report.passed)
    multiplicity_case_count = sum(1 for report in reports if int(report.expected_multiplicity) > 1)
    passed = bool(passed_count == len(reports))
    detail = "ok" if passed else f"{len(reports) - passed_count} Young orthogonal validation cases failed."
    return YoungOrthogonalValidationSuiteReport(
        max_rank=int(max_rank),
        case_count=len(reports),
        passed_count=int(passed_count),
        multiplicity_case_count=int(multiplicity_case_count),
        reports=tuple(reports),
        passed=passed,
        detail=detail,
    )


__all__ = [
    "LittlewoodRichardsonChainLabel",
    "LittlewoodRichardsonTableau",
    "ReducedMultiplicityInnerProduct",
    "TrivialSymmetricSubductionCoefficients",
    "YoungInducedBasisEntry",
    "YoungNaryInducedBasisEntry",
    "YoungOrthogonalNaryCoherenceReport",
    "YoungOrthogonalNaryCoherenceSuiteReport",
    "YoungOrthogonalNarySubductionTensor",
    "YoungOrthogonalNaryValidationReport",
    "YoungOrthogonalValidationReport",
    "YoungOrthogonalValidationSuiteReport",
    "YoungOrthogonalCoupledVector",
    "YoungOrthogonalCouplingTensor",
    "littlewood_richardson_chain_coefficient",
    "littlewood_richardson_chain_labels",
    "littlewood_richardson_coefficient",
    "littlewood_richardson_tableaux",
    "reduced_multiplicity_inner_product",
    "reduced_multiplicity_inner_product_native",
    "subduction_from_matrix_units",
    "trivial_symmetric_subduction_coefficients",
    "validate_young_orthogonal_coupling",
    "validate_young_orthogonal_nary_subduction",
    "validate_young_orthogonal_nary_subduction_coherence",
    "validate_young_orthogonal_nary_subduction_coherence_up_to_rank",
    "validate_young_orthogonal_couplings_up_to_rank",
    "young_orthogonal_nary_coherence_cases",
    "young_orthogonal_validation_cases",
    "young_induced_multiplicity_by_character",
    "young_nary_induced_multiplicity_by_character",
    "young_orthogonal_nary_subduction",
    "young_orthogonal_induced_coupling",
    "young_subgroup_permutations",
]
