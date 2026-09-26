
"""Exact small-``N`` permutation-projector utilities for the generalized path.

These helpers are symbolic throughout:

- irreducible symmetric-group characters are computed exactly
- projector coefficients are exact rationals
- projector matrices are built as exact SymPy matrices

Explicit permutation/projector materialization is a small-``N`` reference
path.  Character evaluation itself uses the generic exact
Murnaghan--Nakayama recursion and is not rank-capped.

Open follow-up:
- add induction/restriction helpers between related Young subgroups
"""
from functools import lru_cache
from fractions import Fraction
from itertools import combinations, permutations
from math import factorial
from math import sqrt

import numpy as np

from ye3t.exact_linalg import exact_matrix_from_entries, exact_matrix_matmul
from ye3t.exact_scalars import ExactRadical
from .generalized_irreps import Partition, PermutationIrrep, PermutationSubgroupFactor
from ye3t._record import recordclass


_MAX_SMALL_N = 8


@recordclass(('factor', 'partition', 'slots', 'matrix', 'rank'), frozen = True)
class SmallNSymmetricGroupProjector:
    """One exact central projector on the raw tensor-product basis."""


@recordclass(('factor', 'partition', 'slots', 'row', 'col', 'matrix', 'provenance', 'codepath'), frozen=True)
class SubgroupMatrixUnit:
    """One exact Young--Yamanouchi matrix unit on a subgroup-factor action."""


@recordclass(('factor', 'partition', 'slots', 'matrix', 'rank', 'matrix_units', 'provenance', 'codepath'), frozen=True)
class SubgroupIsotypicProjector:
    """Exact isotypic projector as the sum of diagonal subgroup matrix units."""


def _sympy():
    from ye3t._optional_sympy import sp

    return sp


def _native_exact_identity(size):
    size = int(size)
    return exact_matrix_from_entries(
        size,
        size,
        {(idx, idx): ExactRadical.rational(1) for idx in range(size)},
    )


def _native_exact_matrix_add(left, right):
    left = tuple(tuple(row) for row in left)
    right = tuple(tuple(row) for row in right)
    if len(left) != len(right):
        raise ValueError("matrix row counts do not match")
    if not left:
        return tuple()
    cols = len(left[0])
    if any(len(row) != cols for row in left) or any(len(row) != cols for row in right):
        raise ValueError("matrix shapes do not match")
    return tuple(
        tuple(left_value + right_value for left_value, right_value in zip(left_row, right_row))
        for left_row, right_row in zip(left, right)
    )


def _native_exact_trace_int(matrix):
    matrix = tuple(tuple(row) for row in matrix)
    total = ExactRadical.rational(0)
    for idx, row in enumerate(matrix):
        total += row[int(idx)]
    if not total.terms:
        return 0
    if set(total.terms) != {Fraction(1)}:
        raise ArithmeticError("native exact projector trace is not rational")
    value = total.terms[Fraction(1)]
    if value.denominator != 1:
        raise ArithmeticError("native exact projector trace is not integral")
    return int(value)


def permutation_cycle_type(perm):
    """Return the cycle type of a permutation in one-line notation."""

    perm = tuple(int(x) for x in perm)
    visited = [False] * len(perm)
    cycle_lengths = []
    for start in range(len(perm)):
        if visited[start]:
            continue
        current = start
        cycle_length = 0
        while not visited[current]:
            visited[current] = True
            current = perm[current]
            cycle_length += 1
        cycle_lengths.append(cycle_length)
    return tuple(sorted(cycle_lengths, reverse=True))


def compose_permutations(left, right):
    """Return the composition used by the raw slot-permutation action."""

    left = tuple(int(x) for x in left)
    right = tuple(int(x) for x in right)
    if len(left) != len(right):
        raise ValueError("Permutations must have the same size.")
    return tuple(int(left[int(right[idx])]) for idx in range(len(left)))


def inverse_permutation(perm):
    perm = tuple(int(x) for x in perm)
    inverse = [0] * len(perm)
    for idx, value in enumerate(perm):
        inverse[int(value)] = int(idx)
    return tuple(inverse)


def adjacent_transposition(n, index):
    n = int(n)
    index = int(index)
    if index < 0 or index >= n - 1:
        raise ValueError(f"Adjacent transposition index must lie in [0, {n - 2}], got {index}.")
    perm = list(range(n))
    perm[index], perm[index + 1] = perm[index + 1], perm[index]
    return tuple(perm)


def _partition_cells(parts):
    return tuple((row, col) for row, row_len in enumerate(parts) for col in range(int(row_len)))


def _cells_to_partition(cells):
    rows = {}
    for row, col in cells:
        rows.setdefault(int(row), set()).add(int(col))
    if not rows:
        return tuple()
    max_row = max(rows)
    lengths = []
    for row in range(max_row + 1):
        cols = rows.get(row, set())
        if not cols:
            lengths.append(0)
            continue
        if cols != set(range(len(cols))):
            return None
        lengths.append(len(cols))
    while lengths and lengths[-1] == 0:
        lengths.pop()
    if any(length == 0 for length in lengths):
        return None
    if any(left < right for left, right in zip(lengths, lengths[1:])):
        return None
    return tuple(int(length) for length in lengths)


def _removable_corners(parts):
    corners = []
    for row, row_len in enumerate(parts):
        next_len = int(parts[row + 1]) if row + 1 < len(parts) else 0
        if int(row_len) > next_len:
            corners.append((int(row), int(row_len) - 1))
    return tuple(corners)


def _remove_corner(parts, row):
    updated = list(int(x) for x in parts)
    updated[int(row)] -= 1
    while updated and updated[-1] == 0:
        updated.pop()
    if any(left < right for left, right in zip(updated, updated[1:])):
        raise RuntimeError(f"Removing a corner broke partition ordering: {tuple(parts)!r} -> {tuple(updated)!r}")
    return tuple(updated)


@lru_cache(maxsize=None)
def standard_tableaux(partition):
    """Return the standard tableaux of one Young diagram shape."""

    partition = tuple(int(part) for part in partition)
    n = sum(partition)
    if n == 0:
        return (tuple(),)
    tableaux = []
    for row, _col in _removable_corners(partition):
        smaller = _remove_corner(partition, row)
        for tableau in standard_tableaux(smaller):
            rows = [list(r) for r in tableau]
            while len(rows) <= int(row):
                rows.append([])
            rows[int(row)].append(int(n))
            tableaux.append(tuple(tuple(int(x) for x in r) for r in rows))
    return tuple(tableaux)


def tableau_positions(tableau):
    return {
        int(value): (int(row_idx), int(col_idx))
        for row_idx, row in enumerate(tableau)
        for col_idx, value in enumerate(row)
    }


def _swap_tableau_entries(tableau, left_value, right_value):
    positions = tableau_positions(tableau)
    pos_left = positions[int(left_value)]
    pos_right = positions[int(right_value)]
    rows = [list(row) for row in tableau]
    rows[pos_left[0]][pos_left[1]], rows[pos_right[0]][pos_right[1]] = rows[pos_right[0]][pos_right[1]], rows[pos_left[0]][pos_left[1]]
    candidate = tuple(tuple(int(x) for x in row) for row in rows)
    if _is_standard_tableau(candidate):
        return candidate
    return None


def _is_standard_tableau(tableau):
    for row in tableau:
        if any(left >= right for left, right in zip(row, row[1:])):
            return False
    max_cols = max((len(row) for row in tableau), default=0)
    for col in range(max_cols):
        col_entries = [row[col] for row in tableau if len(row) > col]
        if any(upper >= lower for upper, lower in zip(col_entries, col_entries[1:])):
            return False
    return True


def _tableau_content(tableau, value):
    row, col = tableau_positions(tableau)[int(value)]
    return int(col) - int(row)


@lru_cache(maxsize=None)
def adjacent_transposition_representation_matrix(partition, index):
    """Return the canonical orthogonal-form matrix of the adjacent transposition ``s_i``."""

    sp = _sympy()
    tableaux = standard_tableaux(tuple(int(part) for part in partition))
    dim = len(tableaux)
    index = int(index)
    matrix = sp.zeros(dim, dim)
    tableau_index = {tableau: idx for idx, tableau in enumerate(tableaux)}
    for col, tableau in enumerate(tableaux):
        value_left = int(index) + 1
        value_right = int(index) + 2
        content_left = _tableau_content(tableau, value_left)
        content_right = _tableau_content(tableau, value_right)
        axial_distance = sp.Integer(int(content_right - content_left))
        swapped = _swap_tableau_entries(tableau, value_left, value_right)
        if swapped is None:
            matrix[col, col] = sp.Integer(1) if axial_distance == 1 else sp.Integer(-1)
            continue
        row = tableau_index[swapped]
        matrix[col, col] = sp.simplify(sp.Integer(1) / axial_distance)
        off_diag = sp.sqrt(sp.simplify(sp.Integer(1) - sp.Integer(1) / (axial_distance ** 2)))
        matrix[row, col] = off_diag
    return matrix


@lru_cache(maxsize=None)
def adjacent_transposition_representation_matrix_native(partition, index):
    """Return the canonical orthogonal-form adjacent transposition over native radicals."""

    tableaux = standard_tableaux(tuple(int(part) for part in partition))
    dim = len(tableaux)
    index = int(index)
    entries = {}
    tableau_index = {tableau: idx for idx, tableau in enumerate(tableaux)}
    for col, tableau in enumerate(tableaux):
        value_left = int(index) + 1
        value_right = int(index) + 2
        content_left = _tableau_content(tableau, value_left)
        content_right = _tableau_content(tableau, value_right)
        axial_distance = int(content_right - content_left)
        swapped = _swap_tableau_entries(tableau, value_left, value_right)
        if swapped is None:
            entries[(col, col)] = ExactRadical.rational(1 if axial_distance == 1 else -1)
            continue
        row = tableau_index[swapped]
        entries[(col, col)] = ExactRadical.rational(Fraction(1, axial_distance))
        entries[(row, col)] = ExactRadical.sqrt(Fraction(axial_distance * axial_distance - 1, axial_distance * axial_distance))
    return exact_matrix_from_entries(dim, dim, entries)


@lru_cache(maxsize=None)
def adjacent_transposition_representation_matrix_numeric(partition, index):
    """Return the canonical orthogonal-form adjacent transposition as floats."""

    tableaux = standard_tableaux(tuple(int(part) for part in partition))
    dim = len(tableaux)
    index = int(index)
    matrix = np.zeros((dim, dim), dtype=np.float64)
    tableau_index = {tableau: idx for idx, tableau in enumerate(tableaux)}
    for col, tableau in enumerate(tableaux):
        value_left = int(index) + 1
        value_right = int(index) + 2
        content_left = _tableau_content(tableau, value_left)
        content_right = _tableau_content(tableau, value_right)
        axial_distance = int(content_right - content_left)
        swapped = _swap_tableau_entries(tableau, value_left, value_right)
        if swapped is None:
            matrix[col, col] = 1.0 if axial_distance == 1 else -1.0
            continue
        row = tableau_index[swapped]
        matrix[col, col] = 1.0 / float(axial_distance)
        matrix[row, col] = sqrt(max(0.0, 1.0 - 1.0 / float(axial_distance * axial_distance)))
    return matrix


@lru_cache(maxsize=None)
def canonical_irrep_matrices(partition):
    """Return canonical exact matrices for every permutation in the irrep ``partition``."""

    sp = _sympy()
    partition = tuple(int(part) for part in partition)
    n = sum(partition)
    if n > _MAX_SMALL_N:
        raise NotImplementedError(f"Small-N symmetric-group support currently stops at S_{_MAX_SMALL_N}.")
    matrices = {tuple(range(n)): sp.eye(len(standard_tableaux(partition)))}
    queue = [tuple(range(n))]
    adjacent = tuple(adjacent_transposition(n, idx) for idx in range(max(n - 1, 0)))
    adjacent_matrices = {adj: adjacent_transposition_representation_matrix(partition, idx) for idx, adj in enumerate(adjacent)}
    while queue:
        current = queue.pop(0)
        current_matrix = matrices[current]
        for adj in adjacent:
            nxt = compose_permutations(current, adj)
            if nxt in matrices:
                continue
            matrices[nxt] = sp.simplify(adjacent_matrices[adj] * current_matrix)
            queue.append(nxt)
    return matrices


@lru_cache(maxsize=None)
def canonical_irrep_matrices_native(partition):
    """Return canonical exact-radical Young-orthogonal matrices for every permutation."""

    partition = tuple(int(part) for part in partition)
    n = sum(partition)
    if n > _MAX_SMALL_N:
        raise NotImplementedError(f"Small-N symmetric-group support currently stops at S_{_MAX_SMALL_N}.")
    dim = len(standard_tableaux(partition))
    identity = exact_matrix_from_entries(
        dim,
        dim,
        {(idx, idx): ExactRadical.rational(1) for idx in range(dim)},
    )
    matrices = {tuple(range(n)): identity}
    queue = [tuple(range(n))]
    adjacent = tuple(adjacent_transposition(n, idx) for idx in range(max(n - 1, 0)))
    adjacent_matrices = {
        adj: adjacent_transposition_representation_matrix_native(partition, idx)
        for idx, adj in enumerate(adjacent)
    }
    while queue:
        current = queue.pop(0)
        current_matrix = matrices[current]
        for adj in adjacent:
            nxt = compose_permutations(current, adj)
            if nxt in matrices:
                continue
            matrices[nxt] = exact_matrix_matmul(adjacent_matrices[adj], current_matrix)
            queue.append(nxt)
    return matrices


@lru_cache(maxsize=None)
def canonical_irrep_matrices_numeric(partition):
    """Return canonical numeric Young-orthogonal matrices for every permutation."""

    partition = tuple(int(part) for part in partition)
    n = sum(partition)
    if n > _MAX_SMALL_N:
        raise NotImplementedError(f"Small-N symmetric-group support currently stops at S_{_MAX_SMALL_N}.")
    dim = len(standard_tableaux(partition))
    matrices = {tuple(range(n)): np.eye(dim, dtype=np.float64)}
    queue = [tuple(range(n))]
    adjacent = tuple(adjacent_transposition(n, idx) for idx in range(max(n - 1, 0)))
    adjacent_matrices = {
        adj: adjacent_transposition_representation_matrix_numeric(partition, idx)
        for idx, adj in enumerate(adjacent)
    }
    while queue:
        current = queue.pop(0)
        current_matrix = matrices[current]
        for adj in adjacent:
            nxt = compose_permutations(current, adj)
            if nxt in matrices:
                continue
            matrices[nxt] = adjacent_matrices[adj] @ current_matrix
            queue.append(nxt)
    return matrices


def _is_edge_connected(cells):
    if not cells:
        return False
    frontier = {next(iter(cells))}
    visited = set()
    while frontier:
        cell = frontier.pop()
        if cell in visited:
            continue
        visited.add(cell)
        row, col = cell
        for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if neighbor in cells and neighbor not in visited:
                frontier.add(neighbor)
    return visited == cells


def _contains_2x2(cells):
    for row, col in cells:
        block = {(row, col), (row + 1, col), (row, col + 1), (row + 1, col + 1)}
        if block.issubset(cells):
            return True
    return False


@lru_cache(maxsize=None)
def rim_hook_removals(parts, hook_length):
    """Enumerate all rim-hook removals of a fixed length.

    Each result is ``(new_partition, height_minus_one)``. The sign contribution
    in the Murnaghan--Nakayama rule is ``(-1)**height_minus_one``.
    """

    hook_length = int(hook_length)
    cells = _partition_cells(parts)
    all_cells = set(cells)
    out = []
    if hook_length <= 0 or hook_length > len(cells):
        return tuple()
    for subset_tuple in combinations(cells, hook_length):
        subset = set(subset_tuple)
        if not _is_edge_connected(subset):
            continue
        if _contains_2x2(subset):
            continue
        new_parts = _cells_to_partition(all_cells - subset)
        if new_parts is None:
            continue
        height_minus_one = len({row for row, _ in subset}) - 1
        out.append((new_parts, int(height_minus_one)))
    return tuple(out)


@lru_cache(maxsize=None)
def _murnaghan_nakayama_character(parts, cycle_type):
    if not cycle_type:
        return 1 if not parts else 0
    if sum(int(part) for part in parts) != sum(int(length) for length in cycle_type):
        return 0
    hook_length = int(cycle_type[0])
    tail = tuple(int(length) for length in cycle_type[1:])
    total = 0
    for new_parts, height_minus_one in rim_hook_removals(parts, hook_length):
        total += ((-1) ** int(height_minus_one)) * _murnaghan_nakayama_character(new_parts, tail)
    return int(total)


def symmetric_group_character(partition, cycle_type):
    """Return the exact irreducible character ``chi_partition(cycle_type)``."""

    cycle_key = tuple(sorted((int(part) for part in cycle_type if int(part) > 0), reverse=True))
    if sum(cycle_key) != int(partition.size):
        raise ValueError(
            f"Cycle type {cycle_key!r} has size {sum(cycle_key)}, but partition {partition.parts!r} has size {partition.size}."
        )
    return _murnaghan_nakayama_character(tuple(partition.parts), cycle_key)


@lru_cache(maxsize=None)
def _integer_cycle_types(n, max_part=None):
    """Return integer partitions of ``n`` as symmetric-group cycle types."""

    n = int(n)
    if n < 0:
        raise ValueError("cycle-type size must be nonnegative")
    if n == 0:
        return (tuple(),)
    cap = n if max_part is None else min(n, int(max_part))
    out = []
    for first in range(cap, 0, -1):
        for tail in _integer_cycle_types(n - first, first):
            out.append((int(first),) + tuple(int(value) for value in tail))
    return tuple(out)


@lru_cache(maxsize=None)
def _cycle_type_class_size(cycle_type):
    """Return the exact size of one conjugacy class in ``S_n``."""

    cycle_type = tuple(int(value) for value in cycle_type)
    n = sum(cycle_type)
    denominator = 1
    for length in set(cycle_type):
        multiplicity = cycle_type.count(length)
        denominator *= int(length) ** int(multiplicity)
        denominator *= factorial(int(multiplicity))
    return int(factorial(n) // denominator)


@lru_cache(maxsize=None)
def _symmetric_group_conjugacy_classes(n):
    """Return ``(cycle_type, class_size)`` records for ``S_n`` exactly."""

    return tuple(
        (cycle_type, _cycle_type_class_size(cycle_type))
        for cycle_type in _integer_cycle_types(int(n))
    )


@lru_cache(maxsize=None)
def _partition_character_classes(partition_parts):
    """Return the exact irreducible character on every conjugacy class.

    Algorithmic reference: finite-group character inner products grouped by
    conjugacy classes. References: Serre, *Linear Representations of Finite
    Groups*; Sagan, *The Symmetric Group*, 2nd ed. This is an independent
    implementation of the standard class-sum identity; no source was copied.
    """

    partition = Partition(tuple(int(value) for value in partition_parts))
    return tuple(
        (
            cycle_type,
            class_size,
            int(symmetric_group_character(partition, cycle_type)),
        )
        for cycle_type, class_size in _symmetric_group_conjugacy_classes(
            int(partition.size)
        )
    )


@lru_cache(maxsize=None)
def all_permutations(n):
    n = int(n)
    if n > _MAX_SMALL_N:
        raise NotImplementedError(f"Small-N symmetric-group support currently stops at S_{_MAX_SMALL_N}.")
    return tuple(tuple(int(x) for x in perm) for perm in permutations(range(n)))


def permute_state_slots(state, slots, perm):
    """Apply a slot permutation to one raw magnetic-basis state."""

    reordered = list(state)
    slot_values = [state[int(slot)] for slot in slots]
    for new_offset, slot in enumerate(slots):
        reordered[int(slot)] = int(slot_values[int(perm[new_offset])])
    return tuple(int(x) for x in reordered)


def projector_matrix_for_factor(
    factor,
    partition,
    slots,
    basis_states,
):
    """Build the exact central idempotent for one subgroup factor on the raw basis."""

    sp = _sympy()
    if int(partition.size) != int(factor.multiplicity):
        raise ValueError(
            f"Partition {partition.parts!r} has size {partition.size}, "
            f"but factor multiplicity is {factor.multiplicity}."
        )
    if len(slots) != int(factor.multiplicity):
        raise ValueError(f"Expected {factor.multiplicity} slots for factor, got {tuple(slots)!r}.")
    if int(partition.size) > _MAX_SMALL_N:
        raise NotImplementedError(f"Small-N symmetric-group support currently stops at S_{_MAX_SMALL_N}.")

    slots = tuple(int(slot) for slot in slots)
    basis_states = tuple(tuple(int(x) for x in state) for state in basis_states)
    index_of_state = {state: idx for idx, state in enumerate(basis_states)}
    entries = {}
    prefactor = sp.Rational(int(partition.dimension), factorial(int(partition.size)))

    for perm in all_permutations(int(partition.size)):
        char = symmetric_group_character(partition, permutation_cycle_type(perm))
        if char == 0:
            continue
        weight = sp.Integer(int(char)) * prefactor
        for col, state in enumerate(basis_states):
            row_state = permute_state_slots(state, slots, perm)
            row = index_of_state[row_state]
            key = (int(row), int(col))
            entries[key] = sp.simplify(entries.get(key, sp.Integer(0)) + weight)

    matrix = sp.SparseMatrix(len(basis_states), len(basis_states), entries)
    rank = int(sp.simplify(matrix.trace()))
    return SmallNSymmetricGroupProjector(
        factor=factor,
        partition=partition,
        slots=slots,
        matrix=matrix,
        rank=rank,
    )


def projector_matrix_for_factor_numeric(
    factor,
    partition,
    slots,
    basis_states,
):
    """Build a numeric central idempotent for one subgroup factor on the raw basis."""

    if int(partition.size) != int(factor.multiplicity):
        raise ValueError(
            f"Partition {partition.parts!r} has size {partition.size}, "
            f"but factor multiplicity is {factor.multiplicity}."
        )
    if len(slots) != int(factor.multiplicity):
        raise ValueError(f"Expected {factor.multiplicity} slots for factor, got {tuple(slots)!r}.")
    if int(partition.size) > _MAX_SMALL_N:
        raise NotImplementedError(f"Small-N symmetric-group support currently stops at S_{_MAX_SMALL_N}.")

    slots = tuple(int(slot) for slot in slots)
    basis_states = tuple(tuple(int(x) for x in state) for state in basis_states)
    index_of_state = {state: idx for idx, state in enumerate(basis_states)}
    matrix = np.zeros((len(basis_states), len(basis_states)), dtype=np.float64)
    prefactor = float(int(partition.dimension)) / float(factorial(int(partition.size)))

    for perm in all_permutations(int(partition.size)):
        char = symmetric_group_character(partition, permutation_cycle_type(perm))
        if char == 0:
            continue
        weight = float(int(char)) * prefactor
        for col, state in enumerate(basis_states):
            row_state = permute_state_slots(state, slots, perm)
            row = index_of_state[row_state]
            matrix[int(row), int(col)] += weight

    rank = int(round(float(np.trace(matrix))))
    return SmallNSymmetricGroupProjector(
        factor=factor,
        partition=partition,
        slots=slots,
        matrix=matrix,
        rank=rank,
    )


def projector_matrix_for_factor_native(
    factor,
    partition,
    slots,
    basis_states,
):
    """Build a native exact central idempotent for one subgroup factor."""

    if int(partition.size) != int(factor.multiplicity):
        raise ValueError(
            f"Partition {partition.parts!r} has size {partition.size}, "
            f"but factor multiplicity is {factor.multiplicity}."
        )
    if len(slots) != int(factor.multiplicity):
        raise ValueError(f"Expected {factor.multiplicity} slots for factor, got {tuple(slots)!r}.")
    if int(partition.size) > _MAX_SMALL_N:
        raise NotImplementedError(f"Small-N symmetric-group support currently stops at S_{_MAX_SMALL_N}.")

    slots = tuple(int(slot) for slot in slots)
    basis_states = tuple(tuple(int(x) for x in state) for state in basis_states)
    index_of_state = {state: idx for idx, state in enumerate(basis_states)}
    entries = {}
    prefactor = ExactRadical.rational(Fraction(int(partition.dimension), factorial(int(partition.size))))

    for perm in all_permutations(int(partition.size)):
        char = symmetric_group_character(partition, permutation_cycle_type(perm))
        if char == 0:
            continue
        weight = ExactRadical.rational(int(char)) * prefactor
        for col, state in enumerate(basis_states):
            row_state = permute_state_slots(state, slots, perm)
            row = index_of_state[row_state]
            key = (int(row), int(col))
            entries[key] = entries.get(key, ExactRadical.rational(0)) + weight

    matrix = exact_matrix_from_entries(len(basis_states), len(basis_states), entries)
    return SmallNSymmetricGroupProjector(
        factor=factor,
        partition=partition,
        slots=slots,
        matrix=matrix,
        rank=_native_exact_trace_int(matrix),
    )


def subgroup_matrix_units_for_factor(
    factor,
    partition,
    slots,
    basis_states,
):
    """Build exact Young--Yamanouchi matrix units for one subgroup factor."""

    sp = _sympy()
    if int(partition.size) != int(factor.multiplicity):
        raise ValueError(
            f"Partition {partition.parts!r} has size {partition.size}, "
            f"but factor multiplicity is {factor.multiplicity}."
        )
    if len(slots) != int(factor.multiplicity):
        raise ValueError(f"Expected {factor.multiplicity} slots for factor, got {tuple(slots)!r}.")
    if int(partition.size) > _MAX_SMALL_N:
        raise NotImplementedError(f"Small-N symmetric-group support currently stops at S_{_MAX_SMALL_N}.")

    slots = tuple(int(slot) for slot in slots)
    basis_states = tuple(tuple(int(x) for x in state) for state in basis_states)
    index_of_state = {state: idx for idx, state in enumerate(basis_states)}
    irrep_matrices = canonical_irrep_matrices(tuple(partition.parts))
    dim = int(partition.dimension)
    prefactor = sp.Rational(dim, factorial(int(partition.size)))
    units = {}
    for row_index in range(dim):
        for col_index in range(dim):
            entries = {}
            for perm in all_permutations(int(partition.size)):
                irrep_inverse = irrep_matrices[inverse_permutation(perm)]
                coeff = sp.simplify(prefactor * irrep_inverse[int(col_index), int(row_index)])
                if coeff == 0:
                    continue
                for col, state in enumerate(basis_states):
                    row_state = permute_state_slots(state, slots, perm)
                    row = index_of_state[row_state]
                    key = (int(row), int(col))
                    entries[key] = sp.simplify(entries.get(key, sp.Integer(0)) + coeff)
            units[(int(row_index), int(col_index))] = SubgroupMatrixUnit(
                factor=factor,
                partition=partition,
                slots=slots,
                row=int(row_index),
                col=int(col_index),
                matrix=sp.SparseMatrix(len(basis_states), len(basis_states), entries),
                provenance="young_yamanouchi_matrix_coefficients",
                codepath="subgroup_matrix_units_for_factor",
            )
    return units


def subgroup_matrix_units_for_factor_numeric(
    factor,
    partition,
    slots,
    basis_states,
):
    """Build numeric Young--Yamanouchi matrix units for one subgroup factor."""

    if int(partition.size) != int(factor.multiplicity):
        raise ValueError(
            f"Partition {partition.parts!r} has size {partition.size}, "
            f"but factor multiplicity is {factor.multiplicity}."
        )
    if len(slots) != int(factor.multiplicity):
        raise ValueError(f"Expected {factor.multiplicity} slots for factor, got {tuple(slots)!r}.")
    if int(partition.size) > _MAX_SMALL_N:
        raise NotImplementedError(f"Small-N symmetric-group support currently stops at S_{_MAX_SMALL_N}.")

    slots = tuple(int(slot) for slot in slots)
    basis_states = tuple(tuple(int(x) for x in state) for state in basis_states)
    index_of_state = {state: idx for idx, state in enumerate(basis_states)}
    irrep_matrices = canonical_irrep_matrices_numeric(tuple(partition.parts))
    dim = int(partition.dimension)
    prefactor = float(dim) / float(factorial(int(partition.size)))
    units = {}
    for row_index in range(dim):
        for col_index in range(dim):
            matrix = np.zeros((len(basis_states), len(basis_states)), dtype=np.float64)
            for perm in all_permutations(int(partition.size)):
                irrep_inverse = irrep_matrices[inverse_permutation(perm)]
                coeff = prefactor * float(irrep_inverse[int(col_index), int(row_index)])
                if abs(coeff) <= 1.0e-15:
                    continue
                for col, state in enumerate(basis_states):
                    row_state = permute_state_slots(state, slots, perm)
                    row = index_of_state[row_state]
                    matrix[int(row), int(col)] += coeff
            units[(int(row_index), int(col_index))] = SubgroupMatrixUnit(
                factor=factor,
                partition=partition,
                slots=slots,
                row=int(row_index),
                col=int(col_index),
                matrix=matrix,
                provenance="young_yamanouchi_matrix_coefficients_numeric",
                codepath="subgroup_matrix_units_for_factor_numeric",
            )
    return units


def subgroup_matrix_units_for_factor_native(
    factor,
    partition,
    slots,
    basis_states,
):
    """Build native exact Young--Yamanouchi matrix units for one subgroup factor."""

    if int(partition.size) != int(factor.multiplicity):
        raise ValueError(
            f"Partition {partition.parts!r} has size {partition.size}, "
            f"but factor multiplicity is {factor.multiplicity}."
        )
    if len(slots) != int(factor.multiplicity):
        raise ValueError(f"Expected {factor.multiplicity} slots for factor, got {tuple(slots)!r}.")
    if int(partition.size) > _MAX_SMALL_N:
        raise NotImplementedError(f"Small-N symmetric-group support currently stops at S_{_MAX_SMALL_N}.")

    slots = tuple(int(slot) for slot in slots)
    basis_states = tuple(tuple(int(x) for x in state) for state in basis_states)
    index_of_state = {state: idx for idx, state in enumerate(basis_states)}
    irrep_matrices = canonical_irrep_matrices_native(tuple(partition.parts))
    dim = int(partition.dimension)
    prefactor = ExactRadical.rational(Fraction(dim, factorial(int(partition.size))))
    units = {}
    for row_index in range(dim):
        for col_index in range(dim):
            entries = {}
            for perm in all_permutations(int(partition.size)):
                irrep_inverse = irrep_matrices[inverse_permutation(perm)]
                coeff = prefactor * irrep_inverse[int(col_index)][int(row_index)]
                if coeff == 0:
                    continue
                for col, state in enumerate(basis_states):
                    row_state = permute_state_slots(state, slots, perm)
                    row = index_of_state[row_state]
                    key = (int(row), int(col))
                    entries[key] = entries.get(key, ExactRadical.rational(0)) + coeff
            units[(int(row_index), int(col_index))] = SubgroupMatrixUnit(
                factor=factor,
                partition=partition,
                slots=slots,
                row=int(row_index),
                col=int(col_index),
                matrix=exact_matrix_from_entries(len(basis_states), len(basis_states), entries),
                provenance="young_yamanouchi_matrix_coefficients_native",
                codepath="subgroup_matrix_units_for_factor_native",
            )
    return units


def _selected_subgroup_matrix_units_for_factor_native(
    factor,
    partition,
    slots,
    basis_states,
    unit_indices,
):
    """Build selected exact Young matrix units through the native scalar path."""
    sp = _sympy()
    if int(partition.size) != int(factor.multiplicity):
        raise ValueError(
            f"Partition {partition.parts!r} has size {partition.size}, "
            f"but factor multiplicity is {factor.multiplicity}."
        )
    if len(slots) != int(factor.multiplicity):
        raise ValueError(
            f"Expected {factor.multiplicity} slots for factor, "
            f"got {tuple(slots)!r}."
        )
    if int(partition.size) > _MAX_SMALL_N:
        raise NotImplementedError(
            "Small-N symmetric-group support currently stops at "
            f"S_{_MAX_SMALL_N}."
        )
    dimension = int(partition.dimension)
    unit_indices = tuple(
        sorted(
            {
                (int(row), int(column))
                for row, column in unit_indices
            }
        )
    )
    if not unit_indices:
        raise ValueError("At least one matrix-unit index is required.")
    if any(
        row < 0
        or row >= dimension
        or column < 0
        or column >= dimension
        for row, column in unit_indices
    ):
        raise ValueError("Matrix-unit index is outside the irrep dimension.")

    slots = tuple(int(slot) for slot in slots)
    basis_states = tuple(
        tuple(int(value) for value in state)
        for state in basis_states
    )
    index_of_state = {state: index for index, state in enumerate(basis_states)}
    irrep_matrices = canonical_irrep_matrices_native(
        tuple(partition.parts)
    )
    prefactor = ExactRadical.rational(
        Fraction(dimension, factorial(int(partition.size)))
    )
    entries = {key: {} for key in unit_indices}
    for perm in all_permutations(int(partition.size)):
        inverse = irrep_matrices[inverse_permutation(perm)]
        coefficients = {
            key: prefactor * inverse[key[1]][key[0]]
            for key in unit_indices
        }
        coefficients = {
            key: value
            for key, value in coefficients.items()
            if value != 0
        }
        if not coefficients:
            continue
        action = tuple(
            (
                index_of_state[permute_state_slots(state, slots, perm)],
                column,
            )
            for column, state in enumerate(basis_states)
        )
        for key, coefficient in coefficients.items():
            unit_entries = entries[key]
            for row, column in action:
                entry = (int(row), int(column))
                unit_entries[entry] = (
                    unit_entries.get(entry, ExactRadical.rational(0))
                    + coefficient
                )
    units = {}
    for row, column in unit_indices:
        matrix_entries = {
            key: value._sympy_()
            for key, value in entries[(row, column)].items()
            if value != 0
        }
        units[(row, column)] = SubgroupMatrixUnit(
            factor=factor,
            partition=partition,
            slots=slots,
            row=int(row),
            col=int(column),
            matrix=sp.SparseMatrix(
                len(basis_states),
                len(basis_states),
                matrix_entries,
            ),
            provenance="young_yamanouchi_matrix_coefficients_native",
            codepath="selected_subgroup_matrix_units_for_factor_native",
        )
    return units


def subgroup_isotypic_projector(
    factor,
    partition,
    slots,
    basis_states,
):
    """Build an exact subgroup isotypic projector from diagonal matrix units."""

    sp = _sympy()
    basis_states = tuple(tuple(int(x) for x in state) for state in basis_states)
    units = subgroup_matrix_units_for_factor(factor, partition, slots, basis_states)
    dim = int(partition.dimension)
    matrix = sp.zeros(len(basis_states), len(basis_states))
    for index in range(dim):
        matrix = sp.simplify(matrix + units[(index, index)].matrix)
    return SubgroupIsotypicProjector(
        factor=factor,
        partition=partition,
        slots=tuple(int(slot) for slot in slots),
        matrix=matrix,
        rank=int(sp.simplify(matrix.trace())),
        matrix_units=units,
        provenance="sum_of_diagonal_young_yamanouchi_matrix_units",
        codepath="subgroup_isotypic_projector",
    )


def subgroup_isotypic_projector_numeric(
    factor,
    partition,
    slots,
    basis_states,
):
    """Build a numeric subgroup isotypic projector from diagonal matrix units."""

    basis_states = tuple(tuple(int(x) for x in state) for state in basis_states)
    units = subgroup_matrix_units_for_factor_numeric(factor, partition, slots, basis_states)
    dim = int(partition.dimension)
    matrix = np.zeros((len(basis_states), len(basis_states)), dtype=np.float64)
    for index in range(dim):
        matrix += units[(index, index)].matrix
    return SubgroupIsotypicProjector(
        factor=factor,
        partition=partition,
        slots=tuple(int(slot) for slot in slots),
        matrix=matrix,
        rank=int(round(float(np.trace(matrix)))),
        matrix_units=units,
        provenance="sum_of_diagonal_young_yamanouchi_matrix_units_numeric",
        codepath="subgroup_isotypic_projector_numeric",
    )


def subgroup_isotypic_projector_native(
    factor,
    partition,
    slots,
    basis_states,
):
    """Build a native exact subgroup isotypic projector from diagonal matrix units."""

    basis_states = tuple(tuple(int(x) for x in state) for state in basis_states)
    units = subgroup_matrix_units_for_factor_native(factor, partition, slots, basis_states)
    dim = int(partition.dimension)
    matrix = exact_matrix_from_entries(len(basis_states), len(basis_states), {})
    for index in range(dim):
        matrix = _native_exact_matrix_add(matrix, units[(index, index)].matrix)
    return SubgroupIsotypicProjector(
        factor=factor,
        partition=partition,
        slots=tuple(int(slot) for slot in slots),
        matrix=matrix,
        rank=_native_exact_trace_int(matrix),
        matrix_units=units,
        provenance="sum_of_diagonal_young_yamanouchi_matrix_units_native",
        codepath="subgroup_isotypic_projector_native",
    )


def projector_matrices_for_irrep(
    permutation_irrep,
    slot_groups,
    basis_states,
):
    """Build one exact factor projector per direct-product subgroup factor."""

    if len(slot_groups) != len(permutation_irrep.subgroup.factors):
        raise ValueError(
            f"Expected {len(permutation_irrep.subgroup.factors)} slot groups, got {len(slot_groups)}."
        )
    return tuple(
        projector_matrix_for_factor(
            factor=factor,
            partition=partition,
            slots=slots,
            basis_states=basis_states,
        )
        for factor, partition, slots in zip(
            permutation_irrep.subgroup.factors,
            permutation_irrep.partitions,
            slot_groups,
            strict=True,
        )
    )


def projector_matrices_for_irrep_numeric(
    permutation_irrep,
    slot_groups,
    basis_states,
):
    """Build numeric factor projectors per direct-product subgroup factor."""

    if len(slot_groups) != len(permutation_irrep.subgroup.factors):
        raise ValueError(
            f"Expected {len(permutation_irrep.subgroup.factors)} slot groups, got {len(slot_groups)}."
        )
    return tuple(
        projector_matrix_for_factor_numeric(
            factor=factor,
            partition=partition,
            slots=slots,
            basis_states=basis_states,
        )
        for factor, partition, slots in zip(
            permutation_irrep.subgroup.factors,
            permutation_irrep.partitions,
            slot_groups,
            strict=True,
        )
    )


def projector_matrices_for_irrep_native(
    permutation_irrep,
    slot_groups,
    basis_states,
):
    """Build native exact factor projectors per direct-product subgroup factor."""

    if len(slot_groups) != len(permutation_irrep.subgroup.factors):
        raise ValueError(
            f"Expected {len(permutation_irrep.subgroup.factors)} slot groups, got {len(slot_groups)}."
        )
    return tuple(
        projector_matrix_for_factor_native(
            factor=factor,
            partition=partition,
            slots=slots,
            basis_states=basis_states,
        )
        for factor, partition, slots in zip(
            permutation_irrep.subgroup.factors,
            permutation_irrep.partitions,
            slot_groups,
            strict=True,
        )
    )


def combined_projector_matrix(
    permutation_irrep,
    slot_groups,
    basis_states,
):
    """Build the combined exact direct-product subgroup projector."""

    sp = _sympy()
    basis_states = tuple(tuple(int(x) for x in state) for state in basis_states)
    factors = projector_matrices_for_irrep(
        permutation_irrep,
        slot_groups,
        basis_states,
    )
    if not factors:
        identity = sp.eye(len(basis_states))
        return identity, tuple()
    matrix = sp.eye(len(basis_states))
    for factor_projector in factors:
        matrix = sp.simplify(matrix * factor_projector.matrix)
    return matrix, factors


def combined_projector_matrix_numeric(
    permutation_irrep,
    slot_groups,
    basis_states,
):
    """Build the combined numeric direct-product subgroup projector."""

    basis_states = tuple(tuple(int(x) for x in state) for state in basis_states)
    factors = projector_matrices_for_irrep_numeric(
        permutation_irrep,
        slot_groups,
        basis_states,
    )
    if not factors:
        return np.eye(len(basis_states), dtype=np.float64), tuple()
    matrix = np.eye(len(basis_states), dtype=np.float64)
    for factor_projector in factors:
        matrix = matrix @ factor_projector.matrix
    return matrix, factors


def combined_projector_matrix_native(
    permutation_irrep,
    slot_groups,
    basis_states,
):
    """Build the combined native exact direct-product subgroup projector."""

    basis_states = tuple(tuple(int(x) for x in state) for state in basis_states)
    factors = projector_matrices_for_irrep_native(
        permutation_irrep,
        slot_groups,
        basis_states,
    )
    if not factors:
        return _native_exact_identity(len(basis_states)), tuple()
    matrix = _native_exact_identity(len(basis_states))
    for factor_projector in factors:
        matrix = exact_matrix_matmul(matrix, factor_projector.matrix)
    return matrix, factors


__all__ = [
    "adjacent_transposition",
    "adjacent_transposition_representation_matrix",
    "adjacent_transposition_representation_matrix_native",
    "adjacent_transposition_representation_matrix_numeric",
    "SmallNSymmetricGroupProjector",
    "SubgroupIsotypicProjector",
    "SubgroupMatrixUnit",
    "all_permutations",
    "canonical_irrep_matrices",
    "canonical_irrep_matrices_native",
    "canonical_irrep_matrices_numeric",
    "combined_projector_matrix",
    "combined_projector_matrix_native",
    "combined_projector_matrix_numeric",
    "compose_permutations",
    "inverse_permutation",
    "permutation_cycle_type",
    "permute_state_slots",
    "projector_matrices_for_irrep",
    "projector_matrices_for_irrep_native",
    "projector_matrices_for_irrep_numeric",
    "projector_matrix_for_factor",
    "projector_matrix_for_factor_native",
    "projector_matrix_for_factor_numeric",
    "rim_hook_removals",
    "standard_tableaux",
    "subgroup_isotypic_projector",
    "subgroup_isotypic_projector_native",
    "subgroup_isotypic_projector_numeric",
    "subgroup_matrix_units_for_factor",
    "subgroup_matrix_units_for_factor_native",
    "subgroup_matrix_units_for_factor_numeric",
    "symmetric_group_character",
    "tableau_positions",
]
