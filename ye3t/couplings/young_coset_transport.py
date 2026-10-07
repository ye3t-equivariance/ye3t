"""Compact Young transport for a fixed-content induced carrier.

The identity-coset restriction fixes the existing multiplicity-copy gauge.
Other coset slices follow from the Young orthogonal representation, so a
compiled evaluator need not retain one dense tableau table per coset.

Algorithmic reference: subgroup-adapted induction in Goff and Thompson,
arXiv:2609.31895, Eqs. (7)--(12). Independent implementation using YE3T's
Young orthogonal adjacent-generator matrices; no external code is copied.
The reduced subgroup-constraint viewpoint is also described by Chilla,
arXiv:math-ph/0606037; Young adjacent-generator representations follow the
standard Young orthogonal construction discussed by Vershik and Okounkov,
arXiv:math/0503040.
"""

from functools import lru_cache


def canonical_factor_coset(factor_types, canonical_types):
    """Group each factor type in canonical order, preserving order within a block."""
    factor_types = tuple(factor_types)
    canonical_types = tuple(canonical_types)
    if len(factor_types) != len(canonical_types) or sorted(factor_types) != sorted(canonical_types):
        raise ValueError("Factor types do not match the fixed-content carrier.")
    return tuple(index for factor_type in dict.fromkeys(canonical_types)
                 for index, candidate in enumerate(factor_types)
                 if candidate == factor_type)


def direct_restricted_young_values(child_partitions, target_partition):
    """Compile the identity-coset map from exact subgroup-generator constraints.

    This avoids the orbit-sized Young coefficient matrix. The returned gamma
    order and signs follow the existing subduction-graph pivot convention.
    """
    import math
    import sympy as sym

    from ye3t.representations.generalized_irreps import Partition
    from ye3t.representations.young_orthogonal import (
        _constructive_subduction_graph_restricted_intertwiner_basis,
        reduced_multiplicity_inner_product,
        standard_tableaux,
        young_nary_induced_multiplicity_by_character,
    )

    children = tuple(tuple(int(value) for value in row) for row in child_partitions)
    target = tuple(int(value) for value in target_partition)
    rank = sum(sum(row) for row in children)
    orbit_count = math.factorial(rank)
    for row in children:
        orbit_count //= math.factorial(sum(row))
    multiplicity = int(young_nary_induced_multiplicity_by_character(
        tuple(Partition(row) for row in children), Partition(target)
    ))
    blocks = _constructive_subduction_graph_restricted_intertwiner_basis(
        children, target, expected_multiplicity=multiplicity,
    )
    target_dim = len(standard_tableaux(target))
    child_dim = int(blocks[0].rows) if blocks else 0
    if not blocks or any(block.shape != (child_dim, target_dim) for block in blocks):
        raise ArithmeticError("Direct Young subduction has inconsistent multiplicity axes.")
    no_generators = all(sum(row) == 1 for row in children)
    reduced = reduced_multiplicity_inner_product(blocks)
    expected_gram = (sym.eye(multiplicity) / target_dim
                     if no_generators else sym.eye(multiplicity))
    if sym.simplify(reduced.gram_matrix - expected_gram) != sym.zeros(
        multiplicity, multiplicity
    ):
        raise ArithmeticError("Direct Young reduced multiplicity Gram is invalid.")
    scale = math.sqrt(target_dim / orbit_count) if no_generators else 1.0 / math.sqrt(orbit_count)
    values = tuple(tuple(float(block[row, col] * scale)
                         for block in blocks for col in range(target_dim))
                   for row in range(child_dim))
    return values, multiplicity


def restricted_young_values(matrix, canonical_rows, canonical_columns,
                            cosets, target_partition, rows_per_coset,
                            target_tableau_dim):
    """Keep the current identity-coset gauge and check every transported slice."""
    import numpy as np

    from ye3t.representations.young_orthogonal import _adjacent_word_to_permutation

    columns = len(canonical_columns)
    if not cosets or cosets[0] != tuple(range(len(cosets[0]))):
        raise ArithmeticError("Young cosets must begin with the identity.")
    if columns % target_tableau_dim or len(canonical_rows) != len(cosets) * rows_per_coset:
        raise ArithmeticError("Young restriction has inconsistent tableau axes.")
    identity = np.asarray(matrix.extract(
        canonical_rows[:rows_per_coset], canonical_columns
    ), dtype=np.float64)
    copy_count = columns // target_tableau_dim
    generators = adjacent_generator_columns(tuple(target_partition))
    maximum_residual = 0.0
    for coset_index, representative in enumerate(cosets):
        expected = identity.reshape(rows_per_coset, copy_count,
                                    target_tableau_dim)
        for generator_index in reversed(_adjacent_word_to_permutation(
            tuple(representative)
        )):
            diagonal, off_diagonal, partner = generators[generator_index]
            expected = (expected * np.asarray(diagonal)
                        + expected[..., list(partner)] * np.asarray(off_diagonal))
        expected = expected.reshape(rows_per_coset, columns)
        actual = np.asarray(matrix.extract(
            canonical_rows[coset_index * rows_per_coset:
                           (coset_index + 1) * rows_per_coset], canonical_columns
        ), dtype=np.float64)
        residual = float(np.max(np.abs(expected - actual)))
        maximum_residual = max(maximum_residual, residual)
    if maximum_residual > 1.0e-8:
        raise ArithmeticError("Compact Young coset transport disagrees with subduction.")
    return tuple(tuple(float(value) for value in row) for row in identity), maximum_residual


@lru_cache(maxsize=128)
def adjacent_generator_columns(target_partition):
    """Store each Young generator in its at-most-two-entry column form."""
    from ye3t.representations.young_orthogonal import (
        adjacent_transposition_representation_matrix,
        standard_tableaux,
    )

    partition = tuple(int(value) for value in target_partition)
    width = len(standard_tableaux(partition))
    plans = []
    for generator_index in range(sum(partition) - 1):
        matrix = adjacent_transposition_representation_matrix(
            partition, generator_index
        )
        diagonal, off_diagonal, partner = [], [], []
        for column in range(width):
            others = tuple(row for row in range(width)
                           if row != column and matrix[row, column] != 0)
            if len(others) > 1:
                raise ArithmeticError("Young adjacent generator is not two-sparse.")
            diagonal.append(float(matrix[column, column]))
            off_diagonal.append(float(matrix[others[0], column]) if others else 0.0)
            partner.append(int(others[0]) if others else column)
        plans.append((tuple(diagonal), tuple(off_diagonal), tuple(partner)))
    return tuple(plans)


def expanded_young_values(young, cosets):
    """Reconstruct a saved full table only for legacy-artifact validation."""
    import numpy as np

    from ye3t.representations.young_orthogonal import _adjacent_word_to_permutation

    rows = int(young["local_tableau_dim"])
    copies = int(young["gamma_count"])
    tableaux = int(young["target_tableau_dim"])
    if young.get("restricted_kind") == "matrix_units":
        identity = np.eye(tableaux, dtype=np.float64).reshape(
            rows, copies, tableaux
        ) * float(young["restricted_scale"])
    else:
        identity = np.asarray(young["restricted_values"], dtype=np.float64)
        identity = identity.reshape(rows, copies, tableaux)
    slices = []
    for representative in cosets:
        result = identity
        for generator_index in reversed(_adjacent_word_to_permutation(
            tuple(representative)
        )):
            diagonal, off_diagonal, partner = young["adjacent_generators"][generator_index]
            result = (result * np.asarray(diagonal)
                      + result[..., list(partner)] * np.asarray(off_diagonal))
        slices.append(result.reshape(rows, copies * tableaux))
    return np.concatenate(slices, axis=0)


def bind_young_restriction(young, dtype, device):
    """Bind the restricted block and sparse Young generators once for Torch."""
    import torch

    rows = int(young["local_tableau_dim"])
    copies = int(young["gamma_count"])
    tableaux = int(young["target_tableau_dim"])
    if "values" in young:
        # Saved pre-compact schedules carry the complete Young table.
        return torch.as_tensor(young["values"], dtype=dtype,
                               device=device).reshape(-1, rows, copies, tableaux)
    restricted = (None if young.get("restricted_kind") == "matrix_units" else
                  torch.as_tensor(young["restricted_values"], dtype=dtype,
                                  device=device).reshape(rows, copies, tableaux))
    generators = tuple((
        torch.as_tensor(diagonal, dtype=dtype, device=device),
        torch.as_tensor(off_diagonal, dtype=dtype, device=device),
        torch.as_tensor(partner, dtype=torch.long, device=device),
    ) for diagonal, off_diagonal, partner in young["adjacent_generators"])
    return restricted, generators


def transported_young_slice(young, representative, gamma, dtype, device,
                            prepared=None, coset_index=None):
    """Return one Young synthesis slice without storing an orbit-sized table."""
    import torch

    from ye3t.representations.young_orthogonal import _adjacent_word_to_permutation

    if "values" in young:
        if coset_index is None:
            raise ValueError("Legacy Young transport needs the saved coset index.")
        bound = (bind_young_restriction(young, dtype, device)
                 if prepared is None else prepared)
        return bound[int(coset_index), :, int(gamma), :]
    restricted, generators = (bind_young_restriction(young, dtype, device)
                              if prepared is None else prepared)
    if restricted is None:
        rows = int(young["local_tableau_dim"])
        copies = int(young["gamma_count"])
        tableaux = int(young["target_tableau_dim"])
        if rows != 1 or copies != tableaux or not 0 <= int(gamma) < copies:
            raise ArithmeticError("Matrix-unit Young restriction has invalid axes.")
        result = torch.zeros((1, tableaux), dtype=dtype, device=device)
        result[0, int(gamma)] = float(young["restricted_scale"])
    else:
        result = restricted[:, int(gamma), :]
    for generator_index in reversed(_adjacent_word_to_permutation(
        tuple(int(value) for value in representative)
    )):
        diagonal, off_diagonal, partner = generators[generator_index]
        result = result * diagonal + result.index_select(-1, partner) * off_diagonal
    return result
