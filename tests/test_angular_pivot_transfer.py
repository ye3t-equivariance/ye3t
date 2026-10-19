"""Exact angular copy coordinates against full native matrix units."""

from itertools import product

import pytest

from ye3t._optional_sympy import sp
from ye3t.representations import builder, coset_units
from ye3t.representations.generalized_irreps import Partition
from ye3t.representations.young_sectors import generalized_sector_counts


CASES = [(3, 1, (2, 1), 1), (3, 2, (2, 1), 2),
         (4, 2, (2, 2), 0), (4, 1, (3, 1), 1),
         (3, 1, (1, 1, 1), 0), (4, 1, (4,), 0)]


@pytest.mark.parametrize("size,angular_l,partition,weight", CASES)
@pytest.mark.parametrize("normalize", [False, True])
def test_angular_pivot_transfer_matches_full_native_units(size, angular_l, partition, weight, normalize):
    nin, lin = (0,)*size, (angular_l,)*size
    labeler = builder.GeneralizedExactSymbolicLabeler(nin, lin, spatial_symmetry="O3")
    irrep = labeler.permutation_irrep((Partition(partition),))
    dimension = int(irrep.partitions[0].dimension)
    states = tuple(word for word in product(range(-angular_l, angular_l+1), repeat=size)
                   if sum(word) == weight)
    raised = tuple(word for word in product(range(-angular_l, angular_l+1), repeat=size)
                   if sum(word) == weight+1)
    raising = builder._weight_raising_matrix(lin, states, raised)
    units = builder._selected_subgroup_matrix_units_for_factor_native(
        irrep.subgroup.factors[0], irrep.partitions[0], tuple(range(size)), states,
        {(tableau, 0) for tableau in range(dimension)})
    reference_basis = sp.Matrix.hstack(*sp.Matrix(units[0, 0].matrix).columnspace())
    coefficients = builder._exact_independent_basis(tuple(
        sp.Matrix(column) for column in builder._exact_domain_nullspace(raising*reference_basis)))
    reference = {}
    for copy, coefficient in enumerate(coefficients):
        vector = builder._pivot_normalize_vector(reference_basis*coefficient)
        for tableau in range(dimension):
            result = units[tableau, 0].matrix*vector
            reference[copy, tableau] = builder._pivot_normalize_vector(result) if normalize else result
    actual, count = builder._angular_scalar_vectors_coset(
        states, raising, tuple(range(size)), angular_l, irrep.partitions[0], dimension,
        normalize_tableaux=normalize)
    expected_count = generalized_sector_counts(nin, lin, permutation_irrep=irrep,
                                               count_only=True, spatial_symmetry="O3")
    assert count == len(coefficients) == int(expected_count.counts_by_L.get(weight, 0))
    assert set(actual) == set(reference)
    for key in reference:
        assert (actual[key]-reference[key]).applyfunc(sp.simplify) == sp.zeros(len(states), 1)
        assert (raising*actual[key]).applyfunc(sp.simplify) == sp.zeros(len(raised), 1)


def test_angular_construction_only_builds_selected_matrix_unit_columns(monkeypatch):
    size, angular_l = 4, 1
    labeler = builder.GeneralizedExactSymbolicLabeler((0,)*size, (angular_l,)*size, spatial_symmetry="O3")
    irrep = labeler.permutation_irrep((Partition((4,)),))

    def forbidden(*args, **kwargs):
        raise AssertionError("Angular pivot construction must not build full word tables or all columns")

    monkeypatch.setattr(coset_units, "class_word_table", forbidden)
    monkeypatch.setattr(coset_units, "unit_columns", forbidden)
    states, vectors, count = builder._exact_single_factor_scalar_vectors(
        (0,)*size, (angular_l,)*size, irrep, backend="exact_coset")
    assert count == 1
    assert len(vectors) == 1
    assert len(states) == 81
