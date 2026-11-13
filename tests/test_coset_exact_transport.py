"""Exact arithmetic agrees with the previous symbolic transport through rank 8."""
from itertools import permutations

import pytest

from ye3t.representations import coset_units as core
from ye3t.representations.generalized_irreps import Partition


@pytest.mark.parametrize("partition,w0", [
    ((4, 2), (0, 0, 0, 0, 1, 2)),
    ((6, 1), (-2, -1, 0, 0, 0, 1, 2)),
    ((6, 2), (0, 0, 0, 0, 0, 0, 1, 1)),
    ((6, 1, 1), (0, 0, 0, 0, 0, 0, 1, 2)),
    ((4, 4), (0, 0, 0, 0, 1, 1, 1, 1)),
    ((5, 2, 1), (0, 0, 0, 0, 0, 1, 1, 2)),
])
@pytest.mark.parametrize('reverse_targets', [False, True])
def test_exact_transport_matches_symbolic_matrix_products(partition, w0, reverse_targets):
    sp = core._sympy()
    size, dimension = len(w0), int(Partition(partition).dimension)
    words = tuple(sorted(set(permutations(w0))))
    generators = core.adjacent_transposition_generators(partition, size)
    stabilizer = core.class_stabilizer_sum(w0, dimension, generators)
    source = next(word for word in words if any(core.class_projected_row(
        w0, word, dimension, generators, stabilizer)))
    if reverse_targets:
        words = tuple(reversed(words))
    initial = core.class_projected_row(w0, source, dimension, generators, stabilizer).T
    reference = {t: {} for t in range(dimension)}
    for index, target in enumerate(words):
        column = initial
        for adjacent in core._word_adjacent_path(w0, target):
            column = generators[adjacent] * column
        for t in range(dimension):
            value = sp.simplify(sp.Rational(dimension, core.factorial(size))*column[t])
            if value != 0:
                reference[t][index] = value
    assert any(reference.values())
    actual = core.selected_unit_column(dimension, size, words, w0, source,
        generators, stabilizer)
    assert actual == reference


@pytest.mark.parametrize("partition,w0", [
    ((3,2), (0,0,0,1,2)),
    ((4,2), (0,0,0,0,1,2)),
    ((5,2,1), (0,0,0,0,0,1,1,2)),
])
def test_reference_columns_and_selected_combinations_equal_complete_units(partition, w0):
    sp = core._sympy()
    size, dimension = len(w0), int(Partition(partition).dimension)
    words = tuple(sorted(set(permutations(w0))))
    generators = core.adjacent_transposition_generators(partition, size)
    stabilizer = core.class_stabilizer_sum(w0, dimension, generators)
    sources = []
    for word in words:
        row = core.class_projected_row(w0, word, dimension, generators, stabilizer)
        if any(row):
            sources.append((word,row))
        if len(sources) == 2:
            break
    complete = [core.selected_unit_column(dimension, size, words, w0, word,
        generators, stabilizer) for word, _ in sources]
    reference = core._reference_unit_columns(dimension, size, words, w0,
        tuple(row for _,row in sources), generators)
    assert reference == tuple(column[0] for column in complete)
    weights = (sp.Rational(2,3), -sp.sqrt(2))
    initial = sum((weight*row for weight, (_,row) in zip(weights,sources)),
                  sp.zeros(1,dimension))
    actual = core._unit_column_from_projected_row(dimension,size,words,w0,initial,generators)
    for tableau in range(dimension):
        expected = {}
        for column,weight in zip(complete,weights):
            for index,value in column[tableau].items():
                expected[index] = expected.get(index,0)+weight*value
        expected = {index:sp.simplify(value) for index,value in expected.items()
                    if sp.simplify(value) != 0}
        assert actual[tableau] == expected
