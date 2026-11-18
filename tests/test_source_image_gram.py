"""Independent exact fixtures for source-image planning and reconstruction."""

from fractions import Fraction

import pytest
import sympy as sp

from ye3t.couplings.source_image_gram import _ExactSourceImageGram
from ye3t.exact_scalars import ExactRadical


@pytest.mark.parametrize("columns, selected", [
    ([{0: 1}, {1: 1}], [0, 1]),
    ([{}, {}], []),
    ([{0: 1}, {0: 1, 1: 1}], [0, 1]),
    ([{0: 1, 1: 1}], [0]),
    ([{0: 1}, {1: 2}, {0: 3, 1: 4}], [0, 1]),
])
def test_exact_images_and_kernel_relations(columns, selected):
    result = _ExactSourceImageGram()
    for column in columns:
        result.append(column)
    assert result.selected == selected
    assert result.validate_reconstruction(columns)
    for output, source in enumerate(selected):
        assert result.reconstructions[source] == {output: 1}
    # An independent exact matrix reference checks all kernel directions.
    keys = sorted(set().union(*(set(column) for column in columns)))
    K = sp.Matrix([[column.get(key, 0) for column in columns] for key in keys])
    assert K.rank() == len(selected)


def test_complex_conjugation_and_positive_metric():
    columns = [{0: sp.Integer(1)}, {0: sp.I}, {1: sp.Integer(1)}]
    for weights in ({0: 2, 1: 3}, {0: 5, 1: 7}):
        result = _ExactSourceImageGram(weights=weights, scalar=sp.expand)
        for column in columns:
            result.append(column)
        assert result.selected == [0, 2]
        assert result.reconstructions[1] == {0: sp.I}
        assert result.validate_reconstruction(columns)


def test_radical_hermitian_ldl_and_metric_independent_selection():
    root = ExactRadical.sqrt(2)
    columns = [{0: root, 1: 1}, {0: 1, 1: root},
               {0: root+2, 1: 1+2*root}, {2: 1}]
    for weights in ({}, {0: Fraction(3, 7), 1: 5, 2: 2}):
        result = _ExactSourceImageGram(weights=weights)
        for column in columns:
            result.append(column)
        assert result.selected == [0, 1, 3]
        assert result.reconstructions[2] == {0: 1, 1: 2}
        assert result.validate_reconstruction(columns)
        L = sp.eye(len(result.selected))
        for i, row in enumerate(result.lower):
            for j, value in row.items():
                L[i, j] = value._sympy_()
        F = sp.Matrix([[sp.sympify(columns[index].get(key, 0)) for index in result.selected]
                       for key in range(3)])
        W = sp.diag(*(weights.get(key, 1) for key in range(3)))
        D = sp.diag(*(value._sympy_() for value in result.diagonal))
        assert (L*D*L.conjugate().T-F.conjugate().T*W*F).applyfunc(sp.simplify) == sp.zeros(3)


def test_nonorthogonal_complex_columns_reconstruct_with_conjugate_transposes():
    columns = [{0: 1, 1: sp.I}, {0: 2*sp.I, 1: 3},
               {0: 1-2*sp.I, 1: 2*sp.I-3}]
    result = _ExactSourceImageGram(weights={0: 2, 1: 7}, scalar=sp.expand)
    for column in columns:
        result.append(column)
    assert result.selected == [0, 1]
    assert result.validate_reconstruction(columns)


def test_source_image_bounds_and_inexact_inputs_are_rejected():
    with pytest.raises(ArithmeticError, match="positive"):
        _ExactSourceImageGram(weights={0: -1})
    with pytest.raises(TypeError, match="floating"):
        _ExactSourceImageGram().append({0: 1.0})
    with pytest.raises(MemoryError, match="maximum_loader_symbolic_cells"):
        _ExactSourceImageGram(maximum_cells=1).append({0: 1})
