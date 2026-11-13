"""Sparse application preserves exact coefficients and rectangular axes."""
import pytest


@pytest.mark.parametrize('partition', [(4,), (1, 1, 1, 1), (3, 2, 1)])
def test_selected_generator_action_matches_complete_matrix(partition):
    import sympy as sp
    from ye3t.representations.projectors import (
        _apply_sparse_action, _sparse_action_columns,
        adjacent_transposition_representation_matrix,
    )
    for generator in range(sum(partition) - 1):
        matrix = adjacent_transposition_representation_matrix(partition, generator)
        width = matrix.rows
        values = sp.SparseMatrix(width, 3, {
            (0, 0): sp.sqrt(2), (width-1, 1): 1+sp.I,
            (width//2, 2): sp.Rational(3, 7),
        })
        columns = _sparse_action_columns(matrix)
        assert max(map(len, columns.values())) <= 2
        actual = _apply_sparse_action(columns, values, width)
        assert (actual - matrix*values).applyfunc(sp.expand) == sp.zeros(width, 3)
        empty = _apply_sparse_action(columns, sp.SparseMatrix(width, 0, {}), width)
        assert empty.shape == (width, 0)


def test_sparse_action_never_visits_unused_ambient_components():
    import sympy as sp
    from ye3t.representations.projectors import _apply_sparse_action
    class SelectedColumns(dict):
        def get(self, source, default=None):
            assert source == 4000
            return super().get(source, default)
        def items(self):
            raise AssertionError('Action scanned all ambient components')
    action = SelectedColumns({4000: ((3, sp.Rational(1, 3)),
                                     (4000, 2*sp.sqrt(2)/3))})
    values = sp.SparseMatrix(5000, 2, {(4000, 1): sp.sqrt(2)})
    expected = sp.SparseMatrix(5000, 2, {(3, 1): sp.sqrt(2)/3, (4000, 1): sp.Rational(4, 3)})
    assert _apply_sparse_action(action, values, 5000) == expected
