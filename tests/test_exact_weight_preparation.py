"""Exact preparation preserves symbolic gauges and ordered raising maps."""
import pytest

from ye3t.representations import builder


@pytest.mark.parametrize('coefficient_domain', ['real', 'complex', 'symbolic'])
def test_exact_tableau_action_check_rejects_corrupted_vector(coefficient_domain):
    from ye3t.representations import coset_units
    states = ((0,0,1), (0,1,0), (1,0,0))
    generators = coset_units.adjacent_transposition_generators((2,1), 3)
    stabilizer = coset_units.class_stabilizer_sum(states[0], 2, generators)
    source = next(word for word in states if any(coset_units.class_projected_row(
        states[0], word, 2, generators, stabilizer)))
    columns = coset_units.selected_unit_column(2, 3, states, states[0],
        source, generators, stabilizer)
    vectors = tuple({states[row]: value for row, value in columns[t].items()} for t in range(2))
    sp = builder._sympy()
    scale = {'real': sp.sqrt(2), 'complex': sp.I, 'symbolic': sp.Symbol('x')}[coefficient_domain]
    vectors = tuple({word: value*scale for word, value in vector.items()} for vector in vectors)
    builder._check_scalar_tableau_actions(vectors, states, (0,1,2), generators)
    vectors[0][states[0]] = vectors[0].get(states[0], 0) + 1
    with pytest.raises(RuntimeError, match='intertwining'):
        builder._check_scalar_tableau_actions(vectors, states, (0,1,2), generators)


def test_repeated_vector_normalization_retains_original_symbolic_gauge():
    sp = builder._sympy()
    x = sp.Symbol('x')
    for values in ([0, sp.sqrt(8), sp.sqrt(2), 0]*50,
                   [0, sp.sqrt(2)+sp.sqrt(3), sp.sqrt(6)-sp.sqrt(2)]*20,
                   [0, sp.Rational(1,7)*sp.sqrt(6), -sp.Rational(3,11)*sp.sqrt(10)]*20,
                   [0, x+x, x, x-x]*20,
                   [0, sp.I*sp.sqrt(3), sp.I*sp.sqrt(12)]*20,
                   [sp.Float(0.2),sp.Float(0.4)]*20, [0]*40):
        entries = [sp.simplify(value) for value in values]
        pivot = next((index for index,value in enumerate(entries) if value != 0),len(entries))
        expected = sp.Matrix(values) if pivot==len(entries) else sp.Matrix([
            sp.simplify(value/entries[pivot]) for value in entries])
        expected_key = (pivot,()) if pivot==len(entries) else (
            pivot,tuple(sp.srepr(value) for value in expected))
        assert builder._pivot_normalize_vector(sp.Matrix(values)) == expected
        assert builder._normalized_vector_key(sp.Matrix(values)) == expected_key


@pytest.mark.parametrize('angular', [(1,), (1,2), (1,1,2), (1,1,1,1)])
def test_requested_raising_equals_full_tensor_generator(angular):
    states = builder._raw_tensor_basis_states(angular)
    _, raising = builder._total_so3_generators(angular)
    for magnetic in (-1,0,1):
        sources = [index for index,state in enumerate(states) if sum(state)==magnetic]
        targets = [index for index,state in enumerate(states) if sum(state)==magnetic+1]
        actual = builder._weight_raising_matrix(angular,
            [states[index] for index in sources], [states[index] for index in targets])
        assert actual == raising.extract(targets,sources)


def test_scalar_compilation_keeps_raising_products_sparse(monkeypatch):
    sp = builder._sympy()
    from ye3t.representations.generalized_irreps import Partition
    original_builder = builder._weight_raising_matrix
    original_multiply = sp.MutableSparseMatrix.multiply
    original_exact = builder._exact_matrix_product
    operators, products, exact_checks = [], [], []

    def record_operator(*args, **kwargs):
        matrix = original_builder(*args, **kwargs)
        operators.append(matrix)
        return matrix

    def checked_multiply(matrix, other, *args, **kwargs):
        if any(matrix is candidate for candidate in operators):
            assert isinstance(other, sp.MutableSparseMatrix)
            products.append(other.shape)
        return original_multiply(matrix, other, *args, **kwargs)

    def checked_exact(matrix, other):
        if any(matrix is candidate for candidate in operators):
            assert isinstance(matrix, sp.MutableSparseMatrix)
            assert isinstance(other, sp.MutableSparseMatrix)
            exact_checks.append(other.shape)
        return original_exact(matrix, other)

    monkeypatch.setattr(builder, '_weight_raising_matrix', record_operator)
    monkeypatch.setattr(sp.MutableSparseMatrix, 'multiply', checked_multiply)
    monkeypatch.setattr(builder, '_exact_matrix_product', checked_exact)
    labeler = builder.GeneralizedExactSymbolicLabeler((0,)*4, (1,)*4, spatial_symmetry='O3')
    irrep = labeler.permutation_irrep((Partition((2,2)),))
    _, vectors, multiplicity = builder._exact_single_factor_scalar_vectors(
        labeler.nin, labeler.lin, irrep, backend='exact_coset')
    assert multiplicity == 1
    assert len(vectors) == 2
    assert products  # The restricted raising map remains sparse.
    assert sum(columns for rows, columns in exact_checks) == 2
    assert len(exact_checks) == 1  # Both complete tableaux share one operator preparation.


def test_complete_tableau_raising_check_rejects_corrupted_scalar(monkeypatch):
    from ye3t.representations.generalized_irreps import Partition
    original = builder._angular_scalar_vectors_coset

    def corrupt(states, raising, *args, **kwargs):
        vectors, count = original(states, raising, *args, **kwargs)
        column = next(column for row,column in raising.todok())
        key = next(iter(vectors))
        vectors[key][column,0] += 1
        return vectors,count

    monkeypatch.setattr(builder, '_angular_scalar_vectors_coset', corrupt)
    labeler = builder.GeneralizedExactSymbolicLabeler((0,)*4,(1,)*4,spatial_symmetry='O3')
    irrep = labeler.permutation_irrep((Partition((2,2)),))
    with pytest.raises(RuntimeError,match='not annihilated'):
        builder._exact_single_factor_scalar_vectors(labeler.nin,labeler.lin,irrep,
                                                    backend='exact_coset')
