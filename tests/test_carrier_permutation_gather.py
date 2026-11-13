"""Exact carrier checks use row permutations without square raw-space arrays."""
from itertools import product
import copy

import pytest

from ye3t.couplings import lifted_cauchy_scalar as scalar


def test_sparse_exact_proof_matrix_roundtrip_and_bounded_storage():
    sp = scalar._sympy()
    matrix = sp.SparseMatrix(1750, 1750, {(0, 0): 1, (3, 7): sp.I,
        (7, 3): -sp.I, (1749, 1749): sp.sqrt(2)+sp.Rational(3, 7)})
    payload = scalar._exact_matrix_payload(matrix, sparse=True)
    assert payload['shape'] == (1750, 1750)
    assert len(payload['entries']) == 4
    assert scalar._exact_matrix_from_payload(payload) == matrix
    assert scalar._exact_matrix_from_payload(scalar._exact_matrix_payload(sp.eye(3))) == sp.eye(3)
    assert scalar._exact_matrix_from_payload(scalar._exact_matrix_payload(sp.zeros(0, 3), sparse=True)).shape == (0, 3)
    for changed in ({**payload, 'shape': (-1, 1750)},
                    {**payload, 'entries': (*payload['entries'], payload['entries'][0])},
                    {**payload, 'entries': ((1750, 0, payload['entries'][0][2]),)},
                    {**payload, 'unexpected': True}):
        with pytest.raises(ValueError, match='Sparse exact matrix'):
            scalar._exact_matrix_from_payload(changed)


def test_restricted_actions_use_same_exact_pivots_as_full_raw_space(monkeypatch):
    from sympy.polys.matrices import DomainMatrix
    sp = scalar._sympy()
    states = tuple(product(range(3), repeat=4))
    basis = sp.SparseMatrix([[state[j]-state[3] for j in range(3)] for state in states])
    basis = basis * sp.Matrix([[sp.sqrt(2), 1, 0], [0, sp.sqrt(3), 1], [1, 0, 1]])
    domain = basis.T.to_DM(extension=True)
    _, pivots = domain.rref()
    pivot_inverse = domain.extract(range(3), pivots).transpose().to_field().inv().to_Matrix()
    expected = []
    for adjacent in range(3):
        rows = scalar._adjacent_permutation_rows(states, 4, adjacent)
        expected.append((pivot_inverse*basis.extract(rows, range(3)).extract(pivots, range(3)))
                        .applyfunc(sp.simplify))
    rref = DomainMatrix.rref
    shapes = []

    def recorded(matrix, *args, **kwargs):
        shapes.append(matrix.shape)
        return rref(matrix, *args, **kwargs)

    monkeypatch.setattr(DomainMatrix, "rref", recorded)
    # Require the production sparse checker and forbid a full permuted copy
    # or full algebraic-field conversion during action recovery.
    monkeypatch.setattr(scalar, '_DENSE_CARRIER_ENTRY_LIMIT', 1)
    extract = sp.MutableSparseMatrix.extract
    to_domain = sp.MutableSparseMatrix.to_DM
    converted = []
    def bounded_extract(matrix, rows, columns):
        rows, columns = tuple(rows), tuple(columns)
        if matrix.shape == basis.shape:
            assert len(rows) < len(states)
        return extract(matrix, rows, columns)
    def bounded_domain(matrix, *args, **kwargs):
        converted.append(matrix.shape)
        assert max(matrix.shape) < len(states)
        return to_domain(matrix, *args, **kwargs)
    monkeypatch.setattr(sp.MutableSparseMatrix, 'extract', bounded_extract)
    monkeypatch.setattr(sp.MutableSparseMatrix, 'to_DM', bounded_domain)
    prepare_rows = scalar._exact_sparse_rows
    prepared = []
    def observe_rows(matrix):
        prepared.append(matrix.shape)
        return prepare_rows(matrix)
    monkeypatch.setattr(scalar, '_exact_sparse_rows', observe_rows)
    actual = scalar._restricted_carrier_actions(states, tuple(basis[:, j] for j in range(3)), 4)
    assert all((a-b).applyfunc(sp.simplify) == sp.zeros(3) for a, b in zip(actual, expected))
    assert shapes and max(shape[1] for shape in shapes) < len(states)
    assert converted
    assert prepared.count(basis.shape) == 1
    assert prepared.count((3, 3)) == 3  # Every adjacent generator is checked.


@pytest.mark.parametrize("independent", [True, False])
def test_carrier_pivot_prefix_expands_until_full_rank_or_all_nonzero_rows(independent, monkeypatch):
    from sympy.polys.matrices import DomainMatrix
    sp = scalar._sympy()
    states = tuple((i,) for i in range(100))
    first = sp.SparseMatrix([1]*100)
    second = sp.SparseMatrix([2]*99+[3 if independent else 2])
    rref = DomainMatrix.rref
    widths = []

    def recorded(matrix, *args, **kwargs):
        widths.append(matrix.shape[1])
        return rref(matrix, *args, **kwargs)

    monkeypatch.setattr(DomainMatrix, "rref", recorded)
    if independent:
        assert scalar._restricted_carrier_actions(states, (first, second), 1) == ()
    else:
        with pytest.raises(RuntimeError, match="not independent"):
            scalar._restricted_carrier_actions(states, (first, second), 1)
    assert widths == [32, 64, 100]


def test_nested_carrier_hash_streams_collections_without_changing_identity(monkeypatch):
    from collections import UserDict
    sp = scalar._sympy()
    columns = tuple((row, column, scalar._exact_scalar_payload(value))
        for row, column, value in ((0, 0, sp.sqrt(2)), (3, 1, sp.I), (7, 2, sp.Rational(2, 3))))
    states = tuple((i, -i) for i in range(8))
    carrier = {'basis_columns_sparse': columns, 'state_order': states,
               'gram': scalar._exact_matrix_payload(sp.diag(2, 1, sp.Rational(4, 9)))}
    payload = {'angular_carrier': carrier, 'nested': UserDict({2: [None, True, -0.0, '\u03bb']}),
               'empty': (), 'text': 'quoted " value'}
    original = scalar._freeze_json
    expected = scalar._stable_hash(original(payload))
    forbidden_ids = {id(value) for value in (payload, carrier, columns, states, carrier['gram'])}

    def bounded_freeze(value):
        assert id(value) not in forbidden_ids, 'Copied a complete carrier collection for hashing'
        return original(value)

    monkeypatch.setattr(scalar, '_freeze_json', bounded_freeze)
    assert scalar._stable_record_hash(payload) == expected
    payload['text'] += ' changed'
    assert scalar._stable_record_hash(payload) != expected
    with pytest.raises(ValueError):
        scalar._stable_record_hash({'nested': {'invalid': float('nan')}})


def test_canonical_encoder_reuse_preserves_threaded_identities_and_bounds():
    from concurrent.futures import ThreadPoolExecutor
    import hashlib
    import json

    payloads = [{"repeated": [{"value": i, "zero": -0.0, "flag": bool(i % 2),
                              "unicode_\u03ba": "\u03bb\n\""}], "long_"+"x"*256: i}
                for i in range(32)]
    expected = [hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=True, allow_nan=False).encode()).hexdigest() for payload in payloads]
    scalar._encoded_short_field.cache_clear()
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(scalar._stable_record_hash, payloads))
    assert actual == expected
    cached = scalar._encoded_short_field.cache_info()
    assert cached.hits > 0
    # Five short schema fields; the long application key must not be retained.
    assert cached.currsize == 5
    payloads[0]["repeated"][0]["value"] = 99
    assert scalar._stable_record_hash(payloads[0]) != expected[0]
    for i in range(1100):
        scalar._stable_record_hash({"distinct_field_"+str(i): i})
    assert scalar._encoded_short_field.cache_info().currsize == 1024


def test_compiled_carrier_and_template_hashes_match_full_canonical_encoding():
    payload = scalar._build_block_template((2, 3, (2, 1), 1, 1))['payload']
    for record, field in ((payload, 'template_hash'),
                          (payload['angular_carrier'], 'carrier_hash'),
                          (payload['role_carrier'], 'carrier_hash')):
        body = {key: value for key, value in record.items() if key != field}
        assert record[field] == scalar._stable_hash(scalar._freeze_json(body))


def test_streamed_monomial_hash_preserves_exact_records_and_canonical_keys(monkeypatch):
    from collections import UserDict
    sp = scalar._sympy()
    terms = [UserDict({'coordinates': ((0, 1, 2), (2, 3, 1)),
                      'coefficient': scalar._exact_scalar_payload(coefficient)})
             for coefficient in (sp.sqrt(2), sp.I/3, -sp.sqrt(3)/7)]
    terms[0].update(occupation=(((0, 1, 2), 1), ((2, 3, 1), 1)), orbit_size=2,
                    degree_factorial=2, occupation_factorial_product=1,
                    lower_degree_occupations=((),),
                    normalized_coefficient_semantics='raw/sqrt(orbit_size)',
                    raw_coefficient_semantics='sum_of_ordered_coefficients')
    payload = {'sources': [{'terms': terms}], 'metadata': {2: ['\u03ba', -0.0, True]}}
    original = scalar._freeze_json
    expected = scalar._stable_hash(original(payload))
    forbidden_ids = {id(payload), id(payload['sources']), id(payload['sources'][0]), id(terms)}

    def bounded_freeze(value):
        assert id(value) not in forbidden_ids, 'Copied an enclosing compiler collection'
        return original(value)

    monkeypatch.setattr(scalar, '_freeze_json', bounded_freeze)
    assert scalar._stable_record_hash(payload) == expected
    terms[0]['coefficient'] = scalar._exact_scalar_payload(sp.sqrt(3))
    assert scalar._stable_record_hash(payload) != expected
    with pytest.raises(ValueError):
        scalar._stable_record_hash({'coordinates': (), 'coefficient': float('nan')})


def test_streamed_hash_retains_lazy_source_sequence_support():
    from collections.abc import Sequence

    class Sources(Sequence):
        def __len__(self):
            return 3

        def __getitem__(self, index):
            if not 0 <= index < len(self):
                raise IndexError(index)
            return {'index': index, 'values': (index, index+1)}

    sources = Sources()
    expected = scalar._stable_hash(scalar._freeze_json({'sources': tuple(sources)}))
    assert scalar._stable_record_hash({'sources': sources}) == expected


@pytest.mark.parametrize('storage', ['dense', 'sparse', 'immutable_sparse'])
def test_single_pass_carrier_columns_preserve_all_entries_and_order(storage):
    sp = scalar._sympy()
    matrix = {'dense': sp.Matrix, 'sparse': sp.SparseMatrix,
              'immutable_sparse': sp.ImmutableSparseMatrix}[storage]
    vectors = tuple(matrix(column) for column in (
        [0, sp.sqrt(2), 0, sp.Rational(3, 7)],
        [sp.I, 0, sp.sqrt(3), 0], [0, 0, 0, 0]))
    expected = sp.Matrix.hstack(*vectors)
    actual = scalar._sparse_carrier_columns(iter(vectors))
    assert isinstance(actual, sp.MutableSparseMatrix)
    assert actual == expected
    assert scalar._sparse_carrier_columns(reversed(vectors)) == expected[:, ::-1]


@pytest.mark.parametrize('kind', ['real_radicals', 'complex', 'symbolic'])
def test_metric_dual_cached_simplification_matches_symbolic_reference(kind):
    sp = scalar._sympy()
    x = sp.Symbol('x', real=True)
    synthesis = sp.Matrix({
        'real_radicals': [[sp.sqrt(2), 1], [1, sp.sqrt(3)], [sp.sqrt(5), sp.sqrt(7)]],
        'complex': [[1, sp.I], [sp.sqrt(2), 1], [sp.I, 2*sp.sqrt(3)]],
        'symbolic': [[x, 1], [1, 0], [0, 1]],
    }[kind])
    reference_gram = sp.simplify(synthesis.conjugate().T*synthesis)
    reference_analysis = sp.simplify(reference_gram.inv()*synthesis.conjugate().T)
    gram, analysis = scalar._metric_dual_analysis(sp.SparseMatrix(synthesis))
    assert (gram-reference_gram).applyfunc(sp.simplify) == sp.zeros(*gram.shape)
    assert (analysis-reference_analysis).applyfunc(sp.simplify) == sp.zeros(*analysis.shape)
    assert (analysis*synthesis).applyfunc(sp.simplify) == sp.eye(synthesis.cols)
    with pytest.raises(RuntimeError, match='not independent'):
        scalar._metric_dual_analysis(sp.Matrix([[1, 2], [2, 4], [3, 6]]))


def test_radical_metric_inverse_matches_exact_ldl_reference():
    sp = scalar._sympy()
    lower = sp.Matrix([[1, 0, 0], [sp.sqrt(2), 1, 0],
                       [sp.sqrt(3)+sp.sqrt(5), sp.sqrt(7), 1]])
    gram = lower * sp.diag(2, 3, 5) * lower.T
    inverse = scalar._radical_gram_inverse(gram)
    assert inverse is not None
    assert (gram*inverse).applyfunc(sp.simplify) == sp.eye(3)
    assert (inverse-gram.inv()).applyfunc(sp.simplify) == sp.zeros(3)
    with pytest.raises(RuntimeError, match='not independent'):
        scalar._radical_gram_inverse(sp.Matrix([[1, 1], [1, 1]]))


def test_radical_metric_inverse_preserves_other_scalar_domains():
    sp = scalar._sympy()
    for gram in (sp.Matrix([[sp.Float('1.5')]]),
                 sp.Matrix([[sp.Symbol('x')]]),
                 sp.Matrix([[2, sp.I], [-sp.I, 2]]),
                 sp.Matrix([[1+sp.sqrt(2)+sp.sqrt(3)]])):
        assert scalar._radical_gram_inverse(gram) is None


@pytest.mark.parametrize('kind', ['radicals', 'complex', 'symbolic', 'floating'])
def test_metric_product_preserves_complete_exact_product(kind):
    sp = scalar._sympy()
    value = {'radicals': sp.sqrt(2)+sp.sqrt(3), 'complex': sp.I,
             'symbolic': sp.Symbol('x'), 'floating': sp.Float('1.25')}[kind]
    left = sp.SparseMatrix([[value, 0, 2], [0, 1, value]])
    right = sp.SparseMatrix([[value, 1], [2, 0], [0, -value]])
    assert scalar._metric_product(left, right) == (left*right).applyfunc(scalar._simplified_scalar)


def test_large_sparse_exact_product_avoids_nonzero_dictionary_copy(monkeypatch):
    sp = scalar._sympy()
    left = sp.SparseMatrix(100000, 3, {(5, 0): sp.sqrt(2), (99999, 2): 3})
    right = sp.SparseMatrix([[sp.sqrt(2), 0], [0, 1], [0, sp.Rational(1, 3)]])

    def forbidden(*args, **kwargs):
        raise AssertionError('Exact multiplication copied the full nonzero dictionary')

    monkeypatch.setattr(sp.MutableSparseMatrix, 'todok', forbidden)
    actual = scalar._metric_product(left, right)
    assert isinstance(actual, sp.MutableSparseMatrix)
    assert actual == sp.SparseMatrix(100000, 2, {(5, 0): 2, (99999, 1): 1})


def test_metric_dual_reuse_checks_exact_input_and_isolates_results(monkeypatch):
    sp = scalar._sympy()
    monkeypatch.setenv('YE3T_CACHE_MODE', 'auto')
    monkeypatch.setenv('YE3T_CACHE_VERIFY', 'hash')
    scalar._COMPUTED_METRIC_DUALS.clear()
    synthesis = sp.Matrix([[1, sp.sqrt(2)], [0, 1], [1, 0]])
    original = scalar._radical_gram_inverse
    calls = []

    def observe(gram):
        calls.append(gram.shape)
        return original(gram)

    monkeypatch.setattr(scalar, '_radical_gram_inverse', observe)
    gram, analysis = scalar._metric_dual_analysis(synthesis)
    gram[0,0] += 17
    analysis[0,0] += 13
    fresh_gram, fresh_analysis = scalar._metric_dual_analysis(synthesis)
    assert len(calls) == 1
    assert fresh_gram == sp.simplify(synthesis.T*synthesis)
    assert sp.simplify(fresh_analysis*synthesis) == sp.eye(2)
    changed = synthesis.copy()
    changed[0,0] += sp.Rational(1, 10**50)
    scalar._metric_dual_analysis(changed)
    assert len(calls) == 2
    for mode, verify in [('auto','full'), ('off','hash'), ('rebuild','hash')]:
        monkeypatch.setenv('YE3T_CACHE_MODE', mode)
        monkeypatch.setenv('YE3T_CACHE_VERIFY', verify)
        scalar._metric_dual_analysis(synthesis)
    assert len(calls) == 5


def test_single_pass_carrier_columns_empty_and_invalid_shapes():
    sp = scalar._sympy()
    assert scalar._sparse_carrier_columns(()) == sp.zeros(0, 0)
    with pytest.raises(ValueError, match='common row count'):
        scalar._sparse_carrier_columns((sp.zeros(2, 1), sp.zeros(3, 1)))
    with pytest.raises(ValueError, match='one column'):
        scalar._sparse_carrier_columns((sp.eye(2),))


@pytest.mark.parametrize('sparse', [False, True])
def test_direct_carrier_vector_decode_preserves_columns_and_exact_entries(sparse):
    sp = scalar._sympy()
    matrix = sp.Matrix([[0, sp.sqrt(2), 0], [sp.I, 0, 0],
                        [sp.Rational(3,7), sp.sqrt(3), 0]])
    coordinates = ((2,1,-1), (0,1,0), (1,0,1))
    if sparse:
        payload = {'storage':'sparse', 'basis_columns_sparse': tuple(
            (row, col, scalar._exact_scalar_payload(value))
            for (row,col),value in reversed(list(matrix.todok().items())))}
    else:
        payload = {'basis_columns':scalar._exact_matrix_payload(matrix)}
    vectors = scalar._basis_vectors_from_carrier_payload(payload, 3, coordinates)
    assert tuple(vectors) == coordinates
    assert all(vector.shape == (3,1) for vector in vectors.values())
    assert sp.Matrix.hstack(*vectors.values()) == matrix
    vectors[coordinates[0]][1,0] = 17
    assert scalar._basis_matrix_from_carrier_payload(payload,3,3) == matrix


def test_direct_sparse_vector_decode_avoids_whole_matrix_and_checks_indices(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Sparse column decode must not allocate a full carrier matrix')
    monkeypatch.setattr(scalar, '_basis_matrix_from_carrier_payload', forbidden)
    value = scalar._exact_scalar_payload(1)
    payload = {'storage':'sparse', 'basis_columns_sparse': ((2,1,value),)}
    vectors = scalar._basis_vectors_from_carrier_payload(payload,3,((0,),(1,)))
    assert vectors[(0,)] == scalar._sympy().zeros(3,1)
    assert vectors[(1,)][2,0] == 1
    for row,column in ((-1,0),(3,0),(0,-1),(0,2)):
        with pytest.raises(ValueError, match='out of range'):
            scalar._basis_vectors_from_carrier_payload(
                {'storage':'sparse', 'basis_columns_sparse': ((row,column,value),)},
                3,((0,),(1,)))
    with pytest.raises(ValueError, match='unique'):
        scalar._basis_vectors_from_carrier_payload(payload,3,((0,),(0,)))


def test_small_complete_high_L_gram_avoids_common_field_construction(monkeypatch):
    sp = scalar._sympy()
    _states, vectors, multiplicity = scalar._angular_schur_vectors(
        3,2,(3,),6,angular_basis_backend='exact_weight_space_v1')
    assert multiplicity == 1
    basis = sp.SparseMatrix.hstack(*(vectors[key] for key in sorted(vectors)))
    assert basis.shape == (125,13)
    expected = sp.simplify(basis.conjugate().T*basis)
    def forbidden(*args,**kwargs):
        raise AssertionError('Complete high-L Gram must avoid common-field construction')
    monkeypatch.setattr(type(basis),'to_DM',forbidden)
    assert scalar._exact_carrier_gram(basis) == expected


def test_carrier_validation_rejects_consistently_reordered_full_columns():
    carrier = copy.deepcopy(scalar._build_block_template((2,3,(2,1),1,1))['payload']['angular_carrier'])
    coordinates = tuple(carrier['coordinate_order'])
    order = list(range(len(coordinates)))
    order[0],order[2] = order[2],order[0]
    basis = scalar._basis_matrix_from_carrier_payload(carrier,len(carrier['state_order']),len(order))
    gram = scalar._exact_matrix_from_payload(carrier['gram'])
    carrier['coordinate_order'] = tuple(coordinates[i] for i in order)
    carrier['basis_columns'] = scalar._exact_matrix_payload(basis[:,order])
    carrier['gram'] = scalar._exact_matrix_payload(gram.extract(order,order))
    carrier.pop('carrier_hash')
    carrier['carrier_hash'] = scalar._stable_hash(scalar._freeze_json(carrier))
    # Every selected local copy and the full Gram remain valid. The payload
    # still fails the explicit full-column/copy-order identity.
    with pytest.raises(ValueError, match='copy ordering'):
        scalar._validate_carrier_payload(carrier)


@pytest.mark.parametrize('gauge_entries', [(1,1), (2,3), ('sqrt2',3)])
def test_domain_pairing_matches_independent_dense_equations(gauge_entries):
    sp = scalar._sympy()
    role = (sp.Matrix([[-1,1],[0,1]]), sp.Matrix([[1,0],[1,-1]]))
    gauge = sp.diag(*(sp.sqrt(2) if value == 'sqrt2' else value for value in gauge_entries))
    angular = tuple(gauge*action*gauge.inv() for action in role)
    rows = []
    for left,right in zip(role,angular):
        for i in range(2):
            for j in range(2):
                rows.append([left[i,a]*right[j,b] - int(i == a and j == b)
                             for a in range(2) for b in range(2)])
    vectors = sp.Matrix(rows).nullspace()
    assert len(vectors) == 1
    expected = vectors[0].reshape(2,2)
    expected = sp.simplify(expected/sp.sqrt(sp.simplify((expected.T*expected).trace())))
    if next(value for value in expected if value != 0).could_extract_minus_sign():
        expected = -expected
    actual = scalar._invariant_pairing(role,angular)
    assert actual == expected
    for left,right in zip(role,angular):
        assert sp.simplify(left*actual*right.T-actual) == sp.zeros(2)
    with pytest.raises(RuntimeError, match='one invariant pairing'):
        scalar._invariant_pairing((sp.eye(2),),(sp.eye(2),))


def test_exact_scalar_decode_reuse_observes_coefficient_mutations():
    sp = scalar._sympy()
    value = sp.sqrt(2)/3 + sp.I*sp.sqrt(5)/7
    payload = scalar._exact_scalar_payload(value)
    scalar._exact_scalar_from_key.cache_clear()
    assert scalar._exact_scalar_from_payload(payload) == value
    assert scalar._exact_scalar_from_payload(copy.deepcopy(payload)) == value
    assert scalar._exact_scalar_from_key.cache_info().hits == 1
    payload['real'][0]['coefficient_numerator'] *= 2
    assert scalar._exact_scalar_from_payload(payload) == 2*sp.sqrt(2)/3 + sp.I*sp.sqrt(5)/7
    payload['imag'][0]['coefficient_denominator'] *= 3
    assert scalar._exact_scalar_from_payload(payload) == 2*sp.sqrt(2)/3 + sp.I*sp.sqrt(5)/21
    payload['binary64'] = (0.,0.)
    assert scalar._exact_scalar_from_payload(payload) == 2*sp.sqrt(2)/3 + sp.I*sp.sqrt(5)/21
    payload['real'][0]['radicand_denominator'] = 0
    with pytest.raises(ZeroDivisionError):
        scalar._exact_scalar_from_payload(payload)


@pytest.mark.parametrize('field', ['radical', 'complex', 'cubic'])
@pytest.mark.parametrize('large', [False, True])
def test_carrier_action_checks_complete_rows_and_rejects_corruption(field, large, monkeypatch):
    sp = scalar._sympy()
    if large:
        monkeypatch.setattr(scalar, '_DENSE_CARRIER_ENTRY_LIMIT', 0)
    scale = {'radical':sp.sqrt(2), 'complex':sp.I, 'cubic':sp.real_root(2,3)}[field]
    basis = sp.SparseMatrix([[scale,sp.sqrt(3)], [scale,-sp.sqrt(3)],
                             [0,0], [2*scale,3*sp.sqrt(3)],
                             [2*scale,-3*sp.sqrt(3)]])
    rows = (1,0,2,4,3)
    action = sp.diag(1,-1)
    assert sp.simplify(basis.extract(rows,(0,1))-basis*action) == sp.zeros(5,2)
    assert scalar._carrier_action_matches(basis,rows,action)
    damaged = basis.copy()
    damaged[2,1] = scale
    assert not scalar._carrier_action_matches(damaged,rows,action)
    damaged = basis.copy()
    damaged[-1,0] += 1
    assert not scalar._carrier_action_matches(damaged,rows,action)
    assert not scalar._carrier_action_matches(basis,rows,sp.eye(2))


@pytest.mark.parametrize('complex_entries', [False, True])
@pytest.mark.parametrize('sparse_products', [False, True])
def test_exact_carrier_gram_preserves_every_cross_copy_entry(complex_entries, sparse_products, monkeypatch):
    sp = scalar._sympy()
    if sparse_products:
        monkeypatch.setattr(scalar, '_DENSE_CARRIER_ENTRY_LIMIT', 0)
    phase = sp.I if complex_entries else sp.Integer(1)
    basis = sp.Matrix([
        [sp.sqrt(2), phase*sp.sqrt(3), sp.Rational(2,3)],
        [sp.sqrt(6), -phase*sp.sqrt(2), sp.sqrt(3)],
        [0, 0, 0],
        [sp.Rational(1,5), phase*sp.sqrt(6), -sp.sqrt(2)],
    ])
    expected = sp.simplify(basis.conjugate().T*basis)
    assert scalar._exact_carrier_gram(basis) == expected
    assert expected[0,1] != 0
    assert sp.simplify(expected - expected.conjugate().T) == sp.zeros(3)


def test_large_gram_preserves_general_exact_fields(monkeypatch):
    sp = scalar._sympy()
    monkeypatch.setattr(scalar, '_DENSE_CARRIER_ENTRY_LIMIT', 0)
    basis = sp.SparseMatrix([[sp.real_root(2,3),1], [1,sp.sqrt(3)], [0,0]])
    assert scalar._exact_carrier_gram(basis) == sp.simplify(basis.T*basis)


def test_computed_gram_reuse_checks_complete_matrix_and_isolates_results(monkeypatch):
    sp = scalar._sympy()
    monkeypatch.setenv('YE3T_CACHE_MODE', 'auto')
    monkeypatch.setenv('YE3T_CACHE_VERIFY', 'hash')
    scalar._COMPUTED_CARRIER_GRAMS.clear()
    basis = sp.SparseMatrix([[sp.sqrt(2), 1], [2, sp.sqrt(3)], [0, 1]])
    expected = sp.simplify(basis.T*basis)
    counter = scalar.Counter
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return counter(*args, **kwargs)

    monkeypatch.setattr(scalar, 'Counter', counted)
    def forbidden_digest(*args, **kwargs):
        raise AssertionError('Transient Gram reuse must not serialize or digest coefficients')
    monkeypatch.setattr(scalar, '_stable_record_hash', forbidden_digest)
    first = scalar._exact_carrier_gram(basis)
    assert first == expected and calls
    first[0, 0] = -7
    calls.clear()
    # Different dictionary insertion order must preserve the exact content key.
    restored = sp.SparseMatrix(3, 2, dict(reversed(tuple(basis.todok().items()))))
    assert scalar._exact_carrier_gram(restored) == expected and not calls
    damaged = basis.copy()
    damaged[2, 1] = 2
    assert scalar._exact_carrier_gram(damaged) == sp.simplify(damaged.T*damaged)
    assert calls
    calls.clear()
    swapped = basis[:, [1, 0]]
    assert scalar._exact_carrier_gram(swapped) == sp.simplify(swapped.T*swapped)
    assert calls
    calls.clear()
    with monkeypatch.context() as strict:
        strict.setenv('YE3T_CACHE_VERIFY', 'full')
        assert scalar._exact_carrier_gram(basis) == expected and calls
    calls.clear()
    with monkeypatch.context() as uncached:
        uncached.setenv('YE3T_CACHE_MODE', 'off')
        assert scalar._exact_carrier_gram(basis) == expected and calls
    assert len(scalar._COMPUTED_CARRIER_GRAMS) <= 8


@pytest.mark.parametrize('complex_phase', [False, True])
def test_schur_gram_retains_nonorthogonal_copies_and_all_tableaux(complex_phase, monkeypatch):
    from ye3t.representations import coset_units
    sp = scalar._sympy()
    states = ((0,0,1), (0,1,0), (1,0,0))
    generators = coset_units.adjacent_transposition_generators((2,1), 3)
    stabilizer = coset_units.class_stabilizer_sum(states[0],2,generators)
    source = next(word for word in states if any(coset_units.class_projected_row(
        states[0],word,2,generators,stabilizer)))
    columns = coset_units.selected_unit_column(2,3,states,states[0],source,generators,stabilizer)
    carrier = sp.Matrix([[columns[t].get(row,0) for t in range(2)] for row in range(3)])
    gauge = sp.diag(2,3)
    phase = sp.I if complex_phase else sp.Integer(1)
    first = carrier.col_join(sp.zeros(3,2))
    second = (2*carrier).col_join(carrier)*gauge*phase
    basis = first.row_join(second)
    actions = (generators, tuple(gauge.inv()*action*gauge for action in generators))
    expected = sp.simplify(basis.conjugate().T*basis)
    direct = scalar._exact_carrier_gram
    sizes = []
    def bounded_gram(matrix):
        sizes.append(matrix.cols)
        assert matrix.cols == 2  # Only the reference component of each copy.
        return direct(matrix)
    monkeypatch.setattr(scalar,'_exact_carrier_gram',bounded_gram)
    actual = scalar._carrier_gram_from_checked_actions(basis,((0,1),(2,3)),actions)
    assert actual == expected
    assert sizes == [2] and actual[0,2] != 0 and actual[1,3] != 0
    assert actual == actual.conjugate().T


def test_schur_gram_requires_proved_uniqueness_and_supported_gauge():
    sp = scalar._sympy()
    basis = sp.eye(4)
    assert scalar._carrier_gram_from_checked_actions(basis,((0,1),(2,3)),
        ((sp.eye(2),),(sp.eye(2),))) is None
    from ye3t.representations import coset_units
    generators = coset_units.adjacent_transposition_generators((2,1),3)
    gauge = sp.Matrix([[1,1],[0,1]])
    transformed = tuple(gauge.inv()*action*gauge for action in generators)
    assert scalar._carrier_gram_from_checked_actions(basis,((0,1),(2,3)),
        (generators,transformed)) is None


def test_carrier_validation_reuse_checks_content_and_isolates_results(monkeypatch):
    monkeypatch.setenv('YE3T_CACHE_MODE', 'auto')
    monkeypatch.setenv('YE3T_CACHE_VERIFY', 'hash')
    carrier = scalar._build_block_template((2,3,(2,1),1,1))['payload']['angular_carrier']
    scalar._VALIDATED_CARRIERS.clear()
    original = scalar._validate_carrier_payload_uncached
    calls = []

    def checked(payload):
        calls.append(payload['carrier_hash'])
        return original(payload)

    monkeypatch.setattr(scalar, '_validate_carrier_payload_uncached', checked)
    actions, grams = scalar._validate_carrier_payload(carrier)
    first = next(iter(actions))
    actions[first][0][0,0] += 1
    grams[first][0,0] += 1
    repeated_actions, repeated_grams = scalar._validate_carrier_payload(copy.deepcopy(carrier))
    assert len(calls) == 1
    assert repeated_actions[first][0] != actions[first][0]
    assert repeated_grams[first] != grams[first]
    damaged = copy.deepcopy(carrier)
    gram = scalar._exact_matrix_from_payload(damaged['gram'])
    gram[0,0] += 1
    damaged['gram'] = scalar._exact_matrix_payload(gram)
    with pytest.raises(ValueError, match='hash'):
        scalar._validate_carrier_payload(damaged)
    assert len(calls) == 1
    damaged.pop('carrier_hash')
    damaged['carrier_hash'] = scalar._stable_hash(scalar._freeze_json(damaged))
    with pytest.raises(ValueError, match='Gram'):
        scalar._validate_carrier_payload(damaged)
    assert len(calls) == 2
    for mode, verify in [('auto','full'), ('off','hash'), ('rebuild','hash')]:
        monkeypatch.setenv('YE3T_CACHE_MODE', mode)
        monkeypatch.setenv('YE3T_CACHE_VERIFY', verify)
        scalar._validate_carrier_payload(carrier)
    assert len(calls) == 5


@pytest.mark.parametrize('local', [False, True])
def test_carrier_validation_rejects_rehashed_gram_corruption(local):
    payload = scalar._build_block_template((2,3,(2,1),1,1))['payload']
    carrier = copy.deepcopy(payload['angular_carrier'])
    record = carrier['copy_records'][0] if local else carrier
    gram = scalar._exact_matrix_from_payload(record['gram'])
    gram[0,0] += 1
    record['gram'] = scalar._exact_matrix_payload(gram)
    carrier.pop('carrier_hash')
    carrier['carrier_hash'] = scalar._stable_hash(scalar._freeze_json(carrier))
    with pytest.raises(ValueError, match='Gram'):
        scalar._validate_carrier_payload(carrier)


def test_pairing_cache_preserves_gauge_and_isolates_mutation():
    sp = scalar._sympy()
    payload = scalar._build_block_template((2,3,(2,1),1,1))['payload']
    role = tuple(scalar._exact_matrix_from_payload(value) for value in
        payload['role_carrier']['copy_records'][0]['adjacent_actions'])
    angular = tuple(scalar._exact_matrix_from_payload(value) for value in
        payload['angular_carrier']['copy_records'][0]['adjacent_actions'])
    scalar._invariant_pairing_cached.cache_clear()
    original = scalar._invariant_pairing(role, angular)
    original[0,0] += 17
    repeated = scalar._invariant_pairing(role, angular)
    assert scalar._invariant_pairing_cached.cache_info().hits == 1
    assert original != repeated
    gauge = sp.diag(2, 1)
    changed = tuple(gauge * action * gauge.inv() for action in angular)
    transformed = scalar._invariant_pairing(role, changed)
    assert scalar._invariant_pairing_cached.cache_info().misses == 2
    independent = scalar._invariant_pairing_cached.__wrapped__(
        tuple(sp.ImmutableMatrix(value) for value in role),
        tuple(sp.ImmutableMatrix(value) for value in changed))
    assert transformed == independent
    for left, right in zip(role, changed):
        assert sp.simplify(left * transformed * right.T - transformed) == sp.zeros(2)


@pytest.mark.parametrize('rank,dimension', [(2,3),(3,3),(4,3),(8,2)])
def test_adjacent_row_gather_matches_independent_permutation_matrix(rank, dimension):
    sp = scalar._sympy()
    states = tuple(product(range(dimension), repeat=rank))
    basis = sp.Matrix(len(states), 3, lambda i,j: sp.Rational((i+1)*(j+2), i+j+3))
    for adjacent in range(rank-1):
        permutation = list(range(rank))
        permutation[adjacent], permutation[adjacent+1] = permutation[adjacent+1], permutation[adjacent]
        raw = scalar._raw_permutation_matrix(states, tuple(permutation))
        expected = sp.Matrix(raw) * basis
        rows = scalar._adjacent_permutation_rows(states, rank, adjacent)
        assert basis.extract(rows, (0,1,2)) == expected
        assert tuple(rows[row] for row in rows) == tuple(range(len(states)))


def test_complete_template_and_validator_do_not_build_raw_square_actions(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Carrier checks must gather rows instead of building raw square actions')

    monkeypatch.setattr(scalar, '_raw_permutation_matrix', forbidden)
    monkeypatch.setattr(scalar, '_DENSE_CARRIER_ENTRY_LIMIT', 0)
    payload = scalar._build_block_template((2,3,(2,1),1,1))['payload']
    for name in ('role_carrier','angular_carrier'):
        carrier = payload[name]
        scalar._validate_carrier_payload(carrier)
        damaged = scalar._freeze_json(carrier)
        damaged['copy_records'][0]['adjacent_actions'] = ()
        damaged['carrier_hash'] = scalar._stable_hash(scalar._freeze_json(
            {key:value for key,value in damaged.items() if key != 'carrier_hash'}))
        with pytest.raises(ValueError, match='generator count'):
            scalar._validate_carrier_payload(damaged)


def test_role_cross_copy_intertwining_rejects_incorrect_block_action(monkeypatch):
    states, vectors, _ = scalar._role_schur_vectors(3, 3, (2, 1))
    order = tuple(sorted(vectors))
    copies = sorted({key[0] for key in order})
    keys = tuple(((index,), tuple(key for key in order if key[0] == index))
                 for index in copies)
    # Each individual copy has valid generators. Corrupt only the combined
    # action, so the full cross-copy identity must reject it independently.
    original = scalar._block_diagonal
    corrupted = []

    def incorrect(matrices):
        result = original(matrices)
        if len(matrices) > 1:
            result[0, -1] += 1
            corrupted.append(result.shape)
        return result

    monkeypatch.setattr(scalar, '_block_diagonal', incorrect)
    with pytest.raises(RuntimeError, match='Full carrier basis fails exact permutation intertwining'):
        scalar._role_carrier_payload(states, order, vectors, keys, 3, 'role_schur')
    assert corrupted
