"""Exact reduced Young/rotation construction against magnetic-space references."""
import pytest

from ye3t.exact_scalars import ExactRadical, exact_scalar
from ye3t.representations.young_angular_paths import (
    young_angular_path_basis, adjacent_action, _inner, _add_scaled)
from ye3t.representations.projectors import adjacent_transposition_representation_matrix
from ye3t.representations.factorized_product_images import (
    _angular_tree_coefficients, _single_factor_carriers)


def expand(column, ell, M):
    result = {}
    for path, coefficient in column.items():
        result = _add_scaled(result, _angular_tree_coefficients(
            (ell,)*len(path), path[1:], M), coefficient)
    return result


@pytest.mark.parametrize("size,ell,partition,L", [
    (1, 1, (1,), 1), (2, 1, (2,), 0), (2, 1, (1, 1), 1),
    (3, 1, (2, 1), 1), (3, 1, (2, 1), 2),
    (4, 1, (3, 1), 1), (4, 1, (2, 2), 0),
    (4, 2, (4,), 2), (4, 2, (3, 1), 3), (4, 2, (2, 2), 0)])
def test_exact_isometry_permutation_action_and_reference_projector(size, ell, partition, L):
    basis = young_angular_path_basis(size, ell, partition, L)
    columns = [dict(column) for copy in basis for column in copy]
    assert columns
    for i, left in enumerate(columns):
        for j, right in enumerate(columns):
            assert _inner(left, right) == int(i == j)
    for index in range(size-1):
        generator = adjacent_transposition_representation_matrix(partition, index)
        for copy in basis:
            for t, column in enumerate(copy):
                expected = {}
                for u, other in enumerate(copy):
                    expected = _add_scaled(expected, dict(other), exact_scalar(generator[u, t]))
                actual = adjacent_action(dict(column), ell, index)
                assert actual == expected
                magnetic = expand(dict(column), ell, L)
                exchanged = {word[:index]+(word[index+1], word[index])+word[index+2:]: value
                             for word, value in magnetic.items()}
                assert expand(actual, ell, L) == exchanged

    legacy = _single_factor_carriers(size, ell, partition, L)
    assert len(legacy) == len(columns)
    for M in range(-L, L+1):
        embeddings = [expand(column, ell, M) for column in columns]
        for old in legacy.values():
            vector = old[M]
            projected = {}
            for column in embeddings:
                projected = _add_scaled(projected, column, _inner(column, vector))
            assert projected == vector


def test_zero_branching_has_no_coordinates():
    assert young_angular_path_basis(4, 1, (3, 1), 0) == ()
    assert young_angular_path_basis(4, 1, (1, 1, 1, 1), 1) == ()


def test_construction_does_not_expand_magnetic_vectors(monkeypatch):
    from ye3t.representations import builder, factorized_product_images
    from ye3t.core import subtree_dag

    def forbidden(*args, **kwargs):
        pytest.fail("Reduced construction expanded a magnetic carrier")
    monkeypatch.setattr(builder, "_exact_single_factor_weight_vectors", forbidden)
    monkeypatch.setattr(factorized_product_images, "_angular_tree_coefficients", forbidden)
    monkeypatch.setattr(subtree_dag, "expand_tree_key_exact", forbidden)
    young_angular_path_basis.cache_clear()
    assert young_angular_path_basis(5, 1, (3, 2), 1)


def test_shared_exact_map_cache_survives_process_cache_clear(tmp_path, monkeypatch):
    from ye3t.representations import young_angular_paths as shared
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("YE3T_CACHE_MODE", "auto")
    monkeypatch.setenv("YE3T_CACHE_VERIFY", "hash")
    shared.young_angular_path_basis.cache_clear()
    arguments = (4,2,(2,2),0)
    expected = shared.young_angular_path_basis(*arguments)
    shared.young_angular_path_basis.cache_clear()
    original = shared._construct_young_angular_path_basis
    def forbidden(*args, **kwargs):
        pytest.fail("Shared representation map was reconstructed on a warm cache")
    monkeypatch.setattr(shared, "_construct_young_angular_path_basis", forbidden)
    assert shared.young_angular_path_basis(*arguments) == expected
    monkeypatch.setattr(shared, "_construct_young_angular_path_basis", original)
    monkeypatch.setenv("YE3T_CACHE_VERIFY", "full")
    shared.young_angular_path_basis.cache_clear()
    assert shared.young_angular_path_basis(*arguments) == expected
    shared.young_angular_path_basis.cache_clear()


def _reference_jucys_murphy_words(vector, ell, rank):
    """Pre-optimization sum of independent transposition words."""
    from ye3t.representations import young_angular_paths as shared

    result = {}
    for first in range(rank - 1):
        current = vector
        word = tuple(range(first, rank - 1)) + tuple(range(rank - 3, first - 1, -1))
        for index in word:
            current = shared.adjacent_action(current, ell, index)
        result = shared._add_scaled(result, current, 1)
    return result


@pytest.mark.parametrize("size,ell,partition,L", [
    (4,2,(4,),2), (4,2,(3,1),3), (5,1,(3,2),1), (4,1,(2,2),0)])
def test_complete_child_space_is_closed_under_independent_jucys_action(size, ell, partition, L):
    from ye3t.representations.projectors import standard_tableaux
    reference = standard_tableaux(partition)[0]
    child = tuple(tuple(value for value in row if value != size) for row in reference)
    child = tuple(row for row in child if row)
    child_partition = tuple(map(len, child))
    component = standard_tableaux(child_partition).index(child)
    candidates = []
    for child_L in range(abs(L-ell), min(L+ell, (size-1)*ell)+1):
        for copy in young_angular_path_basis(size-1,ell,child_partition,child_L):
            candidates.append({path+(L,):value for path,value in copy[component]})
    assert candidates
    for column in candidates:
        action = _reference_jucys_murphy_words(column,ell,size)
        recovered = {}
        for other in candidates:
            recovered = _add_scaled(recovered,other,_inner(other,action))
        assert recovered == action


@pytest.mark.parametrize("ell", [1, 2])
@pytest.mark.parametrize("rank", range(2, 9))
def test_jucys_murphy_reuses_adjacent_actions_exactly(monkeypatch, rank, ell):
    from ye3t.representations import young_angular_paths as shared

    vector = {(ell,) * rank: ExactRadical.rational(1)}
    if rank > 2:
        vector[(ell, 0) + (ell,) * (rank - 2)] = ExactRadical.sqrt(2) / 3
    before = dict(vector)
    original = shared.adjacent_action
    calls = []

    def observed(values, angular, index):
        calls.append(index)
        return original(values, angular, index)

    expected = _reference_jucys_murphy_words(vector, ell, rank)
    monkeypatch.setattr(shared, "adjacent_action", observed)
    actual = shared._last_jucys_murphy(vector, ell, rank)
    assert actual == expected
    assert vector == before
    assert len(calls) == 2 * rank - 3


@pytest.mark.parametrize("arguments", [(4, 2, (2, 2), 0), (4, 2, (2, 1, 1), 1), (5, 1, (3, 2), 1)])
def test_jucys_reuse_preserves_complete_normalized_basis(monkeypatch, arguments):
    from ye3t.representations import young_angular_paths as shared

    monkeypatch.setenv("YE3T_CACHE_MODE", "off")
    shared.young_angular_path_basis.cache_clear()
    actual = shared.young_angular_path_basis(*arguments)
    shared.young_angular_path_basis.cache_clear()
    with monkeypatch.context() as reference:
        reference.setattr(shared, "_last_jucys_murphy", _reference_jucys_murphy_words)
        expected = shared.young_angular_path_basis(*arguments)
    shared.young_angular_path_basis.cache_clear()
    assert actual == expected
    assert actual
