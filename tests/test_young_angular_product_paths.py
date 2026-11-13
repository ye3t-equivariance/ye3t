"""Normalized product maps against bounded explicit magnetic references."""

from itertools import product, permutations

import pytest

from ye3t.exact_scalars import ExactRadical
from ye3t.representations import young_angular_paths as paths
from ye3t.representations.factorized_product_images import (
    _angular_tree_coefficients, _couple_multiplets)


def _chains(spins):
    result = [(spins[0],)]
    for ell in spins[1:]:
        result = [prefix+(L,) for prefix in result
                  for L in range(abs(prefix[-1]-ell), prefix[-1]+ell+1)]
    return result


def _expand(column, spins, M):
    result = {}
    for path, coefficient in column:
        result = paths._add_scaled(result, _angular_tree_coefficients(spins, path[1:], M), coefficient)
    return result


@pytest.mark.parametrize("left_spins,right_spins", [
    ((1,), (1,)), ((1, 2), (2, 1)), ((0, 1), (1, 0, 1)),
    ((2,), (1, 1, 1)),
])
def test_selected_chain_coefficient_equals_full_association(left_spins, right_spins):
    combined = _chains(left_spins+right_spins)
    for left, right in product(_chains(left_spins), _chains(right_spins)):
        for L in range(abs(left[-1]-right[-1]), left[-1]+right[-1]+1):
            reference = dict(paths._coupled_chain_paths(left, right_spins, right, L))
            for target in combined:
                if target[:len(left)] != left or target[-1] != L:
                    continue
                actual = paths._coupled_path_overlap(((left, ExactRadical.rational(1)),),
                    ((right, ExactRadical.rational(1)),), right_spins, L, {target: ExactRadical.rational(1)})
                assert actual == reference.get(target, 0)


@pytest.mark.parametrize("left_spins,right_spins,L", [
    ((1, 1), (1, 1), 0), ((1, 2), (2, 1), 2),
    ((2, 1), (0, 1, 1), 1),
])
def test_selected_overlap_matches_expansion_for_sparse_columns(left_spins, right_spins, L):
    left = tuple((path, ExactRadical.rational(i-2)) for i, path in enumerate(_chains(left_spins))
                 if path[-1] == 1)
    right = tuple((path, ExactRadical.rational(2*i-1)) for i, path in enumerate(_chains(right_spins))
                  if path[-1] == 1)
    target = {path: ExactRadical.rational(i-3) for i, path in enumerate(_chains(left_spins+right_spins))
              if path[-1] == L and i % 3}
    assert left and right and target
    reference = paths._inner(target, paths._coupled_path_columns(left, right, right_spins, L))
    assert paths._coupled_path_overlap(left, right, right_spins, L, target) == reference


def test_selected_overlap_validates_paths_and_empty_inputs():
    column = (((1,), ExactRadical.rational(1)),)
    with pytest.raises(ValueError, match="matching"):
        paths._coupled_path_overlap(column, (((), 1),), (1,), 0, {(1, 0): 1})
    with pytest.raises(ValueError, match="triangle"):
        paths._coupled_path_overlap(column, (((1, 4), 1),), (1, 1), 3, {(1, 1, 3): 1})
    assert paths._coupled_path_overlap(column, column, (1,), 3, {(1, 3): 1}) == 0
    assert paths._coupled_path_overlap(column, (((1, 2), 1),), (1, 1), 3, {(1, 0, 3): 1}) == 0
    assert paths._coupled_path_overlap(column, column, (1,), 0, {}) == 0
    assert paths._coupled_path_overlap((), column, (1,), 0, {(1, 0): 1}) == 0
    with pytest.raises(ValueError, match="Target CG path"):
        paths._coupled_path_overlap(column, column, (1,), 0, {(1,): 1})
    with pytest.raises(ValueError, match="Target CG path"):
        paths._coupled_path_overlap(column, column, (1,), 0, {(1, 2): 1})


@pytest.mark.parametrize("left_spins,right_spins", [
    ((1,), (1, 1)), ((1, 2), (2, 1)), ((0, 1), (1, 0, 1)),
    ((2,), (1, 1, 1)),
])
def test_chain_association_is_exact_and_isometric_for_mixed_degrees(left_spins, right_spins):
    by_L = {}
    for left, right in product(_chains(left_spins), _chains(right_spins)):
        a, b = left[-1], right[-1]
        for L in range(abs(a-b), a+b+1):
            column = paths._coupled_chain_paths(left, right_spins, right, L)
            by_L.setdefault(L, []).append(dict(column))
            lhs = {M: _angular_tree_coefficients(left_spins, left[1:], M) for M in range(-a, a+1)}
            rhs = {M: _angular_tree_coefficients(right_spins, right[1:], M) for M in range(-b, b+1)}
            reference = _couple_multiplets(lhs, a, rhs, b, L)
            for M in range(-L, L+1):
                assert _expand(column, left_spins+right_spins, M) == reference[M]
    for L, columns in by_L.items():
        assert len(columns) == sum(path[-1] == L for path in _chains(left_spins+right_spins))
        for i, left in enumerate(columns):
            for j, right in enumerate(columns):
                assert paths._inner(left, right) == int(i == j)


@pytest.mark.parametrize("left_size,right_size,ell,lp,rp,tp,a,b,L", [
    (2, 2, 1, (2,), (2,), (4,), 2, 2, 0),
    (2, 2, 1, (2,), (2,), (2, 2), 2, 2, 0),
    (2, 2, 1, (1, 1), (1, 1), (2, 2), 1, 1, 0),
    (3, 1, 1, (2, 1), (1,), (3, 1), 1, 1, 2),
    (2, 2, 2, (2,), (1, 1), (3, 1), 2, 3, 2),
])
def test_young_product_overlap_agrees_with_every_explicit_component(left_size, right_size, ell, lp, rp, tp, a, b, L):
    left = paths.young_angular_path_basis(left_size, ell, lp, a)
    right = paths.young_angular_path_basis(right_size, ell, rp, b)
    target = paths.young_angular_path_basis(left_size+right_size, ell, tp, L)
    overlaps = dict(paths._young_angular_product_overlap(left_size, right_size, ell, lp, rp, tp, a, b, L))
    assert overlaps
    for i, copy in enumerate(left):
        for t, column in enumerate(copy):
            lhs = {M: _expand(column, (ell,)*left_size, M) for M in range(-a, a+1)}
            for j, other in enumerate(right):
                for u, other_column in enumerate(other):
                    rhs = {M: _expand(other_column, (ell,)*right_size, M) for M in range(-b, b+1)}
                    direct = _couple_multiplets(lhs, a, rhs, b, L)
                    for c, target_copy in enumerate(target):
                        for v, target_column in enumerate(target_copy):
                            value = overlaps.get((c, v, i, t, j, u), 0)
                            for M in range(-L, L+1):
                                assert paths._inner(_expand(target_column, (ell,)*(left_size+right_size), M), direct[M]) == value


def test_full_young_decomposition_reconstructs_child_product():
    left = paths.young_angular_path_basis(2, 1, (2,), 2)[0][0]
    right = paths.young_angular_path_basis(2, 1, (1, 1), 1)[0][0]
    actual = paths._coupled_path_columns(left, right, (1, 1), 2)
    reconstructed = {}
    norm = ExactRadical.rational(0)
    for tp in ((4,), (3, 1), (2, 2), (2, 1, 1), (1, 1, 1, 1)):
        target = paths.young_angular_path_basis(4, 1, tp, 2)
        overlaps = paths._young_angular_product_overlap(2, 2, 1, (2,), (1, 1), tp, 2, 1, 2)
        for (c, v, i, t, j, u), value in overlaps:
            assert (i, t, j, u) == (0, 0, 0, 0)
            reconstructed = paths._add_scaled(reconstructed, dict(target[c][v]), value)
            norm += value*value
    assert reconstructed == actual
    assert norm == 1


@pytest.mark.parametrize("left_size,right_size,ell,lp,rp,a,b,L", [
    (1, 3, 2, (1,), (3,), 2, 2, 2),
    (2, 3, 2, (2,), (3,), 2, 2, 3),
    (2, 4, 2, (2,), (4,), 2, 2, 0),
    (1, 2, 1, (1,), (1, 1), 1, 1, 1),
])
def test_symmetric_target_exchange_preserves_exact_map_and_original_indices(
        left_size, right_size, ell, lp, rp, a, b, L):
    tp = (left_size+right_size,)
    left = paths.young_angular_path_basis(left_size, ell, lp, a)
    right = paths.young_angular_path_basis(right_size, ell, rp, b)
    target = paths.young_angular_path_basis(left_size+right_size, ell, tp, L)
    assert left and right and target
    expected = []
    for i, copy in enumerate(left):
        for t, column in enumerate(copy):
            for j, other in enumerate(right):
                for u, other_column in enumerate(other):
                    expanded = paths._coupled_path_columns(column, other_column, (ell,)*right_size, L)
                    for c, target_copy in enumerate(target):
                        for v, target_column in enumerate(target_copy):
                            value = paths._inner(dict(target_column), expanded)
                            if value:
                                expected.append(((c, v, i, t, j, u), value))
    actual = paths._young_angular_product_overlap(left_size, right_size, ell, lp, rp, tp, a, b, L)
    assert actual == tuple(expected)
    if rp == (right_size,):
        assert actual


@pytest.mark.parametrize("target,L,expected_sizes", [((4,), 0, (3, 1)), ((3, 1), 2, (1, 3))])
def test_short_child_orientation_depends_on_target_action(monkeypatch, target, L, expected_sizes):
    calls = []
    original = paths._coupled_path_overlap

    def observed(left, right, spins, output_L, column):
        calls.append((len(left[0][0]), len(spins)))
        return original(left, right, spins, output_L, column)

    paths._young_angular_product_overlap.cache_clear()
    monkeypatch.setattr(paths, "_coupled_path_overlap", observed)
    assert paths._young_angular_product_overlap(1, 3, 1, (1,), (3,), target, 1, 1, L)
    assert calls and set(calls) == {expected_sizes}


def test_shared_factor_scalar_routes_use_the_same_normalized_parent():
    # Literal reuse of one vector: the two rotational contraction routes
    # reduce to a single symmetric rank-four scalar with exact coefficients.
    import sympy as sp
    symbols = sp.symbols("xminus xzero xplus")
    values = dict(zip((-1, 0, 1), symbols))
    target = paths.young_angular_path_basis(4, 1, (4,), 0)[0][0]

    def evaluate(column):
        return sp.expand(sum(value._sympy_()*sp.prod(values[m] for m in magnetic)
                             for magnetic, value in _expand(column, (1,)*4, 0).items()))

    for spin in (0, 2):
        child = paths.young_angular_path_basis(2, 1, (2,), spin)[0][0]
        coupled = paths._coupled_path_columns(child, child, (1, 1), 0)
        coefficient = dict(paths._young_angular_product_overlap(
            2, 2, 1, (2,), (2,), (4,), spin, spin, 0))[(0, 0, 0, 0, 0, 0)]
        assert sp.simplify(evaluate(tuple(coupled.items())) - coefficient._sympy_()*evaluate(target)) == 0


def test_rank_eight_product_avoids_magnetic_expansion_and_gram_solve(monkeypatch):
    from ye3t.representations import builder, factorized_product_images
    from ye3t.core import subtree_dag
    import sympy as sp

    def forbidden(*args, **kwargs):
        pytest.fail("Normalized product assembly expanded magnetic states or inverted a Gram matrix")

    monkeypatch.setattr(builder, "_exact_single_factor_weight_vectors", forbidden)
    monkeypatch.setattr(factorized_product_images, "_angular_tree_coefficients", forbidden)
    monkeypatch.setattr(factorized_product_images, "_single_factor_carriers", forbidden)
    monkeypatch.setattr(subtree_dag, "expand_tree_key_exact", forbidden)
    monkeypatch.setattr(sp.matrices.matrixbase.MatrixBase, "inv", forbidden)
    monkeypatch.setattr(paths, "_coupled_path_columns", forbidden)
    monkeypatch.setattr(paths, "_coupled_chain_paths", forbidden)
    paths._young_angular_product_overlap.cache_clear()
    assert paths._young_angular_product_overlap(4, 4, 1, (4,), (4,), (6, 2), 2, 2, 0)


def test_invalid_chain_and_forbidden_triangle():
    with pytest.raises(ValueError, match="matching"):
        paths._coupled_chain_paths((1,), (1,), (), 0)
    with pytest.raises(ValueError, match="triangle"):
        paths._coupled_chain_paths((1,), (1, 1), (1, 4), 3)
    assert paths._coupled_chain_paths((1,), (1,), (1,), 3) == ()
    assert paths._young_angular_product_overlap(2, 2, 1, (2,), (2,), (4,), 0, 0, 2) == ()
    assert paths._young_angular_product_overlap(2, 2, 1, (2,), (2,), (1, 1, 1, 1), 0, 0, 0) == ()


def test_unequal_spin_factor_reordering_matches_explicit_tensors():
    spins = (1, 2, 1)
    for path in _chains(spins):
        column = ((path, ExactRadical.rational(1)),)
        L = path[-1]
        for order in permutations(range(len(spins))):
            changed = paths._permuted_chain_column(column, spins, order)
            assert paths._inner(changed, changed) == 1
            changed_spins = tuple(spins[i] for i in order)
            inverse = tuple(order.index(i) for i in range(len(spins)))
            assert paths._permuted_chain_column(changed.items(), changed_spins, inverse) == dict(column)
            for M in range(-L, L+1):
                expected = {tuple(magnetic[i] for i in order): value
                            for magnetic, value in _expand(column, spins, M).items()}
                assert _expand(changed.items(), changed_spins, M) == expected


def test_outer_shared_block_recoupling_uses_paths_only(monkeypatch):
    from ye3t.representations import factorized_product_images as images

    # Two rank-two block trees share their second block. Canonical parent
    # order also interleaves the two child factor lists.
    args = (((1, 2), (0, 0), (2,)), ((1, 2), (0, 0), (2,)),
            ((1, 1, 2), (0, 0, 0), (0, 2)), (0, 2), (1, 2), 2, 2, 2)
    expected = images._angular_block_recoupling(*args)
    assert expected != 0

    def forbidden(*args, **kwargs):
        pytest.fail("Outer recoupling expanded block magnetic tensors")

    monkeypatch.setattr(images, "_angular_tree_coefficients", forbidden)
    images._angular_block_recoupling.cache_clear()
    assert images._angular_block_recoupling(*args) == expected
