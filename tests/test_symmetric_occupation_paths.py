"""Shared compact occupancy/path changes preserve existing source features."""
import pytest
import sympy as sp

from ye3t.core.basis import homogeneous
from ye3t.representations.young_angular_paths import (
    _symmetric_occupation_path_change, young_angular_path_basis)
from ye3t.representations.factorized_product_images import _angular_tree_coefficients


@pytest.mark.parametrize("size,ell,L", [(1, 0, 0), (1, 2, 2), (2, 1, 0), (3, 1, 1), (3, 2, 3), (4, 2, 2)])
@pytest.mark.parametrize("mode", ["independent", "orthogonal"])
def test_exact_change_reconstructs_original_magnetic_coefficients_at_every_M(size, ell, L, mode):
    change = _symmetric_occupation_path_change(size, ell, L, mode)
    columns = young_angular_path_basis(size, ell, (size,), L)
    states = homogeneous.homogeneous_basis_states_by_L(size, ell, basis_mode=mode)[L]
    assert len(change) == len(states)
    for b, state in enumerate(states):
        source = homogeneous.thaw_magnetic_expansion(homogeneous.occupancy_expansion_to_m_vectors(
            ell, size, state.occupancy_expansion_by_M))
        for M in range(-L, L+1):
            reconstructed = {}
            for a, column in enumerate(columns):
                for path, weight in column[0]:
                    for word, value in _angular_tree_coefficients((ell,)*size, path[1:], M).items():
                        reconstructed[word] = reconstructed.get(word, 0)+(change[a][b]*weight*value)._sympy_()
            expected = source[M]
            for word in set(expected) | set(reconstructed):
                assert sp.simplify(reconstructed.get(word, 0)-expected.get(word, 0)) == 0
    matrix = sp.Matrix([[value._sympy_() for value in row] for row in change])
    if mode == "orthogonal":
        assert (matrix.T*matrix-sp.eye(len(states))).applyfunc(sp.simplify).is_zero_matrix
    else:
        highest = [dict(state.occupancy_weight_map()[L]) for state in states]
        metric = sp.Matrix([[sum(x*highest[b].get(key, 0) for key, x in highest[a].items())
                             for b in range(len(states))] for a in range(len(states))])
        assert (matrix.T*matrix-metric).applyfunc(sp.simplify).is_zero_matrix


def test_rank_eight_conversion_forbids_expanded_magnetic_construction_and_reuses_copy_map(monkeypatch):
    import ye3t.representations.factorized_product_images as products
    def forbidden(*args, **kwargs):
        pytest.fail("compact symmetric conversion expanded magnetic tensors")
    monkeypatch.setattr(homogeneous, "occupancy_expansion_to_m_vectors", forbidden)
    monkeypatch.setattr(homogeneous, "_occupancy_multiset_permutations", forbidden)
    monkeypatch.setattr(products, "_angular_tree_coefficients", forbidden)
    result = _symmetric_occupation_path_change(8, 2, 0)
    assert len(result) == 2 and all(len(row) == 2 for row in result)
    import ye3t.representations.young_angular_paths as paths
    monkeypatch.setattr(paths, "young_angular_path_basis", forbidden)
    monkeypatch.setattr(homogeneous, "_highest_weight_basis", forbidden)
    assert _symmetric_occupation_path_change(8, 2, 0) is result
