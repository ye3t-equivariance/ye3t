"""Numerical shared construction and separately prepared canonical seeds."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from ye3t.representations import projectors as p
from ye3t.representations import _tableau_numeric as n


CASES = [
    ("split", (2, 1), (2, 1), (3, 2, 1)),
    ("split", (3,), (1, 1), (3, 1, 1)),
    ("split", (1, 1), (3,), (3, 1, 1)),
    ("split", (), (2, 1), (2, 1)),
    ("split", (2,), (2,), (1, 1, 1, 1)),
    ("same_action", (3, 1, 1), (3, 1, 1), (3, 2)),
    ("same_action", (3, 2, 1), (3, 2, 1), (4, 2)),
    ("same_action", (3, 2, 1), (3, 2, 1), (3, 2, 1)),
]


@pytest.mark.parametrize("kind,left,right,parent", CASES)
def test_numerical_maps_match_exact_canonical_coefficients(kind, left, right, parent):
    from ye3t.representations.young_orthogonal import _projector_binary_split_matrices
    from ye3t.representations.kronecker_intertwiners import _projector_kronecker_intertwiners
    recipe = n._prepare_tableau_recipe(kind, left, right, parent)
    actual = n._materialize_tableau_recipe(kind, left, right, parent, recipe[0])
    if kind == "split":
        exact = _projector_binary_split_matrices(left, right, parent)
    else:
        exact = _projector_kronecker_intertwiners(parent, left, right)["intertwiners"]
    assert len(actual) == len(exact)
    for matrix, reference in zip(actual, exact, strict=True):
        np.testing.assert_allclose(matrix, np.asarray(reference, dtype=float), atol=1e-12, rtol=1e-12)
        assert not matrix.flags.writeable
    for i, matrix in enumerate(actual):
        for j, other in enumerate(actual):
            np.testing.assert_allclose(matrix.T @ other,
                                       np.eye(matrix.shape[1]) if i == j else 0,
                                       atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("partition", [(3, 1, 1), (3, 2, 1), (4, 2, 1)])
def test_sparse_seminormal_and_numerical_actions_match_independent_existing_generators(partition):
    from fractions import Fraction
    weights = np.sqrt([float(v) for v in n._seminormal_weights(partition)])
    layouts = n._generator_layout(partition)
    actions = n._orthogonal_actions(partition)
    size = len(weights)
    for generator, layout in enumerate(layouts):
        exact = np.zeros((size, size))
        for i in range(size):
            for j, value in n._rational_adjacent(layout, {i: Fraction(1)}).items():
                exact[j, i] = float(value)*weights[j]/weights[i]
        reference = np.asarray(p.adjacent_transposition_representation_matrix(partition, generator), dtype=float)
        np.testing.assert_allclose(exact, reference, atol=5e-16, rtol=0)
        np.testing.assert_allclose(n._numeric_adjacent(actions[generator], np.eye(size)),
                                   reference, atol=5e-16, rtol=0)


def test_numerical_materialization_never_calls_recipe_or_symbolic_arithmetic(monkeypatch):
    request = ("same_action", (3, 2, 1), (3, 2, 1), (4, 2))
    recipe = n._prepare_tableau_recipe(*request)
    def forbidden(*args, **kwargs):
        raise AssertionError("Numerical materialization entered exact preparation or a solver")
    monkeypatch.setattr(n, "_prepare_tableau_recipe", forbidden)
    monkeypatch.setattr(n, "_seminormal_weights", forbidden)
    monkeypatch.setattr(n, "_rational_adjacent", forbidden)
    monkeypatch.setattr(p, "_sympy", forbidden)
    for name in ("svd", "eig", "eigh", "qr", "solve", "cholesky", "matrix_rank"):
        monkeypatch.setattr(np.linalg, name, forbidden)
    assert len(n._materialize_tableau_recipe(*request, recipe[0])) == 3


def test_fresh_recipe_and_materialization_without_sympy(tmp_path):
    # Both phases run in a genuinely fresh interpreter, not a populated module cache.
    script = tmp_path / "fresh.py"
    script.write_text("""
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name in ('sympy', 'torch') or name.startswith(('sympy.', 'torch.')):
        raise AssertionError('Unexpected symbolic/tensor-engine import: '+name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from ye3t.representations._tableau_numeric import _prepare_tableau_recipe, _materialize_tableau_recipe
for request in [('split',(2,1),(2,1),(3,2,1)), ('same_action',(3,2,1),(3,2,1),(4,2))]:
    recipe = _prepare_tableau_recipe(*request)
    matrices = _materialize_tableau_recipe(*request, recipe[0])
    assert len(matrices) in (2,3)
""")
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                            env=environment, cwd=tmp_path, timeout=60)
    assert result.returncode == 0, result.stdout+result.stderr


def test_invalid_recipe_does_not_select_other_coordinates():
    with pytest.raises(ValueError, match="copy count"):
        n._materialize_tableau_recipe("split", (2,1), (2,1), (3,2,1), (0,0))


def test_normalization_arithmetic_failure_does_not_change_gauge():
    with pytest.raises(ArithmeticError, match="determinant"):
        p._numeric_copy_normalization(np.ones((2,2)))


@pytest.mark.parametrize("children,parent", [
    (((1,),)*3, (2,1)),
    (((1,),)*5, (3,2)),
    (((2,1), (1,), (1,)), (3,2)),
    (((2,), (1,1), (2,)), (3,2,1)),
])
def test_nary_numeric_composition_keeps_actual_tree_coordinates(children, parent):
    from ye3t.representations.young_orthogonal import _canonical_split_matrices
    actual = _canonical_split_matrices(children, parent, "balanced", "numeric")
    reference = _canonical_split_matrices(children, parent, "balanced")
    assert len(actual) == len(reference)
    for matrix, expected in zip(actual, reference, strict=True):
        np.testing.assert_allclose(matrix, np.asarray(expected, dtype=float), atol=1e-12, rtol=1e-12)


def test_compact_consumer_cold_build_is_numerical(monkeypatch, tmp_path):
    from ye3t.representations import young_orthogonal as young
    from ye3t.cache import coefficient_arrays
    def forbidden(*args, **kwargs):
        raise AssertionError("Compact numerical consumer entered symbolic coefficients")
    # Capture the real cold builder without an existing disk entry masking dispatch.
    monkeypatch.setattr(coefficient_arrays, "_resolve_coefficient_array",
                        lambda identity, shape, build, directory=None: (build(), "test-cold"))
    monkeypatch.setattr(young, "_sympy", forbidden)
    monkeypatch.setattr(p, "_sympy", forbidden)
    young._canonical_split_matrices.cache_clear()
    young._direct_restricted_young_values_cached.cache_clear()
    values, count = young.direct_restricted_young_values(((2,1), (1,), (1,)), (3,2))
    assert count == 2
    assert np.asarray(values).shape == (2, 10)


def test_nary_numeric_assembly_never_forms_the_child_kronecker_matrix(monkeypatch):
    from ye3t.representations import young_orthogonal as young
    original = np.einsum
    def bounded_contraction(subscripts, *operands, **kwargs):
        assert len(operands) == 2, "A global contraction can form an expanded child product"
        return original(subscripts, *operands, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError("Numerical assembly expanded the child Kronecker matrix")
    monkeypatch.setattr(np, "einsum", bounded_contraction)
    monkeypatch.setattr(np, "kron", forbidden)
    young._canonical_split_matrices.cache_clear()
    assert young._canonical_split_matrices(((2,1), (1,), (1,)), (3,2), "balanced", "numeric")


@pytest.mark.parametrize("rank", range(2, 7))
def test_complete_binary_numeric_inventory_through_rank_six(rank):
    from ye3t.couplings import integer_partitions
    from ye3t.representations.young_orthogonal import _projector_binary_split_matrices
    left_rank = rank//2
    for parent_shape in integer_partitions(rank):
        parent = tuple(parent_shape)
        dimension = len(p.standard_tableaux(parent))
        total = np.zeros((dimension, dimension))
        for left_shape in integer_partitions(left_rank):
            for right_shape in integer_partitions(rank-left_rank):
                left, right = tuple(left_shape), tuple(right_shape)
                request = ("split", left, right, parent)
                indices = n._prepare_tableau_recipe(*request)[0]
                matrices = n._materialize_tableau_recipe(*request, indices)
                reference = _projector_binary_split_matrices(left, right, parent)
                assert len(matrices) == len(reference)
                for matrix, exact in zip(matrices, reference, strict=True):
                    np.testing.assert_allclose(matrix, np.asarray(exact, dtype=float),
                                               atol=1e-12, rtol=1e-12)
                    total += matrix @ matrix.T
        np.testing.assert_allclose(total, np.eye(dimension), atol=1e-12, rtol=1e-12)
