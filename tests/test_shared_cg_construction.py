"""Common normalized coefficients across representation/basis consumers."""

from fractions import Fraction

import pytest
import sympy as sp

from ye3t.core import cg


@pytest.mark.parametrize("left,right,target", [
    (0, 3, 3), (1, 1, 0), (1, 1, 1), (2, 3, 4),
    (Fraction(1, 2), Fraction(1, 2), 0),
    (Fraction(1, 2), 1, Fraction(3, 2)),
])
def test_exact_table_isometry_and_canonical_phase(left, right, target):
    from sympy.physics.wigner import clebsch_gordan

    table = cg._cg_table_exact(left, right, target)
    assert table
    magnetic = [Fraction(-target)+i for i in range(int(2*target)+1)]
    rows = {}
    for m1, m2, M, value in table:
        assert m1+m2 == M
        assert value
        exact = sp.sympify(value)
        assert sp.simplify(exact-clebsch_gordan(left, right, target, m1, m2, M)) == 0
        rows.setdefault(M, {})[m1, m2] = exact
    for M in magnetic:
        for Mp in magnetic:
            overlap = sum(sp.conjugate(value)*rows[Mp].get(word, 0)
                          for word, value in rows[M].items())
            assert sp.simplify(overlap-int(M == Mp)) == 0


@pytest.mark.parametrize("labels", [(1, 3, 0), (1, 1, 3), (1, 1, Fraction(1, 2))])
def test_forbidden_pair_has_no_coefficients(labels):
    assert cg._cg_table_exact(*labels) == ()


def test_negative_angular_label_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        cg._cg_table_exact(-1, 1, 0)


def test_same_pair_reused_by_general_typed_and_cauchy(monkeypatch):
    from ye3t.global_coupler import _pair_cg_table
    from ye3t.couplings.factorized_typed import _cg_tree_payload
    from ye3t.couplings import covariant_cauchy
    from ye3t.core.factorized_dag import cg_matrix_for_node

    cg._cg_table_exact.cache_clear()
    exact = cg._cg_table_exact(2, 2, 2)
    # All consumers must reuse the canonical table, including its zeros.
    monkeypatch.setattr(cg, "cg_exact", lambda *args: pytest.fail("reconstructed cached CG coefficients"))
    assert _pair_cg_table(2, 2, 2) == tuple(
        (int(m1), int(m2), int(M), float(value)) for m1, m2, M, value in exact)
    tree = _cg_tree_payload(("merge", 2, ("leaf", 0, 2), ("leaf", 1, 2)))
    assert tree["entries"] == tuple(
        (int(m1)+2, int(m2)+2, int(M)+2, float(value)) for m1, m2, M, value in exact)
    covariant_cauchy._cached_outer_multiplet_template.cache_clear()
    payload = covariant_cauchy._outer_multiplet_template((2, 2), 2)
    assert payload["copy_count"] == 1
    cg_matrix_for_node.cache_clear()
    matrix = cg_matrix_for_node(2, 2, 2)
    assert (matrix != 0).sum() == len(exact)
    for m1, m2, _, value in exact:
        assert matrix[int(m1)+2, int(m2)+2] == float(value)
    assert cg._cg_table_exact.cache_info().misses == 1
    assert cg._cg_table_exact.cache_info().hits == 4


def _all_tree_reference(spins, bracketing, start=0):
    """Small exhaustive path reference, independent of production pruning."""
    if len(spins) == 1:
        return [(spins[0], ("leaf", start, spins[0]))]
    if bracketing == "balanced":
        # Independent level-by-level topology, including the odd carry.
        widths = [1] * len(spins)
        while len(widths) > 2:
            widths = [sum(widths[i:i+2]) for i in range(0, len(widths), 2)]
        split = widths[0]
    else:
        split = 1 if bracketing == "right" else len(spins)-1
    return [(L, ("merge", L, left, right))
            for l1, left in _all_tree_reference(spins[:split], bracketing, start)
            for l2, right in _all_tree_reference(spins[split:], bracketing, start+split)
            for L in range(abs(l1-l2), l1+l2+1)]


@pytest.mark.parametrize("bracketing", ["left", "right", "balanced"])
@pytest.mark.parametrize("spins,target", [((1, 2, 1, 2), 0), ((1, 1, 1), 2), ((2,), 2), ((2, 0), 3)])
def test_selected_paths_preserve_canonical_indices_and_typed_routes(spins, target, bracketing):
    from ye3t import global_coupler as core

    expected = tuple((index, tree) for index, (L, tree) in enumerate(
        _all_tree_reference(spins, bracketing)) if L == target)
    assert core._indexed_angular_paths(spins, target, bracketing) == expected
    assert core._typed_angular_paths(spins, target, bracketing) == tuple(t for _, t in expected)

    def unbind(tree):
        if tree["kind"] == "leaf":
            return ("leaf", tree["index"], tree["L"])
        assert tree["coefficient_table"] == core._pair_cg_table(
            tree["left_L"], tree["right_L"], tree["L"])
        return ("merge", tree["L"], unbind(tree["left"]), unbind(tree["right"]))

    actual = core._angular_factorized_paths(spins, target, bracketing)
    assert tuple((row["path_index"], unbind(row["tree"])) for row in actual) == expected


def test_only_selected_tree_edges_bind_coefficients(monkeypatch):
    from ye3t import global_coupler as core

    # Two spin-four inputs need only the scalar root table in this request.
    original = core._pair_cg_table
    calls = []
    def selected_only(left, right, target):
        calls.append((left, right, target))
        assert (left, right, target) == (4, 4, 0)
        return original(left, right, target)
    monkeypatch.setattr(core, "_pair_cg_table", selected_only)
    paths = core._angular_factorized_paths((4, 4), 0, "balanced")
    assert len(paths) == 1
    assert calls == [(4, 4, 0)]
    assert core._angular_factorized_paths((4, 4), 9, "balanced") == ()
    assert calls == [(4, 4, 0)]
