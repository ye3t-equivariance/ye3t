"""Exact physical witnesses from tensor_product_math.md, sections 9 and 19."""
import sympy as sp
import pytest

from ye3t import couplings
from ye3t.couplings import lifted_cauchy_scalar as scalar


def _magnetic(vector):
    x, y, z = vector
    return ((x-sp.I*y)/sp.sqrt(2), z, -(x+sp.I*y)/sp.sqrt(2))


@pytest.mark.parametrize("rank,occupation,partition,L", [
    (4, (3, 1), (3, 1), 1), (6, (4, 2), (4, 2), 0)])
def test_compiled_selected_pairing_matches_cartesian_witness(rank, occupation, partition, L):
    request = couplings.covariant_cauchy_request(
        [{"neighbor_species": "X", "radial_channel": 0, "l": 1,
          "source_family_id": "independent"}], [rank], target_L=L,
        role_dimension=2, role_occupations=[occupation])
    counted = couplings.count(request)
    selected = tuple(i for i, label in enumerate(counted["labels"])
                     if tuple(label["block_kappas"][0]) == partition)
    assert len(selected) == 1
    compiled = couplings.compile(couplings.plan({**request, "selected_multiplicities": selected}))
    a = sp.Matrix(sp.symbols("ax ay az", real=True))
    b = sp.Matrix(sp.symbols("bx by bz", real=True))
    inputs = (_magnetic(a), _magnetic(b))
    expected = _magnetic(a.dot(a)*a.cross(b)) if L == 1 else (
        a.dot(a)*(a.dot(a)*b.dot(b)-a.dot(b)**2),)
    actual = []
    for row in compiled["descriptors"][0]["canonical_terms_by_M"]:
        polynomial = sum(value*sp.prod(inputs[role][m] for channel, role, m in coordinates)
                         for coordinates, value in scalar._terms_from_records(row["terms"]).items())
        actual.append(sp.expand(polynomial))
    # Historical coordinate normalization may multiply the whole multiplet by
    # one constant. Establish exact polynomial identity, including its phase.
    variables = tuple(a)+tuple(b)
    reference = sp.Poly(expected[0], *variables)
    monomial, coefficient = reference.terms()[0]
    scale = sp.simplify(sp.Poly(actual[0], *variables).coeff_monomial(monomial)/coefficient)
    assert scale != 0
    for value, witness in zip(actual, expected, strict=True):
        assert sp.expand(value-scale*witness) == 0
    # All evaluated dependencies below are exact, including differentiated
    # identities. No sampled feature-matrix rank is used.
    for variable in variables:
        assert all(sp.expand(sp.diff(value-scale*witness, variable)) == 0
                   for value, witness in zip(actual, expected, strict=True))


def test_retained_tag_pieri_inventory_matches_the_note():
    request = couplings.tagged_cauchy_carriers_request(
        [{"neighbor_species": "X", "radial_channel": 0, "l": 1,
          "source_family_id": "orthogonal_shifted_jacobi_origin_regular_v1"}],
        [4], tag_count=2, target_Ls=(0, 1, 2, 3))
    labels = couplings.count(request)["labels"]
    by_partition = lambda p: {(row["target_L"], tuple(row["tag_partition"]))
                             for row in labels if tuple(row["block_kappas"][0]) == p}
    assert by_partition((3, 1)) == {(L, tau) for L in (1, 2, 3) for tau in ((2,), (1, 1))}
    assert by_partition((2, 2)) == {(0, (2,)), (2, (2,))}


def test_inclusive_tag_marginals_do_not_inherit_formal_nonzero_counts():
    vectors = [sp.Matrix(sp.symbols(f"v{i}x v{i}y v{i}z")) for i in range(3)]
    A = sum(vectors, sp.zeros(3, 1))
    Bsum = sum((A.dot(A)*A.cross(v) for v in vectors), sp.zeros(3, 1))
    assert all(sp.expand(value) == 0 for value in Bsum)
    ordered = sum(A.dot(A)*(A.dot(A)*v.dot(w)-A.dot(v)*A.dot(w))
                  for j, v in enumerate(vectors) for k, w in enumerate(vectors) if j != k)
    expected = -A.dot(A)*sum(A.cross(v).dot(A.cross(v)) for v in vectors)
    assert sp.expand(ordered-expected) == 0
