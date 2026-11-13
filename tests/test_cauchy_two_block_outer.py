"""Direct exact angular coefficients preserve the canonical Cauchy gauge."""
from itertools import product

import pytest
import sympy as sp

from ye3t.couplings import covariant_cauchy as core
from ye3t.representations.builder import GeneralizedExactSymbolicLabeler
from ye3t.representations.generalized_irreps import Partition


def projector_reference(spins, target):
    """Independent, bounded old projector construction, for tests only."""
    data = GeneralizedExactSymbolicLabeler(tuple(range(len(spins))), spins,
        spatial_symmetry="O3").sector_for_partitions(tuple(Partition((1,)) for _ in spins))
    copies = data.joint_multiplicity_by_L[target]
    states = tuple(product(*[range(-value, value+1) for value in spins]))
    rows = []
    gram = None
    for M in range(-target, target+1):
        synthesis = sp.Matrix.hstack(*[
            sp.Matrix(data.lowered_multiplets_by_L[target][(copy, (0,)*len(spins))][M])
            for copy in range(copies)])
        component_gram, analysis = core._scalar._metric_dual_analysis(synthesis)
        if gram is None:
            gram = component_gram
        assert component_gram == gram
        for copy in range(copies):
            terms = {state: sp.simplify(analysis[copy, index])
                     for index, state in enumerate(states) if analysis[copy, index] != 0}
            rows.append({"outer_copy_index": copy, "M": M, "terms": core._scalar._term_records(terms)})
    payload = {"input_Ls": spins, "target_L": target, "copy_count": copies,
        "coordinate_order": tuple((copy, M) for copy in range(copies) for M in range(-target, target+1)),
        "analysis_rows": tuple(sorted(rows, key=lambda row: (row["outer_copy_index"], row["M"]))),
        "multiplicity_gram": core._scalar._exact_matrix_payload(gram),
        "validation_report": {"passed": True, "complete_multiplet": True,
                              "gram_independent_of_M_exact": True}}
    payload["template_id"] = "outer_multiplet_" + core._scalar._stable_hash(
        {"input_Ls": spins, "target_L": target})[:16]
    payload["template_hash"] = core._scalar._stable_hash(core._scalar._freeze_json(payload))
    return payload


@pytest.mark.parametrize("spins,target", [
    ((0, 2), 2), ((1, 1), 0), ((1, 1), 1), ((1, 1), 2),
    ((1, 2), 1), ((1, 2), 2), ((1, 2), 3), ((2, 2), 0), ((2, 2), 2)])
def test_direct_two_block_payload_is_exactly_the_projector_payload(spins, target, monkeypatch):
    expected = projector_reference(spins, target)
    core._cached_outer_multiplet_template.cache_clear()
    monkeypatch.setattr(GeneralizedExactSymbolicLabeler, "sector_for_partitions",
        lambda *args, **kwargs: pytest.fail("Two-block production path constructed a projector"))
    assert core._outer_multiplet_template(spins, target) == expected


@pytest.mark.parametrize("spins,target", [
    ((1, 1, 1), 0), ((1, 1, 1), 1), ((1, 1, 1), 2),
    ((0, 1, 2), 2), ((1, 1, 1, 1), 0), ((0, 1, 1, 2), 1)])
def test_requested_weight_payload_matches_full_projector(spins, target, monkeypatch):
    expected = projector_reference(spins, target)
    core._cached_outer_multiplet_template.cache_clear()
    monkeypatch.setattr(GeneralizedExactSymbolicLabeler, "sector_for_partitions",
        lambda *args, **kwargs: pytest.fail("Outer compilation constructed a full projector"))
    assert core._outer_multiplet_template(spins, target) == expected


@pytest.mark.parametrize("ell", [4, 6, 12])
def test_scalar_closed_form_and_exact_generator_action(ell):
    # DLMF 34.3.1 with the existing first-pivot synthesis normalization.
    payload = core._outer_multiplet_template((ell, ell), 0)
    expected = {(m, -m): sp.Rational((-1)**(ell-m), 2*ell+1)
                for m in range(-ell, ell+1)}
    assert payload["analysis_rows"][0]["terms"] == core._scalar._term_records(expected)
    assert payload["multiplicity_gram"] == core._scalar._exact_matrix_payload(sp.Matrix([[2*ell+1]]))
    raised = {}
    for (left, right), coefficient in expected.items():
        for index, m in enumerate((left, right)):
            if m == ell:
                continue
            state = (left+1, right) if index == 0 else (left, right+1)
            raised[state] = raised.get(state, 0) + coefficient*sp.sqrt((ell-m)*(ell+m+1))
    assert all(sp.simplify(value) == 0 for value in raised.values())


@pytest.mark.parametrize("spins,target", [((1, 3), 0), ((1, 1), 3)])
def test_unreachable_target_fails(spins, target):
    with pytest.raises(ValueError, match="no requested target"):
        core._outer_multiplet_template(spins, target)


@pytest.mark.parametrize("sizes,target", [((2, 1), 1), ((1, 1, 1), 1), ((1, 1, 1, 1), 0)])
def test_complete_compiled_cauchy_source_preserves_coefficients(monkeypatch, sizes, target):
    request = core.covariant_cauchy_request([
        {"neighbor_species": "Ni", "radial_channel": index, "l": 1,
         "source_family_id": "independent"} for index in range(len(sizes))],
        sizes, target_L=target, role_dimension=2)
    actual = core.compile_covariant_cauchy(request)
    assert core.validate_covariant_cauchy(actual)
    monkeypatch.setattr(core, "_outer_multiplet_template", projector_reference)
    expected = core.compile_covariant_cauchy(request)
    assert actual == expected
