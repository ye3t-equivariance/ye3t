"""Normal construction and explicitly requested diagnostics stay distinct."""

from ye3t.representations import young_subgroup_specht_coupling as specht
from ye3t._record import record_replace


def test_specht_construction_does_not_run_mathematical_proof(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("cold construction entered the explicit validator")
    monkeypatch.setattr(specht, "validate_young_orthogonal_nary_subduction", forbidden)
    coupling = specht.build_young_subgroup_specht_coupling(((2,), (1,)), (2,1))
    assert coupling.validation.passed
    assert coupling.validation.gram_matrix is None
    assert "no runtime mathematical diagnostic" in coupling.validation.detail


def test_explicit_specht_diagnostic_still_checks_coefficients():
    coupling = specht.build_young_subgroup_specht_coupling(((2,), (1,)), (2,1))
    report = specht.validate_young_subgroup_specht_coupling(coupling)
    assert report["passed"] and report["validation"]["intermediate_subspaces_match"]
    assert report["validation"]["performed"]
    assert report["projector_validation"]["orthonormal_columns"]


def test_metadata_inspection_does_not_run_projector_diagnostic(monkeypatch):
    coupling = specht.build_young_subgroup_specht_coupling(((2,), (1,)), (2,1))
    def forbidden(*args, **kwargs):
        raise AssertionError("metadata inspection entered a projector diagnostic")
    monkeypatch.setattr(specht.YoungSubgroupSpechtCoupling, "projector_report", forbidden)
    payload = coupling.as_dict()
    assert payload["projector_validation"] is None
    assert payload["validation"]["performed"] is False


def test_explicit_diagnostic_does_not_trust_stale_constructor_metadata():
    coupling = specht.build_young_subgroup_specht_coupling(((1,),)*3, (2,1))
    corrupt = record_replace(coupling.tensor,
                             lr_chain_labels=tuple(reversed(coupling.tensor.lr_chain_labels)))
    wrong = record_replace(coupling, tensor=corrupt)
    assert wrong.validation.passed  # Deliberately stale construction metadata.
    report = specht.validate_young_subgroup_specht_coupling(wrong)
    assert report["validation"]["orthonormal"]
    assert report["validation"]["generator_equivariant"]
    assert report["validation"]["intermediate_subspaces_match"] is False
    assert report["passed"] is False


def test_global_metadata_does_not_invent_unmeasured_residuals():
    from ye3t.global_coupler import YoungSubductionMap
    coupling = specht.build_young_subgroup_specht_coupling(((2,), (1,)), (2,1))
    mapped = YoungSubductionMap.from_coupling(coupling)
    assert mapped.source_validation["performed"] is False
    assert "generator_equivariance" not in mapped.validation_residuals
    assert "multiplicity_character" not in mapped.validation_residuals
    assert mapped.validation_residuals["column_orthonormality"] == (
        mapped.projector_validation["orthonormality_error"])


def test_legacy_numeric_copy_coordinates_require_explicit_conversion(tmp_path):
    import torch
    exact = specht.build_young_subgroup_specht_coupling(((1,),)*3, (2,1), bracketing="right")
    legacy = specht.build_cached_young_subgroup_specht_coupling(
        ((1,),)*3, (2,1), bracketing="right", cache_dir=tmp_path)
    canonical = specht._numeric_coefficients_from_coupling(exact)
    historical = specht._numeric_coefficients_from_coupling(legacy)
    overlap = historical.T @ canonical
    assert not torch.allclose(canonical, historical, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(historical @ overlap, canonical, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(overlap.T @ overlap, torch.eye(4, dtype=torch.float64),
                               rtol=1e-12, atol=1e-12)
    values = torch.tensor([[.2, -.3, .4, .1, -.6, .5]], dtype=torch.float64)
    weights = torch.tensor([.7, .2, -.8, .3], dtype=torch.float64)
    before = values @ historical @ (overlap @ weights)
    after = values @ canonical @ weights
    torch.testing.assert_close(before, after, rtol=1e-12, atol=1e-12)
