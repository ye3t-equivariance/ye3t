"""Angular-table reuse preserves exact coefficients and artifact isolation."""

import copy

import pytest

from ye3t.couplings import covariant_cauchy as core


@pytest.mark.parametrize("target", [1, 2])
def test_combined_real_source_matches_uncached_exact_simplification(target, monkeypatch):
    import sympy as sp
    request = core.covariant_cauchy_request([
        {"neighbor_species": "Ni", "radial_channel": 0, "l": 1, "source_family_id": "independent"},
        {"neighbor_species": "Ni", "radial_channel": 1, "l": 1, "source_family_id": "independent"}],
        (2, 1), target_L=target, role_dimension=3)
    core._scalar._simplified_scalar_cached.cache_clear()
    cached = core.compile_covariant_cauchy(request)
    assert core._scalar._simplified_scalar_cached.cache_info().hits > 0
    with monkeypatch.context() as patch:
        patch.setattr(core._scalar, "_simplified_scalar", sp.simplify)
        uncached = core.compile_covariant_cauchy(request)
    assert cached == uncached
    assert core.validate_covariant_cauchy(cached)


def test_coalesced_rows_share_only_immutable_exact_scalars(monkeypatch):
    import sympy as sp
    scalar = core._scalar
    x, y = (0, 0, -1), (1, 1, 1)
    terms = {(x, y): sp.sqrt(8), (y, x): -sp.sqrt(2),
             (x, x): sp.sqrt(3)**2 - 3}
    scalar._simplified_scalar_cached.cache_clear()
    first = scalar._coalesce_terms(terms)
    second = scalar._coalesce_terms(terms)
    assert scalar._simplified_scalar_cached.cache_info().hits > 0
    with monkeypatch.context() as patch:
        patch.setattr(scalar, "_simplified_scalar", sp.simplify)
        assert first == scalar._coalesce_terms(terms) == {(x, y): sp.sqrt(2)}
    first.clear()
    assert second == {(x, y): sp.sqrt(2)}


@pytest.mark.parametrize("key", [(3, 2, (1, 1), 1, 1), (3, 3, (2, 1), 1, 1)])
def test_metric_dual_scalar_cache_matches_uncached_complete_block(key, monkeypatch):
    import sympy as sp
    scalar = core._scalar
    scalar._simplified_scalar_cached.cache_clear()
    cached = scalar._build_block_template(key)
    assert scalar._simplified_scalar_cached.cache_info().hits > 0
    with monkeypatch.context() as patch:
        patch.setattr(scalar, "_simplified_scalar", sp.simplify)
        patch.setattr(scalar, "_exact_sympy_scalar_payload_cached",
                      scalar._exact_sympy_scalar_payload_cached.__wrapped__)
        uncached = scalar._build_block_template(key)
    assert cached == uncached
    scalar._validate_block_template(cached["payload"])


def test_exact_scalar_simplification_cache_preserves_complex_values_and_payload_isolation():
    import sympy as sp
    scalar = core._scalar
    value = (sp.sqrt(8) + sp.I*sp.sqrt(12))/3
    scalar._simplified_scalar_cached.cache_clear()
    scalar._exact_sympy_scalar_payload_cached.cache_clear()
    first = scalar._exact_scalar_payload(value)
    reference = copy.deepcopy(first)
    assert sp.simplify(scalar._exact_scalar_from_payload(first)-value) == 0
    first["real"][0]["coefficient_numerator"] = 999
    first["binary64"] = (999., 999.)
    assert scalar._exact_scalar_payload(value) == reference
    assert scalar._simplified_scalar_cached.cache_info().hits > 0
    assert scalar._exact_sympy_scalar_payload_cached.cache_info().hits > 0


@pytest.mark.parametrize("spins,target", [((1, 1), 1), ((1, 2), 2), ((1, 1, 1), 1)])
def test_outer_template_cache_matches_exact_builder_and_isolates_mutation(spins, target):
    expected = core._cached_outer_multiplet_template.__wrapped__(spins, target)
    core._cached_outer_multiplet_template.cache_clear()
    first = core._outer_multiplet_template(list(spins), target)
    assert first == expected
    unchanged = copy.deepcopy(first)
    first["analysis_rows"][0]["terms"][0]["coefficient"]["binary64"] = (900., 900.)
    first["validation_report"]["passed"] = False
    second = core._outer_multiplet_template(spins, target)
    assert second == unchanged
    stats = core._cached_outer_multiplet_template.cache_info()
    assert stats.misses == 1 and stats.hits == 1


def test_compiled_source_reuses_only_angular_identity():
    core._cached_outer_multiplet_template.cache_clear()
    def request(radial):
        return core.covariant_cauchy_request([
            {"neighbor_species": "Ni", "radial_channel": radial, "l": 1, "source_family_id": "independent"},
            {"neighbor_species": "Ni", "radial_channel": radial + 1, "l": 1, "source_family_id": "independent"}],
            (1, 1), target_L=1, role_dimension=2, role_occupations=[(1, 1)])
    first = core.compile_covariant_cauchy(request(0))
    before = core._cached_outer_multiplet_template.cache_info()
    second = core.compile_covariant_cauchy(request(2))
    after = core._cached_outer_multiplet_template.cache_info()
    assert after.misses == before.misses
    assert after.hits > before.hits
    assert first["outer_templates"] == second["outer_templates"]
    assert first["request"]["channels"] != second["request"]["channels"]
    assert first["self_hash"] != second["self_hash"]
    assert core.validate_covariant_cauchy(first)
    assert core.validate_covariant_cauchy(second)


def test_each_block_is_decoded_once_for_all_its_role_copies(monkeypatch):
    decoder = core._scalar._block_template_from_payload
    calls = []
    def recorded(payload):
        calls.append(payload["template_hash"])
        return decoder(payload)
    monkeypatch.setattr(core._scalar, "_block_template_from_payload", recorded)
    request = core.covariant_cauchy_request([
        {"neighbor_species": "Ni", "radial_channel": 0, "l": 1, "source_family_id": "independent"}],
        (3,), target_L=1, role_dimension=3)
    result = core.compile_covariant_cauchy(request)
    assert len(result["descriptors"]) > len(result["block_templates"])
    assert sorted(calls) == sorted(row["template_hash"] for row in result["block_templates"])
    assert core.validate_covariant_cauchy(result)
    original = core._canonical_complex_rows
    def decode_per_descriptor(label, request, blocks, outer):
        decoded = {key: decoder(value["payload"]) for key, value in blocks.items()}
        return original(label, request, decoded, outer)
    monkeypatch.setattr(core, "_canonical_complex_rows", decode_per_descriptor)
    reference = core.compile_covariant_cauchy(request)
    assert result == reference


def test_catalogue_local_block_reuse_preserves_coefficients_and_source_identity(monkeypatch):
    def request(radial):
        return core.covariant_cauchy_request([
            {"neighbor_species": "Ni", "radial_channel": radial, "l": 1,
             "source_family_id": "independent"}],
            (2,), target_L=2, role_dimension=2)
    expected = core.compile_covariant_cauchy(request(1))
    shared = {}
    first = core.compile_covariant_cauchy(request(0), _block_templates=shared)
    assert shared
    def forbidden(*args, **kwargs):
        raise AssertionError("Identical blocks were already validated in this catalogue")
    monkeypatch.setattr(core._scalar, "_compile_block_template", forbidden)
    second = core.compile_covariant_cauchy(request(1), _block_templates=shared)
    assert second == expected
    assert second["self_hash"] != first["self_hash"]
    assert core.validate_covariant_cauchy(second)
