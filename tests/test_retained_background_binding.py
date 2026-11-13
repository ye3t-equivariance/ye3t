"""Background evaluation choices retain the shared normalized coefficient maps."""
import pytest

from ye3t import couplings


def request(context="inclusive", tags=2):
    return couplings.tagged_cauchy_carriers_request([
        dict(neighbor_species="Ni", radial_channel=i, l=ell,
             source_family_id="chebyshev_c2_origin_regular_v1") for i, ell in enumerate((1, 2))],
        [2, 2], tag_count=tags, target_Ls=[0, 1, 2], angular_basis_backend="normalized_cg_paths_v1",
        density_context=context)


@pytest.mark.parametrize("tags", [1, 2, 3])
def test_background_count_and_coefficients_share_exact_construction(tags):
    old, new = [couplings.compile(request(context, tags)) for context in ("inclusive", "motif_excluded")]
    assert old["component_count"] == new["component_count"]
    assert old["multiplet_count"] == new["multiplet_count"]
    assert old["factorized_schedule"]["programs"] == new["factorized_schedule"]["programs"]
    assert old["self_hash"] != new["self_hash"]
    assert new["certificate"]["density_context"] == "motif_excluded"
    schedule = couplings.tagged_cauchy_carrier_schedule([new])
    assert schedule["density_context"] == "motif_excluded"
    with pytest.raises(ValueError, match="density_context"):
        couplings.tagged_cauchy_carrier_schedule([old, new])
    with pytest.raises(ValueError, match="marginal"):
        couplings.tagged_cauchy_carrier_schedule([new], support_realization="edge_marginal")


def test_zero_tags_keep_canonical_ordinary_density_identity():
    assert request("inclusive", 0) == request("motif_excluded", 0)
    with pytest.raises(ValueError, match="density_context"):
        request("unknown")
    raw = request()
    with pytest.raises(ValueError, match="normalized"):
        couplings.tagged_cauchy_carriers_request(raw["channels"], raw["block_sizes"], tag_count=2,
                                               density_context="motif_excluded")


def test_source_realization_requires_known_background():
    from ye3t.execution_plan import YE3TSourceRealization
    for context in ("inclusive", "motif_excluded"):
        source = YE3TSourceRealization("tagged_cauchy_occurrence", 3,
            injective=True, retain_role_order=True, metadata=dict(tag_count=2, density_context=context))
        assert source.metadata["density_context"] == context
    for context in (None, "unknown"):
        with pytest.raises(ValueError, match="density_context"):
            YE3TSourceRealization("tagged_cauchy_occurrence", 3, injective=True,
                retain_role_order=True, metadata=dict(tag_count=2, density_context=context))


def test_catalogue_inventory_keeps_background_binding():
    configured = couplings.tagged_cauchy_carriers_request(catalogue=dict(ranks=[2], nmax=1, lmax=1,
        input_Lmax=2, tag_counts=[0, 1, 2], max_source_blocks=1,
        angular_basis_backend="normalized_cg_paths_v1", density_context="motif_excluded"), species=["Ni"])
    report = couplings.count(configured)
    assert report["graph_inventory"]
    for row in report["graph_inventory"]:
        assert row["full_path"].get("density_context", "inclusive") == (
            "motif_excluded" if row["tag_count"] else "inclusive")
