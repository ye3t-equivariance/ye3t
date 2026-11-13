"""Source counts and graph lowering select one complete carrier convention."""

import pytest

from ye3t import couplings
from ye3t.execution_plan import YE3T_O3_PRIMARY_CONVENTION, YE3T_O3_REAL_TESSERAL_CONVENTION
from test_tagged_streamed_graph import _request


@pytest.mark.parametrize("shell", [False, True])
@pytest.mark.parametrize("normalized", [False, True])
def test_real_inventory_and_compilation_preserve_source_coordinates(tmp_path, shell, normalized):
    catalogue = dict(_request(shell)["catalogue"])
    if normalized:
        catalogue["angular_basis_backend"] = "normalized_cg_paths_v1"
    request = couplings.tagged_cauchy_carriers_request(species=["Ni"], catalogue=catalogue)
    real_request = couplings.tagged_cauchy_carriers_request(species=["Ni"], catalogue={
        **catalogue, "carrier_convention_id": YE3T_O3_REAL_TESSERAL_CONVENTION})
    baseline, real = [couplings.count(value) for value in (request, real_request)]
    assert baseline["multiplet_count"] == real["multiplet_count"]
    assert baseline["component_count"] == real["component_count"]
    assert baseline["selected_records"] == real["selected_records"]
    assert [row["coordinate_ids"] for row in baseline["graph_inventory"]] == [
        row["coordinate_ids"] for row in real["graph_inventory"]]
    sectors = [{"L": L, "parity": "even" if p == 1 else "odd"} for L, p in sorted({
        (row["layout"]["key"]["rotation_L"], row["layout"]["key"]["parity"])
        for row in real["graph_inventory"]})]
    graphs = []
    for report, convention in ((baseline, YE3T_O3_PRIMARY_CONVENTION), (real, YE3T_O3_REAL_TESSERAL_CONVENTION)):
        graph = couplings.compile(couplings.plan(report), cache_dir=tmp_path,
            graph_options={"target_L": 0, "parity": 1, "output_sectors": sectors, "support_realization": "ordered_tags"})
        planned = {row["id"]: row["layout"] for row in report["graph_inventory"]}
        for row in graph["records"]:
            layout = row["binding"]["output_layouts"][row["id"]]
            assert layout["key"]["convention_id"] == convention
            assert layout == planned[row["id"]]
            assert row["binding"]["argument_bindings"]["angular_output"] == (
                "real_tesseral" if convention == YE3T_O3_REAL_TESSERAL_CONVENTION else "complex_condon_shortley")
        graphs.append(graph)
    assert [row["component_indices"] for row in graphs[0]["records"]] == [
        row["component_indices"] for row in graphs[1]["records"]]


@pytest.mark.parametrize("value", ["unverified", [], {}])
def test_unsupported_carrier_convention_fails_before_compilation(value):
    with pytest.raises(ValueError, match="carrier_convention_id"):
        couplings.tagged_cauchy_carriers_request(species=["Ni"], catalogue={
            **_request(False)["catalogue"], "carrier_convention_id": value})


@pytest.mark.parametrize("options", [
    {"support_realization": "edge_marginal"},
    {"support_realization": "ordered_tags", "_legacy_zero_tag_action": True},
])
def test_real_graph_rejects_incompatible_legacy_binding(tmp_path, options):
    from ye3t.couplings.tagged_cauchy_carriers import tagged_cauchy_marginal_graph_records
    request = couplings.tagged_cauchy_carriers_request(species=["Ni"], catalogue={
        **_request(False)["catalogue"], "carrier_convention_id": YE3T_O3_REAL_TESSERAL_CONVENTION})
    compiled = couplings.compile(couplings.plan(couplings.count(request)), cache_dir=tmp_path)
    with pytest.raises(ValueError, match="current zero-tag action"):
        tagged_cauchy_marginal_graph_records(compiled, target_L=0, parity=1, **options)
