"""Cache-streamed lowering preserves canonical complete graph artifacts."""
import json

import pytest

from ye3t import couplings
from ye3t.couplings import tagged_cauchy_carriers as core


def _request(shell):
    catalogue = {"ranks": [1, 2, 3], "nmax": 1, "lmax": 1,
                 "input_Lmax": 1, "max_source_blocks": 2, "tag_counts": [0, 1, 2, 3]}
    if shell:
        source = couplings.shell_filtered_source_plan("shell_filtered_chebyshev_c2_v1", 1, 1,
            {"kind": "softmax_gaussian", "centers": [.25, .5, .8], "width": .2},
            evaluation_coordinates="physical_shells")
        catalogue.update(tag_counts=[0], source_family_id=source["family_id"], source_family_plan=source,
                         basis_coordinates="formal_cauchy", shell_role_action="permutation")
    return couplings.tagged_cauchy_carriers_request(species=["Ni"], catalogue=catalogue)


@pytest.mark.parametrize("shell,support", [(True, "ordered_tags"), (False, "ordered_tags"),
                                           (False, "edge_marginal")])
def test_streamed_graph_matches_eager_complete_payload(shell, support, tmp_path, monkeypatch):
    request = _request(shell)
    if support == "edge_marginal":
        request["catalogue"]["tag_counts"] = [0, 1, 2]
        # Rebuild the canonical identity after editing this test request.
        request = couplings.tagged_cauchy_carriers_request(species=["Ni"], catalogue=request["catalogue"])
    plan = couplings.plan(request)
    sectors = None if support == "edge_marginal" else [
        {"L": L, "parity": "even" if parity == 1 else "odd"}
        for L, parity in sorted({(row["layout"]["key"]["rotation_L"], row["layout"]["key"]["parity"])
                                for row in plan["graph_inventory"]})]
    options = {"target_L": 0, "parity": 1, "hidden_channels": 1,
               "output_sectors": sectors, "support_realization": support}
    with monkeypatch.context() as uncached:
        uncached.setattr(core._scalar, "_real_float_from_exact_payload",
                        lambda payload: float(core._scalar._exact_scalar_from_payload(payload)))
        eager = couplings.tagged_cauchy_marginal_graph_records(couplings.compile(plan), **options)
    # Exercise both a cold cache and subsequent reload, without a consolidated
    # full-catalogue object being retained by the streaming implementation.
    streamed = couplings.compile(plan, cache_dir=tmp_path, graph_options=options)
    restored = couplings.compile(plan, cache_dir=tmp_path, graph_options=options)
    assert core._scalar._freeze_json(eager) == core._scalar._freeze_json(streamed)
    assert restored == streamed
    assert not list(tmp_path.glob("catalogue_*.json"))
    json.dumps(streamed)  # Runtime artifacts contain no cache-backed objects.
    assert {row["binding"]["actual_action"] for row in streamed["records"]} == {
        row["binding"]["actual_action"] for row in eager["records"]}


def test_cached_source_rejects_replacement_after_header_capture(tmp_path):
    request = _request(True)
    compiled = core._compile_catalogue(request, cache_dir=tmp_path, _stream_sources=True)
    sources = compiled["sources"]
    path, header = sources.entries[0]
    payload = json.loads(path.read_text())
    payload["validation_report"]["extra_test_metadata"] = True
    payload["self_hash"] = core._scalar._stable_record_hash(
        {key: value for key, value in payload.items() if key != "self_hash"})
    core.validate_tagged_cauchy_carriers(payload)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Cached source changed"):
        sources[0]


def test_validation_reuse_requires_identical_current_file_bytes(tmp_path, monkeypatch):
    compiled = core._compile_catalogue(_request(True), cache_dir=tmp_path, _stream_sources=True)
    sources = compiled["sources"]
    path, header = sources.entries[0]
    original = core.validate_tagged_cauchy_carriers
    calls = []

    def observed(payload, **kwargs):
        calls.append(payload["self_hash"])
        return original(payload, **kwargs)

    monkeypatch.setattr(core, "validate_tagged_cauchy_carriers", observed)
    expected = sources[0]
    assert sources.with_tag_count(header["request"]["tag_count"])[0] == expected
    assert sources[:1][0] == expected
    assert calls == []
    # Even equivalent JSON must receive full validation when its bytes change.
    path.write_text(json.dumps(expected, indent=1))
    assert sources[0] == expected
    assert calls == [expected["self_hash"]]
    assert sources[0] == expected
    assert len(calls) == 1
    damaged = json.loads(path.read_text())
    damaged["self_hash"] = "0"*64
    path.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="Formal shell source hash mismatch"):
        sources[0]
    assert len(calls) == 2


def test_runtime_scalar_cache_preserves_exact_conversion_and_input_changes():
    scalar = core._scalar
    sp = scalar._sympy()
    for value in (sp.Rational(2, 7), sp.sqrt(3)/5-sp.sqrt(2)/7, -sp.sqrt(11), sp.Integer(0)):
        payload = scalar._exact_scalar_payload(value)
        expected = float(scalar._exact_scalar_from_payload(payload))
        assert scalar._real_float_from_exact_payload(payload) == expected
        payload["binary64"] = [123., 456.]
        assert scalar._real_float_from_exact_payload(payload) == expected
        if payload["real"]:
            payload["real"][0]["coefficient_numerator"] += 1
            assert scalar._real_float_from_exact_payload(payload) == float(scalar._exact_scalar_from_payload(payload))
    with pytest.raises(TypeError):
        scalar._real_float_from_exact_payload(scalar._exact_scalar_payload(1+sp.I))
