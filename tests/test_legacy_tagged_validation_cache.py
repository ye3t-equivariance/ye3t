"""Historical saved-map reuse remains isolated from current compilation."""
import json
import pytest
from ye3t import couplings
from ye3t.couplings import tagged_cauchy_carriers as core
from ye3t.cache.artifacts import ArtifactCacheValidationError


def request():
    return couplings.tagged_cauchy_carriers_request(catalogue={
        "ranks": [2], "nmax": 1, "lmax": 1, "tag_counts": [1, 2],
        "input_Lmax": 1, "max_source_blocks": 1}, species=["Ni"])


def test_legacy_cache_replays_exact_maps_and_isolates_production(tmp_path, monkeypatch):
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    expected = core._reconstruct_legacy_tagged_catalogue_for_validation(request())
    path = next((tmp_path / "artifacts/legacy_tagged_validation").glob("*.json"))
    monkeypatch.setattr(core, "_compile_catalogue", lambda *a, **k: pytest.fail("Historical maps were rebuilt"))
    actual = core._reconstruct_legacy_tagged_catalogue_for_validation(request())
    assert actual == expected
    assert actual["schema"] == "ye3t_tagged_cauchy_carrier_catalogue_v1"
    current = couplings.count(request())
    assert current["request"]["catalogue"].get("selection_policy") != actual["request"]["catalogue"]["selection_policy"]
    payload = json.loads(path.read_text())
    payload["payload"]["sources"][0]["multiplet_count"] += 1
    path.write_text(json.dumps(payload))
    monkeypatch.setenv("YE3T_CACHE_MODE", "read_only")
    with pytest.raises(ArtifactCacheValidationError):
        core._reconstruct_legacy_tagged_catalogue_for_validation(request())


def test_historical_cache_rejects_new_selection_interface(tmp_path, monkeypatch):
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    configured = request()
    configured["catalogue"]["source_sectors"] = {"young_lambda": "all"}
    with pytest.raises(ValueError):
        core._reconstruct_legacy_tagged_catalogue_for_validation(configured)
