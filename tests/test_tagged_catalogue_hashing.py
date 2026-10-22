"""Catalogue hash identity without a catalogue-sized normalized copy."""
import json

import pytest

from ye3t import couplings
from ye3t.couplings import tagged_cauchy_carriers as core


@pytest.mark.parametrize("sources", [[], (), [{"values": [0., -0., 1e-12, True, None],
                                            "label": "κ\\\"\n", 2: (1, 3)}]])
def test_sourcewise_hash_matches_canonical_bytes_and_bounds_normalization(sources, monkeypatch):
    payload = {"sources": sources, "schema": "test", "request": {"z": 3, "a": 1}}
    expected = core._scalar._stable_hash(core._scalar._freeze_json(payload))
    original = core._scalar._freeze_json

    def bounded(value):
        assert value is not payload and value is not sources
        return original(value)

    monkeypatch.setattr(core._scalar, "_freeze_json", bounded)
    assert core._catalogue_hash(payload) == expected


def test_catalogue_hash_cache_roundtrip_and_corruption_rejection(tmp_path):
    request = couplings.tagged_cauchy_carriers_request(species=["Ni"], catalogue={
        "ranks": [1, 2], "nmax": 1, "lmax": 1, "input_Lmax": 1,
        "max_source_blocks": 2, "tag_counts": [0, 1]})
    compiled = couplings.compile(request, cache_dir=tmp_path)
    expected = core._scalar._stable_hash(core._scalar._freeze_json(
        {key: value for key, value in compiled.items() if key != "self_hash"}))
    assert compiled["self_hash"] == expected
    restored = couplings.compile(request, cache_dir=tmp_path)
    assert core._scalar._freeze_json(restored) == core._scalar._freeze_json(compiled)
    core.tagged_cauchy_carrier_model_plan(restored)
    path = next(tmp_path.glob("catalogue_*.json"))
    damaged = json.loads(path.read_text())
    damaged["sources"][0]["multiplet_count"] += 1
    path.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="catalogue cache hash mismatch"):
        couplings.compile(request, cache_dir=tmp_path)


def test_nested_sources_and_descriptors_hash_without_normalizing_collections(monkeypatch):
    rows = [{"label": {"L": 2}, "terms": [{"coefficient": [1, 3], "factors": ((1, 0),)}]}]
    source = {"descriptors": rows, "request": {"rank": 4}, "schema": "source"}
    payload = {"sources": [source], "schema": "catalogue"}
    expected = core._scalar._stable_hash(core._scalar._freeze_json(payload))
    original = core._scalar._freeze_json
    forbidden = {id(payload), id(payload["sources"]), id(source), id(rows)}

    def bounded(value):
        assert id(value) not in forbidden
        return original(value)

    monkeypatch.setattr(core._scalar, "_freeze_json", bounded)
    assert core._catalogue_hash(payload) == expected
