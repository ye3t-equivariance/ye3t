"""Persistent reuse of the existing exact occupation subgroup coordinates."""
import json

import pytest

from ye3t.cache.artifacts import ArtifactCacheMiss, artifact_cache_events
from ye3t.representations import young_angular_paths as shared


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("YE3T_CACHE_MODE", "auto")
    monkeypatch.setenv("YE3T_CACHE_VERIFY", "hash")
    shared.occupation_young_vectors.cache_clear()
    yield
    shared.occupation_young_vectors.cache_clear()


@pytest.mark.parametrize("occupation,partition", [
    ((2,1,1),(3,1)), ((3,2,1),(4,2)), ((4,2,2),(6,2))])
def test_exact_copy_order_survives_persistent_reload(occupation,partition,monkeypatch):
    children = tuple((n,) for n in occupation)
    actual = shared.occupation_young_vectors(occupation,partition)
    expected = shared._construct_occupation_young_vectors(children,partition,len(actual))
    assert actual == expected and actual
    shared.occupation_young_vectors.cache_clear()
    def forbidden(*args,**kwargs):
        pytest.fail("Verified occupation cache hit reconstructed the exact subgroup map")
    monkeypatch.setattr(shared,"_construct_occupation_young_vectors",forbidden)
    monkeypatch.setenv("YE3T_CACHE_MODE","read_only")
    assert shared.occupation_young_vectors(occupation,partition) == expected


def test_empty_role_coordinates_share_the_abstract_map(monkeypatch):
    expected = shared.occupation_young_vectors((2,1),(2,1))
    shared.occupation_young_vectors.cache_clear()
    def forbidden(*args,**kwargs):
        pytest.fail("An empty role triggered reconstruction of an unchanged subgroup")
    monkeypatch.setattr(shared,"_construct_occupation_young_vectors",forbidden)
    assert shared.occupation_young_vectors((0,2,0,1),(2,1)) == expected


def test_full_verification_replays_the_existing_constructor(monkeypatch):
    expected = shared.occupation_young_vectors((2,1,1),(3,1))
    shared.occupation_young_vectors.cache_clear()
    original,calls = shared._construct_occupation_young_vectors,[]
    def observed(*args):
        calls.append(args)
        return original(*args)
    monkeypatch.setattr(shared,"_construct_occupation_young_vectors",observed)
    monkeypatch.setenv("YE3T_CACHE_VERIFY","full")
    assert shared.occupation_young_vectors((2,1,1),(3,1)) == expected
    assert len(calls) == 1


def test_read_only_miss_and_disabled_cache(monkeypatch,tmp_path):
    monkeypatch.setenv("YE3T_CACHE_MODE","read_only")
    with pytest.raises(ArtifactCacheMiss):
        shared.occupation_young_vectors((2,1),(2,1))
    monkeypatch.setenv("YE3T_CACHE_MODE","off")
    assert shared.occupation_young_vectors((2,1),(2,1))
    assert not tuple(tmp_path.rglob("*.json"))


def test_corrupt_cache_is_rejected_and_rebuilt(tmp_path):
    expected = shared.occupation_young_vectors((2,1),(2,1))
    path, = (tmp_path/"artifacts"/"occupation_young_vectors").glob("*.json")
    payload = json.loads(path.read_text())
    payload["payload"]["copies"] = []
    path.write_text(json.dumps(payload))
    shared.occupation_young_vectors.cache_clear()
    artifact_cache_events(clear=True)
    assert shared.occupation_young_vectors((2,1),(2,1)) == expected
    assert any(event["status"] == "reject" for event in artifact_cache_events())


def test_fresh_process_reuses_exact_coordinates_without_construction(monkeypatch):
    import os
    from pathlib import Path
    import subprocess
    import sys
    expected = shared.occupation_young_vectors((2,1,1),(3,1))
    result = subprocess.run([sys.executable,str(Path(__file__).resolve())],
        env=dict(os.environ,YE3T_CACHE_MODE="read_only",YE3T_CACHE_VERIFY="hash"),
        text=True,capture_output=True,timeout=60)
    assert result.returncode == 0,result.stderr
    assert json.loads(result.stdout) == [[[t,str(value)] for t,value in row] for row in expected]


if __name__ == "__main__":
    from unittest.mock import patch
    with patch.object(shared,"_construct_occupation_young_vectors",
                      side_effect=RuntimeError("Fresh-process cache hit rebuilt coefficients")):
        rows = shared.occupation_young_vectors((2,1,1),(3,1))
    print(json.dumps([[[t,str(value)] for t,value in row] for row in rows]))
