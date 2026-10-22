"""Reuse independent compiler work without hiding changes in mutable data."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ye3t.cache.artifacts import YE3TArtifactStore, ArtifactCacheValidationError


def test_diagnostic_history_is_bounded(tmp_path):
    from ye3t.cache.artifacts import artifact_cache_events
    artifact_cache_events(clear=True)
    store = YE3TArtifactStore(tmp_path)
    identity = store.identity("event_fixture", "v1", {"rank": 2})
    for index in range(4100):
        store._event(identity, "hit", "user", None, "verified", index)
    assert len(store.events) == 1024
    events = artifact_cache_events(clear=True)
    assert len(events) == 4096 and events[-1]["bytes"] == 4099
    assert artifact_cache_events() == ()


def test_concurrent_catalogue_publication_has_private_temporary_files(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from ye3t.couplings.tagged_cauchy_carriers import _write_compiled_cache
    path = tmp_path / "catalogue.json"
    def write(index):
        _write_compiled_cache(path, {"values": [index] * 1000})
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write, range(12)))
    payload = json.loads(path.read_text())
    assert payload["values"] == [payload["values"][0]] * 1000
    assert list(tmp_path.iterdir()) == [path]


def test_unchanged_store_entry_is_read_once_and_returns_private_values(tmp_path, monkeypatch):
    calls = []
    store = YE3TArtifactStore(tmp_path)

    def resolve(verify="hash"):
        return YE3TArtifactStore(tmp_path, verify=verify).resolve(
            "reuse_fixture", "v1", {"rank": 4},
            builder=lambda: {"rows": [[1, 2]]},
            validator=lambda value: calls.append(value["rows"][0][0]),
            certificate={"passed": True, "checks": {"valid": True}},
            required_certificate_checks=("valid",))

    first = resolve()
    assert calls == [1]
    first["payload"]["rows"][0][0] = 999
    original = Path.read_bytes
    reads = []

    def read(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    assert resolve()["payload"] == {"rows": [[1, 2]]}
    assert reads == []
    assert calls == [1]
    resolve("full")
    assert calls == [1, 1]
    path = Path(first["path"])
    stat = path.stat()
    encoded = original(path)
    assert b'"rows":[[1,2]]' in encoded
    path.write_bytes(encoded.replace(b'"rows":[[1,2]]', b'"rows":[[9,2]]'))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ArtifactCacheValidationError, match="hash mismatch"):
        YE3TArtifactStore(tmp_path, mode="read_only").resolve(
            "reuse_fixture", "v1", {"rank": 4}, builder=None)


def test_tagged_validation_reused_in_fresh_process_and_mutation_rejected(tmp_path, monkeypatch):
    from ye3t import couplings
    from ye3t.couplings import tagged_cauchy_general as general
    from ye3t.couplings.tagged_cauchy_image import CompiledTaggedCauchyImage
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("YE3T_CACHE_MODE", "auto")
    monkeypatch.setenv("YE3T_CACHE_VERIFY", "hash")
    request = couplings.tagged_cauchy_image_request(species=["Ni"], catalogue={
        "nmax_per_rank": {2: 1}, "lmax_per_rank": {2: 1},
        "source_block_partitions_by_rank": {2: [[2]]},
        "tag_counts_by_rank": {2: [0, 1, 2]}})
    compiled = couplings.compile(couplings.plan(couplings.count(request)))
    general._VALIDATED_GENERAL_ARTIFACTS.clear()
    compiled.validate()
    path = tmp_path / "compiled.json"
    path.write_text(json.dumps(compiled.to_dict()))
    program = """
import json, sys
from ye3t.couplings import tagged_cauchy_general as general
from ye3t.couplings.tagged_cauchy_image import CompiledTaggedCauchyImage
def forbidden(value):
    raise AssertionError('unchanged compiler proof was reconstructed')
general._replay_general_artifact = forbidden
artifact = CompiledTaggedCauchyImage.from_dict(json.load(open(sys.argv[1])))
artifact.validate()
print(artifact.self_hash)
"""
    result = subprocess.run([sys.executable, "-c", program, str(path)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == compiled.self_hash
    from types import MappingProxyType
    mapping_payload = compiled.to_dict()
    mapping_payload["payload"] = MappingProxyType(mapping_payload["payload"])
    assert CompiledTaggedCauchyImage.from_dict(mapping_payload).self_hash == compiled.self_hash
    calls = []
    original = general._replay_general_artifact
    monkeypatch.setattr(general, "_replay_general_artifact", lambda value: (calls.append(1), original(value))[1])
    monkeypatch.setenv("YE3T_CACHE_VERIFY", "full")
    compiled.validate()
    assert calls == [1]
    # Disabling persistence keeps the existing process-local validation reuse.
    monkeypatch.setenv("YE3T_CACHE_VERIFY", "hash")
    monkeypatch.setenv("YE3T_CACHE_MODE", "off")
    general._VALIDATED_GENERAL_ARTIFACTS.clear()
    compiled.validate()
    compiled.validate()
    assert calls == [1, 1]
    changed = compiled.to_dict()
    changed["payload"]["tensor_order"] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        CompiledTaggedCauchyImage.from_dict(changed)


@pytest.mark.parametrize("family", ["tagged", "shell"])
def test_individual_source_maps_grow_shared_cache(family, tmp_path, monkeypatch):
    from ye3t.couplings import tagged_cauchy_carriers as tagged
    from ye3t.couplings import shell_cauchy as shell
    from ye3t.couplings.covariant_cauchy import covariant_cauchy_request
    from ye3t.couplings.chebyshev_c2 import CHEBYSHEV_C2_SOURCE_FAMILY
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("YE3T_CACHE_MODE", "auto")
    monkeypatch.setenv("YE3T_CACHE_VERIFY", "hash")
    channel = {"neighbor_species": "Ni", "radial_channel": 0, "l": 1,
               "source_family_id": CHEBYSHEV_C2_SOURCE_FAMILY}
    if family == "shell":
        request = {"coupling_request": covariant_cauchy_request([channel], [2],
            target_L=0, role_dimension=2), "radial_count": 1,
            "shell_role_action": "permutation"}
        compile_source = shell.compile_shell_cauchy
        owner, name = shell, "_compile_shell_cauchy"
    else:
        request = tagged.tagged_cauchy_carriers_request(
            [channel], [2], tag_count=2, target_Ls=(0,))
        compile_source = tagged.compile_tagged_cauchy_carriers
        owner, name = tagged, "_materialize_source_image"
    cold = compile_source(request)
    def forbidden(*args, **kwargs):
        raise AssertionError("warm source rebuilt coefficients")
    original = getattr(owner, name)
    monkeypatch.setattr(owner, name, forbidden)
    warm = compile_source(request)
    assert warm == cold
    warm["descriptors"][0]["label"]["target_L"] = 999
    assert compile_source(request) == cold
    monkeypatch.setattr(owner, name, original)
    monkeypatch.setenv("YE3T_CACHE_VERIFY", "full")
    assert compile_source(request) == cold


def test_tagged_catalogue_extension_reuses_coordinate_lowerings(tmp_path, monkeypatch):
    from ye3t import couplings
    from ye3t.couplings import tagged_cauchy_general as general
    from ye3t.cache.artifacts import artifact_cache_events
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("YE3T_CACHE_MODE", "auto")
    monkeypatch.setenv("YE3T_CACHE_VERIFY", "hash")
    catalogue = {"nmax_per_rank": {2: 1}, "lmax_per_rank": {2: 1},
                 "source_block_partitions_by_rank": {2: [[2]]},
                 "tag_counts_by_rank": {2: [0, 1, 2]}}
    def compile_catalogue(config):
        request = couplings.tagged_cauchy_image_request(species=["Ni"], catalogue=config)
        return couplings.compile(couplings.plan(couplings.count(request)))
    cold = compile_catalogue(catalogue)
    original = general._physical_image_rows
    def forbidden(*args):
        raise AssertionError("warm coordinate repeated physical lowering")
    monkeypatch.setattr(general, "_physical_image_rows", forbidden)
    assert compile_catalogue(catalogue).to_dict() == cold.to_dict()
    monkeypatch.setattr(general, "_physical_image_rows", original)
    artifact_cache_events(clear=True)
    extended = dict(catalogue, nmax_per_rank={2: 2})
    compile_catalogue(extended)
    events = [row for row in artifact_cache_events() if row["artifact_type"] == "tagged_image_coordinate"]
    assert any(row["status"] == "hit" for row in events)
    assert any(row["status"] == "miss" for row in events)
    monkeypatch.setenv("YE3T_CACHE_MODE", "off")
    assert compile_catalogue(catalogue).to_dict() == cold.to_dict()
