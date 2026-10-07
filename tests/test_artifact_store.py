import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


def _certificate(payload):
    return {
        "passed": True,
        "checks": {
            "payload_valid": payload["value"] >= 0,
            "transpose_valid": True,
        },
    }


def _race_resolve(arguments):
    cache_directory, result_path = arguments
    from ye3t.cache.artifacts import YE3TArtifactStore

    store = YE3TArtifactStore(directory=cache_directory)
    result = store.resolve(
        "race_fixture",
        "fixture_v1",
        {"tensor_order": 2, "partition": [1, 1]},
        lambda: {"value": 7, "rows": [[1, 0], [0, 1]]},
        validator=lambda payload: None,
        certificate=_certificate,
        required_certificate_checks=("payload_valid", "transpose_valid"),
        producer={"compiler_convention": "fixture_v1"},
    )
    Path(result_path).write_text(
        json.dumps(
            {
                "artifact_hash": result["artifact_hash"],
                "payload_hash": result["payload_hash"],
                "status": result["status"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _interrupted_resolve(cache_directory, ready):
    from ye3t.cache.artifacts import YE3TArtifactStore

    def build():
        ready.set()
        time.sleep(60)
        return {"value": 7, "rows": [[1, 0], [0, 1]]}

    YE3TArtifactStore(directory=cache_directory).resolve(
        "exact_fixture",
        "fixture_v1",
        {"partition": [2]},
        build,
        certificate=_certificate,
        required_certificate_checks=("payload_valid", "transpose_valid"),
        dependency_hashes={"carrier": "1" * 64},
        producer={"compiler_convention": "fixture_v1"},
    )


def _interrupted_before_replace(cache_directory, ready):
    from ye3t.cache.artifacts import YE3TArtifactStore

    def pause_replace(source, destination):
        ready.set()
        time.sleep(60)

    os.replace = pause_replace
    _resolve(YE3TArtifactStore(directory=cache_directory), {"partition": [2]}, 7)


def _resolve(store, request, value, counter=None, producer=None):
    def build():
        if counter is not None:
            counter.append(value)
        return {"value": value, "rows": [[1, 0], [0, 1]]}

    return store.resolve(
        "exact_fixture",
        "fixture_v1",
        request,
        build,
        validator=lambda payload: None,
        certificate=_certificate,
        required_certificate_checks=("payload_valid", "transpose_valid"),
        dependency_hashes={"carrier": "1" * 64},
        producer={"compiler_convention": "fixture_v1"}
        if producer is None
        else producer,
    )


def test_canonical_json_is_order_independent_and_strict():
    from ye3t.cache.artifacts import artifact_hash, canonical_json_bytes

    left = {"b": (2, 3), "a": {"x": 1.0}}
    right = {"a": {"x": 1.0}, "b": [2, 3]}
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert artifact_hash(left) == artifact_hash(right)
    with pytest.raises(ValueError, match="Non-finite"):
        canonical_json_bytes({"value": float("nan")})
    with pytest.raises(TypeError, match="Non-string"):
        canonical_json_bytes({1: "ambiguous"})
    with pytest.raises(TypeError, match="Unsupported"):
        canonical_json_bytes({"path": Path("cache")})


def test_inspect_and_explicit_prune_keep_valid_entries(tmp_path, monkeypatch):
    from ye3t.cache.artifacts import YE3TArtifactStore

    store = YE3TArtifactStore(directory=tmp_path)
    built = _resolve(store, {"partition": [2]}, 7)
    valid = Path(built["path"])
    damaged = valid.with_name("f" * 64 + ".json")
    damaged.write_bytes(b'{"partial":')
    quarantine = valid.with_name(valid.name + ".invalid.fixture")
    quarantine.write_bytes(b"old invalid entry")
    records = {item["path"]: item for item in store.inspect()}
    assert records[str(valid.relative_to(tmp_path))]["status"] == "integrity_valid"
    assert records[str(damaged.relative_to(tmp_path))]["status"] == "invalid"
    assert records[str(quarantine.relative_to(tmp_path))]["status"] == "quarantined"
    with pytest.raises(ValueError, match="integrity-valid"):
        store.prune(valid.relative_to(tmp_path), dry_run=False)
    outside = tmp_path.parent / "outside.json"
    with pytest.raises(ValueError, match="outside"):
        store.prune(outside, dry_run=False)
    in_root_link = valid.with_name("e" * 64 + ".json")
    in_root_link.symlink_to(damaged)
    outside_link = valid.with_name("d" * 64 + ".json")
    outside_link.symlink_to(outside)
    for link in (in_root_link, outside_link):
        with pytest.raises(ValueError, match="outside"):
            store.prune(link.relative_to(tmp_path), dry_run=False)
    symlinks = {item["path"]: item for item in store.inspect()}
    assert symlinks[str(in_root_link.relative_to(tmp_path))]["status"] == "invalid"
    assert symlinks[str(outside_link.relative_to(tmp_path))]["status"] == "invalid"
    with pytest.raises(ValueError, match="artifact entry"):
        store.prune(valid.with_suffix(".json.lock"), dry_run=False)
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")
              if path.is_file() or path.is_symlink()}
    preview = store.prune([damaged.relative_to(tmp_path),
                           quarantine.relative_to(tmp_path)])
    assert all(item["removed"] is False for item in preview)
    after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")
             if path.is_file() or path.is_symlink()}
    assert after == before
    assert damaged.exists() and quarantine.exists()
    removed = store.prune([damaged.relative_to(tmp_path),
                           quarantine.relative_to(tmp_path)], dry_run=False)
    assert all(item["removed"] is True for item in removed)
    assert valid.is_file()
    assert not damaged.exists() and not quarantine.exists()
    assert store.prune(damaged.relative_to(tmp_path), dry_run=False)[0]["removed"] is False

    intermittent = valid.with_name("c" * 64 + ".json")
    intermittent.write_bytes(b'{"partial":')
    original_stat = Path.stat
    calls = []

    def disappearing_stat(path, *args, **kwargs):
        if path == intermittent:
            calls.append(1)
            if len(calls) == 2:
                raise FileNotFoundError(intermittent)
        return original_stat(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "stat", disappearing_stat)
        assert store._inspect_path(intermittent)["status"] == "missing"


def test_cache_cli_inspects_and_requires_apply_for_prune(tmp_path):
    from ye3t.cache.artifacts import YE3TArtifactStore

    store = YE3TArtifactStore(directory=tmp_path)
    built = _resolve(store, {"partition": [2]}, 7)
    damaged = Path(built["path"]).with_name("f" * 64 + ".json")
    damaged.write_bytes(b'{"partial":')
    prefix = [sys.executable, "-m", "ye3t.cache", "--cache-dir", str(tmp_path)]
    inspected = subprocess.run(prefix + ["inspect"], check=True,
                               capture_output=True, text=True)
    assert {item["status"] for item in json.loads(inspected.stdout)} == {
        "integrity_valid", "invalid"}
    entry = str(damaged.relative_to(tmp_path))
    preview = subprocess.run(prefix + ["prune", "--entry", entry], check=True,
                             capture_output=True, text=True)
    assert json.loads(preview.stdout)[0]["removed"] is False
    assert damaged.exists()
    applied = subprocess.run(prefix + ["prune", "--entry", entry, "--apply"],
                             check=True, capture_output=True, text=True)
    assert json.loads(applied.stdout)[0]["removed"] is True
    assert not damaged.exists()


def test_cache_rejects_symlinked_artifact_root_and_lock(tmp_path):
    from ye3t.cache.artifacts import ArtifactCacheError, YE3TArtifactStore

    external = tmp_path / "external_artifacts"
    external.mkdir()
    linked_cache = tmp_path / "linked_cache"
    linked_cache.mkdir()
    (linked_cache / "artifacts").symlink_to(external, target_is_directory=True)
    linked_store = YE3TArtifactStore(directory=linked_cache)
    with pytest.raises(ValueError, match="symlink"):
        linked_store.inspect()
    with pytest.raises(ValueError, match="symlink"):
        linked_store.prune("artifacts/exact_fixture/" + "a" * 64 + ".json")
    with pytest.raises(ArtifactCacheError, match="symlink"):
        _resolve(linked_store, {"partition": [2]}, 7)
    assert not list(external.rglob("*"))

    source_store = YE3TArtifactStore(directory=tmp_path / "source_cache")
    _resolve(source_store, {"partition": [3]}, 9)
    type_link_cache = tmp_path / "type_link_cache"
    (type_link_cache / "artifacts").mkdir(parents=True)
    (type_link_cache / "artifacts" / "exact_fixture").symlink_to(
        source_store.directory / "artifacts" / "exact_fixture",
        target_is_directory=True)
    with pytest.raises(ArtifactCacheError, match="symlink"):
        _resolve(YE3TArtifactStore(directory=type_link_cache, mode="read_only",
                                   overlays=()), {"partition": [3]}, 99)

    store = YE3TArtifactStore(directory=tmp_path / "regular")
    built = _resolve(store, {"partition": [2]}, 7)
    artifact = Path(built["path"])
    lock = artifact.with_suffix(".json.lock")
    lock.unlink()
    external_lock = tmp_path / "external_lock"
    external_lock.write_bytes(b"")
    lock.symlink_to(external_lock)
    with pytest.raises(ArtifactCacheError, match="symlink"):
        _resolve(YE3TArtifactStore(directory=store.directory, mode="refresh"),
                 {"partition": [2]}, 7)
    assert external_lock.read_bytes() == b""


def test_store_cold_warm_dependency_and_producer_invalidation(tmp_path):
    from ye3t.cache.artifacts import YE3TArtifactStore

    counter = []
    request = {"tensor_order": 2, "partition": [1, 1]}
    cold = _resolve(YE3TArtifactStore(tmp_path), request, 7, counter)
    warm = _resolve(YE3TArtifactStore(tmp_path), request, 99, counter)
    assert cold["status"] == "miss"
    assert warm["status"] == "hit"
    assert warm["payload"] == cold["payload"]
    assert warm["artifact_hash"] == cold["artifact_hash"]
    assert counter == [7]

    changed = _resolve(
        YE3TArtifactStore(tmp_path),
        request,
        9,
        counter,
        producer={"compiler_convention": "fixture_v2"},
    )
    assert changed["status"] == "miss"
    assert changed["artifact_hash"] != cold["artifact_hash"]
    assert counter == [7, 9]

    refreshed = _resolve(
        YE3TArtifactStore(tmp_path, mode="refresh"), request, 11, counter,
    )
    assert refreshed["status"] == "miss"
    assert refreshed["artifact_hash"] == cold["artifact_hash"]
    assert refreshed["payload"]["value"] == 11
    assert counter == [7, 9, 11]


def test_store_rejects_same_shaped_swap_and_recovers_corruption(tmp_path):
    from ye3t.cache.artifacts import (
        ArtifactCacheValidationError,
        YE3TArtifactStore,
    )

    first_store = YE3TArtifactStore(tmp_path)
    first = _resolve(first_store, {"partition": [2]}, 1)
    second = _resolve(first_store, {"partition": [1, 1]}, 2)
    first_path = Path(first_store.path_for_identity(first_store.identity(
        "exact_fixture",
        "fixture_v1",
        {"partition": [2]},
        {"carrier": "1" * 64},
        {"compiler_convention": "fixture_v1"},
    )))
    second_path = Path(first_store.path_for_identity(first_store.identity(
        "exact_fixture",
        "fixture_v1",
        {"partition": [1, 1]},
        {"carrier": "1" * 64},
        {"compiler_convention": "fixture_v1"},
    )))
    first_path.write_bytes(second_path.read_bytes())
    with pytest.raises(ArtifactCacheValidationError, match="different"):
        _resolve(
            YE3TArtifactStore(tmp_path, mode="read_only"),
            {"partition": [2]},
            3,
        )

    first_path.write_bytes(b"not-json")
    rebuilt = []
    recovered = _resolve(
        YE3TArtifactStore(tmp_path, mode="auto"),
        {"partition": [2]},
        4,
        rebuilt,
    )
    assert recovered["status"] == "miss"
    assert recovered["payload"]["value"] == 4
    assert rebuilt == [4]
    assert tuple(tmp_path.rglob("*.invalid.*"))
    assert first["artifact_hash"] != second["artifact_hash"]


def test_store_normalizes_malformed_and_full_validator_failures(tmp_path):
    from ye3t.cache.artifacts import (
        ArtifactCacheValidationError,
        YE3TArtifactStore,
    )

    request = {"partition": [2, 1]}
    store = YE3TArtifactStore(tmp_path)
    built = _resolve(store, request, 5)
    Path(built["path"]).write_bytes(b'{"payload": NaN}')
    recovered = _resolve(YE3TArtifactStore(tmp_path), request, 6)
    assert recovered["status"] == "miss"
    assert recovered["payload"]["value"] == 6

    def reject_payload(payload):
        raise ValueError("semantic fixture failure")

    with pytest.raises(
        ArtifactCacheValidationError, match="semantic fixture failure"
    ):
        YE3TArtifactStore(tmp_path, mode="read_only", verify="full").resolve(
            "exact_fixture",
            "fixture_v1",
            request,
            lambda: {"value": 7, "rows": [[1, 0], [0, 1]]},
            validator=reject_payload,
            certificate=_certificate,
            required_certificate_checks=("payload_valid", "transpose_valid"),
            dependency_hashes={"carrier": "1" * 64},
            producer={"compiler_convention": "fixture_v1"},
        )


def test_read_only_overlay_is_preferred_without_writing_user_store(tmp_path):
    from ye3t.cache.artifacts import YE3TArtifactStore

    overlay = tmp_path / "overlay"
    user = tmp_path / "user"
    built = _resolve(YE3TArtifactStore(overlay), {"rank": 4}, 12)
    resolved = _resolve(
        YE3TArtifactStore(user, mode="read_only", overlays=(overlay,)),
        {"rank": 4},
        99,
    )
    assert resolved["source"] == "overlay"
    assert resolved["payload"] == built["payload"]
    assert not user.exists()


def test_two_processes_resolve_one_atomic_artifact(tmp_path):
    context = multiprocessing.get_context("spawn")
    outputs = (tmp_path / "one.json", tmp_path / "two.json")
    processes = tuple(
        context.Process(
            target=_race_resolve,
            args=((str(tmp_path / "cache"), str(output)),),
        )
        for output in outputs
    )
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    records = tuple(
        json.loads(output.read_text(encoding="utf-8")) for output in outputs
    )
    assert records[0]["artifact_hash"] == records[1]["artifact_hash"]
    assert records[0]["payload_hash"] == records[1]["payload_hash"]
    assert sorted(record["status"] for record in records) == ["hit", "miss"]
    assert len(tuple((tmp_path / "cache").rglob("*.lock"))) == 1
    assert len(tuple((tmp_path / "cache").rglob("*.json"))) == 1


def test_interrupted_writer_and_partial_sibling_do_not_block_rebuild(tmp_path):
    from ye3t.cache.artifacts import YE3TArtifactStore

    directory = tmp_path / "cache"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    process = context.Process(
        target=_interrupted_resolve,
        args=(str(directory), ready),
    )
    process.start()
    try:
        assert ready.wait(timeout=10), "Writer did not enter the build step."
    finally:
        process.terminate()
        process.join(timeout=10)
    assert process.exitcode is not None

    store = YE3TArtifactStore(directory=directory)
    identity = store.identity(
        "exact_fixture",
        "fixture_v1",
        {"partition": [2]},
        {"carrier": "1" * 64},
        {"compiler_convention": "fixture_v1"},
    )
    path = store.path_for_identity(identity)
    partial = path.with_name("." + path.name + ".interrupted")
    partial.write_bytes(b'{"incomplete":')
    started = time.monotonic()
    built = _resolve(store, {"partition": [2]}, 7)
    assert time.monotonic() - started < 5
    assert built["status"] == "miss"
    assert built["artifact_hash"] == identity["semantic_hash"]
    assert partial.read_bytes() == b'{"incomplete":'
    warm = _resolve(YE3TArtifactStore(directory=directory), {"partition": [2]}, 99)
    assert warm["status"] == "hit"
    assert warm["payload"] == built["payload"]


def test_old_mtime_never_steals_a_live_lock(tmp_path):
    from ye3t.cache.artifacts import _ArtifactLock

    path = tmp_path / "live.lock"
    with _ArtifactLock(path, timeout=1):
        os.utime(path, (1, 1))
        with pytest.raises(TimeoutError, match="Timed out"):
            with _ArtifactLock(path, timeout=0.1):
                pytest.fail("A second writer acquired an active lock.")
    with _ArtifactLock(path, timeout=1):
        pass


def test_interruption_after_temp_write_ignores_unpublished_artifact(tmp_path):
    from ye3t.cache.artifacts import YE3TArtifactStore

    directory = tmp_path / "cache"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    process = context.Process(
        target=_interrupted_before_replace,
        args=(str(directory), ready),
    )
    process.start()
    try:
        assert ready.wait(timeout=10), "Writer did not reach atomic replace."
    finally:
        process.terminate()
        process.join(timeout=10)
    assert process.exitcode is not None

    store = YE3TArtifactStore(directory=directory)
    identity = store.identity(
        "exact_fixture",
        "fixture_v1",
        {"partition": [2]},
        {"carrier": "1" * 64},
        {"compiler_convention": "fixture_v1"},
    )
    path = store.path_for_identity(identity)
    assert not path.exists()
    assert len(tuple(path.parent.glob("." + path.name + ".*"))) == 1
    rebuilt = _resolve(store, {"partition": [2]}, 9)
    assert rebuilt["status"] == "miss"
    assert rebuilt["payload"]["value"] == 9
    assert _resolve(YE3TArtifactStore(directory), {"partition": [2]}, 99)["status"] == "hit"


def test_two_processes_recover_one_corrupt_entry(tmp_path):
    from ye3t.cache.artifacts import YE3TArtifactStore

    directory = tmp_path / "cache"
    store = YE3TArtifactStore(directory=directory)
    identity = store.identity(
        "race_fixture",
        "fixture_v1",
        {"tensor_order": 2, "partition": [1, 1]},
        producer={"compiler_convention": "fixture_v1"},
    )
    path = store.path_for_identity(identity)
    path.parent.mkdir(parents=True)
    path.write_bytes(b'{"incomplete":')
    context = multiprocessing.get_context("spawn")
    outputs = (tmp_path / "one.json", tmp_path / "two.json")
    processes = tuple(
        context.Process(target=_race_resolve, args=((str(directory), str(output)),))
        for output in outputs
    )
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    records = tuple(json.loads(output.read_text(encoding="utf-8")) for output in outputs)
    assert sorted(record["status"] for record in records) == ["hit", "miss"]
    assert records[0]["payload_hash"] == records[1]["payload_hash"]
    assert len(tuple(path.parent.glob(path.name + ".invalid.*"))) == 1
    assert json.loads(path.read_text(encoding="utf-8"))["payload_hash"] == records[0]["payload_hash"]
