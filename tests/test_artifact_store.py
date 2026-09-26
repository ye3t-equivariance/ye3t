import json
import multiprocessing
from pathlib import Path

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
    assert not tuple((tmp_path / "cache").rglob("*.lock"))
