"""Exact canonical bytes and bounded repeated-record encoding."""
import hashlib
import json
import random

import pytest

from ye3t.cache import graph_json
from ye3t.execution_plan import _canonical_payload, _stable_hash


def reference(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


@pytest.mark.parametrize("value", [None, True, False, 17, -0.0, 1e-200, 1e200,
    float("inf"), float("-inf"), float("nan"), "\u03ba\n\"\\\ud800", {}, [],
    {"b": [1, False], "a": {"z": 4}}, [[], {}, ["a", "b"]]])
def test_scalar_and_unique_container_bytes(value):
    assert b"".join(graph_json._iter_canonical_json_bytes(value)) == reference(value)


@pytest.mark.parametrize("seed", range(20))
def test_shared_graph_bytes_and_mutation(seed):
    rng = random.Random(seed)
    pool = [None, -0.0, True, "\u03ba", rng.random(), float("nan")]
    for index in range(30):
        children = [rng.choice(pool) for _ in range(rng.randrange(1, 6))]
        pool.append(children if index % 2 else {str(i): child for i, child in enumerate(children)})
    shared = {"values": [rng.random(), "before"]}
    payload = {"first": pool[-1], "second": pool[-2], "shared": [shared]*15}
    assert b"".join(graph_json._iter_canonical_json_bytes(payload)) == reference(payload)
    before = _stable_hash(payload)
    shared["values"][1] = "after"
    assert _stable_hash(payload) == hashlib.sha256(reference(payload)).hexdigest()
    assert _stable_hash(payload) != before


def test_repeated_records_are_encoded_once_without_expanded_root(monkeypatch):
    leaf = {"values": list(range(200))}
    parent = {"left": leaf, "right": leaf}
    payload = [parent]*100
    expected = reference(payload)
    original = json.dumps
    calls = []

    def checked(value, *args, **kwargs):
        assert value is not payload
        calls.append(id(value))
        return original(value, *args, **kwargs)

    monkeypatch.setattr(graph_json.json, "dumps", checked)
    chunks = list(graph_json._iter_canonical_json_bytes(payload))
    assert b"".join(chunks) == expected
    assert calls.count(id(leaf["values"])) == 1
    assert max(map(len, chunks)) < len(expected)//10


def test_canonical_conversions_and_distinct_physical_bindings():
    source = {"role": "tag", "coefficient": 2-3j, "scale": [0.1, 0.2]}
    payload = {"first": source, "again": [source, source], "keys": {1: "a", "1": "b"}}
    normalized = _canonical_payload(payload)
    assert _stable_hash(payload) == hashlib.sha256(reference(normalized)).hexdigest()
    payload["again"][1] = {**source, "scale": [0.1, 0.3]}
    assert _stable_hash(payload) != hashlib.sha256(reference(normalized)).hexdigest()


@pytest.mark.parametrize("kind", ["list", "dict"])
def test_cycles_fail(kind):
    value = [] if kind == "list" else {}
    if kind == "list":
        value.append(value)
    else:
        value["self"] = value
    with pytest.raises(ValueError, match="Circular reference"):
        list(graph_json._iter_canonical_json_bytes(value))


def test_noncanonical_keys_fail():
    with pytest.raises(TypeError, match="keys must be strings"):
        list(graph_json._iter_canonical_json_bytes({1: []}))
