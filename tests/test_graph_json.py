"""Ordered graph snapshots preserve exact compiler records and ownership."""
import json
import math

import pytest

from ye3t.cache.graph_json import _encode_json_graph, _decode_json_graph, _ordered_json_graph_equal


def test_order_values_shared_records_and_independent_decodes():
    shared = {"z": [2**100, -0.0, 1e-100], "a": "\u03ba\n\u96ea"}
    original = {"right": shared, "left": shared, "tuple": (True, None, "x")}
    encoded = _encode_json_graph(original)
    result = _decode_json_graph(encoded)
    assert json.dumps(result) == json.dumps(original)
    assert result["right"] is result["left"]
    assert list(result["right"]) == ["z", "a"]
    result["right"]["z"][0] = 0
    assert shared["z"][0] == 2**100
    assert _decode_json_graph(encoded)["right"]["z"][0] == 2**100
    assert math.copysign(1, result["right"]["z"][1]) == -1


def test_shared_branch_growth_stays_linear_and_comparison_accepts_equal_copies():
    value = {"leaf": [1, 2, 3]}
    for _ in range(18):
        value = {"a": value, "b": value}
    encoded = _encode_json_graph(value)
    assert len(encoded) < 2000
    actual = _decode_json_graph(encoded)
    assert _ordered_json_graph_equal(actual, _decode_json_graph(encoded))
    copied = {"x": [1, 2], "y": [1, 2]}
    shared = {"x": copied["x"], "y": copied["x"]}
    assert _ordered_json_graph_equal(copied, shared)
    assert not _ordered_json_graph_equal(copied, {"y": [1, 2], "x": [1, 2]})
    assert not _ordered_json_graph_equal({"x": -0.0}, {"x": 0.0})
    assert not _ordered_json_graph_equal([True], [1])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object(), {1: "wrong key"}])
def test_encoder_rejects_non_json_values(value):
    with pytest.raises((TypeError, ValueError)):
        _encode_json_graph(value)


def test_encoder_rejects_cycles():
    value = []
    value.append(value)
    with pytest.raises(ValueError, match="cycle"):
        _encode_json_graph(value)


@pytest.mark.parametrize("nodes,root", [
    ([["list", [[0]]]], [0]),
    ([["list", [[1]]]], [0]),
    ([["list", [[-1]]]], [0]),
    ([["list", [[True]]]], [0]),
    ([["dict", [["a", 1], ["a", 2]]]], [0]),
    ([["dict", [[1, "bad"]]]], [0]),
    ([["object", []]], [0]),
    ([["list", [float("nan")]]], [0]),
    ([], {}),
    ([], [0]),
])
def test_decoder_rejects_invalid_records(nodes, root):
    encoded = json.dumps({"schema": "ye3t_ordered_json_graph_v1", "nodes": nodes, "root": root})
    with pytest.raises(ValueError):
        _decode_json_graph(encoded)
