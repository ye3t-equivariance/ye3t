"""Private ordered JSON storage for shared compiler records.

Only JSON values and acyclic container references are supported. This stores
existing records; it does not define labels, couplings or execution semantics.
Each decode owns its containers, while repeated references within it remain
shared. Mutable inputs and decoded values never enter the serialized snapshot.
"""

import json
import math


def _iter_canonical_json_bytes(value):
    """Stream sorted compact JSON, reusing shared containers within this call.

    Input is already normalized to string-keyed dictionaries, lists and JSON
    scalars. Every occurrence contributes its complete bytes to the stream.
    The caller's hash and storage format remain identical to json.dumps; no
    memo survives the call. This avoids expanding a shared compiler graph into
    one large temporary string and then another equally large byte buffer.
    """
    counts, objects, order, active = {}, {}, [], set()

    def visit(item):
        if not isinstance(item, (dict, list)):
            return
        identity = id(item)
        if identity in active:
            raise ValueError("Circular reference detected")
        counts[identity] = counts.get(identity, 0) + 1
        if identity in objects:
            return
        objects[identity] = item
        active.add(identity)
        if isinstance(item, dict) and any(not isinstance(key, str) for key in item):
            raise TypeError("Canonical JSON mapping keys must be strings")
        for child in (item.values() if isinstance(item, dict) else item):
            visit(child)
        active.remove(identity)
        order.append(identity)

    affected, encoded = set(), {}

    def encode(item):
        identity = id(item)
        if identity in encoded:
            return encoded[identity]
        if identity not in affected:
            return json.dumps(item, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if isinstance(item, dict):
            result = b"{" + b",".join(json.dumps(key).encode("utf-8") + b":" + encode(child)
                                     for key, child in sorted(item.items())) + b"}"
        else:
            result = b"[" + b",".join(encode(child) for child in item) + b"]"
        if counts[identity] > 1:
            encoded[identity] = result
        return result

    def chunks(item):
        identity = id(item)
        if identity not in affected or counts[identity] > 1:
            yield encode(item)
        elif isinstance(item, dict):
            yield b"{"
            for index, (key, child) in enumerate(sorted(item.items())):
                if index:
                    yield b","
                yield json.dumps(key).encode("utf-8") + b":"
                yield from chunks(child)
            yield b"}"
        else:
            yield b"["
            for index, child in enumerate(item):
                if index:
                    yield b","
                yield from chunks(child)
            yield b"]"

    try:
        visit(value)
        for identity in order:
            item = objects[identity]
            if counts[identity] > 1 or any(id(child) in affected for child in
                    (item.values() if isinstance(item, dict) else item)):
                affected.add(identity)
        yield from chunks(value)
    finally:
        # Recursive closures otherwise retain this call's compiler graph until
        # cyclic garbage collection, including on a partially consumed stream.
        visit = encode = chunks = None


def _encode_json_graph(value):
    nodes, indices, active = [], {}, set()

    def encode(item):
        kind = type(item)
        if item is None or kind in (str, bool, int):
            return item
        if kind is float:
            if not math.isfinite(item):
                raise ValueError("JSON graph scalars must be finite")
            return item
        if kind not in (dict, list, tuple):
            raise TypeError("JSON graph requires plain JSON containers and scalars")
        identity = id(item)
        if identity in active:
            raise ValueError("JSON graph contains a container cycle")
        if identity not in indices:
            active.add(identity)
            if kind is dict:
                if any(type(key) is not str for key in item):
                    raise TypeError("JSON graph mapping keys must be strings")
                node = ["dict", [[key, encode(child)] for key, child in item.items()]]
            else:
                node = ["list", [encode(child) for child in item]]
            active.remove(identity)
            indices[identity] = len(nodes)
            nodes.append(node)
        return [indices[identity]]

    try:
        root = encode(value)
        return json.dumps({"schema": "ye3t_ordered_json_graph_v1", "nodes": nodes, "root": root},
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    finally:
        encode = None


def _decode_json_graph(encoded):
    record = json.loads(encoded)
    if (type(record) is not dict or set(record) != {"schema", "nodes", "root"}
            or record["schema"] != "ye3t_ordered_json_graph_v1" or type(record["nodes"]) is not list):
        raise ValueError("Invalid ordered JSON graph envelope")
    values, keys = [], {}

    def decode(item):
        kind = type(item)
        if kind is list:
            if (len(item) != 1 or type(item[0]) is not int or not 0 <= item[0] < len(values)):
                raise ValueError("JSON graph references must name an earlier node")
            return values[item[0]]
        if item is None or kind in (str, bool, int):
            return item
        if kind is float and math.isfinite(item):
            return item
        raise ValueError("Invalid JSON graph value")

    for index in range(len(record["nodes"])):
        node = record["nodes"][index]
        record["nodes"][index] = None
        if type(node) is not list or len(node) != 2 or type(node[1]) is not list:
            raise ValueError("Invalid JSON graph node")
        kind, children = node
        if kind == "dict":
            value = {}
            for pair in children:
                if (type(pair) is not list or len(pair) != 2 or type(pair[0]) is not str
                        or pair[0] in value):
                    raise ValueError("JSON graph needs unique string mapping keys")
                key = keys.setdefault(pair[0], pair[0])
                value[key] = decode(pair[1])
        elif kind == "list":
            value = [decode(child) for child in children]
        else:
            raise ValueError("Unknown JSON graph node kind")
        values.append(value)
    return decode(record["root"])


def _ordered_json_graph_equal(left, right):
    """Compare complete decoded values/order without expanding shared subtrees."""
    pending, compared = [(left, right)], set()
    while pending:
        left, right = pending.pop()
        if type(left) is not type(right):
            return False
        if type(left) in (dict, list):
            pair = (id(left), id(right))
            if pair in compared:
                continue
            compared.add(pair)
            if len(left) != len(right):
                return False
            if type(left) is dict:
                if list(left) != list(right):
                    return False
                pending.extend((left[key], right[key]) for key in left)
            else:
                pending.extend(zip(left, right))
        elif left != right or (type(left) is float and left == 0 and
                               math.copysign(1, left) != math.copysign(1, right)):
            return False
    return True
