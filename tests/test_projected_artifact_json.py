import json
from pathlib import Path

import pytest

from ye3t.cache.artifacts import YE3TArtifactStore, ArtifactCacheValidationError, artifact_hash
from ye3t.cache.projected_json import read_projected_payload


def test_projected_cache_checks_entire_artifact_without_decoding_proof(tmp_path, monkeypatch):
    store = YE3TArtifactStore(directory=tmp_path, mode="auto")
    payload = {"rows": [[1, 2], [3, 4]], "proof": [{"escaped": '"}\\[é', "values": list(range(50))}]*200,
               "validation_report": {"passed": True}}
    payload["template_hash"] = artifact_hash(payload)
    result = store.resolve("projection_test", "v1", {"n": 4}, lambda: payload,
        certificate={"passed": True, "checks": {"valid": True}})
    path = Path(result["path"])
    identity = store.identity("projection_test", "v1", {"n": 4})
    original = json.loads
    def bounded(value, *args, **kwargs):
        assert len(value) < 1000, "decoded the complete proof"
        return original(value, *args, **kwargs)
    monkeypatch.setattr(json, "loads", bounded)
    selected = read_projected_payload(path, identity, ("rows", "template_hash"), ("valid",))
    assert selected == {key: payload[key] for key in selected}
    encoded = path.read_bytes()
    path.write_bytes(encoded.replace(b'"values":[0,1,2', b'"values":[9,1,2', 1))
    with pytest.raises(ArtifactCacheValidationError, match="hash mismatch"):
        read_projected_payload(path, identity, ("rows",))


def test_exact_record_sharing_preserves_exact_collisions_and_complete_payload(tmp_path):
    from ye3t.couplings import lifted_cauchy_scalar as scalar
    from copy import deepcopy
    value = scalar._exact_scalar_payload(scalar._sympy().sqrt(2))
    other = deepcopy(value)
    other["real"][0]["coefficient_numerator"] += 1
    payload = {"coefficients": [value]*100+[other]}
    payload["carrier_hash"] = artifact_hash(payload)
    store = YE3TArtifactStore(directory=tmp_path, mode="auto")
    result = store.resolve("angular_read_test", 1, {}, lambda: payload,
        certificate={"passed": True, "checks": {"valid": True}})
    actual = read_projected_payload(Path(result["path"]), store.identity("angular_read_test",1,{}),
        None, ("valid",), share_exact_scalars=True)
    assert actual == json.loads(json.dumps(payload))
    assert len({id(record) for record in actual["coefficients"][:-1]}) == 1
    assert actual["coefficients"][-1] is not actual["coefficients"][0]


@pytest.mark.parametrize("length", [0, 1, 512, 513, 1200])
def test_indexed_sparse_read_is_bounded_exact_and_hash_compatible(tmp_path, monkeypatch, length):
    from ye3t.couplings import lifted_cauchy_scalar as scalar
    from ye3t.cache.projected_json import _IndexedScalarEntries
    coefficients = [scalar._exact_scalar_payload(scalar._sympy().sqrt(2)),
                    scalar._exact_scalar_payload(scalar._sympy().I/3)]
    entries = [[i, i % 3, coefficients[i % 2]] for i in range(length)]
    payload = {"basis_columns_sparse": entries, "other": "preserved"}
    payload["carrier_hash"] = artifact_hash(payload)
    store = YE3TArtifactStore(directory=tmp_path, mode="auto")
    result = store.resolve("indexed_read", 1, {}, lambda: payload,
        certificate={"passed": True, "checks": {"valid": True}})
    original = json.loads
    sizes = []

    def bounded(value, *args, **kwargs):
        decoded = original(value, *args, **kwargs)
        if isinstance(decoded, list) and decoded and isinstance(decoded[0], list):
            sizes.append(len(decoded))
            assert len(decoded) <= 512
        return decoded

    monkeypatch.setattr(json, "loads", bounded)
    actual = read_projected_payload(Path(result["path"]), store.identity("indexed_read", 1, {}),
        None, indexed_sparse=True)
    table = actual["basis_columns_sparse"]
    assert isinstance(table, _IndexedScalarEntries)
    assert scalar._freeze_json(list(table)) == scalar._freeze_json(entries)
    assert len(table.scalars) == min(length, 2)
    assert len(table.rows)*table.rows.itemsize == 8*length
    assert scalar._stable_record_hash(actual) == scalar._stable_record_hash(payload)
    scalar._validate_hash_record(actual, "carrier_hash", "indexed-carrier")
    assert bool(sizes) == bool(length)


def test_execution_migration_drops_unused_monomial_metadata_after_verification(tmp_path):
    from ye3t.couplings import lifted_cauchy_scalar as scalar
    terms = {((0, 0),)*6+((1, 2),)*2: scalar._sympy().sqrt(3)}
    payload = {"analysis_rows": [{"symmetric_power_terms": scalar._term_records(terms)}]}
    payload["template_hash"] = artifact_hash(payload)
    store = YE3TArtifactStore(directory=tmp_path, mode="auto")
    result = store.resolve("minimal_terms", 1, {}, lambda: payload,
        certificate={"passed": True, "checks": {"valid": True}})
    actual = read_projected_payload(Path(result["path"]), store.identity("minimal_terms", 1, {}),
        ("analysis_rows",), ("valid",), minimal_terms=True)
    records = actual["analysis_rows"][0]["symmetric_power_terms"]
    assert scalar._terms_from_records(records) == terms
    assert all(set(record) == {"coordinates", "coefficient"} for record in records)


def test_execution_read_shares_monomials_and_exact_coefficients(tmp_path):
    from ye3t.couplings import lifted_cauchy_scalar as scalar
    term = {"coordinates": [[0, 0]]*4+[[1, 2]]*4,
            "coefficient": scalar._exact_scalar_payload(scalar._sympy().sqrt(3))}
    payload = {"analysis_rows": [{"symmetric_power_terms": [term]} for _ in range(20)]}
    payload["template_hash"] = artifact_hash(payload)
    store = YE3TArtifactStore(directory=tmp_path, mode="auto")
    saved = store.resolve("shared_execution", 1, {}, lambda: payload,
        certificate={"passed": True, "checks": {"valid": True}})
    actual = read_projected_payload(Path(saved["path"]), store.identity("shared_execution", 1, {}),
        None, ("valid",), minimal_terms=True)
    rows = [row["symmetric_power_terms"][0] for row in actual["analysis_rows"]]
    assert len({id(row["coordinates"]) for row in rows}) == 1
    assert len({id(row["coefficient"]) for row in rows}) == 1
    assert scalar._freeze_json(actual) == scalar._freeze_json(payload)
    scalar._validate_hash_record(actual, "template_hash", "shared-execution")
    from ye3t.cache.projected_json import _share_execution_rows
    owned = json.loads(json.dumps(payload))
    _share_execution_rows(owned["analysis_rows"])
    rows = [row["symmetric_power_terms"][0] for row in owned["analysis_rows"]]
    assert len({id(row["coordinates"]) for row in rows}) == 1
    assert len({id(row["coefficient"]) for row in rows}) == 1
    assert scalar._freeze_json(owned) == scalar._freeze_json(payload)
    scalar._validate_hash_record(owned, "template_hash", "shared-owned-execution")
