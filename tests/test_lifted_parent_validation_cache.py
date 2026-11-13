"""Shared parent validation retains exact checks for changed compiler data."""
import copy
import json
import os
import subprocess
import sys

import pytest

from ye3t import couplings
from ye3t.couplings import lifted_cauchy_scalar as scalar


@pytest.fixture
def parent(tmp_path, monkeypatch):
    scalar._VALIDATED_LOADED_SCALARS.clear()
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("YE3T_CACHE_MODE", "off")
    monkeypatch.setenv("YE3T_CACHE_VERIFY", "hash")
    request = couplings.first_lifted_cauchy_scalar_request(
        1, family_ids=("NT_NU4_K22_L0",))
    payload = couplings.compile(request).to_dict()
    monkeypatch.setenv("YE3T_CACHE_MODE", "auto")
    return payload


def observe(monkeypatch):
    calls = []
    original = scalar._validate_compiled_payload
    def checked(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(scalar, "_validate_compiled_payload", checked)
    return calls


def test_parent_validation_reused_without_changing_artifact(parent, monkeypatch, tmp_path):
    calls = observe(monkeypatch)
    cold = scalar.CompiledLiftedCauchyScalar.from_dict(parent)
    warm = scalar.CompiledLiftedCauchyScalar.from_dict(parent)
    assert cold.to_dict() == warm.to_dict() == parent
    assert calls == [1]
    monkeypatch.setenv("YE3T_CACHE_MODE", "read_only")
    assert scalar.CompiledLiftedCauchyScalar.from_dict(parent).to_dict() == parent
    assert calls == [1]
    monkeypatch.setattr(scalar, "_LOADED_PAYLOAD_VALIDATOR_REVISION", 2)
    assert scalar.CompiledLiftedCauchyScalar.from_dict(parent).to_dict() == parent
    assert calls == [1, 1]
    assert not list((tmp_path / "cache").rglob("*.json"))


@pytest.mark.parametrize("mode,verify", [
    ("off", "hash"), ("rebuild", "hash"), ("auto", "full")])
def test_explicit_replay_modes(parent, monkeypatch, mode, verify):
    scalar.CompiledLiftedCauchyScalar.from_dict(parent)
    calls = observe(monkeypatch)
    monkeypatch.setenv("YE3T_CACHE_MODE", mode)
    monkeypatch.setenv("YE3T_CACHE_VERIFY", verify)
    for _ in range(2):
        assert scalar.CompiledLiftedCauchyScalar.from_dict(parent).to_dict() == parent
    assert calls == [1, 1]


def test_readonly_miss_validates_without_writing(parent, tmp_path, monkeypatch):
    calls = observe(monkeypatch)
    monkeypatch.setenv("YE3T_CACHE_MODE", "read_only")
    scalar.CompiledLiftedCauchyScalar.from_dict(parent)
    assert calls == [1]
    assert not list((tmp_path / "cache").rglob("*.json"))


def test_mutated_and_rehashed_parent_cannot_reuse_check(parent, monkeypatch):
    scalar.CompiledLiftedCauchyScalar.from_dict(parent)
    calls = observe(monkeypatch)
    changed = copy.deepcopy(parent)
    changed["payload"]["unexpected"] = True
    with pytest.raises(ValueError, match="hash mismatch"):
        scalar.CompiledLiftedCauchyScalar.from_dict(changed)
    assert calls == []
    changed["self_hash"] = scalar._stable_hash(
        {key: value for key, value in changed.items() if key != "self_hash"})
    with pytest.raises(ValueError):
        scalar.CompiledLiftedCauchyScalar.from_dict(changed)
    assert calls == [1]


def test_fresh_process_validates_once_then_reuses(parent, tmp_path):
    scalar.CompiledLiftedCauchyScalar.from_dict(parent)
    artifact = tmp_path / "parent.json"
    artifact.write_text(json.dumps(parent))
    probe = tmp_path / "load_parent.py"
    probe.write_text(
        "import json, sys\n"
        "from ye3t.couplings import lifted_cauchy_scalar as scalar\n"
        "calls = []\n"
        "original = scalar._validate_compiled_payload\n"
        "def checked(*args):\n"
        "    calls.append(1)\n"
        "    return original(*args)\n"
        "scalar._validate_compiled_payload = checked\n"
        "data = json.load(open(sys.argv[1]))\n"
        "result = scalar.CompiledLiftedCauchyScalar.from_dict(data)\n"
        "assert json.loads(json.dumps(result.to_dict())) == data\n"
        "again = scalar.CompiledLiftedCauchyScalar.from_dict(data)\n"
        "assert again.to_dict() == result.to_dict()\n"
        "assert calls == [1]\n")
    result = subprocess.run([sys.executable, str(probe), str(artifact)],
        env=dict(os.environ), capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_parent_memo_is_bounded_and_revalidates_evicted_entries(parent, monkeypatch):
    monkeypatch.setattr(scalar, "_LOADED_VALIDATION_LIMIT", 2)
    calls = observe(monkeypatch)
    for revision in (1, 2, 1, 3, 2):
        monkeypatch.setattr(scalar, "_LOADED_PAYLOAD_VALIDATOR_REVISION", revision)
        scalar.CompiledLiftedCauchyScalar.from_dict(parent)
        assert len(scalar._VALIDATED_LOADED_SCALARS) <= 2
    assert calls == [1, 1, 1, 1]


def test_failed_parent_check_is_never_memoized(parent, monkeypatch):
    changed = copy.deepcopy(parent)
    changed["payload"]["unexpected"] = True
    changed["self_hash"] = scalar._stable_hash(
        {key: value for key, value in changed.items() if key != "self_hash"})
    calls = observe(monkeypatch)
    for _ in range(2):
        with pytest.raises(ValueError):
            scalar.CompiledLiftedCauchyScalar.from_dict(changed)
    assert calls == [1, 1]
    assert not scalar._VALIDATED_LOADED_SCALARS
