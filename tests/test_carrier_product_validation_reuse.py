"""Reuse only exact compiler-owned maps, retaining canonical validation gates."""
import copy
import json

import pytest
import torch

from ye3t import couplings
from ye3t.couplings import carrier_products as core
from test_carrier_products import layout


def compiled_product(L=0):
    return couplings.compile(couplings.carrier_product_request(
        layout((2,)), layout((2,)), target_partition=(2,), target_L=L))


def test_verified_product_and_runtime_binding_reuse_complete_map(monkeypatch):
    monkeypatch.setenv("YE3T_CACHE_MODE", "auto")
    monkeypatch.setenv("YE3T_CACHE_VERIFY", "hash")
    compiled = compiled_product()

    def forbidden(*args, **kwargs):
        raise AssertionError("Identical canonical product must not be reconstructed")

    monkeypatch.setattr(core, "compile_carrier_product", forbidden)
    actual = json.loads(json.dumps(compiled))
    assert core._validated_carrier_product(actual) == compiled
    bound = core.bind_carrier_product_torch(actual, dtype=torch.complex128, backend="torch")
    values = torch.randn(2, 2, 1, 3, dtype=torch.complex128, requires_grad=True)
    output = bound(values, values.clone())
    gradient, = torch.autograd.grad(output.abs().square().sum(), values)
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("rehash", [False, True])
def test_changed_coefficient_cannot_reuse_receipt_even_with_new_hash(rehash):
    compiled = compiled_product()
    damaged = copy.deepcopy(compiled)
    damaged["angular"][0][-1] = "19"
    if rehash:
        damaged["self_hash"] = core._hash({key: value for key, value in damaged.items() if key != "self_hash"})
    with pytest.raises(ValueError, match="canonical fixed maps"):
        core._validated_carrier_product(damaged)
    assert core._validated_carrier_product(compiled) == compiled


def test_receipt_preserves_constructor_container_validation():
    compiled = compiled_product()
    changed = copy.deepcopy(compiled)
    changed["angular"] = tuple(changed["angular"])
    assert core._hash({key: value for key, value in changed.items() if key != "self_hash"}) == compiled["self_hash"]
    with pytest.raises(ValueError, match="canonical fixed maps"):
        core._validated_carrier_product(changed)


@pytest.mark.parametrize("setting,value", [("YE3T_CACHE_VERIFY", "full"),
                                           ("YE3T_CACHE_VERIFY", " FULL "),
                                           ("YE3T_CACHE_MODE", "off"),
                                           ("YE3T_CACHE_MODE", "rebuild"),
                                           ("YE3T_CACHE_MODE", " Refresh ")])
def test_explicit_verification_modes_repeat_canonical_check(monkeypatch, setting, value):
    compiled = compiled_product()
    original = core.compile_carrier_product
    calls = []

    def observed(request):
        calls.append(True)
        return original(request)

    monkeypatch.setattr(core, "compile_carrier_product", observed)
    monkeypatch.setenv(setting, value)
    assert core._validated_carrier_product(compiled) == compiled
    assert calls == [True]


def test_unknown_maps_require_canonical_check_and_receipts_are_bounded(monkeypatch):
    monkeypatch.setenv("YE3T_CACHE_MODE", "auto")
    monkeypatch.setenv("YE3T_CACHE_VERIFY", "hash")
    monkeypatch.setattr(core, "_verified_product_limit", 2)
    core._verified_product_hashes.clear()
    maps = [compiled_product(L) for L in (0, 1, 2)]
    assert len(core._verified_product_hashes) == 2
    assert maps[0]["self_hash"] not in core._verified_product_hashes
    assert all(value is None for value in core._verified_product_hashes.values())
    original = core.compile_carrier_product
    calls = []

    def observed(request):
        calls.append(True)
        return original(request)

    monkeypatch.setattr(core, "compile_carrier_product", observed)
    assert core._validated_carrier_product(maps[0]) == maps[0]
    assert calls == [True]
    assert len(core._verified_product_hashes) == 2
