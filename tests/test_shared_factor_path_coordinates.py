"""Reduced shared-action coordinates against independent normalized CG chains."""
from functools import lru_cache

import pytest
import torch

from ye3t import couplings
from ye3t.core.cg import cg_exact
from ye3t.core.tesseral import (
    real_tesseral_to_complex_multiplet, complex_multiplet_to_real_tesseral)
from ye3t.couplings.shared_permutation_factors import _shared_factor_path_columns
from ye3t.execution_plan import YE3T_O3_PRIMARY_CONVENTION
from ye3t.representations.young_angular_paths import _inner


def request(partition=(2,), L=0, groups=((0, 1), (2, 3)), spins=(1, 1, 1, 1)):
    content = [None]*len(spins)
    for channel, group in enumerate(groups):
        for factor in group:
            content[factor] = channel
    return dict(family="shared_permutation_factors", factor_groups=groups,
                content=content, input_Ls=spins, target_partition=partition, target_L=L)


CASES = [request(), request((1, 1), 1),
    request((2,), 2, ((0, 2), (1, 3)), (1, 2, 1, 2)),
    request((2, 1), 1, ((0, 1, 2), (3, 4, 5)), (1,)*6),
    request((2,), 1, ((0, 1), (2, 3), (4, 5)), (1,)*6),
    request((2,), 0, ((0, 1), (2, 3), (4, 5), (6, 7)), (1,)*8),
    dict(family="shared_permutation_factors", factor_groups=[[0, 1], [2, 3]],
         fixed_factor_groups=[[4, 5]], content=[0, 0, 1, 1, 2, 2],
         input_Ls=[1]*6, target_partition=[2], target_L=0),
    dict(family="shared_permutation_factors", factor_groups=[[0, 2], [3, 5]],
         fixed_factor_groups=[[1, 4]], content=[0, 2, 0, 1, 2, 1],
         input_Ls=[1, 2, 1, 1, 2, 1], target_partition=[1, 1], target_L=1),
    dict(family="shared_permutation_factors", factor_groups=[[0, 1, 2]],
         fixed_factor_groups=[[3, 4]], content=[0, 0, 0, 1, 1],
         input_Ls=[1]*5, target_partition=[2, 1], target_L=1)]


@lru_cache(maxsize=64)
def cg_table(a, b, L):
    return [[[float(cg_exact(a, m, b, n, L, M)) for M in range(-L, L+1)]
             for n in range(-b, b+1)] for m in range(-a, a+1)]


def reference_paths(report, factors):
    # A bounded test oracle evaluates each normalized chain directly from CGs.
    # It does not call grouped-product or angular-path runtime implementations.
    spins = report["input_Ls"]
    cache = {}
    def chain(path):
        if path in cache:
            return cache[path]
        if len(path) == 1:
            value = factors[0]
        else:
            previous = chain(path[:-1])
            table = torch.tensor(cg_table(path[-2], spins[len(path)-1], path[-1]),
                                 device=previous.device, dtype=previous.dtype)
            value = torch.einsum("...i,...j,ijm->...m", previous, factors[len(path)-1], table)
        cache[path] = value
        return value
    copies = []
    for copy in report["columns"]:
        rows = [sum(float(coefficient)*chain(path) for path, coefficient in column) for column in copy]
        copies.append(torch.stack(rows, -2))
    return torch.stack(copies, -3)


@pytest.mark.parametrize("raw", CASES)
def test_exact_selected_map_isometry_without_magnetic_materialization(raw, monkeypatch):
    from ye3t.couplings import shared_permutation_factors as shared
    from ye3t.representations import young_angular_paths
    def forbidden(*args, **kwargs):
        pytest.fail("reduced coordinate construction materialized a runtime source table")
    monkeypatch.setattr(shared, "compile_shared_permutation", forbidden)
    monkeypatch.setattr(shared, "compile_carrier_product", forbidden)
    monkeypatch.setattr(young_angular_paths, "angular_path_schedule", forbidden)
    report = _shared_factor_path_columns(raw)
    columns = [dict(column) for copy in report["columns"] for column in copy]
    for a, left in enumerate(columns):
        for b, right in enumerate(columns):
            assert _inner(left, right) == int(a == b)
    assert report["magnetic_tensor_materialized"] is False
    assert report["physical_image_resolved"] is False
    count = couplings.count(raw)["output_channel_count"]
    selected = _shared_factor_path_columns({**raw, "selected_copy_indices": [count-1]})
    assert selected["columns"] == report["columns"][-1:]
    assert selected["labels"] == report["labels"][-1:]


@pytest.mark.parametrize("raw", CASES)
@pytest.mark.parametrize("complex_basis", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_every_component_and_complete_force_loss_gradients(raw, complex_basis, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    if complex_basis:
        raw = {**raw, "convention_id": YE3T_O3_PRIMARY_CONVENTION}
    report = _shared_factor_path_columns(raw)
    compiled = couplings.compile(couplings.plan(raw))
    plan = couplings.execution_plan_from_compiled_coupler(compiled)
    runtime = couplings.bind_typed_factorized_execution_plan_torch(plan,
        dtype=torch.complex128 if complex_basis else torch.float64, device=device)
    torch.manual_seed(851)
    spins, L = raw["input_Ls"], raw["target_L"]
    sizes = [2*ell+1 for ell in spins]
    x = torch.randn(2, sum(sizes), dtype=torch.float64, device=device, requires_grad=True)
    real_factors = x.split(sizes, -1)
    complex_factors = tuple(real_tesseral_to_complex_multiplet(value, ell)
        for value, ell in zip(real_factors, spins, strict=True))
    actual = runtime.evaluate(complex_factors if complex_basis else real_factors)
    expected = reference_paths(report, complex_factors)
    if not complex_basis:
        expected = complex_multiplet_to_real_tesseral(
            expected*((-1j)**report["real_phase_exponent"]), L, tuple(range(-L, L+1)))
    torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-12)
    weights = torch.randn(actual.shape[-3], dtype=x.dtype, device=device, requires_grad=True)
    derivatives = []
    for value in (actual, expected):
        energy = (value.abs().square().sum((-1, -2))*weights).sum()
        force = -torch.autograd.grad(energy, x, create_graph=True, retain_graph=True)[0]
        derivatives.append((force, *torch.autograd.grad(
            .1*energy+force.square().sum(), (x, weights), retain_graph=True)))
    for a, b in zip(*derivatives, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-10, atol=2e-9)


def test_collection_requires_explicit_sector_selection():
    raw = request()
    del raw["target_partition"], raw["target_L"]
    raw["source_sectors"] = {"L_max": 1}
    with pytest.raises(ValueError, match="one selected shared-action sector"):
        _shared_factor_path_columns(raw)


def test_selected_copy_only_recouples_required_children(monkeypatch):
    from ye3t.representations import young_angular_paths
    original = young_angular_paths._coupled_path_columns
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(young_angular_paths, "_coupled_path_columns", counted)
    raw = CASES[-1]
    complete = _shared_factor_path_columns(raw)
    all_calls = len(calls)
    calls.clear()
    selected = _shared_factor_path_columns({**raw, "selected_copy_indices": [0]})
    assert selected["columns"] == complete["columns"][:1]
    assert 0 < len(calls) < all_calls


def test_zero_multiplicity_has_no_coordinates():
    result = _shared_factor_path_columns(request((1, 1), 0))
    assert result["columns"] == ()
    assert result["layout"] is None
