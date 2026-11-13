"""Existing compiler maps packed through the canonical sparse linear operator."""
from functools import lru_cache

import pytest
import torch

from test_cauchy_program_batches import schedule, sequential
from ye3t.runtime.execution_plan import _load_extension
from ye3t.runtime.factorized_polynomial import _FactorizedPolynomialRuntime


@lru_cache(None)
def fixture(kind):
    if kind.startswith("tagged"):
        if kind == "tagged_unit":
            return schedule(2, 1)
        return schedule(int(kind[-1]), 2)
    from ye3t.couplings.covariant_cauchy import covariant_cauchy_request
    from ye3t.couplings.factorized_shell import _compile_factorized_shell_source, factorized_shell_schedule
    payloads = []
    for offset in (0, 2):
        request = dict(coupling_request=covariant_cauchy_request([
            dict(neighbor_species="X", radial_channel=offset+i, l=ell, source_family_id="roles")
            for i, ell in enumerate((1, 2))], (2, 2), target_L=1, role_dimension=3,
            angular_basis_backend="normalized_cg_paths_v1"), radial_count=4, shell_role_action="permutation")
        source = _compile_factorized_shell_source(request)
        payloads.append(dict(factorized_schedule=source, self_hash=source["self_hash"]))
    selected = [row["label"]["coordinate_id"] for payload in payloads
                for row in payload["factorized_schedule"]["inventory"]]
    return factorized_shell_schedule(payloads, selected, None)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("kind", ["tagged2", "tagged4", "shell"])
@pytest.mark.parametrize("shape", [(0,), (2,), (2, 2)])
def test_fixed_linear_maps_complete_mixed_derivatives(device, kind, shape, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    _load_extension()
    runtime = _FactorizedPolynomialRuntime(fixture(kind)).to(device)
    torch.manual_seed(20261105)
    x = (torch.randn(*shape, runtime.schedule["input_dimension"], dtype=torch.float64, device=device)/3).requires_grad_()
    expected, actual = sequential(runtime, x), runtime(x)
    assert runtime.packed_output_calls == 1 and runtime.packed_linear_output_calls == 1
    torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-13)
    weights = torch.linspace(.1, .7, actual.shape[-1], dtype=x.dtype, device=device).requires_grad_()
    results = []
    for value in (expected, actual):
        energy = (value*weights).square().sum()
        first = torch.autograd.grad(energy, x, create_graph=True)[0]
        second = torch.autograd.grad(.1*energy+first.square().sum(), (x, weights), retain_graph=True)
        results.append((first, *second))
    for observed, reference in zip(*results, strict=True):
        torch.testing.assert_close(observed, reference, rtol=4e-10, atol=4e-12)
    builds = runtime.packed_output_plan_builds
    changed = x.detach()+.02
    torch.testing.assert_close(runtime(changed), sequential(runtime, changed), rtol=3e-12, atol=3e-13)
    assert runtime.packed_output_plan_builds == builds


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_linear_map_rebinding_and_trainable_map_fallback(device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    _load_extension()
    runtime = _FactorizedPolynomialRuntime(fixture("shell")).to(device)
    x = torch.full((2, runtime.schedule["input_dimension"]), .2, dtype=torch.float64, device=device, requires_grad=True)
    saved = {name: value.clone() for name, value in runtime.state_dict().items()}
    runtime(x)
    with torch.no_grad():
        runtime.path_output_map_0.mul_(1.5)
        runtime.outputs_0.copy_(runtime.outputs_0.flip(0))
    torch.testing.assert_close(runtime(x), sequential(runtime, x), rtol=3e-12, atol=3e-13)
    assert runtime.packed_output_plan_builds == 2
    runtime.load_state_dict(saved, assign=True)
    assert not runtime.packed_output_plans
    torch.testing.assert_close(runtime(x), sequential(runtime, x), rtol=3e-12, atol=3e-13)
    runtime.float().double()
    assert not runtime.packed_output_plans
    torch.testing.assert_close(runtime(x), sequential(runtime, x), rtol=3e-12, atol=3e-13)
    runtime.path_output_map_0.requires_grad_()
    calls = runtime.packed_output_calls
    expected, actual = sequential(runtime, x), runtime(x)
    assert runtime.packed_output_calls == calls
    torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-13)
    for observed, reference in zip(torch.autograd.grad(actual.square().sum(), (x, runtime.path_output_map_0)),
                                   torch.autograd.grad(expected.square().sum(), (x, runtime.path_output_map_0)), strict=True):
        torch.testing.assert_close(observed, reference, rtol=4e-10, atol=4e-12)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_shell_complete_role_action(device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from ye3t.representations.projectors import adjacent_transposition_representation_matrix
    _load_extension()
    runtime = _FactorizedPolynomialRuntime(fixture("shell")).to(device)
    torch.manual_seed(412)
    values = torch.randn(2, runtime.schedule["input_dimension"], device=device, dtype=torch.float64)/3
    output = runtime(values)
    for adjacent in range(2):
        indices, offset = [], 0
        for ell in (1, 2, 1, 2):
            order = list(range(3))
            order[adjacent], order[adjacent+1] = order[adjacent+1], order[adjacent]
            indices.extend(offset+r*(2*ell+1)+m for r in order for m in range(2*ell+1))
            offset += 3*(2*ell+1)
        assert offset == values.shape[-1]
        moved = runtime(values[:, indices])
        for row in runtime.schedule["inventory"]:
            start, end = row["component_slice"]
            partition = tuple(row["label"]["shell_partition"])
            action = torch.tensor(adjacent_transposition_representation_matrix(partition, adjacent).tolist(),
                                  device=device, dtype=values.dtype)
            expected = torch.einsum("ut,btm->bum", action, output[:, start:end].reshape(2, len(action), 3))
            torch.testing.assert_close(moved[:, start:end], expected.flatten(1), atol=3e-11, rtol=3e-11)
    assert runtime.packed_linear_output_calls == 3


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("kind", ["shell", "tagged4"])
def test_existing_explicit_design_row_pullback(device, kind, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    _load_extension()
    runtime = _FactorizedPolynomialRuntime(fixture(kind)).to(device)
    torch.manual_seed(943)
    values = (torch.randn(2, runtime.schedule["input_dimension"], device=device, dtype=torch.float64)/3).requires_grad_()
    output = runtime(values)
    seeds = torch.randn(2, *output.shape, device=device, dtype=values.dtype)
    explicit = runtime.batched_adjoint(values, seeds)
    automatic = torch.stack([torch.autograd.grad(output, values, seed, create_graph=True, retain_graph=True)[0]
                             for seed in seeds])
    torch.testing.assert_close(explicit, automatic, rtol=3e-11, atol=3e-12)
    torch.testing.assert_close(torch.autograd.grad(explicit.square().sum(), values)[0],
                              torch.autograd.grad(automatic.square().sum(), values)[0],
                              rtol=3e-10, atol=3e-11)
    assert runtime.packed_linear_output_calls == 1


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("kind", ["shell", "tagged2", "tagged_unit"])
def test_output_binding_after_inference_warmup(device, kind, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    _load_extension()
    runtime = _FactorizedPolynomialRuntime(fixture(kind)).to(device)
    values = torch.full((2, runtime.schedule["input_dimension"]), .23, device=device, dtype=torch.float64)
    # Prime existing angular bindings separately to isolate the new output cache.
    runtime._path_forward_many(values)
    with torch.inference_mode():
        runtime(values)
    values.requires_grad_()
    actual, expected = runtime(values), sequential(runtime, values)
    torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-13)
    torch.testing.assert_close(torch.autograd.grad(actual.square().sum(), values)[0],
                              torch.autograd.grad(expected.square().sum(), values)[0],
                              rtol=4e-10, atol=4e-12)
