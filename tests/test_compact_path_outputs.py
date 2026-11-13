"""Compact storage of existing compiler maps preserves the dense operation."""
from copy import deepcopy

import pytest
import torch

import ye3t.runtime.factorized_polynomial as runtime_module
from test_packed_path_outputs import fixture
from test_cauchy_program_batches import sequential
from ye3t.runtime.execution_plan import _load_extension


def pair(monkeypatch, kind, device, **kwargs):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    _load_extension()
    with monkeypatch.context() as patch:
        patch.setattr(runtime_module, "_DENSE_OUTPUT_MAP_BYTES", 2**60)
        dense = runtime_module._FactorizedPolynomialRuntime(fixture(kind), **kwargs).to(device)
    monkeypatch.setattr(runtime_module, "_DENSE_OUTPUT_MAP_BYTES", -1)
    compact = runtime_module._FactorizedPolynomialRuntime(fixture(kind), **kwargs).to(device)
    assert not any(name.startswith("path_output_map_") for name in compact._buffers)
    return dense, compact


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("kind", ["tagged2", "tagged4", "shell"])
@pytest.mark.parametrize("shape", [(0,), (2,), (2, 2)])
def test_compact_complete_force_loss_derivatives(monkeypatch, device, kind, shape):
    dense, compact = pair(monkeypatch, kind, device)
    torch.manual_seed(127)
    x = (torch.randn(*shape, dense.schedule["input_dimension"], device=device, dtype=torch.float64)/3).requires_grad_()
    expected, actual = sequential(dense, x), compact(x)
    torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-13)
    weights = torch.linspace(.1, .7, actual.shape[-1], device=device, dtype=x.dtype).requires_grad_()
    derivatives = []
    for output in (expected, actual):
        energy = (output*weights).square().sum()
        first = torch.autograd.grad(energy, x, create_graph=True)[0]
        second = torch.autograd.grad(.1*energy+first.square().sum(), (x, weights), retain_graph=True)
        derivatives.append((first, *second))
    for reference, observed in zip(*derivatives, strict=True):
        torch.testing.assert_close(observed, reference, rtol=4e-10, atol=4e-12)
    assert compact.packed_linear_output_calls == 1
    assert not any(name.startswith("path_output_map_") for name in compact._buffers)
    compact(x.detach()+.01)
    assert compact.packed_output_plan_builds == 1


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("basis", ["real_tesseral", "complex_condon_shortley"])
@pytest.mark.parametrize("backend", ["native", "reference"])
def test_compact_analytical_rows(monkeypatch, device, basis, backend):
    dense, compact = pair(monkeypatch, "tagged2", device, path_basis=basis, backend=backend)
    torch.manual_seed(827)
    values = (torch.randn(2, dense.schedule["input_dimension"], device=device, dtype=torch.float64)/3).requires_grad_()
    actual, expected = compact(values), dense(values)
    torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-13)
    seeds = torch.randn(2, *actual.shape, device=device, dtype=values.dtype)
    adjoint = compact.batched_adjoint(values, seeds)
    reference = torch.stack([torch.autograd.grad(expected, values, seed, create_graph=True, retain_graph=True)[0]
                             for seed in seeds])
    torch.testing.assert_close(adjoint, reference, rtol=3e-11, atol=3e-12)
    torch.testing.assert_close(torch.autograd.grad(adjoint.square().sum(), values)[0],
                              torch.autograd.grad(reference.square().sum(), values)[0], rtol=3e-10, atol=3e-11)
    assert not any(name.startswith("path_output_map_") for name in compact._buffers)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("assign", [False, True])
def test_compact_state_inspection_and_mutation(monkeypatch, device, assign):
    dense, compact = pair(monkeypatch, "shell", device)
    torch.manual_seed(281)
    x = (torch.randn(2, dense.schedule["input_dimension"], device=device, dtype=torch.float64)/3).requires_grad_()
    saved = {name: value.clone() for name, value in compact.state_dict().items()}
    compact(x)
    with torch.no_grad():
        compact.path_output_values_0.mul_(1.5)
        dense.path_output_map_0.mul_(1.5)
        compact.outputs_0.copy_(compact.outputs_0.flip(0))
        dense.outputs_0.copy_(dense.outputs_0.flip(0))
    torch.testing.assert_close(compact(x), sequential(dense, x), rtol=3e-12, atol=3e-13)
    assert compact.packed_output_plan_builds == 2
    # Explicit inspection requests a dense buffer, including all zero entries
    # for coefficient derivatives. No dense allocation happens on normal calls.
    torch.testing.assert_close(compact.path_output_map_0, dense.path_output_map_0, rtol=0, atol=0)
    compact.path_output_map_0.requires_grad_()
    dense.path_output_map_0.requires_grad_()
    for observed, reference in zip(torch.autograd.grad(compact(x).square().sum(), (x, compact.path_output_map_0)),
                                   torch.autograd.grad(sequential(dense, x).square().sum(), (x, dense.path_output_map_0)), strict=True):
        torch.testing.assert_close(observed, reference, rtol=4e-10, atol=4e-12)
    compact.load_state_dict(saved, assign=assign)
    dense.load_state_dict(saved, assign=assign)
    assert "path_output_map_0" not in compact._buffers and "path_output_map_0" not in dense._buffers
    torch.testing.assert_close(compact(x), dense(x), rtol=3e-12, atol=3e-13)
    # Old dense states also load into a compact runtime, preserving their values.
    dense.path_output_map_0
    compact.load_state_dict(dense.state_dict(), assign=assign)
    assert "path_output_map_0" in compact._buffers
    torch.testing.assert_close(compact(x), dense(x), rtol=3e-12, atol=3e-13)


def test_compact_dtype_duplicates_and_invalid_state(monkeypatch):
    request = deepcopy(fixture("tagged2"))
    entries = list(request["programs"][0]["schedule"]["output_map"])
    entries.extend([(0, 0, 1e8), (0, 0, -1e8), (0, 0, 1+1e-8)])
    request["programs"][0]["schedule"]["output_map"] = entries
    monkeypatch.setattr(runtime_module, "_DENSE_OUTPUT_MAP_BYTES", 2**60)
    dense = runtime_module._FactorizedPolynomialRuntime(request, dtype=torch.float32)
    monkeypatch.setattr(runtime_module, "_DENSE_OUTPUT_MAP_BYTES", -1)
    compact = runtime_module._FactorizedPolynomialRuntime(request, dtype=torch.float32)
    for dtype in (torch.float32, torch.float64, torch.float32, torch.float64):
        dense.to(dtype)
        compact.to(dtype)
        for p in compact.path_programs:
            rows, columns, values = compact._compact_output_tables(p)
            expected = getattr(dense, f"path_output_map_{p}")
            torch.testing.assert_close(values, expected[rows, columns], rtol=0, atol=0)
    saved = {name: value.clone() for name, value in compact.state_dict().items()}
    wrong_type = dict(saved, path_output_rows_0=saved["path_output_rows_0"].double())
    with pytest.raises(ValueError, match="Invalid compact"):
        compact.load_state_dict(wrong_type)
    saved["path_output_rows_0"][0] = compact.path_output_shapes[0][0]
    with pytest.raises(ValueError, match="Invalid compact"):
        compact.load_state_dict(saved, assign=True)


def test_compact_storage_and_inference_warmup(monkeypatch):
    dense, compact = pair(monkeypatch, "shell", "cpu")
    dense_bytes = sum(value.numel()*value.element_size() for p in dense.path_programs for value in dense._output_buffers(p))
    compact_bytes = sum(value.numel()*value.element_size() for p in compact.path_programs for value in compact._output_buffers(p))
    assert compact_bytes < dense_bytes
    x = torch.full((2, dense.schedule["input_dimension"]), .23, dtype=torch.float64)
    compact._path_forward_many(x)
    with torch.inference_mode():
        compact(x)
    x.requires_grad_()
    expected, actual = sequential(dense, x), compact(x)
    torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-13)
    torch.testing.assert_close(torch.autograd.grad(actual.square().sum(), x)[0],
                              torch.autograd.grad(expected.square().sum(), x)[0], rtol=4e-10, atol=4e-12)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_compact_complete_shell_role_action(monkeypatch, device):
    from test_packed_path_outputs import test_shell_complete_role_action
    monkeypatch.setattr(runtime_module, "_DENSE_OUTPUT_MAP_BYTES", -1)
    test_shell_complete_role_action(device, monkeypatch)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_compact_repeated_output_selection_and_coefficient_gradients(monkeypatch, device):
    dense, compact = pair(monkeypatch, "shell", device)
    # Reordered and repeated selections retain their separate output positions.
    for model in (dense, compact):
        model.outputs_0 = torch.tensor([2, 0, 2, 1], device=device)
    torch.manual_seed(173)
    x = (torch.randn(2, dense.schedule["input_dimension"], device=device, dtype=torch.float64)/3).requires_grad_()
    expected, actual = sequential(dense, x), compact(x)
    torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-13)
    compact.path_output_values_0.requires_grad_()
    dense.path_output_map_0.requires_grad_()
    calls = compact.packed_output_calls
    expected, actual = sequential(dense, x), compact(x)
    assert compact.packed_output_calls == calls
    gradients = torch.autograd.grad(actual.square().sum(), (x, compact.path_output_values_0), create_graph=True)
    references = torch.autograd.grad(expected.square().sum(), (x, dense.path_output_map_0), create_graph=True)
    rows, columns, _ = compact._compact_output_tables(0)
    torch.testing.assert_close(gradients[0], references[0], rtol=4e-10, atol=4e-12)
    torch.testing.assert_close(gradients[1], references[1][rows, columns], rtol=4e-10, atol=4e-12)
    torch.testing.assert_close(torch.autograd.grad(gradients[1].square().sum(), x)[0],
                              torch.autograd.grad(references[1][rows, columns].square().sum(), x)[0],
                              rtol=4e-10, atol=4e-12)
    # Changed invalid indices must be rejected before entering a native kernel.
    compact.path_output_values_0.requires_grad_(False)
    with torch.no_grad():
        compact.path_output_columns_0[0] = compact.path_output_shapes[0][1]
    with pytest.raises(ValueError, match="Invalid compact"):
        compact(x)


def test_compact_direct_dense_replacement(monkeypatch):
    dense, compact = pair(monkeypatch, "shell", "cpu")
    matrix = dense.path_output_map_0.clone().mul_(1.25)
    # Assignment must replace the registered compact map even if the caller
    # has never inspected or materialized its dense form.
    compact.path_output_map_0 = matrix
    dense.path_output_map_0 = matrix.clone()
    assert compact._buffers["path_output_map_0"] is matrix
    assert "path_output_values_0" not in compact._buffers
    x = torch.randn(2, dense.schedule["input_dimension"], dtype=torch.float64)/3
    torch.testing.assert_close(compact(x), sequential(dense, x), rtol=3e-12, atol=3e-13)
