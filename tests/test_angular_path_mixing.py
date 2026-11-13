"""Existing Young multiplicity maps use bounded native sparse application."""
import pytest
import torch

from ye3t.representations.young_angular_paths import angular_path_schedule
from ye3t.runtime.angular_paths import BoundAngularPaths


@pytest.fixture(autouse=True)
def deterministic_reference_reductions():
    # The mutation tests compare two executions of the same reference
    # index_add bit-for-bit. CUDA atomic summation order is otherwise variable.
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


@pytest.fixture(scope="module")
def schedule():
    return angular_path_schedule(4, 2, (2, 2), 2)


def reference(runtime, values, transpose=False):
    paths, copies = len(runtime.schedule["outputs"]), runtime.schedule["copies"]*runtime.schedule["tableaux"]
    rows, columns = (runtime.columns, runtime.paths) if transpose else (runtime.paths, runtime.columns)
    output = values.new_zeros((*values.shape[:-2], paths if transpose else copies, values.shape[-1]))
    return output.index_add(-2, columns, values[..., rows, :]*runtime.coefficients.conj().unsqueeze(-1))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize("shape", [(0,), (2,), (2, 2)])
@pytest.mark.parametrize("transpose", [False, True])
def test_native_mixing_values_and_complete_derivatives(schedule, device, dtype, shape, transpose, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    runtime = BoundAngularPaths(schedule, dtype=dtype, device=device, backend="native")
    copies = schedule["copies"]*schedule["tableaux"]
    width = copies if transpose else len(schedule["outputs"])
    torch.manual_seed(847)
    values = torch.randn(*shape, width, 5, dtype=dtype, device=device, requires_grad=True)
    operation = "source_analysis_adjoint" if transpose else "source_analysis"
    original, calls = getattr(torch.ops.ye3t_runtime, operation), []

    def observed(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(torch.ops.ye3t_runtime, operation, observed)
    actual, expected = runtime._mix_paths(values, transpose), reference(runtime, values, transpose)
    assert len(calls) == 1
    torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-13)
    weights = torch.linspace(.3, .7, actual.shape[-2], device=device, dtype=torch.float64).requires_grad_()
    results = []
    for output in (actual, expected):
        energy = (output.abs().square()*weights[:, None]).sum()
        first, = torch.autograd.grad(energy, values, create_graph=True, retain_graph=True)
        second = torch.autograd.grad(.1*energy+first.abs().square().sum(), (values, weights), retain_graph=True)
        results.append((first, *second))
    for a, b in zip(*results, strict=True):
        torch.testing.assert_close(a, b, rtol=3e-11, atol=3e-12)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("transpose", [False, True])
def test_mixing_mutation_and_trainable_coefficients_keep_reference_derivatives(schedule, device, transpose, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    runtime = BoundAngularPaths(schedule, dtype=torch.complex128, device=device, backend="native")
    width = schedule["copies"]*schedule["tableaux"] if transpose else len(schedule["outputs"])
    values = torch.randn(2, width, 5, dtype=torch.complex128, device=device, requires_grad=True)
    runtime._mix_paths(values, transpose)
    with torch.no_grad():
        runtime.coefficients.mul_(1+.2j)
    runtime.coefficients.requires_grad_()

    def forbidden(*args):
        pytest.fail("Differentiable/edited coefficient maps must use the reference operation")

    monkeypatch.setattr(torch.ops.ye3t_runtime, "source_analysis", forbidden)
    monkeypatch.setattr(torch.ops.ye3t_runtime, "source_analysis_adjoint", forbidden)
    actual, expected = runtime._mix_paths(values, transpose), reference(runtime, values, transpose)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradients = [torch.autograd.grad(output.abs().square().sum(), (values, runtime.coefficients),
                                    create_graph=True, retain_graph=True) for output in (actual, expected)]
    for a, b in zip(*gradients, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mixing_inference_warmup_and_replaced_indices(schedule, device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    runtime = BoundAngularPaths(schedule, dtype=torch.float64, device=device, backend="native")
    values = torch.randn(2, len(schedule["outputs"]), 5, dtype=torch.float64, device=device)
    with torch.inference_mode():
        runtime._mix_paths(values)
    assert all(not torch.is_inference(value) for value in runtime._mixing_identity)
    values.requires_grad_()
    actual = runtime._mix_paths(values)
    expected = reference(runtime, values)
    torch.testing.assert_close(torch.autograd.grad(actual.square().sum(), values)[0],
                              torch.autograd.grad(expected.square().sum(), values)[0], rtol=3e-12, atol=3e-13)
    runtime.paths = runtime.paths.flip(0)

    def forbidden(*args):
        pytest.fail("Replaced indices require the reference operation")

    monkeypatch.setattr(torch.ops.ye3t_runtime, "source_analysis", forbidden)
    torch.testing.assert_close(runtime._mix_paths(values), reference(runtime, values), rtol=0, atol=0)
