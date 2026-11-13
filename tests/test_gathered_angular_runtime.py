"""Bounded gather/angular derivatives preserve complete compiler coordinates."""
import pytest
import torch

from test_cauchy_program_batches import schedule, sequential
from ye3t.runtime.factorized_polynomial import _FactorizedPolynomialRuntime
from ye3t.runtime.execution_plan import _gathered_factorized_angular


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("basis", ["real_tesseral", "complex_condon_shortley"])
@pytest.mark.parametrize("shape", [(0,), (2,), (2, 2)])
@pytest.mark.parametrize("block_size", [2, 4])
def test_complete_values_and_mixed_derivatives(device, basis, shape, block_size, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    import ye3t.runtime.factorized_polynomial as polynomial
    calls = []
    def counted(*args):
        calls.append(1)
        return _gathered_factorized_angular(*args)
    monkeypatch.setattr(polynomial, "_gathered_factorized_angular", counted)
    model = _FactorizedPolynomialRuntime(schedule(block_size=block_size), path_basis=basis).to(device)
    torch.manual_seed(20261105)
    x = (torch.randn(*shape, model.schedule["input_dimension"], dtype=torch.float64, device=device)/3).requires_grad_()
    expected, actual = sequential(model, x), model(x)
    assert calls
    torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-13)
    weight = torch.linspace(.1, .7, actual.shape[-1], dtype=x.dtype, device=device).requires_grad_()
    derivatives = []
    for value in (expected, actual):
        energy = (value*weight).square().sum()
        first = torch.autograd.grad(energy, x, create_graph=True)[0]
        second = torch.autograd.grad(.1*energy+first.square().sum(), (x, weight), retain_graph=True)
        derivatives.append((first, *second))
    for a, b in zip(*derivatives, strict=True):
        torch.testing.assert_close(a, b, rtol=4e-10, atol=4e-12)
    changed = x.detach()+.03
    torch.testing.assert_close(model(changed), sequential(model, changed), rtol=3e-12, atol=3e-13)


@pytest.mark.parametrize("basis", ["real_tesseral", "complex_condon_shortley"])
def test_finite_differences_and_shared_bank_retention(basis):
    model = _FactorizedPolynomialRuntime(schedule(), path_basis=basis)
    x = torch.zeros(1, model.schedule["input_dimension"], dtype=torch.float64)
    group = model._path_batch_plan(x)[2][0]
    arguments, width, workspace = group["native"]
    output_width = group["output_width"]
    torch.manual_seed(20261105)
    bank = (torch.randn(1, width+2, dtype=arguments[-1].dtype)/3).requires_grad_()
    # Repeated routes share every input; indices have a nonzero offset.
    indices = (torch.arange(2*width) % width + 1).long()
    def function(value):
        return _gathered_factorized_angular(value, indices, arguments, width, workspace, output_width)
    assert torch.autograd.gradcheck(function, (bank,), fast_mode=True)
    assert torch.autograd.gradgradcheck(function, (bank,), fast_mode=True)
    retained = []
    def save(value):
        retained.append(value)
        return value
    with torch.autograd.graph.saved_tensors_hooks(save, lambda value: value):
        output = function(bank)
        cotangent = torch.ones_like(output, requires_grad=True)
        gradient = torch.autograd.grad(output, bank, cotangent, create_graph=True)[0]
    permitted = {bank.data_ptr(), indices.data_ptr(), cotangent.data_ptr(),
                 *(value.data_ptr() for value in arguments)}
    assert retained and all(value.data_ptr() in permitted for value in retained)
    assert gradient.shape == bank.shape


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_value_only_state_dtype_and_device_lifecycle(device, monkeypatch):
    import ye3t.runtime.execution_plan as execution
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    model = _FactorizedPolynomialRuntime(schedule()).to(device)
    torch.manual_seed(20261105)
    x = torch.randn(2, model.schedule["input_dimension"], dtype=torch.float64, device=device)/3
    keys = tuple(model.state_dict())
    expected = sequential(model, x)
    def unexpected_derivative_boundary(*args):
        pytest.fail("Value-only execution created derivative bookkeeping")
    with monkeypatch.context() as patch:
        patch.setattr(execution._GatheredAngular, "apply", unexpected_derivative_boundary)
        torch.testing.assert_close(model(x), expected, rtol=3e-12, atol=3e-13)
        with torch.no_grad():
            torch.testing.assert_close(model(x.requires_grad_()), expected, rtol=3e-12, atol=3e-13)
    for assign in (False, True):
        model.load_state_dict({key: value.clone() for key, value in model.state_dict().items()}, assign=assign)
        torch.testing.assert_close(model(x), expected, rtol=3e-12, atol=3e-13)
        assert tuple(model.state_dict()) == keys
    model.float()
    torch.testing.assert_close(model(x.float()), expected.float(), rtol=3e-5, atol=3e-7)
    model.double().cpu()
    torch.testing.assert_close(model(x.cpu()), expected.cpu(), rtol=3e-12, atol=3e-13)


def test_saved_bank_and_coefficients_reject_mutation():
    model = _FactorizedPolynomialRuntime(schedule())
    x = torch.zeros(1, model.schedule["input_dimension"], dtype=torch.float64)
    group = model._path_batch_plan(x)[2][0]
    original, width, workspace = group["native"]
    for which in ("bank", "coefficient"):
        arguments = tuple(value.clone() for value in original)
        bank = torch.randn(1, width, dtype=torch.float64, requires_grad=True)
        output = _gathered_factorized_angular(bank, torch.arange(width), arguments,
                                             width, workspace, group["output_width"])
        with torch.no_grad():
            (bank if which == "bank" else arguments[-1]).add_(.01)
        with pytest.raises(RuntimeError, match="modified by an inplace operation"):
            torch.autograd.grad(output.sum(), bank)
