"""Shared-polynomial derivatives with sparse factors and wide input channels."""

import os

import pytest
import torch

from ye3t.runtime.execution_plan import (
    native_execution_plan_capabilities,
    symmetric_power_shared_monomial_contraction,
    symmetric_power_shared_monomial_reference,
)


def _table(width, dtype):
    counts = torch.zeros((5, width), dtype=torch.long)
    counts[1, 0] = 2
    counts[2, -1] = 3
    counts[3, [0, width - 1]] = 1
    counts[4, [0, width // 2, width - 1]] = torch.tensor([1, 2, 1])
    return (counts, torch.tensor([0, 3, 6]), torch.tensor([0, 1, 3, 2, 3, 4]),
            torch.tensor([0, 0, 0, 1, 1, 1]),
            torch.tensor([.3, -.5, .7, -.4, .6, 1.2], dtype=dtype))


def _direct(x):
    a, b, c = x[:, 0], x[:, x.shape[1] // 2], x[:, -1]
    return torch.stack((.3 - .5*a*a + .7*a*c,
                        -.4*c*c*c + .6*a*c + 1.2*a*b*b*c), dim=-1)


@pytest.mark.parametrize("backend", ["reference", "native"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_wide_shared_polynomial_complete_derivatives(backend, dtype):
    if backend == "native" and not (
            os.environ.get("YE3T_TEST_CPP_EXTENSION")
            or native_execution_plan_capabilities()["prebuilt_extension"]):
        pytest.skip("requires the optional native extension or YE3T_TEST_CPP_EXTENSION=1")
    torch.manual_seed(192)
    x = torch.randn((3, 200), dtype=dtype)
    x[0, 0] = 0
    x[1, -1] = 0
    x[2, 100] = 0
    x.requires_grad_()
    actual = symmetric_power_shared_monomial_contraction(x, *_table(200, dtype), backend=backend)
    expected = _direct(x)
    torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-13)
    actual_first = torch.autograd.grad(actual.abs().square().sum(), x, create_graph=True)[0]
    expected_first = torch.autograd.grad(expected.abs().square().sum(), x, create_graph=True)[0]
    direction = torch.randn_like(x)
    actual_second = torch.autograd.grad(actual_first, x, direction)[0]
    expected_second = torch.autograd.grad(expected_first, x, direction)[0]
    torch.testing.assert_close(actual_first, expected_first, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(actual_second, expected_second, atol=2e-11, rtol=2e-11)


def test_reference_saved_derivative_storage_ignores_absent_channels():
    sizes = []
    for width in (20, 200):
        saved_bytes = []
        def save(tensor):
            saved_bytes.append(tensor.numel()*tensor.element_size())
            return tensor
        x = torch.full((3, width), .25, dtype=torch.float64, requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
            values = symmetric_power_shared_monomial_reference(x, *_table(width, x.dtype))
            first = torch.autograd.grad(values.square().sum(), x, create_graph=True)[0]
            torch.autograd.grad(first.square().sum(), x)
        sizes.append(sum(saved_bytes))
    # Wider input gradients themselves grow. Absent factors must not each
    # create another full monomial graph during the derivative calculation.
    assert sizes[1] <= sizes[0] + 8 * 3 * (200 - 20) * 8


def test_shared_reference_rejects_negative_powers():
    table = list(_table(3, torch.float64))
    table[0][1, 0] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        symmetric_power_shared_monomial_reference(torch.ones(2, 3), *table)
