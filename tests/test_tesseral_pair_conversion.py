"""Sparse tesseral conversion agrees with the established unitary matrix."""

import pytest
import torch

from ye3t.core.tesseral import (
    _real_to_complex_matrix,
    complex_multiplet_to_real_tesseral,
    real_tesseral_to_complex_multiplet,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pair_conversion_values_and_first_second_derivatives(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    torch.manual_seed(730)
    for ell in range(11):
        values = torch.randn(4, 3, 2*ell+1, dtype=dtype, device=device).requires_grad_()
        matrix = _real_to_complex_matrix(ell, dtype, device)
        actual = real_tesseral_to_complex_multiplet(values, ell)
        expected = values.to(matrix.dtype) @ matrix
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(complex_multiplet_to_real_tesseral(actual, ell, range(-ell, ell+1)), values)
        direction = torch.randn_like(values)
        derivatives = []
        for result in (actual, expected):
            loss = (result.real.sin()+.3*result.imag.cos()).square().sum()
            first = torch.autograd.grad(loss, values, create_graph=True)[0]
            second = torch.autograd.grad((first*direction).sum(), values)[0]
            derivatives.append((first, second))
        for left, right in zip(*derivatives):
            torch.testing.assert_close(left, right)
        empty = values[:0]
        torch.testing.assert_close(real_tesseral_to_complex_multiplet(empty, ell), empty.to(matrix.dtype) @ matrix)
