"""Independent values and mixed second derivatives for small channel products."""

import pytest
import torch
from ye3t.backends.small_matrix import small_bmm


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_small_matrix_values_and_mixed_derivatives(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    if device == "cuda":
        pytest.importorskip("triton")
    torch.manual_seed(174)
    for rows, m, k, n in ((7, 8, 8, 2), (9, 3, 5, 10), (5, 1, 1, 1),
                           (0, 8, 8, 6), (3, 33, 2, 4)):
        left = torch.randn(rows, k, m, device=device, dtype=dtype).transpose(1, 2).requires_grad_()
        right = torch.randn(rows, n, k, device=device, dtype=dtype).transpose(1, 2).requires_grad_()
        directions = (torch.randn_like(left), torch.randn_like(right))
        report = {}
        expected = torch.bmm(left, right)
        actual = small_bmm(left, right, report)
        torch.testing.assert_close(actual, expected)
        assert report == {"triton_small_bmm" if device == "cuda" and m <= 32 else "torch_bmm": 1}
        derivatives = []
        for value in (actual, expected):
            first = torch.autograd.grad(value.sin().square().sum(), (left, right), create_graph=True)
            second = torch.autograd.grad(sum((a*b).sum() for a, b in zip(first, directions)), (left, right))
            derivatives.append((*first, *second))
        tolerance = 3e-4 if dtype == torch.float32 else 3e-11
        for a, b in zip(*derivatives):
            torch.testing.assert_close(a, b, rtol=tolerance, atol=tolerance)


def test_small_matrix_expanded_batch_stride_and_gradgradcheck():
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    left = torch.randn(1, 2, 3, dtype=torch.float64, device="cuda").requires_grad_()
    right = torch.randn(3, 3, 2, dtype=torch.float64, device="cuda").requires_grad_()
    def function(a, b):
        return small_bmm(a.expand(3, -1, -1), b)
    assert torch.autograd.gradcheck(function, (left, right))
    assert torch.autograd.gradgradcheck(function, (left, right))
