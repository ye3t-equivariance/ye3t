"""Numeric indexed-pair derivatives and support-size kernel reuse."""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_indexed_pair_reuses_kernels_across_support_sizes_and_preserves_hvp():
    pytest.importorskip("triton")
    import ye3t.backends._indexed_tagged_pair as runtime

    generator = torch.Generator().manual_seed(19)
    kernels = (runtime._forward_kernel, runtime._left_adjoint_kernel, runtime._right_adjoint_kernel)
    cache_sizes = []
    for rows, left_rows in ((1, 1), (3, 2), (16, 3), (17, 5), (31, 7)):
        row_left = torch.arange(rows) % left_rows
        row_right = (3 * torch.arange(rows) + 1) % 9
        counts = torch.bincount(row_left, minlength=left_rows)
        arrays = (row_left, row_right, torch.cat((counts.new_zeros(1), counts.cumsum(0))),
            torch.argsort(row_left, stable=True), torch.tensor([0, 2, 4]), torch.tensor([0, 2, 1, 3]),
            torch.tensor([0, 2, 3, 4]), torch.tensor([0, 3, 1, 2]),
            torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1, 2, 0]))
        left = torch.randn(left_rows, 4, generator=generator, dtype=torch.float64)
        right = torch.randn(9, 3, generator=generator, dtype=torch.float64)
        left[0, 0], right[0, 0] = 0., 0.
        answers = []
        for device in ("cpu", "cuda"):
            x, y = left.to(device).requires_grad_(), right.to(device).requires_grad_()
            if device == "cuda":
                output = runtime.indexed_tagged_pair(x, y, tuple(value.cuda() for value in arrays), 2)
            else:
                terms = x[row_left] * y[row_right][:, arrays[-1]]
                output = terms.new_zeros(rows, 2).index_add(1, arrays[-2], terms)
            first = torch.autograd.grad(output.square().sum(), (x, y), create_graph=True)
            second = torch.autograd.grad(sum(value.square().sum() for value in first), (x, y))
            answers.append(tuple(value.detach().cpu() for value in (output, *first, *second)))
        for expected, actual in zip(*answers, strict=True):
            torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
        cache_sizes.append(tuple(len(kernel.device_caches[torch.cuda.current_device()][0]) for kernel in kernels))
    # Same schedule and dtype, including singleton, aligned and odd tails.
    assert all(counts == cache_sizes[0] for counts in cache_sizes[1:])
