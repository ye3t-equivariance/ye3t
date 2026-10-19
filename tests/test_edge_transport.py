"""Fused edge transport: values and derivatives through order two against PyTorch."""

import pytest
import torch

edge_transport = pytest.importorskip("ye3t.backends.edge_transport")


def _case(seed, dtype, edges=37, roots=11, rows=9, width=5, coordinates=7):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(roots, width, width, dtype=dtype, generator=generator)
    packed = torch.randn(rows, width, coordinates, dtype=dtype, generator=generator)
    envelope = torch.randn(edges, dtype=dtype, generator=generator)
    root_edges = torch.randint(0, roots, (edges,), generator=generator)
    sender = torch.randint(0, rows, (edges,), generator=generator)
    receiver = torch.randint(0, rows, (edges,), generator=generator)
    return [value.cuda() for value in (matrix, packed, envelope, root_edges, sender, receiver)]


def _reference(matrix, packed, envelope, root_edges, sender, receiver, rows):
    messages = torch.bmm(matrix.index_select(0, root_edges), packed.index_select(0, sender))
    base = packed.new_zeros(rows, packed.shape[1], packed.shape[2])
    return base.index_add(0, receiver, envelope[:, None, None] * messages)


def _requires_cuda_triton():
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    if edge_transport.triton is None:
        pytest.skip("requires optional dependency not installed: triton")


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_fused_transport_matches_reference_values(dtype):
    _requires_cuda_triton()
    matrix, packed, envelope, root_edges, sender, receiver = _case(0, dtype)
    report = {}
    fused = edge_transport.transport_messages(matrix, packed, envelope, root_edges, sender, receiver, packed.shape[0],
                                              execution_report=report)
    expected = _reference(matrix, packed, envelope, root_edges, sender, receiver, packed.shape[0])
    tolerance = 1e-12 if dtype == torch.float64 else 2e-5
    assert torch.allclose(fused, expected, rtol=tolerance, atol=tolerance)
    assert report == {"triton_fused_transport": 1}


def test_fused_transport_is_bitwise_repeatable_and_table_cache_neutral():
    _requires_cuda_triton()
    matrix, packed, envelope, root_edges, sender, receiver = _case(6, torch.float32, edges=500, roots=40, rows=50)
    inputs = [value.requires_grad_(True) for value in (matrix, packed, envelope)]
    tables = edge_transport.transport_tables(root_edges, sender, receiver, packed.shape[0], matrix.shape[0])
    results = []
    for cached in (None, tables, tables):
        out = edge_transport.transport_messages(*inputs, root_edges, sender, receiver, packed.shape[0], tables=cached)
        first = torch.autograd.grad((out * out).sum(), inputs, create_graph=True)
        second = torch.autograd.grad(sum((value * value).sum() for value in first), inputs)
        results.append((out.detach(), *(value.detach() for value in first), *second))
    for repeat in results[1:]:
        for left, right in zip(results[0], repeat, strict=True):
            assert torch.equal(left, right)


def test_fused_transport_empty_message_set_keeps_zero_output_and_gradients():
    _requires_cuda_triton()
    matrix, packed, envelope, root_edges, sender, receiver = _case(1, torch.float64)
    empty = root_edges[:0]
    matrix.requires_grad_(True)
    packed.requires_grad_(True)
    fused = edge_transport.transport_messages(matrix, packed, envelope[:0], empty, empty, empty, packed.shape[0])
    assert fused.shape == packed.shape and not fused.any()
    grads = torch.autograd.grad(fused.sum(), (matrix, packed))
    assert all(not value.any() for value in grads)


def test_fused_transport_first_and_second_order_against_reference_float64():
    _requires_cuda_triton()
    matrix, packed, envelope, root_edges, sender, receiver = _case(2, torch.float64)
    rows = packed.shape[0]
    weights = torch.randn(rows, packed.shape[1], packed.shape[2], dtype=torch.float64, device="cuda")

    def loss_and_first(function):
        inputs = [value.detach().clone().requires_grad_(True) for value in (matrix, packed, envelope)]
        out = function(*inputs, root_edges, sender, receiver, rows)
        first = torch.autograd.grad((out * weights).sum(), inputs, create_graph=True)
        # A nonlinear scalar of the first derivatives exercises every second-order term.
        scalar = sum((value.square() * torch.arange(1, value.numel() + 1, dtype=value.dtype, device=value.device)
                      .reshape(value.shape)).sum() for value in first)
        second = torch.autograd.grad(scalar, inputs)
        return out.detach(), tuple(value.detach() for value in first), tuple(value.detach() for value in second)

    fused_out, fused_first, fused_second = loss_and_first(
        lambda m, p, e, r, s, t, n: edge_transport.transport_messages(m, p, e, r, s, t, n))
    reference_out, reference_first, reference_second = loss_and_first(_reference)
    assert torch.allclose(fused_out, reference_out, rtol=1e-11, atol=1e-11)
    for fused_value, reference_value in zip(fused_first + fused_second, reference_first + reference_second, strict=True):
        assert torch.allclose(fused_value, reference_value, rtol=1e-9, atol=1e-10)


def test_fused_transport_gradcheck_and_gradgradcheck():
    _requires_cuda_triton()
    matrix, packed, envelope, root_edges, sender, receiver = _case(3, torch.float64, edges=13, roots=4, rows=5, width=3, coordinates=4)
    inputs = tuple(value.requires_grad_(True) for value in (matrix, packed, envelope))

    def function(m, p, e):
        return edge_transport.transport_messages(m, p, e, root_edges, sender, receiver, packed.shape[0])

    assert torch.autograd.gradcheck(function, inputs, eps=1e-6, atol=1e-7, rtol=1e-6)
    assert torch.autograd.gradgradcheck(function, inputs, eps=1e-6, atol=1e-7, rtol=1e-6)


def test_fused_transport_rejects_graphs_through_the_second_order_backward():
    """Second order is evaluated; building a further graph through it raises (small-matrix convention)."""
    _requires_cuda_triton()
    matrix, packed, envelope, root_edges, sender, receiver = _case(4, torch.float64)
    inputs = [value.requires_grad_(True) for value in (matrix, packed, envelope)]
    out = edge_transport.transport_messages(*inputs, root_edges, sender, receiver, packed.shape[0])
    first = torch.autograd.grad(out.square().sum(), inputs, create_graph=True)
    second = torch.autograd.grad(sum(value.square().sum() for value in first), inputs)
    assert all(torch.isfinite(value).all() for value in second)
    out = edge_transport.transport_messages(*inputs, root_edges, sender, receiver, packed.shape[0])
    first = torch.autograd.grad(out.square().sum(), inputs, create_graph=True)
    with pytest.raises(RuntimeError, match="order two"):
        torch.autograd.grad(sum(value.square().sum() for value in first), inputs, create_graph=True)


def test_fused_transport_requires_supported_operands():
    _requires_cuda_triton()
    matrix, packed, envelope, root_edges, sender, receiver = _case(5, torch.float64)
    with pytest.raises(RuntimeError, match="fused edge transport"):
        edge_transport.transport_messages(matrix.cpu(), packed.cpu(), envelope.cpu(), root_edges.cpu(), sender.cpu(),
                                          receiver.cpu(), packed.shape[0])
    with pytest.raises(ValueError, match="one entry per message"):
        edge_transport.transport_messages(matrix, packed, envelope[:-1], root_edges, sender, receiver, packed.shape[0])
    wrong = edge_transport.transport_tables(root_edges, sender, receiver, packed.shape[0] + 1, matrix.shape[0])
    with pytest.raises(ValueError, match="transport tables"):
        edge_transport.transport_messages(matrix, packed, envelope, root_edges, sender, receiver, packed.shape[0], tables=wrong)
