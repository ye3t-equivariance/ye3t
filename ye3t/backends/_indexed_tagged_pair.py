"""Numeric indexed contraction for compiler-certified unit pair schedules.

Each left coordinate a occurs once, with compiler-owned destinations o(a)
and right coordinate b(a): Y[p,o] = sum_(a:o(a)=o) H[i(p),a] U[j(p),b(a)].
The left adjoint reduces over the occurrence incidence CSR without atomics.
The small right adjoint is reduced over coordinates before incidence scatter.
No geometry, species, symmetry discovery, or coefficient synthesis lives here.
"""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def _forward_kernel(H, U, I, J, OP, OA, AB, Y, B, LH, RU, O, BLOCK, DOUBLE):
    flat = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    p, o = flat // O, flat % O
    mask = p < B
    i = tl.load(I + p, mask, 0)
    j = tl.load(J + p, mask, 0)
    start = tl.load(OP + o, mask, 0)
    stop = tl.load(OP + o + 1, mask, 0)
    value = tl.full((BLOCK,), 0, tl.float64 if DOUBLE else tl.float32)
    while tl.sum((start < stop).to(tl.int32), 0) > 0:
        active = mask & (start < stop)
        a = tl.load(OA + start, active, 0)
        b = tl.load(AB + a, active, 0)
        value += tl.load(H + i * LH + a, active, 0) * tl.load(U + j * RU + b, active, 0)
        start += 1
    tl.store(Y + flat, value, mask)


def _left_adjoint_kernel(G, U, J, IP, PI, AO, AB, DH, E, LH, RU, O, BLOCK, DOUBLE):
    flat = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    i, a = flat // LH, flat % LH
    mask = i < E
    start = tl.load(IP + i, mask, 0)
    stop = tl.load(IP + i + 1, mask, 0)
    o = tl.load(AO + a, mask, 0)
    b = tl.load(AB + a, mask, 0)
    value = tl.full((BLOCK,), 0, tl.float64 if DOUBLE else tl.float32)
    while tl.sum((start < stop).to(tl.int32), 0) > 0:
        active = mask & (start < stop)
        p = tl.load(PI + start, active, 0)
        j = tl.load(J + p, active, 0)
        value += tl.load(G + p * O + o, active, 0) * tl.load(U + j * RU + b, active, 0)
        start += 1
    tl.store(DH + flat, value, mask)


def _right_adjoint_kernel(G, H, I, BP, BA, AO, DU, B, LH, RU, O, BLOCK, TERMS, DOUBLE):
    flat = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    p, b = flat // RU, flat % RU
    mask = p < B
    i = tl.load(I + p, mask, 0)
    start = tl.load(BP + b, mask, 0)
    stop = tl.load(BP + b + 1, mask, 0)
    term = start[:, None] + tl.arange(0, TERMS)[None, :]
    value = tl.full((BLOCK, TERMS), 0, tl.float64 if DOUBLE else tl.float32)
    while tl.sum(tl.sum((term < stop[:, None]).to(tl.int32), 1), 0) > 0:
        active = mask[:, None] & (term < stop[:, None])
        a = tl.load(BA + term, active, 0)
        o = tl.load(AO + a, active, 0)
        value += tl.load(G + p[:, None] * O + o, active, 0) * tl.load(H + i[:, None] * LH + a, active, 0)
        term += TERMS
    tl.store(DU + flat, tl.sum(value, 1), mask)


if triton is not None:
    # Triton compile-time metadata; no Python typing dependency is introduced.
    for kernel, constants in (
        (_forward_kernel, ("LH", "RU", "O", "BLOCK", "DOUBLE")),
        (_left_adjoint_kernel, ("LH", "RU", "O", "BLOCK", "DOUBLE")),
        (_right_adjoint_kernel, ("LH", "RU", "O", "BLOCK", "TERMS", "DOUBLE")),
    ):
        kernel.__annotations__ = {name: tl.constexpr for name in constants}
    # Row counts only bound masks. Periodic coordination and chunk tails vary
    # between structures; they must not trigger a fresh kernel compilation.
    _forward_kernel = triton.jit(_forward_kernel, do_not_specialize=["B"])
    _left_adjoint_kernel = triton.jit(_left_adjoint_kernel, do_not_specialize=["E"])
    _right_adjoint_kernel = triton.jit(_right_adjoint_kernel, do_not_specialize=["B"])


def _forward(left, right, tables, output_width):
    rows_left, rows_right, left_ptr, left_order, output_ptr, output_left, right_ptr, right_left, left_output, left_right = tables
    output = left.new_empty((rows_left.numel(), output_width))
    if output.numel():
        _forward_kernel[(triton.cdiv(output.numel(), 128),)](
            left, right, rows_left, rows_right, output_ptr, output_left, left_right, output,
            rows_left.numel(), left.shape[1], right.shape[1], output_width, 128, left.dtype == torch.float64,
            num_warps=4, enable_fp_fusion=False)
    return output


def _adjoint(gradient, left, right, tables, output_width):
    rows_left, rows_right, left_ptr, left_order, output_ptr, output_left, right_ptr, right_left, left_output, left_right = tables
    gradient = gradient.contiguous()
    dleft = torch.empty_like(left)
    partial_right = right.new_empty((rows_right.numel(), right.shape[1]))
    if dleft.numel():
        _left_adjoint_kernel[(triton.cdiv(dleft.numel(), 128),)](
            gradient, right, rows_right, left_ptr, left_order, left_output, left_right, dleft,
            left.shape[0], left.shape[1], right.shape[1], output_width, 128, left.dtype == torch.float64,
            num_warps=4, enable_fp_fusion=False)
    if partial_right.numel():
        _right_adjoint_kernel[(triton.cdiv(partial_right.numel(), 8),)](
            gradient, left, rows_left, right_ptr, right_left, left_output, partial_right,
            rows_left.numel(), left.shape[1], right.shape[1], output_width, 8, 128, left.dtype == torch.float64,
            num_warps=4, enable_fp_fusion=False)
    dright = torch.zeros_like(right).index_add_(0, rows_right, partial_right)
    return dleft, dright


class _IndexedPairAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gradient, left, right, *arguments):
        *tables, output_width = arguments
        ctx.save_for_backward(gradient, left, right, *tables)
        ctx.output_width = output_width
        ctx.set_materialize_grads(False)
        return _adjoint(gradient, left, right, tables, output_width)

    @staticmethod
    def backward(ctx, vleft, vright):
        if torch.is_grad_enabled():
            raise RuntimeError("indexed pair execution supports derivatives through order two")
        gradient, left, right, *tables = ctx.saved_tensors
        vleft = torch.zeros_like(left) if vleft is None else vleft.contiguous()
        vright = torch.zeros_like(right) if vright is None else vright.contiguous()
        dgradient = _forward(vleft, right, tables, ctx.output_width) + _forward(left, vright, tables, ctx.output_width)
        dleft, dright = _adjoint(gradient, vleft, vright, tables, ctx.output_width)
        return (dgradient, dleft, dright, *((None,) * (len(tables) + 1)))


class _IndexedPair(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left, right, *arguments):
        *tables, output_width = arguments
        ctx.save_for_backward(left, right, *tables)
        ctx.output_width = output_width
        return _forward(left, right, tables, output_width)

    @staticmethod
    def backward(ctx, gradient):
        left, right, *tables = ctx.saved_tensors
        if torch.is_grad_enabled():
            dleft, dright = _IndexedPairAdjoint.apply(gradient, left, right, *tables, ctx.output_width)
        else:
            dleft, dright = _adjoint(gradient, left, right, tables, ctx.output_width)
        return (dleft, dright, *((None,) * (len(tables) + 1)))


def indexed_tagged_pair(left, right, tables, output_width):
    """Strict CUDA execution of the certified unit pair contraction."""
    if triton is None or not left.is_cuda or right.device != left.device:
        raise RuntimeError("indexed tagged pair execution requires CUDA and Triton")
    if left.dtype not in (torch.float32, torch.float64) or right.dtype != left.dtype:
        raise ValueError("indexed tagged pair execution requires one real f32/f64 dtype")
    if len(tables) != 10 or any(value.device != left.device or value.dtype != torch.long for value in tables):
        raise ValueError("indexed pair tables must be ten same-device int64 compiler/incidence arrays")
    rows_left, rows_right, left_ptr, left_order, output_ptr, output_left, right_ptr, right_left, left_output, left_right = tables
    if (left.ndim != 2 or right.ndim != 2 or any(value.ndim != 1 or not value.is_contiguous() for value in tables)
            or rows_right.numel() != rows_left.numel() or left_order.numel() != rows_left.numel()
            or left_ptr.numel() != left.shape[0] + 1 or output_ptr.numel() != output_width + 1
            or right_ptr.numel() != right.shape[1] + 1
            or any(value.numel() != left.shape[1] for value in (output_left, right_left, left_output, left_right))):
        raise ValueError("indexed pair table shapes differ from the certified coordinate/incidence contract")
    return _IndexedPair.apply(left.contiguous(), right.contiguous(), *tables, int(output_width))
