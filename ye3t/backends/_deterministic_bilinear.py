"""Fixed event-order real bilinear contraction and its two adjoints."""

import torch
from ._deterministic_polynomial import _csr, triton, tl


def bilinear_plan(left, right, output, left_width, right_width, output_width):
    indices = tuple(torch.as_tensor(value, dtype=torch.long, device="cpu") for value in (left, right, output))
    if any(value.ndim != 1 or len(value) != len(indices[0]) for value in indices):
        raise ValueError("bilinear event indices must have matching lengths")
    sizes = (int(left_width), int(right_width), int(output_width))
    if min(sizes) < 1:
        raise ValueError("bilinear widths must be positive")
    for value, width in zip(indices, sizes, strict=True):
        if torch.any((value < 0) | (value >= width)):
            raise ValueError("bilinear event index outside declared width")
    return indices + sum((_csr(value, width) for value, width in zip(indices, sizes, strict=True)), ()), sizes


def _kernel(L, R, C, LI, RI, P, ORDER, Y, B, LW, RW, OW, BLOCK, TERMS, DOUBLE):
    out = tl.program_id(0)
    row = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    cursor = tl.load(P + out) + tl.arange(0, TERMS)
    stop = tl.load(P + out + 1)
    value = tl.full((BLOCK, TERMS), 0, tl.float64 if DOUBLE else tl.float32)
    while tl.min(cursor, 0) < stop:
        active = cursor < stop
        event = tl.load(ORDER + cursor, active, 0)
        left = tl.load(LI + event, active, 0)
        right = tl.load(RI + event, active, 0)
        mask = (row < B)[:, None] & active[None, :]
        value += tl.load(C + event, active, 0)[None, :] * tl.load(L + row[:, None] * LW + left[None, :], mask, 0) * tl.load(R + row[:, None] * RW + right[None, :], mask, 0)
        cursor += TERMS
    tl.store(Y + row * OW + out, tl.sum(value, 1), row < B)


if triton is not None:
    _kernel.__annotations__ = {name: tl.constexpr for name in ("LW", "RW", "OW", "BLOCK", "TERMS", "DOUBLE")}
    _kernel = triton.jit(_kernel, do_not_specialize=["B"])


def _evaluate(left, right, coefficients, tables, sizes, mode):
    # mode = forward, left adjoint (G,R), right adjoint (L,G).
    l_index, r_index, destination = ((0, 1, 2), (2, 1, 0), (0, 2, 1))[mode]
    pointers, order = tables[3 + 2 * destination:5 + 2 * destination]
    result = left.new_empty((left.shape[0], sizes[destination]))
    if result.numel():
        _kernel[(result.shape[1], triton.cdiv(left.shape[0], 8))](left.contiguous(), right.contiguous(), coefficients,
            tables[l_index], tables[r_index], pointers, order, result, left.shape[0], left.shape[1], right.shape[1], result.shape[1],
            8, 32, left.dtype == torch.float64, num_warps=4, enable_fp_fusion=False)
    return result


class _Adjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left, right, gradient, coefficients, tables, sizes):
        ctx.save_for_backward(left, right, gradient, coefficients)
        ctx.tables, ctx.sizes = tables, sizes
        ctx.set_materialize_grads(False)
        return (_evaluate(gradient, right, coefficients, tables, sizes, 1),
                _evaluate(left, gradient, coefficients, tables, sizes, 2))

    @staticmethod
    def backward(ctx, left_tangent, right_tangent):
        if torch.is_grad_enabled():
            raise RuntimeError("deterministic bilinear derivatives support orders through two")
        left, right, gradient, coefficients = ctx.saved_tensors
        dl, dr, dg = None, None, None
        if left_tangent is not None:
            dr = _evaluate(left_tangent, gradient, coefficients, ctx.tables, ctx.sizes, 2)
            dg = _evaluate(left_tangent, right, coefficients, ctx.tables, ctx.sizes, 0)
        if right_tangent is not None:
            dl = _evaluate(gradient, right_tangent, coefficients, ctx.tables, ctx.sizes, 1)
            local = _evaluate(left, right_tangent, coefficients, ctx.tables, ctx.sizes, 0)
            dg = local if dg is None else dg + local
        return dl, dr, dg, None, None, None


class _Bilinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left, right, coefficients, tables, sizes):
        ctx.save_for_backward(left, right, coefficients)
        ctx.tables, ctx.sizes = tables, sizes
        return _evaluate(left, right, coefficients, tables, sizes, 0)

    @staticmethod
    def backward(ctx, gradient):
        left, right, coefficients = ctx.saved_tensors
        dl, dr = _Adjoint.apply(left, right, gradient.contiguous(), coefficients, ctx.tables, ctx.sizes)
        return dl, dr, None, None, None


def deterministic_bilinear(left, right, coefficients, tables, sizes):
    if triton is None or left.device.type != "cuda" or left.dtype not in {torch.float32, torch.float64}:
        raise RuntimeError("deterministic bilinear requires Triton and real CUDA f32/f64")
    if left.shape != (right.shape[0], sizes[0]) or right.shape[1] != sizes[1]:
        raise ValueError("bilinear inputs do not match the numeric schedule")
    if coefficients.requires_grad or coefficients.numel() != len(tables[0]):
        raise ValueError("bilinear requires fixed compiler coefficients")
    if right.dtype != left.dtype or right.device != left.device or coefficients.dtype != left.dtype or coefficients.device != left.device or any(
            value.dtype != torch.long or value.device != left.device for value in tables):
        raise ValueError("bilinear operands/coefficients/tables must share dtype and device")
    return _Bilinear.apply(left, right, coefficients, tables, sizes)
