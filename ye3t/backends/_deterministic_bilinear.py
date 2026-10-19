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
    tables = list(indices)
    # Per destination axis: CSR pointers, the stable event order, and the three
    # index arrays already permuted into that order, so a program reads its
    # events contiguously instead of through the order indirection. The event
    # sequence of every destination column is unchanged.
    for value, width in zip(indices, sizes, strict=True):
        pointers, order = _csr(value, width)
        tables.extend((pointers, order, *(index[order] for index in indices)))
    return tuple(tables), sizes


def _kernel(L, R, C, LI, RI, P, Y, B, LW, RW, OW, BLOCK, TERMS, DOUBLE):
    out = tl.program_id(0)
    row = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    cursor = tl.load(P + out) + tl.arange(0, TERMS)
    stop = tl.load(P + out + 1)
    value = tl.full((BLOCK, TERMS), 0, tl.float64 if DOUBLE else tl.float32)
    while tl.min(cursor, 0) < stop:
        active = cursor < stop
        left = tl.load(LI + cursor, active, 0)
        right = tl.load(RI + cursor, active, 0)
        mask = (row < B)[:, None] & active[None, :]
        value += tl.load(C + cursor, active, 0)[None, :] * tl.load(L + row[:, None] * LW + left[None, :], mask, 0) * tl.load(R + row[:, None] * RW + right[None, :], mask, 0)
        cursor += TERMS
    tl.store(Y + row * OW + out, tl.sum(value, 1), row < B)


if triton is not None:
    _kernel.__annotations__ = {name: tl.constexpr for name in ("LW", "RW", "OW", "BLOCK", "TERMS", "DOUBLE")}
    _kernel = triton.jit(_kernel, do_not_specialize=["B"])


def sorted_coefficients(coefficients, tables):
    """One coefficient vector per destination axis, in that axis's event order."""
    if isinstance(coefficients, (tuple, list)):
        if len(coefficients) != 3:
            raise ValueError("bilinear requires one sorted coefficient vector per destination axis")
        return tuple(coefficients)
    return tuple(coefficients[tables[4 + 5 * axis]] for axis in range(3))


def _evaluate(left, right, coefficients, tables, sizes, mode, launches=None):
    # mode = forward, left adjoint (G,R), right adjoint (L,G).
    l_index, r_index, destination = ((0, 1, 2), (2, 1, 0), (0, 2, 1))[mode]
    pointers = tables[3 + 5 * destination]
    ordered = tables[5 + 5 * destination:8 + 5 * destination]
    result = left.new_empty((left.shape[0], sizes[destination]))
    if result.numel():
        block, terms = launches[mode] if launches is not None and left.shape[0] >= 128 else (8, 32)
        _kernel[(result.shape[1], triton.cdiv(left.shape[0], block))](left.contiguous(), right.contiguous(),
            coefficients[destination], ordered[l_index], ordered[r_index], pointers, result,
            left.shape[0], left.shape[1], right.shape[1], result.shape[1],
            block, terms, left.dtype == torch.float64, num_warps=4, enable_fp_fusion=False)
    return result


class _Adjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left, right, gradient, coefficients, tables, sizes, launches):
        ctx.save_for_backward(left, right, gradient, *coefficients)
        ctx.tables, ctx.sizes = tables, sizes
        ctx.launches = launches
        ctx.set_materialize_grads(False)
        return (_evaluate(gradient, right, coefficients, tables, sizes, 1, launches),
                _evaluate(left, gradient, coefficients, tables, sizes, 2, launches))

    @staticmethod
    def backward(ctx, left_tangent, right_tangent):
        if torch.is_grad_enabled():
            raise RuntimeError("deterministic bilinear derivatives support orders through two")
        left, right, gradient, *coefficients = ctx.saved_tensors
        coefficients = tuple(coefficients)
        dl, dr, dg = None, None, None
        if left_tangent is not None:
            dr = _evaluate(left_tangent, gradient, coefficients, ctx.tables, ctx.sizes, 2, ctx.launches)
            dg = _evaluate(left_tangent, right, coefficients, ctx.tables, ctx.sizes, 0, ctx.launches)
        if right_tangent is not None:
            dl = _evaluate(gradient, right_tangent, coefficients, ctx.tables, ctx.sizes, 1, ctx.launches)
            local = _evaluate(left, right_tangent, coefficients, ctx.tables, ctx.sizes, 0, ctx.launches)
            dg = local if dg is None else dg + local
        return dl, dr, dg, None, None, None, None


class _Bilinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left, right, coefficients, tables, sizes, launches):
        ctx.save_for_backward(left, right, *coefficients)
        ctx.tables, ctx.sizes = tables, sizes
        ctx.launches = launches
        return _evaluate(left, right, coefficients, tables, sizes, 0, launches)

    @staticmethod
    def backward(ctx, gradient):
        left, right, *coefficients = ctx.saved_tensors
        dl, dr = _Adjoint.apply(left, right, gradient.contiguous(), tuple(coefficients), ctx.tables, ctx.sizes, ctx.launches)
        return dl, dr, None, None, None, None


def deterministic_bilinear(left, right, coefficients, tables, sizes, *, launches=None):
    """Fixed-order sparse bilinear contraction; ``coefficients`` is one vector or three sorted vectors."""
    if triton is None or left.device.type != "cuda" or left.dtype not in {torch.float32, torch.float64}:
        raise RuntimeError("deterministic bilinear requires Triton and real CUDA f32/f64")
    if left.shape != (right.shape[0], sizes[0]) or right.shape[1] != sizes[1]:
        raise ValueError("bilinear inputs do not match the numeric schedule")
    supplied = tuple(coefficients) if isinstance(coefficients, (tuple, list)) else (coefficients,)
    if any(value.requires_grad or value.numel() != len(tables[0]) for value in supplied):
        raise ValueError("bilinear requires fixed compiler coefficients")
    if right.dtype != left.dtype or right.device != left.device or any(
            value.dtype != left.dtype or value.device != left.device for value in supplied) or any(
            value.dtype != torch.long or value.device != left.device for value in tables):
        raise ValueError("bilinear operands/coefficients/tables must share dtype and device")
    if len(tables) != 18:
        raise ValueError("bilinear tables must come from bilinear_plan")
    return _Bilinear.apply(left, right, sorted_coefficients(coefficients, tables), tables, sizes, launches)
