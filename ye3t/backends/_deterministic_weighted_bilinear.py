"""Streamed weighted bilinear product-rule adjoints without event matrices.

For F(L,R,w), all three inputs remain distinct autograd operands even when
L and R alias. Each CSR destination/batch tile has one writer; weight
adjoints reduce only a small [batch tiles, weights] temporary. Coupling
indices, coefficients and segment orders are supplied by the existing plan.
"""

import torch
from ._deterministic_polynomial import triton, tl


def _apply_kernel(L, R, W, C, LI, RI, WI, P, ORDER, Y, B, LW, RW, OW, BLOCK, TERMS, DOUBLE):
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
        wi = tl.load(WI + event, active, 0)
        mask = (row < B)[:, None] & active[None, :]
        weight = tl.load(C + event, active, 0) * tl.load(W + wi, active, 0)
        value += weight[None, :] * tl.load(L + row[:, None] * LW + left[None, :], mask, 0) * tl.load(R + row[:, None] * RW + right[None, :], mask, 0)
        cursor += TERMS
    tl.store(Y + row * OW + out, tl.sum(value, 1), row < B)


def _weight_kernel(L, R, G, C, LI, RI, OI, P, ORDER, Y, B, LW, RW, OW, NW, BLOCK, TERMS, DOUBLE):
    out = tl.program_id(0)
    tile = tl.program_id(1)
    row = tile * BLOCK + tl.arange(0, BLOCK)
    cursor = tl.load(P + out) + tl.arange(0, TERMS)
    stop = tl.load(P + out + 1)
    value = tl.full((BLOCK, TERMS), 0, tl.float64 if DOUBLE else tl.float32)
    while tl.min(cursor, 0) < stop:
        active = cursor < stop
        event = tl.load(ORDER + cursor, active, 0)
        left = tl.load(LI + event, active, 0)
        right = tl.load(RI + event, active, 0)
        oi = tl.load(OI + event, active, 0)
        mask = (row < B)[:, None] & active[None, :]
        value += tl.load(C + event, active, 0)[None, :] * tl.load(G + row[:, None] * OW + oi[None, :], mask, 0) * tl.load(L + row[:, None] * LW + left[None, :], mask, 0) * tl.load(R + row[:, None] * RW + right[None, :], mask, 0)
        cursor += TERMS
    tl.store(Y + tile * NW + out, tl.sum(tl.sum(value, 1), 0))


if triton is not None:
    _apply_kernel.__annotations__ = {name: tl.constexpr for name in ("LW", "RW", "OW", "BLOCK", "TERMS", "DOUBLE")}
    _apply_kernel = triton.jit(_apply_kernel, do_not_specialize=["B"])
    _weight_kernel.__annotations__ = {name: tl.constexpr for name in ("LW", "RW", "OW", "NW", "BLOCK", "TERMS", "DOUBLE")}
    _weight_kernel = triton.jit(_weight_kernel, do_not_specialize=["B"])


def _apply(left, right, weight, table, plans, sizes, mode):
    # mode = F(L,R), dL(G,R), dR(L,G).
    indices = (table.left_index, table.right_index, table.output_index)
    li, ri, out = ((0, 1, 2), (2, 1, 0), (0, 2, 1))[mode]
    plan = plans[out]
    result = left.new_empty((left.shape[0], sizes[out]))
    if result.numel():
        _apply_kernel[(result.shape[1], triton.cdiv(left.shape[0], 8))](
            left.contiguous(), right.contiguous(), weight.contiguous(), table.coefficient,
            indices[li], indices[ri], table.weight_index, plan.offsets, plan.order, result,
            left.shape[0], left.shape[1], right.shape[1], result.shape[1], 8, 32,
            left.dtype == torch.float64, num_warps=4, enable_fp_fusion=False)
    return result


def _weight(left, right, gradient, table, plans, sizes):
    tiles = triton.cdiv(left.shape[0], 32)
    partials = left.new_empty((tiles, sizes[3]))
    if partials.numel():
        _weight_kernel[(sizes[3], tiles)](left.contiguous(), right.contiguous(), gradient.contiguous(),
            table.coefficient, table.left_index, table.right_index, table.output_index,
            plans[3].offsets, plans[3].order, partials, left.shape[0], left.shape[1], right.shape[1],
            gradient.shape[1], sizes[3], 32, 32, left.dtype == torch.float64,
            num_warps=4, enable_fp_fusion=False)
    return partials.sum(0)


def _add(left, right):
    return right if left is None else left + right


class _WeightedAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left, right, weight, gradient, table, plans, sizes):
        ctx.save_for_backward(left, right, weight, gradient)
        ctx.table, ctx.plans, ctx.sizes = table, plans, sizes
        ctx.set_materialize_grads(False)
        return (_apply(gradient, right, weight, table, plans, sizes, 1),
                _apply(left, gradient, weight, table, plans, sizes, 2),
                _weight(left, right, gradient, table, plans, sizes))

    @staticmethod
    def backward(ctx, p, q, t):
        if torch.is_grad_enabled():
            raise RuntimeError("streamed weighted bilinear derivatives support orders through two")
        left, right, weight, gradient = ctx.saved_tensors
        table, plans, sizes = ctx.table, ctx.plans, ctx.sizes
        dl, dr, dw, dg = None, None, None, None
        if p is not None:
            dr = _apply(p, gradient, weight, table, plans, sizes, 2)
            dw = _weight(p, right, gradient, table, plans, sizes)
            dg = _apply(p, right, weight, table, plans, sizes, 0)
        if q is not None:
            dl = _apply(gradient, q, weight, table, plans, sizes, 1)
            dw = _add(dw, _weight(left, q, gradient, table, plans, sizes))
            dg = _add(dg, _apply(left, q, weight, table, plans, sizes, 0))
        if t is not None:
            dl = _add(dl, _apply(gradient, right, t, table, plans, sizes, 1))
            dr = _add(dr, _apply(left, gradient, t, table, plans, sizes, 2))
            dg = _add(dg, _apply(left, right, t, table, plans, sizes, 0))
        return dl, dr, dw, dg, None, None, None


class _Weighted(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left, right, weight, table, plans, sizes):
        ctx.save_for_backward(left, right, weight)
        ctx.table, ctx.plans, ctx.sizes = table, plans, sizes
        return _apply(left, right, weight, table, plans, sizes, 0)

    @staticmethod
    def backward(ctx, gradient):
        left, right, weight = ctx.saved_tensors
        dl, dr, dw = _WeightedAdjoint.apply(left, right, weight, gradient,
            ctx.table, ctx.plans, ctx.sizes)
        return dl, dr, dw, None, None, None


def deterministic_weighted_bilinear(left, right, weight, table, left_plan, right_plan, output_plan, weight_plan,
                                    *, indices_certified=False):
    """Execute a validated existing packed weighted plan on real CUDA inputs."""
    if triton is None or left.device.type != "cuda" or left.dtype not in {torch.float32, torch.float64}:
        raise RuntimeError("streamed weighted bilinear requires real CUDA f32/f64 and Triton")
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("weighted bilinear operands require matching batch dimensions")
    if weight.ndim != 1 or weight.numel() != table.weight_count:
        raise ValueError("weighted bilinear weight shape does not match the table")
    if any(value.dtype != left.dtype or value.device != left.device for value in (right, weight, table.coefficient)):
        raise ValueError("weighted bilinear values and coefficients must match dtype and device")
    sizes = (left.shape[1], right.shape[1], table.output_width, weight.numel())
    plans = (left_plan, right_plan, output_plan, weight_plan)
    indices = (table.left_index, table.right_index, table.output_index, table.weight_index)
    terms = table.coefficient.numel()
    if min(sizes) < 0 or table.coefficient.ndim != 1 or not table.coefficient.is_contiguous():
        raise ValueError("weighted bilinear requires nonnegative widths and contiguous coefficient events")
    if table.coefficient.requires_grad:
        raise ValueError("weighted bilinear coupling coefficients must be fixed")
    for index, plan, width in zip(indices, plans, sizes):
        if index.ndim != 1 or index.numel() != terms or index.dtype != torch.long or index.device != left.device or not index.is_contiguous():
            raise ValueError("weighted bilinear index events must be contiguous int64 on the input device")
        if plan.segment_count != width or plan.offsets.numel() != width + 1 or plan.order.numel() != terms:
            raise ValueError("weighted bilinear segment plan dimensions do not match the table")
        if any(value.ndim != 1 or value.dtype != torch.long or value.device != left.device or not value.is_contiguous()
               for value in (plan.offsets, plan.order)):
            raise ValueError("weighted bilinear segment arrays must be contiguous int64 on the input device")
        if not indices_certified:
            # Validate once per immutable compiled binding, rather than synchronizing
            # the device for every support chunk. Stable ordering is part of replay.
            index_cpu = index.detach().cpu()
            if terms and (int(index_cpu.min()) < 0 or int(index_cpu.max()) >= width):
                raise ValueError("weighted bilinear table index is out of range")
            order = torch.argsort(index_cpu, stable=True)
            offsets = torch.cat((torch.zeros(1, dtype=torch.long), torch.bincount(index_cpu, minlength=width).cumsum(0)))
            if not torch.equal(plan.order.cpu(), order) or not torch.equal(plan.offsets.cpu(), offsets):
                raise ValueError("weighted bilinear segment plan is not bound to the table")
    return _Weighted.apply(left, right, weight, table, plans, sizes)
