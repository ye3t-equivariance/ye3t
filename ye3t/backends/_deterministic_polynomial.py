"""Fixed-order real sparse polynomials through second derivatives.

Numeric CSR schedules preserve the supplied coefficient-event order. Each
destination has one writer. Lowered integer powers and a dual product rule
avoid division, including at zero. No symmetry labels are synthesized here.
Independent implementation of the polynomial product rule; custom autograd
contract: https://docs.pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html
"""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def _csr(indices, size):
    order = torch.argsort(indices, stable=True)
    return torch.cat((indices.new_zeros(1), torch.bincount(indices, minlength=size).cumsum(0))), order


def polynomial_plan(term_offsets, components, exponents, coefficient_terms, coefficient_outputs,
                    input_width, output_width):
    """Validate and derive numeric incidence, once on CPU, from a source plan."""
    offsets, components, exponents, terms, outputs = (
        torch.as_tensor(value, dtype=torch.long, device="cpu").contiguous()
        for value in (term_offsets, components, exponents, coefficient_terms, coefficient_outputs))
    if input_width < 1 or output_width < 1 or any(value.ndim != 1 for value in (offsets, components, exponents, terms, outputs)):
        raise ValueError("polynomial widths must be positive and all event arrays one-dimensional")
    if offsets.ndim != 1 or not len(offsets) or offsets[0] != 0 or offsets[-1] != len(components):
        raise ValueError("invalid polynomial support offsets")
    if torch.any(offsets[1:] < offsets[:-1]) or components.shape != exponents.shape:
        raise ValueError("invalid polynomial support lengths")
    if torch.any(exponents <= 0) or torch.any((components < 0) | (components >= input_width)):
        raise ValueError("invalid polynomial components or exponents")
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        local = components[start:stop]
        if local.unique().numel() != local.numel():
            raise ValueError("polynomial supports must have unique components per term")
    count = len(offsets) - 1
    if terms.shape != outputs.shape or torch.any((terms < 0) | (terms >= count)):
        raise ValueError("invalid polynomial coefficient terms")
    if torch.any((outputs < 0) | (outputs >= output_width)):
        raise ValueError("invalid polynomial coefficient outputs")
    event_terms = torch.repeat_interleave(torch.arange(count), offsets[1:] - offsets[:-1])
    component_ptr, component_order = _csr(components, input_width)
    forward_ptr, forward_order = _csr(outputs, output_width)
    adjoint_ptr, adjoint_order = _csr(terms, count)
    return (offsets, components, exponents, event_terms, component_ptr, component_order,
            forward_ptr, forward_order, adjoint_ptr, adjoint_order, terms, outputs), (
                int(input_width), int(output_width), count, int(exponents.max()) if len(exponents) else 0)


def _power(x, exponent, MAX_POWER):
    result = tl.full(x.shape, 1, x.dtype)
    for step in range(MAX_POWER):
        result *= tl.where(step < exponent, x, 1)
    return result


def _products(X, U, TP, C, A, ET, CP, CO, W, Y, B, D, M, MODE, POWER, BLOCK, DOUBLE):
    flat = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    width = M if MODE < 2 else D
    row, destination = flat // width, flat % width
    valid = row < B
    result = tl.full((BLOCK,), 0, tl.float64 if DOUBLE else tl.float32)
    if MODE < 2:
        event = tl.zeros((BLOCK,), tl.int32)
        event_stop = tl.full((BLOCK,), 1, tl.int32)
    else:
        event = tl.load(CP + destination, valid, 0)
        event_stop = tl.load(CP + destination + 1, valid, 0)
    while tl.sum((valid & (event < event_stop)).to(tl.int32), 0) > 0:
        active = valid & (event < event_stop)
        if MODE < 2:
            term = destination
            factor = tl.full((BLOCK,), 1, tl.int64)
        else:
            support = tl.load(CO + event, active, 0)
            term = tl.load(ET + support, active, 0)
            factor = tl.load(A + support, active, 0)
        start = tl.load(TP + term, active, 0)
        stop = tl.load(TP + term + 1, active, 0)
        product = tl.full((BLOCK,), 1, result.dtype)
        directional = tl.full((BLOCK,), 0, result.dtype)
        while tl.sum((active & (start < stop)).to(tl.int32), 0) > 0:
            present = active & (start < stop)
            component = tl.load(C + start, present, 0)
            exponent = tl.load(A + start, present, 0)
            if MODE >= 2:
                exponent -= (component == destination).to(exponent.dtype)
            exponent = tl.where(present, exponent, 0)
            value = tl.load(X + row * D + component, present, 0)
            power = _power(value, exponent, POWER)
            if MODE == 1 or MODE == 3:
                tangent = tl.load(U + row * D + component, present, 0)
                derivative = exponent * _power(value, tl.maximum(exponent - 1, 0), POWER) * tangent
                directional = directional * power + product * derivative
            product *= power
            start += 1
        if MODE == 1 or MODE == 3:
            product = directional
        if MODE >= 2:
            product *= factor * tl.load(W + row * M + term, active, 0)
        result += tl.where(active, product, 0)
        event += 1
    tl.store(Y + flat, result, valid)


def _linear(X, P, ORDER, INDEX, COEFF, Y, B, IN, OUT, BLOCK, TERMS, DOUBLE):
    destination = tl.program_id(0)
    row = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    cursor = tl.load(P + destination) + tl.arange(0, TERMS)
    stop = tl.load(P + destination + 1)
    value = tl.full((BLOCK, TERMS), 0, tl.float64 if DOUBLE else tl.float32)
    while tl.min(cursor, 0) < stop:
        active = cursor < stop
        event = tl.load(ORDER + cursor, active, 0)
        index = tl.load(INDEX + event, active, 0)
        value += tl.load(COEFF + event, active, 0)[None, :] * tl.load(
            X + row[:, None] * IN + index[None, :], (row < B)[:, None] & active[None, :], 0)
        cursor += TERMS
    tl.store(Y + row * OUT + destination, tl.sum(value, 1), row < B)


def _input_adjoint(X, U, TP, C, A, ET, CP, CO, W, Y, B, D, M, MODE, POWER, BLOCK, TERMS, DOUBLE):
    destination = tl.program_id(0)
    row = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    event = tl.load(CP + destination) + tl.arange(0, TERMS)
    event_stop = tl.load(CP + destination + 1)
    result = tl.full((BLOCK, TERMS), 0, tl.float64 if DOUBLE else tl.float32)
    while tl.min(event, 0) < event_stop:
        active = event < event_stop
        support = tl.load(CO + event, active, 0)
        term = tl.load(ET + support, active, 0)
        factor = tl.load(A + support, active, 0)
        start = tl.load(TP + term, active, 0)
        stop = tl.load(TP + term + 1, active, 0)
        product = tl.full((BLOCK, TERMS), 1, result.dtype)
        directional = tl.full((BLOCK, TERMS), 0, result.dtype)
        while tl.sum((active & (start < stop)).to(tl.int32), 0) > 0:
            present = active & (start < stop)
            component = tl.load(C + start, present, 0)
            exponent = tl.load(A + start, present, 0)
            exponent = tl.where(present, exponent - (component == destination).to(exponent.dtype), 0)
            mask = (row < B)[:, None] & present[None, :]
            value = tl.load(X + row[:, None] * D + component[None, :], mask, 0)
            power = _power(value, exponent[None, :], POWER)
            if MODE == 3:
                tangent = tl.load(U + row[:, None] * D + component[None, :], mask, 0)
                derivative = exponent[None, :] * _power(value, tl.maximum(exponent[None, :] - 1, 0), POWER) * tangent
                directional = directional * power + product * derivative
            product *= power
            start += 1
        if MODE == 3:
            product = directional
        weight = tl.load(W + row[:, None] * M + term[None, :], (row < B)[:, None] & active[None, :], 0)
        result += factor[None, :] * product * weight
        event += TERMS
    tl.store(Y + row * D + destination, tl.sum(result, 1), row < B)


if triton is not None:
    _power.__annotations__ = {"MAX_POWER": tl.constexpr}
    _power = triton.jit(_power)
    _products.__annotations__ = {name: tl.constexpr for name in ("D", "M", "MODE", "POWER", "BLOCK", "DOUBLE")}
    _products = triton.jit(_products, do_not_specialize=["B"])
    _linear.__annotations__ = {name: tl.constexpr for name in ("IN", "OUT", "BLOCK", "TERMS", "DOUBLE")}
    _linear = triton.jit(_linear, do_not_specialize=["B"])
    _input_adjoint.__annotations__ = {name: tl.constexpr for name in ("D", "M", "MODE", "POWER", "BLOCK", "TERMS", "DOUBLE")}
    _input_adjoint = triton.jit(_input_adjoint, do_not_specialize=["B"])


def _product(input, tangent, weight, tables, sizes, mode):
    width, _, terms, power = sizes
    result = input.new_empty((input.shape[0], terms if mode < 2 else width))
    if result.numel():
        if mode < 2:
            _products[(triton.cdiv(result.numel(), 128),)](input, tangent, *tables[:6], weight, result,
                input.shape[0], width, terms, mode, power, 128, input.dtype == torch.float64,
                num_warps=4, enable_fp_fusion=False)
        else:
            _input_adjoint[(width, triton.cdiv(input.shape[0], 8))](input, tangent, *tables[:6], weight, result,
                input.shape[0], width, terms, mode, power, 8, 32, input.dtype == torch.float64,
                num_warps=4, enable_fp_fusion=False)
    return result


def _map(values, coefficients, tables, sizes, transpose=False):
    _, outputs, terms, _ = sizes
    pointers, order = tables[8:10] if transpose else tables[6:8]
    indices = tables[11] if transpose else tables[10]
    result = values.new_empty((values.shape[0], terms if transpose else outputs))
    if result.numel():
        _linear[(result.shape[1], triton.cdiv(values.shape[0], 8))](values, pointers, order, indices, coefficients, result,
            values.shape[0], values.shape[1], result.shape[1], 8, 32, values.dtype == torch.float64,
            num_warps=4, enable_fp_fusion=False)
    return result


class _PolynomialAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gradient, coefficients, tables, sizes):
        ctx.save_for_backward(input, gradient, coefficients)
        ctx.tables, ctx.sizes = tables, sizes
        weights = _map(gradient, coefficients, tables, sizes, True)
        return _product(input, input, weights, tables, sizes, 2)

    @staticmethod
    def backward(ctx, tangent):
        if torch.is_grad_enabled():
            raise RuntimeError("deterministic polynomial derivatives support orders through two")
        input, gradient, coefficients = ctx.saved_tensors
        tangent = tangent.contiguous()
        weights = _map(gradient, coefficients, ctx.tables, ctx.sizes, True)
        hessian = _product(input, tangent, weights, ctx.tables, ctx.sizes, 3)
        directional = _product(input, tangent, weights, ctx.tables, ctx.sizes, 1)
        return hessian, _map(directional, coefficients, ctx.tables, ctx.sizes), None, None, None


class _Polynomial(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, coefficients, tables, sizes):
        ctx.save_for_backward(input, coefficients)
        ctx.tables, ctx.sizes = tables, sizes
        values = _product(input, input, input, tables, sizes, 0)
        return _map(values, coefficients, tables, sizes)

    @staticmethod
    def backward(ctx, gradient):
        input, coefficients = ctx.saved_tensors
        return _PolynomialAdjoint.apply(input, gradient.contiguous(), coefficients, ctx.tables, ctx.sizes), None, None, None


def deterministic_polynomial(input, coefficients, tables, sizes):
    if triton is None or input.device.type != "cuda" or input.dtype not in {torch.float32, torch.float64}:
        raise RuntimeError("deterministic sparse polynomial requires Triton and real CUDA f32/f64")
    if coefficients.requires_grad or coefficients.numel() != tables[10].numel():
        raise ValueError("polynomial requires one fixed coefficient per compiler event")
    if coefficients.dtype != input.dtype or coefficients.device != input.device or any(
            value.dtype != torch.long or value.device != input.device for value in tables):
        raise ValueError("polynomial coefficients/tables must match input dtype and device")
    if input.shape[-1] != sizes[0]:
        raise ValueError("polynomial input width differs from its numeric schedule")
    flat = input.reshape(-1, sizes[0]).contiguous()
    result = _Polynomial.apply(flat, coefficients.contiguous(), tables, sizes)
    return result.reshape(*input.shape[:-1], sizes[1])
