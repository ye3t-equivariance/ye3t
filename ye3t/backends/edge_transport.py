"""Fused edge transport of packed real carriers with analytic adjoints.

One message of a carrier graph gathers the sender's packed coordinates
X[s] (channels x coordinates), applies the edge's channel matrix A[r]
(channels x channels, shared by the physical root edge r of the message),
scales by the edge envelope and adds the result to the receiver row. The
kernel below evaluates that gather, product, scale and scatter in one launch
and, through its mode switch, every first- and second-order adjoint of the
same trilinear map:

    Y[t, c, j] = sum_{e: t(e)=t} env[e] sum_k A[r(e), c, k] X[s(e), k, j]       (mode 0)
    dA[r, c, k] = sum_{e: r(e)=r} env[e] sum_j G[t(e), c, j] X[s(e), k, j]      (mode 1)
    dX[s, k, j] = sum_{e: s(e)=s} env[e] sum_c A[r(e), c, k] G[t(e), c, j]      (mode 2)
    denv[e] = sum_{c,j} G[t(e), c, j] sum_k A[r(e), c, k] X[s(e), k, j]         (mode 3)

Each destination row is owned by one program that walks the messages sorted
by that destination (receiver, root edge or sender order), so every sum has a
fixed sequential order, no atomics and no zero fill. The sorted orders depend
only on the graph indices; callers that evaluate the same chunk repeatedly
cache them with ``transport_tables``. The second-order terms are the same four
forms with one operand replaced by a cotangent, so the backward of the adjoint
is analytic (PyTorch custom-function double-backward contract,
https://docs.pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html).
"""

import torch
from ye3t.runtime.triton_config import configure_triton_c_compiler

configure_triton_c_compiler(required=False)
try:
    import triton
    import triton.language as tl
except ImportError:  # optional CUDA backend
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _edge_contraction(A, X, G, ENV, ROOT, SENDER, RECEIVER, PTR, ORDER, OUT, J,
                          W: tl.constexpr, WP: tl.constexpr, JP: tl.constexpr,
                          MODE: tl.constexpr, ACCUMULATE: tl.constexpr):
        pid = tl.program_id(0)
        c = tl.arange(0, WP)
        j = tl.arange(0, JP)
        cm = c < W
        jm = j < J
        square = cm[:, None] & cm[None, :]
        wide = cm[:, None] & jm[None, :]
        if MODE == 3:
            r = tl.load(ROOT + pid)
            s = tl.load(SENDER + pid)
            t = tl.load(RECEIVER + pid)
            a = tl.load(A + (r * W + c[:, None]) * W + c[None, :], square, 0.)
            x = tl.load(X + (s * W + c[:, None]) * J + j[None, :], wide, 0.)
            g = tl.load(G + (t * W + c[:, None]) * J + j[None, :], wide, 0.)
            message = tl.sum(a[:, :, None] * x[None, :, :], axis=1)
            value = tl.sum(tl.sum(g * message, axis=1), axis=0)
            if ACCUMULATE:
                value += tl.load(OUT + pid)
            tl.store(OUT + pid, value)
        else:
            start = tl.load(PTR + pid)
            stop = tl.load(PTR + pid + 1)
            if MODE == 1:
                acc = tl.zeros((WP, WP), dtype=A.dtype.element_ty)
            else:
                acc = tl.zeros((WP, JP), dtype=A.dtype.element_ty)
            for index in range(start, stop):
                e = tl.load(ORDER + index)
                env = tl.load(ENV + e)
                if MODE == 0:
                    r = tl.load(ROOT + e)
                    s = tl.load(SENDER + e)
                    a = tl.load(A + (r * W + c[:, None]) * W + c[None, :], square, 0.)
                    x = tl.load(X + (s * W + c[:, None]) * J + j[None, :], wide, 0.)
                    acc += env * tl.sum(a[:, :, None] * x[None, :, :], axis=1)
                elif MODE == 1:
                    s = tl.load(SENDER + e)
                    t = tl.load(RECEIVER + e)
                    g = tl.load(G + (t * W + c[:, None]) * J + j[None, :], wide, 0.)
                    x = tl.load(X + (s * W + c[:, None]) * J + j[None, :], wide, 0.)
                    acc += env * tl.sum(g[:, None, :] * x[None, :, :], axis=2)
                else:
                    r = tl.load(ROOT + e)
                    t = tl.load(RECEIVER + e)
                    a = tl.load(A + (r * W + c[:, None]) * W + c[None, :], square, 0.)
                    g = tl.load(G + (t * W + c[:, None]) * J + j[None, :], wide, 0.)
                    acc += env * tl.sum(a[:, :, None] * g[:, None, :], axis=0)
            if MODE == 1:
                destination = OUT + (pid * W + c[:, None]) * W + c[None, :]
                mask = square
            else:
                destination = OUT + (pid * W + c[:, None]) * J + j[None, :]
                mask = wide
            if ACCUMULATE:
                acc += tl.load(destination, mask, 0.)
            tl.store(destination, acc, mask)


def _edge_csr(index, count):
    order = torch.argsort(index, stable=True)
    pointers = index.new_zeros(count + 1)
    pointers[1:] = torch.bincount(index, minlength=count).cumsum(0)
    return pointers, order


def transport_tables(root_edges, sender, receiver, rows, roots):
    """Receiver-, root- and sender-sorted message orders with segment pointers.

    They depend only on the message indices of one graph (or chunk) and may be
    cached across every evaluation and replay of that graph.
    """
    receiver_pointers, receiver_order = _edge_csr(receiver, int(rows))
    root_pointers, root_order = _edge_csr(root_edges, int(roots))
    sender_pointers, sender_order = _edge_csr(sender, int(rows))
    return (receiver_pointers, receiver_order, root_pointers, root_order, sender_pointers, sender_order)


def _launch(mode, matrix, packed, gradient, envelope, root_edges, sender, receiver, tables, out, accumulate):
    """Evaluate one trilinear edge form into ``out`` (overwrite or accumulate)."""
    edges = envelope.shape[0]
    if edges == 0:
        if not accumulate:
            out.zero_()
        return out
    width, coordinates = packed.shape[-2], packed.shape[-1]
    if mode == 3:
        programs, pointers, order = edges, tables[0], tables[1]
    else:
        # tables = (receiver, root, sender) x (pointers, order); mode 0, 1, 2 scatter to those destinations.
        pointers, order = tables[2 * mode], tables[2 * mode + 1]
        programs = pointers.shape[0] - 1
    _edge_contraction[(programs,)](
        matrix, packed, packed if gradient is None else gradient, envelope, root_edges, sender, receiver,
        pointers, order, out, coordinates, width, triton.next_power_of_2(width), triton.next_power_of_2(coordinates),
        mode, accumulate, num_warps=4)
    return out


def _contiguous(*values):
    return tuple(None if value is None else value.contiguous() for value in values)


class _TransportAdjoint(torch.autograd.Function):
    """All first-order adjoints of the transport; its backward is analytic."""

    @staticmethod
    def forward(ctx, gradient, matrix, packed, envelope, root_edges, sender, receiver, *tables):
        gradient, matrix, packed, envelope = _contiguous(gradient, matrix, packed, envelope)
        ctx.save_for_backward(gradient, matrix, packed, envelope, root_edges, sender, receiver, *tables)
        indices = (root_edges, sender, receiver, tables)
        matrix_adjoint = _launch(1, matrix, packed, gradient, envelope, *indices, torch.empty_like(matrix), False)
        packed_adjoint = _launch(2, matrix, packed, gradient, envelope, *indices, torch.empty_like(packed), False)
        envelope_adjoint = _launch(3, matrix, packed, gradient, envelope, *indices, torch.empty_like(envelope), False)
        return matrix_adjoint, packed_adjoint, envelope_adjoint

    @staticmethod
    def backward(ctx, matrix_cotangent, packed_cotangent, envelope_cotangent):
        if torch.is_grad_enabled():
            raise RuntimeError("fused edge transport supports derivatives through order two")
        gradient, matrix, packed, envelope, root_edges, sender, receiver, *tables = ctx.saved_tensors
        needs = ctx.needs_input_grad
        indices = (root_edges, sender, receiver, tuple(tables))
        outputs = {}

        def add(name, mode, operands):
            # First term overwrites, later terms accumulate in the fixed program order.
            if name not in outputs:
                template = {"gradient": gradient, "matrix": matrix, "packed": packed, "envelope": envelope}[name]
                outputs[name] = _launch(mode, *operands, *indices, torch.empty_like(template), False)
            else:
                _launch(mode, *operands, *indices, outputs[name], True)

        if matrix_cotangent is not None:
            cotangent = matrix_cotangent.contiguous()
            if needs[0]:
                add("gradient", 0, (cotangent, packed, None, envelope))
            if needs[2]:
                add("packed", 2, (cotangent, packed, gradient, envelope))
            if needs[3]:
                add("envelope", 3, (cotangent, packed, gradient, envelope))
        if packed_cotangent is not None:
            cotangent = packed_cotangent.contiguous()
            if needs[0]:
                add("gradient", 0, (matrix, cotangent, None, envelope))
            if needs[1]:
                add("matrix", 1, (matrix, cotangent, gradient, envelope))
            if needs[3]:
                add("envelope", 3, (matrix, cotangent, gradient, envelope))
        if envelope_cotangent is not None:
            cotangent = envelope_cotangent.contiguous()
            if needs[0]:
                add("gradient", 0, (matrix, packed, None, cotangent))
            if needs[1]:
                add("matrix", 1, (matrix, packed, gradient, cotangent))
            if needs[2]:
                add("packed", 2, (matrix, packed, gradient, cotangent))
        return (outputs.get("gradient"), outputs.get("matrix"), outputs.get("packed"), outputs.get("envelope"),
                None, None, None) + (None,) * len(tables)


class _TransportMessages(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix, packed, envelope, root_edges, sender, receiver, *tables):
        matrix, packed, envelope = _contiguous(matrix, packed, envelope)
        ctx.save_for_backward(matrix, packed, envelope, root_edges, sender, receiver, *tables)
        rows = tables[0].shape[0] - 1
        out = packed.new_empty((rows,) + tuple(packed.shape[-2:]))
        return _launch(0, matrix, packed, None, envelope, root_edges, sender, receiver, tables, out, False)

    @staticmethod
    def backward(ctx, gradient):
        matrix, packed, envelope, root_edges, sender, receiver, *tables = ctx.saved_tensors
        matrix_adjoint, packed_adjoint, envelope_adjoint = _TransportAdjoint.apply(
            gradient, matrix, packed, envelope, root_edges, sender, receiver, *tables)
        needs = ctx.needs_input_grad
        return (matrix_adjoint if needs[0] else None, packed_adjoint if needs[1] else None,
                envelope_adjoint if needs[2] else None, None, None, None) + (None,) * len(tables)


def transport_supported(matrix, packed, envelope):
    """True when the fused kernel can evaluate this transport on this device."""
    if not (triton is not None and packed.is_cuda and matrix.device == packed.device == envelope.device
            and packed.dtype in (torch.float32, torch.float64) and matrix.dtype == packed.dtype == envelope.dtype
            and matrix.ndim == 3 and packed.ndim == 3 and envelope.ndim == 1
            and matrix.shape[1] == matrix.shape[2] == packed.shape[1] and packed.shape[1] > 0 and packed.shape[2] > 0):
        return False
    width, coordinates = (triton.next_power_of_2(int(value)) for value in packed.shape[-2:])
    return width <= 32 and width * width * coordinates <= 8192


def transport_messages(matrix, packed, envelope, root_edges, sender, receiver, rows, tables=None, execution_report=None):
    """Sum every message's enveloped channel map of the sender's packed carrier into its receiver.

    ``matrix`` holds one channel map per physical root edge, ``packed`` the real
    packed coordinates of every row, ``envelope`` one weight per message;
    ``root_edges``, ``sender`` and ``receiver`` index the messages. ``tables``
    are the sorted orders from ``transport_tables`` (built here when omitted).
    The result has ``rows`` rows and the packed shape of the input, with exact
    first and second derivatives with respect to ``matrix``, ``packed`` and
    ``envelope``.
    """
    if not transport_supported(matrix, packed, envelope):
        raise RuntimeError("fused edge transport requires Triton, CUDA f32/f64 operands and channels x coordinates within 32 x 8192")
    if not (root_edges.shape == sender.shape == receiver.shape == envelope.shape):
        raise ValueError("transport index tensors must have one entry per message")
    if any(value.dtype != torch.long for value in (root_edges, sender, receiver)):
        raise ValueError("transport indices must be int64")
    if tables is None:
        tables = transport_tables(root_edges, sender, receiver, rows, matrix.shape[0])
    if tables[0].shape[0] != int(rows) + 1 or tables[2].shape[0] != matrix.shape[0] + 1 or tables[4].shape[0] != int(rows) + 1:
        raise ValueError("transport tables do not match the row and root-edge counts")
    result = _TransportMessages.apply(matrix, packed, envelope, root_edges, sender, receiver, *tables)
    if execution_report is not None:
        execution_report["triton_fused_transport"] = execution_report.get("triton_fused_transport", 0) + 1
    return result
