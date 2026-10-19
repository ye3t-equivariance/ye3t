"""Small real batched matrix products with differentiable adjoints.

Carrier channel maps often have only a few columns per physical occurrence.
Elementwise CUDA reductions avoid the large-matrix scheduling overhead in this
regime. The adjoints are the same matrix operation with transposed operands,
so mixed position/parameter derivatives retain the complete chain rule.
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
    def _matrix_product(A, B, C, R, M, K, N, A0, A1, A2, B0, B1, B2):
        index = tl.program_id(0)*128+tl.arange(0, 128)
        row, m, n = index//(M*N), (index//N)%M, index%N
        total = tl.full((128,), 0., A.dtype.element_ty)
        for k in range(K):
            a = tl.load(A+row*A0+m*A1+k*A2, row < R, 0)
            b = tl.load(B+row*B0+k*B1+n*B2, row < R, 0)
            total += a*b
        tl.store(C+index, total, row < R)

    @triton.jit
    def _matrix_product_pair(P, X, Y, DX, DY, R, M, K, N, P0, P1, P2, X0, X1, X2, Y0, Y1, Y2):
        # Both adjoints of C = X Y with cotangent P in one launch:
        # DX[r,m,k] = sum_n P[r,m,n] Y[r,k,n] and DY[r,k,n] = sum_m X[r,m,k] P[r,m,n].
        # Each output element keeps the sequential ascending reduction of
        # _matrix_product, so the values equal the separate launches exactly.
        index = tl.program_id(0)*128+tl.arange(0, 128)
        if tl.program_id(1) == 0:
            row_x, m_x, k_x = index//(M*K), (index//K)%M, index%K
            total_x = tl.full((128,), 0., P.dtype.element_ty)
            for n_x in range(N):
                p = tl.load(P+row_x*P0+m_x*P1+n_x*P2, row_x < R, 0)
                y = tl.load(Y+row_x*Y0+k_x*Y1+n_x*Y2, row_x < R, 0)
                total_x += p*y
            tl.store(DX+index, total_x, row_x < R)
        else:
            row_y, k_y, n_y = index//(K*N), (index//N)%K, index%N
            total_y = tl.full((128,), 0., P.dtype.element_ty)
            for m_y in range(M):
                x = tl.load(X+row_y*X0+m_y*X1+k_y*X2, row_y < R, 0)
                q = tl.load(P+row_y*P0+m_y*P1+n_y*P2, row_y < R, 0)
                total_y += x*q
            tl.store(DY+index, total_y, row_y < R)

    @triton.jit
    def _matrix_product_sum(A, B, C, D, E, R, M, K, N, A0, A1, A2, B0, B1, B2, C0, C1, C2, D0, D1, D2):
        # E = A B + C D with two complete reductions added once at the end,
        # matching the sum of two separately reduced products.
        index = tl.program_id(0)*128+tl.arange(0, 128)
        row, m, n = index//(M*N), (index//N)%M, index%N
        first = tl.full((128,), 0., A.dtype.element_ty)
        for k in range(K):
            a = tl.load(A+row*A0+m*A1+k*A2, row < R, 0)
            b = tl.load(B+row*B0+k*B1+n*B2, row < R, 0)
            first += a*b
        second = tl.full((128,), 0., A.dtype.element_ty)
        for k in range(K):
            c = tl.load(C+row*C0+m*C1+k*C2, row < R, 0)
            d = tl.load(D+row*D0+k*D1+n*D2, row < R, 0)
            second += c*d
        tl.store(E+index, first+second, row < R)


def _launch_pair(cotangent, left, right):
    """One launch for both adjoints of left @ right; outputs are contiguous."""
    rows, height, inner = left.shape
    width = right.shape[-1]
    left_adjoint = left.new_empty(rows, height, inner)
    right_adjoint = left.new_empty(rows, inner, width)
    if rows and height and inner and width:
        grid = (max(triton.cdiv(left_adjoint.numel(), 128), triton.cdiv(right_adjoint.numel(), 128)), 2)
        _matrix_product_pair[grid](cotangent, left, right, left_adjoint, right_adjoint,
            rows, height, inner, width, *cotangent.stride(), *left.stride(), *right.stride(),
            enable_fp_fusion=False)
    return left_adjoint, right_adjoint


class _SmallMatrixProductAdjoint(torch.autograd.Function):
    """Both first-order adjoints of the small product; its backward is analytic.

    For C = A B with cotangent G: dA = G B^T and dB = A^T G. Given second
    cotangents (cA, cB): dG = cA B + A cB, d(A) = G cB^T, d(B) = cA^T G.
    Reference: PyTorch custom-function double-backward tutorial,
    https://docs.pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html
    """

    @staticmethod
    def forward(ctx, gradient, left, right):
        ctx.save_for_backward(gradient, left, right)
        return _launch_pair(gradient, left, right)

    @staticmethod
    def backward(ctx, left_cotangent, right_cotangent):
        if torch.is_grad_enabled():
            raise RuntimeError("small matrix products support derivatives through order two")
        gradient, left, right = ctx.saved_tensors
        rows, height, inner = left.shape
        width = right.shape[-1]
        gradient_adjoint = left.new_empty(rows, height, width)
        if rows and height and inner and width:
            _matrix_product_sum[(triton.cdiv(gradient_adjoint.numel(), 128),)](
                left_cotangent, right, left, right_cotangent, gradient_adjoint,
                rows, height, inner, width, *left_cotangent.stride(), *right.stride(),
                *left.stride(), *right_cotangent.stride(), enable_fp_fusion=False)
        left_adjoint, right_adjoint = _launch_pair(gradient, left_cotangent, right_cotangent)
        return (gradient_adjoint if ctx.needs_input_grad[0] else None,
                left_adjoint if ctx.needs_input_grad[1] else None,
                right_adjoint if ctx.needs_input_grad[2] else None)


class _SmallMatrixProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left, right):
        ctx.save_for_backward(left, right)
        rows, height, inner = left.shape
        width = right.shape[-1]
        result = left.new_empty(rows, height, width)
        if result.numel():
            _matrix_product[(triton.cdiv(result.numel(), 128),)](
                left, right, result, rows, height, inner, width,
                *left.stride(), *right.stride(), enable_fp_fusion=False)
        return result

    @staticmethod
    def backward(ctx, gradient):
        left, right = ctx.saved_tensors
        left_adjoint, right_adjoint = _SmallMatrixProductAdjoint.apply(gradient, left, right)
        return (left_adjoint if ctx.needs_input_grad[0] else None,
                right_adjoint if ctx.needs_input_grad[1] else None)


def small_bmm(left, right, execution_report=None):
    """Use the small CUDA kernel where supported; report the actual operation."""
    supported = (triton is not None and left.is_cuda and right.device == left.device
        and left.dtype in (torch.float32, torch.float64) and right.dtype == left.dtype
        and left.ndim == right.ndim == 3 and left.shape[0] == right.shape[0]
        and left.shape[2] == right.shape[1]
        and all(0 < dimension <= 32 for dimension in (*left.shape[1:], right.shape[2])))
    if supported:
        result = _SmallMatrixProduct.apply(left, right)
        backend = "triton_small_bmm"
    else:
        result = torch.bmm(left, right)
        backend = "torch_bmm"
    if execution_report is not None:
        execution_report[backend] = execution_report.get(backend, 0)+1
    return result
