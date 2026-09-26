
"""Lie-algebra null-space comparison helpers.

This module implements a compact SO(3) specialization of the kernel-based
construction used by recent general Lie-group equivariant basis generators.
It is intentionally a comparison/diagnostic route, not part of the constructive
exact-ACE label path.

For fixed ACE leaves ``(n_i, l_i)`` and target ``L_R`` we:

1. group repeated channels into Young-subgroup blocks,
2. represent each repeated block in a normalized symmetric occupation basis,
3. build the total SO(3) generators on the permutation-invariant input space,
4. solve the equivariance equations

   ``J_out C - C J_in = 0``

as a dense numerical null-space problem.

The resulting nullity should match the exact multiplicity-space dimension, but
the matrix dimensions and singular spectrum expose where a numerical kernel
route starts to become impractical or tolerance-sensitive.
"""
from math import comb

import numpy as np

from .theory import YoungSubgroupBlock, ace_invariant_subspace_decomposition
from .validation import validate_leaf_quantum_numbers
from ye3t._record import recordclass


def _compositions(total, parts):
    """Yield weak compositions of ``total`` into ``parts`` nonnegative pieces."""
    if parts <= 0:
        return
    if parts == 1:
        yield (int(total),)
        return
    for first in range(int(total) + 1):
        for rest in _compositions(int(total) - first, parts - 1):
            yield (first,) + rest


def symmetric_power_dimension(l, k_b):
    """Dimension of ``Sym^k_b(V_l)``."""
    d = 2 * int(l) + 1
    return int(comb(int(k_b) + d - 1, d - 1))


@recordclass(('l', 'k_b', 'basis', 'Jz', 'Jplus'), frozen = True)
class SymmetricPowerGenerators:

    @property
    def dim(self):
        return int(len(self.basis))


def symmetric_power_so3_generators(l, k_b, *, dtype = np.float64):
    """Return ``J_z`` and ``J_+`` on a normalized symmetric occupation basis."""
    l = int(l)
    k_b = int(k_b)
    if l < 0 or k_b < 0:
        raise ValueError("l and k_b must be nonnegative.")
    d = 2 * l + 1
    basis = tuple(_compositions(k_b, d))
    index = {counts: i for i, counts in enumerate(basis)}
    Jz = np.zeros((len(basis), len(basis)), dtype=dtype)
    Jplus = np.zeros_like(Jz)
    magnetic_values = tuple(range(-l, l + 1))

    for col, counts in enumerate(basis):
        Jz[col, col] = sum(c * m for c, m in zip(counts, magnetic_values, strict=True))
        counts_list = list(counts)
        for m_idx, m in enumerate(range(-l, l)):
            occupied = counts_list[m_idx]
            if occupied == 0:
                continue
            raised_counts = list(counts_list)
            raised_counts[m_idx] -= 1
            raised_counts[m_idx + 1] += 1
            row = index[tuple(raised_counts)]
            single_particle = np.sqrt(float((l - m) * (l + m + 1)))
            bosonic_factor = np.sqrt(float(occupied * (counts_list[m_idx + 1] + 1)))
            Jplus[row, col] += single_particle * bosonic_factor

    return SymmetricPowerGenerators(l=l, k_b=k_b, basis=basis, Jz=Jz, Jplus=Jplus)


def irrep_so3_generators(L, *, dtype = np.float64):
    """Return ``J_z`` and ``J_+`` for the standard spin-``L`` SO(3) irrep."""
    L = int(L)
    if L < 0:
        raise ValueError("L must be nonnegative.")
    dim = 2 * L + 1
    Jz = np.zeros((dim, dim), dtype=dtype)
    Jplus = np.zeros_like(Jz)
    for col, m in enumerate(range(-L, L + 1)):
        Jz[col, col] = m
        if m < L:
            row = (m + 1) + L
            Jplus[row, col] = np.sqrt(float((L - m) * (L + m + 1)))
    return Jz, Jplus


def _kron_all(factors):
    if not factors:
        return np.ones((1, 1), dtype=np.float64)
    out = factors[0]
    for factor in factors[1:]:
        out = np.kron(out, factor)
    return out


def total_input_generators(
    blocks,
    *,
    dtype = np.float64,
):
    """Build total input-space ``J_z`` and ``J_+`` after PI block reduction."""
    block_generators = [
        symmetric_power_so3_generators(block.l, block.multiplicity, dtype=dtype)
        for block in blocks
    ]
    dims = tuple(gen.dim for gen in block_generators)
    if not block_generators:
        zero = np.zeros((1, 1), dtype=dtype)
        return zero, zero.copy(), (1,)

    total_dim = int(np.prod(dims, dtype=np.int64))
    Jz_total = np.zeros((total_dim, total_dim), dtype=dtype)
    Jplus_total = np.zeros_like(Jz_total)
    identities = [np.eye(dim, dtype=dtype) for dim in dims]
    for idx, gen in enumerate(block_generators):
        left = identities[:idx]
        right = identities[idx + 1 :]
        Jz_total += _kron_all([*left, gen.Jz, *right]).astype(dtype, copy=False)
        Jplus_total += _kron_all([*left, gen.Jplus, *right]).astype(dtype, copy=False)
    return Jz_total, Jplus_total, dims


def lie_constraint_shape(
    nin,
    lin,
    L_target,
    *,
    include_lowering = False,
):
    """Return ``(rows, cols, input_dim, block_dims, blocks)`` without building the matrix."""
    validate_leaf_quantum_numbers(nin, lin)
    decomposition = ace_invariant_subspace_decomposition(nin, lin)
    blocks = tuple(decomposition.blocks)
    block_dims = tuple(symmetric_power_dimension(block.l, block.multiplicity) for block in blocks)
    input_dim = int(np.prod(block_dims, dtype=np.int64)) if block_dims else 1
    output_dim = 2 * int(L_target) + 1
    parameter_dim = input_dim * output_dim
    generator_count = 3 if include_lowering else 2
    return generator_count * parameter_dim, parameter_dim, input_dim, block_dims, blocks


def build_lie_constraint_matrix(
    nin,
    lin,
    L_target,
    *,
    include_lowering = False,
    dtype = np.float64,
):
    """Build the dense SO(3) equivariance constraint matrix."""
    rows, cols, input_dim, block_dims, blocks = lie_constraint_shape(
        nin,
        lin,
        L_target,
        include_lowering=include_lowering,
    )
    del rows, cols
    Jz_in, Jplus_in, _ = total_input_generators(blocks, dtype=dtype)
    Jz_out, Jplus_out = irrep_so3_generators(L_target, dtype=dtype)
    output_dim = 2 * int(L_target) + 1
    I_in = np.eye(input_dim, dtype=dtype)
    I_out = np.eye(output_dim, dtype=dtype)

    def constraint(J_out, J_in):
        return np.kron(J_out, I_in) - np.kron(I_out, J_in.T)

    matrices = [
        constraint(Jz_out, Jz_in),
        constraint(Jplus_out, Jplus_in),
    ]
    if include_lowering:
        matrices.append(constraint(Jplus_out.T, Jplus_in.T))
    return np.vstack(matrices).astype(dtype, copy=False), block_dims, blocks


@recordclass(('nin', 'lin', 'L_target', 'status', 'blocks', 'block_dims', 'input_dim', 'parameter_dim', 'constraint_rows', 'estimated_matrix_bytes', 'svd_tol', 'nullity', 'rank', 'singular_values', 'smallest_nonzero_singular_value', 'largest_null_singular_value', 'kernel_gap', 'nonzero_condition_number', 'build_time_s', 'svd_time_s', 'total_time_s', 'error'), frozen = True)
class LieNullspaceAnalysisResult:
    error = ""

    def as_dict(self):
        return {
            "nin": self.nin,
            "lin": self.lin,
            "L_target": self.L_target,
            "status": self.status,
            "blocks": tuple((block.eta, block.l, block.multiplicity) for block in self.blocks),
            "block_dims": self.block_dims,
            "input_dim": self.input_dim,
            "parameter_dim": self.parameter_dim,
            "constraint_rows": self.constraint_rows,
            "estimated_matrix_bytes": self.estimated_matrix_bytes,
            "svd_tol": self.svd_tol,
            "nullity": self.nullity,
            "rank": self.rank,
            "smallest_nonzero_singular_value": self.smallest_nonzero_singular_value,
            "largest_null_singular_value": self.largest_null_singular_value,
            "kernel_gap": self.kernel_gap,
            "nonzero_condition_number": self.nonzero_condition_number,
            "build_time_s": self.build_time_s,
            "svd_time_s": self.svd_time_s,
            "total_time_s": self.total_time_s,
            "error": self.error,
        }


def analyze_lie_nullspace(
    nin,
    lin,
    L_target,
    *,
    rtol = 1e-10,
    atol = 1e-12,
    include_lowering = False,
    dtype = np.float64,
    max_matrix_bytes = 512 * 1024 * 1024,
):
    """Build and SVD the Lie-algebra null-space matrix with size guards."""
    import time

    t0 = time.perf_counter()
    rows, cols, input_dim, block_dims, blocks = lie_constraint_shape(
        nin,
        lin,
        L_target,
        include_lowering=include_lowering,
    )
    itemsize = np.dtype(dtype).itemsize
    estimated_bytes = int(rows) * int(cols) * int(itemsize)
    if estimated_bytes > int(max_matrix_bytes):
        return LieNullspaceAnalysisResult(
            nin=tuple(int(x) for x in nin),
            lin=tuple(int(x) for x in lin),
            L_target=int(L_target),
            status="too_large",
            blocks=blocks,
            block_dims=block_dims,
            input_dim=int(input_dim),
            parameter_dim=int(cols),
            constraint_rows=int(rows),
            estimated_matrix_bytes=estimated_bytes,
            svd_tol=float(max(atol, rtol)),
            nullity=None,
            rank=None,
            singular_values=tuple(),
            smallest_nonzero_singular_value=None,
            largest_null_singular_value=None,
            kernel_gap=None,
            nonzero_condition_number=None,
            build_time_s=0.0,
            svd_time_s=0.0,
            total_time_s=time.perf_counter() - t0,
            error=f"dense matrix estimate exceeds max_matrix_bytes={int(max_matrix_bytes)}",
        )

    try:
        matrix, _, _ = build_lie_constraint_matrix(
            nin,
            lin,
            L_target,
            include_lowering=include_lowering,
            dtype=dtype,
        )
    except MemoryError as exc:
        return LieNullspaceAnalysisResult(
            nin=tuple(int(x) for x in nin),
            lin=tuple(int(x) for x in lin),
            L_target=int(L_target),
            status="memory_error",
            blocks=blocks,
            block_dims=block_dims,
            input_dim=int(input_dim),
            parameter_dim=int(cols),
            constraint_rows=int(rows),
            estimated_matrix_bytes=estimated_bytes,
            svd_tol=float(max(atol, rtol)),
            nullity=None,
            rank=None,
            singular_values=tuple(),
            smallest_nonzero_singular_value=None,
            largest_null_singular_value=None,
            kernel_gap=None,
            nonzero_condition_number=None,
            build_time_s=time.perf_counter() - t0,
            svd_time_s=0.0,
            total_time_s=time.perf_counter() - t0,
            error=str(exc),
        )
    t1 = time.perf_counter()

    if matrix.size == 0:
        svals = np.array([], dtype=np.float64)
    else:
        svals = np.linalg.svd(matrix, compute_uv=False)
    t2 = time.perf_counter()

    sigma_max = float(svals[0]) if svals.size else 0.0
    tol = float(max(float(atol), float(rtol) * sigma_max))
    rank = int(np.sum(svals > tol))
    nullity = int(cols - rank)
    smallest_nonzero = float(svals[rank - 1]) if rank > 0 else None
    largest_null = float(svals[rank]) if rank < len(svals) else 0.0
    if smallest_nonzero is not None and largest_null is not None and largest_null > 0.0:
        kernel_gap = float(smallest_nonzero / largest_null)
    elif smallest_nonzero is not None:
        kernel_gap = float("inf")
    else:
        kernel_gap = None
    nonzero_condition = float(sigma_max / smallest_nonzero) if smallest_nonzero not in {None, 0.0} else None

    return LieNullspaceAnalysisResult(
        nin=tuple(int(x) for x in nin),
        lin=tuple(int(x) for x in lin),
        L_target=int(L_target),
        status="ok",
        blocks=blocks,
        block_dims=block_dims,
        input_dim=int(input_dim),
        parameter_dim=int(cols),
        constraint_rows=int(rows),
        estimated_matrix_bytes=estimated_bytes,
        svd_tol=tol,
        nullity=nullity,
        rank=rank,
        singular_values=tuple(float(x) for x in svals),
        smallest_nonzero_singular_value=smallest_nonzero,
        largest_null_singular_value=largest_null,
        kernel_gap=kernel_gap,
        nonzero_condition_number=nonzero_condition,
        build_time_s=t1 - t0,
        svd_time_s=t2 - t1,
        total_time_s=t2 - t0,
    )
