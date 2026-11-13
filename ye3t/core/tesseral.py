
"""Conversions between complex spherical and real tesseral multiplets."""

from functools import lru_cache

import torch


_REAL_TO_COMPLEX_MATRIX_CACHE = {}


@lru_cache(maxsize=128)
def _real_form_rows(L):
    """Sparse basis change as exact real magnitudes and integer powers of i."""
    from ye3t.exact_scalars import ExactRadical

    L = int(L)
    rows = [[] for _ in range(2*L+1)]
    rows[L] = [(L, ExactRadical.rational(1), 0)]
    root = ExactRadical.sqrt(2) / 2
    for m in range(1, L+1):
        rows[L-m] = [(L-m, root, 0), (L+m, root, 1)]
        rows[L+m] = [(L-m, (-1)**m * root, 0), (L+m, (-1)**m * root, 3)]
    return tuple(tuple(row) for row in rows)


@lru_cache(maxsize=128)
def _exact_real_form_matrix(L):
    """YE3T's column-vector tesseral-to-Condon--Shortley basis change.

    Shared from the existing Cauchy constructor, with the same coordinate order
    and phases. NIST DLMF 14.30.6 gives the spherical-harmonic reality relation.
    The cached immutable matrix cannot be changed by an individual consumer.
    """
    from ye3t._optional_sympy import sp

    L = int(L)
    matrix = sp.zeros(2*L+1)
    for row, entries in enumerate(_real_form_rows(L)):
        for column, value, phase in entries:
            matrix[row, column] = value._sympy_() * sp.I**phase
    if sp.simplify(matrix.conjugate().T * matrix) != sp.eye(2*L+1):
        raise ArithmeticError("Tesseral real-form map is not exactly unitary")
    return sp.ImmutableMatrix(matrix)


@lru_cache(maxsize=512)
def _real_cg_entries_exact(left_L, right_L, output_L):
    """Express existing exact CG maps in the documented real-form convention.

    Only the nonzeros of the three unitary basis changes are contracted.
    Odd left_L + right_L - output_L uses the established -i phase. This is a
    change of coordinates of the canonical map, with no new coupling discovery.
    """
    from ye3t.core.cg import cg_exact_integer
    from ye3t.exact_scalars import ExactRadical

    left_L, right_L, output_L = map(int, (left_L, right_L, output_L))
    rows = [_real_form_rows(L) for L in (left_L, right_L, output_L)]
    phase = -((left_L+right_L-output_L) % 2)
    zero = ExactRadical.rational(0)
    entries = {}
    for m in range(-left_L, left_L+1):
        for n in range(-right_L, right_L+1):
            if abs(m+n) > output_L:
                continue
            cg = cg_exact_integer(left_L, m, right_L, n, output_L, m+n)
            if not cg:
                continue
            for a, x, px in rows[0][m+left_L]:
                for b, y, py in rows[1][n+right_L]:
                    for c, z, pz in rows[2][m+n+output_L]:
                        exponent = (phase+px+py-pz) % 4
                        key = (a, b, c, exponent % 2)
                        entries[key] = entries.get(key, zero) + (-1 if exponent >= 2 else 1)*cg*x*y*z
    result = []
    for (a, b, c, imaginary), value in sorted(entries.items()):
        if value:
            if imaginary:
                raise ArithmeticError("Real-form CG basis change produced a complex coefficient")
            result.append((a, b, c, value))
    return tuple(result)


def _real_to_complex_matrix(L, dtype, device):
    L = int(L)
    complex_dtype = (
        torch.complex128
        if dtype == torch.float64
        else torch.complex64
    )
    key = (L, complex_dtype, str(torch.device(device)))
    cached = _REAL_TO_COMPLEX_MATRIX_CACHE.get(key)
    if cached is not None:
        return cached
    width = 2 * L + 1
    matrix = torch.zeros(
        (width, width),
        dtype=complex_dtype,
        device="cpu",
    )
    for row, entries in enumerate(_real_form_rows(L)):
        for column, value, phase in entries:
            matrix[column, row] = float(value) * 1j**phase
    cached = matrix.to(device=device)
    _REAL_TO_COMPLEX_MATRIX_CACHE[key] = cached
    return cached


def complex_multiplet_to_real_tesseral(values, L, M_values):
    L = int(L)
    if tuple(M_values) != tuple(range(-L, L + 1)):
        raise ValueError('M_values must be the full ordered range -L..L')
    matrix = _real_to_complex_matrix(
        L,
        values.real.dtype,
        values.device,
    )
    transformed = (
        values.to(dtype=matrix.dtype)
        @ matrix.conj().transpose(0, 1)
    )
    if transformed.numel():
        tolerance = 1.0e-5 if transformed.dtype == torch.complex64 else 1.0e-10
        scale = transformed.real.abs().amax().clamp_min(1.0)
        if not torch.isfinite(transformed).all() or transformed.imag.abs().amax() > tolerance * scale:
            raise ValueError("Complex multiplet has a non-negligible imaginary real-form component.")
    return transformed.real


def real_tesseral_to_complex_multiplet(values, L):
    L = int(L)
    if L == 0:
        if values.ndim >= 1 and values.shape[-1] == 1:
            return values.to(dtype=torch.complex128 if values.dtype == torch.float64 else torch.complex64)
        return values.unsqueeze(-1).to(dtype=torch.complex128 if values.dtype == torch.float64 else torch.complex64)
    if values.shape[-1] != 2 * L + 1:
        raise ValueError(f"Expected last dimension {2 * L + 1} for L={L}, got {values.shape[-1]}.")
    # Each +/-m pair has exactly two nonzero coefficients in the unitary
    # tesseral map. Apply these pairs directly, preserving the matrix's
    # Condon--Shortley phases and all derivative orders.
    dtype = torch.complex128 if values.dtype == torch.float64 else torch.complex64
    values = values.to(dtype=dtype)
    cosine, sine = values[..., :L], values[..., L+1:].flip(-1)
    negative = (cosine + 1j*sine) * (2.0 ** -0.5)
    positive = (cosine - 1j*sine).flip(-1) * (2.0 ** -0.5)
    signs = 1 - 2*(torch.arange(1, L+1, device=values.device) % 2)
    return torch.cat((negative, values[..., L:L+1], positive*signs), -1)


__all__ = ['complex_multiplet_to_real_tesseral', 'real_tesseral_to_complex_multiplet']
