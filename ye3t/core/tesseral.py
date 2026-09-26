
"""Conversions between complex spherical and real tesseral multiplets."""

import torch


_REAL_TO_COMPLEX_MATRIX_CACHE = {}


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
    inverse_sqrt_two = 2.0 ** -0.5
    matrix[L, L] = 1.0
    for m in range(1, L + 1):
        cosine = L - m
        sine = L + m
        negative = L - m
        positive = L + m
        sign = (-1) ** m
        matrix[cosine, negative] = inverse_sqrt_two
        matrix[sine, negative] = 1j * inverse_sqrt_two
        matrix[cosine, positive] = sign * inverse_sqrt_two
        matrix[sine, positive] = -1j * sign * inverse_sqrt_two
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
    return (
        values.to(dtype=matrix.dtype)
        @ matrix.conj().transpose(0, 1)
    ).real


def real_tesseral_to_complex_multiplet(values, L):
    L = int(L)
    if L == 0:
        if values.ndim >= 1 and values.shape[-1] == 1:
            return values.to(dtype=torch.complex128 if values.dtype == torch.float64 else torch.complex64)
        return values.unsqueeze(-1).to(dtype=torch.complex128 if values.dtype == torch.float64 else torch.complex64)
    if values.shape[-1] != 2 * L + 1:
        raise ValueError(f"Expected last dimension {2 * L + 1} for L={L}, got {values.shape[-1]}.")
    matrix = _real_to_complex_matrix(
        L,
        values.dtype,
        values.device,
    )
    return values.to(dtype=matrix.dtype) @ matrix


__all__ = ['complex_multiplet_to_real_tesseral', 'real_tesseral_to_complex_multiplet']
