
"""Shared packed real-basis CG backend dispatch.

This module is the single runtime switch for paired channel-wise CG products.
It keeps OpenEquivariance optional: callers can request it, but unsupported
devices, dtypes, or shapes fall back to Triton/Torch unless strict mode is set.
"""

from functools import lru_cache
import os

import torch

from ye3t.core.couplings import clebsch_gordan
from ye3t.core.tesseral import complex_multiplet_to_real_tesseral, real_tesseral_to_complex_multiplet

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None


PACKED_CG_BACKENDS = ("pytorch", "triton", "openequivariance", "auto")

_validated_openequivariance_signatures = set()


def normalize_packed_cg_backend(backend):
    normalized = "pytorch" if backend is None else str(backend).strip().lower().replace("-", "_")
    aliases = {
        "torch": "pytorch",
        "native": "pytorch",
        "oeq": "openequivariance",
        "open_equivariance": "openequivariance",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"pytorch", "triton", "openequivariance", "auto", "torch_compile"}:
        raise ValueError(
            "backend must be one of {'pytorch', 'triton', 'openequivariance', 'oeq', 'auto'}, "
            f"got {backend!r}."
        )
    return "pytorch" if normalized == "torch_compile" else normalized


@lru_cache(maxsize=None)
def _cg_tensor_cpu(L1, L2, L_out):
    tensor = torch.zeros((2 * int(L1) + 1, 2 * int(L2) + 1, 2 * int(L_out) + 1), dtype=torch.complex128)
    for idx_m1, m1 in enumerate(range(-int(L1), int(L1) + 1)):
        for idx_m2, m2 in enumerate(range(-int(L2), int(L2) + 1)):
            M = m1 + m2
            if abs(M) > int(L_out):
                continue
            tensor[idx_m1, idx_m2, M + int(L_out)] = complex(
                clebsch_gordan(int(L1), m1, int(L2), m2, int(L_out), M)
            )
    return tensor


@lru_cache(maxsize=None)
def _real_cg_entries_cpu(
    L1,
    L2,
    L_out,
    *,
    tol = 1.0e-14,
):
    L1 = int(L1)
    L2 = int(L2)
    L_out = int(L_out)
    left_dim = 2 * L1 + 1
    right_dim = 2 * L2 + 1
    out_dim = 2 * L_out + 1
    cg = _cg_tensor_cpu(L1, L2, L_out)
    entries = []
    for idx_left in range(left_dim):
        left = torch.zeros((1, left_dim), dtype=torch.float64)
        left[0, idx_left] = 1.0
        left_complex = real_tesseral_to_complex_multiplet(left, L1)
        for idx_right in range(right_dim):
            right = torch.zeros((1, right_dim), dtype=torch.float64)
            right[0, idx_right] = 1.0
            right_complex = real_tesseral_to_complex_multiplet(right, L2)
            coupled = torch.einsum("nm,np,mpr->nr", left_complex, right_complex, cg)
            if (L1 + L2 - L_out) % 2:
                coupled = -1j * coupled
            real = complex_multiplet_to_real_tesseral(
                coupled,
                L=L_out,
                M_values=tuple(range(-L_out, L_out + 1)),
            ).reshape(out_dim)
            for idx_out, value in enumerate(real):
                scalar = complex(value.item())
                if abs(scalar.imag) > 1.0e-10:
                    raise ValueError("Real-basis CG construction produced a non-negligible imaginary coefficient.")
                if abs(scalar.real) > float(tol):
                    entries.append((idx_left, idx_right, idx_out, float(scalar.real)))
    return tuple(entries)


@lru_cache(maxsize=None)
def _paired_real_cg_path_table_cpu(
    L1,
    L2,
    L_out,
    channel_count,
    dtype_name,
):
    from ye3t.backends.triton_cg import GroupedCGPathTable

    dtype = torch.float32 if dtype_name == "torch.float32" else torch.float64
    entries = _real_cg_entries_cpu(int(L1), int(L2), int(L_out))
    channels = torch.arange(int(channel_count), dtype=torch.long)
    return GroupedCGPathTable(
        path_left=channels,
        path_right=channels.clone(),
        path_out=channels.clone(),
        path_weight=torch.ones(int(channel_count), dtype=dtype),
        cg_m1=torch.tensor([entry[0] for entry in entries], dtype=torch.long),
        cg_m2=torch.tensor([entry[1] for entry in entries], dtype=torch.long),
        cg_M=torch.tensor([entry[2] for entry in entries], dtype=torch.long),
        cg_value=torch.tensor([entry[3] for entry in entries], dtype=dtype),
    )


if triton is not None:  # pragma: no cover - exercised only when Triton is available
    @triton.jit
    def _flat_mul_kernel(left_ptr, right_ptr, out_ptr, numel):
        offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
        mask = offsets < numel
        left = tl.load(left_ptr + offsets, mask=mask, other=0.0)
        right = tl.load(right_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, left * right, mask=mask)


def _triton_flat_mul_available(left, right):
    return bool(
        triton is not None
        and not torch.is_grad_enabled()
        and left.is_cuda
        and right.is_cuda
        and left.is_contiguous()
        and right.is_contiguous()
        and left.shape == right.shape
        and left.dtype in (torch.float32, torch.float64)
        and right.dtype == left.dtype
    )


def _flat_mul(left, right, *, prefer_triton):
    if bool(prefer_triton) and _triton_flat_mul_available(left, right):
        out = torch.empty_like(left)
        block = 256
        grid = (triton.cdiv(left.numel(), block),)
        try:
            _flat_mul_kernel[grid](left, right, out, left.numel())
            return out, "triton_scalar_mul"
        except Exception:
            pass
    return left * right, "torch_scalar_mul"


def _couple_packed_openequivariance(
    left,
    right,
    L1,
    L2,
    Lout,
):
    try:
        from ye3t.backends.openequivariance_bridge import openequivariance_paired_cg_forward

        result = openequivariance_paired_cg_forward(
            left.contiguous(),
            right.contiguous(),
            left_L=int(L1),
            right_L=int(L2),
            out_L=int(Lout),
        )
    except Exception:
        if os.environ.get("YE3T_DEBUG_TRITON") == "1" or os.environ.get("GNE3_DEBUG_TRITON") == "1":
            raise
        return None
    if result is None:
        return None
    out, backend = result
    if int(Lout) == 0:
        return out.squeeze(-1), str(backend)
    return out, str(backend)


def _couple_packed_triton(left, right, L1, L2, Lout, *, allow_triton_autograd = False):
    if left.ndim != 3 or right.ndim != 3 or left.shape[:2] != right.shape[:2]:
        return None
    if left.dtype not in (torch.float32, torch.float64) or right.dtype != left.dtype:
        return None
    try:
        from ye3t.backends.triton_cg import grouped_product_cg_forward

        table = _paired_real_cg_path_table_cpu(
            int(L1),
            int(L2),
            int(Lout),
            int(left.shape[1]),
            str(left.dtype),
        )
        out, backend = grouped_product_cg_forward(
            left.contiguous(),
            right.contiguous(),
            table,
            out_channels=int(left.shape[1]),
            out_m_dim=2 * int(Lout) + 1,
            prefer_triton=True,
            allow_triton_autograd=bool(allow_triton_autograd),
            validate_inputs=False,
            return_backend=True,
        )
    except Exception:
        return None
    if int(Lout) == 0:
        return out.squeeze(-1), str(backend)
    return out, str(backend)


def _couple_packed_torch(left, right, L1, L2, Lout):
    if int(L1) == 0 and int(L2) == 0 and int(Lout) == 0:
        return _flat_mul(left.contiguous(), right.contiguous(), prefer_triton=False)
    left_complex = real_tesseral_to_complex_multiplet(left, int(L1))
    right_complex = real_tesseral_to_complex_multiplet(right, int(L2))
    cg = _cg_tensor_cpu(int(L1), int(L2), int(Lout)).to(dtype=left_complex.dtype, device=left_complex.device)
    coupled = torch.einsum("npa,npb,abc->npc", left_complex, right_complex, cg)
    if (int(L1) + int(L2) - int(Lout)) % 2:
        coupled = -1j * coupled
    real = complex_multiplet_to_real_tesseral(
        coupled,
        L=int(Lout),
        M_values=tuple(range(-int(Lout), int(Lout) + 1)),
    )
    if int(Lout) == 0:
        return real.squeeze(-1), "torch_packed_cg"
    return real, "torch_packed_cg"


def couple_packed_real_tesseral(
    left,
    right,
    L1,
    L2,
    Lout,
    *,
    backend = "pytorch",
    strict_backend = False,
    validate_openequivariance = None,
):
    """Couple paired packed channels in the real tesseral basis.

    ``left`` and ``right`` carry the same channel axis. For scalar irreps they
    may be ``[N, C]``; otherwise they are ``[N, C, 2L+1]``.
    """

    backend_name = normalize_packed_cg_backend(backend)
    prefer_oeq = backend_name in {"openequivariance", "auto"}
    prefer_triton = backend_name in {"triton", "auto"}
    oeq_disabled = (
        os.environ.get("YE3T_DISABLE_OPENEQUIVARIANCE") == "1"
        or os.environ.get("GNE3_DISABLE_OPENEQUIVARIANCE") == "1"
    )
    if bool(prefer_oeq) and int(L1) >= 0 and int(L2) >= 0 and not oeq_disabled:
        oeq_result = _couple_packed_openequivariance(left, right, int(L1), int(L2), int(Lout))
        if oeq_result is not None and oeq_result[1] == "openequivariance_paired_uvu":
            should_validate = (
                os.environ.get(
                    "YE3T_OPENEQUIVARIANCE_VALIDATE",
                    os.environ.get("GNE3_OPENEQUIVARIANCE_VALIDATE", "1"),
                )
                != "0"
            )
            if validate_openequivariance is not None:
                should_validate = bool(validate_openequivariance)
            if should_validate:
                signature = (int(L1), int(L2), int(Lout), int(left.shape[1]), str(left.dtype), str(left.device))
                if signature not in _validated_openequivariance_signatures:
                    reference, _ = _couple_packed_torch(left, right, int(L1), int(L2), int(Lout))
                    atol = 5.0e-5 if left.dtype == torch.float32 else 5.0e-11
                    rtol = 5.0e-5 if left.dtype == torch.float32 else 5.0e-11
                    if not torch.allclose(oeq_result[0], reference, atol=atol, rtol=rtol):
                        if bool(strict_backend):
                            raise RuntimeError("OpenEquivariance paired CG validation failed against Torch.")
                        return reference, "torch_packed_cg"
                    _validated_openequivariance_signatures.add(signature)
            return oeq_result
        if bool(strict_backend) and backend_name == "openequivariance":
            raise RuntimeError("OpenEquivariance backend was requested, but this packed CG block could not use it.")

    if bool(prefer_triton) and int(L1) >= 0 and int(L2) >= 0:
        if int(L1) == 0 and int(L2) == 0 and int(Lout) == 0:
            return _flat_mul(left.contiguous(), right.contiguous(), prefer_triton=True)
        allow_triton_autograd = (
            backend_name == "triton"
            or os.environ.get("YE3T_ENABLE_TRITON_CG_AUTOGRAD", "0").strip().lower() in {"1", "true", "yes", "on"}
        )
        triton_result = _couple_packed_triton(
            left,
            right,
            int(L1),
            int(L2),
            int(Lout),
            allow_triton_autograd=allow_triton_autograd,
        )
        if triton_result is not None and triton_result[1] in {
            "empty",
            "triton_grouped_product_cg",
            "triton_grouped_product_cg_autograd",
        }:
            return triton_result
        if bool(strict_backend) and backend_name == "triton":
            raise RuntimeError("Triton backend was requested, but this packed CG block could not use it.")

    return _couple_packed_torch(left, right, int(L1), int(L2), int(Lout))


__all__ = [
    "PackedCGBackend",
    "couple_packed_real_tesseral",
    "normalize_packed_cg_backend",
]
