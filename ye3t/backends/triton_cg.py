
"""Triton segmented sparse Clebsch-Gordan product kernels.

This module is intentionally small and schedule-shaped.  The public function
accepts an already-resolved path table for one fixed ``(L1, L2, Lout)`` group:
symbolic ACE/G_nu x SO(3) logic stays outside the kernel, and future typed DAG
or subtree fusion passes can feed the same path-table representation.
"""
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import torch
from ye3t._record import recordclass
from ye3t.runtime.triton_config import configure_triton_c_compiler

_cache_base = os.environ.get("XDG_CACHE_HOME") or tempfile.gettempdir()
_default_runtime_root = Path(_cache_base) / "ye3t" / "triton_cg"
_runtime_root = Path(
    os.environ.get("YE3T_RUNTIME_DIR")
    or os.environ.get("GNE3_RUNTIME_DIR")
    or os.environ.get("gne3_ACE_RUNTIME_DIR")
    or str(_default_runtime_root)
)
_tmp_dir = Path(
    os.environ.get("YE3T_TMPDIR")
    or os.environ.get("GNE3_TMPDIR")
    or os.environ.get("gne3_ACE_TMPDIR")
    or str(_runtime_root / "tmp")
)
_triton_cache_dir = Path(os.environ.get("TRITON_CACHE_DIR", str(_runtime_root / "triton_cache")))
_tmp_dir.mkdir(parents=True, exist_ok=True)
_triton_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["YE3T_RUNTIME_DIR"] = str(_runtime_root)
os.environ["YE3T_TMPDIR"] = str(_tmp_dir)
os.environ["TRITON_CACHE_DIR"] = str(_triton_cache_dir)
for _tmp_env_name in ("TMPDIR", "TMP", "TEMP"):
    os.environ[_tmp_env_name] = str(_tmp_dir)
tempfile.tempdir = str(_tmp_dir)


class _WritableTemporaryDirectory:
    """Portable replacement for ``tempfile.TemporaryDirectory``.

    Some Python sandbox combinations create ``mkdtemp`` directories
    with ACLs that deny writes to files inside the new directory.  Triton uses
    ``TemporaryDirectory`` while compiling its CUDA driver helper, so install a
    tiny compatible replacement before importing Triton.
    """

    def __init__(
        self,
        suffix = None,
        prefix = None,
        dir = None,
        ignore_cleanup_errors = False,
        **_,
    ):
        self._ignore_cleanup_errors = bool(ignore_cleanup_errors)
        self.name = self._make_dir(suffix=suffix or "", prefix=prefix or "tmp", directory=dir)

    @staticmethod
    def _make_dir(*, suffix, prefix, directory):
        base = Path(directory) if directory is not None else _tmp_dir
        base.mkdir(parents=True, exist_ok=True)
        for _ in range(100):
            candidate = base / f"{prefix}{uuid.uuid4().hex}{suffix}"
            try:
                candidate.mkdir()
                return str(candidate)
            except FileExistsError:
                continue
        raise FileExistsError(f"Could not create unique temporary directory under {base}")

    def __enter__(self):
        return self.name

    def __exit__(self, exc_type, exc, tb):
        self.cleanup()

    def cleanup(self):
        shutil.rmtree(self.name, ignore_errors=self._ignore_cleanup_errors)


if (
    os.name == "nt"
    and os.environ.get("YE3T_DISABLE_TEMPFILE_WORKAROUND") != "1"
    and os.environ.get("GNE3_DISABLE_TEMPFILE_WORKAROUND") != "1"
):
    tempfile.TemporaryDirectory = _WritableTemporaryDirectory  # type: ignore[assignment]

configure_triton_c_compiler(required=False)

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None


@recordclass(('path_left', 'path_right', 'path_out', 'path_weight', 'cg_m1', 'cg_m2', 'cg_M', 'cg_value'), frozen = True)
class GroupedCGPathTable:
    """One fixed-angular sparse product group.

    Each path contributes
    ``weight[p] * CG[k] * left[:, left[p], m1[k]] * right[:, right[p], m2[k]]``
    into ``out[:, out[p], M[k]]``.  Multiple paths may target the same output
    channel, so the Triton kernel accumulates with atomics.
    """

    def to(self, *, device, dtype):
        return GroupedCGPathTable(
            path_left=self.path_left.to(device=device, dtype=torch.long).contiguous(),
            path_right=self.path_right.to(device=device, dtype=torch.long).contiguous(),
            path_out=self.path_out.to(device=device, dtype=torch.long).contiguous(),
            path_weight=self.path_weight.to(device=device, dtype=dtype).contiguous(),
            cg_m1=self.cg_m1.to(device=device, dtype=torch.long).contiguous(),
            cg_m2=self.cg_m2.to(device=device, dtype=torch.long).contiguous(),
            cg_M=self.cg_M.to(device=device, dtype=torch.long).contiguous(),
            cg_value=self.cg_value.to(device=device, dtype=dtype).contiguous(),
        )

    @property
    def path_count(self):
        return int(self.path_left.numel())

    @property
    def cg_count(self):
        return int(self.cg_value.numel())


def _triton_enabled():
    disabled = (
        os.environ.get("YE3T_DISABLE_TRITON") == "1"
        or os.environ.get("GNE3_DISABLE_TRITON") == "1"
        or os.environ.get("gne3_ACE_DISABLE_TRITON") == "1"
        or os.environ.get("ye3t_ace_DISABLE_TRITON") == "1"
    )
    return bool(triton is not None and not disabled)


if triton is not None:  # pragma: no cover - exercised only on CUDA/Triton systems
    @triton.jit
    def _grouped_product_cg_kernel(
        left_ptr,
        right_ptr,
        out_ptr,
        path_left_ptr,
        path_right_ptr,
        path_out_ptr,
        path_weight_ptr,
        cg_m1_ptr,
        cg_m2_ptr,
        cg_M_ptr,
        cg_value_ptr,
        n_atoms,
        left_channels,
        right_channels,
        out_channels,
        m1_dim,
        m2_dim,
        mout_dim,
        path_count,
        cg_count,
    ):
        pid_term = tl.program_id(0)
        pid_atom = tl.program_id(1)
        atoms = pid_atom * 128 + tl.arange(0, 128)
        atom_mask = atoms < n_atoms

        path_idx = pid_term // cg_count
        cg_idx = pid_term - path_idx * cg_count

        left_channel = tl.load(path_left_ptr + path_idx)
        right_channel = tl.load(path_right_ptr + path_idx)
        out_channel = tl.load(path_out_ptr + path_idx)
        weight = tl.load(path_weight_ptr + path_idx)

        m1 = tl.load(cg_m1_ptr + cg_idx)
        m2 = tl.load(cg_m2_ptr + cg_idx)
        mout = tl.load(cg_M_ptr + cg_idx)
        cg = tl.load(cg_value_ptr + cg_idx)

        left_offsets = (atoms * left_channels + left_channel) * m1_dim + m1
        right_offsets = (atoms * right_channels + right_channel) * m2_dim + m2
        out_offsets = (atoms * out_channels + out_channel) * mout_dim + mout

        left_vals = tl.load(left_ptr + left_offsets, mask=atom_mask, other=0.0)
        right_vals = tl.load(right_ptr + right_offsets, mask=atom_mask, other=0.0)
        contrib = left_vals * right_vals * weight * cg
        tl.atomic_add(out_ptr + out_offsets, contrib, mask=atom_mask)

    @triton.jit
    def _grouped_product_cg_backward_kernel(
        grad_out_ptr,
        left_ptr,
        right_ptr,
        grad_left_ptr,
        grad_right_ptr,
        grad_path_weight_ptr,
        path_left_ptr,
        path_right_ptr,
        path_out_ptr,
        path_weight_ptr,
        cg_m1_ptr,
        cg_m2_ptr,
        cg_M_ptr,
        cg_value_ptr,
        n_atoms,
        left_channels,
        right_channels,
        out_channels,
        m1_dim,
        m2_dim,
        mout_dim,
        path_count,
        cg_count,
    ):
        pid_term = tl.program_id(0)
        pid_atom = tl.program_id(1)
        atoms = pid_atom * 128 + tl.arange(0, 128)
        atom_mask = atoms < n_atoms

        path_idx = pid_term // cg_count
        cg_idx = pid_term - path_idx * cg_count

        left_channel = tl.load(path_left_ptr + path_idx)
        right_channel = tl.load(path_right_ptr + path_idx)
        out_channel = tl.load(path_out_ptr + path_idx)
        weight = tl.load(path_weight_ptr + path_idx)

        m1 = tl.load(cg_m1_ptr + cg_idx)
        m2 = tl.load(cg_m2_ptr + cg_idx)
        mout = tl.load(cg_M_ptr + cg_idx)
        cg = tl.load(cg_value_ptr + cg_idx)
        coeff = weight * cg

        left_offsets = (atoms * left_channels + left_channel) * m1_dim + m1
        right_offsets = (atoms * right_channels + right_channel) * m2_dim + m2
        out_offsets = (atoms * out_channels + out_channel) * mout_dim + mout

        gout = tl.load(grad_out_ptr + out_offsets, mask=atom_mask, other=0.0)
        left_vals = tl.load(left_ptr + left_offsets, mask=atom_mask, other=0.0)
        right_vals = tl.load(right_ptr + right_offsets, mask=atom_mask, other=0.0)

        tl.atomic_add(grad_left_ptr + left_offsets, coeff * gout * right_vals, mask=atom_mask)
        tl.atomic_add(grad_right_ptr + right_offsets, coeff * gout * left_vals, mask=atom_mask)
        grad_weight = tl.sum(cg * gout * left_vals * right_vals, axis=0)
        tl.atomic_add(grad_path_weight_ptr + path_idx, grad_weight)

    @triton.jit
    def _grouped_product_cg_double_backward_kernel(
        grad_out_ptr,
        left_ptr,
        right_ptr,
        grad_grad_left_ptr,
        grad_grad_right_ptr,
        grad_grad_path_weight_ptr,
        grad2_grad_out_ptr,
        grad2_left_ptr,
        grad2_right_ptr,
        grad2_path_weight_ptr,
        path_left_ptr,
        path_right_ptr,
        path_out_ptr,
        path_weight_ptr,
        cg_m1_ptr,
        cg_m2_ptr,
        cg_M_ptr,
        cg_value_ptr,
        n_atoms,
        left_channels,
        right_channels,
        out_channels,
        m1_dim,
        m2_dim,
        mout_dim,
        path_count,
        cg_count,
    ):
        pid_term = tl.program_id(0)
        pid_atom = tl.program_id(1)
        atoms = pid_atom * 128 + tl.arange(0, 128)
        atom_mask = atoms < n_atoms

        path_idx = pid_term // cg_count
        cg_idx = pid_term - path_idx * cg_count

        left_channel = tl.load(path_left_ptr + path_idx)
        right_channel = tl.load(path_right_ptr + path_idx)
        out_channel = tl.load(path_out_ptr + path_idx)
        weight = tl.load(path_weight_ptr + path_idx)

        m1 = tl.load(cg_m1_ptr + cg_idx)
        m2 = tl.load(cg_m2_ptr + cg_idx)
        mout = tl.load(cg_M_ptr + cg_idx)
        cg = tl.load(cg_value_ptr + cg_idx)
        coeff = weight * cg

        left_offsets = (atoms * left_channels + left_channel) * m1_dim + m1
        right_offsets = (atoms * right_channels + right_channel) * m2_dim + m2
        out_offsets = (atoms * out_channels + out_channel) * mout_dim + mout

        gout = tl.load(grad_out_ptr + out_offsets, mask=atom_mask, other=0.0)
        left_vals = tl.load(left_ptr + left_offsets, mask=atom_mask, other=0.0)
        right_vals = tl.load(right_ptr + right_offsets, mask=atom_mask, other=0.0)
        grad_grad_left = tl.load(
            grad_grad_left_ptr + left_offsets,
            mask=atom_mask,
            other=0.0,
        )
        grad_grad_right = tl.load(
            grad_grad_right_ptr + right_offsets,
            mask=atom_mask,
            other=0.0,
        )
        grad_grad_weight = tl.load(grad_grad_path_weight_ptr + path_idx)

        grad2_gout = (
            coeff
            * (
                grad_grad_left * right_vals
                + grad_grad_right * left_vals
            )
            + cg * grad_grad_weight * left_vals * right_vals
        )
        grad2_left = (
            coeff * grad_grad_right * gout
            + cg * grad_grad_weight * gout * right_vals
        )
        grad2_right = (
            coeff * grad_grad_left * gout
            + cg * grad_grad_weight * gout * left_vals
        )
        grad2_weight = tl.sum(
            cg
            * gout
            * (
                grad_grad_left * right_vals
                + grad_grad_right * left_vals
            ),
            axis=0,
        )

        tl.atomic_add(
            grad2_grad_out_ptr + out_offsets,
            grad2_gout,
            mask=atom_mask,
        )
        tl.atomic_add(
            grad2_left_ptr + left_offsets,
            grad2_left,
            mask=atom_mask,
        )
        tl.atomic_add(
            grad2_right_ptr + right_offsets,
            grad2_right,
            mask=atom_mask,
        )
        tl.atomic_add(
            grad2_path_weight_ptr + path_idx,
            grad2_weight,
        )


_LAST_TRITON_CG_BACKWARD_BACKEND = None
_LAST_TRITON_CG_DOUBLE_BACKWARD_BACKEND = None


def _grouped_product_cg_backward_reference(
    grad_out,
    left,
    right,
    table,
    *,
    compute_path_weight_grad = False,
):
    """PyTorch backward for the sparse real-CG product table."""

    table = table.to(device=left.device, dtype=left.dtype)
    grad_left = torch.zeros_like(left)
    grad_right = torch.zeros_like(right)
    grad_path_weight = torch.zeros_like(table.path_weight) if bool(compute_path_weight_grad) else None
    if table.path_count == 0 or table.cg_count == 0:
        return grad_left, grad_right, grad_path_weight
    for path_idx in range(table.path_count):
        lch = int(table.path_left[path_idx].item())
        rch = int(table.path_right[path_idx].item())
        och = int(table.path_out[path_idx].item())
        weight = table.path_weight[path_idx]
        for cg_idx in range(table.cg_count):
            m1 = int(table.cg_m1[cg_idx].item())
            m2 = int(table.cg_m2[cg_idx].item())
            mout = int(table.cg_M[cg_idx].item())
            coeff = weight * table.cg_value[cg_idx]
            gout = grad_out[:, och, mout]
            grad_left[:, lch, m1] = grad_left[:, lch, m1] + coeff * gout * right[:, rch, m2]
            grad_right[:, rch, m2] = grad_right[:, rch, m2] + coeff * gout * left[:, lch, m1]
            if grad_path_weight is not None:
                grad_path_weight[path_idx] = grad_path_weight[path_idx] + (
                    table.cg_value[cg_idx] * gout * left[:, lch, m1] * right[:, rch, m2]
                ).sum()
    return grad_left, grad_right, grad_path_weight


class _GroupedProductCGBackwardAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        grad_out,
        left,
        right,
        path_left,
        path_right,
        path_out,
        path_weight,
        cg_m1,
        cg_m2,
        cg_M,
        cg_value,
        out_channels,
        out_m_dim,
        block_atoms,
    ):
        table = GroupedCGPathTable(
            path_left=path_left,
            path_right=path_right,
            path_out=path_out,
            path_weight=path_weight,
            cg_m1=cg_m1,
            cg_m2=cg_m2,
            cg_M=cg_M,
            cg_value=cg_value,
        ).to(device=left.device, dtype=left.dtype)
        grad_out = grad_out.contiguous()
        grad_left = torch.zeros_like(left)
        grad_right = torch.zeros_like(right)
        grad_path_weight = torch.zeros_like(path_weight)
        grid = (
            int(table.path_count * table.cg_count),
            triton.cdiv(int(left.shape[0]), 128),
        )
        _grouped_product_cg_backward_kernel[grid](
            grad_out,
            left,
            right,
            grad_left,
            grad_right,
            grad_path_weight,
            table.path_left,
            table.path_right,
            table.path_out,
            table.path_weight,
            table.cg_m1,
            table.cg_m2,
            table.cg_M,
            table.cg_value,
            n_atoms=int(left.shape[0]),
            left_channels=int(left.shape[1]),
            right_channels=int(right.shape[1]),
            out_channels=int(out_channels),
            m1_dim=int(left.shape[2]),
            m2_dim=int(right.shape[2]),
            mout_dim=int(out_m_dim),
            path_count=int(table.path_count),
            cg_count=int(table.cg_count),
        )
        ctx.save_for_backward(
            grad_out,
            left,
            right,
            table.path_left,
            table.path_right,
            table.path_out,
            table.path_weight,
            table.cg_m1,
            table.cg_m2,
            table.cg_M,
            table.cg_value,
        )
        ctx.out_channels = int(out_channels)
        ctx.out_m_dim = int(out_m_dim)
        ctx.block_atoms = int(block_atoms)
        ctx.set_materialize_grads(False)
        return grad_left, grad_right, grad_path_weight

    @staticmethod
    def backward(
        ctx,
        grad_grad_left,
        grad_grad_right,
        grad_grad_path_weight,
    ):
        global _LAST_TRITON_CG_DOUBLE_BACKWARD_BACKEND

        (
            grad_out,
            left,
            right,
            path_left,
            path_right,
            path_out,
            path_weight,
            cg_m1,
            cg_m2,
            cg_M,
            cg_value,
        ) = ctx.saved_tensors
        if grad_grad_left is None:
            grad_grad_left = torch.zeros_like(left)
        if grad_grad_right is None:
            grad_grad_right = torch.zeros_like(right)
        if grad_grad_path_weight is None:
            grad_grad_path_weight = torch.zeros_like(path_weight)
        grad_grad_left = grad_grad_left.contiguous()
        grad_grad_right = grad_grad_right.contiguous()
        grad_grad_path_weight = grad_grad_path_weight.contiguous()

        grad2_grad_out = torch.zeros_like(grad_out)
        grad2_left = torch.zeros_like(left)
        grad2_right = torch.zeros_like(right)
        grad2_path_weight = torch.zeros_like(path_weight)
        grid = (
            int(path_left.numel() * cg_value.numel()),
            triton.cdiv(int(left.shape[0]), 128),
        )
        _grouped_product_cg_double_backward_kernel[grid](
            grad_out,
            left,
            right,
            grad_grad_left,
            grad_grad_right,
            grad_grad_path_weight,
            grad2_grad_out,
            grad2_left,
            grad2_right,
            grad2_path_weight,
            path_left,
            path_right,
            path_out,
            path_weight,
            cg_m1,
            cg_m2,
            cg_M,
            cg_value,
            n_atoms=int(left.shape[0]),
            left_channels=int(left.shape[1]),
            right_channels=int(right.shape[1]),
            out_channels=int(ctx.out_channels),
            m1_dim=int(left.shape[2]),
            m2_dim=int(right.shape[2]),
            mout_dim=int(ctx.out_m_dim),
            path_count=int(path_left.numel()),
            cg_count=int(cg_value.numel()),
        )
        _LAST_TRITON_CG_DOUBLE_BACKWARD_BACKEND = (
            "triton_grouped_product_cg_double_backward"
        )
        return (
            grad2_grad_out,
            grad2_left,
            grad2_right,
            None,
            None,
            None,
            grad2_path_weight,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _GroupedProductCGTritonAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        left,
        right,
        path_left,
        path_right,
        path_out,
        path_weight,
        cg_m1,
        cg_m2,
        cg_M,
        cg_value,
        out_channels,
        out_m_dim,
        block_atoms,
    ):
        table = GroupedCGPathTable(
            path_left=path_left,
            path_right=path_right,
            path_out=path_out,
            path_weight=path_weight,
            cg_m1=cg_m1,
            cg_m2=cg_m2,
            cg_M=cg_M,
            cg_value=cg_value,
        )
        ctx.save_for_backward(left, right, path_left, path_right, path_out, path_weight, cg_m1, cg_m2, cg_M, cg_value)
        ctx.out_channels = int(out_channels)
        ctx.out_m_dim = int(out_m_dim)
        ctx.block_atoms = 128
        ctx.compute_path_weight_grad = bool(ctx.needs_input_grad[5])

        table_cuda = table.to(device=left.device, dtype=left.dtype)
        out = torch.zeros((left.shape[0], int(out_channels), int(out_m_dim)), dtype=left.dtype, device=left.device)
        grid = (
            int(table_cuda.path_count * table_cuda.cg_count),
            triton.cdiv(int(left.shape[0]), int(ctx.block_atoms)),
        )
        _grouped_product_cg_kernel[grid](
            left,
            right,
            out,
            table_cuda.path_left,
            table_cuda.path_right,
            table_cuda.path_out,
            table_cuda.path_weight,
            table_cuda.cg_m1,
            table_cuda.cg_m2,
            table_cuda.cg_M,
            table_cuda.cg_value,
            n_atoms=int(left.shape[0]),
            left_channels=int(left.shape[1]),
            right_channels=int(right.shape[1]),
            out_channels=int(out_channels),
            m1_dim=int(left.shape[2]),
            m2_dim=int(right.shape[2]),
            mout_dim=int(out_m_dim),
            path_count=int(table_cuda.path_count),
            cg_count=int(table_cuda.cg_count),
        )
        return out

    @staticmethod
    def backward(ctx, grad_out):
        global _LAST_TRITON_CG_BACKWARD_BACKEND

        left, right, path_left, path_right, path_out, path_weight, cg_m1, cg_m2, cg_M, cg_value = ctx.saved_tensors
        table = GroupedCGPathTable(
            path_left=path_left,
            path_right=path_right,
            path_out=path_out,
            path_weight=path_weight,
            cg_m1=cg_m1,
            cg_m2=cg_m2,
            cg_M=cg_M,
            cg_value=cg_value,
        )
        grad_out_contiguous = grad_out.contiguous()
        if (
            torch.is_grad_enabled()
            and _triton_enabled()
            and left.is_cuda
            and right.is_cuda
            and grad_out_contiguous.is_cuda
            and left.is_contiguous()
            and right.is_contiguous()
            and left.dtype in (torch.float32, torch.float64)
            and right.dtype == left.dtype
            and grad_out_contiguous.dtype == left.dtype
        ):
            grad_left, grad_right, grad_path_weight = (
                _GroupedProductCGBackwardAutograd.apply(
                    grad_out_contiguous,
                    left,
                    right,
                    path_left,
                    path_right,
                    path_out,
                    path_weight,
                    cg_m1,
                    cg_m2,
                    cg_M,
                    cg_value,
                    int(ctx.out_channels),
                    int(ctx.out_m_dim),
                    int(ctx.block_atoms),
                )
            )
            _LAST_TRITON_CG_BACKWARD_BACKEND = (
                "triton_grouped_product_cg_backward_autograd"
            )
            return (
                grad_left,
                grad_right,
                None,
                None,
                None,
                (
                    grad_path_weight
                    if bool(ctx.compute_path_weight_grad)
                    else None
                ),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        if (
            not torch.is_grad_enabled()
            and _triton_enabled()
            and left.is_cuda
            and right.is_cuda
            and grad_out_contiguous.is_cuda
            and left.is_contiguous()
            and right.is_contiguous()
            and left.dtype in (torch.float32, torch.float64)
            and right.dtype == left.dtype
            and grad_out_contiguous.dtype == left.dtype
        ):
            table_cuda = table.to(device=left.device, dtype=left.dtype)
            grad_left = torch.zeros_like(left)
            grad_right = torch.zeros_like(right)
            grad_path_weight = torch.zeros_like(path_weight)
            grid = (
                int(table_cuda.path_count * table_cuda.cg_count),
                triton.cdiv(int(left.shape[0]), int(ctx.block_atoms)),
            )
            try:
                _grouped_product_cg_backward_kernel[grid](
                    grad_out_contiguous,
                    left,
                    right,
                    grad_left,
                    grad_right,
                    grad_path_weight,
                    table_cuda.path_left,
                    table_cuda.path_right,
                    table_cuda.path_out,
                    table_cuda.path_weight,
                    table_cuda.cg_m1,
                    table_cuda.cg_m2,
                    table_cuda.cg_M,
                    table_cuda.cg_value,
                    n_atoms=int(left.shape[0]),
                    left_channels=int(left.shape[1]),
                    right_channels=int(right.shape[1]),
                    out_channels=int(ctx.out_channels),
                    m1_dim=int(left.shape[2]),
                    m2_dim=int(right.shape[2]),
                    mout_dim=int(ctx.out_m_dim),
                    path_count=int(table_cuda.path_count),
                    cg_count=int(table_cuda.cg_count),
                )
                _LAST_TRITON_CG_BACKWARD_BACKEND = "triton_grouped_product_cg_backward"
                return (
                    grad_left,
                    grad_right,
                    None,
                    None,
                    None,
                    grad_path_weight if bool(ctx.compute_path_weight_grad) else None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            except Exception:
                if os.environ.get("YE3T_DEBUG_TRITON") == "1" or os.environ.get("GNE3_DEBUG_TRITON") == "1":
                    raise
        grad_left, grad_right, grad_path_weight = _grouped_product_cg_backward_reference(
            grad_out_contiguous,
            left,
            right,
            table,
            compute_path_weight_grad=bool(ctx.compute_path_weight_grad),
        )
        _LAST_TRITON_CG_BACKWARD_BACKEND = "torch_reference_backward"
        return (
            grad_left,
            grad_right,
            None,
            None,
            None,
            grad_path_weight if bool(ctx.compute_path_weight_grad) else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _validate_inputs(
    left,
    right,
    table,
    *,
    out_channels,
):
    if left.ndim != 3 or right.ndim != 3:
        raise ValueError("left and right must have shape [n_atoms, channels, m_dim].")
    if left.shape[0] != right.shape[0]:
        raise ValueError("left and right must have the same atom dimension.")
    if out_channels < 1:
        raise ValueError("out_channels must be positive.")
    lengths = {
        int(table.path_left.numel()),
        int(table.path_right.numel()),
        int(table.path_out.numel()),
        int(table.path_weight.numel()),
    }
    if len(lengths) != 1:
        raise ValueError("path arrays must all have the same length.")
    cg_lengths = {
        int(table.cg_m1.numel()),
        int(table.cg_m2.numel()),
        int(table.cg_M.numel()),
        int(table.cg_value.numel()),
    }
    if len(cg_lengths) != 1:
        raise ValueError("CG arrays must all have the same length.")
    if table.path_count == 0 or table.cg_count == 0:
        return
    if int(table.path_left.max().item()) >= int(left.shape[1]) or int(table.path_left.min().item()) < 0:
        raise ValueError("path_left contains an out-of-range channel index.")
    if int(table.path_right.max().item()) >= int(right.shape[1]) or int(table.path_right.min().item()) < 0:
        raise ValueError("path_right contains an out-of-range channel index.")
    if int(table.path_out.max().item()) >= int(out_channels) or int(table.path_out.min().item()) < 0:
        raise ValueError("path_out contains an out-of-range channel index.")
    if int(table.cg_m1.max().item()) >= int(left.shape[2]) or int(table.cg_m1.min().item()) < 0:
        raise ValueError("cg_m1 contains an out-of-range magnetic index.")
    if int(table.cg_m2.max().item()) >= int(right.shape[2]) or int(table.cg_m2.min().item()) < 0:
        raise ValueError("cg_m2 contains an out-of-range magnetic index.")


def grouped_product_cg_reference(
    left,
    right,
    table,
    *,
    out_channels,
    out_m_dim,
):
    """PyTorch reference implementation for one grouped sparse-CG product."""

    _validate_inputs(left, right, table, out_channels=int(out_channels))
    table = table.to(device=left.device, dtype=left.dtype)
    out = torch.zeros((left.shape[0], int(out_channels), int(out_m_dim)), dtype=left.dtype, device=left.device)
    if table.path_count == 0 or table.cg_count == 0:
        return out
    if int(table.cg_M.max().item()) >= int(out_m_dim) or int(table.cg_M.min().item()) < 0:
        raise ValueError("cg_M contains an out-of-range output magnetic index.")
    for path_idx in range(table.path_count):
        lch = int(table.path_left[path_idx].item())
        rch = int(table.path_right[path_idx].item())
        och = int(table.path_out[path_idx].item())
        weight = table.path_weight[path_idx]
        for cg_idx in range(table.cg_count):
            m1 = int(table.cg_m1[cg_idx].item())
            m2 = int(table.cg_m2[cg_idx].item())
            mout = int(table.cg_M[cg_idx].item())
            out[:, och, mout] += weight * table.cg_value[cg_idx] * left[:, lch, m1] * right[:, rch, m2]
    return out


def grouped_product_cg_forward(
    left,
    right,
    table,
    *,
    out_channels,
    out_m_dim,
    block_atoms = 128,
    prefer_triton = True,
    allow_triton_autograd = False,
    validate_inputs = True,
    return_backend = False,
):
    """Evaluate one fixed-angular sparse CG product group.

    Triton is used only for CUDA, real floating tensors, contiguous inputs, and
    no active autograd graph.  Otherwise the PyTorch reference path is used so
    training/debugging remains correct while kernel coverage grows.
    """

    if bool(validate_inputs):
        _validate_inputs(left, right, table, out_channels=int(out_channels))
    if int(out_m_dim) < 1:
        raise ValueError("out_m_dim must be positive.")
    if table.path_count == 0 or table.cg_count == 0:
        out = torch.zeros((left.shape[0], int(out_channels), int(out_m_dim)), dtype=left.dtype, device=left.device)
        return (out, "empty") if return_backend else out
    if bool(validate_inputs) and (int(table.cg_M.max().item()) >= int(out_m_dim) or int(table.cg_M.min().item()) < 0):
        raise ValueError("cg_M contains an out-of-range output magnetic index.")
    can_triton = bool(
        prefer_triton
        and _triton_enabled()
        and left.is_cuda
        and right.is_cuda
        and left.is_contiguous()
        and right.is_contiguous()
        and left.dtype in (torch.float32, torch.float64)
        and right.dtype == left.dtype
    )
    if can_triton:
        table_cuda = table.to(device=left.device, dtype=left.dtype)
        use_autograd_kernel = bool(torch.is_grad_enabled() and (left.requires_grad or right.requires_grad))
        if use_autograd_kernel and bool(allow_triton_autograd):
            try:
                out = _GroupedProductCGTritonAutograd.apply(
                    left,
                    right,
                    table_cuda.path_left,
                    table_cuda.path_right,
                    table_cuda.path_out,
                    table_cuda.path_weight,
                    table_cuda.cg_m1,
                    table_cuda.cg_m2,
                    table_cuda.cg_M,
                    table_cuda.cg_value,
                    int(out_channels),
                    int(out_m_dim),
                    128,
                )
                return (out, "triton_grouped_product_cg_autograd") if return_backend else out
            except Exception:
                if os.environ.get("YE3T_DEBUG_TRITON") == "1" or os.environ.get("GNE3_DEBUG_TRITON") == "1":
                    raise
        if use_autograd_kernel:
            out = grouped_product_cg_reference(left, right, table, out_channels=out_channels, out_m_dim=out_m_dim)
            return (out, "torch_reference") if return_backend else out
        out = torch.zeros((left.shape[0], int(out_channels), int(out_m_dim)), dtype=left.dtype, device=left.device)
        grid = (
            int(table_cuda.path_count * table_cuda.cg_count),
            triton.cdiv(int(left.shape[0]), 128),
        )
        try:
            _grouped_product_cg_kernel[grid](
                left,
                right,
                out,
                table_cuda.path_left,
                table_cuda.path_right,
                table_cuda.path_out,
                table_cuda.path_weight,
                table_cuda.cg_m1,
                table_cuda.cg_m2,
                table_cuda.cg_M,
                table_cuda.cg_value,
                n_atoms=int(left.shape[0]),
                left_channels=int(left.shape[1]),
                right_channels=int(right.shape[1]),
                out_channels=int(out_channels),
                m1_dim=int(left.shape[2]),
                m2_dim=int(right.shape[2]),
                mout_dim=int(out_m_dim),
                path_count=int(table_cuda.path_count),
                cg_count=int(table_cuda.cg_count),
            )
            return (out, "triton_grouped_product_cg") if return_backend else out
        except Exception:
            if os.environ.get("YE3T_DEBUG_TRITON") == "1" or os.environ.get("GNE3_DEBUG_TRITON") == "1":
                raise
            pass
    out = grouped_product_cg_reference(left, right, table, out_channels=out_channels, out_m_dim=out_m_dim)
    return (out, "torch_reference") if return_backend else out


def dense_path_table(
    *,
    left_channels,
    right_channels,
    out_channels,
    cg_entries,
    dtype = torch.float64,
):
    """Build a dense channel-pair path table for tests and simple prototypes."""

    path_left = []
    path_right = []
    path_out = []
    path_weight = []
    idx = 0
    for left_idx in range(int(left_channels)):
        for right_idx in range(int(right_channels)):
            path_left.append(left_idx)
            path_right.append(right_idx)
            path_out.append(idx % int(out_channels))
            path_weight.append(1.0)
            idx += 1
    return GroupedCGPathTable(
        path_left=torch.tensor(path_left, dtype=torch.long),
        path_right=torch.tensor(path_right, dtype=torch.long),
        path_out=torch.tensor(path_out, dtype=torch.long),
        path_weight=torch.tensor(path_weight, dtype=dtype),
        cg_m1=torch.tensor([entry[0] for entry in cg_entries], dtype=torch.long),
        cg_m2=torch.tensor([entry[1] for entry in cg_entries], dtype=torch.long),
        cg_M=torch.tensor([entry[2] for entry in cg_entries], dtype=torch.long),
        cg_value=torch.tensor([entry[3] for entry in cg_entries], dtype=dtype),
    )


__all__ = [
    "GroupedCGPathTable",
    "dense_path_table",
    "grouped_product_cg_forward",
    "grouped_product_cg_reference",
]
