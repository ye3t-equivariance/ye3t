"""OpenEquivariance-backed permutation-projected CG tensor adapter.

This module uses OpenEquivariance's ordered Wigner/CG tensor construction as a
coefficient source, then projects the ordered tensor-product path space onto
the orbit-wise permutation-invariant subspace used by YE3T symmetric powers.

The OpenEquivariance dependency is imported lazily. Importing this module does
not require OpenEquivariance to be installed.
"""
from ye3t._record import recordclass
from itertools import permutations, product
import math


@recordclass(('L', 'ordered_paths', 'symmetric_paths', 'projected_norm', 'smallest_kept_singular_value', 'largest_discarded_singular_value', 'ordered_tensor_shape', 'projected_tensor_shape'), frozen = True)
class OEQProjectedSector:
    """One projected target-L sector."""


@recordclass(('n_in', 'l_in', 'orbit_blocks', 'sectors'), frozen = True)
class OEQSymmetricProjectionResult:
    """Summary of OEQ ordered tensors after YE3T-style orbit projection."""

    @property
    def counts_by_L(self):
        return {
            int(sector.L): int(sector.symmetric_paths)
            for sector in self.sectors
            if int(sector.symmetric_paths) > 0
        }

    @property
    def ordered_counts_by_L(self):
        return {
            int(sector.L): int(sector.ordered_paths)
            for sector in self.sectors
            if int(sector.ordered_paths) > 0
        }


_OEQ_IMPORT_ERROR = None
_OEQ_WIGNER_NJ = None


def _load_oeq_wigner_nj():
    global _OEQ_IMPORT_ERROR, _OEQ_WIGNER_NJ
    if _OEQ_WIGNER_NJ is not None:
        return _OEQ_WIGNER_NJ
    if _OEQ_IMPORT_ERROR is not None:
        raise RuntimeError(f"OpenEquivariance symmetric adapter is unavailable: {_OEQ_IMPORT_ERROR}")
    try:
        import torch
        torch.serialization.add_safe_globals([slice])

        # OEQ's symmetric_contraction module imports GroupMM symbols at module
        # import time, although the Wigner tensor construction used here does
        # not need them. Some JIT builds expose DeviceProp/GPUTimer but not
        # GroupMM, so provide stubs to make the pure tensor-construction path
        # importable without modifying the installed package.
        import openequivariance._torch.extlib as extlib

        def _missing_groupmm(*args, **kwargs):
            raise RuntimeError(
                "OpenEquivariance GroupMM symbols are unavailable in this install; "
                "this adapter only uses the OEQ Wigner tensor construction path."
            )

        if not hasattr(extlib, "GroupMM_F32"):
            extlib.GroupMM_F32 = _missing_groupmm
        if not hasattr(extlib, "GroupMM_F64"):
            extlib.GroupMM_F64 = _missing_groupmm

        try:
            from openequivariance._torch.symmetric_contraction.SymmetricContraction import _wigner_nj
        except ModuleNotFoundError:
            from openequivariance._torch.symmetric_contraction.symmetric_contraction import _wigner_nj
    except Exception as exc:
        _OEQ_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(f"OpenEquivariance symmetric adapter is unavailable: {_OEQ_IMPORT_ERROR}")
    _OEQ_WIGNER_NJ = _wigner_nj
    return _OEQ_WIGNER_NJ


def _load_e3nn_o3():
    try:
        import torch
        torch.serialization.add_safe_globals([slice])
        from e3nn import o3
    except Exception as exc:
        raise RuntimeError(f"e3nn is required for the OpenEquivariance symmetric adapter: {type(exc).__name__}: {exc}")
    return o3


def _default_torch_dtype(dtype):
    if dtype is not None:
        return dtype
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"torch is required for the OpenEquivariance symmetric adapter: {type(exc).__name__}: {exc}")
    return torch.float64


def _natural_o3_irrep_string(l_value):
    l_value = int(l_value)
    return f"1x{l_value}{'e' if l_value % 2 == 0 else 'o'}"


def _orbit_blocks(n_in, l_in):
    blocks_by_key = {}
    ordered_keys = []
    for index, (n_value, l_value) in enumerate(zip(n_in, l_in)):
        key = (int(n_value), int(l_value))
        if key not in blocks_by_key:
            blocks_by_key[key] = []
            ordered_keys.append(key)
        blocks_by_key[key].append(int(index))
    return tuple(tuple(blocks_by_key[key]) for key in ordered_keys if len(blocks_by_key[key]) > 1)


def _permutation_count(orbit_blocks):
    count = 1
    for block in orbit_blocks:
        count *= math.factorial(len(block))
    return int(count)


def _validate_inputs(n_in, l_in):
    n_in = tuple(int(x) for x in n_in)
    l_in = tuple(int(x) for x in l_in)
    if len(n_in) != len(l_in):
        raise ValueError("n_in and l_in must have the same length")
    if not n_in:
        raise ValueError("at least one input leaf is required")
    return n_in, l_in


def _canonicalize_oeq_tensor_axes(tensor, rank):
    """Move OEQ tensors to ``leaf_axes..., output_M_axis, path_axis`` order."""

    rank = int(rank)
    if tensor.ndim == rank + 2:
        # OEQ stores non-scalar outputs as output_M, leaf_axes..., paths.
        axis_order = list(range(1, rank + 1)) + [0, rank + 1]
        return tensor.permute(axis_order).contiguous()
    if tensor.ndim == rank + 1:
        # Scalar outputs are squeezed by OEQ, leaving leaf_axes..., paths.
        # Reinsert the singleton output_M axis before the path axis.
        return tensor.unsqueeze(tensor.ndim - 1).contiguous()
    raise ValueError(
        f"unexpected OEQ tensor rank {tensor.ndim}; expected {rank + 1} or {rank + 2} "
        f"for {rank} leaf axes plus output/path axes"
    )


def openequivariance_ordered_wigner_tensors(
    n_in,
    l_in,
    *,
    dtype=None,
    normalization="component",
    max_output_L=None,
):
    """Return OEQ ordered Wigner tensors keyed by target ``L``.

    The returned tensors have leaf magnetic axes first and OEQ path/multiplicity
    as the final axis. This function intentionally calls OEQ's internal
    ``_wigner_nj`` helper instead of ``U_matrix_real`` so rank-4 cases are not
    affected by ``U_matrix_real``'s hardcoded intermediate filter.
    """

    n_in, l_in = _validate_inputs(n_in, l_in)
    o3 = _load_e3nn_o3()
    wigner_nj = _load_oeq_wigner_nj()
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"torch is required for the OpenEquivariance symmetric adapter: {type(exc).__name__}: {exc}")

    rank = len(l_in)
    dtype = _default_torch_dtype(dtype)
    irreps = [o3.Irreps(_natural_o3_irrep_string(l_value)) for l_value in l_in]
    wigners = wigner_nj(irreps, normalization, None, dtype)
    max_l = sum(int(l_value) for l_value in l_in) if max_output_L is None else int(max_output_L)

    stacks_by_L = {}
    for irrep, _path, basis_tensor in wigners:
        L = int(irrep.l)
        if L > max_l:
            continue
        tensor = basis_tensor.squeeze().unsqueeze(-1)
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor, dtype=dtype)
        stacks_by_L.setdefault(L, []).append(tensor)

    out = {}
    for L, tensors in sorted(stacks_by_L.items()):
        out[int(L)] = _canonicalize_oeq_tensor_axes(torch.cat(tensors, dim=-1), rank)
    return out


def symmetrize_leaf_axes(tensor, orbit_blocks, *, leaf_axis_count=None, max_permutations=5040):
    """Average ``tensor`` over permutations within each repeated-leaf orbit."""

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"torch is required for the OpenEquivariance symmetric adapter: {type(exc).__name__}: {exc}")

    leaf_axis_count = max(0, int(tensor.ndim) - 2) if leaf_axis_count is None else int(leaf_axis_count)
    orbit_blocks = tuple(tuple(int(axis) for axis in block) for block in orbit_blocks)
    for block in orbit_blocks:
        if any(axis < 0 or axis >= leaf_axis_count for axis in block):
            raise ValueError(f"orbit block {block!r} is outside the first {leaf_axis_count} leaf axes")

    permutation_count = _permutation_count(orbit_blocks)
    if permutation_count > int(max_permutations):
        raise TimeoutError(
            f"orbit symmetrizer would require {permutation_count} permutations; "
            f"increase max_permutations if this is intentional"
        )
    if permutation_count <= 1:
        return tensor.clone()

    block_permutations = [tuple(permutations(block)) for block in orbit_blocks]
    accum = torch.zeros_like(tensor)
    axis_count = int(tensor.ndim)
    for choices in product(*block_permutations):
        axis_order = list(range(axis_count))
        for block, permuted in zip(orbit_blocks, choices):
            for dst_axis, src_axis in zip(block, permuted):
                axis_order[int(dst_axis)] = int(src_axis)
        accum = accum + tensor.permute(axis_order)
    return accum / float(permutation_count)


def projected_tensor_rank(tensor, *, tol=1e-10):
    """Return the numerical rank of projected path columns."""

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"torch is required for the OpenEquivariance symmetric adapter: {type(exc).__name__}: {exc}")

    if tensor.numel() == 0 or tensor.shape[-1] == 0:
        return 0
    matrix = tensor.reshape(-1, tensor.shape[-1])
    return int(torch.linalg.matrix_rank(matrix, tol=float(tol)).item())


def projected_tensor_singular_diagnostics(tensor, *, tol=1e-10):
    """Return rank and singular-value diagnostics for a projected tensor."""

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"torch is required for the OpenEquivariance symmetric adapter: {type(exc).__name__}: {exc}")

    if tensor.numel() == 0 or tensor.shape[-1] == 0:
        return {
            "rank": 0,
            "projected_norm": 0.0,
            "smallest_kept_singular_value": 0.0,
            "largest_discarded_singular_value": 0.0,
        }
    matrix = tensor.reshape(-1, tensor.shape[-1])
    singular_values = torch.linalg.svdvals(matrix)
    kept = singular_values[singular_values > float(tol)]
    discarded = singular_values[singular_values <= float(tol)]
    return {
        "rank": int(kept.numel()),
        "projected_norm": float(torch.linalg.norm(tensor).item()),
        "smallest_kept_singular_value": float(kept[-1].item()) if kept.numel() else 0.0,
        "largest_discarded_singular_value": float(discarded[0].item()) if discarded.numel() else 0.0,
    }


def orthonormal_projected_tensor_basis(tensor, *, tol=1e-10):
    """Return an orthonormal column basis for the projected tensor span."""

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"torch is required for the OpenEquivariance symmetric adapter: {type(exc).__name__}: {exc}")

    matrix = tensor.reshape(-1, tensor.shape[-1])
    if matrix.numel() == 0:
        return tensor[..., :0]
    q_matrix, r_matrix = torch.linalg.qr(matrix, mode="reduced")
    diag = torch.abs(torch.diagonal(r_matrix))
    rank = int((diag > float(tol)).sum().item())
    return q_matrix[:, :rank].reshape(*tensor.shape[:-1], rank)


def openequivariance_projected_symmetric_tensors(
    n_in,
    l_in,
    *,
    dtype=None,
    normalization="component",
    tol=1e-10,
    max_output_L=None,
    max_permutations=5040,
    orthonormalize=False,
):
    """Return orbit-projected OEQ tensors keyed by target ``L``."""

    n_in, l_in = _validate_inputs(n_in, l_in)
    orbit_blocks = _orbit_blocks(n_in, l_in)
    ordered = openequivariance_ordered_wigner_tensors(
        n_in,
        l_in,
        dtype=dtype,
        normalization=normalization,
        max_output_L=max_output_L,
    )
    projected = {}
    for L, tensor in ordered.items():
        sym_tensor = symmetrize_leaf_axes(
            tensor,
            orbit_blocks,
            leaf_axis_count=len(l_in),
            max_permutations=max_permutations,
        )
        if orthonormalize:
            sym_tensor = orthonormal_projected_tensor_basis(sym_tensor, tol=tol)
        projected[int(L)] = sym_tensor
    return projected


def openequivariance_projected_symmetric_counts(
    n_in,
    l_in,
    *,
    dtype=None,
    normalization="component",
    tol=1e-10,
    max_output_L=None,
    max_permutations=5040,
):
    """Return projected symmetric multiplicities from OEQ ordered tensors."""

    n_in, l_in = _validate_inputs(n_in, l_in)
    orbit_blocks = _orbit_blocks(n_in, l_in)
    ordered = openequivariance_ordered_wigner_tensors(
        n_in,
        l_in,
        dtype=dtype,
        normalization=normalization,
        max_output_L=max_output_L,
    )
    sectors = []
    for L, tensor in ordered.items():
        projected = symmetrize_leaf_axes(
            tensor,
            orbit_blocks,
            leaf_axis_count=len(l_in),
            max_permutations=max_permutations,
        )
        diagnostics = projected_tensor_singular_diagnostics(projected, tol=tol)
        sectors.append(
            OEQProjectedSector(
                L=int(L),
                ordered_paths=int(tensor.shape[-1]),
                symmetric_paths=int(diagnostics["rank"]),
                projected_norm=float(diagnostics["projected_norm"]),
                smallest_kept_singular_value=float(diagnostics["smallest_kept_singular_value"]),
                largest_discarded_singular_value=float(diagnostics["largest_discarded_singular_value"]),
                ordered_tensor_shape=tuple(int(x) for x in tensor.shape),
                projected_tensor_shape=tuple(int(x) for x in projected.shape),
            )
        )
    return OEQSymmetricProjectionResult(
        n_in=n_in,
        l_in=l_in,
        orbit_blocks=orbit_blocks,
        sectors=tuple(sectors),
    )


__all__ = [
    "OEQProjectedSector",
    "OEQSymmetricProjectionResult",
    "openequivariance_ordered_wigner_tensors",
    "openequivariance_projected_symmetric_counts",
    "openequivariance_projected_symmetric_tensors",
    "orthonormal_projected_tensor_basis",
    "projected_tensor_singular_diagnostics",
    "projected_tensor_rank",
    "symmetrize_leaf_axes",
]
