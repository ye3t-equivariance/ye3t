
"""Runtime-facing generalized ``SO(3) x G_\\nu`` descriptors and linear layers.

This module is intentionally separate from the fast ACE / native ye3t
runtime. It provides a lightweight model/runtime surface for the heavier
generalized path:

- an e3nn-style direct-sum container for joint ``SO(3) x G_\\nu`` irreps
- exact subgroup character and permutation-action queries
- block-diagonal rotation/permutation matrices on the canonical magnetic basis
- a symmetry-preserving linear layer that only mixes multiplicities within
  identical joint irreps

``JointYoungCGProduct`` emits fused irreducible target blocks rather than
unfused coupling-path blocks.  The Young/permutation coordinate maps come from
``ExactLoweredChangeOfGroupBasisBranch.coordinate_matrix`` in
``ye3t.representations.tensor_products``; this module applies those maps in
runtime tensors and validates fused dimensions in tests.

The angular action is represented on the canonical magnetic basis used by the
generalized symbolic builder, so the returned SO(3) matrices are generally
complex Wigner-D matrices.
"""
import hashlib
import json
from functools import lru_cache

import numpy as np
import torch

from ye3t.core.rotation import wigner_D_numeric
from ye3t.core.subtree_dag import cg_exact
from ye3t.execution_plan import (
    YE3T_O3_PRIMARY_CONVENTION,
    YE3T_O3_REAL_TESSERAL_CONVENTION,
    YE3T_PRIMARY_CONVENTION,
    YE3T_REAL_TESSERAL_CONVENTION,
    YE3TCarrierKey,
    YE3TCarrierLayout,
)
from ye3t.representations.generalized_irreps import (
    AngularIrrep,
    CoupledIrrepLabel,
    Partition,
    PermutationIrrep,
    PermutationSubgroup,
    PermutationSubgroupFactor,
)
from ye3t.representations.generalized_sector_data import GeneralizedSectorData
from ye3t.representations.projectors import canonical_irrep_matrices_numeric, symmetric_group_character
from ye3t.representations.young_subgroup_specht_coupling import (
    build_cached_young_subgroup_specht_coupling,
    young_subgroup_specht_coupling_multiplicity,
)
from ye3t._record import recordclass


_PACKED_REAL_PRODUCT_PLAN_SCHEMA = "ye3t_packed_real_product_plan_v1"


def _as_python_float(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Generalized runtime angle helpers currently expect scalar tensors.")
        return float(value.detach().cpu().item())
    return float(value)


def _torch_complex_dtype(dtype):
    if dtype in {torch.complex64, torch.complex128}:
        return dtype
    if dtype == torch.float32:
        return torch.complex64
    return torch.complex128


def _torch_real_dtype(dtype):
    if dtype == torch.float32:
        return torch.float32
    return torch.float64


def _require_numerically_real(matrix, operation):
    """Project roundoff-only imaginary residue or reject a non-real action."""

    if not matrix.is_complex():
        return matrix
    real = matrix.real
    imaginary_max = float(matrix.imag.detach().abs().max().cpu())
    real_scale = max(
        1.0,
        float(real.detach().abs().max().cpu()),
    )
    tolerance = 32.0 * float(torch.finfo(real.dtype).eps) * real_scale
    if imaginary_max > tolerance:
        raise RuntimeError(
            str(operation)
            + " did not produce a real-tesseral action: imaginary residual "
            + str(imaginary_max)
            + " exceeds dtype-derived tolerance "
            + str(tolerance)
        )
    return real


def _cached_generalized_tensor_product_composer():
    from ye3t.representations.tensor_products import CachedGeneralizedTensorProductComposer

    return CachedGeneralizedTensorProductComposer()


def _torch_nnz(tensor):
    return int(torch.count_nonzero(torch.abs(tensor) > 0).item())


def _torch_abs_heatmap_data(tensor):
    tensor_cpu = torch.abs(tensor).detach().cpu()
    if tensor_cpu.ndim != 2:
        raise ValueError(f"Expected a rank-2 tensor for heatmap export, got shape {tuple(tensor_cpu.shape)}.")
    return tuple(
        tuple(float(value) for value in row)
        for row in tensor_cpu.tolist()
    )


def _block_diag(matrices):
    if not matrices:
        return torch.zeros((0, 0), dtype=torch.complex128)
    if len(matrices) == 1:
        return matrices[0]
    return torch.block_diag(*matrices)


def _kron_all(matrices, *, dtype, device):
    if not matrices:
        return torch.ones((1, 1), dtype=dtype, device=device)
    out = matrices[0]
    for matrix in matrices[1:]:
        out = torch.kron(out, matrix)
    return out


@lru_cache(maxsize=None)
def _real_angular_change_cached(l_value):
    l_value = int(l_value)
    dim = 2 * l_value + 1
    cosine_columns = {}
    sine_columns = {}

    def mag_index(m):
        return int(m + l_value)

    for m in range(1, l_value + 1):
        cos_col = [0j] * dim
        sin_col = [0j] * dim
        phase = (-1) ** m
        cos_col[mag_index(-m)] = 1.0 / (2.0 ** 0.5)
        cos_col[mag_index(m)] = complex(phase / (2.0 ** 0.5))
        sin_col[mag_index(-m)] = 1j / (2.0 ** 0.5)
        sin_col[mag_index(m)] = complex(0.0, -phase / (2.0 ** 0.5))
        cosine_columns[int(m)] = tuple(cos_col)
        sine_columns[int(m)] = tuple(sin_col)
    zero_col = [0j] * dim
    zero_col[mag_index(0)] = 1.0
    # Match ye3t.core.tesseral exactly: real coordinates are ordered as
    # cosine(-L..-1), m=0, sine(1..L), represented by array indices -L..L.
    columns = (
        [cosine_columns[m] for m in range(l_value, 0, -1)]
        + [tuple(zero_col)]
        + [sine_columns[m] for m in range(1, l_value + 1)]
    )
    matrix = list(zip(*columns))
    return tuple(tuple(complex(value) for value in row) for row in matrix)


def _real_angular_change(
    l_value,
    *,
    dtype,
    device,
):
    return torch.tensor(_real_angular_change_cached(int(l_value)), dtype=_torch_complex_dtype(dtype), device=device)


@lru_cache(maxsize=None)
def _rotation_matrix_zyz(alpha, beta, gamma):
    ca = np.cos(float(alpha))
    sa = np.sin(float(alpha))
    cb = np.cos(float(beta))
    sb = np.sin(float(beta))
    cg = np.cos(float(gamma))
    sg = np.sin(float(gamma))
    rz_alpha = np.array(
        (
            (ca, -sa, 0.0),
            (sa, ca, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=float,
    )
    ry_beta = np.array(
        (
            (cb, 0.0, sb),
            (0.0, 1.0, 0.0),
            (-sb, 0.0, cb),
        ),
        dtype=float,
    )
    rz_gamma = np.array(
        (
            (cg, -sg, 0.0),
            (sg, cg, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=float,
    )
    return tuple(tuple(float(value) for value in row) for row in (rz_alpha @ ry_beta @ rz_gamma))


@lru_cache(maxsize=None)
def _cached_wigner_d_matrix(
    l_value,
    alpha,
    beta,
    gamma,
):
    matrix = wigner_D_numeric(int(l_value), np.array(_rotation_matrix_zyz(float(alpha), float(beta), float(gamma))))
    rows = int(matrix.shape[0])
    cols = int(matrix.shape[1])
    return tuple(
        tuple(complex(matrix[row, col]) for col in range(cols))
        for row in range(rows)
    )


def _angular_matrix(
    l_value,
    *,
    alpha,
    beta,
    gamma,
    dtype,
    device,
):
    cached = _cached_wigner_d_matrix(
        int(l_value),
        _as_python_float(alpha),
        _as_python_float(beta),
        _as_python_float(gamma),
    )
    return torch.tensor(cached, dtype=_torch_complex_dtype(dtype), device=device)


def _proper_rotation_and_inversion(matrix):
    if isinstance(matrix, torch.Tensor):
        array = matrix.detach().cpu().numpy()
    else:
        array = np.asarray(matrix, dtype=float)
    if array.shape != (3, 3):
        raise ValueError("O(3) transformation must have shape (3, 3)")
    if not np.allclose(array.T @ array, np.eye(3), rtol=0.0, atol=1e-10):
        raise ValueError("O(3) transformation must be orthogonal")
    determinant = float(np.linalg.det(array))
    if not np.isclose(abs(determinant), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("O(3) transformation determinant must be +1 or -1")
    inversion = determinant < 0.0
    rotation = -array if inversion else array
    return rotation, inversion


def _normalize_block_permutations(
    blocks,
    permutations,
):
    if len(blocks) == 1:
        first = permutations
        if isinstance(first, tuple) and first and isinstance(first[0], int):
            return ((tuple(int(x) for x in first),),)
        if isinstance(first, (list, tuple)) and first and isinstance(first[0], (list, tuple)):
            return (tuple(tuple(int(x) for x in perm) for perm in first),)
    if not isinstance(permutations, (list, tuple)):
        raise TypeError("permutations must be a nested sequence matching the generalized-irrep blocks.")
    if len(permutations) != len(blocks):
        raise ValueError(f"Expected {len(blocks)} block permutation entries, got {len(permutations)}.")
    normalized_blocks = []
    for block, block_perms in zip(blocks, permutations, strict=True):
        if not isinstance(block_perms, (list, tuple)):
            raise TypeError("Each block permutation entry must be a sequence of subgroup-factor permutations.")
        factor_perms = tuple(tuple(int(x) for x in perm) for perm in block_perms)
        if len(factor_perms) != len(block.label.permutation.partitions):
            raise ValueError(
                f"Expected {len(block.label.permutation.partitions)} subgroup-factor permutations for "
                f"{block.label.permutation.to_string()}, got {len(factor_perms)}."
            )
        normalized_blocks.append(factor_perms)
    return tuple(normalized_blocks)


def _permutation_matrix(
    permutation_irrep,
    factor_permutations,
    *,
    dtype,
    device,
):
    matrices = []
    for partition, perm in zip(permutation_irrep.partitions, factor_permutations, strict=True):
        representation = canonical_irrep_matrices_numeric(tuple(int(x) for x in partition.parts))
        key = tuple(int(x) for x in perm)
        if key not in representation:
            raise KeyError(f"Permutation {key!r} is not valid for partition {partition.parts!r}.")
        matrices.append(torch.as_tensor(representation[key], dtype=dtype, device=device))
    return _kron_all(matrices, dtype=dtype, device=device)


@recordclass(('mul', 'label'), frozen = True)
class GeneralizedMulIrrep:
    """One multiplicity block of a joint ``SO(3) x G_\\nu`` irrep."""

    def __post_init__(self):
        if int(self.mul) <= 0:
            raise ValueError(f"Multiplicity must be positive, got {self.mul!r}.")
        object.__setattr__(self, "mul", int(self.mul))

    @property
    def block_dim(self):
        return int(self.mul) * int(self.label.dim)

    @property
    def angular_dim(self):
        return int(self.label.angular.dim)

    @property
    def permutation_dim(self):
        return int(self.label.permutation.dim)

    def to_string(self):
        return f"{int(self.mul)}x({self.label.to_string()})"


class GeneralizedIrreps:
    """Direct sum of multiplicity copies of joint ``SO(3) x G_\\nu`` irreps."""

    def __init__(self, blocks):
        materialized = []
        for block in blocks:
            if isinstance(block, GeneralizedMulIrrep):
                materialized.append(block)
            else:
                mul, label = block
                materialized.append(GeneralizedMulIrrep(int(mul), label))
        self.blocks = tuple(materialized)
        if not self.blocks:
            raise ValueError("GeneralizedIrreps requires at least one block.")

    def __len__(self):
        return len(self.blocks)

    def __iter__(self):
        return iter(self.blocks)

    def __getitem__(self, index):
        return self.blocks[index]

    def __repr__(self):
        return f"GeneralizedIrreps({self.to_string()})"

    @property
    def dim(self):
        return sum(int(block.block_dim) for block in self.blocks)

    @property
    def num_irreps(self):
        return sum(int(block.mul) for block in self.blocks)

    def to_string(self):
        return " + ".join(block.to_string() for block in self.blocks)

    def slices(self):
        out = []
        start = 0
        for block in self.blocks:
            stop = start + int(block.block_dim)
            out.append(slice(start, stop))
            start = stop
        return out

    def randn(
        self,
        *size,
        dtype = None,
        device = None,
        normalization = "component",
    ):
        if -1 not in size:
            raise ValueError("GeneralizedIrreps.randn expects one '-1' placeholder for the irrep dimension.")
        if normalization not in {"component", "norm"}:
            raise ValueError(f"Unknown normalization {normalization!r}. Use 'component' or 'norm'.")
        shape = list(size)
        dim_index = shape.index(-1)
        shape[dim_index] = int(self.dim)
        dtype = _torch_real_dtype(dtype)
        out = torch.randn(*shape, dtype=dtype, device=device)
        if normalization == "component":
            return out
        return out / out.norm(dim=dim_index, keepdim=True).clamp_min(torch.finfo(dtype).eps)

    def D_from_angles(
        self,
        *,
        alpha,
        beta,
        gamma,
        dtype = None,
        device = None,
    ):
        dtype = _torch_complex_dtype(dtype)
        blocks = []
        for block in self.blocks:
            angular = _angular_matrix(
                int(block.label.angular.l),
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                dtype=dtype,
                device=device,
            )
            perm_eye = torch.eye(int(block.permutation_dim), dtype=dtype, device=device)
            irreducible = torch.kron(perm_eye, angular)
            blocks.extend(irreducible for _ in range(int(block.mul)))
        return _block_diag(blocks)

    def real_basis_change(
        self,
        *,
        dtype = None,
        device = None,
    ):
        dtype = _torch_complex_dtype(dtype)
        blocks = []
        for block in self.blocks:
            angular_change = _real_angular_change(int(block.label.angular.l), dtype=dtype, device=device)
            perm_eye = torch.eye(int(block.permutation_dim), dtype=dtype, device=device)
            irreducible = torch.kron(perm_eye, angular_change)
            blocks.extend(irreducible for _ in range(int(block.mul)))
        return _block_diag(blocks)

    def D_from_angles_real(
        self,
        *,
        alpha,
        beta,
        gamma,
        dtype = None,
        device = None,
    ):
        dtype = _torch_complex_dtype(dtype)
        change = self.real_basis_change(dtype=dtype, device=device)
        change_inv = torch.linalg.inv(change)
        real_matrix = change_inv @ self.D_from_angles(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            dtype=dtype,
            device=device,
        ) @ change
        return _require_numerically_real(
            real_matrix,
            "D_from_angles_real",
        )

    def D_from_matrix(
        self,
        matrix,
        *,
        dtype=None,
        device=None,
    ):
        """Return the declared O(3) action for one orthogonal matrix."""

        dtype = _torch_complex_dtype(dtype)
        rotation, inversion = _proper_rotation_and_inversion(matrix)
        blocks = []
        for block in self.blocks:
            if inversion and block.label.parity is None:
                raise ValueError(
                    "improper transformations require signed O(3) carrier labels"
                )
            angular = torch.as_tensor(
                wigner_D_numeric(int(block.label.angular.l), rotation),
                dtype=dtype,
                device=device,
            )
            if inversion:
                angular = angular * int(block.label.parity)
            perm_eye = torch.eye(
                int(block.permutation_dim),
                dtype=dtype,
                device=device,
            )
            irreducible = torch.kron(perm_eye, angular)
            blocks.extend(irreducible for _ in range(int(block.mul)))
        return _block_diag(blocks)

    def D_from_matrix_real(
        self,
        matrix,
        *,
        dtype=None,
        device=None,
    ):
        dtype = _torch_complex_dtype(dtype)
        change = self.real_basis_change(dtype=dtype, device=device)
        transformed = (
            torch.linalg.inv(change)
            @ self.D_from_matrix(matrix, dtype=dtype, device=device)
            @ change
        )
        return _require_numerically_real(
            transformed,
            "D_from_matrix_real",
        )

    def D_from_permutation(
        self,
        permutations,
        *,
        dtype = None,
        device = None,
    ):
        dtype = _torch_complex_dtype(dtype)
        normalized = _normalize_block_permutations(self.blocks, permutations)
        blocks = []
        for block, factor_perms in zip(self.blocks, normalized, strict=True):
            perm_matrix = _permutation_matrix(
                block.label.permutation,
                factor_perms,
                dtype=_torch_real_dtype(dtype),
                device=device,
            ).to(dtype=dtype)
            ang_eye = torch.eye(int(block.angular_dim), dtype=dtype, device=device)
            irreducible = torch.kron(perm_matrix, ang_eye)
            blocks.extend(irreducible for _ in range(int(block.mul)))
        return _block_diag(blocks)

    def D_from_permutation_real(
        self,
        permutations,
        *,
        dtype = None,
        device = None,
    ):
        dtype = _torch_complex_dtype(dtype)
        change = self.real_basis_change(dtype=dtype, device=device)
        change_inv = torch.linalg.inv(change)
        real_matrix = change_inv @ self.D_from_permutation(permutations, dtype=dtype, device=device) @ change
        return _require_numerically_real(
            real_matrix,
            "D_from_permutation_real",
        )

    def to_real_basis(
        self,
        features,
    ):
        dtype = _torch_complex_dtype(features.dtype)
        change = self.real_basis_change(dtype=dtype, device=features.device)
        return features.to(dtype) @ torch.linalg.inv(change).T

    def from_real_basis(
        self,
        features,
    ):
        dtype = _torch_complex_dtype(features.dtype)
        change = self.real_basis_change(dtype=dtype, device=features.device)
        return features.to(dtype) @ change.T

    def permutation_character(self, permutations):
        normalized = _normalize_block_permutations(self.blocks, permutations)
        total = 0.0
        for block, factor_perms in zip(self.blocks, normalized, strict=True):
            factor_char = 1
            for partition, perm in zip(block.label.permutation.partitions, factor_perms, strict=True):
                factor_char *= symmetric_group_character(partition, _cycle_type_from_permutation(perm))
            total += float(block.mul) * float(factor_char)
        return complex(total)

    def rotation_character(
        self,
        *,
        alpha,
        beta,
        gamma,
        dtype = None,
        device = None,
    ):
        matrix = self.D_from_angles(alpha=alpha, beta=beta, gamma=gamma, dtype=dtype, device=device)
        return complex(torch.trace(matrix).item())

    def joint_character(
        self,
        *,
        alpha,
        beta,
        gamma,
        permutations,
        dtype = None,
        device = None,
    ):
        matrix = self.D_from_permutation(permutations, dtype=dtype, device=device) @ self.D_from_angles(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            dtype=dtype,
            device=device,
        )
        return complex(torch.trace(matrix).item())


def _cycle_type_from_permutation(perm):
    perm = tuple(int(x) for x in perm)
    visited = [False] * len(perm)
    cycles = []
    for start in range(len(perm)):
        if visited[start]:
            continue
        count = 0
        index = start
        while not visited[index]:
            visited[index] = True
            index = perm[index]
            count += 1
        cycles.append(int(count))
    return tuple(sorted(cycles, reverse=True))


class GeneralizedLinear(torch.nn.Module):
    """Symmetry-preserving linear map on direct sums of joint ``SO(3) x G_\\nu`` irreps."""

    def __init__(
        self,
        irreps_in,
        irreps_out,
        *,
        dtype = torch.float64,
    ):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.weight_dtype = _torch_real_dtype(dtype)
        self._weights = torch.nn.ParameterDict()
        for out_index, out_block in enumerate(self.irreps_out.blocks):
            for in_index, in_block in enumerate(self.irreps_in.blocks):
                if in_block.label != out_block.label:
                    continue
                key = f"{out_index}:{in_index}"
                self._weights[key] = torch.nn.Parameter(
                    torch.randn(
                        int(out_block.mul),
                        int(in_block.mul),
                        dtype=self.weight_dtype,
                    ) / max(int(in_block.mul), 1) ** 0.5
                )

    def forward(self, features):
        if features.shape[-1] != int(self.irreps_in.dim):
            raise ValueError(
                f"Expected trailing feature dimension {self.irreps_in.dim}, got {features.shape[-1]}."
            )
        batch_shape = features.shape[:-1]
        flat = features.reshape(-1, int(self.irreps_in.dim))
        out = flat.new_zeros((flat.shape[0], int(self.irreps_out.dim)))
        in_slices = self.irreps_in.slices()
        out_slices = self.irreps_out.slices()
        for out_index, out_block in enumerate(self.irreps_out.blocks):
            out_slice = out_slices[out_index]
            out_dim = int(out_block.label.dim)
            out_block_accum = flat.new_zeros((flat.shape[0], int(out_block.mul), int(out_dim)))
            for in_index, in_block in enumerate(self.irreps_in.blocks):
                key = f"{out_index}:{in_index}"
                if key not in self._weights:
                    continue
                in_slice = in_slices[in_index]
                in_dim = int(in_block.label.dim)
                in_view = flat[:, in_slice].reshape(flat.shape[0], int(in_block.mul), int(in_dim))
                weight = self._weights[key].to(device=flat.device, dtype=in_view.dtype)
                out_block_accum = out_block_accum + torch.einsum(
                    "oi,bid->bod",
                    weight,
                    in_view,
                )
            out[:, out_slice] = out_block_accum.reshape(flat.shape[0], -1)
        return out.reshape(*batch_shape, int(self.irreps_out.dim))


def _runtime_block_basis_labels(
    sector,
    *,
    L,
    copy_index,
):
    labels = []
    for carrier_index in sector.carrier_index_tuples_by_L[int(L)]:
        key = (int(copy_index), tuple(int(x) for x in carrier_index))
        if key not in sector.lowered_multiplets_by_L[int(L)]:
            continue
        for M in range(-int(L), int(L) + 1):
            labels.append((tuple(int(x) for x in carrier_index), int(M)))
    return tuple(labels)


def _runtime_basis_index_map(
    labels,
):
    return {label: idx for idx, label in enumerate(labels)}


@recordclass(('mul', 'sector', 'L', 'copy_index'), frozen = True)
class GeneralizedExactRuntimeBlock:
    """One exact canonical runtime block for a specific joint-irrep copy."""
    copy_index = 0

    def __post_init__(self):
        if int(self.mul) <= 0:
            raise ValueError(f"Multiplicity must be positive, got {self.mul!r}.")
        object.__setattr__(self, "mul", int(self.mul))
        object.__setattr__(self, "L", int(self.L))
        object.__setattr__(self, "copy_index", int(self.copy_index))
        if int(self.L) not in self.sector.labels_by_L:
            raise ValueError(f"Sector does not contain L={self.L}.")
        if int(self.copy_index) >= len(self.sector.labels_by_L[int(self.L)]):
            raise ValueError(
                f"copy_index={self.copy_index} is out of range for L={self.L}; "
                f"sector has {len(self.sector.labels_by_L[int(self.L)])} copies."
            )

    @property
    def label(self):
        return self.sector.labels_by_L[int(self.L)][int(self.copy_index)]

    @property
    def basis_labels(self):
        return _runtime_block_basis_labels(self.sector, L=int(self.L), copy_index=int(self.copy_index))

    @property
    def irreducible_dim(self):
        return len(self.basis_labels)

    @property
    def block_dim(self):
        return int(self.mul) * int(self.irreducible_dim)

    def to_mul_irrep(self):
        return GeneralizedMulIrrep(int(self.mul), self.label)

    def to_string(self):
        return f"{int(self.mul)}x({self.label.to_string()})"


class GeneralizedExactRuntimeIrreps:
    """Direct sum of exact canonical runtime blocks for the generalized path."""

    def __init__(self, blocks):
        self.blocks = tuple(blocks)
        if not self.blocks:
            raise ValueError("GeneralizedExactRuntimeIrreps requires at least one block.")

    @classmethod
    def from_carrier_layout(cls, layout):
        """Construct one exact parent runtime block from a compiled layout."""

        if not isinstance(layout, YE3TCarrierLayout):
            layout = YE3TCarrierLayout.from_dict(layout)
        key = layout.key
        if str(key.convention_id) not in {
            YE3T_PRIMARY_CONVENTION,
            YE3T_REAL_TESSERAL_CONVENTION,
            YE3T_O3_PRIMARY_CONVENTION,
            YE3T_O3_REAL_TESSERAL_CONVENTION,
        }:
            raise ValueError(
                "Exact parent runtime irreps require a declared YE3T "
                "complex or real-tesseral convention."
            )
        expected_tableaux = int(Partition(tuple(key.partition)).dimension)
        expected_magnetic = int(2 * int(key.rotation_L) + 1)
        if int(layout.tableau_count) != expected_tableaux:
            raise ValueError(
                "Carrier layout tableau count does not match the parent "
                "Young partition dimension."
            )
        if int(layout.magnetic_count) != expected_magnetic:
            raise ValueError(
                "Carrier layout magnetic count does not match rotation_L."
            )
        sector = _direct_parent_runtime_sector(
            nin=tuple(
                "formal_slot_" + str(int(index))
                for index in range(int(key.rank))
            ),
            lin=(0,) * int(key.rank),
            partition=tuple(key.partition),
            output_L=int(key.rotation_L),
            multiplicity=1,
            parity=key.parity,
        )
        irreps = cls(
            (
                GeneralizedExactRuntimeBlock(
                    int(layout.channel_count),
                    sector,
                    int(key.rotation_L),
                    0,
                ),
            )
        )
        if int(irreps.dim) != int(layout.width):
            raise ValueError(
                "Exact parent runtime irreps width does not match the "
                "compiled carrier layout."
            )
        return irreps

    @classmethod
    def from_carrier_records(cls, records):
        """Rebuild exact parent blocks from compiler-emitted carrier records."""

        records = tuple(dict(record) for record in records)
        if not records:
            raise ValueError("Exact parent runtime irreps require carrier records.")
        blocks = []
        expected_start = 0
        for block_index, record in enumerate(records):
            if int(record.get("block_index", -1)) != int(block_index):
                raise ValueError(
                    "Exact parent carrier records must use contiguous block indices."
                )
            if str(record.get("carrier_status")) not in {
                "exact_parent_S_N_x_O3",
                "exact_parent_S_N_x_SO3_legacy",
            }:
                raise ValueError(
                    "Packed products require exact parent carrier records."
                )
            layout_payload = record.get("carrier_layout")
            if layout_payload is None:
                raise ValueError(
                    "Packed product carrier record is missing its exact layout."
                )
            layout = YE3TCarrierLayout.from_dict(layout_payload)
            start, stop = tuple(int(value) for value in record["slice"])
            if start != int(expected_start) or stop - start != int(layout.width):
                raise ValueError(
                    "Packed product carrier records must form one contiguous layout."
                )
            copy_index = int(record.get("copy_index", 0))
            sector = _direct_parent_runtime_sector(
                nin=tuple(
                    "formal_slot_" + str(int(index))
                    for index in range(int(layout.key.rank))
                ),
                lin=(0,) * int(layout.key.rank),
                partition=tuple(layout.key.partition),
                output_L=int(layout.key.rotation_L),
                multiplicity=int(copy_index + 1),
                parity=layout.key.parity,
            )
            block = GeneralizedExactRuntimeBlock(
                int(layout.channel_count),
                sector,
                int(layout.key.rotation_L),
                copy_index,
            )
            if block.label.to_string() != str(record["label"]):
                raise ValueError(
                    "Packed product carrier label does not match its exact layout."
                )
            if int(block.block_dim) != int(layout.width):
                raise ValueError(
                    "Packed product carrier block width does not match its layout."
                )
            blocks.append(block)
            expected_start = stop
        irreps = cls(tuple(blocks))
        if int(irreps.dim) != int(expected_start):
            raise ValueError(
                "Packed product output irreps do not cover their serialized layout."
            )
        return irreps

    def __len__(self):
        return len(self.blocks)

    def __iter__(self):
        return iter(self.blocks)

    def __getitem__(self, index):
        return self.blocks[index]

    @property
    def dim(self):
        return sum(int(block.block_dim) for block in self.blocks)

    def slices(self):
        out = []
        start = 0
        for block in self.blocks:
            stop = start + int(block.block_dim)
            out.append(slice(start, stop))
            start = stop
        return out

    def to_string(self):
        return " + ".join(block.to_string() for block in self.blocks)

    def generalized_irreps(self):
        return GeneralizedIrreps(tuple(block.to_mul_irrep() for block in self.blocks))

    def D_from_matrix(
        self,
        matrix,
        *,
        dtype=None,
        device=None,
    ):
        """Return the O(3) action in exact runtime block order."""

        action = self.generalized_irreps().D_from_matrix(
            matrix,
            dtype=dtype,
            device=device,
        )
        if tuple(action.shape) != (int(self.dim), int(self.dim)):
            raise RuntimeError(
                "generalized O(3) action does not match runtime carrier width"
            )
        return action

    def D_from_matrix_real(
        self,
        matrix,
        *,
        dtype=None,
        device=None,
    ):
        """Return the real-tesseral O(3) action in runtime block order."""

        action = self.generalized_irreps().D_from_matrix_real(
            matrix,
            dtype=dtype,
            device=device,
        )
        if tuple(action.shape) != (int(self.dim), int(self.dim)):
            raise RuntimeError(
                "generalized real O(3) action does not match runtime carrier width"
            )
        return action

    def report(self):
        slices = self.slices()
        return {
            "dim": int(self.dim),
            "num_blocks": len(self.blocks),
            "blocks": tuple(
                {
                    "index": int(index),
                    "label": block.label.to_string(),
                    "mul": int(block.mul),
                    "L": int(block.L),
                    "copy_index": int(block.copy_index),
                    "basis_dim": int(block.irreducible_dim),
                    "block_dim": int(block.block_dim),
                    "slice": (int(slices[index].start), int(slices[index].stop)),
                    "basis_labels": tuple(block.basis_labels),
                }
                for index, block in enumerate(self.blocks)
            ),
        }

    def format_report(self):
        report = self.report()
        lines = [
            "GeneralizedExactRuntimeIrreps",
            f"dim={report['dim']} num_blocks={report['num_blocks']}",
        ]
        for block in report["blocks"]:
            lines.append(
                "block[{index}] {label} mul={mul} basis_dim={basis_dim} slice={slice}".format(**block)
            )
        return "\n".join(lines)


@recordclass(('left_block_index', 'right_block_index', 'output_block_index', 'output_L', 'source_tensor', 'coordinate_matrix', 'source_basis_labels', 'target_basis_labels', 'young_multiplicity', 'source_assembly'), frozen = True)
class _FullTensorInstruction:
    young_multiplicity = 1
    source_assembly = None


@recordclass(
    ('left_block_index', 'right_block_index', 'output_block_index'),
    frozen=True,
)
class _PackedBilinearInstruction:
    """Carrier-block binding for an exact precompiled sparse product."""


@recordclass(('irreps_in1', 'irreps_in2', 'irreps_out', 'instructions', 'tree_type', 'product_policy', 'composer_cache', 'provenance'), frozen = True)
class JointYoungCGTensorSchedule:
    """Static lowered tensor schedule for an exact joint Young-CG product."""

    def coefficient_tensors(self, instruction_index = None, *, basis = "canonical"):
        """Return exact CG, Young/intertwiner, and joint coefficient tensors."""

        return _coefficient_tensors_for_instructions(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            self.instructions,
            instruction_index=instruction_index,
            basis=basis,
        )

    def coefficient_report(self, *, include_values = False, tol = 0.0, instruction_index = None):
        """Return exact coefficient metadata for the lowered schedule."""

        return _coefficient_report_for_instructions(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            self.instructions,
            tree_type=str(self.tree_type),
            product_policy=self.product_policy,
            include_values=include_values,
            tol=tol,
            instruction_index=instruction_index,
            composer_cache=self.composer_cache,
            provenance=self.provenance,
        )

    def manifest(self):
        """Return deterministic schedule provenance, policy, cache, and coefficient hashes."""

        return _static_schedule_manifest(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            self.instructions,
            tree_type=str(self.tree_type),
            product_policy=self.product_policy,
            composer_cache=self.composer_cache,
            provenance=self.provenance,
        )


def _select_instruction_indices(instructions, instruction_index):
    if instruction_index is None:
        return tuple(range(len(instructions)))
    if isinstance(instruction_index, int):
        return (int(instruction_index),)
    return tuple(int(index) for index in instruction_index)


def _instruction_young_multiplicity(instruction):
    return int(getattr(instruction, "young_multiplicity", 1))


def _exact_joint_product_fallback_report():
    return {
        "runtime": "torch_complex_exact_schedule",
        "clebsch_gordan_source": "ye3t.core.subtree_dag.cg_exact",
        "young_coupling_source": "ExactLoweredChangeOfGroupBasisBranch.coordinate_matrix",
        "joint_coupling_source": "young_coupling_matrix @ clebsch_gordan_tensor",
        "uses_scalar_proxy": False,
        "uses_norm_shortcut": False,
        "uses_approximate_intertwiner": False,
        "uses_dropped_sector_standin": False,
        "accelerator_backend": None,
        "fallbacks": (),
    }


def _real_joint_product_report():
    report = dict(_exact_joint_product_fallback_report())
    report["runtime"] = "torch_real_basis_exact_schedule"
    report["joint_coupling_source"] = "real-basis transform of young_coupling_matrix @ clebsch_gordan_tensor"
    return report


def _joint_coefficient_tensor(instruction):
    return torch.einsum(
        "ts,sij->tij",
        instruction.coordinate_matrix.to(torch.complex128),
        instruction.source_tensor.to(torch.complex128),
    )


def _magnetic_basis_change_for_labels(labels, L, *, dtype, device):
    labels = tuple(labels)
    L = int(L)
    angular_change = _real_angular_change(L, dtype=dtype, device=device)
    real_labels = []
    seen = set()
    for label in labels:
        carrier = label[:-1]
        if carrier in seen:
            continue
        seen.add(carrier)
        for real_index in range(2 * L + 1):
            real_labels.append((carrier, int(real_index)))
    canonical_index = {
        (label[:-1], int(label[-1])): idx
        for idx, label in enumerate(labels)
    }
    change = torch.zeros((len(labels), len(real_labels)), dtype=angular_change.dtype, device=device)
    for col, (carrier, real_index) in enumerate(real_labels):
        for M in range(-L, L + 1):
            row = canonical_index.get((carrier, int(M)))
            if row is not None:
                change[row, col] = angular_change[M + L, int(real_index)]
    return change


def _real_instruction_tensors(left_block, right_block, out_block, instruction, *, dtype, device):
    complex_dtype = _torch_complex_dtype(dtype)
    left_change = _magnetic_basis_change_for_labels(
        left_block.basis_labels,
        int(left_block.L),
        dtype=complex_dtype,
        device=device,
    )
    right_change = _magnetic_basis_change_for_labels(
        right_block.basis_labels,
        int(right_block.L),
        dtype=complex_dtype,
        device=device,
    )
    source_change = _magnetic_basis_change_for_labels(
        instruction.source_basis_labels,
        int(instruction.output_L),
        dtype=complex_dtype,
        device=device,
    )
    out_change = _magnetic_basis_change_for_labels(
        instruction.target_basis_labels,
        int(out_block.L),
        dtype=complex_dtype,
        device=device,
    )
    source_inv = torch.linalg.inv(source_change)
    out_inv = torch.linalg.inv(out_change)
    cg_phase = -1j if (int(left_block.L) + int(right_block.L) - int(instruction.output_L)) % 2 else 1.0
    source_tensor = instruction.source_tensor.to(device=device, dtype=complex_dtype)
    coordinate_matrix = instruction.coordinate_matrix.to(device=device, dtype=complex_dtype)
    real_cg = torch.einsum(
        "as,sij,ib,jc->abc",
        source_inv,
        source_tensor * cg_phase,
        left_change,
        right_change,
    )
    real_young = torch.einsum(
        "at,ts,sb->ab",
        out_inv,
        coordinate_matrix,
        source_change,
    )
    real_joint = torch.einsum("ts,sij->tij", real_young, real_cg)
    tol = 5.0e-5 if dtype == torch.float32 else 1.0e-12
    if real_joint.numel() and float(torch.max(torch.abs(real_joint.imag)).detach().cpu().item()) > tol:
        raise ValueError("Real-basis joint_coupling_tensor has a non-negligible imaginary component.")
    cg_tensor = real_cg.real.to(dtype=dtype)
    young_matrix = real_young.real.to(dtype=dtype)
    if real_cg.numel() and float(torch.max(torch.abs(real_cg.imag)).detach().cpu().item()) > tol:
        cg_tensor = real_cg
    if real_young.numel() and float(torch.max(torch.abs(real_young.imag)).detach().cpu().item()) > tol:
        young_matrix = real_young
    return {
        "clebsch_gordan_tensor": cg_tensor,
        "young_coupling_matrix": young_matrix,
        "joint_coupling_tensor": real_joint.real.to(dtype=dtype),
    }


def _rank3_sparse_entries(tensor, labels0, labels1, labels2, *, names, tol):
    entries = []
    tol = float(tol)
    tensor = tensor.detach().cpu()
    for idx0 in range(int(tensor.shape[0])):
        for idx1 in range(int(tensor.shape[1])):
            for idx2 in range(int(tensor.shape[2])):
                value = complex(tensor[idx0, idx1, idx2].item())
                if abs(value) <= tol:
                    continue
                entries.append(
                    {
                        f"{names[0]}_index": int(idx0),
                        f"{names[1]}_index": int(idx1),
                        f"{names[2]}_index": int(idx2),
                        f"{names[0]}_label": labels0[idx0],
                        f"{names[1]}_label": labels1[idx1],
                        f"{names[2]}_label": labels2[idx2],
                        "value": _complex_payload(value),
                    }
                )
    return tuple(entries)


def _matrix_sparse_entries(matrix, row_labels, col_labels, *, row_name, col_name, tol):
    entries = []
    tol = float(tol)
    matrix = matrix.detach().cpu()
    for row in range(int(matrix.shape[0])):
        for col in range(int(matrix.shape[1])):
            value = complex(matrix[row, col].item())
            if abs(value) <= tol:
                continue
            entries.append(
                {
                    f"{row_name}_index": int(row),
                    f"{col_name}_index": int(col),
                    f"{row_name}_label": row_labels[row],
                    f"{col_name}_label": col_labels[col],
                    "value": _complex_payload(value),
                }
            )
    return tuple(entries)


def _coefficient_tensors_for_instruction(irreps_in1, irreps_in2, irreps_out, instruction, index, *, basis = "canonical"):
    left_block = irreps_in1.blocks[int(instruction.left_block_index)]
    right_block = irreps_in2.blocks[int(instruction.right_block_index)]
    output_block = irreps_out.blocks[int(instruction.output_block_index)]
    basis = str(basis)
    if basis not in {"canonical", "real"}:
        raise ValueError("basis must be one of {'canonical', 'real'}.")
    if basis == "real":
        tensors = _real_instruction_tensors(
            left_block,
            right_block,
            output_block,
            instruction,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        cg_tensor = tensors["clebsch_gordan_tensor"]
        young_matrix = tensors["young_coupling_matrix"]
        joint_tensor = tensors["joint_coupling_tensor"]
    else:
        cg_tensor = instruction.source_tensor.detach().clone()
        young_matrix = instruction.coordinate_matrix.detach().clone()
        joint_tensor = _joint_coefficient_tensor(instruction).detach().clone()
    return {
        "index": int(index),
        "left_block_index": int(instruction.left_block_index),
        "right_block_index": int(instruction.right_block_index),
        "output_block_index": int(instruction.output_block_index),
        "output_label": output_block.label.to_string(),
        "output_L": int(instruction.output_L),
        "young_multiplicity": _instruction_young_multiplicity(instruction),
        "left_basis_labels": tuple(left_block.basis_labels),
        "right_basis_labels": tuple(right_block.basis_labels),
        "source_basis_labels": tuple(instruction.source_basis_labels),
        "target_basis_labels": tuple(instruction.target_basis_labels),
        "basis": basis,
        "clebsch_gordan_tensor": cg_tensor,
        "young_coupling_matrix": young_matrix,
        "joint_coupling_tensor": joint_tensor,
    }


def _coefficient_tensors_for_instructions(irreps_in1, irreps_in2, irreps_out, instructions, instruction_index = None, *, basis = "canonical"):
    selected = _select_instruction_indices(instructions, instruction_index)
    out = []
    for index in selected:
        out.append(
            _coefficient_tensors_for_instruction(
                irreps_in1,
                irreps_in2,
                irreps_out,
                instructions[index],
                index,
                basis=basis,
            )
        )
    return tuple(out)


def _coefficient_summary_for_instruction(irreps_in1, irreps_in2, irreps_out, instruction, index, *, include_values, tol):
    tensors = _coefficient_tensors_for_instruction(irreps_in1, irreps_in2, irreps_out, instruction, index)
    cg_tensor = tensors["clebsch_gordan_tensor"]
    young_matrix = tensors["young_coupling_matrix"]
    joint_tensor = tensors["joint_coupling_tensor"]
    cg = {
        "source": "ye3t.core.subtree_dag.cg_exact",
        "shape": tuple(int(x) for x in cg_tensor.shape),
        "nnz": _torch_nnz(cg_tensor),
        "coefficient_hash": _tensor_value_hash(cg_tensor),
    }
    source_assembly = getattr(instruction, "source_assembly", None)
    young_source = (
        "YoungSubgroupSpechtCoupling.C_dagger_L_v"
        if source_assembly is not None
        else "ExactLoweredChangeOfGroupBasisBranch.coordinate_matrix"
    )
    young = {
        "source": young_source,
        "shape": tuple(int(x) for x in young_matrix.shape),
        "nnz": _torch_nnz(young_matrix),
        "subgroup_multiplicity": int(tensors["young_multiplicity"]),
        "coefficient_hash": _tensor_value_hash(young_matrix),
    }
    joint = {
        "source": "young_coupling_matrix @ clebsch_gordan_tensor",
        "shape": tuple(int(x) for x in joint_tensor.shape),
        "nnz": _torch_nnz(joint_tensor),
        "coefficient_hash": _tensor_value_hash(joint_tensor),
    }
    if include_values:
        cg["entries"] = _rank3_sparse_entries(
            cg_tensor,
            tensors["source_basis_labels"],
            tensors["left_basis_labels"],
            tensors["right_basis_labels"],
            names=("source", "left_basis", "right_basis"),
            tol=tol,
        )
        young["entries"] = _matrix_sparse_entries(
            young_matrix,
            tensors["target_basis_labels"],
            tensors["source_basis_labels"],
            row_name="target_basis",
            col_name="source",
            tol=tol,
        )
        joint["entries"] = _rank3_sparse_entries(
            joint_tensor,
            tensors["target_basis_labels"],
            tensors["left_basis_labels"],
            tensors["right_basis_labels"],
            names=("target_basis", "left_basis", "right_basis"),
            tol=tol,
        )
    return {
        "index": int(index),
        "left_block_index": int(tensors["left_block_index"]),
        "right_block_index": int(tensors["right_block_index"]),
        "output_block_index": int(tensors["output_block_index"]),
        "output_label": tensors["output_label"],
        "output_L": int(tensors["output_L"]),
        "left_basis_dim": len(tensors["left_basis_labels"]),
        "right_basis_dim": len(tensors["right_basis_labels"]),
        "source_basis_dim": len(tensors["source_basis_labels"]),
        "target_basis_dim": len(tensors["target_basis_labels"]),
        "young_multiplicity": int(tensors["young_multiplicity"]),
        "source_assembly": None if source_assembly is None else dict(source_assembly),
        "clebsch_gordan": cg,
        "young_coupling": young,
        "joint_coupling": joint,
        "source_basis_labels": tuple(tensors["source_basis_labels"]),
        "target_basis_labels": tuple(tensors["target_basis_labels"]),
    }


def _coefficient_report_for_instructions(
    irreps_in1,
    irreps_in2,
    irreps_out,
    instructions,
    *,
    tree_type,
    product_policy,
    include_values = False,
    tol = 0.0,
    instruction_index = None,
    composer_cache = None,
    provenance = None,
):
    selected = _select_instruction_indices(instructions, instruction_index)
    instruction_reports = tuple(
        _coefficient_summary_for_instruction(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions[index],
            index,
            include_values=include_values,
            tol=tol,
        )
        for index in selected
    )
    coefficient_hash = _stable_payload_hash(
        tuple(
            (
                report["index"],
                report["clebsch_gordan"]["coefficient_hash"],
                report["young_coupling"]["coefficient_hash"],
                report["joint_coupling"]["coefficient_hash"],
            )
            for report in instruction_reports
        )
    )
    young_sources = tuple(
        dict.fromkeys(report["young_coupling"]["source"] for report in instruction_reports)
    )
    young_source = young_sources[0] if len(young_sources) == 1 else "instruction_specific_young_coupling"
    return {
        "tree_type": str(tree_type),
        "num_instructions": len(instructions),
        "selected_instruction_indices": selected,
        "coefficient_hash": coefficient_hash,
        "coefficient_sources": {
            "clebsch_gordan": "ye3t.core.subtree_dag.cg_exact",
            "young_coupling": young_source,
            "joint_coupling": "young_coupling_matrix @ clebsch_gordan_tensor",
        },
        "product_policy": product_policy,
        "composer_cache": composer_cache,
        "provenance": provenance,
        "instructions": instruction_reports,
    }


def _static_schedule_manifest(
    irreps_in1,
    irreps_in2,
    irreps_out,
    instructions,
    *,
    tree_type,
    product_policy,
    composer_cache,
    provenance = None,
):
    coefficient_report = _coefficient_report_for_instructions(
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        tree_type=str(tree_type),
        product_policy=product_policy,
        include_values=False,
        composer_cache=composer_cache,
        provenance=provenance,
    )
    payload = {
        "format": "joint_young_cg_tensor_schedule_manifest_v1",
        "tree_type": str(tree_type),
        "num_instructions": len(instructions),
        "irreps_in1": irreps_in1.report(),
        "irreps_in2": irreps_in2.report(),
        "irreps_out": irreps_out.report(),
        "product_policy": product_policy,
        "composer_cache": composer_cache,
        "fallback_report": _exact_joint_product_fallback_report(),
        "coefficient_hash": coefficient_report["coefficient_hash"],
        "instruction_hashes": tuple(
            {
                "index": report["index"],
                "clebsch_gordan": report["clebsch_gordan"]["coefficient_hash"],
                "young_coupling": report["young_coupling"]["coefficient_hash"],
                "joint_coupling": report["joint_coupling"]["coefficient_hash"],
            }
            for report in coefficient_report["instructions"]
        ),
    }
    payload["manifest_hash"] = _stable_payload_hash(payload)
    return payload


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (int, float, str)) or value is None:
        return value
    return str(value)


def _stable_payload_hash(payload):
    encoded = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _complex_payload(value):
    value = complex(value)
    return {"real": float(value.real), "imag": float(value.imag)}


def _tensor_value_hash(tensor):
    tensor = tensor.detach().to(torch.complex128).cpu().contiguous()
    values = torch.view_as_real(tensor).reshape(-1).tolist()
    return _stable_payload_hash(
        {
            "dtype": "complex128",
            "shape": tuple(int(x) for x in tensor.shape),
            "values": tuple(float(x) for x in values),
        }
    )


def _sector_rank(sector):
    if hasattr(sector, "nin"):
        return len(tuple(sector.nin))
    if hasattr(sector, "subgroup"):
        return int(sector.subgroup.degree)
    if hasattr(sector, "permutation_irrep"):
        return int(sector.permutation_irrep.subgroup.degree)
    raise ValueError("Could not infer rank for generalized sector.")


def exact_runtime_carrier_records(
    irreps,
    convention_id=YE3T_REAL_TESSERAL_CONVENTION,
):
    """Describe exact parent carriers and identify unresolved subgroup blocks."""

    records = []
    slices = irreps.slices()
    for block_index, block in enumerate(irreps.blocks):
        permutation = block.label.permutation
        rank = int(_sector_rank(block.sector))
        partitions = tuple(
            tuple(int(value) for value in partition.parts)
            for partition in tuple(permutation.partitions)
        )
        exact_partition = None
        if len(partitions) == 1 and sum(partitions[0]) == int(rank):
            exact_partition = tuple(int(value) for value in partitions[0])
        record = {
            "block_index": int(block_index),
            "label": block.label.to_string(),
            "rank": int(rank),
            "L_R": int(block.L),
            "copy_index": int(block.copy_index),
            "multiplicity": int(block.mul),
            "permutation_representation": (
                "trivial"
                if permutation.is_totally_symmetric()
                else "nontrivial"
            ),
            "permutation_irrep": permutation.to_string(),
            "permutation_partitions": [
                [int(value) for value in partition]
                for partition in partitions
            ],
            "component_dim": int(block.irreducible_dim),
            "block_dim": int(block.block_dim),
            "slice": (
                int(slices[int(block_index)].start),
                int(slices[int(block_index)].stop),
            ),
            "convention_id": str(convention_id),
            "parity": block.label.parity,
        }
        if exact_partition is None:
            record["carrier_status"] = "unresolved_subgroup_product_not_parent_S_N"
            record["carrier_key"] = None
            record["carrier_layout"] = None
        else:
            magnetic_count = int(2 * int(block.L) + 1)
            if int(block.irreducible_dim) % int(magnetic_count):
                raise ValueError(
                    "Exact runtime block width is incompatible with separate tableau and magnetic axes."
                )
            tableau_count = int(block.irreducible_dim) // int(magnetic_count)
            key = YE3TCarrierKey(
                rank=int(rank),
                partition=exact_partition,
                rotation_L=int(block.L),
                convention_id=str(convention_id),
                parity=block.label.parity,
            )
            layout = YE3TCarrierLayout(
                key=key,
                channel_count=int(block.mul),
                tableau_count=int(tableau_count),
                magnetic_count=int(magnetic_count),
            )
            if int(layout.width) != int(block.block_dim):
                raise ValueError("Exact carrier layout width does not match the runtime block.")
            record["carrier_status"] = (
                "exact_parent_S_N_x_O3"
                if block.label.parity is not None
                else "exact_parent_S_N_x_SO3_legacy"
            )
            record["carrier_key"] = key.to_dict()
            record["carrier_layout"] = layout.to_dict()
        records.append(record)
    return tuple(records)


def _permutation_irrep_key(permutation_irrep):
    return permutation_irrep.to_string()


def _target_label_key(label):
    return label.to_string()


def _target_L_from_key(key):
    text = str(key)
    if not text.startswith("L="):
        return None
    head = text.split(" ", 1)[0]
    head = head.split(",", 1)[0]
    try:
        return int(head[2:])
    except ValueError:
        return None


def _target_permutation_from_key(key):
    text = str(key)
    marker = " x "
    if marker not in text:
        return None
    return text.split(marker, 1)[1].split("#", 1)[0]


def _normalize_label_key_set(values, key_function):
    if values is None:
        return None
    out = set()
    for value in values:
        if hasattr(value, "to_string"):
            out.add(key_function(value))
        else:
            out.add(str(value))
    return frozenset(out)


def _normalize_target_parity(value):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "any", "all", "none", "so3"}:
            return None
        if text in {"even", "+", "+1", "1", "positive", "gerade", "g"}:
            return 1
        if text in {"odd", "-", "-1", "negative", "ungerade", "u"}:
            return -1
    numeric = int(value)
    if numeric == 1:
        return 1
    if numeric == -1:
        return -1
    raise ValueError("target_parity must be one of None/'any', 'even'/+1, or 'odd'/-1.")


def _sector_total_parity(sector):
    label_parities = {
        label.parity
        for labels in getattr(sector, "labels_by_L", {}).values()
        for label in labels
    }
    signed = {value for value in label_parities if value is not None}
    if signed:
        if None in label_parities or len(signed) != 1:
            raise ValueError("generalized sector mixes incompatible spatial parity labels")
        return int(next(iter(signed)))
    if not hasattr(sector, "lin"):
        raise ValueError("Cannot infer target parity for a generalized sector without lin metadata.")
    return 1 if (sum(int(l_value) for l_value in tuple(sector.lin)) % 2 == 0) else -1


def _validate_permutation_policy(permutation_policy, allowed_permutation_irreps):
    policy = "mixed_character" if permutation_policy is None else str(permutation_policy)
    if policy not in {"mixed_character", "trivial_only", "explicit"}:
        raise ValueError(
            "permutation_policy must be 'mixed_character', 'trivial_only', or 'explicit', "
            f"got {permutation_policy!r}."
        )
    allow_keys = _normalize_label_key_set(allowed_permutation_irreps, _permutation_irrep_key)
    if policy == "explicit" and allow_keys is None:
        raise ValueError("permutation_policy='explicit' requires allowed_permutation_irreps.")
    return policy, allow_keys


def _target_allowed_by_joint_policy(
    label,
    sector,
    output_L,
    *,
    requested_target_keys,
    rank_cap,
    L_max,
    target_parity,
    permutation_policy,
    allowed_permutation_keys,
):
    if L_max is not None and int(output_L) > int(L_max):
        return False
    if rank_cap is not None and _sector_rank(sector) > int(rank_cap):
        return False
    if requested_target_keys is not None and _target_label_key(label) not in requested_target_keys:
        return False
    if target_parity is not None and _sector_total_parity(sector) != int(target_parity):
        return False

    permutation = label.permutation
    if permutation_policy == "trivial_only" and not permutation.is_totally_symmetric():
        return False
    if allowed_permutation_keys is not None and _permutation_irrep_key(permutation) not in allowed_permutation_keys:
        return False
    return True


def _exact_parent_partition_for_block(block):
    rank = int(_sector_rank(block.sector))
    partitions = tuple(
        tuple(int(value) for value in partition.parts)
        for partition in block.label.permutation.partitions
    )
    if len(partitions) != 1 or sum(partitions[0]) != rank:
        return None
    carrier_labels = []
    for carrier, _ in block.basis_labels:
        carrier = tuple(int(value) for value in carrier)
        if carrier not in carrier_labels:
            carrier_labels.append(carrier)
    expected = tuple((index,) for index in range(int(block.label.permutation.dim)))
    if tuple(carrier_labels) != expected:
        return None
    return tuple(partitions[0])


def _parent_partition_from_permutation_key(key, rank):
    text = str(key)
    prefix = f"S_{int(rank)}:["
    if not text.startswith(prefix) or not text.endswith("]") or " x " in text:
        return None
    body = text[len(prefix) : -1].strip()
    if not body:
        return None
    try:
        partition = tuple(int(value.strip()) for value in body.split(","))
    except ValueError:
        return None
    if any(value <= 0 for value in partition) or sum(partition) != int(rank):
        return None
    if tuple(sorted(partition, reverse=True)) != partition:
        return None
    return partition


def _direct_parent_runtime_sector(
    nin,
    lin,
    partition,
    output_L,
    multiplicity,
    parity=None,
):
    nin = tuple(nin)
    lin = tuple(int(value) for value in lin)
    rank = len(nin)
    if len(lin) != rank:
        raise ValueError("Direct parent runtime sector requires aligned nin/lin metadata.")
    output_L = int(output_L)
    multiplicity = int(multiplicity)
    subgroup = PermutationSubgroup(
        (
            PermutationSubgroupFactor(
                channel_label="formal_slots",
                l=0,
                multiplicity=rank,
            ),
        )
    )
    permutation = PermutationIrrep(
        subgroup=subgroup,
        partitions=(Partition(tuple(partition)),),
    )
    labels = tuple(
        CoupledIrrepLabel(
            angular=AngularIrrep(output_L),
            permutation=permutation,
            multiplicity_index=rho,
            parity=parity,
        )
        for rho in range(multiplicity)
    )
    tableau_count = int(permutation.dim)
    carrier_indices = tuple((index,) for index in range(tableau_count))
    lowered = {
        (rho, carrier): True
        for rho in range(multiplicity)
        for carrier in carrier_indices
    }
    sector_dimension = multiplicity * tableau_count * (2 * output_L + 1)
    return GeneralizedSectorData(
        nin=nin,
        lin=lin,
        permutation_irrep=permutation,
        subgroup=subgroup,
        raw_dim=sector_dimension,
        projected_dim=sector_dimension,
        projector_matrix=None,
        factor_projectors=tuple(),
        labels_by_L={output_L: labels},
        highest_weight_isotypic_basis_by_L={},
        highest_weight_isotypic_dim_by_L={output_L: multiplicity * tableau_count},
        joint_multiplicity_by_L={output_L: multiplicity},
        sector_dim_by_L={output_L: sector_dimension},
        isotypic_symbol_by_L={},
        joint_symbol_by_L={},
        factor_action_matrices_by_L={},
        factor_matrix_units_by_L={},
        full_matrix_units_by_L={},
        carrier_index_tuples_by_L={output_L: carrier_indices},
        canonical_copy_basis_by_L={},
        canonical_highest_weight_vectors_by_L={},
        lowered_multiplets_by_L={output_L: lowered},
        codepath="direct_parent_young_subduction_runtime_sector",
        notes=(
            "Runtime-only exact parent carrier; no raw tensor projector was materialized.",
            "Tableau coordinates use the validated Young-orthogonal subduction ordering.",
        ),
    )


def _direct_parent_source_tensor(left_block, right_block, output_L):
    output_L = int(output_L)
    left_labels = tuple(left_block.basis_labels)
    right_labels = tuple(right_block.basis_labels)
    left_index = _runtime_basis_index_map(left_labels)
    right_index = _runtime_basis_index_map(right_labels)
    left_tableau_count = int(left_block.label.permutation.dim)
    right_tableau_count = int(right_block.label.permutation.dim)
    magnetic_count = 2 * output_L + 1
    source_labels = []
    for left_t in range(left_tableau_count):
        for right_t in range(right_tableau_count):
            for M_out in range(-output_L, output_L + 1):
                source_labels.append(
                    (
                        (int(left_block.copy_index), (left_t,)),
                        (int(right_block.copy_index), (right_t,)),
                        int(M_out),
                    )
                )
    source_tensor = torch.zeros(
        (len(source_labels), int(left_block.irreducible_dim), int(right_block.irreducible_dim)),
        dtype=torch.complex128,
    )
    for source_index, source_label in enumerate(source_labels):
        ((_, left_carrier), (_, right_carrier), M_out) = source_label
        for left_M in range(-int(left_block.L), int(left_block.L) + 1):
            right_M = int(M_out - left_M)
            if right_M < -int(right_block.L) or right_M > int(right_block.L):
                continue
            coefficient = complex(
                cg_exact(
                    int(left_block.L),
                    int(left_M),
                    int(right_block.L),
                    int(right_M),
                    output_L,
                    int(M_out),
                ).evalf()
            )
            source_tensor[
                source_index,
                left_index[(tuple(left_carrier), int(left_M))],
                right_index[(tuple(right_carrier), int(right_M))],
            ] = coefficient
    return source_tensor, tuple(source_labels), left_tableau_count * right_tableau_count, magnetic_count


def _direct_parent_subduction_product(
    irreps_in1,
    irreps_in2,
    *,
    requested_target_keys,
    rank_cap,
    L_max,
    target_parity,
    permutation_policy,
    allowed_permutation_keys,
):
    if requested_target_keys is None:
        return None
    left_partitions = tuple(_exact_parent_partition_for_block(block) for block in irreps_in1.blocks)
    right_partitions = tuple(_exact_parent_partition_for_block(block) for block in irreps_in2.blocks)
    if any(partition is None for partition in left_partitions + right_partitions):
        return None
    total_rank = sum(left_partitions[0]) + sum(right_partitions[0])
    requested = []
    for key in sorted(requested_target_keys):
        output_L = _target_L_from_key(key)
        permutation_key = _target_permutation_from_key(key)
        partition = _parent_partition_from_permutation_key(permutation_key, total_rank)
        if output_L is None or partition is None:
            return None
        requested.append((str(key), int(output_L), partition))

    output_blocks = []
    output_block_key_to_index = {}
    instructions = []
    coupling_cache = {}
    for left_index, (left_block, left_partition) in enumerate(zip(irreps_in1.blocks, left_partitions, strict=True)):
        for right_index, (right_block, right_partition) in enumerate(zip(irreps_in2.blocks, right_partitions, strict=True)):
            if (left_block.label.parity is None) != (right_block.label.parity is None):
                raise ValueError(
                    "cannot mix signed O(3) and unspecified SO(3)-legacy runtime blocks"
                )
            if sum(left_partition) + sum(right_partition) != total_rank:
                return None
            min_L = abs(int(left_block.L) - int(right_block.L))
            max_L = int(left_block.L) + int(right_block.L)
            for requested_key, output_L, target_partition in requested:
                if output_L < min_L or output_L > max_L:
                    continue
                if L_max is not None and output_L > int(L_max):
                    continue
                if rank_cap is not None and total_rank > int(rank_cap):
                    continue
                subgroup_partitions = (tuple(left_partition), tuple(right_partition))
                multiplicity = young_subgroup_specht_coupling_multiplicity(
                    subgroup_partitions,
                    target_partition,
                )
                if multiplicity <= 0:
                    continue
                coupling_key = (subgroup_partitions, tuple(target_partition))
                coupling = coupling_cache.get(coupling_key)
                if coupling is None:
                    coupling = build_cached_young_subgroup_specht_coupling(
                        subgroup_partitions,
                        target_partition,
                    )
                    coupling_cache[coupling_key] = coupling
                nin = tuple(getattr(left_block.sector, "nin", ())) + tuple(
                    getattr(right_block.sector, "nin", ())
                )
                lin = tuple(getattr(left_block.sector, "lin", ())) + tuple(
                    getattr(right_block.sector, "lin", ())
                )
                if len(nin) != total_rank or len(lin) != total_rank:
                    return None
                sector = _direct_parent_runtime_sector(
                    nin,
                    lin,
                    target_partition,
                    output_L,
                    multiplicity,
                    parity=(
                        None
                        if left_block.label.parity is None
                        else int(left_block.label.parity) * int(right_block.label.parity)
                    ),
                )
                coefficient_matrix = coupling.coefficient_matrix()
                source_tensor, source_labels, child_dimension, magnetic_count = _direct_parent_source_tensor(
                    left_block,
                    right_block,
                    output_L,
                )
                if int(coefficient_matrix.rows) != int(coupling.induced_dim):
                    raise ValueError("Young subduction matrix row count does not match its induced basis.")
                if child_dimension > int(coefficient_matrix.rows):
                    raise ValueError("Canonical child placement exceeds the induced Young basis.")
                source_assembly = {
                    "kind": "rooted_motif",
                    "placement": "canonical_binary_tree_identity_coset",
                    "source_dimension": int(child_dimension),
                    "induced_dimension": int(coupling.induced_dim),
                    "active_induced_rows": tuple(range(int(child_dimension))),
                    "analysis_orientation": "C_dagger_L_v",
                    "coefficient_backend": str(coupling.tensor.coefficient_backend),
                    "coefficient_codepath": str(coupling.tensor.codepath),
                    "validation": coupling.as_dict()["validation"],
                }
                for rho in range(multiplicity):
                    candidate_block = GeneralizedExactRuntimeBlock(
                        mul=int(left_block.mul) * int(right_block.mul),
                        sector=sector,
                        L=output_L,
                        copy_index=rho,
                    )
                    candidate_key = _target_label_key(candidate_block.label)
                    if "#" in requested_key:
                        requested_match = candidate_key == requested_key
                    else:
                        requested_match = candidate_key.split("#", 1)[0] == requested_key
                    if not requested_match:
                        continue
                    if not _target_allowed_by_joint_policy(
                        candidate_block.label,
                        sector,
                        output_L,
                        requested_target_keys=None,
                        rank_cap=rank_cap,
                        L_max=L_max,
                        target_parity=target_parity,
                        permutation_policy=permutation_policy,
                        allowed_permutation_keys=allowed_permutation_keys,
                    ):
                        continue
                    output_key = candidate_block.label.to_string()
                    output_block_index = output_block_key_to_index.get(output_key)
                    if output_block_index is None:
                        output_block_index = len(output_blocks)
                        output_blocks.append(candidate_block)
                        output_block_key_to_index[output_key] = output_block_index
                    target_dimension = int(candidate_block.label.permutation.dim)
                    coordinate_matrix = torch.zeros(
                        (target_dimension * magnetic_count, child_dimension * magnetic_count),
                        dtype=torch.complex128,
                    )
                    for target_t in range(target_dimension):
                        coefficient_column = rho * target_dimension + target_t
                        for child_coordinate in range(child_dimension):
                            value = complex(coefficient_matrix[child_coordinate, coefficient_column])
                            for magnetic_index in range(magnetic_count):
                                coordinate_matrix[
                                    target_t * magnetic_count + magnetic_index,
                                    child_coordinate * magnetic_count + magnetic_index,
                                ] = value
                    target_labels = tuple(
                        (rho, (target_t,), M_out)
                        for target_t in range(target_dimension)
                        for M_out in range(-output_L, output_L + 1)
                    )
                    instructions.append(
                        _FullTensorInstruction(
                            left_block_index=left_index,
                            right_block_index=right_index,
                            output_block_index=output_block_index,
                            output_L=output_L,
                            source_tensor=source_tensor,
                            coordinate_matrix=coordinate_matrix,
                            source_basis_labels=source_labels,
                            target_basis_labels=target_labels,
                            young_multiplicity=multiplicity,
                            source_assembly=source_assembly,
                        )
                    )
    if not output_blocks:
        return None
    return tuple(output_blocks), tuple(instructions)


def _match_input_gradient_dtype(grad, original):
    grad = grad.reshape(original.shape)
    if original.is_complex():
        return grad.to(dtype=original.dtype)
    return grad.real.to(dtype=original.dtype)


def _real_joint_tensor_for_instruction(product, instruction, dtype, device):
    cache = getattr(product, "_real_instruction_cache", None)
    if cache is None:
        cache = {}
        product._real_instruction_cache = cache
    key = (
        int(instruction.left_block_index),
        int(instruction.right_block_index),
        int(instruction.output_block_index),
        str(dtype),
        str(device),
    )
    cached = cache.get(key)
    if cached is not None:
        return cached
    left_block = product.irreps_in1.blocks[int(instruction.left_block_index)]
    right_block = product.irreps_in2.blocks[int(instruction.right_block_index)]
    out_block = product.irreps_out.blocks[int(instruction.output_block_index)]
    tensor = _real_instruction_tensors(
        left_block,
        right_block,
        out_block,
        instruction,
        dtype=dtype,
        device=device,
    )["joint_coupling_tensor"]
    cache[key] = tensor
    return tensor


def _packed_real_bilinear_table(product):
    cached = getattr(product, "_packed_real_bilinear_table_cache", None)
    if cached is not None:
        return cached
    from ye3t.backends.triton_joint import SparseBilinearTable

    left_slices = product.irreps_in1.slices()
    right_slices = product.irreps_in2.slices()
    output_slices = product.irreps_out.slices()
    left_indices = []
    right_indices = []
    output_indices = []
    coefficients = []
    dropped_max_abs = 0.0
    retained_min_abs = None
    for instruction in product._instructions:
        left_block = product.irreps_in1.blocks[int(instruction.left_block_index)]
        right_block = product.irreps_in2.blocks[int(instruction.right_block_index)]
        output_block = product.irreps_out.blocks[int(instruction.output_block_index)]
        if int(output_block.mul) != int(left_block.mul) * int(right_block.mul):
            raise ValueError(
                "Packed real Young-CG lowering requires output multiplicity to equal "
                "left multiplicity times right multiplicity."
            )
        tensor = _real_joint_tensor_for_instruction(
            product,
            instruction,
            torch.float64,
            torch.device("cpu"),
        ).detach().cpu()
        scale = float(torch.max(torch.abs(tensor))) if int(tensor.numel()) else 0.0
        threshold = max(1.0, scale) * 64.0 * torch.finfo(torch.float64).eps
        retained = torch.abs(tensor) > float(threshold)
        if bool(torch.any(~retained)):
            dropped_max_abs = max(
                float(dropped_max_abs),
                float(torch.max(torch.abs(tensor[~retained]))),
            )
        nonzero = torch.nonzero(retained, as_tuple=False)
        if int(nonzero.shape[0]) == 0:
            continue
        values = tensor[retained].to(torch.float64)
        retained_min = float(torch.min(torch.abs(values)))
        retained_min_abs = retained_min if retained_min_abs is None else min(retained_min_abs, retained_min)
        left_start = int(left_slices[int(instruction.left_block_index)].start)
        right_start = int(right_slices[int(instruction.right_block_index)].start)
        output_start = int(output_slices[int(instruction.output_block_index)].start)
        target = nonzero[:, 0]
        left_irrep = nonzero[:, 1]
        right_irrep = nonzero[:, 2]
        for left_copy in range(int(left_block.mul)):
            for right_copy in range(int(right_block.mul)):
                left_indices.append(
                    left_start
                    + int(left_copy) * int(left_block.irreducible_dim)
                    + left_irrep
                )
                right_indices.append(
                    right_start
                    + int(right_copy) * int(right_block.irreducible_dim)
                    + right_irrep
                )
                output_indices.append(
                    output_start
                    + (
                        int(left_copy) * int(right_block.mul)
                        + int(right_copy)
                    )
                    * int(output_block.irreducible_dim)
                    + target
                )
                coefficients.append(values)
    if coefficients:
        left_index = torch.cat(tuple(left_indices)).to(torch.long)
        right_index = torch.cat(tuple(right_indices)).to(torch.long)
        output_index = torch.cat(tuple(output_indices)).to(torch.long)
        coefficient = torch.cat(tuple(coefficients)).to(torch.float64)
    else:
        left_index = torch.empty((0,), dtype=torch.long)
        right_index = torch.empty((0,), dtype=torch.long)
        output_index = torch.empty((0,), dtype=torch.long)
        coefficient = torch.empty((0,), dtype=torch.float64)
    table = SparseBilinearTable(
        left_index=left_index,
        right_index=right_index,
        output_index=output_index,
        coefficient=coefficient,
        output_width=int(product.irreps_out.dim),
    )
    product._packed_real_bilinear_table_cache = table
    product._packed_real_bilinear_report = {
        "table_kind": "exact_sparse_bilinear_joint_young_angular",
        "term_count": int(table.term_count),
        "left_width": int(product.irreps_in1.dim),
        "right_width": int(product.irreps_in2.dim),
        "output_width": int(product.irreps_out.dim),
        "instruction_count": int(len(product._instructions)),
        "coefficient_prune_rule": "abs(value) > max(1,max_abs)*64*float64_epsilon",
        "dropped_max_abs": float(dropped_max_abs),
        "retained_min_abs": None if retained_min_abs is None else float(retained_min_abs),
        "basis": "declared_real_unitary_transform_of_complex_primary_convention",
        "logical_output_axes": (
            "channel_or_multiplicity",
            "tableau_t",
            "magnetic_M",
        ),
        "learned_map_axis": "channel_or_multiplicity",
    }
    return table


def _packed_real_bilinear_enabled(product, left, right):
    return bool(
        getattr(product, "use_packed_cuda", True)
        and left.is_cuda
        and right.is_cuda
        and left.dtype in (torch.float32, torch.float64)
        and right.dtype == left.dtype
    )


def _joint_young_cg_product_forward_real(product, features_left, features_right):
    if features_left.is_complex() or features_right.is_complex():
        raise ValueError("Real-basis Young-CG product expects real input coordinates.")
    if features_left.shape[-1] != int(product.irreps_in1.dim):
        raise ValueError(
            f"Expected left trailing dimension {product.irreps_in1.dim}, got {features_left.shape[-1]}."
        )
    if features_right.shape[-1] != int(product.irreps_in2.dim):
        raise ValueError(
            f"Expected right trailing dimension {product.irreps_in2.dim}, got {features_right.shape[-1]}."
        )
    batch_shape = features_left.shape[:-1]
    if batch_shape != features_right.shape[:-1]:
        raise ValueError("Left and right batch shapes must match.")
    dtype = _torch_real_dtype(features_left.dtype)
    left_flat = features_left.reshape(-1, int(product.irreps_in1.dim)).to(dtype=dtype)
    right_flat = features_right.reshape(-1, int(product.irreps_in2.dim)).to(dtype=dtype)
    if (
        bool(getattr(product, "_packed_plan_only", False))
        or _packed_real_bilinear_enabled(product, left_flat, right_flat)
    ):
        from ye3t.backends.triton_joint import sparse_bilinear_forward

        output = sparse_bilinear_forward(
            left_flat.contiguous(),
            right_flat.contiguous(),
            _packed_real_bilinear_table(product),
            prefer_triton=bool(getattr(product, "use_packed_cuda", True)),
            strict=bool(getattr(product, "strict_packed_cuda", False)),
        )
        product._last_real_product_backend = "packed_sparse_bilinear"
        return output.reshape(*batch_shape, int(product.irreps_out.dim))
    left_slices = product.irreps_in1.slices()
    right_slices = product.irreps_in2.slices()
    out_chunks = [left_flat.new_zeros((left_flat.shape[0], int(block.block_dim))) for block in product.irreps_out.blocks]

    for instruction in product._instructions:
        left_block = product.irreps_in1.blocks[instruction.left_block_index]
        right_block = product.irreps_in2.blocks[instruction.right_block_index]
        out_block = product.irreps_out.blocks[instruction.output_block_index]
        left_view = left_flat[:, left_slices[instruction.left_block_index]].reshape(
            left_flat.shape[0],
            int(left_block.mul),
            int(left_block.irreducible_dim),
        )
        right_view = right_flat[:, right_slices[instruction.right_block_index]].reshape(
            right_flat.shape[0],
            int(right_block.mul),
            int(right_block.irreducible_dim),
        )
        joint_tensor = _real_joint_tensor_for_instruction(product, instruction, dtype, left_flat.device)
        out_block_coeffs = torch.einsum(
            "tij,bai,bcj->bact",
            joint_tensor,
            left_view,
            right_view,
        ).reshape(left_flat.shape[0], int(out_block.block_dim))
        out_chunks[instruction.output_block_index] = out_chunks[instruction.output_block_index] + out_block_coeffs

    out = torch.cat(out_chunks, dim=-1) if out_chunks else left_flat.new_zeros((left_flat.shape[0], 0))
    product._last_real_product_backend = "torch_instruction_einsum"
    return out.reshape(*batch_shape, int(product.irreps_out.dim))


def _joint_young_cg_product_backward_real(product, features_left, features_right, grad_output):
    dtype = _torch_real_dtype(features_left.dtype)
    left_flat = features_left.reshape(-1, int(product.irreps_in1.dim)).to(dtype=dtype)
    right_flat = features_right.reshape(-1, int(product.irreps_in2.dim)).to(dtype=dtype)
    grad_out_flat = grad_output.reshape(-1, int(product.irreps_out.dim)).to(dtype=dtype)
    if (
        bool(getattr(product, "_packed_plan_only", False))
        or _packed_real_bilinear_enabled(product, left_flat, right_flat)
    ):
        from ye3t.backends.triton_joint import sparse_bilinear_backward

        grad_left, grad_right = sparse_bilinear_backward(
            grad_out_flat.contiguous(),
            left_flat.contiguous(),
            right_flat.contiguous(),
            _packed_real_bilinear_table(product),
            prefer_triton=bool(getattr(product, "use_packed_cuda", True)),
            strict=bool(getattr(product, "strict_packed_cuda", False)),
        )
        product._last_real_product_backward_backend = "packed_sparse_bilinear"
        return (
            _match_input_gradient_dtype(grad_left, features_left),
            _match_input_gradient_dtype(grad_right, features_right),
        )
    left_slices = product.irreps_in1.slices()
    right_slices = product.irreps_in2.slices()
    out_slices = product.irreps_out.slices()
    grad_left_flat = torch.zeros_like(left_flat)
    grad_right_flat = torch.zeros_like(right_flat)

    for instruction in product._instructions:
        left_block = product.irreps_in1.blocks[instruction.left_block_index]
        right_block = product.irreps_in2.blocks[instruction.right_block_index]
        out_block = product.irreps_out.blocks[instruction.output_block_index]
        left_slice = left_slices[instruction.left_block_index]
        right_slice = right_slices[instruction.right_block_index]
        out_slice = out_slices[instruction.output_block_index]
        left_view = left_flat[:, left_slice].reshape(
            left_flat.shape[0],
            int(left_block.mul),
            int(left_block.irreducible_dim),
        )
        right_view = right_flat[:, right_slice].reshape(
            right_flat.shape[0],
            int(right_block.mul),
            int(right_block.irreducible_dim),
        )
        grad_block = grad_out_flat[:, out_slice].reshape(
            grad_out_flat.shape[0],
            int(left_block.mul),
            int(right_block.mul),
            int(out_block.irreducible_dim),
        )
        joint_tensor = _real_joint_tensor_for_instruction(product, instruction, dtype, left_flat.device)
        grad_left = torch.einsum("tij,bact,bcj->bai", joint_tensor, grad_block, right_view)
        grad_right = torch.einsum("tij,bact,bai->bcj", joint_tensor, grad_block, left_view)
        grad_left_flat[:, left_slice] = grad_left_flat[:, left_slice] + grad_left.reshape(left_flat.shape[0], -1)
        grad_right_flat[:, right_slice] = grad_right_flat[:, right_slice] + grad_right.reshape(right_flat.shape[0], -1)

    product._last_real_product_backward_backend = "torch_instruction_einsum"
    return (
        _match_input_gradient_dtype(grad_left_flat, features_left),
        _match_input_gradient_dtype(grad_right_flat, features_right),
    )


def _joint_young_cg_product_forward(product, features_left, features_right):
    if bool(getattr(product, "use_real_basis_product", False)):
        return _joint_young_cg_product_forward_real(product, features_left, features_right)
    if features_left.shape[-1] != int(product.irreps_in1.dim):
        raise ValueError(
            f"Expected left trailing dimension {product.irreps_in1.dim}, got {features_left.shape[-1]}."
        )
    if features_right.shape[-1] != int(product.irreps_in2.dim):
        raise ValueError(
            f"Expected right trailing dimension {product.irreps_in2.dim}, got {features_right.shape[-1]}."
        )
    batch_shape = features_left.shape[:-1]
    if batch_shape != features_right.shape[:-1]:
        raise ValueError("Left and right batch shapes must match.")
    left_flat = features_left.reshape(-1, int(product.irreps_in1.dim)).to(torch.complex128)
    right_flat = features_right.reshape(-1, int(product.irreps_in2.dim)).to(torch.complex128)
    left_slices = product.irreps_in1.slices()
    right_slices = product.irreps_in2.slices()
    out_chunks = [left_flat.new_zeros((left_flat.shape[0], int(block.block_dim))) for block in product.irreps_out.blocks]

    for instruction in product._instructions:
        left_block = product.irreps_in1.blocks[instruction.left_block_index]
        right_block = product.irreps_in2.blocks[instruction.right_block_index]
        out_block = product.irreps_out.blocks[instruction.output_block_index]
        left_view = left_flat[:, left_slices[instruction.left_block_index]].reshape(
            left_flat.shape[0],
            int(left_block.mul),
            int(left_block.irreducible_dim),
        )
        right_view = right_flat[:, right_slices[instruction.right_block_index]].reshape(
            right_flat.shape[0],
            int(right_block.mul),
            int(right_block.irreducible_dim),
        )
        source = torch.einsum(
            "sij,bai,bcj->bacs",
            instruction.source_tensor.to(device=left_flat.device),
            left_view,
            right_view,
        ).reshape(left_flat.shape[0], int(left_block.mul) * int(right_block.mul), -1)
        out_block_coeffs = torch.einsum(
            "ts,bps->bpt",
            instruction.coordinate_matrix.to(device=left_flat.device),
            source,
        ).reshape(left_flat.shape[0], int(out_block.block_dim))
        out_chunks[instruction.output_block_index] = out_chunks[instruction.output_block_index] + out_block_coeffs

    out = torch.cat(out_chunks, dim=-1) if out_chunks else left_flat.new_zeros((left_flat.shape[0], 0))
    return out.reshape(*batch_shape, int(product.irreps_out.dim))


def _joint_young_cg_product_backward(product, features_left, features_right, grad_output):
    if bool(getattr(product, "use_real_basis_product", False)):
        return _joint_young_cg_product_backward_real(product, features_left, features_right, grad_output)
    left_flat = features_left.reshape(-1, int(product.irreps_in1.dim)).to(torch.complex128)
    right_flat = features_right.reshape(-1, int(product.irreps_in2.dim)).to(torch.complex128)
    grad_out_flat = grad_output.reshape(-1, int(product.irreps_out.dim)).to(torch.complex128)
    left_slices = product.irreps_in1.slices()
    right_slices = product.irreps_in2.slices()
    out_slices = product.irreps_out.slices()
    grad_left_flat = torch.zeros_like(left_flat)
    grad_right_flat = torch.zeros_like(right_flat)

    for instruction in product._instructions:
        left_block = product.irreps_in1.blocks[instruction.left_block_index]
        right_block = product.irreps_in2.blocks[instruction.right_block_index]
        out_block = product.irreps_out.blocks[instruction.output_block_index]
        left_slice = left_slices[instruction.left_block_index]
        right_slice = right_slices[instruction.right_block_index]
        out_slice = out_slices[instruction.output_block_index]
        left_view = left_flat[:, left_slice].reshape(
            left_flat.shape[0],
            int(left_block.mul),
            int(left_block.irreducible_dim),
        )
        right_view = right_flat[:, right_slice].reshape(
            right_flat.shape[0],
            int(right_block.mul),
            int(right_block.irreducible_dim),
        )
        grad_block = grad_out_flat[:, out_slice].reshape(
            grad_out_flat.shape[0],
            int(left_block.mul),
            int(right_block.mul),
            int(out_block.irreducible_dim),
        )
        source_tensor = instruction.source_tensor.to(device=left_flat.device)
        coordinate_matrix = instruction.coordinate_matrix.to(device=left_flat.device)
        grad_source = torch.einsum("ts,bact->bacs", coordinate_matrix.conj(), grad_block)
        grad_left = torch.einsum("sij,bacs,bcj->bai", source_tensor.conj(), grad_source, right_view.conj())
        grad_right = torch.einsum("sij,bacs,bai->bcj", source_tensor.conj(), grad_source, left_view.conj())
        grad_left_flat[:, left_slice] = grad_left_flat[:, left_slice] + grad_left.reshape(left_flat.shape[0], -1)
        grad_right_flat[:, right_slice] = grad_right_flat[:, right_slice] + grad_right.reshape(right_flat.shape[0], -1)

    return (
        _match_input_gradient_dtype(grad_left_flat, features_left),
        _match_input_gradient_dtype(grad_right_flat, features_right),
    )


class _JointYoungCGStreamingVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features_left, features_right, product):
        ctx.product = product
        ctx.save_for_backward(features_left, features_right)
        return _joint_young_cg_product_forward(product, features_left, features_right)

    @staticmethod
    def backward(ctx, grad_output):
        features_left, features_right = ctx.saved_tensors
        grad_left = None
        grad_right = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grad_left, grad_right = _JointYoungCGStreamingVJPBackward.apply(
                features_left,
                features_right,
                grad_output,
                ctx.product,
            )
        if not ctx.needs_input_grad[0]:
            grad_left = None
        if not ctx.needs_input_grad[1]:
            grad_right = None
        return grad_left, grad_right, None


def _zero_like_or_none(value, template):
    if value is not None:
        return value
    return torch.zeros_like(template)


class _JointYoungCGStreamingVJPBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features_left, features_right, grad_output, product):
        ctx.product = product
        ctx.save_for_backward(features_left, features_right, grad_output)
        return _joint_young_cg_product_backward(product, features_left, features_right, grad_output)

    @staticmethod
    def backward(ctx, grad_grad_left, grad_grad_right):
        features_left, features_right, grad_output = ctx.saved_tensors
        product = ctx.product
        grad_features_left = None
        grad_features_right = None
        grad_grad_output = None

        if grad_grad_right is not None and (ctx.needs_input_grad[0] or ctx.needs_input_grad[2]):
            left_term, _ = _joint_young_cg_product_backward(
                product,
                features_left,
                grad_grad_right,
                grad_output,
            )
            if ctx.needs_input_grad[0]:
                grad_features_left = _zero_like_or_none(grad_features_left, features_left) + left_term
            if ctx.needs_input_grad[2]:
                output_term = _joint_young_cg_product_forward(product, features_left, grad_grad_right)
                grad_grad_output = _zero_like_or_none(grad_grad_output, grad_output) + output_term

        if grad_grad_left is not None and (ctx.needs_input_grad[1] or ctx.needs_input_grad[2]):
            _, right_term = _joint_young_cg_product_backward(
                product,
                grad_grad_left,
                features_right,
                grad_output,
            )
            if ctx.needs_input_grad[1]:
                grad_features_right = _zero_like_or_none(grad_features_right, features_right) + right_term
            if ctx.needs_input_grad[2]:
                output_term = _joint_young_cg_product_forward(product, grad_grad_left, features_right)
                grad_grad_output = _zero_like_or_none(grad_grad_output, grad_output) + output_term

        return grad_features_left, grad_features_right, grad_grad_output, None


class GeneralizedFullTensorProduct(torch.nn.Module):
    """Exact runtime/module analogue of e3nn's FullTensorProduct for ``SO(3) x G_\\nu``."""

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        *,
        tree_type = "balanced",
        composer = None,
        requested_targets = None,
        rank_cap = None,
        L_max = None,
        permutation_policy = "mixed_character",
        allowed_permutation_irreps = None,
        output_mode = "fused_irrep",
        target_parity = None,
    ):
        super().__init__()
        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.composer = composer if composer is not None else _cached_generalized_tensor_product_composer()
        self._composer_cache_report = None
        self.use_streaming_vjp = False
        self.use_real_basis_product = False
        self.use_packed_cuda = True
        self.strict_packed_cuda = False
        self._real_instruction_cache = {}
        self._packed_real_bilinear_table_cache = None
        self._packed_real_bilinear_report = None
        self._packed_plan_only = False
        self._packed_real_product_plan = None
        self._static_schedule_provenance = None
        self._last_real_product_backend = None
        self._last_real_product_backward_backend = None
        self.tree_type = str(tree_type)
        self.requested_target_keys = _normalize_label_key_set(requested_targets, _target_label_key)
        self.requested_target_Ls = None
        if self.requested_target_keys is not None:
            parsed_Ls = {_target_L_from_key(key) for key in self.requested_target_keys}
            self.requested_target_Ls = frozenset(int(L) for L in parsed_Ls if L is not None)
        self.requested_target_permutation_keys = None
        if self.requested_target_keys is not None:
            parsed_permutations = {_target_permutation_from_key(key) for key in self.requested_target_keys}
            self.requested_target_permutation_keys = frozenset(
                str(value) for value in parsed_permutations if value is not None
            )
        self.permutation_policy, self.allowed_permutation_keys = _validate_permutation_policy(
            permutation_policy,
            allowed_permutation_irreps,
        )
        self.rank_cap = None if rank_cap is None else int(rank_cap)
        self.L_max = None if L_max is None else int(L_max)
        self.target_parity = _normalize_target_parity(target_parity)
        self.output_mode = "fused_irrep" if output_mode is None else str(output_mode)
        if self.output_mode != "fused_irrep":
            raise ValueError("Only output_mode='fused_irrep' is supported.")
        output_blocks = []
        output_block_key_to_index = {}
        instructions = []
        self._parent_subduction_backend = "legacy_raw_generalized_projector"
        direct_parent_product = _direct_parent_subduction_product(
            self.irreps_in1,
            self.irreps_in2,
            requested_target_keys=self.requested_target_keys,
            rank_cap=self.rank_cap,
            L_max=self.L_max,
            target_parity=self.target_parity,
            permutation_policy=self.permutation_policy,
            allowed_permutation_keys=self.allowed_permutation_keys,
        )
        if direct_parent_product is not None:
            direct_output_blocks, direct_instructions = direct_parent_product
            self.irreps_out = GeneralizedExactRuntimeIrreps(direct_output_blocks)
            self._instructions = tuple(direct_instructions)
            self._parent_subduction_backend = "validated_cached_parent_subduction_C_dagger_L_v"
            return

        for left_index, left_block in enumerate(self.irreps_in1.blocks):
            left_label_index = _runtime_basis_index_map(left_block.basis_labels)
            left_dim = int(left_block.irreducible_dim)
            for right_index, right_block in enumerate(self.irreps_in2.blocks):
                right_label_index = _runtime_basis_index_map(right_block.basis_labels)
                right_dim = int(right_block.irreducible_dim)
                min_L = abs(int(left_block.L) - int(right_block.L))
                max_L = int(left_block.L) + int(right_block.L)
                for output_L in range(int(min_L), int(max_L) + 1):
                    if self.requested_target_Ls is not None and int(output_L) not in self.requested_target_Ls:
                        continue
                    lowered_map = self.composer.lowered_change_of_group_basis_map(
                        left_block.sector,
                        right_block.sector,
                        left_L=int(left_block.L),
                        right_L=int(right_block.L),
                        output_L=int(output_L),
                        tree_type=str(tree_type),
                        target_permutation_keys=self.requested_target_permutation_keys,
                    )
                    for branch in lowered_map.branches:
                        source_cols = [
                            idx
                            for idx, label in enumerate(branch.source_basis_labels)
                            if int(label[0][0]) == int(left_block.copy_index) and int(label[1][0]) == int(right_block.copy_index)
                        ]
                        if not source_cols:
                            continue
                        target_copy_indices = sorted({int(label[0]) for label in branch.target_basis_labels})
                        for target_copy_index in target_copy_indices:
                            row_indices = [
                                idx for idx, label in enumerate(branch.target_basis_labels) if int(label[0]) == int(target_copy_index)
                            ]
                            if not row_indices:
                                continue
                            target_label = branch.output_sector.labels_by_L[int(output_L)][int(target_copy_index)]
                            if not _target_allowed_by_joint_policy(
                                target_label,
                                branch.output_sector,
                                int(output_L),
                                requested_target_keys=self.requested_target_keys,
                                rank_cap=self.rank_cap,
                                L_max=self.L_max,
                                target_parity=self.target_parity,
                                permutation_policy=self.permutation_policy,
                                allowed_permutation_keys=self.allowed_permutation_keys,
                            ):
                                continue
                            candidate_block = GeneralizedExactRuntimeBlock(
                                mul=int(left_block.mul) * int(right_block.mul),
                                sector=branch.output_sector,
                                L=int(output_L),
                                copy_index=int(target_copy_index),
                            )
                            output_key = target_label.to_string()
                            output_block_index = output_block_key_to_index.get(output_key)
                            if output_block_index is None:
                                output_block_index = len(output_blocks)
                                output_blocks.append(candidate_block)
                                output_block_key_to_index[output_key] = int(output_block_index)
                            else:
                                existing = output_blocks[int(output_block_index)]
                                if int(existing.block_dim) != int(candidate_block.block_dim):
                                    raise ValueError(
                                        "Cannot fuse target label with inconsistent runtime block dimension: "
                                        f"{output_key!r} existing={int(existing.block_dim)} "
                                        f"candidate={int(candidate_block.block_dim)}."
                                    )
                            coord = branch.coordinate_matrix.extract(row_indices, source_cols)
                            source_tensor = torch.zeros(
                                (len(source_cols), left_dim, right_dim),
                                dtype=torch.complex128,
                            )
                            for source_offset, source_index in enumerate(source_cols):
                                ((_, left_carrier), (_, right_carrier), M_out) = branch.source_basis_labels[source_index]
                                for left_M in range(-int(left_block.L), int(left_block.L) + 1):
                                    right_M = int(M_out - left_M)
                                    if right_M < -int(right_block.L) or right_M > int(right_block.L):
                                        continue
                                    coeff = complex(
                                        cg_exact(
                                            int(left_block.L),
                                            int(left_M),
                                            int(right_block.L),
                                            int(right_M),
                                            int(output_L),
                                            int(M_out),
                                        ).evalf()
                                    )
                                    left_pos = left_label_index[(tuple(int(x) for x in left_carrier), int(left_M))]
                                    right_pos = right_label_index[(tuple(int(x) for x in right_carrier), int(right_M))]
                                    source_tensor[source_offset, left_pos, right_pos] = coeff
                            instructions.append(
                                _FullTensorInstruction(
                                    left_block_index=int(left_index),
                                    right_block_index=int(right_index),
                                    output_block_index=int(output_block_index),
                                    output_L=int(output_L),
                                    source_tensor=source_tensor,
                                    coordinate_matrix=torch.tensor(
                                        [
                                            [complex(coord[row, col].evalf()) for col in range(coord.cols)]
                                            for row in range(coord.rows)
                                        ],
                                        dtype=torch.complex128,
                                    ),
                                    source_basis_labels=tuple(branch.source_basis_labels[idx] for idx in source_cols),
                                    target_basis_labels=tuple(branch.target_basis_labels[idx] for idx in row_indices),
                                    young_multiplicity=int(getattr(branch, "subgroup_multiplicity", 1)),
                                )
                            )

        if not output_blocks:
            raise ValueError("No valid generalized tensor-product output blocks remain after exact product-policy filters.")
        self.irreps_out = GeneralizedExactRuntimeIrreps(tuple(output_blocks))
        self._instructions = tuple(instructions)

    def enable_streaming_vjp(self, enabled = True):
        """Opt into instruction-streamed exact custom VJP evaluation."""

        self.use_streaming_vjp = bool(enabled)
        return self

    def enable_real_basis_product(self, enabled = True):
        """Opt into exact real-basis Young-CG product evaluation."""

        if bool(getattr(self, "_packed_plan_only", False)) and not bool(enabled):
            raise ValueError(
                "A packed-only exact product cannot disable real-basis execution."
            )
        self.use_real_basis_product = bool(enabled)
        self._real_instruction_cache = {}
        self._packed_real_bilinear_table_cache = None
        self._packed_real_bilinear_report = None
        return self

    def enable_packed_cuda(self, enabled = True, *, strict = False):
        """Enable one-launch packed CUDA evaluation for real-basis schedules."""

        self.use_packed_cuda = bool(enabled)
        self.strict_packed_cuda = bool(strict)
        return self

    def packed_real_bilinear_table(self):
        """Return the compiler-derived packed table for this real-basis product."""

        if not bool(getattr(self, "use_real_basis_product", False)):
            raise ValueError(
                "packed_real_bilinear_table requires enable_real_basis_product(True)."
            )
        return _packed_real_bilinear_table(self)

    def packed_real_output_blocks(self):
        """Return output block slices for multiplicity-only learned maps."""

        slices = self.irreps_out.slices()
        return tuple(
            {
                "block_index": int(block_index),
                "start": int(slices[int(block_index)].start),
                "stop": int(slices[int(block_index)].stop),
                "channel_count": int(block.mul),
                "component_width": int(block.irreducible_dim),
                "logical_axes": (
                    "channel_or_multiplicity",
                    "tableau_t",
                    "magnetic_M",
                ),
                "learned_map_axis": "channel_or_multiplicity",
            }
            for block_index, block in enumerate(self.irreps_out.blocks)
        )

    def packed_real_product_plan(self):
        """Return a portable exact sparse product plan with compiler hashes."""

        cached = getattr(self, "_packed_real_product_plan", None)
        if cached is not None:
            return json.loads(json.dumps(cached))
        if not bool(getattr(self, "use_real_basis_product", False)):
            raise ValueError(
                "packed_real_product_plan requires real-basis product execution."
            )
        table = self.packed_real_bilinear_table()
        output_parities = {block.label.parity for block in self.irreps_out.blocks}
        if None in output_parities and len(output_parities) > 1:
            raise ValueError(
                "Packed products cannot mix O(3) and SO3-legacy outputs."
            )
        output_convention = (
            YE3T_REAL_TESSERAL_CONVENTION
            if output_parities == {None}
            else YE3T_O3_REAL_TESSERAL_CONVENTION
        )
        output_records = exact_runtime_carrier_records(
            self.irreps_out,
            convention_id=output_convention,
        )
        if any(record.get("carrier_layout") is None for record in output_records):
            raise ValueError(
                "Packed product plans require exact parent output carriers."
            )
        table_payload = {
            "left_index": tuple(
                int(value) for value in table.left_index.detach().cpu().tolist()
            ),
            "right_index": tuple(
                int(value) for value in table.right_index.detach().cpu().tolist()
            ),
            "output_index": tuple(
                int(value) for value in table.output_index.detach().cpu().tolist()
            ),
            "coefficient": tuple(
                float(value) for value in table.coefficient.detach().cpu().tolist()
            ),
            "output_width": int(table.output_width),
        }
        payload = {
            "schema": _PACKED_REAL_PRODUCT_PLAN_SCHEMA,
            "left_irreps": str(self.irreps_in1.to_string()),
            "right_irreps": str(self.irreps_in2.to_string()),
            "left_width": int(self.irreps_in1.dim),
            "right_width": int(self.irreps_in2.dim),
            "output_width": int(self.irreps_out.dim),
            "output_carrier_records": tuple(output_records),
            "output_blocks": self.packed_real_output_blocks(),
            "instruction_bindings": tuple(
                {
                    "left_block_index": int(instruction.left_block_index),
                    "right_block_index": int(instruction.right_block_index),
                    "output_block_index": int(instruction.output_block_index),
                }
                for instruction in self._instructions
            ),
            "table": table_payload,
            "table_hash": _stable_payload_hash(table_payload),
            "table_report": dict(self._packed_real_bilinear_report or {}),
            "tree_type": str(self.tree_type),
            "product_policy": self.product_policy_report(),
            "composer_cache": self._composer_cache_report,
            "static_schedule_provenance": dict(
                getattr(self, "_static_schedule_provenance", None) or {}
            ),
        }
        payload["plan_hash"] = _stable_payload_hash(payload)
        self._packed_real_product_plan = payload
        return json.loads(json.dumps(payload))

    @classmethod
    def from_packed_real_product_plan(cls, plan, irreps_in1, irreps_in2):
        """Construct an exact runtime product without rebuilding subduction."""

        payload = json.loads(json.dumps(dict(plan)))
        supplied_hash = str(payload.pop("plan_hash", ""))
        computed_hash = _stable_payload_hash(payload)
        if not supplied_hash or supplied_hash != computed_hash:
            raise ValueError("Packed real product plan hash mismatch.")
        if str(payload.get("schema")) != _PACKED_REAL_PRODUCT_PLAN_SCHEMA:
            raise ValueError("Unsupported packed real product plan schema.")
        if (
            str(payload.get("left_irreps")) != str(irreps_in1.to_string())
            or str(payload.get("right_irreps")) != str(irreps_in2.to_string())
            or int(payload.get("left_width", -1)) != int(irreps_in1.dim)
            or int(payload.get("right_width", -1)) != int(irreps_in2.dim)
        ):
            raise ValueError(
                "Packed real product plan does not match its exact input carriers."
            )
        irreps_out = GeneralizedExactRuntimeIrreps.from_carrier_records(
            payload.get("output_carrier_records", ())
        )
        if int(payload.get("output_width", -1)) != int(irreps_out.dim):
            raise ValueError(
                "Packed real product plan output width does not match its carriers."
            )
        table_payload = dict(payload.get("table", {}))
        if str(payload.get("table_hash", "")) != _stable_payload_hash(
            table_payload
        ):
            raise ValueError("Packed real product coefficient-table hash mismatch.")
        from ye3t.backends.triton_joint import SparseBilinearTable

        table = SparseBilinearTable(
            left_index=torch.tensor(
                table_payload.get("left_index", ()), dtype=torch.long
            ),
            right_index=torch.tensor(
                table_payload.get("right_index", ()), dtype=torch.long
            ),
            output_index=torch.tensor(
                table_payload.get("output_index", ()), dtype=torch.long
            ),
            coefficient=torch.tensor(
                table_payload.get("coefficient", ()), dtype=torch.float64
            ),
            output_width=int(table_payload.get("output_width", -1)),
        )
        term_count = int(table.term_count)
        if not (
            int(table.left_index.numel())
            == int(table.right_index.numel())
            == int(table.output_index.numel())
            == term_count
        ):
            raise ValueError(
                "Packed real product table arrays must have matching lengths."
            )
        if int(table.output_width) != int(irreps_out.dim):
            raise ValueError(
                "Packed real product table output width is inconsistent."
            )
        if term_count:
            if (
                int(torch.min(table.left_index)) < 0
                or int(torch.max(table.left_index)) >= int(irreps_in1.dim)
                or int(torch.min(table.right_index)) < 0
                or int(torch.max(table.right_index)) >= int(irreps_in2.dim)
                or int(torch.min(table.output_index)) < 0
                or int(torch.max(table.output_index)) >= int(irreps_out.dim)
            ):
                raise ValueError(
                    "Packed real product table contains an out-of-range index."
                )
        instructions = tuple(
            _PackedBilinearInstruction(
                int(record["left_block_index"]),
                int(record["right_block_index"]),
                int(record["output_block_index"]),
            )
            for record in payload.get("instruction_bindings", ())
        )
        if not instructions:
            raise ValueError("Packed real product plan has no block bindings.")
        for instruction in instructions:
            if (
                int(instruction.left_block_index) < 0
                or int(instruction.left_block_index) >= len(irreps_in1.blocks)
                or int(instruction.right_block_index) < 0
                or int(instruction.right_block_index) >= len(irreps_in2.blocks)
                or int(instruction.output_block_index) < 0
                or int(instruction.output_block_index) >= len(irreps_out.blocks)
            ):
                raise ValueError(
                    "Packed real product block binding is out of range."
                )
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.irreps_in1 = irreps_in1
        obj.irreps_in2 = irreps_in2
        obj.irreps_out = irreps_out
        obj._instructions = instructions
        obj.use_streaming_vjp = True
        obj.use_real_basis_product = True
        obj.use_packed_cuda = True
        obj.strict_packed_cuda = False
        obj._real_instruction_cache = {}
        obj._packed_real_bilinear_table_cache = table
        obj._packed_real_bilinear_report = dict(
            payload.get("table_report", {})
        )
        obj._last_real_product_backend = None
        obj._last_real_product_backward_backend = None
        obj._packed_plan_only = True
        obj._packed_real_product_plan = {**payload, "plan_hash": supplied_hash}
        obj._static_schedule_provenance = dict(
            payload.get("static_schedule_provenance", {})
        )
        obj.tree_type = str(payload.get("tree_type", "balanced"))
        obj.composer = None
        obj._composer_cache_report = payload.get("composer_cache")
        policy = dict(payload.get("product_policy", {}))
        requested = policy.get("requested_targets")
        allowed = policy.get("allowed_permutation_irreps")
        obj.requested_target_keys = (
            None
            if requested is None
            else frozenset(str(item) for item in requested)
        )
        obj.requested_target_Ls = None
        if obj.requested_target_keys is not None:
            parsed_Ls = {
                _target_L_from_key(key) for key in obj.requested_target_keys
            }
            obj.requested_target_Ls = frozenset(
                int(value) for value in parsed_Ls if value is not None
            )
        obj.requested_target_permutation_keys = None
        if obj.requested_target_keys is not None:
            parsed_permutations = {
                _target_permutation_from_key(key)
                for key in obj.requested_target_keys
            }
            obj.requested_target_permutation_keys = frozenset(
                str(value)
                for value in parsed_permutations
                if value is not None
            )
        obj.allowed_permutation_keys = (
            None
            if allowed is None
            else frozenset(str(item) for item in allowed)
        )
        obj.rank_cap = policy.get("rank_cap")
        obj.L_max = policy.get("L_max")
        obj.target_parity = _normalize_target_parity(
            policy.get("target_parity")
        )
        obj.permutation_policy = str(
            policy.get("permutation_policy", "mixed_character")
        )
        obj.output_mode = str(policy.get("output_mode", "fused_irrep"))
        obj._parent_subduction_backend = str(
            policy.get(
                "parent_subduction_backend",
                "legacy_raw_generalized_projector",
            )
        )
        return obj

    def forward(self, features_left, features_right):
        if bool(getattr(self, "use_streaming_vjp", False)):
            return _JointYoungCGStreamingVJP.apply(features_left, features_right, self)
        return _joint_young_cg_product_forward(self, features_left, features_right)

    def coefficient_tensors(self, instruction_index = None, *, basis = "canonical"):
        """Return exact CG, Young/intertwiner, and joint coefficient tensors.

        For each selected instruction, the returned dense tensors are exactly
        the tensors used by ``forward``:

        - ``clebsch_gordan_tensor`` has shape ``[source, left, right]``.
        - ``young_coupling_matrix`` has shape ``[target, source]``.
        - ``joint_coupling_tensor`` has shape ``[target, left, right]`` and is
          the contraction of the first two objects.
        """

        return _coefficient_tensors_for_instructions(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            self._instructions,
            instruction_index=instruction_index,
            basis=basis,
        )

    def coefficient_report(self, *, include_values = False, tol = 0.0, instruction_index = None):
        """Return exact coefficient provenance and optional sparse entries."""

        return _coefficient_report_for_instructions(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            self._instructions,
            tree_type=str(self.tree_type),
            product_policy=self.product_policy_report(),
            include_values=include_values,
            tol=tol,
            instruction_index=instruction_index,
            composer_cache=self.composer.cache_stats()
            if self.composer is not None
            else self._composer_cache_report,
            provenance=None,
        )

    def product_policy_report(self):
        """Return the exact target filtering policy used to build this product."""

        return {
            "requested_targets": None if self.requested_target_keys is None else tuple(sorted(self.requested_target_keys)),
            "rank_cap": self.rank_cap,
            "L_max": self.L_max,
            "target_parity": self.target_parity,
            "permutation_policy": self.permutation_policy,
            "allowed_permutation_irreps": None
            if self.allowed_permutation_keys is None
            else tuple(sorted(self.allowed_permutation_keys)),
            "output_mode": str(self.output_mode),
            "parent_subduction_backend": str(
                getattr(self, "_parent_subduction_backend", "legacy_raw_generalized_projector")
            ),
        }

    def fallback_report(self):
        """Return the mathematical/runtime fallback status for this exact product."""

        if bool(getattr(self, "use_real_basis_product", False)):
            report = _real_joint_product_report()
            report["packed_cuda_enabled"] = bool(getattr(self, "use_packed_cuda", True))
            report["packed_cuda_strict"] = bool(getattr(self, "strict_packed_cuda", False))
            report["last_real_product_backend"] = getattr(self, "_last_real_product_backend", None)
            report["last_real_product_backward_backend"] = getattr(
                self,
                "_last_real_product_backward_backend",
                None,
            )
            table_report = getattr(self, "_packed_real_bilinear_report", None)
            report["packed_sparse_bilinear_table"] = (
                None if table_report is None else dict(table_report)
            )
            if report["last_real_product_backend"] == "packed_sparse_bilinear":
                report["runtime"] = "triton_real_basis_exact_schedule"
                report["accelerator_backend"] = "triton_sparse_bilinear"
            if getattr(self, "_parent_subduction_backend", "") == "validated_cached_parent_subduction_C_dagger_L_v":
                report["young_coupling_source"] = "YoungSubgroupSpechtCoupling.C_dagger_L_v"
                report["source_assembly"] = "canonical_binary_tree_identity_coset"
            return report
        report = _exact_joint_product_fallback_report()
        if getattr(self, "_parent_subduction_backend", "") == "validated_cached_parent_subduction_C_dagger_L_v":
            report["young_coupling_source"] = "YoungSubgroupSpechtCoupling.C_dagger_L_v"
            report["source_assembly"] = "canonical_binary_tree_identity_coset"
        return report

    def static_schedule_manifest(self):
        """Return deterministic cache/provenance metadata for the lowered schedule."""

        return _static_schedule_manifest(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            self._instructions,
            tree_type=str(self.tree_type),
            product_policy=self.product_policy_report(),
            composer_cache=self.composer.cache_stats()
            if self.composer is not None
            else self._composer_cache_report,
        )

    def instruction_report(self):
        left_report = self.irreps_in1.report()
        right_report = self.irreps_in2.report()
        out_report = self.irreps_out.report()
        instructions = []
        for index, instruction in enumerate(self._instructions):
            output_block = self.irreps_out.blocks[instruction.output_block_index]
            coefficient_summary = _coefficient_summary_for_instruction(
                self.irreps_in1,
                self.irreps_in2,
                self.irreps_out,
                instruction,
                index,
                include_values=False,
                tol=0.0,
            )
            instructions.append(
                {
                    "index": int(index),
                    "left_block_index": int(instruction.left_block_index),
                    "right_block_index": int(instruction.right_block_index),
                    "output_block_index": int(instruction.output_block_index),
                    "output_label": output_block.label.to_string(),
                    "output_L": int(instruction.output_L),
                    "young_multiplicity": _instruction_young_multiplicity(instruction),
                    "source_tensor_shape": tuple(int(x) for x in instruction.source_tensor.shape),
                    "source_tensor_nnz": _torch_nnz(instruction.source_tensor),
                    "source_tensor_hash": coefficient_summary["clebsch_gordan"]["coefficient_hash"],
                    "coordinate_shape": tuple(int(x) for x in instruction.coordinate_matrix.shape),
                    "coordinate_nnz": _torch_nnz(instruction.coordinate_matrix),
                    "coordinate_hash": coefficient_summary["young_coupling"]["coefficient_hash"],
                    "joint_tensor_shape": coefficient_summary["joint_coupling"]["shape"],
                    "joint_tensor_nnz": coefficient_summary["joint_coupling"]["nnz"],
                    "joint_tensor_hash": coefficient_summary["joint_coupling"]["coefficient_hash"],
                    "source_basis_labels": tuple(instruction.source_basis_labels),
                    "target_basis_labels": tuple(instruction.target_basis_labels),
                    "source_assembly": None
                    if getattr(instruction, "source_assembly", None) is None
                    else dict(instruction.source_assembly),
                }
            )
        return {
            "tree_type": str(self.tree_type),
            "num_instructions": len(instructions),
            "irreps_in1": left_report,
            "irreps_in2": right_report,
            "irreps_out": out_report,
            "composer_cache": self.composer.cache_stats()
            if self.composer is not None
            else self._composer_cache_report,
            "product_policy": self.product_policy_report(),
            "fallback_report": self.fallback_report(),
            "instructions": tuple(instructions),
        }

    def static_schedule_report(self):
        """Return deterministic metadata for the lowered exact tensor schedule."""

        report = self.instruction_report()
        manifest = self.static_schedule_manifest()
        return {
            "tree_type": report["tree_type"],
            "num_instructions": report["num_instructions"],
            "irreps_in1": report["irreps_in1"],
            "irreps_in2": report["irreps_in2"],
            "irreps_out": report["irreps_out"],
            "product_policy": report["product_policy"],
            "fallback_report": report["fallback_report"],
            "coefficient_hash": manifest["coefficient_hash"],
            "manifest_hash": manifest["manifest_hash"],
            "instructions": tuple(
                {
                    "index": instruction["index"],
                    "left_block_index": instruction["left_block_index"],
                    "right_block_index": instruction["right_block_index"],
                    "output_block_index": instruction["output_block_index"],
                    "output_label": instruction["output_label"],
                    "output_L": instruction["output_L"],
                    "young_multiplicity": instruction["young_multiplicity"],
                    "source_tensor_shape": instruction["source_tensor_shape"],
                    "source_tensor_nnz": instruction["source_tensor_nnz"],
                    "source_tensor_hash": instruction["source_tensor_hash"],
                    "coordinate_shape": instruction["coordinate_shape"],
                    "coordinate_nnz": instruction["coordinate_nnz"],
                    "coordinate_hash": instruction["coordinate_hash"],
                    "joint_tensor_shape": instruction["joint_tensor_shape"],
                    "joint_tensor_nnz": instruction["joint_tensor_nnz"],
                    "joint_tensor_hash": instruction["joint_tensor_hash"],
                }
                for instruction in report["instructions"]
            ),
        }

    def static_schedule(self):
        """Return the pre-lowered exact tensor schedule used by ``forward``."""

        metadata = self.static_schedule_report()
        provenance = {
            "format": "joint_young_cg_tensor_schedule_v1",
            "codepath": self.__class__.__name__,
            "schedule_hash": _stable_payload_hash(metadata),
            "manifest_hash": metadata["manifest_hash"],
            "coefficient_hash": metadata["coefficient_hash"],
        }
        return JointYoungCGTensorSchedule(
            irreps_in1=self.irreps_in1,
            irreps_in2=self.irreps_in2,
            irreps_out=self.irreps_out,
            instructions=self._instructions,
            tree_type=str(self.tree_type),
            product_policy=metadata["product_policy"],
            composer_cache=self.composer.cache_stats()
            if self.composer is not None
            else self._composer_cache_report,
            provenance=provenance,
        )

    def save_static_schedule(self, path):
        """Save the lowered exact tensor schedule for later runtime loading."""

        torch.save(self.static_schedule(), path)

    @classmethod
    def from_static_schedule(cls, schedule):
        """Construct a product module from a pre-lowered exact tensor schedule."""

        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.irreps_in1 = schedule.irreps_in1
        obj.irreps_in2 = schedule.irreps_in2
        obj.irreps_out = schedule.irreps_out
        obj._instructions = tuple(schedule.instructions)
        obj.use_streaming_vjp = False
        obj.use_real_basis_product = False
        obj.use_packed_cuda = True
        obj.strict_packed_cuda = False
        obj._real_instruction_cache = {}
        obj._packed_real_bilinear_table_cache = None
        obj._packed_real_bilinear_report = None
        obj._packed_plan_only = False
        obj._packed_real_product_plan = None
        obj._static_schedule_provenance = dict(schedule.provenance)
        obj._last_real_product_backend = None
        obj._last_real_product_backward_backend = None
        obj.tree_type = str(schedule.tree_type)
        obj.composer = None
        obj._composer_cache_report = schedule.composer_cache
        policy = dict(schedule.product_policy)
        requested = policy.get("requested_targets")
        allowed = policy.get("allowed_permutation_irreps")
        obj.requested_target_keys = None if requested is None else frozenset(str(item) for item in requested)
        obj.requested_target_Ls = None
        if obj.requested_target_keys is not None:
            parsed_Ls = {_target_L_from_key(key) for key in obj.requested_target_keys}
            obj.requested_target_Ls = frozenset(int(L) for L in parsed_Ls if L is not None)
        obj.requested_target_permutation_keys = None
        if obj.requested_target_keys is not None:
            parsed_permutations = {_target_permutation_from_key(key) for key in obj.requested_target_keys}
            obj.requested_target_permutation_keys = frozenset(
                str(value) for value in parsed_permutations if value is not None
            )
        obj.allowed_permutation_keys = None if allowed is None else frozenset(str(item) for item in allowed)
        obj.rank_cap = policy.get("rank_cap")
        obj.L_max = policy.get("L_max")
        obj.target_parity = _normalize_target_parity(policy.get("target_parity"))
        obj.permutation_policy = str(policy.get("permutation_policy", "mixed_character"))
        obj.output_mode = str(policy.get("output_mode", "fused_irrep"))
        obj._parent_subduction_backend = str(
            policy.get("parent_subduction_backend", "legacy_raw_generalized_projector")
        )
        return obj

    @classmethod
    def load_static_schedule(cls, path, map_location = None):
        """Load a lowered exact tensor schedule and construct a runtime module."""

        try:
            schedule = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            schedule = torch.load(path, map_location=map_location)
        return cls.from_static_schedule(schedule)

    def format_instruction_report(self):
        report = self.instruction_report()
        lines = [
            "GeneralizedFullTensorProduct",
            f"tree_type={report['tree_type']} num_instructions={report['num_instructions']}",
            f"in1: {self.irreps_in1.to_string()}",
            f"in2: {self.irreps_in2.to_string()}",
            f"out: {self.irreps_out.to_string()}",
        ]
        for instruction in report["instructions"]:
            lines.append(
                "instr[{index}] L={output_L} out={output_label} src_shape={source_tensor_shape} "
                "coord_shape={coordinate_shape} coord_nnz={coordinate_nnz}".format(**instruction)
            )
        return "\n".join(lines)

    def instruction_heatmap_data(self):
        """Return plot-ready heatmap payloads for the exact runtime instructions."""

        instructions = []
        for index, instruction in enumerate(self._instructions):
            source_flat = torch.abs(instruction.source_tensor).reshape(int(instruction.source_tensor.shape[0]), -1)
            source_pair_support = torch.linalg.vector_norm(torch.abs(instruction.source_tensor), dim=0)
            instructions.append(
                {
                    "index": int(index),
                    "output_label": self.irreps_out.blocks[instruction.output_block_index].label.to_string(),
                    "output_L": int(instruction.output_L),
                    "source_flat_shape": tuple(int(x) for x in source_flat.shape),
                    "source_pair_support_shape": tuple(int(x) for x in source_pair_support.shape),
                    "coordinate_shape": tuple(int(x) for x in instruction.coordinate_matrix.shape),
                    "source_flat_abs": _torch_abs_heatmap_data(source_flat),
                    "source_pair_support_abs": _torch_abs_heatmap_data(source_pair_support),
                    "coordinate_abs": _torch_abs_heatmap_data(instruction.coordinate_matrix),
                    "source_basis_labels": tuple(instruction.source_basis_labels),
                    "target_basis_labels": tuple(instruction.target_basis_labels),
                }
            )
        return {
            "tree_type": str(self.tree_type),
            "num_instructions": len(instructions),
            "instructions": tuple(instructions),
        }

    def plot_instruction_heatmaps(
        self,
        *,
        instruction_indices = None,
        max_instructions = None,
        cmap = "magma",
        include_colorbar = True,
    ):
        """Plot exact runtime instruction heatmaps for source support and coordinate maps."""

        import matplotlib.pyplot as plt

        heatmap = self.instruction_heatmap_data()
        selected = (
            tuple(int(index) for index in instruction_indices)
            if instruction_indices is not None
            else tuple(range(len(heatmap["instructions"])))
        )
        if max_instructions is not None:
            selected = selected[: int(max_instructions)]
        if not selected:
            raise ValueError("At least one instruction index is required to plot heatmaps.")

        fig, axes = plt.subplots(
            len(selected),
            3,
            figsize=(13.2, max(3.4, 3.2 * len(selected))),
            constrained_layout=True,
            squeeze=False,
        )
        for row_index, instruction_index in enumerate(selected):
            instruction = heatmap["instructions"][instruction_index]
            support_ax = axes[row_index][0]
            flat_ax = axes[row_index][1]
            coord_ax = axes[row_index][2]

            support_im = support_ax.imshow(instruction["source_pair_support_abs"], aspect="auto", cmap=cmap)
            support_ax.set_title(f"instr[{instruction_index}] pair support |.|")
            support_ax.set_xlabel("right basis")
            support_ax.set_ylabel("left basis")

            flat_im = flat_ax.imshow(instruction["source_flat_abs"], aspect="auto", cmap=cmap)
            flat_ax.set_title(f"{instruction['output_label']} source slices |.|")
            flat_ax.set_xlabel("flattened left-right basis")
            flat_ax.set_ylabel("source slice")

            coord_im = coord_ax.imshow(instruction["coordinate_abs"], aspect="auto", cmap=cmap)
            coord_ax.set_title(f"{instruction['output_label']} coordinate |.|")
            coord_ax.set_xlabel("source basis")
            coord_ax.set_ylabel("target basis")

            if include_colorbar:
                fig.colorbar(support_im, ax=support_ax, shrink=0.8)
                fig.colorbar(flat_im, ax=flat_ax, shrink=0.8)
                fig.colorbar(coord_im, ax=coord_ax, shrink=0.8)

        fig.suptitle(
            "Generalized Full Tensor Product Heatmaps\n"
            f"tree_type={heatmap['tree_type']} num_instructions={heatmap['num_instructions']}",
            fontsize=11,
        )
        return fig


class JointYoungCGProduct(GeneralizedFullTensorProduct):
    """Exact joint Young and Clebsch-Gordan tensor product.

    This is the public product surface for joint ``SO(3) x G_\\nu`` sectors.
    It uses the exact generalized tensor-product composer to lower Young /
    permutation-sector couplings and applies the angular Clebsch-Gordan
    contraction on the canonical magnetic basis. Accepted instructions are
    therefore the paths for which both the angular and permutation-side
    couplings exist.
    """

    def __init__(self, *args, output_mode = "fused_irrep", **kwargs):
        super().__init__(*args, output_mode=output_mode, **kwargs)

    def format_instruction_report(self):
        report = self.instruction_report()
        lines = [
            "JointYoungCGProduct",
            f"tree_type={report['tree_type']} num_instructions={report['num_instructions']}",
            f"in1: {self.irreps_in1.to_string()}",
            f"in2: {self.irreps_in2.to_string()}",
            f"out: {self.irreps_out.to_string()}",
        ]
        for instruction in report["instructions"]:
            lines.append(
                "instr[{index}] L={output_L} out={output_label} src_shape={source_tensor_shape} "
                "coord_shape={coordinate_shape} coord_nnz={coordinate_nnz}".format(**instruction)
            )
        return "\n".join(lines)


YoungE3TensorProduct = JointYoungCGProduct


__all__ = [
    "GeneralizedExactRuntimeBlock",
    "GeneralizedExactRuntimeIrreps",
    "GeneralizedFullTensorProduct",
    "GeneralizedIrreps",
    "GeneralizedLinear",
    "GeneralizedMulIrrep",
    "exact_runtime_carrier_records",
    "JointYoungCGTensorSchedule",
    "JointYoungCGProduct",
    "YoungE3TensorProduct",
]
