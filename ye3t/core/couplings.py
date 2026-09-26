
from functools import lru_cache
from collections import OrderedDict
import os

import numpy as np

from ye3t.core.basis.labels import LeafLabel, NodeLabel, SymBlockLabel
from ye3t.core.basis.tree import RawLeaf, get_tree_factory
from ye3t.core.cg import cg_numeric, cg_numeric_integer
from ye3t.core.factorized_dag import (
    cg_matrix_for_node as _fast_cg_matrix_for_node,
    expand_block_m_path_arrays as _fast_expand_block_m_path_arrays,
    expand_block_m_paths_reference as _fast_expand_block_m_paths_reference,
    factorized_path_key as _fast_factorized_path_key,
    label_block_specs as _fast_label_block_specs,
)
from ye3t.core.labels import CompactLabel, normalize_compact_label
from ye3t._record import recordclass


def _env_flag(name, legacy_name = None):
    raw = os.getenv(name)
    if raw is None and legacy_name is not None:
        raw = os.getenv(legacy_name)
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name, default, legacy_name = None):
    if _env_flag("YE3T_DISABLE_CACHES", "gne3_DISABLE_CACHES"):
        return 0
    raw = os.getenv(name)
    if raw is None and legacy_name is not None:
        raw = os.getenv(legacy_name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


@lru_cache(maxsize=1)
def _factorized_cpp_extension_available():
    try:
        from ye3t.core._factorized_runtime_cpp import has_prebuilt_extension
    except Exception:
        return False
    return bool(has_prebuilt_extension()) or _env_flag("YE3T_ENABLE_CPP_RUNTIME_JIT")


class _BoundedSectorCache:
    def __init__(self, max_entries):
        self.max_entries = int(max_entries)
        self._items = OrderedDict()

    def get(self, key):
        item = self._items.get(key)
        if item is None:
            return None
        self._items.move_to_end(key)
        return item

    def put(self, key, value):
        if self.max_entries <= 0:
            return
        self._items.pop(key, None)
        self._items[key] = value
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)


_SECTOR_CACHE_MAX_ENTRIES = _env_int("YE3T_COUPLING_SECTOR_CACHE_MAX_ENTRIES", 64, "gne3_COUPLING_SECTOR_CACHE_MAX_ENTRIES")
_PAYLOAD_CACHE_MAX_ENTRIES = _env_int("YE3T_COUPLING_PAYLOAD_CACHE_MAX_ENTRIES", 1024, "gne3_COUPLING_PAYLOAD_CACHE_MAX_ENTRIES")
_LIBRARY_CACHE_MAX_ENTRIES = _env_int("YE3T_COUPLING_LIBRARY_CACHE_MAX_ENTRIES", 16, "gne3_COUPLING_LIBRARY_CACHE_MAX_ENTRIES")
_STRUCTURED_LABEL_SECTOR_CACHE = _BoundedSectorCache(_SECTOR_CACHE_MAX_ENTRIES)
_SUBTREE_KEY_SECTOR_CACHE = _BoundedSectorCache(_SECTOR_CACHE_MAX_ENTRIES)
_GENERALIZED_PAYLOAD_CACHE = _BoundedSectorCache(_PAYLOAD_CACHE_MAX_ENTRIES)
_GENERALIZED_LIBRARY_CACHE = _BoundedSectorCache(_LIBRARY_CACHE_MAX_ENTRIES)
_LIGHTWEIGHT_STRUCTURED_LABEL_SECTOR_CACHE = _BoundedSectorCache(_SECTOR_CACHE_MAX_ENTRIES)
_LIGHTWEIGHT_STRUCTURED_ALL_LABELS_CACHE = _BoundedSectorCache(_SECTOR_CACHE_MAX_ENTRIES)
_FACTORIZED_SCHEDULES_CACHE = _BoundedSectorCache(_SECTOR_CACHE_MAX_ENTRIES)


def _selected_factorized_constructor_backend(constructor_backend):
    selected = str(constructor_backend or "auto").strip().lower()
    if selected == "auto":
        env_selected = os.getenv("YE3T_FACTORIZED_CONSTRUCTOR_BACKEND")
        if env_selected is None or str(env_selected).strip().lower() == "auto":
            selected = "cpp" if _factorized_cpp_extension_available() else "python"
        else:
            selected = str(env_selected).strip().lower()
    if selected in {"py", "numpy"}:
        selected = "python"
    if selected in {"c++", "torch-cpp", "cpu-cpp"}:
        selected = "cpp"
    if selected in {"cpp-low-memory", "cpp_lowmem", "cpp-lowmem", "low-memory-cpp", "low_memory_cpp"}:
        selected = "cpp_low_memory"
    if selected not in {"python", "cpp", "cpp_low_memory"}:
        raise ValueError(f"Unknown factorized constructor backend: {constructor_backend!r}")
    return selected


@lru_cache(maxsize=None)
def clebsch_gordan(j1, m1, j2, m2, j3, m3):
    return cg_numeric(j1, m1, j2, m2, j3, m3)


@recordclass(('left', 'right', 'L'), frozen = True)
class CoupledNode:

    @property
    def is_leaf(self):
        return False


@recordclass(('l',), frozen = True)
class CoupledLeaf:

    @property
    def is_leaf(self):
        return True


@recordclass(
    (
        'rank',
        'L_R',
        'n_tuple',
        'l_tuple',
        'tree_type',
        'angular_keys',
        'basis_keys',
        'M_R_values',
        'component_label_index',
        'component_M_R',
        'component_offsets',
        'magnetic_tuples',
        'coeffs',
    ),
    eq=False,
)
class CoefficientTable:
    """Flat numeric magnetic-basis coefficient table for one ``(n, l, L_R)`` sector."""

    @property
    def basis_count(self):
        return int(len(self.angular_keys))

    @property
    def component_count(self):
        return int(len(self.component_label_index))

    @property
    def term_count(self):
        return int(len(self.coeffs))

    def component_terms(self, component_index):
        start = int(self.component_offsets[int(component_index)])
        stop = int(self.component_offsets[int(component_index) + 1])
        return self.magnetic_tuples[start:stop], self.coeffs[start:stop]


@recordclass(
    (
        'rank',
        'block_count',
        'L_R',
        'n_tuple',
        'l_tuple',
        'tree_type',
        'block_specs',
        'angular_keys',
        'basis_keys',
        'M_R_values',
        'component_label_index',
        'component_M_R',
        'component_offsets',
        'block_m_tuples',
        'coeffs',
    ),
    eq=False,
)
class FactorizedCoefficientSchedule:
    """Inter-block CG schedule before expanding symmetric blocks to raw leaves."""

    @property
    def basis_count(self):
        return int(len(self.angular_keys))

    @property
    def component_count(self):
        return int(len(self.component_label_index))

    @property
    def term_count(self):
        return int(len(self.coeffs))

    def component_terms(self, component_index):
        start = int(self.component_offsets[int(component_index)])
        stop = int(self.component_offsets[int(component_index) + 1])
        return self.block_m_tuples[start:stop], self.coeffs[start:stop]

    def block_L_by_label(self, *, dtype=np.int16):
        values = [
            [
                int(spec["Lambda"] if spec["kind"] == "sym" else spec["l"])
                for spec in label_specs
            ]
            for label_specs in self.block_specs
        ]
        if not values:
            return np.zeros((0, int(self.block_count)), dtype=dtype)
        return np.asarray(values, dtype=dtype)

    def to_torch(self, *, device=None, dtype=None):
        """Convert this schedule to torch tensors for runtime evaluation."""
        import torch

        if dtype is None:
            dtype = torch.complex128 if np.iscomplexobj(self.coeffs) else torch.float64
        coeff_array = np.asarray(self.coeffs)
        if not torch.is_complex(torch.empty((), dtype=dtype)) and np.iscomplexobj(coeff_array):
            if coeff_array.size and np.max(np.abs(coeff_array.imag)) > 1e-12:
                raise ValueError("Cannot cast genuinely complex factorized coefficients to a real torch dtype.")
            coeff_array = coeff_array.real
        component_offsets = np.asarray(self.component_offsets, dtype=np.int64)
        term_counts = np.diff(component_offsets)
        term_component_index = np.repeat(np.arange(int(self.component_count), dtype=np.int64), term_counts)
        component_label_index_np = np.asarray(self.component_label_index, dtype=np.int64)
        term_label_index = component_label_index_np[term_component_index] if term_component_index.size else np.asarray([], dtype=np.int64)
        block_L = self.block_L_by_label(dtype=np.int64)
        if int(self.term_count) and int(self.block_count):
            block_offsets = block_L[term_label_index]
            block_m_indices = np.asarray(self.block_m_tuples, dtype=np.int64) + block_offsets
        else:
            block_m_indices = np.zeros((0, int(self.block_count)), dtype=np.int64)
        max_block_m_dim = int(2 * int(block_L.max()) + 1) if block_L.size else 0
        return TorchFactorizedCoefficientSchedule(
            rank=int(self.rank),
            block_count=int(self.block_count),
            L_R=int(self.L_R),
            basis_count=int(self.basis_count),
            component_count=int(self.component_count),
            term_count=int(self.term_count),
            max_block_m_dim=max_block_m_dim,
            M_R_values=torch.as_tensor(np.asarray(self.M_R_values, dtype=np.int64), dtype=torch.long, device=device),
            component_label_index=torch.as_tensor(component_label_index_np, dtype=torch.long, device=device),
            component_M_R=torch.as_tensor(np.asarray(self.component_M_R, dtype=np.int64), dtype=torch.long, device=device),
            component_offsets=torch.as_tensor(component_offsets, dtype=torch.long, device=device),
            term_component_index=torch.as_tensor(term_component_index, dtype=torch.long, device=device),
            term_label_index=torch.as_tensor(term_label_index, dtype=torch.long, device=device),
            block_m_indices=torch.as_tensor(block_m_indices, dtype=torch.long, device=device),
            block_L_by_label=torch.as_tensor(block_L, dtype=torch.long, device=device),
            coeffs=torch.as_tensor(coeff_array, dtype=dtype, device=device),
        )


@recordclass(
    (
        'rank',
        'block_count',
        'L_R',
        'basis_count',
        'component_count',
        'term_count',
        'max_block_m_dim',
        'M_R_values',
        'component_label_index',
        'component_M_R',
        'component_offsets',
        'term_component_index',
        'term_label_index',
        'block_m_indices',
        'block_L_by_label',
        'coeffs',
    ),
    eq=False,
)
class TorchFactorizedCoefficientSchedule:
    """Torch tensor form of :class:`FactorizedCoefficientSchedule`."""

    def to(self, *, device=None, dtype=None):
        if dtype is None:
            dtype = self.coeffs.dtype
        coeffs = self.coeffs.to(device=device)
        import torch

        if not torch.is_complex(torch.empty((), dtype=dtype)) and torch.is_complex(coeffs):
            if int(coeffs.numel()) and torch.max(torch.abs(coeffs.imag)).item() > 1e-12:
                raise ValueError("Cannot cast genuinely complex factorized coefficients to a real torch dtype.")
            coeffs = coeffs.real
        return TorchFactorizedCoefficientSchedule(
            rank=int(self.rank),
            block_count=int(self.block_count),
            L_R=int(self.L_R),
            basis_count=int(self.basis_count),
            component_count=int(self.component_count),
            term_count=int(self.term_count),
            max_block_m_dim=int(self.max_block_m_dim),
            M_R_values=self.M_R_values.to(device=device, dtype=self.M_R_values.dtype),
            component_label_index=self.component_label_index.to(device=device, dtype=self.component_label_index.dtype),
            component_M_R=self.component_M_R.to(device=device, dtype=self.component_M_R.dtype),
            component_offsets=self.component_offsets.to(device=device, dtype=self.component_offsets.dtype),
            term_component_index=self.term_component_index.to(device=device, dtype=self.term_component_index.dtype),
            term_label_index=self.term_label_index.to(device=device, dtype=self.term_label_index.dtype),
            block_m_indices=self.block_m_indices.to(device=device, dtype=self.block_m_indices.dtype),
            block_L_by_label=self.block_L_by_label.to(device=device, dtype=self.block_L_by_label.dtype),
            coeffs=coeffs.to(dtype=dtype),
        )


@recordclass(
    (
        'rank',
        'block_count',
        'L_R',
        'n_tuple',
        'l_tuple',
        'tree_type',
        'block_specs',
        'angular_keys',
        'basis_keys',
        'component_indices',
        'component_label_index',
        'component_offsets',
        'block_real_tuples',
        'coeffs',
        'basis_convention',
        'permutation_sector',
        'backend_provenance',
        'complex_term_count',
        'fallback_reason',
    ),
    eq=False,
)
class RealFactorizedCoefficientSchedule:
    """Real-tesseral inter-block CG schedule for collapsed ACE blocks."""

    @property
    def basis_count(self):
        return int(len(self.angular_keys))

    @property
    def component_count(self):
        return int(len(self.component_label_index))

    @property
    def term_count(self):
        return int(len(self.coeffs))

    def component_terms(self, component_index):
        start = int(self.component_offsets[int(component_index)])
        stop = int(self.component_offsets[int(component_index) + 1])
        return self.block_real_tuples[start:stop], self.coeffs[start:stop]

    def block_L_by_label(self, *, dtype=np.int16):
        values = [
            [
                int(spec["Lambda"] if spec["kind"] == "sym" else spec["l"])
                for spec in label_specs
            ]
            for label_specs in self.block_specs
        ]
        if not values:
            return np.zeros((0, int(self.block_count)), dtype=dtype)
        return np.asarray(values, dtype=dtype)

    def metadata(self):
        return {
            "basis_convention": str(self.basis_convention),
            "permutation_sector": str(self.permutation_sector),
            "backend_provenance": str(self.backend_provenance),
            "term_count": int(self.term_count),
            "complex_term_count": int(self.complex_term_count),
            "fallback_reason": None if self.fallback_reason is None else str(self.fallback_reason),
        }

    def to_torch(self, *, device=None, dtype=None):
        """Convert this real schedule to torch tensors for runtime evaluation."""
        import torch

        if dtype is None:
            dtype = torch.float64
        if torch.is_complex(torch.empty((), dtype=dtype)):
            raise ValueError("Real factorized schedules require a real torch dtype.")
        component_offsets = np.asarray(self.component_offsets, dtype=np.int64)
        term_counts = np.diff(component_offsets)
        term_component_index = np.repeat(np.arange(int(self.component_count), dtype=np.int64), term_counts)
        component_label_index_np = np.asarray(self.component_label_index, dtype=np.int64)
        term_label_index = component_label_index_np[term_component_index] if term_component_index.size else np.asarray([], dtype=np.int64)
        block_L = self.block_L_by_label(dtype=np.int64)
        if int(self.term_count) and int(self.block_count):
            block_indices = np.asarray(self.block_real_tuples, dtype=np.int64)
        else:
            block_indices = np.zeros((0, int(self.block_count)), dtype=np.int64)
        max_block_m_dim = int(2 * int(block_L.max()) + 1) if block_L.size else 0
        return TorchRealFactorizedCoefficientSchedule(
            rank=int(self.rank),
            block_count=int(self.block_count),
            L_R=int(self.L_R),
            basis_count=int(self.basis_count),
            component_count=int(self.component_count),
            term_count=int(self.term_count),
            max_block_m_dim=max_block_m_dim,
            component_indices=torch.as_tensor(np.asarray(self.component_indices, dtype=np.int64), dtype=torch.long, device=device),
            component_label_index=torch.as_tensor(component_label_index_np, dtype=torch.long, device=device),
            component_offsets=torch.as_tensor(component_offsets, dtype=torch.long, device=device),
            term_component_index=torch.as_tensor(term_component_index, dtype=torch.long, device=device),
            term_label_index=torch.as_tensor(term_label_index, dtype=torch.long, device=device),
            block_real_indices=torch.as_tensor(block_indices, dtype=torch.long, device=device),
            block_L_by_label=torch.as_tensor(block_L, dtype=torch.long, device=device),
            coeffs=torch.as_tensor(np.asarray(self.coeffs, dtype=np.float64), dtype=dtype, device=device),
            basis_convention=str(self.basis_convention),
            permutation_sector=str(self.permutation_sector),
            backend_provenance=str(self.backend_provenance),
            complex_term_count=int(self.complex_term_count),
            fallback_reason=self.fallback_reason,
        )


@recordclass(
    (
        'rank',
        'block_count',
        'L_R',
        'basis_count',
        'component_count',
        'term_count',
        'max_block_m_dim',
        'component_indices',
        'component_label_index',
        'component_offsets',
        'term_component_index',
        'term_label_index',
        'block_real_indices',
        'block_L_by_label',
        'coeffs',
        'basis_convention',
        'permutation_sector',
        'backend_provenance',
        'complex_term_count',
        'fallback_reason',
    ),
    eq=False,
)
class TorchRealFactorizedCoefficientSchedule:
    """Torch tensor form of :class:`RealFactorizedCoefficientSchedule`."""

    def to(self, *, device=None, dtype=None):
        import torch

        if dtype is None:
            dtype = self.coeffs.dtype
        if torch.is_complex(torch.empty((), dtype=dtype)):
            raise ValueError("Real factorized schedules require a real torch dtype.")
        return TorchRealFactorizedCoefficientSchedule(
            rank=int(self.rank),
            block_count=int(self.block_count),
            L_R=int(self.L_R),
            basis_count=int(self.basis_count),
            component_count=int(self.component_count),
            term_count=int(self.term_count),
            max_block_m_dim=int(self.max_block_m_dim),
            component_indices=self.component_indices.to(device=device, dtype=self.component_indices.dtype),
            component_label_index=self.component_label_index.to(device=device, dtype=self.component_label_index.dtype),
            component_offsets=self.component_offsets.to(device=device, dtype=self.component_offsets.dtype),
            term_component_index=self.term_component_index.to(device=device, dtype=self.term_component_index.dtype),
            term_label_index=self.term_label_index.to(device=device, dtype=self.term_label_index.dtype),
            block_real_indices=self.block_real_indices.to(device=device, dtype=self.block_real_indices.dtype),
            block_L_by_label=self.block_L_by_label.to(device=device, dtype=self.block_L_by_label.dtype),
            coeffs=self.coeffs.to(device=device, dtype=dtype),
            basis_convention=str(self.basis_convention),
            permutation_sector=str(self.permutation_sector),
            backend_provenance=str(self.backend_provenance),
            complex_term_count=int(self.complex_term_count),
            fallback_reason=self.fallback_reason,
        )


def evaluate_factorized_schedule_torch(block_values, schedule, *, backend = "auto"):
    """Evaluate a torch factorized coefficient schedule.

    ``block_values`` must have shape
    ``[batch, basis_count, block_count, max_block_m_dim]``.  For each basis
    label and block, magnetic index ``m + L_block`` stores that block's
    symmetry-adapted value.  Invalid/padded magnetic positions should be zero.

    ``backend='torch'`` uses the pure PyTorch gather/product/index-add path.
    ``backend='cpp'`` uses the compiled CPU C++/PyTorch extension and raises if
    it cannot be loaded.  ``backend='auto'`` uses that extension for CPU tensors
    when it is installed and otherwise falls back to the pure PyTorch path.
    """
    import torch

    if not isinstance(schedule, TorchFactorizedCoefficientSchedule):
        schedule = schedule.to_torch(device=block_values.device, dtype=block_values.dtype)
    else:
        schedule = schedule.to(device=block_values.device)
    if block_values.ndim != 4:
        raise ValueError("block_values must have shape [batch, basis_count, block_count, max_block_m_dim].")
    if int(block_values.shape[1]) < int(schedule.basis_count):
        raise ValueError("block_values basis dimension is smaller than schedule.basis_count.")
    if int(block_values.shape[2]) != int(schedule.block_count):
        raise ValueError("block_values block dimension must equal schedule.block_count.")
    if int(block_values.shape[3]) < int(schedule.max_block_m_dim):
        raise ValueError("block_values magnetic dimension is smaller than schedule.max_block_m_dim.")
    coeffs = schedule.coeffs.to(device=block_values.device)
    if not torch.is_complex(block_values) and torch.is_complex(coeffs):
        if int(coeffs.numel()) and torch.max(torch.abs(coeffs.imag)).item() > 1e-12:
            value_dtype = torch.promote_types(block_values.dtype, coeffs.dtype)
        else:
            coeffs = coeffs.real.to(dtype=block_values.dtype)
            value_dtype = block_values.dtype
    else:
        value_dtype = torch.promote_types(block_values.dtype, coeffs.dtype)
        coeffs = coeffs.to(dtype=value_dtype)
    values = block_values.to(dtype=value_dtype) if block_values.dtype != value_dtype else block_values

    selected_backend = str(backend or "auto").strip().lower()
    if selected_backend == "auto":
        env_selected = os.getenv("YE3T_FACTORIZED_RUNTIME_BACKEND")
        if env_selected is None or str(env_selected).strip().lower() == "auto":
            selected_backend = (
                "cpp"
                if values.device.type == "cpu"
                and not values.requires_grad
                and _factorized_cpp_extension_available()
                else "torch"
            )
        else:
            selected_backend = str(env_selected).strip().lower()
    if selected_backend in {"cpp", "c++", "torch-cpp", "cpu-cpp"}:
        if values.device.type != "cpu":
            raise ValueError("The C++ factorized runtime backend currently supports CPU tensors only.")
        from ye3t.core._factorized_runtime_cpp import evaluate_factorized_schedule_cpu

        return evaluate_factorized_schedule_cpu(
            values.contiguous(),
            schedule.component_offsets.to(device=values.device, dtype=torch.long).contiguous(),
            schedule.component_label_index.to(device=values.device, dtype=torch.long).contiguous(),
            schedule.block_m_indices.to(device=values.device, dtype=torch.long).contiguous(),
            coeffs.to(device=values.device, dtype=value_dtype).contiguous(),
            int(schedule.component_count),
            _env_flag("YE3T_VALIDATE_CPP_RUNTIME"),
        )
    if selected_backend not in {"torch", "pytorch", ""}:
        raise ValueError(f"Unknown factorized runtime backend: {backend!r}")

    out = torch.zeros(
        (int(block_values.shape[0]), int(schedule.component_count)),
        dtype=value_dtype,
        device=block_values.device,
    )
    if int(schedule.term_count) == 0:
        return out
    term_values = coeffs.unsqueeze(0).expand(int(block_values.shape[0]), -1)
    batch_index = torch.arange(int(block_values.shape[0]), device=block_values.device).unsqueeze(1)
    for block_index in range(int(schedule.block_count)):
        term_values = term_values * values[
            batch_index,
            schedule.term_label_index.unsqueeze(0),
            int(block_index),
            schedule.block_m_indices[:, int(block_index)].unsqueeze(0),
        ]
    out.index_add_(1, schedule.term_component_index, term_values)
    return out


def evaluate_real_factorized_schedule_torch(block_values, schedule, *, backend = "auto"):
    """Evaluate a real-tesseral factorized coefficient schedule."""
    import torch

    if not isinstance(schedule, TorchRealFactorizedCoefficientSchedule):
        schedule = schedule.to_torch(device=block_values.device, dtype=block_values.dtype)
    else:
        schedule = schedule.to(device=block_values.device)
    if torch.is_complex(block_values):
        raise ValueError("Real factorized schedule evaluation requires real block values.")
    if block_values.ndim != 4:
        raise ValueError("block_values must have shape [batch, basis_count, block_count, max_block_m_dim].")
    if int(block_values.shape[1]) < int(schedule.basis_count):
        raise ValueError("block_values basis dimension is smaller than schedule.basis_count.")
    if int(block_values.shape[2]) != int(schedule.block_count):
        raise ValueError("block_values block dimension must equal schedule.block_count.")
    if int(block_values.shape[3]) < int(schedule.max_block_m_dim):
        raise ValueError("block_values real-component dimension is smaller than schedule.max_block_m_dim.")
    selected_backend = str(backend or "auto").strip().lower()
    if selected_backend not in {"auto", "torch", "pytorch", ""}:
        raise ValueError(f"Unknown real factorized runtime backend: {backend!r}")
    values = block_values.to(dtype=torch.promote_types(block_values.dtype, schedule.coeffs.dtype))
    coeffs = schedule.coeffs.to(device=values.device, dtype=values.dtype)
    out = torch.zeros(
        (int(values.shape[0]), int(schedule.component_count)),
        dtype=values.dtype,
        device=values.device,
    )
    if int(schedule.term_count) == 0:
        return out
    term_values = coeffs.unsqueeze(0).expand(int(values.shape[0]), -1)
    batch_index = torch.arange(int(values.shape[0]), device=values.device).unsqueeze(1)
    for block_index in range(int(schedule.block_count)):
        term_values = term_values * values[
            batch_index,
            schedule.term_label_index.unsqueeze(0),
            int(block_index),
            schedule.block_real_indices[:, int(block_index)].unsqueeze(0),
        ]
    out.index_add_(1, schedule.term_component_index, term_values)
    return out


def _build_skeleton(l_tuple, tree_type = "balanced"):
    nin = [0] * len(l_tuple)
    return get_tree_factory(tree_type).build_raw_tree(nin, list(l_tuple))



def _assign_internal_Ls_from_postorder(node, internal_Ls, idx_ref):
    if isinstance(node, RawLeaf):
        return CoupledLeaf(l=node.l)
    left = _assign_internal_Ls_from_postorder(node.left, internal_Ls, idx_ref)
    right = _assign_internal_Ls_from_postorder(node.right, internal_Ls, idx_ref)
    L = internal_Ls[idx_ref[0]]
    idx_ref[0] += 1
    return CoupledNode(left=left, right=right, L=L)


@lru_cache(maxsize=None)
def _structured_label_from_compact(label):
    lab = normalize_compact_label(label)
    if lab.basis_key and len(tuple(lab.internal_Ls)) != int(lab.rank) - 1:
        reconstructed = _structured_label_from_collapsed_basis_key(lab)
        if reconstructed is not None:
            return reconstructed
    sector_map = _structured_labels_for_sector(tuple(lab.n_tuple), tuple(lab.l_tuple), int(lab.L_R), str(lab.tree_type))
    return sector_map.get(lab)


def _collapsed_label_blocks(lab):
    blocks = []
    start = 0
    while start < int(lab.rank):
        stop = start + 1
        while stop < int(lab.rank) and int(lab.n_tuple[stop]) == int(lab.n_tuple[start]) and int(lab.l_tuple[stop]) == int(lab.l_tuple[start]):
            stop += 1
        blocks.append(
            {
                "n": int(lab.n_tuple[start]),
                "l": int(lab.l_tuple[start]),
                "k": int(stop - start),
            }
        )
        start = stop
    return tuple(blocks)


def _structured_label_from_collapsed_basis_key(lab):
    blocks = _collapsed_label_blocks(lab)
    internal = tuple(int(value) for value in lab.internal_Ls)

    def build(key, block_pos, internal_pos):
        key = tuple(key)
        if not key:
            if block_pos >= len(blocks):
                return None, block_pos, internal_pos
            block = blocks[block_pos]
            if int(block["k"]) != 1:
                return None, block_pos, internal_pos
            return (
                LeafLabel(n=int(block["n"]), l=int(block["l"]), tree_type=str(lab.tree_type)),
                block_pos + 1,
                internal_pos,
            )
        if str(key[0]) == "sym":
            if block_pos >= len(blocks) or internal_pos >= len(internal):
                return None, block_pos, internal_pos
            block = blocks[block_pos]
            Lambda = int(key[1])
            multiplicity_index = int(key[2])
            if int(internal[internal_pos]) != Lambda:
                return None, block_pos, internal_pos
            return (
                SymBlockLabel(
                    n=int(block["n"]),
                    l=int(block["l"]),
                    k_b=int(block["k"]),
                    Lambda=Lambda,
                    multiplicity_index=multiplicity_index,
                    representative_internal_Ls=tuple(),
                    tree_type=str(lab.tree_type),
                    basis_key=("sym", Lambda, multiplicity_index),
                    occupancy_expansion_by_M=tuple(),
                ),
                block_pos + 1,
                internal_pos + 1,
            )
        if str(key[0]) == "node":
            left, block_pos, internal_pos = build(key[1], block_pos, internal_pos)
            right, block_pos, internal_pos = build(key[2], block_pos, internal_pos)
            if left is None or right is None or internal_pos >= len(internal):
                return None, block_pos, internal_pos
            L = int(internal[internal_pos])
            return (
                NodeLabel(left=left, right=right, L=L, tree_type=str(lab.tree_type)),
                block_pos,
                internal_pos + 1,
            )
        return None, block_pos, internal_pos

    structured, block_pos, internal_pos = build(tuple(lab.basis_key), 0, 0)
    if structured is None or block_pos != len(blocks) or internal_pos != len(internal):
        return None
    if tuple(structured.compact_label()) != (
        tuple(lab.n_tuple),
        tuple(lab.l_tuple),
        tuple(lab.internal_Ls),
        str(lab.tree_type),
        tuple(lab.basis_key),
    ):
        return None
    return structured


@lru_cache(maxsize=None)
def _structured_labels_for_sector(nin, lin, L_R, tree_type):
    cache_key = ("structured_labels", tuple(nin), tuple(lin), int(L_R), str(tree_type))
    cached = _STRUCTURED_LABEL_SECTOR_CACHE.get(cache_key)
    if cached is not None:
        return cached
    from ye3t.core.basis import YE3TBasisLabeler

    labeler = YE3TBasisLabeler(list(nin), list(lin), strict_target_validation=False, tree_type=tree_type)
    sector_map = {}
    if hasattr(labeler, "structured_label_objects_for_target"):
        compact_labels = labeler.compact_labels_for_target(int(L_R))
        structured_labels = labeler.structured_label_objects_for_target(int(L_R))
        sector_map = {
            normalize_compact_label(compact): structured
            for compact, structured in zip(compact_labels, structured_labels)
        }
    else:
        sector = labeler.sector_data_for_target(int(L_R))
        if sector is not None:
            sector_map = {
                normalize_compact_label(entry.compact_label): entry.structured_label
                for entry in sector.entries
            }
    _STRUCTURED_LABEL_SECTOR_CACHE.put(cache_key, sector_map)
    return sector_map


@lru_cache(maxsize=None)
def _lightweight_structured_label_from_compact(label):
    lab = normalize_compact_label(label)
    if lab.basis_key and len(tuple(lab.internal_Ls)) != int(lab.rank) - 1:
        reconstructed = _structured_label_from_collapsed_basis_key(lab)
        if reconstructed is not None:
            return reconstructed
    sector_map = _lightweight_structured_labels_for_sector(
        tuple(lab.n_tuple),
        tuple(lab.l_tuple),
        int(lab.L_R),
        str(lab.tree_type),
    )
    return sector_map.get(lab)


@lru_cache(maxsize=None)
def _lightweight_structured_labels_for_all_targets(nin, lin, tree_type):
    cache_key = ("lightweight_structured_labels_all", tuple(nin), tuple(lin), str(tree_type))
    cached = _LIGHTWEIGHT_STRUCTURED_ALL_LABELS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    from ye3t.core.basis import YE3TBasisLabeler

    labeler = YE3TBasisLabeler(list(nin), list(lin), strict_target_validation=False, tree_type=tree_type)
    if hasattr(labeler.backend, "lightweight_structured_label_objects_by_L"):
        compact_by_l = labeler.compact_labels_by_L()
        structured_by_l = labeler.backend.lightweight_structured_label_objects_by_L()
    else:
        compact_by_l = labeler.compact_labels_by_L()
        structured_by_l = {
            int(L): labeler.lightweight_structured_label_objects_for_target(int(L))
            for L in compact_by_l
        }
    out = {}
    for L, compact_labels in compact_by_l.items():
        structured_labels = structured_by_l.get(int(L), [])
        out[int(L)] = {
            normalize_compact_label(compact): structured
            for compact, structured in zip(compact_labels, structured_labels)
        }
    _LIGHTWEIGHT_STRUCTURED_ALL_LABELS_CACHE.put(cache_key, out)
    return out


@lru_cache(maxsize=None)
def _lightweight_structured_labels_for_sector(nin, lin, L_R, tree_type):
    cache_key = ("lightweight_structured_labels", tuple(nin), tuple(lin), int(L_R), str(tree_type))
    cached = _LIGHTWEIGHT_STRUCTURED_LABEL_SECTOR_CACHE.get(cache_key)
    if cached is not None:
        return cached
    all_targets = _lightweight_structured_labels_for_all_targets(tuple(nin), tuple(lin), str(tree_type))
    sector_map = dict(all_targets.get(int(L_R), {}))
    _LIGHTWEIGHT_STRUCTURED_LABEL_SECTOR_CACHE.put(cache_key, sector_map)
    return sector_map


@lru_cache(maxsize=None)
def _subtree_key_from_compact(label):
    from ye3t.core.subtree_dag import raw_tree_key, structured_label_key

    lab = normalize_compact_label(label)
    cache_key = ("subtree_keys", tuple(lab.n_tuple), tuple(lab.l_tuple), int(lab.L_R), str(lab.tree_type))
    cached = _SUBTREE_KEY_SECTOR_CACHE.get(cache_key)
    if cached is None:
        structured_map = _structured_labels_for_sector(tuple(lab.n_tuple), tuple(lab.l_tuple), int(lab.L_R), str(lab.tree_type))
        cached = {}
        for compact, structured in structured_map.items():
            if getattr(compact, "basis_key", tuple()):
                continue
            cached[compact] = structured_label_key(structured)
        _SUBTREE_KEY_SECTOR_CACHE.put(cache_key, cached)
    subtree_key = cached.get(lab)
    if subtree_key is not None:
        return subtree_key
    return raw_tree_key(lab.l_tuple, lab.internal_Ls, lab.tree_type)


def _build_coupled_tree_from_structured(label_obj):
    if isinstance(label_obj, LeafLabel):
        return CoupledLeaf(l=int(label_obj.l))
    if isinstance(label_obj, SymBlockLabel):
        if label_obj.occupancy_expansion_by_M and not label_obj.representative_internal_Ls:
            raise NotImplementedError(
                "Exact homogeneous symmetric-power labels do not reconstruct to a single coupled tree."
            )
        skeleton = _build_skeleton(label_obj.l_sequence(), tree_type=label_obj.tree_type)
        idx_ref = [0]
        coupled = _assign_internal_Ls_from_postorder(skeleton, label_obj.representative_internal_Ls, idx_ref)
        if idx_ref[0] != len(label_obj.representative_internal_Ls):
            raise RuntimeError("Did not consume all homogeneous-block internal L labels.")
        return coupled
    if isinstance(label_obj, NodeLabel):
        return CoupledNode(
            left=_build_coupled_tree_from_structured(label_obj.left),
            right=_build_coupled_tree_from_structured(label_obj.right),
            L=int(label_obj.L),
        )
    raise TypeError(f"Unsupported structured label type: {type(label_obj)!r}")



def build_coupled_tree(label):
    lab = normalize_compact_label(label)
    structured = _structured_label_from_compact(lab)
    if structured is not None:
        return _build_coupled_tree_from_structured(structured)
    skeleton = _build_skeleton(lab.l_tuple, tree_type=lab.tree_type)
    if lab.rank == 1:
        return CoupledLeaf(l=lab.l_tuple[0])
    if len(lab.internal_Ls) != lab.rank - 1:
        raise ValueError(
            f"internal_Ls must have length rank-1 for full binary labels; got {len(lab.internal_Ls)} for rank {lab.rank}."
        )
    idx_ref = [0]
    coupled = _assign_internal_Ls_from_postorder(skeleton, lab.internal_Ls, idx_ref)
    if idx_ref[0] != len(lab.internal_Ls):
        raise RuntimeError("Did not consume all internal L labels when reconstructing the balanced tree.")
    return coupled



def _expand_m_paths(node):
    if isinstance(node, CoupledLeaf):
        out = {}
        for m in range(-node.l, node.l + 1):
            out[m] = {(m,): 1.0 + 0.0j}
        return out

    left_total = node.left.l if isinstance(node.left, CoupledLeaf) else node.left.L
    right_total = node.right.l if isinstance(node.right, CoupledLeaf) else node.right.L

    left_map = _expand_m_paths(node.left)
    right_map = _expand_m_paths(node.right)
    out = {}

    for M1, d1 in left_map.items():
        for M2, d2 in right_map.items():
            M = M1 + M2
            if abs(M) > node.L:
                continue
            cg = clebsch_gordan(left_total, M1, right_total, M2, node.L, M)
            if abs(cg) < 1e-14:
                continue
            target = out.setdefault(M, {})
            for ms1, c1 in d1.items():
                for ms2, c2 in d2.items():
                    key = ms1 + ms2
                    target[key] = target.get(key, 0.0 + 0.0j) + c1 * c2 * cg
    return out



def generate_generalized_payload(
    label,
    M_R,
    coeff_tol = 1e-14,
):
    lab = normalize_compact_label(label)
    cache_key = (
        tuple(int(x) for x in lab.n_tuple),
        tuple(int(x) for x in lab.l_tuple),
        tuple(lab.internal_Ls),
        str(lab.tree_type),
        tuple(lab.basis_key),
        int(M_R),
        float(coeff_tol),
    )
    cached = _GENERALIZED_PAYLOAD_CACHE.get(cache_key)
    if cached is not None:
        return {
            "rank": int(cached["rank"]),
            "n_tuple": list(cached["n_tuple"]),
            "l_tuple": list(cached["l_tuple"]),
            "internal_Ls": list(cached["internal_Ls"]),
            "basis_key": list(cached.get("basis_key", [])),
            "L_R": int(cached["L_R"]),
            "M_R": int(cached["M_R"]),
            "tree_type": str(cached["tree_type"]),
            "ms_combs": [list(ms) for ms in cached["ms_combs"]],
            "coeffs": [list(coeff) for coeff in cached["coeffs"]],
        }
    L_R = lab.L_R
    if lab.basis_key and len(tuple(lab.internal_Ls)) != int(lab.rank) - 1:
        payload = {
            "rank": lab.rank,
            "n_tuple": list(lab.n_tuple),
            "l_tuple": list(lab.l_tuple),
            "internal_Ls": list(lab.internal_Ls),
            "basis_key": list(lab.basis_key),
            "L_R": int(L_R),
            "M_R": int(M_R),
            "tree_type": lab.tree_type,
            "ms_combs": [],
            "coeffs": [],
        }
        _GENERALIZED_PAYLOAD_CACHE.put(cache_key, payload)
        return {
            "rank": int(payload["rank"]),
            "n_tuple": list(payload["n_tuple"]),
            "l_tuple": list(payload["l_tuple"]),
            "internal_Ls": list(payload["internal_Ls"]),
            "basis_key": list(payload.get("basis_key", [])),
            "L_R": int(payload["L_R"]),
            "M_R": int(payload["M_R"]),
            "tree_type": str(payload["tree_type"]),
            "ms_combs": [],
            "coeffs": [],
        }
    from ye3t.core.subtree_dag import expand_structured_label_numeric, expand_tree_key_numeric

    structured = _structured_label_from_compact(lab)
    if structured is not None:
        root_map = expand_structured_label_numeric(structured)
    else:
        root_map = expand_tree_key_numeric(_subtree_key_from_compact(lab))
    coeff_map = root_map.get(M_R, {})
    items = sorted((ms, coeff) for ms, coeff in coeff_map.items() if abs(coeff) > coeff_tol)
    payload = {
        "rank": lab.rank,
        "n_tuple": list(lab.n_tuple),
        "l_tuple": list(lab.l_tuple),
        "internal_Ls": list(lab.internal_Ls),
        "basis_key": list(lab.basis_key),
        "L_R": int(L_R),
        "M_R": int(M_R),
        "tree_type": lab.tree_type,
        "ms_combs": [list(ms) for ms, _ in items],
        "coeffs": [[float(np.real(c)), float(np.imag(c))] for _, c in items],
    }
    _GENERALIZED_PAYLOAD_CACHE.put(cache_key, payload)
    return {
        "rank": int(payload["rank"]),
        "n_tuple": list(payload["n_tuple"]),
        "l_tuple": list(payload["l_tuple"]),
        "internal_Ls": list(payload["internal_Ls"]),
        "basis_key": list(payload.get("basis_key", [])),
        "L_R": int(payload["L_R"]),
        "M_R": int(payload["M_R"]),
        "tree_type": str(payload["tree_type"]),
        "ms_combs": [list(ms) for ms in payload["ms_combs"]],
        "coeffs": [list(coeff) for coeff in payload["coeffs"]],
    }



def payload_to_library_entry(payload):
    coeffs = [complex(r, i) for r, i in payload["coeffs"]]
    return {
        "rank": int(payload["rank"]),
        "ms_combs": payload["ms_combs"],
        "coeffs": coeffs,
        "n_tuple": payload["n_tuple"],
        "l_tuple": payload["l_tuple"],
        "internal_Ls": payload["internal_Ls"],
        "basis_key": payload.get("basis_key", []),
        "tree_type": payload.get("tree_type", "balanced"),
        "L_R": int(payload["L_R"]),
        "M_R": int(payload["M_R"]),
    }



def compact_label_from_raw(
    nin,
    lin,
    internal_Ls,
    tree_type = "balanced",
    basis_key = tuple(),
):
    return CompactLabel(tuple(nin), tuple(lin), tuple(internal_Ls), tree_type=tree_type, basis_key=tuple(basis_key))



def generate_library_for_labels(
    labels,
    M_R_values = None,
):
    labels = [normalize_compact_label(lab) for lab in labels]
    if not labels:
        return {}
    L_R = labels[0].L_R
    if M_R_values is None:
        M_R_values = range(-L_R, L_R + 1)
    M_R_values = tuple(int(M) for M in M_R_values)
    cache_key = (
        tuple(
            (
                tuple(int(x) for x in lab.n_tuple),
                tuple(int(x) for x in lab.l_tuple),
                tuple(lab.internal_Ls),
                str(lab.tree_type),
                tuple(lab.basis_key),
            )
            for lab in labels
        ),
        M_R_values,
    )
    cached = _GENERALIZED_LIBRARY_CACHE.get(cache_key)
    if cached is not None:
        return {
            int(M_R): {
                int(rank): {str(key): dict(payload) for key, payload in rank_block.items()}
                for rank, rank_block in M_block.items()
            }
            for M_R, M_block in cached.items()
        }

    lib = {}
    for M_R in M_R_values:
        lib[int(M_R)] = {}
        for lab in labels:
            rank_block = lib[int(M_R)].setdefault(lab.rank, {})
            rank_block[lab.angular_key()] = payload_to_library_entry(generate_generalized_payload(lab, M_R=M_R))
    _GENERALIZED_LIBRARY_CACHE.put(cache_key, lib)
    return lib


def generate_coefficient_table_for_labels(
    labels,
    M_R_values = None,
    coeff_tol = 1e-14,
    *,
    coeff_dtype = np.complex128,
    magnetic_dtype = np.int16,
):
    """Return a compact numeric coefficient table for labels sharing one output sector.

    This is the array-oriented companion to :func:`generate_library_for_labels`.
    It keeps one flattened term table plus offsets for each ``(basis label, M_R)``
    component, avoiding the nested payload dictionaries used for serialization.
    """
    labels = [normalize_compact_label(lab) for lab in labels]
    if not labels:
        return CoefficientTable(
            rank=0,
            L_R=0,
            n_tuple=tuple(),
            l_tuple=tuple(),
            tree_type="balanced",
            angular_keys=tuple(),
            basis_keys=tuple(),
            M_R_values=np.asarray([], dtype=np.int64),
            component_label_index=np.asarray([], dtype=np.int64),
            component_M_R=np.asarray([], dtype=np.int64),
            component_offsets=np.asarray([0], dtype=np.int64),
            magnetic_tuples=np.zeros((0, 0), dtype=magnetic_dtype),
            coeffs=np.asarray([], dtype=coeff_dtype),
        )
    first = labels[0]
    L_R = int(first.L_R)
    rank = int(first.rank)
    n_tuple = tuple(int(x) for x in first.n_tuple)
    l_tuple = tuple(int(x) for x in first.l_tuple)
    tree_type = str(first.tree_type)
    for lab in labels:
        if int(lab.L_R) != L_R:
            raise ValueError("All labels must share one final L_R.")
        if int(lab.rank) != rank:
            raise ValueError("All labels must have the same rank.")
    if M_R_values is None:
        M_R_values = range(-L_R, L_R + 1)
    M_R_values = tuple(int(M) for M in M_R_values)

    component_label_index = []
    component_M_R = []
    component_offsets = [0]
    magnetic_rows = []
    coeff_rows = []
    for label_index, lab in enumerate(labels):
        from ye3t.core.subtree_dag import expand_structured_label_numeric, expand_tree_key_numeric

        structured = _structured_label_from_compact(lab)
        if structured is not None:
            root_map = expand_structured_label_numeric(structured)
        else:
            root_map = expand_tree_key_numeric(_subtree_key_from_compact(lab))
        for M_R in M_R_values:
            component_label_index.append(int(label_index))
            component_M_R.append(int(M_R))
            block = root_map.get(int(M_R), {})
            for ms, coeff in sorted(block.items()):
                if abs(coeff) <= coeff_tol:
                    continue
                magnetic_rows.append(tuple(int(m) for m in ms))
                coeff_rows.append(complex(coeff))
            component_offsets.append(len(coeff_rows))

    magnetic_tuples = (
        np.asarray(magnetic_rows, dtype=magnetic_dtype)
        if magnetic_rows
        else np.zeros((0, rank), dtype=magnetic_dtype)
    )
    return CoefficientTable(
        rank=rank,
        L_R=L_R,
        n_tuple=n_tuple,
        l_tuple=l_tuple,
        tree_type=tree_type,
        angular_keys=tuple(lab.angular_key() for lab in labels),
        basis_keys=tuple(tuple(lab.basis_key) for lab in labels),
        M_R_values=np.asarray(M_R_values, dtype=np.int64),
        component_label_index=np.asarray(component_label_index, dtype=np.int64),
        component_M_R=np.asarray(component_M_R, dtype=np.int64),
        component_offsets=np.asarray(component_offsets, dtype=np.int64),
        magnetic_tuples=magnetic_tuples,
        coeffs=np.asarray(coeff_rows, dtype=coeff_dtype),
    )


def _label_block_specs(label_obj):
    if isinstance(label_obj, LeafLabel):
        return (
            {
                "kind": "leaf",
                "n": int(label_obj.n),
                "l": int(label_obj.l),
                "k_b": 1,
                "Lambda": int(label_obj.l),
                "multiplicity_index": 0,
                "basis_key": tuple(),
            },
        )
    if isinstance(label_obj, SymBlockLabel):
        return (
            {
                "kind": "sym",
                "n": int(label_obj.n),
                "l": int(label_obj.l),
                "k_b": int(label_obj.k_b),
                "Lambda": int(label_obj.Lambda),
                "multiplicity_index": int(label_obj.multiplicity_index),
                "basis_key": tuple(label_obj.basis_key),
            },
        )
    if isinstance(label_obj, NodeLabel):
        return _label_block_specs(label_obj.left) + _label_block_specs(label_obj.right)
    raise TypeError(f"Unsupported structured label type: {type(label_obj)!r}")


@lru_cache(maxsize=None)
def _cg_matrix_for_node(left_total, right_total, output_L):
    left_total = int(left_total)
    right_total = int(right_total)
    output_L = int(output_L)
    matrix = np.zeros((2 * left_total + 1, 2 * right_total + 1), dtype=np.complex128)
    for m1 in range(-left_total, left_total + 1):
        for m2 in range(-right_total, right_total + 1):
            M = int(m1 + m2)
            if abs(M) > output_L:
                continue
            value = cg_numeric_integer(left_total, m1, right_total, m2, output_L, M)
            if abs(value) > 1e-14:
                matrix[m1 + left_total, m2 + right_total] = value
    return matrix


@lru_cache(maxsize=None)
def _expand_block_m_path_arrays(label_obj):
    """Return root-M values, block-M tuples, and coefficients for a label tree."""
    if isinstance(label_obj, LeafLabel):
        m_values = np.arange(-int(label_obj.l), int(label_obj.l) + 1, dtype=np.int16)
        return (
            m_values.astype(np.int16, copy=False),
            m_values.reshape(-1, 1),
            np.ones(m_values.shape[0], dtype=np.complex128),
        )
    if isinstance(label_obj, SymBlockLabel):
        m_values = np.arange(-int(label_obj.Lambda), int(label_obj.Lambda) + 1, dtype=np.int16)
        return (
            m_values.astype(np.int16, copy=False),
            m_values.reshape(-1, 1),
            np.ones(m_values.shape[0], dtype=np.complex128),
        )
    if isinstance(label_obj, NodeLabel):
        left_M, left_tuples, left_coeffs = _expand_block_m_path_arrays(label_obj.left)
        right_M, right_tuples, right_coeffs = _expand_block_m_path_arrays(label_obj.right)
        left_count = int(left_M.shape[0])
        right_count = int(right_M.shape[0])
        if left_count == 0 or right_count == 0:
            return (
                np.zeros(0, dtype=np.int16),
                np.zeros((0, left_tuples.shape[1] + right_tuples.shape[1]), dtype=np.int16),
                np.zeros(0, dtype=np.complex128),
            )
        left_total = int(label_obj.left.total_L())
        right_total = int(label_obj.right.total_L())
        output_L = int(label_obj.L)
        left_idx = np.repeat(np.arange(left_count), right_count)
        right_idx = np.tile(np.arange(right_count), left_count)
        root_M = left_M[left_idx].astype(np.int16, copy=False) + right_M[right_idx].astype(np.int16, copy=False)
        valid = np.abs(root_M) <= output_L
        if not np.any(valid):
            return (
                np.zeros(0, dtype=np.int16),
                np.zeros((0, left_tuples.shape[1] + right_tuples.shape[1]), dtype=np.int16),
                np.zeros(0, dtype=np.complex128),
            )
        left_idx = left_idx[valid]
        right_idx = right_idx[valid]
        root_M = root_M[valid].astype(np.int16, copy=False)
        cg_matrix = _cg_matrix_for_node(left_total, right_total, output_L)
        cg_values = cg_matrix[
            left_M[left_idx].astype(np.int64) + left_total,
            right_M[right_idx].astype(np.int64) + right_total,
        ]
        nonzero = np.abs(cg_values) > 1e-14
        if not np.any(nonzero):
            return (
                np.zeros(0, dtype=np.int16),
                np.zeros((0, left_tuples.shape[1] + right_tuples.shape[1]), dtype=np.int16),
                np.zeros(0, dtype=np.complex128),
            )
        left_idx = left_idx[nonzero]
        right_idx = right_idx[nonzero]
        root_M = root_M[nonzero]
        tuples = np.concatenate((left_tuples[left_idx], right_tuples[right_idx]), axis=1)
        coeffs = left_coeffs[left_idx] * right_coeffs[right_idx] * cg_values[nonzero]
        return root_M, tuples, coeffs
    raise TypeError(f"Unsupported structured label type: {type(label_obj)!r}")


@lru_cache(maxsize=None)
def _expand_block_m_paths_reference(label_obj):
    if isinstance(label_obj, LeafLabel):
        return {
            int(m): {tuple([int(m)]): 1.0 + 0.0j}
            for m in range(-int(label_obj.l), int(label_obj.l) + 1)
        }
    if isinstance(label_obj, SymBlockLabel):
        return {
            int(M): {tuple([int(M)]): 1.0 + 0.0j}
            for M in range(-int(label_obj.Lambda), int(label_obj.Lambda) + 1)
        }
    if isinstance(label_obj, NodeLabel):
        left_map = _expand_block_m_paths(label_obj.left)
        right_map = _expand_block_m_paths(label_obj.right)
        left_total = int(label_obj.left.total_L())
        right_total = int(label_obj.right.total_L())
        out = {}
        for M1, d1 in left_map.items():
            for M2, d2 in right_map.items():
                M = int(M1 + M2)
                if abs(M) > int(label_obj.L):
                    continue
                coeff = clebsch_gordan(left_total, int(M1), right_total, int(M2), int(label_obj.L), M)
                if abs(coeff) < 1e-14:
                    continue
                target = out.setdefault(M, {})
                for ms1, c1 in d1.items():
                    for ms2, c2 in d2.items():
                        key = tuple(ms1 + ms2)
                        target[key] = target.get(key, 0.0 + 0.0j) + c1 * c2 * coeff
        return out
    raise TypeError(f"Unsupported structured label type: {type(label_obj)!r}")


def _expand_block_m_paths(label_obj):
    M_values, block_m_tuples, coeffs = _expand_block_m_path_arrays(label_obj)
    out = {}
    for M, ms, coeff in zip(M_values.tolist(), block_m_tuples.tolist(), coeffs.tolist()):
        out.setdefault(int(M), {})[tuple(int(v) for v in ms)] = complex(coeff)
    return out


# Use the SymPy-free factorized DAG helpers for the ACE fast constructor.  The
# local definitions above are retained as readable references; the raw/full
# magnetic paths below import ``subtree_dag`` lazily only when needed.
_label_block_specs = _fast_label_block_specs
_cg_matrix_for_node = _fast_cg_matrix_for_node
_expand_block_m_path_arrays = _fast_expand_block_m_path_arrays
_expand_block_m_paths_reference = _fast_expand_block_m_paths_reference


@lru_cache(maxsize=None)
def _real_cg_entries_cpu_cached(left_L, right_L, output_L):
    from ye3t.runtime.native import _real_cg_entries_cpu

    return tuple(
        (int(left), int(right), int(out), float(value))
        for left, right, out, value in _real_cg_entries_cpu(int(left_L), int(right_L), int(output_L))
        if abs(float(value)) > 1.0e-14
    )


@lru_cache(maxsize=None)
def _expand_block_real_path_arrays(label_obj):
    if isinstance(label_obj, LeafLabel):
        dim = 2 * int(label_obj.l) + 1
        return (
            np.arange(dim, dtype=np.int16),
            np.arange(dim, dtype=np.int16).reshape(dim, 1),
            np.ones(dim, dtype=np.float64),
        )
    if isinstance(label_obj, SymBlockLabel):
        dim = 2 * int(label_obj.Lambda) + 1
        return (
            np.arange(dim, dtype=np.int16),
            np.arange(dim, dtype=np.int16).reshape(dim, 1),
            np.ones(dim, dtype=np.float64),
        )
    if isinstance(label_obj, NodeLabel):
        left_out, left_tuples, left_coeffs = _expand_block_real_path_arrays(label_obj.left)
        right_out, right_tuples, right_coeffs = _expand_block_real_path_arrays(label_obj.right)
        entries = _real_cg_entries_cpu_cached(label_obj.left.total_L(), label_obj.right.total_L(), int(label_obj.L))
        if not entries or left_out.size == 0 or right_out.size == 0:
            return (
                np.zeros(0, dtype=np.int16),
                np.zeros((0, left_tuples.shape[1] + right_tuples.shape[1]), dtype=np.int16),
                np.zeros(0, dtype=np.float64),
            )
        left_by_out = {}
        right_by_out = {}
        for index, out_index in enumerate(left_out.tolist()):
            left_by_out.setdefault(int(out_index), []).append(index)
        for index, out_index in enumerate(right_out.tolist()):
            right_by_out.setdefault(int(out_index), []).append(index)
        out_chunks = []
        tuple_chunks = []
        coeff_chunks = []
        for left_component, right_component, output_component, value in entries:
            left_indices = left_by_out.get(int(left_component), ())
            right_indices = right_by_out.get(int(right_component), ())
            if not left_indices or not right_indices:
                continue
            left_idx = np.asarray(left_indices, dtype=np.int64)
            right_idx = np.asarray(right_indices, dtype=np.int64)
            grid_left, grid_right = np.meshgrid(left_idx, right_idx, indexing="ij")
            flat_left = grid_left.reshape(-1)
            flat_right = grid_right.reshape(-1)
            tuples = np.concatenate((left_tuples[flat_left], right_tuples[flat_right]), axis=1)
            coeffs = left_coeffs[flat_left] * right_coeffs[flat_right] * float(value)
            keep = np.abs(coeffs) > 1.0e-14
            if not np.any(keep):
                continue
            out_chunks.append(np.full(int(np.count_nonzero(keep)), int(output_component), dtype=np.int16))
            tuple_chunks.append(tuples[keep])
            coeff_chunks.append(coeffs[keep])
        if not out_chunks:
            return (
                np.zeros(0, dtype=np.int16),
                np.zeros((0, left_tuples.shape[1] + right_tuples.shape[1]), dtype=np.int16),
                np.zeros(0, dtype=np.float64),
            )
        return (
            np.concatenate(out_chunks, axis=0).astype(np.int16, copy=False),
            np.concatenate(tuple_chunks, axis=0).astype(np.int16, copy=False),
            np.concatenate(coeff_chunks, axis=0).astype(np.float64, copy=False),
        )
    raise TypeError(f"Unsupported structured label type: {type(label_obj)!r}")


def _empty_real_factorized_schedule(*, rank, L_R, n_tuple, l_tuple, tree_type, real_index_dtype, coeff_dtype, fallback_reason=None):
    return RealFactorizedCoefficientSchedule(
        rank=int(rank),
        block_count=0,
        L_R=int(L_R),
        n_tuple=tuple(int(x) for x in n_tuple),
        l_tuple=tuple(int(x) for x in l_tuple),
        tree_type=str(tree_type),
        block_specs=tuple(),
        angular_keys=tuple(),
        basis_keys=tuple(),
        component_indices=np.asarray([], dtype=np.int64),
        component_label_index=np.asarray([], dtype=np.int64),
        component_offsets=np.asarray([0], dtype=np.int64),
        block_real_tuples=np.zeros((0, 0), dtype=real_index_dtype),
        coeffs=np.asarray([], dtype=coeff_dtype),
        basis_convention="real_tesseral",
        permutation_sector="trivial",
        backend_provenance="real_cg_direct",
        complex_term_count=0,
        fallback_reason=fallback_reason,
    )


def _empty_factorized_schedule(*, rank, L_R, n_tuple, l_tuple, tree_type, magnetic_dtype, coeff_dtype):
    return FactorizedCoefficientSchedule(
        rank=int(rank),
        block_count=0,
        L_R=int(L_R),
        n_tuple=tuple(int(x) for x in n_tuple),
        l_tuple=tuple(int(x) for x in l_tuple),
        tree_type=str(tree_type),
        block_specs=tuple(),
        angular_keys=tuple(),
        basis_keys=tuple(),
        M_R_values=np.asarray([], dtype=np.int64),
        component_label_index=np.asarray([], dtype=np.int64),
        component_M_R=np.asarray([], dtype=np.int64),
        component_offsets=np.asarray([0], dtype=np.int64),
        block_m_tuples=np.zeros((0, 0), dtype=magnetic_dtype),
        coeffs=np.asarray([], dtype=coeff_dtype),
    )


def _direct_leaf_block_spec(n, l):
    return {
        "kind": "leaf",
        "n": int(n),
        "l": int(l),
        "k_b": 1,
        "Lambda": int(l),
        "multiplicity_index": 0,
        "basis_key": tuple(),
    }


def _direct_sym_block_spec(n, l, k_b, Lambda, multiplicity_index=0):
    basis_key = ("sym", int(Lambda), int(multiplicity_index))
    return {
        "kind": "sym",
        "n": int(n),
        "l": int(l),
        "k_b": int(k_b),
        "Lambda": int(Lambda),
        "multiplicity_index": int(multiplicity_index),
        "basis_key": basis_key,
    }


def _identity_factorized_schedule_for_one_block(
    *,
    rank,
    L_R,
    n_tuple,
    l_tuple,
    tree_type,
    block_spec,
    basis_key,
    coeff_tol,
    coeff_dtype,
    magnetic_dtype,
):
    label = CompactLabel(
        tuple(int(x) for x in n_tuple),
        tuple(int(x) for x in l_tuple),
        tuple() if int(rank) == 1 else (int(L_R),),
        tree_type=str(tree_type),
        basis_key=tuple(basis_key),
    )
    M_R_values = np.arange(-int(L_R), int(L_R) + 1, dtype=np.int64)
    component_count = int(M_R_values.shape[0])
    keep_identity_terms = not (coeff_tol > 0.0 and 1.0 <= float(coeff_tol))
    if keep_identity_terms:
        component_offsets = np.arange(component_count + 1, dtype=np.int64)
        block_m_tuples = M_R_values.astype(magnetic_dtype, copy=False).reshape(-1, 1)
        coeffs = np.ones(component_count, dtype=coeff_dtype)
    else:
        component_offsets = np.zeros(component_count + 1, dtype=np.int64)
        block_m_tuples = np.zeros((0, 1), dtype=magnetic_dtype)
        coeffs = np.asarray([], dtype=coeff_dtype)
    return FactorizedCoefficientSchedule(
        rank=int(rank),
        block_count=1,
        L_R=int(L_R),
        n_tuple=tuple(int(x) for x in n_tuple),
        l_tuple=tuple(int(x) for x in l_tuple),
        tree_type=str(tree_type),
        block_specs=((block_spec,),),
        angular_keys=(label.angular_key(),),
        basis_keys=(tuple(basis_key),),
        M_R_values=M_R_values,
        component_label_index=np.zeros(component_count, dtype=np.int64),
        component_M_R=M_R_values.copy(),
        component_offsets=component_offsets,
        block_m_tuples=block_m_tuples,
        coeffs=coeffs,
    )


def _rank_two_direct_cg_schedule(
    *,
    n_tuple,
    l_tuple,
    L_R,
    tree_type,
    coeff_tol,
    coeff_dtype,
    magnetic_dtype,
):
    n_tuple = tuple(int(x) for x in n_tuple)
    l_tuple = tuple(int(x) for x in l_tuple)
    l_left = int(l_tuple[0])
    l_right = int(l_tuple[1])
    label = CompactLabel(n_tuple, l_tuple, (int(L_R),), tree_type=str(tree_type), basis_key=tuple())
    M_R_values = np.arange(-int(L_R), int(L_R) + 1, dtype=np.int64)
    component_offsets = [0]
    block_m_rows = []
    coeff_rows = []
    cg_matrix = _cg_matrix_for_node(l_left, l_right, int(L_R))
    for M_R in M_R_values.tolist():
        for m_left in range(-l_left, l_left + 1):
            m_right = int(M_R) - int(m_left)
            if m_right < -l_right or m_right > l_right:
                continue
            coeff = cg_matrix[int(m_left) + l_left, int(m_right) + l_right]
            if coeff_tol > 0.0 and abs(coeff) <= coeff_tol:
                continue
            block_m_rows.append((int(m_left), int(m_right)))
            coeff_rows.append(coeff)
        component_offsets.append(len(coeff_rows))
    return FactorizedCoefficientSchedule(
        rank=2,
        block_count=2,
        L_R=int(L_R),
        n_tuple=n_tuple,
        l_tuple=l_tuple,
        tree_type=str(tree_type),
        block_specs=((_direct_leaf_block_spec(n_tuple[0], l_tuple[0]), _direct_leaf_block_spec(n_tuple[1], l_tuple[1])),),
        angular_keys=(label.angular_key(),),
        basis_keys=(tuple(),),
        M_R_values=M_R_values,
        component_label_index=np.zeros(int(M_R_values.shape[0]), dtype=np.int64),
        component_M_R=M_R_values.copy(),
        component_offsets=np.asarray(component_offsets, dtype=np.int64),
        block_m_tuples=np.asarray(block_m_rows, dtype=magnetic_dtype).reshape(-1, 2),
        coeffs=np.asarray(coeff_rows, dtype=coeff_dtype),
    )


def _low_rank_factorized_schedules_by_L(
    n_tuple,
    l_tuple,
    *,
    tree_type,
    coeff_tol,
    coeff_dtype,
    magnetic_dtype,
):
    if _env_flag("YE3T_DISABLE_LOW_RANK_FACTORIZED_FAST_PATHS"):
        return None
    n_tuple = tuple(int(x) for x in n_tuple)
    l_tuple = tuple(int(x) for x in l_tuple)
    rank = len(l_tuple)
    if rank == 1:
        L_R = int(l_tuple[0])
        return {
            L_R: _identity_factorized_schedule_for_one_block(
                rank=1,
                L_R=L_R,
                n_tuple=n_tuple,
                l_tuple=l_tuple,
                tree_type=tree_type,
                block_spec=_direct_leaf_block_spec(n_tuple[0], l_tuple[0]),
                basis_key=tuple(),
                coeff_tol=coeff_tol,
                coeff_dtype=coeff_dtype,
                magnetic_dtype=magnetic_dtype,
            )
        }
    if rank != 2:
        return None
    n_left, n_right = n_tuple
    l_left, l_right = l_tuple
    if int(n_left) == int(n_right) and int(l_left) == int(l_right):
        out = {}
        for L_R in range(0, 2 * int(l_left) + 1, 2):
            out[int(L_R)] = _identity_factorized_schedule_for_one_block(
                rank=2,
                L_R=int(L_R),
                n_tuple=n_tuple,
                l_tuple=l_tuple,
                tree_type=tree_type,
                block_spec=_direct_sym_block_spec(n_left, l_left, 2, int(L_R)),
                basis_key=("sym", int(L_R), 0),
                coeff_tol=coeff_tol,
                coeff_dtype=coeff_dtype,
                magnetic_dtype=magnetic_dtype,
            )
        return out
    return {
        int(L_R): _rank_two_direct_cg_schedule(
            n_tuple=n_tuple,
            l_tuple=l_tuple,
            L_R=int(L_R),
            tree_type=tree_type,
            coeff_tol=coeff_tol,
            coeff_dtype=coeff_dtype,
            magnetic_dtype=magnetic_dtype,
        )
        for L_R in range(abs(int(l_left) - int(l_right)), int(l_left) + int(l_right) + 1)
    }


def _factorized_schedule_from_structured_labels(
    labels,
    structured_labels,
    *,
    M_R_values = None,
    coeff_tol = 1e-14,
    coeff_dtype = np.complex128,
    magnetic_dtype = np.int16,
    constructor_backend = "auto",
):
    labels = [normalize_compact_label(lab) for lab in labels]
    structured_labels = list(structured_labels)
    if not labels:
        return _empty_factorized_schedule(
            rank=0,
            L_R=0,
            n_tuple=tuple(),
            l_tuple=tuple(),
            tree_type="balanced",
            magnetic_dtype=magnetic_dtype,
            coeff_dtype=coeff_dtype,
        )
    if len(labels) != len(structured_labels):
        raise ValueError("labels and structured_labels must have the same length.")
    first = labels[0]
    L_R = int(first.L_R)
    rank = int(first.rank)
    n_tuple = tuple(int(x) for x in first.n_tuple)
    l_tuple = tuple(int(x) for x in first.l_tuple)
    tree_type = str(first.tree_type)
    if M_R_values is None:
        M_R_values = range(-L_R, L_R + 1)
    M_R_values = tuple(int(M) for M in M_R_values)

    for lab in labels:
        if int(lab.L_R) != L_R:
            raise ValueError("All labels must share one final L_R.")
        if int(lab.rank) != rank:
            raise ValueError("All labels must have the same rank.")

    label_block_specs = tuple(_label_block_specs(structured) for structured in structured_labels)
    block_count = len(label_block_specs[0]) if label_block_specs else 0
    selected_backend = _selected_factorized_constructor_backend(constructor_backend)
    if selected_backend in {"cpp", "cpp_low_memory"}:
        from ye3t.core._factorized_runtime_cpp import (
            assemble_factorized_schedule_from_keys_cpp,
            assemble_factorized_schedule_from_keys_low_memory_cpp,
        )

        keys = [_fast_factorized_path_key(structured) for structured in structured_labels]
        assemble = assemble_factorized_schedule_from_keys_cpp
        if selected_backend == "cpp_low_memory":
            assemble = assemble_factorized_schedule_from_keys_low_memory_cpp
        (
            component_label_index_tensor,
            component_M_R_tensor,
            component_offsets_tensor,
            block_m_tuples_tensor,
            coeffs_tensor,
        ) = assemble(
            keys,
            M_R_values,
            coeff_tol,
        )
        component_label_index = component_label_index_tensor.numpy()
        component_M_R = component_M_R_tensor.numpy()
        component_offsets = component_offsets_tensor.numpy()
        block_m_tuples = np.asarray(block_m_tuples_tensor.numpy(), dtype=magnetic_dtype)
        coeffs = np.asarray(coeffs_tensor.numpy(), dtype=coeff_dtype)
    else:
        component_label_index = []
        component_M_R = []
        component_offsets = [0]
        block_m_chunks = []
        coeff_chunks = []
        for label_index, structured in enumerate(structured_labels):
            root_M, path_block_m_tuples, path_coeffs = _expand_block_m_path_arrays(structured)
            for M_R in M_R_values:
                component_label_index.append(int(label_index))
                component_M_R.append(int(M_R))
                keep = root_M == int(M_R)
                if coeff_tol > 0.0:
                    keep = keep & (np.abs(path_coeffs) > coeff_tol)
                count = int(np.count_nonzero(keep))
                if count:
                    block_m_chunks.append(path_block_m_tuples[keep])
                    coeff_chunks.append(path_coeffs[keep])
                component_offsets.append(component_offsets[-1] + count)

        block_m_tuples = (
            np.concatenate(block_m_chunks, axis=0).astype(magnetic_dtype, copy=False)
            if block_m_chunks
            else np.zeros((0, block_count), dtype=magnetic_dtype)
        )
        coeffs = (
            np.concatenate(coeff_chunks, axis=0).astype(coeff_dtype, copy=False)
            if coeff_chunks
            else np.asarray([], dtype=coeff_dtype)
        )
    return FactorizedCoefficientSchedule(
        rank=rank,
        block_count=block_count,
        L_R=L_R,
        n_tuple=n_tuple,
        l_tuple=l_tuple,
        tree_type=tree_type,
        block_specs=label_block_specs,
        angular_keys=tuple(lab.angular_key() for lab in labels),
        basis_keys=tuple(tuple(lab.basis_key) for lab in labels),
        M_R_values=np.asarray(M_R_values, dtype=np.int64),
        component_label_index=np.asarray(component_label_index, dtype=np.int64),
        component_M_R=np.asarray(component_M_R, dtype=np.int64),
        component_offsets=np.asarray(component_offsets, dtype=np.int64),
        block_m_tuples=block_m_tuples,
        coeffs=coeffs,
    )


def generate_factorized_coefficient_schedule_for_labels(
    labels,
    M_R_values = None,
    coeff_tol = 1e-14,
    *,
    coeff_dtype = np.complex128,
    magnetic_dtype = np.int16,
    constructor_backend = "auto",
):
    """Return a factorized inter-block CG coefficient schedule.

    This schedule keeps repeated-channel symmetric-power blocks collapsed.  It
    therefore constructs the ACE/YE3T coupling coefficients that sit between
    symmetry-adapted blocks without expanding those block basis vectors into all
    raw magnetic leaf tuples.  Full raw tensor materialization remains available
    through :func:`generate_coefficient_table_for_labels`.
    """
    labels = [normalize_compact_label(lab) for lab in labels]
    if not labels:
        return _empty_factorized_schedule(
            rank=0,
            L_R=0,
            n_tuple=tuple(),
            l_tuple=tuple(),
            tree_type="balanced",
            magnetic_dtype=magnetic_dtype,
            coeff_dtype=coeff_dtype,
        )
    first = labels[0]
    L_R = int(first.L_R)
    rank = int(first.rank)
    n_tuple = tuple(int(x) for x in first.n_tuple)
    l_tuple = tuple(int(x) for x in first.l_tuple)
    tree_type = str(first.tree_type)
    if M_R_values is None:
        M_R_values = range(-L_R, L_R + 1)
    M_R_values = tuple(int(M) for M in M_R_values)

    structured_labels = []
    for lab in labels:
        if int(lab.L_R) != L_R:
            raise ValueError("All labels must share one final L_R.")
        if int(lab.rank) != rank:
            raise ValueError("All labels must have the same rank.")
        structured = _lightweight_structured_label_from_compact(lab)
        if structured is None:
            structured = _structured_label_from_compact(lab)
        if structured is None:
            raise ValueError(f"Could not recover structured label for {lab!r}.")
        structured_labels.append(structured)

    return _factorized_schedule_from_structured_labels(
        labels,
        structured_labels,
        M_R_values=M_R_values,
        coeff_tol=coeff_tol,
        coeff_dtype=coeff_dtype,
        magnetic_dtype=magnetic_dtype,
        constructor_backend=constructor_backend,
    )


def _real_factorized_schedule_from_structured_labels(
    labels,
    structured_labels,
    *,
    component_indices = None,
    coeff_tol = 1e-14,
    coeff_dtype = np.float64,
    real_index_dtype = np.int16,
    complex_term_count = None,
):
    # TODO: mirror the validated rank-1/rank-2 direct fast paths
    # from the complex factorized schedule constructor in the real-tesseral
    # schedule path, with reference tests against this structured constructor.
    labels = [normalize_compact_label(lab) for lab in labels]
    structured_labels = list(structured_labels)
    if not labels:
        return _empty_real_factorized_schedule(
            rank=0,
            L_R=0,
            n_tuple=tuple(),
            l_tuple=tuple(),
            tree_type="balanced",
            real_index_dtype=real_index_dtype,
            coeff_dtype=coeff_dtype,
        )
    if len(labels) != len(structured_labels):
        raise ValueError("labels and structured_labels must have the same length.")
    first = labels[0]
    L_R = int(first.L_R)
    rank = int(first.rank)
    n_tuple = tuple(int(x) for x in first.n_tuple)
    l_tuple = tuple(int(x) for x in first.l_tuple)
    tree_type = str(first.tree_type)
    if component_indices is None:
        component_indices = range(0, 2 * L_R + 1)
    component_indices = tuple(int(component) for component in component_indices)
    for component in component_indices:
        if component < 0 or component >= 2 * L_R + 1:
            raise ValueError(f"Real component index {component} is outside V_{L_R}.")
    for lab in labels:
        if int(lab.L_R) != L_R:
            raise ValueError("All labels must share one final L_R.")
        if int(lab.rank) != rank:
            raise ValueError("All labels must have the same rank.")
    label_block_specs = tuple(_label_block_specs(structured) for structured in structured_labels)
    block_count = len(label_block_specs[0]) if label_block_specs else 0
    component_label_index = []
    component_offsets = [0]
    block_chunks = []
    coeff_chunks = []
    for label_index, structured in enumerate(structured_labels):
        root_component, block_real_tuples, coeffs = _expand_block_real_path_arrays(structured)
        for component in component_indices:
            component_label_index.append(int(label_index))
            keep = root_component == int(component)
            if coeff_tol > 0.0:
                keep = keep & (np.abs(coeffs) > float(coeff_tol))
            count = int(np.count_nonzero(keep))
            if count:
                block_chunks.append(block_real_tuples[keep])
                coeff_chunks.append(coeffs[keep])
            component_offsets.append(component_offsets[-1] + count)
    block_real_tuples = (
        np.concatenate(block_chunks, axis=0).astype(real_index_dtype, copy=False)
        if block_chunks
        else np.zeros((0, block_count), dtype=real_index_dtype)
    )
    coeffs = (
        np.concatenate(coeff_chunks, axis=0).astype(coeff_dtype, copy=False)
        if coeff_chunks
        else np.asarray([], dtype=coeff_dtype)
    )
    if complex_term_count is None:
        complex_term_count = -1
    return RealFactorizedCoefficientSchedule(
        rank=rank,
        block_count=block_count,
        L_R=L_R,
        n_tuple=n_tuple,
        l_tuple=l_tuple,
        tree_type=tree_type,
        block_specs=label_block_specs,
        angular_keys=tuple(lab.angular_key() for lab in labels),
        basis_keys=tuple(tuple(lab.basis_key) for lab in labels),
        component_indices=np.asarray(component_indices, dtype=np.int64),
        component_label_index=np.asarray(component_label_index, dtype=np.int64),
        component_offsets=np.asarray(component_offsets, dtype=np.int64),
        block_real_tuples=block_real_tuples,
        coeffs=coeffs,
        basis_convention="real_tesseral",
        permutation_sector="trivial",
        backend_provenance="real_cg_direct",
        complex_term_count=int(complex_term_count),
        fallback_reason=None,
    )


def generate_real_factorized_coefficient_schedule_for_labels(
    labels,
    component_indices = None,
    coeff_tol = 1e-14,
    *,
    coeff_dtype = np.float64,
    real_index_dtype = np.int16,
):
    """Return a native real-tesseral factorized ACE schedule.

    The schedule is assembled from real CG entries between collapsed blocks.
    The first public version targets the trivial permutation sector used by
    scalar ACE descriptors; other permutation sectors should use the existing
    complex/full paths until their real schedule provenance is added.
    """
    labels = [normalize_compact_label(lab) for lab in labels]
    if not labels:
        return _empty_real_factorized_schedule(
            rank=0,
            L_R=0,
            n_tuple=tuple(),
            l_tuple=tuple(),
            tree_type="balanced",
            real_index_dtype=real_index_dtype,
            coeff_dtype=coeff_dtype,
        )
    structured_labels = []
    for lab in labels:
        structured = _lightweight_structured_label_from_compact(lab)
        if structured is None:
            structured = _structured_label_from_compact(lab)
        if structured is None:
            raise ValueError(f"Could not recover structured label for {lab!r}.")
        structured_labels.append(structured)
    complex_schedule = _factorized_schedule_from_structured_labels(
        labels,
        structured_labels,
        M_R_values=range(-int(labels[0].L_R), int(labels[0].L_R) + 1),
        coeff_tol=coeff_tol,
        coeff_dtype=np.complex128,
        magnetic_dtype=np.int16,
        constructor_backend="auto",
    )
    return _real_factorized_schedule_from_structured_labels(
        labels,
        structured_labels,
        component_indices=component_indices,
        coeff_tol=coeff_tol,
        coeff_dtype=coeff_dtype,
        real_index_dtype=real_index_dtype,
        complex_term_count=int(complex_schedule.term_count),
    )


def generate_coefficient_schedule_for_labels(
    labels,
    M_R_values = None,
    coeff_tol = 1e-14,
    *,
    factorized = "auto",
    coeff_dtype = np.complex128,
    magnetic_dtype = np.int16,
    constructor_backend = "auto",
):
    """Return the default coefficient construction schedule for ACE labels.

    ``factorized='auto'`` uses the collapsed symmetric-power schedule for the
    standard ACE/trivial Young-subgroup image.  Pass ``factorized=False`` to
    force full raw magnetic tuple materialization.
    """
    use_factorized = str(factorized).strip().lower() in {"auto", "1", "true", "yes", "on", "factorized"}
    if use_factorized:
        try:
            return generate_factorized_coefficient_schedule_for_labels(
                labels,
                M_R_values=M_R_values,
                coeff_tol=coeff_tol,
                coeff_dtype=coeff_dtype,
                magnetic_dtype=magnetic_dtype,
                constructor_backend=constructor_backend,
            )
        except (TypeError, ValueError, NotImplementedError):
            if str(factorized).strip().lower() != "auto":
                raise
    return generate_coefficient_table_for_labels(
        labels,
        M_R_values=M_R_values,
        coeff_tol=coeff_tol,
        coeff_dtype=coeff_dtype,
        magnetic_dtype=magnetic_dtype,
    )


def _factorized_labels_and_structured_by_L(n_sig, l_sig, tree_type):
    from ye3t.core.basis import YE3TBasisLabeler

    labeler = YE3TBasisLabeler(list(n_sig), list(l_sig), strict_target_validation=False, tree_type=tree_type)
    labels_by_l = {
        int(L): [normalize_compact_label(label) for label in labels]
        for L, labels in labeler.compact_labels_by_L().items()
        if labels
    }
    if hasattr(labeler.backend, "lightweight_structured_label_objects_by_L"):
        structured_by_l = labeler.backend.lightweight_structured_label_objects_by_L()
    else:
        structured_by_l = {
            int(L): labeler.lightweight_structured_label_objects_for_target(int(L))
            for L in labels_by_l
        }
    return labels_by_l, structured_by_l


def iter_factorized_coefficient_schedules_by_L(
    n_in,
    l_in,
    *,
    tree_type = "balanced",
    coeff_tol = 1e-14,
    coeff_dtype = np.complex128,
    magnetic_dtype = np.int16,
    constructor_backend = "auto",
):
    """Yield all target-``L`` factorized schedules without retaining them all."""
    n_sig = tuple(int(x) for x in n_in)
    l_sig = tuple(int(x) for x in l_in)
    selected_constructor_backend = _selected_factorized_constructor_backend(constructor_backend)
    low_rank_schedules = _low_rank_factorized_schedules_by_L(
        n_sig,
        l_sig,
        tree_type=str(tree_type),
        coeff_tol=coeff_tol,
        coeff_dtype=coeff_dtype,
        magnetic_dtype=magnetic_dtype,
    )
    if low_rank_schedules is not None:
        for L in sorted(low_rank_schedules):
            yield int(L), low_rank_schedules[int(L)]
        return
    labels_by_l, structured_by_l = _factorized_labels_and_structured_by_L(n_sig, l_sig, str(tree_type))
    for L, labels in labels_by_l.items():
        yield int(L), _factorized_schedule_from_structured_labels(
            labels,
            structured_by_l.get(int(L), []),
            M_R_values=range(-int(L), int(L) + 1),
            coeff_tol=coeff_tol,
            coeff_dtype=coeff_dtype,
            magnetic_dtype=magnetic_dtype,
            constructor_backend=selected_constructor_backend,
        )


def generate_factorized_coefficient_schedules_by_L(
    n_in,
    l_in,
    *,
    tree_type = "balanced",
    coeff_tol = 1e-14,
    coeff_dtype = np.complex128,
    magnetic_dtype = np.int16,
    constructor_backend = "auto",
):
    """Build all target-``L`` factorized schedules for one ACE signature.

    This is the constructor hot path for repeated use: labels and lightweight
    structured trees are generated once for the full ``(n,l,tree_type)``
    signature, then every target schedule is assembled from that shared cache.
    """
    n_sig = tuple(int(x) for x in n_in)
    l_sig = tuple(int(x) for x in l_in)
    selected_constructor_backend = _selected_factorized_constructor_backend(constructor_backend)
    cache_key = (
        "factorized_schedules_by_L",
        n_sig,
        l_sig,
        str(tree_type),
        float(coeff_tol),
        np.dtype(coeff_dtype).str,
        np.dtype(magnetic_dtype).str,
        selected_constructor_backend,
    )
    cached = _FACTORIZED_SCHEDULES_CACHE.get(cache_key)
    if cached is not None:
        return cached
    low_rank_schedules = _low_rank_factorized_schedules_by_L(
        n_sig,
        l_sig,
        tree_type=str(tree_type),
        coeff_tol=coeff_tol,
        coeff_dtype=coeff_dtype,
        magnetic_dtype=magnetic_dtype,
    )
    if low_rank_schedules is not None:
        _FACTORIZED_SCHEDULES_CACHE.put(cache_key, low_rank_schedules)
        return low_rank_schedules
    schedules = {
        int(L): schedule
        for L, schedule in iter_factorized_coefficient_schedules_by_L(
            n_sig,
            l_sig,
            tree_type=tree_type,
            coeff_tol=coeff_tol,
            coeff_dtype=coeff_dtype,
            magnetic_dtype=magnetic_dtype,
            constructor_backend=selected_constructor_backend,
        )
    }
    _FACTORIZED_SCHEDULES_CACHE.put(cache_key, schedules)
    return schedules


def generate_torch_factorized_coefficient_schedule_for_labels(
    labels,
    M_R_values = None,
    coeff_tol = 1e-14,
    *,
    device = None,
    dtype = None,
    magnetic_dtype = np.int16,
    constructor_backend = "auto",
):
    """Build the default factorized ACE schedule directly in torch tensor form."""
    schedule = generate_factorized_coefficient_schedule_for_labels(
        labels,
        M_R_values=M_R_values,
        coeff_tol=coeff_tol,
        coeff_dtype=np.complex128,
        magnetic_dtype=magnetic_dtype,
        constructor_backend=constructor_backend,
    )
    return schedule.to_torch(device=device, dtype=dtype)


def generate_torch_real_factorized_coefficient_schedule_for_labels(
    labels,
    component_indices = None,
    coeff_tol = 1e-14,
    *,
    device = None,
    dtype = None,
    real_index_dtype = np.int16,
):
    """Build a native real-tesseral factorized schedule in torch tensor form."""
    schedule = generate_real_factorized_coefficient_schedule_for_labels(
        labels,
        component_indices=component_indices,
        coeff_tol=coeff_tol,
        coeff_dtype=np.float64,
        real_index_dtype=real_index_dtype,
    )
    return schedule.to_torch(device=device, dtype=dtype)


def generate_torch_factorized_coefficient_schedules_by_L(
    n_in,
    l_in,
    *,
    tree_type = "balanced",
    coeff_tol = 1e-14,
    device = None,
    dtype = None,
    magnetic_dtype = np.int16,
    constructor_backend = "auto",
):
    """Build all target-``L`` factorized schedules directly in torch tensor form."""
    schedules = generate_factorized_coefficient_schedules_by_L(
        n_in,
        l_in,
        tree_type=tree_type,
        coeff_tol=coeff_tol,
        coeff_dtype=np.complex128,
        magnetic_dtype=magnetic_dtype,
        constructor_backend=constructor_backend,
    )
    return {
        int(L): schedule.to_torch(device=device, dtype=dtype)
        for L, schedule in schedules.items()
    }
