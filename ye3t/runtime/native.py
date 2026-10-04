
"""Native ye3t runtime over ``E3OperatorIR`` and ``ExactSchedule``."""
from functools import lru_cache
import os

import torch

from ye3t.core.couplings import clebsch_gordan
from ye3t.core.basis.sector import ExactBasisHandle, ExactBasisSector
from ye3t.core.labels import CompactLabel, normalize_compact_label
from ye3t.core.product_descriptors import ExactProductColumnDescriptor
from ye3t.core.tesseral import complex_multiplet_to_real_tesseral, real_tesseral_to_complex_multiplet
from ye3t.paired_cg import couple_packed_real_tesseral
from ye3t.ir.model import E3OperatorIR
from ye3t.lowering.schedules import ExactSchedule
from .features import CallablePrimitiveFeatureSource, LabeledPrimitiveFeatureSource, PrimitiveFeatureSource
from .triton_config import configure_triton_c_compiler
from ye3t._record import recordclass

configure_triton_c_compiler(required=False)

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None


_validated_openequivariance_signatures = set()


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
    """Sparse real-basis CG entries matching ``_couple_packed`` semantics."""

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
                    raise ValueError(
                        "Real-basis CG construction produced a non-negligible imaginary coefficient."
                    )
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


def _descriptor_required_labels(descriptor):
    if descriptor.kind == 'primitive':
        if descriptor.basis_label is None:
            return tuple()
        return (normalize_compact_label(descriptor.basis_label),)
    labels = []
    if descriptor.left is not None:
        labels.extend(_descriptor_required_labels(descriptor.left))
    if descriptor.right is not None:
        labels.extend(_descriptor_required_labels(descriptor.right))
    return tuple(labels)


def _descriptor_required_handles(descriptor):
    if descriptor.kind == 'primitive':
        if descriptor.basis_handle is None:
            return tuple()
        return (descriptor.basis_handle,)
    handles = []
    if descriptor.left is not None:
        handles.extend(_descriptor_required_handles(descriptor.left))
    if descriptor.right is not None:
        handles.extend(_descriptor_required_handles(descriptor.right))
    return tuple(handles)


def _descriptor_order(descriptor):
    if descriptor.kind == 'primitive':
        return 1
    if descriptor.left is None or descriptor.right is None:
        return 1
    return _descriptor_order(descriptor.left) + _descriptor_order(descriptor.right)


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


def _flat_mul(left, right):
    if _triton_flat_mul_available(left, right):
        out = torch.empty_like(left)
        block = 256
        grid = (triton.cdiv(left.numel(), block),)
        try:
            _flat_mul_kernel[grid](left, right, out, left.numel())
            return out, "triton_scalar_mul"
        except Exception:
            pass
    return left * right, "torch_scalar_mul"


@recordclass(('left_L', 'right_L', 'out_L', 'descriptors', 'packed_support', 'product_order'), frozen = True)
class NativeRuntimeSegment:
    """Executable descriptor group retained by an exact lowering schedule."""

    @property
    def signature(self):
        return (int(self.left_L), int(self.right_L), int(self.out_L), int(self.product_order))


@recordclass(('ir', 'schedule', 'segments'), frozen = True)
class NativeRuntimePlan:
    """Runtime plan for one ye3t target sector."""

    @property
    def output_L(self):
        return int(self.ir.target_L)

    @property
    def output_dim(self):
        return sum(len(segment.descriptors) for segment in self.segments)


def build_native_runtime_plan(
    schedule,
    *,
    max_output_features_by_L = None,
):
    """Translate an exact schedule into executable descriptor groups."""
    ir = schedule.operator
    cap_by_L = {int(L): int(v) for L, v in dict(max_output_features_by_L or {}).items() if int(v) > 0}
    cap = cap_by_L.get(int(ir.target_L))
    remaining = None if cap is None else int(cap)
    retained_blocks = tuple(schedule.retained_blocks) if getattr(schedule, "retained_blocks", None) else tuple(ir.packed_blocks)
    blocks_by_key = {
        (int(block.key.left_L), int(block.key.right_L), int(block.key.out_L)): block
        for block in retained_blocks
    }
    segments = []
    for segment in schedule.segments:
        block = blocks_by_key.get((int(segment.left_L), int(segment.right_L), int(segment.out_L)))
        if block is None:
            continue
        descriptors = tuple(path.descriptor for path in block.paths)
        if remaining is not None:
            if remaining <= 0:
                descriptors = tuple()
            else:
                descriptors = descriptors[:remaining]
                remaining -= len(descriptors)
        if not descriptors:
            continue
        segments.append(
            NativeRuntimeSegment(
                left_L=int(segment.left_L),
                right_L=int(segment.right_L),
                out_L=int(segment.out_L),
                descriptors=descriptors,
                packed_support=tuple(segment.packed_support),
                product_order=max(_descriptor_order(descriptor) for descriptor in descriptors),
            )
        )
    return NativeRuntimePlan(ir=ir, schedule=schedule, segments=tuple(segments))


class NativeYE3TOperatorModule(torch.nn.Module):
    """Evaluate exact scheduled product descriptors from labeled primitives."""

    def __init__(
        self,
        schedule,
        *,
        labels_by_L = None,
        sectors_by_L = None,
        primitive_source = None,
        max_output_features_by_L = None,
        strict_labels = True,
        optimization_policy = "off",
        prefer_triton_runtime = False,
        prefer_openequivariance_runtime = False,
    ):
        super().__init__()
        self.plan = build_native_runtime_plan(schedule, max_output_features_by_L=max_output_features_by_L)
        self.optimization_policy = str(optimization_policy)
        self.prefer_triton_runtime = bool(prefer_triton_runtime)
        self.prefer_openequivariance_runtime = bool(prefer_openequivariance_runtime)
        if primitive_source is None:
            if labels_by_L is None and sectors_by_L is None:
                raise TypeError("labels_by_L or sectors_by_L is required unless primitive_source is provided.")
            primitive_source = LabeledPrimitiveFeatureSource(labels_by_L=labels_by_L, sectors_by_L=sectors_by_L)
        elif not (
            hasattr(primitive_source, "primitive_feature")
            and hasattr(primitive_source, "validate_required_labels")
        ):
            primitive_source = CallablePrimitiveFeatureSource(primitive_source)
        self.primitive_source = primitive_source
        if sectors_by_L is not None:
            self.sectors_by_L = {int(L): sector for L, sector in sectors_by_L.items()}
            self.labels_by_L = {
                int(L): tuple(normalize_compact_label(entry.compact_label) for entry in sector.entries)
                for L, sector in self.sectors_by_L.items()
            }
        else:
            self.sectors_by_L = {}
            self.labels_by_L = (
                {}
                if labels_by_L is None
                else {
                    int(L): tuple(normalize_compact_label(label) for label in labels)
                    for L, labels in labels_by_L.items()
                }
            )
        self.strict_labels = bool(strict_labels)
        required = sorted(
            {
                label
                for segment in self.plan.segments
                for descriptor in segment.descriptors
                for label in _descriptor_required_labels(descriptor)
            },
            key=lambda label: (label.L_R, label.rank, label.n_tuple, label.l_tuple, label.internal_Ls, label.basis_key),
        )
        required_handles = sorted(
            {
                handle
                for segment in self.plan.segments
                for descriptor in segment.descriptors
                for handle in _descriptor_required_handles(descriptor)
            },
            key=lambda handle: (
                handle.sector.L_R,
                handle.sector.rank,
                handle.sector.nin,
                handle.sector.lin,
                handle.basis_index,
            ),
        )
        missing_handles = tuple()
        if hasattr(self.primitive_source, "validate_required_handles"):
            missing_handles = self.primitive_source.validate_required_handles(tuple(required_handles), strict=bool(strict_labels))
        self.missing_handles = tuple(missing_handles)
        missing = self.primitive_source.validate_required_labels(tuple(required), strict=bool(strict_labels))
        self.missing_labels = tuple(missing)
        self.active_output_hidden_dims_by_L = {self.plan.output_L: self.plan.output_dim} if self.plan.output_dim else {}
        self._last_backend_counts = {}

    def backend_report(self):
        return {
            'counts': dict(self._last_backend_counts),
            'target_L': int(self.plan.output_L),
            'target_nin': tuple(self.plan.ir.target_nin),
            'target_lin': tuple(self.plan.ir.target_lin),
            'schedule_segment_count': len(self.plan.segments),
            'packed_path_count': int(self.plan.schedule.packed_path_count),
            'eliminated_path_count': int(self.plan.schedule.eliminated_path_count),
            'missing_label_count': len(self.missing_labels),
            'missing_handle_count': len(self.missing_handles),
            'active_output_hidden_dims_by_L': dict(self.active_output_hidden_dims_by_L),
            'optimization_policy': str(self.optimization_policy),
            'prefer_triton_runtime': bool(self.prefer_triton_runtime),
            'prefer_openequivariance_runtime': bool(self.prefer_openequivariance_runtime),
            'segment_signatures': [segment.signature for segment in self.plan.segments],
            'flat_schedule_group_count': (
                int(self.plan.schedule.flat_metadata.metadata.get("group_count", 0))
                if self.plan.schedule.flat_metadata is not None
                else 0
            ),
            'materialized_segment_count': (
                int(self.plan.schedule.materialization_plan.materialized_count)
                if self.plan.schedule.materialization_plan is not None
                else 0
            ),
            'recomputed_segment_count': (
                int(self.plan.schedule.materialization_plan.recomputed_count)
                if self.plan.schedule.materialization_plan is not None
                else 0
            ),
        }

    def _primitive_feature(self, descriptor, features_by_L):
        return self.primitive_source.primitive_feature(descriptor, features_by_L)

    @staticmethod
    def _couple_single(left, right, L1, L2, Lout):
        left_complex = real_tesseral_to_complex_multiplet(left, int(L1))
        right_complex = real_tesseral_to_complex_multiplet(right, int(L2))
        cg = _cg_tensor_cpu(int(L1), int(L2), int(Lout)).to(dtype=left_complex.dtype, device=left_complex.device)
        coupled = torch.einsum('nm,np,mpr->nr', left_complex, right_complex, cg)
        if (int(L1) + int(L2) - int(Lout)) % 2:
            coupled = -1j * coupled
        real = complex_multiplet_to_real_tesseral(
            coupled,
            L=int(Lout),
            M_values=tuple(range(-int(Lout), int(Lout) + 1)),
        )
        if int(Lout) == 0:
            return real.squeeze(-1)
        return real

    def _eval_descriptor(
        self,
        descriptor,
        features_by_L,
        memo,
    ):
        if descriptor in memo:
            return memo[descriptor]
        if descriptor.kind == 'primitive':
            value = self._primitive_feature(descriptor, features_by_L)
            if int(descriptor.L_R) == 0 and value.ndim == 2 and value.shape[-1] == 1:
                value = value.squeeze(-1)
        else:
            if descriptor.left is None or descriptor.right is None:
                raise ValueError("Product descriptor is missing a child descriptor.")
            left = self._eval_descriptor(descriptor.left, features_by_L, memo)
            right = self._eval_descriptor(descriptor.right, features_by_L, memo)
            value = self._couple_single(left, right, descriptor.left.L_R, descriptor.right.L_R, descriptor.L_R)
        memo[descriptor] = value
        return value

    @staticmethod
    def _couple_packed_openequivariance(left, right, L1, L2, Lout):
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
            return None
        if result is None:
            return None
        out, backend = result
        if int(Lout) == 0:
            return out.squeeze(-1), str(backend)
        return out, str(backend)

    @staticmethod
    def _couple_packed_triton(left, right, L1, L2, Lout):
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
                return_backend=True,
            )
        except Exception:
            return None
        if int(Lout) == 0:
            return out.squeeze(-1), str(backend)
        return out, str(backend)

    @staticmethod
    def _couple_packed(
        left,
        right,
        L1,
        L2,
        Lout,
        *,
        prefer_openequivariance = False,
        prefer_triton = False,
    ):
        backend = (
            "openequivariance"
            if bool(prefer_openequivariance)
            else "triton"
            if bool(prefer_triton)
            else "pytorch"
        )
        return couple_packed_real_tesseral(
            left,
            right,
            int(L1),
            int(L2),
            int(Lout),
            backend=backend,
        )

    def _eval_segment(
        self,
        segment,
        features_by_L,
        memo,
    ):
        if int(segment.left_L) < 0:
            values = [self._eval_descriptor(descriptor, features_by_L, memo) for descriptor in segment.descriptors]
            return torch.stack(values, dim=1)
        left_values = []
        right_values = []
        product_descriptors = []
        for descriptor in segment.descriptors:
            if descriptor.kind != 'product' or descriptor.left is None or descriptor.right is None:
                values = [self._eval_descriptor(d, features_by_L, memo) for d in segment.descriptors]
                return torch.stack(values, dim=1)
            left_values.append(self._eval_descriptor(descriptor.left, features_by_L, memo))
            right_values.append(self._eval_descriptor(descriptor.right, features_by_L, memo))
            product_descriptors.append(descriptor)
        left = torch.stack(left_values, dim=1)
        right = torch.stack(right_values, dim=1)
        packed, backend = self._couple_packed(
            left,
            right,
            segment.left_L,
            segment.right_L,
            segment.out_L,
            prefer_openequivariance=bool(self.prefer_openequivariance_runtime),
            prefer_triton=bool(self.prefer_triton_runtime),
        )
        self._last_backend_counts[f"packed_backend:{backend}"] = self._last_backend_counts.get(f"packed_backend:{backend}", 0) + len(product_descriptors)
        for idx, descriptor in enumerate(product_descriptors):
            memo[descriptor] = packed[:, idx] if int(segment.out_L) == 0 else packed[:, idx, :]
        return packed

    def forward(self, features_by_L):
        self._last_backend_counts = {}
        memo = {}
        blocks = []
        for segment in self.plan.segments:
            block = self._eval_segment(segment, features_by_L, memo)
            blocks.append(block)
            legacy_key = f"native_segment:{segment.left_L}:{segment.right_L}:{segment.out_L}"
            self._last_backend_counts[legacy_key] = self._last_backend_counts.get(legacy_key, 0) + int(block.shape[1])
            key = f"native_segment:{segment.left_L}:{segment.right_L}:{segment.out_L}:order{segment.product_order}"
            self._last_backend_counts[key] = self._last_backend_counts.get(key, 0) + int(block.shape[1])
        if not blocks:
            return {}
        return {int(self.plan.output_L): torch.cat(blocks, dim=1)}


class MultiNativeYE3TOperatorModule(torch.nn.Module):
    """Merge several target-sector native ye3t modules."""

    def __init__(self, modules):
        super().__init__()
        self.modules_by_index = torch.nn.ModuleList(list(modules))

    def backend_report(self):
        return {
            'target_reports': [
                module.backend_report()
                for module in self.modules_by_index
                if hasattr(module, 'backend_report')
            ]
        }

    def forward(self, features_by_L):
        merged = {}
        for module in self.modules_by_index:
            out = module(features_by_L)
            for L, tensor in out.items():
                merged.setdefault(int(L), []).append(tensor)
        return {L: torch.cat(blocks, dim=1) for L, blocks in merged.items()}

__all__ = [
    'NativeRuntimeSegment',
    'NativeRuntimePlan',
    'NativeYE3TOperatorModule',
    'MultiNativeYE3TOperatorModule',
    'build_native_runtime_plan',
]
