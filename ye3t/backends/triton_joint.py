"""Packed sparse bilinear kernels for exact real-basis YE3T products.

The representation compiler supplies one fixed sparse table whose terms are

``out[b, o] += coefficient[q] * left[b, i] * right[b, j]``.

All Young, angular, source-placement, and basis-convention decisions are made
before this module is called.  The kernel only evaluates the compiled numeric
table.
"""

import os
import warnings

import torch

from ye3t._record import recordclass
from ye3t.runtime.triton_config import configure_triton_c_compiler


configure_triton_c_compiler(required=False)

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None


@recordclass(
    ("left_index", "right_index", "output_index", "coefficient", "output_width"),
    frozen=True,
)
class SparseBilinearTable:
    """One packed exact bilinear schedule."""

    def to(self, *, device, dtype):
        return SparseBilinearTable(
            left_index=self.left_index.to(device=device, dtype=torch.long).contiguous(),
            right_index=self.right_index.to(device=device, dtype=torch.long).contiguous(),
            output_index=self.output_index.to(device=device, dtype=torch.long).contiguous(),
            coefficient=self.coefficient.to(device=device, dtype=dtype).contiguous(),
            output_width=int(self.output_width),
        )

    @property
    def term_count(self):
        return int(self.coefficient.numel())


@recordclass(
    (
        "other_index",
        "output_index",
        "coefficient",
        "source_width",
        "output_width",
    ),
    frozen=True,
)
class SparseQuadraticAdjointTable:
    """Compiler-expanded derivative table for one packed quadratic form."""

    def to(self, *, device, dtype):
        return SparseQuadraticAdjointTable(
            other_index=self.other_index.to(
                device=device,
                dtype=torch.long,
            ).contiguous(),
            output_index=self.output_index.to(
                device=device,
                dtype=torch.long,
            ).contiguous(),
            coefficient=self.coefficient.to(
                device=device,
                dtype=dtype,
            ).contiguous(),
            source_width=int(self.source_width),
            output_width=int(self.output_width),
        )

    @property
    def term_count(self):
        return int(self.coefficient.numel())


@recordclass(
    (
        "left_index",
        "right_index",
        "output_index",
        "weight_index",
        "coefficient",
        "output_width",
        "weight_count",
    ),
    frozen=True,
)
class WeightedSparseBilinearTable:
    """One packed bilinear schedule with trainable multiplicity weights."""

    def to(self, *, device, dtype):
        return WeightedSparseBilinearTable(
            left_index=self.left_index.to(device=device, dtype=torch.long).contiguous(),
            right_index=self.right_index.to(device=device, dtype=torch.long).contiguous(),
            output_index=self.output_index.to(device=device, dtype=torch.long).contiguous(),
            weight_index=self.weight_index.to(device=device, dtype=torch.long).contiguous(),
            coefficient=self.coefficient.to(device=device, dtype=dtype).contiguous(),
            output_width=int(self.output_width),
            weight_count=int(self.weight_count),
        )

    @property
    def term_count(self):
        return int(self.coefficient.numel())


@recordclass(
    (
        "input_index",
        "output_index",
        "weight_index",
        "coefficient",
        "output_width",
        "weight_count",
    ),
    frozen=True,
)
class WeightedSparseLinearTable:
    """One packed linear analysis schedule with trainable channel weights."""

    def to(self, *, device, dtype):
        return WeightedSparseLinearTable(
            input_index=self.input_index.to(
                device=device,
                dtype=torch.long,
            ).contiguous(),
            output_index=self.output_index.to(
                device=device,
                dtype=torch.long,
            ).contiguous(),
            weight_index=self.weight_index.to(
                device=device,
                dtype=torch.long,
            ).contiguous(),
            coefficient=self.coefficient.to(
                device=device,
                dtype=dtype,
            ).contiguous(),
            output_width=int(self.output_width),
            weight_count=int(self.weight_count),
        )

    @property
    def term_count(self):
        return int(self.coefficient.numel())


@recordclass(
    (
        "order",
        "inverse_order",
        "lengths",
        "offsets",
        "segment_ids",
        "segment_count",
        "max_segment_length",
        "bucket_ranges",
        "bucket_segment_ids",
    ),
    frozen=True,
)
class SortedSegmentPlan:
    """One compiler-time fixed ordering for a sparse reduction axis."""

    def to(self, *, device):
        return SortedSegmentPlan(
            order=self.order.to(
                device=device,
                dtype=torch.long,
            ).contiguous(),
            inverse_order=self.inverse_order.to(
                device=device,
                dtype=torch.long,
            ).contiguous(),
            lengths=self.lengths.to(
                device=device,
                dtype=torch.long,
            ).contiguous(),
            offsets=self.offsets.to(
                device=device,
                dtype=torch.long,
            ).contiguous(),
            segment_ids=self.segment_ids.to(
                device=device,
                dtype=torch.long,
            ).contiguous(),
            segment_count=int(self.segment_count),
            max_segment_length=int(self.max_segment_length),
            bucket_ranges=tuple(
                tuple(int(value) for value in record)
                for record in self.bucket_ranges
            ),
            bucket_segment_ids=self.bucket_segment_ids.to(
                device=device,
                dtype=torch.long,
            ).contiguous(),
        )


def _sorted_segment_plan(indices, segment_count):
    indices = torch.as_tensor(indices, dtype=torch.long, device="cpu")
    segment_count = int(segment_count)
    if indices.ndim != 1:
        raise ValueError("Sorted segment indices must be one-dimensional.")
    if segment_count <= 0:
        raise ValueError("Sorted segment count must be positive.")
    if int(indices.numel()):
        if int(indices.min()) < 0 or int(indices.max()) >= segment_count:
            raise ValueError("Sorted segment index is out of range.")
        order = torch.argsort(indices, stable=True)
        sorted_indices = indices.index_select(0, order)
        lengths = torch.bincount(
            sorted_indices,
            minlength=segment_count,
        )
    else:
        order = torch.empty((0,), dtype=torch.long)
        lengths = torch.zeros((segment_count,), dtype=torch.long)
    offsets = torch.cat(
        (
            torch.zeros((1,), dtype=torch.long),
            lengths.cumsum(0),
        )
    )
    segment_ids = torch.repeat_interleave(
        torch.arange(segment_count, dtype=torch.long),
        lengths,
    )
    inverse_order = torch.empty_like(order)
    if int(order.numel()):
        inverse_order[order] = torch.arange(
            int(order.numel()),
            dtype=torch.long,
        )
    bucket_members = {}
    for segment_index, length in enumerate(lengths.tolist()):
        block_terms = 1 << (max(1, int(length)) - 1).bit_length()
        bucket_members.setdefault(int(block_terms), []).append(
            int(segment_index)
        )
    bucket_ranges = []
    bucket_segment_ids = []
    bucket_start = 0
    for block_terms in sorted(bucket_members):
        members = tuple(bucket_members[int(block_terms)])
        bucket_segment_ids.extend(members)
        bucket_stop = int(bucket_start + len(members))
        bucket_ranges.append(
            (int(block_terms), int(bucket_start), int(bucket_stop))
        )
        bucket_start = int(bucket_stop)
    return SortedSegmentPlan(
        order=order.contiguous(),
        inverse_order=inverse_order.contiguous(),
        lengths=lengths.contiguous(),
        offsets=offsets.contiguous(),
        segment_ids=segment_ids.contiguous(),
        segment_count=segment_count,
        max_segment_length=(
            int(lengths.max()) if int(lengths.numel()) else 0
        ),
        bucket_ranges=tuple(bucket_ranges),
        bucket_segment_ids=torch.tensor(
            bucket_segment_ids,
            dtype=torch.long,
        ).contiguous(),
    )


def _validate_sparse_bilinear_table_indices(table, left_width, right_width):
    lengths = {
        int(table.left_index.numel()),
        int(table.right_index.numel()),
        int(table.output_index.numel()),
        int(table.coefficient.numel()),
    }
    if len(lengths) != 1:
        raise ValueError("Packed sparse bilinear table arrays must have the same length.")
    if int(table.output_width) < 1:
        raise ValueError("Packed sparse bilinear output_width must be positive.")
    if int(table.coefficient.numel()) == 0:
        return
    if int(table.left_index.min()) < 0 or int(table.left_index.max()) >= int(left_width):
        raise ValueError("Packed sparse bilinear left_index is out of range.")
    if int(table.right_index.min()) < 0 or int(table.right_index.max()) >= int(right_width):
        raise ValueError("Packed sparse bilinear right_index is out of range.")
    if int(table.output_index.min()) < 0 or int(table.output_index.max()) >= int(table.output_width):
        raise ValueError("Packed sparse bilinear output_index is out of range.")


def _shared_contiguous_packed_view(source_values, source_widths):
    if not source_values:
        return None
    packed_width = int(sum(int(width) for width in source_widths))
    first = source_values[0]
    if int(first.stride(1)) != 1 or int(first.stride(0)) != packed_width:
        return None
    common_base = getattr(first, "_base", None)
    if common_base is None:
        return None
    if (
        common_base.ndim != 2
        or int(common_base.shape[0]) != int(first.shape[0])
        or int(common_base.shape[1]) != packed_width
        or int(common_base.stride(0)) != packed_width
        or int(common_base.stride(1)) != 1
    ):
        return None
    storage_pointer = int(first.untyped_storage().data_ptr())
    expected_offset = int(first.storage_offset())
    for value, width in zip(source_values, source_widths):
        if getattr(value, "_base", None) is not common_base:
            return None
        if int(value.untyped_storage().data_ptr()) != storage_pointer:
            return None
        if int(value.stride(0)) != packed_width or int(value.stride(1)) != 1:
            return None
        if int(value.storage_offset()) != expected_offset:
            return None
        expected_offset += int(width)
    if int(common_base.storage_offset()) != int(first.storage_offset()):
        return None
    return common_base


class PackedSparseBilinearGroup(torch.nn.Module):
    """One packed launch spanning several compiler-resolved binary products."""

    def __init__(
        self,
        tables,
        input_pairs,
        source_widths,
        *,
        strict=False,
        source_storage="declared_real_basis",
        reduction_mode="auto",
    ):
        super().__init__()
        tables = tuple(tables)
        input_pairs = tuple(
            (int(pair[0]), int(pair[1]))
            for pair in tuple(input_pairs)
        )
        source_widths = tuple(int(width) for width in tuple(source_widths))
        if not tables:
            raise ValueError("PackedSparseBilinearGroup requires at least one table.")
        if len(tables) != len(input_pairs):
            raise ValueError("Packed sparse bilinear tables and input pairs must have equal length.")
        if not source_widths or any(int(width) <= 0 for width in source_widths):
            raise ValueError("Packed sparse bilinear source widths must be positive.")
        source_storage = str(source_storage)
        if source_storage not in {
            "declared_real_basis",
            "real_split_primary_complex",
            "interleaved_real_imag_primary_complex",
        }:
            raise ValueError(
                "Packed sparse bilinear source_storage is unsupported."
            )
        source_offsets = []
        offset = 0
        for width in source_widths:
            source_offsets.append(int(offset))
            offset += int(width)
        left_indices = []
        right_indices = []
        output_indices = []
        coefficients = []
        output_slices = []
        output_offset = 0
        for path_index, (table, pair) in enumerate(zip(tables, input_pairs)):
            left_source, right_source = pair
            if left_source < 0 or left_source >= len(source_widths):
                raise ValueError("Packed sparse bilinear left source index is out of range.")
            if right_source < 0 or right_source >= len(source_widths):
                raise ValueError("Packed sparse bilinear right source index is out of range.")
            _validate_sparse_bilinear_table_indices(
                table,
                source_widths[left_source],
                source_widths[right_source],
            )
            left_indices.append(table.left_index.to(torch.long) + int(source_offsets[left_source]))
            right_indices.append(table.right_index.to(torch.long) + int(source_offsets[right_source]))
            output_indices.append(table.output_index.to(torch.long) + int(output_offset))
            coefficients.append(table.coefficient.to(torch.float64))
            output_slices.append(
                {
                    "path_index": int(path_index),
                    "start": int(output_offset),
                    "stop": int(output_offset + int(table.output_width)),
                    "width": int(table.output_width),
                    "left_source_index": int(left_source),
                    "right_source_index": int(right_source),
                }
            )
            output_offset += int(table.output_width)
        self.register_buffer(
            "left_index",
            torch.cat(tuple(left_indices)),
            persistent=False,
        )
        self.register_buffer(
            "right_index",
            torch.cat(tuple(right_indices)),
            persistent=False,
        )
        self.register_buffer(
            "output_index",
            torch.cat(tuple(output_indices)),
            persistent=False,
        )
        self.register_buffer(
            "coefficient",
            torch.cat(tuple(coefficients)),
            persistent=False,
        )
        derivative_destination = torch.cat(
            (self.left_index, self.right_index)
        )
        derivative_other = torch.cat(
            (self.right_index, self.left_index)
        )
        derivative_output = torch.cat(
            (self.output_index, self.output_index)
        )
        derivative_coefficient = torch.cat(
            (self.coefficient, self.coefficient)
        )
        self.register_buffer(
            "quadratic_derivative_other_index",
            derivative_other,
            persistent=False,
        )
        self.register_buffer(
            "quadratic_derivative_output_index",
            derivative_output,
            persistent=False,
        )
        self.register_buffer(
            "quadratic_derivative_coefficient",
            derivative_coefficient,
            persistent=False,
        )
        self.register_buffer(
            "quadratic_cross_left_index",
            derivative_destination,
            persistent=False,
        )
        self.register_buffer(
            "quadratic_cross_right_index",
            derivative_other,
            persistent=False,
        )
        self.register_buffer(
            "quadratic_cross_output_index",
            derivative_output,
            persistent=False,
        )
        self.register_buffer(
            "quadratic_cross_coefficient",
            derivative_coefficient.detach().clone(),
            persistent=False,
        )
        segment_plans = {
            "quadratic_output": _sorted_segment_plan(
                self.output_index,
                int(output_offset),
            ),
            "quadratic_adjoint": _sorted_segment_plan(
                derivative_destination,
                int(offset),
            ),
            "quadratic_cross": _sorted_segment_plan(
                derivative_output,
                int(output_offset),
            ),
        }
        for name, plan in segment_plans.items():
            self.register_buffer(
                name + "_segment_order",
                plan.order,
                persistent=False,
            )
            self.register_buffer(
                name + "_segment_inverse_order",
                plan.inverse_order,
                persistent=False,
            )
            self.register_buffer(
                name + "_segment_lengths",
                plan.lengths,
                persistent=False,
            )
            self.register_buffer(
                name + "_segment_offsets",
                plan.offsets,
                persistent=False,
            )
            self.register_buffer(
                name + "_segment_ids",
                plan.segment_ids,
                persistent=False,
            )
            self.register_buffer(
                name + "_bucket_segment_ids",
                plan.bucket_segment_ids,
                persistent=False,
            )
            setattr(
                self,
                name + "_bucket_ranges",
                tuple(plan.bucket_ranges),
            )
            setattr(
                self,
                name + "_max_segment_length",
                int(plan.max_segment_length),
            )
        self.source_widths = source_widths
        self.source_offsets = tuple(source_offsets)
        self.input_pairs = input_pairs
        self.output_slices = tuple(output_slices)
        self.output_width = int(output_offset)
        self.strict = bool(strict)
        self.source_storage = source_storage
        reduction_mode = str(reduction_mode).strip().lower()
        if reduction_mode not in {
            "auto",
            "atomic",
            "segmented",
            "stable",
        }:
            raise ValueError(
                "Packed sparse bilinear reduction_mode must be 'auto', "
                "'atomic', 'segmented', or 'stable'."
            )
        self.reduction_mode = reduction_mode
        self.last_backend = None
        self.last_source_pack_operation = "not_run"

    def _table(self):
        return SparseBilinearTable(
            left_index=self.left_index,
            right_index=self.right_index,
            output_index=self.output_index,
            coefficient=self.coefficient,
            output_width=int(self.output_width),
        )

    def _segment_plan(self, name):
        return SortedSegmentPlan(
            order=getattr(self, name + "_segment_order"),
            inverse_order=getattr(
                self,
                name + "_segment_inverse_order",
            ),
            lengths=getattr(self, name + "_segment_lengths"),
            offsets=getattr(self, name + "_segment_offsets"),
            segment_ids=getattr(self, name + "_segment_ids"),
            segment_count=int(
                getattr(self, name + "_segment_lengths").numel()
            ),
            max_segment_length=int(
                getattr(self, name + "_max_segment_length")
            ),
            bucket_ranges=tuple(
                getattr(self, name + "_bucket_ranges")
            ),
            bucket_segment_ids=getattr(
                self,
                name + "_bucket_segment_ids",
            ),
        )

    def _quadratic_adjoint_table(self):
        return SparseQuadraticAdjointTable(
            other_index=self.quadratic_derivative_other_index,
            output_index=self.quadratic_derivative_output_index,
            coefficient=self.quadratic_derivative_coefficient,
            source_width=int(sum(self.source_widths)),
            output_width=int(self.output_width),
        )

    def _quadratic_cross_table(self):
        return SparseBilinearTable(
            left_index=self.quadratic_cross_left_index,
            right_index=self.quadratic_cross_right_index,
            output_index=self.quadratic_cross_output_index,
            coefficient=self.quadratic_cross_coefficient,
            output_width=int(self.output_width),
        )

    def pack_sources(self, source_values):
        source_values = tuple(source_values)
        if len(source_values) != len(self.source_widths):
            raise ValueError("Packed product source count does not match the compiled layout.")
        if not source_values:
            raise ValueError("Packed product sources must not be empty.")
        batch_size = int(source_values[0].shape[0])
        dtype = source_values[0].dtype
        device = source_values[0].device
        for source_index, (value, width) in enumerate(zip(source_values, self.source_widths)):
            if value.ndim != 2:
                raise ValueError("Packed product sources must have shape [batch, sector_width].")
            if int(value.shape[0]) != int(batch_size):
                raise ValueError("Packed product sources must have matching batch sizes.")
            if int(value.shape[1]) != int(width):
                raise ValueError(
                    "Packed product source "
                    + str(int(source_index))
                    + " has width "
                    + str(int(value.shape[1]))
                    + ", expected "
                    + str(int(width))
                    + "."
                )
            if value.dtype != dtype or value.device != device:
                raise ValueError("Packed product sources must have matching dtype and device.")
            if value.is_complex():
                raise ValueError(
                    "Packed sparse bilinear groups require real-valued "
                    "numeric storage."
                )
        shared = _shared_contiguous_packed_view(source_values, self.source_widths)
        if shared is not None:
            self.last_source_pack_operation = "zero_copy_shared_packed_carrier_view"
            return shared
        self.last_source_pack_operation = "one_concatenation_of_exact_carrier_blocks"
        return torch.cat(source_values, dim=1).contiguous()

    def evaluate_packed(self, packed_sources):
        global _LAST_SPARSE_BILINEAR_BACKEND

        if packed_sources.ndim != 2:
            raise ValueError("packed_sources must have shape [batch, packed_source_width].")
        if int(packed_sources.shape[1]) != int(sum(self.source_widths)):
            raise ValueError("packed_sources width does not match the compiled source layout.")
        output_plan = self._segment_plan("quadratic_output")
        adjoint_plan = self._segment_plan("quadratic_adjoint")
        cross_plan = self._segment_plan("quadratic_cross")
        use_segmented_quadratic = bool(
            self.reduction_mode != "atomic"
            and
            _can_use_triton(packed_sources, packed_sources)
            and int(output_plan.max_segment_length) <= 4096
            and int(adjoint_plan.max_segment_length) <= 4096
            and int(cross_plan.max_segment_length) <= 4096
        )
        if use_segmented_quadratic:
            table = self._table()
            adjoint_table = self._quadratic_adjoint_table()
            cross_table = self._quadratic_cross_table()
            allow_segment_buckets = bool(
                self.reduction_mode == "segmented"
            )
            if torch.is_grad_enabled() and packed_sources.requires_grad:
                output = _SparseQuadraticSegmentedAutograd.apply(
                    packed_sources,
                    table,
                    output_plan,
                    adjoint_table,
                    adjoint_plan,
                    cross_table,
                    cross_plan,
                    allow_segment_buckets,
                )
                self.last_backend = (
                    "triton_segmented_sparse_quadratic_autograd"
                )
            else:
                output = _sparse_bilinear_segmented_forward_triton(
                    packed_sources,
                    packed_sources,
                    table,
                    output_plan,
                    allow_buckets=allow_segment_buckets,
                )
                self.last_backend = "triton_segmented_sparse_quadratic"
            _LAST_SPARSE_BILINEAR_BACKEND = self.last_backend
            return output
        if (
            self.reduction_mode in {"segmented", "stable"}
            and packed_sources.is_cuda
            and bool(self.strict)
        ):
            raise RuntimeError(
                "Destination-segmented sparse quadratic execution was "
                "required but unavailable."
            )
        output = sparse_bilinear_forward(
            packed_sources,
            packed_sources,
            self._table(),
            prefer_triton=True,
            strict=bool(self.strict),
            indices_certified=True,
        )
        self.last_backend = _LAST_SPARSE_BILINEAR_BACKEND
        return output

    def split_output(self, packed_output):
        if packed_output.ndim != 2 or int(packed_output.shape[1]) != self.output_width:
            raise ValueError("packed_output does not match the compiled output layout.")
        return tuple(
            packed_output[:, int(record["start"]):int(record["stop"])]
            for record in self.output_slices
        )

    def forward_packed(self, packed_sources):
        return self.split_output(self.evaluate_packed(packed_sources))

    def forward(self, source_values):
        return self.forward_packed(self.pack_sources(source_values))

    def report(self):
        return {
            "runtime": "PackedSparseBilinearGroup",
            "path_count": int(len(self.output_slices)),
            "source_count": int(len(self.source_widths)),
            "source_widths": tuple(int(width) for width in self.source_widths),
            "packed_source_width": int(sum(self.source_widths)),
            "output_width": int(self.output_width),
            "term_count": int(self.coefficient.numel()),
            "input_pairs": tuple(tuple(int(value) for value in pair) for pair in self.input_pairs),
            "output_slices": tuple(dict(record) for record in self.output_slices),
            "single_numeric_launch": not bool(
                self.reduction_mode == "segmented"
                and _segment_bucket_padding_report(
                    self._segment_plan("quadratic_output")
                )["auto_eligible"]
            ),
            "destination_segmented_quadratic_available": True,
            "destination_segmented_quadratic_active": bool(
                str(self.last_backend).startswith(
                    "triton_segmented_sparse_quadratic"
                )
            ),
            "segmented_reduction": {
                name: {
                    "segment_count": int(
                        getattr(
                            self,
                            name + "_segment_lengths",
                        ).numel()
                    ),
                    "max_segment_length": int(
                        getattr(
                            self,
                            name + "_max_segment_length",
                        )
                    ),
                    **_segment_bucket_padding_report(
                        self._segment_plan(name)
                    ),
                }
                for name in (
                    "quadratic_output",
                    "quadratic_adjoint",
                    "quadratic_cross",
                )
            },
            "source_pack_operation": str(self.last_source_pack_operation),
            "source_pack_options": (
                "zero_copy_shared_packed_carrier_view",
                "one_concatenation_of_exact_carrier_blocks",
            ),
            "flat_packed_output_available": True,
            "logical_sector_axes": (
                "channel_or_multiplicity",
                "tableau_t",
                "magnetic_M",
            ),
            "last_backend": self.last_backend,
            "strict": bool(self.strict),
            "reduction_mode": str(self.reduction_mode),
            "segment_length_bucketing": (
                "enabled"
                if self.reduction_mode == "segmented"
                else (
                    "disabled_stable_control"
                    if self.reduction_mode == "stable"
                    else "disabled_unpromoted"
                )
            ),
            "source_storage": str(self.source_storage),
            "index_validation": "once_at_packed_group_construction",
        }


class PackedWeightedSparseBilinearGroup(torch.nn.Module):
    """Pack exact products and fuse learned multiplicity maps into one launch.

    A learned map acts only on the multiplicity axis,

    ``y[a, t, m] = sum_r W[a, r] z[r, t, m]``,

    while tableau ``t`` and magnetic ``m`` axes remain complete.  The default
    remains non-expanding.  An explicit capacity-control request may set
    ``allow_output_channel_expansion`` so that ``a`` is wider than ``r``;
    this creates additional copies of the same exact carrier and does not
    change its Young or O(3) action.
    """

    def __init__(
        self,
        tables,
        input_pairs,
        source_widths,
        output_blocks_by_path,
        *,
        output_channel_counts_by_path=None,
        arena_source_offsets=None,
        arena_source_width=None,
        strict=False,
        reduction_mode="auto",
        allow_output_channel_expansion=False,
    ):
        super().__init__()
        tables = tuple(tables)
        input_pairs = tuple(
            (int(pair[0]), int(pair[1]))
            for pair in tuple(input_pairs)
        )
        source_widths = tuple(int(width) for width in tuple(source_widths))
        output_blocks_by_path = tuple(
            tuple(dict(block) for block in blocks)
            for blocks in tuple(output_blocks_by_path)
        )
        if not tables or len(tables) != len(input_pairs):
            raise ValueError("Packed weighted group requires equally many tables and input pairs.")
        if len(output_blocks_by_path) != len(tables):
            raise ValueError("Packed weighted group requires output blocks for every path.")
        if output_channel_counts_by_path is None:
            output_channel_counts_by_path = tuple(
                tuple(int(block["channel_count"]) for block in blocks)
                for blocks in output_blocks_by_path
            )
        else:
            output_channel_counts_by_path = tuple(
                tuple(int(value) for value in counts)
                for counts in tuple(output_channel_counts_by_path)
            )
        if len(output_channel_counts_by_path) != len(tables):
            raise ValueError(
                "Packed weighted group requires output-channel counts for "
                "every path."
            )
        source_offsets = []
        source_offset = 0
        for width in source_widths:
            if int(width) <= 0:
                raise ValueError("Packed weighted source widths must be positive.")
            source_offsets.append(int(source_offset))
            source_offset += int(width)
        direct_arena_indexing_available = bool(
            arena_source_offsets is not None
        )
        if (arena_source_offsets is None) != (arena_source_width is None):
            raise ValueError(
                "Packed weighted arena offsets and width must be supplied together."
            )
        if arena_source_offsets is None:
            arena_source_offsets = tuple(source_offsets)
            arena_source_width = int(source_offset)
        else:
            arena_source_offsets = tuple(
                int(value) for value in arena_source_offsets
            )
            arena_source_width = int(arena_source_width)
            if len(arena_source_offsets) != len(source_widths):
                raise ValueError(
                    "Packed weighted arena offsets must cover every source."
                )
            if arena_source_width <= 0:
                raise ValueError(
                    "Packed weighted arena source width must be positive."
                )
            for offset, width in zip(
                arena_source_offsets,
                source_widths,
            ):
                if offset < 0 or offset + int(width) > arena_source_width:
                    raise ValueError(
                        "Packed weighted arena source slice is out of range."
                    )

        left_indices = []
        right_indices = []
        arena_left_indices = []
        arena_right_indices = []
        output_indices = []
        weight_indices = []
        coefficients = []
        output_slices = []
        weight_blocks = []
        output_offset = 0
        weight_offset = 0
        for path_index, (table, pair, blocks, requested_channels) in enumerate(
            zip(
                tables,
                input_pairs,
                output_blocks_by_path,
                output_channel_counts_by_path,
            )
        ):
            left_source, right_source = pair
            if min(left_source, right_source) < 0:
                raise ValueError("Packed weighted source indices must be nonnegative.")
            if max(left_source, right_source) >= len(source_widths):
                raise ValueError("Packed weighted source index is out of range.")
            covered_width = 0
            compressed_width = 0
            path_weight_blocks = []
            if len(requested_channels) != len(blocks):
                raise ValueError(
                    "Packed weighted output-channel counts must match the "
                    "number of carrier blocks."
                )
            for block_index, (block, output_channel_count) in enumerate(
                zip(blocks, requested_channels)
            ):
                start = int(block["start"])
                stop = int(block["stop"])
                channel_count = int(block["channel_count"])
                component_width = int(block["component_width"])
                output_channel_count = int(output_channel_count)
                if output_channel_count <= 0:
                    raise ValueError(
                        "Packed weighted output-channel count must be positive."
                    )
                if (
                    output_channel_count > channel_count
                    and not bool(allow_output_channel_expansion)
                ):
                    raise ValueError(
                        "Packed weighted output-channel count exceeds the raw "
                        "multiplicity count; explicit channel expansion is "
                        "required."
                    )
                if start != int(covered_width):
                    raise ValueError("Packed weighted output blocks must be contiguous and ordered.")
                if stop - start != channel_count * component_width:
                    raise ValueError("Packed weighted output block dimensions are inconsistent.")
                mask = (
                    (table.output_index >= int(start))
                    & (table.output_index < int(stop))
                )
                selected = torch.nonzero(mask, as_tuple=False).reshape(-1)
                local_output = table.output_index.index_select(0, selected) - int(start)
                raw_channel = torch.div(
                    local_output,
                    int(component_width),
                    rounding_mode="floor",
                )
                component = local_output - raw_channel * int(component_width)
                for output_channel in range(output_channel_count):
                    left_indices.append(
                        table.left_index.index_select(0, selected)
                        + int(source_offsets[left_source])
                    )
                    right_indices.append(
                        table.right_index.index_select(0, selected)
                        + int(source_offsets[right_source])
                    )
                    arena_left_indices.append(
                        table.left_index.index_select(0, selected)
                        + int(arena_source_offsets[left_source])
                    )
                    arena_right_indices.append(
                        table.right_index.index_select(0, selected)
                        + int(arena_source_offsets[right_source])
                    )
                    output_indices.append(
                        int(output_offset)
                        + int(compressed_width)
                        + int(output_channel) * int(component_width)
                        + component
                    )
                    weight_indices.append(
                        int(weight_offset)
                        + int(output_channel) * int(channel_count)
                        + raw_channel
                    )
                    coefficients.append(table.coefficient.index_select(0, selected))
                path_weight_blocks.append(
                    {
                        "path_index": int(path_index),
                        "block_index": int(block_index),
                        "start": int(weight_offset),
                        "stop": int(
                            weight_offset
                            + output_channel_count * channel_count
                        ),
                        "shape": (
                            int(output_channel_count),
                            int(channel_count),
                        ),
                        "raw_channel_count": int(channel_count),
                        "output_channel_count": int(output_channel_count),
                        "output_channel_expansion": bool(
                            output_channel_count > channel_count
                        ),
                        "carrier_component_width": int(component_width),
                        "learned_axis": "channel_or_multiplicity",
                        "preserved_axes": ("tableau_t", "magnetic_M"),
                    }
                )
                weight_blocks.append(path_weight_blocks[-1])
                weight_offset += int(output_channel_count * channel_count)
                covered_width = int(stop)
                compressed_width += int(
                    output_channel_count * component_width
                )
            if int(covered_width) != int(table.output_width):
                raise ValueError("Packed weighted output blocks do not cover the product output.")
            output_slices.append(
                {
                    "path_index": int(path_index),
                    "start": int(output_offset),
                    "stop": int(output_offset + compressed_width),
                    "width": int(compressed_width),
                    "raw_width": int(table.output_width),
                    "left_source_index": int(left_source),
                    "right_source_index": int(right_source),
                    "weight_blocks": tuple(path_weight_blocks),
                }
            )
            output_offset += int(compressed_width)
        self.register_buffer(
            "left_index",
            torch.cat(tuple(left_indices)),
            persistent=False,
        )
        self.register_buffer(
            "right_index",
            torch.cat(tuple(right_indices)),
            persistent=False,
        )
        self.register_buffer(
            "arena_left_index",
            torch.cat(tuple(arena_left_indices)),
            persistent=False,
        )
        self.register_buffer(
            "arena_right_index",
            torch.cat(tuple(arena_right_indices)),
            persistent=False,
        )
        self.register_buffer(
            "arena_compact_gather_index",
            torch.cat(
                tuple(
                    torch.arange(
                        int(offset),
                        int(offset) + int(width),
                        dtype=torch.long,
                    )
                    for offset, width in zip(
                        arena_source_offsets,
                        source_widths,
                    )
                )
            ),
            persistent=False,
        )
        self.register_buffer(
            "output_index",
            torch.cat(tuple(output_indices)),
            persistent=False,
        )
        self.register_buffer(
            "weight_index",
            torch.cat(tuple(weight_indices)),
            persistent=False,
        )
        self.register_buffer(
            "coefficient",
            torch.cat(tuple(coefficients)).to(torch.float64),
            persistent=False,
        )
        segment_plans = {
            "output": _sorted_segment_plan(
                torch.cat(tuple(output_indices)),
                int(output_offset),
            ),
            "left": _sorted_segment_plan(
                torch.cat(tuple(left_indices)),
                int(source_offset),
            ),
            "right": _sorted_segment_plan(
                torch.cat(tuple(right_indices)),
                int(source_offset),
            ),
            "arena_left": _sorted_segment_plan(
                torch.cat(tuple(arena_left_indices)),
                int(arena_source_width),
            ),
            "arena_right": _sorted_segment_plan(
                torch.cat(tuple(arena_right_indices)),
                int(arena_source_width),
            ),
            "weight": _sorted_segment_plan(
                torch.cat(tuple(weight_indices)),
                int(weight_offset),
            ),
        }
        for name, plan in segment_plans.items():
            self.register_buffer(
                name + "_segment_order",
                plan.order,
                persistent=False,
            )
            self.register_buffer(
                name + "_segment_inverse_order",
                plan.inverse_order,
                persistent=False,
            )
            self.register_buffer(
                name + "_segment_lengths",
                plan.lengths,
                persistent=False,
            )
            self.register_buffer(
                name + "_segment_offsets",
                plan.offsets,
                persistent=False,
            )
            self.register_buffer(
                name + "_segment_ids",
                plan.segment_ids,
                persistent=False,
            )
            self.register_buffer(
                name + "_bucket_segment_ids",
                plan.bucket_segment_ids,
                persistent=False,
            )
            setattr(
                self,
                name + "_bucket_ranges",
                tuple(plan.bucket_ranges),
            )
            setattr(
                self,
                name + "_max_segment_length",
                int(plan.max_segment_length),
            )
        mixing_weight = torch.zeros((int(weight_offset),), dtype=torch.float64)
        for block in weight_blocks:
            rows, columns = tuple(int(value) for value in block["shape"])
            view = mixing_weight[int(block["start"]):int(block["stop"])].reshape(rows, columns)
            if rows <= columns:
                view.copy_(torch.eye(rows, columns, dtype=torch.float64))
            else:
                torch.nn.init.xavier_uniform_(view)
        self.mixing_weight = torch.nn.Parameter(mixing_weight)
        self.source_widths = source_widths
        self.source_offsets = tuple(source_offsets)
        self.arena_source_offsets = tuple(arena_source_offsets)
        self.arena_source_width = int(arena_source_width)
        self.direct_arena_indexing_available = bool(
            direct_arena_indexing_available
        )
        self.arena_layout_distinct = bool(
            self.arena_source_offsets != self.source_offsets
            or self.arena_source_width != int(source_offset)
        )
        self.input_pairs = input_pairs
        self.output_slices = tuple(output_slices)
        self.output_channel_counts_by_path = output_channel_counts_by_path
        self.weight_blocks = tuple(weight_blocks)
        self.output_width = int(output_offset)
        self.allow_output_channel_expansion = bool(
            allow_output_channel_expansion
        )
        self.strict = bool(strict)
        reduction_mode = str(reduction_mode).strip().lower()
        if reduction_mode not in {"auto", "atomic", "segmented", "stable"}:
            raise ValueError(
                "Packed weighted reduction_mode must be 'auto', 'atomic', "
                "'segmented', or 'stable'."
            )
        self.reduction_mode = reduction_mode
        self.last_backend = None
        self.last_source_pack_operation = "not_run"
        self.last_source_layout = "not_run"

    def _table(self, source_layout="compact"):
        source_layout = str(source_layout)
        if source_layout not in {"compact", "arena"}:
            raise ValueError("Packed weighted source layout is invalid.")
        return WeightedSparseBilinearTable(
            left_index=(
                self.arena_left_index
                if source_layout == "arena"
                else self.left_index
            ),
            right_index=(
                self.arena_right_index
                if source_layout == "arena"
                else self.right_index
            ),
            output_index=self.output_index,
            weight_index=self.weight_index,
            coefficient=self.coefficient,
            output_width=int(self.output_width),
            weight_count=int(self.mixing_weight.numel()),
        )

    def _segment_plan(self, name):
        return SortedSegmentPlan(
            order=getattr(self, name + "_segment_order"),
            inverse_order=getattr(
                self,
                name + "_segment_inverse_order",
            ),
            lengths=getattr(self, name + "_segment_lengths"),
            offsets=getattr(self, name + "_segment_offsets"),
            segment_ids=getattr(self, name + "_segment_ids"),
            segment_count=int(getattr(self, name + "_segment_lengths").numel()),
            max_segment_length=int(
                getattr(self, name + "_max_segment_length")
            ),
            bucket_ranges=tuple(
                getattr(self, name + "_bucket_ranges")
            ),
            bucket_segment_ids=getattr(
                self,
                name + "_bucket_segment_ids",
            ),
        )

    def pack_sources(self, source_values):
        source_values = tuple(source_values)
        if len(source_values) != len(self.source_widths):
            raise ValueError("Packed weighted source count does not match the compiled layout.")
        batch_size = int(source_values[0].shape[0])
        dtype = source_values[0].dtype
        device = source_values[0].device
        for source_index, (value, width) in enumerate(zip(source_values, self.source_widths)):
            if value.ndim != 2 or int(value.shape[1]) != int(width):
                raise ValueError(
                    "Packed weighted source "
                    + str(int(source_index))
                    + " has an incompatible shape."
                )
            if int(value.shape[0]) != int(batch_size):
                raise ValueError("Packed weighted sources must have matching batch sizes.")
            if value.dtype != dtype or value.device != device:
                raise ValueError("Packed weighted sources must have matching dtype and device.")
            if value.is_complex():
                raise ValueError("Packed weighted groups require declared real-basis sources.")
        shared = _shared_contiguous_packed_view(source_values, self.source_widths)
        if shared is not None:
            self.last_source_pack_operation = "zero_copy_shared_packed_carrier_view"
            return shared
        self.last_source_pack_operation = "one_concatenation_of_exact_carrier_blocks"
        return torch.cat(source_values, dim=1).contiguous()

    def evaluate_packed(self, packed_sources, source_layout="auto"):
        global _LAST_WEIGHTED_SPARSE_BILINEAR_BACKEND

        if packed_sources.ndim != 2:
            raise ValueError("packed_sources must have shape [batch, packed_source_width].")
        compact_width = int(sum(self.source_widths))
        source_layout = str(source_layout).strip().lower()
        if source_layout == "auto":
            compact_match = int(packed_sources.shape[1]) == compact_width
            arena_match = int(packed_sources.shape[1]) == int(
                self.arena_source_width
            )
            if compact_match and arena_match and self.arena_layout_distinct:
                raise ValueError(
                    "Packed weighted source width is ambiguous; choose compact or arena."
                )
            if arena_match and self.arena_layout_distinct:
                source_layout = "arena"
            elif compact_match:
                source_layout = "compact"
            else:
                raise ValueError(
                    "packed_sources width does not match a compiled source layout."
                )
        if source_layout not in {"compact", "arena"}:
            raise ValueError(
                "Packed weighted source_layout must be auto, compact, or arena."
            )
        expected_width = (
            self.arena_source_width
            if source_layout == "arena"
            else compact_width
        )
        if int(packed_sources.shape[1]) != int(expected_width):
            raise ValueError(
                "packed_sources width does not match the selected source layout."
            )
        self.last_source_layout = source_layout
        reduction_mode = self.reduction_mode
        if torch.are_deterministic_algorithms_enabled():
            reduction_mode = "stable"
        if reduction_mode == "auto":
            reduction_mode = "segmented"
        if reduction_mode in {"segmented", "stable"}:
            table = self._table(source_layout)
            output_plan = self._segment_plan("output")
            left_plan = self._segment_plan(
                "arena_left" if source_layout == "arena" else "left"
            )
            right_plan = self._segment_plan(
                "arena_right" if source_layout == "arena" else "right"
            )
            weight_plan = self._segment_plan("weight")
            use_segmented_triton = (
                _can_use_triton(packed_sources, packed_sources)
                and int(output_plan.max_segment_length) <= 4096
            )
            segmented_error = None
            if use_segmented_triton:
                try:
                    if torch.are_deterministic_algorithms_enabled():
                        from ._deterministic_weighted_bilinear import deterministic_weighted_bilinear
                        binding_tensors = (table.left_index, table.right_index, table.output_index,
                            table.weight_index, table.coefficient) + tuple(
                            tensor for plan in (left_plan, right_plan, output_plan, weight_plan)
                            for tensor in (plan.offsets, plan.order))
                        try:
                            binding = (source_layout, expected_width, table.output_width, table.weight_count,
                                tuple((id(tensor), tensor._version) for tensor in binding_tensors))
                        except RuntimeError:
                            # Inference tensors have no version counter: validate
                            # them each time rather than caching a mutable binding.
                            binding = None
                        certified = binding is not None and binding == getattr(self, "_deterministic_binding", None)
                        output = deterministic_weighted_bilinear(packed_sources, packed_sources, self.mixing_weight,
                            table, left_plan, right_plan, output_plan, weight_plan, indices_certified=certified)
                        self._deterministic_binding = binding
                        self.last_backend = "triton_fixed_order_weighted_bilinear_product_rule"
                        return output
                    if torch.is_grad_enabled() and (
                        packed_sources.requires_grad
                        or self.mixing_weight.requires_grad
                    ):
                        if (
                            reduction_mode == "segmented"
                            and source_layout == "arena"
                        ):
                            output = (
                                _WeightedSparseBilinearArenaSegmentedFastAutograd.apply(
                                    packed_sources,
                                    self.mixing_weight,
                                    table,
                                    self._table("compact"),
                                    output_plan,
                                    self.arena_compact_gather_index,
                                )
                            )
                            backend = (
                                "triton_segmented_fast_weighted_sparse_"
                                "bilinear_arena_compact_adjoint_autograd"
                            )
                        else:
                            autograd_kernel = _WeightedSparseBilinearSegmentedAutograd
                            backend = "triton_segmented_weighted_sparse_bilinear_autograd"
                            if reduction_mode == "segmented":
                                autograd_kernel = _WeightedSparseBilinearSegmentedFastAutograd
                                backend = "triton_segmented_fast_weighted_sparse_bilinear_autograd"
                            output = autograd_kernel.apply(
                                packed_sources,
                                packed_sources,
                                self.mixing_weight,
                                table,
                                output_plan,
                                left_plan,
                                right_plan,
                                weight_plan,
                            )
                        self.last_backend = backend
                    else:
                        output = _weighted_sparse_bilinear_segmented_forward_triton(
                            packed_sources,
                            packed_sources,
                            self.mixing_weight,
                            table,
                            output_plan,
                        )
                        self.last_backend = (
                            "triton_segmented_weighted_sparse_bilinear"
                        )
                    return output
                except Exception as exc:
                    segmented_error = exc
                    if bool(self.strict) or torch.are_deterministic_algorithms_enabled() or os.environ.get("YE3T_DEBUG_TRITON") == "1":
                        raise
            if bool(self.strict) and packed_sources.is_cuda:
                raise RuntimeError(
                    "Segmented packed weighted Triton execution was required "
                    "but unavailable."
                )
            if packed_sources.is_cuda:
                reason = (
                    "the compiled segment length exceeds 4096 or Triton is unavailable"
                    if segmented_error is None
                    else str(segmented_error)
                )
                warnings.warn(
                    "YE3T segmented packed CUDA reduction fell back to the "
                    "PyTorch reference path: " + reason,
                    RuntimeWarning,
                    stacklevel=2,
                )
            if reduction_mode == "segmented":
                output = weighted_sparse_bilinear_reference(
                    packed_sources,
                    packed_sources,
                    self.mixing_weight,
                    table,
                )
                self.last_backend = "torch_weighted_sparse_bilinear_reference"
                return output
            output = weighted_sparse_bilinear_stable_reference(
                packed_sources,
                packed_sources,
                self.mixing_weight,
                table,
                output_plan,
                left_plan,
                right_plan,
                weight_plan,
            )
            self.last_backend = "torch_sorted_segment_weighted_sparse_bilinear"
        else:
            output = weighted_sparse_bilinear_forward(
                packed_sources,
                packed_sources,
                self.mixing_weight,
                self._table(source_layout),
                prefer_triton=True,
                strict=bool(self.strict),
            )
            self.last_backend = _LAST_WEIGHTED_SPARSE_BILINEAR_BACKEND
        return output

    def split_output(self, packed_output):
        if packed_output.ndim != 2 or int(packed_output.shape[1]) != self.output_width:
            raise ValueError("packed_output does not match the compiled output layout.")
        return tuple(
            packed_output[:, int(record["start"]):int(record["stop"])]
            for record in self.output_slices
        )

    def forward_packed(self, packed_sources):
        return self.split_output(self.evaluate_packed(packed_sources))

    def forward(self, source_values):
        return self.forward_packed(self.pack_sources(source_values))

    def report(self):
        effective_reduction_mode = self.reduction_mode
        if effective_reduction_mode == "auto":
            effective_reduction_mode = "segmented"
        single_numeric_launch = effective_reduction_mode == "atomic"
        if str(self.last_backend).startswith("triton_segmented_"):
            single_numeric_launch = True
        return {
            "runtime": "PackedWeightedSparseBilinearGroup",
            "path_count": int(len(self.output_slices)),
            "source_count": int(len(self.source_widths)),
            "source_widths": tuple(int(width) for width in self.source_widths),
            "packed_source_width": int(sum(self.source_widths)),
            "arena_source_width": int(self.arena_source_width),
            "arena_source_offsets": tuple(self.arena_source_offsets),
            "direct_arena_indexing_available": bool(
                self.direct_arena_indexing_available
            ),
            "arena_layout_distinct": bool(self.arena_layout_distinct),
            "arena_derivative_strategy": (
                "compact_selected_adjoint_then_single_arena_scatter"
            ),
            "last_source_layout": str(self.last_source_layout),
            "output_width": int(self.output_width),
            "term_count": int(self.coefficient.numel()),
            "base_weight_count": int(self.mixing_weight.numel()),
            "allow_output_channel_expansion": bool(
                self.allow_output_channel_expansion
            ),
            "expanded_weight_block_count": int(
                sum(
                    bool(block["output_channel_expansion"])
                    for block in self.weight_blocks
                )
            ),
            "weight_blocks": tuple(dict(block) for block in self.weight_blocks),
            "input_pairs": tuple(tuple(int(value) for value in pair) for pair in self.input_pairs),
            "output_slices": tuple(dict(record) for record in self.output_slices),
            "single_numeric_launch": bool(single_numeric_launch),
            "source_pack_operation": str(self.last_source_pack_operation),
            "source_pack_options": (
                "zero_copy_shared_packed_carrier_view",
                "one_concatenation_of_exact_carrier_blocks",
            ),
            "flat_packed_output_available": True,
            "fused_operations": (
                "joint_young_angular_analysis",
                "multiplicity_mixing",
                "output_packing",
            ),
            "learned_map_axis": "channel_or_multiplicity",
            "preserved_axes": ("tableau_t", "magnetic_M"),
            "logical_sector_axes": (
                "channel_or_multiplicity",
                "tableau_t",
                "magnetic_M",
            ),
            "last_backend": self.last_backend,
            "strict": bool(self.strict),
            "reduction_mode": str(self.reduction_mode),
            "effective_reduction_mode": str(effective_reduction_mode),
            "promotion_basis": (
                "fp64_value_vjp_hvp_and_interleaved_timing_gate_v1"
                if self.reduction_mode == "auto"
                else "explicit_user_selection"
            ),
            "stable_reduction": {
                name: {
                    "segment_count": int(
                        getattr(self, name + "_segment_lengths").numel()
                    ),
                    "max_segment_length": int(
                        getattr(self, name + "_max_segment_length")
                    ),
                }
                for name in (
                    "output",
                    "left",
                    "right",
                    "arena_left",
                    "arena_right",
                    "weight",
                )
            },
        }


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
    def _sparse_bilinear_forward_kernel(
        left_ptr,
        right_ptr,
        output_ptr,
        left_index_ptr,
        right_index_ptr,
        output_index_ptr,
        coefficient_ptr,
        batch_size,
        left_width,
        right_width,
        output_width,
    ):
        term = tl.program_id(0)
        batch_block = tl.program_id(1)
        batches = batch_block * 128 + tl.arange(0, 128)
        mask = batches < batch_size

        left_index = tl.load(left_index_ptr + term)
        right_index = tl.load(right_index_ptr + term)
        output_index = tl.load(output_index_ptr + term)
        coefficient = tl.load(coefficient_ptr + term)

        left = tl.load(left_ptr + batches * left_width + left_index, mask=mask, other=0.0)
        right = tl.load(right_ptr + batches * right_width + right_index, mask=mask, other=0.0)
        tl.atomic_add(
            output_ptr + batches * output_width + output_index,
            coefficient * left * right,
            mask=mask,
        )

    @triton.jit
    def _sparse_bilinear_backward_kernel(
        grad_output_ptr,
        left_ptr,
        right_ptr,
        grad_left_ptr,
        grad_right_ptr,
        left_index_ptr,
        right_index_ptr,
        output_index_ptr,
        coefficient_ptr,
        batch_size,
        left_width,
        right_width,
        output_width,
    ):
        term = tl.program_id(0)
        batch_block = tl.program_id(1)
        batches = batch_block * 128 + tl.arange(0, 128)
        mask = batches < batch_size

        left_index = tl.load(left_index_ptr + term)
        right_index = tl.load(right_index_ptr + term)
        output_index = tl.load(output_index_ptr + term)
        coefficient = tl.load(coefficient_ptr + term)

        left_offset = batches * left_width + left_index
        right_offset = batches * right_width + right_index
        output_offset = batches * output_width + output_index
        grad_output = tl.load(grad_output_ptr + output_offset, mask=mask, other=0.0)
        left = tl.load(left_ptr + left_offset, mask=mask, other=0.0)
        right = tl.load(right_ptr + right_offset, mask=mask, other=0.0)

        tl.atomic_add(grad_left_ptr + left_offset, coefficient * grad_output * right, mask=mask)
        tl.atomic_add(grad_right_ptr + right_offset, coefficient * grad_output * left, mask=mask)

    @triton.jit
    def _weighted_sparse_bilinear_forward_kernel(
        left_ptr,
        right_ptr,
        weight_ptr,
        output_ptr,
        left_index_ptr,
        right_index_ptr,
        output_index_ptr,
        weight_index_ptr,
        coefficient_ptr,
        batch_size,
        left_width,
        right_width,
        output_width,
    ):
        term = tl.program_id(0)
        batch_block = tl.program_id(1)
        batches = batch_block * 128 + tl.arange(0, 128)
        mask = batches < batch_size
        left_index = tl.load(left_index_ptr + term)
        right_index = tl.load(right_index_ptr + term)
        output_index = tl.load(output_index_ptr + term)
        weight_index = tl.load(weight_index_ptr + term)
        coefficient = tl.load(coefficient_ptr + term)
        weight = tl.load(weight_ptr + weight_index)
        left = tl.load(left_ptr + batches * left_width + left_index, mask=mask, other=0.0)
        right = tl.load(right_ptr + batches * right_width + right_index, mask=mask, other=0.0)
        tl.atomic_add(
            output_ptr + batches * output_width + output_index,
            coefficient * weight * left * right,
            mask=mask,
        )

    def _weighted_sparse_bilinear_segmented_forward_kernel(
        left_ptr,
        right_ptr,
        weight_ptr,
        output_ptr,
        left_index_ptr,
        right_index_ptr,
        weight_index_ptr,
        coefficient_ptr,
        output_order_ptr,
        output_offset_ptr,
        batch_size,
        left_width,
        right_width,
        output_width,
        BLOCK_TERMS,
        BLOCK_BATCH,
    ):
        output_index = tl.program_id(0)
        batch_block = tl.program_id(1)
        batch_offsets = (
            batch_block * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
        )
        segment_start = tl.load(output_offset_ptr + output_index)
        segment_stop = tl.load(output_offset_ptr + output_index + 1)
        sorted_terms = segment_start + tl.arange(0, BLOCK_TERMS)
        term_mask = sorted_terms < segment_stop
        terms = tl.load(
            output_order_ptr + sorted_terms,
            mask=term_mask,
            other=0,
        )
        left_index = tl.load(
            left_index_ptr + terms,
            mask=term_mask,
            other=0,
        )
        right_index = tl.load(
            right_index_ptr + terms,
            mask=term_mask,
            other=0,
        )
        weight_index = tl.load(
            weight_index_ptr + terms,
            mask=term_mask,
            other=0,
        )
        coefficient = tl.load(
            coefficient_ptr + terms,
            mask=term_mask,
            other=0.0,
        )
        weight = tl.load(
            weight_ptr + weight_index,
            mask=term_mask,
            other=0.0,
        )
        value_mask = (
            (batch_offsets[:, None] < batch_size)
            & term_mask[None, :]
        )
        left = tl.load(
            left_ptr
            + batch_offsets[:, None] * left_width
            + left_index[None, :],
            mask=value_mask,
            other=0.0,
        )
        right = tl.load(
            right_ptr
            + batch_offsets[:, None] * right_width
            + right_index[None, :],
            mask=value_mask,
            other=0.0,
        )
        values = (
            left
            * right
            * coefficient[None, :]
            * weight[None, :]
        )
        reduced = tl.sum(values, axis=1)
        tl.store(
            output_ptr
            + batch_offsets * output_width
            + output_index,
            reduced,
            mask=batch_offsets < batch_size,
        )

    _weighted_sparse_bilinear_segmented_forward_kernel.__annotations__ = {
        "BLOCK_TERMS": tl.constexpr,
        "BLOCK_BATCH": tl.constexpr,
    }
    _weighted_sparse_bilinear_segmented_forward_kernel = triton.jit(
        _weighted_sparse_bilinear_segmented_forward_kernel
    )

    def _sparse_bilinear_segmented_forward_kernel(
        left_ptr,
        right_ptr,
        output_ptr,
        segment_index_ptr,
        left_index_ptr,
        right_index_ptr,
        coefficient_ptr,
        output_order_ptr,
        output_offset_ptr,
        batch_size,
        left_width,
        right_width,
        output_width,
        BLOCK_TERMS,
        BLOCK_BATCH,
    ):
        segment_program = tl.program_id(0)
        output_index = tl.load(segment_index_ptr + segment_program)
        batch_block = tl.program_id(1)
        batch_offsets = (
            batch_block * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
        )
        segment_start = tl.load(output_offset_ptr + output_index)
        segment_stop = tl.load(output_offset_ptr + output_index + 1)
        sorted_terms = segment_start + tl.arange(0, BLOCK_TERMS)
        term_mask = sorted_terms < segment_stop
        terms = tl.load(
            output_order_ptr + sorted_terms,
            mask=term_mask,
            other=0,
        )
        left_index = tl.load(
            left_index_ptr + terms,
            mask=term_mask,
            other=0,
        )
        right_index = tl.load(
            right_index_ptr + terms,
            mask=term_mask,
            other=0,
        )
        coefficient = tl.load(
            coefficient_ptr + terms,
            mask=term_mask,
            other=0.0,
        )
        value_mask = (
            (batch_offsets[:, None] < batch_size)
            & term_mask[None, :]
        )
        left = tl.load(
            left_ptr
            + batch_offsets[:, None] * left_width
            + left_index[None, :],
            mask=value_mask,
            other=0.0,
        )
        right = tl.load(
            right_ptr
            + batch_offsets[:, None] * right_width
            + right_index[None, :],
            mask=value_mask,
            other=0.0,
        )
        reduced = tl.sum(
            left * right * coefficient[None, :],
            axis=1,
        )
        tl.store(
            output_ptr
            + batch_offsets * output_width
            + output_index,
            reduced,
            mask=batch_offsets < batch_size,
        )

    _sparse_bilinear_segmented_forward_kernel.__annotations__ = {
        "BLOCK_TERMS": tl.constexpr,
        "BLOCK_BATCH": tl.constexpr,
    }
    _sparse_bilinear_segmented_forward_kernel = triton.jit(
        _sparse_bilinear_segmented_forward_kernel
    )

    def _sparse_quadratic_segmented_adjoint_kernel(
        grad_output_ptr,
        source_ptr,
        grad_source_ptr,
        segment_index_ptr,
        other_index_ptr,
        upstream_output_index_ptr,
        coefficient_ptr,
        derivative_order_ptr,
        derivative_offset_ptr,
        batch_size,
        source_width,
        upstream_output_width,
        BLOCK_TERMS,
        BLOCK_BATCH,
    ):
        segment_program = tl.program_id(0)
        source_index = tl.load(segment_index_ptr + segment_program)
        batch_block = tl.program_id(1)
        batch_offsets = (
            batch_block * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
        )
        segment_start = tl.load(derivative_offset_ptr + source_index)
        segment_stop = tl.load(
            derivative_offset_ptr + source_index + 1
        )
        sorted_terms = segment_start + tl.arange(0, BLOCK_TERMS)
        term_mask = sorted_terms < segment_stop
        terms = tl.load(
            derivative_order_ptr + sorted_terms,
            mask=term_mask,
            other=0,
        )
        other_index = tl.load(
            other_index_ptr + terms,
            mask=term_mask,
            other=0,
        )
        output_index = tl.load(
            upstream_output_index_ptr + terms,
            mask=term_mask,
            other=0,
        )
        coefficient = tl.load(
            coefficient_ptr + terms,
            mask=term_mask,
            other=0.0,
        )
        value_mask = (
            (batch_offsets[:, None] < batch_size)
            & term_mask[None, :]
        )
        upstream = tl.load(
            grad_output_ptr
            + batch_offsets[:, None] * upstream_output_width
            + output_index[None, :],
            mask=value_mask,
            other=0.0,
        )
        other = tl.load(
            source_ptr
            + batch_offsets[:, None] * source_width
            + other_index[None, :],
            mask=value_mask,
            other=0.0,
        )
        reduced = tl.sum(
            upstream * other * coefficient[None, :],
            axis=1,
        )
        tl.store(
            grad_source_ptr
            + batch_offsets * source_width
            + source_index,
            reduced,
            mask=batch_offsets < batch_size,
        )

    _sparse_quadratic_segmented_adjoint_kernel.__annotations__ = {
        "BLOCK_TERMS": tl.constexpr,
        "BLOCK_BATCH": tl.constexpr,
    }
    _sparse_quadratic_segmented_adjoint_kernel = triton.jit(
        _sparse_quadratic_segmented_adjoint_kernel
    )

    @triton.jit
    def _weighted_sparse_bilinear_backward_kernel(
        grad_output_ptr,
        left_ptr,
        right_ptr,
        weight_ptr,
        grad_left_ptr,
        grad_right_ptr,
        grad_weight_ptr,
        left_index_ptr,
        right_index_ptr,
        output_index_ptr,
        weight_index_ptr,
        coefficient_ptr,
        batch_size,
        left_width,
        right_width,
        output_width,
    ):
        term = tl.program_id(0)
        batch_block = tl.program_id(1)
        batches = batch_block * 128 + tl.arange(0, 128)
        mask = batches < batch_size
        left_index = tl.load(left_index_ptr + term)
        right_index = tl.load(right_index_ptr + term)
        output_index = tl.load(output_index_ptr + term)
        weight_index = tl.load(weight_index_ptr + term)
        coefficient = tl.load(coefficient_ptr + term)
        weight = tl.load(weight_ptr + weight_index)
        left_offset = batches * left_width + left_index
        right_offset = batches * right_width + right_index
        output_offset = batches * output_width + output_index
        grad_output = tl.load(grad_output_ptr + output_offset, mask=mask, other=0.0)
        left = tl.load(left_ptr + left_offset, mask=mask, other=0.0)
        right = tl.load(right_ptr + right_offset, mask=mask, other=0.0)
        weighted_coefficient = coefficient * weight
        tl.atomic_add(
            grad_left_ptr + left_offset,
            weighted_coefficient * grad_output * right,
            mask=mask,
        )
        tl.atomic_add(
            grad_right_ptr + right_offset,
            weighted_coefficient * grad_output * left,
            mask=mask,
        )
        grad_weight = tl.sum(coefficient * grad_output * left * right, axis=0)
        tl.atomic_add(grad_weight_ptr + weight_index, grad_weight)

    @triton.jit
    def _weighted_sparse_linear_forward_kernel(
        source_ptr,
        weight_ptr,
        output_ptr,
        input_index_ptr,
        output_index_ptr,
        weight_index_ptr,
        coefficient_ptr,
        batch_size,
        source_width,
        output_width,
    ):
        term = tl.program_id(0)
        batch_block = tl.program_id(1)
        batches = batch_block * 128 + tl.arange(0, 128)
        mask = batches < batch_size
        input_index = tl.load(input_index_ptr + term)
        output_index = tl.load(output_index_ptr + term)
        weight_index = tl.load(weight_index_ptr + term)
        coefficient = tl.load(coefficient_ptr + term)
        weight = tl.load(weight_ptr + weight_index)
        source = tl.load(
            source_ptr + batches * source_width + input_index,
            mask=mask,
            other=0.0,
        )
        tl.atomic_add(
            output_ptr + batches * output_width + output_index,
            coefficient * weight * source,
            mask=mask,
        )

    @triton.jit
    def _weighted_sparse_linear_backward_kernel(
        grad_output_ptr,
        source_ptr,
        weight_ptr,
        grad_source_ptr,
        grad_weight_ptr,
        input_index_ptr,
        output_index_ptr,
        weight_index_ptr,
        coefficient_ptr,
        batch_size,
        source_width,
        output_width,
    ):
        term = tl.program_id(0)
        batch_block = tl.program_id(1)
        batches = batch_block * 128 + tl.arange(0, 128)
        mask = batches < batch_size
        input_index = tl.load(input_index_ptr + term)
        output_index = tl.load(output_index_ptr + term)
        weight_index = tl.load(weight_index_ptr + term)
        coefficient = tl.load(coefficient_ptr + term)
        weight = tl.load(weight_ptr + weight_index)
        source_offset = batches * source_width + input_index
        output_offset = batches * output_width + output_index
        grad_output = tl.load(
            grad_output_ptr + output_offset,
            mask=mask,
            other=0.0,
        )
        source = tl.load(
            source_ptr + source_offset,
            mask=mask,
            other=0.0,
        )
        tl.atomic_add(
            grad_source_ptr + source_offset,
            coefficient * weight * grad_output,
            mask=mask,
        )
        grad_weight = tl.sum(coefficient * grad_output * source, axis=0)
        tl.atomic_add(grad_weight_ptr + weight_index, grad_weight)


def _validate(left, right, table, *, indices_certified=False):
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("Packed sparse bilinear inputs must have shape [batch, width].")
    if int(left.shape[0]) != int(right.shape[0]):
        raise ValueError("Packed sparse bilinear inputs must have the same batch size.")
    if left.dtype != right.dtype or left.device != right.device:
        raise ValueError("Packed sparse bilinear inputs must have matching dtype and device.")
    if not bool(indices_certified):
        _validate_sparse_bilinear_table_indices(
            table,
            int(left.shape[1]),
            int(right.shape[1]),
        )


class _SortedSegmentSumAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, lengths, segment_ids):
        ctx.save_for_backward(segment_ids, lengths)
        return torch.segment_reduce(
            values,
            "sum",
            lengths=lengths,
            axis=0,
        )

    @staticmethod
    def backward(ctx, grad_output):
        segment_ids, lengths = ctx.saved_tensors
        return _SortedSegmentGatherAutograd.apply(
            grad_output,
            segment_ids,
            lengths,
        ), None, None


class _SortedSegmentGatherAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, segment_ids, lengths):
        ctx.save_for_backward(segment_ids, lengths)
        return values.index_select(0, segment_ids)

    @staticmethod
    def backward(ctx, grad_output):
        segment_ids, lengths = ctx.saved_tensors
        return _SortedSegmentSumAutograd.apply(
            grad_output,
            lengths,
            segment_ids,
        ), None, None


def _sorted_segment_sum(values, plan):
    """Reduce batched sparse terms in one fixed compiler-time order."""

    if values.ndim != 2:
        raise ValueError("Sorted segment values must have shape [batch, terms].")
    if int(values.shape[1]) != int(plan.order.numel()):
        raise ValueError("Sorted segment plan term count does not match values.")
    plan = plan.to(device=values.device)
    ordered = values.index_select(1, plan.order)
    return _SortedSegmentSumAutograd.apply(
        ordered.transpose(0, 1),
        plan.lengths,
        plan.segment_ids,
    ).transpose(0, 1)


def _sorted_segment_gather(values, plan):
    """Gather repeated segment values with the fixed reduction as its adjoint."""

    if values.ndim != 2:
        raise ValueError("Sorted segment source must have shape [batch, segments].")
    if int(values.shape[1]) != int(plan.segment_count):
        raise ValueError("Sorted segment source width does not match the plan.")
    plan = plan.to(device=values.device)
    sorted_values = _SortedSegmentGatherAutograd.apply(
        values.transpose(0, 1),
        plan.segment_ids,
        plan.lengths,
    )
    return sorted_values.index_select(
        0,
        plan.inverse_order,
    ).transpose(0, 1)


def sparse_bilinear_reference(left, right, table):
    """Differentiable PyTorch reference for one packed bilinear table."""

    _validate(left, right, table)
    table = table.to(device=left.device, dtype=left.dtype)
    output = left.new_zeros((int(left.shape[0]), int(table.output_width)))
    if int(table.term_count) == 0:
        return output
    terms = (
        left.index_select(1, table.left_index)
        * right.index_select(1, table.right_index)
        * table.coefficient.reshape(1, -1)
    )
    indices = table.output_index.reshape(1, -1).expand(int(left.shape[0]), -1)
    return output.scatter_add(1, indices, terms)


def sparse_bilinear_backward_reference(
    grad_output,
    left,
    right,
    table,
    *,
    indices_certified=False,
):
    """Differentiable PyTorch adjoint for one packed bilinear table."""

    _validate(
        left,
        right,
        table,
        indices_certified=bool(indices_certified),
    )
    table = table.to(device=left.device, dtype=left.dtype)
    grad_left = torch.zeros_like(left)
    grad_right = torch.zeros_like(right)
    if int(table.term_count) == 0:
        return grad_left, grad_right
    selected_grad = grad_output.index_select(1, table.output_index)
    coefficients = table.coefficient.reshape(1, -1)
    left_terms = coefficients * selected_grad * right.index_select(1, table.right_index)
    right_terms = coefficients * selected_grad * left.index_select(1, table.left_index)
    left_indices = table.left_index.reshape(1, -1).expand(int(left.shape[0]), -1)
    right_indices = table.right_index.reshape(1, -1).expand(int(right.shape[0]), -1)
    return (
        grad_left.scatter_add(1, left_indices, left_terms),
        grad_right.scatter_add(1, right_indices, right_terms),
    )


def _validate_weighted(left, right, weight, table):
    _validate(
        left,
        right,
        SparseBilinearTable(
            left_index=table.left_index,
            right_index=table.right_index,
            output_index=table.output_index,
            coefficient=table.coefficient,
            output_width=int(table.output_width),
        ),
    )
    if weight.ndim != 1 or int(weight.numel()) != int(table.weight_count):
        raise ValueError("Packed weighted mixing_weight has an incompatible shape.")
    if weight.dtype != left.dtype or weight.device != left.device:
        raise ValueError("Packed weighted mixing_weight must match input dtype and device.")
    if int(table.term_count):
        if int(table.weight_index.min()) < 0 or int(table.weight_index.max()) >= int(table.weight_count):
            raise ValueError("Packed weighted table weight_index is out of range.")


def weighted_sparse_bilinear_reference(left, right, weight, table):
    """Differentiable PyTorch reference for fused analysis and mixing."""

    _validate_weighted(left, right, weight, table)
    table = table.to(device=left.device, dtype=left.dtype)
    output = left.new_zeros((int(left.shape[0]), int(table.output_width)))
    if int(table.term_count) == 0:
        return output
    terms = (
        left.index_select(1, table.left_index)
        * right.index_select(1, table.right_index)
        * weight.index_select(0, table.weight_index).reshape(1, -1)
        * table.coefficient.reshape(1, -1)
    )
    indices = table.output_index.reshape(1, -1).expand(int(left.shape[0]), -1)
    return output.scatter_add(1, indices, terms)


def weighted_sparse_bilinear_stable_reference(
    left,
    right,
    weight,
    table,
    output_plan,
    left_plan,
    right_plan,
    weight_plan,
):
    """Evaluate a weighted sparse table with one fixed reduction order."""

    _validate_weighted(left, right, weight, table)
    table = table.to(device=left.device, dtype=left.dtype)
    if int(table.term_count) == 0:
        return left.new_zeros(
            (int(left.shape[0]), int(table.output_width))
        )
    terms = (
        _sorted_segment_gather(left, left_plan)
        * _sorted_segment_gather(right, right_plan)
        * _sorted_segment_gather(weight.reshape(1, -1), weight_plan)
        * table.coefficient.reshape(1, -1)
    )
    return _sorted_segment_sum(terms, output_plan)


def weighted_sparse_bilinear_backward_reference(grad_output, left, right, weight, table):
    """Differentiable adjoint for fused analysis and multiplicity mixing."""

    _validate_weighted(left, right, weight, table)
    table = table.to(device=left.device, dtype=left.dtype)
    selected_grad = grad_output.index_select(1, table.output_index)
    coefficients = table.coefficient.reshape(1, -1)
    selected_weight = weight.index_select(0, table.weight_index).reshape(1, -1)
    left_terms = (
        coefficients
        * selected_weight
        * selected_grad
        * right.index_select(1, table.right_index)
    )
    right_terms = (
        coefficients
        * selected_weight
        * selected_grad
        * left.index_select(1, table.left_index)
    )
    left_indices = table.left_index.reshape(1, -1).expand(int(left.shape[0]), -1)
    right_indices = table.right_index.reshape(1, -1).expand(int(right.shape[0]), -1)
    grad_left = torch.zeros_like(left).scatter_add(1, left_indices, left_terms)
    grad_right = torch.zeros_like(right).scatter_add(1, right_indices, right_terms)
    weight_terms = (
        coefficients
        * selected_grad
        * left.index_select(1, table.left_index)
        * right.index_select(1, table.right_index)
    ).sum(dim=0)
    grad_weight = torch.zeros_like(weight).scatter_add(
        0,
        table.weight_index,
        weight_terms,
    )
    return grad_left, grad_right, grad_weight


def weighted_sparse_bilinear_backward_stable_reference(
    grad_output,
    left,
    right,
    weight,
    table,
    output_plan,
    left_plan,
    right_plan,
    weight_plan,
):
    """Apply the weighted adjoint with fixed reductions on every output."""

    _validate_weighted(left, right, weight, table)
    table = table.to(device=left.device, dtype=left.dtype)
    if int(table.term_count) == 0:
        return torch.zeros_like(left), torch.zeros_like(right), torch.zeros_like(weight)
    selected_grad = _sorted_segment_gather(grad_output, output_plan)
    coefficients = table.coefficient.reshape(1, -1)
    selected_weight = _sorted_segment_gather(
        weight.reshape(1, -1),
        weight_plan,
    )
    left_terms = (
        coefficients
        * selected_weight
        * selected_grad
        * _sorted_segment_gather(right, right_plan)
    )
    right_terms = (
        coefficients
        * selected_weight
        * selected_grad
        * _sorted_segment_gather(left, left_plan)
    )
    weight_terms = (
        coefficients
        * selected_grad
        * _sorted_segment_gather(left, left_plan)
        * _sorted_segment_gather(right, right_plan)
    )
    grad_left = _sorted_segment_sum(left_terms, left_plan)
    grad_right = _sorted_segment_sum(right_terms, right_plan)
    grad_weight = _sorted_segment_sum(
        weight_terms,
        weight_plan,
    ).sum(dim=0)
    return grad_left, grad_right, grad_weight


def _validate_weighted_linear(source, weight, table):
    if source.ndim != 2:
        raise ValueError(
            "Packed weighted linear source must have shape [batch, width]."
        )
    if weight.ndim != 1 or int(weight.numel()) != int(table.weight_count):
        raise ValueError("Packed weighted linear weight has an incompatible shape.")
    if weight.dtype != source.dtype or weight.device != source.device:
        raise ValueError(
            "Packed weighted linear weight must match source dtype and device."
        )
    lengths = {
        int(table.input_index.numel()),
        int(table.output_index.numel()),
        int(table.weight_index.numel()),
        int(table.coefficient.numel()),
    }
    if len(lengths) != 1:
        raise ValueError(
            "Packed weighted linear table arrays must have the same length."
        )
    if int(table.output_width) < 1:
        raise ValueError(
            "Packed weighted linear output_width must be positive."
        )
    if int(table.term_count) == 0:
        return
    if (
        int(table.input_index.min()) < 0
        or int(table.input_index.max()) >= int(source.shape[1])
    ):
        raise ValueError("Packed weighted linear input_index is out of range.")
    if (
        int(table.output_index.min()) < 0
        or int(table.output_index.max()) >= int(table.output_width)
    ):
        raise ValueError("Packed weighted linear output_index is out of range.")
    if (
        int(table.weight_index.min()) < 0
        or int(table.weight_index.max()) >= int(table.weight_count)
    ):
        raise ValueError("Packed weighted linear weight_index is out of range.")


def weighted_sparse_linear_reference(source, weight, table):
    """Differentiable reference for fused source analysis and channel mixing."""

    _validate_weighted_linear(source, weight, table)
    table = table.to(device=source.device, dtype=source.dtype)
    output = source.new_zeros(
        (int(source.shape[0]), int(table.output_width))
    )
    if int(table.term_count) == 0:
        return output
    terms = (
        source.index_select(1, table.input_index)
        * weight.index_select(0, table.weight_index).reshape(1, -1)
        * table.coefficient.reshape(1, -1)
    )
    indices = table.output_index.reshape(1, -1).expand(
        int(source.shape[0]),
        -1,
    )
    return output.scatter_add(1, indices, terms)


def weighted_sparse_linear_backward_reference(
    grad_output,
    source,
    weight,
    table,
):
    """Differentiable adjoint for fused source analysis and channel mixing."""

    _validate_weighted_linear(source, weight, table)
    table = table.to(device=source.device, dtype=source.dtype)
    selected_grad = grad_output.index_select(1, table.output_index)
    coefficients = table.coefficient.reshape(1, -1)
    selected_weight = weight.index_select(
        0,
        table.weight_index,
    ).reshape(1, -1)
    source_terms = coefficients * selected_weight * selected_grad
    source_indices = table.input_index.reshape(1, -1).expand(
        int(source.shape[0]),
        -1,
    )
    grad_source = torch.zeros_like(source).scatter_add(
        1,
        source_indices,
        source_terms,
    )
    weight_terms = (
        coefficients
        * selected_grad
        * source.index_select(1, table.input_index)
    ).sum(dim=0)
    grad_weight = torch.zeros_like(weight).scatter_add(
        0,
        table.weight_index,
        weight_terms,
    )
    return grad_source, grad_weight


def _can_use_triton(left, right):
    return bool(
        _triton_enabled()
        and left.is_cuda
        and right.is_cuda
        and left.is_contiguous()
        and right.is_contiguous()
        and left.dtype in (torch.float32, torch.float64)
        and right.dtype == left.dtype
    )


def _sparse_bilinear_forward_triton(left, right, table):
    table = table.to(device=left.device, dtype=left.dtype)
    output = left.new_zeros((int(left.shape[0]), int(table.output_width)))
    if int(table.term_count) == 0:
        return output
    grid = (int(table.term_count), triton.cdiv(int(left.shape[0]), 128))
    _sparse_bilinear_forward_kernel[grid](
        left,
        right,
        output,
        table.left_index,
        table.right_index,
        table.output_index,
        table.coefficient,
        batch_size=int(left.shape[0]),
        left_width=int(left.shape[1]),
        right_width=int(right.shape[1]),
        output_width=int(table.output_width),
    )
    return output


def _sparse_bilinear_backward_triton(grad_output, left, right, table):
    table = table.to(device=left.device, dtype=left.dtype)
    grad_left = torch.zeros_like(left)
    grad_right = torch.zeros_like(right)
    if int(table.term_count) == 0:
        return grad_left, grad_right
    grid = (int(table.term_count), triton.cdiv(int(left.shape[0]), 128))
    _sparse_bilinear_backward_kernel[grid](
        grad_output.contiguous(),
        left,
        right,
        grad_left,
        grad_right,
        table.left_index,
        table.right_index,
        table.output_index,
        table.coefficient,
        batch_size=int(left.shape[0]),
        left_width=int(left.shape[1]),
        right_width=int(right.shape[1]),
        output_width=int(table.output_width),
    )
    return grad_left, grad_right


def _weighted_sparse_bilinear_forward_triton(left, right, weight, table):
    table = table.to(device=left.device, dtype=left.dtype)
    output = left.new_zeros((int(left.shape[0]), int(table.output_width)))
    if int(table.term_count) == 0:
        return output
    grid = (int(table.term_count), triton.cdiv(int(left.shape[0]), 128))
    _weighted_sparse_bilinear_forward_kernel[grid](
        left,
        right,
        weight,
        output,
        table.left_index,
        table.right_index,
        table.output_index,
        table.weight_index,
        table.coefficient,
        batch_size=int(left.shape[0]),
        left_width=int(left.shape[1]),
        right_width=int(right.shape[1]),
        output_width=int(table.output_width),
    )
    return output


def _weighted_sparse_bilinear_segmented_forward_triton(
    left,
    right,
    weight,
    table,
    output_plan,
):
    table = table.to(device=left.device, dtype=left.dtype)
    output_plan = output_plan.to(device=left.device)
    output = left.new_empty(
        (int(left.shape[0]), int(table.output_width))
    )
    if int(table.term_count) == 0:
        return output.zero_()
    block_terms = triton.next_power_of_2(
        max(1, int(output_plan.max_segment_length))
    )
    if int(block_terms) > 4096:
        raise RuntimeError(
            "Stable segmented weighted reduction exceeds the supported "
            "4096-term output segment."
        )
    if int(block_terms) <= 256:
        block_batch = 32
    elif int(block_terms) <= 1024:
        block_batch = 16
    else:
        block_batch = 8
    grid = (
        int(table.output_width),
        triton.cdiv(int(left.shape[0]), int(block_batch)),
    )
    _weighted_sparse_bilinear_segmented_forward_kernel[grid](
        left,
        right,
        weight,
        output,
        table.left_index,
        table.right_index,
        table.weight_index,
        table.coefficient,
        output_plan.order,
        output_plan.offsets,
        batch_size=int(left.shape[0]),
        left_width=int(left.shape[1]),
        right_width=int(right.shape[1]),
        output_width=int(table.output_width),
        BLOCK_TERMS=int(block_terms),
        BLOCK_BATCH=int(block_batch),
    )
    return output


def _segment_bucket_padding_report(plan):
    global_block_terms = 1 << (
        max(1, int(plan.max_segment_length)) - 1
    ).bit_length()
    ranges = tuple(plan.bucket_ranges)
    bucket_work = int(
        sum(
            int(block_terms) * int(stop - start)
            for block_terms, start, stop in ranges
        )
    )
    global_work = int(global_block_terms) * int(plan.segment_count)
    eligible = bool(
        len(ranges) > 1
        and int(plan.segment_count) >= 1024
        and int(bucket_work) * 5 <= int(global_work) * 4
    )
    return {
        "bucket_count": int(len(ranges)),
        "bucket_ranges": tuple(
            tuple(int(value) for value in record)
            for record in ranges
        ),
        "global_block_terms": int(global_block_terms),
        "global_padded_term_count": int(global_work),
        "bucketed_padded_term_count": int(bucket_work),
        "predicted_padding_reduction": (
            float(global_work) / float(bucket_work)
            if int(bucket_work)
            else 1.0
        ),
        "auto_eligible": bool(eligible),
    }


def _segment_launch_ranges(plan, allow_buckets):
    report = _segment_bucket_padding_report(plan)
    ranges = tuple(plan.bucket_ranges)
    use_buckets = bool(allow_buckets and report["auto_eligible"])
    if use_buckets:
        return ranges, True
    return (
        (
            int(report["global_block_terms"]),
            0,
            int(plan.segment_count),
        ),
    ), False


def _sparse_bilinear_segmented_forward_triton(
    left,
    right,
    table,
    output_plan,
    allow_buckets=True,
):
    table = table.to(device=left.device, dtype=left.dtype)
    output_plan = output_plan.to(device=left.device)
    output = left.new_empty(
        (int(left.shape[0]), int(table.output_width))
    )
    if int(table.term_count) == 0:
        return output.zero_()
    if int(output_plan.max_segment_length) > 4096:
        raise RuntimeError(
            "Segmented sparse reduction exceeds the supported "
            "4096-term output segment."
        )
    launch_ranges, _used_buckets = _segment_launch_ranges(
        output_plan,
        bool(allow_buckets),
    )
    for block_terms, segment_start, segment_stop in launch_ranges:
        if int(block_terms) <= 256:
            block_batch = 32
        elif int(block_terms) <= 1024:
            block_batch = 16
        else:
            block_batch = 8
        segment_indices = output_plan.bucket_segment_ids[
            int(segment_start):int(segment_stop)
        ]
        grid = (
            int(segment_stop - segment_start),
            triton.cdiv(int(left.shape[0]), int(block_batch)),
        )
        _sparse_bilinear_segmented_forward_kernel[grid](
            left,
            right,
            output,
            segment_indices,
            table.left_index,
            table.right_index,
            table.coefficient,
            output_plan.order,
            output_plan.offsets,
            batch_size=int(left.shape[0]),
            left_width=int(left.shape[1]),
            right_width=int(right.shape[1]),
            output_width=int(table.output_width),
            BLOCK_TERMS=int(block_terms),
            BLOCK_BATCH=int(block_batch),
        )
    return output


def _sparse_quadratic_segmented_adjoint_triton(
    grad_output,
    source,
    table,
    derivative_plan,
    allow_buckets=True,
):
    table = table.to(device=source.device, dtype=source.dtype)
    derivative_plan = derivative_plan.to(device=source.device)
    grad_source = torch.empty_like(source)
    if int(table.term_count) == 0:
        return grad_source.zero_()
    if int(derivative_plan.max_segment_length) > 4096:
        raise RuntimeError(
            "Segmented sparse adjoint exceeds the supported "
            "4096-term source segment."
        )
    launch_ranges, _used_buckets = _segment_launch_ranges(
        derivative_plan,
        bool(allow_buckets),
    )
    contiguous_grad_output = grad_output.contiguous()
    for block_terms, segment_start, segment_stop in launch_ranges:
        if int(block_terms) <= 256:
            block_batch = 32
        elif int(block_terms) <= 1024:
            block_batch = 16
        else:
            block_batch = 8
        segment_indices = derivative_plan.bucket_segment_ids[
            int(segment_start):int(segment_stop)
        ]
        grid = (
            int(segment_stop - segment_start),
            triton.cdiv(int(source.shape[0]), int(block_batch)),
        )
        _sparse_quadratic_segmented_adjoint_kernel[grid](
            contiguous_grad_output,
            source,
            grad_source,
            segment_indices,
            table.other_index,
            table.output_index,
            table.coefficient,
            derivative_plan.order,
            derivative_plan.offsets,
            batch_size=int(source.shape[0]),
            source_width=int(table.source_width),
            upstream_output_width=int(table.output_width),
            BLOCK_TERMS=int(block_terms),
            BLOCK_BATCH=int(block_batch),
        )
    return grad_source


def _weighted_sparse_bilinear_backward_triton(grad_output, left, right, weight, table):
    table = table.to(device=left.device, dtype=left.dtype)
    grad_left = torch.zeros_like(left)
    grad_right = torch.zeros_like(right)
    grad_weight = torch.zeros_like(weight)
    if int(table.term_count) == 0:
        return grad_left, grad_right, grad_weight
    grid = (int(table.term_count), triton.cdiv(int(left.shape[0]), 128))
    _weighted_sparse_bilinear_backward_kernel[grid](
        grad_output.contiguous(),
        left,
        right,
        weight,
        grad_left,
        grad_right,
        grad_weight,
        table.left_index,
        table.right_index,
        table.output_index,
        table.weight_index,
        table.coefficient,
        batch_size=int(left.shape[0]),
        left_width=int(left.shape[1]),
        right_width=int(right.shape[1]),
        output_width=int(table.output_width),
    )
    return grad_left, grad_right, grad_weight


def _weighted_sparse_linear_forward_triton(source, weight, table):
    table = table.to(device=source.device, dtype=source.dtype)
    output = source.new_zeros(
        (int(source.shape[0]), int(table.output_width))
    )
    if int(table.term_count) == 0:
        return output
    grid = (int(table.term_count), triton.cdiv(int(source.shape[0]), 128))
    _weighted_sparse_linear_forward_kernel[grid](
        source,
        weight,
        output,
        table.input_index,
        table.output_index,
        table.weight_index,
        table.coefficient,
        batch_size=int(source.shape[0]),
        source_width=int(source.shape[1]),
        output_width=int(table.output_width),
    )
    return output


def _weighted_sparse_linear_backward_triton(
    grad_output,
    source,
    weight,
    table,
):
    table = table.to(device=source.device, dtype=source.dtype)
    grad_source = torch.zeros_like(source)
    grad_weight = torch.zeros_like(weight)
    if int(table.term_count) == 0:
        return grad_source, grad_weight
    grid = (int(table.term_count), triton.cdiv(int(source.shape[0]), 128))
    _weighted_sparse_linear_backward_kernel[grid](
        grad_output.contiguous(),
        source,
        weight,
        grad_source,
        grad_weight,
        table.input_index,
        table.output_index,
        table.weight_index,
        table.coefficient,
        batch_size=int(source.shape[0]),
        source_width=int(source.shape[1]),
        output_width=int(table.output_width),
    )
    return grad_source, grad_weight


class _SparseBilinearTritonAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        left,
        right,
        left_index,
        right_index,
        output_index,
        coefficient,
        output_width,
    ):
        ctx.save_for_backward(
            left,
            right,
            left_index,
            right_index,
            output_index,
            coefficient,
        )
        ctx.output_width = int(output_width)
        table = SparseBilinearTable(
            left_index=left_index,
            right_index=right_index,
            output_index=output_index,
            coefficient=coefficient,
            output_width=int(output_width),
        )
        return _sparse_bilinear_forward_triton(left, right, table)

    @staticmethod
    def backward(ctx, grad_output):
        left, right, left_index, right_index, output_index, coefficient = ctx.saved_tensors
        if torch.is_grad_enabled():
            grad_left, grad_right = _SparseBilinearBackwardTritonAutograd.apply(
                grad_output,
                left,
                right,
                left_index,
                right_index,
                output_index,
                coefficient,
                int(ctx.output_width),
            )
        else:
            table = SparseBilinearTable(
                left_index=left_index,
                right_index=right_index,
                output_index=output_index,
                coefficient=coefficient,
                output_width=int(ctx.output_width),
            )
            grad_left, grad_right = _sparse_bilinear_backward_triton(
                grad_output,
                left,
                right,
                table,
            )
        return grad_left, grad_right, None, None, None, None, None


class _SparseBilinearBackwardTritonAutograd(torch.autograd.Function):
    """Fused differentiable adjoint for force-training double backward."""

    @staticmethod
    def forward(
        ctx,
        grad_output,
        left,
        right,
        left_index,
        right_index,
        output_index,
        coefficient,
        output_width,
    ):
        ctx.save_for_backward(
            grad_output,
            left,
            right,
            left_index,
            right_index,
            output_index,
            coefficient,
        )
        ctx.output_width = int(output_width)
        table = SparseBilinearTable(
            left_index=left_index,
            right_index=right_index,
            output_index=output_index,
            coefficient=coefficient,
            output_width=int(output_width),
        )
        return _sparse_bilinear_backward_triton(
            grad_output.contiguous(),
            left,
            right,
            table,
        )

    @staticmethod
    def backward(ctx, grad_grad_left, grad_grad_right):
        (
            grad_output,
            left,
            right,
            left_index,
            right_index,
            output_index,
            coefficient,
        ) = ctx.saved_tensors
        if grad_grad_left is None:
            grad_grad_left = torch.zeros_like(left)
        else:
            grad_grad_left = grad_grad_left.contiguous()
        if grad_grad_right is None:
            grad_grad_right = torch.zeros_like(right)
        else:
            grad_grad_right = grad_grad_right.contiguous()
        table = SparseBilinearTable(
            left_index=left_index,
            right_index=right_index,
            output_index=output_index,
            coefficient=coefficient,
            output_width=int(ctx.output_width),
        )
        grad_grad_output = _sparse_bilinear_forward_triton(
            grad_grad_left,
            right,
            table,
        )
        grad_grad_output = grad_grad_output + _sparse_bilinear_forward_triton(
            left,
            grad_grad_right,
            table,
        )
        grad_left, grad_right = _sparse_bilinear_backward_triton(
            grad_output.contiguous(),
            grad_grad_left,
            grad_grad_right,
            table,
        )
        return (
            grad_grad_output,
            grad_left,
            grad_right,
            None,
            None,
            None,
            None,
            None,
        )


class _SparseQuadraticSegmentedAutograd(torch.autograd.Function):
    """Destination-segmented packed quadratic form and analytic adjoints."""

    @staticmethod
    def forward(
        ctx,
        source,
        table,
        output_plan,
        adjoint_table,
        adjoint_plan,
        cross_table,
        cross_plan,
        allow_segment_buckets,
    ):
        ctx.save_for_backward(source)
        ctx.adjoint_table = adjoint_table
        ctx.adjoint_plan = adjoint_plan
        ctx.cross_table = cross_table
        ctx.cross_plan = cross_plan
        ctx.allow_segment_buckets = bool(allow_segment_buckets)
        return _sparse_bilinear_segmented_forward_triton(
            source,
            source,
            table,
            output_plan,
            allow_buckets=ctx.allow_segment_buckets,
        )

    @staticmethod
    def backward(ctx, grad_output):
        (source,) = ctx.saved_tensors
        if torch.is_grad_enabled():
            grad_source = _SparseQuadraticSegmentedBackwardAutograd.apply(
                grad_output.contiguous(),
                source,
                ctx.adjoint_table,
                ctx.adjoint_plan,
                ctx.cross_table,
                ctx.cross_plan,
                ctx.allow_segment_buckets,
            )
        else:
            grad_source = _sparse_quadratic_segmented_adjoint_triton(
                grad_output,
                source,
                ctx.adjoint_table,
                ctx.adjoint_plan,
                allow_buckets=ctx.allow_segment_buckets,
            )
        return grad_source, None, None, None, None, None, None, None


class _SparseQuadraticSegmentedBackwardAutograd(torch.autograd.Function):
    """Differentiable adjoint for force-training double backward."""

    @staticmethod
    def forward(
        ctx,
        grad_output,
        source,
        adjoint_table,
        adjoint_plan,
        cross_table,
        cross_plan,
        allow_segment_buckets,
    ):
        ctx.save_for_backward(grad_output, source)
        ctx.adjoint_table = adjoint_table
        ctx.adjoint_plan = adjoint_plan
        ctx.cross_table = cross_table
        ctx.cross_plan = cross_plan
        ctx.allow_segment_buckets = bool(allow_segment_buckets)
        return _sparse_quadratic_segmented_adjoint_triton(
            grad_output,
            source,
            adjoint_table,
            adjoint_plan,
            allow_buckets=ctx.allow_segment_buckets,
        )

    @staticmethod
    def backward(ctx, grad_grad_source):
        grad_output, source = ctx.saved_tensors
        if grad_grad_source is None:
            grad_grad_source = torch.zeros_like(source)
        else:
            grad_grad_source = grad_grad_source.contiguous()
        grad_grad_output = _sparse_bilinear_segmented_forward_triton(
            grad_grad_source,
            source,
            ctx.cross_table,
            ctx.cross_plan,
            allow_buckets=ctx.allow_segment_buckets,
        )
        grad_source = _sparse_quadratic_segmented_adjoint_triton(
            grad_output,
            grad_grad_source,
            ctx.adjoint_table,
            ctx.adjoint_plan,
            allow_buckets=ctx.allow_segment_buckets,
        )
        return (
            grad_grad_output,
            grad_source,
            None,
            None,
            None,
            None,
            None,
        )


class _WeightedSparseBilinearTritonAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        left,
        right,
        weight,
        left_index,
        right_index,
        output_index,
        weight_index,
        coefficient,
        output_width,
    ):
        ctx.save_for_backward(
            left,
            right,
            weight,
            left_index,
            right_index,
            output_index,
            weight_index,
            coefficient,
        )
        ctx.output_width = int(output_width)
        table = WeightedSparseBilinearTable(
            left_index=left_index,
            right_index=right_index,
            output_index=output_index,
            weight_index=weight_index,
            coefficient=coefficient,
            output_width=int(output_width),
            weight_count=int(weight.numel()),
        )
        return _weighted_sparse_bilinear_forward_triton(left, right, weight, table)

    @staticmethod
    def backward(ctx, grad_output):
        (
            left,
            right,
            weight,
            left_index,
            right_index,
            output_index,
            weight_index,
            coefficient,
        ) = ctx.saved_tensors
        table = WeightedSparseBilinearTable(
            left_index=left_index,
            right_index=right_index,
            output_index=output_index,
            weight_index=weight_index,
            coefficient=coefficient,
            output_width=int(ctx.output_width),
            weight_count=int(weight.numel()),
        )
        if torch.is_grad_enabled():
            grad_left, grad_right, grad_weight = weighted_sparse_bilinear_backward_reference(
                grad_output,
                left,
                right,
                weight,
                table,
            )
        else:
            grad_left, grad_right, grad_weight = _weighted_sparse_bilinear_backward_triton(
                grad_output,
                left,
                right,
                weight,
                table,
            )
        return (
            grad_left,
            grad_right,
            grad_weight,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _WeightedSparseBilinearSegmentedAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        left,
        right,
        weight,
        table,
        output_plan,
        left_plan,
        right_plan,
        weight_plan,
    ):
        ctx.save_for_backward(left, right, weight)
        ctx.table = table
        ctx.output_plan = output_plan
        ctx.left_plan = left_plan
        ctx.right_plan = right_plan
        ctx.weight_plan = weight_plan
        return _weighted_sparse_bilinear_segmented_forward_triton(
            left,
            right,
            weight,
            table,
            output_plan,
        )

    @staticmethod
    def backward(ctx, grad_output):
        left, right, weight = ctx.saved_tensors
        grad_left, grad_right, grad_weight = (
            weighted_sparse_bilinear_backward_stable_reference(
                grad_output,
                left,
                right,
                weight,
                ctx.table,
                ctx.output_plan,
                ctx.left_plan,
                ctx.right_plan,
                ctx.weight_plan,
            )
        )
        return (
            grad_left,
            grad_right,
            grad_weight,
            None,
            None,
            None,
            None,
            None,
        )


class _WeightedSparseBilinearArenaSegmentedFastAutograd(
    torch.autograd.Function
):
    """Read a wide arena directly and reduce its adjoint on active support."""

    @staticmethod
    def forward(
        ctx,
        source,
        weight,
        arena_table,
        compact_table,
        output_plan,
        compact_gather_index,
    ):
        ctx.save_for_backward(source, weight, compact_gather_index)
        ctx.compact_table = compact_table
        return _weighted_sparse_bilinear_segmented_forward_triton(
            source,
            source,
            weight,
            arena_table,
            output_plan,
        )

    @staticmethod
    def backward(ctx, grad_output):
        source, weight, compact_gather_index = ctx.saved_tensors
        compact_source = source.index_select(
            1,
            compact_gather_index,
        )
        if torch.is_grad_enabled():
            grad_left, grad_right, grad_weight = (
                weighted_sparse_bilinear_backward_reference(
                    grad_output,
                    compact_source,
                    compact_source,
                    weight,
                    ctx.compact_table,
                )
            )
            grad_source = torch.zeros_like(source).index_add(
                1,
                compact_gather_index,
                grad_left + grad_right,
            )
        else:
            grad_left, grad_right, grad_weight = (
                _weighted_sparse_bilinear_backward_triton(
                    grad_output,
                    compact_source,
                    compact_source,
                    weight,
                    ctx.compact_table,
                )
            )
            grad_source = torch.zeros_like(source)
            grad_source.index_add_(
                1,
                compact_gather_index,
                grad_left + grad_right,
            )
        return (
            grad_source,
            grad_weight,
            None,
            None,
            None,
            None,
        )


class _WeightedSparseBilinearSegmentedFastAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        left,
        right,
        weight,
        table,
        output_plan,
        left_plan,
        right_plan,
        weight_plan,
    ):
        ctx.save_for_backward(left, right, weight)
        ctx.table = table
        ctx.output_plan = output_plan
        return _weighted_sparse_bilinear_segmented_forward_triton(
            left,
            right,
            weight,
            table,
            output_plan,
        )

    @staticmethod
    def backward(ctx, grad_output):
        left, right, weight = ctx.saved_tensors
        if torch.is_grad_enabled():
            grad_left, grad_right, grad_weight = (
                weighted_sparse_bilinear_backward_reference(
                    grad_output,
                    left,
                    right,
                    weight,
                    ctx.table,
                )
            )
        else:
            grad_left, grad_right, grad_weight = (
                _weighted_sparse_bilinear_backward_triton(
                    grad_output,
                    left,
                    right,
                    weight,
                    ctx.table,
                )
            )
        return (
            grad_left,
            grad_right,
            grad_weight,
            None,
            None,
            None,
            None,
            None,
        )


class _WeightedSparseLinearTritonAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        source,
        weight,
        input_index,
        output_index,
        weight_index,
        coefficient,
        output_width,
    ):
        ctx.save_for_backward(
            source,
            weight,
            input_index,
            output_index,
            weight_index,
            coefficient,
        )
        ctx.output_width = int(output_width)
        table = WeightedSparseLinearTable(
            input_index=input_index,
            output_index=output_index,
            weight_index=weight_index,
            coefficient=coefficient,
            output_width=int(output_width),
            weight_count=int(weight.numel()),
        )
        return _weighted_sparse_linear_forward_triton(
            source,
            weight,
            table,
        )

    @staticmethod
    def backward(ctx, grad_output):
        (
            source,
            weight,
            input_index,
            output_index,
            weight_index,
            coefficient,
        ) = ctx.saved_tensors
        table = WeightedSparseLinearTable(
            input_index=input_index,
            output_index=output_index,
            weight_index=weight_index,
            coefficient=coefficient,
            output_width=int(ctx.output_width),
            weight_count=int(weight.numel()),
        )
        if torch.is_grad_enabled():
            grad_source, grad_weight = (
                weighted_sparse_linear_backward_reference(
                    grad_output,
                    source,
                    weight,
                    table,
                )
            )
        else:
            grad_source, grad_weight = (
                _weighted_sparse_linear_backward_triton(
                    grad_output,
                    source,
                    weight,
                    table,
                )
            )
        return (
            grad_source,
            grad_weight,
            None,
            None,
            None,
            None,
            None,
        )


_LAST_SPARSE_BILINEAR_BACKEND = None
_LAST_SPARSE_BILINEAR_BACKWARD_BACKEND = None
_LAST_WEIGHTED_SPARSE_BILINEAR_BACKEND = None
_LAST_WEIGHTED_SPARSE_LINEAR_BACKEND = None


def sparse_bilinear_forward(
    left,
    right,
    table,
    *,
    prefer_triton=True,
    strict=False,
    indices_certified=False,
):
    """Evaluate a compiled sparse bilinear table."""

    global _LAST_SPARSE_BILINEAR_BACKEND

    _validate(
        left,
        right,
        table,
        indices_certified=bool(indices_certified),
    )
    if bool(prefer_triton) and _can_use_triton(left, right):
        table_cuda = table.to(device=left.device, dtype=left.dtype)
        try:
            if torch.is_grad_enabled() and (left.requires_grad or right.requires_grad):
                output = _SparseBilinearTritonAutograd.apply(
                    left,
                    right,
                    table_cuda.left_index,
                    table_cuda.right_index,
                    table_cuda.output_index,
                    table_cuda.coefficient,
                    int(table_cuda.output_width),
                )
                _LAST_SPARSE_BILINEAR_BACKEND = "triton_sparse_bilinear_autograd"
                return output
            output = _sparse_bilinear_forward_triton(left, right, table_cuda)
            _LAST_SPARSE_BILINEAR_BACKEND = "triton_sparse_bilinear"
            return output
        except Exception:
            if bool(strict) or os.environ.get("YE3T_DEBUG_TRITON") == "1":
                raise
    if bool(strict):
        raise RuntimeError("Packed sparse bilinear Triton execution was required but unavailable.")
    _LAST_SPARSE_BILINEAR_BACKEND = "torch_sparse_bilinear_reference"
    return sparse_bilinear_reference(left, right, table)


def sparse_bilinear_backward(grad_output, left, right, table, *, prefer_triton=True, strict=False):
    """Apply the adjoint of a compiled sparse bilinear table."""

    global _LAST_SPARSE_BILINEAR_BACKWARD_BACKEND

    _validate(left, right, table)
    if bool(prefer_triton) and _can_use_triton(left, right) and grad_output.is_cuda:
        try:
            result = _sparse_bilinear_backward_triton(
                grad_output,
                left,
                right,
                table,
            )
            _LAST_SPARSE_BILINEAR_BACKWARD_BACKEND = "triton_sparse_bilinear_backward"
            return result
        except Exception:
            if bool(strict) or os.environ.get("YE3T_DEBUG_TRITON") == "1":
                raise
    if bool(strict):
        raise RuntimeError("Packed sparse bilinear Triton adjoint was required but unavailable.")
    _LAST_SPARSE_BILINEAR_BACKEND = "torch_sparse_bilinear_backward_reference"
    return sparse_bilinear_backward_reference(grad_output, left, right, table)


def weighted_sparse_bilinear_forward(
    left,
    right,
    weight,
    table,
    *,
    prefer_triton=True,
    strict=False,
):
    """Evaluate fused exact bilinear analysis and multiplicity mixing."""

    global _LAST_WEIGHTED_SPARSE_BILINEAR_BACKEND

    _validate_weighted(left, right, weight, table)
    if bool(prefer_triton) and _can_use_triton(left, right):
        table_cuda = table.to(device=left.device, dtype=left.dtype)
        try:
            if torch.is_grad_enabled() and (
                left.requires_grad or right.requires_grad or weight.requires_grad
            ):
                output = _WeightedSparseBilinearTritonAutograd.apply(
                    left,
                    right,
                    weight,
                    table_cuda.left_index,
                    table_cuda.right_index,
                    table_cuda.output_index,
                    table_cuda.weight_index,
                    table_cuda.coefficient,
                    int(table_cuda.output_width),
                )
                _LAST_WEIGHTED_SPARSE_BILINEAR_BACKEND = (
                    "triton_weighted_sparse_bilinear_autograd"
                )
                return output
            output = _weighted_sparse_bilinear_forward_triton(
                left,
                right,
                weight,
                table_cuda,
            )
            _LAST_WEIGHTED_SPARSE_BILINEAR_BACKEND = (
                "triton_weighted_sparse_bilinear"
            )
            return output
        except Exception:
            if bool(strict) or os.environ.get("YE3T_DEBUG_TRITON") == "1":
                raise
    if bool(strict):
        raise RuntimeError("Packed weighted Triton execution was required but unavailable.")
    _LAST_WEIGHTED_SPARSE_BILINEAR_BACKEND = (
        "torch_weighted_sparse_bilinear_reference"
    )
    return weighted_sparse_bilinear_reference(left, right, weight, table)


def weighted_sparse_linear_forward(
    source,
    weight,
    table,
    *,
    prefer_triton=True,
    strict=False,
):
    """Evaluate fused exact source analysis and channel mixing."""

    global _LAST_WEIGHTED_SPARSE_LINEAR_BACKEND

    _validate_weighted_linear(source, weight, table)
    can_use_triton = _can_use_triton(source, source)
    if bool(prefer_triton) and can_use_triton:
        table_cuda = table.to(device=source.device, dtype=source.dtype)
        try:
            if torch.is_grad_enabled() and (
                source.requires_grad or weight.requires_grad
            ):
                output = _WeightedSparseLinearTritonAutograd.apply(
                    source,
                    weight,
                    table_cuda.input_index,
                    table_cuda.output_index,
                    table_cuda.weight_index,
                    table_cuda.coefficient,
                    int(table_cuda.output_width),
                )
                _LAST_WEIGHTED_SPARSE_LINEAR_BACKEND = (
                    "triton_weighted_sparse_linear_autograd"
                )
                return output
            output = _weighted_sparse_linear_forward_triton(
                source,
                weight,
                table_cuda,
            )
            _LAST_WEIGHTED_SPARSE_LINEAR_BACKEND = (
                "triton_weighted_sparse_linear"
            )
            return output
        except Exception:
            if bool(strict) or os.environ.get("YE3T_DEBUG_TRITON") == "1":
                raise
    if bool(strict):
        raise RuntimeError(
            "Packed weighted sparse-linear Triton execution was required "
            "but unavailable."
        )
    _LAST_WEIGHTED_SPARSE_LINEAR_BACKEND = (
        "torch_weighted_sparse_linear_reference"
    )
    return weighted_sparse_linear_reference(source, weight, table)


__all__ = [
    "PackedSparseBilinearGroup",
    "PackedWeightedSparseBilinearGroup",
    "SparseBilinearTable",
    "WeightedSparseBilinearTable",
    "WeightedSparseLinearTable",
    "sparse_bilinear_backward",
    "sparse_bilinear_backward_reference",
    "sparse_bilinear_forward",
    "sparse_bilinear_reference",
    "weighted_sparse_bilinear_backward_reference",
    "weighted_sparse_bilinear_forward",
    "weighted_sparse_bilinear_reference",
    "weighted_sparse_linear_backward_reference",
    "weighted_sparse_linear_forward",
    "weighted_sparse_linear_reference",
]
