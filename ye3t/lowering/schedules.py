
from dataclasses import field
import hashlib

from ye3t.ir.model import E3OperatorIR, PackedPathBlock
from ye3t._record import recordclass


@recordclass(('block_index', 'left_L', 'right_L', 'out_L', 'num_paths', 'packed_support'), frozen = True)
class LoweringSegment:
    pass


@recordclass(('segment_ids', 'segment_support_offsets', 'segment_support_indices', 'segment_coefficient_offsets', 'segment_coefficient_lengths', 'segment_group_ids', 'op_signatures', 'label_texts', 'descriptor_label_offsets', 'descriptor_label_ids', 'metadata'), frozen = True)
class FlatScheduleMetadata:
    """Flat integer metadata generated from symbolic exact schedule blocks."""
    metadata = field(default_factory=dict)

    def as_dict(self):
        return {
            "segment_ids": [int(x) for x in self.segment_ids],
            "segment_support_offsets": [int(x) for x in self.segment_support_offsets],
            "segment_support_indices": [int(x) for x in self.segment_support_indices],
            "segment_coefficient_offsets": [int(x) for x in self.segment_coefficient_offsets],
            "segment_coefficient_lengths": [int(x) for x in self.segment_coefficient_lengths],
            "segment_group_ids": [int(x) for x in self.segment_group_ids],
            "op_signatures": [tuple(int(v) for v in row) for row in self.op_signatures],
            "label_texts": [str(x) for x in self.label_texts],
            "descriptor_label_offsets": [int(x) for x in self.descriptor_label_offsets],
            "descriptor_label_ids": [int(x) for x in self.descriptor_label_ids],
            "metadata": dict(self.metadata),
        }


@recordclass(('segment_index', 'reuse_count', 'estimated_compute', 'estimated_bytes', 'action', 'reason'), frozen = True)
class MaterializationDecision:
    """Compile-time recommendation for one schedule segment."""

    def as_dict(self):
        return {
            "segment_index": int(self.segment_index),
            "reuse_count": int(self.reuse_count),
            "estimated_compute": int(self.estimated_compute),
            "estimated_bytes": int(self.estimated_bytes),
            "action": str(self.action),
            "reason": str(self.reason),
        }


@recordclass(('policy', 'backend', 'decisions', 'materialized_count', 'recomputed_count', 'estimated_saved_compute', 'estimated_memory_bytes', 'metadata'), frozen = True)
class MaterializationPlan:
    """Schedule-level materialization/recompute recommendations."""
    metadata = field(default_factory=dict)

    def as_dict(self):
        return {
            "policy": str(self.policy),
            "backend": str(self.backend),
            "decisions": [decision.as_dict() for decision in self.decisions],
            "materialized_count": int(self.materialized_count),
            "recomputed_count": int(self.recomputed_count),
            "estimated_saved_compute": int(self.estimated_saved_compute),
            "estimated_memory_bytes": int(self.estimated_memory_bytes),
            "metadata": dict(self.metadata),
        }


@recordclass(('backend', 'reuse_threshold', 'aggressive_reuse_threshold', 'max_materialize_bytes', 'compute_per_byte_threshold'), frozen = True)
class ScheduleCostModel:
    """Small deterministic cost model for schedule planning decisions."""
    reuse_threshold = 2
    aggressive_reuse_threshold = 1
    max_materialize_bytes = 1 << 20
    compute_per_byte_threshold = 1


@recordclass(('operator', 'segments', 'eliminated_path_count', 'packed_path_count', 'primitive_sparsified', 'retained_blocks', 'flat_metadata', 'materialization_plan', 'metadata'), frozen = True)
class ExactSchedule:
    retained_blocks = tuple()
    flat_metadata = None
    materialization_plan = None
    metadata = field(default_factory=dict)


@recordclass(('static_path_elimination', 'better_block_packing', 'primitive_first_fusion', 'quotient_aware_sparsification', 'prefer_contiguous_support', 'primitive_segments_first', 'max_paths_per_segment', 'max_segment_product_order'), frozen = True)
class ScheduleOptions:
    static_path_elimination = True
    better_block_packing = True
    primitive_first_fusion = False
    quotient_aware_sparsification = False
    prefer_contiguous_support = True
    primitive_segments_first = True
    max_paths_per_segment = None
    max_segment_product_order = None


@recordclass(('left_L', 'right_L', 'out_L', 'product_order', 'support_width', 'primitive'), frozen = True)
class TritonSegmentSignature:
    """Static specialization key for one grouped Triton lowering family."""

    @property
    def angular_key(self):
        return (int(self.left_L), int(self.right_L), int(self.out_L))

    def as_tuple(self):
        return (
            int(self.left_L),
            int(self.right_L),
            int(self.out_L),
            int(self.product_order),
            int(self.support_width),
            1 if self.primitive else 0,
        )


@recordclass(('signature', 'segment_indices', 'total_paths', 'support_union', 'estimated_flops', 'estimated_bytes', 'fusion_priority', 'recommended_kernel'), frozen = True)
class TritonLoweringGroup:
    """A regular path group intended to become one Triton kernel family."""
    estimated_flops = 0
    estimated_bytes = 0
    fusion_priority = "reference"
    recommended_kernel = "reference_pytorch"

    @property
    def kernel_key(self):
        payload = "|".join(
            [
                ",".join(str(x) for x in self.signature.as_tuple()),
                ",".join(str(i) for i in self.segment_indices),
                ",".join(str(i) for i in self.support_union),
                str(int(self.total_paths)),
            ]
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    def as_dict(self):
        return {
            "signature": {
                "left_L": int(self.signature.left_L),
                "right_L": int(self.signature.right_L),
                "out_L": int(self.signature.out_L),
                "product_order": int(self.signature.product_order),
                "support_width": int(self.signature.support_width),
                "primitive": bool(self.signature.primitive),
            },
            "segment_indices": [int(i) for i in self.segment_indices],
            "total_paths": int(self.total_paths),
            "support_width": int(len(self.support_union)),
            "estimated_flops": int(self.estimated_flops),
            "estimated_bytes": int(self.estimated_bytes),
            "fusion_priority": str(self.fusion_priority),
            "recommended_kernel": str(self.recommended_kernel),
            "kernel_key": self.kernel_key,
        }


@recordclass(('groups', 'total_segments', 'total_paths', 'singleton_group_count', 'eager_kernel_group_count', 'streamable_group_count', 'metadata'), frozen = True)
class TritonLoweringPlan:
    """Backend-agnostic Triton fusion plan derived from an exact schedule.

    This is intentionally a plan, not a custom CUDA kernel. It records the
    static path grouping and specialization keys needed by future Triton
    sparse-CG kernels while the PyTorch reference runtime remains available.
    """
    metadata = field(default_factory=dict)

    def as_dict(self):
        return {
            "total_segments": int(self.total_segments),
            "total_paths": int(self.total_paths),
            "group_count": int(len(self.groups)),
            "singleton_group_count": int(self.singleton_group_count),
            "eager_kernel_group_count": int(self.eager_kernel_group_count),
            "streamable_group_count": int(self.streamable_group_count),
            "groups": [group.as_dict() for group in self.groups],
            "metadata": dict(self.metadata),
        }


def _descriptor_order(descriptor):
    if getattr(descriptor, "kind", "") == "primitive":
        return 1
    left = getattr(descriptor, "left", None)
    right = getattr(descriptor, "right", None)
    return _descriptor_order(left) + _descriptor_order(right) if left is not None and right is not None else 1


def _segment_product_order(block):
    if not block.paths:
        return 0
    return max(_descriptor_order(path.descriptor) for path in block.paths)


def _normalize_optimization_policy(policy):
    normalized = "off" if policy is None else str(policy).strip().lower().replace("-", "_")
    if normalized not in {"off", "auto", "aggressive"}:
        raise ValueError(
            "optimization_policy must be one of {'off', 'auto', 'aggressive'}, "
            f"got {policy!r}."
        )
    return normalized


def _descriptor_label_texts(descriptor):
    if getattr(descriptor, "kind", "") == "primitive":
        handle = getattr(descriptor, "basis_handle", None)
        if handle is not None:
            return (str(handle),)
        label = getattr(descriptor, "basis_label", None)
        if label is not None:
            return (str(label),)
        return tuple()
    labels = []
    left = getattr(descriptor, "left", None)
    right = getattr(descriptor, "right", None)
    if left is not None:
        labels.extend(_descriptor_label_texts(left))
    if right is not None:
        labels.extend(_descriptor_label_texts(right))
    return tuple(labels)


def _block_signature(block, segment):
    return (
        int(segment.left_L),
        int(segment.right_L),
        int(segment.out_L),
        int(_segment_product_order(block)),
        int(len(segment.packed_support)),
        1 if int(segment.left_L) < 0 or int(segment.right_L) < 0 else 0,
    )


def _segment_coefficient_length(block):
    return sum(int(getattr(path, "coefficient_count", 0)) for path in block.paths)


def build_flat_schedule_metadata(schedule):
    """Lower symbolic schedule blocks into deterministic flat metadata arrays."""

    segments = tuple(schedule.segments)
    blocks = tuple(schedule.retained_blocks)
    label_to_id = {}
    label_texts = []
    segment_ids = []
    support_offsets = [0]
    support_indices = []
    coefficient_offsets = []
    coefficient_lengths = []
    group_ids = []
    op_signatures = []
    signature_to_group = {}
    descriptor_label_offsets = [0]
    descriptor_label_ids = []
    coeff_cursor = 0

    for index, segment in enumerate(segments):
        block = blocks[index] if index < len(blocks) else None
        segment_ids.append(int(index))
        support_indices.extend(int(value) for value in segment.packed_support)
        support_offsets.append(len(support_indices))
        if block is None:
            signature = (
                int(segment.left_L),
                int(segment.right_L),
                int(segment.out_L),
                0,
                int(len(segment.packed_support)),
                1 if int(segment.left_L) < 0 or int(segment.right_L) < 0 else 0,
            )
            coeff_len = int(segment.num_paths)
            path_iter = tuple()
        else:
            signature = _block_signature(block, segment)
            coeff_len = _segment_coefficient_length(block)
            path_iter = tuple(block.paths)
        op_signatures.append(signature)
        group_id = signature_to_group.setdefault(signature, len(signature_to_group))
        group_ids.append(int(group_id))
        coefficient_offsets.append(int(coeff_cursor))
        coefficient_lengths.append(int(coeff_len))
        coeff_cursor += int(coeff_len)

        for path in path_iter:
            for label_text in _descriptor_label_texts(path.descriptor):
                label_id = label_to_id.get(label_text)
                if label_id is None:
                    label_id = len(label_texts)
                    label_to_id[label_text] = label_id
                    label_texts.append(label_text)
                descriptor_label_ids.append(int(label_id))
        descriptor_label_offsets.append(len(descriptor_label_ids))

    return FlatScheduleMetadata(
        segment_ids=tuple(segment_ids),
        segment_support_offsets=tuple(support_offsets),
        segment_support_indices=tuple(support_indices),
        segment_coefficient_offsets=tuple(coefficient_offsets),
        segment_coefficient_lengths=tuple(coefficient_lengths),
        segment_group_ids=tuple(group_ids),
        op_signatures=tuple(op_signatures),
        label_texts=tuple(label_texts),
        descriptor_label_offsets=tuple(descriptor_label_offsets),
        descriptor_label_ids=tuple(descriptor_label_ids),
        metadata={
            "group_count": int(len(signature_to_group)),
            "coefficient_count": int(coeff_cursor),
            "label_count": int(len(label_texts)),
            "runtime_shape": "flat_grouped_schedule",
        },
    )


def _angular_width(L):
    return 1 if int(L) < 0 else 2 * int(L) + 1


def _estimate_group_flops(signature, total_paths):
    if signature.primitive:
        return int(total_paths)
    return int(total_paths) * _angular_width(signature.left_L) * _angular_width(signature.right_L)


def _estimate_group_bytes(signature, total_paths, support_width):
    width = _angular_width(signature.left_L) + _angular_width(signature.right_L) + _angular_width(signature.out_L)
    return 8 * int(max(1, total_paths)) * int(max(1, support_width)) * int(max(1, width))


def _estimate_signature_compute(signature, total_paths):
    if isinstance(signature, TritonSegmentSignature):
        return _estimate_group_flops(signature, total_paths)
    left_L, right_L, out_L, product_order, support_width, primitive = signature
    if primitive:
        return int(max(1, total_paths))
    return int(max(1, total_paths)) * _angular_width(left_L) * _angular_width(right_L) * max(1, int(product_order))


def _estimate_signature_bytes(signature, total_paths, support_width):
    if isinstance(signature, TritonSegmentSignature):
        return _estimate_group_bytes(signature, total_paths, support_width)
    left_L, right_L, out_L, product_order, _, primitive = signature
    width = _angular_width(left_L) + _angular_width(right_L) + _angular_width(out_L)
    if primitive:
        width = max(1, _angular_width(out_L))
    return 8 * int(max(1, total_paths)) * int(max(1, support_width)) * int(max(1, width))


def build_materialization_plan(
    schedule,
    *,
    policy = "off",
    backend = "pytorch",
    cost_model = None,
):
    """Choose deterministic materialize/recompute recommendations for segments."""

    normalized = _normalize_optimization_policy(policy)
    model = cost_model or ScheduleCostModel(backend=str(backend))
    metadata = schedule.flat_metadata or build_flat_schedule_metadata(schedule)
    support_use = {}
    offsets = tuple(metadata.segment_support_offsets)
    flat_support = tuple(metadata.segment_support_indices)
    for index in range(len(metadata.segment_ids)):
        for support in set(flat_support[offsets[index] : offsets[index + 1]]):
            support_use[int(support)] = support_use.get(int(support), 0) + 1

    decisions = []
    saved_compute = 0
    memory_bytes = 0
    threshold = int(model.aggressive_reuse_threshold if normalized == "aggressive" else model.reuse_threshold)
    max_bytes = int(model.max_materialize_bytes)
    compute_per_byte = int(model.compute_per_byte_threshold)
    for index, signature in enumerate(metadata.op_signatures):
        support = flat_support[offsets[index] : offsets[index + 1]]
        reuse_count = max([support_use.get(int(value), 0) for value in support] or [0])
        total_paths = int(metadata.segment_coefficient_lengths[index] or 1)
        estimated_compute = _estimate_signature_compute(signature, total_paths)
        estimated_bytes = _estimate_signature_bytes(signature, total_paths, len(support))
        if normalized == "off":
            action = "recompute"
            reason = "policy_off"
        elif reuse_count < threshold:
            action = "recompute"
            reason = "low_reuse"
        elif estimated_bytes > max_bytes:
            action = "recompute"
            reason = "memory_budget"
        elif estimated_compute < estimated_bytes * compute_per_byte:
            action = "recompute"
            reason = "cheap_to_recompute"
        else:
            action = "materialize"
            reason = "reuse_outweighs_memory"
            saved_compute += int(max(0, reuse_count - 1) * estimated_compute)
            memory_bytes += int(estimated_bytes)
        decisions.append(
            MaterializationDecision(
                segment_index=int(index),
                reuse_count=int(reuse_count),
                estimated_compute=int(estimated_compute),
                estimated_bytes=int(estimated_bytes),
                action=action,
                reason=reason,
            )
        )
    return MaterializationPlan(
        policy=normalized,
        backend=str(backend),
        decisions=tuple(decisions),
        materialized_count=sum(1 for decision in decisions if decision.action == "materialize"),
        recomputed_count=sum(1 for decision in decisions if decision.action == "recompute"),
        estimated_saved_compute=int(saved_compute),
        estimated_memory_bytes=int(memory_bytes),
        metadata={
            "cost_model": {
                "reuse_threshold": int(model.reuse_threshold),
                "aggressive_reuse_threshold": int(model.aggressive_reuse_threshold),
                "max_materialize_bytes": int(model.max_materialize_bytes),
                "compute_per_byte_threshold": int(model.compute_per_byte_threshold),
            }
        },
    )


def with_schedule_planning(
    schedule,
    *,
    policy = "off",
    backend = "pytorch",
    cost_model = None,
):
    """Return ``schedule`` with flat metadata and materialization decisions."""

    flat = build_flat_schedule_metadata(schedule)
    planned = ExactSchedule(
        operator=schedule.operator,
        segments=schedule.segments,
        eliminated_path_count=schedule.eliminated_path_count,
        packed_path_count=schedule.packed_path_count,
        primitive_sparsified=schedule.primitive_sparsified,
        retained_blocks=schedule.retained_blocks,
        flat_metadata=flat,
        materialization_plan=None,
        metadata=dict(schedule.metadata),
    )
    materialization = build_materialization_plan(
        planned,
        policy=policy,
        backend=backend,
        cost_model=cost_model,
    )
    metadata = dict(schedule.metadata)
    metadata.update(
        {
            "flat_schedule": flat.as_dict(),
            "materialization_plan": materialization.as_dict(),
            "optimization_policy": _normalize_optimization_policy(policy),
            "planning_backend": str(backend),
        }
    )
    return ExactSchedule(
        operator=schedule.operator,
        segments=schedule.segments,
        eliminated_path_count=schedule.eliminated_path_count,
        packed_path_count=schedule.packed_path_count,
        primitive_sparsified=bool(
            schedule.primitive_sparsified or materialization.materialized_count > 0
        ),
        retained_blocks=schedule.retained_blocks,
        flat_metadata=flat,
        materialization_plan=materialization,
        metadata=metadata,
    )


def _recommend_triton_kernel(
    signature,
    *,
    total_paths,
    segment_count,
):
    if signature.primitive:
        return "reference", "primitive_lookup_reference"
    if int(total_paths) >= 8 or int(segment_count) >= 2:
        return "eager", "grouped_product_cg_forward"
    if int(total_paths) >= 2:
        return "opportunistic", "grouped_product_cg_forward"
    return "reference", "reference_pytorch"


def build_exact_schedule(
    ir,
    options = None,
    *,
    optimization_policy = "off",
    backend = "pytorch",
):
    opts = options or ScheduleOptions()
    retained_blocks = []
    eliminated = 0
    keep_primitive_paths = bool(ir.primitive_first or ir.metadata.get('include_target_primitive', False))
    for block in ir.packed_blocks:
        if opts.static_path_elimination and block.key.left_L < 0 and not keep_primitive_paths:
            # Primitive-only leaves are kept only when they are explicitly requested.
            eliminated += len(block.paths)
            continue
        paths = tuple(block.paths)
        if opts.max_segment_product_order is not None:
            paths = tuple(
                path for path in paths
                if _descriptor_order(path.descriptor) <= int(opts.max_segment_product_order)
            )
        if opts.max_paths_per_segment is not None and int(opts.max_paths_per_segment) > 0:
            paths = paths[: int(opts.max_paths_per_segment)]
        if not paths:
            eliminated += len(block.paths)
            continue
        if paths is not block.paths:
            support_union = tuple(sorted({idx for path in paths for idx in path.support_indices}))
            block = PackedPathBlock(key=block.key, paths=paths, support_union=support_union)
        retained_blocks.append(block)
    if opts.better_block_packing:
        retained_blocks = sorted(
            retained_blocks,
            key=lambda b: (
                0 if (opts.primitive_segments_first and b.key.left_L < 0) else 1,
                _segment_product_order(b),
                b.key.out_L,
                b.key.left_L,
                b.key.right_L,
                -len(b.support_union),
                -len(b.paths),
            ),
        )
    segments = tuple(
        LoweringSegment(
            block_index=i,
            left_L=int(block.key.left_L),
            right_L=int(block.key.right_L),
            out_L=int(block.key.out_L),
            num_paths=len(block.paths),
            packed_support=tuple(block.support_union),
        )
        for i, block in enumerate(retained_blocks)
    )
    base = ExactSchedule(
        operator=ir,
        segments=segments,
        eliminated_path_count=int(eliminated),
        packed_path_count=int(sum(len(block.paths) for block in retained_blocks)),
        primitive_sparsified=bool(opts.primitive_first_fusion or opts.quotient_aware_sparsification),
        retained_blocks=tuple(retained_blocks),
        flat_metadata=None,
        materialization_plan=None,
        metadata={
            'todo_kernel_specialization': 'Lower segments to backend-specific fused kernels keyed by (L_in,L_edge,L_out,product_order,packed_support).',
            'todo_autotuning': 'Benchmark alternative segment orders and tile shapes for Triton/CUDA backends.',
            'primitive_segments_first': bool(opts.primitive_segments_first),
            'max_paths_per_segment': opts.max_paths_per_segment,
            'max_segment_product_order': opts.max_segment_product_order,
        },
    )
    return with_schedule_planning(base, policy=optimization_policy, backend=backend)


def build_triton_lowering_plan(
    schedule,
    *,
    max_groups = None,
):
    """Group exact lowering segments into stable Triton specialization families.

    The grouping mirrors OpenEquivariance/cuEquivariance-style segmented tensor
    product lowering: symbolic ACE/G_nu x SO(3) logic is resolved before
    runtime, then kernels see regular path families keyed by angular tuple,
    product order, support width, and primitive/product status.
    """

    grouped = {}
    blocks = tuple(schedule.retained_blocks)
    for idx, segment in enumerate(schedule.segments):
        if idx >= len(blocks):
            continue
        block = blocks[idx]
        signature = TritonSegmentSignature(
            left_L=int(segment.left_L),
            right_L=int(segment.right_L),
            out_L=int(segment.out_L),
            product_order=int(_segment_product_order(block)),
            support_width=int(len(segment.packed_support)),
            primitive=bool(segment.left_L < 0 or segment.right_L < 0),
        )
        grouped.setdefault(signature, []).append((idx, segment, block))

    groups = []
    for signature, items in grouped.items():
        segment_indices = tuple(int(idx) for idx, _, _ in items)
        support_union = tuple(sorted({int(v) for _, segment, _ in items for v in segment.packed_support}))
        total_paths = sum(int(segment.num_paths) for _, segment, _ in items)
        priority, kernel = _recommend_triton_kernel(
            signature,
            total_paths=int(total_paths),
            segment_count=len(items),
        )
        estimated_flops = _estimate_group_flops(signature, int(total_paths))
        estimated_bytes = _estimate_group_bytes(signature, int(total_paths), len(support_union))
        groups.append(
            TritonLoweringGroup(
                signature=signature,
                segment_indices=segment_indices,
                total_paths=int(total_paths),
                support_union=support_union,
                estimated_flops=int(estimated_flops),
                estimated_bytes=int(estimated_bytes),
                fusion_priority=priority,
                recommended_kernel=kernel,
            )
        )

    groups.sort(
        key=lambda group: (
            0 if group.signature.primitive else 1,
            group.signature.product_order,
            group.signature.out_L,
            group.signature.left_L,
            group.signature.right_L,
            -group.total_paths,
            group.kernel_key,
        )
    )
    if max_groups is not None and int(max_groups) > 0:
        groups = groups[: int(max_groups)]

    return TritonLoweringPlan(
        groups=tuple(groups),
        total_segments=int(len(schedule.segments)),
        total_paths=int(schedule.packed_path_count),
        singleton_group_count=sum(1 for group in groups if len(group.segment_indices) == 1),
        eager_kernel_group_count=sum(1 for group in groups if group.fusion_priority == "eager"),
        streamable_group_count=sum(1 for group in groups if group.fusion_priority in {"eager", "opportunistic"}),
        metadata={
            "kernel_strategy": "segmented_sparse_cg_triton",
            "custom_cuda_required": False,
            "reference_runtime": "native_pytorch",
            "next_kernel_family": "grouped_product_cg_forward",
            "aggressive_but_safe_policy": (
                "Launch Triton for grouped nonprimitive CG/product segments with "
                "enough paths or repeated segment signatures; keep primitive and "
                "tiny singleton groups on the reference path until benchmarks prove a win."
            ),
        },
    )


__all__ = [
    'LoweringSegment',
    'FlatScheduleMetadata',
    'MaterializationDecision',
    'MaterializationPlan',
    'ScheduleCostModel',
    'ExactSchedule',
    'ScheduleOptions',
    'TritonSegmentSignature',
    'TritonLoweringGroup',
    'TritonLoweringPlan',
    'build_flat_schedule_metadata',
    'build_materialization_plan',
    'build_exact_schedule',
    'build_triton_lowering_plan',
    'with_schedule_planning',
]
