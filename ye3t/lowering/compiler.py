
from collections.abc import Sequence as SequenceABC
import torch

from ye3t.ir import build_operator_ir, build_operator_irs, E3OperatorIR
from ye3t.runtime import PrimitiveFeatureSource
from ye3t.runtime import MultiNativeYE3TOperatorModule, NativeYE3TOperatorModule
from .schedules import ExactSchedule, ScheduleOptions, build_exact_schedule, build_triton_lowering_plan, with_schedule_planning
from ye3t._record import recordclass


@recordclass(('ir', 'schedule', 'module', 'backend', 'metadata'))
class CompiledYE3TOperator:

    def forward(self, features_by_L):
        return self.module(features_by_L)

    __call__ = forward


@recordclass(('operators', 'module', 'backend', 'metadata'))
class MultiCompiledYE3TOperator:

    @property
    def irs(self):
        return tuple(op.ir for op in self.operators)

    @property
    def schedules(self):
        return tuple(op.schedule for op in self.operators)

    def forward(self, features_by_L):
        return self.module(features_by_L)

    __call__ = forward


def _maybe_torch_compile(module, backend, enabled):
    if not enabled:
        return module
    compile_fn = getattr(torch, 'compile', None)
    if compile_fn is None:
        return module
    try:
        return compile_fn(module, fullgraph=False, dynamic=True)
    except Exception:
        return module


def _normalize_optimization_policy(policy):
    normalized = "off" if policy is None else str(policy).strip().lower().replace("-", "_")
    if normalized not in {"off", "auto", "aggressive"}:
        raise ValueError(
            "optimization_policy must be one of {'off', 'auto', 'aggressive'}, "
            f"got {policy!r}."
        )
    return normalized


def _normalize_backend_name(backend):
    normalized = str(backend).strip().lower().replace("-", "_")
    aliases = {
        "torch": "pytorch",
        "native": "pytorch",
        "oeq": "openequivariance",
        "open_equivariance": "openequivariance",
    }
    return aliases.get(normalized, normalized)


def _resolve_schedule_options(
    schedule_options,
    *,
    primitive_segments_first = None,
    max_paths_per_segment = None,
    max_segment_product_order = None,
):
    if (
        primitive_segments_first is None
        and max_paths_per_segment is None
        and max_segment_product_order is None
    ):
        return schedule_options
    opts = schedule_options or ScheduleOptions()
    return ScheduleOptions(
        static_path_elimination=opts.static_path_elimination,
        better_block_packing=opts.better_block_packing,
        primitive_first_fusion=opts.primitive_first_fusion,
        quotient_aware_sparsification=opts.quotient_aware_sparsification,
        prefer_contiguous_support=opts.prefer_contiguous_support,
        primitive_segments_first=(
            opts.primitive_segments_first
            if primitive_segments_first is None
            else bool(primitive_segments_first)
        ),
        max_paths_per_segment=(
            opts.max_paths_per_segment
            if max_paths_per_segment is None
            else int(max_paths_per_segment)
        ),
        max_segment_product_order=(
            opts.max_segment_product_order
            if max_segment_product_order is None
            else int(max_segment_product_order)
        ),
    )


def compile_ye3t_operator(
    nin = None,
    lin = None,
    L_R = None,
    *,
    ir = None,
    schedule = None,
    labels_by_L=None,
    primitive_source = None,
    tree_type = 'balanced',
    factorization_policy = 'full',
    primitive_first = False,
    include_target_primitive = False,
    generator_ranks = None,
    generator_Ls = None,
    max_generator_rank = None,
    max_generator_L = None,
    max_recoupling_L = None,
    max_internal_L = None,
    max_target_L = None,
    max_output_features_by_L = None,
    backend = 'pytorch',
    optimization_policy = None,
    schedule_options = None,
    primitive_segments_first = None,
    max_paths_per_segment = None,
    max_segment_product_order = None,
    strict_labels = True,
    torch_compile = None,
):
    """Compile ye3t product paths through the IR/schedule runtime.

    ``nin`` may be the legacy first positional pattern argument, an
    ``E3OperatorIR``, or an ``ExactSchedule``.  Passing ``L_R='all'`` builds one
    target-sector IR/schedule per allowed SO(3) irrep and returns a merged
    multi-operator.
    """
    if isinstance(nin, ExactSchedule):
        schedule = nin
        ir = schedule.operator
        nin = None
    elif isinstance(nin, E3OperatorIR):
        ir = nin
        nin = None
    if schedule is not None:
        ir = schedule.operator
    optimization_policy = _normalize_optimization_policy(optimization_policy)
    schedule_options = _resolve_schedule_options(
        schedule_options,
        primitive_segments_first=primitive_segments_first,
        max_paths_per_segment=max_paths_per_segment,
        max_segment_product_order=max_segment_product_order,
    )
    if ir is not None:
        backend_name = _normalize_backend_name(str(backend))
        if schedule is None:
            schedule = build_exact_schedule(
                ir,
                schedule_options,
                optimization_policy=optimization_policy,
                backend=backend_name,
            )
        else:
            schedule = with_schedule_planning(
                schedule,
                policy=optimization_policy,
                backend=backend_name,
            )
        triton_plan = build_triton_lowering_plan(schedule) if backend_name == 'triton' else None
        openeq_spec = None
        if backend_name == "openequivariance":
            try:
                from ye3t.backends.openequivariance_bridge import export_to_openequivariance_specs

                openeq_spec = export_to_openequivariance_specs(schedule)
            except Exception as exc:
                openeq_spec = {"availability": {"available": False, "reason": f"export failed: {exc}"}}
        module = NativeYE3TOperatorModule(
            schedule,
            labels_by_L=labels_by_L,
            primitive_source=primitive_source,
            max_output_features_by_L=max_output_features_by_L,
            strict_labels=bool(strict_labels),
            optimization_policy=optimization_policy,
            prefer_triton_runtime=bool(backend_name == 'triton'),
            prefer_openequivariance_runtime=bool(backend_name == 'openequivariance'),
        )
        should_compile = bool(torch_compile) or backend_name == 'torch_compile'
        module = _maybe_torch_compile(module, str(backend), should_compile)
        metadata = {
            'lowering': 'native_ye3t_runtime',
            'torch_compile_requested': bool(should_compile),
            'optimization_policy': str(optimization_policy),
            'flat_schedule': schedule.flat_metadata.as_dict() if schedule.flat_metadata is not None else {},
            'materialization_plan': (
                schedule.materialization_plan.as_dict()
                if schedule.materialization_plan is not None
                else {}
            ),
        }
        if triton_plan is not None:
            metadata.update(
                {
                    'triton_lowering': 'planned_segmented_sparse_cg',
                    'triton_plan': triton_plan.as_dict(),
                    'triton_execution': 'native_runtime_segment_kernels_when_supported',
                }
            )
        elif openeq_spec is not None:
            metadata.update(
                {
                    'openequivariance_lowering': 'optional_paired_uvu_cg_blocks',
                    'openequivariance_spec': openeq_spec,
                    'openequivariance_execution': 'native_runtime_paired_uvu_when_supported_else_fallback',
                }
            )
        else:
            metadata['todo_segment_triton'] = 'Generate one specialized kernel family per ExactSchedule segment signature.'
        return CompiledYE3TOperator(
            ir=ir,
            schedule=schedule,
            module=module,
            backend=backend_name,
            metadata=metadata,
        )
    if nin is None or lin is None or L_R is None:
        raise TypeError("compile_ye3t_operator requires either an IR/schedule or nin, lin, and L_R.")
    if isinstance(L_R, str) or isinstance(L_R, SequenceABC):
        return compile_ye3t_operators(
            nin,
            lin,
            L_R,
            labels_by_L=labels_by_L,
            primitive_source=primitive_source,
            tree_type=tree_type,
            factorization_policy=factorization_policy,
            primitive_first=primitive_first,
            include_target_primitive=include_target_primitive,
            generator_ranks=generator_ranks,
            generator_Ls=generator_Ls,
            max_generator_rank=max_generator_rank,
            max_generator_L=max_generator_L,
            max_recoupling_L=max_recoupling_L,
            max_internal_L=max_internal_L,
            max_target_L=max_target_L,
            max_output_features_by_L=max_output_features_by_L,
            backend=backend,
            optimization_policy=optimization_policy,
            schedule_options=schedule_options,
            primitive_segments_first=primitive_segments_first,
            max_paths_per_segment=max_paths_per_segment,
            max_segment_product_order=max_segment_product_order,
            strict_labels=strict_labels,
            torch_compile=torch_compile,
        )
    ir = build_operator_ir(
        nin,
        lin,
        int(L_R),
        tree_type=tree_type,
        factorization_policy=factorization_policy,
        primitive_first=primitive_first,
        include_target_primitive=include_target_primitive,
        generator_ranks=generator_ranks,
        generator_Ls=generator_Ls,
        max_generator_rank=max_generator_rank,
        max_generator_L=max_generator_L,
        max_recoupling_L=max_recoupling_L,
        max_internal_L=max_internal_L,
    )
    return compile_ye3t_operator(
        ir=ir,
        labels_by_L=labels_by_L,
        primitive_source=primitive_source,
        backend=backend,
        optimization_policy=optimization_policy,
        schedule_options=schedule_options,
        primitive_segments_first=primitive_segments_first,
        max_paths_per_segment=max_paths_per_segment,
        max_segment_product_order=max_segment_product_order,
        strict_labels=strict_labels,
        max_output_features_by_L=max_output_features_by_L,
        torch_compile=torch_compile,
    )


def compile_ye3t_operators(
    nin,
    lin,
    L_R,
    *,
    labels_by_L=None,
    primitive_source = None,
    tree_type = 'balanced',
    factorization_policy = 'full',
    primitive_first = False,
    include_target_primitive = False,
    generator_ranks = None,
    generator_Ls = None,
    max_generator_rank = None,
    max_generator_L = None,
    max_recoupling_L = None,
    max_internal_L = None,
    max_target_L = None,
    max_output_features_by_L = None,
    backend = 'pytorch',
    optimization_policy = None,
    schedule_options = None,
    primitive_segments_first = None,
    max_paths_per_segment = None,
    max_segment_product_order = None,
    strict_labels = True,
    torch_compile = None,
):
    schedule_options = _resolve_schedule_options(
        schedule_options,
        primitive_segments_first=primitive_segments_first,
        max_paths_per_segment=max_paths_per_segment,
        max_segment_product_order=max_segment_product_order,
    )
    optimization_policy = _normalize_optimization_policy(optimization_policy)
    backend_name = _normalize_backend_name(str(backend))
    irs = build_operator_irs(
        nin,
        lin,
        L_R,
        tree_type=tree_type,
        factorization_policy=factorization_policy,
        primitive_first=primitive_first,
        include_target_primitive=include_target_primitive,
        generator_ranks=generator_ranks,
        generator_Ls=generator_Ls,
        max_generator_rank=max_generator_rank,
        max_generator_L=max_generator_L,
        max_recoupling_L=max_recoupling_L,
        max_internal_L=max_internal_L,
        max_target_L=max_target_L,
    )
    operators = tuple(
        compile_ye3t_operator(
            ir=one_ir,
            labels_by_L=labels_by_L,
            primitive_source=primitive_source,
            backend=backend_name,
            optimization_policy=optimization_policy,
            schedule_options=schedule_options,
            primitive_segments_first=primitive_segments_first,
            max_paths_per_segment=max_paths_per_segment,
            max_segment_product_order=max_segment_product_order,
            strict_labels=strict_labels,
            max_output_features_by_L=max_output_features_by_L,
            torch_compile=torch_compile,
        )
        for one_ir in irs
    )
    module = MultiNativeYE3TOperatorModule([op.module for op in operators])
    return MultiCompiledYE3TOperator(
        operators=operators,
        module=module,
        backend=backend_name,
        metadata={
            'target_Ls': [int(op.ir.target_L) for op in operators],
            'lowering': 'native_ye3t_runtime_multi_target',
            'optimization_policy': str(optimization_policy),
            'max_target_L': None if max_target_L is None else int(max_target_L),
            'max_internal_L': None if (max_internal_L if max_internal_L is not None else max_recoupling_L) is None else int(max_internal_L if max_internal_L is not None else max_recoupling_L),
        },
    )


__all__ = [
    'CompiledYE3TOperator',
    'MultiCompiledYE3TOperator',
    'compile_ye3t_operator',
    'compile_ye3t_operators',
]
