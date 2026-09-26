
from ye3t.lowering import compile_ye3t_operator
from ye3t.lowering.schedules import build_triton_lowering_plan


def compile_triton_ye3t_operator(*args, **kwargs):
    kwargs['backend'] = 'triton'
    compiled = compile_ye3t_operator(*args, **kwargs)
    if 'triton_plan' not in compiled.metadata:
        compiled.metadata.update({'triton_plan': build_triton_lowering_plan(compiled.schedule).as_dict()})
    compiled.metadata.update(
        {
            'triton_backend_status': 'planned_reference_runtime',
            'triton_kernel_families': [
                'grouped_product_cg_forward',
                'grouped_product_cg_backward',
                'edge_tp_scatter_fused',
            ],
            'triton_design_goal': (
                'Use exact YE3T/G_nu x SO(3) schedules as Triton specialization keys '
                'rather than introducing custom CUDA kernels.'
            ),
        }
    )
    return compiled


__all__ = ['compile_triton_ye3t_operator']
