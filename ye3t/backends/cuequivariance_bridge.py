

from ye3t.ir import E3OperatorIR
from ye3t.lowering import ExactSchedule
from ye3t._record import recordclass


@recordclass(('operator_name', 'path_count', 'segment_count', 'metadata'), frozen = True)
class CuEquivarianceBridgeSpec:
    """Backend-agnostic summary for segmented tensor-product export."""


def export_to_cuequivariance_ir(ir):
    if isinstance(ir, ExactSchedule):
        schedule = ir
        ir = schedule.operator
        segment_count = len(schedule.segments)
    else:
        segment_count = len(ir.packed_blocks)
    return CuEquivarianceBridgeSpec(
        operator_name=ir.name,
        path_count=ir.num_paths,
        segment_count=segment_count,
        metadata={
            'bridge_kind': 'segmented_tensor_product_summary',
        },
    )


__all__ = ['CuEquivarianceBridgeSpec', 'export_to_cuequivariance_ir']
