
from dataclasses import field

from ye3t.core.irreps import IrrepTerm
from ye3t.core.product_descriptors import ExactProductColumnDescriptor
from ye3t._record import recordclass


@recordclass(('left_L', 'right_L', 'out_L'), frozen = True)
class PathKey:
    pass


@recordclass(('key', 'descriptor', 'support_indices', 'coefficient_count', 'primitive_first'), frozen = True)
class ExactPath:
    primitive_first = False


@recordclass(('key', 'paths', 'support_union'), frozen = True)
class PackedPathBlock:
    pass


@recordclass(('name', 'input_irreps', 'output_irreps', 'target_nin', 'target_lin', 'target_L', 'primitive_first', 'factorization_policy', 'paths', 'packed_blocks', 'metadata'), frozen = True)
class E3OperatorIR:
    metadata = field(default_factory=dict)

    @property
    def num_paths(self):
        return len(self.paths)
