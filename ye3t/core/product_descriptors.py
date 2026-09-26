"""Runtime-safe product descriptor records."""

from ye3t._record import recordclass


@recordclass(('kind', 'space_nin', 'space_lin', 'L_R', 'basis_index', 'basis_label', 'basis_handle', 'left', 'right'), frozen = True)
class ExactProductColumnDescriptor:
    """Recursive descriptor for one exact product column."""
    basis_index = None
    basis_label = None
    basis_handle = None
    left = None
    right = None

    @property
    def product_order(self):
        if self.kind == "primitive":
            return len(self.space_nin)
        left_order = 0 if self.left is None else int(self.left.product_order)
        right_order = 0 if self.right is None else int(self.right.product_order)
        return left_order + right_order


__all__ = ["ExactProductColumnDescriptor"]
