"""Exact ordered image selection in independent target coefficient coordinates.

This arithmetic is for a source map on multiplicity coordinates. It does not
construct representation couplings or determine independence from environments.
The input coordinate functions must already have proved independence.
"""

from ye3t.exact_scalars import ExactRadical, exact_scalar


def _divide_exact(left, right):
    """Use the existing radical field, with exact denominator rationalization."""
    try:
        return left / right
    except ValueError:
        import sympy as sp
        expression = sp.radsimp(left._sympy_() / right._sympy_(),
                               symbolic=False, max_terms=len(right.terms))
        quotient = exact_scalar(sp.expand(expression))
        if quotient * right != left:
            raise ArithmeticError("Exact radical source-image division failed.")
        return quotient


class _ExactSourceImageGram:
    """Stream sparse image columns and exact Hermitian LDL reconstruction.

    G=K^dagger W K for positive diagonal W. With accepted columns F, solve
    (F^dagger W F)c=F^dagger W k. The Schur complement is zero precisely
    when k is in their span. No square roots or dense target matrix are used.
    The prescribed column order fixes which original coordinates survive.

    Independent implementation of Hermitian LDL/Schur-complement identities.
    Algorithmic reference: Golub & Van Loan, Matrix Computations, 4th ed.,
    section 4.2 (positive definite systems). The coefficient metric is not a
    physical or sampled-environment metric. Each instance owns its workspaces.
    """

    def __init__(self, *, weights=None, scalar=exact_scalar, divide=_divide_exact,
                 maximum_cells=40000000):
        self.scalar = scalar
        self.divide = divide
        self.zero, self.one = scalar(0), scalar(1)
        self.weights = {key: self._scalar(value) for key, value in (weights or {}).items()}
        for weight in self.weights.values():
            self._positive(weight)
        self.maximum_cells = int(maximum_cells)
        if self.maximum_cells < 1:
            raise ValueError("Source-image coefficient-cell bound must be positive.")
        self.selected = []
        self.feature_rows = []
        self.reconstructions = []
        self.lower = []
        self.diagonal = []
        self.cells = 0

    def _scalar(self, value):
        if isinstance(value, (float, complex)) or getattr(value, "is_Float", False):
            raise TypeError("Exact source-image planning does not accept floating coefficients.")
        if hasattr(value, "atoms") and any(getattr(atom, "is_Float", False) for atom in value.atoms()):
            raise TypeError("Exact source-image planning does not accept floating coefficients.")
        return self.scalar(value)

    def _positive(self, value):
        if value != value.conjugate():
            raise ArithmeticError("Source-image Gram residual must be real.")
        if isinstance(value, ExactRadical):
            if value.terms and all(coefficient > 0 for coefficient in value.terms.values()):
                return
            # This rare mixed-radical sign check is exact, never a tolerance.
            positive = value._sympy_().is_positive
        else:
            positive = getattr(value, "is_positive", None)
            if positive is None:
                positive = value > 0
        if positive is not True:
            raise ArithmeticError("Source-image Gram residual is not certified positive.")

    def _pair(self, left, right):
        if len(left) > len(right):
            return self._pair(right, left).conjugate()
        return self.scalar(sum((value.conjugate() * self.weights.get(key, self.one) * right[key]
                               for key, value in left.items() if key in right), self.zero))

    def append(self, raw):
        """Return whether this column adds an independent source function."""
        if self.cells + len(raw) + 3*len(self.selected) + 1 > self.maximum_cells:
            raise MemoryError("Exact source-image planning exceeds maximum_loader_symbolic_cells.")
        row = {key: self._scalar(value) for key, value in raw.items() if value}
        h = [self._pair(previous, row) for previous in self.feature_rows]
        transformed = []
        for index, value in enumerate(h):
            transformed.append(self.scalar(value - sum((coefficient*transformed[k]
                for k, coefficient in self.lower[index].items()), self.zero)))
        scaled = [self.divide(value, diagonal) if value else self.zero
                  for value, diagonal in zip(transformed, self.diagonal)]
        delta = self.scalar(self._pair(row, row) - sum((value.conjugate()*coefficient
                             for value, coefficient in zip(transformed, scaled)), self.zero))
        if delta:
            self._positive(delta)
            self.selected.append(len(self.reconstructions))
            self.feature_rows.append(row)
            self.lower.append({index: value.conjugate() for index, value in enumerate(scaled) if value})
            self.diagonal.append(delta)
            reconstruction = {len(self.selected)-1: self.one}
            self.cells += len(row) + len(self.lower[-1]) + 1
        else:
            coefficients = list(scaled)
            for j in range(len(coefficients)-1, -1, -1):
                if coefficients[j]:
                    for index, value in self.lower[j].items():
                        coefficients[index] = self.scalar(coefficients[index] - value.conjugate()*coefficients[j])
            reconstruction = {index: value for index, value in enumerate(coefficients) if value}
        self.reconstructions.append(reconstruction)
        self.cells += len(reconstruction)
        return bool(delta)

    def validate_reconstruction(self, raw_rows):
        """Explicit development diagnostic; never called during evaluation."""
        if len(raw_rows) != len(self.reconstructions):
            raise ValueError("Source-image reconstruction column count differs.")
        for raw, coordinates in zip(raw_rows, self.reconstructions):
            difference = {key: self._scalar(value) for key, value in raw.items() if value}
            for index, coefficient in coordinates.items():
                for key, value in self.feature_rows[index].items():
                    result = self.scalar(difference.get(key, self.zero) - coefficient*value)
                    if result:
                        difference[key] = result
                    else:
                        difference.pop(key, None)
            if difference:
                raise ArithmeticError("Exact source-image reconstruction failed.")
        return True
