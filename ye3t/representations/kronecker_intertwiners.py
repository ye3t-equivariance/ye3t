r"""Exact Kronecker intertwiners with a declared multiplicity gauge.

For partitions ``mu, alpha, beta`` of ``n`` this module constructs a basis of

    Hom_{S_n}( S^mu , S^alpha (x) S^beta ),

whose dimension is the Kronecker coefficient ``g_{mu alpha beta}``. These are
the maps that realize ``S_mu(W (x) V) = (+) g S_alpha(W) (x) S_beta(V)``; the
symmetric and skew Cauchy pairings are the cases ``mu = (n)`` and
``mu = (1^n)``.

Algorithm. The intertwining equations are solved over the rationals in
Young's seminormal form, where every adjacent transposition is a rational
matrix, and then transported to the project's Young-orthogonal form by the
diagonal seminormal norms ``gamma_T``, with
``gamma_{s_i T} = (1 - 1/d^2) gamma_T`` for axial distance ``d > 0``. Path
independence of ``gamma_T`` is asserted during construction.

Multiplicity gauge (declared, deterministic). The rational solution space is
taken in reduced row echelon pivot order, mapped to the orthogonal form, and
orthonormalized by exact Gram-Schmidt under ``<A, B> = tr(A^T B) / dim S^mu``;
by Schur's lemma ``A^T B`` is that inner product times the identity. Each
intertwiner is scaled to an isometry and its first nonzero entry is positive.

Algorithmic reference: Young's seminormal and orthogonal forms of the
symmetric group. Paper/reference: James and Kerber, *The Representation Theory
of the Symmetric Group* (1981); Okounkov and Vershik, "A new approach to
representation theory of symmetric groups" (1996). REFERENCE_TODO: record the
exact section and equation numbers before publication-facing use.
Implementation note: independent implementation; no source copied.
"""

from functools import lru_cache
from math import factorial

from ye3t.representations.projectors import (
    _murnaghan_nakayama_character,
    _swap_tableau_entries,
    _symmetric_group_conjugacy_classes,
    _tableau_content,
    adjacent_transposition_representation_matrix,
    standard_tableaux,
)


def _sympy():
    import sympy as sp

    return sp


def kronecker_multiplicity(parent, left, right):
    """Exact ``g_{parent,left,right}`` from the character inner product."""

    parent, left, right = (tuple(int(v) for v in value) for value in (parent, left, right))
    if any(any(v <= 0 for v in parts) or tuple(sorted(parts, reverse=True)) != parts
           for parts in (parent, left, right)):
        raise ValueError("Kronecker labels must be partitions with positive nonincreasing parts")
    order = sum(parent)
    if sum(left) != order or sum(right) != order:
        raise ValueError("Kronecker partitions must share one tensor order.")
    if order == 0:
        # S_0 has one element and one one-dimensional irrep, labelled ().
        return 1
    for fixed, a, b in ((parent, left, right), (left, parent, right), (right, parent, left)):
        if fixed == (order,):
            return int(a == b)
        if fixed == (1,)*order:
            return int(a == _conjugate(b))
    total = 0
    for cycle_type, size in _symmetric_group_conjugacy_classes(order):
        cycle_key = tuple(sorted((int(part) for part in cycle_type), reverse=True))
        total += (
            int(size)
            * int(_murnaghan_nakayama_character(parent, cycle_key))
            * int(_murnaghan_nakayama_character(left, cycle_key))
            * int(_murnaghan_nakayama_character(right, cycle_key))
        )
    if total % factorial(order):
        raise RuntimeError("Kronecker character sum is not an integer multiple of n!.")
    return total // factorial(order)


@lru_cache(maxsize=None)
def _seminormal_data(partition):
    """Rational seminormal generators and the norms ``gamma_T`` of one irrep."""

    sp = _sympy()
    tableaux = standard_tableaux(partition)
    index = {tableau: position for position, tableau in enumerate(tableaux)}
    order = sum(partition)
    gamma = {0: sp.Integer(1)}
    frontier = [0]
    while frontier:
        current = frontier.pop()
        for generator in range(order - 1):
            swapped = _swap_tableau_entries(
                tableaux[current], generator + 1, generator + 2
            )
            if swapped is None:
                continue
            distance = sp.Integer(
                _tableau_content(tableaux[current], generator + 2)
                - _tableau_content(tableaux[current], generator + 1)
            )
            ratio = 1 - 1 / distance**2
            value = gamma[current] * ratio if distance > 0 else gamma[current] / ratio
            target = index[swapped]
            if target in gamma:
                if sp.simplify(gamma[target] - value) != 0:
                    raise RuntimeError("Seminormal norms are path dependent.")
                continue
            gamma[target] = value
            frontier.append(target)
    if len(gamma) != len(tableaux):
        raise RuntimeError("Standard tableaux are not connected by adjacent swaps.")
    generators = []
    for generator in range(order - 1):
        matrix = sp.zeros(len(tableaux), len(tableaux))
        for column, tableau in enumerate(tableaux):
            distance = sp.Integer(
                _tableau_content(tableau, generator + 2)
                - _tableau_content(tableau, generator + 1)
            )
            swapped = _swap_tableau_entries(tableau, generator + 1, generator + 2)
            if swapped is None:
                matrix[column, column] = 1 if distance == 1 else -1
                continue
            matrix[column, column] = 1 / distance
            matrix[index[swapped], column] = (
                1 if distance > 0 else 1 - 1 / distance**2
            )
        generators.append(matrix)
    scale = sp.diag(*[sp.sqrt(gamma[position]) for position in range(len(tableaux))])
    for generator, matrix in enumerate(generators):
        orthogonal = adjacent_transposition_representation_matrix(partition, generator)
        if (scale * matrix * scale.inv() - orthogonal).applyfunc(sp.simplify) != sp.zeros(
            *matrix.shape
        ):
            raise RuntimeError(
                "Seminormal generators do not transport to the Young-orthogonal form."
            )
    return tuple(generators), scale


def _conjugate(partition):
    return tuple(sum(row > column for row in partition) for column in range(partition[0])) if partition else ()


def _sign_twist(partition):
    """Signed tableau-transpose map, in the existing positive-pivot gauge."""
    sp = _sympy()
    tableaux = standard_tableaux(partition)
    transposed = {tableau: index for index, tableau in enumerate(standard_tableaux(_conjugate(partition)))}
    result = sp.zeros(len(tableaux))
    for row, tableau in enumerate(tableaux):
        transpose = tuple(tuple(tableau[i][j] for i in range(len(tableau)) if j < len(tableau[i]))
                          for j in range(len(tableau[0])))
        word = tuple(value for line in tableau for value in line)
        sign = (-1)**sum(word[i] > word[j] for i in range(len(word)) for j in range(i+1, len(word)))
        result[row, transposed[transpose]] = sign
    if next(value for value in result if value != 0) < 0:
        result = -result
    return result


@lru_cache(maxsize=None)
def exact_kronecker_intertwiners(parent, left, right):
    """Exact maps with identity, invariant-metric and sign-twist fast paths.

    Tensoring by sign transposes a Young diagram. The signed tableau-transpose
    map follows the same Young-orthogonal adjacent-generator relations and
    positive-pivot isometry gauge as the general exact solver below.
    """
    sp = _sympy()
    parent, left, right = (tuple(map(int, value)) for value in (parent, left, right))
    multiplicity = kronecker_multiplicity(parent, left, right)
    if multiplicity == 0:
        return {"multiplicity": 0, "intertwiners": (), "gauge": _GAUGE}
    if not parent:
        return {"multiplicity": 1, "intertwiners": (sp.eye(1),), "gauge": _GAUGE}
    trivial, sign = (sum(parent),), (1,)*sum(parent)
    matrix = None
    if left == trivial or right == trivial:
        matrix = sp.eye(len(standard_tableaux(parent)))
    elif left == sign or right == sign:
        matrix = _sign_twist(right if left == sign else left)
    elif parent == trivial:
        dimension = len(standard_tableaux(left))
        matrix = sp.eye(dimension).reshape(dimension**2, 1)/sp.sqrt(dimension)
    elif parent == sign:
        matrix = _sign_twist(left)
        matrix = matrix.reshape(matrix.rows**2, 1)/sp.sqrt(matrix.rows)
    if matrix is not None:
        return {"multiplicity": 1, "intertwiners": (matrix,), "gauge": _GAUGE}
    return _exact_kronecker_intertwiners_general(parent, left, right)


@lru_cache(maxsize=None)
def _exact_kronecker_intertwiners_general(parent, left, right):
    """Orthonormal exact basis of ``Hom_{S_n}(S^parent, S^left (x) S^right)``.

    Returns a dict with ``multiplicity`` and ``intertwiners``: SymPy matrices
    of shape ``[dim left * dim right, dim parent]`` in the Young-orthogonal
    bases, rows in Kronecker-product order ``left (x) right``.
    """

    sp = _sympy()
    parent, left, right = (tuple(int(v) for v in value) for value in (parent, left, right))
    multiplicity = kronecker_multiplicity(parent, left, right)
    if multiplicity == 0:
        return {"multiplicity": 0, "intertwiners": tuple(), "gauge": _GAUGE}
    parent_generators, parent_scale = _seminormal_data(parent)
    left_generators, left_scale = _seminormal_data(left)
    right_generators, right_scale = _seminormal_data(right)
    parent_dimension = parent_scale.rows
    child_dimension = left_scale.rows * right_scale.rows
    if sum(parent) == 1:
        if parent != (1,) or left != (1,) or right != (1,) or multiplicity != 1:
            raise ArithmeticError("Rank-one Kronecker labels are inconsistent.")
        return {"multiplicity": 1, "intertwiners": (sp.eye(1),), "gauge": _GAUGE}
    # vec(T) with T[child, parent] stored at child * parent_dimension + parent.
    equations = []
    for parent_matrix, left_matrix, right_matrix in zip(
        parent_generators, left_generators, right_generators, strict=True
    ):
        child = sp.kronecker_product(left_matrix, right_matrix)
        equations.append(
            sp.kronecker_product(child, sp.eye(parent_dimension))
            - sp.kronecker_product(sp.eye(child_dimension), parent_matrix.T)
        )
    solutions = sp.Matrix.vstack(*equations).nullspace()
    if len(solutions) != multiplicity:
        raise RuntimeError(
            "Exact intertwiner space dimension disagrees with the Kronecker character count."
        )
    child_scale = sp.kronecker_product(left_scale, right_scale)
    basis = []
    for solution in solutions:
        candidate = (
            child_scale
            * sp.Matrix(solution).reshape(child_dimension, parent_dimension)
            * parent_scale.inv()
        ).applyfunc(sp.simplify)
        for previous in basis:
            overlap = sp.simplify((previous.T * candidate).trace() / parent_dimension)
            candidate = (candidate - overlap * previous).applyfunc(sp.simplify)
        norm = sp.sqrt(sp.simplify((candidate.T * candidate).trace() / parent_dimension))
        candidate = (candidate / norm).applyfunc(sp.simplify)
        pivot = next(value for value in candidate if sp.simplify(value) != 0)
        if pivot.is_negative:
            candidate = -candidate
        basis.append(candidate)
    for position, candidate in enumerate(basis):
        for generator in range(sum(parent) - 1):
            child = sp.kronecker_product(
                adjacent_transposition_representation_matrix(left, generator),
                adjacent_transposition_representation_matrix(right, generator),
            )
            residual = child * candidate - candidate * (
                adjacent_transposition_representation_matrix(parent, generator)
            )
            if residual.applyfunc(sp.simplify) != sp.zeros(*residual.shape):
                raise RuntimeError("An emitted map is not an exact intertwiner.")
        for other in basis[: position + 1]:
            expected = sp.eye(parent_dimension) if other is candidate else sp.zeros(
                parent_dimension, parent_dimension
            )
            if (other.T * candidate - expected).applyfunc(sp.simplify) != sp.zeros(
                parent_dimension, parent_dimension
            ):
                raise RuntimeError("Intertwiner gauge is not orthonormal.")
    return {
        "multiplicity": int(multiplicity),
        "intertwiners": tuple(basis),
        "gauge": _GAUGE,
    }


_GAUGE = (
    "seminormal_rational_rref_pivot_order_then_exact_gram_schmidt_"
    "isometry_first_nonzero_entry_positive_v1"
)


__all__ = ["exact_kronecker_intertwiners", "kronecker_multiplicity"]
