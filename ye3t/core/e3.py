
"""Utilities for E(3)-equivariant irrep products and repeated-channel blocks."""
from functools import lru_cache
from itertools import product
import re

import numpy as np
import torch

from ye3t.core.basis.tree import RawLeaf, get_tree_factory
from ye3t.core.basis.validation import validate_tree_type
from ye3t.core.irreps import IrrepTerm
from ye3t.core.product_engine import ExactProductExpansionEngine
from ye3t.core.couplings import clebsch_gordan
from ye3t.core.rotation import wigner_D_numeric
from ye3t._record import recordclass


_IRREP_RE = re.compile(r"\s*(?:(\d+)x)?(\d+)([eo])\s*")


def parse_irreps(irreps):
    if isinstance(irreps, str):
        terms = []
        for part in irreps.split("+"):
            part = part.strip()
            if not part:
                continue
            match = _IRREP_RE.fullmatch(part)
            if match is None:
                raise ValueError(f"Could not parse irreps term: {part!r}")
            terms.append(IrrepTerm(int(match.group(1) or 1), int(match.group(2)), match.group(3)))
        return terms
    return [IrrepTerm(int(term.mul), int(term.l), str(term.parity)) for term in irreps]


def format_irreps(terms):
    return " + ".join(term.to_string() for term in terms)


def total_dim(terms):
    return sum(term.dim for term in parse_irreps(terms))


@recordclass(('L_left', 'L_right', 'output_Ls', 'change_of_basis', 'column_slices_by_L'), frozen = True)
class TensorProductReduction:
    """CG reduction table for one pair of SO(3) irreps."""

    @property
    def raw_dim(self):
        return (2 * int(self.L_left) + 1) * (2 * int(self.L_right) + 1)

    @property
    def reduced_dim(self):
        return int(self.change_of_basis.shape[1])


@recordclass(('l', 'rank', 'tree_type', 'raw_dim', 'symmetry_adapted_dim', 'irreps_out', 'labels_by_L', 'change_of_basis', 'column_slices_by_L'), frozen = True)
class RepeatedChannelReduction:
    """Symmetry-adapted reduction for a homogeneous repeated-channel product."""

    @property
    def sparsity_fraction(self):
        return 1.0 - float(np.count_nonzero(np.abs(self.change_of_basis) > 1e-14)) / float(self.change_of_basis.size)


@recordclass(('reduction', 'rotation_matrix', 'D_wig_reduced', 'off_block_max_abs', 'zero_fraction'), frozen = True)
class RepeatedChannelBlockDiagonalization:
    """Rotation action in the repeated-channel symmetry-adapted basis."""


@recordclass(('nin', 'lin', 'tree_type', 'raw_dim', 'symmetry_adapted_dim', 'irreps_out', 'labels_by_L', 'change_of_basis', 'column_slices_by_L'), frozen = True)
class YE3TChannelReduction:
    """ye3t reduction for a fixed canonical leaf pattern."""

    @property
    def rank(self):
        return len(self.lin)

    @property
    def sparsity_fraction(self):
        return 1.0 - float(np.count_nonzero(np.abs(self.change_of_basis) > 1e-14)) / float(self.change_of_basis.size)


@recordclass(('nin', 'lin', 'tree_type', 'raw_dim', 'symmetry_adapted_dim', 'irreps_out', 'labels_by_L', 'column_slices_by_L', 'label_repr_bytes'), frozen = True)
class YE3TSymbolicSummary:
    """Symbolic/count-only ye3t summary for a fixed leaf pattern.

    This path records exact labels and multiplicities without expanding the
    labels into uncoupled magnetic-coordinate vectors.  In particular, it does
    not materialize the dense ``raw_dim x symmetry_adapted_dim`` basis matrix
    used by diagnostic block-diagonalization helpers.
    """

    @property
    def rank(self):
        return len(self.lin)

    @property
    def label_count(self):
        return sum(len(labels) for labels in self.labels_by_L.values())


@recordclass(('reduction', 'rotation_matrix', 'D_wig_reduced', 'off_block_max_abs', 'zero_fraction'), frozen = True)
class YE3TBlockDiagonalization:
    """Rotation action in a ye3t symmetry-adapted basis."""


@recordclass(('nin', 'lin', 'L_R', 'factorization_policy', 'full_dim', 'decomposable_dim', 'primitive_quotient_dim'), frozen = True)
class PrimitiveQuotientDimensionSummary:
    """Small exact dimension summary for one primitive quotient."""


@recordclass(('lin', 'tree_type', 'raw_dim', 'coupled_dim', 'irreps_out', 'change_of_basis', 'column_slices_by_L'), frozen = True)
class NaiveCoupledTreeReduction:
    """Sequential CG coupled-tree reduction without repeated-channel quotienting."""

    @property
    def sparsity_fraction(self):
        return 1.0 - float(np.count_nonzero(np.abs(self.change_of_basis) > 1e-14)) / float(self.change_of_basis.size)


@recordclass(('reduction', 'rotation_matrix', 'D_wig_reduced', 'off_block_max_abs', 'zero_fraction'), frozen = True)
class NaiveCoupledTreeBlockDiagonalization:
    """Rotation action in a naive sequential CG coupled-tree basis."""


def _complex_value(value):
    if hasattr(value, "evalf"):
        return complex(value.evalf())
    return complex(value)


def wigner_D_wig_from_matrix(L, rotation_matrix, n_samples = 25):
    """Return the complex Wigner matrix for one SO(3) irrep.

    The function name intentionally keeps the Wigner matrix distinct from
    decomposable subspaces commonly denoted ``D`` in the ACE algebra.
    """
    return np.asarray(wigner_D_numeric(int(L), np.asarray(rotation_matrix, dtype=float), n_samples=int(n_samples)))


def wigner_D_wig_z(L, angle):
    """Return the complex Wigner matrix for a rotation about the z axis."""
    m_values = np.arange(-int(L), int(L) + 1, dtype=float)
    return np.diag(np.exp(-1j * m_values * float(angle)))


def block_diag_dense(blocks):
    """Dense block diagonal assembly without requiring SciPy."""
    if not blocks:
        return np.zeros((0, 0), dtype=np.complex128)
    rows = sum(int(block.shape[0]) for block in blocks)
    cols = sum(int(block.shape[1]) for block in blocks)
    out = np.zeros((rows, cols), dtype=np.result_type(*[block.dtype for block in blocks], np.complex128))
    r0 = 0
    c0 = 0
    for block in blocks:
        r1 = r0 + int(block.shape[0])
        c1 = c0 + int(block.shape[1])
        out[r0:r1, c0:c1] = block
        r0 = r1
        c0 = c1
    return out


@lru_cache(maxsize=None)
def cg_tensor_product_reduction(L_left, L_right):
    """Build the coupled-basis change of basis for ``V_L_left tensor V_L_right``."""
    L_left = int(L_left)
    L_right = int(L_right)
    product_ms = [(m1, m2) for m1 in range(-L_left, L_left + 1) for m2 in range(-L_right, L_right + 1)]
    output_Ls = tuple(range(abs(L_left - L_right), L_left + L_right + 1))
    columns = []
    slices = {}
    start = 0
    for L_out in output_Ls:
        for M_out in range(-L_out, L_out + 1):
            col = np.zeros(len(product_ms), dtype=np.complex128)
            for row, (m1, m2) in enumerate(product_ms):
                col[row] = clebsch_gordan(L_left, m1, L_right, m2, L_out, M_out)
            columns.append(col)
        stop = start + 2 * L_out + 1
        slices[int(L_out)] = slice(start, stop)
        start = stop
    change_of_basis = np.stack(columns, axis=1) if columns else np.zeros((len(product_ms), 0), dtype=np.complex128)
    return TensorProductReduction(
        L_left=L_left,
        L_right=L_right,
        output_Ls=output_Ls,
        change_of_basis=change_of_basis,
        column_slices_by_L=slices,
    )


def reduce_tensor_product_D_wig(
    L_left,
    L_right,
    D_wig_left,
    D_wig_right,
):
    """Transform a raw tensor-product rotation matrix to coupled coordinates."""
    reduction = cg_tensor_product_reduction(int(L_left), int(L_right))
    raw_action = np.kron(np.asarray(D_wig_left), np.asarray(D_wig_right))
    Q = reduction.change_of_basis
    return Q.conj().T @ raw_action @ Q


def tensor_product_output_irreps(
    irreps_left,
    irreps_right,
):
    """Return the irrep inventory generated by pairwise tensor products."""
    out = {}
    for left in parse_irreps(irreps_left):
        for right in parse_irreps(irreps_right):
            parity = "e" if left.parity == right.parity else "o"
            for L_out in range(abs(left.l - right.l), left.l + right.l + 1):
                key = (int(L_out), parity)
                out[key] = out.get(key, 0) + int(left.mul) * int(right.mul)
    return [IrrepTerm(mul=mul, l=L, parity=parity) for (L, parity), mul in sorted(out.items())]


def _couple_branch_states(left_states, right_states):
    states = []
    for L_left, Q_left in left_states:
        for L_right, Q_right in right_states:
            pair_reduction = cg_tensor_product_reduction(int(L_left), int(L_right))
            kron_basis = np.kron(Q_left, Q_right)
            for L_out, slc in pair_reduction.column_slices_by_L.items():
                states.append((int(L_out), kron_basis @ pair_reduction.change_of_basis[:, slc]))
    return states


def _normalize_basis_columns(matrix):
    """Return a copy with nonzero columns normalized to unit 2-norm."""
    if matrix.size == 0:
        return matrix
    out = np.array(matrix, dtype=np.complex128, copy=True)
    norms = np.linalg.norm(out, axis=0)
    if np.any(norms <= 0):
        raise ValueError("Basis construction produced a zero column.")
    out /= norms[np.newaxis, :]
    return out


def _naive_branch_states_on_raw_tree(node):
    if isinstance(node, RawLeaf):
        l_value = int(node.l)
        return [(l_value, np.eye(2 * l_value + 1, dtype=np.complex128))]
    left_states = _naive_branch_states_on_raw_tree(node.left)
    right_states = _naive_branch_states_on_raw_tree(node.right)
    return _couple_branch_states(left_states, right_states)


@lru_cache(maxsize=None)
def naive_coupled_tree_reduction(lin, tree_type = "balanced"):
    """Build a sequential CG coupled-tree basis without repeated-channel reduction.

    The resulting basis is unitary-equivalent to the raw tensor product; it
    block diagonalizes rotations but keeps multiplicities generated by all
    coupling paths.
    """
    lin = tuple(int(l_value) for l_value in lin)
    if not lin:
        raise ValueError("lin must contain at least one leaf angular momentum.")
    tree_type = validate_tree_type(tree_type)
    raw_tree = get_tree_factory(tree_type).build_raw_tree([0] * len(lin), list(lin))
    states = _naive_branch_states_on_raw_tree(raw_tree)
    grouped = {}
    for L_out, basis in states:
        grouped.setdefault(int(L_out), []).append(basis)
    columns = []
    irreps = []
    slices = {}
    start = 0
    for L_out in sorted(grouped):
        for basis in grouped[L_out]:
            for col_idx in range(basis.shape[1]):
                columns.append(basis[:, col_idx])
        multiplicity = len(grouped[L_out])
        irreps.append(IrrepTerm(mul=int(multiplicity), l=int(L_out), parity="e" if L_out % 2 == 0 else "o"))
        stop = start + int(multiplicity) * (2 * int(L_out) + 1)
        slices[int(L_out)] = slice(start, stop)
        start = stop
    change_of_basis = np.stack(columns, axis=1) if columns else np.zeros((0, 0), dtype=np.complex128)
    raw_dim = 1
    for l_value in lin:
        raw_dim *= 2 * int(l_value) + 1
    return NaiveCoupledTreeReduction(
        lin=lin,
        tree_type=tree_type,
        raw_dim=int(raw_dim),
        coupled_dim=int(change_of_basis.shape[1]),
        irreps_out=tuple(irreps),
        change_of_basis=change_of_basis,
        column_slices_by_L=slices,
    )


def irrep_direct_sum_D_wig(
    irreps,
    rotation_matrix,
    *,
    n_samples = 25,
):
    """Return a block diagonal Wigner action for an irrep inventory."""
    blocks = []
    for term in parse_irreps(irreps):
        D_wig_L = wigner_D_wig_from_matrix(term.l, rotation_matrix, n_samples=n_samples)
        blocks.extend([D_wig_L] * int(term.mul))
    return block_diag_dense(blocks)


@lru_cache(maxsize=None)
def repeated_channel_reduction(l, rank, tree_type = "balanced"):
    """Build the exact symmetry-adapted basis for ``Sym^rank(V_l)``.

    Rows of ``change_of_basis`` are ordered by the raw tensor-product magnetic
    basis.  Columns are ordered by ye3t repeated-channel labels and then by
    magnetic component ``M`` inside each label.
    """
    l = int(l)
    rank = int(rank)
    tree_type = validate_tree_type(tree_type)
    if rank < 1:
        raise ValueError("rank must be positive.")
    engine = ExactProductExpansionEngine(tree_type=tree_type)
    nin = tuple(1 for _ in range(rank))
    lin = tuple(l for _ in range(rank))
    raw_ms = list(product(range(-l, l + 1), repeat=rank))
    columns = []
    labels_by_L = {}
    slices = {}
    irreps_out = []
    start = 0
    for L_out in engine.available_L_for_pattern(nin, lin):
        space = engine.feature_space(nin, lin, int(L_out))
        labels_by_L[int(L_out)] = tuple(space.labels)
        irreps_out.append(IrrepTerm(mul=int(space.dim), l=int(L_out), parity="e" if L_out % 2 == 0 else "o"))
        for label in space.labels:
            vectors_by_M = engine._m_vectors(label)
            for M_out in range(-int(L_out), int(L_out) + 1):
                vec = vectors_by_M[int(M_out)]
                columns.append(np.asarray([_complex_value(vec.get(ms, 0)) for ms in raw_ms], dtype=np.complex128))
        stop = start + int(space.dim) * (2 * int(L_out) + 1)
        slices[int(L_out)] = slice(start, stop)
        start = stop
    change_of_basis = np.stack(columns, axis=1) if columns else np.zeros((len(raw_ms), 0), dtype=np.complex128)
    change_of_basis = _normalize_basis_columns(change_of_basis)
    return RepeatedChannelReduction(
        l=l,
        rank=rank,
        tree_type=tree_type,
        raw_dim=(2 * l + 1) ** rank,
        symmetry_adapted_dim=int(change_of_basis.shape[1]),
        irreps_out=tuple(irreps_out),
        labels_by_L=labels_by_L,
        change_of_basis=change_of_basis,
        column_slices_by_L=slices,
    )


@lru_cache(maxsize=None)
def ace_channel_reduction(
    nin,
    lin,
    tree_type = "balanced",
):
    """Alias for :func:`ye3t_channel_reduction`."""
    return ye3t_channel_reduction(tuple(nin), tuple(lin), tree_type=str(tree_type))


@lru_cache(maxsize=None)
def ye3t_channel_reduction(
    nin,
    lin,
    tree_type = "balanced",
):
    """Build the ye3t symmetry-adapted basis for one leaf pattern."""
    tree_type = validate_tree_type(tree_type)
    engine = ExactProductExpansionEngine(tree_type=tree_type)
    target = engine.feature_space(tuple(nin), tuple(lin), 0)
    nin_c, lin_c = target.nin, target.lin
    raw_ms = list(product(*[range(-int(l), int(l) + 1) for l in lin_c]))
    columns = []
    labels_by_L = {}
    slices = {}
    irreps_out = []
    start = 0
    for L_out in engine.available_L_for_pattern(nin_c, lin_c):
        space = engine.feature_space(nin_c, lin_c, int(L_out))
        labels_by_L[int(L_out)] = tuple(space.labels)
        irreps_out.append(IrrepTerm(mul=int(space.dim), l=int(L_out), parity="e" if L_out % 2 == 0 else "o"))
        for label in space.labels:
            vectors_by_M = engine._m_vectors(label)
            for M_out in range(-int(L_out), int(L_out) + 1):
                vec = vectors_by_M[int(M_out)]
                columns.append(np.asarray([_complex_value(vec.get(ms, 0)) for ms in raw_ms], dtype=np.complex128))
        stop = start + int(space.dim) * (2 * int(L_out) + 1)
        slices[int(L_out)] = slice(start, stop)
        start = stop
    change_of_basis = np.stack(columns, axis=1) if columns else np.zeros((len(raw_ms), 0), dtype=np.complex128)
    change_of_basis = _normalize_basis_columns(change_of_basis)
    raw_dim = 1
    for l_value in lin_c:
        raw_dim *= 2 * int(l_value) + 1
    return YE3TChannelReduction(
        nin=tuple(nin_c),
        lin=tuple(lin_c),
        tree_type=tree_type,
        raw_dim=int(raw_dim),
        symmetry_adapted_dim=int(change_of_basis.shape[1]),
        irreps_out=tuple(irreps_out),
        labels_by_L=labels_by_L,
        change_of_basis=change_of_basis,
        column_slices_by_L=slices,
    )


@lru_cache(maxsize=None)
def ye3t_symbolic_summary(
    nin,
    lin,
    tree_type = "balanced",
):
    """Return ye3t labels and multiplicities without dense realization."""
    tree_type = validate_tree_type(tree_type)
    engine = ExactProductExpansionEngine(tree_type=tree_type)
    target = engine.feature_space(tuple(nin), tuple(lin), 0)
    nin_c, lin_c = target.nin, target.lin
    labels_by_L = {}
    slices = {}
    irreps_out = []
    start = 0
    label_repr_bytes = 0
    for L_out in engine.available_L_for_pattern(nin_c, lin_c):
        space = engine.feature_space(nin_c, lin_c, int(L_out))
        labels = tuple(space.labels)
        labels_by_L[int(L_out)] = labels
        label_repr_bytes += len(repr(labels).encode("utf-8"))
        irreps_out.append(IrrepTerm(mul=int(space.dim), l=int(L_out), parity="e" if L_out % 2 == 0 else "o"))
        stop = start + int(space.dim) * (2 * int(L_out) + 1)
        slices[int(L_out)] = slice(start, stop)
        start = stop
    raw_dim = 1
    for l_value in lin_c:
        raw_dim *= 2 * int(l_value) + 1
    return YE3TSymbolicSummary(
        nin=tuple(nin_c),
        lin=tuple(lin_c),
        tree_type=tree_type,
        raw_dim=int(raw_dim),
        symmetry_adapted_dim=int(start),
        irreps_out=tuple(irreps_out),
        labels_by_L=labels_by_L,
        column_slices_by_L=slices,
        label_repr_bytes=int(label_repr_bytes),
    )


def primitive_quotient_dimension_summary(
    nin,
    lin,
    L_R,
    *,
    tree_type = "balanced",
    factorization_policy = "full",
):
    """Return exact ``F/D/P`` dimensions for one ACE primitive quotient."""
    tree_type = validate_tree_type(tree_type)
    engine = ExactProductExpansionEngine(tree_type=tree_type)
    summary = engine.primitive_quotient(tuple(nin), tuple(lin), int(L_R), mode=str(factorization_policy))
    return PrimitiveQuotientDimensionSummary(
        nin=tuple(summary.target_space.nin),
        lin=tuple(summary.target_space.lin),
        L_R=int(summary.target_space.L_R),
        factorization_policy=str(summary.factorization_policy) if hasattr(summary, "factorization_policy") else str(factorization_policy),
        full_dim=int(summary.target_space.dim),
        decomposable_dim=int(summary.generated_rank),
        primitive_quotient_dim=int(summary.primitive_rank),
    )


def block_diagonalize_repeated_channel_D_wig(
    l,
    rank,
    rotation_matrix,
    *,
    tree_type = "balanced",
    n_samples = 25,
):
    """Reduce the repeated-channel tensor rotation to symmetry-adapted blocks."""
    reduction = repeated_channel_reduction(int(l), int(rank), tree_type=str(tree_type))
    D_wig_leaf = wigner_D_wig_from_matrix(int(l), rotation_matrix, n_samples=n_samples)
    return block_diagonalize_repeated_channel_from_leaf_D_wig(
        D_wig_leaf,
        reduction=reduction,
        rotation_matrix=rotation_matrix,
    )


def block_diagonalize_repeated_channel_from_leaf_D_wig(
    D_wig_leaf,
    *,
    reduction,
    rotation_matrix = None,
):
    """Reduce a repeated-channel tensor action from a precomputed leaf irrep matrix."""
    D_wig_leaf = np.asarray(D_wig_leaf)
    raw_action = D_wig_leaf
    for _ in range(int(reduction.rank) - 1):
        raw_action = np.kron(raw_action, D_wig_leaf)
    Q = reduction.change_of_basis
    D_wig_reduced = Q.conj().T @ raw_action @ Q
    mask = np.zeros_like(D_wig_reduced, dtype=bool)
    for slc in reduction.column_slices_by_L.values():
        mask[slc, slc] = True
    off_block = D_wig_reduced.copy()
    off_block[mask] = 0
    zero_fraction = 1.0 - float(np.count_nonzero(np.abs(D_wig_reduced) > 1e-12)) / float(D_wig_reduced.size)
    return RepeatedChannelBlockDiagonalization(
        reduction=reduction,
        rotation_matrix=np.eye(3) if rotation_matrix is None else np.asarray(rotation_matrix, dtype=float),
        D_wig_reduced=D_wig_reduced,
        off_block_max_abs=float(np.max(np.abs(off_block))) if off_block.size else 0.0,
        zero_fraction=float(zero_fraction),
    )


def block_diagonalize_ace_channel_from_leaf_D_wigs(
    D_wig_by_l,
    *,
    reduction,
    rotation_matrix = None,
):
    """Alias for :func:`block_diagonalize_ye3t_from_leaf_D_wigs`."""
    return block_diagonalize_ye3t_from_leaf_D_wigs(
        D_wig_by_l,
        reduction=reduction,
        rotation_matrix=rotation_matrix,
    )


def block_diagonalize_ye3t_from_leaf_D_wigs(
    D_wig_by_l,
    *,
    reduction,
    rotation_matrix = None,
):
    """Reduce a mixed ye3t tensor action from leaf irrep matrices."""
    raw_action = np.asarray(D_wig_by_l[int(reduction.lin[0])])
    for l_value in reduction.lin[1:]:
        raw_action = np.kron(raw_action, np.asarray(D_wig_by_l[int(l_value)]))
    Q = reduction.change_of_basis
    D_wig_reduced = Q.conj().T @ raw_action @ Q
    mask = np.zeros_like(D_wig_reduced, dtype=bool)
    for slc in reduction.column_slices_by_L.values():
        mask[slc, slc] = True
    off_block = D_wig_reduced.copy()
    off_block[mask] = 0
    zero_fraction = 1.0 - float(np.count_nonzero(np.abs(D_wig_reduced) > 1e-12)) / float(D_wig_reduced.size)
    return YE3TBlockDiagonalization(
        reduction=reduction,
        rotation_matrix=np.eye(3) if rotation_matrix is None else np.asarray(rotation_matrix, dtype=float),
        D_wig_reduced=D_wig_reduced,
        off_block_max_abs=float(np.max(np.abs(off_block))) if off_block.size else 0.0,
        zero_fraction=float(zero_fraction),
    )


def block_diagonalize_ace_channel_D_wig(
    nin,
    lin,
    rotation_matrix,
    *,
    tree_type = "balanced",
    n_samples = 25,
):
    """Alias for :func:`block_diagonalize_ye3t_D_wig`."""
    return block_diagonalize_ye3t_D_wig(
        nin,
        lin,
        rotation_matrix,
        tree_type=tree_type,
        n_samples=n_samples,
    )


def block_diagonalize_naive_coupled_tree_from_leaf_D_wigs(
    D_wig_by_l,
    *,
    reduction,
    rotation_matrix = None,
):
    """Reduce a raw tensor action to a naive sequential CG coupled-tree basis."""
    raw_action = np.asarray(D_wig_by_l[int(reduction.lin[0])])
    for l_value in reduction.lin[1:]:
        raw_action = np.kron(raw_action, np.asarray(D_wig_by_l[int(l_value)]))
    Q = reduction.change_of_basis
    D_wig_reduced = Q.conj().T @ raw_action @ Q
    mask = np.zeros_like(D_wig_reduced, dtype=bool)
    for slc in reduction.column_slices_by_L.values():
        mask[slc, slc] = True
    off_block = D_wig_reduced.copy()
    off_block[mask] = 0
    zero_fraction = 1.0 - float(np.count_nonzero(np.abs(D_wig_reduced) > 1e-12)) / float(D_wig_reduced.size)
    return NaiveCoupledTreeBlockDiagonalization(
        reduction=reduction,
        rotation_matrix=np.eye(3) if rotation_matrix is None else np.asarray(rotation_matrix, dtype=float),
        D_wig_reduced=D_wig_reduced,
        off_block_max_abs=float(np.max(np.abs(off_block))) if off_block.size else 0.0,
        zero_fraction=float(zero_fraction),
    )


def block_diagonalize_naive_coupled_tree_D_wig(
    lin,
    rotation_matrix,
    *,
    tree_type = "balanced",
    n_samples = 25,
):
    """Reduce a raw tensor rotation to a naive sequential CG coupled-tree basis."""
    reduction = naive_coupled_tree_reduction(tuple(int(l_value) for l_value in lin), tree_type=str(tree_type))
    D_wig_by_l = {
        int(l_value): wigner_D_wig_from_matrix(int(l_value), rotation_matrix, n_samples=n_samples)
        for l_value in sorted(set(reduction.lin))
    }
    return block_diagonalize_naive_coupled_tree_from_leaf_D_wigs(
        D_wig_by_l,
        reduction=reduction,
        rotation_matrix=rotation_matrix,
    )


def block_diagonalize_ye3t_D_wig(
    nin,
    lin,
    rotation_matrix,
    *,
    tree_type = "balanced",
    n_samples = 25,
):
    """Reduce a mixed ye3t leaf-pattern tensor rotation to ye3t blocks."""
    reduction = ye3t_channel_reduction(tuple(int(x) for x in nin), tuple(int(x) for x in lin), tree_type=str(tree_type))
    D_wig_by_l = {
        int(l_value): wigner_D_wig_from_matrix(int(l_value), rotation_matrix, n_samples=n_samples)
        for l_value in sorted(set(reduction.lin))
    }
    return block_diagonalize_ye3t_from_leaf_D_wigs(
        D_wig_by_l,
        reduction=reduction,
        rotation_matrix=rotation_matrix,
    )


ACEChannelReduction = YE3TChannelReduction
ACEChannelBlockDiagonalization = YE3TBlockDiagonalization


def torch_cg_tensor_product(
    left,
    right,
    L_left,
    L_right,
    L_out,
):
    """Couple two complex multiplets with a CG tensor.

    ``left`` and ``right`` are expected to have final dimensions ``2L+1`` in
    magnetic order ``m=-L,...,L``.  Leading batch dimensions are broadcast by
    PyTorch.
    """
    tensor = torch.zeros(
        (2 * int(L_left) + 1, 2 * int(L_right) + 1, 2 * int(L_out) + 1),
        dtype=torch.complex128,
        device=left.device,
    )
    for i, m1 in enumerate(range(-int(L_left), int(L_left) + 1)):
        for j, m2 in enumerate(range(-int(L_right), int(L_right) + 1)):
            M = m1 + m2
            if abs(M) <= int(L_out):
                tensor[i, j, M + int(L_out)] = complex(clebsch_gordan(int(L_left), m1, int(L_right), m2, int(L_out), M))
    if left.dtype == torch.complex64 or right.dtype == torch.complex64:
        target_dtype = torch.complex64
    else:
        target_dtype = torch.complex128
    tensor = tensor.to(dtype=target_dtype)
    left = left.to(dtype=target_dtype)
    right = right.to(dtype=target_dtype)
    return torch.einsum("...m,...n,mnr->...r", left, right, tensor)


__all__ = [
    "IrrepTerm",
    "YE3TBlockDiagonalization",
    "YE3TChannelReduction",
    "YE3TSymbolicSummary",
    "NaiveCoupledTreeBlockDiagonalization",
    "NaiveCoupledTreeReduction",
    "PrimitiveQuotientDimensionSummary",
    "TensorProductReduction",
    "RepeatedChannelReduction",
    "RepeatedChannelBlockDiagonalization",
    "block_diag_dense",
    "ye3t_channel_reduction",
    "ye3t_symbolic_summary",
    "block_diagonalize_ye3t_D_wig",
    "block_diagonalize_ye3t_from_leaf_D_wigs",
    "block_diagonalize_naive_coupled_tree_D_wig",
    "block_diagonalize_naive_coupled_tree_from_leaf_D_wigs",
    "block_diagonalize_repeated_channel_D_wig",
    "block_diagonalize_repeated_channel_from_leaf_D_wig",
    "cg_tensor_product_reduction",
    "format_irreps",
    "irrep_direct_sum_D_wig",
    "naive_coupled_tree_reduction",
    "parse_irreps",
    "primitive_quotient_dimension_summary",
    "reduce_tensor_product_D_wig",
    "repeated_channel_reduction",
    "tensor_product_output_irreps",
    "torch_cg_tensor_product",
    "total_dim",
    "wigner_D_wig_from_matrix",
    "wigner_D_wig_z",
    "ACEChannelBlockDiagonalization",
    "ACEChannelReduction",
    "ace_channel_reduction",
    "block_diagonalize_ace_channel_D_wig",
    "block_diagonalize_ace_channel_from_leaf_D_wigs",
]
