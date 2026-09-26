
"""Naive Gram/SVD comparison helpers.

Nothing in this module is used to construct the runtime algebraic exact ACE
basis. These classes/functions exist only to compare the exact block-first
Schur-Weyl + Young-subgroup basis against an overcomplete candidate list whose
rank is recovered numerically from a Gram/SVD calculation.
"""

import itertools
import time
from functools import lru_cache

import numpy as np

from .characters import cg_allowed
from .tree import RawLeaf, PairNode, get_tree_factory
from ye3t.core.cg import cg_numeric
from ye3t._record import recordclass


@lru_cache(maxsize=None)
def cg_coeff(j1, m1, j2, m2, J, M):
    """Cached Clebsch-Gordan coefficient as a float."""
    if m1 + m2 != M:
        return 0.0
    return float(cg_numeric(j1, m1, j2, m2, J, M).real)


@recordclass(('n', 'l'), frozen = True)
class CoupledLeaf:

    @property
    def J(self):
        return self.l

    @property
    def num_leaves(self):
        return 1


@recordclass(('left', 'right', 'J'), frozen = True)
class CoupledNode:

    @property
    def num_leaves(self):
        return self.left.num_leaves + self.right.num_leaves


class NaiveBalancedPairwiseGenerator:
    """
    Generate an overcomplete candidate set for validation/testing.

    This is a comparison tool only. It is not part of the constructive exact
    Schur-Weyl + Young-subgroup basis generation path.
    """

    def __init__(self, nin, lin, tree_type = "balanced"):
        self.nin = tuple(nin)
        self.lin = tuple(lin)
        self.tree_type = tree_type
        self.tree_factory = get_tree_factory(tree_type)
        self.root = self.tree_factory.build_raw_tree(list(nin), list(lin))

    @lru_cache(maxsize=None)
    def _candidates_by_L(self, sig):
        node = self.tree_factory.reconstruct_raw_from_signature(sig)
        if isinstance(node, RawLeaf):
            return {node.l: (tuple(),)}

        left_map = self._candidates_by_L(node.left.signature())
        right_map = self._candidates_by_L(node.right.signature())
        out = {}
        for L1, left_tups in left_map.items():
            for L2, right_tups in right_map.items():
                allowed = cg_allowed(L1, L2)
                for t1 in left_tups:
                    for t2 in right_tups:
                        for Lout in allowed:
                            out.setdefault(Lout, []).append(t1 + t2 + (Lout,))
        return {L: tuple(sorted(set(v))) for L, v in out.items()}

    def candidates_for_target(self, L_target):
        return list(self._candidates_by_L(self.root.signature()).get(L_target, tuple()))

    def count_candidates_for_target(self, L_target):
        return len(self.candidates_for_target(L_target))


class CoupledTensorBuilder:
    """
    Build uncoupled-basis tensors for naive validation candidates.

    Notes
    -----
    This class is used only in Gram/SVD comparisons. It is not used by the
    runtime exact ACE implementation.

    This is intended for modest body order / angular momentum. The uncoupled basis
    dimension grows as prod_i (2 l_i + 1), so this becomes expensive quickly.
    """

    def __init__(self, nin, lin, tree_type = "balanced"):
        if len(nin) != len(lin):
            raise ValueError("nin and lin must have the same length.")
        self.nin = tuple(nin)
        self.lin = tuple(lin)
        self.tree_type = tree_type
        self.tree_factory = get_tree_factory(tree_type)
        self.root = self.tree_factory.build_raw_tree(list(nin), list(lin))
        self.dims = tuple(2 * l + 1 for l in self.lin)
        self.identical_groups = self._identical_index_groups()

    def _identical_index_groups(self):
        groups = {}
        for i, pair in enumerate(zip(self.nin, self.lin)):
            groups.setdefault(pair, []).append(i)
        return [tuple(v) for _, v in sorted(groups.items()) if len(v) >= 2]

    def _build_coupled_tree(self, node, internal_Ls, pos = 0):
        if isinstance(node, RawLeaf):
            return CoupledLeaf(n=node.n, l=node.l), pos

        left_c, pos = self._build_coupled_tree(node.left, internal_Ls, pos)
        right_c, pos = self._build_coupled_tree(node.right, internal_Ls, pos)
        if pos >= len(internal_Ls):
            raise ValueError("internal_Ls tuple is too short for the tree.")
        J = internal_Ls[pos]
        pos += 1
        return CoupledNode(left=left_c, right=right_c, J=J), pos

    def coupled_tree_from_tuple(self, internal_Ls):
        tree, pos = self._build_coupled_tree(self.root, tuple(internal_Ls), 0)
        if pos != len(internal_Ls):
            raise ValueError("internal_Ls tuple is too long for the tree.")
        return tree

    @lru_cache(maxsize=None)
    def _m_tensors(self, key):
        kind = key[0]
        if kind == "leaf":
            _, n, l = key
            out = {}
            for m in range(-l, l + 1):
                vec = np.zeros((2 * l + 1,), dtype=np.float64)
                vec[m + l] = 1.0
                out[m] = vec
            return out

        if kind == "node":
            _, left_key, right_key, J = key
            left_map = self._m_tensors(left_key)
            right_map = self._m_tensors(right_key)
            out = {}
            for M in range(-J, J + 1):
                acc = None
                for M1, left_tensor in left_map.items():
                    M2 = M - M1
                    if M2 not in right_map:
                        continue
                    coeff = cg_coeff(self._J_from_key(left_key), M1, self._J_from_key(right_key), M2, J, M)
                    if abs(coeff) < 1e-14:
                        continue
                    term = coeff * np.tensordot(left_tensor, right_map[M2], axes=0)
                    acc = term if acc is None else acc + term
                if acc is not None:
                    out[M] = acc
            return out

        raise ValueError(f"Unknown coupled-tree key kind: {kind}")

    def _key_from_coupled_tree(self, node):
        if isinstance(node, CoupledLeaf):
            return ("leaf", node.n, node.l)
        return ("node", self._key_from_coupled_tree(node.left), self._key_from_coupled_tree(node.right), node.J)

    def _J_from_key(self, key):
        if key[0] == "leaf":
            return key[2]
        return key[3]

    def candidate_tensor(self, internal_Ls, M_target = 0):
        coupled = self.coupled_tree_from_tuple(internal_Ls)
        key = self._key_from_coupled_tree(coupled)
        m_map = self._m_tensors(key)
        if M_target not in m_map:
            raise ValueError(f"Candidate tuple {tuple(internal_Ls)} does not support M_target={M_target}.")
        return np.array(m_map[M_target], dtype=np.float64, copy=True)

    def symmetrize_tensor(self, tensor):
        """
        Average a coefficient tensor over permutations of axes corresponding to
        identical (n,l) channels.
        """
        out = np.array(tensor, dtype=np.float64, copy=True)
        for group in self.identical_groups:
            perms = list(itertools.permutations(group))
            acc = np.zeros_like(out)
            group = tuple(group)
            for perm in perms:
                axes = list(range(out.ndim))
                for src, dst in zip(group, perm):
                    axes[src] = dst
                acc += np.transpose(out, axes=axes)
            out = acc / float(len(perms))
        return out

    def candidate_vector(self, internal_Ls, M_target = 0, symmetrize = True):
        tensor = self.candidate_tensor(internal_Ls, M_target=M_target)
        if symmetrize:
            tensor = self.symmetrize_tensor(tensor)
        return tensor.reshape(-1)


@recordclass(('nin', 'lin', 'L_target', 'M_target', 'candidate_count', 'svd_rank', 'singular_values', 'generation_time_s', 'vector_build_time_s', 'svd_time_s', 'total_time_s'), frozen = True)
class GramianAnalysisResult:
    pass


def gramian_svd_rank(
    nin,
    lin,
    L_target,
    M_target = 0,
    svd_tol = 1e-10,
    tree_type = "balanced",
):
    """
    Validation-only naive Dusson-like workflow:
      1. generate an overcomplete balanced-pairwise candidate set,
      2. build the symmetrized coefficient vectors in the uncoupled m-basis,
      3. compute the Gram matrix / SVD rank.

    Returns candidate count, SVD rank, singular values, and timings.

    This numerical route is not used by the constructive exact basis builder.
    """
    t0 = time.perf_counter()
    gen = NaiveBalancedPairwiseGenerator(nin, lin, tree_type=tree_type)
    candidates = gen.candidates_for_target(L_target)
    t1 = time.perf_counter()

    tensor_builder = CoupledTensorBuilder(nin, lin, tree_type=tree_type)
    if candidates:
        mat = np.vstack([
            tensor_builder.candidate_vector(cand, M_target=M_target, symmetrize=True)
            for cand in candidates
        ])
    else:
        mat = np.zeros((0, int(np.prod([2 * l + 1 for l in lin]))), dtype=np.float64)
    t2 = time.perf_counter()

    if mat.size == 0:
        svals = np.array([], dtype=np.float64)
        rank = 0
    else:
        # SVD of the candidate matrix is numerically more stable than on the Gram matrix.
        svals = np.linalg.svd(mat, compute_uv=False)
        rank = int(np.sum(svals > svd_tol))
    t3 = time.perf_counter()

    return GramianAnalysisResult(
        nin=tuple(nin),
        lin=tuple(lin),
        L_target=L_target,
        M_target=M_target,
        candidate_count=len(candidates),
        svd_rank=rank,
        singular_values=tuple(float(x) for x in svals),
        generation_time_s=t1 - t0,
        vector_build_time_s=t2 - t1,
        svd_time_s=t3 - t2,
        total_time_s=t3 - t0,
    )
