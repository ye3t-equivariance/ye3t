
from ye3t._record import recordclass


@recordclass(('idx', 'n', 'l'), frozen = True)
class RawLeaf:

    def signature(self):
        return ("leaf", self.n, self.l)

    def n_multiset(self):
        return (self.n,)

    def l_multiset(self):
        return (self.l,)


@recordclass(('left', 'right'), frozen = True)
class PairNode:

    def signature(self):
        return ("node", self.left.signature(), self.right.signature())

    def n_multiset(self):
        return tuple(sorted(self.left.n_multiset() + self.right.n_multiset()))

    def l_multiset(self):
        return tuple(sorted(self.left.l_multiset() + self.right.l_multiset()))


@recordclass(('size', 'left', 'right'), frozen = True)
class HomShape:
    left = None
    right = None

    @property
    def is_leaf(self):
        return self.left is None and self.right is None

    def signature(self):
        if self.is_leaf:
            return ("leaf", 1)
        return ("node", self.left.signature(), self.right.signature())


class BinaryTreeFactory:
    """Base helper for recursive full binary trees used in exact ACE."""

    tree_type = "balanced"

    @classmethod
    def _combine_raw_level(cls, level):
        raise NotImplementedError

    @classmethod
    def _combine_shape_level(cls, level):
        raise NotImplementedError

    @classmethod
    def build_raw_tree(cls, nin, lin):
        if len(nin) != len(lin):
            raise ValueError("nin and lin must have the same length.")
        if not nin:
            raise ValueError("nin and lin must be non-empty.")
        level = [RawLeaf(i, n, l) for i, (n, l) in enumerate(zip(nin, lin))]
        while len(level) > 1:
            level = cls._combine_raw_level(level)
        return level[0]

    @classmethod
    def build_hom_shape(cls, k_b):
        if k_b < 1:
            raise ValueError("k_b must be >= 1")
        level = [HomShape(size=1) for _ in range(k_b)]
        while len(level) > 1:
            level = cls._combine_shape_level(level)
        return level[0]

    @staticmethod
    def subtree_leaves(node):
        if isinstance(node, RawLeaf):
            return [(node.n, node.l)]
        return BinaryTreeFactory.subtree_leaves(node.left) + BinaryTreeFactory.subtree_leaves(node.right)

    @staticmethod
    def homogeneous_subtree_info(node):
        leaves = BinaryTreeFactory.subtree_leaves(node)
        if len(leaves) < 2:
            return False, None, None, None
        first = leaves[0]
        if all(x == first for x in leaves):
            n, l = first
            return True, n, l, len(leaves)
        return False, None, None, None

    @staticmethod
    def reconstruct_raw_from_signature(sig):
        tag = sig[0]
        if tag == "leaf":
            _, n, l = sig
            return RawLeaf(-1, n, l)
        if tag == "node":
            _, left_sig, right_sig = sig
            return PairNode(
                BinaryTreeFactory.reconstruct_raw_from_signature(left_sig),
                BinaryTreeFactory.reconstruct_raw_from_signature(right_sig),
            )
        raise ValueError(f"Unknown raw signature tag: {tag}")

    @staticmethod
    def reconstruct_hom_from_signature(sig):
        tag = sig[0]
        if tag == "leaf":
            return HomShape(size=1)
        if tag == "node":
            _, left_sig, right_sig = sig
            left = BinaryTreeFactory.reconstruct_hom_from_signature(left_sig)
            right = BinaryTreeFactory.reconstruct_hom_from_signature(right_sig)
            return HomShape(size=left.size + right.size, left=left, right=right)
        raise ValueError(f"Unknown homogeneous signature tag: {tag}")


class BalancedPairwiseTreeFactory(BinaryTreeFactory):
    """Factories and helpers for recursive balanced pairwise trees."""

    tree_type = "balanced"

    @classmethod
    def _combine_raw_level(cls, level):
        nxt = []
        i = 0
        while i < len(level):
            if i + 1 < len(level):
                nxt.append(PairNode(level[i], level[i + 1]))
                i += 2
            else:
                nxt.append(level[i])
                i += 1
        return nxt

    @classmethod
    def _combine_shape_level(cls, level):
        nxt = []
        i = 0
        while i < len(level):
            if i + 1 < len(level):
                left = level[i]
                right = level[i + 1]
                nxt.append(HomShape(size=left.size + right.size, left=left, right=right))
                i += 2
            else:
                nxt.append(level[i])
                i += 1
        return nxt


class LeftJustifiedTreeFactory(BinaryTreeFactory):
    """Factories and helpers for recursive left-justified full binary trees."""

    tree_type = "left"

    @classmethod
    def _combine_raw_level(cls, level):
        if len(level) < 2:
            return level
        return [PairNode(level[0], level[1])] + level[2:]

    @classmethod
    def _combine_shape_level(cls, level):
        if len(level) < 2:
            return level
        left = level[0]
        right = level[1]
        return [HomShape(size=left.size + right.size, left=left, right=right)] + level[2:]


def get_tree_factory(tree_type):
    normalized = "balanced" if tree_type is None else str(tree_type).strip().lower()
    if normalized in {"balanced", "balanced_pairwise", "pairwise"}:
        return BalancedPairwiseTreeFactory
    if normalized in {"left", "left_justified", "left-justified"}:
        return LeftJustifiedTreeFactory
    raise ValueError(f"Unknown tree_type '{tree_type}'. Use 'balanced' or 'left'.")
