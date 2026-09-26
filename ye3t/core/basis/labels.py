
from ye3t._record import recordclass


@recordclass(('n', 'l', 'tree_type'), frozen = True)
class LeafLabel:
    tree_type = "balanced"

    def total_L(self):
        return self.l

    def n_sequence(self):
        return (self.n,)

    def l_sequence(self):
        return (self.l,)

    def internal_Ls_postorder(self):
        return tuple()

    def compact_label(self):
        if self.tree_type == "balanced":
            return (self.n_sequence(), self.l_sequence(), self.internal_Ls_postorder())
        return (self.n_sequence(), self.l_sequence(), self.internal_Ls_postorder(), self.tree_type)

    def compact_basis_key(self):
        return tuple()

    def sort_key(self):
        return ("leaf", self.tree_type, self.n, self.l)

    def pretty(self):
        return f"Leaf(n={self.n}, l={self.l})"


@recordclass(('left', 'right', 'L', 'tree_type'), frozen = True)
class NodeLabel:
    tree_type = "balanced"

    def total_L(self):
        return self.L

    def n_sequence(self):
        return self.left.n_sequence() + self.right.n_sequence()

    def l_sequence(self):
        return self.left.l_sequence() + self.right.l_sequence()

    def internal_Ls_postorder(self):
        return self.left.internal_Ls_postorder() + self.right.internal_Ls_postorder() + (self.L,)

    def compact_label(self):
        basis_key = self.compact_basis_key()
        if not basis_key and self.tree_type == "balanced":
            return (self.n_sequence(), self.l_sequence(), self.internal_Ls_postorder())
        if not basis_key:
            return (self.n_sequence(), self.l_sequence(), self.internal_Ls_postorder(), self.tree_type)
        return (self.n_sequence(), self.l_sequence(), self.internal_Ls_postorder(), self.tree_type, basis_key)

    def compact_basis_key(self):
        left_key = self.left.compact_basis_key()
        right_key = self.right.compact_basis_key()
        if not left_key and not right_key:
            return tuple()
        return ("node", left_key, right_key)

    def sort_key(self):
        return ("node", self.tree_type, self.L, self.left.sort_key(), self.right.sort_key())

    def pretty(self, depth = 0):
        pad = "  " * depth
        out = [f"{pad}Node(L={self.L})"]
        if isinstance(self.left, NodeLabel):
            out.append(self.left.pretty(depth + 1))
        else:
            out.append("  " * (depth + 1) + self.left.pretty())
        if isinstance(self.right, NodeLabel):
            out.append(self.right.pretty(depth + 1))
        else:
            out.append("  " * (depth + 1) + self.right.pretty())
        return "\n".join(out)


@recordclass(('n', 'l', 'k_b', 'Lambda', 'multiplicity_index', 'representative_internal_Ls', 'tree_type', 'basis_key', 'occupancy_expansion_by_M'), frozen = True)
class SymBlockLabel:
    """Collapsed label for a fully homogeneous subtree ``Sym^{k_b}(V_l)``.

    The field ``multiplicity_index`` labels basis vectors inside the block's
    multiplicity space. It is *not* the ACE chemical index ``mu`` used later in
    descriptor/channel notation.

    When the same final block angular momentum ``Lambda`` appears more than once in
    the exact Schur-Weyl / Young-subgroup decomposition of ``Sym^{k_b}(V_l)``,
    ``multiplicity_index = 0, 1, ...`` selects one canonical basis vector in
    that block multiplicity space.
    """
    representative_internal_Ls = tuple()
    tree_type = "balanced"
    basis_key = tuple()
    occupancy_expansion_by_M = tuple()

    def total_L(self):
        return self.Lambda

    def n_sequence(self):
        return tuple([self.n] * self.k_b)

    def l_sequence(self):
        return tuple([self.l] * self.k_b)

    def internal_Ls_postorder(self):
        if self.representative_internal_Ls:
            return self.representative_internal_Ls
        return (int(self.Lambda),)

    def compact_label(self):
        basis_key = self.compact_basis_key()
        if not basis_key and self.tree_type == "balanced":
            return (self.n_sequence(), self.l_sequence(), self.internal_Ls_postorder())
        if not basis_key:
            return (self.n_sequence(), self.l_sequence(), self.internal_Ls_postorder(), self.tree_type)
        return (self.n_sequence(), self.l_sequence(), self.internal_Ls_postorder(), self.tree_type, basis_key)

    def compact_basis_key(self):
        if self.basis_key:
            return tuple(self.basis_key)
        if self.multiplicity_index == 0:
            return tuple()
        return ("sym", int(self.Lambda), int(self.multiplicity_index))

    def sort_key(self):
        return (
            "symblock",
            self.tree_type,
            self.n,
            self.l,
            self.k_b,
            self.Lambda,
            self.multiplicity_index,
            self.representative_internal_Ls,
            self.compact_basis_key(),
        )

    def pretty(self):
        return (
            f"SymBlock(n={self.n}, l={self.l}, k_b={self.k_b}, "
            f"Lambda={self.Lambda}, multiplicity_index={self.multiplicity_index}, "
            f"repLs={self.representative_internal_Ls})"
        )
