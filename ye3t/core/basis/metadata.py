
r"""Rich exact-label metadata for the constructive Schur--Weyl + Young basis.

The original compact label

.. math::

    (\eta_1,\ldots,\eta_N; l_1,\ldots,l_N; L_{\mathrm{int}})

is intentionally minimal.  It is sufficient to *name* a canonical exact basis
vector, but it does not carry enough structured information to support exact
product-expansion, recoupling, and primitive-quotient computations without some
reconstruction work.

This module adds a rigorous metadata layer while leaving the compact label
untouched.  The metadata records:

* the full leaf/channel multiset in canonical order,
* the blockwise Schur--Weyl / Young multiplicity labels,
* the canonical block-coupling tree topology,
* every internal angular label on that tree,
* a canonical phase / tensor-basis reconstruction recipe,
* and an optional cached tensor-basis expansion.

The tensor-basis cache is expressed in the uncoupled spherical-tensor basis,
not in a dipole-specific Cartesian basis.  It therefore applies equally well to
higher moments such as quadrupoles and octupoles, and in fact to any target
angular momentum supported by the exact basis.
"""

from dataclasses import field
from ye3t._record import recordclass


@recordclass(('eta', 'l', 'block_size', 'Lambda', 'multiplicity_index', 'representative_internal_Ls', 'basis_key'), frozen = True)
class BlockMultiplicityMetadata:
    r"""Metadata for one repeated-channel block ``(\eta,l)_b^{k_b}``.

    Parameters
    ----------
    eta
        Channel index ``\eta`` used in the paper's block notation.  In the
        current runtime implementation this coincides with the stored radial /
        channel index carried by the compact exact label.
    l
        Angular momentum label of the block channel.
    block_size
        Multiplicity ``k_b`` of the repeated block.
    Lambda
        Final block angular momentum ``\Lambda_b``.
    multiplicity_index
        Exact multiplicity-space index inside the Schur--Weyl / Young-reduced
        block sector.  This is *not* the chemical/channel index and therefore
        deliberately avoids the overloaded symbol ``mu``.
    representative_internal_Ls
        Canonical internal angular labels for the representative homogeneous
        coupling used to realize this block basis vector.
    basis_key
        Additional exact basis identifier used when the block basis is named by
        a tree-independent symmetric-power construction rather than by a single
        coupled-tree tuple alone.
    """
    representative_internal_Ls = tuple()
    basis_key = tuple()


@recordclass(('postorder_index', 'coupled_L', 'left_leaf_span', 'right_leaf_span'), frozen = True)
class InternalAngularNodeMetadata:
    r"""One internal node of the canonical coupling tree.

    The exact basis is tree agnostic at the representation-theory level, but it
    is *represented* in a chosen canonical tree coordinate system.  Each node
    stores the precise coupled angular momentum label used at that point of the
    recursion.

    Parameters
    ----------
    postorder_index
        Position of the internal node in postorder traversal.  This matches the
        ordering convention used for ``internal_Ls`` in the compact exact label.
    coupled_L
        Angular momentum attached to this internal node.
    left_leaf_span, right_leaf_span
        Leaf-index spans identifying which canonical leaves are coupled on the
        left and right children at this node.
    """


@recordclass(('component_index', 'magnetic_tuple', 'coefficient'), frozen = True)
class TensorBasisTerm:
    r"""One coefficient in an uncoupled spherical-tensor basis expansion.

    Parameters
    ----------
    component_index
        Tensor component index of the output irrep.  For SO(3) irreps this is
        the magnetic quantum number ``M``.
    magnetic_tuple
        Tuple ``(m_1,\ldots,m_N)`` naming one uncoupled tensor-product basis
        vector of the leaf irreps.
    coefficient
        Exact coefficient multiplying that uncoupled basis vector.  The runtime
        algebra engine may store a SymPy expression here; callers should treat
        it as an exact symbolic scalar.
    """


@recordclass(('basis_kind', 'coefficients_by_component'), frozen = True)
class TensorBasisExpansionCache:
    r"""Optional cached expansion in the uncoupled spherical-tensor basis.

    Notes
    -----
    This cache is intentionally tensor-general.  It is not restricted to
    dipoles and therefore supports higher tensor moments such as quadrupoles
    (``L=2``) and octupoles (``L=3``), as well as arbitrary target irreps used
    in the ACE covariant basis.
    """

    basis_kind = "uncoupled_spherical_tensor_m_basis"
    coefficients_by_component = field(default_factory=dict)


@recordclass(('tree_type', 'tree_signature', 'internal_Ls_postorder', 'phase_convention', 'coupling_recipe', 'tensor_basis_kind'), frozen = True)
class ReconstructionRecipe:
    r"""Canonical recipe for reconstructing the exact basis vector.

    The tree topology alone does not fully determine a coupled basis state: one
    also needs the internal node labels and a phase convention.  This metadata
    records the exact reconstruction prescription used throughout the package.
    """
    phase_convention = "Condon-Shortley via exact SymPy Clebsch-Gordan coefficients"
    coupling_recipe = "postorder recursive CG coupling on the canonical tree"
    tensor_basis_kind = "uncoupled spherical-tensor m-basis"


@recordclass(('compact_label', 'eta_tuple', 'l_tuple', 'root_L', 'tree_type', 'canonical_tree_signature', 'young_block_multiplicities', 'internal_nodes', 'reconstruction', 'tensor_basis_expansion'), frozen = True)
class ExactLabelMetadata:
    r"""Full metadata for one canonical exact ACE basis vector.

    This is the rich companion of the compact exact label.  It records the
    Schur--Weyl + Young block structure and the chosen coupling coordinates
    needed for exact graded algebra/module computations.
    """
    tensor_basis_expansion = None
