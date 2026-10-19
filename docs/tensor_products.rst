General YE3T Tensor Products
============================

Complete carrier products
-------------------------

``couplings.carrier_product_request`` selects a product of complete Young and
O(3) representations. Use ``count`` for multiplicities, ``plan`` for its fixed
operation, and ``compile`` for the coefficients. Bind the result with
``couplings.bind_carrier_product_torch`` and supply arrays with final axes
``(channel, t, M)``. All output copies are returned. Young and angular maps
are stored separately, and evaluation supports first and second derivatives.

.. list-table:: Product choices
   :header-rows: 1
   :widths: 28 36 36

   * - Operation
     - Permutation coupling
     - Input requirement
   * - ``same_action_tensor_product``
     - Kronecker product under the diagonal :math:`S_N` action
     - Both arguments use the same permutation action and factor identities.
   * - ``pointwise_role_product``
     - Coupling followed by the role-function multiplication map
     - The source supplies its role algebra. The supported normalized
       two-role characters give a factor :math:`1/\sqrt{2}`.
   * - ``disjoint_induction``
     - LR induction from :math:`S_{N_1}\times S_{N_2}` to :math:`S_{N_1+N_2}`
     - Every compiler-ordered placement of the two child factor sets is supplied.
   * - ``independent_action_tensor_product``
     - External tensor product retaining both independently acting permutation groups
     - Separately bound child actions and identities; the output retains their ordered group factors.

Angular couplings satisfy the triangle rule, and output parity is
:math:`p=p_1p_2`. Equal argument identities request a self-product. For
:math:`V=\mathbb C^c\otimes[\lambda]\otimes V_L`, the compiler selects
:math:`\operatorname{Sym}^2(V)`: exchange acts on channel pairs, Young copies,
and angular coordinates together. Distinct argument identities retain the
full tensor product. These counts describe representation multiplicities;
relations imposed by a particular physical source are handled by that source.

Products of symmetric groups
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``YE3TCarrierKey`` represents a product group with ``permutation_blocks`` and
an empty single-group ``partition``. For example, ``((2,), (1, 1))`` selects
the trivial representation of the first :math:`S_2` and the sign representation
of the second. ``rank`` is their total factor degree. The ordered block
degrees identify the acting group, and the full tableau count is the product
of the block dimensions.

An independent-action product concatenates the child group factors and
couples their angular momenta. A same-action product couples corresponding
group factors block by block. For inputs :math:`\boldsymbol\lambda_1`,
:math:`\boldsymbol\lambda_2` and output :math:`\boldsymbol\lambda`, its
permutation multiplicity is

.. math::

   g_{\boldsymbol\lambda_1,\boldsymbol\lambda_2}^{\boldsymbol\lambda}
   =\prod_b g_{\lambda_{1,b},\lambda_{2,b}}^{\lambda_b}.

The compiler retains every copy and composes the sparse block maps. Equal
argument identities apply the exchange projection across all permutation
blocks, angular components and channel pairs. CUDA evaluation uses the same
sparse bilinear kernel and supports complete first and second derivatives.
The direct-product decomposition follows
`Putman, Representations of products, Theorem A
<https://www3.nd.edu/~andyp/notes/RepOfProducts.pdf>`_.

Application graph stages bind independent input action identities in
``independent_action_ids``. Physical source adapters additionally identify
the associated occurrences and their equality rules. The explicit-Phi
two-pair example in ``ye3t-ace`` retains these identities through root changes.

The following example supplies mixed Young vector inputs and obtains every
quadrupole coordinate in the selected output sector:

.. literalinclude:: ../examples/coupled_carrier_products.py
   :language: python

Disjoint products have an additional placement axis immediately before the
channel axis. Supply the corresponding values for every placement returned by
the compiler. Permutations crossing the child sets act on this axis as well
as on the child coordinates.

.. literalinclude:: ../examples/induced_carrier_products.py
   :language: python

Existing joint-product schedules
--------------------------------

``JointYoungCGProduct`` couples already-constructed angular and Young sectors.
For a fixed-content rank-``N`` YE3T basis, the parent symmetry is
``S_N x SO(3)``; the repeated-content subgroup ``G_nu`` enters the internal
induction or restriction step. Each accepted product instruction combines:

.. math::

   C^{L M}_{L_1 M_1, L_2 M_2}
   \quad\text{and}\quad
   Y_{\lambda_1,\lambda_2}^{\lambda},

The first factor is a Clebsch-Gordan coefficient; the second is the Young
coupling map for the relevant group change. The runtime composes the maps.

The output is a fused irreducible target space: if several pairwise coupling
branches reach the same output sector, they share one output block. The
runtime adds their coordinate maps into that block.

By default, products are ``SO(3)``-resolved and do not discard pseudoscalar or
odd-parity sectors. Pass ``target_parity="even"`` or ``target_parity="odd"``
to request an ``O(3)`` parity filter. For spherical-harmonic input factors,
the total parity is ``(-1)^{sum_i l_i}``. A scalar from three ``l=1`` factors is an
odd-parity ``SO(3)`` scalar and is excluded by ``target_parity="even"``.

Low-Level Product Construction
------------------------------

Programs that already hold coupled sectors can use ``ye3t.workflows`` to build
their runtime product from explicit dictionaries. For a new fixed-content
calculation, use the seven-section configuration in
``examples/user_supplied_factors.py`` with ``ye3t.couplings.count``, ``plan``,
and ``compile``.

.. code-block:: python

   from ye3t.workflows import build_joint_young_cg_product_from_config

   product = build_joint_young_cg_product_from_config(
       {
           "left": {"blocks": [{"nin": [1], "lin": [1], "L": 1, "mul": 2}]},
           "right": {"blocks": [{"nin": [1], "lin": [1], "L": 1}]},
           "tree_type": "balanced",
           "L_max": 1,
           "permutation_policy": "trivial_only",
       }
   )

   print(product.format_instruction_report())

Each block describes one exact canonical sector:

``nin`` and ``lin``
   Leaf channel/radial and angular labels. The repeated-channel subgroup is
   inferred from these labels.

``partition`` or ``partitions``
   Optional Young diagrams for the subgroup factors. If omitted, the trivial
   Young sector is used.

``L`` and ``copy_index``
   The angular copy selected from the symbolic sector. ``L`` is required when
   the sector contains multiple angular channels.

``mul``
   Multiplicity copies carried by the runtime feature tensor.

The product-level controls ``rank_cap``, ``L_max``, ``requested_targets``, and
``permutation_policy`` select which valid branches appear in this schedule.

Static Schedules
----------------

Products can be lowered once and reloaded without reconstructing the symbolic
composer:

.. code-block:: python

   schedule = product.static_schedule()
   manifest = product.static_schedule_manifest()
   loaded = product.from_static_schedule(schedule)

The manifest records coefficient hashes and provenance for the Clebsch-Gordan,
Young, and joint coupling tensors.

For a runnable walkthrough of joint permutation-plus-rotation coupling, see
:doc:`basic_examples`.
