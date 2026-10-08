General YE3T Tensor Products
============================

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
