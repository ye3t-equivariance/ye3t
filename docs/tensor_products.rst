General YE3T Tensor Products
============================

``JointYoungCGProduct`` is the public exact product for joint
``SO(3) x G_nu`` sectors. Each accepted instruction has two exact pieces:

.. math::

   C^{L M}_{L_1 M_1, L_2 M_2}
   \quad\text{and}\quad
   Y_{\lambda_1,\lambda_2}^{\lambda},

where the first factor is a Clebsch-Gordan coefficient and the second is the
lowered Young/change-of-group coupling matrix. The runtime contraction uses the
joint tensor obtained by composing those two exact maps.

The output is a fused irreducible target space: if several pairwise coupling
branches reach the same canonical ``SO(3) x G_nu`` label, they share one output
block and their exact coordinate maps are accumulated into that block.  The
runtime does not expose unfused coupling-path blocks as a public backend.

By default, products are ``SO(3)``-resolved and do not discard pseudoscalar or
odd-parity sectors.  Pass ``target_parity="even"`` or ``target_parity="odd"``
to request an ``O(3)``-style parity filter using the total input parity
``(-1)^{sum_i l_i}``.  For example, a scalar from three ``l=1`` slots is an
odd-parity ``SO(3)`` scalar and is excluded by ``target_parity="even"``.

Dictionary Construction
-----------------------

For examples and configuration files, use ``ye3t.workflows`` to build the
runtime irreps and product from explicit dictionaries.

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
``permutation_policy`` are truncation and admissibility controls. They do not
change the mathematical coupling rules; they only decide which exact valid
branches are retained.

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
