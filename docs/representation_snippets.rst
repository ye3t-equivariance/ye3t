Representation Snippets
=======================

Status: stable

Requires:
   ``ye3t`` installed.

These snippets replace small script-style examples that only demonstrated
core representation queries. They are copy/paste friendly and use public
``ye3t`` workflow or utility functions directly.

Count Basis Labels
------------------

Use count-only queries when you need dimensions before materializing coupling
coefficients.

.. doctest::

   >>> import contextlib
   >>> import io
   >>> from ye3t.utils.printing import print_label_count_case
   >>> case = {
   ...     "name": "rank-2 repeated channel",
   ...     "rank": 2,
   ...     "target_l_avs": (0.0, 1.0),
   ...     "strict_max_li": 2,
   ...     "homogeneous_n": False,
   ...     "print_limit": 2,
   ...     "spec": {
   ...         "mode": "restricted",
   ...         "n_orbits": ((1, 1),),
   ...         "l_orbits": ((1, 1),),
   ...         "pair_orbits": ((1, 1),),
   ...     },
   ... }
   >>> stream = io.StringIO()
   >>> with contextlib.redirect_stdout(stream):
   ...     print_label_count_case(case)
   >>> text = stream.getvalue()
   >>> "rank-2 repeated channel" in text and "count:" in text
   True

Enumerate Basis Sectors
-----------------------

The same sector can be queried by target angular momentum ``L`` without
constructing all coefficient tensors.

.. doctest::

   >>> import contextlib
   >>> import io
   >>> from ye3t.utils.printing import print_basis_enumeration
   >>> config = {
   ...     "mode": "counts_by_L",
   ...     "max_target_L": 3,
   ...     "include_rank16": False,
   ...     "print_limit": 2,
   ...     "tree_type": "balanced",
   ...     "cases": (
   ...         {"name": "rank-2 repeated", "nin": (1, 1), "lin": (1, 1)},
   ...     ),
   ... }
   >>> stream = io.StringIO()
   >>> with contextlib.redirect_stdout(stream):
   ...     print_basis_enumeration(config)
   >>> text = stream.getvalue()
   >>> "rank-2 repeated" in text and "mode=counts_by_L" in text
   True

Structured Basis Coordinates
----------------------------

Use structured-sector printing when you want representative coordinate labels
for a fixed ``(eta_i, l_i)`` sector.

.. doctest::

   >>> import contextlib
   >>> import io
   >>> from ye3t.utils.printing import print_structured_basis_sector
   >>> case = {
   ...     "name": "rank4_mixed_scalar",
   ...     "nin": (1, 1, 1, 2),
   ...     "lin": (1, 1, 2, 2),
   ...     "L_R": 0,
   ...     "print_limit": 2,
   ...     "tree_type": "balanced",
   ... }
   >>> stream = io.StringIO()
   >>> with contextlib.redirect_stdout(stream):
   ...     print_structured_basis_sector(case)
   >>> text = stream.getvalue()
   >>> "rank4_mixed_scalar" in text and "Basis:" in text
   True

Generalized Characters And Change Of Group
------------------------------------------

The generalized utilities build actual representation and tensor-product
objects. Keep these calls small in documentation; increase the ranks and
angular caps in scripts or benchmarks.

.. doctest::

   >>> import contextlib
   >>> import io
   >>> from ye3t.utils.illustrators import (
   ...     illustrate_generalized_permutation_characters,
   ...     illustrate_generalized_tensor_product_change_of_group,
   ... )
   >>> stream = io.StringIO()
   >>> with contextlib.redirect_stdout(stream):
   ...     illustrate_generalized_permutation_characters({
   ...         "nin": (1, 1),
   ...         "lin": (1, 1),
   ...         "symmetric_L": 1,
   ...         "antisymmetric_L": 0,
   ...         "antisymmetric_partition": (1, 1),
   ...         "swap": (1, 0),
   ...         "alpha": 0.1,
   ...         "batch": 2,
   ...     })
   >>> "Generalized irreps" in stream.getvalue()
   True
   >>> stream = io.StringIO()
   >>> with contextlib.redirect_stdout(stream):
   ...     illustrate_generalized_tensor_product_change_of_group({
   ...         "left_L": 1,
   ...         "right_L": 1,
   ...         "output_L": 1,
   ...         "tree_type": "balanced",
   ...         "node_span": (0, 2),
   ...         "seed": 3,
   ...         "batch": 2,
   ...         "write_diagrams": False,
   ...         "max_diagram_instructions": 4,
   ...     })
   >>> "GeneralizedFullTensorProduct" in stream.getvalue()
   True

Schedule Metadata
-----------------

Schedule metadata checks should use existing schedule printers and write
diagrams only when you explicitly request output.

.. doctest::

   >>> import contextlib
   >>> import io
   >>> from ye3t.utils.printing import print_schedule_planning_policy
   >>> stream = io.StringIO()
   >>> with contextlib.redirect_stdout(stream):
   ...     print_schedule_planning_policy("auto")
   >>> "policy=auto" in stream.getvalue()
   True

Common failure modes
--------------------

- Omitted config keys can change a snippet from a bounded count/report into an
  unbounded enumeration. Keep ``include_rank16``, label limits, and output
  controls explicit.
- Snippets that need coefficient tensors should use ``ye3t.couplings.plan`` or
  ``ye3t.couplings.compile`` explicitly instead of relying on a count-only
  printer.
- Diagram output is opt-in; do not write generated figures from copy/paste
  snippets unless the output path is explicit.
