Representation Snippets
=======================

Status: stable

Requires:
   ``ye3t`` installed.

These short queries use public ``ye3t`` coupling and representation functions.
For a complete editable seven-section config, see
``examples/coupling_multiplicity_counts.py``.

Count Basis Labels
------------------

Use count-only queries when you need dimensions before materializing coupling
coefficients.

.. doctest::

   >>> from ye3t.couplings import count
   >>> report = count(content=(1, 1), input_Ls=(1, 1), target_L=0,
   ...                target_permutation="trivial", carrier="ACE_density")
   >>> len(report.labels_for_target(0)) > 0
   True

Enumerate Basis Sectors
-----------------------

Query the same factor content for several output angular momenta ``L``
without constructing coefficient tensors.

.. doctest::

   >>> counts = {L: len(count(content=(1, 1), input_Ls=(1, 1), target_L=L,
   ...                        target_permutation="trivial", carrier="ACE_density").labels_for_target(L))
   ...           for L in (0, 1, 2)}
   >>> counts[0] > 0 and counts[2] > 0
   True

Structured Basis Coordinates
----------------------------

Inspect compiler-issued coordinate labels for a fixed factor-content and
angular-momentum sector.

.. doctest::

   >>> report = count(content=(1, 1, 2, 2), input_Ls=(1, 1, 2, 2),
   ...                target_L=0, target_permutation="trivial", carrier="ACE_density")
   >>> labels = report.labels_for_target(0)
   >>> len(labels) > 0
   True
   >>> report.require_label(labels[0], target_L=0) is not None
   True

Explicit physical contents and shared coordinates
------------------------------------------------

The scalar tagged catalogue request accepts ``additional_source_contents_by_rank``
and ``generated_ranks`` to combine selected complete channels with generated
contents. A complete channel retains its chemical identity, radial index,
angular degree and source family. ``ye3t.couplings.catalogue_inputs`` supplies
the shared aligned-input normalization used by the application resolvers.
Counts, valid intermediate paths and physical independence still come from
``count``, ``plan`` and ``compile``. Exact pooled image counts report whether
source-coefficient preparation occurred; the abstract count examples above
have a different, coefficient-free counting boundary.

For already compiled scalar physical images, ``source.coordinates_in(target)``
returns a sparse exact matrix K satisfying ``F_source = F_target @ K.T``.
The method reuses matching IDs and stored source relations, then the existing
exact source-image operation where required. It rejects a missing target span
and persists successful maps under the two compiler identities. It changes no
rotation/permutation coupling normalization. Atomistic root-chemistry masks
and numerical fitting/readout conversion remain in the application packages.
The editable root-specific workflow is documented in YE3T Methods' standardized
linear fitting guide and ``examples/quickstart/explicit_catalogue_descriptors.py``.

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
- Snippets that need coefficients should use ``ye3t.couplings.plan`` followed
  by ``ye3t.couplings.compile`` or the applicable ACE/Cauchy compiler. Ordered
  external factors use the complete factorized typed route by default. The
  full angular typed-orbit matrix is a bounded reference requiring explicit
  opt-in.
- Diagram output is opt-in; do not write generated figures from copy/paste
  snippets unless the output path is explicit.
