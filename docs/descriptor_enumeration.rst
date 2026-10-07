Descriptor Enumeration And Label Printing
=========================================

The core enumerator lists canonical rank labels before any ACE chemistry or
radial basis is attached. It is useful for checking counts, printing compact
labels, and deciding which ranks should be exposed to a downstream descriptor
builder.

Exact enumeration works over orbit partitions of the repeated leaf labels. For
each rank ``nu``, ``enumerate_rank_labels`` returns records containing
``n_in``, ``l_in``, orbit partitions, and realized average angular momentum.

.. code-block:: python

   from ye3t import enumerate_rank_labels, format_ye3t_basis

   labels = enumerate_rank_labels(
       rank=4,
       target_l_avs=(0.0, 0.5, 1.0),
       strict_max_li=2,
       spec={"mode": "exhaustive"},
       max_labels=8,
   )

   for index, label in enumerate(labels):
       print(format_ye3t_basis(index, label))

Basis labels are schema-backed.  The public ``BasisLabel`` object can carry
fixed content, radial content, Young labels, angular paths, multiplicity
coordinates, conventions, and carrier type when those fields are known from the
compiled plan or exact basis metadata.  Use ``to_json`` for programmatic
serialization, ``to_latex`` for papers, ``to_human`` / ``format_ye3t_basis``
for reports, and ``to_filename_safe`` for artifacts.  Unknown fields remain
explicitly absent; formatters must not invent Young, LR, tableau, parity, or
convention data from a compact legacy tuple alone.

ACE descriptors add chemistry, radial channels, target angular momentum, and
optionally charge axes. Physical materialization and models live in the
separate ``ye3t-methods`` package; ``ye3t`` supplies their exact
representation-side labels and coupling coefficients.

Coupling Namespace
------------------

Valid descriptor labels, multiplicity counts, backend plans, and coupling
coefficients should be requested through ``ye3t.couplings``:

.. code-block:: python

   from ye3t.couplings import count, plan, compile_ace_factorized_schedules_by_L

   report = count(content=(1, 1, 2), input_Ls=(1, 1, 0), target_L=0)
   labels = report.labels_for_target(0)
   report.require_label(labels[0], target_L=0)

   coupler_plan = plan(report)
   compiled = compile_ace_factorized_schedules_by_L(
       content=report.content, input_Ls=(1, 1, 0),
   )
   assert compiled.schedules_by_L[0].basis_count == len(labels)

``count`` is the public fixed-content multiplicity report.  ``plan`` records
the backend choice and count provenance before coefficient construction.
The factorized ACE compiler materializes practical coupling schedules and
attaches a validation report. Downstream packages such as
``ye3t-methods`` may materialize descriptors from these reports and plans, but
they should not decide locally which symmetry labels are valid.

For nonzero angular factors in ordinary density,
``compile(..., allow_dense_reference=True)`` can check the full
``(content, l)`` block decomposition and every angular path. It projects the
globally trivial Young output onto commutative density and verifies that the
compact ACE labels span the same physical space with one common change of
basis across all magnetic components. The compact coefficients remain the
evaluation and serialization convention. This joint reference compilation
has a bounded dense-table size cap and is meant for tests and comparisons;
``count`` and ``plan`` do not materialize the table.

For an ordinary commutative ACE-density label, ``compile_ace_coordinate``
materializes all ``2L+1`` complex-magnetic components. It composes the
compiler's symmetric-power block coordinates with its angular schedule,
then collects equivalent density monomials. The returned certificate checks
fixed-content membership, magnetic sign reversal, and the SO(3) raising
generator. The algebraic map is exact for compiler-issued labels with maximal
fixed-content blocks through rank eight; stored binary64 coefficients use
tolerance pruning and are numerically certified. ``L=0`` delegates to the
existing scalar compiler so its coefficient convention stays compatible.

.. code-block:: python

   from ye3t.couplings import count, compile_ace_coordinate

   report = count(content=(1, 1, 1), input_Ls=(1, 1, 1),
                  target_L=1, target_permutation="trivial",
                  carrier="ACE_density")
   coordinate = compile_ace_coordinate(report.labels_for_target(1)[0])
   table = coordinate["coefficient_table"]
   print(table.M_R_values.tolist())  # [-1, 0, 1]
