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
optionally charge axes. Those workflows live in ``ye3t-ace`` (separate
package, not yet public); the core package only enumerates the exact
representation-side labels.

Coupling Namespace
------------------

Valid descriptor labels, multiplicity counts, backend plans, and coupling
coefficients should be requested through ``ye3t.couplings``:

.. code-block:: python

   from ye3t.couplings import count, plan, compile

   report = count(content=(1, 1, 1), input_Ls=(0, 1, 1), target_L=0)
   labels = report.labels_for_target(0)
   report.require_label(labels[0], target_L=0)

   coupler_plan = plan(report)
   compiled = compile(coupler_plan)

``count`` is the public fixed-content multiplicity report.  ``plan`` records
the backend choice and count provenance before coefficient construction.
``compile`` materializes coefficients through the selected lower-level YE3T
compiler and attaches the validation certificate.  Downstream packages such as
``ye3t-ace`` may materialize descriptors from these reports and plans, but
they should not decide locally which symmetry labels are valid.
