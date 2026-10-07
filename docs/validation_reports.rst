Validation Reports
==================

Status: stable
Requires: ``ye3t``
Runs in: no runtime dependency; reports are returned by compiler calls
Produces: provenance dictionaries attached to count, plan, and compile results

Validation reports are ordinary dictionaries attached to public report objects.
They are not formal proof objects.  They record what was validated, which
backend produced the result, and which convention hash identifies the
coefficient table.

Typical fields include:

``passed``
   Overall validation status for the requested operation.

``backend`` or ``coefficient_backend``
   Backend that produced the count, subduction, or coefficient table.

``convention_hash``
   Hash of normalization, carrier, target, and backend metadata.

``residuals`` / ``rank_gap`` / ``singular_values``
   Numeric-backend diagnostics when a numeric nullspace or subduction backend
   is used.

``valid_labels_from``
   Source of truth for label validity, usually ``ye3t.couplings.count`` or a
   lower-level exact fixed-content decomposition called by it.

.. code-block:: python

   from ye3t.couplings import count, plan, compile

   report = count(content=(1, 1, 1), input_Ls=(0, 1, 1), target_L=0)
   print(report.validation_report["passed"])
   print(report.validation_report["valid_labels_from"])
   compiled = compile(plan(report))
   print(compiled.certificate.passed)

This case has the same radial/content channel at both ``l=0`` and ``l=1``.
Compilation checks the full content-and-angular blocks, then binds the
existing compact ACE coefficient to the counted label. The compiled
ordinary-density table records its physical projection and every magnetic
component. A count report alone still does not certify coefficient
materialization.

Common failure modes:

- A numeric coefficient table is not valid just because it exists; check its
  validation report.
- A label printed by an old compact tuple is not necessarily valid until a
  current report accepts it.
- Do not call report dictionaries certificates unless the object is a true
  formal proof artifact.

Validation link: ``tests/test_couplings_namespace.py`` and
``tests/test_numeric_subduction_fastpath.py`` exercise these fields.

