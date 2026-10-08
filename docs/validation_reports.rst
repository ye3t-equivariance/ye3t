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

   from ye3t.couplings import count, compile_ace_coordinate

   report = count(content=(1, 1, 1), input_Ls=(0, 1, 1), target_L=0)
   print(report.validation_report["passed"])
   print(report.validation_report["valid_labels_from"])
   assert report.counts_by_target[0] == 1
   coordinate = compile_ace_coordinate(report.labels_for_target(0)[0])
   magnetic_rows, coefficients = coordinate["coefficient_table"].component_terms(0)
   print(len(magnetic_rows), len(coefficients))

This case has the same radial/content channel at both ``l=0`` and ``l=1``.
The compact ACE compiler materializes the counted scalar coordinate and its
magnetic coefficients. The general ``compile(plan(report))`` path handles a
bounded single-angular-path ordinary-density case with distinct content; use
``compile_ace_coordinate`` for this repeated-content example.

Common failure modes:

- A numeric coefficient table is not valid just because it exists; check its
  validation report.
- A label printed by an old compact tuple is not necessarily valid until a
  current report accepts it.
- Do not call report dictionaries certificates unless the object is a true
  formal proof artifact.

Validation link: ``tests/test_couplings_namespace.py`` and
``tests/test_numeric_subduction_fastpath.py`` exercise these fields.

