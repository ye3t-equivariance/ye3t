Fixed-Content Coupler Quickstart
================================

Status: stable
Requires: ``ye3t``
Runs in: under one second for the shown scalar ACE-density sector
Produces: ``count -> plan -> compile`` provenance
Uses: ``ye3t.couplings.count``, ``ye3t.couplings.plan``, and
``ye3t.couplings.compile``

All valid labels, multiplicity counts, coupling paths, and coefficient tables
come from ``ye3t.couplings``.  Downstream packages may materialize descriptors
from these reports, but they do not decide which labels are valid.

.. code-block:: python

   from ye3t.couplings import compile, count, plan

   report = count(content=(1, 1), input_Ls=(0, 0), target_L=0)
   labels = report.labels_for_target(0)
   report.require_label(labels[0], target_L=0)

   coupler_plan = plan(report)
   compiled = compile(coupler_plan, subduction_materialization_backend="exact")

   print(report.carrier)
   print(coupler_plan.backend)
   print(compiled.validation_report["compiled_certificate_passed"])

Expected output shape:

.. code-block:: text

   ACE_density
   symmetric_power_fast_path
   True

Common failure modes:

- Invalid manual labels are rejected by ``MultiplicityReport.require_label``.
- Ordinary ACE density products only realize the globally trivial Young sector.
- ``A_s``, ``Phi``, and message-state carriers must declare their carrier
  constraints before planning.
- This zero-angular example uses the direct scalar path. For nonzero angular
  ACE inputs, use ``compile_ace_factorized_schedules_by_L``. Full generic
  angular typed-orbit matrix assembly is a bounded reference and requires
  ``allow_dense_reference=True`` for tests or comparisons.

Validation link: ``tests/test_couplings_namespace.py`` checks count, plan,
compile, and invalid-label rejection.

