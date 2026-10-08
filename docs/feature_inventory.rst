Feature Inventory and Migration
===============================

Status: preservation inventory

Package boundary
----------------

``ye3t`` owns representation theory, coupling labels, coupling plans,
coefficient materialization, validation reports, and compiler-side fast-path
plans. It must not absorb atomistic datasets, ASE calculators, fitting loops,
or model-training workflows from ``ye3t-methods``.

Migration records
-----------------

Feature: Public coupling namespace
  Current file: ``ye3t/couplings/__init__.py``
  Current tests: ``tests/test_couplings_namespace.py``
  Current docs/examples: ``docs/quickstart_fixed_content_couplers.rst``
  Keep / move / deprecate / delete: keep
  Replacement: none
  Backward compatibility: stable public compiler entry point
  Notes: materialization performance changes must not alter label or
  coefficient provenance.

Feature: Fixed-content and Young-sector decomposition
  Current file: ``ye3t/fixed_content.py``
  Current tests: ``tests/test_fixed_content.py``
  Current docs/examples: ``docs/descriptor_enumeration.rst``
  Keep / move / deprecate / delete: keep
  Replacement: none
  Backward compatibility: stable compiler primitive
  Notes: symbolic internals may be replaced only by validated equivalents that
  leave the decomposition results unchanged.

Feature: Numeric subduction and projector validation
  Current file: ``ye3t/representations/numeric_subduction.py``
  Current tests: ``tests/test_numeric_subduction_fastpath.py``
  Current docs/examples: ``docs/validation_reports.rst``
  Keep / move / deprecate / delete: keep
  Replacement: none
  Backward compatibility: stable when validation reports pass
  Notes: coefficient-materialization benchmarks are documented in
  ``docs/coefficient_benchmarks.rst``.

Feature: Symmetric-power compiler/runtime kernels
  Current file: ``ye3t/runtime/symmetric_power.py``
  Current tests: ``tests/test_symmetric_power_kernels.py``
  Current docs/examples: ``examples/benchmarks/symmetric_power_kernels.py``
  Keep / move / deprecate / delete: keep
  Replacement: compiler-owned plan paths in ``ye3t.couplings``
  Backward compatibility: keep as compiler/runtime implementation detail
  Notes: validated for representative linear ACE, scheduler, benchmark, and
  grouped-CG message paths.

Feature: Exterior/sign and fermion carrier support
  Current file: ``ye3t/fermion_wedge.py``
  Current tests: ``tests/test_su2_spinful_fermions.py``
  Current docs/examples: ``docs/operator_ir_and_schedules.rst``
  Keep / move / deprecate / delete: keep
  Replacement: none
  Backward compatibility: stable only with explicit carrier conventions
  Notes: spinful fermion support requires ``SU(2)`` or a specified double-group
  convention; spatial exterior powers alone do not provide it.

Feature: Rank-graded message-passing compiler core
  Current file: ``ye3t/message_passing.py``
  Current tests: ``tests/test_rank_graded_message_passing_core.py``
  Current docs/examples: ``docs/tensor_products.rst``
  Keep / move / deprecate / delete: keep
  Replacement: none
  Backward compatibility: stable compiler-side representation logic
  Notes: ``ye3t-methods`` message paths take labels and coupling plans from this
  compiler, with no local label enumeration.

Feature: Visualization helpers
  Current file: ``ye3t/utils/illustrators.py``
  Current tests: ``tests/test_examples.py``
  Current docs/examples: ``docs/representation_snippets.rst``,
  ``examples/benchmarks/symmetric_power_kernels.py``
  Keep / move / deprecate / delete: keep and preserve
  Replacement: none
  Backward compatibility: development/paper utility, not a core public API
  Notes: diagram output is opt-in and is written only to an explicitly
  requested output path.
