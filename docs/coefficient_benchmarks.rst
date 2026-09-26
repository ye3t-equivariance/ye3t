Coefficient Construction Smoke Benchmarks
=========================================

``local_coefficient_materialization_benchmark_table(...)`` returns a bounded
local smoke table for central coefficient-construction paths:

* symmetric-power ``lambda=(N)`` ACE;
* exterior/sign ``lambda=(1^N)``;
* a small generic global Young--E3 coupler;
* a repeated-content balanced-tree case;
* a balanced Young--E3 message-passing schedule-construction smoke case;
* a finite same-rank Kronecker reference case for
  ``[2,1] tensor [2,1]`` under the diagonal ``S_3`` action.

Each local row records the selected backend, elapsed construction time for the
small case, a conservative ``target_elapsed_seconds`` guardrail,
``within_target`` status, sparse table kinds, sparse entry counts, factorized
table kinds, certificate status, selected fast-path policy, main certificate
checks, angular coefficient normalization status, dimension-sum status,
coefficient hash, fast-path policy mode, any forced backend, and whether the row
is a normal-test guardrail or an optional external placeholder.  These rows are
guardrails for compiler regressions and metadata shape changes.  They are not
evidence for production performance, GPU scaling, or broad comparison claims.
Rows also record ``global_young_label`` when available,
``global_target_partition``, ``block_mu_labels``, and ``label_scope`` so the
benchmark table checks that symmetric-power ACE uses global ``lambda=(N)`` and
exterior/sign paths use global ``lambda=(1^N)`` without treating block
``mu_b`` labels as global Young labels.
The balanced schedule row additionally records the sector schedule count and
hashes the scheduled dispatch-coupler coefficient hashes; it does not benchmark
or claim a full recursive message-passing runtime.
Exterior/sign rows also carry the finite sign-vector table scope and
carrier-level wedge-vanish metadata.  This keeps benchmark output aligned with
the runtime distinction between raw permutation-basis sign-vector evaluation
and a realized fermionic carrier/operator model.
The same-rank Kronecker row records the exact character-count decomposition,
exact finite intertwiner table shapes, finite torch evaluator shapes, and a
local timing smoke target.  It is a reference-table guardrail for the diagonal
``S_N`` tensor product path; it does not claim descriptor evaluation, message
passing, nonlinear feature products, or task readout support.

Optional rows for cuEquivariance, OpenEquivariance, and Dusson/Barthelemy-style
external scripts are included only as skipped placeholders unless a separate
benchmark workflow runs those dependencies and records hardware, dtype, input
size, warmup/timed runs, and baseline details.

Example:

.. code-block:: python

   from ye3t import local_coefficient_materialization_benchmark_table

   rows = local_coefficient_materialization_benchmark_table(
       repeat=1,
       include_optional_external=True,
   )
   for row in rows:
       print(row["name"], row["status"], row["backend"])


Artifact Benchmarks
===================

``coefficient_materialization_benchmark_artifacts(...)`` writes raw benchmark
artifacts for representative coefficient materialization families:

* rotation-only ``AngularCGMap`` construction;
* permutation-only cached numeric Young-subgroup subduction;
* joint Young--E3 global-coupler construction;
* optional external-library comparison placeholders.

The writer separates cold and cached construction rows, records validation
reports on every local timed row, and writes:

* ``coefficient_materialization_rows.json`` with full nested metadata;
* ``coefficient_materialization_rows.csv`` for quick plotting and spreadsheet
  inspection;
* ``coefficient_materialization_timings.png`` with a transparent background
  when matplotlib is installed and plotting is enabled.

Each timed row records the benchmark family, carrier, backend, construction
mode, elapsed seconds, multiplicity, basis dimension, coefficient nonzeros,
memory estimate, cache status, validation report, residuals, tolerance,
convention hash, and coefficient hash.  Numeric subduction rows include the
validation report produced by the numeric backend, including residual and
rank-gap metadata.  Joint Young--E3 rows include compiler certificate checks
and coefficient hashes.

External rows are not run by default.  They document capability overlap for
e3nn-style rotation CG, cuEquivariance-style sparse CG kernels, and generic
Lie/nullspace basis-generation methods.  Those rows are deliberately marked as
optional comparison placeholders so the package never depends on external
libraries for local benchmark generation.

Example:

.. code-block:: python

   from pathlib import Path
   from ye3t.coefficient_benchmarks import coefficient_materialization_benchmark_artifacts

   artifact = coefficient_materialization_benchmark_artifacts(
       Path("examples/generated/coefficient_materialization_benchmarks"),
       {"repeat": 1, "make_plots": True},
   )
   print(artifact["files"]["json"])
