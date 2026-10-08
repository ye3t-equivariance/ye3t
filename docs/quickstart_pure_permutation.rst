Pure Permutation Quickstart
===========================

Status: stable
Requires: ``ye3t``
Runs in: under one second for the shown rank-3 cached numeric subduction
Produces: a Young-subgroup Specht subduction report with validation metadata
Uses: cached ``ye3t`` permutation-coupler materialization

Use this when the calculation only concerns tensor-factor permutation
representations.  No atomistic package is involved.  ``ye3t`` owns the Young
subgroup, target partition, coefficient backend, and validation report.

.. code-block:: python

   from ye3t import compile_permutation_coupler

   coupler = compile_permutation_coupler(
       ((2,), (1,)),
       (2, 1),
       backend="numeric_cached",
   )
   payload = coupler.as_dict()

   print(payload["subgroup_partitions"])
   print(payload["target_partition"])
   print(payload["validation"]["passed"])

Expected output shape:

.. code-block:: text

   [[2], [1]]
   [2, 1]
   True

Common failure modes:

- Rank-additive induction/LR and same-rank Kronecker products are different
  operations.
- Numeric subduction is valid only with the returned residuals and validation
  report.
- Descriptor workflows in ``ye3t-methods`` should consume this kind of plan/report
  and obtain labels from the compiler.

Validation link: ``tests/test_pure_couplers.py`` checks this cached
subduction path.

