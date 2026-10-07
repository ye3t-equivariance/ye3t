Backend Capability Table
========================

Status: stable documentation of current public capability
Requires: ``ye3t``; optional accelerator extras are noted per backend
Runs in: documentation only
Produces: a support matrix for compiler and runtime choices

.. list-table::
   :header-rows: 1

   * - Area
     - Stable
     - Experimental
     - Planned / limited
   * - Coupling labels
     - ``ye3t.couplings.count`` for fixed-content multiplicities
     - Carrier-specific reports for ``A_s``, ``Phi``, and message states
     - Very large exact projector materialization may require numeric reports
   * - Coefficients
     - Factorized ACE schedules and physical Cauchy compilers, with provenance and validation reports
     - Full typed factorized evaluation across every valid multiplicity coordinate, ordered-role Cauchy factor evaluation, and numeric cached Young subduction
     - Native packed lowering for the general typed and ordered-role schedules remains open; bounded dense typed-orbit assembly is explicitly a reference
   * - Pure rotation
     - ``compile_rotation_coupler`` for ``SO(3)``/``O(3)`` angular couplers
     - Optional accelerator bridges
     - Broader non-integer spin replacement kernels are tracked separately
   * - Pure permutation
     - ``compile_permutation_coupler`` and same-rank Kronecker tables
     - Numeric cached subgroup subduction
     - Unbounded exact projector construction

Package boundary:

- ``ye3t`` owns this table for representation and coupling backends.
- ``ye3t-methods`` owns atomistic descriptor, calculator, fitting, and ASE-facing
  runtime capability.

