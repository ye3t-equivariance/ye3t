YE3T Core Documentation
=======================

``ye3t`` contains the exact representation and product machinery used by ACE
and message-passing workflows. The core package is intentionally mathematical:
it describes tensor factors, Young-subgroup irreps, angular irreps, exact
Clebsch-Gordan couplings, and lowered tensor schedules without assuming a
particular training task.

LAMMPS inference for compiled models is provided by the separate
`ye3t-lammps <https://github.com/ye3t-equivariance/ye3t-lammps>`_ package.
The separate `ye3t-methods <https://github.com/ye3t-equivariance/ye3t-methods>`_
package provides descriptor construction, fitting, and ASE calculators. It
consumes the ``ye3t`` compiler; its ``ye3t_ace`` module path remains a
compatibility implementation.

.. toctree::
   :maxdepth: 2

   descriptor_enumeration
   quickstart_fixed_content_couplers
   quickstart_pure_rotation
   quickstart_pure_permutation
   validation_reports
   backend_capability_table
   basic_examples
   representation_snippets
   workflow_utilities
   tensor_products
   coefficient_benchmarks
   operator_ir_and_schedules
   message_passing_equation_coverage
   native_execution_plan
   young_primitive_construction
   feature_inventory
   api_reference

Notation
--------

For a descriptor rank ``nu``, the repeated-channel subgroup is inferred from
the leaf labels ``(n_i, l_i)``. A joint sector is a tensor product

.. math::

   V_L \otimes W_\lambda,

where ``V_L`` is an irreducible ``SO(3)`` angular representation and
``W_\lambda`` is a Young-subgroup representation. Scalar invariant readouts are
the special case ``L=0`` with the trivial permutation sector.

The provisional project standard for fixed-content message passing is available
as :download:`message_passing_mathematical_standard.md
<message_passing_mathematical_standard.md>`. Its ``STD`` and ``DERIVED`` entries
are implementation contracts. Entries marked ``PROPOSED`` or ``VERIFY`` remain
under validation and are not production-completeness claims.

The fixed-rank basis theorem, source-injectivity distinction, and compiler
certification requirements for linear lifted-Cauchy descriptors are available
as :download:`linear_lifted_cauchy_basis.md
<linear_lifted_cauchy_basis.md>`.

The general chemically resolved tagged physical-image construction, exact
pivot certificates and coordinate conventions are documented in
:download:`general_tagged_physical_image.md <general_tagged_physical_image.md>`.

The covariant Cauchy basis derivation used by ``ye3t.couplings.covariant_cauchy``
is available as :download:`covariant_cauchy_basis.md <covariant_cauchy_basis.md>`.
