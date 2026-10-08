YE3T Core Documentation
=======================

``ye3t`` constructs the symmetry couplings used to combine tensor factors in
descriptors, linear models, and rotation/permutation-equivariant message
passing. It counts independent couplings, builds their coefficients, and
evaluates supplied factor arrays. The factors and model architecture can come
from any application that preserves the required rotation and permutation
actions.

The separate `ye3t-methods <https://github.com/ye3t-equivariance/ye3t-methods>`_
package provides descriptor construction, fitting, and ASE calculators. It
uses couplings from ``ye3t`` and provides a ``ye3t_ace`` import shim for
compatible saved models. LAMMPS inference is provided by
`ye3t-lammps <https://github.com/ye3t-equivariance/ye3t-lammps>`_.

.. toctree::
   :maxdepth: 2

   descriptor_enumeration
   first_workflows
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

For tensor-product rank ``N``, let :math:`\boldsymbol{\nu}` denote the fixed
factor-channel content. Repeated channels determine the stabilizer
:math:`G_{\boldsymbol{\nu}}=\prod_b S_{k_b}` within :math:`S_N`. The joint
decomposition selects parent sectors

.. math::

   [\lambda]\otimes V_L, \qquad \lambda\vdash N,

where :math:`[\lambda]` is an irreducible representation of :math:`S_N` and
:math:`V_L` is an angular representation of :math:`SO(3)`. The
permutation-invariant sector has :math:`\lambda=(N)`. A scalar invariant
readout also has :math:`L=0` and even parity when using the :math:`O(3)`
extension. See the `YE3T paper <https://arxiv.org/abs/2609.31895>`_,
Eqs. (5), (9), and (12), for the carrier and coupling conventions.

The fixed-content message-passing mathematical standard is available as
:download:`message_passing_mathematical_standard.md
<message_passing_mathematical_standard.md>`. Its ``STD`` and ``DERIVED`` entries
are implementation contracts. Entries marked ``PROPOSED`` or ``VERIFY``
identify work that still needs validation.

The fixed-rank basis theorem, source-injectivity distinction, and compiler
validation requirements for linear lifted-Cauchy descriptors are available
as :download:`linear_lifted_cauchy_basis.md
<linear_lifted_cauchy_basis.md>`.

The general chemically resolved tagged physical-image construction, exact
pivot validation records, and coordinate conventions are documented in
:download:`general_tagged_physical_image.md <general_tagged_physical_image.md>`.

The covariant Cauchy basis derivation used by ``ye3t.couplings.covariant_cauchy``
is available as :download:`covariant_cauchy_basis.md <covariant_cauchy_basis.md>`.
