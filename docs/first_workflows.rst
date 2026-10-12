Count and couple tensor factors
===============================

``ye3t`` selects valid permutation and rotation couplings for tensor factors
supplied by an application. It does not construct atomic neighbors or radial
functions. For ASE ``Atoms`` objects, use the ``Basis`` workflows in
`ye3t-methods <https://github.com/ye3t-equivariance/ye3t-methods>`_.
Run the scripts below from an extracted ``ye3t`` source archive.

Count independent coordinates
-----------------------------

Edit ``representation.parent`` to choose the output Young type, angular
momentum, and parity. Each ``basis.cases_by_rank`` entry supplies ``content``
and ``input_Ls`` for one fixed-content request. The seven-section config below
counts fully symmetric scalar coordinates across ranks 1–8.
Use the external-factor examples below for a nontrivial parent Young type.

``count_coupling_cases`` calls ``representation.count_fixed_content(...)``
for each configured case. Reports contain valid labels and multiplicities
before any coefficient build.

.. literalinclude:: ../examples/coupling_multiplicity_counts.py
   :language: python
   :caption: examples/coupling_multiplicity_counts.py

The selected contents have 162 labels at ``L=0``, including 91 at rank eight.
A label names one independent coupling coordinate. Atomistic descriptors
also require physical single-factor functions supplied by the application.

Apply the coupling to supplied factors
--------------------------------------

This example supplies complete magnetic multiplets in complex
Condon--Shortley order ``m=-l,...,l`` across ranks 2–8. Its repeated channel
blocks reach sizes ``(4,4)`` at rank eight. The requested parent has Young
type ``(N-1,1)`` and ``L=1``; the second rank-two case uses ``l=2`` inputs.
The script counts and compiles every valid multiplicity route, binds a
factorized execution plan, and evaluates one factor set and a batch. Its
Torch gradient demonstrates use inside a trainable model. This example
requires PyTorch.

The factor evaluation itself is a direct call:

.. code-block:: python

   runtime = couplings.bind_typed_factorized_execution_plan_torch(
       execution_plan, dtype=factors.dtype, device=factors.device,
   )
   values = runtime.evaluate(factors)

The full runnable script uses ``compile_coupling_cases`` and
``evaluate_coupling_cases`` to construct and bind those plans for every
configured rank. Results are dictionaries keyed by ``(rank, case)``.

.. literalinclude:: ../examples/user_supplied_factors.py
   :language: python
   :caption: examples/user_supplied_factors.py

At rank eight the output shape is ``(12, 7, 3)`` for ``(a, t, M)``: twelve
independent multiplicity coordinates, seven tableau components, and three
magnetic components. The 16-input batch has shape ``(16, 12, 7, 3)``. Change the
factor values or replace their source while preserving their channel,
parity, permutation action, and magnetic convention. Physical aggregation
over neighbors or clusters belongs to the application supplying the factors.
