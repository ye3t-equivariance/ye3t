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
momentum, and parity. Edit ``basis.fixed_content`` and ``basis.input_Ls``
to describe the input factors. The seven-section config below counts the
fully symmetric scalar sector of four factors. No coefficient table is built.

The public call is ``representation.count_fixed_content(...)``; the report
contains valid labels and multiplicities before any coefficient build.

.. literalinclude:: ../examples/coupling_multiplicity_counts.py
   :language: python
   :caption: examples/coupling_multiplicity_counts.py

For the shown content and angular inputs, the report has two labels at
``L=0`` and reports ``validation passed: True``. A label names one independent
coupling coordinate; it is not a radial basis function or an ASE descriptor
column by itself.

Apply the coupling to supplied factors
--------------------------------------

This example supplies three complete ``l=1`` magnetic multiplets in complex
Condon--Shortley order ``m=-1,0,1``. The first two factors have the same
channel type. The requested parent has Young type ``(2,1)`` and ``L=1``.
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

The full runnable script supplies the config, ``execution_plan``, and
``factors`` used by these lines.

.. literalinclude:: ../examples/user_supplied_factors.py
   :language: python
   :caption: examples/user_supplied_factors.py

The output shape is ``(3, 2, 3)`` for ``(a, t, M)``: three independent
multiplicity coordinates, two tableau components, and three magnetic
components. The 64-input batch has shape ``(64, 3, 2, 3)``. Change the
factor values or replace their source while preserving their channel,
parity, permutation action, and magnetic convention. Physical aggregation
over neighbors or clusters belongs to the application supplying the factors.
