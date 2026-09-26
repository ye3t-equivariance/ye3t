Pure Rotation Quickstart
========================

Status: stable
Requires: ``ye3t``
Runs in: under one second for the shown rank-2 coupler
Produces: a cached ``SO(3)`` Clebsch-Gordan coupling report
Uses: ``ye3t.couplings`` indirectly through the public pure-coupler API

Use this when the calculation is only about angular momentum coupling and does
not need atom positions, neighbor lists, ASE, or descriptor materialization.
The package boundary is simple: ``ye3t`` owns the representation and
coefficient table.

.. code-block:: python

   from ye3t import compile_rotation_coupler

   coupler = compile_rotation_coupler((1, 1), 0, group="SO3")
   payload = coupler.to_dict()

   print(payload["map_kind"])
   print(payload["input_Ls"], "->", payload["output_L"])
   print(payload["coefficient_validation"]["passed"])

Expected output shape:

.. code-block:: text

   AngularCGMap
   (1, 1) -> 0
   True

Common failure modes:

- ``output_L`` must satisfy the angular triangle rule.
- ``group="O3"`` parity filters require a declared parity convention.
- Large scans should use cached calls rather than rebuilding coefficient
  tables inside loops.

Validation link: ``tests/test_pure_couplers.py`` checks this exact public path.

