Core API Reference
==================

.. automodule:: ye3t
   :members:
   :undoc-members:

.. automodule:: ye3t.couplings
   :members:
   :undoc-members:

Shared permutation actions
--------------------------

The ``shared_permutation_factors`` request describes equal-size factor groups
on which the same permutation acts simultaneously. For example,
``factor_groups=[[0, 1], [2, 3]]`` has four factors and one S2 action.
``content`` and ``input_Ls`` identify every factor; each group preserves its
complete input type. The existing normalized Young/angular and Kronecker/CG
constructors supply the maps.

Optional ``fixed_factor_groups`` adds symmetric powers of repeated complete
inputs fixed under this shared action. For example, ``[[4, 5]]`` adds two uses
of one input; its angular decomposition is retained with trivial S2 action.
Active and fixed groups together cover all input factors exactly once. Use
separate singleton groups for distinct independent fixed inputs. Physical
background evaluation and its product/marginal image require their own binding.

Select a single ``target_partition`` and ``target_L``, or supply
``source_sectors={"young_lambda": "all", "L_max": 2, "copies": 1}``.
The selector resolves partitions against the acting group degree. Count/plan
reports give available and selected independent copies, complete tableau and
magnetic dimensions, and an individual compiler request per selected sector.
They do not materialize runtime coefficients.

For a sector collection, compilation returns selected ``records``. Use
``execution_plan_from_compiled_coupler(compiled, table_index=index)`` for each
record. Every resulting plan uses the existing typed-source evaluator.
These counts describe independent abstract tensor coordinates. A physical
adapter must also verify its evaluation image and actual action.

.. automodule:: ye3t.workflows
   :members:
   :undoc-members:

.. automodule:: ye3t.utils.printing
   :members:
   :undoc-members:

.. automodule:: ye3t.utils.illustrators
   :members:
   :undoc-members:

.. automodule:: ye3t.ir
   :members:
   :undoc-members:

.. automodule:: ye3t.lowering.schedules
   :members:
   :undoc-members:

.. automodule:: ye3t.lowering.compiler
   :members:
   :undoc-members:

.. automodule:: ye3t.representations.young_sectors
   :members:
   :undoc-members:

.. automodule:: ye3t.representations.young_orthogonal
   :members:
   :undoc-members:

.. automodule:: ye3t.representations.young_yamanouchi_runtime
   :members:
   :undoc-members:

.. automodule:: ye3t.representations.partition_centralizer
   :members:
   :undoc-members:

.. automodule:: ye3t.runtime.generalized
   :members: JointYoungCGProduct, YoungE3TensorProduct, GeneralizedExactRuntimeBlock, GeneralizedExactRuntimeIrreps
   :undoc-members:

.. automodule:: ye3t.runtime.schur_weyl_tree
   :members:
   :undoc-members:
