Workflow Utility Map
====================

The public examples are importable recipes over reusable utility modules.
They can also run directly as scripts from the package root. The files stay
focused on workflow composition while exposing the same durable building blocks
to users.

``ye3t.workflows``
   Rank-indexed counting, compilation, and evaluation of configured tensor
   contents, plus runtime-irrep builders and joint Young/``SO(3)`` products.

``ye3t.utils.printing``
   Stable summaries for basis labels, exact sector enumeration, primitive
   quotient coverage, coupling payload previews, cache policies, and schedule
   planning metadata.

``ye3t.utils.illustrators``
   Reusable demonstrations for generalized permutation-representation spaces,
   lowered tensor-product change-of-group maps, symmetric-power output
   catalogs, optional folded symmetric-power kernels, and object-driven SVG
   diagrams. The SVG helpers accept constructed tensor products or exact
   schedules and render their actual instructions, segments, dimensions,
   supports, and materialization decisions.

Example modules expose a visible editable ``cfg_ye3t`` dictionary. Core
examples construct public objects and call their operations directly.
Specialized catalogues and benchmarks use a purpose-named ``run_*`` function.

Coupling Several Ranks
----------------------

The core examples use this sequence:

.. code-block:: python

   from ye3t import YE3TRepresentation
   from ye3t.workflows import (
       count_coupling_cases, compile_coupling_cases, evaluate_coupling_cases,
   )

   representation = YE3TRepresentation.from_config(cfg_ye3t["representation"])
   cases = count_coupling_cases(representation, cfg_ye3t["basis"])
   compiled_cases = compile_coupling_cases(cases)
   values = evaluate_coupling_cases(
       compiled_cases, cfg_ye3t["basis"]["multiplets"],
       dtype=cfg_ye3t["runtime"]["dtype"], device=cfg_ye3t["runtime"]["device"],
   )

``basis.cases_by_rank`` maps every selected rank to a list of fixed contents.
For ``ACE_density`` and ``external_tensor``, each entry supplies ``content``
and ``input_Ls``. Cauchy entries supply ``channel_indices`` into
``basis.channels`` and matching ``block_sizes``; ``role_dimension`` and
``role_kappa_policy`` belong in ``basis``. Counts, compiled objects, and
values are keyed by ``(rank, case_index)``. Zero-dimensional sectors remain
visible in the count result and raise an error if selected for compilation.

For ordinary density or external factors with parent ``(N)``,
``representation.intermediates.young_kappa="symmetric_only"`` explicitly
selects symmetric repeated-content blocks. Restricting the trivial parent
to those subgroups already gives only trivial Young types, so this setting
preserves the full multiplicity space. Role-resolved Cauchy carriers have
additional role representations and retain their separate intermediate policy.

For density inputs, ``multiplets[l][..., n-1, :]`` supplies channel ``(n,l)``.
For external inputs, ``multiplets[l][..., f, :]`` supplies factor ``f``.
The index ``f`` is its position in the complete ordered factor sequence,
including when different positions have different angular momenta.
Both use complete complex magnetic multiplets in ``m=-l,...,l`` order.
Ordered Cauchy inputs add a role axis: ``(..., factor, role, magnetic)``.
Their outputs preserve ``(..., a, t, M)``. Ordinary density outputs use
``(..., a, M)``. Torch evaluation preserves batch axes and derivatives.

The ``A_s`` Cauchy example supplies ``basis.role_densities`` as the second
evaluation argument: a mapping from configured channel index to a real
tesseral ``(role, magnetic)`` array. This evaluator uses CPU float64.
See :doc:`basic_examples` for complete configurations and expected counts.

Selecting source types and copies
--------------------------------

For ``ACE_density``, ``external_tensor`` and ``ordered_role`` cases, the optional
``representation.source_sectors`` setting controls the independent source
inventory before compilation:

.. code-block:: python

   "source_sectors": {
       "young_lambda": "all", "L_max": 2, "parity": "all", "copies": 2,
   }

The Young policy accepts ``all``, ``symmetric_only``, ``antisymmetric_only``, or
a list of partition templates such as ``["(N)", "(N-1,1)"]``. Templates use
the degree of the source action. Ordinary density permits the symmetric type.
``L`` requests particular angular momenta; ``L_max`` sets an upper bound.
When both are omitted, all physically available angular outputs are considered.
Parity follows the input factors, including explicit factor parities in ordered
Cauchy inputs. Invalid explicit requests raise errors with available alternatives.

``copies`` limits independent copies within each fixed content and output type;
``"all"`` keeps every available copy. Selected copies retain their original
compiler labels and all tableau and magnetic components. Counts include
``available_multiplicity_count``, ``selected_multiplicities`` and
``selected_labels``. Keys become ``(rank, source_id)`` because one input case
can supply several output types. The same count, compile and evaluate calls
shown above apply.

Each counted case also exposes ``full_path``: complete input channel content,
the original multiplicity indices and canonical labels. This information is
available before coefficient materialization and follows selected sources into
application planning. Repeated identical input cases raise an error in both
the default and sector-selected workflows.

At the lower-level fixed-content interface,
``coupler_plan.select_multiplicities((0, 2))`` selects original independent
copies 0 and 2 for factorized compilation. Indices must be sorted, unique and
within the exact multiplicity count. This selection preserves complete
irreducible multiplets and supports differentiation through supplied factors.

Config Dictionary Conventions
-----------------------------

The public examples keep user-editable knobs in the module-level
``cfg_ye3t`` dictionary. This is intentionally similar to the source comments
in the example files:

- ``cases_by_rank`` contains the core fixed-content coupling requests.
- ``cases`` is used for specialized catalogues or diagnostic cases.
- ``name`` is only for printed output.
- ``rank`` is the feature/product rank ``N``.
- ``nin`` and ``lin`` define a concrete sector and must have matching length.
- ``L_R`` is the target angular momentum.
- ``print_limit`` limits printed labels, not exact counts.
- ``tree_type`` selects the recoupling tree used for compact paths.
- ``mode`` switches between count-only and materialized-label views in
  enumeration examples.
- ``write_diagrams=True`` writes SVG diagrams from runtime objects built from
  the coupling report.
- ``max_diagram_instructions`` and ``max_diagram_segments`` keep actual
  instruction/segment graphs compact when a product has many branches.

Utility functions may still provide reusable defaults for tests and advanced
users, but the examples are meant to be readable recipes: edit ``cfg_ye3t``,
then run the file directly or call its public construction and evaluation steps.
