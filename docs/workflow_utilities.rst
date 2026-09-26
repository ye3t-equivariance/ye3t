Workflow Utility Map
====================

The public examples are importable recipes over reusable utility modules.
They can also run directly as scripts from the package root. The files stay
focused on workflow composition while exposing the same durable building blocks
to users.

``ye3t.workflows``
   Recursive config merging, runtime-irrep builders, reusable construction
   helpers for joint Young/``SO(3)`` products, and utility defaults used by
   tests.

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

Example modules expose a visible editable ``cfg_ye3t`` dictionary plus one
purpose-named ``run_*`` function. Presentation logic comes from
``ye3t.utils``.

Config Dictionary Conventions
-----------------------------

The public examples keep user-editable knobs in the module-level
``cfg_ye3t`` dictionary. This is intentionally similar to the source comments
in the example files:

- ``cases`` is the usual list of sectors or diagnostic cases.
- ``name`` is only for printed output.
- ``rank`` is the feature/product rank ``N``.
- ``nin`` and ``lin`` define a concrete sector and must have matching length.
- ``L_R`` is the target angular momentum.
- ``print_limit`` limits printed labels, not exact counts.
- ``tree_type`` selects the recoupling tree used for compact paths.
- ``mode`` switches between count-only and materialized-label views in
  enumeration examples.
- ``write_diagrams=True`` writes SVG diagrams from constructed runtime objects
  instead of drawing a fixed conceptual sketch.
- ``max_diagram_instructions`` and ``max_diagram_segments`` keep actual
  instruction/segment graphs compact when a product has many branches.

Utility functions may still provide reusable defaults for tests and advanced
users, but the examples are meant to be readable recipes: edit ``cfg_ye3t``,
then call the single ``run_*`` workflow or run the file directly.
