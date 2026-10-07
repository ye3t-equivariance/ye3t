Basic YE3T Examples
===================

The remaining examples in ``examples/`` are deterministic importable workflows
for the core ``ye3t`` representation and tensor-product API. Small API
demonstrations that only print counts, characters, or schedule metadata live in
:doc:`representation_snippets` as tested copy/paste snippets.

General Joint Coupling
----------------------

Use ``JointYoungCGProduct`` or the dictionary helper
``build_joint_young_cg_product_from_config`` when a workflow needs a product
between explicit joint sectors

.. math::

   V_{L_1}\otimes W_{\lambda_1}
   \;\otimes\;
   V_{L_2}\otimes W_{\lambda_2}
   \longrightarrow
   V_L\otimes W_\lambda.

The angular factor is an exact Clebsch-Gordan map and the permutation factor is
the exact lowered Young/change-of-group map. The product schedules only retain
branches allowed by both factors.

.. code-block:: python

   from ye3t.workflows import build_joint_young_cg_product_from_config

   product = build_joint_young_cg_product_from_config(
       {
           "left": {
               "blocks": [
                   {"nin": [1], "lin": [1], "L": 1, "mul": 2},
               ],
           },
           "right": {
               "blocks": [
                   {"nin": [1], "lin": [1], "L": 1, "mul": 1},
               ],
           },
           "tree_type": "balanced",
           "L_max": 1,
           "permutation_policy": "mixed_character",
       }
   )

   print(product.format_instruction_report())
   schedule = product.static_schedule()
   loaded = product.from_static_schedule(schedule)

``L_max`` and ``permutation_policy`` are admissibility controls. They do not
replace the Clebsch-Gordan or Young-product rules; they only restrict which
already-valid branches are materialized.

Importable Examples
-------------------

Each example exposes a visible, editable seven-section ``cfg_ye3t``
dictionary. The fixed-content count, coefficient, and supplied-factor
examples read top-to-bottom through public YE3T objects and coupling calls;
specialized catalogues and benchmarks retain a purpose-named ``run_*``
function where they cover several cases.
The recipe files keep their rank, sector, label-limit, and runtime knobs
visible so users can edit them directly.

Reading ``cfg_ye3t``
~~~~~~~~~~~~~~~~~~~~

The examples and the tested snippets in :doc:`representation_snippets` share a
small vocabulary:

``content`` / ``input_Ls`` / ``target_L``
   A fixed-content coupling request: the non-angular channel content, the
   slot angular momenta, and target ``L_R``. The migrated
   ``examples/coupling_multiplicity_counts.py`` places the target under
   ``representation.parent.L`` and the first two values under ``basis``;
   ``coupling_coefficient_materialization.py`` uses the same seven-section shape.

``rank``
   The product rank ``N`` of a feature. In atomistic language this is related
   to the body/correlation order of the descriptor term.

``nin`` / ``lin``
   Tuples that define one product sector. They must have the same length. Each
   slot pairs a non-angular channel ``eta_i`` from ``nin`` with an angular
   momentum ``l_i`` from ``lin``. Repeated pairs induce Young/permutation
   subgroups such as ``S2{0,1}``.

``L_R``
   Target output angular momentum. ``L_R=0`` is scalar; ``L_R=1`` is vector;
   larger values are higher covariants.

``target_l_avs`` / ``strict_max_li``
   Sampling controls for the count-only label snippet. ``target_l_avs`` lists
   average input angular momenta to inspect, while ``strict_max_li`` caps the
   enumerated input ``l_i`` values.

``spec["mode"] = "restricted"``
   Count labels compatible with the explicit orbit partitions in ``n_orbits``,
   ``l_orbits``, and ``pair_orbits`` rather than scanning every possible
   subgroup pattern.

``n_orbits`` / ``l_orbits`` / ``pair_orbits``
   Partitions of ``N`` describing the non-angular subgroup ``G_eta``, angular
   subgroup ``G_l``, and combined feature subgroup ``G_nu``.

``mode``
   For the basis enumeration snippet, ``"counts_by_L"`` prints exact
   dimensions by target ``L_R`` and ``"labels_by_L"`` materializes compact
   basis labels. The primitive catalog uses ``"full"`` with a matched-pair
   factor policy.

``print_limit``
   Bound on representative labels printed per section. It does not change the
   exact count.

Use :doc:`representation_snippets` for small copy/paste count and enumeration
snippets, and use ``examples/coupling_multiplicity_counts.py`` or
``examples/coupling_coefficient_materialization.py`` when you want a dedicated
script that exercises the public coupling report/compiler surface. To choose
your own sector, edit ``cases`` directly:

.. code-block:: python

   from ye3t.utils.printing import print_basis_enumeration

   config = {
       "mode": "labels_by_L",
       "include_rank16": False,
       "max_target_L": 3,
       "print_limit": 3,
       "tree_type": "balanced",
       "cases": (),
   }
   config["cases"] = (
       {
           "name": "my rank-3 sector",
           "nin": (1, 1, 2),
           "lin": (0, 1, 2),
       },
   )
   config["max_target_L"] = 3
   print_basis_enumeration(config)

The snippet page also covers generalized permutation/rotation characters,
exact change-of-group tensor products, structured basis coordinates, and
schedule metadata.

``examples/coupling_multiplicity_counts.py``
   Counts valid labels for one fixed-content ACE sector through
   ``ye3t.couplings.count``. The example prints the content, carrier, target,
   counts by target, a small label preview, and the validation status. This is
   the canonical script for basis-size and label-validity checks.

``examples/coupling_coefficient_materialization.py``
   Builds the matching ``ye3t.couplings.plan`` and the ACE factorized
   coefficient schedule, then compares its independent basis count with the
   exact multiplicity. It prints the factorized term count, convention hash,
   and validation status before downstream runtimes consume the schedule.

``examples/symmetric_count_formula.py``
   Compares the independent symmetric fixed-content count formula used for a
   GE-PI dimension check with compiler counts for every allowed output ``L``.
   It does not download or execute an external code archive.

``examples/symmetric_density_factors.py``
   Applies a globally symmetric scalar coupler to supplied complete density
   multiplets. The single-factor radial basis can come from another package.

``examples/user_supplied_factors.py``
   Evaluates all three valid multiplicity coordinates for a nontrivial
   ``(2,1)`` Young and ``L=1`` coupler on user-supplied complex Condon–Shortley
   multiplets. Its shape is ``(a,t,M)=(3,2,3)``. Tests check a general
   rotation and nontrivial factor permutations. The full typed-orbit matrix
   remains a bounded reference that requires explicit opt-in.

``examples/symmetric_external_factors.py``
   Evaluates every ``(N)`` multiplet for an editable output ``L``. A repeated
   factor block uses occupation contraction, and the example checks factor
   permutation invariance.

``examples/antisymmetric_external_factors.py``
   Evaluates every ``(1^N)`` multiplet for an editable output ``L``. A repeated
   factor block uses a wedge contraction; the example checks sign changes and
   annihilation when its two ordered factor values become identical.

``examples/cauchy_supplied_density.py``
   Counts and applies a nontrivial local ``κ=(1,1)`` Cauchy path to a pair of
   role-resolved real tesseral densities. The output is an axial ``L=1``
   multiplet with globally symmetric factor character.

``examples/tagged_cauchy_linear_coordinates.py``
   Counts and compiles the scalar tagged-Cauchy physical coordinates for the
   rank-four, two-tag catalogue used by the linear fit in
   ``ye3t-methods/examples/quickstart/tagged_fit.py``. It reports the raw
   coordinate count and the independent physical-image count.

``examples/compile_scalar_ace_lammps_plans.py``
   Compiles four representative Ta scalar ACE coordinates (rank 3,
   homogeneous rank 4, the ``[4,4]`` rank-8 block pair, and homogeneous rank
   16) into exact coefficient tables and repeated-angular-block execution
   plans, adds a rank-3 coupled-product plan candidate, and writes them with a
   manifest under ``runtime.output_dir``. The directory must be new or empty.
   Edit ``CASES`` to change the coordinates and ``runtime`` to change the
   resource caps or output location.

``examples/exact_full_primitive_catalog.py``
   Shows the saved-object workflow: configure ``YE3TFixedContentBasis`` with
   integer rank keys, call ``build_basis()`` and ``calculate_pd()``, then inspect
   ``yb.representations``, ``yb.basis``, ``yb.primitive`` and
   ``yb.decomposable``. Rank 4 restricts rank-2 child Young partitions and
   compares angular caps for ``S_4:[3,1]``. Rank 6 adds a three-pair local
   sector and the full mixed ``S_6:[3,2,1]`` sector. Rank 16 uses the same
   global API. ``yb.primitive.basis.blocks(rank, cap)`` gives exact reduced
   coefficients, and ``vector(rank, cap, scope="local")`` selects one normalized
   primitive vector. Edit both rank maps to add a case; the shared
   ``representation["L"]`` sets the default target angular momentum.

``examples/symbolic_young_partition_catalogue.py``
   Expands symbolic parent-partition templates such as ``("N/2", "N/2")`` for
   the configured ``basis.ranks`` through
   ``ye3t.couplings.expand_partition_templates``, reports inapplicable and
   duplicate templates, and prints the positive O(3) multiplicities of each
   expanded partition through ``ye3t.couplings.counts_for_partitions`` for
   the fixed ``n_in_by_rank``/``l_in_by_rank`` content. This is the count-only
   entry point for choosing parent partitions before compiling a catalogue.

``examples/tagged_cauchy_image_catalogue.py``
   Builds the exact product algebra of a user-chosen shifted-Jacobi source set
   through ``ye3t.couplings.build_radial_species_product_record``, binds it to
   the Racah angular product plan from ``ye3t.couplings.racah_harmonic_product_plan``,
   and runs ``ye3t.couplings.count`` and ``plan`` on the resulting
   ``tagged_cauchy_image_request``. The report gives the raw label count and
   the exact physical-image dimension without materializing coefficients;
   ``runtime.materialize_coefficients`` additionally compiles the hash-bound
   artifact. Edit ``basis.single_factors.sources`` and
   ``basis.tensor_product.selected_raw_tag_counts`` for another source set. The public rung
   is bounded to tensor order 4 with two unit tags and l=1 primitive sources.

Benchmarks
~~~~~~~~~~

Bounded benchmark and diagnostic workflows live in ``examples/benchmarks/``.
They use the same ``cfg_ye3t``/``run_*`` shape and keep their output and
cache paths visible in the config.

``examples/benchmarks/benchmark_permutation_subduction_fastpath.py``
   Times the exact symbolic Young-subgroup subduction path against the numeric
   generator-nullspace path for each configured case, validating the numeric
   result by generator residuals and, for small cases, exact restricted
   projector comparison. ``runtime.benchmark.include_slow`` enables the
   larger cases.

``examples/benchmarks/coefficient_materialization_benchmarks.py``
   Writes cold and cached coefficient-materialization timing rows, a CSV
   table, and an optional plot for the rotation-only, permutation-only, and
   joint Young/O(3) families under ``runtime.benchmark.output_dir``. See
   :doc:`coefficient_benchmarks` for the row contents.

``examples/benchmarks/primitive_cache_policies.py``
   Compares primitive-cache lookup policies for one exact sector. The
   ``runtime.benchmark.case`` block chooses ``nin``, ``lin``, ``target_L``, and the
   maximum rank stored in the primitive cache.

``examples/benchmarks/symmetric_power_kernels.py``
   Compares optional folded symmetric-power kernels with reference evaluation.
   ``runtime.benchmark.catalog_cases`` lists powers and input ``L`` values; timing cases use
   ``batch``, ``warmups``, and ``runs`` only for bounded benchmarking.

See :doc:`tensor_products` for the product rules,
:doc:`operator_ir_and_schedules` for the IR/schedule planning boundary, and
:doc:`young_primitive_construction` for the direct Young and primitive quotient
APIs.
