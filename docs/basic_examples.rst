Basic YE3T Examples
===================

The examples in ``examples/`` show editable configurations for the core
``ye3t`` representation and tensor-product API. Short API demonstrations that
print counts, representation traces, or schedule metadata live in
:doc:`representation_snippets` as tested copy/paste snippets.

Joint Rotation and Permutation Coupling
---------------------------------------

For rank ``N``, YE3T couples ordered tensor factors into a parent sector

.. math::

   [\lambda]\otimes V_L, \qquad \lambda\vdash N.

The multiplicity count identifies independent coupling paths; each path has
Young-carrier and magnetic components. The complete config and runnable
factor-array example are in ``examples/user_supplied_factors.py``. Change its
``representation.parent`` and ``basis.cases_by_rank`` to select a different
Young type, angular momentum, and factor content. The symmetric and
antisymmetric factor examples cover their respective parent types for any
reachable output ``L``.

Programs that multiply already-coupled sectors can use the lower-level
``JointYoungCGProduct`` schedule API described in
:doc:`operator_ir_and_schedules`. Rank-increasing products use induction and
Littlewood--Richardson multiplicities; same-rank products under one diagonal
``S_N`` action use Kronecker multiplicities.

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
   factor angular momenta, and target ``L``. The
   ``examples/coupling_multiplicity_counts.py`` config places the target under
   ``representation.parent.L`` and the first two values in each ``basis.cases_by_rank`` entry;
   ``coupling_coefficient_materialization.py`` uses the same seven-section shape.

``rank``
   The product rank ``N`` of a feature. In atomistic language this is related
   to the body/correlation order of the descriptor term.

``nin`` / ``lin``
   Tuples that define one product sector. They must have the same length. Each
   factor pairs a non-angular channel ``eta_i`` from ``nin`` with an angular
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

Use :doc:`representation_snippets` for short copy/paste count and enumeration
queries. For an editable seven-section configuration, run
``examples/coupling_multiplicity_counts.py`` and change its parent Young type,
target ``L``, factor content, and factor angular momenta. The companion
``examples/coupling_coefficient_materialization.py`` compiles coefficients for
the same kind of request.

The snippet page also covers permutation/rotation representations,
exact change-of-group tensor products, structured basis coordinates, and
schedule metadata.

``examples/coupling_multiplicity_counts.py``
   Counts 162 scalar labels across ranks 1–8 through
   ``ye3t.couplings.count``. The example prints counts keyed by ``(rank, case)``
   and a small label preview. This is
   the canonical script for basis-size and label-validity checks.

``examples/coupling_coefficient_materialization.py``
   Builds the matching ``ye3t.couplings.plan`` and the ACE factorized
   coefficient schedule, then compares its independent basis count with the
   exact multiplicity. It prints the independent label and
   nonzero term counts for each rank 1–8 case.

``examples/symmetric_count_formula.py``
   Compares the independent symmetric fixed-content count formula used for a
   GE-PI dimension check with compiler counts for every allowed output ``L``.
   It does not download or execute an external code archive.

``examples/symmetric_density_factors.py``
   Applies a globally symmetric scalar coupler to supplied complete density
   multiplets across ranks 1–8, including 91 rank-eight coordinates. The
   single-factor radial basis can come from another package. It sets
   ``young_kappa="symmetric_only"`` and parent ``young_lambda="(N)"``:
   every permutation intermediate and the parent are symmetric. Angular
   intermediates keep every allowed value that couples to ``L=0``. This is
   the ACE-sector counterpart to ``user_supplied_factors.py``.

``examples/user_supplied_factors.py``
   Evaluates all valid ``(N-1,1)``, ``L=1`` coordinates across ranks 2–8
   on supplied complex Condon–Shortley multiplets. At rank eight its shape
   is ``(a,t,M)=(12,7,3)``. A batch of 16 inputs preserves Torch gradients.
   Tests check general rotations, inversion, adjacent factor permutations,
   and three-cycle orientation throughout the rank range.

``examples/symmetric_external_factors.py``
   Evaluates every ``(N)`` multiplet for an editable output ``L``. A repeated
   factor block uses occupation contraction; its ``symmetric_only`` Young
   intermediates are permutation invariant. The rank 2–8 test checks factor
   permutation invariance, rotation covariance, and natural parity.

``examples/antisymmetric_external_factors.py``
   Evaluates every ``(1^N)`` multiplet for an editable output ``L``. A repeated
   factor block uses a wedge contraction. The rank 2–8 test checks sign
   changes, rotation covariance, parity, and annihilation when two factors
   of the same type have identical values.

``examples/cauchy_supplied_density.py``
   Counts and applies every local Cauchy path to role-resolved real tesseral
   densities across ranks 2–8. The parent is globally symmetric with ``L=1``;
   the output parity follows the input factors. Vector triples combine with
   scalar-density powers, giving 36 independent multiplets at rank eight.
   A second rank-two case uses quadrupole inputs.

``examples/ordered_role_cauchy_factors.py``
   Evaluates all Cauchy coordinates for parents ``(N-1,1)``, ``L=1`` across
   ranks 2–8. Each factor supplies a role-by-magnetic array. The rank-eight
   result has shape ``(108,7,3)``. Tests cover rotations, inversion, and
   permutations across equal and different channel blocks.

``examples/tagged_cauchy_linear_coordinates.py``
   Counts and compiles the scalar tagged-Cauchy physical coordinates for the
   rank-four, two-tag catalogue used by the linear fit in
   ``ye3t-methods/examples/quickstart/tagged_fit.py``. It reports the raw
   coordinate count and the independent physical-image count.

``examples/compile_scalar_ace_lammps_plans.py``
   Exports all 20 scalar coordinates in the configured rank 1–8 contents,
   including three paths for the rank-eight ``[4,4]`` block pair. It also
   exports a homogeneous rank-16 coordinate and a rank-three coupled-product
   plan candidate. Edit ``basis.cases_by_rank`` to choose contents and
   ``basis.additional_coordinates`` for constructive high-rank coordinates.
   ``runtime.output_dir`` must be new or empty. The manifest records compiler
   labels, exact counts, coordinate validation, and execution-plan hashes.
   Physical source bindings and fitted weights are supplied by ye3t-methods.

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
   Counts and compiles the scalar tagged physical image across ranks 1–8.
   Edit the radial/angular caps, source-block partitions, angular patterns,
   and tag counts in the visible config. The raw counts are
   ``2, 6, 18, 81, 60, 90, 126, 168``. Exact physical-image dimension is
   obtained during coefficient compilation: ``1, 1, 3, 14, 15, 26, 39, 56``
   by rank, totaling 155 coordinates. Higher ranks combine scalar
   powers with vector pairs; rank four includes two radial channels.
   Counts and planning run by default. Set
   ``runtime.materialize_coefficients=True`` to compile the full physical image.
   Each rank selects the same physical coordinates when compiled separately
   or together in this catalogue; rank four keeps the fitting example's
   14 coordinates.

``examples/extras/tagged_cauchy_rank4_product_algebra.py``
   Builds and inspects the exact shifted-Jacobi source-product record and
   Racah angular plan. This specialized rank-four calculation has six raw
   labels and a four-dimensional physical image. Coefficient compilation is
   optional in its config.

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
