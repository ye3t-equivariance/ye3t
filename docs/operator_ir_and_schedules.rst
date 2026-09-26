Operator IR And Schedules
=========================

The core runtime has three public levels:

``E3OperatorIR``
   A symbolic exact-product operator built from ``n_in``, ``l_in``, and a
   target angular channel ``L_R``.

``ExactSchedule``
   A grouped lowering of the IR into deterministic path segments, flat integer
   metadata, and optional materialization decisions.

``compile_ye3t_operator``
   A convenience entry point that builds or accepts an IR/schedule and returns
   a callable native runtime module.

The IR keeps the mathematical selection rules visible.  Its paths are exact product
columns and its packed blocks group paths by angular signature
``(left_L, right_L, out_L)``.  Primitive quotient paths use ``left_L=-1`` and
``right_L=-1`` until they are explicitly retained by
``include_target_primitive`` or ``primitive_first``.

.. code-block:: python

   from ye3t.api import build_exact_schedule, build_operator_ir

   ir = build_operator_ir(
       (1, 1, 1),
       (1, 2, 1),
       2,
       primitive_first=True,
       include_target_primitive=True,
       max_internal_L=3,
   )

   schedule = build_exact_schedule(
       ir,
       optimization_policy="auto",
       backend="pytorch",
   )

   print(ir.num_paths)
   print(schedule.flat_metadata.metadata["group_count"])
   print(schedule.materialization_plan.as_dict())

Schedule Options
----------------

``ScheduleOptions`` controls the shape of the lowered schedule without
changing the exact target sector:

``static_path_elimination``
   Drop primitive-only paths unless the caller requested primitive output
   support.

``max_paths_per_segment``
   Keep only a bounded number of paths per angular segment.  This is useful for
   smoke tests and diagnostics.

``max_segment_product_order``
   Drop paths whose decomposable-product order exceeds a chosen cap.

``primitive_segments_first``
   Sort primitive segments before higher-order decomposable products.

The materialization policy is separate from those structural options.  Use
``optimization_policy="off"``, ``"auto"``, or ``"aggressive"`` to ask
``with_schedule_planning`` for deterministic recompute/materialize decisions.
The current policy is a planning recommendation; it records reuse, estimated
compute, estimated bytes, and a reason for each decision.

Multi-Target Compilation
------------------------

Passing ``L_R="all"`` or an explicit list to ``compile_ye3t_operator`` builds
one target-sector schedule per allowed output irrep and returns a merged
multi-operator.  ``max_target_L`` can cap that expansion.

.. code-block:: python

   from ye3t.api import compile_ye3t_operator

   compiled = compile_ye3t_operator(
       (1, 1, 1),
       (1, 1, 2),
       L_R="all",
       max_target_L=2,
       optimization_policy="auto",
   )

Central Young--E3 Coupler Records
---------------------------------

``CompileGlobalYE3TCouplers`` builds the shared finite-rank Young--E3
coefficient artifact used by descriptor and message-passing planning.  The
record separates:

``YoungSubductionMap``
   The subgroup-adapted restriction map from the induced child/block basis into
   the requested global Specht sector.

``YoungInductionCoupler``
   The induced-basis, coset/shuffle, Frobenius-lift, and normalization metadata
   used before the target Specht projection.

``AngularCGMap``
   The SO(3)/O(3) angular selection and coefficient metadata, including
   factorized binary CG tables for n-ary balanced/left/right schedules.

``JointYoungE3Coupler``
   The composed Young/block/angular coefficient record, including complete
   alpha labels ``(mu, Lambda, beta, gamma, pi)``, sparse coefficient tables,
   factorized coefficient tables, backend provenance, and a validation
   certificate.

``JointYoungE3Coupler.to_dict()["component_inventory"]`` summarizes which
coefficient-object families are present: complete labels, block maps,
subduction maps, induction/coset couplers with Frobenius-lift metadata,
angular maps including parity metadata, sparse tables, factorized tables, and
normalization keys.  The
inventory is a downstream completeness check; the individual maps and
certificate remain the source of the tested finite-rank identities.

``RepeatedContentImageMap`` records the exact projector image used when a
balanced tree split separates repeated full-channel content, for example the
``1123`` split ``((1,2),(1,3))``.  It emits an orthonormal image isometry, the
corresponding projector, sparse tables, hashes, split metadata, and a
``validation["passed"]`` summary.  The validation is representation-level: it
checks idempotency, image rank, agreement with the direct global coupler image,
and the absence of descriptor-level SVD or numerical descriptor reduction.
``CompileBalancedTree`` accepts an optional ``metadata["balanced_content_split"]``
with two child index blocks so tests and examples can state the split being
certified; otherwise it records the default midpoint split.  The emitted image
map also records whether repeated channel support crosses the selected split.
The current raw-path table is the exact projected induced-basis Gram
``G=P^T P`` for the emitted Young image projector ``P=S S^T``.  Its pivot
columns give the recorded exact rank profile.  This is a representation-level
Gram table over induced-basis candidate paths; a future lowerer may still
factor earlier local path graphs before this global image projector is formed.
The balanced-tree compilation also emits ``balanced_tree_node_ledger`` records
for the recursive binary tree.  Each internal node records its slot indices,
child split, child content, repeated labels crossing the split, whether image
reduction is required, and whether the current compiler materialized that
node's image map.  At present the root global image map is materialized when
needed.  Non-root scalar-trivial repeated-content nodes are also materialized
as exact local image maps, while unsupported intermediate sectors remain
recorded as requirements for the future local tree lowerer.

The current certificate checks small-rank dimension sums, projector
idempotency, pairwise projector orthogonality, sum of projectors on the induced
space, subgroup generator/multiplicity residuals, induction/coset counts,
trivial-target normalized orbit sums, angular admissibility, angular
coefficient normalization, parity, and coefficient hashes.  These checks
validate the emitted coefficient artifact for the tested finite cases.  They
do not by themselves mean a descriptor or trainable message-passing model has
realized geometry carriers, hidden updates, readouts, or forces.

``CompileGlobalYE3TCouplerFamily`` expands
``target_permutation="full_irrep_decomposition"`` into all reachable concrete
Young target sectors and carries the same dimension-sum and orthogonality
report across the family.

Shared spec files may either use the canonical nested rotation target

.. code-block:: yaml

   target_rotation:
     L_R: 2
     M_R_values: [-2, -1, 0, 1, 2]
     parity: even

or the workflow-friendly top-level aliases ``target_L_R`` / ``L_R``,
``target_M_R_values`` / ``M_R_values``, ``target_parity`` / ``parity``, and
``target_rotation_group`` / ``rotation_group``.  Parsing normalizes both forms
to the same ``YE3TSpec.target_rotation`` object, validates that every requested
``M_R`` lies in ``[-L_R, L_R]``, and writes the canonical nested form from
``to_dict()``.  This keeps homogeneous config inputs available across
``ye3t`` and ``ye3t-ace`` without changing the mathematical target sector.
Carrier-specific runtime options should be placed in ``carrier_options``.  For
example, A_s role-coordinate settings such as
``role_coordinate_policy="role_resolved"`` or ``"commutative_density"`` are
parsed there, while the Young/rotation labels remain in the main spec fields.
``YE3TSpec.carrier_policy_report()`` returns the corresponding config-level
A_s role-coordinate consistency report.  It is a selection-rule diagnostic,
not a descriptor runtime validation.

Balanced Message-State Schedule Records
---------------------------------------

``CompileBalancedYE3TMessagePassingSchedule`` consumes
``BalancedYE3TMessageStateSpec`` and emits balanced-tree sector schedules plus
the central coupler certificates used by the reference evaluator.  The schedule
is rank-graded: each retained hidden sector carries its own slot group
``S_N`` for the corresponding rank/content.  The default pair-product schedule
combines disjoint child slot sets and grows rank through induction/LR
multiplicities.  It is not a same-rank nonlinear product of two features that
already transform under one common ``S_N`` action; such products require a
separate Kronecker-coupling implementation before they can be claimed as part
of the YE3T message-passing runtime.  Shared message-state specs record this
choice as ``rank_coupling_mode``: ``"rank_additive_induction"`` is the
implemented-under-validation schedule mode, while ``"same_rank_kronecker"`` is
carried as an unsupported planned mode.
``BalancedYE3TRankCouplingPolicy(...)`` and schedule payloads expose the same
boundary as structured metadata.  For rank-additive schedules the policy names
induction from disjoint slot sets and Littlewood--Richardson multiplicities;
for ``"same_rank_kronecker"`` it reports
``implemented_in_balanced_schedule=False`` and
``required_backend="same_rank_kronecker_coupler"`` before schedule
compilation.
The schedule
record now includes ``input_value_specs``: one entry per scheduled hidden
sector, with the expected input feature-axis width, expected output coefficient
width, backend, and coefficient-table kind.  Generic sectors report the sparse
subduction/coefficient matrix shape; exterior/sign sectors report the exterior
sign-vector basis size because that is the table consumed by the reference
evaluator.  The direct exterior/sign evaluator keeps its historical default of
returning the contracted tensor, while balanced message-sector evaluation uses
the singleton-axis form so the hidden-state feature axis remains explicit; the
metadata records ``coefficient_axis_kept`` and ``coefficient_axis_status``.
``BalancedYE3TMessageStateSpec`` also carries optional
``input_Ls_by_content`` records, for example
``{"content": [1, 1], "input_Ls": [1, 1]}``, so JSON/YAML and Python configs
can specify the input angular labels used when compiling each content sector.
The low-level compiler still accepts an explicit ``input_Ls_by_content``
override for reference tests.
For exterior/sign sectors with repeated content labels, the emitted metadata
also carries ``content_label_wedge_vanish_report`` and
``carrier_realization_required_to_enforce_vanish``.  The sign-vector reference
evaluator is an alternating linear functional on caller-supplied permutation
basis values; it does not silently zero arbitrary raw inputs.  Vanishing of
duplicate one-particle exterior factors is a carrier-realization condition.

The reference evaluator and direct-sum state view remain validation and wiring
tools.  They apply scheduled coefficient tables and package sector outputs with
task/readout metadata, but they do not perform recursive message aggregation,
nonlinear updates, scalar energy readout, force differentiation, or fermionic
operator assembly.
``BalancedYE3TMessageStateReferenceView.to_tensor_container()`` exposes the
implemented-under-validation direct-sum hidden-state tensor container and
layout checks.  The container is a tensor layout object, not a recursive
message update.
``ApplyBalancedYE3TMessageSumAggregation`` and
``BalancedYE3TMessageStateTensorContainer.apply_message_sum_aggregation(...)``
apply incoming-edge sum aggregation along a node/site/particle axis while
leaving the Young--E3 feature axis and sector slices unchanged.  This is the
implemented aggregation primitive under validation; recursive aggregation with
sector projection between layers is still a missing full-runtime stage.
``ApplyBalancedYE3TSectorwiseScalarUpdate`` and
``BalancedYE3TMessageStateTensorContainer.apply_sectorwise_scalar_update(...)``
apply one scalar intertwiner per retained hidden sector.  This preserves the
direct-sum sector layout and does not mix representation coordinates between
sectors.  It is a narrow hidden-state update primitive under validation, not
message aggregation or a general multiplicity-resolved trainable update.
``ApplyBalancedYE3TChannelLinearUpdate`` and
``BalancedYE3TMessageStateTensorContainer.apply_channel_linear_update(...)``
apply a linear map along an explicit channel or multiplicity axis while
preserving the Young--E3 feature axis and sector slices.  This is only valid for
axes carrying trivial group action; it is not a general map on representation
coordinates.
``ApplyBalancedYE3TPairProductMerge`` and
``BalancedYE3TMessageStateTensorContainer.apply_pair_product_merge(...)`` take
one selected sector from each of two direct-sum containers, form the explicit
pair-product input basis along the Young--E3 feature axes, and project it with
a scheduled output-sector coupler.  This is the first implemented pair-merge
reference primitive with sector projection; it is still a finite checked
operation, not a recursive layer loop, nonlinear update, energy/force model, or
fermion-operator assembly.
``ApplyBalancedYE3TPairProductReferenceMessageLayer`` and
``BalancedYE3TMessageStateTensorContainer.apply_pair_product_reference_message_layer(...)``
compose incoming-edge sum aggregation of the right/message state with that
single scheduled pair-product projection.  This is a one-step checked
reference layer for wiring balanced product merges into message-state tests; it
is still not a recursive layer stack, nonlinear update, general
multiplicity-resolved trainable map, readout, force model, or fermion-operator
assembly.
``ApplyBalancedYE3TRecursivePairProductReferenceStack`` and
``BalancedYE3TMessageStateTensorContainer.apply_recursive_pair_product_reference_stack(...)``
iterate compatible one-sector versions of the same sum-then-pair-product
reference layer.  The stack records the layer history and the scheduled
sector projection applied at each step, but it remains a reference stack under
validation: it does not add nonlinear hidden updates, trainable
multiplicity-space maps, task readouts, force validation, or
fermion-operator assembly.
``BalancedYE3TRecursivePairProductReferenceStackLayer.initialize(...)`` wraps
the same compatible schedule sequence in a reusable object.  It has no
trainable parameters by default.  When ``use_scalar_gains=True`` it applies one
scalar intertwiner after each projected pair-product layer; with
``trainable_scalar_gains=True`` those gains are parameter objects.  This does
not mix representation coordinates or multiplicity spaces, and does not change
the status of the full balanced message-passing runtime.
``ApplyBalancedYE3TChannelLinearReferenceMessageLayer`` and
``BalancedYE3TChannelLinearReferenceMessageLayer.initialize(...)`` compose the
incoming-edge sum with that channel-axis linear map and provide the
corresponding parameterized reference object.
``ApplyBalancedYE3TReferenceMessageLayer`` and
``BalancedYE3TMessageStateTensorContainer.apply_reference_message_layer(...)``
compose the incoming-edge sum with the sectorwise scalar intertwiners.  This is
the implemented linear reference layer on direct-sum hidden states; it still
does not perform a balanced product merge, nonlinear update, energy/force
evaluation, or fermion-operator assembly.
``BalancedYE3TReferenceMessageLayer.initialize(...)`` creates a small
parameterized object with one scalar gain per retained sector and a
``parameters()`` hook for trainable-gain experiments.  It is still the same
linear reference layer, not a full balanced recursive model.
``ApplyBalancedYE3TReferenceLinearReadout`` and
``BalancedYE3TMessageStateReferenceView.apply_linear_readout(...)`` add a
caller-weighted linear readout map on top of that packaged reference state.
For scalar aggregations such as ``site_sum`` and ``global_sum`` they reduce the
linear site/state values according to the shared readout spec.  This is still a
reference map on scheduled coefficient outputs, not a trainable balanced
recursive message-passing layer.
``ValidateBalancedYE3TAtomicScalarReferenceReadout`` checks that a caller-built
atomic scalar reference readout has pre-aggregation site values that relabel
covariantly and an aggregated scalar that is unchanged under the supplied
node/site permutation.  ``ValidateBalancedYE3TFermionExchangeReferenceReadout``
checks the requested exchange sign on a caller-built fermion/operator
reference readout.  These helpers validate the emitted reference tensors for
the supplied group action; force checks, nontrivial rotation tests, and full
operator-symmetry validation remain separate obligations.
``ValidateBalancedYE3TAtomicScalarReferenceReadoutHiddenJacobian`` additionally
checks the autograd Jacobian of the scalar reference readout with respect to
the packaged direct-sum hidden-state tensor under the same node/site relabeling.
This is a hidden-state tensor check only.  It is not a geometry-carrier force
check, finite-difference force validation, or MLIP suitability claim.
All of these schedule/reference objects now expose
``runtime_scope="balanced_coefficient_schedule_reference_only"`` and
``missing_recursive_message_passing_stages`` so downstream descriptor/model
code can check the difference between a certified coefficient schedule and a
trainable recursive message-passing runtime.
The balanced schedule also records ``task_family``, ``readout_target``,
``readout_selection_rule_status``, and ``required_runtime_validation``.  Atomic
scalar schedules list energy/force invariance and covariance checks that remain
outside the schedule compiler; fermion/operator schedules list exchange-sign,
operator-symmetry, and exterior-fast-path equivalence checks.  Each sector
schedule also exposes the ``balanced_tree_node_ledger`` from the underlying
balanced compiler plus aggregate ``local_repeated_content_image_maps`` and
``nonroot_image_map_requirements``.  These records distinguish local
repeated-content reductions already materialized for scalar-trivial nodes from
remaining local reductions that a future recursive tree lowerer must
materialize before the schedule can be treated as a full balanced
message-passing runtime.
The corresponding ``runtime_validation_status`` remains
``not_performed_by_balanced_schedule_compiler`` until those model-level tests
are run.

Balanced-tree compilation also records exact recoupling evidence for checked
small ranks.  In addition to comparing the left/right/balanced image
projectors, the certificate provenance includes overlap matrices
``C_comparison^T C_balanced`` between the orthonormal subduction bases.  The
checks require these overlaps to be orthogonal and to reconstruct the balanced
subduction basis from the left/right bases.  This is a representation-level
recoupling check on the emitted Young/Specht maps; it is separate from later
descriptor carrier evaluation and trainable message updates.

Schur-Weyl Guided Tree Backend
------------------------------

``compile_schur_weyl_guided_tree_product_from_coupler`` consumes the central
global coupler record and carries its certificate hash, backend, target
partition metadata, and angular/parity target into the runtime-tree report.
The backend certificate now records whether the global coupler certificate was
present and passed, whether the coefficient hash is present, whether node
dimensions match the cached Schur-Weyl/Specht plans, and whether requested
labels are missing or duplicated.
When the runtime tree is built through ``CompileBalancedTree(...,
build_runtime_tree=True)``, the backend provenance also receives the balanced
node ledger, local repeated-content image maps, and remaining non-root image-map
requirements.  The backend certificate checks that this balanced compiler
metadata is present when requested and that materialized local image maps
validate.

For roots whose subgroup-adapted tree has one symmetric-group factor, the
adapter also passes the global target partition into the tree selector and the
certificate records ``root_permutation_target_status`` as
``"enforced_single_root_symmetric_group_factor"``.  The report includes the
enforced partition, ``single_root_factor_target_enforced=True``,
``full_global_induction_coset_lift_required=False``, the root subgroup factor
count/multiplicities, and the actual root output partition signatures.  For
multi-factor fixed-content roots, the tree selector can enforce only the
subgroup-local Specht restriction; the full global ``S_N`` target still requires
the global induction/coset lift.  Those cases now report
``root_permutation_target_directly_enforceable=False``,
``full_global_induction_coset_lift_required=True``, and
``root_permutation_target_direct_enforcement_kind="requires_global_induction_coset_lift"``.
They are reported as a remaining global assembly stage rather than as a
completed global Young-sector runtime.

Backends And Adapter Summaries
------------------------------

The native PyTorch runtime is the reference execution path.  Backend names such
as ``"triton"`` and ``"openequivariance"`` currently add lowering metadata and
use specialized paths only when supported; otherwise the native runtime remains
available.

The accelerator bridge helpers are intentionally summary-level.
``export_to_cuequivariance_ir`` records a segmented tensor-product bridge
summary. It is useful for comparing selection rules with an external
accelerator stack, but it is not a full external-code generator.

The tested :doc:`representation_snippets` page includes flat schedule
metadata and materialization summaries for ``off``, ``auto``, and
``aggressive`` policies.
