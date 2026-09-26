Message-Passing Equation Coverage
=================================

This ledger maps the equations in
:download:`message_passing_mathematical_standard.md
<message_passing_mathematical_standard.md>` to current implementation evidence.
``Reference`` means a mathematically explicit implementation exists but is not
a production-performance claim. ``Under validation`` preserves the source
document's ``PROPOSED`` or ``VERIFY`` status.

Implementation and validation records use the MP-EQ-00--MP-EQ-30 numbering
below.

.. list-table::
   :header-rows: 1
   :widths: 10 23 25 42

   * - Equation
     - Contract
     - Current status
     - Primary implementation/evidence
   * - MP-EQ-00
     - Linear maps preserve ``(N, lambda, L)``
     - Axis scope enforced
     - Exact carrier layouts and channel-only maps act on the declared trivial
       axis and never on tableau or magnetic components
   * - MP-EQ-00a
     - Cross-``L`` interaction uses angular coupling
     - Implemented
     - Cached complex/real CG tables and exact joint binary products
   * - MP-EQ-01
     - Orbit-closed fixed-content carrier
     - Implemented
     - ``fixed_content`` orbit/stabilizer reports and global-coupler plans
   * - MP-EQ-02
     - Commuting slot and rotation actions
     - Validated at representative ranks
     - Global-coupler permutation and rotation action tests
   * - MP-EQ-03
     - Raw product basis and coordinate vector
     - Implemented
     - Fixed-content raw-index conventions and typed source-coordinate layouts
       keep adapted tableau/path labels out of the raw index
   * - MP-EQ-04
     - Repeated-block decomposition
     - Implemented reference
     - Repeated-content decomposition and block multiplicity records
   * - MP-EQ-05
     - Block-to-parent permutation induction
     - Implemented
     - Young-subgroup LR/subduction compiler
   * - MP-EQ-06
     - Block-to-parent angular restriction
     - Implemented
     - Angular path counting and cached CG schedules
   * - MP-EQ-07
     - Full joint decomposition, multiplicity, and dimension identity
     - Under validation
     - ``ye3t.couplings.count``, block ``kappa/Lambda`` reports, global
       couplers, and exact low-rank dimension audits; theorem status remains
       ``VERIFY``
   * - MP-EQ-08
     - Coupled synthesis basis and ``C^\dagger`` analysis
     - Implemented and native CPU/CUDA validated
     - Versioned synthesis tables, global-coupler evaluators, and native
       source-analysis forward/adjoint operators
   * - MP-EQ-09
     - Orthonormality, completeness, and selected projectors
     - Validated where the retained table scope permits
     - Coupler certificates, matrix-unit tests, and complete low-rank
       projector/dimension audits
   * - MP-EQ-10
     - Joint intertwining
     - Validated at representative ranks
     - Non-vacuous permutation and rotation action tests
   * - MP-EQ-11
     - Complete-tree recoupling
     - Implemented reference
     - Balanced coupled-sector recouplers; pruned/nonlinear trees remain
       architecture-distinct
   * - MP-EQ-12
     - General equivariant linear map
     - Exact carrier and axis scope enforced
     - Logical ``[trivial_axis, tableau_t, magnetic_M]`` layouts permit
       mixing of declared multiplicity, source/content, path, radial,
       chemical, and learned-channel copies only after they share one exact
       carrier
   * - MP-EQ-13
     - Primitive one-interaction feature
     - Implemented in ``ye3t-ace``
     - Radial/angular site-basis evaluators with scalar species/charge
       conditioning
   * - MP-EQ-14
     - Ordinary and edge-filtered lifted densities
     - Implemented
     - Ordinary ``A`` and role-resolved ``A_s`` materialization with
       rotation-scalar, edge-data-dependent filters
   * - MP-EQ-15
     - Exact post-pooling radial-role map
     - Interface requirement recorded; general public path pending
     - Current role filters are evaluated before pooling. A general serialized
       ``K_s^(l)`` path must prove the radial-span relation before promotion
   * - MP-EQ-16
     - Exact density-product expansion
     - Implemented reference
     - Role-orbit product evaluators retain repeated-neighbor terms
   * - MP-EQ-17
     - Explicit rooted-motif realization
     - Implemented reference
     - ``cluster_phi`` motif materialization and decorated automorphism reports;
       carrier-generic optimized enumeration remains open
   * - MP-EQ-18
     - Unified raw source coordinates
     - Typed reference and physical role binding implemented
     - ``ordinary_density``, ``lifted_density_roles``, and ``rooted_motif``
       records lower source-specific coordinates without adding adapted-basis
       labels to the raw index
   * - MP-EQ-19
     - Density selection and role covariance
     - Implemented and validated at representative ranks
     - ACE carrier guards, nontrivial-sector rejection, ``A_s`` slot
       permutation/value/VJP tests, rank-3 parent ``(2,1)`` tableau action, and
       repeated-role collapse tests
   * - MP-EQ-19b
     - Typed pair-dependent source families
     - Contract recorded; runtime generalization pending
     - Pair-specific source metadata exists in the application layer, while a
       compiler-owned typed family spanning distinct bond cutoffs, radial
       bases, and role counts remains under implementation
   * - MP-EQ-19c
     - Pair-dependent radial filters and role counts remain intertwiners
     - Implemented for scalar pair filters; general role-count validation
       pending
     - Species-pair radial parameters act on the trivial channel axis. Exact
       tests for heterogeneous role counts and every retained carrier remain
       required before promotion
   * - MP-EQ-19d
     - Role occupancy determines the physical source image
     - Exact count reference validated; physical ``L_v`` validation pending
     - Stabilizer/Kostka and LR counts agree at rank six, but the serialized
       pair-dependent source image, automorphism reduction, and angular
       intersection remain ``VERIFY`` work
   * - MP-EQ-19e
     - Formal rank may exceed the number of distinct role labels
     - Compiler contract recorded; optimized runtime pending
     - Rank-three plus rank-three LR growth provides the targeted rank-six
       construction without inventing extra physical role labels
   * - MP-EQ-19f
     - Pair-source/rank-growth validation obligations
     - Under validation
     - Exact dimension/intertwining, role reordering, source-image, angular
       intersection, value, VJP, and HVP records are required for each
       promoted heterogeneous-role path
   * - MP-EQ-20
     - Rank-additive LR induction
     - Implemented; rank-6 targeted parent path validated
     - ``RankAdditiveInductionPath``, coupled-sector recouplers, direct cached
       parent subduction, and the dedicated
       ``rank_additive_lr_induction`` execution opcode
   * - MP-EQ-21
     - Same-rank Kronecker product
     - Finite exact tables and scalar-gate model runtime implemented; general
       output-carrier runtime pending
     - ``SameRankKroneckerPath``, character counts, finite sparse
       intertwiners/evaluators, the separate ``same_rank_kronecker`` opcode,
       and channel-preserving ``[channel, tableau, M]`` scalar gates
   * - MP-EQ-22
     - Angular binary coupling
     - Implemented reference, native CPU, and packed CUDA
     - Typed angular DAGs, sparse pair-CG tables, complex/real-tesseral paths,
       and one-launch sparse bilinear tables
   * - MP-EQ-23
     - Exact joint binary coupler
     - Implemented with static schedules and packed CUDA evaluation
     - Joint Young-CG schedules apply ``J_p^\dagger`` and multiplicity-only
       weights through packed forward/adjoint/double-backward tables
   * - MP-EQ-24
     - Source-specific assembly and binary-tree recursion
     - Typed source-specific references and accelerated nodes under validation
     - Explicit ordinary-density, lifted-role, and rooted-motif ``L_v`` maps
       precede analysis; related placements are serialized rather than
       discovered by runtime rank testing. No universal source-free reduction
       is claimed
   * - MP-EQ-25
     - Rank-graded hidden state
     - Implemented with exact carrier/axis contract
     - ``(N, lambda, L, convention_id)`` keys and separate trivial,
       tableau, and magnetic axes
   * - MP-EQ-26
     - Source-to-message analysis
     - Application integration implemented, broader paths under validation
     - Physical ``A_s`` role tuples pass through source-specific
       ``L_v -> C^\dagger -> W`` with packed channel-only hidden mixing
   * - MP-EQ-27
     - Carrier-aligned aggregation
     - Implemented for current exact-carrier state operations
     - Addition requires identical carrier signatures, basis conventions, and
       aligned output-channel bases even when flat widths match
   * - MP-EQ-28
     - Equivariant update
     - Under validation
     - Compiled path-coupled hidden updates preserve exact direct-sum carrier
       layouts; broader nonlinear tree-node coverage remains
   * - MP-EQ-29
     - Architecture-neutral master layer
     - Under validation, not production fast
     - K2SO4/H2O mixed-sector reference runtime and packed CUDA binary/source
       nodes; complete physical-source rank-6 layer fusion remains open
   * - MP-EQ-30
     - Scalar gate
     - Implemented
     - Per-copy gates broadcast uniformly over complete tableau and magnetic
       blocks with generator-equivariance tests

Validation Coverage Required by the Updated Standard
----------------------------------------------------

- Exact dimensions, intertwining, orthonormality/projectors, completeness, and
  complete-tree recoupling have representative low-rank records.
- Linear-type safety rejects cross-``N``, cross-``lambda``, and cross-``L``
  weights.
- Rank-additive LR and same-rank Kronecker schedules have separate types,
  opcodes, tests, and failure guards; substituting one mode for the other is
  rejected.
- Ordinary-density and role-resolved selection tests are present.
- Density tests retain repeated physical-neighbor indices, whereas explicit
  rooted-motif tests require the declared injective/automorphism semantics.
- Neighbor-list reordering and scalar charge typing have dedicated application
  tests; global charge conservation remains a separate model constraint.

The performance program must update this ledger when a reference row becomes a
validated native path. It must not change mathematical status solely because a
kernel is faster.
