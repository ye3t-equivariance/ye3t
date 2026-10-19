Sources and executable model stages
===================================

``YE3TSourceRealization`` records how tensor factors are supplied. Its
``bind_outputs`` method associates the source with complete carrier layouts,
canonical fixed-map identities, argument bindings and derivative capabilities.
``compile_carrier_arena_plan(..., model_stage_records=...)`` validates those
bindings and records their dependencies in ``model_stage_program``.

The application owns geometry, source evaluation and learned operations.
``ye3t`` owns multiplicities, coupled labels, representation maps and the
structural conditions on their composition. The private ``ye3t-ace``
``carrier_graph`` examples provide an executable density and supplied-factor
workflow using these records.

Source rank and emitted action
------------------------------

For a source of rank :math:`N`, the input word identifies every factor
:math:`f=1,\ldots,N`. A selected source image may carry a different action
after a restriction or marginal sum. The fixed two-role source, for example,
has four construction factors and an emitted auxiliary :math:`S_2` action.
Its construction parents, exact analysis map and emitted action remain separate
fields in the source record.

An action specification gives its group, degree, scope and generator
convention. An action ID identifies the aligned acting domain; the carrier key
specifies the Young type, angular momentum and parity within that domain.
Consequently an invariant readout retains the action ID and changes its
representation to :math:`(N),L=0,p=+1`.

Source bindings include:

* ordinary inclusive density, restricted to the symmetric Young parent;
* complete supplied spherical factors with their ordered channel word;
* supplied Cauchy factors with a retained role dimension;
* selected marginals with exact restriction/analysis maps.
* explicit Phi factors tied to a root, neighbor atom and periodic image.

All output layouts retain the full axes :math:`(a,t,M)`, following
:math:`\alpha=(\lambda,L,a,t,M)` in the paper. The tableau dimension follows
the Young representation and the magnetic dimension is :math:`2L+1`.

Operations and dependencies
--------------------------------

Executable stages name their operation, complete carrier inputs and outputs,
external arguments, auxiliary tensors, trainable parameter identities and
fixed maps. Operation kinds distinguish sums, channel maps, invariant gates,
invariant reductions, scalar context injection and compiler-owned polynomial
couplings. Channel maps preserve the representation and act on its multiplicity
or learned channel axis. Polynomial terms must match their canonical coupling
plan, including normalization and input/output types.

``source_carrier_ids`` describes the direct representation map. The operation's
``input_bindings`` also records conditioning values and auxiliary geometry.
Both contribute to lifetime analysis. Each executable stage produces its
outputs, including a first stage driven entirely by external positions.
Auxiliary lifetimes keep their row domains and derivative dependencies separate
from carrier coordinate storage. Loading a plan rechecks these dependencies.

Derivative modes identify value evaluation, vector-Jacobian products and
double backward. The latter is required when a force loss differentiates through
position derivatives and trainable source parameters. These records describe
required behavior; runtime tests compare actual values and derivatives against
independent calculations.

Physical occurrences and disjoint products
----------------------------------------------------

An atomistic induction stage combines two disjoint sets of physical factors.
For child ranks :math:`N_l,N_r`, its inputs contain all
:math:`\binom{N_l+N_r}{N_l}` placements in the order supplied by
``couplings.compile``. Each placement binds both child tensors to subsets of
the same parent occurrence. Shared learned parameters retain distinct left
and right argument identities.

The application records the root, atom and periodic-image identities, child
row indices, parent roots and row offsets. The compiler validates the product
type, complete placement order, carrier layouts and argument dependencies.
The physical binding and canonical coupling plan have separate content hashes.
This preserves the distinction between an abstract representation map and its
evaluation on particular atoms.

Occurrence counts depend on geometry. A topology stage supplies integer row
tables through ``row_count_outputs``; dependent carrier stages select their
table through ``row_count_key``. The executor checks carrier dimensions against
the current table. Integer topology can be reused while its input tensors are
unchanged. Radial values, angular values and learned features remain connected
to positions and parameters on every evaluation.

The private explicit-Phi example uses injective depth-one stars. It couples
every child placement, retains all tableau and magnetic components, and forms
real Hermitian invariants on each occurrence before summing at its root.
Learned channel maps can compress the intermediate channels. The resulting
model features carry the selected symmetries; this architecture makes no claim
that they form the complete independent polynomial basis of the primitive
source.

The application graph uses functional PyTorch operations and factorized core
source plans. Complete carrier products can lower the canonical nonzero Young,
angular, channel and placement entries to deterministic sparse CUDA kernels.
Complex inputs retain interleaved real/imaginary coordinates and their original
phase convention. Native products provide derivatives through order two.
``bind_carrier_product_torch(..., backend="torch")`` selects the factorized
implementation explicitly; ``backend="native"`` requires CUDA/Triton execution.
The default ``auto`` mode bounds sparse expansion to 262144 terms and uses the
factorized evaluator for larger schedules. This budget controls additional
schedule storage; it imposes no restriction on valid coupled coordinates.
Runtime reports identify actual dispatch, schedule size and selection reason.

LAMMPS communication and neighbor ownership require an application runtime.
