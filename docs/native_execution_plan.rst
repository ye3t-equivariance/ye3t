Native Execution Plans
======================

Status: implemented under validation.

``YE3TExecutionPlan`` is the versioned boundary between representation
compilation and numeric execution. It records:

- exact carrier keys ``(N, lambda, L, convention_id)``;
- logical ``[channel_or_multiplicity, tableau_t, magnetic_M]`` layouts;
- typed ordinary-density, lifted-role, or rooted-motif source semantics;
- sparse source-placement maps ``L_v``;
- synthesis coefficient tables ``C``;
- typed factorized angular DAG nodes, binary CG tables, root-path order, and
  coset-placement order;
- forward, reverse, and second-order instruction schedules;
- convention, coefficient, plan, certificate, and provenance hashes.

Feature evaluation applies ``C^\dagger L_v``. Learned channel maps are not
part of coupling instructions and may act only on the channel/multiplicity
axis.

.. note::

   Passages on this page that mention ``ye3t-methods``,
   ``SiteBasisV2``, ``ExteriorPowerEvaluator``, ``YE3TMessagePassing``, or
   named force-training workloads are downstream integration notes. They
   record how the separate application package
   consumes the runtime operators described here and which measured
   workloads motivated each operator. None of those objects is part of the
   ``ye3t`` API, and the quoted timings are local engineering measurements on
   the validation hardware, not portable performance claims.

Python interface
----------------

Use ``ye3t.couplings.compile_execution_plan`` for explicit records, or
``execution_plan_from_compiled_coupler`` for dense rank-one/rank-two angular
lowering or rank-greater-than-two factorized angular-tree lowering. The
factorized plan deduplicates common subtrees and keeps each binary CG edge as
a sparse synthesis table.

``ye3t.runtime.YE3TSourceAnalysisModule`` prepares one instruction as stable
Torch buffers. Its ``native`` backend is a registered CPU operator with a
native adjoint, double backward, fake/meta registration, ``opcheck`` coverage,
and ``torch.compile`` support at the tensor-operator boundary.

Native build
------------

``libye3t_runtime`` is an independent C++17 numeric core:

.. code-block:: bash

   cmake -S . -B build -DYE3T_BUILD_TORCH_ADAPTER=OFF
   cmake --build build

The explicit Torch adapter can be built against the PyTorch installed in the
active Python environment without loading Caffe2's toolkit-dependent CUDA
CMake configuration:

.. code-block:: bash

   cmake -S . -B build-torch \
     -DYE3T_BUILD_TORCH_ADAPTER=ON \
     -DYE3T_BUILD_CUDA_ADAPTER=OFF \
     -DYE3T_TORCH_DISCOVERY=PYTHON
   cmake --build build-torch --target ye3t_runtime_torch
   (cd build-torch && ctest --output-on-failure)

This mode queries the wheel's include/library paths, CPython extension suffix,
C++ ABI, and pybind11 ABI from the selected ``Python3_EXECUTABLE``.
``YE3T_BUILD_CUDA_ADAPTER`` defaults to ``ON`` and then requires a CUDA
toolkit; with it set to ``OFF`` the CPU adapter builds even when the installed
PyTorch wheel has CUDA enabled but no CUDA toolkit is installed.
``YE3T_TORCH_DISCOVERY=LIBTORCH`` retains the standard ``find_package(Torch)``
route for standalone LibTorch installations.
``cmake --install`` stages the core archive/header and the adapter under
``ye3t/runtime``.

The opt-in CUDA adapter adds native radial/angular source primitives,
density scatter/gather, normalized compact symmetric/exterior pair products,
sparse source-analysis forward/adjoint, and fixed linear readout/adjoint
operators. Set the target architectures explicitly for the deployment
machines:

.. code-block:: bash

   cmake -S . -B build-torch-cuda \
     -DYE3T_BUILD_TORCH_ADAPTER=ON \
     -DYE3T_BUILD_CUDA_ADAPTER=ON \
     -DYE3T_TORCH_DISCOVERY=PYTHON \
     -DCMAKE_CUDA_ARCHITECTURES=89
   cmake --build build-torch-cuda --target ye3t_runtime_torch
   (cd build-torch-cuda && ctest --output-on-failure)

Replace ``89`` with the CUDA architecture list for the target systems. A
setuptools source build auto-enables CUDA when the build interpreter has
CUDA-enabled Torch and a visible CUDA toolkit. An explicit required build is
still available for release wheels:

.. code-block:: bash

   YE3T_BUILD_CUDA_EXTENSION=1 \
   TORCH_CUDA_ARCH_LIST=8.9 \
   python -m pip wheel . --no-build-isolation

Set ``YE3T_BUILD_CUDA_EXTENSION=0`` for an intentional CPU-only source build.
After installation, ``native_execution_plan_capabilities()`` should report
``cuda`` as true and include ``factorized_angular`` in ``cuda_operations``.

The Python wheel builds ``ye3t.runtime._execution_plan_native`` through the
same core source. ``YE3T_REQUIRE_NATIVE=1`` forbids reference fallback.
Native source builds must use the target runtime environment:

.. code-block:: bash

   python -m pip install "setuptools>=77,<82" torch wheel
   python -m pip install -e . --no-build-isolation

Torch is deliberately absent from ``build-system.requires``. A PEP 517
isolated source build can otherwise compile against a temporary Torch whose
ABI differs from the Torch that later loads the extension. Binary wheels do
not require build-time Torch at installation.

Linux and macOS native modules use a relative loader path from the YE3T
subpackages to the target environment's ``torch/lib`` directory. Linux wheel
CI runs ``cmake/audit_linux_wheel.py`` and rejects absolute build-environment
RPATH/RUNPATH entries. When ``patchelf`` is already available, the setuptools
build also normalizes the Linux extension RPATH after linking; this removes
paths injected unconditionally by some conda compiler specifications.
The CMake Linux install applies the same normalization when ``patchelf`` is
available. Its ordinary system-compiler path does not require that tool.

Current boundary
----------------

The initial native CUDA boundary implements built-in ``ChebExpCos`` radial
values and analytic ``dR/dr`` as fixed-index and packed all-index tables,
fixed-``L`` and packed all-``L`` real or complex spherical harmonics and
Cartesian derivatives, ordinary edge-to-center density accumulation and
its gather adjoint, sparse source placement followed by synthesis analysis and
its exact adjoint, normalized compact symmetric/exterior rank-two products and
their exact adjoints, generic compact exterior powers through order eight,
compiler-grouped repeated-block symmetric-power monomial contractions and
their exact adjoints, and fixed linear readout with source/readout/bias
adjoints.
These operators support FP32/FP64 and real/complex tensors where applicable.
The public ``YE3TSourceAnalysisModule`` reports ``native_cuda`` only when its
installed operators execute. Registered autograd uses native adjoints where
available and preserves double backward; the CMake and installed-wheel smokes
compare values, VJPs, mixed second derivatives, and readout contractions with
independent Torch formulas. ``native_execution_plan_capabilities()`` lists the
supported CUDA operations and measured automatic-dispatch thresholds
explicitly.

The CUDA spherical kernels have no fixed ``lmax``. They evaluate the derivatives
``P_L^(m)(z)`` with a three-row degree recurrence per edge, requiring
``O(edges * L)`` scratch, and applies the same real-to-complex unitary
conversion as the CPU convention. FP64 values and Cartesian derivatives agree
through ``L=12`` in the permanent smoke, and FP32 agrees through ``L=8`` at
the production ``5e-5`` tolerance. Both real and complex value paths pass VJP,
HVP, empty-input, and ``opcheck`` validation. The packed table stores degree
``L`` in columns ``[L^2, (L+1)^2)`` and evaluates the recurrence only once
through the requested maximum.

On the validation GPU, native density scatter became consistently faster than
Torch ``index_add_`` at approximately ``2^20`` edge-channel work items.
Automatic CUDA density dispatch therefore uses that conservative threshold;
explicit ``backend="native"`` bypasses the threshold. The capability report
publishes the active threshold. Broader hardware calibration remains required
before treating it as a universal crossover.

The compact rank-two kernels emit lexicographically ordered coordinates
directly, without constructing the ordered ``dimension**2`` product. The
exterior path annihilates duplicate indices and uses the normalized
``(x_i y_j - x_j y_i) / sqrt(2)`` convention; the symmetric path uses the
corresponding plus sign and retains diagonal products. A one-thread-per-input
adjoint avoids atomic accumulation. Real/complex FP32/FP64 values, explicit
adjoints, VJPs, mixed second derivatives, and ``opcheck`` pass on CPU and
CUDA. On the validation GPU, native forward was already 2.2--3.1 times faster
than the current exact Torch reference at batch one and dimension two, while
the adjoint was 7.6--9.1 times faster. Automatic CUDA dispatch therefore uses
the native operation for every eligible compact pair when installed.

Generic compact exterior powers use the same normalized determinant convention
through order eight on CUDA. Forward evaluates one pivoted local determinant
per lexicographic coordinate. The signed adjoint evaluates one local adjugate
per coordinate and atomically accumulates only compact factor gradients;
exactly singular blocks use explicit signed minors so duplicate annihilation
does not erase valid derivatives. Orders 1, 2, 3, 4, 6, and 8 pass
real/complex FP32/FP64 value, adjoint, HVP, duplicate-factor, and ``opcheck``
coverage. On the validation GPU, native forward and adjoint beat a batched
``torch.linalg.det`` baseline for all measured orders through eight, batch
sizes 1--512, and dimensions through 10. Automatic CUDA dispatch therefore
uses native for every supported order when installed.

Compiler-generated real-tesseral symmetric-power tables lower
``Sym^p(V_L) -> V_LR`` to grouped integer-exponent monomials. The native
forward contracts each output's contiguous term range without expanding an
ordered ``(2L+1)^p`` product. Reverse evaluates each term once, uses
zero-safe polynomial derivatives, and accumulates directly into the input
coordinates. The low-level operator supports real/complex FP32/FP64; the
current compiler integration uses its established real-tesseral coefficient
tables and preserves output multiplicity.

CUDA forward and adjoint were faster than the existing product evaluator for
all measured powers 2--8, batch sizes 1, 32, and 1024, so eligible installed
CUDA operations dispatch natively. CPU measurements favored native execution
for flattened batches through 32 and for powers at least 6; those conservative
thresholds are published by ``native_execution_plan_capabilities()`` and used
by ``optimization_policy="auto"``. Explicit aggressive or required-native
execution bypasses the crossover policy. Actual compiler tables with
nontrivial output multiplicity pass value, adjoint, fixed-cotangent HVP,
squared-loss HVP, fake/meta, and ``opcheck`` coverage. This is not yet a
generic mixed-block Schur/Young instruction or cross-descriptor subtree
schedule.

Selected output irreps from one repeated block can be concatenated into a
single grouped native call. The result is sliced back into the established
``L_R`` and multiplicity shapes without copying. On the validation GPU,
combined execution was 2.1--3.6 times faster than separate native calls for
``Sym^4(V_1)``, ``Sym^4(V_2)``, and ``Sym^8(V_1)`` over batches 1, 32, and
1024, except ``Sym^4(V_2)`` at batch 1024 where it remained slightly faster
at 1.03 times.

``SymmetricPowerProductPlan`` also lowers all full-channel exponent vectors
to one grouped table. ``ye3t-methods`` uses this tensor-valued runtime for descriptor
values and batches compiler-selected output seeds for explicit root rows.
The complex native adjoint is conjugated back to the application's established
holomorphic ``dB/dA`` convention before radial/angular position VJPs. Real and
complex root rows remain available inside ``torch.no_grad()``. For a complete
15-component ``Sym^4(V_1)`` descriptor plan, grouped CUDA values plus root
rows were 19.5 and 24.0 times faster than the Python exponent-loop baseline
at batches 8 and 256 respectively.

The compiler also lowers repeated exponent rows to a unique
monomial table and a sparse coefficient map. Native CPU/CUDA forward computes
each unique monomial once; reverse first accumulates output adjoints onto the
unique monomials and then applies their zero-safe polynomial derivatives.
This adds ``symmetric_power_shared_monomial`` and its exact adjoint. Real and
complex values, explicit adjoints, fixed-cotangent and squared-loss HVPs,
zeros, fake/meta execution, and ``opcheck`` pass on CPU and CUDA.

Measured automatic dispatch retains the grouped kernel where launch and
workspace overhead dominates. CPU selects sharing at coefficient-to-monomial
reuse at least 8, or reuse at least 3 from batch 8. CUDA selects it at reuse
at least 8 from batch 256 when autograd is active and from batch 1024 for
forward-only execution. On the validation GPU, high-reuse forward-plus-adjoint
speedups were 1.33--2.57 times at batch 256 and 4.73--8.36 times at batch
1024. Low-reuse ``Sym^4(V_1)`` remained slower and therefore stays grouped.
CPU high-reuse speedups reached 2.0--3.4 times for forward plus adjoint over
the measured batches.

ABI 29 adds a native CUDA double-backward operator for the plain sparse
monomial symmetric-power schedule. It evaluates the output tangent and the
zero-safe polynomial Hessian action directly from the compiler-owned exponent
table, avoiding many small PyTorch complex reductions in force training. The
capability report exposes
``symmetric_power_monomial_double_backward``. Real and complex FP64 reference
tests include zero coordinates, and rank-general value/VJP/HVP coverage
includes ranks 8, 12, 16, and 32.

On the bounded crossed rank-6+8 CUDA gate, ABI 29 reduced the per-batch
force-training time from 174.46 to 99.33 ms for the repetition-oriented
catalogue and from 136.98 to 86.47 ms for the diversity-oriented catalogue.
The maximum trained-parameter residuals against bracketing reference runs were
``8.55e-7`` and ``1.01e-6``. The developer-only environment variable
``YE3T_SYMMETRIC_POWER_MONOMIAL_DOUBLE_BACKWARD_POLICY`` can force
``reference`` or ``native`` for an A/B gate; normal ``auto`` dispatch selects
the native CUDA operation when ABI 29 is available and retains the exact
generated reference elsewhere.

A rank-general one-warp-per-sample factorized-angular traversal was also
validated against the serial CUDA reference, including shared-memory
double-backward workspaces and a rank-12 two-block plan. It improved the two
crossed catalogue gates by only 0.9 percent and -0.15 percent, below the fixed
10 percent promotion threshold. Production ``auto`` therefore remains the
stable serial traversal. ``YE3T_FACTORIZED_ANGULAR_CUDA_POLICY=warp`` is a
developer-only diagnostic until a compiler-generated destination-segmented
implementation passes an end-to-end gate.

ABI 30 adds a fused residual gated carrier scatter, and ABI 31 adds an exact
destination-segmented form with native forward, adjoint, and double backward.
The segmented schedule stores target-major, source-major, and channel-major
edge or feature permutations. Kernels therefore perform no runtime sorting or
path discovery and use one destination-owned reduction rather than atomic
fan-in. Real or complex carriers with real invariant gates follow the same
specified carrier convention.

The application policy is ``atomic``, ``auto``, or strict ``segmented``.
``atomic`` remains the production default. On the retained 474-atom,
7,432-edge force-training workload, segmented execution improved the median
from 877.500 to 815.963 ms (7.5 percent); on the eligible 79-atom ASE workload
it improved 125.249 to 121.789 ms (2.8 percent). Both are below the fixed
10-percent end-to-end promotion gate. ABI 31 remains an exact forced
diagnostic for future destination/fan-in shapes rather than an automatic
dispatch claim.

The same force-training profile exposed a rank-general CUDA lookup cost in packed
carrier channel updates. Replacing a linear scan over hundreds of carrier
block offsets with a hybrid linear/binary lookup reduced the unchanged atomic
force-training profile from 1004.637 to 877.500 ms and the 79-atom ASE profile
from 142.520 to 125.249 ms. This 12--14 percent end-to-end improvement is
retained for every real/complex forward, adjoint, and double-adjoint layout;
it changes indexing only, not carrier content or arithmetic order.

The shared CUDA forward and ordinary adjoint allocate a
batch-by-unique-monomial workspace; measured peak temporary allocation was
approximately twice the grouped path for these compact tables. Descriptor
root rows use a separate broadcast batched adjoint. Its logical cotangent
shape is ``[seed, batch, output]``, but it reads one physical
``[batch, input]`` tensor and returns ``[seed, batch, input]`` without
expanding the input or evaluating an unused expanded forward result.

Against the expanded-root baseline, the broadcast adjoint was
1.84--2.65 times faster for low-reuse ``Sym^2(V_1)`` and ``Sym^4(V_1)`` cases
at batches 32 and 256 with 4 or 16 roots, while reducing measured peak
allocation by 40--46 percent. High-reuse ``Sym^4(V_2)`` cases at batches
32--1024 were 1.28--2.28 times faster and reduced peak allocation by
approximately 48 percent. ``ye3t-methods`` uses the broadcast root path
unconditionally when a compiler-owned shared-monomial product table is
available.

This first implementation launches source placement and synthesis analysis as
two CUDA kernels. Its linear readout wrapper contracts the native analysis
with an ATen matrix product. It is therefore a functional native boundary,
not a claim of final one-kernel fusion or measured performance leadership.
Primary-complex factorized angular DAG forward/adjoint and fixed-linear-readout
forward/adjoint run on CUDA.
Direct physical-source-to-readout fusion, mixed Schur/Young exterior
projections, Hermitian operator actions on CUDA, real-basis factorized
execution with a serialized unitary transform, and full message-layer fusion
remain open.

ABI 13 retains the Torch-independent CPU kernels for the built-in ``ChebExpCos``
radial family and fixed-``L`` complex Condon--Shortley spherical harmonics,
and adds packed radial and all-``L`` table operators on CPU/CUDA.
Both return analytic radial or Cartesian derivatives. The real-tesseral
operator is obtained through the specified unitary convention and returns the
same signed-``m`` ordering. Value autograd uses the analytic first derivative;
the Torch reference in the registered backward preserves double backward.
The same core owns ordinary edge-to-center density accumulation and its exact
gather adjoint for real or complex buffers; both are registered for double
backward, fake/meta operation, and ``opcheck``.
``SiteBasisV2`` uses one radial-table and one angular-table call per geometry
in its explicit-edge-derivative and streaming force-VJP paths. On the
validation SM 8.9 GPU, the two smaller representative cases improved native
edge-derivative speedups over Torch from 5.3--7.1x before table dispatch to
9.2--10.0x. At the largest ABI-13 case, grouped edge-product/index assembly
dominated.

ABI 14 adds ``plain_site_basis_product_with_derivative`` on CPU/CUDA. One
launch consumes the cached radial and angular tables, chemical/charge
prefactors, radial directions, and compiler-derived group/channel placement
arrays and returns edge values, Cartesian derivatives, and center/neighbor
charge derivatives. ``SiteBasisV2.compute_channel_edges_with_dx`` dispatches
it only when the complete channel schedule consists of ordinary plain
channels; auxiliary, fallback, and specialized streaming-VJP paths
retain their established implementations. CUDA placement bounds use device
assertions, avoiding host synchronization in the hot path.
Against the ABI-13 table path plus grouped PyTorch product/index assembly on
the validation SM 8.9 GPU, synchronized FP32/complex-FP32 source construction
improved by 1.74x at 4,096 edges and 100 channels, 1.20x at 16,384 edges and
294 channels, and 1.02x at 65,536 edges and 648 channels. The largest case is
dominated by output and downstream memory traffic. Values and all three
explicit derivative outputs match the reference for real/complex conventions
and scalar-charge sources. Value autograd supports double backward; explicit
derivative outputs remain first-order analytic runtime products.

ABI 15 adds ``plain_site_basis_product_adjoint`` on CPU/CUDA. It consumes the
same packed source tables plus edge weights, weight derivatives, and an
edge-channel cotangent, then reduces directly to edge Cartesian and
center/neighbor charge cotangents. The operation does not materialize an
``edge x channel x xyz`` Jacobian. ``SiteBasisV2`` uses it in both the
specialized real and general complex streaming position/strain VJP paths for
complete ordinary plain schedules, including soft-neighbor normalization.
Real/complex values, VJPs over every differentiable operator input, coupled
HVPs, fake/meta behavior, and ``opcheck`` pass. On the validation SM 8.9 GPU,
native speedups over the prior streaming paths were 2.05x, 1.79x, and 1.24x
for real workloads and 1.48x, 1.19x, and 1.58x for complex workloads at
4,096/100, 16,384/294, and 65,536/648 edges/channels, respectively.

ABI 16 adds the CUDA
``spherical_harmonics_table_double_backward`` contraction for real spherical
tables. It evaluates the directional Cartesian Hessian contraction and the
corresponding value-adjoint gradient in one kernel without materializing an
``edge x angular_width x 3 x 3`` Hessian. The implementation differentiates
the normalized Cartesian solid-harmonic recurrence with exact directional
jets carrying value, Cartesian gradient, directional derivative, and its
Cartesian gradient. FP64 results match the Torch reference for maximum
angular momenta 1, 3, and 5 through both the raw operator and registered
autograd path. ``backend="native"`` and automatic CUDA ``SiteBasisV2`` source
dispatch use this kernel for force-loss HVPs. On an interleaved validation
run of the 31-atom, 409-parameter force-training workload, automatic angular
dispatch reduced CUDA events from 25,988 to 25,200 and the warmed median step
from 340.1 ms to 335.2 ms. Separate-process medians were 319.5 and 328.7 ms
for the angular path versus 339.0 ms for radial-only dispatch. These are local
SM 8.9 break-even results, not general throughput claims.

The CUDA factorized angular path follows the compiler's topological subtree
DAG, source-placement projection, and ``C^\dagger`` analysis orientation. It
supports ordinary and lifted-role source coordinates in real/complex
FP32/FP64. Real/primary-complex FP64 values, VJPs, HVPs, and ``opcheck`` pass
for ordinary and nontrivial ``lambda=(2,1)`` role-resolved plans. Against the
existing Torch DAG reference on the validation GPU, forward measured
2.4--13.4 times faster and forward plus adjoint measured 1.82--11.3 times
faster across batches 1--1024.
These older DAG tests exercise algebraic source-coordinate buffers. The
complete typed compiler is required to enumerate all independent angular
and local Young routes; a raw child-tableau buffer alone is not a physical
``A_s`` realization.
ABI 20 adds a heterogeneous-root factorized operator for exact block
composition across parent angular momenta. Each root carries compiler-validated
projection-column bounds, magnetic dimension through its root node, and an
output offset. Projection tables remain source-major, plan hashes and logical
output slices remain independent, and the operation changes execution
grouping only: it does not mix different ``L`` sectors. CPU validates that
root output intervals exactly partition the specified output; CUDA validates
bounds in the kernel without host reads.
Forward, adjoint, and double-backward operators pass mixed-``L`` lifted
``A_s`` value, VJP, HVP, fake/meta, ``opcheck``, and ``torch.compile`` tests.
For a controlled batch-31 four-parent rank-3 ``lambda=(2,1)`` lifted-role
source/HVP workload, one heterogeneous call replaced three compatible
factorized calls and reduced the interleaved median CUDA time from 3.871 ms
to 3.757 ms on the validation SM 8.9 GPU. ``ye3t-methods`` uses this path only
inside its existing measured direct-physical-bank batch policy.

Within that one call, compiler-owned exact subtree identities permit
cross-plan common-subexpression elimination when parent plans consume the same
physical source bank. Leaf identity includes physical binding, magnetic
dimension, and convention. Merge identity includes ordered child identities,
dimensions, synthesis and analysis orientations, convention IDs, and the
exact sparse coefficient-table hash; matching shape or ``L`` alone is
insufficient. For the rank-3 lifted ``A_s``, ``lambda=(2,1)``,
``L_R=1,2,3`` validation group, this reduces 23 plan-local nodes to 13 shared
nodes and workspace width from 103 to 61. A controlled batch-31 CUDA
forward-plus-VJP-plus-HVP comparison against the prior one-call unshared
forest measured 2.875 ms versus 2.513 ms over 40 interleaved repeats on the
same validation GPU.

ABI 21 adds ``edge_outer_accumulate`` for role-resolved physical source
materialization. It evaluates
``Y[a,s,c] = sum_(e:center(e)=a) q[e,s] A[e,c]`` without collapsing the role
axis and provides exact adjoint and double-backward operators. ``ye3t-methods``
uses the same boundary for the lifted numerator and role-resolved soft-count
normalizer. CPU/CUDA values, position VJPs/HVPs, edge-order invariance, and
empty-neighbor behavior match the Torch reference. Automatic dispatch uses
measured work-item thresholds of ``2^16`` on CPU and ``2^18`` on CUDA;
explicit native requests bypass those thresholds without fallback.

ABI 22 implements MP-EQ-27 carrier-aligned graph aggregation through
``carrier_gated_scatter``:

.. math::

   y_{j,a t M}
   =
   \sum_{e:i\to j}
   g_{e,a}\,x_{i,a t M}.

``YE3TCarrierLayout.feature_channel_indices`` compiles the flattened feature
to invariant-channel map from exact ``[channel_or_multiplicity, tableau_t,
magnetic_M]`` layouts. A gate is therefore shared over every ``t,M``
component of one carrier copy, and heterogeneous ``(N,lambda,L)`` blocks may
share one execution call without becoming one representation or mixing
sectors. This operation transports already-typed carriers; it is not an LR,
Kronecker, or angular coupling operation.

Real and primary-complex CPU/CUDA values, adjoints, coupled double backward,
empty edges, fake/meta, ``opcheck``, and full-graph ``torch.compile`` match an
independent gather/multiply/scatter reference. At a representative
216-component, 24-channel complex layout, automatic dispatch uses ``2^15``
CPU and ``2^18`` CUDA work-item thresholds. Native CPU was about 5.7 times
faster at 2,048 edges; native CUDA was about 2.9 times faster at 8,192 edges
on the validation SM 8.9 GPU. These thresholds are local calibration data,
not architecture-independent guarantees.

ABI 23 fuses the residual, block-local channel map, and invariant gate for
packed carrier blocks:

.. math::

   y_{b,o t M}
   =
   x_{b,o t M}
   +
   g_{b,o}
   \sum_i W_{o i} x_{b,i t M}.

Compiler-derived feature, channel, and map offsets delimit heterogeneous
``(N,lambda,L)`` blocks. ``W`` may mix only the channel-or-multiplicity axis
inside one block, while the same gate and map are applied to every tableau and
magnetic component. The operation therefore does not mix carrier keys and
does not replace source assembly, LR induction, Kronecker products, or angular
analysis.

Real and primary-complex CPU/CUDA forward, adjoint, and double-backward
operators match the independent blockwise ``einsum`` reference. Fake/meta,
``opcheck``, full-graph ``torch.compile``, gradcheck, gradgradcheck, CUDA VJP,
and coupled HVP tests pass. This is a small-work fusion: for the exact
216-feature, 20-channel, four-block lifted-carrier layout, the 124-row CUDA
forward-plus-VJP-plus-HVP boundary measured 6.19 ms for the blockwise
reference and 2.90 ms for native. Native lost at 512 rows, so automatic CPU
and CUDA dispatch uses a conservative maximum of ``2^15`` packed value items
rather than a minimum-work threshold.

In the lifted-density message-passing integration, one complex packed carrier
buffer persists across hidden layers. Graph transport and the ABI-23 channel
update consume that buffer directly; typed sector records are views rather
than repacked copies. Distance features and inverse target degrees are also
computed once per model forward and shared by the graph layers. This does not
erase carrier boundaries: compiler offsets and sector views continue to
preserve each ``(N,lambda,L,convention)`` block and its independent ``t,M``
axes.

ABI 24 adds
``scheduled_radial_angular_channels_with_derivative``. It consumes complete
radial and angular value/derivative tables plus immutable per-channel radial
indices, angular indices, normalization scales, and optional integer source
masks. One CPU/CUDA operation produces ``[edge,channel]`` values and their
Cartesian derivatives without per-channel table slicing, transposes, stacks,
or mask tensors. The contract is a generic numeric table schedule; atom types,
neighbor lists, role filters, and physical source semantics remain owned by
``ye3t-methods``.

CPU/CUDA values and analytic derivatives match the independent Torch table
selection and product reference. VJP, HVP, gradcheck, gradgradcheck,
fake/meta, and ``opcheck`` pass. On the retained 124-atom, 1,312-edge
role-resolved K2SO4(H2O) force-training workload, ABI 24 reduces lifted
edge/site-basis time from 2.80 to 1.35--1.41 ms, complete source analysis from
4.20 to 2.71--2.91 ms, CUDA events from 1,899 to 1,335, and peak allocation
from 77.60 to 76.65 MB. Four 100-repeat full-step medians were 24.12, 24.38,
27.43, and 28.99 ms versus 30.97 ms immediately before ABI 24; phase and event
reductions are the more stable evidence because the end-to-end measurements
remain bimodal.

ABI 25 adds ``softmax_gaussian_role_density`` and its native adjoint and
double-backward operations. The operation consumes distances, fixed per-edge
cutoffs, immutable Gaussian role centers and width, packed edge channels, and
center indices. It evaluates normalized role filters and accumulates directly
to ``[atom,role,channel]`` without materializing an edge-by-role filter tensor
or an edge-by-role-by-channel outer product. Cutoffs and role-filter parameters
are schedule data; the differentiated inputs are distances and edge channels.
The role coordinate remains an explicit output axis.

CPU/CUDA values, VJPs, HVPs, fake/meta, and ``opcheck`` match an independent
Torch softmax-and-scatter reference. On the retained 124-atom, 1,312-edge
role-resolved K2SO4(H2O) force-training workload, the fused role-density phase
is 0.207 ms. The 100-repeat full-step median is 23.23 ms, CUDA events decrease
from 1,335 to 1,224, and peak allocation decreases from 76.65 to 75.07 MB
relative to ABI 24. Energy remains exact and the maximum force residual is
``2.1e-22``. Other filter families and pair-specific filter overrides retain
the exact separate-filter/edge-outer fallback.

ABI 26 reuses the primal and first-adjoint workspaces produced by the CUDA
heterogeneous factorized-angular adjoint when evaluating its double backward.
The public factorized analysis result and the CPU reference path are
unchanged. Internal workspace-returning adjoint and workspace-consuming
double-backward operations preserve the same ``C^\dagger L_v`` analysis,
source coordinates, and ``[channel_or_multiplicity,tableau_t,magnetic_M]``
carrier layout. The retained workspaces are treated as immutable internal
autograd state rather than differentiable public outputs.

CUDA values, VJPs, HVPs, fake/meta behavior, and ``opcheck`` match the
independent complex reference. The cached adjoint and second derivative are
also bitwise equal to the prior recomputing native operations in the focused
heterogeneous mixed-sector test. On the retained role-resolved 124-atom
K2SO4(H2O) force-training workload, factorized-angular double backward falls
from 1.706 to 1.078 ms, force-loss backward from 11.787 to 10.826 ms, and the
100-repeat full-step median from 23.231 to 22.342 ms. CUDA events remain 1,224
and peak allocation remains 75.07 MB. Energy is exact and the maximum force
residual remains ``2.1e-22``.

The ABI-26 carrier-channel adjoint and double-backward kernels also use a
cooperative block reduction over the flattened batch, tableau, and magnetic
plane. One block owns each channel-map entry, so parameter reductions no
longer serialize the complete batch in one CUDA thread. Small CPU/reference
kernels are retained below the measured dispatch thresholds. For the retained
four-block complex mixed-sector layout, native forward, VJP, and HVP take
1.44--11.45 ms for batches 32--4096, compared with 5.95--30.50 ms for the
Torch reference. Values, gradients, and HVPs match at every measured size.
Automatic CUDA dispatch therefore permits the native path through 1,048,576
work items. At 992 atoms the optimized native force-training step measures
40.27 ms, compared with 45.82 ms for the reference channel update and 56.04
ms for the prior serial native reduction.

The current canonical 124-atom rerun uses automatic dispatch and selects the
native carrier update. Its 100-repeat median is 22.72 ms, with 5.91 ms energy
forward, 5.67 ms force VJP, and 10.94 ms force-loss backward. It retains 1,224
CUDA events and 75.07 MB peak allocation; energy is exact and the maximum
force residual is ``3.2e-22``. End-to-end timings remain mildly variable, so
the operator and phase reductions are the primary performance evidence.

Native lifted-density forward evaluation is compatible with full-graph
``torch.compile`` when species and source schedules are static. Compiled
force-loss training is not yet a valid replacement for the native
second-order path: PyTorch AOTAutograd currently rejects double backward. The
runtime therefore keeps explicit native adjoint and double-backward dispatch
for force training rather than silently compiling only the forward pass.

Factorized-angular CUDA traversal uses one independent serial DAG traversal
per batch/source coordinate. Launching those register-heavy traversals in
256-thread blocks left the canonical 744-coordinate workload with only three
resident blocks. ABI 26 uses one-warp blocks and computes the grid from
that block size. This changes neither traversal order nor arithmetic. On the
canonical complex128 workload, heterogeneous cached double backward decreases
from 1.084 to 0.467 ms and its two adjoints from 0.657 to 0.317 ms. The final
100-repeat force-step median is 22.15 ms, with exact energy and a ``3.2e-22``
force residual. Complex64 double backward is 0.646 ms, its two adjoints are
0.478 ms, and the FP32 production probe uses 47.70 MB peak allocation versus
75.07 MB for FP64. Its full-step median is 21.95 ms and its force-reference
residual is ``5.0e-13``. At 992 atoms the one-warp launch remains neutral
within run variability: 40.94 ms versus 40.36 ms for two-warp blocks, with
unchanged native operator phases and exact reference agreement.

ABI 27 permits invariant graph gates to retain the real component dtype when
they act on complex carriers. Forward evaluation multiplies the complex
``[channel,t,M]`` carrier by a real channel gate; the adjoint returns a complex
carrier gradient and the real part of the gate contraction. Native CPU/CUDA
double backward preserves the same mixed tangent types. Legacy same-dtype
real and complex gates remain supported. This makes the implementation match
the carrier contract directly: scalar gates are uniform over ``t,M`` and do
not acquire an artificial complex learned-parameter axis.

CPU/CUDA mixed-dtype values, gradients, HVPs, gradcheck, gradgradcheck,
fake/meta behavior, ``opcheck``, and full-graph forward compilation match the
independent Torch reference. The role-resolved lifted-density model passes
its real graph gates directly to the native scatter. On the canonical FP64
force-training profile this removes six CUDA events and 0.63 MB peak
allocation; graph-scatter double backward falls from approximately 0.126 to
0.102 ms and its four adjoints from approximately 0.089 to 0.064 ms. FP32
also removes six events, four copies, two fills, and 0.31 MB peak allocation.
The corresponding wall medians remain bimodal, so these structural and
operator reductions are the retained evidence.

ABI 27 also permits the carrier channel-update gates and learned channel maps
to retain the real component dtype over complex carriers. Forward evaluation
still acts on packed complex ``[channel,t,M]`` values; native CPU/CUDA
adjoints return complex value gradients and real gate/map gradients, and
double backward preserves those mixed tangent types. Same-dtype real and
legacy complex controls remain supported. The role-resolved lifted-density
model keeps learned controls real while evaluating ``A_s`` through its explicit
role axis and compiler-owned ``L_v -> C^dagger`` analysis.

Mixed-control values, VJPs, HVPs, gradcheck, gradgradcheck, fake/meta,
``opcheck``, and full-graph forward compilation match the independent Torch
reference on CPU and CUDA. Relative to the preceding ABI-27 graph-gate run,
the canonical FP64 profile removes ten CUDA events and 143,872 peak bytes.
The FP32 profile removes ten events, nine copies, one fill, and 72,704 peak
bytes. FP64 and FP32 maximum force residuals are respectively ``2.1e-22`` and
``4.5e-13``. Wall medians remain variable and are not used as standalone speed
evidence.

The lifted application stores each layer's compiler-ordered real channel maps
as one persistent packed parameter and uses it during forward and force
derivative evaluation. Strict checkpoint loading accepts the
``channel_maps.N`` layout and converts it to the packed layout.
The canonical FP64 profile removes two additional CUDA events and two
``aten::to`` calls, with exact energy and a ``2.6e-22`` force residual. This is
a storage/layout correction, not a claimed standalone wall-time speedup.

The application computes each active edge distance once and reuses it for
pair selection, native radial tables, radial directions, role filters, and
soft-neighbor normalization. The explicit VJP record and ASE model paths use
the same geometry contract. Runtime reports expose ``shared_edge_geometry``;
the source remains role-resolved ``A_s`` and all unsupported filter families
retain their established exact fallback.

On the canonical FP64 profile, edge geometry plus site-basis construction
decreases from 1.480 to 1.136 ms and complete source analysis from 2.376 to
2.086 ms. CUDA events decrease from 1,206 to 1,135, ``aten::to`` calls from
151 to 144, and copies from 148 to 145. Energy is exact and the force residual
is ``2.6e-22``. The end-to-end wall median remains bimodal and is not retained
as speed evidence.

The runtime also exposes fused dense source-analysis and factorized-angular linear
readouts with native source and parameter adjoints. They avoid publishing or
retaining a Torch descriptor tensor and pass double backward, ``opcheck``,
and full-graph ``torch.compile`` validation. Dense readout contracts the
readout weights through ``C^\dagger L_v`` before each source row. Factorized
readout contracts weights directly at DAG roots and seeds the reverse
traversal there. Neither fused path allocates a
``batch x descriptor_dimension`` feature or feature-adjoint buffer.
For factorized plans, the ABI-12 fused kernel returns source, readout-weight,
and bias adjoints directly. It passes real/complex ordinary and lifted-role
FP64 value, VJP, coupled HVP, fake/meta, and ``opcheck`` validation. Relative
to materializing native factorized features before the readout, the measured
forward-plus-backward peak allocation fell by approximately 24 percent for the
ordinary source and 44--45 percent for the lifted-role source. Real arithmetic
and ordinary complex arithmetic were generally faster; the current fused
complex lifted-role kernel was approximately 18 percent slower at large
batches. Explicit ``backend="native"`` remains fused, while automatic dispatch
uses the measured faster all-native materialized factorized path for complex
multi-coordinate role sources. The runtime report distinguishes both choices.

ABI 8 adds generic compact spinless exterior powers through order eight on
CPU. Coordinates are normalized determinants in lexicographic increasing
one-particle index order. The C++ core evaluates them without ordered
``dimension**rank`` storage and provides a signed cofactor adjoint for real or
complex FP32/FP64 tensors. Rank-3 and rank-4 values, factor-swap signs,
duplicate annihilation, first derivatives, second derivatives, fake/meta
behavior, and ``opcheck`` are validated. The application
``ExteriorPowerEvaluator`` rescales this normalized runtime result explicitly
to its pre-existing unnormalized determinant convention; stored model
normalization therefore does not change silently.

The other native CPU operators cover sparse source placement followed by
linear analysis and packed factorized rank-greater-than-two angular DAG
traversal for real or complex FP32/FP64 tensors. The factorized operator has
an analytic reverse traversal, double-backward registration, fake/meta
behavior, ``opcheck``, and ``torch.compile`` coverage. Both native and
reference paths follow the stored ``L_v`` source identifications without
materializing ordered tensor products.

Typed multi-coordinate lifted-role buffers are supported by the same native
CPU traversal and retain the sparse ``L_v`` source-column identities. Dense
real-tesseral source-analysis instructions also compile the complete sparse
``C^\dagger L_v`` map into one packed Triton launch. An optional learned map
is expanded only over the channel/multiplicity index and is shared over all
tableau and magnetic components. Explicit Triton dispatch requires the
serialized
``real_tesseral_from_complex_condon_shortley_young_orthogonal_v1``
convention; it never infers a real basis merely from numerically real
coefficients. FP64/FP32 values, source and parameter adjoints, HVPs, complete
role-placement invariance, and strict no-fallback dispatch are validated.
``native_execution_plan_capabilities()`` reports C++ extension capabilities
and optional Triton availability separately, including the required real
convention ID and supported packed source-analysis operations.
Factorized source-analysis CUDA, rooted-motif buffers, direct
edge-source-to-readout fusion in the C++ core, and native geometry double
backward remain open.
The application package materializes the preceding physical
``[atom, role, channel]`` lifted-density source through the native center
scatter without flattening away the logical role coordinate. That source
materialization is not itself a Young projection; complete role placements
must still be supplied by the execution plan's ``L_v``.
For lifted seed roles, ``L_v`` serializes each source coordinate's exact coset
representative and child-tableau coordinate. Equal role tuples with the same
child-tableau coordinate are merged at compile time with
``1/sqrt(placement_multiplicity)`` weights, giving an explicit isometric
source map without runtime rank discovery. The application can therefore
build complete distinct- or repeated-role tuple orbits without equating role
indices with induced-basis indices. A rank-3 mixed parent validates the parent
Young action on separate tableau axes, a repeated ``(a,a,b)`` source validates
the expected nontrivial-sector selection rule, and a totally symmetric scalar
parent validates native energy and analytic atomic forces.
Exact real-basis ``JointYoungCGProduct`` binary nodes have a separate packed
Triton path: every instruction in one product is lowered to one sparse
bilinear table and evaluated in one launch. The table is derived from the
serialized complex-primary coefficients through the specified fixed real
unitary transformation; it preserves logical
``[channel_or_multiplicity, tableau_t, magnetic_M]`` output axes. Its forward,
analytic adjoint, and double-backward behavior match the instruction-einsum
reference in FP64, and FP32 construction uses the project ``5e-5`` production
tolerance. Several exact product paths can also be packed into one weighted
group. That group applies trainable maps only on multiplicity/channel axes and
packs the outputs without mixing tableau or magnetic components.
Its flat packed output can feed the next compiled group directly. When the
next group's source blocks are ordered adjacent views that cover that packed
state exactly, source packing reconstructs a zero-copy contiguous view after
checking storage identity, offsets, strides, and compiled widths. Reordered,
partial, unrelated, or noncontiguous source blocks use the explicit
one-concatenation fallback.
The serialized layout uses ``YE3TPackedCarrierSlice`` records with explicit
buffer IDs, exact ``YE3TCarrierLayout`` values, and half-open offsets.
``YE3TExecutionPlanWiring`` declares complete packed-buffer widths and binds
each runtime instruction input position and output to one slice. Validation
requires gap-free, nonoverlapping direct sums and exact
``(N, lambda, L, convention)`` agreement between each binding and instruction.
The current mixed-sector hidden product group emits this wiring across all
packed paths and records zero-then-accumulate output semantics when several
exact product instructions contribute to one carrier slice. Plans that do not
yet use packed wiring retain their prior serialized payload and hash.

``YE3TMessagePassing`` can attach the compiler-owned dense source evaluator
through its execution-plan entry point, including explicit dtype, device,
channel-mixing, and strict-backend settings. This validates the input-node
handoff ``physical source -> L_v -> C^\dagger -> W -> packed carrier``. It
does not make an isolated-child binary product a complete LR induction map,
and it does not yet fuse physical source construction, subsequent tree nodes,
aggregation, and nonlinear updates into one GPU layer.

The application real-force path uses compile-time-specialized Triton kernels
for ProductDAG output initialization/depth reverse and factorized schedules
through eight blocks. Unsupported tile overrides and legacy ProductDAG
full/chunk or complex-split modes use explicitly reported reference execution;
they do not silently claim Triton dispatch.

For explicitly requested exact parent ``S_N`` targets, the compiler bypasses
the legacy raw-sector symbolic projector and lowers the validated cached
Young-subgroup subduction matrix directly. The emitted analysis is
``C^\dagger L_v`` for the canonical binary-tree placement and records the
active identity-coset rows. A representative
``(2,1) x (2,1) -> (3,2,1)`` rank-6 target emits both LR copies and is
validated through packed CUDA double backward. This canonical placement is
one typed source map; a physical lifted-density or motif layer must still
materialize all source placements required by its complete ``L_v``.

The application
artifact binds explicit ordinary-density source factors and performs
validated streaming energy/force/virial traversal without descriptor or
edge-derivative buffers. The analytic
derivative outputs themselves are first-order data and are not differentiable
operator outputs; value paths have validated first and second autograd
derivatives.

The compiled atomistic model also exposes fixed-basis energy, force, strain,
and ASE-order stress design rows independently of saved readout weights.
Feature adjoints are evaluated in bounded chunks and may be accumulated into
``X^T X``, ``X^T y``, and ``y^T y`` across structures. This fitting path
retains one structure's force rows and compact edge-derivative record, but
never retains a dataset-wide design matrix. Dense rank-2 and factorized
rank-3 plans match full autograd and explicit normal equations.
