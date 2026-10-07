# ye3t Examples

These examples are small, deterministic demonstrations of the
representation-theory package surface.

| example | purpose |
|---|---|
| `coupling_multiplicity_counts.py` | Construct `YE3TRepresentation`, then count valid fixed-content labels through its compiler-backed method and inspect validation. |
| `coupling_coefficient_materialization.py` | Count a fixed-content ACE sector and compile its factorized coefficient schedule; inspect the exact count, term count, convention hash, and validation metadata. |
| `symmetric_count_formula.py` | Compare the GE-PI symmetric fixed-content multiplicity formula against YE3T's exact compiler counts for every allowed output `L`. No coefficients are compiled. |
| `symmetric_density_factors.py` | Supply your own complete density multiplets, apply a symmetric scalar YE3T coupler, and check factor-exchange invariance. |
| `user_supplied_factors.py` | Supply complex Condon–Shortley multiplets from any source, bind the factorized runtime, and evaluate every `(2,1)`, `L=1` coordinate on one factor set or a batch of message inputs with gradients. |
| `symmetric_external_factors.py` | Evaluate every `(N)`, configurable-`L` coordinate of ordered external factors with a repeated symmetric block. |
| `antisymmetric_external_factors.py` | Evaluate every `(1^N)`, configurable-`L` coordinate of ordered external factors with a repeated alternating block. |
| `cauchy_supplied_density.py` | Supply role-resolved real tesseral densities, count and compile the nontrivial local Cauchy path `κ=(1,1)`, and evaluate its axial `L=1` output. |
| `compile_scalar_ace_lammps_plans.py` | Compile representative rank-3, H4, `[4,4]`, and H16 scalar ACE coordinates plus block and v3 coupled-product execution plans for downstream LAMMPS binding through `ye3t-lammps`. |
| `exact_full_primitive_catalog.py` | Build a saved fixed-content basis for ranks 4, 6, and 16; inspect primitive and decomposable representations and request exact coefficient blocks or one normalized vector. |
| `symbolic_young_partition_catalogue.py` | Expand symbolic Young partition templates by rank through `ye3t.couplings.expand_partition_templates` and print the positive O(3) multiplicities of each expanded partition. |
| `tagged_cauchy_image_catalogue.py` | Build the exact shifted-Jacobi source product algebra with `ye3t.couplings.build_radial_species_product_record`, bind it to the Racah angular plan, and count, plan, and optionally compile the bounded tagged-Cauchy physical image through `ye3t.couplings.count`, `plan`, and `compile`. |
| `tagged_cauchy_linear_coordinates.py` | Count and compile the rank-four scalar tagged-Cauchy physical coordinates matching the catalogue used by `ye3t-methods/examples/quickstart/tagged_fit.py`. |
| `ordered_role_cauchy_factors.py` | Supply ordered role-by-angular factor multiplets, compile every nontrivial Young/Cauchy coordinate, and check the factor action on the tableau axis. |

Default output checks for the short core examples:

| example | expected result |
|---|---|
| `symmetric_count_formula.py` | `{0: 1, 2: 1, 4: 1}` for both formula and compiler counts. |
| `symmetric_density_factors.py` | One scalar label, feature shape `(1, 1)`, factor-exchange invariant. |
| `user_supplied_factors.py` | Full count three and output shape `(3, 2, 3)` for `(a,t,M)`. |
| `symmetric_external_factors.py` | Two independent paths, shape `(2, 1, 3)`, with occupation contraction inside the repeated block. |
| `antisymmetric_external_factors.py` | One independent path, shape `(1, 1, 3)`, with a wedge contraction inside the repeated block. |
| `cauchy_supplied_density.py` | One `κ=(1,1)` multiplet with output shape `(1, 3)`. |
| `tagged_cauchy_linear_coordinates.py` | 81 compiler raw coordinates and 14 independent scalar physical coordinates for the rank-four catalogue. |
| `ordered_role_cauchy_factors.py` | Three independent copies and output shape `(3, 2, 3)` for `(a,t,M)`. |

Bounded benchmarks and diagnostics live in `benchmarks/`. They keep their
output and cache paths visible in `cfg_ye3t`.

For the shortest core route, run `coupling_multiplicity_counts.py`: its visible
seven-section config creates `YE3TRepresentation`, and one
`count_fixed_content` call returns the compiler's count and valid labels.
For ACE coefficients, run `coupling_coefficient_materialization.py` with the
same seven-section shape. It compares the exact count with the factorized
schedule size and reports the nonzero schedule terms.
Start with `symmetric_density_factors.py` when replacing another package's
permutation-invariant angular coupling. Its input is a complete `m` multiplet
for each channel. For a nontrivial Young parent, use the ordered-factor contract
in `user_supplied_factors.py`: preserve factor order and supply the complete
rotation multiplet in the stated convention. An ordinary commutative density
cannot realize a nontrivial global Young sector. `cauchy_supplied_density.py`
shows how a globally symmetric product can still have a nontrivial *local*
Cauchy path when role coordinates are retained. The latter uses real tesseral
input order; the user-supplied example uses complex Condon–Shortley order.
`ordered_role_cauchy_factors.py` keeps role-resolved factors distinct, so it
can evaluate nontrivial global Young outputs. The commutative role-density
example supports a globally symmetric parent with nontrivial local Cauchy
paths. The user-supplied example evaluates every valid multiplicity coordinate from
the full exact count. The default external-factor compiler stores separate
local, binary angular, and Young maps and never builds the full orbit matrix.
The symmetric and alternating repeated blocks use occupation and determinant
contractions. Their bound runtimes reuse coefficient tensors, symmetric
occupation transitions, and alternating minor indices across evaluations.
`user_supplied_factors.py` also demonstrates a batched message input and its
factor gradient. These examples evaluate one ordered orbit fiber; physical
density or motif pooling belongs to the application. The bounded dense
typed-orbit matrix remains available with `allow_dense_reference=True` for
tests and comparisons. The current PyTorch route loop is a correctness
baseline; production message-passing lowering into the packed execution plan
and high-rank compiler scaling remain open.

In the paper notation, `a` indexes independent multiplicity copies and the
output axes are `(a,t,M)`. The saved compiler schema retains the older
`alpha_index` key for compatibility. On a selected-coordinate artifact it is
the local output-axis index (zero); `full_alpha_index` in the public resolved
label identifies the coordinate in the complete `(lambda,L)` sector. For a scalar tagged potential, the
`tagged_cauchy_linear_coordinates.py` catalogue matches the basis choices in
the `ye3t-methods` tagged fit example. Core `ye3t` compiles its coordinates;
`ye3t-methods` constructs atomistic descriptor rows and fits energy, force,
and stress labels. The methods example compiles the core tagged-Cauchy artifact
and passes it to `Basis.from_config(..., compiled_cauchy_artifact=artifact)`.
The basis checks the request and coefficient hash before evaluating descriptors.

`symmetric_count_formula.py` compares YE3T's counts with the independently
implemented symmetric formula associated with GE-PI. The example runs offline;
it compares multiplicities for the same fixed content and every output `L`,
without assuming the two methods use the same coefficient coordinates.
The [GE-PI paper](https://arxiv.org/html/2604.01975v2) links its
[replication archive](https://doi.org/10.5281/zenodo.20331400) for users who
also want to run the authors' implementation. This example compares the
published count formula locally; it does not execute that archive.
If your feature package uses a real spherical-harmonic basis, convert each
complete multiplet to the complex Condon–Shortley convention before using the
user-supplied or symmetric-density examples. Check its `m` order and phases;
radial functions alone do not determine a compatible coupling convention.

`ye3t` deliberately does not accept ASE atoms; use the
`ye3t-methods/examples/quickstart/ase_descriptors.py` example for atomistic
descriptor arrays.

The exact primitive catalog is an advanced saved-object workflow: edit the
integer-keyed rank entries in `cfg_ye3t`, construct `YE3TFixedContentBasis`,
call `build_basis()` and `calculate_pd()`, then inspect the saved objects.
`yb.primitive.basis.blocks(rank, cap)` returns exact reduced coefficient
matrices; `yb.primitive.basis.vector(rank, cap, scope="local")` selects a
normalized primitive vector with exact coordinates in `.terms`. Global blocks
also carry their Young induction multiplicity. `build_basis()` constructs the
reduced exact product matrices; full Young placement coefficient tables are
compiled only when `yb.basis.induction_map(rank, cap, partitions)` is called.

The catalog's former diagnostic cases remain in
`tests/test_core_counts_and_characters.py`: legacy rank-2/3 normalization,
rank-3 child angular/Young limits, rank-6 route filters and recoupling, the
mixed `S_6` induction checks, and `S_8`/`S_16` rank checks. The example keeps
only the cases needed to explain the quotient and its output.

| benchmark | purpose |
|---|---|
| `benchmarks/benchmark_permutation_subduction_fastpath.py` | Compare exact symbolic and numeric generator-nullspace permutation-subduction construction, with residual and exact-projector validation. |
| `benchmarks/coefficient_materialization_benchmarks.py` | Write cold and cached coefficient-materialization timing rows, a CSV table, and an optional plot for rotation-only, permutation-only, and joint Young/O(3) families. |
| `benchmarks/primitive_cache_policies.py` | Compare conservative and eager primitive-cache lookup policies on a small exact sector. |
| `benchmarks/symmetric_power_kernels.py` | Compare optional folded symmetric-power monomial kernels, including a rank-8 case, with reference paths. |

Tested Sphinx snippets in `docs/representation_snippets.rst` cover copy/paste
counting, basis enumeration, generalized representation traces, exact
change-of-group metadata, structured sector labels, and schedule metadata.
The Sphinx pages `operator_ir_and_schedules` and `young_primitive_construction`
describe the lower-level rules behind the schedule and primitive examples.

Generated reports, long benchmark outputs, and paper figure pipelines should not
be package examples.
