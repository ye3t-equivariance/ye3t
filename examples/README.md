# ye3t Examples

These examples are small, deterministic demonstrations of the
representation-theory package surface.

| example | purpose |
|---|---|
| `coupling_multiplicity_counts.py` | Count valid fixed-content coupling labels through `ye3t.couplings.count`, including provenance and validation status. |
| `coupling_coefficient_materialization.py` | Plan and compile coupling coefficients through `ye3t.couplings.plan` and `ye3t.couplings.compile`, including backend, convention hash, validation, and certificate metadata. |
| `compile_scalar_ace_lammps_plans.py` | Compile representative rank-3, H4, `[4,4]`, and H16 scalar ACE coordinates plus block and v3 coupled-product execution plans for downstream LAMMPS binding through `ye3t-lammps`. |
| `exact_full_primitive_catalog.py` | Build a saved fixed-content basis for ranks 4, 6, and 16; inspect primitive and decomposable representations and request exact coefficient blocks or one normalized vector. |
| `symbolic_young_partition_catalogue.py` | Expand symbolic Young partition templates by rank through `ye3t.couplings.expand_partition_templates` and print the positive O(3) multiplicities of each expanded partition. |
| `tagged_cauchy_image_catalogue.py` | Build the exact shifted-Jacobi source product algebra with `ye3t.couplings.build_radial_species_product_record`, bind it to the Racah angular plan, and count, plan, and optionally compile the bounded tagged-Cauchy physical image through `ye3t.couplings.count`, `plan`, and `compile`. |

Bounded benchmarks and diagnostics live in `benchmarks/`. They keep their
output and cache paths visible in `cfg_ye3t`.

The catalog is the current example of the short object workflow: edit the
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
