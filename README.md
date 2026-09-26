# ye3t

General `G_N x SO(3)` / E(3) equivariance infrastructure.

This package is the stable core for:

- permutation-representation and angular representation labels,
- fast multiplicity and basis-count enumeration,
- exact/full/primitive coupling catalogs,
- Clebsch-Gordan tensor products,
- backend schedules and runtime lowering,
- optional accelerator bridges.

The stable API should preserve the descriptive mathematical metadata carried by
the combined research package: permutation representation, Young-diagram labels,
angular targets, multiplicity provenance, exact/full/primitive distinctions,
count sources, and backend fallback reasons.

## Related packages

- [ye3t-lammps](https://github.com/ye3t-equivariance/ye3t-lammps) provides
  LAMMPS inference (`pair_style ye3t` and `ye3t/kk`) for models compiled with
  this package.
- `ye3t-ace` is the separate application package for ACE and YE3T-equivariant
  message-passing model construction, fitting, and ASE calculators. It is not
  yet publicly released; references to it in this repository describe how it
  consumes the `ye3t` API and are not part of this package.

## Requirements

- Python 3.11 or newer.
- A C++20 compiler; the editable and wheel builds compile the tracked C++
  sources.
- `torch` and `numpy`.
- Optional: a CUDA toolkit matching a CUDA-enabled Torch wheel for the CUDA
  runtime, `patchelf` for relocatable Linux wheels, and `sympy` for
  `examples/exact_full_primitive_catalog.py` and
  `examples/tagged_cauchy_image_catalogue.py` (`pip install ".[reference]"`).

Set `YE3T_DISABLE_TRITON=1` to force the native/PyTorch paths when Triton is
installed but should not be used.

## Installation

Source install from a clone:

```bash
git clone https://github.com/ye3t-equivariance/ye3t.git
cd ye3t
python -m pip install "setuptools>=77,<82" torch wheel
python -m pip install -e . --no-build-isolation
```

Use `-e ".[dev]"` to add the test dependencies and `".[reference]"` for the
sympy-based exact examples. The package is not yet published on PyPI; source
install is the supported route for now.

The first `pip install` is required because the editable build compiles the
tracked C++ and optional CUDA sources. Both commands must use the same Python
interpreter. Torch is intentionally absent from `build-system.requires`, so a
native source install must use `--no-build-isolation`.

When that interpreter has CUDA-enabled Torch and a visible CUDA toolkit, the
CUDA runtime is built automatically. Set `YE3T_BUILD_CUDA_EXTENSION=0` only for
an intentional CPU-only build, or `YE3T_BUILD_CUDA_EXTENSION=1` to require a
CUDA build and fail if its toolchain is unavailable. Confirm the installed
runtime with:

```bash
python -c "from ye3t.runtime import native_execution_plan_capabilities as c; print(c())"
```

Optional accelerators are grouped for convenience:

```bash
python -m pip install -e ".[accelerators]" --no-build-isolation
```

Individual extras are also available for narrow environments: `triton`,
`oeq`, and `cueq`.

## Running tests

```bash
python -m pip install -e ".[dev]" --no-build-isolation
python -m pytest -m fast
```

`-m fast` is the refactor-safe subset run by CI. Omit the marker for the full
suite, which includes slow validation and optional GPU/accelerator tests that
skip when their hardware or packages are absent.

## Building the documentation

```bash
python -m pip install -e ".[docs]" --no-build-isolation
sphinx-build docs docs/_build/html
```

The package must be installed because the API reference uses autodoc.

## Import Surface

`import ye3t` is intentionally lightweight. It exposes stable mathematical
objects such as `YE3TAPI`, `ExactProductExpansionEngine`,
`PermutationIrrep`, `CoupledIrrepLabel`, and count-only helpers such as
`enumerate_rank_labels`.

Common entry points include `YE3TAPI.reduce_ye3t`,
`YE3TAPI.summarize_ye3t`, `format_ye3t_basis`,
`compile_ye3t_operator`, and `compile_ye3t_operators`.

Runtime compilers, native modules, and accelerator bridges are
available from explicit submodules and lazy symbols in `ye3t.api`. Importing
`ye3t` or `ye3t.api` should not eagerly import Triton, OpenEquivariance,
cuEquivariance, ASE, or ACE code.

## Intended Extras

- `ye3t[accelerators]`: Triton, OpenEquivariance, and cuEquivariance bridges.
- `ye3t[all]`: currently equivalent to `ye3t[accelerators]`.

All optional stacks should import lazily and fall back to native/PyTorch paths
where possible.

## Rigor Rules

This package should expose exact algebra with provenance, not only convenience
counts. Stable labels and catalogs are expected to retain the permutation
character, angular target, multiplicity index or source, coupling/tree
provenance, exact/full/primitive status, and dimension source.

## Current Capability Limits

These limits are intentional first-release boundaries rather than silent
fallbacks:

- Some exact projector paths are capped to small permutation ranks; larger
  ranks should use count-only, character, cached, or fallback-reported APIs.
- Selected exact basis builders require explicit backend metadata and will
  raise a descriptive error when an unsupported enumeration mode is requested.
- Generic graph optimization and whole-DAG fusion remain deferred from the
  stable API until bounded end-to-end benchmarks justify a stable cost model.
- Accelerator bridges are optional, lazy, and skip/fallback aware.
  They are not imported by `import ye3t` and should always preserve native or
  PyTorch paths when an optional dependency is missing.

## Examples

Small, deterministic examples live in `examples/` and are included in source
distributions. Importable public scripts expose visible editable `cfg_ye3t`
dictionaries and one purpose-named `run_*` workflow when the script represents
a reusable workflow. Smaller count, representation, and schedule demonstrations
live in `docs/representation_snippets.rst` as tested copy/paste snippets.

- `coupling_multiplicity_counts.py`: counts valid fixed-content coupling
  labels through `ye3t.couplings.count`.
- `coupling_coefficient_materialization.py`: plans and compiles coupling
  coefficients through `ye3t.couplings.plan` and `ye3t.couplings.compile`.
- `compile_scalar_ace_lammps_plans.py`: compiles representative scalar ACE
  coordinates and execution plans for downstream LAMMPS binding through
  `ye3t-lammps`.
- `exact_full_primitive_catalog.py`: compares exact/full/primitive product
  catalog dimensions for a small sector.
- `symbolic_young_partition_catalogue.py`: expands symbolic Young partition
  templates by rank and counts the valid O(3) sectors of each partition.
- `tagged_cauchy_image_catalogue.py`: builds the exact shifted-Jacobi source
  product algebra for a user-chosen source set and counts, plans, and
  optionally compiles the bounded tagged-Cauchy physical image.

Bounded benchmarks and diagnostics live in `examples/benchmarks/`:

- `benchmark_permutation_subduction_fastpath.py`: compares exact symbolic and
  numeric permutation-subduction construction with validation.
- `coefficient_materialization_benchmarks.py`: writes coefficient
  materialization timing artifacts for rotation-only, permutation-only, and
  joint Young/O(3) families.
- `primitive_cache_policies.py`: compares primitive-cache lookup policies on
  a small exact sector.
- `symmetric_power_kernels.py`: validates optional folded symmetric-power
  monomial kernels, including a rank-8 case, against reference paths and
  reports bounded timing comparisons.

## License and authors

BSD-3-Clause, copyright (c) 2026 James M. Goff; see `LICENSE` and `AUTHORS.md`.

## Citation

Citation metadata is in `CITATION.cff`.
