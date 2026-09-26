# ye3t

`ye3t` is a representation-theory compiler for `G_N x SO(3)` / E(3)
equivariant bases. Given the content of a product of atomic (or other)
factors, it enumerates the valid permutation- and rotation-symmetry-adapted
labels, counts their multiplicities exactly, materializes the coupling
coefficients with a validation certificate, and lowers the result to
execution plans that run on its native CPU/CUDA runtime or through optional
accelerator kernels. The ordinary symmetric sector of that basis is the
linear Atomic Cluster Expansion (ACE); the nontrivial Young sectors extend it.

## Quick start

```python
from ye3t.couplings import count, plan, compile

report = count(content=(1, 1, 1), input_Ls=(0, 1, 1), target_L=0)
labels = report.labels_for_target(0)
report.require_label(labels[0], target_L=0)

coupler_plan = plan(report)
compiled = compile(coupler_plan)
print(len(labels), coupler_plan.backend, compiled.convention_hash)
```

`count` returns the exact multiplicity report, `plan` records the backend and
count provenance, and `compile` materializes the coefficients and attaches the
validation report. Every result carries its convention hash and provenance.
The pages under `docs/` walk through fixed-content couplers, pure rotation and
pure permutation cases, validation reports, execution plans, and the native
runtime.

## Related packages

- [ye3t-lammps](https://github.com/ye3t-equivariance/ye3t-lammps) provides
  LAMMPS inference (`pair_style ye3t` and `ye3t/kk`) for models compiled with
  this package.
- `ye3t-ace` is the separate application package for descriptor
  construction, model fitting, and ASE calculators. It is not yet publicly
  released; the documentation here mentions it where it consumes the `ye3t`
  API.

## Requirements

- Python 3.11 or 3.12 (the versions exercised by CI).
- A C++20 compiler; the source install compiles the tracked C++ sources.
- `torch` and `numpy`.
- Optional: a CUDA toolkit matching a CUDA-enabled Torch wheel for the CUDA
  runtime, `patchelf` for relocatable Linux wheels, and `sympy` for the
  exact symbolic examples and benchmarks (`pip install ".[reference]"`).

## Installation

`ye3t` is not yet published on PyPI; install it from a source clone:

```bash
git clone https://github.com/ye3t-equivariance/ye3t.git
cd ye3t
python -m pip install "setuptools>=77,<82" torch wheel
python -m pip install -e . --no-build-isolation
```

The first `pip install` provides the build tools and the Torch headers the
extension build needs; Torch is intentionally absent from
`build-system.requires`, so the install must use `--no-build-isolation`. Both
commands must use the same interpreter. On a machine without CUDA, add
`--extra-index-url https://download.pytorch.org/whl/cpu` to the first
command to get the CPU-only Torch wheel. Use `-e ".[dev]"` for the test
dependencies, `".[reference]"` for sympy, and `".[docs]"` for Sphinx.

When the interpreter has CUDA-enabled Torch and a visible CUDA toolkit, the
CUDA runtime is built automatically. Set `YE3T_BUILD_CUDA_EXTENSION=0` for
an intentional CPU-only build, or `YE3T_BUILD_CUDA_EXTENSION=1` to require a
CUDA build and fail if its toolchain is unavailable. Confirm the installed
runtime with:

```bash
python -c "from ye3t.runtime import native_execution_plan_capabilities as c; print(c())"
```

Optional accelerator extras: `accelerators` (Triton, OpenEquivariance, and
cuEquivariance together), or individually `triton` (Triton kernels), `oeq`
(the OpenEquivariance bridge), and `cueq` (the cuEquivariance IR export).
They load lazily, and the native/PyTorch paths remain available without
them; `dependency_matrix.md` lists every extra.

```bash
python -m pip install -e ".[accelerators]" --no-build-isolation
```

### Building a wheel

```bash
python -m pip install build
python -m build --no-isolation
python cmake/audit_linux_wheel.py dist/*.whl
python -m pip install --force-reinstall --no-deps dist/*.whl
python cmake/smoke_installed_package.py
```

### Standalone C++ runtime

The native runtime under `ye3t/runtime/csrc` also builds as a standalone
CMake project (`ye3t-lammps` compiles the same runtime source from a `ye3t`
checkout):

```bash
cmake -S . -B build -DYE3T_TORCH_DISCOVERY=PYTHON
cmake --build build
(cd build && ctest --output-on-failure)
```

`YE3T_BUILD_TORCH_ADAPTER` (default `ON`) adds the PyTorch adapter and
`YE3T_BUILD_CUDA_ADAPTER` (default `ON`) its CUDA operators; set the latter
to `OFF` on hosts without a CUDA toolkit. `docs/native_execution_plan.rst`
describes the options and the install layout.

## Environment variables

| Variable | Effect |
| --- | --- |
| `YE3T_BUILD_CUDA_EXTENSION` | `auto` (default), `1` to require the CUDA build, `0` for CPU only |
| `YE3T_SKIP_CPP_EXTENSION=1` | install without compiling the native extensions |
| `YE3T_CACHE_DIR` | root directory of the on-disk artifact cache |
| `YE3T_CACHE_MODE` | `auto` (default), `read_only`, `rebuild`, or `off` |
| `YE3T_CACHE_VERIFY` | `hash` (default) or `full` verification of cached artifacts |
| `YE3T_DISABLE_TRITON=1` | disable the Triton coupling kernels in `ye3t.backends`; the native/PyTorch paths are used instead |
| `YE3T_DISABLE_OPENEQUIVARIANCE=1` | skip the OpenEquivariance bridge in automatic packed-CG dispatch (an explicit `backend="openequivariance"` still uses it) |
| `YE3T_REQUIRE_NATIVE=1` | raise instead of falling back to reference implementations |
| `YE3T_DEBUG_TRITON=1` | verbose Triton diagnostics |

## Package layout

- `ye3t.couplings`: the public entry point for labels, multiplicity counts,
  plans, and coefficient materialization, plus the scalar ACE,
  tagged-Cauchy, lifted-Cauchy, and covariant-Cauchy compilers.
- `ye3t.core`: labels, spherical and tesseral harmonics, rotations, exact
  couplings, basis construction, and the factorized-runtime extension.
- `ye3t.representations`: Young orthogonal, Yamanouchi, and Specht
  constructions, numeric subduction, SU(2), projectors, and the
  permutation-subduction extension.
- `ye3t.ir` and `ye3t.lowering`: operator IR records, exact schedules,
  materialization plans, and `compile_ye3t_operator(s)`.
- `ye3t.runtime`: versioned execution plans and the native CPU/CUDA
  operators.
- `ye3t.backends` and `ye3t.adapters`: PyTorch and Triton lowering, the
  OpenEquivariance bridge, and the cuEquivariance IR export.
- `ye3t.cache`: the hash-bound on-disk artifact store and symbolic caches.
- `ye3t.utils`: printing and illustration helpers used by the docs.
- `ye3t.api`: a lazy facade over the compilers, runtime, and accelerators.
  `import ye3t` exposes `YE3TAPI`, `ExactProductExpansionEngine`,
  `PermutationIrrep`, `CoupledIrrepLabel`, `enumerate_rank_labels`, and
  `format_ye3t_basis` without importing the native runtime modules or the
  optional accelerator packages.

## Examples

Each script in `examples/` shows its editable `cfg_ye3t` dictionary and one
`run_*` function; run it with `python examples/<name>.py`. `examples/README.md`
describes them in full.

- `coupling_multiplicity_counts.py`: count valid fixed-content coupling
  labels through `ye3t.couplings.count`.
- `coupling_coefficient_materialization.py`: plan and compile coupling
  coefficients through `ye3t.couplings.plan` and `ye3t.couplings.compile`.
- `compile_scalar_ace_lammps_plans.py`: compile scalar ACE coordinates and
  execution plans for `ye3t-lammps`.
- `exact_full_primitive_catalog.py`: compare exact, full, and primitive
  product catalog dimensions for a small sector.
- `symbolic_young_partition_catalogue.py`: expand symbolic Young partition
  templates by rank and count the valid O(3) sectors.
- `tagged_cauchy_image_catalogue.py`: build the shifted-Jacobi source
  product algebra for a chosen source set and count, plan, and optionally
  compile the tagged-Cauchy physical image.

`examples/benchmarks/` holds bounded timing and validation benchmarks for
permutation subduction, coefficient materialization, primitive caches, and
symmetric-power kernels; see `examples/benchmarks/README.md`.

## Tests

```bash
python -m pip install -e ".[dev]" --no-build-isolation
python -m pytest -m fast
```

`-m fast` is the subset run by CI. The full suite adds the `slow` tests and
the `optional`, `gpu`, `triton`, `oeq`, and `cueq` tests, which skip when
their hardware or packages are absent.

## Documentation

```bash
python -m pip install -e ".[docs]" --no-build-isolation
sphinx-build -W docs docs/_build/html
```

## License and authors

BSD-3-Clause, copyright (c) 2026 James M. Goff; see `LICENSE` and `AUTHORS.md`.

## Citation

Citation metadata is in `CITATION.cff`.
