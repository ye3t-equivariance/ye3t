# YE3T benchmarks

Benchmark, profiling, timing, and ablation examples live here.

Rules:

- Keep full inline homogeneous configs in runnable benchmark scripts.
- Put benchmark-specific settings under visible runtime/benchmark config keys.
- Write exploratory outputs to `examples/generated/...`.
- Commit only small representative JSON timing artifacts that support docs,
  paper figures, or reproducible engineering decisions.
- Keep machine metadata optional and disabled by default.

Current scripts:

- `benchmark_permutation_subduction_fastpath.py`: exact symbolic versus
  numeric permutation-subduction construction with validation.
- `coefficient_materialization_benchmarks.py`: coefficient materialization
  timing artifacts for rotation-only, permutation-only, and joint
  Young/O(3) families.
- `primitive_cache_policies.py`: primitive-cache lookup policies on a small
  exact sector.
- `symmetric_power_kernels.py`: folded symmetric-power monomial kernels
  against reference paths with bounded timing comparisons.
