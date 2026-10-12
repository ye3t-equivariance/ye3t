# Rank coverage in compiler examples

- `compile_scalar_ace_lammps_plans.py`: expand to every compiler-selected scalar
  coordinate in the configured rank 1–8 contents. Keep the homogeneous rank-16
  coordinate and the rank-three coupled-product plan. Move export orchestration
  into `ye3t.scalar_ace_export`; preserve plan JSON checks and coefficient validation.
- `tagged_cauchy_image_catalogue.py`: use the general tagged catalogue for ranks
  1–8. Preserve the specialized rank-four source-product calculation in
  `extras/tagged_cauchy_rank4_product_algebra.py`.
- `tagged_cauchy_linear_coordinates.py`: retain the standalone rank-four
  coordinate system shared with `ye3t-methods/examples/quickstart/tagged_fit.py`.
  The rank 1–8 catalogue selects exactly the same physical coordinates when
  each rank is compiled separately or together. Its rank-four count remains 14.
  The fitting example keeps its smaller configuration for a focused end-to-end fit.

The standard config and compiler own all selected contents, labels, and paths.
The specialized example and fitted coordinate system remain runnable.
