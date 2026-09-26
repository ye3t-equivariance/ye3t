# General tagged-Cauchy physical image

The experimental `tagged_cauchy_image_request(catalogue=..., species=...)`
uses count/plan/compile. Chemical channels are indicators, not learned embeddings.
The repeated-channel key is neighbor species, radial degree, angular degree and
source family; the application binds central species and directed-pair cutoff.
The certified tag-count scope is `s=0,1,2`. The algebra has no fixed tensor-order
or angular-degree restriction; compiler resource checks still apply.

For a fixed center, collided factors of one species share `x=r/r_cut(pair)`.
Products of different species indicators vanish. Racah harmonic products use
`C_lm C_kn = sum_L CG(l,0;k,0|L,0) CG(l,m;k,n|L,m+n) C_L,m+n`.

For `k` radial factors, remove the output source's `x^L (1-x)^2`. The remaining
polynomial is `x^(sum l-L) (1-x)^(2(k-1)) prod P_q^(4,2l+2)(2x-1)`.
Exact triangular back-substitution expands it in the output Jacobi basis,
including `sqrt(prod norm_squared / output_norm_squared)`. Closure is untruncated.
Distinct tags use Möbius inclusion-exclusion: for two tags, `M[f]M[g]-M[fg]`.
Density contexts remain inclusive.

Sparse moment polynomials are reduced by exact pivots. Survivors remain original
compiler coordinates. Every discarded attempted row has an exact reconstruction.
Selection in ascending tensor order eliminates cross-rank dependencies. Named
coordinates must survive that policy. Feature caps bound the selected subspace
and do not assert the complete uncapped mathematical image dimension.

Physical independence follows because the finite one-body species/Jacobi/Racah
functions are linearly independent: choose `d` points with invertible evaluation
matrix `F`. Environments with multiplicities `n` give moments `Fn`. A polynomial
vanishing on all these environments vanishes on `N^d`, hence identically.
Coincident samples can be approached continuously by distinct points. The claim
concerns **all finite coordination numbers**, not a fixed maximum coordination
or a finite dataset. Each directed species pair has one fixed radial support.

Version-4 compiler artifacts embed compiler-produced, hash-bound, internally
validated lifted-Cauchy parents. Loading replays exact physical lowering from
those parents, pivots, source inventory and real schedules. It does not regenerate
the parent's named coupling coefficients: this preserves the existing compiler
artifact/cache trust boundary. Process-local validation caches rehash current
complete bytes before reuse. Old N=4 orthonormal artifacts retain version-3 meaning.

General V4 loaders accept `compiler_validation="certificate"` to trust the
stored symmetry and physical-image proof instead of replaying it. The default
`"full"` preserves the exact replay described above. Certificate mode checks
request/plan/certificate identities, source polynomial tables, real forms,
selected-coordinate bindings, exact/binary64 executable coefficients and the
independently derived runtime adjoint. It does not reconstruct parent couplings,
perform physical lowering or repeat the sparse independence calculation.
Integrity hashes are not signatures. Use full mode to re-prove the artifact's
algebraic claims; a certificate-only load never populates the full-replay cache.
The application example exposes the same policy for cache loads and model
reloads, while newly compiled cache entries always receive full validation.

Image reduction uses no Gram–Schmidt. Existing role and scalar-angular templates
use exact pivots/nullspaces. The general workflow selects
`angular_basis_backend="exact_weight_space_v1"` for positive angular momenta:
content-class coset matrix units construct the Young first-row image in weight
`M=L`; the kernel of `J+` selects highest weights; normalized `J-` steps generate
the other magnetic components without repivoting. Exact multiplicities, Young
actions, ladders and low-rank legacy spans validate this construction. It avoids
full magnetic projectors and unrelated L sectors. The new copy gauge is bound
into requests, labels and template caches; scalar templates remain unchanged.

The equal-value stabilizer sum is factored through the chain of symmetric
subgroups. With adjacent matrices G and an interval ending at p,
`S_r=(I+G_(p-1)+G_(p-2)G_(p-1)+...+G_a...G_(p-1))*S_(r-1)`.
The left multiplication follows from the established anti-representation
convention and left-coset decomposition. Exact small-group sums, generator
invariance, symmetry and `S_r^2=r!*S_r` certify the replacement. Its cost is
quadratic in interval length in matrix products, rather than factorial
permutation enumeration; matrix dimensions and exact arithmetic still matter.
The new backend therefore uses explicit resource limits without a fixed
eight-slot block ceiling. Legacy full-permutation requests retain that guard.

The legacy full-sector path remains for old artifacts and as an oracle. Outer
coupling still uses its existing cached construction. Numerical Young subduction
solves a different intertwiner problem; its redundant QR after SVD was removed,
while the independent projector-validation oracle retains QR.

## Why selecting two order-eight coordinates can still be expensive

The current selection boundary is too late for some costs. In
`tagged_cauchy_general.py`, `_general_count` enumerates complete-content parent
labels before `_general_compile` applies named-coordinate and feature limits.
The selected candidate is parent-compiled with one manual label, but
`lifted_cauchy_scalar._build_block_template` still constructs all role/angular
multiplicity copies, tableaux and magnetic components for its block key.
`_instantiate_descriptor` subsequently reads only the requested rows.

For two tags the role dimension is three. The present ordered-state preflight
for the `[4,4]` partition estimates `9^8 = 43,046,721` states for `[1]*8`, and
`9^4 * 15^4 = 332,150,625` for `[1]*4+[2]*4`. The former already exceeds the
40M-cell guard and represents about 2.57 GiB at 64 bytes per cell; the latter
also exceeds the 100M ordered-state guard. These are conservative bounds for
the current full-template representation, not measured minimum resources for
one or two final scalar descriptors.

This path uses exact coset/weight-space carriers and exact metric algebra;
switching the separate numerical subduction backend does not remove this
all-copy template construction. The required optimization is selection-aware
counting and a requested-row template compiler, with bounds based on proved
content/weight support. A requested metric-dual row can depend on other copies
in the same Gram block: those coupled columns must remain in the closure to
preserve the existing coordinate gauge, phases, indices and normalization.
Exact selected-row equality against feasible full templates and the N=4 oracle
is required before claiming this optimization. The general image reducer must
still account for zero/dependent candidates and retained lower-order rows.
This optimization is not implemented; merely raising the limits leaves the
excess construction in place.

References (independent implementations; no external code copied):

- R. Drautz, *Atomic cluster expansion for accurate and transferable interatomic
  potentials*, Phys. Rev. B **99**, 014104 (2019), section II A and appendices A/B,
  [DOI](https://doi.org/10.1103/PhysRevB.99.014104): chemistry, density products and
  nonorthogonal expansion bases.
- [DLMF 34.3(vii)](https://dlmf.nist.gov/34.3#vii): harmonic products and angular
  momentum coefficients, using existing compiler CG implementation.
- [LAMMPS ZBL](https://docs.lammps.org/pair_zbl.html) and
  [GROMACS switch](https://docs.lammps.org/pair_gromacs.html): mathematical and
  behavioral references for portable ZBL; no GPL implementation was translated.
