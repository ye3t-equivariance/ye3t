# Covariant Cauchy bases for arbitrary `(L, parity)` and Young outputs

## Ordered role-resolved factors and general Young outputs

The `ordered_role` carrier is the factor-resolved extension of the commuting
`A_s` construction. For complete channel blocks `b` with multiplicities `k_b`,
each input factor has a role coordinate in `W_b = C^d` and a full angular
multiplet in `V_{l_b}`. The fixed-content carrier is

\[
\widehat{\mathcal H}_{\boldsymbol\nu}
=\bigoplus_{\sigma\in S_N/(\prod_b S_{k_b})}
\bigotimes_b (W_b\otimes V_{l_b})^{\otimes k_b}.
\]

Within block `b`, a local Young type `mu_b` splits into role type `kappa_b`,
angular type `rho_b`, their Kronecker copy, a role Schur copy, and an angular
copy at `Lambda_b`. Global Young subduction contributes a Littlewood-Richardson
copy, while binary angular coupling contributes an angular path. The number
of independent output multiplets is

\[
m_{\lambda L}=
\sum_{\boldsymbol\mu,\boldsymbol\Lambda}
c_{\boldsymbol\mu}^{\lambda}\,\mathcal N_{\boldsymbol\Lambda}^{L}
\prod_b\left(
  \sum_{\kappa_b,\rho_b}
  g_{\mu_b\kappa_b\rho_b}
  \dim\mathbb S_{\kappa_b}(W_b)
  d_b^{\rho_b\Lambda_b}
\right).
\]

Each copy `a` yields every tableau coordinate `t` and magnetic coordinate `M`
of the paper's `B_{i,alpha}`, with `alpha=(lambda,L,a,t,M)`. This follows the
block Schur-Weyl, multiplicity, and coupling construction in
[Goff and Thompson, Eqs. (7)–(12)](https://arxiv.org/abs/2609.31895).

The local compiler checks exact factor-permutation action and all role,
angular, and Kronecker copies. Distinct local copy families can overlap even
when each family is normalized. If `S` contains their synthesis vectors, the
compiler forms `G=S^H S=H H^H` and uses `Q=S H^{-H}`. Thus `Q^H Q=I` for the
complete local copy space. The original family dual `S G^{-1}` remains in the
low-level report for comparison; the ordered-role evaluator uses `Q`. Young
and binary angular maps then act on these orthonormal local coordinates. The
full orbit-by-coupled-coordinate matrix `C_{q alpha}` is never assembled.
For larger repeated blocks, the compiler also keeps role and angular vectors
separate. The local evaluation contracts the ordered factors first against
role vectors, then against angular vectors and the orthonormal copy kernel.
If `r` and `s` are the numbers of retained role and angular columns, this
stores `r d^k + s(2l+1)^k` component entries plus the copy kernel, instead of
the joint `d^k(2l+1)^k` product table for every output column. Small blocks retain a sparse local
table that serves as an independent reference for the tensor network.

The copy indices in each route name the lexicographically ordered seed
families before this transform. The saved local table records their
`family_order`, `copy_gram`, and
`copy_gauge="full_family_cholesky_orthonormal_v1"`. An orthonormal output copy
can combine several seed families; its label is an index in that fixed copy
gauge, not a claim that its vector has only one pure role/angular/Kronecker
factorization.

The public path is `covariant_cauchy_request(..., carrier="ordered_role",
target_permutation=..., target_L=...)` followed by `ye3t.couplings.count`,
`plan`, and `compile`. Call `ye3t.couplings.validate_covariant_cauchy` when
loading a saved artifact. For repeated Torch evaluation, bind its constant
tensors once with `ye3t.couplings.bind_ordered_role_cauchy_torch`; see
`examples/ordered_role_cauchy_factors.py`. Inputs are ordered
`(..., N, d, 2l+1)` for a common `l`, or one `(d, 2l_b+1)` multiplet per factor
when angular degrees differ. The output shape is `(..., a, t, M)`.

For commuting `A_s` inputs, the product itself is globally symmetric and only
`lambda=(N)` is a physical output. The ordered-role API keeps distinct factor
information and supports symmetric, antisymmetric, and general Young outputs.
This core evaluator couples supplied factors; physical source construction,
motif sums, and packed native message-passing lowering are separate application
operations. The current Torch evaluator reuses bound constants but still loops
over coupled routes in Python, so this path is experimental for high-throughput
message-passing workloads.

Status: **derivation with exact checks; referenced as the mathematical
standard by `ye3t/couplings/covariant_cauchy.py`.** It generalizes
`linear_lifted_cauchy_basis.md`, which is the special case of a commutative
feature product (`lambda = (N)`) and a global even scalar target (`L = 0`,
`epsilon = +`). Nothing here changes that standard; the scalar theorem is
recovered exactly in the section "Recovered special cases".

The dimension identities quoted below were checked numerically with exact
integer arithmetic (the `S_4` character table and the hook-content formula).
They are listed in "Exact checks" so that they can become unit tests.

New `covariant_cauchy_request(...)` calls use
`angular_basis_backend="exact_weight_space_v1"`. This exact compiler builds
the requested angular weight and lowers it to the full multiplet. Set
`angular_basis_backend="legacy_exact"` to use the full-sector oracle. Saved
requests without this field retain the legacy interpretation. Neither setting
selects the generic `numeric_cached` Young subduction backend.

The compiled real-tesseral output applies `(-i)^sigma`, where
`sigma = (sum_b k_b l_b - L) mod 2`. Imaginary entries in an intermediate
complex-basis transformation are expected; the compiler rejects any nonzero
imaginary coefficient after this phase and the real-basis transformation.

## 1. What is being constructed

A linear covariant model predicts a target `T` that transforms in a
representation `Y` of a symmetry group `G` from one-particle features
`a in U`:

\[
T(X)=\sum_N C_N\,\bigl[\mathcal P^N a(X)\bigr],
\qquad
C_N\in\operatorname{Hom}_G\!\bigl(\mathcal P^N(U),\,Y\bigr).
\]

`P^N` is the degree-`N` product functor realized by the physical carrier:

- `Sym^N(U)` for ordinary commuting densities;
- an ordered product of distinct slot carriers
  `U^(1) (x) ... (x) U^(N)` for slot- or role-resolved sources;
- `Lambda^N(U)`, realized only as the antisymmetrized image of `N`
  slot-resolved carriers (a single commuting vector has `a ^ a = 0`). Spinful
  fermionic carriers additionally require `SU(2)` or a declared double-group
  convention; spatial exterior powers alone do not support that claim.

Lower-order and constant terms are part of the model: a rank-`n` product
cannot represent a geometry-independent block unless each slot carrier
contains a constant scalar channel or the lower ranks are included
explicitly. Translation invariance is assumed throughout because every
one-particle feature is a function of relative positions.

A **covariant Cauchy basis** is an explicit, exactly counted, orthogonal basis
of the finite-dimensional model space `Hom_G(P^N(U), Y)`. The scalar
descriptor basis is the case `Y = V_0^+`.

The symmetry group is `G = O(3) x Gamma`, where `Gamma` is a finite group of
physical relabellings (identical atoms). Formal tensor-slot permutations are
**not** part of `G`; they organize the spaces through Schur-Weyl duality and
become physical only through the stabilizer construction of section 6.

### Schur's lemma gives the whole model structure

Decompose both sides into `G`-isotypic components,

\[
\mathcal P^N(U)\cong\bigoplus_\omega \mathcal F_\omega\otimes R_\omega,
\qquad
Y\cong\bigoplus_\omega \mathcal Y_\omega\otimes R_\omega ,
\]

with `R_omega` the irreducible `G`-carriers and `F_omega`, `Y_omega` the
feature and output multiplicity spaces. Then

\[
\operatorname{Hom}_G\bigl(\mathcal P^N(U),Y\bigr)
\cong
\bigoplus_\omega \operatorname{Hom}\bigl(\mathcal F_\omega,\mathcal Y_\omega\bigr).
\tag{1}
\]

**Field and type hypothesis.** (1) is stated over the real numbers with
real-tesseral carriers, and requires every irrep involved to be of real type
(`End_G(R_omega) = R`). `O(3)` irreps are of real type. Stabilizers generated
by relabelling identical atoms are products of symmetric and wreath-product
groups, which are rational groups, so their irreps are of real type as well.
The hypothesis fails for point groups with complex-type irreps (`C_n` with
`n >= 3`, `T`, `C_3h`), lattice translations, and half-integer `SU(2)` or
double-group spinors; there the commutant is `C` or `H` and coefficients act
as `a + bJ` on irrep components rather than as the identity. An
implementation certifies a Frobenius-Schur indicator of `+1` for every
stabilizer it emits.

Consequences:

1. the model is block diagonal in `omega = (L, epsilon, gamma)`;
2. the fitted coefficients are unconstrained matrices between multiplicity
   spaces, and act as the identity on the magnetic index `M` and on every
   component of a `Gamma`-irrep. This is the complete-carrier rule: learned
   maps mix only multiplicity and channel axes. At a binding readout
   (section 6) the formal tableau axis of `S^lambda` is a multiplicity axis
   of the physical group `O(3) x Gamma_b`, so a readout may combine tableau
   components and different `lambda_geom`. This is legitimate for a final
   readout but differs from the message-passing contract rule that forbids
   learned maps from mixing `lambda` or tableau components of hidden
   carriers, where formal `S_N` equivariance is imposed. See open point 5;
3. constructing the model basis reduces to constructing bases of the two
   multiplicity spaces. Sections 2-4 do this for the features and section 5
   does it for the outputs, with the same three identities.

## 2. The three identities

Let `S_mu` denote the Schur functor of a partition `mu` of `n`, and `S^mu` the
corresponding Specht module of `S_n`.

**Direct sums (Littlewood-Richardson).** For `U = (+)_b U_b`,

\[
\mathbb S_\mu\Bigl(\bigoplus_b U_b\Bigr)
\cong
\bigoplus_{(k_b),\;\alpha_b\vdash k_b}
\mathbb C^{\,c^{\mu}_{(\alpha_b)}}\otimes
\bigotimes_b \mathbb S_{\alpha_b}(U_b),
\qquad \sum_b k_b=n,
\tag{2}
\]

where `c^mu_(alpha_b)` is the iterated Littlewood-Richardson coefficient, the
multiplicity of `(x)_b S^{alpha_b}` in the restriction of `S^mu` to the Young
subgroup `prod_b S_{k_b}`.

**Tensor products (Kronecker).** For `U = W (x) V`,

\[
\mathbb S_\mu(W\otimes V)
\cong
\bigoplus_{\alpha,\beta\vdash n}
\mathbb C^{\,g_{\mu\alpha\beta}}\otimes
\mathbb S_\alpha(W)\otimes\mathbb S_\beta(V),
\tag{3}
\]

where `g_{mu alpha beta}` is the Kronecker coefficient, the multiplicity of
`S^mu` in `S^alpha (x) S^beta`.

*Derivation.* `(W (x) V)^{(x)n} = W^{(x)n} (x) V^{(x)n}` with `S_n` acting
diagonally. Apply Schur-Weyl duality to each factor,
`W^{(x)n} = (+)_alpha S_alpha(W) (x) S^alpha`, and likewise for `V`. The
`S^mu`-isotypic component of `S^alpha (x) S^beta` has multiplicity
`g_{mu alpha beta}`, which gives (3). Identity (2) follows in the same way
from restricting `S^mu` to a Young subgroup.

**Cauchy identities are the two extreme cases of (3).** Since
`S^(n)` is trivial and `S^(1^n)` is the sign representation,

\[
g_{(n)\alpha\beta}=\delta_{\alpha\beta},
\qquad
g_{(1^n)\alpha\beta}=\delta_{\alpha\beta'},
\]

with `beta'` the conjugate partition. Hence

\[
\operatorname{Sym}^n(W\otimes V)\cong\bigoplus_{\kappa\vdash n}
\mathbb S_\kappa(W)\otimes\mathbb S_\kappa(V),
\qquad
\Lambda^n(W\otimes V)\cong\bigoplus_{\kappa\vdash n}
\mathbb S_\kappa(W)\otimes\mathbb S_{\kappa'}(V).
\]

The symmetric form is the identity used by `linear_lifted_cauchy_basis.md`.
The skew form is the natural identity for spin-orbital or particle-slot
fermionic carriers. A general output or parent sector `mu` needs the full
Kronecker form (3); neither Cauchy identity alone is sufficient.

This matches the project invariant: rank-additive products of disjoint slot
sets use LR (2), and same-rank diagonal products use Kronecker (3).

**O(3) branching.** For an angular carrier `V_l` with its natural parity
`(-1)^l`,

\[
\mathbb S_\beta(V_l)\cong\bigoplus_{\Lambda}
\mathcal D^{\beta\Lambda}_l\otimes V_\Lambda^{\epsilon},
\qquad \epsilon=(-1)^{l|\beta|},
\tag{4}
\]

where `D^{beta Lambda}_l` is the multiplicity space of `V_Lambda` in
`S_beta(V_l)`. The parity is fixed by the block, so carriers with
`epsilon != (-1)^Lambda` (pseudo-tensors) occur from `|beta| >= 2` onward; for
example `Lambda^2(V_1) = V_1^+` is the axial vector.

## 3. Feature-side theorem

Fix the tensor rank `N`, the parent sector `lambda` of `N`, the block content
`U = (+)_b W_b (x) V_{l_b}` with block degrees `k_b`, and the target
`(L, epsilon)`. Combining (2), (3), and (4) and coupling the block angular
carriers to the target,

\[
\operatorname{Hom}_{O(3)}\!\Bigl(V_L^{\epsilon},\;
\mathbb S_\lambda\bigl(\textstyle\bigoplus_b W_b\otimes V_{l_b}\bigr)_{(k_b)}\Bigr)
\cong
\bigoplus_{(\lambda_b),(\kappa_b),(\rho_b),(\Lambda_b)}
\mathbb C^{\,c^{\lambda}_{(\lambda_b)}}\otimes
\Bigl[\bigotimes_b
\mathbb C^{\,g_{\lambda_b\kappa_b\rho_b}}\otimes
\mathbb S_{\kappa_b}(W_b)\otimes
\mathcal D_b^{\rho_b\Lambda_b}\Bigr]
\otimes
\mathcal M^{L,\epsilon}_{\boldsymbol\Lambda}.
\tag{5}
\]

The subscript `(k_b)` denotes the component of multidegree `(k_b)`, and the
blocks are maximal with distinct complete channels, so that repeated content
never appears in two blocks. `S_alpha(W)` vanishes when `alpha` has more rows
than `dim W`, and likewise for `V_l`. For carriers with a declared parity
`p_b` different from `(-1)^{l_b}` (hidden or message carriers) the parity rule
generalizes to `epsilon = prod_b p_b^{k_b}`.

`M^{L,epsilon}_Lambda` is the multiplicity space of `V_L` in
`(x)_b V_{Lambda_b}`, and it is nonzero only when
`epsilon = prod_b (-1)^{l_b k_b}`. The role Schur modules `S_{kappa_b}(W_b)`
are retained whole: every basis vector (semistandard tableau of shape
`kappa_b` in the role labels) is a separate model coordinate.

A basis label is therefore

```text
(N, content, lambda; LR copy;
 per block: lambda_b, kappa_b, rho_b, Kronecker copy, role tableau,
            Lambda_b, angular copy;
 outer copy; L, epsilon)  with magnetic component M = -L..L
```

**Which `lambda` is physical.** `lambda` is not a free modelling choice:

- commuting densities realize only `lambda = (N)`; then `lambda_b = (k_b)`,
  `c = 1`, and `g` forces `rho_b = kappa_b`;
- anticommuting carriers realize only `lambda = (1^N)`, which forces
  `rho_b = kappa_b'`;
- role-resolved commuting densities stay in the parent `(N)`; their role
  content appears only as the internal `kappa_b`;
- only slot-resolved ordered products realize a general `lambda`, in the
  sectors allowed by section 6. For `f^lambda > 1` the carrier is
  `S^lambda (x) V_L`, the label gains a tableau component, and in
  Young-orthogonal form the Gram structure is
  `D (x) I_{f^lambda} (x) I_{2L+1}`.

**Orthogonality and completeness.** `U(W_b)`, `U(V)`, `O(3)`, and `S_N` act
unitarily under the induced tensor inner product, so every direct sum in (2)-(5)
is orthogonal. Choosing orthonormal bases inside each LR, Kronecker, role,
angular, and outer multiplicity space gives a complete orthonormal basis of the
fixed-rank, fixed-content space. Within one carrier `V_L` the Gram matrix of
the emitted multiplets has the form

\[
G = D\otimes I_{2L+1},
\]

by Schur's lemma, so the orthogonal-output certificate
`C^dagger M C = D` of the scalar standard carries over unchanged and its size
does not grow with `L`. Two kinds of restriction apply to orthogonalization
and must not be confused:

- *mathematically forbidden*: mixing different `lambda`, `L`, `epsilon`,
  magnetic components `M`, or tableau components. Different `N` or content
  are different spaces altogether;
- *forbidden by convention*: mixing different
  `lambda_b, kappa_b, rho_b, Lambda_b`. These are labels of the larger group
  `prod_b U(U_b)`, which is not a symmetry of the target, and the fitted
  matrix of (1) mixes them freely. The sectors are already mutually
  orthogonal, so orthogonalization never needs to cross them, and crossing
  them would destroy label attribution and factorization.

**Complete multiplets.** For `L > 0` a basis element is the whole
`(2L+1)`-component multiplet. All components are emitted together from one
highest-weight (or any fixed reference) vector by the lowering operator, with
one declared real-tesseral or complex convention. The basis chosen in each
highest-weight multiplicity space must be real under the conjugation
structure; an arbitrary complex orthonormal basis of a multiplicity space
does not yield real multiplets. The injective-source and
source-orthogonality statements of the scalar standard are independent of `L`
and carry over verbatim.

## 4. Recovered special cases

| Setting | Result |
|---|---|
| `lambda=(N)`, `L=0`, `epsilon=+` | exactly `linear_lifted_cauchy_basis.md` |
| `lambda=(N)`, general `(L, epsilon)` | equivariant lifted-Cauchy features; only the outer space `M^{0,+}` is replaced by `M^{L,epsilon}` |
| `lambda=(N)`, one role (`dim W = 1`) | ordinary equivariant ACE features; only `kappa_b=(k_b)` survives |
| `lambda=(1^N)` | skew Cauchy, `rho_b = kappa_b'`; fermionic particle-slot carriers |
| general `lambda` | slot/role-resolved message or operator carriers |

## 5. Output-side theorem: arbitrary Young-symmetric targets

Many physical targets are tensors over a one-particle orbital space

\[
H=\bigoplus_{i} V_{l_i},
\]

a direct sum over **shell occurrences** `i` (one angular shell attached to one
site). `H` is taken to be a **real orthogonal carrier** with bra and ket
indices identified (real orbitals in an orthonormal, for example Lowdin,
basis). For complex orbitals the two-electron integrals have only fourfold
symmetry with conjugation, the relevant space is `Lambda^2 H (x) Lambda^2 H*`,
and the `S_n` statements below do not apply. A target with index symmetry
`mu` of `n` lives in `S_mu(H)`. Applying
(2) with one block per shell occurrence, then (4) and the outer coupling,

\[
\mathbb S_\mu(H)\cong
\bigoplus_{\text{content }(k_i)}\;
\bigoplus_{\alpha_i\vdash k_i}
\mathbb C^{\,c^\mu_{(\alpha_i)}}\otimes
\bigotimes_i \mathbb S_{\alpha_i}(V_{l_i}),
\qquad
\bigotimes_i\mathbb S_{\alpha_i}(V_{l_i})
\cong\bigoplus_{L}\mathcal Y^{L,\epsilon}\otimes V_L^{\epsilon},
\tag{6}
\]

with `epsilon = prod_i (-1)^{l_i k_i}`. A content `(k_i)` is a multiset of
shell occurrences, which the ERI workflow calls a **binding**; bindings
related by `Gamma` form a **binding type**. The output label is dual to the
feature label:

```text
(mu; binding content; alpha_i; LR copy; Lambda_i, angular copy; outer copy;
 L, epsilon)  with magnetic component M
```

The equivalent site-symmetric form groups identical shells,
`H = (+)_sigma W_sigma (x) V_{l_sigma}` with `W_sigma` the site space of shell
type `sigma`, and applies (3): `S_mu(W (x) V) = (+) g_{mu alpha beta}
S_alpha(W) (x) S_beta(V)`. The two forms are related by the weight-space
decomposition of `S_alpha(W_sigma)`; the exact relation is

\[
\sum_\alpha g_{\mu\alpha\beta}\,K_{\alpha,k}
=\sum_{(\alpha_i)} c^{\mu}_{(\alpha_i)}\,c^{\beta}_{(\alpha_i)},
\]

with `K` the Kostka numbers. The binding-attached form (6) is the one suited
to local, transferable models.

### Worked sectors

| Target | Space | Decomposition used |
|---|---|---|
| energy, scalar property | `V_0^+` | section 3 with `L=0` |
| dipole, polarizability, EFG | `V_1^-`, `Sym^2(V_1)=V_0+V_2`, `V_2^+` | section 3 with general `(L, epsilon)` |
| symmetric one-body operator (Fock, overlap, 1-RDM) | `Sym^2(H)`, `mu=(2)` | same occurrence `Sym^2(V_l)`, distinct occurrences `V_l (x) V_l'` |
| antisymmetric one-body operator | `Lambda^2(H)`, `mu=(1,1)` | same occurrence `Lambda^2(V_l)`, distinct occurrences `V_l (x) V_l'` |
| antisymmetrized same-spin ERI / Coulomb wedge operator | `S_(2,2)(H)` in `Sym^2(Lambda^2 H)` | table below |
| full spatial ERI `(ij|kl)` | `Sym^2(Sym^2 H) = S_(4)(H) + S_(2,2)(H)` | `mu=(4)` and `mu=(2,2)` |
| same-spin two-particle reduced density matrix | `Sym^2(Lambda^2 H) = S_(2,2)(H) + Lambda^4(H)` | `mu=(2,2)` and `mu=(1,1,1,1)` |
| density-fitted three-index tensor | `Sym^2(H) (x) H_aux` | product of two output functors |

"Same occurrence" and "distinct occurrences" replace the looser on-site and
off-site: two different shells on one atom form a distinct-occurrence block.
Blocks vanish for small shells; for an `s` shell `Lambda^2`, `S_(2,1)`, and
`S_(2,2)` are zero. The wedge operator is the spatial or same-spin object: it
does not determine `J` and `K` separately, and opposite-spin terms need the
`mu=(4)` part of the full-ERI row. Spin-orbital or spin-free reduced density
matrices require the `SU(2)` convention noted in section 1.

Products of Schur functors of different spaces (the last row) are handled by
applying (6) to each factor and coupling the resulting carriers; no new
identity is needed.

**The `mu = (2,2)` sector.** The wedge operator
`A_(ij),(kl) = (ik|jl) - (il|jk)` is antisymmetric within each pair, symmetric
under pair exchange, and satisfies the first Bianchi identity. The first two
properties place it in `Sym^2(Lambda^2 H) = S_(2,2)(H) + Lambda^4(H)`, and the
Bianchi identity removes the `Lambda^4` part. Its binding blocks are, by (6),

| Binding content | Blocks `c^{(2,2)}_{(alpha_i)}` |
|---|---|
| four distinct shell occurrences `(1,1,1,1)` | **2** copies of `V_{l_1} (x) V_{l_2} (x) V_{l_3} (x) V_{l_4}` |
| `(2,1,1)` | `Sym^2(V_{l_1}) (x) V_{l_2} (x) V_{l_3}` and `Lambda^2(V_{l_1}) (x) V_{l_2} (x) V_{l_3}` |
| `(2,2)` | `Sym^2 (x) Sym^2` and `Lambda^2 (x) Lambda^2`; no mixed term |
| `(3,1)` | `S_(2,1)(V_{l_1}) (x) V_{l_2}` |
| `(4)` | `S_(2,2)(V_{l_1})` |

The count of two for distinct occurrences is the number of standard tableaux
of shape `(2,2)`, equal to the two independent components of an algebraic
curvature tensor with four distinct indices.

In the site-symmetric form, (3) gives nine terms for `mu=(2,2)`:

```text
S_(2,2)(W (x) V) =  S_4 W (x) S_22 V   +  S_22 W (x) S_4 V   +  S_22 W (x) S_22 V
                  + S_22 W (x) S_1111 V + S_1111 W (x) S_22 V
                  + S_31 W (x) S_31 V   + S_31 W (x) S_211 V
                  + S_211 W (x) S_31 V  + S_211 W (x) S_211 V
```

This is neither the symmetric nor the skew Cauchy pairing. The pair space alone
does follow skew Cauchy, `Lambda^2(W (x) V) = Sym^2 W (x) Lambda^2 V +
Lambda^2 W (x) Sym^2 V`, and the nine terms are what remains of its symmetric
square after removing `Lambda^4`.

## 6. Binding-attached models and the physical meaning of `lambda_geom`

For a binding `b` with `n` index slots, let `Y_b` be its output block from (6)
and let `Gamma_b` be the stabilizer of `b` in `Gamma`. A slot-resolved feature
attaches one one-particle carrier to every slot,

\[
F_b = U^{(1)}\otimes\cdots\otimes U^{(n)}
\;\cong\;\bigoplus_{\lambda\vdash n}\mathbb S_\lambda(U)\otimes S^\lambda
\quad\text{(when the slot carriers are copies of one }U\text{)},
\]

and the label `lambda = lambda_geom` is the formal slot sector. The model block
for a binding type `tau` is

\[
C_\tau\in\operatorname{Hom}_{O(3)\times\Gamma_b}(F_b,\,Y_b),
\]

tied across all bindings of type `tau`, and block diagonal by (1).

Only `Gamma_b` must be respected. It acts on the slots through a homomorphic
image in `N(K_b)/K_b`, where `K_b` is the Young subgroup of coincident slots;
it is not in general a subgroup of `S_n` acting by slot permutation alone.
Formal `S_n` equivariance is **not** required. (If one imposed equivariance
under formal permutations of the feature factors with `S_n` acting on `Y_b`
through `S^mu` only, it would force `lambda_geom = mu`; that is not the
physical action, as the next paragraph shows.)

**The induced action on the output block.** A relabelling in `Gamma_b` does
two things to `Y_b`: it acts on the LR/Specht multiplicity space, **and** it
permutes the equivalent angular carriers `V_{l_i}` inside the block, with the
canonical-order signs of the index symmetry. Both must be included. This is
what the existing binding action does: it lowers a position permutation to
the induced one-particle AO-block relabelling and acts by `Lambda^2` of it on
both wedge axes.

Selection rule: a sector `lambda_geom` contributes at binding `b` if and only if

1. its physical source image is nonzero. For identical carriers on
   coincident slots with content `k`, this holds exactly when the Kostka
   number `K_{lambda,k}` is positive; and
2. `Hom_{O(3) x Gamma_b}( S_lambda(U) (x) Res S^lambda , Y_b )` is nonzero,
   with the full induced action on `Y_b`.

In the site-symmetric form (3), `lambda_geom` must match the **site** type
`alpha` in `g_{mu alpha beta} S_alpha(W) (x) S_beta(V)`, not `mu`. It is
forced to equal `mu` only when every permuted shell is `s`-type, because then
`beta = (n)` and `g_{mu alpha (n)} = delta_{alpha mu}`.

*Worked example (one-body, two equivalent `p` shells).* Take `mu=(2)`, shell
occurrences `a, b` related by `Gamma_b = S_2`. The block is
`F_ab in V_1 (x) V_1`, and the swap sends `F_ab` to `F_ba = F_ab^T`. Hence
`Y_b = Sym^2(V_1) (swap-even) + Lambda^2(V_1) (swap-odd)`. The swap-odd part,
an `L=1` even-parity pseudo-vector, is reached **only** by
`lambda_geom = (1,1)`. It is physically nonzero: a mirror plane exchanging two
equivalent atoms forces `F_yz = -F_zy` for their `p` functions.

Source sectors by content for rank four (Kostka rule): content `(1,1,1,1)`
admits all five sectors with multiplicities `1,3,2,3,1`; `(2,1,1)` admits all
but `(1,1,1,1)`; `(2,2)` admits `(4),(3,1),(2,2)`; `(3,1)` admits
`(4),(3,1)`; `(4)` admits only `(4)`. For four equivalent distinct `p` shells
under the full `S_4`, the `mu=(2,2)` block has output multiplicity dimensions
`6/18/21/18/6` by `lambda_geom` (total `162 = 2 x 81`), so all five sectors
are reachable; with `s` shells only `(2,2)` survives.

Two-centre and pair carriers additionally carry orientation signs under
`Gamma_b` (a bond vector picks up `(-1)^l` under reversal), so their action
is the induced slot permutation together with those signs.

Two limits make the meaning concrete.

- **All slots coincide** (same site, same one-particle carrier). The slot
  features commute, `F_b` collapses to `Sym^n(U)`, and only
  `lambda_geom = (n)` survives. This is the rule that ordinary commutative
  density products carry only the trivial sector.
- **All slots distinct, `Gamma_b` trivial.** Every `lambda_geom` with a nonzero
  source image contributes, and together they span the full slot-resolved
  space `U^{(x)n}`. The `(n)` sector alone is the strict subspace of
  slot-symmetrized features, which no longer record which index slot saw
  which environment.

For a composite source built from two pair carriers of type `(1,1)`, the LR
product rule `S_a(U) (x) S_b(U) = (+)_lambda c^lambda_{ab} S_lambda(U)` gives
`(1,1) . (1,1) = (2,2) + (2,1,1) + (1,1,1,1)`, so `(4)` and `(3,1)` are absent
by an exact LR result, not by a missing software path.

**Consequence for the sector-utility question.** In a linear model the span of
a mixed-sector feature set contains the span of the symmetric-sector set at
the same rank, content, and one-particle source. Containment is strict unless
all slots share one carrier, `dim U = 1`, `lambda` has more rows than
`dim U`, no `O(3) x Gamma_b` route exists for any nontrivial sector, the
physical source image is degenerate over configurations, or, on a finite
training set, the `(n)`-sector design already has full row rank.

There are two distinct mechanisms, and only the first is a capacity effect:

- *more features*: for a binding with trivial `Gamma_b`, the nontrivial
  sectors add slot-resolved coordinates. Their benefit is the decrease in the
  in-sample, unregularized least-squares projection residual. With ridge
  only the penalized objective is monotone, and nothing is implied about
  test error;
- *symmetry-forced necessity*: for a binding with nontrivial `Gamma_b` and
  equivalent shells of `l > 0`, output components that are odd under the
  relabelling are **unreachable** from the `(n)` sector at any width. In the
  worked example the target `F_ab = z_a z_b^T` with independent isotropic
  `z_a, z_b` leaves a relative residual of exactly `1/3` when `(1,1)` is
  removed and `0` when it is kept. This is a representational gap, not an
  optimization or capacity effect, and it grows with the number of
  symmetry-equivalent `l > 0` shells.

Neither statement contains optimizer, seed, or training-budget dependence.

## 7. Linear fitting

Let the output synthesis basis of (6) be orthonormal. The Frobenius loss then
separates over output coordinates, and by (1) over blocks:

\[
\|T-\hat T\|_F^2
=\sum_{\tau}\sum_{(L,\epsilon)}
\bigl\|\,y_{\tau}^{L\epsilon}-C_{\tau}^{L\epsilon}B_{\tau}^{L\epsilon}\bigr\|^2 .
\]

Each block `(tau, L, epsilon, gamma_b)` is an independent multi-output ridge
problem. Its unknown `C` lies in the `Gamma_b`-commutant `Hom_{Gamma_b}`; it
is a free (output multiplicity) x (feature multiplicity) matrix only when
`Gamma_b` is trivial. An unconstrained fit on canonically oriented rows makes
predictions depend on atom ordering. Either impose the commutant through the
compiler's intertwiner basis, or Reynolds-symmetrize the normal equations
with a ridge that commutes with the action. Every
geometry, every binding of type `tau`, and every magnetic component
contributes one row per output copy with the same `C`, so the normal equations
have the size of the feature multiplicity, not of the operator. They can be
accumulated in a stream, and the blocks can be fitted in parallel.

Conditions and caveats:

- the ridge must itself be block separable, and sample weights must be
  constant across `M` and across `Gamma`-related bindings. Feature scaling
  must use one scale per multiplet and no per-component centering for
  `L > 0`, or covariance is broken;
- "orthonormal" refers to the declared Frobenius inner product on the full
  tensor or wedge matrix; storing only canonical components requires
  orbit-size weights, the analogue of the inverse-orbit-size metric of the
  scalar standard;
- decoupling requires an orthonormal output basis and a loss that is a sum of
  squared output coordinates. A transformation that mixes blocks before the
  loss, such as projection into a geometry-dependent active-orbital frame,
  couples the blocks and requires either a coupled solve or fitting the parent
  operator first;
- identity ridge is not invariant under nonunitary basis changes; fit in
  normalized orthogonal coordinates as the scalar standard requires;
- a downstream contraction (Fock matrix, fixed-density energy) is linear in
  `T`, so a linear model gives it in closed form, and it may be added to the
  quadratic loss without losing linearity, at the cost of block decoupling.

## 8. Multiplet-valued evaluation and derivatives

The forward and adjoint structure is the scalar one with a vector-valued final
contraction. With density coordinates `A` and an exact coefficient tensor
`K^{(label) L}_{M; m_1...m_N}`,

\[
B^{(\text{label})L}_{M}(A)=\sum_{m_1\ldots m_N}
K^{(\text{label})L}_{M;\,m_1\ldots m_N}\prod_{j=1}^{N}A_{c_j m_j},
\qquad
\bar A_{c_j m_j}=\sum_{M}\bar B_M\,
\frac{\partial B_M}{\partial A_{c_j m_j}} .
\]

All `2L+1` components share every intermediate density product; only the final
coupling coefficients depend on `M`. A single kernel family therefore serves
every target: the scalar energy and force path is `L = 0` with output
cotangent `1`. The adjoint is the algebraic covector pullback from the chain
rule; a metric (Riesz) adjoint must not be substituted. The density cotangent
is indexed by `(c, m)`, not by slot: coincident slots, or slots bound to the
same site feature, accumulate into the same entry. For complex carriers the
analysis or synthesis orientation and the conjugation convention of the
pullback must be declared, because automatic-differentiation conventions
differ.

### Real-form phase convention for pseudo-tensors

With Condon-Shortley harmonics and real Clebsch-Gordan coefficients, a coupled
multiplet built from real densities satisfies

\[
\overline{B_{LM}}=(-1)^{M}\,(-1)^{\sum_b k_b l_b-L}\,B_{L,-M}.
\]

When `sum_b k_b l_b - L` is even (natural parity, `epsilon = (-1)^L`) this is
the standard reality condition and the real-tesseral transform is real. When
it is odd (pseudo-tensor sectors, `epsilon != (-1)^L`) the multiplet is
anti-real and its tesseral transform is purely imaginary. The familiar instance
is `[a (x) b]^1 = (i / sqrt 2) (a x b)`.

**Adopted convention.** Emit

\[
B^{\mathrm{real}}_{L}=U_L\,\bigl[(-i)^{\sigma}B_L\bigr],
\qquad
\sigma=\Bigl(\sum_b k_b l_b-L\Bigr)\bmod 2 ,
\]

which is the rule already used for every binary real coupling in
`ye3t/paired_cg.py` (`_real_cg_entries_cpu` multiplies by `-1j` when
`L1 + L2 - L_out` is odd) and therefore by the message-passing stack and the
`(2,2)` algebraic-curvature synthesis. Properties:

- natural-parity outputs carry no phase, so the scalar `L = 0` artifact is
  byte-identical to the existing compiler;
- covariant Cauchy multiplets and the existing real coupling tables share one
  convention, so no conversion layer is needed at the ERI output chart;
- a multi-node coupling tree evaluated with the per-node rule differs from the
  single output phase by the real sign `(-1)^{floor(o/2)}`, with `o` the number
  of odd nodes. The emitted sign is a basis gauge; it is recorded in the
  artifact and bound by its hash.

Required exact certificate: every emitted real coefficient has zero imaginary
part after the phase, and the `L = 1` pseudo-vector from `1 (x) 1` equals the
`_real_cg_entries_cpu(1, 1, 1)` contraction of the same inputs.

**Declared backup.** The Fano-Racah convention `Y_lm -> i^l Y_lm`, under which
every tensor satisfies `conj(T_LM) = (-1)^{L-M} T_{L,-M}` and the property is
closed under coupling with real Clebsch-Gordan coefficients for all `L` and
parities, so no parity case distinction is needed. It differs from the adopted
rule by a real sign per sector and would change the sign of existing
natural-parity quantities when `sum l - L = 2 mod 4`; it is therefore not the
default, because it would break byte identity of the hash-bound scalar
artifacts. REFERENCE_TODO: verify the primary citation for the `i^l`
convention before it is quoted in publication-facing text.

Covariance to be certified on complete multiplets, for `R` in `SO(3)` and the
inversion `P`:

\[
B^{L\epsilon}(RX)=D^{L}(R)\,B^{L\epsilon}(X),
\qquad
B^{L\epsilon}(PX)=\epsilon\,B^{L\epsilon}(X),
\]

together with `Gamma` covariance: relabelling identical atoms permutes
bindings within a type and acts on slot-resolved features by the induced slot
permutation, with orientation signs for pair carriers, and tableau-axis
covariance when `f^lambda > 1`. `epsilon` is the total inversion eigenvalue
and equals the signed parity `p` of the `O(3)` amendments MP-EQ-O3-01..07 in
`message_passing_mathematical_standard.md` (`D^{(L,p)}(Q) = p^k D^L(R)`),
because densities transform as
`A_nlm(PX) = (-1)^l A_nlm(X)`. The improper test must be written as
`Q = R(-I)` with `B(QX) = epsilon D^L(R) B(X)`: evaluating a polynomial real
`D` at an improper `Q` silently assumes natural parity and is wrong for
pseudo-tensors.

## 9. Exact checks

Dimension identities that hold exactly and should become tests. All were
verified numerically.

1. `g_{(4) alpha beta} = delta_{alpha beta}` and
   `g_{(1^4) alpha beta} = delta_{alpha beta'}` from the `S_4` character
   table.
2. The nine nonzero `g_{(2,2) alpha beta}` listed in section 5, each equal to
   one.
3. `dim S_mu(C^{wv}) = sum g_{mu alpha beta} dim S_alpha(C^w) dim S_beta(C^v)`
   for every `mu` of 4 and `(w,v)` in `{(2,2),(2,3),(3,3),(2,5),(4,3)}`.
4. `dim Sym^2(Lambda^2 C^d) = dim S_(2,2)(C^d) + dim Lambda^4(C^d)` with
   `dim S_(2,2)(C^d) = d^2(d^2-1)/12`.
5. The binding-block table of section 5: for `m` shells of dimension `v`,
   the block dimensions sum to `dim S_(2,2)(C^{mv})`.
6. `S_(2,2)(V_1) = V_0^+ + V_2^+` (dimension 6), the three-dimensional
   curvature tensor.
7. At `lambda=(N)`, `L=0`, `epsilon=+`, the general count reduces to the
   existing scalar compiler count; the algebraic reduction is exact.
8. `sum_alpha g_{mu alpha beta} K_{alpha,k} = sum c^mu c^beta` for all `mu`,
   `beta`, and contents at `n=4`.
9. The Kostka source-sector table and the `6/18/21/18/6` and `0/0/1/0/0`
   stabilizer dimensions of section 6.
10. `SO(3)` content of `S_lambda(V_l)`, for example
    `S_(3,1)(V_2) = {1:2, 2:2, 3:3, 4:2, 5:2, 6:1, 7:1}`, cross-checked against
    an independent tabulation of the available `L` values.
11. The assembled dimension identity of (5) for every `lambda` of 4 in mixed
    role and angular configurations.
12. Multiplicity-label coverage: every Kronecker coefficient at `n=4` is at
    most one, so rank four never exercises the Kronecker copy label. The
    first `g=2` is at `n=5`, `((3,1,1),(3,1,1),(3,2))`; the first two-block
    LR multiplicity of two is at `n=6`, `c^{(3,2,1)}_{(2,1),(2,1)}`; a
    three-block LR multiplicity of two already occurs at `n=4`.
13. The worked one-body fit: residual `1/3` without `(1,1)` and `0` with it,
    and a relabel-covariance test in which a free `C` that does not commute
    with `Gamma_b` must fail.
14. Finite-difference VJP on a content-`(2,1,1)` binding with a shared site
    carrier, to catch a missing accumulation factor.

Additional required certificates for an implementation: complete-multiplet
proper and improper `O(3)` covariance, `Gamma` covariance, Gram structure
`D (x) I_{2L+1}`, canonical versus factored forward and VJP agreement,
finite-difference VJP with random output cotangents, and exact reduction to
the scalar path.

## 10. Open points for review

1. **Real-tesseral convention for emitted multiplets.** The statement is
   convention independent, but the emitted coefficient tensors are not. The
   implementation must declare one convention and reuse the existing
   real/complex equivalence tests.
2. **Pseudo-tensor conventions.** The real-form phase is resolved in section 8
   (adopted: the `paired_cg.py` odd-coupling `-i` rule; backup: Fano-Racah).
   Still to confirm: that `epsilon`, defined here by `B(PX) = epsilon B(X)`,
   matches the signed-parity convention of the `O(3)` amendments
   MP-EQ-O3-01..07 in `message_passing_mathematical_standard.md`.
3. **Slot carriers that differ between slots.** Section 6 writes the Schur-Weyl
   form for identical slot carriers. When slot carriers differ (for example
   shell-conditioned sources), `lambda_geom` remains a valid organizing label
   through the subgroup preserving the carrier assignment, defined on the
   orbit closure; the exact statement should be reviewed against the current
   source-image construction, and the correspondence between a content
   binding and the ordered occurrence patterns of the ERI code was not
   verified in full.
4. **`Gamma`-irrep versus orbit form.** Section 6 ties parameters over binding
   orbits rather than decomposing into `Gamma`-irreps. The two are equivalent
   for binding-diagonal maps only: `Hom_Gamma(Ind F_b, Ind Y_b)` also has
   Mackey cross-binding terms, which the orbit form drops by locality rather
   than by symmetry. The orbit form is local and transferable, and is the one
   proposed for implementation.
5. **Readout mixing versus the message-passing contract.** Section 6 lets a
   binding readout combine tableau components and different `lambda_geom`
   under `O(3) x Gamma_b`. The boundary is: formal `S_N` equivariance on
   hidden carriers, and an exact compiler-owned `Hom_{O(3) x Gamma_b}` basis
   at a final physical readout.
6. **Scope not yet covered.** Dual and conjugate carriers for `H (x) H*` with
   complex bases (a walled-Brauer rather than `S_n` theory); Hermiticity as
   `Sym^2(H_R) + i Lambda^2(H_R)`; time reversal as an extra `Z_2` selection
   rule; quaternionic spinor irreps; the non-orthogonal AO metric and
   frame-dependent losses; periodic systems; and active-orbital frames, which
   are not functions into a fixed representation space and need either a
   covariant gauge rule or gauge-invariant targets.

## References

REFERENCE_TODO: before this becomes publication-facing, verify and record
exact section, theorem, and equation numbers from the paper bibliography.
Candidate standard sources, not yet checked against
the text: Schur-Weyl duality and Schur functors of direct sums and tensor
products (Fulton and Harris, *Representation Theory: A First Course*;
Macdonald, *Symmetric Functions and Hall Polynomials*, for the Cauchy
identities and the internal product); `(GL_n, GL_m)` and skew duality (Howe,
"Perspectives on invariant theory"). Do not cite edition, page, or DOI
metadata until checked. No external source code was consulted; every
statement above is derived from standard representation theory and verified by
the exact checks of section 9.
