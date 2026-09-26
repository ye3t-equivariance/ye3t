---
title: YE3T message-passing mathematical standards
role: mathematical-standards
status: PROVISIONAL_PROJECT_STANDARD
scope: fixed-content S_N x SO(3) carriers, exact coupling, density/motif realizations, and equivariant message passing
---

# YE3T message-passing mathematics

## 0. Purpose, authority, and immediate coding contract

This file is the compact mathematical contract for implementing or reviewing YE3T representation and message-passing machinery. It consolidates the current carrier decomposition, basis maps, multiplicity spaces, density and motif realizations, binary couplers, arbitrary coupling trees, learned maps, and validation identities.

Status tags:

- **`[STD]`**: standard representation theory, angular-momentum theory, or graph/message-passing mathematics.
- **`[DERIVED]`**: a direct application of standard results to the stated YE3T carrier.
- **`[PROPOSED]`**: a YE3T architecture or realization interface requiring project proof and validation.
- **`[VERIFY]`**: plausible or draft-supported, but not yet promoted to a proved project theorem.

Project-source precedence:

1. latest explicit project decision or correction;
2. this file for message-passing mathematics;
3. the manuscript, which is evidence rather than automatic authority.

### 0.1 Non-negotiable implementation rules

1. **A representation type is at least**

   ```text
   theta = (N, lambda, L, convention_id)
   ```

   where `lambda` partitions `N`. A future parity extension must add a parity field explicitly.

2. **A learned linear equivariant map does not mix representation types.** For
   \(U_\theta=[\lambda]\otimes V_L\),

   \[
   T_{\mathrm{lin}}
   =
   \bigoplus_{\theta}
   \left(I_{U_\theta}\otimes W_\theta\right).
   \tag{MP-EQ-00}
   \]

   \(W_\theta\) acts only on axes carrying the trivial group action: exact multiplicity copies, source/content channels, radial/chemical channels, coupling-path coordinates after a basis is fixed, and ordinary learned channels. It never acts arbitrarily on Young components \(t\), magnetic components \(M\), or a different \((N,\lambda,L)\) block.

3. **Different angular momenta do interact, but this is not linear “mixing of \(L\).”** They interact through tensor products and exact Clebsch--Gordan projection:

   \[
   V_{L_A}\otimes V_{L_B}
   \longrightarrow
   V_L,
   \qquad
   |L_A-L_B|\le L\le L_A+L_B.
   \tag{MP-EQ-00a}
   \]

   The permutation labels are coupled at the same operation by the appropriate finite-group intertwiner. The output \(L\) and \(\lambda\) may differ from both child labels.

4. **If coupled kets are the columns of \(\mathsf C\), feature analysis uses \(\mathsf C^\dagger\), not \(\mathsf C\).**

5. **Raw coordinates are indexed by the raw product label** \(q=(\boldsymbol\mu,\mathbf m)\). Do not append an independent subgroup-tableau or coupling-path index to \(q\); those labels belong to adapted/coupled bases.

6. **Orbit closure is not symmetrization.** It produces an \(S_N\)-stable carrier. Irrep selection occurs only through a projector or an intertwiner such as \(\mathsf C^\dagger\).

7. **Density products and explicit motif embeddings are different source spaces.** Density products use unrestricted neighbor-index sums and allow repeated neighbors. Injective motif embeddings do not.

8. **Do not use runtime SVD, QR, eigendecomposition, Gram--Schmidt, or tolerance-based rank discovery to define the representation.** Exact or precomputed intertwiners are primary.

9. **Do not use Littlewood--Richardson induction for every hidden tensor product.** Rank-additive factor concatenation uses induction; a product of two same-rank \(S_N\) carriers under one diagonal \(S_N\) action uses Kronecker coefficients. Section 6 distinguishes these cases.

10. **No external-model containment or implementation comparison is part of this standards file.**

### 0.2 Scope boundary

The production equations use

\[
S_N\times O(3).
\]

Translation invariance is obtained from relative positions, giving the spatial
behavior needed for \(E(3)\)-equivariant atomistic models without introducing a
separate translation carrier. `parity=None` denotes only an explicit
`SO3_legacy` carrier. All spaces are finite-dimensional complex inner-product
spaces unless a fixed real-basis conversion is declared and serialized.

### 0.3 O(3) carrier and product convention

#### MP-EQ-O3-01 -- irreducible carrier `[STD]`

Every \(Q\in O(3)\) has the unique form \(Q=R(-I)^k\), with
\(R\in SO(3)\) and \(k\in\{0,1\}\). The irrep labelled by angular momentum
\(L\) and parity \(p\in\{-1,+1\}\) acts as

\[
D^{(L,p)}(Q)=p^kD^{(L)}(R).
\tag{MP-EQ-O3-01}
\]

Parity is independent of \(L\). In particular, \(L=0,p=-1\) is a
pseudoscalar rather than an invariant scalar.

#### MP-EQ-O3-02 -- polar spherical-harmonic sources `[STD]`

For polar relative-position vectors and scalar radial, chemical, and role
channels,

\[
Y_l^m(-\hat{\mathbf r})=(-1)^lY_l^m(\hat{\mathbf r}),
\qquad
p_{\mathrm{source}}=(-1)^{\sum_f l_f}.
\tag{MP-EQ-O3-02}
\]

Any axial, pseudoscalar, spinful, or otherwise intrinsic-parity source must
declare that source convention separately; it must not reuse the polar-source
default silently.

#### MP-EQ-O3-03 -- product and coupling parity `[STD]`

For a binary tensor product,

\[
(L_A,p_A)\otimes(L_B,p_B)
\longrightarrow (L,p_Ap_B),
\qquad |L_A-L_B|\le L\le L_A+L_B.
\tag{MP-EQ-O3-03}
\]

The ordinary \(SO(3)\) Clebsch--Gordan coefficients therefore remain the
numeric angular intertwiner. Parity changes compile-time path eligibility,
carrier identity, serialization, and cache identity; it must not add a
data-dependent branch to numeric contraction kernels. A \(k\)-fold symmetric,
exterior, or mixed Young block product has spatial parity \(p^k\), independently
of its permutation image.

#### MP-EQ-O3-04 -- joint carrier and learned maps `[DERIVED]`

The joint carrier is

\[
U_\theta=[\lambda]\otimes V_{L,p},
\qquad \theta=(N,\lambda,L,p),
\tag{MP-EQ-O3-04}
\]

and learned maps may mix only multiplicity/channel copies at fixed
\((N,\lambda,L,p,\text{convention})\).

#### MP-EQ-O3-05 -- atomistic energy readout `[DERIVED]`

A scalar potential energy readout accepts only

\[
\lambda=(N),\qquad L=0,\qquad p=+1.
\tag{MP-EQ-O3-05}
\]

Odd pseudoscalar hidden carriers are valid, but cannot enter the energy
linearly. Forces obtained as \(-\nabla_{\mathbf r}E\) transform as polar vectors
under proper and improper orthogonal transformations.

### 0.4 O(3) validation and runtime contract

#### MP-EQ-O3-06 -- improper equivariance `[VERIFY]`

For every retained optimized path and a representative \(Q\in O(3)\), tests
must verify carrier transformation, inversion, product parity, invariant energy,
polar-force covariance, and the corresponding VJP/HVP or double-backward
identity. Proper-rotation tests alone are insufficient.

#### MP-EQ-O3-07 -- migration and performance `[PROPOSED]`

New exact plans serialize signed parity and an O(3) convention ID. Existing
schema-v1 plans may load only through an explicit `SO3_legacy` migration that
preserves their original hash in provenance and never guesses parity. O(3)
selection must happen before runtime so fused CG, graph-scatter, symmetric-power,
reverse, and second-order kernels remain parity-branch-free. Acceptance requires
paired benchmarks against the equivalent legacy numeric schedule.

### 0.5 References for the O(3) convention

- Batzner et al., *E(3)-equivariant graph neural networks for data-efficient and
  accurate interatomic potentials*, Nature Communications 13, 2453 (2022),
  especially the \((l,p)\) irreps and tensor-product parity rule in Eq. 7:
  https://doi.org/10.1038/s41467-022-29939-5
- Thomas et al., *Tensor Field Networks: Rotation- and Translation-Equivariant
  Neural Networks for 3D Point Clouds* (2018), Section 3.3 and supplementary
  Clebsch--Gordan intertwining identities: https://arxiv.org/abs/1802.08219
- Geiger and Smidt, *e3nn: Euclidean Neural Networks* (2022), O(3) irrep and
  parity conventions: https://arxiv.org/abs/2207.09453
- e3nn project documentation, *Irreducible representations of O(3)*, for the
  independently labelled \((l,p)\) convention and spherical-harmonic parity:
  https://docs.e3nn.org/en/stable/api/o3/o3_irreps.html
- A. R. Edmonds, *Angular Momentum in Quantum Mechanics*, Princeton University
  Press, Chapters 3--5, for rotations, angular-momentum coupling, and spherical
  tensors: https://doi.org/10.1515/9781400884186

YE3T implements these representation-theory rules independently. No external
implementation source is copied or mechanically translated.

---

## 1. Conventions and notation

### 1.1 Basis conventions

- Orthonormal complex spherical harmonics \(Y_l^m\), with the Condon--Shortley phase.
- Active rotations and a fixed Wigner-\(D\) convention.
- Condon--Shortley/Edmonds Clebsch--Gordan coefficients.
- Young--Yamanouchi labels in a Young-orthogonal basis for \(S_N\) carriers.
- A real spherical-harmonic implementation must use a stored, fixed unitary basis change.
- Multiplicity-space bases, copy ordering, phases, tree convention, and serialization order are not fixed by symmetry and must be stored explicitly.

### 1.2 Core symbols

| Symbol | Meaning |
|---|---|
| \(i\) | central/root atom |
| \(j,j_1,\ldots,j_N\) | physical-neighbor indices |
| \(f=1,\ldots,N\) | tensor-product factor index; not a neighbor label |
| \(\eta\) | collected non-angular one-factor channel label |
| \(\nu=(\eta,l)\) | complete one-factor channel |
| \(\mathcal V_\nu\cong V_l\) | channel-labelled copy of the \(SO(3)\) irrep \(V_l\) |
| \(\boldsymbol\nu\) | fixed channel content/multiset |
| \(\boldsymbol\nu^\circ\) | provisional canonical ordered representative of \(\boldsymbol\nu\) |
| \(\boldsymbol\mu\) | an ordering in \(\operatorname{Orb}_{S_N}(\boldsymbol\nu^\circ)\) |
| \(B\), \(k_b\) | number and sizes of repeated-content blocks, \(N=\sum_bk_b\) |
| \(G_{\boldsymbol\nu}\) | stabilizer/Young subgroup \(\prod_bS_{k_b}\) |
| \(\kappa_b\), \(\boldsymbol\kappa\) | block partition and tuple of block partitions |
| \(\lambda\vdash N\) | parent \(S_N\) partition |
| \([\kappa_b]\), \([\lambda]\) | block and parent permutation carriers |
| \(\Lambda_b\), \(\boldsymbol\Lambda\) | block angular momentum and tuple |
| \(L,M\) | parent angular momentum and magnetic component |
| \(t\) | Young-basis component of \([\lambda]\) |
| \(a\) | basis index in a full joint multiplicity space |
| \(q=(\boldsymbol\mu,\mathbf m)\) | raw product-basis label |
| \(\theta=(N,\lambda,L)\) | representation type, excluding convention metadata |
| \(U_\theta=[\lambda]\otimes V_L\) | irreducible carrier of \(S_N\times SO(3)\) |
| \(\mathcal A_\theta\) | multiplicity/ordinary-channel space carrying the trivial group action |
| \(\mathcal T\) | coupling tree/factorization |
| \(p\) | typed binary coupling path/table entry |
| \(\mathsf C\) | raw-to-coupled synthesis matrix |
| \(W\) | learned map on \(\mathcal A_\theta\), never on \(t,M\) |
| \(s\) | optional role/filter label for a lifted density |
| \(b=(Z_i,Z_j)\) | declared directed or unordered chemical pair type |
| \(K_b\) | number of available lifted source roles for pair type \(b\) |
| \(\chi\) | source realization: role tuple, motif embedding, or another typed source |
| \(Q_i\) | scalar per-atom charge or charge-like variable |

The role label \(s\) has two legitimate uses:

- if it is an intrinsic one-factor radial/channel label, it may be included in \(\eta\);
- if it is an architecture-level factor role, branch, or support label, keep it separate.

The mathematical requirement is not a particular symbol placement. It is that ordered factor-role/channel information is retained through orbit closure and coupling rather than collapsed to an unordered commutative product.

---

## 2. Orbit-closed fixed-content carrier

### MP-EQ-01 — content, stabilizer, and orbit carrier `[DERIVED]`

Let

\[
\boldsymbol\nu^\circ
=
(\nu_1^{k_1},\ldots,\nu_B^{k_B}),
\qquad
N=\sum_{b=1}^{B}k_b,
\qquad
G_{\boldsymbol\nu}
=
\operatorname{Stab}_{S_N}(\boldsymbol\nu^\circ)
=
\prod_{b=1}^{B}S_{k_b}.
\]

For an ordering \(\boldsymbol\mu=(\mu_1,\ldots,\mu_N)\), define

\[
\mathcal H_{\boldsymbol\mu}
=
\bigotimes_{f=1}^{N}\mathcal V_{\mu_f}.
\]

Then

\[
\boxed{
\widehat{\mathcal H}_{\boldsymbol\nu}
=
\bigoplus_{\boldsymbol\mu\in
\operatorname{Orb}_{S_N}(\boldsymbol\nu^\circ)}
\mathcal H_{\boldsymbol\mu}
\cong
\mathcal H_{\boldsymbol\nu^\circ}
\uparrow_{G_{\boldsymbol\nu}}^{S_N}.
}
\tag{MP-EQ-01}
\]

The orbit contains distinct orderings, not one duplicate for every stabilizer element.

### MP-EQ-02 — commuting factor-permutation and rotation actions `[DERIVED]`

Use

\[
(\sigma\!\cdot\!\boldsymbol\mu)_f
=
\mu_{\sigma^{-1}(f)},
\qquad
(\sigma\!\cdot\!\mathbf m)_f
=
m_{\sigma^{-1}(f)}.
\]

On the raw basis,

\[
P(\sigma)|\boldsymbol\mu,\mathbf m\rangle
=
|\sigma\!\cdot\!\boldsymbol\mu,
  \sigma\!\cdot\!\mathbf m\rangle,
\]

\[
D(R)|\boldsymbol\mu,\mathbf m\rangle
=
\sum_{\mathbf m'}
\left[
\prod_{f=1}^{N}
D^{(l(\mu_f))}_{m_f'm_f}(R)
\right]
|\boldsymbol\mu,\mathbf m'\rangle.
\]

Therefore

\[
\boxed{
[P(\sigma),D(R)]=0,
\qquad
(\sigma,R)\in S_N\times SO(3).
}
\tag{MP-EQ-02}
\]

This factor-position action is distinct from reordering a physical neighbor list.

### MP-EQ-03 — raw basis and raw coordinate vector `[STD/DERIVED]`

The orthonormal raw product basis satisfies

\[
\boxed{
\langle\boldsymbol\mu,\mathbf m
\mid
\boldsymbol\mu',\mathbf m'\rangle
=
\delta_{\boldsymbol\mu,\boldsymbol\mu'}
\prod_{f=1}^{N}\delta_{m_fm_f'}.
}
\tag{MP-EQ-03}
\]

Define the raw index set

\[
\mathcal Q_{\boldsymbol\nu}
=
\left\{
q=(\boldsymbol\mu,\mathbf m):
\boldsymbol\mu\in\operatorname{Orb}_{S_N}(\boldsymbol\nu^\circ)
\right\}.
\]

A physical realization supplies scalar product-basis functions/coordinates \(Z_{i,\chi;q}\). Runtime may store only

\[
\boxed{
\mathbf Z_{i,\chi}^{\boldsymbol\nu}
:=
\left(Z_{i,\chi;q}\right)_{q\in\mathcal Q_{\boldsymbol\nu}}
\in
\mathbb C^{\mathcal Q_{\boldsymbol\nu}}
\cong
\widehat{\mathcal H}_{\boldsymbol\nu}.
}
\tag{MP-EQ-03a}
\]

Here \(Z_{i,\chi;q}\) is an evaluated product-basis function/coordinate; \(|q\rangle\) is an abstract carrier basis vector. They are not the same object. The coordinate action is

\[
\left[P(\sigma)\mathbf Z\right]_q
=
Z_{\sigma^{-1}\!\cdot q}.
\]

Enumerating every orbit coordinate does not make \(\mathbf Z\) invariant. Invariance requires \(Z_q=Z_{\sigma^{-1}\cdot q}\) for every \(q,\sigma\).

---

## 3. Exact decomposition and multiplicity spaces

### MP-EQ-04 — repeated-block decomposition `[STD/DERIVED]`

For repeated block \(b\),

\[
\boxed{
\mathcal V_{\nu_b}^{\otimes k_b}
\cong
\bigoplus_{\kappa_b\vdash k_b}
\bigoplus_{\Lambda_b}
[\kappa_b]\otimes V_{\Lambda_b}
\otimes
\mathcal D_b^{\kappa_b\Lambda_b}.
}
\tag{MP-EQ-04}
\]

The block multiplicity space and count are

\[
\mathcal D_b^{\kappa_b\Lambda_b}
:=
\operatorname{Hom}_{S_{k_b}\times SO(3)}
\!\left(
[\kappa_b]\otimes V_{\Lambda_b},
\mathcal V_{\nu_b}^{\otimes k_b}
\right),
\qquad
d_b^{\kappa_b\Lambda_b}
:=
\dim\mathcal D_b^{\kappa_b\Lambda_b}.
\]

Define

\[
[\boldsymbol\kappa]
:=
\boxtimes_{b=1}^{B}[\kappa_b],
\qquad
\mathcal D_{\boldsymbol\kappa\boldsymbol\Lambda}
:=
\bigotimes_{b=1}^{B}
\mathcal D_b^{\kappa_b\Lambda_b},
\qquad
d_{\boldsymbol\kappa\boldsymbol\Lambda}
=
\prod_bd_b^{\kappa_b\Lambda_b}.
\]

The symbol \(\boxtimes\) is the external tensor product for the product group \(G_{\boldsymbol\nu}=\prod_bS_{k_b}\).

### MP-EQ-05 — block-to-parent permutation induction `[STD]`

\[
\boxed{
[\boldsymbol\kappa]
\uparrow_{G_{\boldsymbol\nu}}^{S_N}
\cong
\bigoplus_{\lambda\vdash N}
[\lambda]\otimes
\mathbb C^{c_{\boldsymbol\kappa}^{\lambda}}.
}
\tag{MP-EQ-05}
\]

By Frobenius reciprocity,

\[
c_{\boldsymbol\kappa}^{\lambda}
=
\dim\operatorname{Hom}_{S_N}
\!\left(
[\boldsymbol\kappa]\uparrow_{G_{\boldsymbol\nu}}^{S_N},
[\lambda]
\right)
=
\dim\operatorname{Hom}_{G_{\boldsymbol\nu}}
\!\left(
[\boldsymbol\kappa],
[\lambda]\downarrow_{G_{\boldsymbol\nu}}^{S_N}
\right).
\]

### MP-EQ-06 — block-to-parent angular restriction `[STD]`

\[
\boxed{
\left(
\bigotimes_{b=1}^{B}V_{\Lambda_b}
\right)
\downarrow_{\Delta SO(3)}^{\prod_bSO(3)_b}
\cong
\bigoplus_L
V_L\otimes\mathcal M_{\boldsymbol\Lambda}^{L}.
}
\tag{MP-EQ-06}
\]

\[
M_{\boldsymbol\Lambda}^{L}
:=
\dim\mathcal M_{\boldsymbol\Lambda}^{L}
=
\dim\operatorname{Hom}_{SO(3)}
\!\left(
V_L,
\bigotimes_bV_{\Lambda_b}
\right).
\]

Different complete coupling trees give different path bases but the same total \(M_{\boldsymbol\Lambda}^{L}\).

### MP-EQ-07 — full joint decomposition `[DERIVED; theorem status VERIFY]`

\[
\boxed{
\widehat{\mathcal H}_{\boldsymbol\nu}
\cong
\bigoplus_{\lambda\vdash N}
\bigoplus_L
\left([\lambda]\otimes V_L\right)
\otimes
\mathscr M_{\boldsymbol\nu}^{\lambda L}.
}
\tag{MP-EQ-07}
\]

A basis-independent multiplicity-space decomposition is

\[
\boxed{
\mathscr M_{\boldsymbol\nu}^{\lambda L}
\cong
\bigoplus_{\boldsymbol\kappa,\boldsymbol\Lambda}
\mathcal D_{\boldsymbol\kappa\boldsymbol\Lambda}
\otimes
\mathbb C^{c_{\boldsymbol\kappa}^{\lambda}}
\otimes
\mathcal M_{\boldsymbol\Lambda}^{L}.
}
\tag{MP-EQ-07a}
\]

Hence

\[
\boxed{
\alpha_{\boldsymbol\nu}^{\lambda L}
:=
\dim\mathscr M_{\boldsymbol\nu}^{\lambda L}
=
\sum_{\boldsymbol\kappa,\boldsymbol\Lambda}
d_{\boldsymbol\kappa\boldsymbol\Lambda}
c_{\boldsymbol\kappa}^{\lambda}
M_{\boldsymbol\Lambda}^{L}.
}
\tag{MP-EQ-07b}
\]

The dimension identity is

\[
\boxed{
\frac{N!}{\prod_bk_b!}
\prod_{b=1}^{B}(2l_b+1)^{k_b}
=
\sum_{\lambda\vdash N}\sum_L
\dim[\lambda]\,(2L+1)\,
\alpha_{\boldsymbol\nu}^{\lambda L}.
}
\tag{MP-EQ-07c}
\]

---

## 4. Coupled orthonormal basis, intertwiners, and learned maps

### MP-EQ-08 — synthesis basis and feature analysis `[DERIVED]`

Let \(\upalpha=(\lambda,L,a,t,M)\). For a selected tree \(\mathcal T\),

\[
\boxed{
|\upalpha;\mathcal T\rangle
=
\sum_{q\in\mathcal Q_{\boldsymbol\nu}}
\mathsf C^{(\mathcal T)}_{q\upalpha}|q\rangle.
}
\tag{MP-EQ-08}
\]

Thus \(\mathsf C\) is a synthesis matrix. Coupled feature coordinates are obtained by the adjoint analysis map:

\[
\boxed{
B_{i,\upalpha}^{(\mathcal T)}
=
\sum_q
\overline{\mathsf C^{(\mathcal T)}_{q\upalpha}}
Z_{i,q}
=
\left[\mathsf C^{(\mathcal T)\dagger}\mathbf Z_i\right]_{\upalpha}.
}
\tag{MP-EQ-08a}
\]

### MP-EQ-09 — orthonormality, completeness, and selected projectors `[STD/DERIVED]`

For retained columns,

\[
\boxed{
\mathsf C^\dagger\mathsf C=I_{\mathrm{cpl}}.
}
\tag{MP-EQ-09}
\]

For a complete table,

\[
\boxed{
\mathsf C\mathsf C^\dagger=I_{\mathrm{raw}}.
}
\tag{MP-EQ-09a}
\]

For a selected subset \(\mathcal S\),

\[
\boxed{
\mathsf C_{\mathcal S}^\dagger\mathsf C_{\mathcal S}=I_{\mathcal S},
\qquad
\mathsf C_{\mathcal S}\mathsf C_{\mathcal S}^\dagger
=
\Pi_{\mathcal S}.
}
\tag{MP-EQ-09b}
\]

The character projector onto the \(\lambda\)-isotypic component is

\[
\Pi_\lambda
=
\frac{\dim[\lambda]}{N!}
\sum_{\sigma\in S_N}
\overline{\chi^\lambda(\sigma)}P(\sigma).
\]

It does not select an individual tableau component or multiplicity copy.

### MP-EQ-10 — intertwining property `[STD/DERIVED]`

With \(\rho_{\mathrm{raw}}(\sigma,R)=P(\sigma)D(R)\),

\[
\boxed{
\rho_{\mathrm{raw}}(\sigma,R)\mathsf C
=
\mathsf C
\bigoplus_{\lambda,L}
\left[
D^\lambda(\sigma)\otimes
D^L(R)\otimes
I_{\mathscr M_{\boldsymbol\nu}^{\lambda L}}
\right].
}
\tag{MP-EQ-10}
\]

Tableaux, diagrams, and paths label states or copies; they do not themselves supply the numerical coefficients in \(\mathsf C\).

### MP-EQ-11 — change of complete coupling tree `[STD/DERIVED]`

For complete orthonormal trees \(\mathcal T\) and \(\mathcal T'\), using the same parent Young and magnetic bases,

\[
\boxed{
\mathsf C_{\mathcal T'}^{\lambda L}
=
\mathsf C_{\mathcal T}^{\lambda L}
\left(
I_{[\lambda]\otimes V_L}
\otimes
\mathsf R_{\mathcal T\to\mathcal T'}^{\lambda L}
\right),
\qquad
\mathsf R^\dagger\mathsf R=I.
}
\tag{MP-EQ-11}
\]

The recoupling unitary acts on path/multiplicity coordinates. A complete linear tree does not change \(\alpha_{\boldsymbol\nu}^{\lambda L}\). Pruning, truncation, or nonlinear operations between partial couplings can change the retained subspace and therefore define a different architecture.

### MP-EQ-12 — most general linear equivariant map `[STD]`

For each fixed rank \(N\), let

\[
\mathcal H_{N,\mathrm{hid}}
=
\bigoplus_{\lambda\vdash N,L}
U_{N\lambda L}\otimes\mathcal A_{N\lambda L},
\qquad
U_{N\lambda L}=[\lambda]\otimes V_L.
\]

Over the declared complex convention, Schur's lemma gives

\[
\boxed{
\operatorname{Hom}_{S_N\times SO(3)}
\!\left(
U_{N\lambda L}\otimes\mathcal A_{N\lambda L},
U_{N\lambda' L'}\otimes\mathcal B_{N\lambda' L'}
\right)
\cong
\begin{cases}
\operatorname{Hom}(\mathcal A_{N\lambda L},\mathcal B_{N\lambda L}),
&(\lambda,L)=(\lambda',L'),\\
0,&(\lambda,L)\ne(\lambda',L').
\end{cases}
}
\tag{MP-EQ-12}
\]

The concrete map in the matching block is \(I_{U_{N\lambda L}}\otimes W_{N\lambda L}\). Different ranks carry representations of different symmetric groups; a generic linear layer must therefore keep \(N\) fixed. A rank-changing operation requires an explicit induction, restriction, insertion/removal, or other declared group-changing map.

**Interpretation of \(W_\theta\):**

- it may mix all coordinates that transform trivially under the group;
- this may include exact multiplicity copies, content/source channels, radial/chemical channels, and a selected path basis;
- it may mix copies originating from different \(\boldsymbol\nu\) or source types if they have already been placed in the same \(\theta\) block and the architecture permits it;
- preserving source/path structure is an optional sparsity or interpretability constraint, not an equivariance requirement;
- it must not directly mix different \(N\), \(\lambda\), or \(L\), and must not depend arbitrarily on \(t\) or \(M\).

This is the precise replacement for the vague sentence “\(W\) mixes only multiplicity copies.”

---

## 5. Physical realization maps

The YE3T analysis map is independent of how the raw coordinates are physically realized. The main realizations here are lifted-density products and explicit rooted motifs.

### MP-EQ-13 — primitive one-interaction feature `[PROPOSED input convention]`

\[
\boxed{
\phi_{ij,\eta lm}
=
R_{\eta l}
\!\left(
 r_{ij};
 x_i^{\mathrm{sc}},x_j^{\mathrm{sc}},Q_i,Q_j,\ldots
\right)
Y_l^m(\widehat{\mathbf r}_{ij}),
\qquad
\nu=(\eta,l).
}
\tag{MP-EQ-13}
\]

Here \(x_i^{\mathrm{sc}}\) denotes scalar atom attributes. Scalar charge may condition a radial/filter map without changing the \(V_l\) carrier. A vectorial atom variable must carry its own nontrivial rotational representation.

### MP-EQ-14 — ordinary and edge-filtered lifted densities `[PROPOSED realization]`

\[
\boxed{
A_{i,\eta lm}
=
\sum_{j\in\mathcal N_i}
\phi_{ij,\eta lm},
\qquad
A_{i,s,\eta lm}
=
\sum_{j\in\mathcal N_i}
 w_{is}(j)\phi_{ij,\eta lm}.
}
\tag{MP-EQ-14}
\]

The scalar weight \(w_{is}(j)\):

- must be rotation invariant;
- must depend on atom/edge data rather than arbitrary neighbor-list position;
- may be a smooth shell/window, radial filter, scalar attention weight, role map, or another permutation-compatible scalar;
- does not create a physical symmetry that exchanges different radial shells.

### MP-EQ-15 — formal post-pooling radial-role map `[PROPOSED; exact under span condition]`

When the filtered radial functions lie in the span of a base radial basis, define

\[
\widetilde R_{i,s,\eta l}
=
\sum_{\eta'}
K_{i,s;\eta\eta'}^{(l)}R_{\eta'l}.
\]

Then

\[
\boxed{
A_{i,s,\eta lm}
=
\sum_{\eta'}
K_{i,s;\eta\eta'}^{(l)}A_{i,\eta'lm},
\qquad
K_{i,s}
=
\bigoplus_l
\left(K_{i,s}^{(l)}\otimes I_{V_l}\right).
}
\tag{MP-EQ-15}
\]

This is the formal meaning of an equivariant radial-channel/role lift: \(K_{i,s}^{(l)}\) acts on non-angular copies of the same \(V_l\) and leaves \(m\) untouched. It can be applied after pooling when the span relation is exact. A neighbor-dependent filter outside that span must be evaluated before the neighbor sum.

No radial polynomial family automatically creates mixed permutation symmetry. Nontrivial factor-permutation content arises from retaining different ordered factor vectors/roles through the YE3T coupling.

### MP-EQ-16 — exact density-product expansion `[STD/DERIVED]`

\[
\boxed{
\prod_{f=1}^{N}
A_{i,s_f,\mu_fm_f}
=
\sum_{j_1\in\mathcal N_i}\cdots
\sum_{j_N\in\mathcal N_i}
\prod_{f=1}^{N}
\left[
 w_{is_f}(j_f)
 \phi_{ij_f,\mu_fm_f}
\right].
}
\tag{MP-EQ-16}
\]

Repeated neighbors \(j_f=j_{f'}\) are allowed. Injectivity, exclusion of repeated neighbors, or removal of self-interaction is an additional modeling choice.

### MP-EQ-17 — explicit rooted motif realization `[PROPOSED realization]`

Let \(\mathsf M\) be a typed rooted motif with root \(v_\star\) and factor-support edges \(e_f=(u_f,v_f)\). For a root-preserving, type-preserving injective embedding

\[
\iota\in\operatorname{Emb}_i(\mathsf M,\mathcal G),
\qquad
\iota(v_\star)=i,
\]

define

\[
\boxed{
\Phi_{i,\iota;\boldsymbol\mu,\mathbf m}^{\mathsf M}
=
\prod_{f=1}^{N}
\phi_{\iota(u_f)\,\iota(v_f),\mu_fm_f}.
}
\tag{MP-EQ-17}
\]

Distinct unlabeled physical occurrences may be represented by

\[
\operatorname{Occ}_i(\mathsf M;\mathcal G)
=
\operatorname{Emb}_i(\mathsf M,\mathcal G)
\big/
\operatorname{Aut}_\star(\mathsf M).
\]

Quotienting physical embeddings must not silently project away a desired nontrivial internal representation of motif roles. Use an automorphism-adapted internal carrier or a declared orientation/transformation convention.

### MP-EQ-18 — unified raw coordinates `[PROPOSED interface]`

For \(q=(\boldsymbol\mu,\mathbf m)\), let \(\chi\) be either a motif embedding or a lifted-density role tuple \(\boldsymbol s=(s_1,\ldots,s_N)\). Define

\[
\boxed{
Z_{i,\chi;q}
=
\begin{cases}
\Phi_{i,\iota;\boldsymbol\mu,\mathbf m}^{\mathsf M},
&\chi=(\mathsf M,\iota),\\[4pt]
\displaystyle
\prod_{f=1}^{N}
A_{i,s_f,\mu_fm_f},
&\chi=\boldsymbol s.
\end{cases}
}
\tag{MP-EQ-18}
\]

The raw source input to coupling is the coordinate vector

\[
\boxed{
\mathbf Z_{i,\chi}^{\boldsymbol\nu}
=
\left(Z_{i,\chi;q}\right)_{q\in\mathcal Q_{\boldsymbol\nu}}.
}
\tag{MP-EQ-18a}
\]

This avoids the notational ambiguity of calling both \(Z_q\) and \(Z_q|q\rangle\) “the uncoupled basis.”

### MP-EQ-19 — ordinary-density selection rule and lifted-role covariance `[DERIVED]`

For one shared density vector, commutativity gives

\[
\boxed{
P(\sigma)\mathbf Z_i^{\mathrm{dens}}
=
\mathbf Z_i^{\mathrm{dens}}
\quad\forall\sigma\in S_N,
\qquad
\Pi_\lambda\mathbf Z_i^{\mathrm{dens}}=0
\quad\text{for }\lambda\ne(N).
}
\tag{MP-EQ-19}
\]

For an ordered role tuple, with

\[
(\sigma\!\cdot\!\boldsymbol s)_f
=s_{\sigma^{-1}(f)},
\]

\[
\boxed{
P(\sigma)\mathbf Z_{i,\boldsymbol s}^{\mathrm{lift}}
=
\mathbf Z_{i,\sigma\cdot\boldsymbol s}^{\mathrm{lift}}.
}
\tag{MP-EQ-19a}
\]

The span of the role orbit may contain nontrivial \([\lambda]\) carriers. Such components vanish when the relevant role vectors are identical, proportional, or collapsed before coupling.

The formal left \(S_N\) action above must not be confused with relabeling the
typed role/filter alphabet itself. For right-coset coordinates \(gH\), the
physical lifted coordinate is serialized with
\(s_f=s^\circ_{g^{-1}(f)}\). A formal slot permutation acts by left
multiplication on \(gH\) and therefore on the Young tableau axis. A permutation
of distinct role labels acts through the right normalizer when defined and may
mix multiplicity/source channels. Different radial shells are not implicitly
exchangeable. If an architecture declares such a role automorphism as a
symmetry, its learned maps must be compiled to commute with that additional
action; otherwise no role-label-swap invariance may be claimed.

### MP-EQ-19b — pair-dependent lifted source families `[DERIVED/PROPOSED interface]`

Let \(b(i,j)=(Z_i,Z_j)\) denote a declared directed chemical pair type, or its
unordered analogue when the model enforces pair symmetry. Pair-dependent
cutoffs, radial decay rates, and radial basis sizes may be included in the
scalar radial map:

\[
\boxed{
\phi^{(b)}_{ij,\eta lm}
=
\mathbf 1_{b(i,j)=b}\,
R^{(b)}_{\eta l}
\!\left(r_{ij};r_{\mathrm c}^{(b)},\beta_b,\ldots\right)
Y_l^m(\widehat{\mathbf r}_{ij}),
\qquad
A_{i,b,s,\eta lm}
=
\sum_{j\in\mathcal N_i}
w^{(b)}_{is}(j)\phi^{(b)}_{ij,\eta lm},
\quad 1\le s\le K_b.
}
\tag{MP-EQ-19b}
\]

Here \(r_{\mathrm c}^{(b)}\), \(\beta_b\), the radial basis size, and \(K_b\)
may depend on \(b\). They carry the trivial rotational action. Dependence only
on chemical types and invariant edge data also preserves permutation
equivariance under relabeling atoms of the same species. A directed pair table
may distinguish central and neighbor species; an unordered table must serialize
and enforce its exchange-sharing convention. Smooth cutoff regularity must be
sufficient for every requested force, stress, and higher derivative.

For each destination type
\(\theta=(N,\lambda,L,\mathrm{convention\_id})\), a pair source has its own
typed domain \(X_{b,N}\) and exact analysis map

\[
\boxed{
F_{b,\theta}:X_{b,N}\longrightarrow
U_\theta\otimes\mathcal A_{b,\theta},
\qquad
F_{b,\theta}\rho_{b,N}(g)
=
\left(D_\theta(g)\otimes I\right)F_{b,\theta},
\quad g\in S_N\times SO(3).
}
\tag{MP-EQ-19c}
\]

Source families with different dimensions or role counts may therefore be
combined as a direct sum and mixed or summed after analysis into aligned copies
of the same \(\theta\). Raw source coordinates of different types must not be
added before their typed intertwiners. A packed runtime may use offsets or
source-signature buckets. Padding is valid only when the padded zero subspace is
invariant under the complete declared source action.

The role count \(K_b\), formal tensor rank \(N\), and radial basis size are
independent quantities. For an ordered rank-\(N\) role assignment with
occupancy

\[
\mathbf m=(m_1,\ldots,m_{K_b}),
\qquad \sum_s m_s=N,
\qquad
H_{\mathbf m}=\prod_{s=1}^{K_b}S_{m_s},
\]

the assignment orbit carries the Young permutation module

\[
\boxed{
\mathbb C[\operatorname{Orb}_{S_N}(\boldsymbol s^\circ)]
\cong
\mathbf 1\!\uparrow_{H_{\mathbf m}}^{S_N},
\qquad
a_{\lambda}(\mathbf m)
=
\dim\operatorname{Hom}_{H_{\mathbf m}}
\!\left(
\mathbf 1,
\operatorname{Res}_{H_{\mathbf m}}^{S_N}[\lambda]
\right).
}
\tag{MP-EQ-19d}
\]

When the sorted nonzero occupancy is the partition \(\mu\), this multiplicity
is the Kostka number \(K_{\lambda\mu}\). If a repeated block carries a declared
nontrivial internal source representation \(\rho\), replace the trivial
representation in MP-EQ-19d by \(\rho\). These finite-group multiplicities must
still be intersected with angular coupling multiplicities for the selected
input \(l\) values and parent \(L\).

### MP-EQ-19e — rank growth with fewer role types `[DERIVED/VERIFY source image]`

Formal rank is not bounded by role count. Let a rank-3 child use three role
types \((a,b,c)\). A rank-additive product of two such children has six formal
factor positions and role occupancy \((2,2,2)\):

\[
(a,b,c)\,\Vert\,(a,b,c),
\qquad
H_{(2,2,2)}=S_2\times S_2\times S_2.
\tag{MP-EQ-19e}
\]

The child permutation carriers couple by LR induction

\[
\left([\lambda_A]\boxtimes[\lambda_B]\right)
\uparrow_{S_3\times S_3}^{S_6},
\]

while their angular carriers couple by the ordinary CG map. For example, the
unrestricted finite-group candidate decomposition is

\[
[2,1]*[2,1]
=
[4,2]+[4,1,1]+[3,3]+2[3,2,1]
+[3,1,1,1]+[2,2,2]+[2,2,1,1].
\tag{MP-EQ-19f}
\]

The role occupancy module
\(\mathbf 1\uparrow_{S_2^3}^{S_6}\) permits only partitions dominating
\((2,2,2)\). Intersecting that source condition with MP-EQ-19f removes at least
\([3,1,1,1]\) and \([2,2,1,1]\) for this realization. The exact
\(\mathsf L_v\) image, angular multiplicity, and any relation between identical
child copies may reduce the remaining copies further. In particular, the
self-product of one identical child vector can occupy only a symmetric-square
subspace. Compiler-distinct source branches or channels may realize a larger
image, but code must not declare every formal LR path independent without the
source-image calculation.

The same principle gives rank 4 from three roles through occupancy \((2,1,1)\),
whose role module contains
\([4]\oplus2[3,1]\oplus[2,2]\oplus[2,1,1]\). With two roles, occupancy
\((2,2)\) contains \([4]\oplus[3,1]\oplus[2,2]\), whereas occupancy
\((4,0)\) contains only \([4]\). None of these statements imposes
\(\kappa_b=(k_b)\) globally.

**`[VERIFY]` requirements for MP-EQ-19b--MP-EQ-19f:**

1. Compare all stated low-rank occupancy and LR intersections with exact
   projectors, fixed-content images, and `ye3t.couplings.count`.
2. Verify rotations, same-species atom relabeling, role covariance, and
   neighbor-list reordering for mixed pair tables and unequal \(K_b\).
3. Verify every per-pair \(F_{b,\theta}\) satisfies MP-EQ-19c and that combining
   aligned outputs preserves equivariance.
4. Verify energy/force continuity at pair-dependent cutoffs and derivatives
   with respect to geometry and trainable radial parameters.
5. Serialize directed/unordered pair semantics, cutoffs, decay rates, radial
   basis sizes, role counts, source signatures, and destination conventions.
6. Compare formal LR multiplicities with the realized image of identical and
   compiler-distinct rank-3 child pairs before enabling rank-6 fast paths.

Validation record: YE3T's LR-tableau counters and its independent
induced-character counters agree for every partition of 6 in the
`(2,1) x (2,1)` child product and the `(2),(2),(2)` repeated-role induction.
This validates the formal multiplicities and the candidate intersection stated
above. It does not yet validate a physical `L_v` image or angular coupling.

`REFERENCE_TODO`: record the precise theorem locations for Young's rule/Kostka
multiplicities and LR induction in James--Kerber or Sagan before promoting the
pair-source realization from `VERIFY` to a completed project theorem. The
standard references are listed in Section 10; no external source code is used.

---

## 6. Binary coupling and arbitrary coupling trees

There are two distinct finite-group products that code must not conflate.

### 6.1 Rank-additive factor concatenation `[STD]`

For \(N=N_A+N_B\),

\[
\boxed{
\left([\lambda_A]\boxtimes[\lambda_B]\right)
\uparrow_{S_{N_A}\times S_{N_B}}^{S_N}
\cong
\bigoplus_{\lambda\vdash N}
[\lambda]\otimes
\mathbb C^{c_{\lambda_A,\lambda_B}^{\lambda}}.
}
\tag{MP-EQ-20}
\]

This is the Littlewood--Richardson/outer-product coupling appropriate when child factor sets are concatenated and the parent rank is additive.

### 6.2 Same-rank diagonal product `[STD]`

If two features both carry representations of the same \(S_N\) and their product is taken under the diagonal \(S_N\) action, rank remains \(N\) and

\[
\boxed{
\left([\lambda_A]\boxtimes[\lambda_B]\right)
\downarrow_{\Delta S_N}^{S_N\times S_N}
\cong
\bigoplus_{\lambda\vdash N}
[\lambda]\otimes
\mathbb C^{g_{\lambda_A,\lambda_B}^{\lambda}},
}
\tag{MP-EQ-21}
\]

where \(g_{\lambda_A,\lambda_B}^{\lambda}\) is a Kronecker coefficient. Do not replace this by LR induction.

For rotations, both coupling modes use

\[
\boxed{
V_{L_A}\otimes V_{L_B}
\cong
\bigoplus_{L=|L_A-L_B|}^{L_A+L_B}V_L.
}
\tag{MP-EQ-22}
\]

### 6.3 Exact joint binary coupler `[STD/PROPOSED interface]`

For a typed path \(p\), let \(\mathsf Y_p\) be the appropriate finite-group synthesis intertwiner (LR-induced or same-rank Kronecker, as declared) and let \(\mathsf A_p\) be the angular synthesis intertwiner. Define

\[
\boxed{
\mathsf J_p
=
\mathsf Y_p\otimes\mathsf A_p.
}
\tag{MP-EQ-23}
\]

Given a correctly assembled child-product coordinate vector, the local analysis/update is

\[
\boxed{
x_v
=
\bigoplus_{\theta_v}
\sum_{p_v\to\theta_v}
\left(I_{U_{\theta_v}}\otimes W_{p_v}\right)
\mathsf J_{p_v}^{\dagger}
\mathsf L_v
\left(x_{v_{\mathrm L}}\otimes x_{v_{\mathrm R}}\right).
}
\tag{MP-EQ-24}
\]

Here:

- \(\mathsf L_v\) is a fixed, exact source/placement assembly map;
- \(\mathsf J_{p_v}^{\dagger}\) performs the finite-group and angular analysis/projection;
- \(W_{p_v}\) acts only on LR/Kronecker copies, child multiplicities/channels, and other trivial axes;
- \(U_{\theta_v}=[\lambda_v]\otimes V_{L_v}\) is the intermediate parent carrier.

For rank-additive induction, \(\mathsf L_v\) must supply or lazily represent the required placement/coset orbit. If repeated content or source automorphisms impose relations among placements, code must not treat every induced coordinate as an independent physical feature. This reduction must be exact/source-defined, not discovered by runtime rank testing.

Balanced, left-recursive, block-aligned, and graph-aligned trees are permitted. If all allowed intermediates are retained and the tree remains linear, they are related by recoupling. Intermediate learned nonlinearities, pruning, or truncation define architecture-dependent subspaces.

**This equation is where different \(L\) and \(\lambda\) values interact. The subsequent \(W_{p_v}\) does not itself mix them.**

---

## 7. Master YE3T message-passing equations

### MP-EQ-25 — rank-graded hidden state `[PROPOSED interface]`

\[
\boxed{
h_i^{(r)}
\in
\bigoplus_{N\ge0}
\bigoplus_{\lambda\vdash N}
\bigoplus_L
U_{N\lambda L}
\otimes
\mathcal A_{i,r,N\lambda L},
\qquad
U_{N\lambda L}=[\lambda]\otimes V_L.
}
\tag{MP-EQ-25}
\]

This direct-sum notation is a **rank-graded storage/type declaration**, not one representation of a single symmetric group acting simultaneously on every rank. Each fixed-\(N\) block carries its own \(S_N\times SO(3)\) action. Different ranks must not be merged merely because arrays have similar shapes.

### MP-EQ-26 — source-to-message analysis `[PROPOSED]`

For source \(\chi\) with content \(\boldsymbol\nu_\chi\) and tree \(\mathcal T_\chi\),

\[
\boxed{
\widetilde m_{i,\chi}^{(r),N\lambda L}
=
\left(
I_{[\lambda]\otimes V_L}
\otimes
W_{\chi}^{(r),N\lambda L}
\right)
\left(
\mathsf C_{\boldsymbol\nu_\chi,\mathcal T_\chi}^{\lambda L}
\right)^\dagger
\mathbf Z_{i,\chi}^{(r),\boldsymbol\nu_\chi}.
}
\tag{MP-EQ-26}
\]

The source-specific multiplicity/channel space may be mapped to a common output channel basis before aggregation. Equivalent physical sources must use tied parameterization or another explicitly equivariant sharing rule.

### MP-EQ-27 — carrier-aligned aggregation and update `[PROPOSED]`

\[
\boxed{
m_i^{(r),N\lambda L}
=
\sum_{\chi\in\mathcal X_i^{(r),N\lambda L}}
\widetilde m_{i,\chi}^{(r),N\lambda L},
\qquad
m_i^{(r)}
=
\bigoplus_{N,\lambda,L}
m_i^{(r),N\lambda L}.
}
\tag{MP-EQ-27}
\]

A plain sum is valid only after all terms share the same \((N,\lambda,L)\) type, the same \(t,M\) basis convention, and the same output-channel basis. Different types remain in a direct sum or are tensor-product coupled to a selected parent.

The update satisfies

\[
\boxed{
h_i^{(r+1)}
=
\mathcal U^{(r)}
\left(h_i^{(r)},m_i^{(r)}\right).
}
\tag{MP-EQ-28}
\]

For every fixed-rank carrier block, \(\mathcal U^{(r)}\) must intertwine the declared \(S_N\times SO(3)\) action. Any rank-changing part of \(\mathcal U^{(r)}\) must be assembled explicitly from typed couplers such as MP-EQ-20--MP-EQ-24; there is no single common \(S_N\) action across all ranks.

### MP-EQ-29 — compact architecture-neutral master layer `[PROPOSED]`

\[
\boxed{
h_i^{(r+1)}
=
\mathcal U^{(r)}\!\left(
h_i^{(r)},
\bigoplus_{N,\lambda\vdash N,L}
\sum_{\chi\in\mathcal X_i^{(r),N\lambda L}}
\left(I_{[\lambda]\otimes V_L}\otimes
W_\chi^{(r),N\lambda L}\right)
\left(
\mathsf C_{\boldsymbol\nu_\chi,\mathcal T_\chi}^{\lambda L}
\right)^\dagger
\mathbf Z_{i,\chi}^{(r),\boldsymbol\nu_\chi}
\right).
}
\tag{MP-EQ-29}
\]

Use this collapsed equation only when \(\mathcal T_\chi\) is a factorization of one linear analysis map. If learned maps, gates, nonlinearities, pruning, or aggregation occur at internal nodes, use the local recursion MP-EQ-24.

### MP-EQ-30 — safe scalar gate `[STD/PROPOSED architecture]`

\[
\boxed{
\left[\operatorname{Gate}(g_i)x_i\right]_{a t M}^{N\lambda L}
=
g_{i,a}^{N\lambda L}
x_{i,a t M}^{N\lambda L},
}
\tag{MP-EQ-30}
\]

where \(g_{i,a}^{N\lambda L}\) is invariant and shared across all \(t,M\) components of that irrep copy. Arbitrary componentwise nonlinearities on nontrivial carrier components are not generally equivariant.

### 7.1 Charge and other atom variables

A scalar charge \(Q_i\) may:

- condition \(R_{\eta l}\), \(w_{is}\), or \(K_{i,s}^{(l)}\);
- appear as an \(L=0\) node feature with a declared trivial permutation type;
- be predicted by a scalar readout and reused in later layers.

Equivariance does not impose charge conservation. A constraint such as

\[
\sum_iQ_i=Q_{\mathrm{tot}}
\]

is a separate global constraint or projection. Vectorial or tensorial atom variables must be typed by their own rotational carriers and coupled through exact intertwiners.

---

## 8. Type and storage contracts

### 8.1 Carrier key and axes

Minimum key:

```text
(N, lambda, L, convention_id)
```

Logical axes for one block:

```text
[trivial_axis_a, Young_component_t, magnetic_component_M]
```

The `trivial_axis_a` may combine exact multiplicity copies, content/source labels, radial/chemical channels, path-basis labels, and learned channels. The code must document that packing.

### 8.2 Required operations

```text
Linear(theta):
    x[theta, a, t, M] -> y[theta, a_out, t, M]
    via W_theta[a_out, a]
```

```text
CoupleRankAdditive(theta_A, theta_B, path):
    N_out = N_A + N_B
    permutation coupling = LR induction
    rotation coupling = CG
```

```text
CoupleSameRank(theta_A, theta_B, path):
    require N_A = N_B = N_out
    permutation coupling = Kronecker/diagonal S_N product
    rotation coupling = CG
```

```text
Aggregate(theta):
    sum only messages already aligned to the same theta and channel basis
```

Do not implement a generic dense matrix that maps an \(L\) axis to another \(L'\) axis. Cross-\(L\) interactions belong in `Couple*`, not `Linear`.

### 8.3 Coefficient-table shapes

For fixed content \(\boldsymbol\nu\),

\[
n_{\mathrm{raw}}
=
\frac{N!}{\prod_bk_b!}
\prod_b(2l_b+1)^{k_b},
\qquad
n_{\lambda L}
=
\dim[\lambda](2L+1)
\alpha_{\boldsymbol\nu}^{\lambda L}.
\]

Store

```text
C[nu, tree, lambda, L] : (n_raw, n_lambdaL)   # synthesis
C_dagger               : (n_lambdaL, n_raw)   # analysis
W[theta]               : (a_out, a_in)
```

A factorized implementation may store sparse local couplers rather than a dense full \(\mathsf C\), but it must implement the same declared map.

### 8.4 Required metadata

Serialize with every table/cache:

- ordered-content convention;
- factor and raw-index order;
- Young basis and tableau order;
- spherical basis and CG/Wigner convention;
- multiplicity/copy ordering and phases;
- coupling tree and path order;
- synthesis-versus-analysis orientation;
- real/complex basis conversion;
- exact or floating coefficient representation;
- table/schema version.

### 8.5 Runtime separation

- **Offline algebra:** exact multiplicities, representation matrices, intertwiners, projectors, and recoupling tables.
- **Compiler:** select paths, sparse schedules, layouts, caching, and kernel fusion.
- **Runtime:** evaluate already-defined maps on features.
- **Validation:** compare exact identities and floating implementations.

Runtime must not discover multiplicities, ranks, or basis vectors from evaluated data.

---

## 9. Required validation identities

1. **Dimensions:** verify MP-EQ-07c exactly for representative ranks and contents.

2. **Intertwining:** for generators \(\sigma\) and representative rotations \(R\), verify

   \[
   \rho_{\mathrm{raw}}(\sigma,R)\mathsf C
   =
   \mathsf C\rho_{\mathrm{cpl}}(\sigma,R).
   \]

3. **Orthonormality/projectors:**

   \[
   \mathsf C^\dagger\mathsf C=I,
   \qquad
   \Pi_{\mathcal S}^\dagger=\Pi_{\mathcal S},
   \qquad
   \Pi_{\mathcal S}^2=\Pi_{\mathcal S}.
   \]

4. **Completeness:** for complete tables, \(\mathsf C\mathsf C^\dagger=I_{\mathrm{raw}}\).

5. **Tree equivalence:** compare complete trees after the recoupling unitary in MP-EQ-11.

6. **Linear-type safety:** confirm `Linear(theta)` commutes with the group action and has no cross-\(N\), cross-\(\lambda\), or cross-\(L\) weights.

7. **Coupling-type safety:** separately test rank-additive LR coupling and same-rank Kronecker coupling. A regression test must fail if one table type is substituted for the other.

8. **Density rule:** verify \(\Pi_\lambda\mathbf Z_i^{\mathrm{dens}}=0\) for \(\lambda\ne(N)\), and construct role-resolved examples with nonzero mixed-symmetry projection.

9. **Neighbor reordering:** reordering an input neighbor list must not change pooled densities or aggregation over the same physical motif occurrences.

10. **Density versus motif source semantics:** repeated neighbor indices must be included in density-product tests and excluded in injective motif-embedding tests.

11. **Charge typing:** scalar charge conditioning must commute with rotations; any global conservation constraint must be tested separately.

---

## 10. References for online lookup

### Symmetric groups, induction, Kronecker products, tableaux, and Schur--Weyl theory

1. G. D. James and A. Kerber, *The Representation Theory of the Symmetric Group*, Cambridge University Press (1984), DOI `10.1017/CBO9781107340732`.
2. B. E. Sagan, *The Symmetric Group*, 2nd ed., Springer (2001), DOI `10.1007/978-1-4757-6804-6`.
3. W. Fulton, *Young Tableaux*, Cambridge University Press (1997), DOI `10.1017/CBO9780511626241`.
4. W. Fulton and J. Harris, *Representation Theory: A First Course*, Springer (1991), DOI `10.1007/978-1-4612-0979-9`.
5. J.-P. Serre, *Linear Representations of Finite Groups*, Springer. Use for induction, characters, semisimplicity, and Schur's lemma.
6. V. Chilla, “On the linear equation method for the subduction problem in symmetric groups,” *J. Phys. A* **39**, 7657 (2006), DOI `10.1088/0305-4470/39/24/004`.
7. V. Chilla, “A reduced subduction graph and higher multiplicity in S_n transformation coefficients,” *J. Phys. A* **39**, 12395 (2006), DOI `10.1088/0305-4470/39/40/008`.
8. I. G. Macdonald, *Symmetric Functions and Hall Polynomials*, 2nd ed., Oxford University Press. Use for symmetric functions, LR coefficients, and Kronecker-product context.

### Angular momentum and recoupling

9. A. R. Edmonds, *Angular Momentum in Quantum Mechanics*, Princeton University Press.
10. D. A. Varshalovich, A. N. Moskalev, and V. K. Khersonskii, *Quantum Theory of Angular Momentum*, World Scientific (1988), DOI `10.1142/0270`.
11. A. P. Yutsis, I. B. Levinson, and V. V. Vanagas, *Mathematical Apparatus of the Theory of Angular Momentum*.

### Atomic-density, cluster-basis, and equivariant-network foundations

12. R. Drautz, “Atomic cluster expansion for accurate and transferable interatomic potentials,” *Phys. Rev. B* **99**, 014104 (2019), DOI `10.1103/PhysRevB.99.014104`.
13. R. Drautz, “Atomic cluster expansion of scalar, vectorial, and tensorial properties including magnetism and charge transfer,” *Phys. Rev. B* **102**, 024104 (2020), DOI `10.1103/PhysRevB.102.024104`.
14. Y. Lysogorskiy et al., “Performant implementation of the atomic cluster expansion (PACE),” *npj Comput. Mater.* **7**, 97 (2021), DOI `10.1038/s41524-021-00559-9`.
15. G. Dusson et al., “Atomic cluster expansion: Completeness, efficiency and stability,” *J. Comput. Phys.* **454**, 110946 (2022), DOI `10.1016/j.jcp.2022.110946`.
16. J. Nigam, S. Pozdnyakov, and M. Ceriotti, “Recursive evaluation and iterative contraction of N-body equivariant features,” *J. Chem. Phys.* **153**, 121101 (2020), DOI `10.1063/5.0021116`.
17. J. M. Goff, C. Sievers, M. A. Wood, and A. P. Thompson, “Permutation-adapted complete and independent basis for atomic cluster expansion descriptors,” *J. Comput. Phys.* **510**, 113073 (2024), DOI `10.1016/j.jcp.2024.113073`.
18. T. S. Cohen and M. Welling, “Group Equivariant Convolutional Networks,” *ICML* (2016), arXiv `1602.07576`.
19. N. Thomas et al., “Tensor Field Networks: Rotation- and Translation-Equivariant Neural Networks for 3D Point Clouds,” arXiv `1802.08219`.
20. J. Gilmer et al., “Neural Message Passing for Quantum Chemistry,” *ICML* (2017), arXiv `1704.01212`.
21. R. Curticapean, H. Dell, and D. Marx, “Homomorphisms are a good basis for counting small subgraphs,” *STOC* (2017), DOI `10.1145/3055399.3055502`.

---

## 11. Status summary

- The finite-group, Schur--Weyl, induction/restriction, Schur-lemma, angular-CG, orthogonality, projector, and recoupling ingredients are standard mathematics.
- The full fixed-content decomposition MP-EQ-07 through MP-EQ-07c remains the current YE3T target theorem and should retain project status `VERIFY` until its formal scope, conventions, and proof record are finalized.
- The lifted-density realization, optional post-pooling \(K_s\) lift, motif realization, local binary-tree interface, and master message-passing equations are proposed YE3T constructions.
- **Do not implement \(W\) as a map that mixes \(L\), \(\lambda\), or \(N\). Implement cross-type interaction only through exact coupling/projection.**
- **Distinguish rank-additive LR induction from same-rank Kronecker coupling.**
- **Treat \(Z_{i,\chi;q}\) as raw coordinates and use \(\mathsf C^\dagger\mathbf Z\) for analysis; do not append subgroup-tableau/path indices to the raw basis.**
- **Retain role ordering through the lifted-density coupling; the radial basis family alone does not create permutation resolution.**
- **Keep source/placement assembly exact and separate from learned channel mixing; do not infer source-rank reductions with runtime SVD/QR.** The generic map \(\mathsf L_v\) is presently an interface, not a completed universal algorithm; implement only explicitly defined source-specific assembly maps.
- No implementation, testing, or benchmark claim follows from this file alone.
