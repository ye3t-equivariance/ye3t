# Linear lifted-Cauchy descriptor basis

Status: mathematical standard for the linear lifted-Cauchy basis. The
fixed-rank decomposition below
is exact. Statements about the coordinates emitted by the current compiler and
the atomistic source map are separated explicitly from that theorem.

## Fixed-rank, fixed-content scope

Fix all of the following:

- tensor rank \(N\);
- complete-channel content
  \(\boldsymbol\nu=\{\nu_b^{k_b}\}_b\), with
  \(N=\sum_b k_b\);
- the role space \(W_b\) and angular carrier \(V_{l_b}\) for every repeated
  block;
- total angular momentum \(L=0\); and
- the desired parity \(\epsilon\).

With the standard tensor-product inner product and normalized symmetrization,
the finite invariant space is

\[
\mathcal I_{N,\boldsymbol\nu}^{0,\epsilon}
=
\operatorname{Hom}_{O(3)}
\!\left(
V_0^\epsilon,
\bigotimes_b
\operatorname{Sym}^{k_b}(W_b\otimes V_{l_b})
\right).
\]

For an energy descriptor, \(\epsilon=+\). The \(\epsilon=-\), \(L=0\) sector
is a pseudoscalar target, not an \(O(3)\)-invariant energy.

For each repeated block, the Cauchy identity gives the orthogonal direct sum

\[
\operatorname{Sym}^{k_b}(W_b\otimes V_{l_b})
\cong
\bigoplus_{\kappa_b\vdash k_b}
\mathbb S_{\kappa_b}(W_b)
\otimes
\mathbb S_{\kappa_b}(V_{l_b}).
\]

After resolving angular multiplicities and the final scalar coupling, this can
be written

\[
\mathcal I_{N,\boldsymbol\nu}^{0,\epsilon}
\cong
\bigoplus_{\boldsymbol\kappa,\boldsymbol\Lambda}
\left[
\bigotimes_b
\mathbb S_{\kappa_b}(W_b)
\otimes
\mathcal D_b^{\kappa_b\Lambda_b}
\right]
\otimes
\mathcal M_{\boldsymbol\Lambda}^{0,\epsilon},
\]

where \(\mathcal D_b^{\kappa_b\Lambda_b}\) is the full multiplicity space of
\(V_{\Lambda_b}\) inside \(\mathbb S_{\kappa_b}(V_{l_b})\), and
\(\mathcal M_{\boldsymbol\Lambda}^{0,\epsilon}\) is the outer scalar-coupling
multiplicity space.

This is the precise analogue of the usual fixed-rank YE3T basis statement.
Choosing orthogonal, or normalized orthogonal, bases in every role-Schur,
angular-multiplicity, and outer-coupling space gives a complete orthogonal, or
orthonormal, basis of this fixed-\(N\), fixed-content invariant space. A
selected subset of those basis vectors remains orthogonal and linearly
independent. No orthogonality between different ranks is asserted or needed.

Completeness in this statement requires every valid \(\kappa_b\), block
angular momentum \(\Lambda_b\), role copy, angular copy, and outer coupling
copy. A practical catalogue may select only some of these vectors and may
select only some ranks or contents. That makes the catalogue incomplete, but
does not make its retained vectors dependent.

The internal \(\kappa_b\) labels do not change the global permutation
symmetry. They are paired role/angular intermediate labels inside the globally
trivial parent \((N)\) sector.

The decomposition theorem is valid for every finite \(N\). The first Ta
workflow uses a bounded rank-4 catalogue, and current software remains subject
to its declared per-block and resource limits. Higher-rank plans are described
as implemented only after their `count`, `plan`, `compile`, and resource gates
pass.

## Current compiler coordinates and orthogonal-output plan

An orthogonal basis exists by the direct-sum construction above. The current
general compiler coordinates are exact and independent, but they are selected
by lexicographic pivots and analyzed with the exact metric dual

\[
C^\sharp=(C^\dagger M C)^{-1}C^\dagger M,
\]

where \(M\) is the declared ordered-carrier or orbit-weighted monomial metric.

That remains the independence and reconstruction certificate for pivot
coordinates. The implemented, hash-bound orthogonal-output plan additionally
constructs $O=RC$ inside each strict fixed sector and certifies
$R(C^\dagger M C)R^\dagger=D>0$. Thus emitted orthogonal-output coordinates
are independent and orthogonal, and normalization by $D^{-1/2}$ is optional.
They can be extended to a complete orthogonal basis of the full fixed-
$N$, fixed-content sector when the request itself is complete.

Orthogonalization may mix compiler-owned copies within one fixed role,
angular, or outer multiplicity space. It must not mix different rank, content,
\(\kappa\), block \(\Lambda\), target \(L\), or parity sectors. The copy
transformation is serialized, applied offline, and must certify

\[
C^\dagger M C=D,
\]

with positive diagonal \(D\), or \(D=I\) for normalized coordinates. Raw
coalesced monomials use the exact inverse-orbit-size metric; \(M=I\) only in an
orthonormal ordered or normalized-symmetric carrier. Linear training uses
normalized orthogonal coordinates by default and lowers the fitted readout
exactly back to executable pivot/canonical coordinates. Runtime therefore
performs no coordinate Gram, whitening, or dense change of basis. A dense Gram
matrix or inverse remains a validation oracle, not part of fitting iterations
or LAMMPS execution.

The required certificates are exact dimension counts, diagonal coordinate
Gram, completeness-projector rank for complete requests, unique valid labels,
all multiplicity copies emitted once, and canonical/factored forward and VJP
agreement.

## Abstract basis versus atomistic source realization

Let there be \(S\) roles and \(C\) radial/source-family indices. The general
atomistic source map is

\[
J(e_s\otimes e_n)=\psi_{snl}(r).
\]

The separable form

\[
\psi_{snl}(r)=w_s(r)R_{nl}(r)
\]

is an optional implementation choice, not part of the lifted-Cauchy theorem.
In particular, a lifted-density implementation must not require a common
radial function multiplied by every role filter.

Use the following terms without overloading ``rank``:

- \(N\) is tensor rank, equivalently density-product or correlation order;
- \(q\) is a radial polynomial degree or exponent;
- \(d_{\rm src}=\dim\operatorname{span}\{\psi_{snl}\}_{s,n}\) is the
  source-span dimension at fixed \(l\);
- the role/source map is **injective** when \(d_{\rm src}=SC\); and
- empirical linear algebra is qualified as design-matrix rank or
  coefficient-matrix rank.

If \(J\) is an injective, \(O(3)\)-equivariant linear source map, its tensor and
symmetric-power maps are injective, and their restrictions to Schur and
invariant subspaces remain injective. Thus an independent
internal-\(\kappa\) basis stays independent as polynomial functionals on the
full source-density space. The role index remains an explicit carrier
coordinate throughout the Cauchy coupling, so an injective source does not
remove the nontrivial permutation information.

An injective map need not be isometric under a physical radial inner product.
For a declared positive measure \(d\mu_l\), define

\[
G^{\rm src}_{sn,s'n'}
=
\int_0^{r_c}
\psi_{snl}(r)\overline{\psi_{s'n'l}(r)}\,d\mu_l(r).
\]

The source family is physically orthogonal when this Gram matrix is positive
diagonal, and orthonormal when it is the identity. There are therefore three
different notions that must be reported separately:

1. physical one-particle source/radial orthogonality under \(d\mu_l\);
2. algebraic orthogonality of the fixed-\(N\), fixed-content Cauchy
   coordinates under the exact carrier or orbit metric; and
3. empirical feature covariance on a finite atomistic dataset.

The third is neither a basis certificate nor a replacement for the first two.
If \(J\) is isometric and the compiler emits normalized orthogonal Cauchy
coordinates, the realized fixed-\(N\) descriptors inherit the corresponding
orthogonality. If \(J\) is only injective, they remain independent but are
orthogonal only in the transported abstract metric, not necessarily in the
declared physical radial metric.

## Source-orthogonal primary construction

The primary lifted-density construction should choose the joint functions
\(\psi_{snl}\) directly from an orthogonal radial/source family. For example,
let \(Q_{ql}\), \(q=0,\ldots,SC-1\), be orthogonal under the declared
\(d\mu_l\), choose a bijection \(q=q(s,n)\), and set

\[
\psi_{snl}=Q_{q(s,n),l}.
\]

This makes the role/source map injective and diagonalizes
\(G^{\rm src}\). Normalizing the \(Q_{ql}\) makes it an isometry. The role
coordinate is still explicit in \(A_{s,n,lm}\) and in every Cauchy coupling;
separability into a shell filter and a common radial basis is not required to
retain nontrivial internal \(\kappa\) information.
The \(s\) labels are then auxiliary source coordinates, not necessarily
physical inner/outer shells. Their ordering and the bijection \(q(s,n)\) are
part of the serialized convention.

An equivalent orthogonal one-particle family may be obtained by an exact or
certified offline basis change. A transform block-diagonal in source-family
index \(n\) preserves the fixed-content grading. A transform mixing different
\(n\) maps one fixed-content sector into a sum of sectors under
\(\operatorname{Sym}^N(T)\). In that case exact use requires either defining
the transformed source first and recompiling in its new channel grouping, or
applying the full induced transform over every affected content sector and
proving that the retained catalogue is closed under it. A bounded incomplete
catalogue is not assumed closed. Role action, compiler and model coefficients,
derivative schedules, and content metadata must be transported consistently.
The transformed functions and schedules are serialized. Dense source-Gram
construction, matrix factorization, or a linear solve is forbidden in fitting
loops and LAMMPS timesteps. A fixed compiler-emitted source-coordinate
transform may run as a separately benchmarked bounded kernel; it is not a
runtime Gram construction.

For a chosen finite candidate span \(Q_{ql}\), an offline certificate may form

\[
G^Q_{qq'}=\langle Q_{ql},Q_{q'l}\rangle_{\mu_l},
\qquad
\widehat Q_{al}=\sum_qT_{aq}Q_{ql},
\qquad
TG^QT^\dagger=D,
\]

with positive diagonal \(D\), or \(D=I\) after normalization. There are two
exact runtime lowerings to benchmark:

1. evaluate \(\widehat Q_{al}\) and its derivative directly with a compact
   polynomial or orthogonal-polynomial recurrence; or
2. accumulate the factorized source \(A^f\), then apply the fixed transform

   \[
   A^{\widehat Q}_{a,lm}=\sum_uT_{au}A^f_{u,lm},
   \qquad
   \overline A^f_u=\sum_aT_{au}\overline A^{\widehat Q}_a,
   \]

   once per center and angular component. In matrix notation the reverse map
   is \(\overline A^f=T^{\mathsf T}\overline A^{\widehat Q}\). This is the
   algebraic covector pullback from the chain rule, independent of whether the
   computational source functions are orthogonal. A metric/Riesz adjoint is a
   different map and must not be used for this VJP.

The second path evaluates the same orthogonal-coordinate model and preserves
its fixed-content catalogue because the descriptor receives
\(A^{\widehat Q}\), not \(A^f\). It must not apply a dense transform on every
edge. The source Gram is certificate metadata; \(T\) and its explicitly stored
and validated transpose are compact executable lowering records.

The exact polynomial family, envelope, measure, normalization, and origin
regularity are model choices and must be displayed and certified before a
fit. Direct source-orthogonal functions are the mathematical primary;
separable \(w_sR_{nl}\) sources remain useful measured fast-path ablations.

## Why the first polynomial source was dependent

The current Ta pilot uses \(x=r/r_c\), an envelope \(e(x)\),

\[
R_n=e(x)x^n,
\qquad
w_{\mathrm{in}}=1-x,
\qquad
w_{\mathrm{out}}=x.
\]

Writing \(Q_n=e(x)x^nY_{lm}\) gives

\[
A_{\mathrm{in},n}=Q_n-Q_{n+1},
\qquad
A_{\mathrm{out},n}=Q_{n+1}.
\]

For \(C=2\), for example,

\[
A_{\mathrm{out},0}
=
A_{\mathrm{in},1}+A_{\mathrm{out},1}.
\]

Four nominal role/source coordinates are therefore evaluated on a
three-dimensional physical source image. More generally this construction has
\(2C\) nominal coordinates but source-span dimension \(C+1\). The
abstract Cauchy descriptors are still independent; the dependence appears
only after restricting them through this noninjective source map.

This source remains useful as an exact folded-source scientific control. An
exact source-image quotient and pivot columns may diagnose its image, but they
are not the primary basis construction: quotient pivots are generally
nonorthogonal and can obscure unique \(\kappa\)-family attribution.

## Candidate injective factorized source

A simple candidate keeps the two smooth partition-of-unity roles but separates
their radial powers:

\[
w_0(x)=1-x,
\qquad
w_1(x)=x,
\qquad
R_{nl}(x)=e(x)x^{l+2n},
\qquad
e(x)=(1-x)^2.
\]

Then

\[
w_0R_{nl}=e(x)\left(x^{l+2n}-x^{l+2n+1}\right),
\qquad
w_1R_{nl}=e(x)x^{l+2n+1}.
\]

Indeed,

\[
\sum_n\left[a_nw_0R_{nl}+b_nw_1R_{nl}\right]
=
e(x)x^l
\sum_n\left[a_nx^{2n}+(b_n-a_n)x^{2n+1}\right].
\]

On \(0<x<1\), \(e(x)\ne0\), and all displayed powers are distinct. Therefore
\(a_n=0\) and \(b_n-a_n=0\), hence \(a_n=b_n=0\): the role/source map is
injective and has source-span dimension \(d_{\rm src}=2C\). The
\(x^lY_{lm}\) factor removes the angular direction
singularity and gives a direction-independent regular solid-harmonic limit.
The required Cartesian differentiability order of the complete
radial-envelope product is validated separately. This factorized span is the
frozen computational source for the first \(l=1\) implementation; it is not
the coordinate system used for fitting.

It is not source-orthogonal under a positive radial measure in its displayed
factorized coordinates. In particular,

\[
\left\langle (1-x)R_{nl},xR_{nl}\right\rangle
=
\int_0^{r_c}x(1-x)|R_{nl}|^2\,d\mu_l>0
\]

for a nonzero \(R_{nl}\). It can be re-orthogonalized within its same
\(2C\)-dimensional span by the offline transformation above. The resulting
\(\widehat Q\) family is a valid source-orthogonal basis. After its
\((s,n)\leftrightarrow q\) grouping is frozen and the Cauchy catalogue is
compiled for that grouping, it retains an explicit role/source coordinate and
valid internal-\(\kappa\) construction. A cross-\(n\) transform does not,
however, map each descriptor in an incomplete old fixed-content catalogue to
one descriptor with the same label. It also generally loses the simple
per-function product form \(w_sR_{nl}\). The displayed source is therefore an
injective factorized performance/localization ablation; its independently
compiled re-orthogonalized form is a candidate primary.

The displayed factorized functions may also serve as an exact computational
backend for the orthogonal primary through the post-accumulation
\(T\)/\(T^{\mathsf T}\) pair above. The compiler and descriptor catalogue remain
defined in \(\widehat Q\) coordinates, so this backend changes neither the
model nor its ridge coordinates. The transform is fixed, geometry-independent,
hash-bound, and may mix only sources with the same species/pair context,
cutoff, \(l\), parity, and spherical convention.

Its span is exactly that of

\[
Q_{q,l}(x)=(1-x)^2x^{l+q},
\qquad q=0,\ldots,2C-1,
\]

because \(Q_{2n,l}=w_0R_{nl}+w_1R_{nl}\) and
\(Q_{2n+1,l}=w_1R_{nl}\). Any ordinary ACE capacity control for this ablation
must use this identical \(2C\)-dimensional one-particle span. Comparing it only
with a \(C\)-function ordinary radial basis confounds source capacity with the
lifted-Cauchy organization.

For the first \(l=1\) Ta study, the source-orthonormal primary is frozen
explicitly. With \(d\mu=r^2dr/r_c^3=x^2dx\), \(q=2n+s\), and
\(q=0,\ldots,2C-1\), use

\[
\widehat Q_{ql}(x)=h_{ql}^{-1/2}(1-x)^2x^l
P_q^{(4,2l+2)}(2x-1),
\]

where

\[
h_{ql}=
\frac{\Gamma(q+5)\Gamma(q+2l+3)}
{(2q+2l+7)q!\Gamma(q+2l+7)}.
\]

These functions have exact identity radial/source Gram under the declared
measure. Tensor rank \(N\), Jacobi degree \(q\), total radial polynomial
degree \(l+q+2\), source-span dimension \(2C\), and empirical design-matrix
rank remain distinct quantities. The role label is retained through the
bijection \(q=2n+s\), so the source change does not remove internal
\(\kappa\) information. Direct Horner/recurrence evaluation and the exact
factorized \(T\)/\(T^{\mathsf T}\) backend execute the same fitted model.

Thus the primary, factorized computational backend, and ordinary ACE controls
can have the same one-particle capacity. Their bounded descriptor catalogues
need not span identical degree-\(N\) spaces unless the induced source
transformation closes on every retained content sector.

Performance decisions first compare direct-recurrence and
factorized-then-transform source backends on the identical orthogonal model.
A separate factorized-coordinate ablation uses identical source-span
dimension, polynomial-degree envelope, and abstract label schedule, while
recording whether its realized catalogue is transform-closed. Source/VJP and
end-to-end costs, conditioning, accuracy, and model width are reported as a
Pareto comparison. No performance threshold can convert nonorthogonal model
coordinates into orthogonal ones.

Use three distinct ordinary controls:

- a **basis-matched ordinary** control uses the exact same orthogonal
  \(\psi_{snl}\) functions as combined ordinary channels and includes every
  role-content multiset corresponding to each lifted block;
- a **span-matched ordinary** control uses the same one-particle span in a
  different basis and is a capacity control, not by itself a causal control
  for a truncated catalogue; and
- an **exhaustive-closure** control includes all affected degree-\(N\) content
  sectors and may support an exact functional-space equivalence statement.

Identity ridge is not invariant under a nonunitary source or descriptor basis
change. Exact comparisons therefore fit in common orthonormal descriptor
coordinates and lower the selected model afterward, or transport the
quadratic coefficient penalty by congruence.

## Validation checklist

Before fitting or native deployment, require:

1. exact dimension identities for a bounded matrix of ranks, contents, role
   dimensions, angular carriers, and internal partitions;
2. direct orthogonal-coordinate or exact change-of-basis certificates;
3. an exact source-span/injectivity certificate and a physical source-Gram
   certificate under the declared measure for every selected source family;
4. nonzero generic examples for each retained nontrivial \(\kappa\);
5. synthetic carrier collapse \(A_s=c_sA\) for every source family, plus
   physical filter collapse for factorized sources;
6. canonical, symmetric-power/block, and factored forward/VJP agreement; and
7. the identical compiler artifact in training, export, CPU PairYE3T, and
   PairYE3T/Kokkos; and
8. a basis-matched ordinary control for bounded accuracy claims, with
   span-matched and exhaustive-closure controls labeled separately; and
9. a common orthonormal ridge metric or an exactly transported quadratic
   penalty across every nonunitary basis change.

## References

This standard records the construction reviewed from the refined
lifted-density note and the exact-compiler evidence. Before this becomes
publication-facing documentation, verify and copy the exact bibliographic
entries and relevant sections for the Cauchy/Schur identity and ACE
conventions from the paper bibliography.
`REFERENCE_TODO`: do not invent edition, page, or DOI metadata here.
