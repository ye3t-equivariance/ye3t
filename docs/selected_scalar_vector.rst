Selected scalar/vector source maps
==================================

The canonical coupling API provides an analytical rank-four specialization
for content :math:`q^3v`, where :math:`q` is scalar and :math:`v` is a polar
vector. This core construction operates on representations and coefficients;
the application supplies physical factor values and occurrence identities::

   from ye3t.couplings import count, plan, compile

   request = {"kind": "selected_scalar_vector_marginal",
              "rank": 4, "hidden_width": 2}
   multiplicities = count(request)
   compiler_plan = plan(multiplicities)
   coefficients = compile(compiler_plan)
   print(multiplicities["selected_dimension"])  # 9
   print(compiler_plan["output_order"])

The permutation module :math:`\mathrm{Ind}_{S_3\times S_1}^{S_4}\mathbf1`
decomposes into :math:`[4]\oplus[3,1]`. The compiler uses the existing exact
Young-orthogonal subduction engine, then applies the pooled :math:`S_2`
invariance and selected :math:`S_2` character projectors. The selected space
has two even vector copies and one odd vector copy. Source records retain
parent and restriction multiplicities, copy indices, actual types, parity,
and stable IDs.

The hashed plan fixes output ordering, analytical backend, moment bindings,
marginal normalization, and the same-action role-product algebra. Compilation
checks a supplied plan against these choices. Exact canonical-to-selected
coordinate maps are stored with the coefficients. The role basis is
:math:`e_\pm=(1,\pm1)/\sqrt2` with the counting metric.

The compiled hidden-product inventory includes all scalar/scalar,
scalar/vector, and vector-dot routes for the configured channel width.
Repeated complete channel pairs occur once. Their physical role coefficient
is obtained from the character-basis multiplication map. The vector-dot
metric is checked against the core Condon--Shortley coupling using
:math:`v\cdot w=-\sqrt3\,[v\otimes w]_{00}` and magnetic order
:math:`(-1,0,1)`.

Disjoint-support LR induction and overlapping-support coupling require their
own placement and argument maps. This specialization selects same-action
pointwise multiplication. Its finite rank is the scope of the first source
profile; other YE3T coupling families keep their existing general interfaces.
