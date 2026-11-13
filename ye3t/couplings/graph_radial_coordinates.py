"""Finite exact physical coordinates on a freely variable rooted graph.

Implements tensor_product_math.md (54a), (54b), (55) and (60). Vertices, rather
than independent edges, resolve cycle relations. One positive radius belongs
to each unordered pair. Sparse binary radical powers and a common denominator
then give independent scalar function coordinates for a bounded expression set.

These are physical function coordinates, not new representation labels or
coupling coefficients. No Young/CG maps are constructed here. Periodic aliases,
rigid constraints, radial masks and pooled moments need their own bindings and
certificates before this operation can admit a model's product image.
"""
from functools import lru_cache, reduce
from itertools import product
from math import gcd, isfinite

import sympy as sp


def _free_graph_coordinates(vertex_count, edges):
    """Immutable coordinate specification for a connected free 3D graph.

    Root zero is fixed at the origin. Other vertices vary independently on a
    collision-free open domain. Duplicate/reversed edges share their radius;
    their oriented Cartesian coordinates remain explicit substitution inputs.
    The tuple result contains vertices, oriented inputs, radii, substitutions
    and squared radii. It contains no geometry samples or atomistic imports.
    """
    if type(vertex_count) is not int or vertex_count < 2 or not isinstance(edges, tuple) or not edges:
        raise ValueError("Free graph coordinates need at least two vertices and an immutable nonempty edge list")
    if any(not isinstance(edge, tuple) or len(edge) != 2 or
           any(type(v) is not int or v < 0 or v >= vertex_count for v in edge) or edge[0] == edge[1]
           for edge in edges):
        raise ValueError("Free graph edges need distinct in-range integer endpoints; periodic aliases require another binding")
    reached = {0}
    while True:
        extended = reached | {v for edge in edges if reached.intersection(edge) for v in edge}
        if extended == reached:
            break
        reached = extended
    if len(reached) != vertex_count:
        raise ValueError("Free graph coordinates require a connected rooted graph")
    return _linear_graph_coordinates(tuple(tuple(int(v == i)-int(u == i) for i in range(1, vertex_count))
                                           for u, v in edges))


def _linear_graph_coordinates(edge_forms):
    """Coordinate binding for integer combinations of independent 3-vectors.

    Columns can represent rooted vertices and variable lattice columns. Actual
    periodic occurrence bindings belong to the application adapter. All columns
    here vary independently on an open domain; fixed-cell/rigid bindings must
    not be passed as if they supplied independent columns.

    For each primitive integer edge form, its squared length is an irreducible
    rank-three quadratic. Two such quadratics are associates precisely when
    their edge forms are proportional. Primitive sign/gcd canonicalization
    resolves this case before applying (54b). This extends its free-vertex
    proof to integer linear forms, without making images independent vertices.
    """
    if not isinstance(edge_forms, tuple) or not edge_forms or not isinstance(edge_forms[0], tuple) or not edge_forms[0]:
        raise ValueError("Graph edge forms require a nonempty tuple of integer coefficient tuples")
    width = len(edge_forms[0])
    if any(not isinstance(form, tuple) or len(form) != width or not any(form) or
           any(type(c) is not int for c in form) for form in edge_forms):
        raise ValueError("Graph edges require nonzero integer linear forms of equal dimension")
    if any(not any(form[i] for form in edge_forms) for i in range(width)):
        raise ValueError("Graph coordinate columns must each occur in at least one edge")
    # Validate before cache lookup: nested tuples containing False compare
    # equal to tuples containing 0 and must not inherit a previous admission.
    return _linear_graph_coordinates_cached(edge_forms)


@lru_cache(maxsize=32)
def _linear_graph_coordinates_cached(edge_forms):
    columns = len(edge_forms[0])
    vertices = tuple(sp.Symbol(f"coordinate_{v}_{axis}", real=True)
                     for v in range(columns) for axis in range(3))
    canonical = []
    for form in edge_forms:
        scale = reduce(gcd, form, 0)
        sign = 1 if next(c for c in form if c) > 0 else -1
        canonical.append((tuple(sign*c//scale for c in form), scale))
    forms = tuple(sorted({form for form, _ in canonical}))
    radii = tuple(sp.Symbol(f"radius_{i}", positive=True) for i in range(len(forms)))
    def vector(form):
        return tuple(sum(c*vertices[3*i+a] for i, c in enumerate(form)) for a in range(3))
    squared = tuple(sum(x**2 for x in vector(form)) for form in forms)
    oriented = tuple(tuple(sp.Symbol(f"edge_{i}_{a}", real=True) for a in range(3)) for i in range(len(edge_forms)))
    substitutions = tuple((oriented[i][a], value) for i, form in enumerate(edge_forms)
                          for a, value in enumerate(vector(form)))
    edge_radii = tuple(scale*radii[forms.index(form)] for form, scale in canonical)
    return vertices, oriented, edge_radii, radii, substitutions, squared


def _graph_radial_rows(expressions, coordinates, *, _component_terms=False, polynomial_generators=()):
    """Resolve bounded Laurent-radius expressions into independent rows.

    The returned keys are (binary radius powers, radial polynomial powers,
    polynomial monomial powers). Polynomial variables are the vertices followed
    by any caller-certified extra generators. Each independent component has one
    recorded denominator product(q_e**h_e), shared across all input rows.
    Rational coefficients must never be treated as independent constants or as
    independent rows merely because they use different denominators. Different
    binary/radial components need no common denominator. Only populated
    coordinates are built.
    """
    vertices, oriented, _, radii, substitutions, squared = coordinates
    allowed = set(vertices) | set(radii) | {x for edge in oriented for x in edge}
    polynomial_generators = tuple(polynomial_generators)
    if (any(not isinstance(value, sp.Symbol) for value in polynomial_generators)
            or len(set(polynomial_generators)) != len(polynomial_generators)
            or set(polynomial_generators) & allowed):
        raise ValueError("Additional graph polynomial generators must be distinct symbols outside the geometry coordinates")
    # The caller supplies the physical injectivity certificate (e.g. free
    # background moments); arbitrary evaluated functions are not generators.
    polynomial_variables = vertices+polynomial_generators
    allowed.update(polynomial_generators)
    @lru_cache(maxsize=32)
    def radial_terms(expression):
        expression = sp.sympify(expression)
        if expression.atoms(sp.Float) or not expression.free_symbols <= allowed:
            raise ValueError("Graph image coefficients require exact constants and only the configured coordinates")
        numerator, denominator = sp.fraction(sp.together(expression.xreplace(dict(substitutions))))
        powers = denominator.as_powers_dict()
        inverse = tuple(powers.get(r, sp.S.Zero) for r in radii)
        constant = denominator/sp.prod(r**p for r, p in zip(radii, inverse, strict=True))
        if constant.free_symbols or any(not p.is_Integer for p in inverse):
            raise ValueError("Graph image denominators must be Laurent radius monomials")
        try:
            terms = (sp.Poly(numerator, *radii).terms() if numerator.has(*radii)
                     else [((0,)*len(radii), numerator)])
        except sp.PolynomialError as error:
            raise ValueError("Graph image requires integer radius and radial polynomial powers") from error
        result = []
        for powers, coefficient in terms:
            coefficient /= constant
            if not coefficient.is_polynomial(*polynomial_variables):
                raise ValueError("Graph image vertex coefficients must be polynomial; encode inverse radii explicitly")
            radial = tuple(p-d for p, d in zip(powers, inverse, strict=True))
            halves = tuple(int(p)//2 for p in radial)
            binary = tuple(int(p) % 2 for p in radial)
            result.append((binary, halves, coefficient))
        return tuple(result)
    reduced, denominators = [], {}
    for feature in expressions:
        groups = {}
        terms = feature if _component_terms else [((), sp.S.One, feature)]
        for powers, weight, expression in terms:
            for binary, halves, coefficient in radial_terms(sp.sympify(expression)):
                component = binary, powers
                denominator = denominators.setdefault(component, [0]*len(radii))
                for i, power in enumerate(halves):
                    denominator[i] = max(denominator[i], -power)
                key = component, halves
                groups[key] = groups.get(key, sp.S.Zero)+weight*coefficient
        reduced.append(groups)
    radial_terms.cache_clear()
    rows = []
    # Common denominators preserve all cross-edge cancellations, including
    # relations between radii and dot products on cyclic unions. Sparse exact
    # polynomial arithmetic preserves them; no sampled rank or Gram is used.
    @lru_cache(maxsize=32)
    def polynomial_terms(numerator):
        # Polynomial-ring conversion performs sparse arithmetic on the factored
        # expression. Poly(expr) first expands a large symbolic Add/Mul tree,
        # even when those terms immediately cancel. Both give the same exact
        # normal form; only planning temporaries change.
        from sympy.polys.rings import ring
        domain, *_ = ring(polynomial_variables, sp.EXRAW)
        return tuple((monomial, sp.expand(domain.domain.to_sympy(coefficient)))
                     for monomial, coefficient in domain.from_expr(numerator).items())
    for groups in reduced:
        components = {}
        for (component, halves), coefficient in groups.items():
            denominator = denominators[component]
            numerator = coefficient*sp.prod(q**(p+h) for q, p, h in zip(squared, halves, denominator, strict=True))
            components[component] = components.get(component, sp.S.Zero)+numerator
        row = {}
        for (binary, radial), numerator in components.items():
            scale, primitive = numerator.as_content_primitive()
            for monomial, value in polynomial_terms(primitive):
                if value != 0:
                    row[binary, radial, monomial] = scale*value
        rows.append(row)
    polynomial_terms.cache_clear()
    return {"rows": rows, "denominator_powers": {key: tuple(value) for key, value in denominators.items()},
            "coordinate_convention": "linear_graph_radial_component_denominators_v1",
            "radical_component_count": len({key[0] for row in rows for key in row}),
            "coordinate_count": len({key for row in rows for key in row})}


def _chebexp_cos_graph_rows(features, coordinates, profiles, *, polynomial_generators=()):
    """Use the existing radial polynomials with exact graph-coordinate rows.

    Each feature is a sum of (angular expression, radial-index tuples per edge)
    terms. Empty tuples omit an edge; retained factors require n>=1. Every
    primitive radius must have one effective cutoff and at most one warp among
    its n>1 factors. A proportional edge of length c*r has effective cutoff
    rc/c. Other profiles need common rate/support analysis and fail here.

    Finite envelope/warp polynomials are independent over the radical vertex
    field on the common positive-cutoff interior: expand into distinct
    exponential rates, choose generic coordinates, and dilate all independent
    columns. The proof is in YE3T_CROSS_ROOT_PHYSICAL_IMAGE_EXTENSION.md.
    Polynomial reconstruction preserves each envelope's zero extension.
    """
    from .chebexp_cos import chebexp_cos_radial_product
    features = tuple(tuple(feature) for feature in features)
    _, _, edge_radii, radii, _, _ = coordinates
    if len(profiles) != len(edge_radii):
        raise ValueError("Graph radial profiles must match the oriented edge bindings")
    for profile in profiles:
        if (not isinstance(profile, dict) or set(profile) != {"family", "cutoff", "lambda"}
                or profile["family"] != "chebexp_cos"
                or any(type(profile[key]) not in {int, float} or not isfinite(profile[key])
                       for key in ("cutoff", "lambda"))
                or profile["cutoff"] <= 0 or profile["lambda"] == 0):
            raise ValueError("Graph ChebExpCos images need positive finite cutoffs and finite nonzero real warps")
    assignments = []
    for length, profile in zip(edge_radii, profiles, strict=True):
        scale, radius = length.as_coeff_Mul()
        assignments.append((radii.index(radius), sp.Rational(profile["cutoff"])/scale,
                            sp.Rational(profile["lambda"])))
    effective = {}
    for feature in features:
        for _, factors in feature:
            if len(factors) != len(edge_radii) or any(type(n) is not int or n < 1 for ns in factors for n in ns):
                raise ValueError("Graph radial factors require one n>=1 index tuple per oriented edge")
            for indices, (i, cutoff, warp) in zip(factors, assignments, strict=True):
                if not indices:
                    continue
                previous = effective.setdefault(i, {"cutoff": cutoff, "warp": None})
                if previous["cutoff"] != cutoff:
                    raise NotImplementedError("Profiles sharing a radius need the same effective cutoff; support-cell resolution remains required")
                if any(n > 1 for n in indices):
                    if previous["warp"] is not None and previous["warp"] != warp:
                        raise NotImplementedError("Profiles sharing a radius need one nonconstant warp; common-rate resolution remains required")
                    previous["warp"] = warp
    symbols = {i: (sp.Symbol(f"envelope_{i}"), sp.Symbol(f"warp_{i}")) for i in sorted(effective)}
    radial_symbols = tuple(symbol for i, pair in symbols.items()
                           for symbol in (pair if effective[i]["warp"] is not None else pair[:1]))
    indices = {symbol: i for i, symbol in enumerate(radial_symbols)}
    expressions = []
    for feature in features:
        terms = []
        for angular, factors in feature:
            choices = [chebexp_cos_radial_product(tuple(indices)) if indices else ((0, 0, 1),)
                       for indices in factors]
            for combination in product(*choices):
                powers, weight = [0]*len(radial_symbols), sp.S.One
                for (h, p, coefficient), (i, _, _) in zip(combination, assignments, strict=True):
                    weight *= sp.Rational(coefficient)
                    if h or p:
                        envelope, warp = symbols[i]
                        powers[indices[envelope]] += h
                        if p:
                            powers[indices[warp]] += p
                terms.append((tuple(powers), weight, sp.sympify(angular)))
        expressions.append(terms)
    report = _graph_radial_rows(expressions, coordinates, _component_terms=True,
                                polynomial_generators=polynomial_generators)
    report.update(coordinate_convention="linear_graph_chebexp_cos_radial_laurent_v1",
                  radial_symbols=radial_symbols, radial_generators=tuple(
                      (i, effective[i]["cutoff"], effective[i]["warp"]) for i in symbols))
    return report


def _graph_radial_derivative(expression, variable, coordinates):
    """Differentiate on the physical vertex domain, including every radius.

    The result is another finite Laurent-radius expression consumed by the
    same row operation. The derivative follows dr_e=dq_e/(2*r_e) and therefore
    descends through cycle and squared-radius relations, including repeated
    and reversed edge uses. Edge values are differentiated through vertices.
    """
    vertices, _, _, radii, substitutions, squared = coordinates
    if variable not in vertices:
        raise ValueError("Graph derivative variable must be a configured independent vertex coordinate")
    expression = sp.sympify(expression).xreplace(dict(substitutions))
    return sp.diff(expression, variable) + sum(
        sp.diff(expression, radius)*sp.diff(q, variable)/(2*radius)
        for radius, q in zip(radii, squared, strict=True))


@lru_cache(maxsize=32)
def _racah_solid_polynomials(ell):
    """Exact physical harmonics in the existing Condon--Shortley convention.

    Returns r**ell * sqrt(4*pi/(2*ell+1)) * Y_ell,m, m=-ell,...,ell.
    DLMF14.30.1 and the Legendre derivative identity used by core.spherical
    give homogeneous Cartesian polynomials. This is source-function evaluation,
    not construction of rotation/permutation coupling coefficients.
    """
    if type(ell) is not int or ell < 0:
        raise ValueError("Solid harmonics require a nonnegative integer angular degree")
    x, y, z = sp.symbols("harmonic_x harmonic_y harmonic_z", real=True)
    q = x*x+y*y+z*z
    values = [sp.S.Zero]*(2*ell+1)
    for m in range(ell+1):
        coefficients = sp.Poly(sp.diff(sp.legendre(ell, z), z, m), z).terms()
        polynomial = sum(c*z**power[0]*q**((ell-m-power[0])//2) for power, c in coefficients)
        value = (-1)**m*sp.sqrt(sp.Rational(sp.factorial(ell-m), sp.factorial(ell+m)))*(x+sp.I*y)**m*polynomial
        values[ell+m] = value
        if m:
            values[ell-m] = (-1)**m*sp.conjugate(value)
    return (x, y, z), tuple(values)


def _graph_angular_multiplets(input_Ls, columns, factor_edges, coordinates, *, phase_exponent=0,
                             factor_values=None):
    """Evaluate selected canonical CG-path columns on oriented graph edges.

    columns[copy][tableau] contains existing (path, exact coefficient) pairs.
    Prefixes are contracted once and shared across selected paths. At no point
    is the tensor product of all magnetic indices materialized. Results retain
    every output magnetic component, in complex harmonic coordinates; the
    caller's existing real-form phase is applied without changing its gauge.
    Supplied factor_values use the same contraction for complete physical
    multiplets (including background moments and their affine corrections).
    """
    from ye3t.core.cg import cg_exact_integer
    _, edges, radii, _, _, _ = coordinates
    spins = tuple(input_Ls)
    if (not spins or any(type(ell) is not int or ell < 0 for ell in spins) or
            type(phase_exponent) is not int or phase_exponent not in {0, 1}):
        raise ValueError("Selected graph paths need complete angular/edge assignments and the canonical real-form phase")
    factors, source_values = [], {}
    if factor_values is None:
        factor_edges = tuple(factor_edges)
        if (len(spins) != len(factor_edges) or any(type(edge) is not int or edge < 0 or edge >= len(edges)
                                                for edge in factor_edges)):
            raise ValueError("Selected graph paths need one valid edge per angular input")
        for ell, edge in zip(spins, factor_edges, strict=True):
            key = ell, edge
            if key not in source_values:
                variables, polynomials = _racah_solid_polynomials(ell)
                replacement = dict(zip(variables, edges[edge], strict=True))
                source_values[key] = tuple(value.xreplace(replacement)/radii[edge]**ell for value in polynomials)
            factors.append(source_values[key])
    else:
        if factor_edges is not None:
            raise ValueError("Supply physical factor values or edge assignments, not both")
        if len(factor_values) != len(spins) or any(len(row) != 2*ell+1 for row, ell in zip(factor_values, spins, strict=True)):
            raise ValueError("Physical factor values must retain every magnetic component")
        factors = [tuple(sp.sympify(value) for value in row) for row in factor_values]
    prefixes = {}
    def evaluate(path):
        if path in prefixes:
            return prefixes[path]
        if len(path) == 1:
            result = factors[0]
        else:
            previous = evaluate(path[:-1])
            a, b, L = path[-2], spins[len(path)-1], path[-1]
            result = []
            for M in range(-L, L+1):
                terms = []
                for m in range(max(-a, M-b), min(a, M+b)+1):
                    n = M-m
                    coefficient = cg_exact_integer(a, m, b, n, L, M)
                    if coefficient:
                        terms.append(coefficient._sympy_()*previous[m+a]*factors[len(path)-1][n+b])
                result.append(sp.Add(*terms))
            result = tuple(result)
        prefixes[path] = result
        return result
    output, target_L, tableaux = [], None, None
    for copy in columns:
        if tableaux is None:
            tableaux = len(copy)
        if not copy or len(copy) != tableaux:
            raise ValueError("Selected graph angular copies need complete matching tableaux")
        rows = []
        for column in copy:
            terms = []
            for path, coefficient in (column.items() if isinstance(column, dict) else column):
                path = tuple(path)
                if (len(path) != len(spins) or path[0] != spins[0] or
                        any(type(L) is not int or L < 0 for L in path) or
                        any(not abs(a-b) <= L <= a+b for a, b, L in zip(path[:-1], spins[1:], path[1:], strict=True))):
                    raise ValueError("Selected graph angular columns require valid canonical CG paths")
                if target_L is None:
                    target_L = path[-1]
                if target_L != path[-1]:
                    raise ValueError("Selected graph angular columns must share one output L")
                value = coefficient._sympy_() if hasattr(coefficient, "_sympy_") else sp.sympify(coefficient)
                if value.free_symbols or value.atoms(sp.Float):
                    raise ValueError("Selected graph angular maps need exact parameter-independent coefficients")
                terms.append(tuple(value*entry for entry in evaluate(path)))
            if not terms:
                raise ValueError("Selected graph angular columns must not omit a complete tableau")
            rows.append(tuple((-sp.I)**phase_exponent*sp.Add(*(term[M] for term in terms))
                              for M in range(2*target_L+1)))
        output.append(tuple(rows))
    # The recursive function otherwise holds a cycle retaining its expressions
    # until cyclic GC; no expanded function tables belong in a persistent cache.
    evaluate = None
    return tuple(output)
