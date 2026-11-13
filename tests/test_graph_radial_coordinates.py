"""Exact graph-image equations from tensor_product_math.md 54a--55 and 60."""
import pytest
import sympy as sp
import torch

from ye3t.couplings.graph_radial_coordinates import (
    _free_graph_coordinates, _linear_graph_coordinates, _linear_graph_coordinates_cached,
    _graph_radial_rows, _graph_radial_derivative, _chebexp_cos_graph_rows)


def _expression(row, report, coordinates):
    vertices, _, _, radii, _, squared = coordinates
    return sum(value*sp.prod(r**b for r, b in zip(radii, binary, strict=True))*
               sp.prod(x**p for x, p in zip(vertices, monomial, strict=True))/sp.prod(
                   q**h for q, h in zip(squared, report["denominator_powers"][binary, radial], strict=True))
               for (binary, radial, monomial), value in row.items())


def test_cycle_aliases_and_radical_cancellation_have_exact_zero_rows():
    coordinates = _free_graph_coordinates(3, ((0, 1), (1, 2), (0, 2), (2, 0)))
    _, edge, edge_radii, radii, _, _ = coordinates
    assert edge_radii[2] == edge_radii[3]
    assert len(radii) == 3
    cycles = [edge[0][a]+edge[1][a]-edge[2][a] for a in range(3)]
    reversal = [edge[2][a]+edge[3][a] for a in range(3)]
    cosine_identity = (edge_radii[2]**2-edge_radii[0]**2-edge_radii[1]**2-
                       2*sum(edge[0][a]*edge[1][a] for a in range(3)))
    report = _graph_radial_rows(cycles+reversal+[cosine_identity], coordinates)
    assert report["rows"] == [{}]*7
    assert report["coordinate_count"] == 0


def test_mixed_denominators_reconstruct_and_do_not_create_extra_coordinates():
    coordinates = _free_graph_coordinates(3, ((0, 1), (1, 2), (0, 2)))
    vertices, edge, _, radii, _, squared = coordinates
    r, s, t = coordinates[2]
    x = vertices[0]
    expressions = [x/r, x*sum(e**2 for e in edge[0])/r**3]
    expressions += [r*s/t**3, r**-3+2*r**-1, sum(e**2 for e in edge[0])/r**3]
    report = _graph_radial_rows(expressions, coordinates)
    assert report["rows"][0] == report["rows"][1]
    for expression, row in zip(expressions, report["rows"], strict=True):
        rebuilt = _expression(row, report, coordinates)
        replacements = dict(zip(radii, [sp.sqrt(q) for q in squared], strict=True))
        difference = (rebuilt-expression.xreplace(dict(coordinates[4]))).subs(replacements)
        assert sp.simplify(difference) == 0
    assert report["radical_component_count"] < 2**len(radii)


@pytest.mark.parametrize("order", [1, 2])
def test_derivatives_preserve_cycle_and_radius_ideals(order):
    coordinates = _free_graph_coordinates(3, ((0, 1), (1, 2), (0, 2)))
    vertices, edge, edge_radii, _, _, _ = coordinates
    identities = [edge[0][0]+edge[1][0]-edge[2][0],
                  edge_radii[2]**2-sum(e**2 for e in edge[2])]
    for identity in identities:
        value = identity/edge_radii[0]**3
        for variable in vertices[:order]:
            value = _graph_radial_derivative(value, variable, coordinates)
        assert _graph_radial_rows([value], coordinates)["rows"] == [{}]


def test_derivative_matches_positive_branch_and_product_rule():
    coordinates = _free_graph_coordinates(3, ((0, 1), (1, 2), (0, 2)))
    vertices, edge, edge_radii, radii, substitutions, squared = coordinates
    left = edge[1][0]/edge_radii[0]**3
    right = edge[2][1]*edge_radii[1]/edge_radii[2]
    variable = vertices[0]
    derivative = _graph_radial_derivative(left*right, variable, coordinates)
    product_rule = (_graph_radial_derivative(left, variable, coordinates)*right+
                    left*_graph_radial_derivative(right, variable, coordinates))
    report = _graph_radial_rows([derivative, product_rule], coordinates)
    assert report["rows"][0] == report["rows"][1]
    positive = dict(zip(radii, [sp.sqrt(q) for q in squared], strict=True))
    reference = sp.diff((left*right).xreplace(dict(substitutions)).subs(positive), variable)
    assert sp.simplify(derivative.subs(positive)-reference) == 0


def test_nonzero_binary_components_stay_distinct():
    coordinates = _free_graph_coordinates(3, ((0, 1), (1, 2), (0, 2)))
    radii = coordinates[3]
    report = _graph_radial_rows([sp.prod(radii), radii[0], radii[1], radii[2], 1], coordinates)
    assert all(len(row) == 1 for row in report["rows"])
    assert report["coordinate_count"] == report["radical_component_count"] == 5


@pytest.mark.parametrize("bad", ["float", "unknown", "fractional_radius", "radius_sum", "vertex_inverse", "analytic"])
def test_unsupported_function_domains_fail(bad):
    coordinates = _free_graph_coordinates(2, ((0, 1),))
    x, r = coordinates[0][0], coordinates[3][0]
    expression = {"float": sp.Float(.5)*x, "unknown": sp.Symbol("outside"),
                  "fractional_radius": sp.sqrt(r), "radius_sum": 1/(1+r),
                  "vertex_inverse": 1/x, "analytic": sp.exp(x)}[bad]
    with pytest.raises(ValueError):
        _graph_radial_rows([expression], coordinates)


@pytest.mark.parametrize("vertices,edges", [(1, ((0, 0),)), (3, ((0, 1),)),
    (2, ((0, 2),)), (2, ((0, 0),)), (2, ((False, 1),))])
def test_invalid_graphs_fail(vertices, edges):
    with pytest.raises(ValueError):
        _free_graph_coordinates(vertices, edges)


def test_coordinate_cache_is_immutable_and_bounded():
    coordinates = _free_graph_coordinates(3, ((0, 1), (1, 2), (0, 2)))
    assert coordinates is _free_graph_coordinates(3, ((0, 1), (1, 2), (0, 2)))
    assert isinstance(coordinates, tuple) and all(isinstance(item, tuple) for item in coordinates)
    assert _linear_graph_coordinates_cached.cache_info().maxsize == 32


def test_periodic_linear_forms_resolve_proportional_and_reversed_radii():
    # Two position columns and one variable lattice column. The first two
    # edges end at different images of one atom. The last pair are self-images.
    coordinates = _linear_graph_coordinates(((1, 0, 1), (1, 0, -1), (0, 1, 0),
        (-1, 1, -1), (0, 0, 1), (0, 0, -2)))
    _, edge, lengths, radii, _, _ = coordinates
    assert lengths[-1] == 2*lengths[-2]
    assert len(radii) == 5
    identities = [edge[0][a]+edge[3][a]-edge[2][a] for a in range(3)]
    identities += [edge[0][a]-edge[1][a]-2*edge[4][a] for a in range(3)]
    identities += [lengths[-1]**2-4*lengths[-2]**2]
    assert _graph_radial_rows(identities, coordinates)["rows"] == [{}]*7


@pytest.mark.parametrize("bad", [(), ((0, 0),), ((True,),), ((1, 0),), ((1,), (1, 2)), ((1.0,),)])
def test_invalid_linear_bindings_fail_before_cache(bad):
    _linear_graph_coordinates(((1,),))
    with pytest.raises(ValueError):
        _linear_graph_coordinates(bad)


def test_chebexp_graph_reuses_radial_algebra_and_resolves_scaled_edge_aliases():
    from ye3t.couplings.tagged_cauchy_general import _exact_pivot_image
    coordinates = _linear_graph_coordinates(((1,), (-2,)))
    profiles = [dict(family="chebexp_cos", cutoff=4.6, **{"lambda": 1.5}),
                dict(family="chebexp_cos", cutoff=9.2, **{"lambda": 1.5})]
    features = [[(1, ((4,), (1,)))], [(1, ((2,), (1,)))], [(1, ((2,), (2,)))]]
    report = _chebexp_cos_graph_rows(features, coordinates, profiles)
    first, second, third = report["rows"]
    keys = first.keys() | second.keys() | third.keys()
    assert all(first.get(key, 0) == 4*second.get(key, 0)-4*third.get(key, 0) for key in keys)
    assert len(report["radial_generators"]) == 1
    assert len(report["radial_symbols"]) == 2
    selected, reconstruction = _exact_pivot_image(report["rows"])
    assert selected == (0, 1)
    assert reconstruction[2] == {0: -sp.Rational(1, 4), 1: 1}


def test_radial_components_keep_their_own_bounded_denominators():
    coordinates = _linear_graph_coordinates(((1, 0), (0, 1)))
    profiles = [dict(family="chebexp_cos", cutoff=4.6, **{"lambda": 1.5})]*2
    r, s = coordinates[2]
    features = [[(r**-8, ((1,)*8, ()))], [(s**-8, ((), (1,)*8))]]
    report = _chebexp_cos_graph_rows(features, coordinates, profiles)
    assert report["coordinate_count"] == 2
    assert sorted(report["denominator_powers"].values()) == [(0, 4), (4, 0)]


def test_rank_eight_factored_radials_match_expanded_polynomial_reference():
    coordinates = _free_graph_coordinates(4, ((0, 2), (0, 3), (1, 2), (1, 3)))
    _, edge, radius, radii, _, _ = coordinates
    left = sum(edge[0][a]*edge[1][a] for a in range(3))/(radius[0]*radius[1])
    right = sum(edge[2][a]*edge[3][a] for a in range(3))/(radius[2]*radius[3])
    profiles = [dict(family="chebexp_cos", cutoff=4.6, **{"lambda": 1.5})]*4
    factors = ((1, 2),)*4
    actual = _chebexp_cos_graph_rows([[(left**2*right**2, factors)]], coordinates, profiles)
    # Explicit bounded reference expands the textbook Chebyshev polynomials;
    # it does not call chebexp_cos_radial_product or the packed radial assembly.
    expression = sp.S.One
    for length in radius:
        i = radii.index(length)
        h, x = sp.Symbol(f"envelope_{i}"), sp.Symbol(f"warp_{i}")
        expression *= h*h*(1-sp.chebyshevt(2, x))/2
    terms = [(powers, coefficient, left**2*right**2) for powers, coefficient in
             sp.Poly(expression, *actual["radial_symbols"]).terms()]
    expected = _graph_radial_rows([terms], coordinates, _component_terms=True)
    assert actual["rows"] == expected["rows"]
    assert actual["denominator_powers"] == expected["denominator_powers"]


def test_relabelled_free_vertices_preserve_polynomial_and_radius_relations():
    coordinates = _free_graph_coordinates(3, ((0, 1), (1, 2), (0, 2)))
    vertices, edge, lengths, radii, _, squared = coordinates
    expression = sum(edge[0][a]*edge[2][a] for a in range(3))/(lengths[0]*lengths[2])
    report = _graph_radial_rows([expression], coordinates)
    swap = {vertices[a]: vertices[a+3] for a in range(3)}
    swap.update({vertices[a+3]: vertices[a] for a in range(3)})
    positive = dict(zip(radii, [sp.sqrt(q) for q in squared], strict=True))
    reconstructed = _expression(report["rows"][0], report, coordinates).subs(positive)
    assert sp.simplify(reconstructed.xreplace(swap)-reconstructed) == 0


def test_envelope_alias_is_independent_of_unused_warp():
    coordinates = _linear_graph_coordinates(((1,), (-1,)))
    profiles = [dict(family="chebexp_cos", cutoff=4.6, **{"lambda": warp}) for warp in (1.5, 2.5)]
    report = _chebexp_cos_graph_rows([[(1, ((1,), ()))], [(1, ((), (1,)))]], coordinates, profiles)
    assert report["rows"][0] == report["rows"][1]
    assert len(report["radial_symbols"]) == 1
    assert report["radial_generators"][0][2] is None


@pytest.mark.parametrize("change", ["cutoff", "lambda"])
def test_unresolved_shared_radius_profiles_are_not_admitted(change):
    coordinates = _linear_graph_coordinates(((1,), (-1,)))
    profiles = [dict(family="chebexp_cos", cutoff=4.6, **{"lambda": 1.5}) for _ in range(2)]
    profiles[1][change] += .5
    with pytest.raises(NotImplementedError, match="sharing a radius"):
        _chebexp_cos_graph_rows([[(1, ((2,), (2,)))]], coordinates, profiles)


def test_radial_multipliers_preserve_all_cycle_cancellations():
    coordinates = _free_graph_coordinates(3, ((0, 1), (1, 2), (0, 2)))
    edge = coordinates[1]
    profiles = [dict(family="chebexp_cos", cutoff=4.6, **{"lambda": 1.5})]*3
    radial = ((2,), (3,), (4,))
    feature = [(edge[0][0], radial), (edge[1][0], radial), (-edge[2][0], radial)]
    assert _chebexp_cos_graph_rows([feature], coordinates, profiles)["rows"] == [{}]


def test_selected_cg_path_uses_existing_coefficients_on_cyclic_geometry():
    from ye3t.core.cg import cg_exact_integer
    coordinates = _free_graph_coordinates(3, ((0, 1), (1, 2), (0, 2)))
    _, edge, radius, _, _, _ = coordinates
    # Exact Racah l=1 physical values in the existing Condon--Shortley gauge.
    harmonics = [[(x-sp.I*y)/(sp.sqrt(2)*r), z/r, -(x+sp.I*y)/(sp.sqrt(2)*r)]
                 for (x, y, z), r in zip(edge, radius, strict=True)]
    contraction = sum(cg_exact_integer(1, m, 1, -m, 0, 0)._sympy_()*harmonics[0][m+1]*harmonics[1][1-m]
                      for m in range(-1, 2))
    cosine = -sum(edge[0][a]*edge[1][a] for a in range(3))/(sp.sqrt(3)*radius[0]*radius[1])
    law_of_cosines = -(radius[2]**2-radius[0]**2-radius[1]**2)/(2*sp.sqrt(3)*radius[0]*radius[1])
    report = _graph_radial_rows([contraction, cosine, law_of_cosines], coordinates)
    assert report["rows"][0] == report["rows"][1] == report["rows"][2]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_coordinate_image_preserves_forces_and_force_loss_gradients(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    coordinates = _linear_graph_coordinates(((1, 0), (0, 1), (-1, 1)))
    _, edge, radius, _, _, squared = coordinates
    expressions = [sum(edge[0][a]*edge[2][a] for a in range(3))/(radius[0]*radius[2]),
                   radius[1]/radius[0]**3, edge[2][0]/radius[2]]
    report = _graph_radial_rows(expressions, coordinates)
    x = torch.tensor([[.7, .2, -.3], [-.2, .8, .5]], dtype=torch.float64, device=device, requires_grad=True)
    vectors = [x[0], x[1], x[1]-x[0]]
    lengths = [torch.linalg.vector_norm(v) for v in vectors]
    direct = torch.stack([(vectors[0]*vectors[2]).sum()/(lengths[0]*lengths[2]),
                          lengths[1]/lengths[0]**3, vectors[2][0]/lengths[2]])
    # Evaluate only requested sparse physical coordinates, with the actual
    # canonical radius ordering supplied by the core operation.
    replacements = dict(zip(coordinates[0], x.flatten(), strict=True))
    q_values = []
    for q in squared:
        terms = []
        for powers, c in sp.Poly(q, *coordinates[0]).terms():
            factors = [replacements[v]**p for v, p in zip(coordinates[0], powers, strict=True)]
            terms.append(float(c)*torch.prod(torch.stack(factors)))
        q_values.append(sum(terms))
    canonical_radii = torch.sqrt(torch.stack(q_values))
    denominators = {key: torch.prod(torch.stack([q**h for q, h in zip(q_values, powers, strict=True)]))
                    for key, powers in report["denominator_powers"].items()}
    converted = []
    for row in report["rows"]:
        converted.append(sum(float(coefficient)*torch.prod(canonical_radii**torch.tensor(binary, device=device))*
            torch.prod(x.flatten()**torch.tensor(monomial, device=device))/denominators[binary, radial]
            for (binary, radial, monomial), coefficient in row.items()))
    converted = torch.stack(converted)
    weights = torch.tensor([.7, -.2, .4], dtype=x.dtype, device=device, requires_grad=True)
    def derivatives(features):
        energy = weights@features
        force = -torch.autograd.grad(energy, x, create_graph=True, retain_graph=True)[0]
        gradient = torch.autograd.grad(force.square().sum(), weights, retain_graph=True)[0]
        return energy, force, gradient
    for actual, reference in zip(derivatives(converted), derivatives(direct), strict=True):
        torch.testing.assert_close(actual, reference, rtol=2e-11, atol=2e-11)
