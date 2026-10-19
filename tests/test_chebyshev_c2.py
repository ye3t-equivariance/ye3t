"""Independent polynomial and exact source-image checks for the C2 family."""
import numpy as np
import pytest
import sympy as sp

from ye3t.couplings.chebyshev_c2 import (
    CHEBYSHEV_C2_SOURCE_FAMILY as FAMILY,
    chebyshev_c2_radial_product,
    shifted_chebyshev_ladder_with_derivative,
    shifted_chebyshev_power_coefficients,
)
from ye3t.couplings import compile, tagged_cauchy_carriers_request
from ye3t.couplings.tagged_cauchy_carriers import (
    tagged_cauchy_carrier_schedule, tagged_cauchy_carrier_physical_image_plan,
)
from ye3t.couplings.tagged_cauchy_general import _physical_source_product


@pytest.mark.parametrize("degree", [0, 1, 2, 5, 12])
def test_exact_chebyshev_coefficients_and_runtime_derivative(degree):
    x = sp.Symbol("x")
    polynomial = sum(value*x**power for power, value in enumerate(shifted_chebyshev_power_coefficients(degree)))
    assert sp.expand(polynomial - sp.chebyshevt(degree, 2*x-1)) == 0
    points = np.array([0., 1e-9, .17, .53, .93, 1-1e-9, 1.])
    coefficients = [0]*degree + [1]
    values, derivatives = shifted_chebyshev_ladder_with_derivative(degree, points)
    np.testing.assert_allclose(values[-1], np.polynomial.chebyshev.chebval(2*points-1, coefficients), atol=2e-13)
    expected = 2*np.polynomial.chebyshev.chebval(2*points-1, np.polynomial.chebyshev.chebder(coefficients))
    np.testing.assert_allclose(derivatives[-1], expected, atol=3e-12)


@pytest.mark.parametrize("factors,output_l", [(((0,0),(0,0)),0), (((2,1),(3,2)),1),
    (((2,1),(1,2),(3,3)),2), (((4,2),(5,3)),5)])
def test_untruncated_collision_product_exact_and_two_derivatives(factors, output_l):
    x = sp.Symbol("x")
    source = lambda q,l: x**l*(1-x)**3*sp.chebyshevt(q, 2*x-1)
    direct = sp.prod(source(q,l) for q,l in factors)
    expansion = chebyshev_c2_radial_product(factors, output_l)
    lowered = sum(sp.Rational(c.numerator,c.denominator)*source(q,output_l) for q,c in expansion)
    assert sp.expand(direct-lowered) == 0
    expected_degree = sum(q+l for q,l in factors)-output_l+3*(len(factors)-1)
    assert max(q for q,_ in expansion) == expected_degree
    assert expansion == chebyshev_c2_radial_product(tuple(reversed(factors)), output_l)
    for order in range(3):
        assert sp.diff(direct,x,order).subs(x,1) == 0
        assert sp.diff(lowered,x,order).subs(x,1) == 0
    # Independent exact-rational points also cover cancellations near cutoff.
    for point in (sp.Rational(1,7), sp.Rational(9,10), sp.Rational(9999,10000)):
        assert direct.subs(x,point) == lowered.subs(x,point)


def test_tagged_chebyshev_image_uses_exact_collision_algebra_and_preserves_sectors():
    channel = {"neighbor_species":"Ni", "radial_channel":1,"l":1,"source_family_id":FAMILY}
    payload = compile(tagged_cauchy_carriers_request([channel],[2],tag_count=2,target_Ls=[0,1,2]))
    assert payload["multiplet_count"] == 3
    labels = [record["label"] for record in payload["descriptors"]]
    assert {(row["target_L"],row["target_parity"],row["tag_character"]) for row in labels} == {(0,1,1),(1,1,-1),(2,1,1)}
    schedule = tagged_cauchy_carrier_schedule([payload],support_realization="edge_marginal")
    assert len(schedule["inventory"]) == 3
    proof = schedule["marginal_image_certificate"]
    assert proof["exact_all_M_reconstruction"]
    assert proof["physical_collision_lowering"] == "exact_species_chebyshev_c2_racah_real_tesseral"
    assert schedule["radial_source"]["zero_extension"] == "C2"
    pooled = tagged_cauchy_carrier_physical_image_plan([payload])
    assert len(pooled["selected_coordinate_ids"]) == 2
    assert pooled["complete_multiplet_reconstruction"]
    jacobi = dict(channel, source_family_id="orthogonal_shifted_jacobi_origin_regular_v1")
    old = compile(tagged_cauchy_carriers_request([jacobi],[2],tag_count=2,target_Ls=[0,1,2]))
    assert old["self_hash"] != payload["self_hash"]
    with pytest.raises(ValueError, match="one supported radial family"):
        tagged_cauchy_carrier_physical_image_plan([old,payload])


def test_cross_species_zero_and_unsupported_radial_product_rejected():
    left = ("Ni",0,1,FAMILY,0)
    assert _physical_source_product((left,("Cu",0,1,FAMILY,2))) == {}
    with pytest.raises(ValueError, match="one supported radial family"):
        _physical_source_product((left,("Ni",0,1,"orthogonal_shifted_jacobi_origin_regular_v1",2)))
