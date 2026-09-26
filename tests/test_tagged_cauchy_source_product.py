from ye3t.couplings.lifted_cauchy_scalar import _exact_scalar_from_payload
from ye3t.couplings.tagged_cauchy_image import (
    RACAH_HARMONIC_PRODUCT_SCHEMA,
    racah_harmonic_product_plan,
)


def _sympy():
    from ye3t._optional_sympy import sp

    return sp


def _coefficient(output, left_m, right_m):
    for term in output["terms"]:
        if (
            int(term["left_m"]) == int(left_m)
            and int(term["right_m"]) == int(right_m)
        ):
            return _exact_scalar_from_payload(term["coefficient"])
    return _sympy().Integer(0)


def test_racah_product_plan_has_exact_scalar_and_l1_identities():
    sp = _sympy()
    plan = racah_harmonic_product_plan(
        (0, 1), maximum_collision_arity=3
    )
    assert plan["schema"] == RACAH_HARMONIC_PRODUCT_SCHEMA
    assert plan["certificate"]["passed"] is True
    assert plan["certificate"]["checks"]["C00_identity_exact"] is True
    assert plan["certificate"]["checks"]["C10_squared_identity_exact"] is True

    pairs = {
        (int(pair["left_l"]), int(pair["right_l"])): pair
        for pair in plan["pairs"]
    }
    scalar_times_l1 = pairs[(0, 1)]["outputs"]
    assert len(scalar_times_l1) == 1
    assert int(scalar_times_l1[0]["L"]) == 1
    for magnetic in (-1, 0, 1):
        assert sp.simplify(_coefficient(scalar_times_l1[0], 0, magnetic) - 1) == 0

    square = {
        int(output["L"]): output for output in pairs[(1, 1)]["outputs"]
    }
    assert set(square) == {0, 2}
    assert sp.simplify(_coefficient(square[0], 0, 0) - sp.Rational(1, 3)) == 0
    assert sp.simplify(_coefficient(square[2], 0, 0) - sp.Rational(2, 3)) == 0


def test_racah_product_plan_is_exactly_commutative_and_closes_to_arity_three():
    sp = _sympy()
    plan = racah_harmonic_product_plan(
        (0, 1, 2), maximum_collision_arity=3
    )
    pairs = {
        (int(pair["left_l"]), int(pair["right_l"])): pair
        for pair in plan["pairs"]
    }
    for left_l, right_l in ((0, 2), (1, 2), (2, 1)):
        forward = {
            int(output["L"]): output
            for output in pairs[(left_l, right_l)]["outputs"]
        }
        if (right_l, left_l) in pairs:
            reverse = {
                int(output["L"]): output
                for output in pairs[(right_l, left_l)]["outputs"]
            }
        else:
            continue
        assert set(forward) == set(reverse)
        for output_l in forward:
            for left_m in range(-left_l, left_l + 1):
                for right_m in range(-right_l, right_l + 1):
                    assert sp.simplify(
                        _coefficient(forward[output_l], left_m, right_m)
                        - _coefficient(reverse[output_l], right_m, left_m)
                    ) == 0
    reachable = {
        int(record["collision_arity"]): tuple(record["angular_degrees"])
        for record in plan["reachable_angular_degrees"]
    }
    assert reachable[1] == (0, 1, 2)
    assert reachable[2] == (0, 1, 2, 3, 4)
    assert reachable[3] == (0, 1, 2, 3, 4, 5, 6)
