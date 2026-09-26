"""Exact algebra for the origin-regular shifted-Jacobi source family.

This module owns the logical radial polynomials of the tagged-Cauchy source
family and their exact same-neighbor multiplication record.  A user supplies
the source keys of their own radial functions (neighbor species, Jacobi degree
``q``, angular degree ``l``); the module returns the hash-bound product-algebra
record that ``tagged_cauchy_image_request`` binds to the Racah angular product
plan.  Geometry, neighbor lists, angular coupling coefficients, and descriptor
enumeration remain elsewhere.
"""

import hashlib
import json
import math
from fractions import Fraction


ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY = (
    "orthogonal_shifted_jacobi_origin_regular_v1"
)
ORTHOGONAL_SHIFTED_JACOBI_PRODUCT_SCHEMA = (
    "ye3t_orthogonal_shifted_jacobi_product_algebra_v1"
)


def _fraction_payload(value):
    value = Fraction(value)
    return {
        "numerator": int(value.numerator),
        "denominator": int(value.denominator),
    }


def _fraction_from_payload(payload):
    return Fraction(int(payload["numerator"]), int(payload["denominator"]))


def _stable_hash(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def shifted_jacobi_power_coefficients(degree, alpha, beta):
    """Coefficients of ``P_degree^(alpha,beta)(2*x-1)``, low power first."""

    degree = int(degree)
    alpha = int(alpha)
    beta = int(beta)
    if degree < 0 or alpha < 0 or beta < 0:
        raise ValueError("Jacobi degree and integer parameters must be nonnegative.")
    result = [0] * (degree + 1)
    for left_power in range(degree + 1):
        scale = math.comb(degree + alpha, left_power) * math.comb(
            degree + beta, degree - left_power
        )
        remainder = degree - left_power
        for shifted_power in range(remainder + 1):
            power = left_power + shifted_power
            result[power] += (
                scale
                * math.comb(remainder, shifted_power)
                * (-1) ** (remainder - shifted_power)
            )
    return tuple(int(value) for value in result)


def shifted_jacobi_ladder_with_derivative(maximum_degree, alpha, beta, coordinate):
    """Evaluate shifted Jacobi values and ``d/dx`` by a stable recurrence.

    The logical polynomial is still
    ``P_q^(alpha,beta)(2*x-1)``.  This routine changes only its floating-point
    realization: it avoids evaluating the increasingly ill-conditioned
    expanded power coefficients used for exact compiler provenance.
    ``coordinate`` may be a scalar or an array/tensor supporting ordinary
    arithmetic.
    """

    maximum_degree = int(maximum_degree)
    alpha = int(alpha)
    beta = int(beta)
    if maximum_degree < 0 or alpha < 0 or beta < 0:
        raise ValueError(
            "Jacobi degree and integer parameters must be nonnegative."
        )

    one = coordinate * 0.0 + 1.0
    zero = coordinate * 0.0
    values = [one]
    derivatives = [zero]
    if maximum_degree == 0:
        return tuple(values), tuple(derivatives)

    first_scale = float(alpha + beta + 2)
    values.append(first_scale * coordinate - float(beta + 1))
    derivatives.append(zero + first_scale)
    shifted = 2.0 * coordinate - 1.0
    for degree in range(1, maximum_degree):
        total = 2 * degree + alpha + beta
        a_n = (
            (total + 1) * (total + 2)
            / (2.0 * (degree + 1) * (degree + alpha + beta + 1))
        )
        b_n = (
            (alpha * alpha - beta * beta) * (total + 1)
            / (
                2.0
                * (degree + 1)
                * (degree + alpha + beta + 1)
                * total
            )
        )
        c_n = (
            (degree + alpha)
            * (degree + beta)
            * (total + 2)
            / (
                (degree + 1)
                * (degree + alpha + beta + 1)
                * total
            )
        )
        multiplier = a_n * shifted + b_n
        values.append(multiplier * values[-1] - c_n * values[-2])
        derivatives.append(
            2.0 * a_n * values[-2]
            + multiplier * derivatives[-1]
            - c_n * derivatives[-2]
        )
    return tuple(values), tuple(derivatives)


def shifted_jacobi_normalization_squared(q, angular_l):
    """Exact reciprocal norm for the declared ``x^2 dx`` source metric."""

    q = int(q)
    angular_l = int(angular_l)
    if q < 0 or angular_l < 0:
        raise ValueError("q and angular_l must be nonnegative.")
    return Fraction(
        (2 * q + 2 * angular_l + 7)
        * math.factorial(q)
        * math.factorial(q + 2 * angular_l + 6),
        math.factorial(q + 4) * math.factorial(q + 2 * angular_l + 2),
    )


def _trim_polynomial(coefficients):
    coefficients = [Fraction(value) for value in coefficients]
    while len(coefficients) > 1 and coefficients[-1] == 0:
        coefficients.pop()
    return tuple(coefficients or (Fraction(0),))


def _multiply_polynomials(left, right):
    result = [Fraction(0)] * (len(left) + len(right) - 1)
    for left_power, left_coefficient in enumerate(left):
        for right_power, right_coefficient in enumerate(right):
            result[left_power + right_power] += Fraction(
                left_coefficient
            ) * Fraction(right_coefficient)
    return _trim_polynomial(result)


def _differentiate_polynomial(coefficients):
    if len(coefficients) <= 1:
        return (Fraction(0),)
    return _trim_polynomial(
        tuple(
            Fraction(power) * Fraction(coefficients[power])
            for power in range(1, len(coefficients))
        )
    )


def _shift_polynomial(coefficients, power):
    power = int(power)
    if power < 0:
        raise ValueError("Polynomial shifts must be nonnegative.")
    return tuple(Fraction(0) for _ in range(power)) + tuple(
        Fraction(value) for value in coefficients
    )


def _one_minus_x_power(power):
    power = int(power)
    if power < 0:
        raise ValueError("Envelope powers must be nonnegative.")
    return tuple(
        Fraction(((-1) ** index) * math.comb(power, index))
        for index in range(power + 1)
    )


def shifted_jacobi_expansion(power_coefficients, angular_l):
    """Expand one rational polynomial in the shifted-Jacobi basis exactly."""

    angular_l = int(angular_l)
    if angular_l < 0:
        raise ValueError("angular_l must be nonnegative.")
    remainder = list(_trim_polynomial(power_coefficients))
    maximum_degree = len(remainder) - 1
    expansion = [Fraction(0)] * (maximum_degree + 1)
    for degree in range(maximum_degree, -1, -1):
        basis = tuple(
            Fraction(value)
            for value in shifted_jacobi_power_coefficients(
                degree, 4, 2 * angular_l + 2
            )
        )
        leading = basis[-1]
        if leading == 0:
            raise RuntimeError("Shifted-Jacobi basis lost triangularity.")
        coefficient = remainder[degree] / leading
        expansion[degree] = coefficient
        for power, value in enumerate(basis):
            remainder[power] -= coefficient * value
    if any(value != 0 for value in remainder):
        raise RuntimeError("Exact shifted-Jacobi back-substitution did not close.")
    return tuple(expansion)


def radial_product_expansion(q1, l1, q2, l2, output_l):
    """Return the exact radial linearization for one allowed angular output."""

    q1 = int(q1)
    l1 = int(l1)
    q2 = int(q2)
    l2 = int(l2)
    output_l = int(output_l)
    if min(q1, l1, q2, l2, output_l) < 0:
        raise ValueError("Radial degrees and angular momenta must be nonnegative.")
    if output_l < abs(l1 - l2) or output_l > l1 + l2:
        raise ValueError("output_l violates the angular triangle rule.")
    if (l1 + l2 + output_l) % 2:
        raise ValueError("output_l has zero Racah product parity coefficient.")

    first = shifted_jacobi_power_coefficients(q1, 4, 2 * l1 + 2)
    second = shifted_jacobi_power_coefficients(q2, 4, 2 * l2 + 2)
    origin_power = l1 + l2 - output_l
    polynomial = _multiply_polynomials(first, second)
    polynomial = _multiply_polynomials(polynomial, _one_minus_x_power(2))
    polynomial = _shift_polynomial(polynomial, origin_power)
    expansion = shifted_jacobi_expansion(polynomial, output_l)
    expected_degree = q1 + q2 + origin_power + 2
    if len(expansion) != expected_degree + 1:
        raise RuntimeError("Radial product degree closure is inconsistent.")

    input_norm_squared = (
        shifted_jacobi_normalization_squared(q1, l1)
        * shifted_jacobi_normalization_squared(q2, l2)
    )
    outputs = []
    for degree, coefficient in enumerate(expansion):
        if coefficient == 0:
            continue
        output_norm_squared = shifted_jacobi_normalization_squared(
            degree, output_l
        )
        outputs.append(
            {
                "q": int(degree),
                "jacobi_coefficient": _fraction_payload(coefficient),
                "normalization_ratio_squared": _fraction_payload(
                    input_norm_squared / output_norm_squared
                ),
            }
        )
    return {
        "left": {"q": q1, "l": l1},
        "right": {"q": q2, "l": l2},
        "output_l": output_l,
        "origin_power": int(origin_power),
        "required_output_degree": int(expected_degree),
        "unnormalized_power_coefficients": tuple(
            _fraction_payload(value) for value in polynomial
        ),
        "outputs": tuple(outputs),
    }


def _normalize_source_key(channel, support_id):
    channel = dict(channel)
    species = str(channel["neighbor_species"])
    q = int(channel.get("q", channel.get("radial_channel", -1)))
    angular_l = int(channel["l"])
    family = str(
        channel.get("source_family_id", ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY)
    )
    channel_support = str(channel.get("support_id", support_id))
    if not species or q < 0 or angular_l < 0 or not channel_support:
        raise ValueError("Source keys require species, q>=0, l>=0, and support_id.")
    if family != ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY:
        raise ValueError(
            "The exact product record requires the role-independent "
            "orthogonal shifted-Jacobi source family."
        )
    return {
        "neighbor_species": species,
        "q": q,
        "l": angular_l,
        "source_family_id": family,
        "support_id": channel_support,
    }


def _source_key_tuple(key):
    return (
        str(key["neighbor_species"]),
        int(key["q"]),
        int(key["l"]),
        str(key["source_family_id"]),
        str(key["support_id"]),
    )


def build_radial_species_product_record(
    channels,
    angular_product_plan,
    *,
    maximum_collision_arity=2,
    support_id="common_normalized_cutoff_support_v1",
):
    """Build the cutoff-independent radial/species half of the product algebra."""

    maximum_collision_arity = int(maximum_collision_arity)
    if maximum_collision_arity < 2:
        raise ValueError("maximum_collision_arity must be at least 2.")
    angular_product_plan = dict(angular_product_plan)
    if angular_product_plan.get("schema") != "ye3t_racah_harmonic_product_v1":
        raise ValueError("Expected a compiler-owned YE3T Racah product plan.")
    if int(angular_product_plan.get("maximum_collision_arity", -1)) < (
        maximum_collision_arity
    ):
        raise ValueError("The angular product plan has insufficient collision arity.")

    primitive = tuple(
        _normalize_source_key(channel, support_id) for channel in channels
    )
    if not primitive or len({_source_key_tuple(key) for key in primitive}) != len(
        primitive
    ):
        raise ValueError("Primitive source keys must be nonempty and unique.")
    supports = {key["support_id"] for key in primitive}
    if len(supports) != 1:
        raise ValueError("The first exact product record requires one common support.")

    pair_outputs = {}
    for pair in angular_product_plan["pairs"]:
        pair_key = (int(pair["left_l"]), int(pair["right_l"]))
        pair_outputs[pair_key] = tuple(int(item["L"]) for item in pair["outputs"])

    primitive_by_tuple = {_source_key_tuple(key): key for key in primitive}
    frontier = {1: set(primitive_by_tuple)}
    all_keys = dict(primitive_by_tuple)
    products = {}
    closure = {1: set(primitive_by_tuple)}
    for arity in range(2, maximum_collision_arity + 1):
        reachable = set()
        for left_tuple in sorted(frontier[arity - 1]):
            left = all_keys[left_tuple]
            for right_tuple in sorted(primitive_by_tuple):
                right = primitive_by_tuple[right_tuple]
                if left["support_id"] != right["support_id"]:
                    raise ValueError("Product operands have incompatible support identities.")
                operation_key = (left_tuple, right_tuple)
                if left["neighbor_species"] != right["neighbor_species"]:
                    products.setdefault(
                        operation_key,
                        {
                            "left": left,
                            "right": right,
                            "species_product": "zero",
                            "outputs": (),
                        },
                    )
                    continue
                angular_key = (int(left["l"]), int(right["l"]))
                if angular_key not in pair_outputs:
                    raise ValueError(
                        "The angular product plan does not cover a required pair."
                    )
                operation_outputs = []
                for output_l in pair_outputs[angular_key]:
                    radial = radial_product_expansion(
                        left["q"],
                        left["l"],
                        right["q"],
                        right["l"],
                        output_l,
                    )
                    for output in radial["outputs"]:
                        output_key = {
                            "neighbor_species": left["neighbor_species"],
                            "q": int(output["q"]),
                            "l": int(output_l),
                            "source_family_id": ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY,
                            "support_id": left["support_id"],
                        }
                        output_tuple = _source_key_tuple(output_key)
                        all_keys.setdefault(output_tuple, output_key)
                        reachable.add(output_tuple)
                    operation_outputs.append(
                        {
                            "L": int(output_l),
                            "radial": radial,
                        }
                    )
                products.setdefault(
                    operation_key,
                    {
                        "left": left,
                        "right": right,
                        "species_product": "same_species_idempotent",
                        "outputs": tuple(operation_outputs),
                    },
                )
        frontier[arity] = reachable
        closure[arity] = set(reachable)

    closure_report = tuple(
        {
            "collision_arity": int(arity),
            "source_count": len(keys),
            "maximum_q_by_l": tuple(
                {
                    "l": int(angular_l),
                    "maximum_q": max(
                        int(key[1]) for key in keys if int(key[2]) == angular_l
                    ),
                }
                for angular_l in sorted({int(key[2]) for key in keys})
            ),
        }
        for arity, keys in sorted(closure.items())
    )
    normalized_support = {
        "radial_coordinate": "x=r/r_c",
        "normalized_interval": (0, 1),
        "envelope": "(1-x)^2",
        "radial_measure": "x^2_dx",
        "support_id": next(iter(supports)),
        "cartesian_C1_at_origin_for_all_channels": False,
        "exact_zero_distance_force_policy": "reject_before_direction_evaluation_v1",
    }
    body = {
        "schema": ORTHOGONAL_SHIFTED_JACOBI_PRODUCT_SCHEMA,
        "source_family_id": ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY,
        "jacobi_alpha": 4,
        "jacobi_beta_rule": "2*l+2",
        "normalized_support": normalized_support,
        "normalized_support_hash": _stable_hash(normalized_support),
        "angular_product_plan_hash": str(angular_product_plan["plan_hash"]),
        "primitive_source_keys": primitive,
        "source_inventory": tuple(
            {
                "source_key": all_keys[key],
                "shifted_jacobi_power_coefficients": tuple(
                    _fraction_payload(value)
                    for value in shifted_jacobi_power_coefficients(
                        int(key[1]), 4, 2 * int(key[2]) + 2
                    )
                ),
                "normalization_squared": _fraction_payload(
                    shifted_jacobi_normalization_squared(
                        int(key[1]), int(key[2])
                    )
                ),
            }
            for key in sorted(all_keys)
        ),
        "maximum_collision_arity": maximum_collision_arity,
        "species_product_rule": "indicator_idempotent_same_species_zero_cross_species",
        "binary_products": tuple(
            products[key] for key in sorted(products, key=repr)
        ),
        "required_degree_closure": closure_report,
        "certificates": {
            "exact_rational_polynomial_linearization": True,
            "normalization_ratio_stored_as_exact_square": True,
            "cutoff_independent_normalized_algebra": True,
            "common_support_required": True,
            "exact_origin_smoothness_not_overclaimed": True,
        },
    }
    return {**body, "record_hash": _stable_hash(body)}


def validate_radial_species_product_record(record, angular_product_plan):
    """Fail closed unless a product record exactly matches its reconstruction."""

    record = dict(record)
    supplied_hash = str(record.get("record_hash", ""))
    body = {key: value for key, value in record.items() if key != "record_hash"}
    if not supplied_hash or _stable_hash(body) != supplied_hash:
        raise ValueError("Shifted-Jacobi product record hash mismatch.")
    primitive = tuple(dict(value) for value in record["primitive_source_keys"])
    support_ids = {str(value["support_id"]) for value in primitive}
    if len(support_ids) != 1:
        raise ValueError("Shifted-Jacobi product record support identity is invalid.")
    expected = build_radial_species_product_record(
        primitive,
        angular_product_plan,
        maximum_collision_arity=int(record["maximum_collision_arity"]),
        support_id=next(iter(support_ids)),
    )
    if _stable_hash(expected) != _stable_hash(record):
        raise ValueError(
            "Shifted-Jacobi product record differs from exact reconstruction."
        )
    return True


__all__ = [
    "ORTHOGONAL_SHIFTED_JACOBI_PRODUCT_SCHEMA",
    "ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY",
    "build_radial_species_product_record",
    "radial_product_expansion",
    "shifted_jacobi_expansion",
    "shifted_jacobi_ladder_with_derivative",
    "shifted_jacobi_normalization_squared",
    "shifted_jacobi_power_coefficients",
    "validate_radial_species_product_record",
]
