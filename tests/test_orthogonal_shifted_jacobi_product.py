import copy
from fractions import Fraction

import numpy as np
import pytest

import ye3t.couplings.tagged_cauchy_image as tagged_image_module
from ye3t.cache import artifact_cache_events
from ye3t.couplings import (
    CompiledTaggedCauchyImage,
    compile as compile_coupling,
    count as count_coupling,
    plan as plan_coupling,
    racah_harmonic_product_plan,
    tagged_cauchy_image_request,
)
from ye3t.couplings.tagged_cauchy_image import (
    TAGGED_CAUCHY_N4_CATALOGUE_SCHEMA,
    TAGGED_CAUCHY_N4_COMPANION_SCHEMA,
    _physical_term_map,
    _validate_radial_species_product_algebra,
)
from ye3t.couplings.lifted_cauchy_scalar import (
    _exact_matrix_from_payload,
    _exact_scalar_from_payload,
)
from ye3t.couplings.orthogonal_shifted_jacobi import (
    ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY,
    _differentiate_polynomial,
    _fraction_from_payload,
    build_radial_species_product_record,
    radial_product_expansion,
    shifted_jacobi_ladder_with_derivative,
    shifted_jacobi_normalization_squared,
    shifted_jacobi_power_coefficients,
    validate_radial_species_product_record,
)


def _sympy():
    from ye3t._optional_sympy import sp

    return sp


def _power_add(left, right):
    size = max(len(left), len(right))
    return tuple(
        (left[index] if index < len(left) else Fraction(0))
        + (right[index] if index < len(right) else Fraction(0))
        for index in range(size)
    )


def _power_scale(scale, coefficients):
    return tuple(Fraction(scale) * Fraction(value) for value in coefficients)


def _power_multiply(left, right):
    result = [Fraction(0)] * (len(left) + len(right) - 1)
    for left_power, left_value in enumerate(left):
        for right_power, right_value in enumerate(right):
            result[left_power + right_power] += left_value * right_value
    return tuple(result)


def _integrate_unit_interval(coefficients):
    return sum(
        Fraction(value) / (power + 1)
        for power, value in enumerate(coefficients)
    )


def _normalized_radial_value_derivative(degree, angular_l, x):
    coefficients = shifted_jacobi_power_coefficients(
        degree, 4, 2 * angular_l + 2
    )
    normalization = float(
        shifted_jacobi_normalization_squared(degree, angular_l)
    ) ** 0.5
    polynomial = sum(value * x**power for power, value in enumerate(coefficients))
    derivative = sum(
        power * value * x ** (power - 1)
        for power, value in enumerate(coefficients)
        if power
    )
    envelope = x**angular_l * (1.0 - x) ** 2
    envelope_derivative = (
        angular_l * x ** (angular_l - 1) * (1.0 - x) ** 2
        if angular_l
        else 0.0
    ) - 2.0 * x**angular_l * (1.0 - x)
    return (
        normalization * envelope * polynomial,
        normalization
        * (envelope_derivative * polynomial + envelope * derivative),
    )


def _reconstruct(record):
    result = (Fraction(0),)
    output_l = int(record["output_l"])
    for output in record["outputs"]:
        degree = int(output["q"])
        coefficient = _fraction_from_payload(output["jacobi_coefficient"])
        basis = shifted_jacobi_power_coefficients(
            degree, 4, 2 * output_l + 2
        )
        result = _power_add(result, _power_scale(coefficient, basis))
    return result


def _normalized_product(q1, l1, q2, l2, output_l):
    sp = _sympy()
    record = radial_product_expansion(q1, l1, q2, l2, output_l)
    result = {}
    for output in record["outputs"]:
        coefficient = _fraction_from_payload(output["jacobi_coefficient"])
        ratio = _fraction_from_payload(output["normalization_ratio_squared"])
        value = sp.Rational(coefficient.numerator, coefficient.denominator) * sp.sqrt(
            sp.Rational(ratio.numerator, ratio.denominator)
        )
        result[int(output["q"])] = sp.simplify(value)
    return result


def _evaluate_emitted_schedule(schedule, values):
    total = 0.0
    for term in schedule["forward_terms"]:
        coefficient = float(_exact_scalar_from_payload(term["coefficient"]))
        total += coefficient * np.prod(
            [values[int(index)] for index in term["source_indices"]]
        )
    return float(total)


def _evaluate_emitted_schedule_vjp(schedule, values):
    result = np.zeros(len(values), dtype=np.float64)
    for term in schedule["adjoint_terms"]:
        coefficient = float(_exact_scalar_from_payload(term["coefficient"]))
        result[int(term["source_index"])] += coefficient * np.prod(
            [
                values[int(index)]
                for index in term["remaining_source_indices"]
            ]
        )
    return result


def test_shifted_jacobi_coefficients_norm_and_binary_product_are_exact():
    assert shifted_jacobi_power_coefficients(0, 4, 2) == (1,)
    assert shifted_jacobi_power_coefficients(1, 4, 2) == (-3, 8)
    assert shifted_jacobi_normalization_squared(0, 0) == 105

    record = radial_product_expansion(0, 0, 0, 0, 0)
    original = tuple(
        _fraction_from_payload(value)
        for value in record["unnormalized_power_coefficients"]
    )
    reconstructed = _reconstruct(record)
    assert reconstructed == original
    assert _differentiate_polynomial(reconstructed) == _differentiate_polynomial(
        original
    )
    assert record["required_output_degree"] == 2


def test_shifted_jacobi_sources_have_exact_radial_gram_identity():
    one_minus_x_fourth = (1, -4, 6, -4, 1)
    for angular_l in (0, 1, 3):
        rows = [
            shifted_jacobi_power_coefficients(
                degree, 4, 2 * angular_l + 2
            )
            for degree in range(4)
        ]
        weight = (Fraction(0),) * (2 * angular_l + 2) + tuple(
            Fraction(value) for value in one_minus_x_fourth
        )
        for left_degree, left in enumerate(rows):
            for right_degree, right in enumerate(rows):
                overlap = _integrate_unit_interval(
                    _power_multiply(_power_multiply(left, right), weight)
                )
                if left_degree == right_degree:
                    assert (
                        overlap
                        * shifted_jacobi_normalization_squared(
                            left_degree, angular_l
                        )
                        == 1
                    )
                else:
                    assert overlap == 0


def test_normalized_radial_product_values_and_derivatives_match_on_grid():
    expansion = _normalized_product(1, 1, 2, 2, 1)
    for x in (0.07, 0.23, 0.51, 0.83, 0.97):
        left_value, left_derivative = _normalized_radial_value_derivative(
            1, 1, x
        )
        right_value, right_derivative = _normalized_radial_value_derivative(
            2, 2, x
        )
        expected_value = left_value * right_value
        expected_derivative = (
            left_derivative * right_value + left_value * right_derivative
        )
        actual_value = 0.0
        actual_derivative = 0.0
        for degree, coefficient in expansion.items():
            value, derivative = _normalized_radial_value_derivative(
                degree, 1, x
            )
            actual_value += float(coefficient) * value
            actual_derivative += float(coefficient) * derivative
        # The exact polynomial identity is checked above.  This grid check
        # deliberately exercises the exported binary64 coefficients, whose
        # high-degree cancellation near the interval ends loses a few ulps.
        assert actual_value == pytest.approx(
            expected_value, rel=2.0e-11, abs=2.0e-11
        )
        assert actual_derivative == pytest.approx(
            expected_derivative, rel=5.0e-11, abs=5.0e-11
        )


def test_radial_product_is_commutative_and_associative_through_three_factors():
    sp = _sympy()
    forward = radial_product_expansion(1, 1, 2, 2, 1)
    reverse = radial_product_expansion(2, 2, 1, 1, 1)
    assert forward["required_output_degree"] == reverse["required_output_degree"]
    assert forward["unnormalized_power_coefficients"] == reverse[
        "unnormalized_power_coefficients"
    ]
    assert forward["outputs"] == reverse["outputs"]

    left = {}
    for middle_q, first_coefficient in _normalized_product(0, 0, 1, 0, 0).items():
        for final_q, second_coefficient in _normalized_product(
            middle_q, 0, 2, 0, 0
        ).items():
            left[final_q] = sp.simplify(
                left.get(final_q, 0) + first_coefficient * second_coefficient
            )
    right = {}
    for middle_q, first_coefficient in _normalized_product(1, 0, 2, 0, 0).items():
        for final_q, second_coefficient in _normalized_product(
            0, 0, middle_q, 0, 0
        ).items():
            right[final_q] = sp.simplify(
                right.get(final_q, 0) + first_coefficient * second_coefficient
            )
    assert set(left) == set(right)
    assert all(sp.simplify(left[key] - right[key]) == 0 for key in left)


def test_product_record_tracks_degree_closure_species_and_tampering():
    angular = racah_harmonic_product_plan(
        (0,), maximum_collision_arity=3
    )
    channels = (
        {
            "neighbor_species": "Ta",
            "q": 0,
            "l": 0,
            "source_family_id": ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY,
        },
        {
            "neighbor_species": "W",
            "q": 0,
            "l": 0,
            "source_family_id": ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY,
        },
    )
    record = build_radial_species_product_record(
        channels, angular, maximum_collision_arity=3
    )
    assert validate_radial_species_product_record(record, angular) is True
    assert _validate_radial_species_product_algebra(record, angular)["passed"] is True
    assert record["required_degree_closure"][-1]["maximum_q_by_l"] == (
        {"l": 0, "maximum_q": 4},
    )
    assert any(
        product["species_product"] == "zero"
        for product in record["binary_products"]
    )
    assert any(
        product["species_product"] == "same_species_idempotent"
        for product in record["binary_products"]
    )
    assert record["normalized_support"][
        "cartesian_C1_at_origin_for_all_channels"
    ] is False
    assert record["normalized_support"]["exact_zero_distance_force_policy"] == (
        "reject_before_direction_evaluation_v1"
    )

    tampered = copy.deepcopy(record)
    tampered["required_degree_closure"][-1]["source_count"] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_radial_species_product_record(tampered, angular)

    with pytest.raises(ValueError, match="insufficient collision arity"):
        build_radial_species_product_record(
            channels,
            racah_harmonic_product_plan(
                (0,), maximum_collision_arity=2
            ),
            maximum_collision_arity=3,
        )


def test_radial_source_value_and_first_derivative_vanish_at_cutoff():
    for angular_l in (0, 1, 3):
        for degree in (0, 1, 3):
            jacobi = shifted_jacobi_power_coefficients(
                degree, 4, 2 * angular_l + 2
            )
            source = [Fraction(0)] * angular_l + list(jacobi)
            envelope = (Fraction(1), Fraction(-2), Fraction(1))
            product = [Fraction(0)] * (len(source) + 2)
            for left_power, left_value in enumerate(source):
                for right_power, right_value in enumerate(envelope):
                    product[left_power + right_power] += left_value * right_value
            value = sum(product)
            derivative = sum(
                power * coefficient
                for power, coefficient in enumerate(product)
            )
            assert value == 0
            assert derivative == 0


def test_n4_s2_nontrivial_tag_physical_companion_is_exact_and_nonzero(
    monkeypatch,
    tmp_path,
):
    sp = _sympy()
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path / "cache"))
    artifact_cache_events(clear=True)
    angular = racah_harmonic_product_plan(
        (1,), maximum_collision_arity=2
    )
    source = {
        "neighbor_species": "Ta",
        "q": 0,
        "l": 1,
        "source_family_id": ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY,
    }
    product_record = build_radial_species_product_record(
        (source,), angular, maximum_collision_arity=2
    )
    request = tagged_cauchy_image_request(
        product_record, angular, source_key=source
    )
    with monkeypatch.context() as context:
        context.setattr(
            tagged_image_module,
            "_compile_s2_n4_physical_companion",
            lambda *args, **kwargs: pytest.fail(
                "count/plan materialized coefficients"
            ),
        )
        report = count_coupling(request)
        compiler_plan = plan_coupling(report)
    assert report.raw_label_count == 3
    assert report.image_dimension_upper_bound == 2
    assert report.exact_image_dimension == 2
    assert report.resource_report["dimension_certificate"][
        "predicted_image_dimension"
    ] == 2
    assert report.resource_report["coefficient_materialization_performed"] is False
    assert compiler_plan.validation_report[
        "coefficient_materialization_performed"
    ] is False

    artifact_cache_events(clear=True)
    artifact = compile_coupling(compiler_plan)
    replayed = compile_coupling(compiler_plan)
    cache_events = tuple(
        event
        for event in artifact_cache_events(clear=True)
        if event["artifact_type"] == "tagged_cauchy_n4_s2_physical_image"
    )
    assert tuple(event["status"] for event in cache_events) == ("miss", "hit")
    assert replayed.to_dict() == artifact.to_dict()
    assert CompiledTaggedCauchyImage.from_dict(artifact.to_dict()) == artifact
    warm_report = count_coupling(request)
    assert warm_report.exact_image_dimension == 2
    assert warm_report.convention_hash == report.convention_hash
    assert warm_report.resource_report["cache_status"][
        "physical_companion"
    ] == (
        "verified_hit"
    )
    assert warm_report.resource_report["cache_status"]["catalogue_image"] == (
        "verified_hit"
    )
    compiled_report = count_coupling(artifact)
    assert compiled_report.exact_image_dimension == 2
    assert "cache_status" not in compiled_report.resource_report
    assert compiled_report.validation_report[
        "exact_image_dimension_known_from_compiled_payload"
    ] is True
    warm_artifact = compile_coupling(plan_coupling(warm_report))
    assert warm_artifact.to_dict() == artifact.to_dict()
    mismatched_payload = copy.deepcopy(artifact.payload)
    mismatched_payload["image_rows"] = mismatched_payload["image_rows"][:-1]
    with monkeypatch.context() as context:
        context.setattr(
            tagged_image_module,
            "_compile_n4_s012_physical_image",
            lambda *args, **kwargs: mismatched_payload,
        )
        with pytest.raises(
            ValueError, match="compiled payload dimensions disagree"
        ):
            compile_coupling(plan_coupling(warm_report))
    assert count_coupling(warm_artifact) == compiled_report
    assert plan_coupling(warm_artifact) == plan_coupling(artifact)
    compiled = artifact.payload
    assert compiled["schema"] == TAGGED_CAUCHY_N4_CATALOGUE_SCHEMA
    assert compiled["certificate"]["passed"] is True
    assert len(compiled["raw_coordinate_labels"]) == 3
    assert len(compiled["image_rows"]) == 2
    assert compiled["dropped"] == (
        {
            "raw_coordinate_index": 1,
            "reason": "exact_physical_image_kernel",
        },
    )
    image_from_raw = _exact_matrix_from_payload(compiled["image_from_raw"])
    raw_from_image = _exact_matrix_from_payload(compiled["raw_from_image"])
    raw_gram = _exact_matrix_from_payload(compiled["raw_gram"])
    assert sp.simplify(image_from_raw * raw_from_image) == sp.eye(2)
    assert sp.simplify(
        image_from_raw * raw_gram * image_from_raw.conjugate().T
    ) == sp.eye(2)
    assert sp.simplify(
        raw_from_image - raw_gram * image_from_raw.conjugate().T
    ) == sp.zeros(3, 2)
    assert _exact_matrix_from_payload(compiled["image_gram"]) == sp.eye(2)

    assert len(compiled["physical_companions"]) == 1
    companion = compiled["physical_companions"][0]
    assert companion["schema"] == TAGGED_CAUCHY_N4_COMPANION_SCHEMA
    assert companion["certificate"]["passed"] is True
    assert companion["tag_kappa"] == (1, 1)
    assert companion["placement_parent_lambda"] == (3, 1)
    assert companion["role_kappa"] == (2, 1, 1)
    assert companion["angular_kappa"] == (2, 2)
    assert companion["matching_physical_source_bound"] is True
    assert companion["physical_image_lowered"] is True
    assert len(_physical_term_map(companion["paired_source_terms"])) == 216
    assert companion["raw_moment_terms"]
    assert companion["image_moment_terms"]
    raw_to_image = _exact_matrix_from_payload(companion["raw_to_image_P"])
    image_to_raw = _exact_matrix_from_payload(companion["image_to_raw_H"])
    assert sp.simplify(raw_to_image * image_to_raw) == sp.eye(1)
    assert sp.simplify(image_to_raw * raw_to_image) == sp.eye(1)
    source_witness = companion["source_image_witness"]
    direct_value = _exact_scalar_from_payload(
        source_witness["direct_distinct_tag_value"]
    )
    raw_value = _exact_scalar_from_payload(source_witness["raw_moment_value"])
    image_value = _exact_scalar_from_payload(
        source_witness["normalized_image_value"]
    )
    image_norm = _exact_scalar_from_payload(companion["image_metric_norm"])
    assert sp.simplify(direct_value - raw_value) == 0
    assert sp.simplify(direct_value) != 0
    assert sp.simplify(image_value - direct_value / sp.sqrt(image_norm)) == 0
    three_neighbor = companion["three_neighbor_source_image_witness"]
    three_direct = _exact_scalar_from_payload(
        three_neighbor["direct_distinct_tag_value"]
    )
    three_raw = _exact_scalar_from_payload(three_neighbor["raw_moment_value"])
    three_reordered = _exact_scalar_from_payload(
        three_neighbor["reordered_direct_value"]
    )
    assert sp.simplify(three_direct - three_raw) == 0
    assert sp.simplify(three_direct - three_reordered) == 0
    assert sp.simplify(three_direct) != 0
    assert all(companion["certificate"]["checks"].values())
    assert sp.simplify(
        _exact_scalar_from_payload(companion["two_neighbor_witness"]["value"])
        + 1
    ) == 0
    assert sp.simplify(
        _exact_scalar_from_payload(
            companion["two_neighbor_witness"]["second_value"]
        )
        + 4
    ) == 0

    for schedule in compiled["moment_schedules"]:
        values = np.linspace(
            0.31,
            1.07,
            len(schedule["source_generators"]),
            dtype=np.float64,
        )
        analytic = _evaluate_emitted_schedule_vjp(schedule, values)
        numerical = np.zeros_like(analytic)
        step = 1.0e-6
        for index in range(len(values)):
            plus = values.copy()
            minus = values.copy()
            plus[index] += step
            minus[index] -= step
            numerical[index] = (
                _evaluate_emitted_schedule(schedule, plus)
                - _evaluate_emitted_schedule(schedule, minus)
            ) / (2.0 * step)
        np.testing.assert_allclose(analytic, numerical, rtol=2.0e-8, atol=2.0e-9)

    assert {int(record["angular_l"]) for record in compiled["real_forms"]} == {
        int(generator["l"])
        for schedule in compiled["moment_schedules"]
        for generator in schedule["source_generators"]
    }

    tampered = copy.deepcopy(request)
    tampered["source_keys"][0]["q"] += 1
    with pytest.raises(ValueError, match="request hash mismatch"):
        count_coupling(tampered)

    tampered_artifact = copy.deepcopy(artifact)
    tampered_artifact.payload["image_rows"] = ()
    with pytest.raises(ValueError, match="compiled payload hash mismatch"):
        count_coupling(tampered_artifact)

    tampered_serialized = copy.deepcopy(artifact.to_dict())
    tampered_serialized["payload"]["real_forms"][0]["angular_l"] += 1
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        CompiledTaggedCauchyImage.from_dict(tampered_serialized)


def test_n4_tagged_physical_image_is_catalogue_wide_across_radial_content(
    monkeypatch,
    tmp_path,
):
    sp = _sympy()
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path / "cache"))
    angular = racah_harmonic_product_plan(
        (1,), maximum_collision_arity=2
    )
    sources = tuple(
        {
            "neighbor_species": "Ta",
            "q": degree,
            "l": 1,
            "source_family_id": ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY,
        }
        for degree in (0, 1)
    )
    product_record = build_radial_species_product_record(
        sources, angular, maximum_collision_arity=2
    )
    request = tagged_cauchy_image_request(product_record, angular)
    report = count_coupling(request)
    assert report.raw_label_count == 6
    assert report.image_dimension_upper_bound == 4
    assert report.exact_image_dimension == 4
    assert report.resource_report["dimension_certificate"][
        "predicted_image_dimension"
    ] == 4

    artifact = compile_coupling(plan_coupling(report))
    payload = artifact.payload
    assert tuple(value["q"] for value in payload["source_keys"]) == (0, 1)
    assert len(payload["raw_rows"]) == 6
    assert len(payload["image_rows"]) == 4
    assert tuple(
        int(value["raw_coordinate_index"]) for value in payload["dropped"]
    ) == (1, 4)
    image_from_raw = _exact_matrix_from_payload(payload["image_from_raw"])
    raw_from_image = _exact_matrix_from_payload(payload["raw_from_image"])
    assert sp.simplify(image_from_raw * raw_from_image) == sp.eye(4)
    assert _exact_matrix_from_payload(payload["image_gram"]) == sp.eye(4)
    assert all(payload["certificate"]["checks"].values())

    warm_report = count_coupling(request)
    assert warm_report.exact_image_dimension == 4
    assert warm_report.convention_hash == report.convention_hash
    warm_artifact = compile_coupling(plan_coupling(warm_report))
    assert warm_artifact.to_dict() == artifact.to_dict()


def test_n4_tagged_selected_opportunity_images_are_exact_and_canonical(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path / "cache"))
    angular = racah_harmonic_product_plan(
        (1,), maximum_collision_arity=2
    )
    sources = tuple(
        {
            "neighbor_species": "Ta",
            "q": degree,
            "l": 1,
            "source_family_id": ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY,
        }
        for degree in (0, 1)
    )
    product_record = build_radial_species_product_record(
        sources, angular, maximum_collision_arity=2
    )
    selections = (
        (0,),
        (1,),
        (2,),
        (0, 1),
        (0, 2),
        (1, 2),
        (0, 1, 2),
    )
    payloads = {}
    for source_count in (1, 2):
        selected_sources = sources[:source_count]
        for selected in selections:
            request = tagged_cauchy_image_request(
                product_record,
                angular,
                source_keys=selected_sources,
                selected_raw_tag_counts=selected,
            )
            report = count_coupling(request)
            expected_dimension = source_count * (
                int(bool(set(selected) & {0, 1})) + int(2 in selected)
            )
            assert report.raw_label_count == source_count * len(selected)
            assert report.exact_image_dimension == expected_dimension
            assert report.image_dimension_upper_bound == expected_dimension
            assert report.resource_report[
                "coefficient_materialization_performed"
            ] is False
            assert tuple(
                report.resource_report["dimension_certificate"][
                    "selected_raw_tag_counts"
                ]
            ) == selected
            plan = plan_coupling(report)
            assert (
                "compile_selected_ordinary_density_moment_controls"
                in plan.materialization_steps
            ) is bool(set(selected) & {0, 1})
            artifact = compile_coupling(plan)
            payload = artifact.payload
            assert tuple(payload["selected_raw_tag_counts"]) == selected
            assert len(payload["raw_rows"]) == source_count * len(selected)
            assert len(payload["image_rows"]) == expected_dimension
            assert len(payload["image_coordinate_provenance"]) == (
                expected_dimension
            )
            assert payload["matching_physical_source_bound"] is (2 in selected)
            assert payload["matching_physical_source_status"] == (
                "bound" if 2 in selected else "not_required"
            )
            assert all(
                set(record["represented_raw_tag_counts"]).issubset(selected)
                and "single_selected_raw_tag_count_construction_support"
                in record
                for record in payload["image_coordinate_provenance"]
            )
            real_schedule = payload["real_schedule_core"]
            assert tuple(real_schedule["tag_counts"]) == selected
            assert tuple(real_schedule["selected_raw_tag_counts"]) == selected
            assert int(real_schedule["tag_count"]) == max(selected)
            assert {
                int(label["tag_count"])
                for label in payload["raw_coordinate_labels"]
            } == set(selected)
            payloads[(source_count, selected)] = payload

    for source_count in (1, 2):
        assert payloads[(source_count, (0,))]["image_rows"] == payloads[
            (source_count, (1,))
        ]["image_rows"]
        assert payloads[(source_count, (0,))]["image_rows"] == payloads[
            (source_count, (0, 1))
        ]["image_rows"]
        assert payloads[(source_count, (0, 2))]["image_rows"] == payloads[
            (source_count, (1, 2))
        ]["image_rows"]
        assert payloads[(source_count, (0, 2))]["image_rows"] == payloads[
            (source_count, (0, 1, 2))
        ]["image_rows"]

    q1_only_request = tagged_cauchy_image_request(
        product_record,
        angular,
        source_keys=(sources[1],),
        selected_raw_tag_counts=(0, 2),
    )
    q1_only = compile_coupling(plan_coupling(q1_only_request)).payload
    q1_only_ids = {
        int(label["tag_count"]): label["raw_opportunity_id"]
        for label in q1_only["raw_coordinate_labels"]
    }
    extended_ids = {
        int(label["tag_count"]): label["raw_opportunity_id"]
        for label in payloads[(2, (0, 2))]["raw_coordinate_labels"]
        if int(label["source_key"]["q"]) == 1
    }
    assert q1_only_ids == extended_ids

    canonical = tagged_cauchy_image_request(
        product_record,
        angular,
        source_keys=sources,
        selected_raw_tag_counts=(2, 0),
    )
    ordered = tagged_cauchy_image_request(
        product_record,
        angular,
        source_keys=sources,
        selected_raw_tag_counts=(0, 2),
    )
    assert canonical == ordered
    with pytest.raises(ValueError, match="nonempty"):
        tagged_cauchy_image_request(
            product_record, angular, selected_raw_tag_counts=()
        )
    with pytest.raises(ValueError, match="duplicates"):
        tagged_cauchy_image_request(
            product_record, angular, selected_raw_tag_counts=(0, 0)
        )
    with pytest.raises(ValueError, match="contain integers"):
        tagged_cauchy_image_request(
            product_record, angular, selected_raw_tag_counts=(0, "2")
        )
    with pytest.raises(ValueError, match="supports raw tag counts"):
        tagged_cauchy_image_request(
            product_record, angular, selected_raw_tag_counts=(3,)
        )


def _exact_power_value_derivative(coefficients, coordinate):
    coordinate = Fraction(coordinate)
    value = sum(
        Fraction(coefficient) * coordinate**power
        for power, coefficient in enumerate(coefficients)
    )
    derivative = sum(
        power * Fraction(coefficient) * coordinate ** (power - 1)
        for power, coefficient in enumerate(coefficients)
        if power
    )
    return float(value), float(derivative)


def test_shifted_jacobi_stable_recurrence_matches_exact_polynomials():
    coordinates = (
        Fraction(0),
        Fraction(1, 1000),
        Fraction(1, 7),
        Fraction(1, 2),
        Fraction(999, 1000),
        Fraction(1),
    )
    for angular_l in (0, 1, 2, 9):
        beta = 2 * angular_l + 2
        for coordinate in coordinates:
            values, derivatives = shifted_jacobi_ladder_with_derivative(
                18, 4, beta, float(coordinate)
            )
            for degree in range(19):
                exact = _exact_power_value_derivative(
                    shifted_jacobi_power_coefficients(degree, 4, beta),
                    coordinate,
                )
                np.testing.assert_allclose(
                    (values[degree], derivatives[degree]),
                    exact,
                    rtol=3.0e-12,
                    atol=3.0e-10,
                )
    values, derivatives = shifted_jacobi_ladder_with_derivative(2, 4, 2, 0.37)
    np.testing.assert_allclose(values[0], 1.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(values[1], 8.0 * 0.37 - 3.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        values[2], 6.0 - 36.0 * 0.37 + 45.0 * 0.37**2, rtol=0.0, atol=2.0e-15
    )
    np.testing.assert_allclose(derivatives[2], -36.0 + 90.0 * 0.37, atol=2.0e-14)


