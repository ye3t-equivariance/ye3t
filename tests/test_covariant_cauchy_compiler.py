"""Exact-count, covariance, and derivative tests for covariant Cauchy multiplets."""

from math import comb

import numpy as np
import pytest
import torch

from ye3t.core.spherical import spherical_harmonics_l
from ye3t.couplings.covariant_cauchy import (
    compile_covariant_cauchy,
    covariant_cauchy_count,
    covariant_cauchy_request,
    evaluate_covariant_cauchy,
)
from ye3t.couplings.lifted_cauchy_scalar import (
    _real_form_matrix,
    compile_lifted_cauchy_scalar,
    evaluate_lifted_cauchy_scalar,
    lifted_cauchy_fixed_content_scalar_request,
)
from ye3t.paired_cg import _real_cg_entries_cpu


def channel(l, radial_channel):
    return {
        "neighbor_species": "X",
        "radial_channel": int(radial_channel),
        "l": int(l),
        "source_family_id": "test_source",
    }


def real_harmonics(l, positions):
    """Real harmonics in the compiler's tesseral layout (cosine l..1, m=0, sine 1..l).

    ``ye3t.core.spherical`` orders its real harmonics by signed ``m`` instead,
    so the inputs are built from the complex Condon-Shortley harmonics through
    the compiler's own unitary real form.
    """

    unit = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    complex_harmonics = spherical_harmonics_l(
        int(l),
        torch.as_tensor(np.arccos(np.clip(unit[:, 2], -1.0, 1.0))),
        torch.as_tensor(np.arctan2(unit[:, 1], unit[:, 0])),
    ).numpy().T
    real_to_complex = np.asarray(
        [[complex(value) for value in row] for row in _real_form_matrix(int(l)).tolist()]
    )
    real = complex_harmonics @ real_to_complex.conj()
    assert np.max(np.abs(real.imag)) < 1.0e-12
    return real.real


def densities(request, positions):
    """Role-resolved real densities with rotation-invariant radial role weights."""

    radius = np.linalg.norm(positions, axis=1)
    values = {}
    for item in request["channels"]:
        harmonics = real_harmonics(item["l"], positions)
        weights = np.stack(
            [
                radius ** (role + item["radial_channel"])
                for role in range(request["role_dimension"])
            ]
        )
        values[item["channel_index"]] = weights @ harmonics
    return values


def random_rotation(rng):
    matrix, triangular = np.linalg.qr(rng.normal(size=(3, 3)))
    matrix = matrix @ np.diag(np.sign(np.diag(triangular)))
    if np.linalg.det(matrix) < 0.0:
        matrix[:, 0] *= -1.0
    return matrix


def wigner_real(L, rotation, rng):
    """D^L in the project's real basis, fitted from degree-L harmonics alone."""

    points = rng.normal(size=(12 * (2 * L + 1), 3))
    return np.linalg.lstsq(
        real_harmonics(L, points), real_harmonics(L, points @ rotation.T), rcond=None
    )[0].T


def test_component_counts_fill_the_symmetric_power_space():
    role_dimension = 2
    contents = (
        ((channel(1, 0),), (2,)),
        ((channel(1, 0), channel(2, 1)), (2, 1)),
        ((channel(1, 0), channel(1, 1)), (1, 1)),
    )
    for channels, sizes in contents:
        expected = 1
        for item, size in zip(channels, sizes, strict=True):
            dimension = role_dimension * (2 * item["l"] + 1)
            expected *= comb(dimension + size - 1, size)
        total = 0
        for L in range(sum(size * item["l"] for item, size in zip(channels, sizes)) + 1):
            report = covariant_cauchy_count(
                covariant_cauchy_request(
                    channels, sizes, target_L=L, role_dimension=role_dimension
                )
            )
            assert report["component_count"] == report["multiplet_count"] * (2 * L + 1)
            total += report["component_count"]
        assert total == expected


def test_couplings_entry_points_own_the_labels_and_coefficients():
    import ye3t.couplings as couplings

    request = couplings.covariant_cauchy_request((channel(1, 0),), (2,), target_L=2)
    report = couplings.count(request)
    assert couplings.plan(request) == report
    assert "block_templates" not in report
    compiled = couplings.compile(request)
    assert tuple(item["label"] for item in compiled["descriptors"]) == report["labels"]
    assert compiled["validation_report"]["protected_axes"] == ("magnetic",)


def test_parity_is_fixed_by_content():
    with pytest.raises(ValueError, match="fixes O\\(3\\) parity"):
        covariant_cauchy_request((channel(1, 0),), (2,), target_L=2, target_parity=-1)
    request = covariant_cauchy_request((channel(1, 0),), (3,), target_L=1)
    assert request["target"]["o3_parity"] == -1


@pytest.mark.parametrize(
    "channels,sizes,target_L,pseudo",
    (
        ((channel(1, 0),), (2,), 2, False),
        ((channel(1, 0), channel(2, 1)), (2, 1), 1, True),
        ((channel(1, 0), channel(1, 1)), (1, 1), 1, True),
        ((channel(2, 0),), (1,), 2, False),
    ),
)
def test_complete_multiplets_are_o3_covariant(channels, sizes, target_L, pseudo):
    rng = np.random.default_rng(7 + target_L + len(sizes))
    request = covariant_cauchy_request(channels, sizes, target_L=target_L)
    compiled = compile_covariant_cauchy(request)
    assert compiled["descriptors"]
    parity = request["target"]["o3_parity"]
    assert pseudo == (parity != (-1) ** target_L)
    positions = rng.normal(size=(6, 3))
    rotation = random_rotation(rng)
    reference, _ = evaluate_covariant_cauchy(compiled, densities(request, positions))
    assert np.max(np.abs(reference)) > 1.0e-6
    wigner = wigner_real(target_L, rotation, rng)
    rotated, _ = evaluate_covariant_cauchy(
        compiled, densities(request, positions @ rotation.T)
    )
    assert np.allclose(rotated, reference @ wigner.T, atol=1.0e-10)
    # Improper element Q = R(-I): the multiplet acquires the signed parity, which
    # differs from the natural (-1)^L for a pseudo-tensor.
    improper, _ = evaluate_covariant_cauchy(
        compiled, densities(request, -positions @ rotation.T)
    )
    assert np.allclose(improper, parity * reference @ wigner.T, atol=1.0e-10)


def test_single_block_outer_map_is_the_identity_multiplet():
    request = covariant_cauchy_request((channel(2, 0),), (1,), target_L=2, role_dimension=1)
    compiled = compile_covariant_cauchy(request)
    values = {0: np.random.default_rng(3).normal(size=(1, 5))}
    outputs, _ = evaluate_covariant_cauchy(compiled, values)
    assert outputs.shape == (1, 5)
    assert np.allclose(outputs[0], values[0][0], atol=1.0e-13)


def test_vjp_matches_finite_differences_for_a_random_cotangent():
    rng = np.random.default_rng(11)
    request = covariant_cauchy_request((channel(1, 0), channel(2, 1)), (2, 1), target_L=1)
    compiled = compile_covariant_cauchy(request)
    values = densities(request, rng.normal(size=(5, 3)))
    count = len(compiled["descriptors"])
    cotangent = rng.normal(size=(count, 3))
    _, gradients = evaluate_covariant_cauchy(compiled, values, upstream=cotangent)
    step = 1.0e-6
    for key, value in values.items():
        numeric = np.zeros_like(value)
        for index in np.ndindex(value.shape):
            shifted = {name: item.copy() for name, item in values.items()}
            shifted[key][index] += step
            plus, _ = evaluate_covariant_cauchy(compiled, shifted)
            shifted[key][index] -= 2.0 * step
            minus, _ = evaluate_covariant_cauchy(compiled, shifted)
            numeric[index] = np.sum(cotangent * (plus - minus)) / (2.0 * step)
        assert np.allclose(gradients[key], numeric, rtol=1.0e-6, atol=1.0e-7)


def test_scalar_target_reproduces_the_scalar_compiler():
    rng = np.random.default_rng(5)
    channels = (channel(1, 0), channel(2, 1))
    sizes = (2, 1)
    request = covariant_cauchy_request(channels, sizes, target_L=0)
    compiled = compile_covariant_cauchy(request)
    scalar = compile_lifted_cauchy_scalar(
        lifted_cauchy_fixed_content_scalar_request(channels, sizes, role_dimension=2)
    )
    values = densities(request, rng.normal(size=(6, 3)))
    outputs, _ = evaluate_covariant_cauchy(compiled, values)
    scalar_outputs, _ = evaluate_lifted_cauchy_scalar(
        scalar, values, realization="factored", input_basis="real_tesseral"
    )
    key_fields = (
        "block_kappas",
        "block_Lambdas",
        "role_copy_indices",
        "angular_copy_indices",
        "outer_copy_index",
    )
    general = {
        tuple(descriptor["label"][name] for name in key_fields): outputs[index, 0]
        for index, descriptor in enumerate(compiled["descriptors"])
    }
    reference = {
        tuple(
            getattr(label, name) if name != "block_kappas" else tuple(
                tuple(part) for part in label.block_kappas
            )
            for name in key_fields
        ): scalar_outputs[index]
        for index, label in enumerate(scalar.plan.report.labels)
    }
    assert set(general) == set(reference)
    for key, value in reference.items():
        assert general[key] == pytest.approx(value, abs=1.0e-12)


def test_pseudovector_uses_the_paired_cg_real_phase():
    rng = np.random.default_rng(13)
    request = covariant_cauchy_request(
        (channel(1, 0), channel(1, 1)), (1, 1), target_L=1, role_dimension=1
    )
    compiled = compile_covariant_cauchy(request)
    assert compiled["real_forms"]["output"]["phase_exponent"] == 1
    left = rng.normal(size=3)
    right = rng.normal(size=3)
    outputs, _ = evaluate_covariant_cauchy(
        compiled, {0: left[None, :], 1: right[None, :]}
    )
    assert outputs.shape == (1, 3)
    expected = np.zeros(3)
    for index_left, index_right, index_out, value in _real_cg_entries_cpu(1, 1, 1):
        expected[index_out] += value * left[index_left] * right[index_right]
    ratio = outputs[0] @ expected / (expected @ expected)
    assert abs(ratio) > 1.0e-3
    assert np.allclose(outputs[0], ratio * expected, atol=1.0e-12)


def flat_real_outputs(plan, vector):
    outputs = np.zeros(plan["row_count"])
    for term in range(len(plan["term_row"])):
        start, stop = plan["factor_offsets"][term], plan["factor_offsets"][term + 1]
        outputs[plan["term_row"][term]] += plan["term_coefficient"][term] * np.prod(
            vector[plan["factors"][start:stop]]
        )
    return outputs


@pytest.mark.parametrize(
    "channels,sizes,target_L",
    (
        ((channel(1, 0),), (2,), 2),
        ((channel(1, 0), channel(1, 1)), (1, 1), 1),
        ((channel(1, 0), channel(2, 1)), (2, 1), 1),
    ),
)
def test_exact_real_schedule_matches_the_factored_realization(channels, sizes, target_L):
    from ye3t.couplings.covariant_cauchy import covariant_cauchy_flat_real_plan

    rng = np.random.default_rng(17 + target_L)
    request = covariant_cauchy_request(channels, sizes, target_L=target_L)
    compiled = compile_covariant_cauchy(request)
    assert compiled["validation_report"]["real_schedule_coefficients_exactly_real"]
    values = densities(request, rng.normal(size=(5, 3)))
    factored, _ = evaluate_covariant_cauchy(compiled, values)
    plan = covariant_cauchy_flat_real_plan(compiled)
    vector = np.concatenate(
        [values[item["channel_index"]].reshape(-1) for item in request["channels"]]
    )
    assert vector.shape == (plan["input_count"],)
    assert np.allclose(
        flat_real_outputs(plan, vector).reshape(factored.shape), factored, atol=1.0e-12
    )


def test_orthogonal_output_plan_acts_on_multiplicity_copies_only():
    from ye3t.couplings.covariant_cauchy import (
        _binary_matrix,
        covariant_cauchy_orthogonal_output_plan,
    )

    request = covariant_cauchy_request((channel(1, 0), channel(2, 1)), (2, 1), target_L=1)
    compiled = compile_covariant_cauchy(request)
    plan = covariant_cauchy_orthogonal_output_plan(compiled)
    report = plan["validation_report"]
    assert report["gram_independent_of_M_exact"] and report["gram_diagonal_exact"]
    transform = _binary_matrix(plan["transform"])
    sectors = [
        (tuple(map(tuple, item["label"]["block_kappas"])), tuple(item["label"]["block_Lambdas"]))
        for item in compiled["descriptors"]
    ]
    for left, right in np.argwhere(np.abs(transform) > 0.0):
        assert sectors[left] == sectors[right]
    # Orthogonal coordinates stay covariant because R never touches M.
    rng = np.random.default_rng(23)
    positions = rng.normal(size=(6, 3))
    rotation = random_rotation(rng)
    wigner = wigner_real(1, rotation, rng)
    reference, _ = evaluate_covariant_cauchy(compiled, densities(request, positions))
    rotated, _ = evaluate_covariant_cauchy(
        compiled, densities(request, positions @ rotation.T)
    )
    assert np.allclose(
        transform.real @ rotated, (transform.real @ reference) @ wigner.T, atol=1.0e-10
    )


def test_validator_rejects_a_tampered_artifact():
    from ye3t.couplings.covariant_cauchy import validate_covariant_cauchy

    compiled = compile_covariant_cauchy(
        covariant_cauchy_request((channel(1, 0),), (2,), target_L=2)
    )
    assert validate_covariant_cauchy(compiled)
    tampered = dict(compiled)
    tampered["descriptors"] = compiled["descriptors"][:-1]
    with pytest.raises(ValueError, match="self hash"):
        validate_covariant_cauchy(tampered)
