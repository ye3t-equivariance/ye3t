"""Frozen shell-filtered radial family: Gram identities, orientation, nesting, catalogue binding."""

import copy

import pytest
import torch

from ye3t.couplings import (
    compile, count, is_shell_filtered_family, plan, shell_filtered_radial_values,
    shell_filtered_source_plan, tagged_cauchy_carrier_model_plan, tagged_cauchy_carriers_request,
    tagged_cauchy_marginal_graph_records, validate_shell_filtered_source_plan,
)
from ye3t.couplings.shell_filtered_radial import _gram, shell_filtered_raw_values, shell_windows


SHELLS = {"kind": "softmax_gaussian", "centers": [0.25, 0.75], "width": 0.25}
FAMILY = "shell_filtered_chebyshev_c2_v1"


@pytest.mark.parametrize("coordinates", ["physical_shells", "orthogonal_channels"])
def test_empty_neighbor_radial_arrays_preserve_width_and_derivatives(coordinates):
    payload = shell_filtered_source_plan(FAMILY, 2, 2, SHELLS, evaluation_coordinates=coordinates)
    x = torch.empty(0, dtype=torch.float64, requires_grad=True)
    values = shell_filtered_radial_values(payload, x, 2)
    assert values.shape == (4, 0)
    derivative = torch.autograd.grad(values.sum(), x)[0]
    assert derivative.shape == (0,)


def test_weighted_qr_resolves_requested_shell_bank_without_changing_span_or_tolerance():
    from ye3t.couplings.shell_filtered_radial import _quadrature
    shells = {"kind": "softmax_gaussian", "centers": [.54, .77, .94], "width": .1}
    payload = shell_filtered_source_plan(FAMILY, 4, 4, shells, evaluation_coordinates="physical_shells")
    assert payload["orthogonalization"]["tolerance"] == 1e-8
    assert payload["radial_count"] == 4 and payload["shell_count"] == 3
    report = validate_shell_filtered_source_plan(payload, recompute=True)
    assert report["passed"] and report["independent_radial_channels_per_l"] == 12
    nodes, weights = _quadrature(1024)
    for ell in range(5):
        transform = _transform(payload, ell)
        assert payload["diagnostics"][str(ell)]["transform_construction"] == "weighted_unpivoted_qr"
        torch.testing.assert_close(transform, transform.tril(), rtol=0, atol=0)
        assert bool((torch.diagonal(transform) > 0).all())
        raw = shell_filtered_raw_values(payload, nodes, ell)
        values = transform @ raw * (weights * nodes**(2*ell+2)).sqrt()
        error = torch.linalg.matrix_norm(values @ values.T-torch.eye(12), ord=2)
        assert error < payload["orthogonalization"]["tolerance"]
        torch.testing.assert_close(shell_filtered_radial_values(payload, nodes, ell), raw, rtol=0, atol=0)


def test_weighted_qr_and_cholesky_have_the_same_ordered_gauge():
    from ye3t.couplings.shell_filtered_radial import _weighted_sample_transform
    payload = shell_filtered_source_plan(FAMILY, 2, 1, SHELLS)
    for ell in (0, 1):
        actual = _weighted_sample_transform(payload, ell, payload["orthogonalization"])
        torch.testing.assert_close(actual, _transform(payload, ell), rtol=3e-11, atol=1e-10)


@pytest.mark.parametrize("factorization", ["cholesky", "lowdin"])
def test_physical_occupation_uses_inverse_coordinate_transport_with_derivatives(factorization):
    options = {"factorization": factorization}
    orthogonal = shell_filtered_source_plan(FAMILY, 2, 2, SHELLS, options)
    physical = shell_filtered_source_plan(FAMILY, 2, 2, SHELLS, options,
                                           evaluation_coordinates="physical_shells")
    assert orthogonal["family_id"] != physical["family_id"]
    assert orthogonal["transforms"] == physical["transforms"]
    assert validate_shell_filtered_source_plan(physical, recompute=True)["passed"]
    x = torch.tensor([.13, .37, .68, .91], dtype=torch.float64, requires_grad=True)
    for ell in (0, 1, 2):
        transformed = shell_filtered_radial_values(orthogonal, x, ell)
        reconstructed = torch.linalg.solve(_transform(orthogonal, ell), transformed)
        direct = shell_filtered_radial_values(physical, x, ell)
        torch.testing.assert_close(direct, reconstructed, rtol=2e-12, atol=2e-13)
        # A physical (3,1) occupation is transported as the full subspace,
        # even when T mixes every shell/radial coordinate.
        actual, expected = direct[0]**3 * direct[2], reconstructed[0]**3 * reconstructed[2]
        derivatives = [torch.autograd.grad(row.sum(), x, create_graph=True)[0] for row in (actual, expected)]
        torch.testing.assert_close(*derivatives, rtol=1e-11, atol=1e-13)
        second = [torch.autograd.grad(row.sum(), x)[0] for row in derivatives]
        torch.testing.assert_close(*second, rtol=1e-10, atol=1e-12)
        assert (actual - transformed[0]**3 * transformed[2]).abs().max() > 1e-5


def test_physical_shell_catalogue_requires_matching_coordinate_choice():
    physical = shell_filtered_source_plan(FAMILY, 1, 1, SHELLS, evaluation_coordinates="physical_shells")
    catalogue = {"ranks": [2], "nmax": 1, "lmax": 1, "input_Lmax": 1, "tag_counts": [0],
                 "source_family_id": physical["family_id"], "source_family_plan": physical,
                 "shell_occupancies_by_rank": {2: [[1, 1]]}, "shell_occupation_coordinates": "physical_shells"}
    request = tagged_cauchy_carriers_request(species=["Ni"], catalogue=catalogue)
    assert count(request)["graph_inventory"]
    catalogue["shell_occupation_coordinates"] = "orthogonal_channels"
    with pytest.raises(ValueError, match="occupation coordinates"):
        tagged_cauchy_carriers_request(species=["Ni"], catalogue=catalogue)


def _transform(plan_payload, angular_l):
    return torch.tensor(plan_payload["transforms"][str(angular_l)], dtype=torch.float64)


@pytest.mark.parametrize("family", [FAMILY, "shell_filtered_shifted_jacobi_v1"])
def test_joint_ordered_transform_is_lower_triangular_and_orthonormal_on_a_third_rule(family):
    payload = shell_filtered_source_plan(family, 2, 1, SHELLS)
    assert is_shell_filtered_family(payload["family_id"]) and payload["family_id"].startswith(family + ":")
    assert payload["shell_count"] == 2 and payload["radial_count"] == 2
    for angular_l in (0, 1):
        transform = _transform(payload, angular_l)
        torch.testing.assert_close(transform, transform.tril(), rtol=0, atol=0)
        gram = _gram(payload, angular_l, 1024)
        assert torch.linalg.matrix_norm(transform @ gram @ transform.T - torch.eye(4, dtype=torch.float64), ord=2) < 1e-10
        diagnostics = payload["diagnostics"][str(angular_l)]
        assert diagnostics["identity_residual_check"] < 1e-10 and diagnostics["cross_shell_overlap_check"] == 0.
        assert diagnostics["equilibrated_condition_number"] > 1.
    report = validate_shell_filtered_source_plan(payload, recompute=True)
    assert report["passed"] and set(report["residuals"]) == {"0", "1"}
    assert report["source_independence"] == "distinct_exponents_and_polynomial_degrees"
    assert report["independent_radial_channels_per_l"] == 4
    corrupted = copy.deepcopy(payload)
    corrupted["transforms"]["0"][0][0] += 1e-6
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_shell_filtered_source_plan(corrupted)


@pytest.mark.parametrize("factorization", ["cholesky", "lowdin"])
def test_rehashed_singular_source_transform_fails_before_materialization(factorization):
    from ye3t.couplings.shell_filtered_radial import _stable_hash
    payload = shell_filtered_source_plan(FAMILY, 2, 0, SHELLS, {"factorization": factorization})
    payload["transforms"]["0"][1] = list(payload["transforms"]["0"][0])
    body = {key: value for key, value in payload.items() if key not in {"plan_hash", "family_id"}}
    payload["plan_hash"] = _stable_hash(body)
    payload["family_id"] = payload["family"] + ":" + payload["plan_hash"][:16]
    with pytest.raises(ValueError, match="singular"):
        validate_shell_filtered_source_plan(payload)


def test_orientation_samples_in_rows_use_transpose_and_pullback_is_the_transpose():
    payload = shell_filtered_source_plan(FAMILY, 2, 1, SHELLS)
    transform = _transform(payload, 1)
    assert not torch.allclose(transform.T, torch.linalg.inv(transform))
    x = torch.linspace(0.05, 0.95, 7, dtype=torch.float64, requires_grad=True)
    raw = shell_filtered_raw_values(payload, x, 1)
    transformed = shell_filtered_radial_values(payload, x, 1)
    torch.testing.assert_close(transformed, transform @ raw, rtol=1e-13, atol=1e-14)
    torch.testing.assert_close(transformed.T, raw.T @ transform.T, rtol=1e-13, atol=1e-14)
    covector = torch.randn(4, 7, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    pulled = torch.autograd.grad((transformed * covector).sum(), x, retain_graph=True)[0]
    reference = torch.autograd.grad((raw * (transform.T @ covector)).sum(), x)[0]
    torch.testing.assert_close(pulled, reference, rtol=1e-12, atol=1e-13)
    finite = torch.zeros_like(pulled)
    with torch.no_grad():
        for index in range(len(x)):
            step = torch.zeros_like(x)
            step[index] = 1e-6
            finite[index] = ((shell_filtered_radial_values(payload, x + step, 1) * covector).sum()
                             - (shell_filtered_radial_values(payload, x - step, 1) * covector).sum()) / 2e-6
    torch.testing.assert_close(pulled, finite, rtol=1e-6, atol=1e-8)
    windows = shell_windows(x.detach(), payload["shells"])
    torch.testing.assert_close(windows.sum(-1), torch.ones(7, dtype=torch.float64), rtol=0, atol=1e-14)


def test_ordered_prefixes_nest_under_an_appended_shell_and_lowdin_is_symmetric():
    # Raw Gaussian windows leave earlier functions unchanged when a shell is
    # appended, so the ordered transform of the prefix is nested. Softmax
    # windows renormalize every shell and have no such nesting.
    raw = {"kind": "gaussian", "centers": [0.25, 0.75], "width": 0.25}
    two = shell_filtered_source_plan(FAMILY, 2, 0, raw)
    three = shell_filtered_source_plan(FAMILY, 2, 0, {**raw, "centers": [0.25, 0.75, 0.95]})
    torch.testing.assert_close(_transform(three, 0)[:4, :4], _transform(two, 0), rtol=1e-11, atol=1e-12)
    assert two["family_id"] != three["family_id"]
    softmax_three = shell_filtered_source_plan(FAMILY, 2, 0, {**SHELLS, "centers": [0.25, 0.75, 0.95]})
    assert not torch.allclose(_transform(softmax_three, 0)[:4, :4],
                              _transform(shell_filtered_source_plan(FAMILY, 2, 0, SHELLS), 0), atol=1e-6)
    lowdin = shell_filtered_source_plan(FAMILY, 2, 0, SHELLS, {"factorization": "lowdin"})
    transform = _transform(lowdin, 0)
    gram = torch.tensor(lowdin["gram"]["0"], dtype=torch.float64)
    torch.testing.assert_close(transform, transform.T, rtol=1e-11, atol=1e-12)
    assert bool((torch.linalg.eigvalsh(transform) > 0).all())
    assert torch.linalg.matrix_norm(transform @ gram @ transform.T - torch.eye(4, dtype=torch.float64), ord=2) < 1e-10
    assert lowdin["orthogonalization"]["convention"] == "symmetric"
    lowdin_three = shell_filtered_source_plan(FAMILY, 2, 0, {**SHELLS, "centers": [0.25, 0.75, 0.95]},
                                              {"factorization": "lowdin"})
    assert not torch.allclose(_transform(lowdin_three, 0)[:4, :4], transform, atol=1e-6)


def test_lowdin_matches_closed_form_of_an_unequally_scaled_two_function_gram():
    from ye3t.couplings.shell_filtered_radial import _block_transform
    gram = torch.tensor([[4., 1.], [1., 1.]], dtype=torch.float64)
    # Cayley-Hamilton gives sqrt(G) = (G + sqrt(det(G)) I) /
    # sqrt(tr(G) + 2 sqrt(det(G))) for a positive definite 2x2 matrix.
    root_det = torch.linalg.det(gram).sqrt()
    identity = torch.eye(2, dtype=torch.float64)
    expected = torch.linalg.solve(gram + root_det * identity, identity)
    expected *= (gram.trace() + 2 * root_det).sqrt()
    actual, _ = _block_transform(gram, "lowdin")
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-14)
    torch.testing.assert_close(actual, actual.T, rtol=0, atol=1e-14)


def test_shell_local_scope_orthonormalizes_blocks_and_reports_cross_shell_overlap():
    payload = shell_filtered_source_plan(FAMILY, 2, 0, SHELLS, {"scope": "shell_local"})
    transform = _transform(payload, 0)
    assert transform[:2, 2:].abs().max() == 0 and transform[2:, :2].abs().max() == 0
    identity = transform @ torch.tensor(payload["gram"]["0"], dtype=torch.float64) @ transform.T
    torch.testing.assert_close(identity[:2, :2], torch.eye(2, dtype=torch.float64), rtol=0, atol=1e-10)
    torch.testing.assert_close(identity[2:, 2:], torch.eye(2, dtype=torch.float64), rtol=0, atol=1e-10)
    assert payload["diagnostics"]["0"]["cross_shell_overlap_check"] > 0.5


def test_dependent_or_malformed_windows_fail_without_silent_reduction():
    with pytest.raises(ValueError, match="strictly increasing"):
        shell_filtered_source_plan(FAMILY, 2, 0, {**SHELLS, "centers": [0.5, 0.5]})
    with pytest.raises(ArithmeticError):
        shell_filtered_source_plan(FAMILY, 2, 0, {"kind": "gaussian", "centers": [0.5, 0.5 + 1e-13], "width": 0.25})
    with pytest.raises(ArithmeticError):
        shell_filtered_source_plan(FAMILY, 2, 0, {"kind": "gaussian", "centers": [0.3, 0.7], "width": 1e4})
    with pytest.raises(ValueError, match="scope"):
        shell_filtered_source_plan(FAMILY, 2, 0, SHELLS, {"scope": "global"})
    with pytest.raises(ValueError, match="Unknown shell-filtered"):
        shell_filtered_source_plan("shell_filtered_unknown_v1", 2, 0, SHELLS)


def _catalogue(payload, **overrides):
    catalogue = {"ranks": [1, 2], "nmax": 2, "lmax": 1, "input_Lmax": 1, "max_source_blocks": 2,
                 "tag_counts": [0], "source_family_id": payload["family_id"], "source_family_plan": payload,
                 "shell_occupancies_by_rank": {1: [[1, 0], [0, 1]], 2: [[1, 1]]}}
    catalogue.update(overrides)
    return catalogue


def test_catalogue_enumerates_shell_decorated_channels_filters_occupancies_and_binds_the_plan(tmp_path):
    payload = shell_filtered_source_plan(FAMILY, 2, 1, SHELLS)
    request = tagged_cauchy_carriers_request(catalogue=_catalogue(payload), species=["Ni"])
    assert request["catalogue"]["source_family_plan"]["plan_hash"] == payload["plan_hash"]
    assert request["catalogue"]["shell_occupancies_by_rank"] == {"1": ((0, 1), (1, 0)), "2": ((1, 1),)}
    report = count(request)
    by_rank = {}
    for record in report["candidate_records"]:
        channels = record["request"]["channels"]
        occupancy = tuple(sum(size for channel, size in zip(channels, record["pattern"], strict=True)
                              if channel["radial_channel"] // 2 == shell) for shell in range(2))
        by_rank.setdefault(record["rank"], []).append(occupancy)
        assert all(0 <= channel["radial_channel"] < 4 for channel in channels)
    # Rank one: two shells, two radial functions, l = 0, 1. Rank two: one
    # channel from each shell (4 x 4 ordered pairs of distinct channels).
    assert sorted(by_rank[1]) == [(0, 1)] * 4 + [(1, 0)] * 4
    assert by_rank[2] == [(1, 1)] * 16
    unrestricted = tagged_cauchy_carriers_request(
        catalogue={key: value for key, value in _catalogue(payload).items() if key != "shell_occupancies_by_rank"},
        species=["Ni"])
    assert len(count(unrestricted)["candidate_records"]) > len(report["candidate_records"])
    assert unrestricted["request_hash"] != request["request_hash"]
    assert tagged_cauchy_carriers_request(catalogue=_catalogue(payload), species=["Ni"])["request_hash"] == request["request_hash"]
    wider = shell_filtered_source_plan(FAMILY, 2, 1, {**SHELLS, "width": 0.3})
    assert tagged_cauchy_carriers_request(catalogue=_catalogue(wider), species=["Ni"])["request_hash"] != request["request_hash"]
    compiled = compile(plan(report), cache_dir=str(tmp_path))
    model_plan = tagged_cauchy_carrier_model_plan(compiled, hidden_channels=2)
    assert {schedule["tag_count"] for schedule in model_plan["schedules"]} == {0}
    assert all(schedule["radial_source"]["family_id"] == payload["family_id"] for schedule in model_plan["schedules"])
    graph = tagged_cauchy_marginal_graph_records(compiled, target_L=0, parity=1, hidden_channels=2,
        output_sectors=[{"L": 0, "parity": "even"}, {"L": 1, "parity": "odd"}], support_realization="ordered_tags")
    assert graph["source_smoothness"]["radial_family"] == payload["family_id"]
    assert all(row["binding"]["source_realization"]["metadata"]["radial_family"] == payload["family_id"]
               for row in graph["records"])
    assert {row["rank"] for row in graph["records"]} == {1, 2}
    # Different shell/angular channels support an odd vector at rank two.
    assert any(row["rank"] == 2 and row["binding"]["output_layouts"][row["id"]]["key"]["rotation_L"] == 1
               for row in graph["records"])
    with pytest.raises(ValueError, match="edge-marginal"):
        tagged_cauchy_marginal_graph_records(compiled, target_L=0, parity=1, support_realization="edge_marginal")


def test_catalogue_rejects_missing_plan_mismatched_caps_and_plans_on_polynomial_families():
    payload = shell_filtered_source_plan(FAMILY, 2, 1, SHELLS)
    with pytest.raises(ValueError, match="frozen source_family_plan"):
        tagged_cauchy_carriers_request(catalogue={key: value for key, value in _catalogue(payload).items()
                                                  if key != "source_family_plan"}, species=["Ni"])
    with pytest.raises(ValueError, match="exceed the frozen"):
        tagged_cauchy_carriers_request(catalogue=_catalogue(payload, nmax=3), species=["Ni"])
    with pytest.raises(ValueError, match="summing to its rank"):
        tagged_cauchy_carriers_request(catalogue=_catalogue(payload, shell_occupancies_by_rank={1: [[1, 1]], 2: [[1, 1]]}),
                                       species=["Ni"])
    with pytest.raises(ValueError, match="apply only to shell-filtered"):
        tagged_cauchy_carriers_request(catalogue={"ranks": [1], "nmax": 2, "lmax": 1, "input_Lmax": 1, "tag_counts": [0],
            "source_family_id": "chebyshev_c2_origin_regular_v1", "shell_occupancies_by_rank": {1: [[1]]}}, species=["Ni"])
    polynomial = tagged_cauchy_carriers_request(catalogue={"ranks": [1], "nmax": 2, "lmax": 1, "input_Lmax": 1,
        "tag_counts": [0], "source_family_id": "chebyshev_c2_origin_regular_v1"}, species=["Ni"])
    assert "source_family_plan" not in polynomial["catalogue"] and "shell_occupancies_by_rank" not in polynomial["catalogue"]
