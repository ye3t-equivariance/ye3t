"""Ordered role Cauchy coordinates, copy axes, and factor symmetries."""

import json
from itertools import product
import numpy as np
import pytest
import sympy as sp
import torch

from ye3t.couplings import (
    bind_ordered_role_cauchy_torch, compile, count,
    covariant_cauchy_request, evaluate_covariant_cauchy, plan,
)
from ye3t.couplings.ordered_role_cauchy import validate_ordered_role_cauchy


def channel(l, radial):
    return {"neighbor_species": "X", "radial_channel": radial, "l": l,
            "source_family_id": "ordered_role_test"}


def test_ordered_role_cauchy_full_dimension_identity_rank_two():
    # (W tensor V_1)^{tensor 2} has dimension (2*3)^2 = 36.  The exact
    # Young/O(3) inventory must account for every copy, tableau, and M.
    dimensions = {}
    for parent in ("trivial", "antisymmetric"):
        for L in range(3):
            request = covariant_cauchy_request(
                (channel(1, 0),), (2,), target_L=L, role_dimension=2,
                target_permutation=parent, carrier="ordered_role",
            )
            dimensions[(parent, L)] = count(request)["component_count"]
    assert sum(dimensions.values()) == 36
    assert dimensions[("trivial", 1)] > 0
    assert dimensions[("antisymmetric", 1)] > 0


def test_ordered_role_cauchy_multiblock_dimension_identity():
    # The fixed-content orbit has three placements of a repeated pair.
    # Its full role/angular dimension is 3 * (2*3)^3 = 648.
    tableau_dimensions = {
        "trivial": 1, "young:2,1": 2, "antisymmetric": 1,
    }
    component_count = 0
    for parent, tableau_dimension in tableau_dimensions.items():
        for L in range(4):
            report = count(covariant_cauchy_request(
                (channel(1, 0), channel(1, 1)), (2, 1),
                target_L=L, role_dimension=2,
                target_permutation=parent, carrier="ordered_role",
            ))
            assert report["tableau_count"] == tableau_dimension
            component_count += report["component_count"]
    assert component_count == 648


def test_ordered_role_cauchy_all_route_isometry_on_full_small_orbit():
    request = covariant_cauchy_request(
        (channel(1, 0), channel(1, 1)), (2, 1),
        target_L=1, role_dimension=1,
        target_permutation="young:2,1", carrier="ordered_role",
    )
    compiled = compile(plan(count(request)))
    raw_states = tuple(product(range(3), repeat=3))
    columns = []
    for word in ((0, 0, 1), (0, 1, 0), (1, 0, 0)):
        factors = torch.zeros((len(raw_states), 3, 1, 3), dtype=torch.complex128)
        for row, state in enumerate(raw_states):
            for factor, magnetic in enumerate(state):
                factors[row, factor, 0, magnetic] = 1
        values, _ = evaluate_covariant_cauchy(
            compiled, factors, factor_types=word,
            basis="complex_condon_shortley",
        )
        columns.append(values.reshape(len(raw_states), -1))
    analysis = torch.cat(columns, dim=0)
    identity = torch.eye(analysis.shape[1], dtype=analysis.dtype)
    torch.testing.assert_close(analysis.conj().T @ analysis, identity,
                               atol=1e-11, rtol=1e-11)
    projector = analysis @ analysis.conj().T
    torch.testing.assert_close(projector @ projector, projector,
                               atol=1e-11, rtol=1e-11)


def test_ordered_role_cauchy_cross_family_whitening_survives_runtime():
    from ye3t.couplings.lifted_cauchy_scalar import _exact_matrix_from_payload

    request = covariant_cauchy_request(
        (channel(1, 0),), (3,), target_L=1, role_dimension=3,
        target_permutation="trivial", carrier="ordered_role",
    )
    compiled = compile(plan(count(request)))
    mixed = next(table for table in compiled["local_tables"]
                 if tuple(table["key"][1]) == (3,)
                 and tuple(table["key"][2]) == (2, 1)
                 and tuple(table["key"][3]) == (2, 1))
    assert mixed["strategy"] == "role_angular_network"
    assert not mixed["entries_by_family"]
    gram = _exact_matrix_from_payload(mixed["copy_gram"])
    assert gram.rows == 8 and gram[2, 4] == gram[4, 2] == sp.Rational(1, 2)
    raw_states = tuple(product(range(9), repeat=3))
    factors = torch.zeros((len(raw_states), 3, 3, 3), dtype=torch.complex128)
    for row, state in enumerate(raw_states):
        for factor, coordinate in enumerate(state):
            factors[row, factor, coordinate // 3, coordinate % 3] = 1
    runtime = bind_ordered_role_cauchy_torch(
        compiled, dtype=factors.dtype, device=factors.device,
        basis="complex_condon_shortley",
    )
    values, _ = runtime.evaluate(factors)
    analysis = values.reshape(len(raw_states), -1)
    torch.testing.assert_close(
        analysis.conj().T @ analysis,
        torch.eye(analysis.shape[1], dtype=analysis.dtype),
        atol=1e-11, rtol=1e-11,
    )
    restored = json.loads(json.dumps(compiled))
    assert validate_ordered_role_cauchy(restored)
    replay, _ = evaluate_covariant_cauchy(
        restored, factors, basis="complex_condon_shortley",
    )
    torch.testing.assert_close(replay, values, atol=1e-12, rtol=1e-12)


def test_factorized_cauchy_local_network_matches_explicit_vectors():
    from ye3t.couplings.covariant_cauchy import compile_slot_resolved_cauchy_block

    arguments = (2, 2, (2,), (2,), (2,), 1, 0)
    network = compile_slot_resolved_cauchy_block(
        *arguments, factorized_only=True,
    )
    explicit = compile_slot_resolved_cauchy_block(
        *arguments, factorized_only=False,
    )
    assert network["validation_report"]["joint_role_angular_tensor_not_materialized"]
    assert tuple(network["family_order"]) == tuple(dict.fromkeys(
        key[:3] for key in explicit["orthonormal_vectors"]
    ))
    roles = tuple(product(range(2), repeat=2))
    magnetic = tuple(product(range(-1, 2), repeat=2))
    role_rows = {state: row for row, state in enumerate(roles)}
    magnetic_rows = {state: row for row, state in enumerate(magnetic)}
    for copy, family in enumerate(network["family_order"]):
        for row, state in enumerate(explicit["joint_states"]):
            role_state, magnetic_state = tuple(zip(*state))
            role_index = role_rows[role_state]
            angular_index = magnetic_rows[magnetic_state]
            for M_index, M in enumerate(range(-arguments[-1], arguments[-1] + 1)):
                actual = sum(
                    network["role_values"][r][role_index]
                    * network["angular_values"][a][angular_index][M_index]
                    * network["orthogonal_kernel"][copy][r][a][0]
                    for r in range(len(network["role_values"]))
                    for a in range(len(network["angular_values"]))
                )
                expected = float(explicit["orthonormal_vectors"][
                    (*family, 0, M)
                ][row])
                assert abs(actual - expected) < 1e-12


def test_large_ordered_role_network_retains_general_young_action_and_gradients():
    from ye3t.representations.projectors import (
        adjacent_transposition_representation_matrix_numeric,
    )

    request = covariant_cauchy_request(
        (channel(1, 0),), (3,), target_L=1, role_dimension=3,
        target_permutation="young:2,1", carrier="ordered_role",
    )
    compiled = compile(plan(count(request)))
    assert all(table["strategy"] == "role_angular_network"
               for table in compiled["local_tables"])
    factors = torch.tensor([
        [[0.2, 0.4, -0.1], [0.3, 0.1, 0.7], [-0.4, 0.3, 0.1]],
        [[-0.3, 0.1, 0.7], [0.6, -0.2, 0.1], [0.2, 0.5, 0.3]],
        [[0.8, -0.2, 0.5], [-0.1, 0.4, 0.3], [0.6, 0.2, -0.5]],
    ], dtype=torch.complex128, requires_grad=True)
    runtime = bind_ordered_role_cauchy_torch(
        compiled, dtype=factors.dtype, device=factors.device,
        basis="complex_condon_shortley",
    )
    base, _ = runtime.evaluate(factors)
    direct, _ = evaluate_covariant_cauchy(
        compiled, factors, basis="complex_condon_shortley",
    )
    torch.testing.assert_close(base, direct, atol=1e-12, rtol=1e-12)
    assert base.shape == (count(request)["multiplet_count"], 2, 3)
    moved, _ = runtime.evaluate(factors[[1, 0, 2]])
    action = torch.as_tensor(
        adjacent_transposition_representation_matrix_numeric((2, 1), 0),
        dtype=factors.dtype,
    )
    torch.testing.assert_close(
        moved, torch.einsum("tu,aum->atm", action, base),
        atol=1e-11, rtol=1e-11,
    )
    gradient = torch.autograd.grad(base.abs().square().sum(), factors)[0]
    assert torch.isfinite(gradient).all()


def test_bound_ordered_role_network_cuda_matches_cpu():
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    request = covariant_cauchy_request(
        (channel(1, 0),), (3,), target_L=1, role_dimension=3,
        target_permutation="young:2,1", carrier="ordered_role",
    )
    compiled = compile(plan(count(request)))
    factors = torch.randn(3, 3, 3, 3, dtype=torch.float64)
    cpu = factors.clone().requires_grad_(True)
    gpu = factors.cuda().requires_grad_(True)
    cpu_runtime = bind_ordered_role_cauchy_torch(
        compiled, dtype=cpu.dtype, device=cpu.device, basis="real_tesseral",
    )
    gpu_runtime = bind_ordered_role_cauchy_torch(
        compiled, dtype=gpu.dtype, device=gpu.device, basis="real_tesseral",
    )
    cpu_output, _ = cpu_runtime.evaluate(cpu)
    gpu_output, _ = gpu_runtime.evaluate(gpu)
    torch.testing.assert_close(gpu_output.cpu(), cpu_output,
                               atol=1e-11, rtol=1e-11)
    cpu_gradient = torch.autograd.grad(cpu_output.square().sum(), cpu)[0]
    gpu_gradient = torch.autograd.grad(gpu_output.square().sum(), gpu)[0]
    torch.testing.assert_close(gpu_gradient.cpu(), cpu_gradient,
                               atol=1e-10, rtol=1e-10)


def test_mixed_block_network_retains_all_young_and_angular_routes():
    from ye3t.representations.projectors import (
        adjacent_transposition_representation_matrix_numeric,
    )

    request = covariant_cauchy_request(
        (channel(1, 0), channel(1, 1)), (3, 1),
        target_L=1, role_dimension=3,
        target_permutation="young:3,1", carrier="ordered_role",
    )
    report = count(request)
    compiled = compile(plan(report))
    assert any(table["strategy"] == "role_angular_network"
               for table in compiled["local_tables"])
    assert len(compiled["routes"]) == report["multiplet_count"]
    factors = torch.randn(2, 4, 3, 3, dtype=torch.complex128)
    runtime = bind_ordered_role_cauchy_torch(
        compiled, dtype=factors.dtype, device=factors.device,
        basis="complex_condon_shortley",
    )
    values, _ = runtime.evaluate(factors)
    assert values.shape == (2, report["multiplet_count"], 3, 3)
    moved, _ = runtime.evaluate(factors[:, [1, 0, 2, 3]])
    action = torch.as_tensor(
        adjacent_transposition_representation_matrix_numeric((3, 1), 0),
        dtype=factors.dtype,
    )
    torch.testing.assert_close(
        moved, torch.einsum("tu,...aum->...atm", action, values),
        atol=1e-10, rtol=1e-10,
    )


def test_cauchy_request_accepts_standard_parent_and_intermediate_spellings():
    legacy = covariant_cauchy_request(
        (channel(1, 0),), (2,), target_L=0, role_dimension=2,
    )
    standardized = covariant_cauchy_request(
        (channel(1, 0),), (2,), target_L=0, role_dimension=2,
        target_permutation="(N)", kappa_policy="all_valid",
    )
    assert standardized == legacy


@pytest.mark.parametrize("basis,dtype", (
    ("real_tesseral", torch.float64),
    ("complex_condon_shortley", torch.complex128),
))
def test_bound_ordered_role_runtime_matches_unbound_values_and_gradients(basis, dtype):
    request = covariant_cauchy_request(
        (channel(1, 0), channel(1, 1)), (2, 1),
        target_L=1, role_dimension=2,
        target_permutation="young:2,1", carrier="ordered_role",
    )
    compiled = compile(plan(count(request)))
    factors = torch.tensor([
        [[0.2, 0.4, -0.1], [0.3, 0.1, 0.7]],
        [[-0.3, 0.1, 0.7], [0.6, -0.2, 0.1]],
        [[0.8, -0.2, 0.5], [-0.1, 0.4, 0.3]],
    ], dtype=dtype, requires_grad=True)
    runtime = bind_ordered_role_cauchy_torch(
        compiled, dtype=dtype, device=factors.device, basis=basis,
    )
    direct, _ = evaluate_covariant_cauchy(compiled, factors, basis=basis)
    actual, _ = runtime.evaluate(factors)
    torch.testing.assert_close(actual, direct, atol=1e-12, rtol=1e-12)
    seed = torch.ones_like(actual)
    grad_bound = torch.autograd.grad(actual, factors, seed, retain_graph=True)[0]
    grad_direct = torch.autograd.grad(direct, factors, seed)[0]
    torch.testing.assert_close(grad_bound, grad_direct,
                               atol=1e-12, rtol=1e-12)
    with pytest.raises(ValueError, match="Bound Cauchy tensors"):
        runtime.evaluate(factors.to(torch.float32 if dtype == torch.float64
                                    else torch.complex64))


@pytest.mark.parametrize("parent,L,sign", (
    ("trivial", 0, 1), ("antisymmetric", 1, -1),
))
def test_ordered_role_cauchy_one_block_swap_and_vjp(parent, L, sign):
    request = covariant_cauchy_request(
        (channel(1, 0),), (2,), target_L=L, role_dimension=2,
        target_permutation=parent, carrier="ordered_role",
    )
    report = count(request)
    compiled = compile(plan(report))
    assert report["multiplet_count"] == compiled["multiplet_count"] > 0
    assert report["component_count"] == (
        report["multiplet_count"] * report["tableau_count"] * (2 * L + 1)
    )
    assert compiled["validation_report"]["full_orbit_matrix_not_materialized"]
    factors = torch.tensor(
        [[[0.2, 0.6, -0.1], [0.7, -0.2, 0.3]],
         [[-0.4, 0.1, 0.8], [0.5, 0.9, -0.3]]],
        dtype=torch.float64, requires_grad=True,
    )
    base, _ = evaluate_covariant_cauchy(compiled, factors)
    swapped, _ = evaluate_covariant_cauchy(compiled, factors[[1, 0]])
    torch.testing.assert_close(swapped, sign * base, atol=1e-11, rtol=1e-11)
    assert base.shape == (report["multiplet_count"], 1, 2 * L + 1)
    assert torch.autograd.gradcheck(
        lambda x: evaluate_covariant_cauchy(compiled, x)[0],
        (factors,), atol=1e-6, rtol=1e-5,
    )
    assert torch.autograd.gradgradcheck(
        lambda x: evaluate_covariant_cauchy(compiled, x)[0],
        (factors,), atol=1e-6, rtol=1e-5,
    )
    if sign == -1:
        repeated, _ = evaluate_covariant_cauchy(
            compiled, factors[[0, 0]]
        )
        torch.testing.assert_close(repeated, torch.zeros_like(repeated),
                                   atol=1e-12, rtol=0)
    seed = torch.ones_like(base)
    _, gradients = evaluate_covariant_cauchy(
        compiled, factors, upstream=seed
    )
    direct = torch.autograd.grad(base, factors, grad_outputs=seed)[0]
    torch.testing.assert_close(gradients, direct, atol=1e-11, rtol=1e-11)


def test_ordered_role_cauchy_general_young_has_all_copies_and_tableaux():
    request = covariant_cauchy_request(
        (channel(1, 0), channel(1, 1), channel(1, 2)),
        (1, 1, 1), target_L=1, role_dimension=1,
        target_permutation="young:2,1", carrier="ordered_role",
    )
    report = count(request)
    compiled = compile(plan(report))
    assert report["tableau_count"] == 2
    assert report["multiplet_count"] == len(compiled["routes"]) > 1
    assert all(row["local_copy_gauge"] ==
               "full_family_cholesky_orthonormal_v1" for row in report["labels"])
    assert all(table["copy_gauge"] ==
               "full_family_cholesky_orthonormal_v1"
               and len(table["family_order"]) == len(table["entries_by_family"])
               for table in compiled["local_tables"])
    assert len({row["young_copy"] for row in report["labels"]}) == 2
    factors = torch.tensor(
        [[[[0.2, 0.3, -0.5]], [[0.7, -0.1, 0.4]], [[0.1, 0.9, -0.2]]]],
        dtype=torch.complex128,
    )
    base, _ = evaluate_covariant_cauchy(
        compiled, factors, basis="complex_condon_shortley"
    )
    assert base.shape == (1, report["multiplet_count"], 2, 3)
    order = (1, 0, 2)
    moved, _ = evaluate_covariant_cauchy(
        compiled, factors[:, order], basis="complex_condon_shortley",
        factor_types=order,
    )
    from ye3t.representations.projectors import (
        adjacent_transposition_representation_matrix_numeric,
    )
    action = torch.as_tensor(
        adjacent_transposition_representation_matrix_numeric((2, 1), 0),
        dtype=torch.complex128,
    )
    torch.testing.assert_close(moved, torch.einsum("tu,...aum->...atm", action, base),
                               atol=1e-11, rtol=1e-11)
    axis = np.array([0.4, -0.7, 0.2])
    axis /= np.linalg.norm(axis)
    cross = np.array([
        [0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    angle = 0.37
    rotation = (np.cos(angle) * np.eye(3) + np.sin(angle) * cross
                + (1 - np.cos(angle)) * np.outer(axis, axis))
    spherical = np.array([
        [1 / np.sqrt(2), -1j / np.sqrt(2), 0],
        [0, 0, 1],
        [-1 / np.sqrt(2), -1j / np.sqrt(2), 0],
    ])
    spin_one = torch.as_tensor(spherical @ rotation @ np.linalg.inv(spherical),
                               dtype=torch.complex128)
    rotated, _ = evaluate_covariant_cauchy(
        compiled, factors @ spin_one.T, basis="complex_condon_shortley"
    )
    torch.testing.assert_close(rotated, base @ spin_one.T,
                               atol=1e-11, rtol=1e-11)
    inverted, _ = evaluate_covariant_cauchy(
        compiled, -factors, basis="complex_condon_shortley"
    )
    torch.testing.assert_close(inverted, -base, atol=1e-11, rtol=1e-11)


def test_ordered_role_cauchy_artifact_roundtrip_and_coefficient_hash():
    request = covariant_cauchy_request(
        (channel(1, 0),), (2,), target_L=0, role_dimension=2,
        target_permutation="trivial", carrier="ordered_role",
    )
    compiled = compile(plan(count(request)))
    restored = json.loads(json.dumps(compiled))
    assert validate_ordered_role_cauchy(restored)
    factors = np.asarray([
        [[0.2, -0.4, 0.1], [0.3, 0.7, -0.2]],
        [[0.6, 0.1, 0.5], [-0.3, 0.2, 0.4]],
    ])
    original, _ = evaluate_covariant_cauchy(compiled, factors)
    replay, _ = evaluate_covariant_cauchy(restored, factors)
    np.testing.assert_allclose(replay, original, atol=1e-12)
    restored["local_tables"][0]["entries_by_family"][0]["terms"][0][2] += 0.1
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_ordered_role_cauchy(restored)


def test_ordered_cauchy_low_rank_projector_matches_role_symmetry_and_spin_singlet():
    from itertools import product

    request = covariant_cauchy_request(
        (channel(1, 0),), (2,), target_L=0, role_dimension=2,
        target_permutation="trivial", carrier="ordered_role",
    )
    compiled = compile(plan(count(request)))
    assert compiled["multiplet_count"] == 3
    raw_states = tuple(product(product(range(2), range(-1, 2)), repeat=2))
    raw_index = {tuple(role * 3 + m + 1 for role, m in state): index
                 for index, state in enumerate(raw_states)}
    independent = np.zeros((len(raw_states), 3))
    for row, state in enumerate(raw_states):
        (role0, m0), (role1, m1) = state
        if m0 + m1:
            continue
        angular = (-1) ** (1 - m0) / np.sqrt(3)
        if role0 == role1:
            independent[row, role0 * 2] = angular
        else:
            independent[row, 1] = angular / np.sqrt(2)
    np.testing.assert_allclose(independent.T @ independent, np.eye(3), atol=1e-12)
    analysis = np.zeros((len(raw_states), compiled["multiplet_count"]))
    for route in compiled["routes"]:
        local = compiled["local_tables"][route["local_indices"][0]]
        choice = route["local_choices"][0]
        family = (choice["role_copy"], choice["angular_copy"],
                  choice["kronecker_copy"])
        terms = next(row["terms"] for row in local["entries_by_family"]
                     if tuple(row["family"]) == family)
        for state, column, coefficient in terms:
            assert column == 0
            analysis[raw_index[tuple(state)], route["multiplicity_index"]] += coefficient
    gram = analysis.T @ analysis
    projector = analysis @ np.linalg.solve(gram, analysis.T)
    np.testing.assert_allclose(projector, independent @ independent.T,
                               atol=1e-11)
    factors = np.asarray([
        [[0.2, -0.4, 0.1], [0.3, 0.7, -0.2]],
        [[0.6, 0.1, 0.5], [-0.3, 0.2, 0.4]],
    ])
    raw = np.asarray([
        factors[0, role0, m0 + 1] * factors[1, role1, m1 + 1]
        for ((role0, m0), (role1, m1)) in raw_states
    ])
    evaluated, _ = evaluate_covariant_cauchy(
        compiled, factors, basis="complex_condon_shortley"
    )
    np.testing.assert_allclose(evaluated.reshape(-1), raw @ analysis,
                               atol=1e-12)


def test_commuting_role_density_rejects_general_parent():
    with pytest.raises(ValueError, match="trivial global Young"):
        covariant_cauchy_request(
            (channel(1, 0),), (2,), target_L=1, role_dimension=2,
            target_permutation="antisymmetric", carrier="A_s",
        )


def test_ordered_role_cauchy_repeated_and_cross_block_permutations():
    from ye3t.representations.projectors import (
        adjacent_transposition_representation_matrix_numeric,
    )

    request = covariant_cauchy_request(
        (channel(1, 0), channel(1, 1)), (2, 1),
        target_L=1, role_dimension=1,
        target_permutation="young:2,1", carrier="ordered_role",
    )
    compiled = compile(plan(count(request)))
    factors = torch.tensor([
        [[0.2, 0.4, -0.1]], [[-0.3, 0.1, 0.7]], [[0.8, -0.2, 0.5]],
    ], dtype=torch.complex128)
    base, _ = evaluate_covariant_cauchy(
        compiled, factors, basis="complex_condon_shortley"
    )
    internal, _ = evaluate_covariant_cauchy(
        compiled, factors[[1, 0, 2]], basis="complex_condon_shortley"
    )
    internal_action = torch.as_tensor(
        adjacent_transposition_representation_matrix_numeric((2, 1), 0),
        dtype=torch.complex128,
    )
    torch.testing.assert_close(
        internal, torch.einsum("tu,aum->atm", internal_action, base),
        atol=1e-11, rtol=1e-11,
    )
    cross, _ = evaluate_covariant_cauchy(
        compiled, factors[[0, 2, 1]], factor_types=(0, 1, 0),
        basis="complex_condon_shortley"
    )
    action = torch.as_tensor(
        adjacent_transposition_representation_matrix_numeric((2, 1), 1),
        dtype=torch.complex128,
    )
    torch.testing.assert_close(cross, torch.einsum("tu,aum->atm", action, base),
                               atol=1e-11, rtol=1e-11)


def test_ordered_role_cauchy_complex_vjp_uses_bilinear_transpose():
    request = covariant_cauchy_request(
        (channel(1, 0),), (1,), target_L=1, role_dimension=1,
        carrier="ordered_role",
    )
    compiled = compile(plan(count(request)))
    factor = torch.tensor([[[0.2 + 0.4j, -0.1 + 0.3j, 0.5 - 0.2j]]],
                          dtype=torch.complex128)
    result, _ = evaluate_covariant_cauchy(
        compiled, factor, basis="complex_condon_shortley"
    )
    seed = torch.full_like(result, 1.0 + 2.0j)
    _, gradient = evaluate_covariant_cauchy(
        compiled, factor, upstream=seed,
        basis="complex_condon_shortley",
    )
    torch.testing.assert_close(gradient.reshape(-1), seed.reshape(-1),
                               atol=1e-12, rtol=1e-12)
