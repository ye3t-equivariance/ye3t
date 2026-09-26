import io
import math

import numpy as np
import pytest

from ye3t.couplings import (
    algebraic_curvature_output_count,
    algebraic_curvature_output_plan,
    algebraic_curvature_pair_index,
    compile_algebraic_curvature_output,
    pack_algebraic_curvature_numpy,
    project_algebraic_curvature_numpy,
    unpack_algebraic_curvature_numpy,
)


def _ordered_tensor_from_wedge(matrix, one_particle_dim):
    n = int(one_particle_dim)
    tensor = np.zeros((n, n, n, n), dtype=np.asarray(matrix).dtype)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            first_sign = 1.0 if i < j else -1.0
            first = algebraic_curvature_pair_index(min(i, j), max(i, j), n)
            for k in range(n):
                for l in range(n):
                    if k == l:
                        continue
                    second_sign = 1.0 if k < l else -1.0
                    second = algebraic_curvature_pair_index(min(k, l), max(k, l), n)
                    tensor[i, j, k, l] = first_sign * second_sign * matrix[first, second]
    return tensor


def _wedge_from_ordered_tensor(tensor):
    n = int(tensor.shape[0])
    pair_dim = math.comb(n, 2)
    matrix = np.zeros((pair_dim, pair_dim), dtype=np.asarray(tensor).dtype)
    for i in range(n):
        for j in range(i + 1, n):
            first = algebraic_curvature_pair_index(i, j, n)
            for k in range(n):
                for l in range(k + 1, n):
                    second = algebraic_curvature_pair_index(k, l, n)
                    matrix[first, second] = tensor[i, j, k, l]
    return matrix


def _matrix_free_exact_s4_y22_projector(tensor):
    from ye3t.representations import Partition
    from ye3t.representations.projectors import (
        all_permutations,
        permutation_cycle_type,
        permute_state_slots,
        symmetric_group_character,
    )

    partition = Partition((2, 2))
    output = np.zeros_like(tensor)
    prefactor = float(partition.dimension) / float(math.factorial(4))
    n = int(tensor.shape[0])
    states = tuple(np.ndindex((n, n, n, n)))
    for permutation in all_permutations(4):
        character = symmetric_group_character(partition, permutation_cycle_type(permutation))
        if not character:
            continue
        weight = prefactor * float(character)
        for state in states:
            target = permute_state_slots(state, (0, 1, 2, 3), permutation)
            output[target] += weight * tensor[state]
    return output


def _wedge_action(orthogonal):
    orthogonal = np.asarray(orthogonal, dtype=np.float64)
    n = int(orthogonal.shape[0])
    pairs = tuple((i, j) for i in range(n) for j in range(i + 1, n))
    action = np.zeros((len(pairs), len(pairs)), dtype=np.float64)
    for row, (i, j) in enumerate(pairs):
        for col, (a, b) in enumerate(pairs):
            action[row, col] = (
                orthogonal[i, a] * orthogonal[j, b]
                - orthogonal[i, b] * orthogonal[j, a]
            )
    return action


def _wedge_signed_permutation(one_particle_indices, one_particle_signs):
    indices = np.asarray(one_particle_indices, dtype=np.int64)
    signs = np.asarray(one_particle_signs, dtype=np.float64)
    n = int(indices.size)
    pairs = tuple((i, j) for i in range(n) for j in range(i + 1, n))
    lookup = {pair: index for index, pair in enumerate(pairs)}
    pair_indices = []
    pair_signs = []
    for left, right in pairs:
        source_left = int(indices[left])
        source_right = int(indices[right])
        orientation = 1.0
        if source_left > source_right:
            source_left, source_right = source_right, source_left
            orientation = -1.0
        pair_indices.append(lookup[(source_left, source_right)])
        pair_signs.append(signs[left] * signs[right] * orientation)
    return np.asarray(pair_indices), np.asarray(pair_signs)


def _list_schedule_reference(one_particle_dim):
    n = int(one_particle_dim)
    pair_dim = math.comb(n, 2)
    free_rows = list(range(pair_dim))
    free_cols = list(range(pair_dim))
    for a, b, c in __import__("itertools").combinations(range(n), 3):
        ab = algebraic_curvature_pair_index(a, b, n)
        ac = algebraic_curvature_pair_index(a, c, n)
        bc = algebraic_curvature_pair_index(b, c, n)
        free_rows.extend((ab, ab, ac))
        free_cols.extend((ac, bc, bc))
    quadruple_rows = []
    quadruple_cols = []
    for a, b, c, d in __import__("itertools").combinations(range(n), 4):
        pairs = (
            (algebraic_curvature_pair_index(a, b, n), algebraic_curvature_pair_index(c, d, n)),
            (algebraic_curvature_pair_index(a, c, n), algebraic_curvature_pair_index(b, d, n)),
            (algebraic_curvature_pair_index(a, d, n), algebraic_curvature_pair_index(b, c, n)),
        )
        quadruple_rows.append(tuple(min(left, right) for left, right in pairs))
        quadruple_cols.append(tuple(max(left, right) for left, right in pairs))
    return (
        np.asarray(free_rows, dtype=np.int32),
        np.asarray(free_cols, dtype=np.int32),
        np.asarray(quadruple_rows, dtype=np.int32).reshape(-1, 3),
        np.asarray(quadruple_cols, dtype=np.int32).reshape(-1, 3),
    )


@pytest.mark.fast
@pytest.mark.parametrize("one_particle_dim", range(0, 17))
def test_algebraic_curvature_count_matches_exact_dimension_identities(one_particle_dim):
    report = algebraic_curvature_output_count(one_particle_dim=one_particle_dim)
    n = int(one_particle_dim)
    pair_dim = math.comb(n, 2)
    assert report["pair_dim"] == pair_dim
    assert report["compact_dim"] == n * n * (n * n - 1) // 12
    assert report["symmetric_pair_dim"] == pair_dim * (pair_dim + 1) // 2
    assert report["four_form_dim"] == math.comb(n, 4)
    assert report["compact_dim"] + report["four_form_dim"] == report["symmetric_pair_dim"]


@pytest.mark.fast
def test_compiled_schedule_is_deterministic_compact_and_serializable():
    first = compile_algebraic_curvature_output(one_particle_dim=8)
    second = compile_algebraic_curvature_output(one_particle_dim=8)
    assert first.convention_hash == second.convention_hash
    assert first.schedule_hash == second.schedule_hash
    assert first.plan.target["young_partition"] == (2, 2)
    assert first.schedule.free_rows.shape == (first.plan.free_coordinate_dim,)
    assert first.schedule.quadruple_rows.shape == (math.comb(8, 4), 3)
    payload = first.to_dict()
    assert payload["plan"]["resource_report"]["ambient_projector_materialized"] is False
    assert payload["schedule"]["resource_report"]["coefficient_values_materialized"] is False
    assert "free_rows" not in payload["schedule"]
    with pytest.raises(ValueError, match="young_partition"):
        algebraic_curvature_output_plan(
            one_particle_dim=8,
            target={"young_partition": (4,)},
        )


@pytest.mark.fast
@pytest.mark.parametrize("one_particle_dim", (0, 1, 2, 4, 7, 9))
def test_preallocated_schedule_preserves_exact_legacy_coordinate_order(one_particle_dim):
    compiled = compile_algebraic_curvature_output(one_particle_dim=one_particle_dim)
    expected = _list_schedule_reference(one_particle_dim)
    actual = (
        compiled.schedule.free_rows,
        compiled.schedule.free_cols,
        compiled.schedule.quadruple_rows,
        compiled.schedule.quadruple_cols,
    )
    for actual_array, expected_array in zip(actual, expected):
        np.testing.assert_array_equal(actual_array, expected_array)


@pytest.mark.fast
@pytest.mark.parametrize("one_particle_dim", (3, 4, 6, 8))
def test_compact_chart_is_frobenius_isometric_and_projector_is_orthogonal(one_particle_dim):
    compiled = compile_algebraic_curvature_output(one_particle_dim=one_particle_dim)
    rng = np.random.default_rng(1821 + int(one_particle_dim))
    compact = rng.normal(size=(3, compiled.plan.compact_dim))
    matrix = unpack_algebraic_curvature_numpy(compact, compiled)
    recovered = pack_algebraic_curvature_numpy(matrix, compiled)
    np.testing.assert_allclose(recovered, compact, atol=2.0e-13, rtol=2.0e-13)
    np.testing.assert_allclose(
        np.sum(matrix * matrix, axis=(-2, -1)),
        np.sum(compact * compact, axis=-1),
        atol=5.0e-13,
        rtol=5.0e-13,
    )
    raw = rng.normal(size=(2, compiled.plan.pair_dim, compiled.plan.pair_dim))
    projected = project_algebraic_curvature_numpy(raw, compiled)
    repeated = project_algebraic_curvature_numpy(projected, compiled)
    np.testing.assert_allclose(repeated, projected, atol=3.0e-13, rtol=3.0e-13)
    residual = raw - projected
    compact_residual = pack_algebraic_curvature_numpy(residual, compiled)
    np.testing.assert_allclose(compact_residual, 0.0, atol=3.0e-13, rtol=0.0)


@pytest.mark.fast
@pytest.mark.parametrize("one_particle_dim", (4, 5, 6, 7, 8))
def test_implicit_chart_matches_matrix_free_exact_s4_y22_projector(one_particle_dim):
    compiled = compile_algebraic_curvature_output(one_particle_dim=one_particle_dim)
    rng = np.random.default_rng(2200 + int(one_particle_dim))
    raw = rng.normal(size=(compiled.plan.pair_dim, compiled.plan.pair_dim))
    pair_symmetric = 0.5 * (raw + raw.T)
    expected_tensor = _matrix_free_exact_s4_y22_projector(
        _ordered_tensor_from_wedge(pair_symmetric, one_particle_dim)
    )
    expected = _wedge_from_ordered_tensor(expected_tensor)
    actual = project_algebraic_curvature_numpy(pair_symmetric, compiled)
    np.testing.assert_allclose(actual, expected, atol=2.0e-12, rtol=2.0e-12)


@pytest.mark.fast
@pytest.mark.parametrize("one_particle_dim", (4, 6, 8))
def test_projector_intertwines_arbitrary_orthogonal_one_particle_actions(one_particle_dim):
    compiled = compile_algebraic_curvature_output(one_particle_dim=one_particle_dim)
    rng = np.random.default_rng(3400 + int(one_particle_dim))
    raw_q = rng.normal(size=(one_particle_dim, one_particle_dim))
    orthogonal, _ = np.linalg.qr(raw_q)
    wedge = _wedge_action(orthogonal)
    raw = rng.normal(size=(compiled.plan.pair_dim, compiled.plan.pair_dim))
    transformed = wedge @ raw @ wedge.T
    left = project_algebraic_curvature_numpy(transformed, compiled)
    right = wedge @ project_algebraic_curvature_numpy(raw, compiled) @ wedge.T
    np.testing.assert_allclose(left, right, atol=2.0e-12, rtol=2.0e-12)


@pytest.mark.fast
def test_torch_runtime_matches_numpy_and_supports_vjp_hvp_and_state_roundtrip():
    torch = pytest.importorskip("torch")
    from ye3t.runtime import YE3TAlgebraicCurvatureOutput

    compiled = compile_algebraic_curvature_output(one_particle_dim=5)
    module = YE3TAlgebraicCurvatureOutput(compiled).double()
    compact = torch.randn(2, compiled.plan.compact_dim, dtype=torch.float64, requires_grad=True)
    torch_matrix = module(compact)
    numpy_matrix = unpack_algebraic_curvature_numpy(compact.detach().numpy(), compiled)
    np.testing.assert_allclose(torch_matrix.detach().numpy(), numpy_matrix, atol=2.0e-13, rtol=2.0e-13)
    recovered = module.pack(torch_matrix)
    torch.testing.assert_close(recovered, compact, atol=2.0e-13, rtol=2.0e-13)
    assert float(module.bianchi_residual(torch_matrix).max().detach()) < 2.0e-15
    assert float(module.hermiticity_residual(torch_matrix).max().detach()) == 0.0

    direction = torch.randn_like(compact)
    first = torch.autograd.grad((torch_matrix.square()).sum(), compact, create_graph=True)[0]
    second = torch.autograd.grad((first * direction).sum(), compact)[0]
    expected_second = 2.0 * direction
    torch.testing.assert_close(second, expected_second, atol=3.0e-12, rtol=3.0e-12)

    state = io.BytesIO()
    torch.save(module.state_dict(), state)
    state.seek(0)
    restored = YE3TAlgebraicCurvatureOutput(compiled).double()
    restored.load_state_dict(torch.load(state, weights_only=True))
    torch.testing.assert_close(restored(compact.detach()), torch_matrix.detach())
    assert restored.backend_report()["convention_hash"] == compiled.convention_hash


@pytest.mark.fast
@pytest.mark.parametrize("one_particle_dim", (4, 5, 7, 8))
def test_compact_signed_permutation_matches_dense_action_and_roundtrip(one_particle_dim):
    torch = pytest.importorskip("torch")
    from ye3t.runtime import YE3TAlgebraicCurvatureOutput

    compiled = compile_algebraic_curvature_output(
        one_particle_dim=one_particle_dim
    )
    rng = np.random.default_rng(18700 + int(one_particle_dim))
    one_particle_indices = rng.permutation(one_particle_dim)
    one_particle_signs = rng.choice((-1.0, 1.0), size=one_particle_dim)
    pair_indices, pair_signs = _wedge_signed_permutation(
        one_particle_indices, one_particle_signs
    )
    inverse_indices = np.empty_like(pair_indices)
    inverse_signs = np.empty_like(pair_signs)
    inverse_indices[pair_indices] = np.arange(len(pair_indices))
    inverse_signs[pair_indices] = pair_signs
    module = YE3TAlgebraicCurvatureOutput(compiled).double()
    report = module.register_signed_permutation(
        "forward", pair_indices, pair_signs, dtype=torch.float64
    )
    module.register_signed_permutation(
        "inverse", inverse_indices, inverse_signs, dtype=torch.float64
    )
    assert report["maximum_bianchi_plane_residual"] < 2.0e-12
    assert report["maximum_orthogonality_residual"] < 2.0e-12

    compact = torch.randn(
        2, module.compact_dim, dtype=torch.float64, requires_grad=True
    )
    matrix = module.unpack(compact)
    selected = torch.as_tensor(pair_indices, dtype=torch.long)
    phases = torch.as_tensor(pair_signs, dtype=torch.float64)
    dense = matrix.index_select(-2, selected).index_select(-1, selected)
    dense = dense * phases.unsqueeze(-1) * phases.unsqueeze(-2)
    expected = module.pack(dense)
    actual = module.apply_signed_permutation(compact, "forward")
    torch.testing.assert_close(actual, expected, atol=4.0e-13, rtol=4.0e-13)
    recovered = module.apply_signed_permutation(actual, "inverse")
    torch.testing.assert_close(recovered, compact, atol=4.0e-13, rtol=4.0e-13)

    direction = torch.randn_like(compact)
    first = torch.autograd.grad(actual.square().sum(), compact, create_graph=True)[0]
    second = torch.autograd.grad((first * direction).sum(), compact)[0]
    torch.testing.assert_close(second, 2.0 * direction, atol=4.0e-12, rtol=4.0e-12)


@pytest.mark.fast
@pytest.mark.parametrize("one_particle_dim", (3, 4, 6, 8))
def test_direct_compact_exterior_density_matches_dense_pack_and_vjp(one_particle_dim):
    torch = pytest.importorskip("torch")
    from ye3t.runtime import YE3TAlgebraicCurvatureOutput

    compiled = compile_algebraic_curvature_output(
        one_particle_dim=one_particle_dim
    )
    module = YE3TAlgebraicCurvatureOutput(compiled).double()
    density = torch.randn(
        2,
        one_particle_dim,
        one_particle_dim,
        dtype=torch.float64,
        requires_grad=True,
    )
    pairs = module.pair_basis
    left_i = pairs[:, 0][:, None]
    left_j = pairs[:, 1][:, None]
    right_k = pairs[:, 0][None, :]
    right_l = pairs[:, 1][None, :]
    dense_exterior = (
        density[..., left_i, right_k] * density[..., left_j, right_l]
        - density[..., left_i, right_l] * density[..., left_j, right_k]
    )
    expected = module.pack(dense_exterior)
    actual = module.compact_exterior_density(density)
    torch.testing.assert_close(actual, expected, atol=4.0e-13, rtol=4.0e-13)
    cotangent = torch.randn_like(actual)
    expected_vjp = torch.autograd.grad(
        (expected * cotangent).sum(), density, retain_graph=True
    )[0]
    actual_vjp = torch.autograd.grad((actual * cotangent).sum(), density)[0]
    torch.testing.assert_close(actual_vjp, expected_vjp, atol=8.0e-12, rtol=8.0e-12)


@pytest.mark.fast
def test_production_dimensions_report_bounded_construction_memory():
    for one_particle_dim in (56, 66):
        plan = algebraic_curvature_output_plan(one_particle_dim=one_particle_dim)
        report = plan.resource_report
        assert plan.compact_dim == one_particle_dim ** 2 * (one_particle_dim ** 2 - 1) // 12
        assert report["ambient_projector_materialized"] is False
        assert report["estimated_schedule_bytes"] < report["dense_wedge_matrix_bytes_fp64"]
        assert report["forbidden_dense_ambient_projector_bytes_fp64"] > 10 ** 12
