"""Physical same-direction identities using the shared normalized CG maps."""
from math import pi, sqrt

import pytest
import torch

from ye3t.core.spherical import spherical_harmonics_l
from ye3t.couplings.retained_product_images import _same_occurrence_angular_image
from ye3t.exact_scalars import ExactRadical, exact_scalar
from ye3t.global_coupler import _indexed_angular_paths
from test_shared_factor_path_coordinates import cg_table


def paths(spins, L):
    result = []
    for _, tree in _indexed_angular_paths(spins, L, "left"):
        tail = []
        while tree[0] == "merge":
            tail.append(tree[1])
            tree = tree[2]
        result.append((tree[2],)+tuple(reversed(tail)))
    return result


def harmonics(ell, points):
    unit = points/points.norm(dim=-1, keepdim=True)
    theta, phi = unit[..., 2].acos(), torch.atan2(unit[..., 1], unit[..., 0])
    return spherical_harmonics_l(ell, theta, phi).movedim(0, -1)*sqrt(4*pi/(2*ell+1))


def chain_value(spins, path, factors):
    value = factors[0]
    for i, (ell, L) in enumerate(zip(spins[1:], path[1:], strict=True), 1):
        cg = torch.tensor(cg_table(path[i-1], ell, L), dtype=value.dtype, device=value.device)
        value = torch.einsum("...i,...j,ijm->...m", value, factors[i], cg)
    return value


def image_value(image, roles, points, L):
    result = torch.zeros((*points.shape[:-2], 2*L+1), dtype=torch.complex128, device=points.device)
    for (spins, path), coefficient in image.items():
        factors = tuple(harmonics(ell, points[..., role, :]) for ell, role in zip(spins, roles, strict=True))
        result = result+float(coefficient)*chain_value(spins, path, factors)
    return result


@pytest.mark.parametrize("spins,roles", [((1, 1), (0, 0)), ((1, 2), (0, 0)),
    ((1, 1, 1), (0, 1, 0)), ((1, 1, 1, 1), (0, 1, 0, 1)),
    ((2, 1, 2, 1), (2, 0, 2, 0)), ((1, 2, 1), (0, 0, 0))])
def test_every_path_component_matches_physical_harmonic_products(spins, roles):
    torch.manual_seed(195)
    points = torch.randn(3, max(roles)+1, 3, dtype=torch.float64)
    factors = tuple(harmonics(ell, points[:, role]) for ell, role in zip(spins, roles, strict=True))
    for L in range(sum(spins)+1):
        for path in paths(spins, L):
            image = _same_occurrence_angular_image(spins, roles, {path: exact_scalar(1)})
            actual = image_value(image, sorted(set(roles)), points, L)
            expected = chain_value(spins, path, factors)
            torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-12)


def test_exact_cancellations_and_distinct_physical_occurrences():
    assert _same_occurrence_angular_image((1, 1), (4, 4), {(1, 1): exact_scalar(1)}) == {}
    assert _same_occurrence_angular_image((1, 1), (4, 7), {(1, 1): exact_scalar(1)}) == {
        ((1, 1), (1, 1)): exact_scalar(1)}
    scalar = _same_occurrence_angular_image((1, 1), (4, 4), {(1, 0): exact_scalar(1)})
    assert scalar == {((0,), (0,)): -exact_scalar(1)/ExactRadical.sqrt(3)}


def test_complex_and_normalization_coefficients_pass_through_linear_maps():
    import sympy as sp
    spins, roles = (1, 1, 1, 1), (0, 1, 0, 1)
    column = {path: exact_scalar(i+1) for i, path in enumerate(paths(spins, 1))}
    reference = _same_occurrence_angular_image(spins, roles, column)
    scale = sp.I/sp.pi**2
    observed = _same_occurrence_angular_image(spins, roles,
        {path: scale*coefficient._sympy_() for path, coefficient in column.items()})
    assert {key for key, value in observed.items() if sp.simplify(value)} == set(reference)
    assert all(sp.simplify(observed[key]-scale*coefficient._sympy_()) == 0 for key, coefficient in reference.items())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_complete_second_derivatives_and_parameter_gradients(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    spins, roles, L = (1, 2, 1, 2), (0, 1, 0, 1), 2
    column = {path: exact_scalar(i+1) for i, path in enumerate(paths(spins, L))}
    image = _same_occurrence_angular_image(spins, roles, column)
    torch.manual_seed(196)
    points = torch.randn(3, 2, 3, dtype=torch.float64, device=device, requires_grad=True)
    weights = torch.randn(2*L+1, dtype=points.dtype, device=device, requires_grad=True)
    factors = tuple(harmonics(ell, points[:, role]) for ell, role in zip(spins, roles, strict=True))
    expected = sum(float(value)*chain_value(spins, path, factors) for path, value in column.items())
    actual = image_value(image, [0, 1], points, L)
    torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-12)
    derivatives = []
    for value in (actual, expected):
        energy = (value.abs().square()*weights).sum()
        force = -torch.autograd.grad(energy, points, create_graph=True, retain_graph=True)[0]
        derivatives.append((force, *torch.autograd.grad(.1*energy+force.square().sum(),
            (points, weights), retain_graph=True)))
    for a, b in zip(*derivatives, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-10, atol=2e-9)


def test_invalid_paths_and_bindings_reject():
    for spins, roles, column in [((1, 1), (0,), {(1, 0): 1}),
        ((1, 1), (0, 0), {(1, 3): 1}), ((1, 1), (0, 0), {(1, 0): 1, (1, 1): 1})]:
        with pytest.raises(ValueError):
            _same_occurrence_angular_image(spins, roles, column)


@pytest.mark.parametrize("spins,roles,L", [
    ((1, 2, 1, 1, 2), (4, 0, 4, 4, 0), 1),
    ((1, 1, 2, 1, 1), (5, 0, 2, 5, 0), 2),
    ((1, 1, 1, 1, 1, 1), (2, 0, 1, 2, 0, 1), 1)])
def test_incremental_reduction_preserves_linear_assembly_and_physical_values(spins, roles, L):
    # Mixed degrees, three repeated factors, interleaved repeated arguments and
    # a distinct pooled argument exercise the intermediate smaller chains.
    column = {path: exact_scalar((-1)**i*(i+1)) for i, path in enumerate(paths(spins, L))}
    combined = _same_occurrence_angular_image(spins, roles, column)
    separate = {}
    for path, weight in column.items():
        for key, value in _same_occurrence_angular_image(spins, roles, {path: exact_scalar(1)}).items():
            separate[key] = separate.get(key, 0)+weight*value
    assert combined == {key: value for key, value in separate.items() if value}
    torch.manual_seed(199)
    points = torch.randn(2, max(roles)+1, 3, dtype=torch.float64)
    factors = tuple(harmonics(ell, points[:, role]) for ell, role in zip(spins, roles, strict=True))
    expected = sum(float(weight)*chain_value(spins, path, factors) for path, weight in column.items())
    actual = image_value(combined, sorted(set(roles)), points, L)
    torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize("word", [((2, 1), (2, 1), (19, 1)),
    ((4, 1), (4, 2), (9, 1), (9, 2)), ((5, 2), (5, 2), (11, 1), (17, 1))])
def test_channel_renaming_cache_preserves_exact_ordered_coordinates(word):
    from ye3t.couplings.density_product_images import (
        _canonical_sector_path_columns, _normalized_sector_path_columns)
    for L in range(sum(ell for _, ell in word)+1):
        # The uncached constructor on the original identities is the prior
        # implementation, including its original fixed-content copy ordering.
        expected = _canonical_sector_path_columns.__wrapped__(word, L)
        actual = _normalized_sector_path_columns(word, L)
        assert actual == expected
        before = _canonical_sector_path_columns.cache_info()
        renamed = tuple((token*7+100, ell) for token, ell in word)
        assert _normalized_sector_path_columns(renamed, L) is actual
        after = _canonical_sector_path_columns.cache_info()
        assert after.hits == before.hits+1
        assert after.currsize == before.currsize


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_shared_action_products_resolve_the_actual_directional_image(device):
    import json
    from ye3t import couplings
    from ye3t.couplings.carrier_products import _compile_young, bind_carrier_product_torch
    from ye3t.couplings.density_product_images import _fixed_operation_multiplets
    from ye3t.couplings.shared_permutation_factors import _shared_factor_path_columns
    from ye3t.execution_plan import YE3T_O3_PRIMARY_CONVENTION
    from ye3t.representations.young_angular_paths import _coupled_path_columns
    from test_shared_factor_path_coordinates import request

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    raw = {**request((1, 1), 1), "convention_id": YE3T_O3_PRIMARY_CONVENTION}
    source = _shared_factor_path_columns(raw)
    source_plan = couplings.execution_plan_from_compiled_coupler(couplings.compile(couplings.plan(raw)))
    source_runtime = couplings.bind_typed_factorized_execution_plan_torch(
        source_plan, dtype=torch.complex128, device=device)
    product_request = couplings.carrier_product_request(source["layout"], source["layout"],
        target_partition=(2,), target_L=0, argument_ids=("same_source", "same_source"))
    selected = _compile_young(json.dumps(product_request, sort_keys=True))
    product_runtime = bind_carrier_product_torch(couplings.compile(couplings.plan(product_request)),
        dtype=torch.complex128, device=device, backend="torch")
    routes = []
    for channel in selected["channels"]:
        a, b = channel["left_channel"], channel["right_channel"]
        column = _coupled_path_columns(source["columns"][a][0], source["columns"][b][0], (1,)*4, 0)
        scale = ExactRadical.sqrt(2) if a != b else exact_scalar(1)
        image = _same_occurrence_angular_image((1,)*8, tuple(range(4))*2, column)
        routes.append([{key: (value*scale)._sympy_() for key, value in image.items()}])
    targets, matrices, assembly, _ = _fixed_operation_multiplets(routes, 1)
    assert assembly["validation"]["complete_tableau_reconstruction"]
    assert assembly["validation"]["complete_tableau_isometry"]
    assert assembly["independent_map_count"] <= len(routes)
    torch.manual_seed(197)
    points = torch.randn(2, 4, 3, dtype=torch.float64, device=device, requires_grad=True)
    values = source_runtime.evaluate(tuple(harmonics(1, points[:, i]) for i in range(4)))
    actual = product_runtime(values, values)
    coordinates = torch.stack([image_value({key: exact_scalar(1)}, range(4), points, 0)
                               for key in targets], -2)
    matrix = torch.tensor([[complex(item) for item in row] for row in matrices[0].T.tolist()],
                          dtype=torch.complex128, device=device)
    expected = torch.einsum("rc,...cm->...rm", matrix, coordinates).unsqueeze(-2)
    torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-12)
    weight = torch.randn(actual.shape[-3], dtype=points.dtype, device=device, requires_grad=True)
    derivatives = []
    for value in (actual, expected):
        energy = (value.real.sum((-1, -2))*weight).sum()
        force = -torch.autograd.grad(energy, points, create_graph=True, retain_graph=True)[0]
        derivatives.append((force, *torch.autograd.grad(force.square().sum(), (points, weight), retain_graph=True)))
    for a, b in zip(*derivatives, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-10, atol=2e-9)
