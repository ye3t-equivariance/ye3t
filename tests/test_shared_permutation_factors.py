"""Shared factor-group action through the canonical compiler and runtime."""
import json
from itertools import product

import pytest
import torch

from ye3t import couplings
from ye3t.execution_plan import (YE3TExecutionPlan, YE3TSourceRealization, YE3TCarrierLayout,
    YE3T_O3_PRIMARY_CONVENTION, YE3T_O3_REAL_TESSERAL_CONVENTION)
from ye3t.core.tesseral import real_tesseral_to_complex_multiplet, complex_multiplet_to_real_tesseral


def request(partition=(2,), L=0, groups=2, degree=2, ell=1):
    return {"family": "shared_permutation_factors",
        "factor_groups": [list(range(i*degree, (i+1)*degree)) for i in range(groups)],
        "content": [i for i in range(groups) for _ in range(degree)],
        "input_Ls": [ell]*(groups*degree), "target_partition": list(partition), "target_L": L}


def compile_runtime(raw, device="cpu", complex_basis=False):
    if complex_basis:
        raw = {**raw, "convention_id": YE3T_O3_PRIMARY_CONVENTION}
    compiled = couplings.compile(couplings.plan(couplings.count(raw)))
    plan = couplings.execution_plan_from_compiled_coupler(compiled)
    restored = YE3TExecutionPlan.from_dict(json.loads(plan.to_json()))
    runtime = couplings.bind_typed_factorized_execution_plan_torch(restored,
        dtype=torch.complex128 if complex_basis else torch.float64, device=device)
    return compiled, restored, runtime


@pytest.mark.parametrize("degree", [2, 3])
def test_sector_inventory_is_complete_and_precedes_coefficient_construction(monkeypatch, degree):
    from ye3t.couplings import shared_permutation_factors as shared
    raw = request(degree=degree)
    del raw["target_partition"], raw["target_L"]
    raw["source_sectors"] = {"young_lambda": "all"}
    def forbidden(*args, **kwargs):
        pytest.fail("sector inventory materialized coupling coefficients")
    monkeypatch.setattr(shared, "compile_carrier_product", forbidden)
    monkeypatch.setattr(shared, "compile_shared_permutation", forbidden)
    inventory = couplings.plan(couplings.count(raw))
    assert inventory["factor_rank"] == 2*degree
    assert inventory["action_degree"] == degree
    assert inventory["available_component_count"] == inventory["selected_component_count"] == 3**(2*degree)
    assert sum(row["layout"]["width"] for row in inventory["records"]) == 3**(2*degree)
    assert inventory["coefficient_materialization"] is False
    limited = couplings.count({**raw, "source_sectors": {"L_max": 2, "copies": 1}})
    assert limited["available_component_count"] == inventory["available_component_count"]
    assert all(row["layout"]["key"]["rotation_L"] <= 2 and row["layout"]["channel_count"] == 1
               for row in limited["records"])
    assert limited["selected_component_count"] < inventory["selected_component_count"]
    with pytest.raises(ValueError, match="Unavailable explicit Young"):
        couplings.count({**raw, "source_sectors": {"young_lambda": ["(N-4,4)"]}})
    with pytest.raises(ValueError, match="Unavailable explicit angular"):
        couplings.count({**raw, "source_sectors": {"L": 2*degree+1}})
    with pytest.raises(ValueError, match="replaces target_partition"):
        couplings.count({**raw, "target_L": 0})


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_selected_sector_collection_lowers_to_the_existing_runtime(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    raw = request()
    del raw["target_partition"], raw["target_L"]
    raw["source_sectors"] = {"L": [0, 1], "copies": 1}
    compiled = couplings.compile(couplings.plan(raw))
    assert len(compiled["records"]) == 3
    torch.manual_seed(54)
    x = torch.randn(4, 4, 3, dtype=torch.float64, device=device, requires_grad=True)
    factors = tuple(x[:, i] for i in range(4))
    for index, row in enumerate(compiled["records"]):
        plan = couplings.execution_plan_from_compiled_coupler(compiled, table_index=index)
        restored = YE3TExecutionPlan.from_dict(json.loads(plan.to_json()))
        runtime = couplings.bind_typed_factorized_execution_plan_torch(restored, dtype=torch.float64, device=device)
        _, _, reference = compile_runtime(row["request"], device)
        value, expected = runtime.evaluate(factors), reference.evaluate(factors)
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
        assert list(value.shape[1:]) == [row["layout"][name] for name in ("channel_count", "tableau_count", "magnetic_count")]
        actual = torch.autograd.grad(value.square().sum(), x, create_graph=True, retain_graph=True)[0]
        wanted = torch.autograd.grad(expected.square().sum(), x, create_graph=True, retain_graph=True)[0]
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    with pytest.raises(ValueError, match="available shared-action sector"):
        couplings.execution_plan_from_compiled_coupler(compiled, table_index=len(compiled["records"]))


def test_counts_do_not_compile_maps_and_cover_complete_spaces(monkeypatch):
    from ye3t.couplings import shared_permutation_factors as shared
    from ye3t.representations import young_angular_paths
    def forbidden(*args, **kwargs):
        pytest.fail("coefficient-free shared-action counting materialized maps")
    monkeypatch.setattr(shared, "compile_carrier_product", forbidden)
    monkeypatch.setattr(young_angular_paths, "angular_path_schedule", forbidden)
    expected = {(2,): [3, 2, 4, 1, 1], (1, 1): [0, 4, 2, 2, 0]}
    for partition, counts in expected.items():
        for L, count in enumerate(counts):
            result = couplings.count(request(partition, L))
            assert result["output_channel_count"] == count
            assert result["factor_rank"] == 4 and result["action_degree"] == 2
            assert result["coefficient_materialization"] is False
            assert result["physical_image_status"] == "requires_physical_binding"
    # Three groups under the same S2: full 3^6 independent tensor coordinates.
    dimension = sum(couplings.count(request(partition, L, groups=3))["output_channel_count"]*(2*L+1)
                    for partition in expected for L in range(7))
    assert dimension == 3**6


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_full_two_branch_real_decomposition_is_orthonormal(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    words = torch.tensor(list(product(range(3), repeat=4)), device=device)
    factors = tuple(torch.nn.functional.one_hot(words[:, i], 3).to(torch.float64) for i in range(4))
    matrices = []
    for partition in ((2,), (1, 1)):
        for L in range(5):
            raw = request(partition, L)
            if not couplings.count(raw)["output_channel_count"]:
                continue
            _, _, runtime = compile_runtime(raw, device)
            output = runtime.evaluate(factors)
            swapped = runtime.evaluate((factors[1], factors[0], factors[3], factors[2]))
            torch.testing.assert_close(swapped, output*(1 if partition == (2,) else -1), atol=2e-12, rtol=2e-12)
            assert not output.is_complex()
            assert all(not value.is_complex() for bound in runtime.local.values() for value in bound.tables)
            assert all(not value.is_complex() for bound in runtime.products.values() for value in bound.buffers())
            matrices.append(output.flatten(1))
    matrix = torch.cat(matrices, 1)
    identity = torch.eye(81, device=device, dtype=matrix.dtype)
    torch.testing.assert_close(matrix.T@matrix, identity, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(matrix@matrix.T, identity, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("raw", [request((1, 1), 1), request((2,), 1, groups=3),
    request((2, 1), 1, groups=1, degree=3), request((2, 1), 0, degree=3),
    request((4,), 2, groups=1, degree=4, ell=2),
    {**request((1, 1), 2), "input_Ls": [1, 1, 2, 2]}])
def test_real_complex_complete_derivatives_and_selection(raw, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    compiled, _, real = compile_runtime(raw, device)
    _, _, complex_runtime = compile_runtime(raw, device, True)
    torch.manual_seed(71)
    ells, L = raw["input_Ls"], raw["target_L"]
    x = torch.randn(2, sum(2*ell+1 for ell in ells), dtype=torch.float64, device=device, requires_grad=True)
    factors = x.split([2*ell+1 for ell in ells], -1)
    actual = real.evaluate(factors)
    transformed = tuple(real_tesseral_to_complex_multiplet(value, ell) for value, ell in zip(factors, ells, strict=True))
    complex_value = complex_runtime.evaluate(transformed)*((-1j)**((sum(ells)-L) % 2))
    expected = complex_multiplet_to_real_tesseral(complex_value, L, tuple(range(-L, L+1)))
    torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-12)
    weights = torch.randn(actual.shape[-3], dtype=x.dtype, device=device, requires_grad=True)
    derivatives = []
    for value in (actual, expected):
        energy = (value.square().sum((-1, -2))*weights).sum()
        force = -torch.autograd.grad(energy, x, create_graph=True, retain_graph=True)[0]
        derivatives.append((force, *torch.autograd.grad(.1*energy+force.square().sum(), (x, weights), retain_graph=True)))
    for a, b in zip(*derivatives, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-10, atol=2e-9)
    index = compiled["output_channel_count"]-1
    _, _, selected = compile_runtime({**raw, "selected_copy_indices": [index]}, device)
    torch.testing.assert_close(selected.evaluate(factors), actual[..., index:index+1, :, :], rtol=2e-12, atol=2e-12)
    empty = real.evaluate(tuple(value[:0] for value in factors))
    first = torch.autograd.grad(empty.square().sum(), x, create_graph=True)[0]
    second = torch.autograd.grad(first.square().sum(), x)[0]
    torch.testing.assert_close(second, torch.zeros_like(x), rtol=0, atol=0)


def test_nontrivial_three_branch_tableaux_follow_the_actual_simultaneous_action():
    from ye3t.representations.projectors import adjacent_transposition_representation_matrix
    raw = request((2, 1), 1, degree=3)
    _, _, runtime = compile_runtime(raw)
    x = torch.randn(2, 6, 3, dtype=torch.float64)
    factors = tuple(x[:, i] for i in range(6))
    original = runtime.evaluate(factors)
    assert original.shape[-2] == 2
    for adjacent in range(2):
        transformed = list(factors)
        for group in raw["factor_groups"]:
            a, b = group[adjacent:adjacent+2]
            transformed[a], transformed[b] = transformed[b], transformed[a]
        matrix = torch.tensor(adjacent_transposition_representation_matrix((2, 1), adjacent).tolist(), dtype=torch.float64)
        expected = torch.einsum("tu,...cum->...ctm", matrix, original)
        torch.testing.assert_close(runtime.evaluate(transformed), expected, rtol=3e-12, atol=3e-12)


def test_source_action_binding_and_invalid_requests():
    compiled, _, _ = compile_runtime(request((1, 1), 1))
    source = YE3TSourceRealization("rooted_motif", 4, content=(0, 0, 1, 1),
        retain_role_order=True, injective=True, automorphisms=((1, 0, 3, 2),),
        metadata={"emitted_action": "shared_factor_groups", "factor_groups": [[0, 1], [2, 3]]})
    binding = source.bind_outputs({"branch": YE3TCarrierLayout.from_dict(compiled["output_layout"])},
        actual_action="rooted_branch_S2", construction_parents=[],
        fixed_maps=[{"plan_hash": compiled["hash"].split(":")[1], "map_name": "shared_permutation_factors", "orientation": "analysis"}],
        argument_bindings={"factors": "distinct physical root/leaf edge identities"}, parameter_names=[],
        row_domain="sample", derivative_modes=["value", "vjp", "double_backward"])
    assert binding["source_realization"]["rank"] == 4
    assert binding["action_spec"]["degree"] == 2
    assert binding["action_spec"]["scope"] == "shared_factor_groups"
    bad = source.to_dict()
    bad["automorphisms"] = [[1, 0, 2, 3]]
    with pytest.raises(ValueError, match="simultaneous"):
        YE3TSourceRealization.from_dict(bad)
    for changes, message in [({"factor_groups": [[0, 1], [1, 3]]}, "partition all"),
        ({"content": [0, 1, 2, 2]}, "complete input type"),
        ({"target_partition": [4]}, "acting group degree"),
        ({"selected_copy_indices": [1000]}, "exact shared-action multiplicity"),
        ({"surprise": 1}, "requires")]:
        with pytest.raises(ValueError, match=message):
            couplings.count({**request(), **changes})
