"""Fixed repeated inputs through the shared action compiler, including derivatives."""
from functools import lru_cache
from itertools import product
import json

import pytest
import torch

from ye3t import couplings
from ye3t.core.cg import cg_exact
from ye3t.execution_plan import (YE3TExecutionPlan, YE3TSourceRealization,
    YE3T_O3_PRIMARY_CONVENTION)


def request(fixed=((4, 5),), spins=(1,)*6, partition=(2,), L=0):
    return dict(family="shared_permutation_factors", factor_groups=[[0, 1], [2, 3]],
        fixed_factor_groups=fixed, content=[0, 0, 1, 1]+[2]*(len(spins)-4),
        input_Ls=spins, target_partition=partition, target_L=L)


def runtime(raw, device="cpu", complex_basis=False):
    if complex_basis:
        raw = {**raw, "convention_id": YE3T_O3_PRIMARY_CONVENTION}
    compiled = couplings.compile(couplings.plan(raw))
    plan = couplings.execution_plan_from_compiled_coupler(compiled)
    restored = YE3TExecutionPlan.from_dict(json.loads(plan.to_json()))
    return couplings.bind_typed_factorized_execution_plan_torch(restored,
        dtype=torch.complex128 if complex_basis else torch.float64, device=device)


@pytest.mark.parametrize("fixed,spins,dimension,scalars,parity", [
    (((4, 5),), (1,)*6, 486, (7, 2), 1),
    (((4,),), (1, 1, 1, 1, 2), 405, (4, 2), 1),
    (((4,),), (1,)*5, 243, (2, 4), -1),
    (((4,), (5,)), (1,)*6, 729, (9, 6), 1)])
def test_exact_inventory_before_materialization(monkeypatch, fixed, spins, dimension, scalars, parity):
    from ye3t.couplings import shared_permutation_factors as shared
    from ye3t.representations import young_angular_paths
    def forbidden(*args, **kwargs):
        pytest.fail("count/plan materialized coefficients")
    monkeypatch.setattr(shared, "compile_carrier_product", forbidden)
    monkeypatch.setattr(shared, "compile_shared_permutation", forbidden)
    monkeypatch.setattr(young_angular_paths, "angular_path_schedule", forbidden)
    raw = request(fixed, spins)
    for partition, expected in zip(((2,), (1, 1)), scalars, strict=True):
        counted = couplings.plan({**raw, "target_partition": partition})
        assert counted["output_channel_count"] == expected
        assert counted["output_layout"]["key"]["parity"] == parity
    del raw["target_partition"], raw["target_L"]
    counted = couplings.plan({**raw, "source_sectors": {"young_lambda": "all"}})
    assert counted["factor_rank"] == len(spins)
    assert counted["action_degree"] == 2
    assert counted["selected_component_count"] == counted["available_component_count"] == dimension
    assert counted["physical_image_status"] == "requires_physical_binding"
    assert counted["coefficient_materialization"] is False


@lru_cache(maxsize=None)
def cg(a, b, L):
    return torch.tensor([[[float(cg_exact(a, m, b, n, L, M)) for M in range(-L, L+1)]
        for n in range(-b, b+1)] for m in range(-a, a+1)], dtype=torch.complex128)


def singlets(factors):
    # Test-only triangle enumeration supplies an independent angular projector.
    active = {(1,): factors[0]}
    for index in range(1, 6):
        next_rows = {}
        for path, value in active.items():
            for L in range(abs(path[-1]-1), path[-1]+2):
                if L <= 5-index:
                    next_rows[path+(L,)] = torch.einsum("bi,bj,ijm->bm", value,
                        factors[index], cg(path[-1], 1, L).to(value.device))
        active = next_rows
    assert len(active) == 15
    return torch.cat(tuple(active.values()), -1)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_selected_projector_and_literal_reuse_derivatives(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    bound = runtime(request(), device, complex_basis=True)
    words = torch.tensor(list(product(range(3), repeat=6)), dtype=torch.long, device=device)
    factors = tuple(torch.nn.functional.one_hot(words[:, i], 3).to(torch.complex128) for i in range(6))
    Q = bound.evaluate(factors).flatten(1)
    assert Q.shape == (729, 7)
    torch.testing.assert_close(Q.conj().T@Q, torch.eye(7, dtype=Q.dtype, device=device), rtol=2e-12, atol=2e-12)
    R = singlets(factors)
    torch.testing.assert_close(R.conj().T@R, torch.eye(15, dtype=R.dtype, device=device), rtol=2e-12, atol=2e-12)
    angular = R@R.conj().T
    powers = torch.tensor([3**i for i in range(5, -1, -1)], device=device)
    permutations = ((0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5),
                    (0, 1, 2, 3, 5, 4), (1, 0, 3, 2, 5, 4))
    rows = [(words[:, order]*powers).sum(-1) for order in permutations]
    projector = sum(angular.index_select(0, order) for order in rows)/4
    torch.testing.assert_close(Q@Q.conj().T, projector, rtol=3e-12, atol=3e-12)
    for order in rows:
        torch.testing.assert_close(Q[order], Q, rtol=2e-12, atol=2e-12)

    generator = torch.Generator().manual_seed(720)
    x = (torch.randn(2, 5, 3, generator=generator, dtype=torch.complex128)/3).to(device).requires_grad_()
    reused = tuple(x[:, i] for i in (0, 1, 2, 3, 4, 4))
    actual = bound.evaluate(reused).flatten(1)
    tensor = reused[0]
    for factor in reused[1:]:
        tensor = (tensor[..., None]*factor[:, None, :]).flatten(1)
    expected = tensor@Q
    torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-12)
    weights = torch.linspace(.1, .7, 7, dtype=torch.float64, device=device).requires_grad_()
    derivatives = []
    for value in (actual, expected):
        energy = (value.abs().square()*weights).sum()
        force = torch.autograd.grad(energy, x, create_graph=True)[0]
        derivatives.append((force, *torch.autograd.grad(.1*energy+force.abs().square().sum(),
                            (x, weights), retain_graph=True)))
    for a, b in zip(*derivatives, strict=True):
        torch.testing.assert_close(a, b, rtol=3e-10, atol=3e-12)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fixed_inputs_preserve_complete_three_branch_action(device):
    from ye3t.representations.projectors import adjacent_transposition_representation_matrix
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    raw = dict(family="shared_permutation_factors", factor_groups=[[0, 1, 2]],
        fixed_factor_groups=[[3, 4]], content=[0, 0, 0, 1, 1], input_Ls=[1]*5,
        target_partition=[2, 1], target_L=1)
    bound = runtime(raw, device)
    torch.manual_seed(90)
    x = torch.randn(2, 4, 3, dtype=torch.float64, device=device)
    factors = tuple(x[:, i] for i in (0, 1, 2, 3, 3))
    original = bound.evaluate(factors)
    assert original.shape[-2:] == (2, 3)
    assert not original.is_complex()
    for adjacent in range(2):
        changed = list(factors)
        changed[adjacent], changed[adjacent+1] = changed[adjacent+1], changed[adjacent]
        matrix = torch.tensor(adjacent_transposition_representation_matrix((2, 1), adjacent).tolist(),
                              dtype=torch.float64, device=device)
        expected = torch.einsum("tu,...cum->...ctm", matrix, original)
        torch.testing.assert_close(bound.evaluate(changed), expected, rtol=3e-12, atol=3e-12)


def test_action_binding_and_unchanged_historical_request():
    source = YE3TSourceRealization("rooted_motif", 6, content=(0, 0, 1, 1, 2, 2),
        retain_role_order=True, injective=True, automorphisms=((1, 0, 3, 2, 4, 5),),
        metadata={"emitted_action": "shared_factor_groups", "factor_groups": [[0, 1], [2, 3]],
                  "fixed_factor_groups": [[4, 5]]})
    assert YE3TSourceRealization.from_dict(source.to_dict()).to_dict() == source.to_dict()
    bad = source.to_dict()
    bad["automorphisms"] = [[1, 0, 3, 2, 5, 4]]
    with pytest.raises(ValueError, match="simultaneous"):
        YE3TSourceRealization.from_dict(bad)
    raw = request((), (1,)*4)
    old = {key: value for key, value in raw.items() if key != "fixed_factor_groups"}
    assert couplings.compile(raw) == couplings.compile(old)


@pytest.mark.parametrize("changes,message", [
    ({"fixed_factor_groups": [[4, 4]]}, "partition all"),
    ({"fixed_factor_groups": [[3, 4, 5]]}, "partition all"),
    ({"fixed_factor_groups": [[4]]}, "partition all"),
    ({"fixed_factor_groups": [[]]}, "nonempty"),
    ({"fixed_factor_groups": [4]}, "nonempty"),
    ({"fixed_factor_groups": None}, "nonempty"),
    ({"content": [0, 0, 1, 1, 2, 3]}, "complete input type"),
    ({"input_Ls": [1, 1, 1, 1, 1, 2]}, "complete input type")])
def test_invalid_fixed_input_requests_fail_early(changes, message):
    with pytest.raises(ValueError, match=message):
        couplings.plan({**request(), **changes})
