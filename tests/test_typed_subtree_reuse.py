"""Shared CG subtrees preserve complete typed-source values and derivatives."""
import pytest
import torch

from ye3t import couplings
from ye3t.couplings.factorized_typed import (
    YE3TFactorizedTorchRuntime, evaluate_typed_factorized_torch,
)


@pytest.fixture(scope="module", params=[
    ((1, 2, 3, 4, 5, 6), (1,)*6, (6,), 0),
    ((1, 1, 2, 2), (1,)*4, (3, 1), 1),
    ((1, 1, 1, 1, 2, 3), (2, 2, 2, 2, 1, 1), (6,), 2),
])
def compiled_source(request):
    content, ells, partition, L = request.param
    plan = couplings.plan(content=content, input_Ls=ells, target_L=L,
        target_permutation="young:"+",".join(map(str, partition)), carrier="Phi")
    return couplings.compile(plan, subduction_materialization_backend="exact").coupler


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bound_shared_subtrees_match_uncached_routes(compiled_source, device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    table = compiled_source.factorized_coefficient_tables[0]
    types = tuple(map(tuple, table["input_factor_types"]))
    runtime = YE3TFactorizedTorchRuntime(compiled_source, dtype=torch.complex128, device=device)
    torch.manual_seed(173)
    x = torch.randn(2, sum(2*ell+1 for _, ell in types), dtype=torch.float64,
                    device=device, requires_grad=True)
    assert runtime.device == x.device
    values = tuple(value+.13j*value.square() for value in x.split([2*ell+1 for _, ell in types], -1))
    actual = runtime.evaluate(values)
    expected = evaluate_typed_factorized_torch(compiled_source, values)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    weight = torch.linspace(.2, .7, actual[0].numel(), dtype=x.dtype, device=device).requires_grad_()
    derivatives = []
    for output in (actual, expected):
        energy = (output.flatten(1).abs().square()*weight).sum()
        force = -torch.autograd.grad(energy, x, create_graph=True, retain_graph=True)[0]
        gradients = torch.autograd.grad(.1*energy+force.square().sum(), (x, weight), retain_graph=True)
        derivatives.append((force, *gradients))
    for observed, reference in zip(*derivatives, strict=True):
        torch.testing.assert_close(observed, reference, rtol=2e-10, atol=2e-10)

    # Reordered input words and changed geometry must get fresh values while
    # retaining the static map/copy/factor identities.
    changed = tuple(.7*value+.11 for value in reversed(values))
    reversed_types = tuple(reversed(types))
    torch.testing.assert_close(runtime.evaluate(changed, factor_types=reversed_types),
        evaluate_typed_factorized_torch(compiled_source, changed, factor_types=reversed_types),
        rtol=2e-12, atol=2e-12)
    empty = runtime.evaluate(tuple(value[:0] for value in values))
    first = torch.autograd.grad(empty.abs().square().sum(), x, create_graph=True)[0]
    second = torch.autograd.grad(first.square().sum(), x)[0]
    torch.testing.assert_close(first, torch.zeros_like(x), rtol=0, atol=0)
    torch.testing.assert_close(second, torch.zeros_like(x), rtol=0, atol=0)

    if types == tuple((i, 1) for i in range(1, 7)):
        counts = []
        original = torch.einsum
        for bound in (None, runtime):
            calls = []
            def traced(equation, *args, **kwargs):
                if equation == "...am,...bn,mnc->...abc":
                    calls.append(equation)
                return original(equation, *args, **kwargs)
            with monkeypatch.context() as patch:
                patch.setattr(torch, "einsum", traced)
                evaluate_typed_factorized_torch(compiled_source, values, bound=bound)
            counts.append(len(calls))
        assert 0 < counts[1] < counts[0]
        assert len({id(value) for value in runtime.cg.values()}) < len(runtime.cg)
