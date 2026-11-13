"""Qualify candidate elementary maps through public planning and execution.

The historical reference is injected only for the baseline. Candidate requests
use production dispatch, and both pass through count/plan/compile.
"""
import pytest
import sympy as sp
import torch

from ye3t import couplings
from ye3t.couplings import carrier_products as products
from ye3t.execution_plan import YE3TCarrierKey, YE3TCarrierLayout, YE3T_O3_PRIMARY_CONVENTION
from ye3t.representations import young_orthogonal as young
from ye3t.representations.kronecker_intertwiners import (
    _projector_kronecker_intertwiners, _exact_kronecker_intertwiners_general,
)

pytestmark = pytest.mark.filterwarnings(
    "once:bitcount function is deprecated:DeprecationWarning:mpmath.libmp.libintmath")


def clear_maps():
    young._canonical_split_matrices.cache_clear()
    young._young_orthogonal_nary_subduction_cached.cache_clear()
    products._compile_young.cache_clear()
    products._compile.cache_clear()


@pytest.mark.parametrize("operation", ["disjoint_induction", "same_action_tensor_product"])
def test_projector_maps_public_execution_and_two_derivatives(operation, monkeypatch):
    if operation == "disjoint_induction":
        child, parent = (2, 1), (3, 2, 1)
        old_maps = young._reference_binary_split_matrices(child, child, parent)
        new_maps = young._projector_binary_split_matrices(child, child, parent)
    else:
        child, parent = (3, 1, 1), (3, 2)
        old_maps = _exact_kronecker_intertwiners_general(parent, child, child)["intertwiners"]
        new_maps = _projector_kronecker_intertwiners(parent, child, child)["intertwiners"]
    change = torch.tensor([[float(sp.simplify(sp.trace(a.T*b)/a.cols))
                            for b in old_maps] for a in new_maps], dtype=torch.float64)
    key = YE3TCarrierKey(sum(child), child, 1, YE3T_O3_PRIMARY_CONVENTION, -1)
    layout = YE3TCarrierLayout(key, 1, key.tableau_dimension, 3)
    request = couplings.carrier_product_request(layout, layout, target_partition=parent,
            target_L=1, operation=operation, argument_ids=("x", "y"))
    with monkeypatch.context() as patch:
        if operation == "disjoint_induction":
            patch.setattr(young, "_binary_split_matrices", young._reference_binary_split_matrices)
        else:
            patch.setattr(products, "exact_kronecker_intertwiners", _exact_kronecker_intertwiners_general)
        clear_maps()
        try:
            baseline = couplings.compile(couplings.plan(couplings.count(request)))
        finally:
            clear_maps()
    with monkeypatch.context() as patch:
        from ye3t.representations import young_subgroup_specht_coupling as specht
        def forbidden(*args, **kwargs):
            raise AssertionError("Production compilation repeated a mathematical proof")
        patch.setattr(specht, "validate_young_orthogonal_nary_subduction", forbidden)
        clear_maps()
        try:
            candidate = couplings.compile(couplings.plan(couplings.count(request)))
        finally:
            clear_maps()
    assert baseline["output_channel_count"] == candidate["output_channel_count"] == 2
    assert baseline["placements"] == candidate["placements"]
    assert candidate["output_layout"]["tableau_count"] == len(young.standard_tableaux(parent))
    generator = torch.Generator().manual_seed(811)
    shape = (2, 1, key.tableau_dimension, 3)
    options = {}
    if operation == "disjoint_induction":
        shape = (2, candidate["placement_count"], *shape[1:])
        options["placements"] = candidate["placements"]
    x = torch.randn(shape, generator=generator, dtype=torch.float64, requires_grad=True)
    y = torch.randn(shape, generator=generator, dtype=torch.float64, requires_grad=True)
    before = couplings.bind_carrier_product_torch(baseline, dtype=x.dtype, backend="torch")(x, y, **options)
    after = couplings.bind_carrier_product_torch(candidate, dtype=x.dtype, backend="torch")(x, y, **options)
    expected = torch.einsum("ab,...btm->...atm", change, before)
    torch.testing.assert_close(after, expected, rtol=1e-12, atol=1e-12)
    # A nonuniform target coefficient tests coefficient freedom, not only a
    # rotation-invariant norm that could conceal a copy-coordinate error.
    weight = torch.randn(after.shape, generator=generator, dtype=x.dtype)
    losses = [((value*weight).sum().sin()+value.square().sum()) for value in (after, expected)]
    first = [torch.autograd.grad(loss, (x, y), create_graph=True) for loss in losses]
    for a, b in zip(*first, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
    second = [torch.autograd.grad(sum(g.square().sum() for g in pair), (x, y)) for pair in first]
    for a, b in zip(*second, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
