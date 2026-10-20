"""Independent real-form and physical-image checks for local seed products."""

import pytest
import torch

from ye3t import couplings
from ye3t.core.tesseral import real_tesseral_to_complex_multiplet
from ye3t.execution_plan import YE3TCarrierKey, YE3TCarrierLayout, YE3T_O3_PRIMARY_CONVENTION
from ye3t.exact_scalars import exact_scalar
from ye3t.couplings.carrier_seed_image import _product_rows, _independent_multiplets, _physical_tag_rows


def _variables(channels, tableaux, magnetic, offset=0):
    return tuple(tuple({((offset+(a*tableaux+t)*magnetic+m,),): exact_scalar(1)}
                       for t in range(tableaux) for m in range(magnetic)) for a in range(channels))


def _evaluate(rows, values):
    return torch.stack([torch.stack([sum((float(coefficient)*torch.prod(values[..., [factor[0] for factor in monomial]], dim=-1)
                                         for monomial, coefficient in row.items()), values[..., 0]*0)
                                    for row in components], dim=-1) for components in rows], dim=-2)


@pytest.mark.parametrize("parent", [(3,), (2, 1), (1, 1, 1)])
@pytest.mark.parametrize("L", [0, 1, 2])
@pytest.mark.parametrize("same", [False, True])
def test_real_form_symbolic_product_matches_complete_runtime_and_derivatives(parent, L, same):
    key = YE3TCarrierKey(3, (2, 1), 1, YE3T_O3_PRIMARY_CONVENTION, -1)
    layout = YE3TCarrierLayout(key, 2, 2, 3)
    request = couplings.carrier_product_request(layout, layout, target_partition=parent, target_L=L,
                                                argument_ids=("x", "x" if same else "y"))
    compiled = couplings.compile(request)
    left = _variables(2, 2, 3)
    right = left if same else _variables(2, 2, 3, 12)
    rows = _product_rows(compiled, left, right)
    values = torch.randn(3, 12 if same else 24, dtype=torch.float64, requires_grad=True)
    x = real_tesseral_to_complex_multiplet(values[:, :12].reshape(3, 2, 2, 3), 1)
    y = x if same else real_tesseral_to_complex_multiplet(values[:, 12:].reshape(3, 2, 2, 3), 1)
    expected = couplings.bind_carrier_product_torch(compiled, dtype=x.dtype)(x, y)
    expected = expected * (-1j if (2-L) % 2 else 1)
    out = compiled["output_layout"]
    actual = real_tesseral_to_complex_multiplet(_evaluate(rows, values).reshape(
        3, out["channel_count"], out["tableau_count"], out["magnetic_count"]), L)
    torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-13)
    first = [torch.autograd.grad(value.abs().square().sum(), values, create_graph=True)[0]
             for value in (actual, expected)]
    torch.testing.assert_close(*first, rtol=2e-12, atol=2e-12)
    second = [torch.autograd.grad(value.square().sum(), values)[0] for value in first]
    torch.testing.assert_close(*second, rtol=3e-12, atol=3e-11)


def test_known_duplicate_product_has_one_copy_with_exact_reconstruction():
    x2 = {((0,), (0,)): exact_scalar(1)}
    selected, reconstruction = _independent_multiplets([(x2,), (x2,), ({((1,),): exact_scalar(1)},)])
    assert selected == (0, 2)
    assert reconstruction == ({0: exact_scalar(1)}, {0: exact_scalar(1)}, {1: exact_scalar(1)})


def test_one_component_pivots_cannot_hide_wrong_other_components():
    x = {((0,),): exact_scalar(1)}
    y = {((1,),): exact_scalar(1)}
    with pytest.raises(ArithmeticError, match="tableau/magnetic"):
        _independent_multiplets([(x, y), (x, x)])


def test_tagged_collision_species_rule_and_density_distinction():
    family = "orthogonal_shifted_jacobi_origin_regular_v1"
    h = (0, "H", 0, 0, family, 0)
    o = (0, "O", 0, 0, family, 0)
    assert _physical_tag_rows(({tuple(sorted((h, o))): exact_scalar(1)},), 1) == ({},)
    # Different atoms in inclusive moments can contribute both species.
    density = tuple(sorted(((1, *h[1:]), (1, *o[1:]), h)))
    assert _physical_tag_rows(({density: exact_scalar(1)},), 1)[0]
