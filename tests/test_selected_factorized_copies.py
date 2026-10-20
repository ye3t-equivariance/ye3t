import json

import pytest
import torch

from ye3t.couplings import (
    bind_typed_factorized_execution_plan_torch,
    compile as compile_coupling,
    execution_plan_from_compiled_coupler,
    plan,
)
from ye3t.execution_plan import YE3TExecutionPlan


@pytest.mark.parametrize("content,ells,parent,L", [
    ((1, 1, 2), (1, 1, 1), "trivial", 1),
    ((1, 1, 2), (1, 1, 1), "antisymmetric", 1),
    ((1, 1, 2), (1, 1, 1), "young:2,1", 1),
    ((1, 1, 2, 2), (1, 1, 1, 1), "young:2,2", 0),
    ((1, 2, 3), (0, 0, 0), "young:2,1", 0),
])
def test_selected_factorized_copies_values_derivatives_and_reload(content, ells, parent, L):
    request = plan(content=content, input_Ls=ells, target_L=L,
                   target_permutation=parent, carrier="Phi")
    count = request.report.counts_by_target[L]
    assert count > 0
    indices = tuple(sorted({0, count - 1}))
    selected_plan = request.select_multiplicities(indices)
    selected = compile_coupling(selected_plan, subduction_materialization_backend="exact")
    # Force the general factorized route for the all-scalar reference too.
    full = compile_coupling(request.select_multiplicities(tuple(range(count))),
                            subduction_materialization_backend="exact")
    table = selected.coupler.factorized_coefficient_tables[0]
    assert not selected.coupler.sparse_coefficient_tables
    assert tuple(table["selected_full_alphas"]) == indices
    assert table["full_target_count"] == count
    assert len(table["routes"]) == len(indices)
    assert selected.coupler.certificate.checks["selected_alpha_axes_valid"]
    assert tuple(label["full_alpha_index"] for label in selected.coupler.alpha_labels()) == indices
    generator = torch.Generator().manual_seed(42)
    values = torch.randn(2, len(content), 2 * ells[0] + 1, generator=generator,
                         dtype=torch.complex128, requires_grad=True)
    actual = selected.coupler.evaluate_factorized_factors_torch(values)
    expected = full.coupler.evaluate_factorized_factors_torch(values)[:, indices]
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    for output in (actual, expected):
        gradient = torch.autograd.grad(output.abs().square().sum(), values,
                                       create_graph=True, retain_graph=True)[0]
        hessian = torch.autograd.grad(gradient.real.sum(), values, retain_graph=True)[0]
        if output is actual:
            actual_gradient, actual_hessian = gradient, hessian
        else:
            torch.testing.assert_close(actual_gradient, gradient, atol=1e-10, rtol=1e-10)
            torch.testing.assert_close(actual_hessian, hessian, atol=1e-10, rtol=1e-10)
    execution = execution_plan_from_compiled_coupler(selected)
    restored = YE3TExecutionPlan.from_dict(json.loads(execution.to_json()))
    runtime = bind_typed_factorized_execution_plan_torch(restored, dtype=values.dtype,
                                                        device=values.device)
    torch.testing.assert_close(runtime.evaluate(values), actual, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("indices", [(), (-1,), (True,), (0, 0), (1, 0), (10000,)])
def test_invalid_copy_selection_rejected_during_plan(indices):
    request = plan(content=(1, 2, 3), input_Ls=(1, 1, 1), target_L=1,
                   target_permutation="young:2,1", carrier="Phi")
    with pytest.raises(ValueError, match="sorted unique"):
        request.select_multiplicities(indices)
