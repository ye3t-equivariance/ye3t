"""Canonical Cauchy compile/evaluate/reload with normalized shared maps."""
import json

import numpy as np
import pytest
import torch

from ye3t.couplings import compile as compile_coupling
from ye3t.couplings.covariant_cauchy import (
    covariant_cauchy_request, covariant_cauchy_count,
    evaluate_covariant_cauchy, validate_covariant_cauchy,
    covariant_cauchy_orthogonal_output_plan, covariant_cauchy_flat_real_plan)
from ye3t.runtime.angular_paths import BoundCauchyPaths


@pytest.mark.parametrize("sizes,ells,L,occupations", [
    ((3,), (1,), 1, ((2, 1),)), ((2, 1), (1, 2), 1, ((1, 2), (2, 1))),
    ((1, 1, 1), (1, 1, 1), 1, None), ((4,), (2,), 0, ((2, 2),))])
def test_shared_compiler_count_complete_derivatives_and_reload(sizes, ells, L, occupations, monkeypatch):
    channels = tuple({"neighbor_species": "Ni", "radial_channel": i, "l": ell,
                      "source_family_id": "supplied_roles"} for i, ell in enumerate(ells))
    request = covariant_cauchy_request(channels, sizes, target_L=L, role_dimension=2,
        role_occupations=occupations, angular_basis_backend="normalized_cg_paths_v1")
    count = covariant_cauchy_count(request)
    compiled = compile_coupling(request)
    assert compiled["multiplet_count"] == count["multiplet_count"] > 0
    assert validate_covariant_cauchy(compiled)
    assert "block_templates" not in compiled
    assert all(row["label"]["convention_id"] == "canonical_split_role_paired_angular_paths_v2"
               for row in compiled["descriptors"])
    restored = json.loads(json.dumps(compiled))
    assert validate_covariant_cauchy(restored)
    orthogonal = covariant_cauchy_orthogonal_output_plan(restored)
    assert orthogonal["validation_report"]["normalized_shared_maps"]
    assert not orthogonal["validation_report"]["physical_sample_orthogonality"]
    with pytest.raises(ValueError, match="retains factorized couplings"):
        covariant_cauchy_flat_real_plan(restored)
    generator = torch.Generator().manual_seed(13)
    values = {i: torch.randn(2, 2, 2*ell+1, generator=generator, dtype=torch.float64,
                            requires_grad=True) for i, ell in enumerate(ells)}
    runtime = BoundCauchyPaths(restored["path_schedule"], dtype=torch.float64, device="cpu")
    output = runtime.evaluate(values)
    assert output.shape == (2, count["multiplet_count"], 2*L+1)
    seeds = torch.randn(3, *output.shape, generator=generator, dtype=torch.float64)
    adjoint = runtime.batched_adjoint(values, seeds)
    automatic = [torch.autograd.grad((output*seed).sum(), tuple(values.values()),
                                    create_graph=True, retain_graph=True) for seed in seeds]
    for i, result in adjoint.items():
        torch.testing.assert_close(result, torch.stack([row[i] for row in automatic]), atol=1e-10, rtol=1e-10)
    actual_second = torch.autograd.grad(sum(v.square().sum() for v in adjoint.values()),
                                      tuple(values.values()), retain_graph=True)
    expected_second = torch.autograd.grad(sum(v.square().sum() for row in automatic for v in row),
                                        tuple(values.values()))
    for actual, expected in zip(actual_second, expected_second):
        torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-10)

    # Reload/evaluation must not reconstruct Young, CG, or magnetic coefficients.
    from ye3t.representations import young_angular_paths
    from ye3t.core import cg
    def forbidden(*args, **kwargs):
        pytest.fail("Evaluation reconstructed coefficient maps")
    monkeypatch.setattr(young_angular_paths, "young_angular_path_basis", forbidden)
    monkeypatch.setattr(cg, "_cg_table_exact", forbidden)
    arrays = {i: value[0].detach().numpy() for i, value in values.items()}
    result, gradient = evaluate_covariant_cauchy(restored, arrays, upstream=np.ones(output.shape[1:]))
    assert np.isfinite(result).all()
    # Independent finite differences include the real-form and pseudo-tensor phase.
    step = 1e-6
    for i, array in arrays.items():
        index = (0, min(1, array.shape[1]-1))
        plus, minus = dict(arrays), dict(arrays)
        plus[i], minus[i] = array.copy(), array.copy()
        plus[i][index] += step
        minus[i][index] -= step
        finite = (evaluate_covariant_cauchy(restored, plus)[0].sum()
                  - evaluate_covariant_cauchy(restored, minus)[0].sum())/(2*step)
        assert gradient[i][index] == pytest.approx(finite, abs=2e-7, rel=2e-7)


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_shared_paths_gpu_values_derivatives_and_parameter_gradients(backend):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU for CPU/CUDA derivative comparison")
    request = covariant_cauchy_request([
        dict(neighbor_species="X", radial_channel=i, l=ell, source_family_id="roles")
        for i, ell in enumerate((1, 2))], (2, 2), target_L=1, role_dimension=3,
        angular_basis_backend="normalized_cg_paths_v1")
    compiled = compile_coupling(request)
    schedule = compiled["path_schedule"]
    generator = torch.Generator().manual_seed(871)
    inputs = [torch.randn(2,3,2*ell+1,generator=generator,dtype=torch.float64) for ell in (1,2)]
    weight = torch.randn(len(schedule["routes"]),3,generator=generator,dtype=torch.float64)
    results = []
    for device in ("cpu", "cuda"):
        values = {i: value.to(device).requires_grad_(True) for i,value in enumerate(inputs)}
        parameters = weight.to(device).detach().requires_grad_(True)
        runtime = BoundCauchyPaths(schedule, dtype=torch.float64, device=device, backend=backend)
        output = runtime.evaluate(values)
        seed = parameters.unsqueeze(0).expand(2,-1,-1)
        gradient = runtime.batched_adjoint(values, seed.unsqueeze(0))
        loss = sum(v.square().mean() for v in gradient.values()) + (output*seed).square().mean()
        parameter_gradient = torch.autograd.grad(loss, parameters, retain_graph=True)[0]
        reference_rows = torch.autograd.grad((output*seed).sum(), tuple(values.values()),
                                             create_graph=True, retain_graph=True)
        reference_loss = sum(v.square().mean() for v in reference_rows) + (output*seed).square().mean()
        reference_gradient = torch.autograd.grad(reference_loss, parameters)[0]
        torch.testing.assert_close(parameter_gradient, reference_gradient, atol=2e-9, rtol=2e-10)
        results.append([output.detach().cpu(), parameter_gradient.detach().cpu()]
                       + [v.detach().cpu() for v in gradient.values()])
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-10)


@pytest.mark.parametrize("field", ["dependency", "multiplicity", "magnetic", "role"])
def test_saved_path_binding_rejects_malformed_maps(field):
    from copy import deepcopy
    request = covariant_cauchy_request([
        dict(neighbor_species="X", radial_channel=0,l=1,source_family_id="roles")],
        (3,), target_L=1,role_dimension=2,angular_basis_backend="normalized_cg_paths_v1")
    schedule = deepcopy(compile_coupling(request)["path_schedule"])
    # JSON decoding produces mutable lists, as in a relocated model.
    schedule = json.loads(json.dumps(schedule))
    local = schedule["local"][0]["schedule"]
    if field == "dependency":
        local["nodes"][0][0] = 999
    elif field == "multiplicity":
        local["mixing"][0][1] = local["copies"]*local["tableaux"]
    elif field == "magnetic":
        local["cg_tables"][0]["entries"][0][0] = -1
    else:
        local["factor_roles"][0] = schedule["role_dimension"]
    with pytest.raises(ValueError):
        BoundCauchyPaths(schedule, dtype=torch.float64, device="cpu")


def test_shared_prefixes_preserve_original_copy_coordinates_and_zero_seed_gradients():
    from ye3t.runtime.angular_paths import BoundAngularPaths
    request = covariant_cauchy_request([
        dict(neighbor_species="X",radial_channel=0,l=2,source_family_id="roles")],
        (4,),target_L=0,role_dimension=2,role_occupations=((2,2),),
        angular_basis_backend="normalized_cg_paths_v1")
    schedule = compile_coupling(request)["path_schedule"]
    runtime = BoundCauchyPaths(schedule,dtype=torch.float64,device="cpu",backend="native")
    assert sum(len(item.schedule["nodes"]) for item,members in runtime.groups) < sum(
        len(row["schedule"]["nodes"]) for row in schedule["local"])
    values = {0:torch.randn(2,2,5,dtype=torch.float64,requires_grad=True)}
    independent = [BoundAngularPaths(row["schedule"],dtype=torch.float64,device="cpu").evaluate(
        [values[0][:,role,:] for role in row["schedule"]["factor_roles"]]) for row in schedule["local"]]
    reference = torch.stack([independent[i][:,copy,0,:] for route in schedule["routes"]
                             for i,copy in route["local"]],dim=-2)*schedule["parent_factor"]
    output = runtime.evaluate(values)
    torch.testing.assert_close(output,reference,atol=1e-12,rtol=1e-12)
    seeds = torch.zeros((1,)+tuple(output.shape),dtype=output.dtype,requires_grad=True)
    explicit = runtime.batched_adjoint(values,seeds)[0]
    expected = torch.autograd.grad((output*seeds[0]).sum(),values[0],create_graph=True)[0]
    torch.testing.assert_close(torch.autograd.grad(explicit.sum(),seeds)[0],
                              torch.autograd.grad(expected.sum(),seeds)[0],atol=1e-11,rtol=1e-11)
