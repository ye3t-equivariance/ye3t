"""Real stored paths preserve the documented complex-source coefficient gauge."""
import pytest
import torch

from ye3t import couplings
from ye3t.core.tesseral import _real_to_complex_matrix
from ye3t.runtime.angular_paths import BoundCauchyPaths, _bound_cg_table


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape,seeds", [((0,), 2), ((2, 0), 2), ((2,), 0)])
def test_real_empty_rows_and_seeds_preserve_second_derivatives(device, dtype, shape, seeds):
    from ye3t.representations.young_angular_paths import angular_path_schedule
    from test_angular_paths_empty import test_empty_angular_values_and_seed_dependent_second_derivatives
    test_empty_angular_values_and_seed_dependent_second_derivatives(
        angular_path_schedule(3, 1, (2, 1), 1), device, dtype, shape, seeds, basis="real_tesseral")


@pytest.mark.parametrize("angular", [(1, 1, 0), (1, 1, 1), (1, 2, 2), (2, 2, 3), (3, 2, 4)])
def test_bound_real_tables_match_existing_exact_coefficients(angular):
    from ye3t.core.cg import cg_exact_integer
    from ye3t.core.tesseral import _real_cg_entries_exact
    a, b, c = angular
    entries = [(m+a, n+b, m+n+c, float(value))
               for m in range(-a, a+1) for n in range(-b, b+1) if abs(m+n) <= c
               if (value := cg_exact_integer(a, m, b, n, c, m+n))]
    actual, _ = _bound_cg_table(angular, entries, torch.float64, "cpu", True)
    expected = torch.zeros_like(actual)
    for i, j, k, value in _real_cg_entries_exact(*angular):
        expected[i, j, k] = float(value)
    torch.testing.assert_close(actual, expected, rtol=3e-14, atol=3e-14)
    # A stale/tampered saved map must not be silently replaced by the canonical
    # map. Check the saved coefficients before reusing their exact real form.
    with pytest.raises(ValueError, match="canonical normalized map"):
        _bound_cg_table(angular, [(i,j,k,2*v) for i,j,k,v in entries], torch.float64, "cpu", True)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_factorized_real_storage_dtype_move_and_empty_adjoint(device):
    from ye3t.runtime.factorized_polynomial import _FactorizedPolynomialRuntime
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    source = couplings.compile(couplings.tagged_cauchy_carriers_request([
        dict(neighbor_species="X", radial_channel=i, l=ell,
             source_family_id="chebyshev_c2_origin_regular_v1") for i, ell in enumerate((1, 2))],
        [2, 1], tag_count=2, target_Ls=[1, 2], angular_basis_backend="normalized_cg_paths_v1"))
    schedule = couplings.tagged_cauchy_carrier_schedule([source])
    model = _FactorizedPolynomialRuntime(schedule).to(device)
    control = _FactorizedPolynomialRuntime(schedule, path_basis="complex_condon_shortley").to(device)
    x = torch.linspace(.1, .6, schedule["input_dimension"], dtype=torch.float64, device=device)[None, :].requires_grad_()
    expected = control(x)
    torch.testing.assert_close(model(x), expected, rtol=4e-11, atol=4e-12)
    assert model.path_bindings
    model = model.float()
    assert not model.path_bindings
    torch.testing.assert_close(model(x.float()).double(), expected, rtol=4e-6, atol=4e-7)
    model = model.double()
    assert not model.path_bindings
    torch.testing.assert_close(model(x), expected, rtol=4e-11, atol=4e-12)
    for rows, count in ((0, 2), (2, 0)):
        values = torch.zeros(rows, schedule["input_dimension"], dtype=torch.float64, device=device, requires_grad=True)
        seeds = torch.zeros(count, rows, schedule["output_dimension"], dtype=values.dtype, device=device, requires_grad=True)
        output = model(values)
        adjoint = model.batched_adjoint(values, seeds)
        assert output.shape == (rows, schedule["output_dimension"])
        assert adjoint.shape == (count, rows, schedule["input_dimension"])
        gradients = torch.autograd.grad(adjoint.sum(), (values, seeds))
        assert all(torch.count_nonzero(gradient) == 0 for gradient in gradients)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["reference", "native"])
@pytest.mark.parametrize("sizes,ells", [((3,), (1,)), ((4,), (1,)), ((2, 2), (1, 2))])
@pytest.mark.parametrize("L", [0, 1, 2])
def test_real_cauchy_values_adjoint_and_force_loss_match_complex(sizes, ells, L, device, backend):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    request = couplings.covariant_cauchy_request([
        dict(neighbor_species="X", radial_channel=i, l=ell, source_family_id="independent")
        for i, ell in enumerate(ells)], sizes, target_L=L, role_dimension=3,
        angular_basis_backend="normalized_cg_paths_v1")
    schedule = couplings.compile(request)["path_schedule"]
    real = BoundCauchyPaths(schedule, dtype=torch.float64, device=device, backend=backend, basis="real_tesseral")
    reference = BoundCauchyPaths(schedule, dtype=torch.complex128, device=device, backend=backend)
    generator = torch.Generator().manual_seed(132)
    channels = {i: (torch.randn(2, 3, 2*ell+1, generator=generator, dtype=torch.float64)/3).to(device).requires_grad_()
                for i, ell in enumerate(ells)}
    forms = {i: _real_to_complex_matrix(ell, torch.float64, device) for i, ell in enumerate(ells)}
    transform = _real_to_complex_matrix(L, torch.float64, device)
    complex_channels = {i: x.to(torch.complex128)@forms[i] for i, x in channels.items()}
    expected = ((-1j)**schedule["real_phase_exponent"]*reference.evaluate(complex_channels)@transform.conj().T).real
    actual = real.evaluate(channels)
    assert actual.dtype == torch.float64 and actual.abs().max() > 1e-8
    torch.testing.assert_close(actual, expected, rtol=5e-11, atol=5e-12)
    # Real execution retains no complex coefficient or native workspace tensors.
    for local, members in real.groups:
        assert all(not table.is_complex() for table in local.tables)
        assert not local.coefficients.is_complex()
        if local.native_arguments is not None:
            assert all(not value.is_complex() for value in local.native_arguments)
    assert all(not table.is_complex() for table in real.cg.values())
    weights = torch.linspace(.2, .8, actual.numel(), dtype=actual.dtype, device=device).reshape_as(actual).requires_grad_()
    gradients = []
    for output in (actual, expected):
        energy = (output*weights).square().sum()
        forces = torch.autograd.grad(energy, tuple(channels.values()), create_graph=True)
        loss = .1*energy+sum(force.square().sum() for force in forces)
        gradients.append(torch.autograd.grad(loss, (*channels.values(), weights), retain_graph=True))
    for a, b in zip(*gradients, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-9, atol=2e-10)
    seeds = torch.stack((weights, 2*weights))
    adjoints = real.batched_adjoint(channels, seeds)
    for seed_index in range(2):
        expected_gradients = torch.autograd.grad((actual*seeds[seed_index]).sum(), tuple(channels.values()),
                                                create_graph=True, retain_graph=True)
        for index, value in enumerate(expected_gradients):
            torch.testing.assert_close(adjoints[index][seed_index], value, rtol=8e-11, atol=8e-12)
    objective = sum(value.square().sum() for value in adjoints.values())
    first = torch.autograd.grad(objective, (*channels.values(), weights), retain_graph=True)
    reference_adjoints = [torch.autograd.grad((expected*seed).sum(), tuple(channels.values()), create_graph=True, retain_graph=True)
                          for seed in seeds]
    reference_objective = sum(value.square().sum() for row in reference_adjoints for value in row)
    second = torch.autograd.grad(reference_objective, (*channels.values(), weights))
    for a, b in zip(first, second, strict=True):
        torch.testing.assert_close(a, b, rtol=3e-9, atol=3e-10)
