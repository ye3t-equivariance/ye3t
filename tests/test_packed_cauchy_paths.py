"""Packed outer routes preserve every compiler coordinate and derivative."""
import pytest
import torch

from ye3t import couplings
from ye3t.runtime.angular_paths import BoundCauchyPaths


def ungrouped(runtime, channels, *, _local_cache=None):
    """Previous per-route evaluation retained only as an independent test oracle."""
    local = runtime._local_values(channels, cache=_local_cache)
    outputs = []
    for route in runtime.schedule["routes"]:
        inputs = [local[i][..., copy, 0, :] for i, copy in route["local"]]
        outputs.append(runtime._tree_values(runtime.schedule["outer"][route["outer"]], inputs, {}))
    if not outputs:
        return channels[0].new_zeros((*channels[0].shape[:-2], 0, 2*runtime.schedule["output_L"]+1))
    return torch.stack(outputs, dim=-2)*runtime.schedule["parent_factor"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["reference", "native"])
@pytest.mark.parametrize("basis", ["real_tesseral", "complex_condon_shortley"])
@pytest.mark.parametrize("sizes,ells", [((3,), (1,)), ((2, 2), (1, 2)), ((1, 1, 2), (0, 1, 2))])
@pytest.mark.parametrize("rows", [0, 2])
def test_packed_values_force_loss_and_explicit_adjoint(sizes, ells, rows, basis, backend, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    request = couplings.covariant_cauchy_request([
        dict(neighbor_species="X", radial_channel=i, l=ell, source_family_id="independent")
        for i, ell in enumerate(ells)], sizes, target_L=1, role_dimension=3,
        angular_basis_backend="normalized_cg_paths_v1")
    schedule = couplings.compile(request)["path_schedule"]
    dtype = torch.float64 if basis == "real_tesseral" else torch.complex128
    runtime = BoundCauchyPaths(schedule, dtype=dtype, device=device, backend=backend, basis=basis)
    expected_native = {id(tree) for tree, leaves in runtime.outer_groups
                       if backend == "native" and tree["kind"] != "leaf"}
    assert set(runtime.native_outer) == expected_native
    if basis == "real_tesseral":
        assert all(not value.is_complex() for arguments, width, workspace in runtime.native_outer.values()
                   for value in arguments)
    assert len(schedule["routes"]) > len(runtime.outer_groups)
    generator = torch.Generator().manual_seed(736)
    channels = {i: (torch.randn(rows, 3, 2*ell+1, generator=generator, dtype=dtype)/3).to(device).requires_grad_()
                for i, ell in enumerate(ells)}
    expected, actual = ungrouped(runtime, channels), runtime.evaluate(channels)
    torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-13)
    if rows:
        assert actual.abs().max() > 1e-8
    weights = torch.linspace(.2, .8, actual.shape[-2], dtype=torch.float64, device=device).requires_grad_()
    derivatives = []
    for output in (expected, actual):
        energy = (output.abs().square()*weights[:, None]).sum()
        forces = torch.autograd.grad(energy, tuple(channels.values()), create_graph=True)
        loss = .1*energy+sum(value.abs().square().sum() for value in forces)
        derivatives.append(torch.autograd.grad(loss, (*channels.values(), weights), retain_graph=True))
    for expected_derivative, actual_derivative in zip(*derivatives, strict=True):
        torch.testing.assert_close(actual_derivative, expected_derivative, rtol=3e-10, atol=3e-12)
    seeds = torch.ones((2,)+tuple(actual.shape), dtype=dtype, device=device)
    explicit = runtime.batched_adjoint(channels, seeds)
    # The explicit adjoint uses algebraic covectors. PyTorch conjugates its
    # derivative in the complex gradient convention.
    automatic = torch.autograd.grad((actual*seeds[0]).real.sum(), tuple(channels.values()),
                                    create_graph=True, retain_graph=True)
    for i, value in enumerate(automatic):
        torch.testing.assert_close(explicit[i][0], value.conj(), rtol=3e-11, atol=3e-12)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("basis", ["real_tesseral", "complex_condon_shortley"])
def test_shared_local_maps_use_exact_inputs_and_call_local_activations(device, basis, monkeypatch):
    from ye3t.runtime.angular_paths import BoundAngularPaths
    from ye3t.runtime.factorized_polynomial import _FactorizedPolynomialRuntime
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    payloads = [couplings.compile(couplings.tagged_cauchy_carriers_request([
        dict(neighbor_species="X", radial_channel=offset+i, l=ell,
             source_family_id="chebyshev_c2_origin_regular_v1") for i, ell in enumerate((1, 2))],
        [2, 2], tag_count=2, target_Ls=[0, 1, 2], angular_basis_backend="normalized_cg_paths_v1"))
        for offset in (0, 2)]
    schedule = couplings.tagged_cauchy_carrier_schedule(payloads)
    # Exercise the per-map activation cache explicitly. Native execution batches
    # these maps; test_cauchy_program_batches covers that dispatch and reuse.
    model = _FactorizedPolynomialRuntime(schedule, path_basis=basis, backend="reference").to(device)
    generator = torch.Generator().manual_seed(153)
    values = (torch.randn(2, schedule["input_dimension"], generator=generator, dtype=torch.float64)/3).to(device).requires_grad_()
    calls = {}
    original_evaluate = BoundAngularPaths.evaluate
    def counted(runtime, factors):
        calls[runtime] = calls.get(runtime, 0)+1
        return original_evaluate(runtime, factors)
    monkeypatch.setattr(BoundAngularPaths, "evaluate", counted)
    output = model(values)
    assert calls and max(calls.values()) == 1
    assert sum(len(runtime.groups) for runtime in model.path_bindings.values()) > len(calls)
    # Equal angular maps on different radial/input coordinates cannot share
    # their activations. The uncached reference below evaluates each separately.
    all_maps = [item for candidates in model.local_path_bindings.values() for item in candidates]
    assert len(set(all_maps)) == len(all_maps)
    original_local = BoundCauchyPaths._local_values
    def uncached(runtime, channels, needed=None, cache=None):
        return original_local(runtime, channels, needed=needed)
    with monkeypatch.context() as control:
        control.setattr(BoundCauchyPaths, "_local_values", uncached)
        expected = model(values)
    torch.testing.assert_close(output, expected, rtol=2e-12, atol=2e-13)
    weights = torch.linspace(.1, .7, output.shape[-1], dtype=values.dtype, device=device).requires_grad_()
    derivatives = []
    for result in (output, expected):
        energy = (result*weights).square().sum()
        force = torch.autograd.grad(energy, values, create_graph=True)[0]
        derivatives.append(torch.autograd.grad(.1*energy+force.square().sum(), (values, weights), retain_graph=True))
    for actual, reference in zip(*derivatives, strict=True):
        torch.testing.assert_close(actual, reference, rtol=4e-10, atol=4e-12)
    seeds = torch.ones((2,)+output.shape, dtype=values.dtype, device=device)
    explicit = model.batched_adjoint(values, seeds)
    automatic = torch.autograd.grad(output.sum(), values, create_graph=True, retain_graph=True)[0]
    torch.testing.assert_close(explicit[0], automatic, rtol=3e-11, atol=3e-12)
    torch.testing.assert_close(torch.autograd.grad(explicit[0].square().sum(), values)[0],
                              torch.autograd.grad(automatic.square().sum(), values)[0], rtol=3e-10, atol=3e-11)
    calls.clear()
    changed = model(values+.03)
    assert max(calls.values()) == 1
    assert (changed-output).abs().max() > 1e-6
    with monkeypatch.context() as control:
        control.setattr(BoundCauchyPaths, "_local_values", uncached)
        torch.testing.assert_close(changed, model(values+.03), rtol=2e-12, atol=2e-13)
    model.float()
    assert not model.path_bindings and not model.local_path_bindings
    model.double()
    torch.testing.assert_close(model(values), output, rtol=2e-12, atol=2e-13)
