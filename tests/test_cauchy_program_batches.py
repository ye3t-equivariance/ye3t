"""Compatible program batching preserves compiler coordinates and derivatives."""
import pytest
import torch

from ye3t import couplings
from ye3t.runtime.factorized_polynomial import _FactorizedPolynomialRuntime


def schedule(block_size=2, tag_count=2):
    payloads = [couplings.compile(couplings.tagged_cauchy_carriers_request([
        dict(neighbor_species="X", radial_channel=offset+i, l=ell,
             source_family_id="chebyshev_c2_origin_regular_v1") for i, ell in enumerate((1, 2))],
        [block_size, block_size], tag_count=tag_count, target_Ls=[0, 1, 2], angular_basis_backend="normalized_cg_paths_v1"))
        for offset in (0, 2)]
    return couplings.tagged_cauchy_carrier_schedule(payloads)


def sequential(model, values):
    # Per-program evaluation plus the original dense output operation remains
    # independent of both grouped execution and fixed-output shortcuts.
    local_cache = {}
    outputs = []
    for p in model.path_programs:
        current = values.index_select(-1, getattr(model, f"inputs_{p}"))
        runtime, channels, _ = model._path_inputs(p, current)
        output = runtime.evaluate(channels, _local_cache=local_cache)
        outputs.append(dense_output(model, p, output, current).index_select(-1, getattr(model, f"outputs_{p}")))
    return torch.cat(outputs, dim=-1)


def dense_output(model, p, output, values):
    if model.path_basis != "real_tesseral":
        from ye3t.core.tesseral import _real_to_complex_matrix
        path = model.path_programs[p]
        transform = _real_to_complex_matrix(path["output_L"], values.dtype, values.device)
        output = (((-1j)**path["real_phase_exponent"])*output @ transform.conj().T).real
    return output.flatten(-2) @ getattr(model, f"path_output_map_{p}")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("basis", ["real_tesseral", "complex_condon_shortley"])
@pytest.mark.parametrize("shape", [(0,), (2,), (2, 2)])
def test_grouped_program_values_complete_gradients_and_explicit_adjoint(device, basis, shape, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    model = _FactorizedPolynomialRuntime(schedule(), path_basis=basis).to(device)
    generator = torch.Generator().manual_seed(20261104)
    values = (torch.randn((*shape, model.schedule["input_dimension"]), dtype=torch.float64,
                          generator=generator)/3).to(device).requires_grad_()
    from ye3t.runtime.execution_plan import _load_extension
    _load_extension()
    counts = []
    original = torch.ops.ye3t_runtime.factorized_angular
    def counted(*args, **kwargs):
        counts.append(1)
        return original(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(torch.ops.ye3t_runtime, "factorized_angular", counted)
        expected = sequential(model, values)
        sequential_calls = len(counts)
        counts.clear()
        actual = model(values)
        grouped_calls = len(counts)
    assert grouped_calls < sequential_calls
    torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-13)
    weights = torch.linspace(.1, .7, actual.shape[-1], device=device, dtype=torch.float64).requires_grad_()
    results = []
    for output in (expected, actual):
        energy = (output*weights).square().sum()
        first = torch.autograd.grad(energy, values, create_graph=True)[0]
        second = torch.autograd.grad(.1*energy+first.square().sum(), (values, weights), retain_graph=True)
        results.append((first, *second))
    for observed, reference in zip(results[0], results[1], strict=True):
        torch.testing.assert_close(observed, reference, rtol=4e-10, atol=4e-12)
    if shape == (2,):
        seeds = torch.ones((2,)+tuple(actual.shape), device=device, dtype=values.dtype)
        explicit = model.batched_adjoint(values, seeds)[0]
        automatic = torch.autograd.grad(actual.sum(), values, create_graph=True, retain_graph=True)[0]
        torch.testing.assert_close(explicit, automatic, rtol=3e-11, atol=3e-12)
        torch.testing.assert_close(torch.autograd.grad(explicit.square().sum(), values)[0],
                                  torch.autograd.grad(automatic.square().sum(), values)[0],
                                  rtol=3e-10, atol=3e-11)
    changed = values.detach()+.03
    changed_reference = sequential(model, changed)
    plan = model._path_batch_plan(values)
    for _, indices, transform in plan[1].values():
        assert not indices.requires_grad
        assert transform is None or (not transform.requires_grad and transform.grad_fn is None)
    def unexpected_binding(*args, **kwargs):
        pytest.fail("Changed geometry rebuilt immutable angular bindings")
    with monkeypatch.context() as patch:
        patch.setattr(model, "_path_inputs", unexpected_binding)
        changed_actual = model(changed)
        assert model._path_batch_plan(changed) is plan
    torch.testing.assert_close(changed_actual, changed_reference, rtol=3e-12, atol=3e-13)


def test_batched_program_workspace_limit_and_dtype_rebinding(monkeypatch):
    import ye3t.runtime.factorized_polynomial as module
    model = _FactorizedPolynomialRuntime(schedule())
    values = torch.linspace(-.3, .3, 2*model.schedule["input_dimension"], dtype=torch.float64).reshape(2, -1)
    expected = sequential(model, values)
    with monkeypatch.context() as patch:
        patch.setattr(module, "_ANGULAR_BATCH_WORKSPACE_BYTES", 1)
        actual = model(values)
        assert model.last_local_group_report["workspace_coordinate_capacity"] == 1
        assert (model.last_local_group_report["native_dispatches"]
                == model.last_local_group_report["sequential_native_dispatches"])
    torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-13)
    assert model.path_batch_plans
    model.float()
    assert not model.path_batch_plans
    assert not model.local_group_plans
    torch.testing.assert_close(model(values.float()), expected.float(), rtol=3e-5, atol=3e-7)
    model.double()
    assert not model.path_batch_plans
    assert not model.local_group_plans
    torch.testing.assert_close(model(values), expected, rtol=3e-12, atol=3e-13)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_local_group_bindings_reuse_geometry_and_invalidate_state_load(device, monkeypatch):
    import ye3t.runtime.execution_plan as execution
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    model = _FactorizedPolynomialRuntime(schedule()).to(device)
    state_keys = tuple(model.state_dict())
    x = torch.randn(2, model.schedule["input_dimension"], dtype=torch.float64, device=device)/3
    expected = sequential(model, x)
    torch.testing.assert_close(model(x), expected, rtol=3e-12, atol=3e-13)
    report = model.last_local_group_report
    assert report["native_dispatches"] < report["sequential_native_dispatches"]
    assert model.local_group_plans
    assert tuple(model.state_dict()) == state_keys
    for plans in model.local_group_plans.values():
        for _, indices, transform, groups in plans:
            assert not indices.requires_grad
            assert transform is None or transform.grad_fn is None
            for group, members in groups:
                assert all(value.grad_fn is None and not value.requires_grad for value in group.buffers())
                assert all(bound._native_plan.compiler_schedule is bound.schedule for bound, _, _ in members)
                assert all(row["plan_hash"] is None for row in group.output_slices)
    def unexpected_group(*args, **kwargs):
        pytest.fail("Changed geometry rebuilt fixed local angular groups")
    with monkeypatch.context() as patch:
        patch.setattr(execution, "_PackedFactorizedAngularPlanGroup", unexpected_group)
        torch.testing.assert_close(model(x+.02), sequential(model, x+.02), rtol=3e-12, atol=3e-13)
    for assign in (False, True):
        model.load_state_dict({key: value.clone() for key, value in model.state_dict().items()}, assign=assign)
        assert not model.local_group_plans and not model.path_batch_plans
        torch.testing.assert_close(model(x), expected, rtol=3e-12, atol=3e-13)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize("rows", [0, 3])
def test_bound_gather_linear_adjoint_retains_only_indices(device, dtype, rows):
    from ye3t.runtime.execution_plan import _load_extension, prepare_source_arena_schedule
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    _load_extension()
    schedule = prepare_source_arena_schedule((3, 1, 3, 8, 0, 7, 1), producer_width=12)
    binding = tuple(torch.tensor(schedule[name], dtype=torch.long, device=device)
                    for name in ("gather_indices", "reverse_offsets", "reverse_output_indices", "center_types"))
    values = torch.randn(rows, 12, dtype=dtype, device=device, requires_grad=True)
    weights = torch.randn(rows, 7, dtype=dtype, device=device, requires_grad=True)
    direction = torch.randn_like(values)
    result = torch.ops.ye3t_runtime.source_arena_gather(
        values, *binding, torch.zeros(rows, dtype=torch.long, device=device))
    expected = values.index_select(-1, binding[0])
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    saved_kinds = []
    def save(value):
        saved_kinds.append((value.is_floating_point(), value.is_complex()))
        return value.detach()
    with torch.autograd.graph.saved_tensors_hooks(save, lambda value: value):
        first, = torch.autograd.grad(result, values, weights, create_graph=True)
    second, = torch.autograd.grad(first, weights, direction)
    reference_first, = torch.autograd.grad(expected, values, weights, create_graph=True)
    reference_second, = torch.autograd.grad(reference_first, weights, direction)
    assert saved_kinds and not any(real or complex_value for real, complex_value in saved_kinds)
    torch.testing.assert_close(first, reference_first, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(second, reference_second, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_rank_eight_mixed_angular_blocks_and_force_loss_gradients(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    # Complete compiler-selected rank-eight products of four l=1 and four l=2
    # factors, with two distinct radial assignments and two retained tags.
    model = _FactorizedPolynomialRuntime(schedule(block_size=4)).to(device)
    generator = torch.Generator().manual_seed(20261104)
    values = (torch.randn(2, model.schedule["input_dimension"], dtype=torch.float64,
                          generator=generator)/2).to(device).requires_grad_()
    weights = torch.linspace(.1, .7, model.schedule["output_dimension"],
                             device=device, dtype=values.dtype).requires_grad_()
    reference, observed = sequential(model, values), model(values)
    torch.testing.assert_close(observed, reference, rtol=3e-12, atol=3e-13)
    derivatives = []
    for output in (reference, observed):
        energy = (output*weights).sum()
        forces = -torch.autograd.grad(energy, values, create_graph=True)[0]
        derivatives.append((forces, *torch.autograd.grad(
            .1*energy.square()+(forces-.013).square().mean(), (values, weights))))
    for expected, actual in zip(*derivatives, strict=True):
        torch.testing.assert_close(actual, expected, rtol=4e-10, atol=4e-12)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fixed_outputs_follow_buffer_lifecycle_and_coefficient_derivatives(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    model = _FactorizedPolynomialRuntime(schedule(tag_count=0)).to(device)
    p = next(p for p in model.path_programs
             if (binding := model._fixed_output_binding(f"path_output_map_{p}")) is not None
             and binding[2] is None and binding[0].shape[0] > 1)
    name = f"path_output_map_{p}"
    matrix = getattr(model, name)
    output = torch.linspace(-.3, .5, 2*matrix.shape[0], device=device, dtype=matrix.dtype).reshape(2, 1, -1)
    # Identity reuse is an actual view; it allocates no matrix-product output.
    assert model._path_output(p, output, output).data_ptr() == output.data_ptr()
    assert model._select_outputs(p, output.flatten(-2)).data_ptr() == output.data_ptr()
    for assign in (False, True):
        state = {key: value.clone() for key, value in model.state_dict().items()}
        model.load_state_dict(state, assign=assign)
        assert model._path_output(p, output, output).data_ptr() == output.data_ptr()
        assert model._select_outputs(p, output.flatten(-2)).data_ptr() == output.data_ptr()
    matrix = getattr(model, name)

    matrix.requires_grad_(True)
    observed = model._path_output(p, output, output)
    expected = dense_output(model, p, output, output)
    torch.testing.assert_close(torch.autograd.grad(observed.square().sum(), matrix)[0],
                               torch.autograd.grad(expected.square().sum(), matrix)[0])
    assert name not in model.fixed_output_bindings
    matrix.requires_grad_(False)

    # A changed buffer must take the dense operation without a stale binding.
    model._bind_path_output(p, matrix.cpu())
    matrix.mul_(2)
    torch.testing.assert_close(model._path_output(p, output, output), dense_output(model, p, output, output))
    assert name not in model.fixed_output_bindings
    replacement = torch.eye(*matrix.shape, device=device, dtype=matrix.dtype)
    setattr(model, name, replacement)
    model._bind_path_output(p, replacement.cpu())
    setattr(model, name, replacement*3)
    torch.testing.assert_close(model._path_output(p, output, output), dense_output(model, p, output, output))
    assert name not in model.fixed_output_bindings

    # Unit gathers, including repeated inputs, use the exact stored map.
    replacement.zero_()
    replacement[0, :] = 1
    setattr(model, name, replacement)
    model._bind_path_output(p, replacement.cpu())
    assert model._fixed_output_binding(name)[2] is not None
    inputs = output.clone().requires_grad_()
    results = []
    for result in (model._path_output(p, inputs, inputs), dense_output(model, p, inputs, inputs)):
        first = torch.autograd.grad(result.square().sum(), inputs, create_graph=True)[0]
        results.append((result, first, torch.autograd.grad(first.square().sum(), inputs)[0]))
    for observed, expected in zip(*results, strict=True):
        torch.testing.assert_close(observed, expected)

    # A state load invalidates an old gather; evaluation still uses loaded data.
    state = {key: value.clone() for key, value in model.state_dict().items()}
    state[name] *= .5
    model.load_state_dict(state)
    torch.testing.assert_close(model._path_output(p, output, output), dense_output(model, p, output, output))
    assert name not in model.fixed_output_bindings
    indices = getattr(model, f"outputs_{p}")
    indices.zero_()
    torch.testing.assert_close(model._select_outputs(p, output.flatten(-2)),
                               output.flatten(-2).index_select(-1, indices))


def test_fixed_outputs_inference_tensors_and_dtype_rounding():
    from copy import deepcopy
    planned = deepcopy(schedule(tag_count=0))
    # Exercise a map entry that rounds to one only in float32. Classification
    # must use the materialized coefficients, not schedule-level Python floats.
    p = next(p for p, program in enumerate(planned["programs"])
             if program["schedule"]["output_map"])
    entries = planned["programs"][p]["schedule"]["output_map"]
    planned["programs"][p]["schedule"]["output_map"] = [
        (i, j, value*(1+1e-8)) for i, j, value in entries]
    model = _FactorizedPolynomialRuntime(planned)
    name = f"path_output_map_{p}"
    assert model._fixed_output_binding(name) is None
    model.float()
    assert model._fixed_output_binding(name) is not None
    model.double()
    assert model._fixed_output_binding(name) is None
    with torch.inference_mode():
        model = _FactorizedPolynomialRuntime(schedule(), backend="reference")
        assert not model.fixed_output_bindings
        values = torch.full((1, model.schedule["input_dimension"]), .2, dtype=torch.float64)
        torch.testing.assert_close(model(values), sequential(model, values))
