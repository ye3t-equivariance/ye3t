"""Existing segmented execution preserves packed graphs and their lifecycle."""
from functools import lru_cache

import pytest
import torch

from test_execution_plan_native import _rank3_factorized_role_plan
from ye3t.runtime.execution_plan import (
    YE3TFactorizedAngularModule, _PackedFactorizedAngularPlanGroup, _load_extension,
)


@lru_cache(maxsize=1)
def _plans():
    return tuple(_rank3_factorized_role_plan(L) for L in (1, 2))


def _group(device, dtype, shared):
    _load_extension()
    modules = [YE3TFactorizedAngularModule(plan, backend="native", dtype=dtype, device=device)
               for plan in _plans()]
    options = dict(heterogeneous_roots=True)
    if shared:
        options.update(physical_leaf_offsets=((0, 3, 8), (0, 3, 8)), physical_input_dimension=11)
    return _PackedFactorizedAngularPlanGroup(modules, (0, 1), **options)


def _original(group, values):
    return torch.ops.ye3t_runtime.factorized_angular_heterogeneous(
        values, *group._native_arguments(), group.source_dimension,
        group.workspace_dimension, group.output_dimension)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("rows", [0, 3])
def test_packed_dispatch_complete_derivatives_and_buffer_reuse(device, dtype, shared, rows, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    monkeypatch.setenv("YE3T_FACTORIZED_ANGULAR_CUDA_POLICY", "serial")
    monkeypatch.setenv("YE3T_FACTORIZED_ANGULAR_SEGMENTED_CUDA_POLICY", "auto")
    group = _group(device, dtype, shared)
    assert group.source_dimension > 1
    if shared:
        assert group.subtree_evaluation_shared
    width = group.physical_input_dimension or sum(group.input_dimensions)
    generator = torch.Generator().manual_seed(1108)
    values = (.2*torch.randn(rows, group.source_dimension*width, dtype=dtype,
                            generator=generator)).to(device).requires_grad_()
    weights = torch.linspace(.1, .5, group.output_dimension, dtype=torch.float64,
                             device=device).requires_grad_()
    original_arrays = tuple(value.clone() for value in group._native_arguments())
    expected = _original(group, values)
    actual = group._evaluate_packed_output(values)
    assert group.last_backend == ("native_segmented_heterogeneous_root_factorized_angular"
        if device == "cuda" else "native_heterogeneous_root_factorized_angular")
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
    derivatives = []
    for output in (expected, actual):
        energy = (output*weights).abs().square().sum()
        first = torch.autograd.grad(energy, values, create_graph=True, retain_graph=True)[0]
        loss = .1*energy+first.abs().square().sum()
        derivatives.append((first, *torch.autograd.grad(loss, (values, weights), retain_graph=True)))
    for observed, wanted in zip(*derivatives, strict=True):
        torch.testing.assert_close(observed, wanted, rtol=2e-9, atol=2e-11)
    for observed, wanted in zip(group._native_arguments(), original_arrays, strict=True):
        torch.testing.assert_close(observed, wanted, rtol=0, atol=0)
    buffer_ids = {name: id(value) for name, value in group.named_buffers()}
    changed = values.detach()+.013
    torch.testing.assert_close(group._evaluate_packed_output(changed), _original(group, changed),
                              rtol=2e-11, atol=2e-12)
    assert buffer_ids == {name: id(value) for name, value in group.named_buffers()}
    # Execution metadata is regenerated from the compiler plan, not serialized
    # as additional model state or retained geometry.
    assert not group.state_dict()


@pytest.mark.parametrize("shared", [False, True])
def test_segment_metadata_moves_with_existing_buffers(shared, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    monkeypatch.setenv("YE3T_FACTORIZED_ANGULAR_SEGMENTED_CUDA_POLICY", "auto")
    group = _group("cpu", torch.float64, shared)
    for device, dtype in (("cuda", torch.float64), ("cpu", torch.complex128), ("cuda", torch.complex128)):
        group.to(device=device, dtype=dtype)
        width = group.physical_input_dimension or sum(group.input_dimensions)
        values = torch.full((2, group.source_dimension*width), .2, dtype=dtype,
                            device=device, requires_grad=True)
        expected, actual = _original(group, values), group._evaluate_packed_output(values)
        torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
        for buffer in group.buffers():
            assert buffer.device == values.device
        for name, buffer in group.named_buffers():
            if name.startswith("segment_"):
                assert buffer.dtype == torch.int64
        first = torch.autograd.grad(actual.abs().square().sum(), values, create_graph=True)[0]
        assert torch.isfinite(torch.autograd.grad(first.abs().square().sum(), values)[0]).all()


@pytest.mark.parametrize("field", ["source_dimension", "workspace_dimension", "output_dimension", "physical_input_dimension"])
@pytest.mark.parametrize("rows", [0, 2])
def test_prepared_segment_rejects_stale_dimensions_before_launch(field, rows, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    group = _group("cuda", torch.float64, True)
    values = torch.full((rows, group.source_dimension*group.physical_input_dimension),
                        .2, dtype=torch.float64, device="cuda")
    setattr(group, field, getattr(group, field)+1)
    def forbidden(*args, **kwargs):
        pytest.fail("stale segment dimensions reached native allocation")
    monkeypatch.setattr(torch.ops.ye3t_runtime, "factorized_angular_segmented", forbidden)
    with pytest.raises(RuntimeError, match="dimensions changed after preparation"):
        group._evaluate_packed_output(values)
