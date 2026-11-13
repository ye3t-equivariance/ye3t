"""Empty physical batches and empty adjoint seed batches retain derivatives."""
import pytest
import torch

from ye3t.representations.young_angular_paths import angular_path_schedule
from ye3t.runtime.angular_paths import BoundAngularPaths


@pytest.fixture(scope="module")
def schedule():
    # A genuine mixed Young type: two tableau components and complete L=1.
    return angular_path_schedule(3, 1, (2, 1), 1)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128],
                         ids=["float32", "float64", "complex64", "complex128"])
@pytest.mark.parametrize("shape,seeds_count", [((0,), 2), ((2, 0), 2), ((2,), 0)],
                         ids=["empty_rows", "empty_inner_batch", "empty_seeds"])
def test_empty_angular_values_and_seed_dependent_second_derivatives(schedule, device, dtype,
                                                                  shape, seeds_count, basis="complex_condon_shortley"):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA for native empty angular execution")
    generator = torch.Generator().manual_seed(271)
    original = [torch.randn(*shape, 3, dtype=dtype, generator=generator) for _ in range(3)]
    seeds_data = torch.randn(seeds_count, *shape, schedule["copies"], schedule["tableaux"], 3,
                             dtype=dtype, generator=generator)
    results = []
    for backend in ("reference", "native"):
        factors = tuple(value.to(device).detach().requires_grad_(True) for value in original)
        seeds = seeds_data.to(device).detach().requires_grad_(True)
        runtime = BoundAngularPaths(schedule, dtype=dtype, device=device, backend=backend, basis=basis)
        output = runtime.evaluate(factors)
        assert output.shape == (*shape, schedule["copies"], schedule["tableaux"], 3)
        gradient = torch.autograd.grad(output.real.sum(), factors, create_graph=True, retain_graph=True)
        second = torch.autograd.grad(sum(value.real.sum() for value in gradient), factors,
                                     retain_graph=True)
        explicit = runtime.batched_adjoint(factors, seeds)
        # Dependence on both the input values and trainable upstream weights must
        # survive zero-sized axes. Disconnected new_zeros() is not sufficient.
        derivatives = torch.autograd.grad(sum(value.real.sum() for value in explicit),
                                          factors + (seeds,))
        results.append((output, *gradient, *second, *explicit, *derivatives))
    for expected, actual in zip(*results):
        assert actual.shape == expected.shape
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_native_empty_operator_keeps_registered_autograd(schedule, device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA for native empty angular operator")
    runtime = BoundAngularPaths(schedule, dtype=dtype, device=device, backend="native")
    packed = torch.empty(0, 9, dtype=dtype, device=device, requires_grad=True)
    width = len(schedule["outputs"])*3
    output = torch.ops.ye3t_runtime.factorized_angular(
        packed, *runtime.native_arguments, 1, runtime.workspace_dimension, width)
    seed = torch.empty_like(output, requires_grad=True)
    first, = torch.autograd.grad(output, packed, seed, create_graph=True)
    second = torch.autograd.grad(first.real.sum(), (packed, seed))
    assert output.shape == (0, width)
    assert first.shape == packed.shape
    assert second[0].shape == packed.shape
    assert second[1].shape == seed.shape
