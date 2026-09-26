import pytest
import torch

from ye3t.runtime import packed_concatenate


pytestmark = pytest.mark.fast


def _objective_and_derivatives(parts, operation, directions):
    output = operation(parts)
    objective = (
        output.real.square()
        + 0.7 * output.imag.square()
        if output.is_complex()
        else output.square()
    ).square().sum()
    first = torch.autograd.grad(objective, parts, create_graph=True)
    directional = sum(
        (
            (gradient.conj() * direction).real.sum()
            if gradient.is_complex()
            else (gradient * direction).sum()
        )
        for gradient, direction in zip(first, directions)
    )
    second = torch.autograd.grad(directional, parts)
    return output, first, second


@pytest.mark.parametrize(
    "dtype",
    (torch.float32, torch.float64, torch.complex64, torch.complex128),
)
@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_packed_concatenate_matches_cat_value_vjp_and_hvp(dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    generator = torch.Generator().manual_seed(731)

    def sample(shape):
        real = torch.randn(shape, dtype=torch.float64, generator=generator)
        if dtype == torch.complex128:
            imaginary = torch.randn(
                shape,
                dtype=torch.float64,
                generator=generator,
            )
            value = torch.complex(real, imaginary)
        else:
            value = real
        return value.to(device=device, dtype=dtype)

    reference_parts = tuple(
        sample((3, width)).requires_grad_(True)
        for width in (2, 0, 5, 1)
    )
    candidate_parts = tuple(
        value.detach().clone().requires_grad_(True)
        for value in reference_parts
    )
    directions = tuple(sample(value.shape) for value in reference_parts)
    reference = _objective_and_derivatives(
        reference_parts,
        lambda values: torch.cat(values, dim=-1),
        directions,
    )
    candidate = _objective_and_derivatives(
        candidate_parts,
        packed_concatenate,
        directions,
    )
    for reference_values, candidate_values in zip(reference, candidate):
        if isinstance(reference_values, tuple):
            for reference_value, candidate_value in zip(
                reference_values,
                candidate_values,
            ):
                torch.testing.assert_close(
                    candidate_value,
                    reference_value,
                    rtol=2.0e-5 if dtype in {torch.float32, torch.complex64} else 1.0e-11,
                    atol=2.0e-6 if dtype in {torch.float32, torch.complex64} else 1.0e-12,
                )
        else:
            torch.testing.assert_close(
                candidate_values,
                reference_values,
                rtol=2.0e-5 if dtype in {torch.float32, torch.complex64} else 1.0e-11,
                atol=2.0e-6 if dtype in {torch.float32, torch.complex64} else 1.0e-12,
            )


@pytest.mark.parametrize("dtype", (torch.float64, torch.complex128))
def test_packed_concatenate_gradcheck_and_gradgradcheck(dtype):
    parts = tuple(
        torch.randn(2, width, dtype=dtype, requires_grad=True)
        for width in (2, 3, 1)
    )
    operation = lambda *values: packed_concatenate(values)
    assert torch.autograd.gradcheck(operation, parts)
    assert torch.autograd.gradgradcheck(operation, parts)


def test_packed_concatenate_single_part_is_identity():
    value = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    assert packed_concatenate((value,)) is value


def test_packed_concatenate_validates_layout():
    with pytest.raises(ValueError, match="at least one"):
        packed_concatenate(())
    with pytest.raises(ValueError, match="leading axes"):
        packed_concatenate((torch.zeros(2, 3), torch.zeros(3, 4)))
    with pytest.raises(TypeError, match="dtype and device"):
        packed_concatenate(
            (torch.zeros(2, 3), torch.zeros(2, 4, dtype=torch.float64))
        )
