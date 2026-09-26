import pytest
import torch


def test_openequivariance_basis_bridge_closes_angular_triples():
    pytest.importorskip("openequivariance")
    from ye3t.backends.openequivariance_bridge import (
        _openequivariance_basis_transform_cpu,
        _openequivariance_cg_scale_cpu,
        _openequivariance_component_cg_tensor_cpu,
        _ye3t_real_cg_tensor_cpu,
    )

    triples = (
        (0, 1, 1),
        (1, 1, 0),
        (1, 1, 1),
        (1, 1, 2),
        (2, 1, 1),
        (2, 1, 2),
        (2, 1, 3),
        (2, 2, 0),
        (2, 2, 1),
        (2, 2, 2),
        (2, 2, 3),
        (2, 2, 4),
        (3, 2, 1),
        (3, 2, 2),
        (3, 2, 3),
        (3, 2, 4),
    )
    for left_L, right_L, out_L in triples:
        left_transform = _openequivariance_basis_transform_cpu(left_L)
        right_transform = _openequivariance_basis_transform_cpu(right_L)
        output_inverse = _openequivariance_basis_transform_cpu(out_L).T
        candidate = torch.einsum(
            "ai,bj,ijk,kc->abc",
            left_transform,
            right_transform,
            _openequivariance_component_cg_tensor_cpu(
                left_L, right_L, out_L
            ),
            output_inverse,
        )
        candidate = (
            _openequivariance_cg_scale_cpu(left_L, right_L, out_L)
            * candidate
        )
        torch.testing.assert_close(
            candidate,
            _ye3t_real_cg_tensor_cpu(left_L, right_L, out_L),
            atol=5.0e-12,
            rtol=5.0e-12,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_openequivariance_paired_cg_matches_ye3t_value_vjp_and_hvp():
    pytest.importorskip("openequivariance")
    from ye3t.paired_cg import couple_packed_real_tesseral

    generator = torch.Generator().manual_seed(811)
    left = torch.randn(5, 3, 5, dtype=torch.float64, generator=generator).cuda()
    right = torch.randn(5, 3, 3, dtype=torch.float64, generator=generator).cuda()
    cotangent = torch.randn(
        5, 3, 5, dtype=torch.float64, generator=generator
    ).cuda()
    left_direction = torch.randn(
        left.shape, dtype=torch.float64, generator=generator
    ).cuda()
    right_direction = torch.randn(
        right.shape, dtype=torch.float64, generator=generator
    ).cuda()

    def evaluate(backend, left_value, right_value):
        value, selected_backend = couple_packed_real_tesseral(
            left_value,
            right_value,
            2,
            1,
            2,
            backend=backend,
            strict_backend=True,
        )
        return value, selected_backend

    reference, _reference_backend = evaluate("pytorch", left, right)
    actual, actual_backend = evaluate("openequivariance", left, right)
    assert actual_backend == "openequivariance_paired_uvu"
    torch.testing.assert_close(actual, reference, atol=5.0e-11, rtol=5.0e-11)

    def derivatives(backend):
        left_value = left.detach().requires_grad_(True)
        right_value = right.detach().requires_grad_(True)
        output, _selected_backend = evaluate(
            backend, left_value, right_value
        )
        gradients = torch.autograd.grad(
            torch.sum(output * cotangent),
            (left_value, right_value),
            create_graph=True,
        )
        hvp = torch.autograd.grad(
            torch.sum(gradients[0] * left_direction)
            + torch.sum(gradients[1] * right_direction),
            (left_value, right_value),
        )
        return gradients, hvp

    reference_gradients, reference_hvp = derivatives("pytorch")
    actual_gradients, actual_hvp = derivatives("openequivariance")
    for actual_value, reference_value in zip(
        actual_gradients + actual_hvp,
        reference_gradients + reference_hvp,
    ):
        torch.testing.assert_close(
            actual_value,
            reference_value,
            atol=5.0e-11,
            rtol=5.0e-11,
        )
