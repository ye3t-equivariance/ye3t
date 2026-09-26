import math

import pytest
import torch

from ye3t.runtime import (
    compact_exterior_pair_product,
    compact_exterior_power_adjoint_reference,
    compact_exterior_power_product,
    compact_exterior_power_product_reference,
    compact_pair_product_adjoint_reference,
    compact_pair_product_reference,
    compact_symmetric_pair_product,
)


pytestmark = pytest.mark.fast


def test_compact_symmetric_pair_product_uses_normalized_basis():
    left = torch.tensor([[2.0, 3.0]], dtype=torch.float64)
    right = torch.tensor([[5.0, 7.0]], dtype=torch.float64)
    actual = compact_symmetric_pair_product(
        left,
        right,
        backend="reference",
    )
    expected = torch.tensor(
        [[10.0, (14.0 + 15.0) / math.sqrt(2.0), 21.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-14)


def test_compact_exterior_pair_product_annihilates_duplicates():
    value = torch.tensor([[2.0, -3.0, 5.0]], dtype=torch.float64)
    actual = compact_exterior_pair_product(
        value,
        value,
        backend="reference",
    )
    torch.testing.assert_close(
        actual,
        torch.zeros_like(actual),
        rtol=0.0,
        atol=1e-14,
    )


def test_compact_exterior_power_rank_three_uses_normalized_determinants():
    factors = torch.tensor(
        [
            [
                [2.0, 0.0, 0.0, 3.0],
                [0.0, 5.0, 0.0, 7.0],
                [0.0, 0.0, 11.0, 13.0],
            ]
        ],
        dtype=torch.float64,
    )
    actual = compact_exterior_power_product(
        factors,
        backend="reference",
    )
    scale = 1.0 / math.sqrt(6.0)
    expected = torch.stack(
        (
            torch.linalg.det(factors[..., (0, 1, 2)]),
            torch.linalg.det(factors[..., (0, 1, 3)]),
            torch.linalg.det(factors[..., (0, 2, 3)]),
            torch.linalg.det(factors[..., (1, 2, 3)]),
        ),
        dim=-1,
    ) * scale
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-14)


def test_compact_exterior_power_has_signed_factor_action_and_duplicate_zero():
    factors = torch.randn(2, 3, 5, dtype=torch.float64)
    actual = compact_exterior_power_product_reference(factors)
    swapped = compact_exterior_power_product_reference(
        factors[:, (1, 0, 2), :]
    )
    duplicate = factors.clone()
    duplicate[:, 1, :] = duplicate[:, 0, :]

    torch.testing.assert_close(swapped, -actual, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(
        compact_exterior_power_product_reference(duplicate),
        torch.zeros_like(actual),
        rtol=0.0,
        atol=1e-12,
    )


def test_compact_exterior_power_rank_two_matches_pair_and_adjoint():
    factors = torch.randn(
        2,
        2,
        4,
        dtype=torch.float64,
        requires_grad=True,
    )
    actual = compact_exterior_power_product_reference(factors)
    pair = compact_pair_product_reference(
        factors[:, 0, :],
        factors[:, 1, :],
        antisymmetric=True,
    )
    torch.testing.assert_close(actual, pair, rtol=0.0, atol=1e-12)

    output_adjoint = torch.randn_like(actual)
    expected = torch.autograd.grad(
        actual,
        factors,
        output_adjoint,
        create_graph=True,
    )[0]
    adjoint = compact_exterior_power_adjoint_reference(
        output_adjoint,
        factors,
    )
    torch.testing.assert_close(adjoint, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("antisymmetric", [False, True])
def test_compact_pair_reference_adjoint_matches_autograd(antisymmetric):
    left = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    right = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    output = compact_pair_product_reference(
        left,
        right,
        antisymmetric=antisymmetric,
    )
    output_adjoint = torch.randn_like(output)
    expected = torch.autograd.grad(
        output,
        (left, right),
        output_adjoint,
        create_graph=True,
    )
    actual = compact_pair_product_adjoint_reference(
        output_adjoint,
        left,
        right,
        antisymmetric=antisymmetric,
    )
    torch.testing.assert_close(actual[0], expected[0], rtol=0.0, atol=1e-12)
    torch.testing.assert_close(actual[1], expected[1], rtol=0.0, atol=1e-12)
