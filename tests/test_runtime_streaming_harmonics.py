"""The streaming harmonic recurrence of the native core is reachable from Python."""

import numpy as np
import pytest
import torch

from ye3t.core.spherical import spherical_harmonics_l
from ye3t.runtime.execution_plan import _NATIVE_EXECUTION_PLAN_ABI, _load_extension


def recurrence(displacements, maximum_l):
    extension = _load_extension()
    assert int(extension.core_abi_version()) == _NATIVE_EXECUTION_PLAN_ABI
    return torch.ops.ye3t_runtime.complex_spherical_harmonics_recurrence(
        displacements, int(maximum_l), 1.0
    )


def test_streaming_recurrence_matches_the_reference_harmonics():
    maximum_l = 4
    displacements = torch.as_tensor(np.random.default_rng(3).normal(size=(9, 3)))
    values, derivatives = recurrence(displacements, maximum_l)
    width = (maximum_l + 1) * (maximum_l + 2) // 2
    assert values.shape == (9, width) and derivatives.shape == (9, width, 3)
    unit = displacements / torch.linalg.vector_norm(displacements, dim=1, keepdim=True)
    theta = torch.arccos(unit[:, 2].clamp(-1.0, 1.0))
    phi = torch.atan2(unit[:, 1], unit[:, 0])
    for l in range(maximum_l + 1):
        reference = spherical_harmonics_l(l, theta, phi)
        for m in range(l + 1):
            assert torch.allclose(
                values[:, l * (l + 1) // 2 + m], reference[l + m], atol=1.0e-12
            )


def test_streaming_recurrence_derivatives_match_finite_differences():
    maximum_l = 3
    displacements = torch.as_tensor(np.random.default_rng(5).normal(size=(4, 3)))
    _, derivatives = recurrence(displacements, maximum_l)
    step = 1.0e-6
    for axis in range(3):
        shift = torch.zeros_like(displacements)
        shift[:, axis] = step
        plus, _ = recurrence(displacements + shift, maximum_l)
        minus, _ = recurrence(displacements - shift, maximum_l)
        assert torch.allclose(
            derivatives[:, :, axis], (plus - minus) / (2.0 * step), atol=1.0e-7
        )


def test_zero_length_displacements_are_rejected():
    with pytest.raises(RuntimeError, match="positive length"):
        recurrence(torch.zeros((1, 3), dtype=torch.float64), 2)
