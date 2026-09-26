import torch


def test_real_tesseral_cartesian_derivatives_are_finite_on_axes():
    from ye3t.core.spherical import real_spherical_harmonics_l_from_cartesian_with_derivatives

    vectors = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -2.0],
            [1.0, 0.0, 0.0],
            [0.0, -3.0, 0.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=torch.float64,
    )
    for l_in in range(0, 5):
        values, dxyz = real_spherical_harmonics_l_from_cartesian_with_derivatives(l_in, vectors)
        assert values.shape == (2 * l_in + 1, vectors.shape[0])
        assert dxyz.shape == (2 * l_in + 1, vectors.shape[0], 3)
        assert torch.isfinite(values).all()
        assert torch.isfinite(dxyz).all()


def test_real_tesseral_cartesian_values_match_angle_backend_off_axes():
    from ye3t.core.spherical import (
        real_spherical_harmonics_l,
        real_spherical_harmonics_l_from_cartesian_with_derivatives,
    )

    vectors = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [-2.5, 0.7, 1.1],
            [0.4, -1.3, -2.2],
        ],
        dtype=torch.float64,
    )
    r = torch.linalg.norm(vectors, dim=-1)
    theta = torch.atan2(torch.linalg.norm(vectors[:, :2], dim=-1), vectors[:, 2])
    phi = torch.atan2(vectors[:, 1], vectors[:, 0])
    for l_in in range(0, 5):
        values, _ = real_spherical_harmonics_l_from_cartesian_with_derivatives(l_in, vectors)
        reference = real_spherical_harmonics_l(l_in, theta, phi)
        assert torch.allclose(values, reference, atol=1e-10, rtol=1e-10)
        assert torch.isfinite(r).all()
