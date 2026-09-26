

import math
from functools import lru_cache

import torch

_PI = math.pi
_LOG_4PI = math.log(4.0 * math.pi)


def _as_real_tensor(x, *, like = None):
    if torch.is_tensor(x):
        return x
    if like is not None:
        return torch.as_tensor(x, dtype=like.dtype, device=like.device)
    return torch.as_tensor(x)


def _broadcast_angles(theta, phi):
    theta, phi = torch.broadcast_tensors(theta, phi)
    return theta, phi


def associated_legendre_l(l, x):
    """
    Return [P_l^0(x), ..., P_l^l(x)] using stable three-term recurrences.
    Includes the Condon-Shortley phase.
    """
    if l < 0:
        raise ValueError('l must be non-negative')

    one = torch.ones_like(x)
    if l == 0:
        return [one]

    # Build P_m^m for all m iteratively.
    sqrt_term = torch.sqrt(torch.clamp(1.0 - x * x, min=0.0))
    Pmm = one
    diag = [Pmm]
    for m in range(1, l + 1):
        Pmm = -(2 * m - 1) * sqrt_term * Pmm
        diag.append(Pmm)

    out = []
    for m in range(0, l + 1):
        if m == l:
            out.append(diag[m])
            continue

        P_lm_minus2 = diag[m]
        P_lm_minus1 = (2 * m + 1) * x * P_lm_minus2
        if l == m + 1:
            out.append(P_lm_minus1)
            continue

        for ell in range(m + 2, l + 1):
            P_lm = ((2 * ell - 1) * x * P_lm_minus1 - (ell + m - 1) * P_lm_minus2) / (ell - m)
            P_lm_minus2, P_lm_minus1 = P_lm_minus1, P_lm
        out.append(P_lm_minus1)

    return out


@lru_cache(maxsize=None)
def _normalization_constants(l):
    vals = []
    for m in range(0, l + 1):
        log_norm = 0.5 * (
            math.log(2 * l + 1)
            - _LOG_4PI
            + math.lgamma(l - m + 1)
            - math.lgamma(l + m + 1)
        )
        vals.append(math.exp(log_norm))
    return tuple(vals)


@lru_cache(maxsize=None)
def _legendre_coefficients_ascending(l):
    """Ordinary Legendre coefficients in ascending powers.

    The Cartesian real-tesseral path below uses
    ``P_l^m(z)=(-1)^m (1-z^2)^{m/2} d^m P_l(z)/dz^m`` so that the
    ``(1-z^2)^{m/2}`` factor cancels the azimuthal ``rho^{-m}`` term before
    numerical evaluation. This is the regular solid-harmonic construction,
    expressed in the same Condon-Shortley convention as DLMF 14.30:
    https://dlmf.nist.gov/14.30
    """

    if l < 0:
        raise ValueError("l must be non-negative")
    if l == 0:
        return (1.0,)
    p_nm1 = [1.0]
    p_n = [0.0, 1.0]
    for n in range(1, l):
        out = [0.0] * (n + 2)
        for power, coeff in enumerate(p_n):
            out[power + 1] += ((2 * n + 1) / (n + 1)) * coeff
        for power, coeff in enumerate(p_nm1):
            out[power] -= (n / (n + 1)) * coeff
        p_nm1, p_n = p_n, out
    return tuple(float(v) for v in p_n)


@lru_cache(maxsize=None)
def _legendre_derivative_coefficients_ascending(l, m):
    coeffs = list(_legendre_coefficients_ascending(l))
    for _ in range(int(m)):
        coeffs = [(power + 1) * coeffs[power + 1] for power in range(len(coeffs) - 1)]
        if not coeffs:
            coeffs = [0.0]
            break
    return tuple(float(v) for v in coeffs)


def _polyval_ascending(coeffs, x):
    out = torch.zeros_like(x)
    for coeff in reversed(tuple(coeffs)):
        out = out * x + torch.as_tensor(coeff, dtype=x.dtype, device=x.device)
    return out


def real_spherical_harmonics_l_from_unit_cartesian(l, unit_xyz):
    r"""
    Return real tesseral harmonics from unit Cartesian directions.

    This avoids differentiating through ``atan2`` or ``rho=sqrt(x^2+y^2)``.
    For ``m>0`` the product

    ``P_l^m(z) cos(m phi)`` or ``P_l^m(z) sin(m phi)``

    is evaluated after analytically cancelling the azimuthal denominator:

    ``P_l^m(z) e^{i m phi} = (-1)^m d^m P_l(z)/dz^m (x+i y)^m``

    on the unit sphere.  This gives a regular Cartesian polynomial on the
    sphere, with the radial ``xyz / ||xyz||`` normalization handled outside
    this helper.
    """

    if unit_xyz.shape[-1] != 3:
        raise ValueError("unit_xyz must have trailing dimension 3")
    if l < 0:
        raise ValueError("l must be non-negative")
    x = unit_xyz[..., 0]
    y = unit_xyz[..., 1]
    z = unit_xyz[..., 2]
    norms = _normalization_constants(l)
    rt2 = torch.sqrt(torch.as_tensor(2.0, dtype=unit_xyz.dtype, device=unit_xyz.device))

    out = [None] * (2 * l + 1)
    out[l] = torch.as_tensor(norms[0], dtype=unit_xyz.dtype, device=unit_xyz.device) * _polyval_ascending(
        _legendre_coefficients_ascending(l),
        z,
    )
    if l == 0:
        return torch.stack(out, dim=0)

    z_complex = torch.complex(x, y)
    power = torch.ones_like(z_complex)
    for m in range(1, l + 1):
        power = power * z_complex
        derivative = _polyval_ascending(_legendre_derivative_coefficients_ascending(l, m), z)
        norm = torch.as_tensor(norms[m], dtype=unit_xyz.dtype, device=unit_xyz.device)
        out[l + m] = rt2 * norm * derivative * power.real
        out[l - m] = -rt2 * norm * derivative * power.imag
    return torch.stack(out, dim=0)


def spherical_harmonics_l(l, theta, phi):
    """
    Return all complex spherical harmonics for fixed ``l`` in m-order
    ``[-l, ..., l]`` with shape ``(2*l+1, *theta.shape)``.
    """
    theta = _as_real_tensor(theta)
    phi = _as_real_tensor(phi, like=theta)
    theta, phi = _broadcast_angles(theta, phi)
    x = torch.cos(theta)
    P = associated_legendre_l(l, x)
    norms = _normalization_constants(l)
    complex_dtype = torch.complex64 if theta.dtype == torch.float32 else torch.complex128

    out = [None] * (2 * l + 1)
    for m in range(0, l + 1):
        norm = torch.as_tensor(norms[m], dtype=theta.dtype, device=theta.device)
        phase = torch.exp((1j * m) * phi).to(complex_dtype)
        y_pos = (norm * P[m]).to(complex_dtype) * phase
        out[l + m] = y_pos
        if m > 0:
            out[l - m] = ((-1) ** m) * torch.conj(y_pos)

    return torch.stack(out, dim=0)


def spherical_harmonics_l_with_derivatives(l, theta, phi):
    """
    Return ``Y_lm``, ``dY_lm/dtheta``, and ``dY_lm/dphi`` in m-order
    ``[-l, ..., l]``.

    The angular derivatives use representation identities rather than asking
    autograd to differentiate the associated-Legendre recurrence.
    """
    theta = _as_real_tensor(theta)
    phi = _as_real_tensor(phi, like=theta)
    theta, phi = _broadcast_angles(theta, phi)
    y = spherical_harmonics_l(l, theta, phi)
    complex_dtype = y.dtype

    m_values = torch.arange(-l, l + 1, dtype=theta.dtype, device=theta.device)
    dphi = (1j * m_values).to(complex_dtype).reshape((2 * l + 1,) + (1,) * theta.ndim) * y

    dtheta_terms = []
    exp_pos = torch.exp(1j * phi).to(complex_dtype)
    exp_neg = torch.exp(-1j * phi).to(complex_dtype)
    zero = torch.zeros_like(y[0])
    for m in range(-l, l + 1):
        upper = y[m + 1 + l] if m + 1 <= l else zero
        lower = y[m - 1 + l] if m - 1 >= -l else zero
        upper_coeff = math.sqrt(max((l - m) * (l + m + 1), 0))
        lower_coeff = math.sqrt(max((l + m) * (l - m + 1), 0))
        dtheta_terms.append(0.5 * (upper_coeff * exp_neg * upper - lower_coeff * exp_pos * lower))
    dtheta = torch.stack(dtheta_terms, dim=0)

    return y, dtheta, dphi


def real_spherical_harmonics_l(l, theta, phi):
    r"""
    Return real orthonormal spherical harmonics for fixed ``l``.

    The returned rows are ordered by signed ``m`` as ``[-l, ..., l]`` and use
    the tesseral convention induced by the complex Condon-Shortley harmonics
    returned by :func:`spherical_harmonics_l`:

    .. math::

        Y^{real}_{l,m} =
        \begin{cases}
          \sqrt{2}(-1)^{|m|+1} \operatorname{Im} Y_l^{|m|}, & m < 0,\\
          Y_l^0, & m = 0,\\
          \sqrt{2}(-1)^m \operatorname{Re} Y_l^m, & m > 0.
        \end{cases}

    Equivalently, for ``m > 0`` with the associated Legendre functions used by
    the complex backend,

    .. math::

        Y^{real}_{l,m} =
          \sqrt{2}(-1)^m N_{l,m} P_l^m(\cos\theta)\cos(m\phi),

        Y^{real}_{l,-m} =
          \sqrt{2}(-1)^{m+1} N_{l,m} P_l^m(\cos\theta)\sin(m\phi),

    where ``N_{l,m} = sqrt((2l+1)/(4*pi) * (l-m)!/(l+m)!)``. These equations
    match the real-form convention described by the spherical harmonics
    real-form equations and the DLMF complex harmonic normalization
    (https://dlmf.nist.gov/14.30), with the Condon-Shortley phase already
    included in ``P_l^m`` by :func:`associated_legendre_l`.
    """

    y = spherical_harmonics_l(l, theta, phi)
    rt2 = torch.sqrt(torch.as_tensor(2.0, dtype=y.real.dtype, device=y.device))
    out = []
    for m in range(-l, l + 1):
        if m < 0:
            sign = 1.0 if ((-m) % 2) else -1.0
            out.append(rt2 * sign * y[l + abs(m)].imag)
        elif m == 0:
            out.append(y[l].real)
        else:
            sign = -1.0 if (m % 2) else 1.0
            out.append(rt2 * sign * y[l + m].real)
    return torch.stack(out, dim=0)


def real_spherical_harmonics_l_with_derivatives(l, theta, phi):
    r"""
    Return real ``Y_lm``, ``dY_lm/dtheta``, and ``dY_lm/dphi``.

    This applies the same real/tesseral transformation as
    :func:`real_spherical_harmonics_l` to the complex analytic derivative
    identities used by :func:`spherical_harmonics_l_with_derivatives`.
    """

    y, dtheta_complex, dphi_complex = spherical_harmonics_l_with_derivatives(l, theta, phi)
    rt2 = torch.sqrt(torch.as_tensor(2.0, dtype=y.real.dtype, device=y.device))
    values = []
    dtheta = []
    dphi = []
    for m in range(-l, l + 1):
        if m < 0:
            sign = 1.0 if ((-m) % 2) else -1.0
            row = l + abs(m)
            values.append(rt2 * sign * y[row].imag)
            dtheta.append(rt2 * sign * dtheta_complex[row].imag)
            dphi.append(rt2 * sign * dphi_complex[row].imag)
        elif m == 0:
            values.append(y[l].real)
            dtheta.append(dtheta_complex[l].real)
            dphi.append(dphi_complex[l].real)
        else:
            sign = -1.0 if (m % 2) else 1.0
            row = l + m
            values.append(rt2 * sign * y[row].real)
            dtheta.append(rt2 * sign * dtheta_complex[row].real)
            dphi.append(rt2 * sign * dphi_complex[row].real)
    return torch.stack(values, dim=0), torch.stack(dtheta, dim=0), torch.stack(dphi, dim=0)


def real_spherical_harmonics_l_from_cartesian_with_derivatives(
    l,
    xyz,
    *,
    eps = 1.0e-12,
):
    """
    Return real spherical harmonics and Cartesian derivatives for edge vectors.

    The derivative tensor has shape ``(2*l+1, *xyz.shape[:-1], 3)`` and follows
    the signed-``m`` real row order ``[-l, ..., l]``.
    """

    if xyz.shape[-1] != 3:
        raise ValueError("xyz must have trailing dimension 3")
    xyz = _as_real_tensor(xyz)
    with torch.enable_grad():
        work = xyz.detach().clone().requires_grad_(True)
        r = torch.linalg.norm(work, dim=-1)
        safe_r = torch.clamp(r, min=eps)
        unit = work / safe_r.unsqueeze(-1)
        values = real_spherical_harmonics_l_from_unit_cartesian(l, unit)
        rows = []
        for row in values:
            grad = torch.autograd.grad(row.sum(), work, retain_graph=True, create_graph=False)[0]
            rows.append(grad)
    return values.detach(), torch.stack(rows, dim=0).detach()


def spherical_harmonics_l_from_cartesian_with_derivatives(
    l,
    xyz,
    *,
    eps = 1.0e-12,
):
    """
    Return ``Y_lm`` and Cartesian derivatives ``dY_lm/dxyz`` for edge vectors.

    The derivative tensor has shape ``(2*l+1, *xyz.shape[:-1], 3)``. Inputs
    should avoid the polar singularity for high-accuracy derivative checks.
    """
    if xyz.shape[-1] != 3:
        raise ValueError("xyz must have trailing dimension 3")
    xyz = _as_real_tensor(xyz)
    x = xyz[..., 0]
    y_cart = xyz[..., 1]
    z = xyz[..., 2]
    rho2 = x * x + y_cart * y_cart
    rho = torch.sqrt(torch.clamp(rho2, min=eps * eps))
    r2 = rho2 + z * z
    safe_r2 = torch.clamp(r2, min=eps * eps)
    theta = torch.atan2(rho, z)
    phi = torch.atan2(y_cart, x)
    harmonics, dtheta, dphi = spherical_harmonics_l_with_derivatives(l, theta, phi)

    dtheta_dxyz = torch.stack(
        (
            z * x / (safe_r2 * rho),
            z * y_cart / (safe_r2 * rho),
            -rho / safe_r2,
        ),
        dim=-1,
    )
    safe_rho2 = torch.clamp(rho2, min=eps * eps)
    dphi_dxyz = torch.stack(
        (
            -y_cart / safe_rho2,
            x / safe_rho2,
            torch.zeros_like(x),
        ),
        dim=-1,
    )
    dxyz = dtheta.unsqueeze(-1) * dtheta_dxyz.unsqueeze(0) + dphi.unsqueeze(-1) * dphi_dxyz.unsqueeze(0)
    return harmonics, dxyz


def spherical_harmonic(m, l, theta, phi):
    """
    Legacy-compatible accessor for a single ``Y_l^m``.
    ``m`` and ``l`` are expected to be scalars (Python ints or 0-d tensors).
    """
    if torch.is_tensor(m):
        m = int(m.item())
    if torch.is_tensor(l):
        l = int(l.item())
    if abs(m) > l:
        return torch.zeros_like(theta, dtype=torch.complex64 if theta.dtype == torch.float32 else torch.complex128)
    ys = spherical_harmonics_l(l, theta, phi)
    return ys[m + l]


def trc_arctan2(y, x):
    """Compatibility wrapper around torch.atan2."""
    return torch.atan2(y, x)
