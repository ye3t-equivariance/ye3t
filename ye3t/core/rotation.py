
"""Generic rotation helpers for SO(3) representation diagnostics."""

import numpy as np
import torch

from .spherical import spherical_harmonics_l


def wigner_D_numeric(L_R, R, n_samples = 25):
    """Numerically estimate the complex Wigner-D matrix for rotation ``R``."""
    L_R = int(L_R)
    rng = np.random.default_rng(123 + L_R)
    pts = []
    while len(pts) < max(int(n_samples), 2 * L_R + 3):
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        pts.append(v)
    pts = np.asarray(pts)
    pts_rot = pts @ np.asarray(R, dtype=float).T

    def ang(v):
        theta = np.arccos(np.clip(v[:, 2], -1.0, 1.0))
        phi = np.arctan2(v[:, 1], v[:, 0])
        return theta, phi

    th, ph = ang(pts)
    thr, phr = ang(pts_rot)
    Y = spherical_harmonics_l(L_R, torch.tensor(th, dtype=torch.float64), torch.tensor(ph, dtype=torch.float64)).cpu().numpy()
    Yrot = spherical_harmonics_l(L_R, torch.tensor(thr, dtype=torch.float64), torch.tensor(phr, dtype=torch.float64)).cpu().numpy()
    return Yrot @ np.linalg.pinv(Y)


__all__ = ['wigner_D_numeric']
