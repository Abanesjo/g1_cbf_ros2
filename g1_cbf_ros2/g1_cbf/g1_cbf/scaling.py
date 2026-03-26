"""Scaling functions for CBF collision bodies.

Ellipsoid3D: F(p) = (p-c)^T M (p-c), quadratic scaling
BoxBody3D:   Halfspace representation, linear scaling (paper eq 41)
"""

import numpy as np


def _skew(v):
    """Skew-symmetric matrix [v]_x."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


class Ellipsoid3D:
    """F(p) = (p-c)^T M (p-c), M = R diag(1/a^2) R^T."""

    def __init__(self, center, R, semi_axes):
        self.semi_axes = semi_axes
        self.P = np.diag([1.0 / a**2 for a in semi_axes])
        self.P_inv = np.diag([a**2 for a in semi_axes])
        self.center = np.asarray(center, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self._update_matrices()

    def _update_matrices(self):
        self.M = self.R @ self.P @ self.R.T
        self.M_inv = self.R @ self.P_inv @ self.R.T

    def update(self, center, R):
        self.center = np.asarray(center, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self._update_matrices()

    def scaling(self, p):
        d = p - self.center
        return float(d @ self.M @ d)

    def gradient(self, p):
        return 2.0 * self.M @ (p - self.center)

    def hessian(self, p):
        return 2.0 * self.M

    def dscaling_dqi(self, p, Jt_i, Jr_i):
        """(dF/dqi, d∇F/dqi) through pose change."""
        d = p - self.center
        S = _skew(Jr_i)
        dM = S @ self.M - self.M @ S
        dF = float(d @ dM @ d - 2.0 * d @ self.M @ Jt_i)
        dgrad = 2.0 * dM @ d - 2.0 * self.M @ Jt_i
        return dF, dgrad


class BoxBody3D:
    """Box defined by center, rotation, half-dimensions.

    Uses halfspace representation for the CBF (paper eq 41).
    Linear scaling: face at distance α * h_i from center.
    """

    def __init__(self, center, R, half_dims):
        self.half_dims = np.asarray(half_dims, dtype=float)
        self.center = np.asarray(center, dtype=float)
        self.R = np.asarray(R, dtype=float)

    def update(self, center, R):
        self.center = np.asarray(center, dtype=float)
        self.R = np.asarray(R, dtype=float)
