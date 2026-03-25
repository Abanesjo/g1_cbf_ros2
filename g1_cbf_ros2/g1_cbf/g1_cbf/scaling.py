"""3D ellipsoid scaling function for CBF collision detection.

Generalizes the 2D ellipse from g1_cbf/python_cbf/ellipse.py to 3D.
Scaling function: F(p) = (p - c)^T M (p - c)
where M = R @ diag(1/a^2, 1/b^2, 1/c^2) @ R^T.
"""

import numpy as np


class Ellipsoid3D:
    """3D ellipsoid defined by center, rotation, and semi-axes (a, b, c).

    The scaling function F(p) = (p - c)^T M (p - c) equals 1 on the
    ellipsoid surface, < 1 inside, > 1 outside.
    """

    def __init__(self, center: np.ndarray, R: np.ndarray, semi_axes: tuple):
        self.semi_axes = semi_axes
        self.P = np.diag([1.0 / a**2 for a in semi_axes])
        self.center = np.asarray(center, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.M = self.R @ self.P @ self.R.T

    def update(self, center: np.ndarray, R: np.ndarray):
        """Update pose after FK. Recomputes M = R P R^T."""
        self.center = np.asarray(center, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.M = self.R @ self.P @ self.R.T

    def scaling(self, p: np.ndarray) -> float:
        """Evaluate F(p) = (p - c)^T M (p - c)."""
        d = p - self.center
        return float(d @ self.M @ d)
