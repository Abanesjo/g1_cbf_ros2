"""Capsule CBF: alpha from line-segment distance, analytical gradient.

Alpha = d_segments / (r_A + r_B) where d_segments is the minimum
distance between two line segments. Gradient uses Jacobians directly
(dt terms cancel at closest points).
"""

import numpy as np
from g1_cbf.scaling import _skew


def _segment_closest_points(capA, capB):
    """Find closest points on two line segments.

    Returns (t_A, t_B, d, w_hat) where:
      t_A, t_B: parameters along each segment
      d: distance between closest points
      w_hat: unit vector from q_B to q_A (or zeros if d≈0)
    """
    cA, vA, lA = capA.center, capA.direction, capA.seg_half_len
    cB, vB, lB = capB.center, capB.direction, capB.seg_half_len

    d_AB = cA - cB
    cos_th = vA @ vB
    sin2 = 1.0 - cos_th * cos_th

    if sin2 < 1e-8:
        # Near-parallel segments: project centers
        tA = 0.0
        tB = d_AB @ vB
        tB = np.clip(tB, -lB, lB)
    else:
        # Standard closest point on two lines
        a = d_AB @ vA
        b = d_AB @ vB
        tA = (a - cos_th * b) / sin2
        tB = (cos_th * a - b) / sin2

        # Clamp and re-project
        if abs(tA) > lA:
            tA = np.clip(tA, -lA, lA)
            tB = (cA + tA * vA - cB) @ vB
            tB = np.clip(tB, -lB, lB)
        if abs(tB) > lB:
            tB = np.clip(tB, -lB, lB)
            tA = (cB + tB * vB - cA) @ vA
            tA = np.clip(tA, -lA, lA)

    qA = cA + tA * vA
    qB = cB + tB * vB
    w = qA - qB
    d = np.linalg.norm(w)
    w_hat = w / d if d > 1e-12 else np.zeros(3)

    return float(tA), float(tB), float(d), w_hat


class CapsuleCBF:
    """CBF for capsule collision pairs."""

    def __init__(self, beta: float = 1.05, gamma: float = 5.0):
        self.beta = beta
        self.gamma = gamma

    @staticmethod
    def solve_alpha(capA, capB):
        """Alpha = segment_distance / (r_A + r_B)."""
        tA, tB, d, w_hat = _segment_closest_points(
            capA, capB,
        )
        r_sum = capA.radius + capB.radius
        alpha = d / r_sum if r_sum > 1e-12 else 1e6
        return alpha, tA, tB, w_hat

    @staticmethod
    def compute_dalpha_dq(
        tA, tB, w_hat, capA, capB, J_A, J_B,
    ):
        """Gradient dalpha/dq using Jacobians.

        dα/dq_i = (1/(rA+rB)) * ŵ · (
            dc_A + t_A·dv_A - dc_B - t_B·dv_B
        )
        where dv = skew(Jr) @ v.
        dt terms cancel because w ⊥ v at closest points.
        """
        r_sum = capA.radius + capB.radius
        vA = capA.direction
        vB = capB.direction
        n_q = J_A.shape[1]
        dalpha = np.zeros(n_q)

        for i in range(n_q):
            dc_A = J_A[:3, i]
            dv_A = _skew(J_A[3:, i]) @ vA
            dc_B = J_B[:3, i]
            dv_B = _skew(J_B[3:, i]) @ vB

            dd = w_hat @ (
                dc_A + tA * dv_A - dc_B - tB * dv_B
            )
            dalpha[i] = dd / r_sum

        return dalpha

    def build_constraint(
        self, capA, capB, J_A, J_B,
        nu_warm=0.5, p_warm=None,
    ):
        """Build one CBF constraint.

        Returns (alpha, A_row, b_val, nu_warm, p_star).
        """
        alpha, tA, tB, w_hat = self.solve_alpha(
            capA, capB,
        )
        dalpha = self.compute_dalpha_dq(
            tA, tB, w_hat, capA, capB, J_A, J_B,
        )

        h = alpha - self.beta
        # Contact midpoint (for debug/warm-start compat)
        qA = capA.center + tA * capA.direction
        qB = capB.center + tB * capB.direction
        p_star = 0.5 * (qA + qB)

        return (
            alpha, dalpha, -self.gamma * h,
            nu_warm, p_star,
        )
