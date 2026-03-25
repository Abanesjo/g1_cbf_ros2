"""CBF solver: minimum scaling alpha, gradient via implicit differentiation.

Generalizes g1_cbf/python_cbf/cbf_solver.py from 2D to 3D.
Reference: Dai et al., "Safe Navigation and Obstacle Avoidance Using
Differentiable Optimization Based Control Barrier Functions", RA-L 2023.
"""

import numpy as np
from g1_cbf.scaling import Ellipsoid3D


def _skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix [v]_x such that [v]_x @ w = v x w."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


class EllipsoidCBF3D:
    """Computes minimum scaling factor alpha between two 3D ellipsoids
    and its gradient w.r.t. joint configuration via implicit differentiation.
    """

    def __init__(self, beta: float = 1.05, gamma: float = 5.0):
        self.beta = beta
        self.gamma = gamma

    @staticmethod
    def solve_alpha(
        ellA: Ellipsoid3D, ellB: Ellipsoid3D, nu_init: float = 0.5
    ) -> tuple:
        """Find minimum uniform scaling alpha* via 1-D Newton on KKT.

        At the optimum, F_A(p*) = F_B(p*) = alpha*.
        Returns (alpha, p_star, nuA).
        """
        cA, MA = ellA.center, ellA.M
        cB, MB = ellB.center, ellB.M

        nuA = np.clip(nu_init, 0.01, 0.99)

        for _ in range(30):
            nuB = 1.0 - nuA
            Mc = nuA * MA + nuB * MB
            rhs = nuA * MA @ cA + nuB * MB @ cB
            p = np.linalg.solve(Mc, rhs)

            dA = p - cA
            dB = p - cB
            fA = dA @ MA @ dA
            fB = dB @ MB @ dB

            res = fA - fB
            if abs(res) < 1e-12:
                break

            dp = np.linalg.solve(Mc, MA @ cA - MB @ cB - (MA - MB) @ p)
            dres = 2.0 * dA @ MA @ dp - 2.0 * dB @ MB @ dp
            if abs(dres) < 1e-15:
                break

            nuA = np.clip(nuA - res / dres, 0.001, 0.999)

        nuB = 1.0 - nuA
        Mc = nuA * MA + nuB * MB
        p_star = np.linalg.solve(Mc, nuA * MA @ cA + nuB * MB @ cB)
        alpha = float((p_star - cA) @ MA @ (p_star - cA))
        return alpha, p_star, nuA

    @staticmethod
    def compute_dalpha_dq(
        p: np.ndarray,
        alpha: float,
        nuA: float,
        ellA: Ellipsoid3D,
        ellB: Ellipsoid3D,
        J_A: np.ndarray,
        J_B: np.ndarray,
    ) -> np.ndarray:
        """Compute dalpha/dq (n_q,) via implicit differentiation of KKT.

        Parameters
        ----------
        p : (3,) optimal intersection point
        alpha : scalar optimal scaling
        nuA : scalar KKT dual variable
        ellA, ellB : Ellipsoid3D objects with current poses
        J_A : (6, n_q) Jacobian at ellipsoid A center [trans; rot]
        J_B : (6, n_q) Jacobian at ellipsoid B center [trans; rot]

        Returns
        -------
        dalpha_dq : (n_q,)
        """
        nuB = 1.0 - nuA
        MA, MB = ellA.M, ellB.M
        cA, cB = ellA.center, ellB.center
        dA = p - cA
        dB = p - cB
        n_q = J_A.shape[1]

        # Build dg/dz (5x5) for KKT system:
        # g1: 2*nu*MA*(p-cA) + 2*(1-nu)*MB*(p-cB) = 0  [3]
        # g2: (p-cA)^T MA (p-cA) - alpha = 0            [1]
        # g3: (p-cB)^T MB (p-cB) - alpha = 0            [1]
        # z = [p(3), alpha(1), nu(1)]
        Dz = np.zeros((5, 5))
        Dz[:3, :3] = 2 * nuA * MA + 2 * nuB * MB
        Dz[:3, 3] = 0.0  # dg1/dalpha
        Dz[:3, 4] = 2 * MA @ dA - 2 * MB @ dB  # dg1/dnu
        Dz[3, :3] = 2 * dA @ MA                  # dg2/dp
        Dz[3, 3] = -1.0                           # dg2/dalpha
        Dz[3, 4] = 0.0                            # dg2/dnu
        Dz[4, :3] = 2 * dB @ MB                  # dg3/dp
        Dz[4, 3] = -1.0                           # dg3/dalpha
        Dz[4, 4] = 0.0                            # dg3/dnu

        # Solve (dg/dz)^T @ lam = e_alpha once
        e_alpha = np.zeros(5)
        e_alpha[3] = 1.0
        lam = np.linalg.solve(Dz.T, e_alpha)  # (5,)

        # For each joint qi, compute dg/dqi (5-vector) and
        # dalpha/dqi = -lam^T @ dg/dqi
        dalpha_dq = np.zeros(n_q)

        for i in range(n_q):
            dg_dqi = np.zeros(5)
            # Contribution from body A
            Jt_A = J_A[:3, i]  # translational Jacobian column
            Jr_A = J_A[3:, i]  # rotational Jacobian column
            # dcA/dqi = Jt_A
            # dMA/dqi = skew(Jr_A) @ MA - MA @ skew(Jr_A)
            S_A = _skew(Jr_A)
            dMA_dqi = S_A @ MA - MA @ S_A

            # g1 depends on MA and cA:
            # d/dqi [2*nu*MA*(p-cA)] = 2*nu*(dMA*(p-cA) + MA*(-dcA))
            dg_dqi[:3] += 2 * nuA * (dMA_dqi @ dA - MA @ Jt_A)
            # g2: d/dqi [(p-cA)^T MA (p-cA)] = dA^T dMA dA - 2 dA^T MA dcA
            dg_dqi[3] += dA @ dMA_dqi @ dA - 2 * dA @ MA @ Jt_A

            # Contribution from body B
            Jt_B = J_B[:3, i]
            Jr_B = J_B[3:, i]
            S_B = _skew(Jr_B)
            dMB_dqi = S_B @ MB - MB @ S_B

            # g1: d/dqi [2*(1-nu)*MB*(p-cB)]
            dg_dqi[:3] += 2 * nuB * (dMB_dqi @ dB - MB @ Jt_B)
            # g3: d/dqi [(p-cB)^T MB (p-cB)]
            dg_dqi[4] += dB @ dMB_dqi @ dB - 2 * dB @ MB @ Jt_B

            dalpha_dq[i] = -lam @ dg_dqi

        return dalpha_dq

    def build_constraint(
        self,
        ellA: Ellipsoid3D,
        ellB: Ellipsoid3D,
        J_A: np.ndarray,
        J_B: np.ndarray,
        nu_warm: float,
    ) -> tuple:
        """Build one CBF constraint for a collision pair.

        Returns (alpha, A_row, b_val, nu_warm_out) where the constraint is:
            A_row @ dq >= b_val
        i.e.  dalpha/dq @ dq >= -gamma * (alpha - beta)
        """
        alpha, p_star, nuA = self.solve_alpha(ellA, ellB, nu_warm)
        dalpha_dq = self.compute_dalpha_dq(
            p_star, alpha, nuA, ellA, ellB, J_A, J_B
        )

        h = alpha - self.beta
        A_row = dalpha_dq          # (n_q,)
        b_val = -self.gamma * h    # scalar

        return alpha, A_row, b_val, nuA
