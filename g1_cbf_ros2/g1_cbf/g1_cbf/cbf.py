"""CBF solver: minimum scaling alpha, gradient via implicit diff.

Handles ellipsoid-ellipsoid (1D Newton) and box-ellipsoid
(analytical per-face, from Dai et al. RA-L 2023 eq 41).
"""

import numpy as np
from g1_cbf.scaling import Ellipsoid3D, BoxBody3D, _skew


class EllipsoidCBF3D:
    """CBF for 3D collision body pairs."""

    def __init__(self, beta: float = 1.05, gamma: float = 5.0):
        self.beta = beta
        self.gamma = gamma

    # ----------------------------------------------------------
    # Ellipsoid-ellipsoid (existing, fast 1D Newton)
    # ----------------------------------------------------------

    @staticmethod
    def _solve_ellipsoid(ellA, ellB, nu_init):
        """1D Newton on KKT dual for two ellipsoids."""
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

            dp = np.linalg.solve(
                Mc,
                MA @ cA - MB @ cB - (MA - MB) @ p,
            )
            dres = (
                2.0 * dA @ MA @ dp
                - 2.0 * dB @ MB @ dp
            )
            if abs(dres) < 1e-15:
                break

            nuA = np.clip(
                nuA - res / dres, 0.001, 0.999,
            )

        nuB = 1.0 - nuA
        Mc = nuA * MA + nuB * MB
        p_star = np.linalg.solve(
            Mc, nuA * MA @ cA + nuB * MB @ cB,
        )
        alpha = float(
            (p_star - cA) @ MA @ (p_star - cA)
        )
        return alpha, p_star, nuA

    @staticmethod
    def _dalpha_ellipsoid(
        p, alpha, nuA, ellA, ellB, J_A, J_B,
    ):
        """dalpha/dq for two ellipsoids via 5x5 KKT."""
        nuB = 1.0 - nuA
        MA, MB = ellA.M, ellB.M
        dA = p - ellA.center
        dB = p - ellB.center
        n_q = J_A.shape[1]

        Dz = np.zeros((5, 5))
        Dz[:3, :3] = 2 * nuA * MA + 2 * nuB * MB
        Dz[:3, 4] = 2 * MA @ dA - 2 * MB @ dB
        Dz[3, :3] = 2 * dA @ MA
        Dz[3, 3] = -1.0
        Dz[4, :3] = 2 * dB @ MB
        Dz[4, 3] = -1.0

        e_alpha = np.zeros(5)
        e_alpha[3] = 1.0
        try:
            lam = np.linalg.solve(Dz.T, e_alpha)
        except np.linalg.LinAlgError:
            return np.zeros(n_q)

        dalpha_dq = np.zeros(n_q)
        for i in range(n_q):
            Jt_A, Jr_A = J_A[:3, i], J_A[3:, i]
            Jt_B, Jr_B = J_B[:3, i], J_B[3:, i]
            S_A = _skew(Jr_A)
            dMA = S_A @ MA - MA @ S_A
            S_B = _skew(Jr_B)
            dMB = S_B @ MB - MB @ S_B

            dg = np.zeros(5)
            dg[:3] = (
                2 * nuA * (dMA @ dA - MA @ Jt_A)
                + 2 * nuB * (dMB @ dB - MB @ Jt_B)
            )
            dg[3] = (
                dA @ dMA @ dA - 2 * dA @ MA @ Jt_A
            )
            dg[4] = (
                dB @ dMB @ dB - 2 * dB @ MB @ Jt_B
            )
            dalpha_dq[i] = -lam @ dg

        return dalpha_dq

    # ----------------------------------------------------------
    # Box-ellipsoid (analytical per-face solver)
    # ----------------------------------------------------------

    @staticmethod
    def _solve_box_ell(box, ell):
        """Analytical alpha for box-ellipsoid pair.

        Tries all 6 faces of the box. For each face with
        world normal n and half-dim h:
          e = n^T (c_ell - c_box)
          d = M_inv @ n,  q = n^T d
          k = (-1 + sqrt(1 + 4he/q)) / (2h)
          alpha = k^2 q,  p = c_ell - k d

        Returns (alpha, p, n, h, lam, nu).
        """
        M_inv = ell.M_inv
        c_ell = ell.center
        c_box = box.center
        hd = box.half_dims

        best = None

        for axis in range(3):
            h = hd[axis]
            for sign in (+1.0, -1.0):
                n = sign * box.R[:, axis]
                e = n @ (c_ell - c_box)
                d = M_inv @ n
                q = n @ d

                disc = 1.0 + 4.0 * h * e / q
                if disc < 0:
                    continue
                k = (-1.0 + np.sqrt(disc)) / (2.0 * h)
                if k <= 1e-12:
                    continue

                alpha = k * k * q
                p = c_ell - k * d

                # Check other face constraints
                xl = box.R.T @ (p - c_box)
                ok = True
                for j in range(3):
                    if abs(xl[j]) > alpha * hd[j] + 1e-8:
                        ok = False
                        break

                if ok and (
                    best is None or alpha < best[0]
                ):
                    nu = 1.0 / (2.0 * k * h + 1.0)
                    lam = 2.0 * k * nu
                    best = (
                        alpha, p.copy(), n.copy(),
                        h, lam, nu,
                    )

        if best is None:
            # Fallback: bodies overlapping
            mid = 0.5 * (c_ell + c_box)
            return (0.0, mid, box.R[:, 0], hd[0], 0.5, 0.5)

        return best

    @staticmethod
    def _dalpha_box_ell(
        p, alpha, n, h, lam, nu,
        box, ell, J_box, J_ell,
    ):
        """dalpha/dq for box-ellipsoid via 6x6 KKT.

        KKT system:
          g1: lam*n + 2*nu*M*(p-c_ell) = 0      [3]
          g2: n^T(p-c_box) - alpha*h = 0         [1]
          g3: (p-c_ell)^T M (p-c_ell) - alpha = 0 [1]
          g4: lam*h + nu - 1 = 0                  [1]
        """
        M = ell.M
        d_ell = p - ell.center
        n_q = J_box.shape[1]

        # Dz (6x6)
        Dz = np.zeros((6, 6))
        Dz[:3, :3] = 2.0 * nu * M
        # Dz[:3, 3] = 0 (dg1/dalpha)
        Dz[:3, 4] = n
        Dz[:3, 5] = 2.0 * M @ d_ell
        Dz[3, :3] = n
        Dz[3, 3] = -h
        Dz[4, :3] = 2.0 * d_ell @ M
        Dz[4, 3] = -1.0
        Dz[5, 4] = h
        Dz[5, 5] = 1.0

        e_a = np.zeros(6)
        e_a[3] = 1.0
        try:
            xi = np.linalg.solve(Dz.T, e_a)
        except np.linalg.LinAlgError:
            return np.zeros(n_q)

        dalpha = np.zeros(n_q)

        for i in range(n_q):
            Jt_T = J_box[:3, i]
            Jr_T = J_box[3:, i]
            Jt_A = J_ell[:3, i]
            Jr_A = J_ell[3:, i]

            # Box: dn/dqi, dc_box/dqi
            dn = _skew(Jr_T) @ n
            dc_box = Jt_T

            # Ell: dM/dqi, dc_ell/dqi
            S_A = _skew(Jr_A)
            dM = S_A @ M - M @ S_A
            dc_ell = Jt_A

            dg = np.zeros(6)
            # dg1 = lam*dn + 2nu*(dM*d_ell - M*dc_ell)
            dg[:3] = (
                lam * dn
                + 2.0 * nu * (dM @ d_ell - M @ dc_ell)
            )
            # dg2 = dn^T(p-c_box) - n^T dc_box
            dg[3] = (
                dn @ (p - box.center) - n @ dc_box
            )
            # dg3 = d_ell^T dM d_ell - 2 d_ell^T M dc_ell
            dg[4] = (
                d_ell @ dM @ d_ell
                - 2.0 * d_ell @ M @ dc_ell
            )
            # dg4 = 0

            dalpha[i] = -xi @ dg

        return dalpha

    # ----------------------------------------------------------
    # Dispatch
    # ----------------------------------------------------------

    def build_constraint(
        self, bodyA, bodyB, J_A, J_B,
        nu_warm, p_warm=None,
    ):
        """Build one CBF constraint for a collision pair.

        Returns (alpha, A_row, b_val, nu_out, p_star).
        """
        both_ell = (
            isinstance(bodyA, Ellipsoid3D)
            and isinstance(bodyB, Ellipsoid3D)
        )

        if both_ell:
            alpha, p_star, nuA = self._solve_ellipsoid(
                bodyA, bodyB, nu_warm,
            )
            dalpha = self._dalpha_ellipsoid(
                p_star, alpha, nuA,
                bodyA, bodyB, J_A, J_B,
            )
            h = alpha - self.beta
            return (
                alpha, dalpha, -self.gamma * h,
                nuA, p_star,
            )

        # Box-ellipsoid: identify which is which
        if isinstance(bodyA, BoxBody3D):
            box, ell = bodyA, bodyB
            J_box, J_ell = J_A, J_B
        else:
            box, ell = bodyB, bodyA
            J_box, J_ell = J_B, J_A

        result = self._solve_box_ell(box, ell)
        alpha, p_star, n_f, h_f, lam_f, nu_f = result

        dalpha = self._dalpha_box_ell(
            p_star, alpha, n_f, h_f, lam_f, nu_f,
            box, ell, J_box, J_ell,
        )

        h = alpha - self.beta
        return (
            alpha, dalpha, -self.gamma * h,
            nu_warm, p_star,
        )
