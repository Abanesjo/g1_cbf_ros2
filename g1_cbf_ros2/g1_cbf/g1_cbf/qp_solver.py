"""CBF-QP solver using OSQP.

Solves:  min ||dq - dq_ref||^2
         s.t. A_cbf @ dq >= b_cbf  (CBF constraints)
              dq_min <= dq <= dq_max  (velocity limits)

Uses a slack variable with large penalty for feasibility when in violation.
"""

import numpy as np
from scipy import sparse

try:
    import osqp
    _HAS_OSQP = True
except ImportError:
    _HAS_OSQP = False


class CBFQPSolver:
    """QP solver for CBF safety filter."""

    def __init__(self, n_joints: int = 8, n_cbf: int = 3, slack_weight: float = 1e4):
        self.n = n_joints
        self.n_cbf = n_cbf
        self.slack_weight = slack_weight
        self._solver = None

    def solve(
        self,
        dq_ref: np.ndarray,
        cbf_constraints: list,
        dq_min: np.ndarray,
        dq_max: np.ndarray,
    ) -> np.ndarray:
        """Solve the CBF-QP.

        Parameters
        ----------
        dq_ref : (n,) reference joint velocity
        cbf_constraints : list of (A_row, b_val) tuples
            Each constraint: A_row @ dq >= b_val
        dq_min, dq_max : (n,) joint velocity bounds

        Returns
        -------
        dq_safe : (n,) safe joint velocity
        """
        if _HAS_OSQP:
            return self._solve_osqp(dq_ref, cbf_constraints, dq_min, dq_max)
        return self._solve_scipy(dq_ref, cbf_constraints, dq_min, dq_max)

    def _solve_osqp(self, dq_ref, cbf_constraints, dq_min, dq_max):
        n = self.n
        n_slack = 1
        N = n + n_slack  # decision: [dq(n), slack(1)]
        n_cbf = len(cbf_constraints)

        # Objective: ||dq - dq_ref||^2 + M * slack^2
        # = dq^T I dq - 2 dq_ref^T dq + M * s^2
        P_diag = np.ones(N)
        P_diag[n:] = self.slack_weight
        P = sparse.diags(P_diag, format='csc') * 2.0
        q = np.zeros(N)
        q[:n] = -2.0 * dq_ref

        # Constraints: l <= A x <= u
        rows = []
        l_list = []
        u_list = []

        # CBF constraints: A_row @ dq - slack >= b_val
        # => A_row @ dq - s >= b  => [A_row, -1] @ [dq, s] >= b
        for A_row, b_val in cbf_constraints:
            row = np.zeros(N)
            row[:n] = A_row
            row[n] = -1.0  # slack
            rows.append(row)
            l_list.append(b_val)
            u_list.append(np.inf)

        # Velocity bounds: dq_min <= dq <= dq_max
        for i in range(n):
            row = np.zeros(N)
            row[i] = 1.0
            rows.append(row)
            l_list.append(dq_min[i])
            u_list.append(dq_max[i])

        # Slack >= 0
        row = np.zeros(N)
        row[n] = 1.0
        rows.append(row)
        l_list.append(0.0)
        u_list.append(np.inf)

        A = sparse.csc_matrix(np.array(rows))
        l = np.array(l_list)
        u = np.array(u_list)

        solver = osqp.OSQP()
        solver.setup(
            P, q, A, l, u,
            verbose=False,
            eps_abs=1e-6,
            eps_rel=1e-6,
            max_iter=200,
        )

        result = solver.solve()
        if result.info.status in ('solved', 'solved_inaccurate'):
            return result.x[:n]

        # Fallback: return clamped reference
        return np.clip(dq_ref, dq_min, dq_max)

    def _solve_scipy(self, dq_ref, cbf_constraints, dq_min, dq_max):
        """Fallback using scipy SLSQP."""
        from scipy.optimize import minimize

        n = self.n

        constraints = []
        for A_row, b_val in cbf_constraints:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, A=A_row, b=b_val: A @ x - b,
                'jac': lambda x, A=A_row: A,
            })

        bounds = list(zip(dq_min, dq_max))

        res = minimize(
            lambda x: np.sum((x - dq_ref) ** 2),
            x0=np.clip(dq_ref, dq_min, dq_max),
            jac=lambda x: 2.0 * (x - dq_ref),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-10},
        )

        if res.success:
            return res.x
        return np.clip(dq_ref, dq_min, dq_max)
